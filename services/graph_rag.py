"""GraphRAG 混合检索管道：向量检索 + 图谱遍历 + 路径推理 + 社区摘要"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.doc_parser_agent import DocType
from config import settings
from services.knowledge_graph import KnowledgeGraphService
from services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


@dataclass
class GraphRAGContext:
    content: str
    source_type: str  # vector | subgraph | path | community
    score: float
    doc_type: str = DocType.TEXT.value
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrieveStep:
    """6 步管道的单步执行记录，给前端时间线渲染用"""
    name: str          # vector / entity_linking / subgraph / path / community / rerank
    label: str         # 中文标签
    hits: int = 0      # 该步命中的上下文条数
    cost_ms: int = 0   # 该步耗时
    detail: str = ""   # 可选细节，如命中的实体名


_SIX_STEP_LABELS = {
    "vector": "向量检索",
    "entity_linking": "实体链接",
    "subgraph": "子图遍历",
    "path": "路径推理",
    "community": "社区摘要",
    "rerank": "交叉重排",
}


ENTITY_LINKING_PROMPT = """\
从以下问题中提取所有可能的实体名称。
返回 JSON: {"entities": ["实体1", "实体2"]}
只返回 JSON。
"""

COMMUNITY_SUMMARY_PROMPT = """\
你是知识图谱分析专家。根据以下子图信息生成结构化摘要：
1. 概述核心实体和关系
2. 突出实体间关键联系
3. 指出有价值的推理链
"""


class GraphRAGPipeline:
    """混合检索管道：向量语义 + 图谱结构 + 路径推理 + 社区摘要"""

    def __init__(
        self,
        vector_store: VectorStoreService,
        knowledge_graph: KnowledgeGraphService,
    ) -> None:
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
        )

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        steps: list[RetrieveStep] | None = None,
        vector_queries: list[str] | None = None,
        entities_hint: list[str] | None = None,
    ) -> list[GraphRAGContext]:
        steps = steps if steps is not None else []

        search_queries = self._merge_unique(vector_queries or [query])
        (vector_results, vector_ms), (linked_entities, entity_ms) = await asyncio.gather(
            self._timed(self._vector_search_many(search_queries, top_k=top_k)),
            self._timed(self._entity_linking(query)),
        )
        steps.append(RetrieveStep(
            name="vector", label=_SIX_STEP_LABELS["vector"],
            hits=len(vector_results), cost_ms=vector_ms,
        ))

        entities = self._merge_unique([
            *(entities_hint or []),
            *linked_entities,
        ])
        steps.append(RetrieveStep(
            name="entity_linking", label=_SIX_STEP_LABELS["entity_linking"],
            hits=len(entities), cost_ms=entity_ms,
            detail="、".join(entities[:5]),
        ))

        (subgraph_results, subgraph_ms), (path_results, path_ms) = await asyncio.gather(
            self._timed(self._subgraph_search(entities)),
            self._timed(self._path_search(entities)),
        )
        steps.append(RetrieveStep(
            name="subgraph", label=_SIX_STEP_LABELS["subgraph"],
            hits=len(subgraph_results), cost_ms=subgraph_ms,
        ))

        steps.append(RetrieveStep(
            name="path", label=_SIX_STEP_LABELS["path"],
            hits=len(path_results), cost_ms=path_ms,
        ))

        all_results = vector_results + subgraph_results + path_results

        t0 = time.perf_counter()
        community_hits = 0
        if subgraph_results:
            community_ctx = await self._community_summary(subgraph_results)
            all_results.append(community_ctx)
            community_hits = 1
        steps.append(RetrieveStep(
            name="community", label=_SIX_STEP_LABELS["community"],
            hits=community_hits, cost_ms=int((time.perf_counter() - t0) * 1000),
        ))

        t0 = time.perf_counter()
        reranked = self._cross_rerank(all_results, query)
        steps.append(RetrieveStep(
            name="rerank", label=_SIX_STEP_LABELS["rerank"],
            hits=len(reranked), cost_ms=int((time.perf_counter() - t0) * 1000),
        ))

        return reranked[:top_k]

    async def _timed(self, awaitable) -> tuple[Any, int]:
        t0 = time.perf_counter()
        result = await awaitable
        return result, int((time.perf_counter() - t0) * 1000)

    async def _vector_search_many(self, queries: list[str], top_k: int = 5) -> list[GraphRAGContext]:
        contexts: list[GraphRAGContext] = []
        for query in queries:
            contexts.extend(await self._vector_search(query, top_k=top_k))

        seen: set[str] = set()
        unique: list[GraphRAGContext] = []
        for ctx in contexts:
            key = self._context_key(ctx)
            if key not in seen:
                seen.add(key)
                unique.append(ctx)
        return unique

    async def _vector_search(self, query: str, top_k: int = 5) -> list[GraphRAGContext]:
        results = await self.vector_store.search(query, top_k=top_k)
        contexts: list[GraphRAGContext] = []
        for doc, score in results:
            metadata = dict(doc.get("metadata", {}))
            source = metadata.get("source") or doc.get("source", "")
            if source:
                metadata["source"] = source
            contexts.append(GraphRAGContext(
                content=doc["content"],
                source_type="vector",
                score=score,
                doc_type=metadata.get("doc_type", DocType.TEXT.value),
                metadata=metadata,
            ))
        return contexts

    @staticmethod
    def _merge_unique(values: list[Any]) -> list[str]:
        items: list[str] = []
        seen: set[str] = set()
        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if text and text not in seen:
                seen.add(text)
                items.append(text)
        return items

    @staticmethod
    def _context_key(ctx: GraphRAGContext) -> str:
        metadata = ctx.metadata
        for key in ("chunk_id", "id"):
            value = metadata.get(key)
            if value:
                return f"{key}:{value}"
        doc_id = metadata.get("doc_id")
        chunk_index = metadata.get("chunk_index")
        if doc_id is not None and chunk_index is not None:
            return f"doc:{doc_id}:chunk:{chunk_index}"
        return f"content:{ctx.content[:120]}"

    async def _entity_linking(self, query: str) -> list[str]:
        messages = [
            SystemMessage(content=ENTITY_LINKING_PROMPT),
            HumanMessage(content=query),
        ]
        resp = await self.llm.ainvoke(messages)
        try:
            cleaned = resp.content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(cleaned)
            return data.get("entities", [])
        except (json.JSONDecodeError, IndexError):
            return []

    async def _subgraph_search(self, entities: list[str], hops: int = 2) -> list[GraphRAGContext]:
        results = await asyncio.gather(*[
            self._subgraph_for_entity(entity_name, hops)
            for entity_name in entities
        ])
        return [ctx for group in results for ctx in group]

    async def _subgraph_for_entity(self, entity_name: str, hops: int) -> list[GraphRAGContext]:
        contexts: list[GraphRAGContext] = []
        neighbors = await self.knowledge_graph.get_neighbors(entity_name, hops=hops)
        for record in neighbors:
            source_docs = self._record_sources(record)
            metadata = {"entity": entity_name, "hops": hops, "source_docs": source_docs}
            if source_docs:
                metadata["source"] = source_docs[0]
            content = (
                f"{record.get('source', '')} "
                f"--[{', '.join(record.get('relations', []))}]--> "
                f"{record.get('target', '')} "
                f"({record.get('target_type', '')}): "
                f"{record.get('target_desc', '')}"
            )
            contexts.append(GraphRAGContext(
                content=content,
                source_type="subgraph",
                score=0.75,
                metadata=metadata,
            ))
        return contexts

    async def _path_search(self, entities: list[str]) -> list[GraphRAGContext]:
        if len(entities) < 2:
            return []

        pairs = [
            (entities[i], entities[j])
            for i in range(len(entities))
            for j in range(i + 1, min(i + 3, len(entities)))
        ]
        results = await asyncio.gather(*[
            self._path_between(name_a, name_b)
            for name_a, name_b in pairs
        ])
        return [ctx for group in results for ctx in group]

    async def _path_between(self, name_a: str, name_b: str) -> list[GraphRAGContext]:
        contexts: list[GraphRAGContext] = []
        cypher = """
        MATCH path = shortestPath(
            (a:Entity {name: $name_a})-[*..5]-(b:Entity {name: $name_b})
        )
        RETURN
            [n IN nodes(path) | n.name] AS node_names,
            [r IN relationships(path) | type(r)] AS rel_types,
            [r IN relationships(path) | r.source] AS source_docs
        LIMIT 3
        """
        try:
            records = await self.knowledge_graph.execute_cypher(
                cypher, {"name_a": name_a, "name_b": name_b}
            )
            for rec in records:
                nodes = rec.get("node_names", [])
                rels = rec.get("rel_types", [])
                path_str = ""
                for k, node in enumerate(nodes):
                    path_str += node
                    if k < len(rels):
                        path_str += f" --[{rels[k]}]--> "
                source_docs = self._record_sources(rec)
                metadata = {
                    "from": name_a,
                    "to": name_b,
                    "path": nodes,
                    "source_docs": source_docs,
                }
                if source_docs:
                    metadata["source"] = source_docs[0]
                contexts.append(GraphRAGContext(
                    content=f"推理路径: {path_str}",
                    source_type="path",
                    score=0.85,
                    metadata=metadata,
                ))
        except Exception:
            logger.exception("路径检索失败: %s → %s", name_a, name_b)
        return contexts

    async def _community_summary(self, subgraph_results: list[GraphRAGContext]) -> GraphRAGContext:
        subgraph_text = "\n".join(r.content for r in subgraph_results[:20])
        messages = [
            SystemMessage(content=COMMUNITY_SUMMARY_PROMPT),
            HumanMessage(content=f"子图信息:\n{subgraph_text}"),
        ]
        resp = await self.llm.ainvoke(messages)
        source_docs = self._merge_unique([
            source
            for result in subgraph_results
            for source in result.metadata.get("source_docs", [])
        ])
        metadata: dict[str, Any] = {"type": "community_summary", "source_docs": source_docs}
        if source_docs:
            metadata["source"] = source_docs[0]
        return GraphRAGContext(
            content=resp.content,
            source_type="community",
            score=0.9,
            metadata=metadata,
        )

    @classmethod
    def _record_sources(cls, record: dict) -> list[str]:
        values: list[str] = []
        raw = record.get("source_docs", [])
        if isinstance(raw, list):
            values.extend(raw)
        elif raw:
            values.append(raw)
        for key in ("source_doc", "target_source"):
            value = record.get(key)
            if value:
                values.append(value)
        return cls._merge_unique(values)

    @staticmethod
    def _cross_rerank(contexts: list[GraphRAGContext], query: str) -> list[GraphRAGContext]:
        """交叉重排序：来源权重 × 多模态权重，统一打分"""
        # 来源权重：路径推理最有价值，社区摘要次之，子图再次，向量基础
        source_weight = {"vector": 1.0, "subgraph": 1.15, "path": 1.25, "community": 1.1}
        # 多模态权重：纯文本质量最高，图片是转文本再嵌入的所以打折
        # 从原 multimodal.py 合并而来，保留跨模态加权重排能力
        doc_type_weight = {
            DocType.TEXT.value: 1.0,
            DocType.MARKDOWN.value: 1.0,
            DocType.PDF.value: 0.95,
            DocType.TABLE.value: 0.9,
            DocType.IMAGE.value: 0.85,
        }
        for ctx in contexts:
            ctx.score *= source_weight.get(ctx.source_type, 1.0)
            ctx.score *= doc_type_weight.get(ctx.doc_type, 1.0)

        seen: set[str] = set()
        unique: list[GraphRAGContext] = []
        for ctx in contexts:
            key = ctx.content[:80]
            if key not in seen:
                seen.add(key)
                unique.append(ctx)

        unique.sort(key=lambda c: c.score, reverse=True)
        return unique
