"""GraphRAG 混合检索管道：向量检索 + 图谱遍历 + 路径推理 + 社区摘要"""

from __future__ import annotations

import json
import logging
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

    async def retrieve(self, query: str, top_k: int = 10) -> list[GraphRAGContext]:
        vector_results = await self._vector_search(query, top_k=top_k)
        entities = await self._entity_linking(query)
        subgraph_results = await self._subgraph_search(entities)
        path_results = await self._path_search(entities)

        all_results = vector_results + subgraph_results + path_results

        if subgraph_results:
            community_ctx = await self._community_summary(subgraph_results)
            all_results.append(community_ctx)

        reranked = self._cross_rerank(all_results, query)
        return reranked[:top_k]

    async def _vector_search(self, query: str, top_k: int = 5) -> list[GraphRAGContext]:
        results = await self.vector_store.search(query, top_k=top_k)
        return [
            GraphRAGContext(
                content=doc["content"],
                source_type="vector",
                score=score,
                doc_type=doc.get("metadata", {}).get("doc_type", DocType.TEXT.value),
                metadata=doc.get("metadata", {}),
            )
            for doc, score in results
        ]

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
        contexts: list[GraphRAGContext] = []
        for entity_name in entities:
            neighbors = await self.knowledge_graph.get_neighbors(entity_name, hops=hops)
            for record in neighbors:
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
                    metadata={"entity": entity_name, "hops": hops},
                ))
        return contexts

    async def _path_search(self, entities: list[str]) -> list[GraphRAGContext]:
        if len(entities) < 2:
            return []

        contexts: list[GraphRAGContext] = []
        for i in range(len(entities)):
            for j in range(i + 1, min(i + 3, len(entities))):
                # 参数化查询，避免 Cypher 注入
                cypher = """
                MATCH path = shortestPath(
                    (a:Entity {name: $name_a})-[*..5]-(b:Entity {name: $name_b})
                )
                RETURN
                    [n IN nodes(path) | n.name] AS node_names,
                    [r IN relationships(path) | type(r)] AS rel_types
                LIMIT 3
                """
                try:
                    records = await self.knowledge_graph.execute_cypher(
                        cypher, {"name_a": entities[i], "name_b": entities[j]}
                    )
                    for rec in records:
                        nodes = rec.get("node_names", [])
                        rels = rec.get("rel_types", [])
                        path_str = ""
                        for k, node in enumerate(nodes):
                            path_str += node
                            if k < len(rels):
                                path_str += f" --[{rels[k]}]--> "
                        contexts.append(GraphRAGContext(
                            content=f"推理路径: {path_str}",
                            source_type="path",
                            score=0.85,
                            metadata={"from": entities[i], "to": entities[j]},
                        ))
                except Exception:
                    logger.exception("路径检索失败: %s → %s", entities[i], entities[j])
                    continue
        return contexts

    async def _community_summary(self, subgraph_results: list[GraphRAGContext]) -> GraphRAGContext:
        subgraph_text = "\n".join(r.content for r in subgraph_results[:20])
        messages = [
            SystemMessage(content=COMMUNITY_SUMMARY_PROMPT),
            HumanMessage(content=f"子图信息:\n{subgraph_text}"),
        ]
        resp = await self.llm.ainvoke(messages)
        return GraphRAGContext(
            content=resp.content,
            source_type="community",
            score=0.9,
            metadata={"type": "community_summary"},
        )

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
