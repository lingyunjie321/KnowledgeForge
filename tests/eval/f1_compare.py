"""F1 评测脚本：纯向量 RAG vs GraphRAG 检索质量对比

用法：python -m tests.eval.f1_compare
默认降级模式（规则抽实体，不调 LLM）；设 EVAL_USE_LLM=1 用真 LLM 跑 entity_linking 和 community_summary。
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any

from agents.doc_parser_agent import DocType, DocumentChunk
from services.graph_rag import GraphRAGContext, GraphRAGPipeline
from services.vector_store import VectorStoreService
from tests.eval.in_memory_graph import InMemoryGraph


class BagOfWordsEmbeddings:
    """词袋向量：中文按字、英文按词，共现词越多向量越相似，给纯向量检索提供稳定语义"""

    DIM = 512

    def _tokenize(self, text: str) -> list[str]:
        # 中文单字 + 英文/数字连续串
        return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text)

    def embed_query(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        for tok in self._tokenize(text):
            vec[hash(tok) % self.DIM] += 1.0
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


@dataclass
class QAItem:
    question: str
    answer_keywords: list[str]
    note: str = ""


@dataclass
class CorpusDoc:
    doc_id: str
    content: str


# 多跳问题（同事关系、跨实体推理）是 GraphRAG 的优势场景
CORPUS = [
    CorpusDoc("d1", "张三是 ACME 公司的资深工程师，负责知识图谱模块的研发工作。"),
    CorpusDoc("d2", "李四在 ACME 公司担任产品经理，主导企业知识库产品规划。"),
    CorpusDoc("d3", "ACME 公司总部位于北京，成立于 2015 年。"),
    CorpusDoc("d4", "王五就职于 Beta 科技公司，该公司总部在上海。"),
    CorpusDoc("d5", "Neo4j 是一款高性能的图数据库，广泛用于存储实体与关系数据。"),
    CorpusDoc("d6", "张三毕业于清华大学计算机系，研究方向是自然语言处理。"),
]

ENTITIES = [
    ("张三", "Person", "ACME 工程师"),
    ("李四", "Person", "ACME 产品经理"),
    ("王五", "Person", "Beta 员工"),
    ("ACME", "Organization", "科技公司"),
    ("Beta", "Organization", "科技公司"),
    ("北京", "Location", "城市"),
    ("上海", "Location", "城市"),
    ("Neo4j", "Technology", "图数据库"),
    ("清华大学", "Organization", "高校"),
    ("知识图谱", "Concept", "技术模块"),
]

RELATIONS = [
    ("张三", "WORKS_AT", "ACME"),
    ("李四", "WORKS_AT", "ACME"),
    ("王五", "WORKS_AT", "Beta"),
    ("ACME", "LOCATED_IN", "北京"),
    ("Beta", "LOCATED_IN", "上海"),
    ("张三", "USES", "Neo4j"),
    ("张三", "PART_OF", "知识图谱"),
    ("张三", "BELONGS_TO", "清华大学"),
]

DATASET = [
    QAItem("张三在哪家公司工作？", ["ACME"], "单跳事实"),
    QAItem("张三和李四是什么关系？", ["ACME", "同事", "同公司"], "多跳推理"),
    QAItem("ACME 公司总部在哪里？", ["北京"], "单跳事实"),
    QAItem("张三的同事是谁？", ["李四"], "多跳推理"),
    QAItem("张三使用什么数据库？", ["Neo4j"], "关联推理"),
    QAItem("张三毕业于哪所大学？", ["清华大学"], "单跳事实"),
]


async def build_corpus(vs: VectorStoreService, kg: InMemoryGraph) -> None:
    chunks = [
        DocumentChunk(content=d.content, doc_id=d.doc_id, chunk_index=0, doc_type=DocType.TEXT, metadata={"source": d.doc_id})
        for d in CORPUS
    ]
    await vs.add_chunks(chunks)
    for name, etype, desc in ENTITIES:
        kg.add_entity(name, etype, desc)
    for head, rel, tail in RELATIONS:
        kg.add_relation(head, rel, tail)


def _rule_based_entity_link(question: str, known: list[str]) -> list[str]:
    return [e for e in known if e in question]


async def eval_vector_only(vs: VectorStoreService, questions: list[str]) -> list[list[str]]:
    out: list[list[str]] = []
    for q in questions:
        results = await vs.search(q, top_k=3)
        out.append([doc["content"] for doc, _ in results])
    return out


async def eval_graphrag(
    vs: VectorStoreService,
    kg: InMemoryGraph,
    questions: list[str],
    use_llm: bool,
) -> list[list[str]]:
    pipeline = GraphRAGPipeline(vector_store=vs, knowledge_graph=kg)

    known_entities = [name for name, _, _ in ENTITIES]

    if not use_llm:
        # 降级：规则抽实体 + 拼接式社区摘要，避免调 LLM
        async def _rule_link(query: str) -> list[str]:
            return _rule_based_entity_link(query, known_entities)

        async def _concat_community(subgraph_results: list[GraphRAGContext]) -> GraphRAGContext:
            text = "\n".join(r.content for r in subgraph_results[:10])
            return GraphRAGContext(content=f"子图摘要:\n{text}", source_type="community", score=0.9)

        pipeline._entity_linking = _rule_link  # type: ignore
        pipeline._community_summary = _concat_community  # type: ignore

    out: list[list[str]] = []
    for q in questions:
        contexts = await pipeline.retrieve(q, top_k=3)
        out.append([c.content for c in contexts])
    return out


def compute_metrics(retrieved: list[list[str]], dataset: list[QAItem]) -> dict[str, float]:
    hits = 0
    total_relevant_retrieved = 0
    total_retrieved = 0
    for contents, item in zip(retrieved, dataset):
        total_retrieved += len(contents)
        hit = False
        for content in contents:
            if any(kw in content for kw in item.answer_keywords):
                total_relevant_retrieved += 1
                hit = True
        if hit:
            hits += 1
    recall = hits / len(dataset) if dataset else 0.0
    precision = total_relevant_retrieved / total_retrieved if total_retrieved else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"recall@k": recall, "precision@k": precision, "f1": f1, "hits": float(hits)}


def _print_table(name_a: str, metrics_a: dict, name_b: str, metrics_b: dict) -> None:
    print(f"\n{'指标':<16}{name_a:<16}{name_b:<16}{'提升':<12}")
    print("-" * 60)
    for key in ("recall@k", "precision@k", "f1"):
        a, b = metrics_a[key], metrics_b[key]
        delta = b - a
        pct = (delta / a * 100) if a > 0 else float("inf")
        pct_str = f"+{pct:.1f}%" if pct != float("inf") and delta >= 0 else (f"{pct:.1f}%" if pct != float("inf") else "N/A")
        print(f"{key:<16}{a:<16.4f}{b:<16.4f}{pct_str:<12}")
    print("-" * 60)


async def main() -> int:
    import tempfile
    from config import settings
    # 脚本独立运行时用临时目录，避免 chroma_data 残留污染结果
    settings.upload_dir = tempfile.mkdtemp(prefix="kf_eval_")

    use_llm = os.environ.get("EVAL_USE_LLM") == "1"
    if use_llm:
        # 探测 LLM 是否可用，不可用则回退降级模式并明确提示
        try:
            from langchain_openai import ChatOpenAI
            from config import settings
            from langchain_core.messages import HumanMessage
            llm = ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key, base_url=settings.openai_base_url, temperature=0)
            await llm.ainvoke([HumanMessage(content="ping")])
        except Exception as e:
            print(f"⚠️  LLM 不可用 ({type(e).__name__})，自动回退降级模式（规则抽实体）")
            print(f"    充值或配置 OPENAI_API_KEY 后可用 EVAL_USE_LLM=1 跑真 LLM 评测\n")
            use_llm = False

    mode = "真 LLM" if use_llm else "降级（规则抽实体）"
    print(f"评测模式: {mode}")
    print(f"数据集规模: {len(DATASET)} 个 QA, {len(CORPUS)} 篇文档, {len(ENTITIES)} 个实体, {len(RELATIONS)} 条关系")

    vs = VectorStoreService()
    vs._embeddings = BagOfWordsEmbeddings()
    await vs.init()
    kg = InMemoryGraph()
    await build_corpus(vs, kg)

    questions = [item.question for item in DATASET]

    vector_contents = await eval_vector_only(vs, questions)
    graphrag_contents = await eval_graphrag(vs, kg, questions, use_llm)

    vector_metrics = compute_metrics(vector_contents, DATASET)
    graphrag_metrics = compute_metrics(graphrag_contents, DATASET)

    _print_table("纯向量RAG", vector_metrics, "GraphRAG", graphrag_metrics)

    print("\n逐题命中情况:")
    for i, item in enumerate(DATASET):
        v_hit = any(any(kw in c for kw in item.answer_keywords) for c in vector_contents[i])
        g_hit = any(any(kw in c for kw in item.answer_keywords) for c in graphrag_contents[i])
        flag = "多跳" if "多跳" in item.note else "单跳"
        print(f"  [{flag}] {item.question:<22} 向量:{'✓' if v_hit else '✗'}  GraphRAG:{'✓' if g_hit else '✗'}")

    await kg.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
