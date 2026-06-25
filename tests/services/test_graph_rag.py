"""GraphRAGPipeline 单测：交叉重排权重、路径检索 Cypher 参数化、全流程"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.doc_parser_agent import DocType
from services.graph_rag import GraphRAGContext, GraphRAGPipeline


def test_cross_rerank_applies_source_weight():
    # 同分同类型，只差来源；path 权重最高应排最前
    contexts = [
        GraphRAGContext(content="A", source_type="vector", score=0.8, doc_type=DocType.TEXT.value),
        GraphRAGContext(content="B", source_type="subgraph", score=0.8, doc_type=DocType.TEXT.value),
        GraphRAGContext(content="C", source_type="path", score=0.8, doc_type=DocType.TEXT.value),
        GraphRAGContext(content="D", source_type="community", score=0.8, doc_type=DocType.TEXT.value),
    ]
    ranked = GraphRAGPipeline._cross_rerank(contexts, "q")
    sources = [c.source_type for c in ranked]
    # path(1.25) > community(1.1) > subgraph(1.15) > vector(1.0)
    # 0.8*1.25=1.0, 0.8*1.1=0.88, 0.8*1.15=0.92, 0.8*1.0=0.8
    assert sources == ["path", "subgraph", "community", "vector"]


def test_cross_rerank_applies_doc_type_weight():
    # 同来源同分，只差 doc_type；text 权重最高
    contexts = [
        GraphRAGContext(content="img", source_type="vector", score=0.9, doc_type=DocType.IMAGE.value),
        GraphRAGContext(content="txt", source_type="vector", score=0.9, doc_type=DocType.TEXT.value),
        GraphRAGContext(content="tbl", source_type="vector", score=0.9, doc_type=DocType.TABLE.value),
    ]
    ranked = GraphRAGPipeline._cross_rerank(contexts, "q")
    types = [c.doc_type for c in ranked]
    # text(1.0) > table(0.9) > image(0.85)
    assert types == [DocType.TEXT.value, DocType.TABLE.value, DocType.IMAGE.value]


def test_cross_rerank_combined_source_and_doc_type():
    # path+text 应高于 vector+image
    contexts = [
        GraphRAGContext(content="low", source_type="vector", score=1.0, doc_type=DocType.IMAGE.value),
        GraphRAGContext(content="high", source_type="path", score=0.8, doc_type=DocType.TEXT.value),
    ]
    ranked = GraphRAGPipeline._cross_rerank(contexts, "q")
    # high: 0.8*1.25*1.0 = 1.0；low: 1.0*1.0*0.85 = 0.85
    assert ranked[0].content == "high"


def test_cross_rerank_deduplicates_by_content_prefix():
    # 前 80 字符相同视为重复，只保留首个
    prefix = "x" * 80
    contexts = [
        GraphRAGContext(content=prefix + "extra1", source_type="vector", score=0.9),
        GraphRAGContext(content=prefix + "extra2", source_type="path", score=0.9),
    ]
    ranked = GraphRAGPipeline._cross_rerank(contexts, "q")
    assert len(ranked) == 1
    assert ranked[0].source_type == "vector"


def test_cross_rerank_empty_returns_empty():
    assert GraphRAGPipeline._cross_rerank([], "q") == []


async def test_path_search_uses_parameterized_cypher(mock_llm):
    # 验证 Cypher 用 $name_a/$name_b 参数化，不拼字符串，且 params 传入正确
    mock_kg = MagicMock()
    mock_kg.execute_cypher = AsyncMock(return_value=[
        {"node_names": ["张三", "ACME"], "rel_types": ["WORKS_AT"], "source_docs": ["/docs/a.md"]},
    ])
    mock_vs = MagicMock()
    pipeline = GraphRAGPipeline(vector_store=mock_vs, knowledge_graph=mock_kg)

    contexts = await pipeline._path_search(["张三", "ACME"])

    assert len(contexts) == 1
    assert contexts[0].source_type == "path"
    assert "张三" in contexts[0].content and "ACME" in contexts[0].content
    assert contexts[0].metadata["source"] == "/docs/a.md"
    assert contexts[0].metadata["source_docs"] == ["/docs/a.md"]
    mock_kg.execute_cypher.assert_awaited_once()
    call_args = mock_kg.execute_cypher.call_args
    cypher = call_args.args[0]
    params = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("params")
    # 关键：参数化占位符存在，实体名只出现在 params 不在 cypher 字面量里
    assert "$name_a" in cypher and "$name_b" in cypher
    assert params == {"name_a": "张三", "name_b": "ACME"}
    assert "张三" not in cypher


async def test_path_search_skips_single_entity(mock_llm):
    mock_kg = MagicMock()
    mock_kg.execute_cypher = AsyncMock()
    pipeline = GraphRAGPipeline(vector_store=MagicMock(), knowledge_graph=mock_kg)
    contexts = await pipeline._path_search(["仅一个实体"])
    assert contexts == []
    mock_kg.execute_cypher.assert_not_awaited()


async def test_subgraph_search_builds_content(mock_llm):
    mock_kg = MagicMock()
    mock_kg.get_neighbors = AsyncMock(return_value=[
        {"source": "张三", "relations": ["WORKS_AT"], "target": "ACME",
         "target_type": "Organization", "target_desc": "科技公司",
         "source_docs": ["/docs/a.md"]},
    ])
    pipeline = GraphRAGPipeline(vector_store=MagicMock(), knowledge_graph=mock_kg)

    contexts = await pipeline._subgraph_search(["张三"])

    assert len(contexts) == 1
    assert contexts[0].source_type == "subgraph"
    assert "张三" in contexts[0].content and "ACME" in contexts[0].content
    assert "WORKS_AT" in contexts[0].content
    assert contexts[0].metadata["source"] == "/docs/a.md"
    assert contexts[0].metadata["source_docs"] == ["/docs/a.md"]


async def test_retrieve_uses_vector_queries_and_deduplicates(mock_llm, fake_llm_response):
    mock_vs = MagicMock()
    mock_vs.search = AsyncMock(side_effect=[
        [
            ({"content": "张三是工程师", "metadata": {"doc_id": "doc1", "chunk_id": "c1"}}, 0.9),
            ({"content": "ACME 是科技公司", "metadata": {"doc_id": "doc1", "chunk_id": "c2"}}, 0.8),
        ],
        [
            ({"content": "张三是工程师", "metadata": {"doc_id": "doc1", "chunk_id": "c1"}}, 0.95),
            ({"content": "张三负责 GraphRAG", "metadata": {"doc_id": "doc2", "chunk_id": "c3"}}, 0.7),
        ],
    ])
    mock_kg = MagicMock()
    mock_kg.get_neighbors = AsyncMock(return_value=[])
    mock_kg.execute_cypher = AsyncMock(return_value=[])
    mock_llm.ainvoke.return_value = fake_llm_response('{"entities": []}')

    pipeline = GraphRAGPipeline(vector_store=mock_vs, knowledge_graph=mock_kg)
    results = await pipeline.retrieve(
        "张三在哪工作",
        top_k=10,
        vector_queries=["张三 工作", "ACME 张三"],
    )

    searched = [call.args[0] for call in mock_vs.search.await_args_list]
    assert searched == ["张三 工作", "ACME 张三"]
    assert [r.content for r in results].count("张三是工程师") == 1
    assert {r.metadata["chunk_id"] for r in results} == {"c1", "c2", "c3"}


async def test_retrieve_merges_entities_hint_with_entity_linking(mock_llm, fake_llm_response):
    mock_vs = MagicMock()
    mock_vs.search = AsyncMock(return_value=[])
    mock_kg = MagicMock()
    mock_kg.get_neighbors = AsyncMock(return_value=[])
    mock_kg.execute_cypher = AsyncMock(return_value=[])
    mock_llm.ainvoke.return_value = fake_llm_response('{"entities": ["ACME", "张三"]}')

    steps = []
    pipeline = GraphRAGPipeline(vector_store=mock_vs, knowledge_graph=mock_kg)
    await pipeline.retrieve(
        "张三在哪工作",
        top_k=5,
        steps=steps,
        entities_hint=["张三"],
    )

    entity_step = next(step for step in steps if step.name == "entity_linking")
    assert entity_step.hits == 2
    assert entity_step.detail == "张三、ACME"
    assert mock_kg.get_neighbors.await_count == 2


async def test_retrieve_full_flow_merges_and_reranks(mock_llm, fake_llm_response, deterministic_embeddings):
    # vector_store 返回向量结果；llm 返回实体链接；kg 返回子图和路径
    mock_vs = MagicMock()
    mock_vs.search = AsyncMock(return_value=[
        ({"content": "张三是工程师", "metadata": {"doc_type": DocType.TEXT.value}}, 0.9),
    ])
    mock_kg = MagicMock()
    mock_kg.get_neighbors = AsyncMock(return_value=[
        {"source": "张三", "relations": ["WORKS_AT"], "target": "ACME",
         "target_type": "Organization", "target_desc": "公司"},
    ])
    mock_kg.execute_cypher = AsyncMock(return_value=[
        {"node_names": ["张三", "ACME"], "rel_types": ["WORKS_AT"]},
    ])
    # _entity_linking 用 LLM 抽实体，_community_summary 用 LLM 生成摘要
    mock_llm.ainvoke.side_effect = [
        fake_llm_response('{"entities": ["张三", "ACME"]}'),
        fake_llm_response("社区摘要：张三在 ACME 工作"),
    ]

    pipeline = GraphRAGPipeline(vector_store=mock_vs, knowledge_graph=mock_kg)
    results = await pipeline.retrieve("张三和 ACME 的关系", top_k=5)

    # 至少包含 vector + subgraph + path + community 四种来源
    sources = {r.source_type for r in results}
    assert "vector" in sources
    assert "subgraph" in sources
    assert "path" in sources
    assert "community" in sources
    # 结果已按 rerank 分数降序
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


async def test_retrieve_empty_when_no_vector_results_and_no_entities(mock_llm, fake_llm_response):
    mock_vs = MagicMock()
    mock_vs.search = AsyncMock(return_value=[])
    mock_kg = MagicMock()
    mock_kg.get_neighbors = AsyncMock(return_value=[])
    mock_kg.execute_cypher = AsyncMock(return_value=[])
    mock_llm.ainvoke.return_value = fake_llm_response('{"entities": []}')

    pipeline = GraphRAGPipeline(vector_store=mock_vs, knowledge_graph=mock_kg)
    results = await pipeline.retrieve("空查询", top_k=5)
    # 无实体→无子图/路径/社区，只剩可能的 vector（空），结果为空
    assert results == []


async def test_retrieve_runs_vector_and_entity_linking_concurrently(mock_llm):
    pipeline = GraphRAGPipeline(vector_store=MagicMock(), knowledge_graph=MagicMock())
    started: set[str] = set()
    both_started = asyncio.Event()

    async def wait_for_peer(name: str):
        started.add(name)
        if len(started) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=0.1)

    async def vector_search_many(_queries, top_k=5):
        await wait_for_peer("vector")
        return []

    async def entity_linking(_query):
        await wait_for_peer("entity")
        return []

    pipeline._vector_search_many = vector_search_many  # type: ignore[method-assign]
    pipeline._entity_linking = entity_linking  # type: ignore[method-assign]
    pipeline._subgraph_search = AsyncMock(return_value=[])
    pipeline._path_search = AsyncMock(return_value=[])

    await pipeline.retrieve("问题")

    assert started == {"vector", "entity"}


async def test_retrieve_runs_subgraph_and_path_concurrently(mock_llm):
    pipeline = GraphRAGPipeline(vector_store=MagicMock(), knowledge_graph=MagicMock())
    pipeline._vector_search_many = AsyncMock(return_value=[])
    pipeline._entity_linking = AsyncMock(return_value=["张三", "李四"])
    started: set[str] = set()
    both_started = asyncio.Event()

    async def wait_for_peer(name: str):
        started.add(name)
        if len(started) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=0.1)

    async def subgraph_search(_entities):
        await wait_for_peer("subgraph")
        return []

    async def path_search(_entities):
        await wait_for_peer("path")
        return []

    pipeline._subgraph_search = subgraph_search  # type: ignore[method-assign]
    pipeline._path_search = path_search  # type: ignore[method-assign]

    await pipeline.retrieve("问题")

    assert started == {"subgraph", "path"}


async def test_subgraph_search_queries_entities_concurrently(mock_llm):
    mock_kg = MagicMock()
    started: set[str] = set()
    both_started = asyncio.Event()

    async def get_neighbors(entity_name: str, hops: int = 2):
        started.add(entity_name)
        if len(started) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=0.1)
        return []

    mock_kg.get_neighbors = get_neighbors
    pipeline = GraphRAGPipeline(vector_store=MagicMock(), knowledge_graph=mock_kg)

    await pipeline._subgraph_search(["张三", "李四"])

    assert started == {"张三", "李四"}


async def test_path_search_queries_pairs_concurrently(mock_llm):
    mock_kg = MagicMock()
    active = 0
    max_active = 0

    async def execute_cypher(_cypher: str, params: dict):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        return []

    mock_kg.execute_cypher = execute_cypher
    pipeline = GraphRAGPipeline(vector_store=MagicMock(), knowledge_graph=mock_kg)

    await pipeline._path_search(["张三", "李四", "ACME"])

    assert max_active >= 2
