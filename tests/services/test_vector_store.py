"""VectorStoreService 单测：chroma 后端真入库真检索、删除、降级"""

from __future__ import annotations

import pytest

from agents.doc_parser_agent import DocType, DocumentChunk
from services.vector_store import VectorStoreService
from tests.conftest import make_chunk


def _make_vs_with_embeddings(deterministic_embeddings) -> VectorStoreService:
    vs = VectorStoreService()
    # 跳过 create_embeddings，直接注入确定性向量，让 chromadb cosine 能正常算
    vs._embeddings = deterministic_embeddings
    return vs


async def test_add_and_search_roundtrip(deterministic_embeddings):
    vs = _make_vs_with_embeddings(deterministic_embeddings)
    await vs.init()

    chunks = [
        make_chunk("知识图谱用 Neo4j 存储", doc_id="d1", chunk_index=0),
        make_chunk("向量检索用 ChromaDB", doc_id="d1", chunk_index=1),
        make_chunk("今天天气不错", doc_id="d2", chunk_index=0),
    ]
    added = await vs.add_chunks(chunks)
    assert added == 3

    results = await vs.search("知识图谱存储", top_k=2)
    # P0-1 验收：真入库后 search 必须返回带分数的结果，不再 return []
    assert len(results) >= 1
    for doc, score in results:
        assert "content" in doc
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


async def test_search_returns_metadata(deterministic_embeddings):
    vs = _make_vs_with_embeddings(deterministic_embeddings)
    await vs.init()

    chunk = make_chunk("带元数据的 chunk", doc_id="docA", chunk_index=0, doc_type=DocType.MARKDOWN, source="readme.md")
    await vs.add_chunks([chunk])

    results = await vs.search("元数据", top_k=1)
    assert len(results) == 1
    doc, _ = results[0]
    assert doc["metadata"]["doc_id"] == "docA"
    assert doc["metadata"]["doc_type"] == DocType.MARKDOWN.value
    assert doc["source"] == "readme.md"


async def test_delete_by_doc_id_removes_chunks(deterministic_embeddings):
    vs = _make_vs_with_embeddings(deterministic_embeddings)
    await vs.init()

    await vs.add_chunks([
        make_chunk("内容A", doc_id="d1", chunk_index=0),
        make_chunk("内容B", doc_id="d2", chunk_index=0),
    ])
    deleted = await vs.delete_by_doc_id("d1")
    assert deleted == 1

    # 删除后再检索不应返回 d1 的内容
    results = await vs.search("内容A", top_k=5)
    for doc, _ in results:
        assert doc["metadata"]["doc_id"] != "d1"


async def test_search_empty_store_returns_empty(deterministic_embeddings):
    vs = _make_vs_with_embeddings(deterministic_embeddings)
    await vs.init()

    results = await vs.search("任意查询", top_k=5)
    assert results == []


async def test_search_returns_empty_when_embeddings_unavailable(monkeypatch):
    # create_embeddings 抛异常时 embeddings 为 None，search 直接降级返回空
    import services.vector_store as vs_mod
    monkeypatch.setattr(vs_mod, "create_embeddings", lambda: (_ for _ in ()).throw(RuntimeError("不可用")))
    vs = VectorStoreService()
    await vs.init()

    results = await vs.search("q", top_k=5)
    assert results == []


async def test_add_chunks_returns_zero_when_embeddings_unavailable(monkeypatch):
    import services.vector_store as vs_mod
    monkeypatch.setattr(vs_mod, "create_embeddings", lambda: (_ for _ in ()).throw(RuntimeError("不可用")))
    vs = VectorStoreService()
    await vs.init()
    count = await vs.add_chunks([make_chunk("x")])
    assert count == 0


async def test_get_stats_returns_count(deterministic_embeddings):
    vs = _make_vs_with_embeddings(deterministic_embeddings)
    await vs.init()

    await vs.add_chunks([
        make_chunk("a", doc_id="d1"),
        make_chunk("b", doc_id="d1", chunk_index=1),
    ])
    stats = await vs.get_stats()
    assert stats["backend"] == "chroma"
    assert stats["total_vectors"] == 2
