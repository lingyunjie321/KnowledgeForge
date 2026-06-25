"""测试公共夹具：隔离环境、mock LLM、内存版向量库与图谱"""

from __future__ import annotations

import hashlib
import importlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.doc_parser_agent import DocumentChunk, DocType


# 这些模块在 __init__ 里实例化 ChatOpenAI，patch 源头让它返回统一 mock
_LLM_MODULES = [
    "agents.doc_parser_agent",
    "agents.knowledge_extract_agent",
    "agents.qa_agent",
    "services.graph_rag",
]


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "disabled")
    monkeypatch.setattr("config.settings.upload_dir", str(tmp_path / "uploads"))


class FakeLLMResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeLLMChunk:
    """模拟 LangChain astream 产出的 chunk，只有 .content"""
    def __init__(self, content: str) -> None:
        self.content = content


@pytest.fixture
def fake_llm_response() -> type:
    return FakeLLMResponse


@pytest.fixture
def fake_llm_chunk() -> type:
    return FakeLLMChunk


@pytest.fixture
def mock_llm(monkeypatch) -> MagicMock:
    instance = MagicMock(name="FakeChatOpenAI")
    instance.ainvoke = AsyncMock()
    instance.astream_text = ""

    async def _fake_astream(messages, *_a, **_kw):
        if instance.astream_text:
            yield FakeLLMChunk(instance.astream_text)
            instance.astream_text = ""
        else:
            yield FakeLLMChunk("")

    instance.astream = _fake_astream
    for mod_path in _LLM_MODULES:
        mod = importlib.import_module(mod_path)
        monkeypatch.setattr(mod, "ChatOpenAI", lambda *a, **kw: instance)
    return instance


class DeterministicEmbeddings:
    """把文本 sha256 映射成固定向量，相同文本同向量，给真 chromadb 检索用"""

    DIM = 32

    def embed_query(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [((h[i % len(h)] / 255.0) - 0.5) * 2 for i in range(self.DIM)]
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


class FakeVectorStore:
    def __init__(self) -> None:
        self._chunks: dict[str, tuple[str, dict]] = {}
        self._preset_results: list[tuple[dict, float]] = []
        self.add_calls: int = 0
        self.delete_calls: list[str] = []
        self.delete_chunk_calls: list[list[str]] = []

    def set_search_results(self, results: list[tuple[dict, float]]) -> None:
        self._preset_results = results

    async def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        self.add_calls += 1
        for c in chunks:
            self._chunks[c.chunk_id] = (
                c.content,
                {"doc_id": c.doc_id, "doc_type": c.doc_type.value, "source": c.metadata.get("source", "")},
            )
        return len(chunks)

    async def search(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        return list(self._preset_results)[:top_k]

    async def delete_by_doc_id(self, doc_id: str) -> int:
        self.delete_calls.append(doc_id)
        ids = [k for k, (_, m) in self._chunks.items() if m.get("doc_id") == doc_id]
        for k in ids:
            del self._chunks[k]
        return len(ids)

    async def delete_chunks(self, chunk_ids: list[str]) -> int:
        self.delete_chunk_calls.append(chunk_ids)
        found = [chunk_id for chunk_id in chunk_ids if chunk_id in self._chunks]
        for chunk_id in found:
            del self._chunks[chunk_id]
        return len(found)

    async def get_stats(self) -> dict:
        return {"backend": "fake", "total_vectors": len(self._chunks)}


class FakeKnowledgeGraph:
    def __init__(self) -> None:
        self._entities: dict[str, Any] = {}
        self._relations: list[Any] = []
        self._neighbors: dict[str, list[dict]] = {}
        self._cypher_results: dict[str, list[dict]] = {}
        self.deleted_sources: list[str] = []

    def set_neighbors(self, entity: str, records: list[dict]) -> None:
        self._neighbors[entity] = records

    def set_cypher_result(self, cypher: str, records: list[dict]) -> None:
        self._cypher_results[cypher] = records

    async def init(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def upsert_entity(self, entity, version: int = 1, source: str = "") -> None:
        self._entities[entity.name] = entity

    async def add_relation(self, relation, source: str = "") -> None:
        self._relations.append(relation)

    async def get_neighbors(self, entity_name: str, hops: int = 2) -> list[dict]:
        return self._neighbors.get(entity_name, [])

    async def execute_cypher(self, cypher: str, params: dict | None = None) -> list[dict]:
        return self._cypher_results.get(cypher.strip(), [])

    async def delete_by_source(self, source: str) -> int:
        self.deleted_sources.append(source)
        return 0

    async def get_stats(self) -> dict:
        return {"total_entities": len(self._entities), "total_relations": len(self._relations)}


@pytest.fixture
def fake_vector_store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture
def fake_knowledge_graph() -> FakeKnowledgeGraph:
    return FakeKnowledgeGraph()


@pytest.fixture
def deterministic_embeddings() -> DeterministicEmbeddings:
    return DeterministicEmbeddings()


def make_chunk(
    content: str,
    doc_id: str = "doc1",
    chunk_index: int = 0,
    doc_type: DocType = DocType.TEXT,
    source: str = "test.txt",
) -> DocumentChunk:
    return DocumentChunk(
        content=content,
        doc_id=doc_id,
        chunk_index=chunk_index,
        doc_type=doc_type,
        metadata={"source": source},
    )


@pytest.fixture
def make_chunk_factory():
    return make_chunk
