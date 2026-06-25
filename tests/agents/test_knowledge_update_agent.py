"""KnowledgeUpdateAgent 单测：变更检测、增删改处理、批处理、失败兜底"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.knowledge_extract_agent import Entity, ExtractionResult, Relation
from agents.knowledge_update_agent import (
    ChangeType,
    DocumentChange,
    KnowledgeUpdateAgent,
)
from tests.conftest import make_chunk


def _make_extraction(entities=None, relations=None):
    return ExtractionResult(
        entities=entities or [],
        relations=relations or [],
        events=[],
        source_chunk_id="c0",
    )


def test_detect_changes_new_file(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello", encoding="utf-8")

    agent = KnowledgeUpdateAgent()
    changes = agent.detect_changes([str(f)])
    assert len(changes) == 1
    assert changes[0].change_type is ChangeType.CREATED
    assert changes[0].new_hash != ""


def test_detect_changes_modified_file(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("v1", encoding="utf-8")

    agent = KnowledgeUpdateAgent()
    agent.detect_changes([str(f)])

    f.write_text("v2-changed", encoding="utf-8")
    changes = agent.detect_changes([str(f)])
    assert len(changes) == 1
    assert changes[0].change_type is ChangeType.MODIFIED
    assert changes[0].old_hash != changes[0].new_hash
    assert changes[0].before_content == "v1"
    assert changes[0].after_content == "v2-changed"


def test_detect_changes_deleted_file(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x", encoding="utf-8")

    agent = KnowledgeUpdateAgent()
    agent.detect_changes([str(f)])
    changes = agent.detect_changes([])
    assert len(changes) == 1
    assert changes[0].change_type is ChangeType.DELETED


async def test_handle_create_calls_all_dependencies(fake_vector_store, fake_knowledge_graph):
    chunk = make_chunk("内容", doc_id="d1")
    extraction = _make_extraction(
        entities=[Entity(name="张三", type="Person")],
        relations=[Relation(head="张三", relation="works_at", tail="ACME")],
    )

    mock_parser = MagicMock()
    mock_parser.parse = AsyncMock(return_value=[chunk])
    mock_extractor = MagicMock()
    mock_extractor.extract = AsyncMock(return_value=[extraction])

    agent = KnowledgeUpdateAgent(
        doc_parser=mock_parser,
        knowledge_extractor=mock_extractor,
        vector_store=fake_vector_store,
        knowledge_graph=fake_knowledge_graph,
    )

    change = DocumentChange(file_path="/x/a.txt", change_type=ChangeType.CREATED)
    result = await agent.process_change(change)

    assert result.success is True
    assert result.vectors_added == 1
    assert result.entities_added == 1
    assert result.relations_added == 1
    mock_parser.parse.assert_awaited_once_with("/x/a.txt")
    assert fake_vector_store.add_calls == 1
    assert "张三" in fake_knowledge_graph._entities


async def test_handle_create_passes_source_to_knowledge_graph(fake_vector_store):
    chunk = make_chunk("张三在 ACME 工作", doc_id="d1", source="/x/a.txt")
    extraction = ExtractionResult(
        entities=[Entity(name="张三", type="Person")],
        relations=[Relation(head="张三", relation="works_at", tail="ACME")],
        events=[],
        source_chunk_id=chunk.chunk_id,
    )
    mock_parser = MagicMock()
    mock_parser.parse = AsyncMock(return_value=[chunk])
    mock_extractor = MagicMock()
    mock_extractor.extract = AsyncMock(return_value=[extraction])
    mock_kg = MagicMock()
    mock_kg.upsert_entity = AsyncMock()
    mock_kg.add_relation = AsyncMock()

    agent = KnowledgeUpdateAgent(
        doc_parser=mock_parser,
        knowledge_extractor=mock_extractor,
        vector_store=fake_vector_store,
        knowledge_graph=mock_kg,
    )

    result = await agent.process_change(DocumentChange(file_path="/x/a.txt", change_type=ChangeType.CREATED))

    assert result.success is True
    assert mock_kg.upsert_entity.await_args.kwargs["source"] == "/x/a.txt"
    assert mock_kg.add_relation.await_args.kwargs["source"] == "/x/a.txt"


async def test_handle_modify_deletes_then_creates(fake_vector_store, fake_knowledge_graph):
    chunk = make_chunk("新内容", doc_id="d1")
    mock_parser = MagicMock()
    mock_parser.parse = AsyncMock(return_value=[chunk])
    mock_extractor = MagicMock()
    mock_extractor.extract = AsyncMock(return_value=[_make_extraction()])

    agent = KnowledgeUpdateAgent(
        doc_parser=mock_parser,
        knowledge_extractor=mock_extractor,
        vector_store=fake_vector_store,
        knowledge_graph=fake_knowledge_graph,
    )

    change = DocumentChange(file_path="/x/a.txt", change_type=ChangeType.MODIFIED)
    result = await agent.process_change(change)

    assert result.success is True
    assert len(fake_vector_store.delete_calls) == 1
    assert result.vectors_added == 1
    assert result.vectors_deleted == 0


async def test_handle_modify_with_minor_diff_updates_affected_chunks():
    before = "\n".join(f"line{i}" for i in range(20))
    after = before + "\n新增知识"
    unchanged = make_chunk("line0\nline1", doc_id="d1", chunk_index=0, source="/x/a.txt")
    changed = make_chunk("新增知识", doc_id="d1", chunk_index=1, source="/x/a.txt")
    extraction = ExtractionResult(
        entities=[Entity(name="新增知识", type="Concept")],
        relations=[],
        events=[],
        source_chunk_id=changed.chunk_id,
    )
    mock_parser = MagicMock()
    mock_parser.parse = AsyncMock(return_value=[unchanged, changed])
    mock_extractor = MagicMock()
    mock_extractor.extract = AsyncMock(return_value=[extraction])
    mock_vs = MagicMock()
    mock_vs.delete_by_doc_id = AsyncMock()
    mock_vs.delete_chunks = AsyncMock(return_value=1)
    mock_vs.add_chunks = AsyncMock(return_value=1)
    mock_kg = MagicMock()
    mock_kg.upsert_entity = AsyncMock()
    mock_kg.add_relation = AsyncMock()

    agent = KnowledgeUpdateAgent(
        doc_parser=mock_parser,
        knowledge_extractor=mock_extractor,
        vector_store=mock_vs,
        knowledge_graph=mock_kg,
    )

    change = DocumentChange(
        file_path="/x/a.txt",
        change_type=ChangeType.MODIFIED,
        before_content=before,
        after_content=after,
    )
    result = await agent.process_change(change)

    assert result.success is True
    assert result.diff["is_major_change"] is False
    mock_vs.delete_by_doc_id.assert_not_awaited()
    mock_vs.delete_chunks.assert_awaited_once_with([changed.chunk_id])
    mock_vs.add_chunks.assert_awaited_once_with([changed])
    mock_extractor.extract.assert_awaited_once_with([changed])
    assert result.vectors_deleted == 1
    assert result.vectors_added == 1
    assert result.entities_added == 1


async def test_handle_modify_with_major_diff_deletes_then_creates():
    before = "a\nb\nc"
    after = "x\ny\nz\nw"
    chunk = make_chunk("新内容", doc_id="d1", source="/x/a.txt")
    mock_parser = MagicMock()
    mock_parser.parse = AsyncMock(return_value=[chunk])
    mock_extractor = MagicMock()
    mock_extractor.extract = AsyncMock(return_value=[])
    mock_vs = MagicMock()
    mock_vs.delete_by_doc_id = AsyncMock(return_value=2)
    mock_vs.add_chunks = AsyncMock(return_value=1)
    mock_kg = MagicMock()
    mock_kg.upsert_entity = AsyncMock()
    mock_kg.add_relation = AsyncMock()

    agent = KnowledgeUpdateAgent(
        doc_parser=mock_parser,
        knowledge_extractor=mock_extractor,
        vector_store=mock_vs,
        knowledge_graph=mock_kg,
    )

    result = await agent.process_change(DocumentChange(
        file_path="/x/a.txt",
        change_type=ChangeType.MODIFIED,
        before_content=before,
        after_content=after,
    ))

    assert result.success is True
    assert result.diff["is_major_change"] is True
    mock_vs.delete_by_doc_id.assert_awaited_once()
    mock_vs.add_chunks.assert_awaited_once_with([chunk])
    assert result.vectors_deleted == 2
    assert result.vectors_added == 1


async def test_handle_delete_removes_from_stores(fake_vector_store, fake_knowledge_graph):
    agent = KnowledgeUpdateAgent(
        vector_store=fake_vector_store,
        knowledge_graph=fake_knowledge_graph,
    )
    change = DocumentChange(file_path="/x/a.txt", change_type=ChangeType.DELETED)
    result = await agent.process_change(change)

    assert result.success is True
    assert len(fake_vector_store.delete_calls) == 1
    assert "/x/a.txt" in fake_knowledge_graph.deleted_sources


async def test_process_change_records_processing_time():
    agent = KnowledgeUpdateAgent()
    change = DocumentChange(file_path="/x/none.txt", change_type=ChangeType.DELETED)
    result = await agent.process_change(change)
    assert result.processing_time_ms >= 0


async def test_process_change_failure_records_error():
    mock_parser = MagicMock()
    mock_parser.parse = AsyncMock(side_effect=RuntimeError("解析炸了"))
    agent = KnowledgeUpdateAgent(doc_parser=mock_parser)

    change = DocumentChange(file_path="/x/a.txt", change_type=ChangeType.CREATED)
    result = await agent.process_change(change)

    assert result.success is False
    assert "解析炸了" in result.error


async def test_process_batch_returns_one_result_per_change(fake_vector_store, fake_knowledge_graph):
    agent = KnowledgeUpdateAgent(
        vector_store=fake_vector_store,
        knowledge_graph=fake_knowledge_graph,
    )
    changes = [
        DocumentChange(file_path="/x/a.txt", change_type=ChangeType.DELETED),
        DocumentChange(file_path="/x/b.txt", change_type=ChangeType.DELETED),
    ]
    results = await agent.process_batch(changes)
    assert len(results) == 2
    assert all(r.success for r in results)


def test_bump_version_increments_per_entity():
    agent = KnowledgeUpdateAgent()
    assert agent._bump_version("张三") == 1
    assert agent._bump_version("张三") == 2
    assert agent._bump_version("李四") == 1
