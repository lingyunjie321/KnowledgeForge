"""编排引擎单测：三条流水线构建、QA 流水线运行"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.knowledge_extract_agent import Entity, ExtractionResult, Relation
from agents.qa_agent import QAResult, QueryIntent
from orchestrator.graph import _build_ingest_graph, _build_qa_graph, build_knowledge_graph_workflow
from tests.conftest import make_chunk


def test_build_workflow_returns_three_graphs(mock_llm, fake_vector_store, fake_knowledge_graph):
    workflows = build_knowledge_graph_workflow(
        vector_store=fake_vector_store,
        knowledge_graph=fake_knowledge_graph,
    )
    assert set(workflows.keys()) == {"ingest", "qa", "update", "qa_agent"}
    for name in ("ingest", "qa", "update"):
        assert workflows[name] is not None, f"{name} 流水线未构建"
    assert workflows["qa_agent"] is not None


async def test_qa_workflow_runs_with_mock_agent():
    mock_qa = MagicMock()
    expected = QAResult(
        question="测试问题",
        answer="测试答案",
        contexts=[],
        intent=QueryIntent.FACTOID,
        confidence=0.0,
    )
    mock_qa.answer = AsyncMock(return_value=expected)

    graph = _build_qa_graph(mock_qa)
    result = await graph.ainvoke({"question": "测试问题"})

    assert result["result"] is expected
    mock_qa.answer.assert_awaited_once_with("测试问题")


async def test_ingest_workflow_runs_end_to_end(mock_llm, fake_llm_response, tmp_path):
    # 不接向量库和图谱（传 None），只验证 parse → extract 数据流
    fake_llm_response('{"entities": [{"name": "知识图谱", "type": "Concept"}], "relations": [], "events": []}')
    mock_llm.ainvoke.return_value = fake_llm_response('{"entities": [{"name": "知识图谱", "type": "Concept"}], "relations": [], "events": []}')

    f = tmp_path / "doc.txt"
    f.write_text("知识图谱是结构化知识的表示方式，Neo4j 是常用图数据库。", encoding="utf-8")

    workflows = build_knowledge_graph_workflow(vector_store=None, knowledge_graph=None)
    ingest = workflows["ingest"]

    result = await ingest.ainvoke({"file_paths": [str(f)]})

    chunks = result.get("chunks", [])
    assert len(chunks) >= 1
    assert "知识图谱" in chunks[0].content
    extractions = result.get("extractions", [])
    assert len(extractions) >= 1
    # 无 vector_store/kg，store 步骤产出 0
    assert result.get("vectors_stored", 0) == 0
    assert result.get("entities_stored", 0) == 0


async def test_ingest_workflow_writes_graph_source_from_chunk_metadata():
    chunk = make_chunk("张三在 ACME 工作", doc_id="doc1", source="/docs/a.md")
    extraction = ExtractionResult(
        entities=[Entity(name="张三", type="Person")],
        relations=[Relation(head="张三", relation="works_at", tail="ACME")],
        events=[],
        source_chunk_id=chunk.chunk_id,
    )
    mock_parser = MagicMock()
    mock_parser.parse_batch = AsyncMock(return_value=[chunk])
    mock_extractor = MagicMock()
    mock_extractor.extract = AsyncMock(return_value=[extraction])
    mock_kg = MagicMock()
    mock_kg.upsert_entity = AsyncMock()
    mock_kg.add_relation = AsyncMock()

    graph = _build_ingest_graph(mock_parser, mock_extractor, vector_store=None, knowledge_graph=mock_kg)
    await graph.ainvoke({"file_paths": ["/docs/a.md"]})

    assert mock_kg.upsert_entity.await_args.kwargs["source"] == "/docs/a.md"
    assert mock_kg.add_relation.await_args.kwargs["source"] == "/docs/a.md"


async def test_update_workflow_processes_changes():
    # update 流水线内部 new KnowledgeUpdateAgent，这里直接验证编译后的图能跑
    workflows = build_knowledge_graph_workflow(vector_store=None, knowledge_graph=None)
    update = workflows["update"]

    from agents.knowledge_update_agent import ChangeType, DocumentChange
    change = DocumentChange(file_path="/nonexistent.txt", change_type=ChangeType.DELETED)
    result = await update.ainvoke({"changes": [change]})

    results = result.get("results", [])
    assert len(results) == 1
    # 删除空文件不报错
    assert results[0].success is True
