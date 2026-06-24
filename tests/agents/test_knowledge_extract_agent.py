"""KnowledgeExtractAgent 单测：响应解析、跨 chunk 去重、全流程"""

from __future__ import annotations

import pytest

from agents.doc_parser_agent import DocType, DocumentChunk
from agents.knowledge_extract_agent import (
    Entity,
    ExtractionResult,
    KnowledgeExtractAgent,
    Relation,
)


def _make_result(entities=None, relations=None, events=None, sid="c0"):
    return ExtractionResult(
        entities=entities or [],
        relations=relations or [],
        events=events or [],
        source_chunk_id=sid,
    )


def test_parse_response_valid_json(mock_llm):
    agent = KnowledgeExtractAgent()
    raw = '{"entities": [{"name": "张三", "type": "Person", "description": "工程师"}], "relations": [{"head": "张三", "relation": "works_at", "tail": "ACME", "confidence": 0.9}], "events": [{"trigger": "加入", "type": "人事", "participants": ["张三"]}]}'
    result = agent._parse_response(raw, "c1")

    assert len(result.entities) == 1
    assert result.entities[0].name == "张三"
    assert result.entities[0].type == "Person"
    assert len(result.relations) == 1
    assert result.relations[0].confidence == 0.9
    assert len(result.events) == 1
    assert result.events[0].participants == ["张三"]
    assert result.source_chunk_id == "c1"


def test_parse_response_markdown_wrapped(mock_llm):
    agent = KnowledgeExtractAgent()
    raw = '```json\n{"entities": [{"name": "X", "type": "Concept"}], "relations": [], "events": []}\n```'
    result = agent._parse_response(raw, "c2")
    assert len(result.entities) == 1
    assert result.entities[0].name == "X"


def test_parse_response_invalid_json_returns_empty(mock_llm):
    agent = KnowledgeExtractAgent()
    result = agent._parse_response("这不是 JSON", "c3")
    assert result.entities == []
    assert result.relations == []
    assert result.events == []
    assert result.source_chunk_id == "c3"


def test_parse_response_skips_entities_without_name(mock_llm):
    agent = KnowledgeExtractAgent()
    raw = '{"entities": [{"name": "", "type": "Person"}, {"name": "有效", "type": "Org"}], "relations": [{"head": "", "tail": "x", "relation": "r"}, {"head": "a", "tail": "b", "relation": "r"}], "events": []}'
    result = agent._parse_response(raw, "c4")
    assert len(result.entities) == 1
    assert result.entities[0].name == "有效"
    assert len(result.relations) == 1
    assert result.relations[0].head == "a"


def test_deduplicate_merges_same_name_type(mock_llm):
    agent = KnowledgeExtractAgent()
    r1 = _make_result(entities=[Entity(name="张三", type="Person"), Entity(name="Neo4j", type="Technology")], sid="c1")
    r2 = _make_result(entities=[Entity(name="张三", type="Person"), Entity(name="李四", type="Person")], sid="c2")

    merged = agent._deduplicate([r1, r2])
    assert len(merged[0].entities) == 2
    assert {e.name for e in merged[0].entities} == {"张三", "Neo4j"}
    assert len(merged[1].entities) == 1
    assert merged[1].entities[0].name == "李四"


def test_deduplicate_keeps_different_types(mock_llm):
    agent = KnowledgeExtractAgent()
    r1 = _make_result(entities=[Entity(name="Java", type="Technology")], sid="c1")
    r2 = _make_result(entities=[Entity(name="Java", type="Location")], sid="c2")
    merged = agent._deduplicate([r1, r2])
    assert len(merged[0].entities) == 1
    assert len(merged[1].entities) == 1


def test_deduplicate_relations_by_triple(mock_llm):
    agent = KnowledgeExtractAgent()
    rel = Relation(head="张三", relation="works_at", tail="ACME")
    r1 = _make_result(relations=[rel, Relation(head="张三", relation="lives_in", tail="北京")], sid="c1")
    r2 = _make_result(relations=[rel], sid="c2")
    merged = agent._deduplicate([r1, r2])
    assert len(merged[0].relations) == 2
    assert len(merged[1].relations) == 0


async def test_extract_calls_llm_and_dedupes(mock_llm, fake_llm_response):
    mock_llm.ainvoke.side_effect = [
        fake_llm_response('{"entities": [{"name": "张三", "type": "Person"}], "relations": [{"head": "张三", "relation": "works_at", "tail": "ACME"}], "events": []}'),
        fake_llm_response('{"entities": [{"name": "张三", "type": "Person"}], "relations": [], "events": []}'),
    ]
    agent = KnowledgeExtractAgent()
    chunks = [
        DocumentChunk(content="A", doc_id="d", chunk_index=0, doc_type=DocType.TEXT, metadata={}),
        DocumentChunk(content="B", doc_id="d", chunk_index=1, doc_type=DocType.TEXT, metadata={}),
    ]

    results = await agent.extract(chunks)

    assert len(results) == 2
    assert len(results[0].entities) == 1
    assert len(results[0].relations) == 1
    # 第二个 result 的张三被去重
    assert len(results[1].entities) == 0
    assert mock_llm.ainvoke.await_count == 2


async def test_extract_single_returns_parsed(mock_llm, fake_llm_response):
    mock_llm.ainvoke.return_value = fake_llm_response('{"entities": [{"name": "X", "type": "Concept"}], "relations": [], "events": []}')
    agent = KnowledgeExtractAgent()
    result = await agent.extract_single("text", "sid")
    assert len(result.entities) == 1
    assert result.source_chunk_id == "sid"
