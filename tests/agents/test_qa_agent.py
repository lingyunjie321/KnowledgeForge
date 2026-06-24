"""QAAgent 单测：意图分类、查询改写、置信度、全流程"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.qa_agent import QAAgent, QAResult, QueryIntent, RetrievedContext
from services.graph_rag import GraphRAGContext


async def test_classify_intent_parses_llm_response(mock_llm, fake_llm_response):
    mock_llm.ainvoke.return_value = fake_llm_response("analytical")
    agent = QAAgent()
    intent = await agent._classify_intent("为什么天是蓝的")
    assert intent is QueryIntent.ANALYTICAL


async def test_classify_intent_fallback_to_factoid_on_garbage(mock_llm, fake_llm_response):
    mock_llm.ainvoke.return_value = fake_llm_response("我不太明白你在说什么???")
    agent = QAAgent()
    intent = await agent._classify_intent("乱码")
    assert intent is QueryIntent.FACTOID


async def test_rewrite_query_parses_json(mock_llm, fake_llm_response):
    mock_llm.ainvoke.return_value = fake_llm_response('{"queries": ["q1", "q2"], "entities": ["e1"], "keywords": ["k1"]}')
    agent = QAAgent()
    result = await agent._rewrite_query("问题")
    assert result["queries"] == ["q1", "q2"]
    assert result["entities"] == ["e1"]


async def test_rewrite_query_fallback_on_invalid_json(mock_llm, fake_llm_response):
    mock_llm.ainvoke.return_value = fake_llm_response("不是 JSON")
    agent = QAAgent()
    result = await agent._rewrite_query("问题")
    assert result["queries"] == ["问题"]
    assert result["entities"] == []


def test_calc_confidence_empty_returns_zero():
    assert QAAgent._calc_confidence([]) == 0.0


def test_calc_confidence_averages_and_caps_at_one():
    ctxs = [RetrievedContext(content="a", source="s", score=0.6, retrieval_type="vector"),
            RetrievedContext(content="b", source="s", score=0.8, retrieval_type="vector")]
    assert QAAgent._calc_confidence(ctxs) == pytest.approx(0.7)
    high = [RetrievedContext(content="x", source="s", score=1.2, retrieval_type="vector")]
    assert QAAgent._calc_confidence(high) == 1.0


async def test_answer_full_flow_with_mock_graph_rag(mock_llm, fake_llm_response):
    mock_llm.ainvoke.side_effect = [
        fake_llm_response("factoid"),
        fake_llm_response('{"queries": ["q"], "entities": [], "keywords": []}'),
        fake_llm_response("最终答案文本"),
    ]
    mock_rag = MagicMock()
    mock_rag.retrieve = AsyncMock(return_value=[
        GraphRAGContext(content="向量命中", source_type="vector", score=0.9),
        GraphRAGContext(content="子图命中", source_type="subgraph", score=0.8),
        GraphRAGContext(content="路径命中", source_type="path", score=0.85),
        GraphRAGContext(content="社区摘要", source_type="community", score=0.7),
    ])

    agent = QAAgent(graph_rag=mock_rag)
    result = await agent.answer("张三在哪工作")

    assert isinstance(result, QAResult)
    assert result.intent is QueryIntent.FACTOID
    assert result.answer == "最终答案文本"
    assert len(result.contexts) == 4
    assert result.confidence > 0
    steps_text = " ".join(result.reasoning_steps)
    assert "vector" in steps_text and "subgraph" in steps_text
    assert mock_rag.retrieve.await_count == 1


async def test_answer_no_graph_rag_yields_empty_contexts(mock_llm, fake_llm_response):
    mock_llm.ainvoke.side_effect = [
        fake_llm_response("factoid"),
        fake_llm_response('{"queries": ["q"], "entities": [], "keywords": []}'),
        fake_llm_response("空库兜底答案"),
    ]
    agent = QAAgent()
    result = await agent.answer("任意问题")
    assert result.contexts == []
    assert result.confidence == 0.0
    assert result.answer == "空库兜底答案"


async def test_answer_truncates_to_top_eight(mock_llm, fake_llm_response):
    mock_llm.ainvoke.side_effect = [
        fake_llm_response("factoid"),
        fake_llm_response('{"queries": ["q"], "entities": [], "keywords": []}'),
        fake_llm_response("答案"),
    ]
    mock_rag = MagicMock()
    mock_rag.retrieve = AsyncMock(return_value=[
        GraphRAGContext(content=f"c{i}", source_type="vector", score=0.5) for i in range(15)
    ])
    agent = QAAgent(graph_rag=mock_rag)
    result = await agent.answer("q")
    assert len(result.contexts) == 8
