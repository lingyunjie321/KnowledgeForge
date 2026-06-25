"""API 端到端 smoke test：health、stats、问答、上传鉴权"""

from __future__ import annotations

from fastapi.testclient import TestClient

from agents.qa_agent import QAResult, QueryIntent, RetrievedContext
from api.main import app, workflows


def test_health_returns_ok():
    with TestClient(app) as client:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["service"] == "knowledgeforge"


def test_stats_returns_structure():
    with TestClient(app) as client:
        resp = client.get("/api/admin/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert "vector_store" in body
        assert "knowledge_graph" in body
        # 降级时也应有默认字段
        assert "backend" in body["vector_store"] or "total_vectors" in body["vector_store"]


def test_ask_question_returns_answer(mock_llm, fake_llm_response):
    # qa_agent.answer 走 4 次 LLM：意图、改写、实体链接、答案生成
    mock_llm.ainvoke.side_effect = [
        fake_llm_response("factoid"),
        fake_llm_response('{"queries": ["q"], "entities": [], "keywords": []}'),
        fake_llm_response('{"entities": []}'),
        fake_llm_response("空知识库兜底回答"),
    ]
    with TestClient(app) as client:
        resp = client.post("/api/qa/ask", json={"question": "测试问题"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["question"] == "测试问题"
        assert body["answer"] == "空知识库兜底回答"
        assert body["intent"] == "factoid"
        assert isinstance(body["sources"], list)
        assert isinstance(body["reasoning_steps"], list)


def test_ask_question_returns_source_metadata():
    class FakeQAWorkflow:
        async def ainvoke(self, state):
            return {
                "result": QAResult(
                    question=state["question"],
                    answer="答案",
                    contexts=[
                        RetrievedContext(
                            content="命中内容",
                            source="/docs/a.md",
                            score=0.9,
                            retrieval_type="vector",
                            metadata={"doc_id": "doc1", "chunk_id": "doc1#chunk-0"},
                        ),
                    ],
                    intent=QueryIntent.FACTOID,
                    confidence=0.9,
                    reasoning_steps=["答案生成完成"],
                )
            }

    with TestClient(app) as client:
        old_workflow = workflows.get("qa")
        workflows["qa"] = FakeQAWorkflow()
        try:
            resp = client.post("/api/qa/ask", json={"question": "测试问题"})
        finally:
            if old_workflow is None:
                workflows.pop("qa", None)
            else:
                workflows["qa"] = old_workflow

    assert resp.status_code == 200
    source = resp.json()["sources"][0]
    assert source["type"] == "vector"
    assert source["source"] == "/docs/a.md"
    assert source["metadata"]["chunk_id"] == "doc1#chunk-0"


def test_ask_rejects_missing_question_field():
    with TestClient(app) as client:
        resp = client.post("/api/qa/ask", json={})
        assert resp.status_code == 422


def test_upload_unsupported_extension_returns_400():
    with TestClient(app) as client:
        resp = client.post(
            "/api/ingest/upload",
            files={"file": ("doc.docx", b"placeholder", "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert "不支持" in resp.json()["detail"]


def test_root_serves_frontend():
    with TestClient(app) as client:
        resp = client.get("/")
        assert resp.status_code == 200


def test_graph_data_returns_available_flag():
    with TestClient(app) as client:
        resp = client.get("/api/graph/data")
        assert resp.status_code == 200
        body = resp.json()
        assert "entities" in body
        assert "relations" in body
        assert "available" in body
        # 测试环境无 Neo4j，应降级返回 available=false
        assert body["available"] is False
        assert body["entities"] == []


def test_ask_stream_returns_sse(mock_llm, fake_llm_response):
    mock_llm.ainvoke.side_effect = [
        fake_llm_response("factoid"),
        fake_llm_response('{"queries": ["q"], "entities": [], "keywords": []}'),
        fake_llm_response('{"entities": []}'),
    ]
    mock_llm.astream_text = "流式答案文本"
    with TestClient(app) as client:
        resp = client.post("/api/qa/ask_stream", json={"question": "测试"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        body = resp.text
        assert "event: meta" in body
        assert "event: token" in body
        assert "流式答案文本" in body
        assert "event: done" in body
