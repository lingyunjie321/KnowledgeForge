"""API 端到端 smoke test：health、stats、问答、上传鉴权"""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


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
