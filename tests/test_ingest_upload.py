"""入库接口 smoke test"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from agents.doc_parser_agent import DocParserAgent
from api.main import app


def test_upload_rejects_unsupported_file_extension_before_workflow():
    client = TestClient(app)

    response = client.post(
        "/api/ingest/upload",
        files={
            "file": (
                "briefing.docx",
                b"PK\x03\x04binary-docx-placeholder",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )

    assert response.status_code == 400
    assert "不支持" in response.json()["detail"]


def test_doc_parser_rejects_unsupported_file_extension(tmp_path):
    file_path = tmp_path / "briefing.docx"
    file_path.write_bytes(b"PK\x03\x04binary-docx-placeholder")

    agent = DocParserAgent()

    with pytest.raises(ValueError, match="不支持"):
        asyncio.run(agent.parse(str(file_path)))
