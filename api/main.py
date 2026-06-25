"""FastAPI 入口"""

from __future__ import annotations

import json
import logging
import os
import shutil
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents.doc_parser_agent import DocParserAgent
from agents.knowledge_extract_agent import KnowledgeExtractAgent
from agents.knowledge_update_agent import ChangeType, DocumentChange, KnowledgeUpdateAgent
from agents.qa_agent import QAAgent
from config import settings
from orchestrator.graph import build_knowledge_graph_workflow
from services.knowledge_graph import KnowledgeGraphService
from services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

vector_store = VectorStoreService()
knowledge_graph = KnowledgeGraphService()
workflows: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(settings.upload_dir, exist_ok=True)

    # 向量库初始化失败应该让进程挂掉，而不是带病运行
    try:
        await vector_store.init()
    except Exception:
        logger.exception("向量库初始化失败")
        raise

    # Neo4j 连不上不致命，图谱检索会降级，但要有明确日志
    try:
        await knowledge_graph.init()
    except Exception:
        logger.exception("知识图谱初始化失败，图谱检索将降级")

    workflows.update(
        build_knowledge_graph_workflow(vector_store=vector_store, knowledge_graph=knowledge_graph)
    )
    yield
    await knowledge_graph.close()


app = FastAPI(
    title="KnowledgeForge — 企业知识库管理系统",
    version="1.0.0",
    lifespan=lifespan,
)

static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(static_dir, "index.html"))


class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    intent: str
    sources: list[dict[str, Any]]
    reasoning_steps: list[str]


class IngestResponse(BaseModel):
    file_name: str
    chunks_count: int
    entities_count: int
    relations_count: int
    status: str


class StatsResponse(BaseModel):
    vector_store: dict[str, Any]
    knowledge_graph: dict[str, Any]


class UpdateRequest(BaseModel):
    file_path: str
    change_type: str = "modified"


class UpdateResponse(BaseModel):
    file_path: str
    vectors_added: int
    vectors_deleted: int
    entities_added: int
    relations_added: int
    success: bool
    processing_time_ms: float


@app.post("/api/ingest/upload", response_model=IngestResponse, tags=["文档入库"])
async def upload_document(file: UploadFile = File(...)):
    file_name = file.filename or "unknown"
    if not DocParserAgent.is_supported_file(file_name):
        supported = ", ".join(sorted(DocParserAgent.SUPPORTED_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"不支持的文件类型，当前支持: {supported}")

    # 防路径穿越：只取文件名，不带目录
    safe_name = os.path.basename(file_name)
    save_path = os.path.join(settings.upload_dir, safe_name)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    ingest_wf = workflows.get("ingest")
    if not ingest_wf:
        raise HTTPException(status_code=503, detail="Ingest workflow not initialized")

    result = await ingest_wf.ainvoke({"file_paths": [save_path]})
    chunks = result.get("chunks", [])
    extractions = result.get("extractions", [])
    total_entities = sum(len(e.entities) for e in extractions)
    total_relations = sum(len(e.relations) for e in extractions)

    return IngestResponse(
        file_name=safe_name,
        chunks_count=len(chunks),
        entities_count=total_entities,
        relations_count=total_relations,
        status="success",
    )


@app.post("/api/ingest/batch", response_model=list[IngestResponse], tags=["文档入库"])
async def upload_batch(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        resp = await upload_document(file)
        results.append(resp)
    return results


@app.post("/api/qa/ask", response_model=QuestionResponse, tags=["智能问答"])
async def ask_question(req: QuestionRequest):
    qa_wf = workflows.get("qa")
    if not qa_wf:
        raise HTTPException(status_code=503, detail="QA workflow not initialized")

    result = await qa_wf.ainvoke({"question": req.question})
    qa_result = result.get("result")
    if not qa_result:
        raise HTTPException(status_code=500, detail="QA failed")

    return QuestionResponse(
        question=qa_result.question,
        answer=qa_result.answer,
        confidence=qa_result.confidence,
        intent=qa_result.intent.value,
        sources=[
            {"content": c.content[:200], "source": c.source, "score": c.score, "type": c.retrieval_type}
            for c in qa_result.contexts
        ],
        reasoning_steps=qa_result.reasoning_steps,
    )


@app.post("/api/qa/ask_stream", tags=["智能问答"])
async def ask_question_stream(req: QuestionRequest):
    """SSE 流式问答：meta 事件推意图/来源/检索步，token 事件逐字推答案，done 事件收尾"""
    qa_agent = workflows.get("qa_agent")
    if qa_agent is None:
        raise HTTPException(status_code=503, detail="QA agent not initialized")

    async def event_gen():
        try:
            async for evt in qa_agent.answer_stream(req.question):
                yield f"event: {evt['type']}\ndata: {json.dumps(evt, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.exception("SSE 流式问答失败")
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/admin/stats", response_model=StatsResponse, tags=["系统管理"])
async def get_stats():
    try:
        vs_stats = await vector_store.get_stats()
    except Exception:
        logger.exception("获取向量库统计失败")
        vs_stats = {"backend": "chroma", "total_vectors": 0}
    try:
        kg_stats = await knowledge_graph.get_stats()
    except Exception:
        logger.exception("获取图谱统计失败")
        kg_stats = {"total_entities": 0, "total_relations": 0}
    return StatsResponse(vector_store=vs_stats, knowledge_graph=kg_stats)


class GraphDataResponse(BaseModel):
    entities: list[dict[str, Any]]
    relations: list[dict[str, Any]]
    available: bool


@app.get("/api/graph/data", response_model=GraphDataResponse, tags=["知识图谱"])
async def get_graph_data(limit: int = 200):
    """查实体+关系给前端图谱可视化，Neo4j 不可用时 available=False"""
    if not knowledge_graph.is_connected:
        return GraphDataResponse(entities=[], relations=[], available=False)
    try:
        entities = await knowledge_graph.list_entities(limit=limit)
        relations = await knowledge_graph.list_relations(limit=limit * 2)
        return GraphDataResponse(entities=entities, relations=relations, available=True)
    except Exception:
        logger.exception("查图谱数据失败")
        return GraphDataResponse(entities=[], relations=[], available=False)


@app.post("/api/admin/update", response_model=UpdateResponse, tags=["系统管理"])
async def trigger_update(req: UpdateRequest):
    update_wf = workflows.get("update")
    if not update_wf:
        raise HTTPException(status_code=503, detail="Update workflow not initialized")

    change = DocumentChange(
        file_path=req.file_path,
        change_type=ChangeType(req.change_type),
    )
    result = await update_wf.ainvoke({"changes": [change]})
    results = result.get("results", [])
    if not results:
        raise HTTPException(status_code=500, detail="Update failed")

    r = results[0]
    return UpdateResponse(
        file_path=r.change.file_path,
        vectors_added=r.vectors_added,
        vectors_deleted=r.vectors_deleted,
        entities_added=r.entities_added,
        relations_added=r.relations_added,
        success=r.success,
        processing_time_ms=r.processing_time_ms,
    )


@app.get("/api/health", tags=["系统管理"])
async def health():
    return {"status": "ok", "service": "knowledgeforge"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.api_host, port=settings.api_port, reload=True)

