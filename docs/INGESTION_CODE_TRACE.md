# 文档入库代码追踪文档

> 面向要改代码的人。以 `上传 report.txt` 为例，逐函数追踪从 HTTP 到落库的完整调用链，每行关键代码都标位置。

## 1. 场景设定

用户通过前端上传 `report.txt`，内容：
```
张三在 ACME 公司工作，担任高级工程师。
ACME 公司总部位于北京，主要研发知识图谱产品。
李四也在 ACME 公司，和张三是同事。
```

追踪目标：从 HTTP 请求到 ChromaDB 和 Neo4j 各写入什么。

## 2. 调用链总览

```
[HTTP] POST /api/ingest/upload                          api/main.py:113
  └─ upload_document()                                  api/main.py:114
       ├─ DocParserAgent.is_supported_file()            agents/doc_parser_agent.py:96
       ├─ shutil.copyfileobj()                          api/main.py:123
       └─ ingest_wf.ainvoke({"file_paths": [...]})      api/main.py:130
            │
            ▼ LangGraph ingest 流水线
            ┌─────────────────────────────────────────┐
            │ parse_documents()                        orchestrator/graph.py:62
            │   └─ DocParserAgent.parse_batch()        agents/doc_parser_agent.py:86
            │        └─ DocParserAgent.parse()         agents/doc_parser_agent.py:66
            │             ├─ _classify()               agents/doc_parser_agent.py:92
            │             ├─ _make_doc_id()            agents/doc_parser_agent.py:101
            │             ├─ _parse_text()             agents/doc_parser_agent.py:221
            │             └─ _chunk_texts()            agents/doc_parser_agent.py:226
            │                                          state["chunks"] = [DocumentChunk×N]
            ├─────────────────────────────────────────┤
            │ extract_knowledge()                      orchestrator/graph.py:67
            │   └─ KnowledgeExtractAgent.extract()     agents/knowledge_extract_agent.py:86
            │        ├─ _extract_from_chunk()          agents/knowledge_extract_agent.py:99
            │        │    └─ _extract_from_text()      agents/knowledge_extract_agent.py:102
            │        │         └─ llm.ainvoke()        agents/knowledge_extract_agent.py:107
            │        ├─ _parse_response()              agents/knowledge_extract_agent.py:110
            │        └─ _deduplicate()                 agents/knowledge_extract_agent.py:154
            │                                          state["extractions"] = [ExtractionResult×N]
            ├─────────────────────────────────────────┤
            │ store_vectors()                          orchestrator/graph.py:72
            │   └─ VectorStoreService.add_chunks()     services/vector_store.py:76
            │        ├─ embeddings.embed_documents()   services/vector_store.py:94
            │        └─ _store.add()                   services/vector_store.py:95
            │                                          state["vectors_stored"] = N
            ├─────────────────────────────────────────┤
            │ store_graph()                            orchestrator/graph.py:83
            │   ├─ KnowledgeGraphService.upsert_entity()  services/knowledge_graph.py:53
            │   └─ KnowledgeGraphService.add_relation()   services/knowledge_graph.py:81
            │                                          state["entities_stored"] = N
            └─────────────────────────────────────────┘
            │
            ▼
[返回] IngestResponse                                  api/main.py:136
```

## 3. 逐函数追踪

### 3.1 HTTP 入口

**文件**：`api/main.py`

```python
# 第 113 行
@app.post("/api/ingest/upload", response_model=IngestResponse, tags=["文档入库"])
async def upload_document(file: UploadFile = File(...)):
    # 第 115 行：取文件名
    file_name = file.filename or "unknown"

    # 第 116-118 行：校验扩展名
    if not DocParserAgent.is_supported_file(file_name):
        supported = ", ".join(sorted(DocParserAgent.SUPPORTED_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"不支持的文件类型，当前支持: {supported}")

    # 第 121-124 行：防路径穿越 + 落盘
    safe_name = os.path.basename(file_name)  # "report.txt"
    save_path = os.path.join(settings.upload_dir, safe_name)  # "uploads/report.txt"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 第 126-128 行：取流水线
    ingest_wf = workflows.get("ingest")
    if not ingest_wf:
        raise HTTPException(status_code=503, detail="Ingest workflow not initialized")

    # 第 130 行：触发流水线
    result = await ingest_wf.ainvoke({"file_paths": [save_path]})

    # 第 131-142 行：组装响应
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
```

**此时的 state**：`{"file_paths": ["uploads/report.txt"]}`

### 3.2 parse 节点

**文件**：`orchestrator/graph.py:62`

```python
async def parse_documents(state: dict) -> dict:
    file_paths = state.get("file_paths", [])  # ["uploads/report.txt"]
    chunks = await doc_parser.parse_batch(file_paths)
    return {**state, "chunks": chunks}
```

进入 `DocParserAgent.parse_batch`（`agents/doc_parser_agent.py:86`），逐文件调 `parse`。

**`parse` 函数**（`agents/doc_parser_agent.py:66`）：

```python
async def parse(self, file_path: str) -> list[DocumentChunk]:
    # 第 67 行：分类
    doc_type = self._classify(file_path)
    # "uploads/report.txt" → ext=".txt" → DocType.TEXT

    # 第 68-70 行：不支持就报错
    if doc_type == DocType.UNKNOWN:
        raise ValueError(...)

    # 第 71 行：生成 doc_id
    doc_id = self._make_doc_id(file_path)
    # hashlib.sha256("uploads/report.txt".encode()).hexdigest()[:16]
    # = "a3f8b2c1d4e5f6a7"（示例）

    # 第 80-81 行：TEXT 走 _parse_text
    raw_texts = self._parse_text(file_path)
    # 返回 [整个文件内容字符串]

    # 第 83 行：分块
    chunks = self._chunk_texts(raw_texts, doc_id, doc_type, file_path)
    return chunks
```

**`_parse_text`**（`agents/doc_parser_agent.py:221`）：
```python
@staticmethod
def _parse_text(file_path: str) -> list[str]:
    with open(file_path, encoding="utf-8") as f:
        return [f.read()]
# 返回 ["张三在 ACME 公司工作，担任高级工程师。\nACME 公司总部位于北京..."]
```

**`_chunk_texts`**（`agents/doc_parser_agent.py:226`）：

滑动窗口分块，`CHUNK_SIZE=512`，`CHUNK_OVERLAP=64`。

示例文本约 80 字符，不足 512，所以只产生 1 个 chunk：

```python
chunks = [
    DocumentChunk(
        content="张三在 ACME 公司工作，担任高级工程师。\nACME 公司总部位于北京，主要研发知识图谱产品。\n李四也在 ACME 公司，和张三是同事。",
        doc_id="a3f8b2c1d4e5f6a7",
        chunk_index=0,
        doc_type=DocType.TEXT,
        metadata={
            "source": "uploads/report.txt",
            "char_start": 0,
            "char_end": 512
        },
        embedding=None
    )
]
```

**此时的 state**：
```python
{
    "file_paths": ["uploads/report.txt"],
    "chunks": [DocumentChunk(...)]  # 1 个 chunk
}
```

### 3.3 extract 节点

**文件**：`orchestrator/graph.py:67`

```python
async def extract_knowledge(state: dict) -> dict:
    chunks = state.get("chunks", [])
    extractions = await extractor.extract(chunks)
    return {**state, "extractions": extractions}
```

进入 `KnowledgeExtractAgent.extract`（`agents/knowledge_extract_agent.py:86`）：

```python
async def extract(self, chunks: list[DocumentChunk]) -> list[ExtractionResult]:
    results: list[ExtractionResult] = []
    for i in range(0, len(chunks), self.BATCH_SIZE):  # BATCH_SIZE=5
        batch = chunks[i : i + self.BATCH_SIZE]
        for chunk in batch:
            result = await self._extract_from_chunk(chunk)
            results.append(result)
    merged = self._deduplicate(results)
    return merged
```

**`_extract_from_chunk`**（第 99 行）→ **`_extract_from_text`**（第 102 行）：

```python
async def _extract_from_text(self, text: str, source_id: str) -> ExtractionResult:
    messages = [
        SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),  # 第 18-34 行的 prompt
        HumanMessage(content=f"请从以下文本中抽取知识：\n\n{text}"),
    ]
    resp = await self.llm.ainvoke(messages)
    return self._parse_response(resp.content, source_id)
```

LLM 返回（示例）：
```json
{
  "entities": [
    {"name": "张三", "type": "Person", "description": "ACME 公司高级工程师"},
    {"name": "ACME", "type": "Organization", "description": "研发知识图谱产品的公司"},
    {"name": "北京", "type": "Location", "description": "ACME 总部所在地"},
    {"name": "李四", "type": "Person", "description": "张三的同事"}
  ],
  "relations": [
    {"head": "张三", "relation": "WORKS_AT", "tail": "ACME", "confidence": 0.95},
    {"head": "ACME", "relation": "LOCATED_IN", "tail": "北京", "confidence": 0.9},
    {"head": "李四", "relation": "WORKS_AT", "tail": "ACME", "confidence": 0.9}
  ],
  "events": []
}
```

**`_parse_response`**（第 110 行）：解析 JSON，构造 `ExtractionResult`：

```python
ExtractionResult(
    entities=[
        Entity(name="张三", type="Person", description="ACME 公司高级工程师"),
        Entity(name="ACME", type="Organization", description="研发知识图谱产品的公司"),
        Entity(name="北京", type="Location", description="ACME 总部所在地"),
        Entity(name="李四", type="Person", description="张三的同事"),
    ],
    relations=[
        Relation(head="张三", relation="WORKS_AT", tail="ACME", confidence=0.95),
        Relation(head="ACME", relation="LOCATED_IN", tail="北京", confidence=0.9),
        Relation(head="李四", relation="WORKS_AT", tail="ACME", confidence=0.9),
    ],
    events=[],
    source_chunk_id="a3f8b2c1d4e5f6a7#chunk-0"
)
```

**`_deduplicate`**（第 154 行）：本例只有 1 个 chunk，无重复可去。

**此时的 state**：
```python
{
    "file_paths": ["uploads/report.txt"],
    "chunks": [DocumentChunk(...)],
    "extractions": [ExtractionResult(...)]  # 4 实体 3 关系
}
```

### 3.4 store_vectors 节点

**文件**：`orchestrator/graph.py:72`

```python
async def store_vectors(state: dict) -> dict:
    chunks = state.get("chunks", [])
    count = 0
    if vector_store and chunks:
        try:
            count = await vector_store.add_chunks(chunks)
        except Exception:
            logger.exception("向量入库失败，%d 个 chunk 丢失", len(chunks))
    return {**state, "vectors_stored": count}
```

进入 `VectorStoreService.add_chunks`（`services/vector_store.py:76`）：

```python
async def add_chunks(self, chunks: list[DocumentChunk]) -> int:
    # 第 77 行：embedding 不可用直接返回 0
    if not chunks or not self.embeddings_available:
        return 0

    # 第 80-89 行：准备三个并行列表
    texts = [c.content for c in chunks]
    # ["张三在 ACME 公司工作..."]

    metadatas = [
        {
            "doc_id": c.doc_id,         # "a3f8b2c1d4e5f6a7"
            "chunk_id": c.chunk_id,     # "a3f8b2c1d4e5f6a7#chunk-0"
            "source": c.metadata.get("source", ""),  # "uploads/report.txt"
            "doc_type": c.doc_type.value,  # "text"
        }
        for c in chunks
    ]

    ids = [c.chunk_id for c in chunks]
    # ["a3f8b2c1d4e5f6a7#chunk-0"]

    # 第 92-102 行：ChromaDB 后端
    if self._backend == "chroma":
        # 算向量（线程池）
        vectors = await self._run_sync(self.embeddings.embed_documents, texts)
        # vectors = [[0.12, -0.34, ...], ...]  (api 档 1024 维，disabled 档 8 维)

        # 写入（线程池）
        await self._run_sync(
            self._store.add,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )
        return len(chunks)  # 1
```

**写入 ChromaDB 的实际数据**：
```
collection: knowledge_chunks
record:
  id:        "a3f8b2c1d4e5f6a7#chunk-0"
  document:  "张三在 ACME 公司工作，担任高级工程师。\nACME 公司总部位于北京..."
  embedding: [0.12, -0.34, ...]  (维度取决于 embedding 档位)
  metadata:  {
    "doc_id":   "a3f8b2c1d4e5f6a7",
    "chunk_id": "a3f8b2c1d4e5f6a7#chunk-0",
    "source":   "uploads/report.txt",
    "doc_type": "text"
  }
```

**此时的 state**：
```python
{
    "file_paths": ["uploads/report.txt"],
    "chunks": [DocumentChunk(...)],
    "extractions": [ExtractionResult(...)],
    "vectors_stored": 1
}
```

### 3.5 store_graph 节点

**文件**：`orchestrator/graph.py:83`

```python
async def store_graph(state: dict) -> dict:
    extractions = state.get("extractions", [])
    entity_count = 0
    if knowledge_graph:
        try:
            for ext in extractions:
                for ent in ext.entities:
                    await knowledge_graph.upsert_entity(ent)
                    entity_count += 1
                for rel in ext.relations:
                    await knowledge_graph.add_relation(rel)
        except Exception:
            logger.exception("图谱写入失败，部分实体/关系丢失")
    return {**state, "entities_stored": entity_count}
```

#### 3.5.1 upsert_entity（写实体）

**文件**：`services/knowledge_graph.py:53`

对每个 Entity 执行：

```cypher
MERGE (e:Entity {name: $name})
ON CREATE SET
    e.type = $type,
    e.description = $description,
    e.version = $version,
    e.source = $source,
    e.created_at = $now,
    e.updated_at = $now
ON MATCH SET
    e.description = CASE WHEN $description <> '' THEN $description ELSE e.description END,
    e.version = $version,
    e.updated_at = $now
```

参数：
```python
{
    "name": "张三",
    "type": "Person",
    "description": "ACME 公司高级工程师",
    "version": 1,
    "source": "",
    "now": 1719253935
}
```

写入 Neo4j 的节点：
```
(:Entity {
  name: "张三",
  type: "Person",
  description: "ACME 公司高级工程师",
  version: 1,
  source: "",
  created_at: 1719253935,
  updated_at: 1719253935
})
```

4 个实体各执行一次，共 4 个节点。

#### 3.5.2 add_relation（写关系）

**文件**：`services/knowledge_graph.py:81`

```python
async def add_relation(self, relation: Relation, source: str = "") -> None:
    if not self._driver:
        return
    # 第 84 行：标准化关系类型
    rel_type = relation.relation.upper().replace(" ", "_")
    # "WORKS_AT" → "WORKS_AT"（已是大写）

    # 第 85-87 行：白名单校验
    if rel_type not in _ALLOWED_REL_TYPES:
        logger.warning("关系类型不在白名单，已拒绝: %s", rel_type)
        return

    # 第 88-93 行：参数化 Cypher
    cypher = f"""
    MATCH (h:Entity {{name: $head}})
    MATCH (t:Entity {{name: $tail}})
    MERGE (h)-[r:{rel_type}]->(t)
    SET r.confidence = $confidence, r.source = $source, r.updated_at = $now
    """
    # 注意：rel_type 是从白名单取的，安全拼进 Cypher
```

对 `Relation(head="张三", relation="WORKS_AT", tail="ACME", confidence=0.95)`：

```cypher
MATCH (h:Entity {name: $head})   # $head = "张三"
MATCH (t:Entity {name: $tail})   # $tail = "ACME"
MERGE (h)-[r:WORKS_AT]->(t)
SET r.confidence = 0.95, r.source = "", r.updated_at = 1719253935
```

写入 Neo4j 的关系：
```
(张三)-[:WORKS_AT {confidence: 0.95, source: "", updated_at: 1719253935}]->(ACME)
```

3 个关系各执行一次。如果 LLM 抽出的关系类型不在 8 种白名单内（比如 "FRIEND_OF"），会被拒绝并记日志。

**此时的 state**：
```python
{
    "file_paths": ["uploads/report.txt"],
    "chunks": [DocumentChunk(...)],
    "extractions": [ExtractionResult(...)],
    "vectors_stored": 1,
    "entities_stored": 4
}
```

### 3.6 返回响应

**文件**：`api/main.py:130-142`

```python
result = await ingest_wf.ainvoke({"file_paths": [save_path]})
# result = {
#     "file_paths": ["uploads/report.txt"],
#     "chunks": [DocumentChunk(...)],
#     "extractions": [ExtractionResult(entities=[4个], relations=[3个])],
#     "vectors_stored": 1,
#     "entities_stored": 4
# }

chunks = result.get("chunks", [])           # 1 个
extractions = result.get("extractions", []) # 1 个
total_entities = sum(len(e.entities) for e in extractions)    # 4
total_relations = sum(len(e.relations) for e in extractions)  # 3

return IngestResponse(
    file_name="report.txt",
    chunks_count=1,
    entities_count=4,
    relations_count=3,
    status="success",
)
```

## 4. 最终落库数据汇总

### ChromaDB（collection: knowledge_chunks）
| id | document | embedding | metadata.doc_id | metadata.doc_type |
|----|----------|-----------|-----------------|-------------------|
| a3f8b2c1d4e5f6a7#chunk-0 | 张三在 ACME 公司工作... | [0.12, -0.34, ...] | a3f8b2c1d4e5f6a7 | text |

### Neo4j
**4 个节点**：
| name | type | description |
|------|------|-------------|
| 张三 | Person | ACME 公司高级工程师 |
| ACME | Organization | 研发知识图谱产品的公司 |
| 北京 | Location | ACME 总部所在地 |
| 李四 | Person | 张三的同事 |

**3 条关系**：
| head | relation | tail | confidence |
|------|----------|------|------------|
| 张三 | WORKS_AT | ACME | 0.95 |
| ACME | LOCATED_IN | 北京 | 0.9 |
| 李四 | WORKS_AT | ACME | 0.9 |

## 5. 关键代码位置速查

| 关注点 | 文件 | 行号 |
|--------|------|------|
| HTTP 入口 | api/main.py | 113 |
| 文件类型校验 | agents/doc_parser_agent.py | 96 |
| 防路径穿越 | api/main.py | 121 |
| doc_id 生成 | agents/doc_parser_agent.py | 101 |
| 分块窗口大小 | agents/doc_parser_agent.py | 55-56 |
| LLM 抽取 prompt | agents/knowledge_extract_agent.py | 18-34 |
| 跨 chunk 去重 | agents/knowledge_extract_agent.py | 154 |
| embedding 线程池隔离 | services/vector_store.py | 30, 32 |
| chromadb 写入 | services/vector_store.py | 92-102 |
| 实体 MERGE Cypher | services/knowledge_graph.py | 57-69 |
| 关系类型白名单 | services/knowledge_graph.py | 15 |
| 关系 MERGE Cypher | services/knowledge_graph.py | 88-93 |
| 流水线编排 | orchestrator/graph.py | 55-110 |
| 入库失败不静默 | orchestrator/graph.py | 79-80, 94-95 |

## 相关文档

- [项目整体文档](./PROJECT_OVERVIEW.md)
- [新手代码导读](./CODE_GUIDE.md)
- [文档入库流程](./INGESTION_FLOW.md)（本文是代码视角，流程文档是业务视角）
- [项目架构设计详解](./ARCHITECTURE.md)
- [项目文件树](./FILE_TREE.md)
