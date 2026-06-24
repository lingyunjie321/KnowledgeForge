# 文档入库流程文档

> 面向理解业务的人。从用户上传文件到知识可被检索，每一步做什么、数据怎么变、失败怎么办。

## 1. 流程总览

```
用户上传文件
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  HTTP 层：POST /api/ingest/upload                        │
│  - 校验文件类型（9 种扩展名白名单）                        │
│  - 防路径穿越（只取 basename）                            │
│  - 落盘到 uploads/                                       │
└─────────────────────────────────────────────────────────┘
    │
    ▼ file_paths: [save_path]
┌─────────────────────────────────────────────────────────┐
│  LangGraph ingest 流水线                                 │
│                                                         │
│  ┌─────────┐    ┌─────────┐    ┌──────────────┐    ┌──────────────┐
│  │  parse  │ →  │ extract │ →  │ store_vectors│ →  │ store_graph  │
│  └─────────┘    └─────────┘    └──────────────┘    └──────────────┘
│      │              │                │                    │
│      │              │                │                    │
│   文档解析       知识抽取         向量入库             图谱写入
│   ↓              ↓                ↓                    ↓
│   chunks        extractions     vectors_stored       entities_stored
└─────────────────────────────────────────────────────────┘
    │
    ▼
返回 IngestResponse
{
  "file_name": "xxx.pdf",
  "chunks_count": 12,
  "entities_count": 35,
  "relations_count": 18,
  "status": "success"
}
```

## 2. 各阶段详解

### 阶段 1：HTTP 接收与落盘

**入口**：`api/main.py:113` `upload_document`

**做的事**：
1. 从 `file.filename` 取文件名
2. `DocParserAgent.is_supported_file()` 校验扩展名，不支持直接 400
3. `os.path.basename()` 防路径穿越（去掉目录部分）
4. 拼接 `settings.upload_dir` 保存路径
5. `shutil.copyfileobj` 流式落盘
6. 从 `workflows["ingest"]` 取编译好的 LangGraph
7. `await ingest_wf.ainvoke({"file_paths": [save_path]})` 进入流水线

**数据形态**：原始文件 → 磁盘上的 `uploads/xxx.pdf`

**失败处理**：
- 文件类型不支持 → HTTP 400，不入流水线
- workflow 没初始化 → HTTP 503

### 阶段 2：文档解析（parse 节点）

**代码位置**：`orchestrator/graph.py:62` `parse_documents` → `agents/doc_parser_agent.py:66` `parse`

**做的事**：
1. `_classify(file_path)` 按扩展名分到 5 种 DocType 之一
2. 生成 doc_id（文件路径 SHA256 前 16 位）
3. 按 DocType 走不同解析路径：
   - **PDF**：PyPDF2 提文字 → 失败降级 pdf2image 转图片 → LLM 视觉描述
   - **IMAGE**：pytesseract OCR + LLM 视觉描述（双路拼接）
   - **TABLE**：CSV 用 csv 模块、Excel 用 openpyxl，按行展开成 `表头: 值 | 表头: 值` 格式
   - **TEXT/MARKDOWN**：直接 `open().read()`
4. `_chunk_texts` 滑动窗口分块：512 字符窗口，64 字符重叠
5. 每个 chunk 带 metadata：`{source, char_start, char_end}`

**数据形态变化**：
```
输入: file_paths = ["/app/uploads/report.pdf"]
输出: chunks = [
  DocumentChunk(
    content="第一段文本...",
    doc_id="a3f8b2c1d4e5f6a7",
    chunk_index=0,
    doc_type=DocType.PDF,
    metadata={"source": "/app/uploads/report.pdf", "char_start": 0, "char_end": 512}
  ),
  DocumentChunk(content="第二段...", chunk_index=1, ...),
  ...
]
```

**失败处理**：
- PDF 解析失败 → 文本变成 `[PDF 解析失败] xxx.pdf`，不中断流程
- 图片 OCR 失败 → 返回空字符串，仍有 LLM 描述
- 不支持的扩展名 → `parse()` 直接 raise ValueError

### 阶段 3：知识抽取（extract 节点）

**代码位置**：`orchestrator/graph.py:67` `extract_knowledge` → `agents/knowledge_extract_agent.py:86` `extract`

**做的事**：
1. 按 `BATCH_SIZE=5` 分批
2. 逐 chunk 调 LLM（DeepSeek），prompt 要求返回 JSON：
   ```json
   {
     "entities": [{"name": "...", "type": "...", "description": "..."}],
     "relations": [{"head": "...", "relation": "...", "tail": "...", "confidence": 0.95}],
     "events": [{"trigger": "...", "type": "...", "participants": ["..."]}]
   }
   ```
3. `_parse_response` 解析 JSON（去掉 ```json 包裹）
4. `_deduplicate` 跨 chunk 去重：
   - 实体按 `name::type` 去重
   - 关系按 `(head, relation, tail)` 三元组去重

**数据形态变化**：
```
输入: chunks = [DocumentChunk(...), ...]
输出: extractions = [
  ExtractionResult(
    entities=[Entity(name="张三", type="Person", description="..."), ...],
    relations=[Relation(head="张三", relation="WORKS_AT", tail="ACME", confidence=0.95), ...],
    events=[KnowledgeEvent(trigger="入职", type="Business", participants=["张三"]), ...],
    source_chunk_id="a3f8b2c1d4e5f6a7#chunk-0"
  ),
  ...
]
```

**失败处理**：
- LLM 返回不是合法 JSON → `_parse_response` 返回空 ExtractionResult，不中断
- LLM 调用失败 → 异常向上抛，被 LangGraph 捕获

### 阶段 4：向量入库（store_vectors 节点）

**代码位置**：`orchestrator/graph.py:72` `store_vectors` → `services/vector_store.py:76` `add_chunks`

**做的事**：
1. 检查 `embeddings_available`（embedding 初始化过没）
2. 把 chunks 转成 texts / metadatas / ids 三个并行列表
3. ChromaDB 后端：
   - `embeddings.embed_documents(texts)` 算向量（线程池执行）
   - `_store.add(embeddings=vectors, documents=texts, metadatas=metadatas, ids=ids)` 写入（线程池执行）
4. PGVector 后端：`aadd_texts` 一步到位

**数据形态变化**：
```
输入: chunks = [DocumentChunk(content="...", doc_id="...", ...), ...]
写入 ChromaDB:
  collection: knowledge_chunks
  每条记录:
    id: "a3f8b2c1d4e5f6a7#chunk-0"
    document: "第一段文本..."
    embedding: [0.12, -0.34, ...]  (1024 维 or 8 维 disabled)
    metadata: {doc_id, chunk_id, source, doc_type}
输出: vectors_stored = 12
```

**失败处理**：
- `vector_store` 是 None → 跳过，vectors_stored=0
- embedding 不可用 → `add_chunks` 返回 0
- chromadb 写入异常 → `logger.exception` 记录，**不静默吞**（重构期改的，原来是 `except: pass`）

**关键设计**：
- `ThreadPoolExecutor(max_workers=2)` 隔离 chromadb 的 C 扩展，避免 asyncio 事件循环 segfault
- cosine 距离存的是原始距离，检索时转成相似度 `1.0 - dist`

### 阶段 5：图谱写入（store_graph 节点）

**代码位置**：`orchestrator/graph.py:83` `store_graph` → `services/knowledge_graph.py:53` `upsert_entity` + `add_relation`

**做的事**：
1. 遍历每个 ExtractionResult
2. 对每个 Entity 执行 Cypher MERGE：
   ```cypher
   MERGE (e:Entity {name: $name})
   ON CREATE SET e.type=$type, e.description=$desc, e.version=1, e.created_at=$now, ...
   ON MATCH SET e.description=..., e.version=$version, e.updated_at=$now
   ```
3. 对每个 Relation：
   - `relation.relation.upper().replace(" ", "_")` 标准化
   - 校验是否在 8 种白名单内，不在则拒绝并记日志
   - 通过则 MATCH 头尾实体 + MERGE 关系

**数据形态变化**：
```
输入: extractions = [ExtractionResult(entities=[...], relations=[...]), ...]
写入 Neo4j:
  节点: (:Entity {name:"张三", type:"Person", description:"...", version:1, source:"", created_at:..., updated_at:...})
  关系: (张三)-[:WORKS_AT {confidence:0.95, source:"", updated_at:...}]->(ACME)
输出: entities_stored = 35
```

**失败处理**：
- `knowledge_graph` 是 None → 跳过
- Neo4j 连不上 → `init()` 阶段已降级，此处不执行
- Cypher 执行异常 → `logger.exception` 记录

**关键设计**：
- 关系类型白名单 `_ALLOWED_REL_TYPES`（8 种）防注入，LLM 抽的关系类型如果不在白名单会被拒
- 所有 Cypher 参数化（`$name` / `$type`），不拼字符串
- 实体有版本号 `version`，增量更新时递增

### 阶段 6：返回响应

**代码位置**：`api/main.py:130-142`

**做的事**：
1. 从流水线结果取 `chunks`、`extractions`
2. 统计 `total_entities` / `total_relations`
3. 返回 `IngestResponse`

## 3. 状态管理

入库过程是**无状态**的（除了向量库和图谱的持久化）。每次入库：
- doc_id 由文件路径 hash 生成，同文件多次入库会重复（MERGE 语义保证幂等）
- chunk_id 由 `doc_id#chunk-{index}` 生成
- 实体版本号初始为 1，增量更新时递增

## 4. 失败处理矩阵

| 失败点 | 处理方式 | 影响 |
|--------|---------|------|
| 文件类型不支持 | HTTP 400 | 不入库 |
| 文件落盘失败 | 异常向上抛 → HTTP 500 | 不入库 |
| workflow 没初始化 | HTTP 503 | 不入库 |
| PDF 解析失败 | 文本变成占位符 | 继续流程，质量降级 |
| LLM 抽取返回非 JSON | 返回空结果 | 继续流程，该 chunk 无知识 |
| LLM 调用 402 | 异常向上抛 | 流水线中断 |
| 向量入库失败 | `logger.exception` | 该批 chunk 丢失，有日志 |
| Neo4j 连不上 | init 阶段降级 | 图谱检索不可用，向量检索正常 |
| Cypher 执行失败 | `logger.exception` | 该实体/关系丢失，有日志 |

## 5. 并发与性能

- `parse_batch` 串行处理多个文件（LLM 调用是瓶颈，并行收益不大且容易触发限流）
- `extract` 按 BATCH_SIZE=5 分批，每批内串行调 LLM
- `add_chunks` 一次性批量写入 chromadb
- `store_graph` 逐实体/关系写 Neo4j（可优化为批量 UNWIND，当前未做）

## 6. 幂等性

- 同一文件多次上传：doc_id 相同，chunk_id 相同，chromadb 用 MERGE 语义覆盖
- Neo4j 实体 MERGE，关系 MERGE，天然幂等
- 但实体的 `version` 不会因重复入库递增（只有增量更新调 `upsert_entity` 时传 version 才递增）

## 相关文档

- [项目整体文档](./PROJECT_OVERVIEW.md)
- [新手代码导读](./CODE_GUIDE.md)
- [文档入库代码追踪](./INGESTION_CODE_TRACE.md)（本文的业务视角，代码追踪是代码视角）
- [项目架构设计详解](./ARCHITECTURE.md)
- [项目文件树](./FILE_TREE.md)
