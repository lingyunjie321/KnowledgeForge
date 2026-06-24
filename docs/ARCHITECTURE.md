# 项目架构设计详解

> 面向架构评审。讲清楚分层、Agent 设计、GraphRAG 管道、Embedding 三档、CDC 增量、扩展点、技术决策与权衡。

## 1. 分层架构

```
┌─────────────────────────────────────────────────────────────┐
│  表现层  static/（HTML/CSS/JS）+ api/main.py（FastAPI）      │
├─────────────────────────────────────────────────────────────┤
│  编排层  orchestrator/graph.py（LangGraph 三流水线）         │
├─────────────────────────────────────────────────────────────┤
│  Agent 层  agents/                                           │
│    DocParserAgent  KnowledgeExtractAgent                     │
│    QAAgent          KnowledgeUpdateAgent                     │
├─────────────────────────────────────────────────────────────┤
│  服务层  services/                                           │
│    VectorStoreService  KnowledgeGraphService                 │
│    GraphRAGPipeline    CDCProcessor                          │
│    Embedding（三档）   EmbeddingWorker（子进程）              │
├─────────────────────────────────────────────────────────────┤
│  配置层  config/settings.py（pydantic-settings + .env）      │
├─────────────────────────────────────────────────────────────┤
│  存储层  ChromaDB（向量）+ Neo4j（图谱）+ uploads/（原文）    │
└─────────────────────────────────────────────────────────────┘
```

**依赖方向**：上层依赖下层，下层不反向依赖。Agent 层不直接调 ChromaDB/Neo4j，通过服务层间接访问。

## 2. Agent 设计

### 2.1 为什么是 4 个 Agent

| Agent | 职责边界 | 为什么独立 |
|-------|---------|-----------|
| DocParserAgent | 文件 → DocumentChunk | 多模态解析逻辑复杂，独立可单测 |
| KnowledgeExtractAgent | DocumentChunk → ExtractionResult | 强依赖 LLM，prompt 和去重逻辑独立 |
| QAAgent | 问题 → QAResult | 检索委托 GraphRAG，只管意图/改写/生成 |
| KnowledgeUpdateAgent | DocumentChange → UpdateResult | CDC 逻辑和全量入库不同，独立演化 |

### 2.2 Agent 间通信

Agent 之间**不直接调用**，通过 LangGraph state 传递：

```
parse 节点输出 chunks → extract 节点输入 chunks
extract 节点输出 extractions → store_vectors/store_graph 输入
```

state 是普通 dict，LangGraph 负责节点间流转。

### 2.3 Agent 的依赖注入

`KnowledgeUpdateAgent` 和 `QAAgent` 通过构造函数注入依赖：

```python
# QAAgent 可以外部注入 graph_rag，也可以惰性构造
def __init__(self, vector_store=None, knowledge_graph=None, graph_rag=None):
    self._graph_rag = graph_rag  # 外部注入优先

@property
def graph_rag(self):
    if self._graph_rag is None and self.vector_store and self.knowledge_graph:
        from services.graph_rag import GraphRAGPipeline
        self._graph_rag = GraphRAGPipeline(...)
    return self._graph_rag
```

好处：单测可注入 mock，生产可惰性构造。

## 3. LangGraph 编排

### 3.1 三条流水线

```
ingest 流水线（入库）:
  parse → extract → store_vectors → store_graph → END

qa 流水线（问答）:
  answer → END

update 流水线（增量更新）:
  process → should_continue?
              ├─ "done" → END
              └─ "retry" → retry → END
```

### 3.2 为什么用 LangGraph

- **可视化流转**：StateGraph 的节点和边清晰表达数据流
- **条件分支**：update 流水线的 retry 逻辑用 `add_conditional_edges`
- **状态管理**：state 在节点间自动传递，不用手动串
- **可编译**：`graph.compile()` 返回可执行对象，支持 `ainvoke` 异步调用

### 3.3 编译时机

`build_knowledge_graph_workflow()` 在 `api/main.py` 的 `lifespan` 里调用一次，编译好的三个 graph 存到全局 `workflows` dict，HTTP handler 直接取用。**不在每次请求时重新编译**（编译有开销）。

### 3.4 入库失败处理策略

```python
# orchestrator/graph.py:72-81
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

**设计决策**：入库失败**不中断流水线**（store_graph 还能跑），但**不静默吞**（`logger.exception` 记录）。

原代码是 `except: pass`，重构期改成 `logger.exception`。理由：数据悄悄丢会让问答质量下降却查不出原因。

## 4. GraphRAG 6 步管道

### 4.1 管道结构

```
输入: query
  │
  ▼
Step 1: _vector_search       向量检索（chromadb）
  │  → vector_results
  ▼
Step 2: _entity_linking      LLM 从 query 抽实体名
  │  → entities: ["张三", "ACME"]
  ▼
Step 3: _subgraph_search     Neo4j 多跳邻居遍历
  │  → subgraph_results
  ▼
Step 4: _path_search         Neo4j shortestPath 路径推理
  │  → path_results
  ▼
Step 5: _community_summary   LLM 对子图生成摘要（仅 subgraph 非空时）
  │  → community_ctx
  ▼
Step 6: _cross_rerank        双权重重排 + 去重
  │  → reranked[:top_k]
  ▼
输出: list[GraphRAGContext]
```

### 4.2 各步设计

#### Step 1：向量检索
- 调 `VectorStoreService.search`
- 走线程池（chromadb C 扩展隔离）
- 返回带 `doc_type` 的 GraphRAGContext（重排用）

#### Step 2：实体链接
- LLM prompt 要求返回 JSON `{"entities": [...]}`
- 失败返回空列表（不中断）
- **这是 GraphRAG 和纯向量 RAG 的关键差异**：把问题映射到图谱节点

#### Step 3：子图检索
- 对每个实体，Neo4j `MATCH path = (start)-[*1..2]-(neighbor)`
- hops=2 默认，可配
- LIMIT 50 防爆
- 把邻居关系格式化成文本上下文

#### Step 4：路径推理
- **只有 ≥2 个实体时才执行**（单实体无路径可言）
- 对实体两两组合调 `shortestPath`
- `[*..5]` 限制最长 5 跳
- LIMIT 3 防爆
- **这是多跳推理的关键**：纯向量检索无法回答"张三的同事在哪个城市"

#### Step 5：社区摘要
- **仅当 Step 3 有结果时才执行**（没子图没摘要）
- 把子图结果拼成文本（最多 20 条）
- LLM 生成结构化摘要
- 这是 Microsoft GraphRAG 论文的社区检测思路的简化版

#### Step 6：交叉重排
- **双权重**：
  - `source_weight`：vector 1.0 / subgraph 1.15 / path 1.25 / community 1.1
  - `doc_type_weight`：text/markdown 1.0 / pdf 0.95 / table 0.9 / image 0.85
- `final_score = original_score × source_weight × doc_type_weight`
- 去重：content 前 80 字符相同视为重复，保留首个
- 按分数降序取 top_k

### 4.3 设计权衡

**为什么 path 权重最高（1.25）？**
路径推理是多跳关联的硬证据，A→B→C 的链条比单条向量相似度可信。

**为什么 image 权重最低（0.85）？**
图片是 OCR/LLM 描述转文本再嵌入的，信息损耗大，质量不如原生文本。

**为什么社区摘要单独有一类？**
社区摘要是 LLM 对子图的综合理解，比单条邻居信息密度高，但比路径推理的确定性弱。

## 5. Embedding 三档架构

### 5.1 接口设计

```python
# services/embedding.py
class Embeddings(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]: ...
    async def aembed_query(self, text: str) -> list[float]: ...
```

三个实现（`NullEmbeddings` / `ApiEmbeddings` / `LocalEmbeddings`）都遵守这个接口，上层 `VectorStoreService` 通过 `create_embeddings()` 工厂获取，**无感知切换**。

### 5.2 local 档的子进程隔离

**问题**：sentence-transformers 依赖 PyTorch，PyTorch 的 C 扩展在 asyncio 事件循环里会 segfault。

**解决**：`services/embedding_worker.py` 用 `multiprocessing` spawn 一个子进程跑模型，主进程通过 `Queue` 通信。

```
主进程（asyncio 事件循环）
    │
    ├── request_queue.put((msg_id, texts))
    │
    ▼
子进程（spawn）
    └── SentenceTransformer.encode(texts)
    │
    ├── response_queue.put((msg_id, vectors))
    ▼
主进程
    └── response_queue.get() → vectors
```

**关键点**：
- spawn 模式（不是 fork），子进程不继承父进程内存
- 模型名和 device 通过环境变量传（spawn 不能直接传对象）
- 启动时等子进程 warmup 完成才返回
- daemon=True，主进程退出子进程自动死

### 5.3 disabled 档的存在意义

- CI/单测环境没装 torch，要能跑测试
- 返回 8 维零向量，检索降级为"无结果"
- `embeddings_available` 为 False 时，`search` 直接返回 `[]`，`add_chunks` 返回 0

### 5.4 切换方式

只改 `.env`，不改代码：
```bash
EMBEDDING_PROVIDER=local    # 三选一
# EMBEDDING_PROVIDER=api
# EMBEDDING_PROVIDER=disabled
```

`create_embeddings()` 读 `settings.embedding_provider` 返回对应实现。

## 6. CDC 增量更新架构

### 6.1 两种来源

```
来源 1: Watchdog 文件监听
  KnowledgeUpdateAgent.start_watching(directory)
    └── Observer 线程
        └── on_created/on_modified/on_deleted
            └── run_coroutine_threadsafe(process_change(...))  # 丢回主循环

来源 2: Kafka CDC 消费
  KnowledgeUpdateAgent.start_kafka_consumer()
    └── Consumer.poll() 循环
        └── from_kafka_message() 归一化
            └── process_change()
```

### 6.2 变更处理

```
CREATED  → _handle_create:  parse + add_chunks + upsert_entity + add_relation
MODIFIED → _handle_modify:  delete_by_doc_id + _handle_create（删旧建新）
DELETED  → _handle_delete:  delete_by_doc_id + delete_by_source
```

### 6.3 差量计算（CDCProcessor）

```python
# services/cdc_processor.py:77
def compute_diff(before: str, after: str) -> dict:
    before_lines = set(before.splitlines())
    after_lines = set(after.splitlines())
    added = after_lines - before_lines
    removed = before_lines - after_lines
    change_ratio = len(added | removed) / max(len(before_lines) + len(after_lines), 1)
    return {
        ...,
        "change_ratio": round(change_ratio, 4),
        "is_major_change": change_ratio > 0.3,  # 30% 阈值
    }
```

**设计**：差量超 30% 触发全量重建，否则增量更新。避免小改动也全量重建浪费资源。

### 6.4 Watchdog 的 asyncio 并发 bug

**问题**：Watchdog 回调在独立线程触发，直接 `await agent.process_change()` 会报 "no running event loop"。

**解决**（`agents/knowledge_update_agent.py:135-141`）：
```python
loop = asyncio.get_event_loop()  # 主循环
pending: set = set()

def _schedule(coro):
    task = asyncio.run_coroutine_threadsafe(coro, loop)
    pending.add(task)
    task.add_done_callback(pending.discard)
```

把协程通过 `run_coroutine_threadsafe` 丢回主事件循环执行，`pending` set 防止 task 被 GC。

### 6.5 版本追踪

`_version_counter: dict[str, int]` 记录每个实体的版本号，每次 `upsert_entity` 递增。Neo4j 节点的 `version` 属性随之更新，可追溯实体变更历史。

## 7. 安全设计

### 7.1 Cypher 注入防护

**关系类型白名单**（`services/knowledge_graph.py:15`）：
```python
_ALLOWED_REL_TYPES = {
    "BELONGS_TO", "WORKS_AT", "LOCATED_IN", "DEVELOPED_BY",
    "RELATED_TO", "PART_OF", "USES", "DEPENDS_ON",
}
```

LLM 抽的关系类型经 `.upper().replace(" ", "_")` 标准化后，不在白名单直接拒绝。白名单内的类型才拼进 Cypher（`f"...[:{rel_type}]..."`），否则用参数化。

### 7.2 路径穿越防护

`api/main.py:121`：
```python
safe_name = os.path.basename(file_name)  # 去掉目录部分
```

即使客户端传 `../../etc/passwd`，也只取 `passwd`，落盘到 `uploads/passwd`。

### 7.3 配置隔离

`.env` 文件不入仓库（`.gitignore` 排除），`.env.example` 提供模板。密钥通过环境变量注入，不硬编码。

## 8. 扩展点

### 8.1 加新文档类型
1. `DocParserAgent.SUPPORTED_EXTENSIONS` 加扩展名映射
2. `DocType` 枚举加新类型（如果需要）
3. `parse()` 方法加 `elif` 分支调新解析器
4. `_cross_rerank` 的 `doc_type_weight` 加新权重

### 8.2 加新关系类型
1. `services/knowledge_graph.py:15` `_ALLOWED_REL_TYPES` 加新类型
2. 抽取 prompt（`knowledge_extract_agent.py:32`）的关系类型列表同步更新

### 8.3 换向量库后端
1. `VectorStoreService._init_*` 加新后端初始化
2. `add_chunks` / `search` / `delete_by_doc_id` 加新分支
3. `config/settings.py` 加 `vector_store_type` 新值

### 8.4 加新 CDC 来源
1. `CDCProcessor.from_*` 加新归一化方法
2. `KnowledgeUpdateAgent.start_*` 加新监听方法

### 8.5 加新 GraphRAG 检索步
1. `GraphRAGPipeline` 加新 `_xxx_search` 方法
2. `retrieve()` 里插入调用
3. `_cross_rerank` 的 `source_weight` 加新来源权重

## 9. 技术决策与权衡

### 9.1 为什么 ChromaDB 而不是 Milvus/Qdrant
- ChromaDB 纯 Python，嵌入式部署，单机够用
- 企业私有化场景数据量不会太大（百万级 chunk）
- 重构期锁定，不引入新向量库

### 9.2 为什么 Neo4j 而不是其他图库
- Cypher 表达力强，`shortestPath` 原生支持
- AsyncGraphDatabase 异步驱动成熟
- 企业图数据库事实标准，简历加分

### 9.3 为什么 DeepSeek 而不是 GPT-4
- OpenAI API 兼容，LangChain 无缝接入
- 中文场景效果好
- 成本低（重构期可承受）

### 9.4 为什么 LangGraph 而不是直接 async 函数串
- 流水线可视化，状态管理清晰
- 条件分支（retry 逻辑）原生支持
- 简历加分（Agent 编排是热点）

### 9.5 为什么子进程隔离 Embedding
- PyTorch C 扩展在 asyncio 里 segfault 是真实问题
- 子进程崩了不影响主进程，daemon 自动重启
- 牺牲一点启动时间换稳定性

### 9.6 为什么增量更新是"删旧建新"而不是 diff patch
- 文本 diff patch 实现复杂，边界 case 多
- 删旧建新语义简单，幂等可重试
- 性能损耗在可接受范围（基准脚本验证 92-97% 效率提升）

## 10. 已知限制

1. **store_graph 逐条写 Neo4j**：可优化为 UNWIND 批量写，当前未做
2. **extract 串行调 LLM**：BATCH_SIZE=5 但批内串行，可改并发（注意限流）
3. **社区摘要没有社区检测**：用的是子图直接摘要，不是 Louvain 算法分社区
4. **path_search 两两组合**：实体多时 Cypher 调用次数是 O(n²)
5. **没有缓存层**：相同问题每次都重新检索，可加 Redis 缓存

## 相关文档

- [项目整体文档](./PROJECT_OVERVIEW.md)
- [新手代码导读](./CODE_GUIDE.md)
- [文档入库流程](./INGESTION_FLOW.md)
- [文档入库代码追踪](./INGESTION_CODE_TRACE.md)
- [项目文件树](./FILE_TREE.md)
