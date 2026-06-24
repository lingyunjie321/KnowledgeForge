# 新手代码导读

> 面向接手代码的新手。按推荐顺序读，2 小时上手；按索引查，5 分钟定位。

## 1. 推荐阅读顺序

```
config/settings.py          先看配置，知道项目依赖哪些外部服务
    ↓
agents/doc_parser_agent.py  入库起点，理解 DocumentChunk 数据结构
    ↓
agents/knowledge_extract_agent.py  理解三元组抽取和跨 chunk 去重
    ↓
services/vector_store.py    向量怎么存怎么查，理解线程池隔离
    ↓
services/knowledge_graph.py Cypher 怎么写，关系类型白名单
    ↓
services/graph_rag.py       核心难点：6 步混合检索管道
    ↓
agents/qa_agent.py          检索委托给 GraphRAG，答案生成
    ↓
orchestrator/graph.py       LangGraph 编排，三条流水线怎么串
    ↓
api/main.py                 HTTP 入口，lifespan 初始化时序
    ↓
services/cdc_processor.py   增量更新，差量计算和版本追踪
    ↓
agents/knowledge_update_agent.py  Watchdog/Kafka 双模式监听
```

## 2. 核心数据结构

### 2.1 DocumentChunk（文档块）
位置：`agents/doc_parser_agent.py:26`

```python
@dataclass
class DocumentChunk:
    content: str                    # 块文本
    doc_id: str                     # 文档 ID（文件路径 SHA256 前 16 位）
    chunk_index: int                # 块序号
    doc_type: DocType               # 文档类型枚举
    metadata: dict[str, Any]        # source / char_start / char_end
    embedding: list[float] | None   # 向量（实际不入库，由 vector_store 现算）
```

`chunk_id` 是 property：`f"{doc_id}#chunk-{chunk_index}"`，作为向量库主键。

### 2.2 ExtractionResult（抽取结果）
位置：`agents/knowledge_extract_agent.py:65`

```python
@dataclass
class ExtractionResult:
    entities: list[Entity]          # 实体列表
    relations: list[Relation]       # 关系三元组
    events: list[KnowledgeEvent]    # 事件
    source_chunk_id: str            # 来自哪个 chunk
```

### 2.3 GraphRAGContext（检索上下文）
位置：`services/graph_rag.py:22`

```python
@dataclass
class GraphRAGContext:
    content: str           # 上下文文本
    source_type: str       # vector | subgraph | path | community
    score: float           # 原始分数（重排前）
    doc_type: str          # 文档类型，用于重排加权
    metadata: dict[str, Any]
```

### 2.4 QAResult（问答结果）
位置：`agents/qa_agent.py:37`

```python
@dataclass
class QAResult:
    question: str
    answer: str
    contexts: list[RetrievedContext]   # 检索到的上下文（最多 8 条）
    intent: QueryIntent                # 意图分类
    confidence: float                  # 置信度（上下文平均分）
    reasoning_steps: list[str]         # 推理步骤（可追溯）
```

## 3. 各模块入口与关键函数索引

### 3.1 agents/

#### DocParserAgent
| 方法 | 行号 | 作用 |
|------|------|------|
| `parse(file_path)` | 66 | 单文件解析入口，返回 `list[DocumentChunk]` |
| `parse_batch(file_paths)` | 86 | 批量解析 |
| `_classify(file_path)` | 92 | 按扩展名分类 |
| `_parse_pdf(file_path)` | 105 | PDF 解析（PyPDF2 失败降级视觉） |
| `_parse_image(file_path)` | 135 | 图片：OCR + LLM 描述 |
| `_parse_table(file_path)` | 173 | CSV/Excel 解析 |
| `_chunk_texts(...)` | 226 | 滑动窗口分块（512/64） |

#### KnowledgeExtractAgent
| 方法 | 行号 | 作用 |
|------|------|------|
| `extract(chunks)` | 86 | 批量抽取入口 |
| `_extract_from_text(text, id)` | 102 | 单文本抽取，调 LLM |
| `_parse_response(raw, id)` | 110 | 解析 LLM JSON 返回 |
| `_deduplicate(results)` | 154 | 跨 chunk 去重（静态方法） |

#### QAAgent
| 方法 | 行号 | 作用 |
|------|------|------|
| `answer(question)` | 102 | 问答入口 |
| `graph_rag` (property) | 92 | 惰性构造 GraphRAGPipeline |
| `_classify_intent(question)` | 133 | LLM 意图分类 |
| `_rewrite_query(question)` | 145 | LLM 查询改写 |
| `_generate_answer(...)` | 159 | 拼上下文调 LLM 生成 |
| `_calc_confidence(contexts)` | 193 | 静态方法，平均分 |

#### KnowledgeUpdateAgent
| 方法 | 行号 | 作用 |
|------|------|------|
| `process_change(change)` | 68 | 单变更处理入口 |
| `process_batch(changes)` | 87 | 批量处理 |
| `detect_changes(file_paths)` | 93 | 对比文件 hash 检测变更 |
| `start_watching(directory)` | 126 | 启动 Watchdog 监听 |
| `start_kafka_consumer()` | 177 | 启动 Kafka 消费 |
| `_handle_create/modify/delete` | 207/227/237 | 三种变更处理 |

### 3.2 services/

#### VectorStoreService
| 方法 | 行号 | 作用 |
|------|------|------|
| `init()` | 50 | 初始化（chroma/pgvector） |
| `add_chunks(chunks)` | 76 | 入库，返回入库数 |
| `search(query, top_k)` | 108 | 检索，返回 `list[(doc, score)]` |
| `delete_by_doc_id(doc_id)` | 139 | 按文档删除（增量更新用） |
| `get_stats()` | 148 | 统计信息 |

**关键设计**：`_run_sync` 把 chromadb 的 C 扩展调用丢到线程池，避免 asyncio 事件循环里 segfault。见 `vector_store.py:32`。

#### KnowledgeGraphService
| 方法 | 行号 | 作用 |
|------|------|------|
| `init()` | 27 | 建 driver + 建索引 |
| `upsert_entity(entity, version, source)` | 53 | MERGE 语义，存在则更新版本号 |
| `add_relation(relation, source)` | 81 | 关系类型白名单校验后写入 |
| `get_neighbors(name, hops)` | 116 | 多跳子图遍历（GraphRAG 第 3 步） |
| `execute_cypher(cypher, params)` | 103 | 通用 Cypher 执行（参数化） |
| `delete_by_source(source)` | 139 | 按来源删除（增量更新用） |

**关键设计**：`_ALLOWED_REL_TYPES`（第 15 行）是关系类型白名单，防 Cypher 注入。

#### GraphRAGPipeline
| 方法 | 行号 | 作用 |
|------|------|------|
| `retrieve(query, top_k)` | 61 | 检索入口，6 步管道 |
| `_vector_search(query, top_k)` | 76 | 第 1 步：向量检索 |
| `_entity_linking(query)` | 89 | 第 2 步：LLM 抽实体 |
| `_subgraph_search(entities, hops)` | 104 | 第 3 步：子图遍历 |
| `_path_search(entities)` | 124 | 第 4 步：shortestPath 路径推理 |
| `_community_summary(subgraph)` | 164 | 第 5 步：LLM 生成社区摘要 |
| `_cross_rerank(contexts, query)` | 178 | 第 6 步：双权重重排（静态方法） |

#### CDCProcessor
| 方法 | 行号 | 作用 |
|------|------|------|
| `from_filesystem_event(...)` | 51 | 文件系统事件归一化 |
| `from_kafka_message(message)` | 63 | Kafka 消息归一化（兼容 Debezium） |
| `compute_diff(before, after)` | 77 | 差量计算，返回 change_ratio |
| `bump_version(resource)` | 99 | 版本递增 |
| `process_event(event)` | 108 | 单事件处理 |

### 3.3 orchestrator/

#### build_knowledge_graph_workflow
位置：`orchestrator/graph.py:33`

返回 `{"ingest": graph, "qa": graph, "update": graph}` 三个编译好的 LangGraph。

- `_build_ingest_graph`（第 55 行）：parse → extract → store_vectors → store_graph
- `_build_qa_graph`（第 113 行）：单节点 answer
- `_build_update_graph`（第 128 行）：process → 条件分支 → retry / done

### 3.4 api/

#### lifespan
位置：`api/main.py:31`

启动时序：
1. 建 uploads 目录
2. `vector_store.init()` —— 失败直接 raise（带病运行更危险）
3. `knowledge_graph.init()` —— 失败降级但有日志（图谱检索降级不致命）
4. `build_knowledge_graph_workflow()` —— 构建三条流水线

#### 关键端点
| 路由 | 方法 | 行号 | 作用 |
|------|------|------|------|
| `/api/ingest/upload` | POST | 113 | 单文件入库 |
| `/api/ingest/batch` | POST | 145 | 批量入库 |
| `/api/qa/ask` | POST | 154 | 智能问答 |
| `/api/admin/stats` | GET | 178 | 库统计 |
| `/api/admin/update` | POST | 193 | 触发增量更新 |
| `/api/health` | GET | 220 | 健康检查 |

## 4. 调试技巧

### 4.1 单步调试入库
```python
import asyncio
from agents.doc_parser_agent import DocParserAgent

async def main():
    agent = DocParserAgent()
    chunks = await agent.parse("uploads/test.txt")
    for c in chunks:
        print(c.chunk_id, c.doc_type, len(c.content))

asyncio.run(main())
```

### 4.2 单步调试 GraphRAG
```python
import asyncio
from services.vector_store import VectorStoreService
from services.knowledge_graph import KnowledgeGraphService
from services.graph_rag import GraphRAGPipeline

async def main():
    vs = VectorStoreService()
    await vs.init()
    kg = KnowledgeGraphService()
    await kg.init()
    pipeline = GraphRAGPipeline(vector_store=vs, knowledge_graph=kg)
    contexts = await pipeline.retrieve("张三在哪家公司工作？")
    for c in contexts:
        print(c.source_type, c.score, c.content[:80])

asyncio.run(main())
```

### 4.3 看检索来源分布
问答结果 `reasoning_steps` 里有 4 种来源的命中数：
```python
result = await qa_agent.answer("...")
print(result.reasoning_steps)
# ['识别问题意图: factoid',
#  'GraphRAG 检索到 5 条上下文',
#  '  - vector: 2 条',
#  '  - subgraph: 2 条',
#  '  - path: 1 条',
#  '答案生成完成']
```

### 4.4 跑测试
```bash
# 全量
EMBEDDING_PROVIDER=disabled python -m pytest tests/ -v

# 只跑 Agent
EMBEDDING_PROVIDER=disabled python -m pytest tests/agents/ -v

# 跑 F1 评测
EMBEDDING_PROVIDER=disabled python -m tests.eval.f1_compare

# 跑增量更新基准
EMBEDDING_PROVIDER=disabled python -m tests.bench.update_benchmark
```

### 4.5 看 Neo4j 图谱
```bash
# 启动 Neo4j 后浏览器打开
http://localhost:7474

# 查所有实体
MATCH (e:Entity) RETURN e LIMIT 50

# 查某实体的 2 跳邻居
MATCH path = (e:Entity {name: "张三"})-[*1..2]-(n) RETURN path
```

## 5. 常见问题

### Q1: 启动报 "向量库初始化失败"
A: ChromaDB 初始化失败会直接 raise。检查 `chroma_data/` 目录权限，或 `EMBEDDING_PROVIDER` 配置。CI/单测环境用 `disabled`。

### Q2: Neo4j 连不上但服务能起
A: 这是设计如此。`api/main.py:43` 的 `knowledge_graph.init()` 失败只记日志不 raise，图谱检索降级为纯向量。生产环境要保证 Neo4j 可用。

### Q3: 上传文件后问答答不出来
A: 三种可能：
1. `EMBEDDING_PROVIDER=disabled`，向量是零向量，检索无效 → 改 `local` 或 `api`
2. DeepSeek 没充值，LLM 调用 402 → 充值
3. 文档太短，没抽出实体 → 看入库日志的 `entities_count`

### Q4: pytest 报 asyncio 相关错误
A: 项目用 `pytest-asyncio` STRICT 模式（见 `pytest.ini`）。async 测试函数必须加 `@pytest.mark.asyncio` 装饰器，或用 `asyncio_mode=auto`。conftest.py 里有公共夹具。

### Q5: 修改文件后没触发增量更新
A: `start_watching()` 是非阻塞的，要在事件循环里跑。Watchdog 回调在独立线程，已用 `run_coroutine_threadsafe` 丢回主循环。确认调用了 `start_watching()` 且主进程没退出。

### Q6: Cypher 报关系类型不在白名单
A: 看 `services/knowledge_graph.py:15` 的 `_ALLOWED_REL_TYPES`。LLM 抽的关系会被 `.upper().replace(" ", "_")`，如果抽出来不在白名单会被拒。如需扩展，加到白名单集合里。

## 6. 代码风格约定（重构期定下）

1. 注释只写"为什么"，不写"是什么"
2. docstring 最多 2 行
3. 禁用装饰分隔条（`# ── xxx ──`、`# === xxx ===`）
4. 禁用营销话（"核心技术亮点"、"高效智能"）
5. 中文为主，技术术语保留英文
6. 函数命名表意即可，不加 `Helper/Manager/Service` 后缀（已有的保留）

改代码前必读 `AGENTS.md`（项目根，不入仓库），违反风格规则即返工。

## 相关文档

- [项目整体文档](./PROJECT_OVERVIEW.md)
- [文档入库流程](./INGESTION_FLOW.md)
- [文档入库代码追踪](./INGESTION_CODE_TRACE.md)
- [项目架构设计详解](./ARCHITECTURE.md)
- [项目文件树](./FILE_TREE.md)
