# 项目文件树

> 完整文件树 + 每个文件一行职责说明 + 行数统计 + 模块依赖关系。

## 1. 完整文件树

```
KnowledgeForge/
│
├── agents/                          # Agent 层：4 个核心 Agent
│   ├── __init__.py                  #   导出 4 个 Agent 类（11 行）
│   ├── doc_parser_agent.py          #   文档解析：多模态文件 → DocumentChunk（250 行）
│   ├── knowledge_extract_agent.py   #   知识抽取：chunk → 实体/关系/事件三元组（182 行）
│   ├── qa_agent.py                  #   问答：意图分类 + 查询改写 + 答案生成（198 行）
│   └── knowledge_update_agent.py    #   增量更新：Watchdog/Kafka 监听变更（258 行）
│
├── services/                        # 服务层：6 个服务模块
│   ├── __init__.py                  #   导出 VectorStoreService / KnowledgeGraphService（4 行）
│   ├── vector_store.py              #   向量库：ChromaDB/PGVector 双后端，线程池隔离（154 行）
│   ├── knowledge_graph.py           #   Neo4j 图谱：实体/关系 CRUD + Cypher 参数化（155 行）
│   ├── graph_rag.py                 #   GraphRAG 6 步混合检索管道（205 行）
│   ├── embedding.py                 #   Embedding 三档统一出口（102 行）
│   ├── embedding_worker.py          #   local 档子进程 worker，隔离 PyTorch（124 行）
│   └── cdc_processor.py             #   CDC 事件归一化 + 差量计算 + 版本追踪（186 行）
│
├── orchestrator/                    # 编排层：LangGraph 三流水线
│   ├── __init__.py                  #   导出 build_knowledge_graph_workflow（3 行）
│   └── graph.py                     #   构建入库/问答/更新三条 StateGraph（157 行）
│
├── api/                             # 表现层：FastAPI HTTP 入口
│   ├── __init__.py                  #   空文件
│   └── main.py                      #   FastAPI app + 6 个端点 + lifespan 初始化（228 行）
│
├── config/                          # 配置层
│   ├── __init__.py                  #   导出 settings 单例（3 行）
│   └── settings.py                  #   pydantic-settings 配置类，读 .env（49 行）
│
├── utils/                           # 工具函数（预留）
│   └── __init__.py                  #   空文件（0 行）
│
├── static/                          # 前端三件套
│   ├── index.html                   #   单页应用入口，3 个 tab（196 行）
│   ├── style.css                    #   玻璃拟态样式（1124 行）
│   └── app.js                       #   前端逻辑：问答/上传/统计（369 行）
│
├── tests/                           # 测试 + 评测 + 基准
│   ├── __init__.py                  #   空文件
│   ├── conftest.py                  #   公共夹具：mock LLM / FakeVectorStore / 临时目录（177 行）
│   ├── test_settings.py             #   配置加载测试（16 行）
│   ├── test_ingest_upload.py        #   入库上传测试（37 行）
│   ├── test_orchestrator.py         #   三流水线构建与执行测试（77 行）
│   ├── test_api_smoke.py            #   API smoke test（68 行）
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── test_doc_parser_agent.py     #   解析/分块/分类测试（107 行）
│   │   ├── test_knowledge_extract_agent.py  # 抽取/去重测试（124 行）
│   │   ├── test_qa_agent.py             #   意图分类/全流程 mock 测试（107 行）
│   │   └── test_knowledge_update_agent.py   # 增删改/失败兜底测试（167 行）
│   ├── services/
│   │   ├── __init__.py
│   │   ├── test_vector_store.py     #   入库检索往返/删除/disabled 降级（110 行）
│   │   ├── test_graph_rag.py        #   双权重重排/Cypher 参数化验证（162 行）
│   │   └── test_cdc_processor.py    #   差量阈值/版本递增/事件归一化（142 行）
│   ├── eval/                        # F1 评测
│   │   ├── __init__.py
│   │   ├── in_memory_graph.py       #   内存图谱（BFS 子图+最短路径，不依赖 Neo4j）（87 行）
│   │   └── f1_compare.py            #   F1 评测脚本：纯向量 vs GraphRAG（241 行）
│   └── bench/                       # 基准测试
│       ├── __init__.py
│       └── update_benchmark.py      #   增量更新 vs 全量重建效率对比（132 行）
│
├── docs/                            # 项目文档（本目录）
│   ├── PROJECT_OVERVIEW.md          #   项目整体文档（180 行）
│   ├── CODE_GUIDE.md                #   新手代码导读（324 行）
│   ├── INGESTION_FLOW.md            #   文档入库流程（256 行）
│   ├── INGESTION_CODE_TRACE.md      #   文档入库代码追踪（556 行）
│   ├── ARCHITECTURE.md              #   项目架构设计详解（421 行）
│   └── FILE_TREE.md                 #   项目文件树（本文档）
│
├── AGENTS.md                        # 重构行为准则（不入仓库，本地保留）
├── HANDOFF.md                       # 交接说明（不入仓库，本地保留）
├── Dockerfile                       # 容器化部署（16 行）
├── requirements.txt                 # 依赖锁定（29 行）
├── pytest.ini                       # pytest 配置：asyncio STRICT 模式（6 行）
├── .env.example                     # 环境变量模板（39 行）
├── .env                             # 实际环境变量（不入仓库）
└── .gitignore                       # Git 忽略规则（81 行）
```

## 2. 行数统计总览

| 目录 | 文件数 | 总行数 | 说明 |
|------|--------|--------|------|
| agents/ | 5 | 899 | 4 个核心 Agent |
| services/ | 7 | 930 | 6 个服务模块 |
| orchestrator/ | 2 | 160 | LangGraph 编排 |
| api/ | 2 | 228 | FastAPI 入口 |
| config/ | 2 | 52 | 配置加载 |
| utils/ | 1 | 0 | 预留 |
| static/ | 3 | 1689 | 前端三件套 |
| tests/ | 17 | 1566 | 测试 + 评测 + 基准 |
| docs/ | 6 | 1937 | 项目文档 |
| 根目录 | 7 | 317 | Dockerfile/requirements 等 |
| **总计** | **52** | **7778** | |

## 3. 模块依赖关系

```
                    ┌─────────────────┐
                    │  api/main.py    │
                    │  （HTTP 入口）   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
    ┌─────────────────┐ ┌──────────┐ ┌──────────────┐
    │ orchestrator/   │ │ services/│ │   config/    │
    │   graph.py      │ │ vector_  │ │  settings.py │
    │ （LangGraph）   │ │  store   │ └──────────────┘
    └────────┬────────┘ └────┬─────┘
             │               │
    ┌────────┴────────┐      │
    │                 │      │
    ▼                 ▼      ▼
┌────────┐  ┌────────────┐ ┌──────────────┐
│agents/ │  │  agents/   │ │  services/   │
│doc_    │  │  qa_agent  │ │  graph_rag   │
│parser  │  │            │ └──────┬───────┘
└───┬────┘  └─────┬──────┘        │
    │             │               │
    │             │   ┌───────────┴───────────┐
    │             │   ▼                       ▼
    │             │ ┌──────────────┐  ┌──────────────┐
    │             │ │ services/    │  │ services/    │
    │             │ │ vector_store │  │ knowledge_   │
    │             │ └──────┬───────┘  │   graph      │
    │             │        │          └──────┬───────┘
    │             │        │                 │
    │             │        ▼                 │
    │             │ ┌──────────────┐         │
    │             │ │ services/    │         │
    │             │ │ embedding    │         │
    │             │ └──────┬───────┘         │
    │             │        │                 │
    │             │        ▼                 │
    │             │ ┌──────────────┐         │
    │             │ │ services/    │         │
    │             │ │ embedding_   │         │
    │             │ │   worker     │         │
    │             │ └──────────────┘         │
    │             │                          │
    ▼             ▼                          ▼
┌────────────────────────┐        ┌──────────────────┐
│  agents/                │        │   存储层          │
│  knowledge_extract_     │        │  ChromaDB        │
│    agent                │        │  Neo4j           │
│  （产出 Entity/Relation）│        │  uploads/        │
└────────────────────────┘        └──────────────────┘
```

### 依赖说明

| 模块 | 依赖 | 说明 |
|------|------|------|
| api/main.py | orchestrator, services, agents, config | HTTP 入口，组装所有依赖 |
| orchestrator/graph.py | agents (4个), services (2个) | 编排层，不直接访问存储 |
| agents/doc_parser_agent | config | 只依赖配置（LLM client） |
| agents/knowledge_extract_agent | agents/doc_parser_agent, config | 依赖 DocumentChunk 数据结构 |
| agents/qa_agent | services/graph_rag (惰性), config | 检索委托 GraphRAG |
| agents/knowledge_update_agent | agents (doc_parser, extract), services (vector, kg) | 增量更新需要全部依赖 |
| services/vector_store | services/embedding, config | 通过 embedding 工厂获取实现 |
| services/knowledge_graph | agents/knowledge_extract (Entity/Relation), config | 只依赖数据结构，不依赖 Agent |
| services/graph_rag | services/vector_store, services/knowledge_graph, config | 6 步管道串联两个服务 |
| services/embedding | services/embedding_worker (local 档), config | 工厂模式，三档切换 |
| services/cdc_processor | config | 纯计算，不依赖外部服务 |

## 4. 入口文件

| 入口 | 文件 | 启动命令 |
|------|------|---------|
| HTTP 服务 | api/main.py | `python -m api.main` 或 `uvicorn api.main:app --port 8080` |
| pytest | pytest.ini + tests/ | `EMBEDDING_PROVIDER=disabled python -m pytest tests/` |
| F1 评测 | tests/eval/f1_compare.py | `EMBEDDING_PROVIDER=disabled python -m tests.eval.f1_compare` |
| 增量基准 | tests/bench/update_benchmark.py | `EMBEDDING_PROVIDER=disabled python -m tests.bench.update_benchmark` |
| Docker | Dockerfile | `docker build -t knowledgeforge . && docker run -p 8080:8080 knowledgeforge` |

## 5. 文件分类索引

### 按职责

| 职责 | 文件 |
|------|------|
| 文档解析 | agents/doc_parser_agent.py |
| 知识抽取 | agents/knowledge_extract_agent.py |
| 向量存储 | services/vector_store.py |
| 图谱存储 | services/knowledge_graph.py |
| 混合检索 | services/graph_rag.py |
| 问答生成 | agents/qa_agent.py |
| 增量更新 | agents/knowledge_update_agent.py, services/cdc_processor.py |
| Embedding | services/embedding.py, services/embedding_worker.py |
| 编排 | orchestrator/graph.py |
| HTTP | api/main.py |
| 配置 | config/settings.py, .env.example |
| 前端 | static/index.html, static/style.css, static/app.js |
| 测试 | tests/**/*.py |
| 评测 | tests/eval/*.py |
| 基准 | tests/bench/*.py |
| 文档 | docs/*.md, README.md |

### 按数据结构定义位置

| 数据结构 | 文件 | 行号 |
|---------|------|------|
| DocType (枚举) | agents/doc_parser_agent.py | 17 |
| DocumentChunk | agents/doc_parser_agent.py | 26 |
| Entity | agents/knowledge_extract_agent.py | 37 |
| Relation | agents/knowledge_extract_agent.py | 49 |
| KnowledgeEvent | agents/knowledge_extract_agent.py | 58 |
| ExtractionResult | agents/knowledge_extract_agent.py | 65 |
| QueryIntent (枚举) | agents/qa_agent.py | 19 |
| RetrievedContext | agents/qa_agent.py | 27 |
| QAResult | agents/qa_agent.py | 37 |
| ChangeType (枚举) | agents/knowledge_update_agent.py | 18 |
| DocumentChange | agents/knowledge_update_agent.py | 24 |
| UpdateResult | agents/knowledge_update_agent.py | 34 |
| GraphRAGContext | services/graph_rag.py | 22 |
| CDCEvent | services/cdc_processor.py | 17 |
| CDCProcessResult | services/cdc_processor.py | 29 |
| WorkflowType (枚举) | orchestrator/graph.py | 27 |
| Settings | config/settings.py | 6 |

### 按关键常量

| 常量 | 文件 | 行号 | 值 |
|------|------|------|-----|
| SUPPORTED_EXTENSIONS | agents/doc_parser_agent.py | 43 | 9 种扩展名 |
| CHUNK_SIZE | agents/doc_parser_agent.py | 55 | 512 |
| CHUNK_OVERLAP | agents/doc_parser_agent.py | 56 | 64 |
| BATCH_SIZE (extract) | agents/knowledge_extract_agent.py | 76 | 5 |
| MAX_RETRY | agents/knowledge_update_agent.py | 51 | 3 |
| COLLECTION_NAME | services/vector_store.py | 23 | "knowledge_chunks" |
| _ALLOWED_REL_TYPES | services/knowledge_graph.py | 15 | 8 种关系 |
| MAJOR_CHANGE_THRESHOLD | services/cdc_processor.py | 44 | 0.3 |
| source_weight (rerank) | services/graph_rag.py | 182 | vector 1.0 / subgraph 1.15 / path 1.25 / community 1.1 |
| doc_type_weight (rerank) | services/graph_rag.py | 185 | text 1.0 / pdf 0.95 / table 0.9 / image 0.85 |

## 6. 不入仓库的文件

`.gitignore` 排除的文件（本地保留，不进 Git）：

| 文件/目录 | 原因 |
|----------|------|
| .env | 含密钥 |
| .workbuddy/ | CodeBuddy 工作目录 |
| AGENTS.md | 重构行为准则（内部文档） |
| 重构计划.md | 重构计划（内部文档） |
| HANDOFF.md | 交接说明（内部文档） |
| chroma_data/ | ChromaDB 持久化数据 |
| uploads/ | 上传的原始文件 |
| neo4j_data/ | Neo4j 数据（如本地部署） |
| kafka_data/ | Kafka 数据（如本地部署） |
| __pycache__/ | Python 缓存 |
| .venv/ | 虚拟环境 |
| .pytest_cache/ | pytest 缓存 |
| .idea/ | IDE 配置 |

## 相关文档

- [项目整体文档](./PROJECT_OVERVIEW.md)
- [新手代码导读](./CODE_GUIDE.md)
- [文档入库流程](./INGESTION_FLOW.md)
- [文档入库代码追踪](./INGESTION_CODE_TRACE.md)
- [项目架构设计详解](./ARCHITECTURE.md)
