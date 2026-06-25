# KnowledgeForge

企业私有知识库管理系统：文档自动入库 → 知识抽取 → GraphRAG 混合检索问答 → CDC 增量更新。

## 技术栈

- 后端：Python 3.12 + FastAPI + LangGraph + LangChain
- 向量库：ChromaDB（主）/ PGVector（备）
- 图库：Neo4j
- LLM：DeepSeek（OpenAI API 兼容）
- Embedding：三档可切换（local BGE-M3 子进程 / api OpenAI 兼容 / disabled 零向量）
- CDC：Watchdog 监听本地文件 / Kafka 消费数据库变更
- 前端：Vue 3 + Vite + TypeScript + Tailwind CSS

## 目录结构

```
.
├── agents/                  # 4 个核心 Agent
│   ├── doc_parser_agent.py        # 多模态文档解析（10 种扩展名）
│   ├── knowledge_extract_agent.py # NER/RE/事件抽取，跨 chunk 去重
│   ├── qa_agent.py                # 意图分类 + 查询改写 + 答案生成
│   └── knowledge_update_agent.py  # CDC 增量同步，Watchdog/Kafka 双模式
├── orchestrator/            # LangGraph 三流水线编排（入库/问答/更新）
├── services/                # 服务层
│   ├── embedding.py               # 三档 Embedding 统一出口
│   ├── vector_store.py            # ChromaDB/PGVector 双后端
│   ├── graph_rag.py               # 6 步混合检索管道
│   ├── knowledge_graph.py         # Neo4j 操作
│   ├── cdc_processor.py           # CDC 差量计算 + 版本追踪
│   └── embedding_worker.py        # local 档子进程 worker
├── api/                     # FastAPI 入口
├── config/                  # 配置（pydantic-settings）
├── frontend/                # Vue 3 前端（Vite + TS + Tailwind）
│   ├── src/
│   │   ├── api/                   # 类型化 API 调用
│   │   ├── views/                 # QA / Graph / Upload / Update / Dashboard
│   │   ├── router/
│   │   └── App.vue
│   └── dist/                      # 构建产物，FastAPI 直接挂载
├── tests/                   # 单测 + 评测脚本 + 基准脚本
│   ├── agents/                    # 4 个 Agent 单测
│   ├── services/                  # vector_store / graph_rag / cdc_processor 单测
│   ├── eval/f1_compare.py         # 纯向量 vs GraphRAG F1 对比
│   ├── bench/update_benchmark.py  # 全量重建 vs CDC 增量基准
│   └── ...
├── Dockerfile
├── requirements.txt
└── pytest.ini
```

## 快速开始

### 1. 安装依赖

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> local 档 embedding 需要额外装 torch + sentence-transformers（已在 requirements.txt 中）。
> PDF/图片解析需要系统级依赖：`tesseract-ocr`、`tesseract-ocr-chi-sim`、`poppler-utils`。

### 2. 配置

复制 `.env.example` 为 `.env`，填入实际密钥和连接信息：

```bash
cp .env.example .env
```

关键配置项：

| 配置 | 说明 | 取值 |
|------|------|------|
| `OPENAI_API_KEY` | LLM 密钥（DeepSeek 兼容） | 必填 |
| `OPENAI_BASE_URL` | LLM 接口地址 | DeepSeek: `https://api.deepseek.com/v1` |
| `OPENAI_MODEL` | 模型名 | `deepseek-chat` |
| `EMBEDDING_PROVIDER` | Embedding 档位 | `local` / `api` / `disabled` |
| `EMBEDDING_LOCAL_MODEL` | local 档模型名 | `BAAI/bge-m3` |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | Neo4j 连接 | 连不上时图谱检索自动降级 |
| `VECTOR_STORE_TYPE` | 向量后端 | `chroma` / `pgvector` |
| `KAFKA_BOOTSTRAP_SERVERS` | Kafka 地址（CDC 用） | 不填则只能用 Watchdog 监听 |

### 3. 启动服务

**开发模式**（前后端分离，前端热更新）：

```bash
# 终端 1：后端
EMBEDDING_PROVIDER=disabled uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 终端 2：前端（Vite dev server，自动代理 /api 到 8000）
cd frontend && npm run dev
```

打开 `http://localhost:5173`，前端改动热更新，后端 API 走 Vite 代理。

**生产模式**（构建后单端口）：

```bash
cd frontend && npm run build      # 产物到 frontend/dist/
cd .. && uvicorn api.main:app --host 0.0.0.0 --port 8080
```

打开 `http://localhost:8080`，FastAPI 直接挂载 `frontend/dist/`，SPA history fallback 由后端处理。

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| GET | `/api/admin/stats` | 向量库 + 图谱统计 |
| GET | `/api/graph/data` | 图谱实体+关系（前端可视化用） |
| POST | `/api/ingest/upload` | 上传单文档入库 |
| POST | `/api/ingest/batch` | 批量上传 |
| POST | `/api/qa/ask` | 智能问答（非流式，JSON 响应） |
| POST | `/api/qa/ask_stream` | 智能问答（SSE 流式，meta→token→done） |
| POST | `/api/admin/update` | 触发增量更新 |

## 测试与评测

### 单元测试

```bash
EMBEDDING_PROVIDER=disabled pytest tests/ -v
```

覆盖 4 个 Agent、3 个核心 service、orchestrator 编排、API 端到端 smoke test（87 个测试，含 SSE 流式协议验证）。

### F1 评测（纯向量 RAG vs GraphRAG）

```bash
# 降级模式（不调 LLM，规则抽实体，开箱即用）
python -m tests.eval.f1_compare

# 真 LLM 模式（需 DeepSeek 充值，不可用时自动回退降级）
EVAL_USE_LLM=1 python -m tests.eval.f1_compare
```

输出 recall@k、precision@k、f1 三项指标的对比表，以及逐题命中情况。

### 增量更新基准

```bash
python -m tests.bench.update_benchmark

# 调整文档规模
BENCH_DOC_COUNT=100 python -m tests.bench.update_benchmark
```

对比全量重建与 CDC 增量更新的耗时、吞吐和效率提升。

当前 `/api/admin/update` 手动触发只传文件路径和变更类型；`modified` 没有 before/after 快照时会回退为删旧建新。`DocumentChange` 带文本快照或 chunk hash 快照时，`KnowledgeUpdateAgent` 会调用 `CDCProcessor.compute_diff()`：小于等于 30% 的改动只重写受影响 chunk，超过 30% 走删旧建新。

## Embedding 三档策略

通过 `EMBEDDING_PROVIDER` 切换，上层检索逻辑无感知，只改 `.env` 不改代码：

| 档位 | 走什么 | 适用场景 |
|------|--------|---------|
| `local` | 子进程 + sentence-transformers + BGE-M3 | 企业私有化、无网络、数据不出域 |
| `api` | OpenAI 兼容 API | 开发演示、不想下 2GB 模型 |
| `disabled` | 返回零向量，检索降级 | 单测、CI、没装 torch 的环境 |

> local 档保留子进程隔离：PyTorch C 扩展在 asyncio 环境会 segfault，必须隔离。

## GraphRAG 6 步检索管道

`services/graph_rag.py` 中的 `GraphRAGPipeline.retrieve()` 按序执行：

1. 向量检索（vector）
2. 实体链接（entity_linking）
3. 子图扩展（subgraph，2 跳）
4. 路径推理（path，最短路径）
5. 社区摘要（community_summary）
6. 交叉重排（cross_rerank，来源权重 × doc_type 权重）

问答时 `qa_agent` 把检索委托给这条管道，自身专注意图分类、查询改写和答案生成。

## Docker

```bash
cp .env.example .env
# 填入 OPENAI_API_KEY；演示时可先保留 EMBEDDING_PROVIDER=disabled
docker compose up --build
```

启动后访问（默认端口；如果 `.env` 改了 `API_PORT`，以实际映射端口为准）：

- Web/API: `http://localhost:8080`
- Neo4j Browser: `http://localhost:7474`

Compose 会启动 Neo4j 5 和 API。ChromaDB 使用当前代码里的 embedded `PersistentClient`，数据挂载到宿主机 `./chroma_data`；上传文件挂载到 `./uploads`，方便演示后查看入库文件。

只构建 API 镜像时也可以单独跑：

```bash
docker build -t knowledgeforge .
docker run -p 8080:8080 --env-file .env knowledgeforge
```
