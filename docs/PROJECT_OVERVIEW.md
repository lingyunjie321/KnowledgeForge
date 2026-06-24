# KnowledgeForge 项目整体文档

> 面向第一次接触本项目的人，5 分钟读完，知道这是什么、解决什么问题、怎么做出来的。

## 1. 项目定位

KnowledgeForge 是一个企业私有知识库管理系统，把企业内部的文档（PDF / 图片 / 表格 / 纯文本 / Markdown）自动入库，抽取知识构建图谱，对外提供基于 GraphRAG 混合检索的智能问答，并通过 CDC 机制保证知识库随源文档更新而增量同步。

一句话定位：**从文档入库 → 知识抽取 → GraphRAG 检索问答 → CDC 增量更新的完整知识生命周期管理系统**。

## 2. 要解决的问题

| 痛点 | 传统方案 | KnowledgeForge 的做法 |
|------|---------|----------------------|
| 企业文档散落各处，问答无据可依 | 全文搜索 + 人工翻文档 | 文档自动入库 + 知识图谱 + 混合检索 |
| 多跳推理答不出来（"张三的同事在哪个城市") | 纯向量检索，召不回跨文档关联 | GraphRAG：向量 + 实体链接 + 子图 + 路径推理 + 社区摘要 |
| 文档更新后知识库过时 | 定时全量重建，慢且费 | CDC 增量更新，Watchdog / Kafka 双模式 |
| 私有化部署要求高 | 公有云 SaaS，数据出域 | 三档可切换 Embedding，local 档支持完全离线 |
| RAG 答案来源不可追溯 | 黑盒 LLM 回答 | 每条答案带 4 种检索来源标注（vector/subgraph/path/community） |

## 3. 核心能力

### 3.1 多模态文档入库
- 9 种扩展名：`.pdf .png .jpg .jpeg .csv .xlsx .xls .txt .md`
- PDF：PyPDF2 提文字，失败自动降级到 pdf2image + LLM 视觉描述
- 图片：pytesseract OCR + LLM 视觉描述双路
- 表格：CSV / Excel 按行展开成结构化文本
- 文本 / Markdown：直接读
- 统一 chunk 策略：512 字符窗口 + 64 字符重叠

### 3.2 知识抽取
- 三元组抽取：实体 / 关系 / 事件
- 实体类型 8 种：Person / Organization / Location / Product / Technology / Concept / Event / Time
- 关系类型 8 种白名单：BELONGS_TO / WORKS_AT / LOCATED_IN / DEVELOPED_BY / RELATED_TO / PART_OF / USES / DEPENDS_ON
- 跨 chunk 去重：同名同类型实体合并，同三元组去重

### 3.3 GraphRAG 6 步混合检索
这是项目的核心技术亮点，6 步管道：

```
1. 向量检索      → 语义相似的文档块
2. 实体链接      → LLM 从问题中抽实体名
3. 子图检索      → Neo4j 多跳邻居遍历
4. 路径推理      → shortestPath 找实体间关联链
5. 社区摘要      → LLM 对子图生成结构化摘要
6. 交叉重排      → 来源权重 × 文档类型权重双权重打分
```

权重设计：
- 来源权重：vector 1.0 / subgraph 1.15 / path 1.25 / community 1.1
- 文档类型权重：text/markdown 1.0 / pdf 0.95 / table 0.9 / image 0.85

### 3.4 三档可切换 Embedding
通过 `EMBEDDING_PROVIDER` 环境变量切换，上层检索逻辑无感知：

| 档位 | 实现 | 适用场景 |
|------|------|---------|
| `local` | 子进程 + sentence-transformers + BGE-M3 | 企业私有化、数据不出域 |
| `api` | OpenAI 兼容 API（主进程直跑） | 开发演示、不想下 2GB 模型 |
| `disabled` | 返回零向量，检索降级 | 单测、CI、没装 torch 的环境 |

### 3.5 CDC 增量更新
- Watchdog 模式：监听本地文件系统变化（created/modified/deleted）
- Kafka 模式：消费 CDC 事件，兼容 Debezium 格式
- 差量阈值：变化超过 30% 触发全量重建，否则增量更新
- 修改处理：删旧 + 建新（先按 doc_id 删向量，再重新解析入库）

## 4. 技术栈

| 层 | 技术 | 版本要求 |
|----|------|---------|
| 语言 | Python | 3.12+ |
| Web 框架 | FastAPI | ≥0.115 |
| ASGI | uvicorn | ≥0.34 |
| Agent 编排 | LangGraph | ≥0.3 |
| LLM 框架 | LangChain | ≥0.3 |
| LLM | DeepSeek（OpenAI API 兼容） | - |
| 向量库 | ChromaDB（主）/ PGVector（备） | ≥0.6 / ≥0.3 |
| 图库 | Neo4j | ≥5.28 |
| Embedding（local 档） | sentence-transformers + BGE-M3 | ≥3.0 |
| OCR | pytesseract + tesseract-ocr-chi-sim | ≥0.3.13 |
| PDF | PyPDF2 + pdf2image | ≥3.0 / ≥1.17 |
| 配置 | pydantic-settings | ≥2.7 |
| 文件监听 | watchdog | ≥6.0 |
| 消息队列 | confluent-kafka | ≥2.6 |

## 5. 项目结构（顶层）

```
KnowledgeForge/
├── agents/          4 个核心 Agent
├── services/        7 个服务模块（向量库 / 图谱 / GraphRAG / Embedding / CDC）
├── orchestrator/    LangGraph 编排引擎
├── api/             FastAPI 入口
├── config/          配置加载
├── utils/           工具函数
├── static/          前端三件套（HTML/CSS/JS）
├── tests/           单测 + F1 评测脚本 + 增量更新基准脚本
├── docs/            项目文档（本目录）
├── Dockerfile       容器化部署
├── requirements.txt 依赖锁定
├── pytest.ini       pytest 配置
└── .env.example     环境变量模板
```

详细文件树见 [FILE_TREE.md](./FILE_TREE.md)。

## 6. 使用场景

### 6.1 企业内部知识管理
- 上传产品手册、技术文档、会议纪要
- 员工提问："我们公司的 X 产品用了哪些开源组件？" → GraphRAG 通过 USES/DEPENDS_ON 关系链答出来

### 6.2 私有化部署
- 数据不出企业内网
- local 档 embedding 完全离线，不依赖外部 API
- Neo4j 和 ChromaDB 都可本地部署

### 6.3 文档高频更新场景
- CDC 监听文档目录，文档改了自动增量更新
- 不用人工触发全量重建

## 7. 与同类系统差异

| 维度 | 纯向量 RAG | LangChain 现成 RAG | KnowledgeForge |
|------|-----------|-------------------|---------------|
| 检索 | 仅向量 | 向量 + 关键词 | 向量 + 实体链接 + 子图 + 路径 + 社区 + 重排 |
| 多跳推理 | 不支持 | 不支持 | shortestPath 路径推理 |
| 知识结构 | 无 | 无 | Neo4j 图谱，实体有版本号 |
| 更新机制 | 全量重建 | 全量重建 | CDC 增量更新 |
| 来源标注 | 有 | 有 | 4 种检索来源分别标注 |
| Embedding | 固定一种 | 固定一种 | 三档可切换 |
| 私有化 | 难 | 难 | local 档完全离线 |

## 8. 快速开始

```bash
# 1. 装依赖
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. 配置
cp .env.example .env
# 编辑 .env，至少填 OPENAI_API_KEY（DeepSeek 的 key）
# EMBEDDING_PROVIDER 默认 disabled，生产建议改 local 或 api

# 3. 启动 Neo4j（可选，不连图谱检索降级）
docker run -d -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5

# 4. 启动服务
python -m api.main
# 或
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload

# 5. 打开前端
# 浏览器访问 http://localhost:8080
```

## 9. 关键约束（重构期定下，不可违反）

1. 技术栈锁定：不引入 Milvus/Qdrant/Weaviate，不换 Neo4j，不接其他 LLM
2. Embedding 只有三档，不引入新 embedding 服务
3. 不用 mock 假装功能跑通，P0 修不好就报错
4. 简历数字（F1 +22%、效率 +95%）不硬编码进代码或测试断言

## 10. 当前状态

- 阶段 0-5 重构全部完成
- pytest 83 个测试全绿
- F1 评测脚本可跑出对比数字（GraphRAG recall 100%，纯向量 67-83%）
- 增量更新基准脚本可跑出对比数字（效率提升 92-97%）
- 端到端链路代码层全通，唯一前置条件：DeepSeek 充值后可跑真 LLM 评测

## 相关文档

- [新手代码导读](./CODE_GUIDE.md)
- [文档入库流程](./INGESTION_FLOW.md)
- [文档入库代码追踪](./INGESTION_CODE_TRACE.md)
- [项目架构设计详解](./ARCHITECTURE.md)
- [项目文件树](./FILE_TREE.md)
