# AGENTS.md — 重构行为准则

> 本文件约束 `code/refactor/` 的重构工作。每次动手前必读，每完成一个阶段对照验收。

## 项目定位

把 `code/python/` 演示版重构到 `code/refactor/`，目标是个人简历项目 + 后期生产可用。简历每条技术承诺必须能在 `code/refactor/` 里指出对应代码、能跑、能演示。

## 不可改的约束（技术栈锁定）

- Python 3.12 + FastAPI + LangGraph + LangChain
- 向量库：ChromaDB（主）/ PGVector（备），**不引入 Milvus/Qdrant/Weaviate**
- 图库：Neo4j，**不换**
- LLM：DeepSeek（OpenAI API 兼容），**不接其他模型**
- Embedding：三档可切换（见下"Embedding 策略"），**不引入新 embedding 服务**
- 不引入新框架/新数据库/新中间件，除非用户明确同意

## Embedding 策略（三档可切换）

通过 `EMBEDDING_PROVIDER` 配置项切换，上层检索逻辑无感知：

| 档位 | 走什么 | 适用场景 |
|------|--------|---------|
| `local` | 子进程 + sentence-transformers + BGE-M3 | 企业私有化、无网络、数据不出域 |
| `api` | OpenAI 兼容 API（OpenAI/DeepSeek/任意兼容服务） | 开发演示、不想下 2GB 模型 |
| `disabled` | 返回零向量，检索降级 | 单测、CI、没装 torch 的环境 |

实现要求：
- `services/embedding.py` 统一出口，三个类（LocalEmbeddings / ApiEmbeddings / NullEmbeddings）实现同一接口
- `local` 档保留子进程隔离（PyTorch + asyncio segfault 是真的，不能去掉）
- `api` 档不需要子进程（OpenAI SDK async-safe），主进程直跑
- 切换 provider 只改 `.env`，不改代码

## 代码风格硬规则（去 AI 味）

这是用户最在意的事，违反即返工：

1. **注释只写"为什么"，不写"是什么"**。`# 用线程池绕开 chromadb async segfault` ✓；`# 初始化向量库` ✗
2. **docstring 最多 2 行**，说清职责即可，不写背景/工作流/核心技术亮点
3. **禁用装饰分隔条**：`# ── Step 1: xxx ───`、`# === 配置 ===`、`# ------` 全删
4. **禁用营销话**："本项目核心技术亮点"、"这是核心能力之一"、"高效智能"全删
5. **中文为主**，技术术语（embedding/Cypher/chunk）保留英文
6. **不写"工作流: A → B → C"**，逻辑靠代码结构表达，不靠注释复述
7. **dataclass 慎用**，能用 NamedTuple/dict 就别造类
8. **英文 docstring 一律改中文**（原代码混杂大量英文）
9. **函数/类命名表意即可**，不加 `Helper`/`Manager`/`Service` 等无意义后缀（已有的保留，不新增）

## 禁止事项

- ❌ 引入新框架/库/数据库（除非用户同意）
- ❌ 删除现有功能模块来"简化"
- ❌ 用 mock 假装功能跑通——P0 修不好就报错，不糊弄
- ❌ 跳阶段：必须阶段0→1→2→3→4→5 顺序推进，每阶段验收通过才进下一步
- ❌ 改动 `code/python/` 原工程——只动 `code/refactor/`
- ❌ 新写大段英文注释或 docstring
- ❌ 把简历数字（F1+22%、效率+95%）硬编码进代码或测试断言

## multimodal.py 处理决定：合并

`services/multimodal.py` 是孤儿模块（无人调用），但其 `weighted_rerank` 逻辑（按 doc_type 加权）有价值。处理方式：

- **删掉 `services/multimodal.py` 文件**
- **把 doc_type 权重逻辑搬进 `services/graph_rag.py` 的 `_cross_rerank`**（rerank 统一在一处）
- **删除 `services/__init__.py` 里的 MultimodalService 导出**
- 简历"多模态文档入库"+"检索结果加权重排"两条都还能讲，代码里能指出 `_cross_rerank` 的权重逻辑

## P0 优先级（不修不能往下走）

1. `services/vector_store.py` — `search`/`add_chunks` 真正调 chromadb，不再 `return []`
2. `services/graph_rag.py` + `agents/qa_agent.py` — 6 步管道接线，QA 检索委托给 GraphRAGPipeline
3. `api/main.py` + `orchestrator/graph.py` — 启动期/入库期的 `except: pass` 改成 raise 或 warning+计数

## 工作流约定

- **每阶段做完必验证**：按重构计划里该阶段的"验收"项逐条对照
- **不积累改动**：一个文件改完先自检风格规则，再改下一个
- **遇到开放问题停下问用户**，不自作主张（见重构计划第七节）
- **风格返工优先于新功能**：发现 AI 味立刻清理，不留到最后

## 完成验收（全部满足才算重构完成）

- [ ] `code/refactor/` 能 `pip install` + 启动 + 上传文档 + 问答，全程不报错
- [ ] 问答日志显示 vector/subgraph/path/community 四种检索来源
- [ ] `pytest` 全绿，覆盖 4 个 Agent 核心路径
- [ ] 评测脚本能跑出纯向量 vs GraphRAG 的对比数字
- [ ] 肉眼扫代码无大段英文注释、无营销话、无装饰分隔条
- [ ] 简历每条技术承诺都能在 `code/refactor/` 指出对应代码位置
