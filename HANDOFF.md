# HANDOFF.md — 交接给新窗口助手

> 这个文件是写给"下一个我"的。如果你是新开窗口的助手，先读这个，再读 AGENTS.md 和 重构计划.md。

## 项目是什么

**KnowledgeForge** — 企业私有知识库管理系统，个人简历项目 + 后期生产可用。
从文档自动入库 → 知识抽取 → GraphRAG 混合检索问答 → CDC 增量更新的知识生命周期管理。

## 当前状态（2026-06-24）

重构阶段0-5 全部完成，代码在 `~/Desktop/KnowledgeForge/`。

**已完成**：
- ✅ 阶段0 搭骨架（从原 code/python/ 复制到 code/refactor/，现已挪到项目根）
- ✅ 阶段1 修 P0（vector_store 真调 chromadb，except:pass 改 raise/log）
- ✅ 阶段2 接 GraphRAG 6 步管道（qa_agent 检索委托给 GraphRAGPipeline）
- ✅ 阶段3 增量更新稳定性（Watchdog asyncio 并发 bug 修复）
- ✅ 阶段4 去 AI 味（13 文件全部重写）
- ✅ 阶段5 补测试 + 评测脚本（83 测试全绿 + F1 评测 + 基准脚本）

## 验证状态

- **pytest**：83 个测试全过（覆盖 4 Agent + 3 services + orchestrator + API smoke）
- **F1 评测**：`python -m tests.eval.f1_compare` 跑出对比数字（GraphRAG recall 稳定 100%，f1 +70%+）
- **基准测试**：`python -m tests.bench.update_benchmark` 跑出对比数字（CDC 增量 vs 全量重建，效率提升 92-97%）
- 服务能启动，/api/health 返回 200
- 端到端 LLM 调用仍需 DeepSeek 充值（评测脚本默认降级模式不依赖 LLM）

## 新窗口怎么接上下文

按这个顺序读文件（都在项目根目录）：

1. **本文件**（HANDOFF.md）— 你现在在读
2. **`.workbuddy/memory/MEMORY.md`** — 项目长期记忆，技术栈和架构全貌
3. **`AGENTS.md`** — 重构行为准则，**硬规则不能违反**（技术栈锁定、去 AI 味 9 条、禁止事项）
4. **`重构计划.md`** — 6 阶段执行计划（已全部完成）
5. **`.workbuddy/memory/2026-06-24.md`** — 当天完整工作日志

## 关键约束（来自 AGENTS.md，不能违反）

- **技术栈锁定**：ChromaDB/Neo4j/DeepSeek 不换，不引入新框架
- **去 AI 味 9 条硬规则**：注释只写"为什么"不写"是什么"；禁装饰分隔条；禁营销话；docstring≤2行；中文为主；英文 docstring 一律改中文
- **不动原工程**：`~/Desktop/企业私有知识库管理调度智能体v2/` 是历史参考，只看不动
- **不硬编码简历数字**（F1+22%、效率+95%）到代码或测试断言

## 测试与评测脚本

### 跑测试
```bash
EMBEDDING_PROVIDER=disabled /Users/papa/.workbuddy/binaries/python/envs/refactor/bin/python3 -m pytest tests/ -v
```

### F1 评测（纯向量 vs GraphRAG 检索质量）
```bash
# 降级模式（默认，不调 LLM）
python -m tests.eval.f1_compare
# 真 LLM 模式（需 DeepSeek 充值，不可用时自动回退降级）
EVAL_USE_LLM=1 python -m tests.eval.f1_compare
```

### 增量更新基准（全量重建 vs CDC 增量）
```bash
python -m tests.bench.update_benchmark
# 调整文档规模
BENCH_DOC_COUNT=100 python -m tests.bench.update_benchmark
```

### 测试目录结构
```
tests/
├── conftest.py              # 公共夹具：mock_llm / FakeVectorStore / FakeKnowledgeGraph / DeterministicEmbeddings
├── agents/                  # 4 个 Agent 单测（39 个）
├── services/                # vector_store / graph_rag / cdc_processor 单测（30 个）
├── test_orchestrator.py     # 三流水线编排测试
├── test_api_smoke.py        # API 端到端 smoke test
├── test_settings.py         # 配置测试（原有）
├── test_ingest_upload.py    # 上传测试（原有）
├── eval/
│   ├── in_memory_graph.py   # 内存版知识图谱（BFS 子图遍历 + 最短路径），评测用
│   └── f1_compare.py        # F1 评测脚本
└── bench/
    └── update_benchmark.py  # 增量更新基准脚本
```

## venv 路径

`/Users/papa/.workbuddy/binaries/python/envs/refactor/bin/python3`
依赖已装齐（torch + sentence-transformers + chromadb + langchain + langgraph 等）

## 简历承诺兑现对照（全部已兑现到代码）

| 简历承诺 | 代码位置 |
|---------|---------|
| 4 Agent 分工 | `agents/*.py` |
| LangGraph 三条流水线 | `orchestrator/graph.py` |
| 多模态文档入库（9种格式） | `agents/doc_parser_agent.py` |
| 知识抽取+跨chunk去重 | `agents/knowledge_extract_agent.py` `_deduplicate` |
| GraphRAG 6步管道 | `services/graph_rag.py` + `agents/qa_agent.py` 接线 |
| 多权重重排（含多模态权重） | `services/graph_rag.py` `_cross_rerank` |
| Embedding 可插拔 | `services/embedding.py` 三档 |
| CDC 增量更新 | `services/cdc_processor.py` + `agents/knowledge_update_agent.py` |
| Cypher 参数化防注入 | `services/graph_rag.py` + `knowledge_graph.py` 白名单 |
| F1 提升对比 | `tests/eval/f1_compare.py`（跑出真实数字，未硬编码） |
| 效率提升对比 | `tests/bench/update_benchmark.py`（跑出真实数字，未硬编码） |
