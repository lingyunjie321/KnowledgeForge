"""增量更新基准脚本：全量重建 vs CDC 增量更新耗时对比

用法：python -m tests.bench.update_benchmark
默认 N=50 篇文档，可用 BENCH_DOC_COUNT 环境变量调规模。
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock

from agents.doc_parser_agent import DocType, DocumentChunk
from agents.knowledge_update_agent import ChangeType, DocumentChange, KnowledgeUpdateAgent
from services.vector_store import VectorStoreService
from tests.conftest import DeterministicEmbeddings, FakeKnowledgeGraph


def _make_chunks(doc_id: str, count: int, prefix: str = "内容") -> list[DocumentChunk]:
    return [
        DocumentChunk(
            content=f"{prefix} {doc_id} chunk {i} " + "字" * 60,
            doc_id=doc_id,
            chunk_index=i,
            doc_type=DocType.TEXT,
            metadata={"source": f"{doc_id}.txt"},
        )
        for i in range(count)
    ]


def _make_mock_parser(chunks_per_doc: int = 3):
    """mock doc_parser，parse 直接返回预制 chunks，跳过真实文件 IO 和 LLM"""
    parser = MagicMock()

    async def _parse(file_path: str) -> list[DocumentChunk]:
        doc_id = file_path.rsplit("/", 1)[-1].replace(".txt", "")
        return _make_chunks(doc_id, chunks_per_doc)

    parser.parse = AsyncMock(side_effect=_parse)
    return parser


async def bench_full_rebuild(vs: VectorStoreService, n_docs: int, chunks_per_doc: int) -> float:
    """全量重建：N 个文档从零入库"""
    start = time.perf_counter()
    for i in range(n_docs):
        chunks = _make_chunks(f"doc{i}", chunks_per_doc)
        await vs.add_chunks(chunks)
    return time.perf_counter() - start


async def bench_incremental_update(
    vs: VectorStoreService,
    kg: FakeKnowledgeGraph,
    target_doc_id: str,
    chunks_per_doc: int,
) -> float:
    """CDC 增量更新：修改 1 个已入库文档，delete 旧 + add 新"""
    parser = _make_mock_parser(chunks_per_doc)
    extractor = MagicMock()
    extractor.extract = AsyncMock(return_value=[])

    agent = KnowledgeUpdateAgent(
        doc_parser=parser,
        knowledge_extractor=extractor,
        vector_store=vs,
        knowledge_graph=kg,
    )
    change = DocumentChange(file_path=f"/bench/{target_doc_id}.txt", change_type=ChangeType.MODIFIED)
    start = time.perf_counter()
    await agent.process_change(change)
    return time.perf_counter() - start


def _print_result(n_docs: int, chunks_per_doc: int, full_time: float, incr_time: float) -> None:
    total_chunks = n_docs * chunks_per_doc
    speedup = full_time / incr_time if incr_time > 0 else float("inf")
    efficiency_gain = (1 - incr_time / full_time) * 100 if full_time > 0 else 0.0

    print(f"\n文档规模: {n_docs} 篇 × {chunks_per_doc} chunks/篇 = {total_chunks} chunks")
    print(f"{'场景':<20}{'耗时(秒)':<14}{'吞吐(chunks/s)':<18}")
    print("-" * 52)
    print(f"{'全量重建':<20}{full_time:<14.4f}{total_chunks / full_time:<18.1f}")
    print(f"{'增量更新(1篇)':<20}{incr_time:<14.4f}{chunks_per_doc / incr_time:<18.1f}")
    print("-" * 52)
    print(f"加速比: {speedup:.1f}x")
    print(f"效率提升: {efficiency_gain:.1f}%  (增量更新耗时仅为全量的 {100 - efficiency_gain:.1f}%)")


async def main() -> int:
    n_docs = int(os.environ.get("BENCH_DOC_COUNT", "50"))
    chunks_per_doc = 3

    print(f"基准测试: 全量重建 vs CDC 增量更新")
    print(f"向量后端: chromadb (本地持久化), embedding: 确定性哈希向量")

    vs = VectorStoreService()
    vs._embeddings = DeterministicEmbeddings()
    await vs.init()
    kg = FakeKnowledgeGraph()

    # 先全量入库 N 篇，作为增量更新的基线
    print(f"阶段1: 全量重建 {n_docs} 篇文档...")
    full_time = await bench_full_rebuild(vs, n_docs, chunks_per_doc)

    # 修改其中 1 篇，触发增量更新
    target = f"doc{n_docs // 2}"
    print(f"阶段2: 增量更新 1 篇文档 ({target})...")
    incr_time = await bench_incremental_update(vs, kg, target, chunks_per_doc)

    _print_result(n_docs, chunks_per_doc, full_time, incr_time)

    # 多次增量取平均，减少抖动
    runs = 3
    print(f"\n稳定性: 增量更新连续 {runs} 次取平均")
    times = []
    for r in range(runs):
        t = await bench_incremental_update(vs, kg, f"doc{r}", chunks_per_doc)
        times.append(t)
    avg_incr = sum(times) / len(times)
    print(f"  单次: {['%.4f' % t for t in times]}  平均: {avg_incr:.4f}s")
    print(f"  对应全量 {full_time:.4f}s, 效率提升 {(1 - avg_incr / full_time) * 100:.1f}%")

    await kg.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
