"""Embedding 子进程 worker，隔离 PyTorch 避免主进程 segfault"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import queue
import sys
from typing import Any

# spawn 模式不能直接传对象，模型名和 device 由主进程通过环境变量传入
_MODEL_NAME = os.environ.get("EMBEDDING_LOCAL_MODEL", "BAAI/bge-m3")
_DEVICE = os.environ.get("EMBEDDING_LOCAL_DEVICE", "cpu")
_SHUTDOWN_TIMEOUT = 30


def _worker_process(request_queue: multiprocessing.Queue, response_queue: multiprocessing.Queue):
    """加载模型并循环处理 encode 请求"""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(_MODEL_NAME, device=_DEVICE)
        model.encode("warmup", show_progress_bar=False)
        print(f"[embedding_worker] 模型已加载: {_MODEL_NAME}", flush=True)
    except Exception as e:
        response_queue.put(("error", str(e)))
        return

    while True:
        try:
            msg = request_queue.get(timeout=1)
        except queue.Empty:
            continue

        if msg is None or msg[0] == "shutdown":
            break

        msg_id, texts = msg
        try:
            vectors = model.encode(texts, show_progress_bar=False).tolist()
            response_queue.put((msg_id, vectors))
        except Exception as e:
            response_queue.put((msg_id, f"ERROR: {e}"))


class EmbeddingClient:
    """与子进程通信的客户端"""

    def __init__(self):
        self._request_queue: multiprocessing.Queue | None = None
        self._response_queue: multiprocessing.Queue | None = None
        self._process: multiprocessing.Process | None = None
        self._counter = 0

    def start(self):
        if self._process is not None:
            return
        ctx = multiprocessing.get_context("spawn")
        self._request_queue = ctx.Queue()
        self._response_queue = ctx.Queue()
        self._process = ctx.Process(
            target=_worker_process,
            args=(self._request_queue, self._response_queue),
            daemon=True,
        )
        self._process.start()
        try:
            result = self._response_queue.get(timeout=120)
            if result[0] == "error":
                raise RuntimeError(f"子进程加载模型失败: {result[1]}")
        except queue.Empty:
            raise RuntimeError("embedding 子进程启动超时")

    def stop(self):
        if self._process is None:
            return
        try:
            self._request_queue.put(("shutdown", None))
        except Exception:
            pass
        self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.terminate()
        self._process = None
        self._request_queue = None
        self._response_queue = None

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self._process is None or not self._process.is_alive():
            raise RuntimeError("embedding 子进程未运行")
        self._counter += 1
        msg_id = f"enc_{self._counter}"
        self._request_queue.put((msg_id, texts))
        try:
            msg_id_resp, result = self._response_queue.get(timeout=300)
            if isinstance(result, str) and result.startswith("ERROR:"):
                raise RuntimeError(result)
            return result
        except queue.Empty:
            raise RuntimeError("embedding 子进程响应超时")

    async def aencode(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.encode, texts)

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.is_alive()


_embedding_client: EmbeddingClient | None = None


def get_embedding_client() -> EmbeddingClient | None:
    global _embedding_client
    if os.environ.get("EMBEDDING_PROVIDER") == "disabled":
        return None
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
        try:
            _embedding_client.start()
        except Exception:
            _embedding_client = None
    return _embedding_client
