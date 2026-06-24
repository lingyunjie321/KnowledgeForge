"""Embedding 统一出口，三档可切换：local / api / disabled"""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol

from config import settings

logger = logging.getLogger(__name__)


class Embeddings(Protocol):
    """三档实现都遵守这个接口，上层无感知"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]: ...
    async def aembed_query(self, text: str) -> list[float]: ...


class NullEmbeddings:
    """disabled 档：返回零向量，给 CI/单测用"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 8 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * 8

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


class ApiEmbeddings:
    """api 档：走 OpenAI 兼容 endpoint，主进程直跑"""

    def __init__(self) -> None:
        from langchain_openai import OpenAIEmbeddings
        self._client = OpenAIEmbeddings(
            model=settings.embedding_api_model,
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_api_base,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._client.aembed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await self._client.aembed_query(text)


class LocalEmbeddings:
    """local 档：子进程跑 sentence-transformers，隔离 PyTorch C 扩展的 segfault"""

    def __init__(self) -> None:
        from services.embedding_worker import get_embedding_client
        self._client = get_embedding_client()
        if self._client is None:
            raise RuntimeError("本地 embedding 子进程启动失败")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._client.encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._client.encode([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._client.aencode(texts)

    async def aembed_query(self, text: str) -> list[float]:
        result = await self._client.aencode([text])
        return result[0]


def create_embeddings() -> Any:
    """根据 settings.embedding_provider 返回对应实现，上层无感知"""
    provider = settings.embedding_provider

    if provider == "disabled":
        logger.info("embedding 已禁用（disabled 档），检索将降级")
        return NullEmbeddings()

    if provider == "api":
        logger.info("使用 api 档 embedding: %s", settings.embedding_api_base)
        return ApiEmbeddings()

    if provider == "local":
        logger.info("使用 local 档 embedding: %s", settings.embedding_local_model)
        return LocalEmbeddings()

    raise ValueError(f"未知 embedding_provider: {provider}")
