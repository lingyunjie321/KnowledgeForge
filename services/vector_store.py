"""向量存储服务，支持 ChromaDB / PGVector 双后端"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_openai import OpenAIEmbeddings

from agents.doc_parser_agent import DocumentChunk
from config import settings
from services.embedding import create_embeddings

logger = logging.getLogger(__name__)


class VectorStoreService:
    """向量库统一接口，底层可切换 ChromaDB / PGVector"""

    COLLECTION_NAME = "knowledge_chunks"

    def __init__(self) -> None:
        self._embeddings: Any = None
        self._store: Any = None
        self._backend = settings.vector_store_type
        # chromadb 的 C 扩展在 asyncio 事件循环里会 segfault，必须放到线程池
        self._executor = ThreadPoolExecutor(max_workers=2)

    async def _run_sync(self, fn, *args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, lambda: fn(*args, **kwargs))

    @property
    def embeddings(self):
        if self._embeddings is None:
            try:
                self._embeddings = create_embeddings()
            except Exception:
                logger.exception("embedding 初始化失败")
                self._embeddings = None
        return self._embeddings

    @property
    def embeddings_available(self) -> bool:
        return self.embeddings is not None

    async def init(self) -> None:
        if self._backend == "chroma":
            await self._init_chroma()
        else:
            await self._init_pgvector()

    async def _init_chroma(self) -> None:
        def _init():
            import chromadb
            persist_dir = os.path.join(settings.upload_dir, "..", "chroma_data")
            os.makedirs(persist_dir, exist_ok=True)
            client = chromadb.PersistentClient(path=persist_dir)
            return client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        self._store = await self._run_sync(_init)

    async def _init_pgvector(self) -> None:
        from langchain_community.vectorstores import PGVector
        self._store = PGVector(
            connection_string=settings.pgvector_dsn,
            collection_name=self.COLLECTION_NAME,
            embedding_function=self.embeddings,
        )

    async def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        if not chunks or not self.embeddings_available:
            return 0

        texts = [c.content for c in chunks]
        metadatas = [
            {
                "doc_id": c.doc_id,
                "chunk_id": c.chunk_id,
                "source": c.metadata.get("source", ""),
                "doc_type": c.doc_type.value,
            }
            for c in chunks
        ]
        ids = [c.chunk_id for c in chunks]

        if self._backend == "chroma":
            # 先算向量，再丢给 chromadb；两步都走线程池，避免 C 扩展 segfault
            vectors = await self._run_sync(self.embeddings.embed_documents, texts)
            await self._run_sync(
                self._store.add,
                embeddings=vectors,
                documents=texts,
                metadatas=metadatas,
                ids=ids,
            )
            return len(chunks)

        # pgvector 走 langchain 的 async 接口
        await self._store.aadd_texts(texts=texts, metadatas=metadatas, ids=ids)
        return len(chunks)

    async def search(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        if not self.embeddings_available:
            return []

        if self._backend == "chroma":
            query_vec = await self._run_sync(self.embeddings.embed_query, query)
            raw = await self._run_sync(
                self._store.query,
                query_embeddings=[query_vec],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            results: list[tuple[dict, float]] = []
            docs = raw.get("documents", [[]])[0]
            metas = raw.get("metadatas", [[]])[0]
            dists = raw.get("distances", [[]])[0]
            for doc, meta, dist in zip(docs, metas, dists):
                # cosine 距离转相似度
                score = 1.0 - float(dist)
                results.append((
                    {"content": doc, "source": meta.get("source", ""), "metadata": meta},
                    score,
                ))
            return results

        results = await self._store.asimilarity_search_with_score(query, k=top_k)
        return [
            ({"content": doc.page_content, "source": doc.metadata.get("source", ""), "metadata": doc.metadata}, score)
            for doc, score in results
        ]

    async def delete_by_doc_id(self, doc_id: str) -> int:
        if self._backend == "chroma":
            existing = await self._run_sync(self._store.get, where={"doc_id": doc_id}, include=[])
            ids = existing.get("ids", [])
            if ids:
                await self._run_sync(self._store.delete, ids=ids)
            return len(ids)
        return 0

    async def delete_chunks(self, chunk_ids: list[str]) -> int:
        if not chunk_ids:
            return 0
        if self._backend == "chroma":
            existing = await self._run_sync(self._store.get, ids=chunk_ids, include=[])
            ids = existing.get("ids", [])
            if ids:
                await self._run_sync(self._store.delete, ids=ids)
            return len(ids)
        return 0

    async def get_stats(self) -> dict:
        if self._backend == "chroma":
            if self._store is None:
                return {"backend": "chroma", "total_vectors": 0, "collection": self.COLLECTION_NAME}
            count = await self._run_sync(self._store.count)
            return {"backend": "chroma", "total_vectors": count, "collection": self.COLLECTION_NAME}
        return {"backend": "pgvector", "collection": self.COLLECTION_NAME}
