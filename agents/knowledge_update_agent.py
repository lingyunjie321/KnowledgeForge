"""知识更新 Agent：监听文档变更，增量更新向量库和知识图谱"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from config import settings
from services.cdc_processor import CDCProcessor

logger = logging.getLogger(__name__)


class ChangeType(str, Enum):
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"


@dataclass
class DocumentChange:
    file_path: str
    change_type: ChangeType
    timestamp: float = field(default_factory=time.time)
    old_hash: str = ""
    new_hash: str = ""
    before_content: str = ""
    after_content: str = ""
    before_chunk_hashes: dict[str, str] = field(default_factory=dict)
    after_chunk_hashes: dict[str, str] = field(default_factory=dict)
    diff_chunks: list[str] = field(default_factory=list)
    diff: dict[str, Any] = field(default_factory=dict)


@dataclass
class UpdateResult:
    change: DocumentChange
    vectors_added: int = 0
    vectors_deleted: int = 0
    entities_added: int = 0
    entities_updated: int = 0
    relations_added: int = 0
    diff: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str = ""
    processing_time_ms: float = 0


class KnowledgeUpdateAgent:
    """两种模式：Watchdog 监听本地文件 / Kafka 消费 CDC 事件"""

    # 重试退避参数
    MAX_RETRY = 3
    RETRY_BACKOFF_BASE = 1.0  # 秒，指数退避基数

    def __init__(
        self,
        doc_parser: Any = None,
        knowledge_extractor: Any = None,
        vector_store: Any = None,
        knowledge_graph: Any = None,
    ) -> None:
        self.doc_parser = doc_parser
        self.knowledge_extractor = knowledge_extractor
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self._file_hashes: dict[str, str] = {}
        self._file_contents: dict[str, str] = {}
        self._version_counter: dict[str, int] = {}
        self.cdc_processor = CDCProcessor()

    async def process_change(self, change: DocumentChange) -> UpdateResult:
        start = time.time()
        result = UpdateResult(change=change)

        try:
            if change.change_type == ChangeType.DELETED:
                await self._handle_delete(change, result)
            elif change.change_type == ChangeType.CREATED:
                await self._handle_create(change, result)
            elif change.change_type == ChangeType.MODIFIED:
                await self._handle_modify(change, result)
        except Exception as e:
            result.success = False
            result.error = str(e)
            logger.exception("文档变更处理失败: %s", change.file_path)

        result.processing_time_ms = (time.time() - start) * 1000
        return result

    async def process_batch(self, changes: list[DocumentChange]) -> list[UpdateResult]:
        results: list[UpdateResult] = []
        for change in changes:
            results.append(await self.process_change(change))
        return results

    def detect_changes(self, file_paths: list[str]) -> list[DocumentChange]:
        changes: list[DocumentChange] = []
        current_files = set(file_paths)

        for fp in current_files:
            new_hash = self._compute_hash(fp)
            old_hash = self._file_hashes.get(fp, "")
            new_content = self._read_text(fp)
            old_content = self._file_contents.get(fp, "")

            if not old_hash:
                changes.append(DocumentChange(
                    file_path=fp,
                    change_type=ChangeType.CREATED,
                    new_hash=new_hash,
                    after_content=new_content,
                ))
            elif new_hash != old_hash:
                changes.append(DocumentChange(
                    file_path=fp,
                    change_type=ChangeType.MODIFIED,
                    old_hash=old_hash,
                    new_hash=new_hash,
                    before_content=old_content,
                    after_content=new_content,
                ))
            self._file_hashes[fp] = new_hash
            self._file_contents[fp] = new_content

        for fp in set(self._file_hashes) - current_files:
            changes.append(DocumentChange(
                file_path=fp,
                change_type=ChangeType.DELETED,
                old_hash=self._file_hashes[fp],
                before_content=self._file_contents.get(fp, ""),
            ))
            del self._file_hashes[fp]
            self._file_contents.pop(fp, None)

        return changes

    def start_watching(self, directory: str) -> None:
        """启动 Watchdog 文件监听"""
        import threading
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        agent = self
        # Watchdog 回调在独立线程触发，需要用 asyncio.run_coroutine_threadsafe
        # 把协程丢回主事件循环，否则并发会出错
        loop = asyncio.get_event_loop()
        pending: set = set()

        def _schedule(coro):
            task = asyncio.run_coroutine_threadsafe(coro, loop)
            pending.add(task)
            task.add_done_callback(pending.discard)

        class _Handler(FileSystemEventHandler):
            def on_created(self, event):
                if not event.is_directory:
                    _schedule(agent.process_change(
                        DocumentChange(file_path=event.src_path, change_type=ChangeType.CREATED)
                    ))

            def on_modified(self, event):
                if not event.is_directory:
                    _schedule(agent.process_change(
                        DocumentChange(file_path=event.src_path, change_type=ChangeType.MODIFIED)
                    ))

            def on_deleted(self, event):
                if not event.is_directory:
                    _schedule(agent.process_change(
                        DocumentChange(file_path=event.src_path, change_type=ChangeType.DELETED)
                    ))

        observer = Observer()
        observer.schedule(_Handler(), directory, recursive=True)

        def _run():
            observer.start()
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                observer.stop()
            observer.join()

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    async def start_kafka_consumer(self) -> None:
        import json
        from confluent_kafka import Consumer

        conf = {
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "group.id": "knowledge-update-agent",
            "auto.offset.reset": "latest",
        }
        consumer = Consumer(conf)
        consumer.subscribe([settings.kafka_topic_doc_changes])

        try:
            while True:
                msg = consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    continue
                payload = json.loads(msg.value().decode("utf-8"))
                change = DocumentChange(
                    file_path=payload["file_path"],
                    change_type=ChangeType(payload["change_type"]),
                    old_hash=payload.get("old_hash", ""),
                    new_hash=payload.get("new_hash", ""),
                )
                await self.process_change(change)
        finally:
            consumer.close()

    async def _handle_create(self, change: DocumentChange, result: UpdateResult) -> None:
        if not self.doc_parser:
            return
        chunks = await self.doc_parser.parse(change.file_path)

        if self.vector_store:
            await self.vector_store.add_chunks(chunks)
            result.vectors_added = len(chunks)

        await self._extract_and_store_graph(change, chunks, result)

    async def _handle_modify(self, change: DocumentChange, result: UpdateResult) -> None:
        diff = self._diff_for_change(change)
        if diff:
            change.diff = diff
            result.diff = diff
            if not diff["is_major_change"]:
                await self._handle_minor_modify(change, result, diff)
                return

        doc_id = hashlib.sha256(change.file_path.encode()).hexdigest()[:16]

        if self.vector_store:
            deleted = await self.vector_store.delete_by_doc_id(doc_id)
            result.vectors_deleted = deleted

        await self._handle_create(change, result)

    async def _handle_minor_modify(self, change: DocumentChange, result: UpdateResult, diff: dict[str, Any]) -> None:
        if not self.doc_parser:
            return

        chunks = await self.doc_parser.parse(change.file_path)
        affected = self._affected_chunks(change, chunks, diff)
        change.diff_chunks = [chunk.chunk_id for chunk in affected]
        result.diff = {**diff, "affected_chunks": change.diff_chunks}
        if not affected:
            return

        if self.vector_store:
            if hasattr(self.vector_store, "delete_chunks"):
                result.vectors_deleted = await self.vector_store.delete_chunks(change.diff_chunks)
            added = await self.vector_store.add_chunks(affected)
            result.vectors_added = added if isinstance(added, int) else len(affected)

        await self._extract_and_store_graph(change, affected, result)

    async def _handle_delete(self, change: DocumentChange, result: UpdateResult) -> None:
        doc_id = hashlib.sha256(change.file_path.encode()).hexdigest()[:16]

        if self.vector_store:
            deleted = await self.vector_store.delete_by_doc_id(doc_id)
            result.vectors_deleted = deleted

        if self.knowledge_graph:
            await self.knowledge_graph.delete_by_source(change.file_path)

    @staticmethod
    def _compute_hash(file_path: str) -> str:
        try:
            with open(file_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except FileNotFoundError:
            return ""

    @staticmethod
    def _read_text(file_path: str) -> str:
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                return f.read()
        except FileNotFoundError:
            return ""

    def _bump_version(self, entity_name: str) -> int:
        ver = self._version_counter.get(entity_name, 0) + 1
        self._version_counter[entity_name] = ver
        return ver

    def _diff_for_change(self, change: DocumentChange) -> dict[str, Any]:
        if change.before_content or change.after_content:
            return self.cdc_processor.compute_diff(change.before_content, change.after_content)
        if change.before_chunk_hashes or change.after_chunk_hashes:
            changed = [
                chunk_id
                for chunk_id, after_hash in change.after_chunk_hashes.items()
                if change.before_chunk_hashes.get(chunk_id) != after_hash
            ]
            removed = [
                chunk_id
                for chunk_id in change.before_chunk_hashes
                if chunk_id not in change.after_chunk_hashes
            ]
            total = max(len(change.before_chunk_hashes) + len(change.after_chunk_hashes), 1)
            ratio = (len(changed) + len(removed)) / total
            return {
                "added_lines": [],
                "removed_lines": [],
                "added_count": len(changed),
                "removed_count": len(removed),
                "change_ratio": round(ratio, 4),
                "is_major_change": ratio > CDCProcessor.MAJOR_CHANGE_THRESHOLD,
                "changed_chunks": changed,
                "removed_chunks": removed,
            }
        return {}

    def _affected_chunks(self, change: DocumentChange, chunks: list[Any], diff: dict[str, Any]) -> list[Any]:
        changed_chunks = set(diff.get("changed_chunks", []))
        if changed_chunks:
            return [chunk for chunk in chunks if chunk.chunk_id in changed_chunks]

        added_lines = [line for line in diff.get("added_lines", []) if str(line).strip()]
        affected = [
            chunk
            for chunk in chunks
            if any(line in chunk.content for line in added_lines)
        ]
        if affected:
            return affected
        return chunks[:1]

    async def _extract_and_store_graph(self, change: DocumentChange, chunks: list[Any], result: UpdateResult) -> None:
        if not (self.knowledge_extractor and self.knowledge_graph):
            return

        extractions = await self.knowledge_extractor.extract(chunks)
        source_by_chunk = {
            chunk.chunk_id: chunk.metadata.get("source") or change.file_path
            for chunk in chunks
        }
        for ext in extractions:
            source = source_by_chunk.get(ext.source_chunk_id, change.file_path)
            for ent in ext.entities:
                version = self._bump_version(ent.name)
                await self.knowledge_graph.upsert_entity(ent, version=version, source=source)
                result.entities_added += 1
            for rel in ext.relations:
                await self.knowledge_graph.add_relation(rel, source=source)
                result.relations_added += 1
