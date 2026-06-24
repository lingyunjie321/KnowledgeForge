"""CDCProcessor 单测：差量计算、阈值判定、版本追踪、事件归一化"""

from __future__ import annotations

import json

import pytest

from services.cdc_processor import CDCEvent, CDCProcessor


def test_compute_diff_added_and_removed():
    before = "line1\nline2\nline3"
    after = "line2\nline3\nline4"
    diff = CDCProcessor.compute_diff(before, after)

    assert "line4" in diff["added_lines"]
    assert "line1" in diff["removed_lines"]
    assert diff["added_count"] == 1
    assert diff["removed_count"] == 1


def test_compute_diff_major_change_above_threshold():
    before = "a\nb\nc"
    after = "x\ny\nz\nw"
    diff = CDCProcessor.compute_diff(before, after)
    # 改动行数占比高，应判为大变更
    assert diff["is_major_change"] is True
    assert diff["change_ratio"] > CDCProcessor.MAJOR_CHANGE_THRESHOLD


def test_compute_diff_minor_change_below_threshold():
    before = "\n".join(f"line{i}" for i in range(100))
    after = before + "\nline100"  # 只加一行，占比极小
    diff = CDCProcessor.compute_diff(before, after)
    assert diff["is_major_change"] is False
    assert diff["change_ratio"] <= CDCProcessor.MAJOR_CHANGE_THRESHOLD


def test_compute_diff_empty_strings():
    diff = CDCProcessor.compute_diff("", "")
    assert diff["added_count"] == 0
    assert diff["removed_count"] == 0
    assert diff["change_ratio"] == 0.0
    assert diff["is_major_change"] is False


def test_bump_version_increments_per_resource():
    proc = CDCProcessor()
    assert proc.bump_version("/a.txt") == 1
    assert proc.bump_version("/a.txt") == 2
    assert proc.bump_version("/b.txt") == 1


def test_get_version_default_zero():
    proc = CDCProcessor()
    assert proc.get_version("/never.txt") == 0


def test_from_filesystem_event_maps_operation():
    event = CDCProcessor.from_filesystem_event("modified", "/x/a.txt", "old", "new")
    assert event.source_type == "filesystem"
    assert event.operation == "UPDATE"
    assert event.resource_path == "/x/a.txt"
    assert event.before == {"content": "old"}
    assert event.after == {"content": "new"}

    deleted = CDCProcessor.from_filesystem_event("deleted", "/x/a.txt")
    assert deleted.operation == "DELETE"


def test_from_kafka_message_debezium_format():
    # 模拟 Debezium 风格的 CDC 消息
    payload = {
        "id": "evt-1",
        "op": "u",
        "ts_ms": 1700000000000,
        "source": {"table": "documents"},
        "before": {"id": 1, "content": "old"},
        "after": {"id": 1, "content": "new"},
    }
    event = CDCProcessor.from_kafka_message(json.dumps(payload).encode("utf-8"))
    assert event.event_id == "evt-1"
    assert event.source_type == "database"
    assert event.resource_path == "documents"
    assert event.after["content"] == "new"


async def test_process_event_insert():
    proc = CDCProcessor()
    event = CDCProcessor.from_filesystem_event("created", "/a.txt")
    result = await proc.process_event(event)

    assert result.success is True
    assert result.version == 1
    assert result.chunks_affected == 1


async def test_process_event_delete():
    proc = CDCProcessor()
    event = CDCProcessor.from_filesystem_event("deleted", "/a.txt")
    result = await proc.process_event(event)

    assert result.success is True
    assert result.chunks_affected == -1


async def test_process_event_update_with_diff():
    proc = CDCProcessor()
    event = CDCProcessor.from_filesystem_event(
        "modified", "/a.txt",
        content_before="a\nb\nc",
        content_after="a\nb\nc\nd\ne\nf\ng\nh\ni\nj",  # 大量新增
    )
    result = await proc.process_event(event)

    assert result.success is True
    assert result.event.diff is not None
    assert result.event.diff["is_major_change"] is True
    # 大变更按 added_count 计
    assert result.chunks_affected == result.event.diff["added_count"]


async def test_process_batch_returns_one_per_event():
    proc = CDCProcessor()
    events = [
        CDCProcessor.from_filesystem_event("created", "/a.txt"),
        CDCProcessor.from_filesystem_event("created", "/b.txt"),
    ]
    results = await proc.process_batch(events)
    assert len(results) == 2
    assert all(r.success for r in results)


def test_get_stats_tracks_events_and_versions():
    proc = CDCProcessor()
    proc.bump_version("/a.txt")
    proc.bump_version("/a.txt")
    stats = proc.get_stats()
    assert stats["tracked_resources"] == 1
    assert "/a.txt" in stats["versions"]
    assert stats["versions"]["/a.txt"] == 2
