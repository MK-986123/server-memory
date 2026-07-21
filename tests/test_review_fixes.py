"""Regression tests for PR #13 post-review hardening fixes."""

from __future__ import annotations

import sqlite3
import threading
import time

import pytest

from server_memory import embeddings, paths
from server_memory.db import SCHEMA_VERSION, Database, DatabaseBusyError


def test_write_timeout_bounds_sqlite_wait_and_restores_busy_timeout(tmp_path):
    db_path = tmp_path / "bounded-write.db"
    db = Database(db_path, write_timeout_seconds=0.25)
    db.open()
    locker = sqlite3.connect(db_path, check_same_thread=False)
    locker.execute("BEGIN EXCLUSIVE")

    started = time.monotonic()
    try:
        with pytest.raises(DatabaseBusyError):
            with db.transaction() as cx:
                cx.execute("INSERT INTO entities (name, entity_type) VALUES ('blocked', 'note')")
        elapsed = time.monotonic() - started
        assert elapsed < 1.5
        assert db.cx.execute("PRAGMA busy_timeout").fetchone()[0] == Database.BUSY_TIMEOUT_MS
        assert db._busy_timeout_ms == Database.BUSY_TIMEOUT_MS
    finally:
        locker.rollback()
        locker.close()
        db.close()


def test_submit_bounded_releases_slot_when_executor_submit_fails(monkeypatch):
    slots = threading.BoundedSemaphore(1)

    class BrokenExecutor:
        def submit(self, fn, *args):
            raise RuntimeError("executor stopped")

    monkeypatch.setattr(embeddings, "_timeout_slots", slots)
    monkeypatch.setattr(embeddings, "_timeout_executor", BrokenExecutor())

    with pytest.raises(RuntimeError, match="executor stopped"):
        embeddings._submit_bounded(lambda: None)

    assert slots.acquire(blocking=False)
    slots.release()


def test_find_workspace_marker_returns_none_when_start_cannot_be_statted(tmp_path):
    assert paths._find_workspace_marker(tmp_path / "missing" / "workspace") is None


def test_utf8_payload_limit_counts_bytes_without_full_encoded_copy(monkeypatch):
    from server_memory import server

    monkeypatch.setattr(server, "IMPORT_SIZE_CHUNK_CHARS", 2)

    assert not server._utf8_payload_exceeds_limit("abc", 3)
    assert server._utf8_payload_exceeds_limit("abcd", 3)
    assert not server._utf8_payload_exceeds_limit("éé", 4)
    assert server._utf8_payload_exceeds_limit("ééé", 5)

    with pytest.raises(ValueError, match="non-negative"):
        server._utf8_payload_exceeds_limit("", -1)

    monkeypatch.setattr(server, "IMPORT_PAYLOAD_LIMIT_BYTES", 5)
    import_tool = server.create_server()._tool_manager._tools["import_graph"].fn
    with pytest.raises(ValueError, match="50 MiB UTF-8 limit"):
        import_tool(None, data="ééé")


def test_snapshot_import_rejects_incompatible_schema_without_mutation(tmp_path):
    source = Database(tmp_path / "source.db")
    target = Database(tmp_path / "target.db")
    source.open()
    target.open()
    try:
        source.cx.execute("INSERT INTO entities (name, entity_type) VALUES ('Source', 'note')")
        source.cx.commit()
        snapshot = source.export_snapshot()
        snapshot["schema_version"] = SCHEMA_VERSION - 1
        before = target.export_snapshot()

        with pytest.raises(ValueError, match="snapshot schema version"):
            target.import_snapshot(snapshot)

        assert target.export_snapshot() == before
    finally:
        target.close()
        source.close()


def test_snapshot_conflict_checks_every_semantic_table(tmp_path):
    source = Database(tmp_path / "source.db")
    target = Database(tmp_path / "target.db")
    source.open()
    target.open()
    try:
        snapshot = source.export_snapshot()
        target.cx.execute(
            "INSERT INTO tags (name, description, color, is_system) "
            "VALUES ('custom-only', '', '', 0)"
        )
        target.cx.commit()
        before = target.export_snapshot()

        with pytest.raises(ValueError, match="target store is not empty"):
            target.import_snapshot(snapshot)

        assert target.export_snapshot() == before
    finally:
        target.close()
        source.close()


def test_snapshot_conflict_rejects_forged_system_tag(tmp_path):
    source = Database(tmp_path / "source-system.db")
    target = Database(tmp_path / "target-system.db")
    source.open()
    target.open()
    try:
        snapshot = source.export_snapshot()
        target.cx.execute(
            "INSERT INTO tags (name, description, color, is_system) "
            "VALUES ('forged-system', 'not seeded', '', 1)"
        )
        target.cx.commit()
        before = target.export_snapshot()

        with pytest.raises(ValueError, match="target store is not empty"):
            target.import_snapshot(snapshot)

        assert target.export_snapshot() == before
    finally:
        target.close()
        source.close()
