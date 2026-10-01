"""Tests for database layer."""

import sqlite3
import struct
import threading
import time
from pathlib import Path

import pytest

from server_memory.db import SCHEMA_VERSION, SYSTEM_TAGS, Database, DatabaseBusyError


def test_schema_creation(db):
    """Schema tables should exist after init."""
    tables = db.cx.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    table_names = {r["name"] for r in tables}
    assert "entities" in table_names
    assert "observations" in table_names
    assert "relations" in table_names
    assert "tags" in table_names
    assert "entity_tags" in table_names
    assert "activity_log" in table_names
    assert "observation_history" in table_names
    assert "schema_version" in table_names


def test_wal_mode(db):
    """WAL mode should be enabled (in-memory DBs report 'memory' instead of 'wal')."""
    mode = db.cx.execute("PRAGMA journal_mode").fetchone()[0]
    # In-memory databases can't use WAL, so they report 'memory'
    assert mode in ("wal", "memory")


def test_foreign_keys_on(db):
    """Foreign keys should be enforced."""
    fk = db.cx.execute("PRAGMA foreign_keys").fetchone()[0]
    assert fk == 1


def test_system_tags_seeded(db):
    """System tags should be present after init."""
    rows = db.cx.execute("SELECT name FROM tags WHERE is_system = 1 ORDER BY name").fetchall()
    system_names = {r["name"] for r in rows}
    for name, *_ in SYSTEM_TAGS:
        assert name in system_names, f"System tag '{name}' missing"


def test_fts_entity_trigger(db):
    """FTS index should auto-update on entity insert."""
    db.cx.execute("INSERT INTO entities (name, entity_type) VALUES ('TestApp', 'project')")
    db.cx.commit()
    row = db.cx.execute("SELECT * FROM fts_entities WHERE fts_entities MATCH 'TestApp'").fetchone()
    assert row is not None


def test_fts_observation_trigger(db):
    """FTS index should auto-update on observation insert."""
    db.cx.execute("INSERT INTO entities (name, entity_type) VALUES ('E1', 'test')")
    eid = db.cx.execute("SELECT id FROM entities WHERE name='E1'").fetchone()["id"]
    db.cx.execute(
        "INSERT INTO observations (entity_id, content) VALUES (?, 'uses React for frontend')",
        (eid,),
    )
    db.cx.commit()
    row = db.cx.execute(
        "SELECT * FROM fts_observations WHERE fts_observations MATCH 'React'"
    ).fetchone()
    assert row is not None


def test_foreign_key_enforcement(db):
    """Inserting observation with bad entity_id should fail."""
    import sqlite3

    try:
        db.cx.execute("INSERT INTO observations (entity_id, content) VALUES (9999, 'orphan')")
        db.cx.commit()
        assert False, "Should have raised IntegrityError"
    except sqlite3.IntegrityError:
        db.cx.rollback()


def test_schema_version(db):
    """Schema version should be recorded."""
    row = db.cx.execute("SELECT version FROM schema_version").fetchone()
    assert row is not None
    assert row["version"] == SCHEMA_VERSION


def test_cleanup_expired(db):
    """cleanup_expired should soft-delete entities with expired ephemeral tags."""
    # Create entity tagged 'ephemeral' (24h expire)
    db.cx.execute(
        "INSERT INTO entities (name, entity_type, created_at) VALUES ('OldE', 'test', '2020-01-01T00:00:00.000Z')"
    )
    eid = db.cx.execute("SELECT id FROM entities WHERE name='OldE'").fetchone()["id"]
    tag_id = db.cx.execute("SELECT id FROM tags WHERE name='ephemeral'").fetchone()["id"]
    db.cx.execute("INSERT INTO entity_tags (entity_id, tag_id) VALUES (?, ?)", (eid, tag_id))
    db.cx.commit()

    cleaned = db.cleanup_expired()
    assert cleaned >= 1

    row = db.cx.execute("SELECT deleted_at FROM entities WHERE name='OldE'").fetchone()
    assert row["deleted_at"] is not None


def test_cleanup_expired_cascades_relations(db):
    """Expiry must not leave an active relation pointing at a deleted entity."""
    db.cx.execute(
        "INSERT INTO entities (name, entity_type, created_at) "
        "VALUES ('Expired', 'test', '2020-01-01T00:00:00.000Z')"
    )
    db.cx.execute("INSERT INTO entities (name, entity_type) VALUES ('Live', 'test')")
    expired_id = db.cx.execute(
        "SELECT id FROM entities WHERE name='Expired'"
    ).fetchone()["id"]
    live_id = db.cx.execute("SELECT id FROM entities WHERE name='Live'").fetchone()["id"]
    tag_id = db.cx.execute("SELECT id FROM tags WHERE name='ephemeral'").fetchone()["id"]
    db.cx.execute(
        "INSERT INTO entity_tags (entity_id, tag_id) VALUES (?, ?)",
        (expired_id, tag_id),
    )
    db.cx.execute(
        "INSERT INTO relations (from_entity_id, to_entity_id, relation_type) "
        "VALUES (?, ?, 'references')",
        (expired_id, live_id),
    )
    db.cx.commit()

    db.cleanup_expired()

    row = db.cx.execute("SELECT deleted_at FROM relations").fetchone()
    assert row["deleted_at"] is not None


def test_lossless_snapshot_round_trip_preserves_semantic_tables(tmp_path):
    source = Database(tmp_path / "source.db")
    source.open()
    source.cx.execute(
        "INSERT INTO entities (name, entity_type, metadata_json, deleted_at) "
        "VALUES ('Snapshot', 'decision', '{\"owner\":\"team\"}', NULL)"
    )
    entity_id = source.cx.execute("SELECT id FROM entities WHERE name='Snapshot'").fetchone()["id"]
    source.cx.execute(
        "INSERT INTO observations "
        "(entity_id, content, source, confidence, importance, obs_type, version, metadata_json) "
        "VALUES (?, 'precise fact', 'test', 0.7, 0.9, 'decision', 2, '{\"k\":1}')",
        (entity_id,),
    )
    observation_id = source.cx.execute("SELECT id FROM observations").fetchone()["id"]
    source.cx.execute(
        "INSERT INTO entities (name, entity_type, metadata_json, deleted_at) "
        "VALUES ('Deleted target', 'component', '{\"state\":\"retired\"}', "
        "'2026-07-01T00:00:00.000Z')"
    )
    deleted_entity_id = source.cx.execute(
        "SELECT id FROM entities WHERE name='Deleted target'"
    ).fetchone()["id"]
    source.cx.execute(
        "INSERT INTO observation_history (observation_id, content, version) "
        "VALUES (?, 'old fact', 1)",
        (observation_id,),
    )
    source.cx.execute(
        "INSERT INTO relations "
        "(from_entity_id, to_entity_id, relation_type, weight, metadata_json, deleted_at) "
        "VALUES (?, ?, 'replaced', 0.25, '{\"reason\":\"migration\"}', "
        "'2026-07-02T00:00:00.000Z')",
        (entity_id, deleted_entity_id),
    )
    relation_id = source.cx.execute("SELECT id FROM relations").fetchone()["id"]
    source.cx.execute(
        "INSERT INTO tags (name, description, color, is_system, auto_expire_hours) "
        "VALUES ('snapshot-custom', 'round trip', '#123456', 0, 36)"
    )
    tag_id = source.cx.execute("SELECT id FROM tags WHERE name='snapshot-custom'").fetchone()["id"]
    source.cx.execute("INSERT INTO entity_tags VALUES (?, ?)", (entity_id, tag_id))
    source.cx.execute("INSERT INTO observation_tags VALUES (?, ?)", (observation_id, tag_id))
    source.cx.execute("INSERT INTO relation_tags VALUES (?, ?)", (relation_id, tag_id))
    source.cx.execute(
        "INSERT INTO activity_log (action, summary, metadata_json) "
        "VALUES ('decision_made', 'snapshot test', '{\"source\":\"unit\"}')"
    )
    vector = struct.pack("4f", 1.0, 0.0, -1.0, 0.5)
    source.cx.execute(
        "INSERT INTO entity_embeddings "
        "(entity_id, embedding, model_name, dimension, bucket0, bucket1, bucket2, bucket3) "
        "VALUES (?, ?, 'snapshot-model', 4, 1, 2, 3, 4)",
        (entity_id, vector),
    )
    source.cx.execute(
        "INSERT INTO observation_embeddings "
        "(observation_id, embedding, model_name, dimension, bucket0, bucket1, bucket2, bucket3) "
        "VALUES (?, ?, 'snapshot-model', 4, 4, 3, 2, 1)",
        (observation_id, vector),
    )
    source.cx.commit()

    snapshot = source.export_snapshot()
    target = Database(tmp_path / "target.db")
    target.open()
    target.import_snapshot(snapshot)

    assert target.export_snapshot()["tables"] == snapshot["tables"]
    source.close()
    target.close()


def test_snapshot_import_is_atomic_on_invalid_input(db):
    db.cx.execute("INSERT INTO entities (name, entity_type) VALUES ('Existing', 'note')")
    db.cx.commit()
    before = db.export_snapshot()

    try:
        db.import_snapshot({"format": "server-memory-snapshot", "version": 1, "tables": {}})
        assert False, "invalid snapshot should fail"
    except ValueError:
        pass

    assert db.export_snapshot() == before


def test_snapshot_import_rolls_back_after_mid_apply_foreign_key_failure(db):
    db.cx.execute("INSERT INTO entities (name, entity_type) VALUES ('Existing', 'note')")
    db.cx.commit()
    before = db.export_snapshot()

    source = Database(":memory:")
    source.open()
    source.cx.execute("INSERT INTO entities (name, entity_type) VALUES ('Imported', 'note')")
    source.cx.commit()
    snapshot = source.export_snapshot()
    snapshot["tables"]["relations"].append(
        {
            "id": 1,
            "from_entity_id": 1,
            "to_entity_id": 99999,
            "relation_type": "broken",
            "weight": 1.0,
            "metadata_json": "{}",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "deleted_at": None,
        }
    )

    with pytest.raises(sqlite3.IntegrityError):
        db.import_snapshot(snapshot, conflict="replace")

    assert db.export_snapshot() == before
    source.close()


def test_embedding_tables_exist(db):
    """Embedding tables should be created in schema v2."""
    tables = db.cx.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    table_names = {r["name"] for r in tables}
    assert "entity_embeddings" in table_names
    assert "observation_embeddings" in table_names


def test_embedding_rows_record_dimension(db):
    for table in ("entity_embeddings", "observation_embeddings"):
        columns = {row["name"] for row in db.cx.execute(f"PRAGMA table_info({table})")}
        assert "dimension" in columns


def test_observation_columns_exist(db):
    """Observation schema should match graph/runtime expectations."""
    rows = db.cx.execute("PRAGMA table_info(observations)").fetchall()
    columns = {r["name"] for r in rows}
    assert "importance" in columns
    assert "obs_type" in columns


def test_schema_migration_v1_to_v2():
    """Simulate opening a v1 database and verify migration to v2."""
    # Create a bare v1 database manually (just the version table, no embedding tables)
    db = Database(":memory:")
    db.conn = sqlite3.connect(":memory:")
    db.conn.row_factory = sqlite3.Row
    db._configure()

    # Create a minimal v1 schema — just schema_version + entities (enough to test migration)
    db.conn.executescript("""
        CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY);
        INSERT INTO schema_version (version) VALUES (1);
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            entity_type TEXT NOT NULL DEFAULT '',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            deleted_at TEXT DEFAULT NULL
        );
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            content TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT '',
            confidence REAL NOT NULL DEFAULT 1.0,
            version INTEGER NOT NULL DEFAULT 1,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            deleted_at TEXT DEFAULT NULL
        );
    """)
    db.conn.commit()

    # Insert test data
    db.conn.execute("INSERT INTO entities (name, entity_type) VALUES ('TestEntity', 'test')")
    db.conn.commit()

    # Verify v1 state — no embedding tables
    tables_before = {
        r["name"]
        for r in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert "entity_embeddings" not in tables_before

    # Close and reopen (simulating server restart which runs _init_schema)
    db.close()

    # Re-open with full schema initialization
    db2 = Database(":memory:")
    db2.open()

    # Verify v2/v3 schema has embedding tables
    tables_after = {
        r["name"]
        for r in db2.cx.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert "entity_embeddings" in tables_after
    assert "observation_embeddings" in tables_after

    # Verify schema version matches current
    row = db2.cx.execute("SELECT MAX(version) as v FROM schema_version").fetchone()
    assert row["v"] == SCHEMA_VERSION

    db2.close()


def test_embedding_table_foreign_key(db):
    """Embedding tables should cascade delete with parent entity/observation."""
    # Create an entity
    db.cx.execute("INSERT INTO entities (name, entity_type) VALUES ('E1', 'test')")
    eid = db.cx.execute("SELECT id FROM entities WHERE name='E1'").fetchone()["id"]

    # Insert an embedding
    db.cx.execute(
        "INSERT INTO entity_embeddings (entity_id, embedding, model_name) VALUES (?, X'00000000', 'test')",
        (eid,),
    )
    db.cx.commit()

    # Delete entity (hard) should cascade to embedding
    db.cx.execute("DELETE FROM entities WHERE id = ?", (eid,))
    db.cx.commit()

    row = db.cx.execute(
        "SELECT COUNT(*) c FROM entity_embeddings WHERE entity_id = ?", (eid,)
    ).fetchone()
    assert row["c"] == 0


def test_default_lock_timeouts_are_thirty_seconds():
    assert Database.CONNECT_TIMEOUT_SECONDS == 30.0
    assert Database.BUSY_TIMEOUT_MS == 30_000
    assert Database.WRITE_TIMEOUT_SECONDS == 30.0
    assert Database.MIGRATION_BUSY_TIMEOUT_MS >= Database.BUSY_TIMEOUT_MS


def test_configured_write_timeout_is_applied(tmp_path):
    db = Database(tmp_path / "timeout.db", write_timeout_seconds=2.5)
    db.open()
    try:
        assert db.write_timeout_seconds == 2.5
        busy = db.cx.execute("PRAGMA busy_timeout").fetchone()[0]
        assert busy == Database.BUSY_TIMEOUT_MS
    finally:
        db.close()


def test_write_lock_succeeds_when_released_inside_timeout(tmp_path):
    db_path = tmp_path / "lock-success.db"
    db = Database(db_path, write_timeout_seconds=5.0)
    db.open()
    locker = sqlite3.connect(db_path, check_same_thread=False)
    locker.execute("BEGIN EXCLUSIVE")

    def release() -> None:
        time.sleep(0.25)
        locker.rollback()
        locker.close()

    thread = threading.Thread(target=release)
    thread.start()
    try:
        with db.transaction() as cx:
            cx.execute("INSERT INTO entities (name, entity_type) VALUES ('ok', 'note')")
        row = db.cx.execute("SELECT name FROM entities WHERE name='ok'").fetchone()
        assert row is not None
    finally:
        thread.join()
        db.close()


def test_write_lock_exceeding_deadline_raises_database_busy(tmp_path):
    db_path = tmp_path / "lock-fail.db"
    db = Database(db_path, write_timeout_seconds=0.35)
    db.open()
    db.cx.execute("PRAGMA busy_timeout=50")
    locker = sqlite3.connect(db_path, check_same_thread=False)
    locker.execute("BEGIN EXCLUSIVE")
    try:
        with (
            pytest.raises(DatabaseBusyError) as exc_info,
            db.transaction() as cx,
        ):
            cx.execute("INSERT INTO entities (name, entity_type) VALUES ('blocked', 'note')")
        assert exc_info.value.retryable is True
        assert "0.35" in str(exc_info.value)
    finally:
        locker.rollback()
        locker.close()
        db.close()


def _pack_embedding(values: list[float]) -> bytes:
    return struct.pack(f"{len(values)}f", *values)


def _create_v4_database(path: Path, *, embedding_rows: int = 0) -> None:
    """Create a schema-v4 on-disk DB (pre embedding dimension/buckets)."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
        INSERT INTO schema_version (version) VALUES (4);
        CREATE TABLE entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            entity_type TEXT NOT NULL DEFAULT '',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            last_accessed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            deleted_at TEXT DEFAULT NULL
        );
        CREATE TABLE observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            content TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT '',
            confidence REAL NOT NULL DEFAULT 1.0,
            importance REAL NOT NULL DEFAULT 0.5,
            obs_type TEXT NOT NULL DEFAULT '',
            version INTEGER NOT NULL DEFAULT 1,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            deleted_at TEXT DEFAULT NULL
        );
        CREATE TABLE entity_embeddings (
            entity_id INTEGER PRIMARY KEY REFERENCES entities(id) ON DELETE CASCADE,
            embedding BLOB NOT NULL,
            model_name TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );
        CREATE TABLE observation_embeddings (
            observation_id INTEGER PRIMARY KEY REFERENCES observations(id) ON DELETE CASCADE,
            embedding BLOB NOT NULL,
            model_name TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );
        CREATE TABLE tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            color TEXT NOT NULL DEFAULT '',
            is_system INTEGER NOT NULL DEFAULT 0,
            auto_expire_hours INTEGER DEFAULT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );
        CREATE TABLE relations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            to_entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            relation_type TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 1.0,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            deleted_at TEXT DEFAULT NULL,
            UNIQUE(from_entity_id, to_entity_id, relation_type)
        );
        CREATE TABLE activity_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL DEFAULT '',
            action TEXT NOT NULL,
            summary TEXT NOT NULL DEFAULT '',
            entity_ids_json TEXT NOT NULL DEFAULT '[]',
            tags_json TEXT NOT NULL DEFAULT '[]',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );
        CREATE TABLE observation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            observation_id INTEGER NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
            content TEXT NOT NULL,
            version INTEGER NOT NULL,
            changed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );
        CREATE TABLE entity_tags (
            entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
            PRIMARY KEY (entity_id, tag_id)
        );
        CREATE TABLE observation_tags (
            observation_id INTEGER NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
            tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
            PRIMARY KEY (observation_id, tag_id)
        );
        CREATE TABLE relation_tags (
            relation_id INTEGER NOT NULL REFERENCES relations(id) ON DELETE CASCADE,
            tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
            PRIMARY KEY (relation_id, tag_id)
        );
        """
    )
    for index in range(embedding_rows):
        conn.execute(
            "INSERT INTO entities (name, entity_type) VALUES (?, 'note')",
            (f"Entity-{index}",),
        )
        eid = conn.execute(
            "SELECT id FROM entities WHERE name = ?", (f"Entity-{index}",)
        ).fetchone()["id"]
        emb = _pack_embedding([float((index + 1) % 7), 0.25, -0.5, 0.125])
        conn.execute(
            "INSERT INTO entity_embeddings (entity_id, embedding, model_name) VALUES (?, ?, 'test')",
            (eid, emb),
        )
        conn.execute(
            "INSERT INTO observations (entity_id, content) VALUES (?, ?)",
            (eid, f"observation {index}"),
        )
        oid = conn.execute(
            "SELECT id FROM observations WHERE entity_id = ?", (eid,)
        ).fetchone()["id"]
        conn.execute(
            "INSERT INTO observation_embeddings (observation_id, embedding, model_name) "
            "VALUES (?, ?, 'test')",
            (oid, emb),
        )
    conn.commit()
    conn.close()


def test_v4_to_v6_migration_is_idempotent_with_many_embeddings(tmp_path):
    path = tmp_path / "v4-many.db"
    _create_v4_database(path, embedding_rows=120)

    first = Database(path)
    first.open()
    try:
        version = first.cx.execute("SELECT MAX(version) v FROM schema_version").fetchone()["v"]
        assert version == SCHEMA_VERSION
        entity_count = first.cx.execute("SELECT COUNT(*) c FROM entity_embeddings").fetchone()["c"]
        assert entity_count == 120
        for table in ("entity_embeddings", "observation_embeddings"):
            columns = {row["name"] for row in first.cx.execute(f"PRAGMA table_info({table})")}
            assert {"dimension", "bucket0", "bucket1", "bucket2", "bucket3"}.issubset(columns)
            nonzero = first.cx.execute(
                f"SELECT COUNT(*) c FROM {table} WHERE dimension > 0"
            ).fetchone()["c"]
            assert nonzero == 120
        integrity = first.cx.execute("PRAGMA integrity_check").fetchone()[0]
        assert integrity == "ok"
    finally:
        first.close()

    second = Database(path)
    second.open()
    try:
        version = second.cx.execute("SELECT MAX(version) v FROM schema_version").fetchone()["v"]
        assert version == SCHEMA_VERSION
        entity_count = second.cx.execute("SELECT COUNT(*) c FROM entities").fetchone()["c"]
        assert entity_count == 120
    finally:
        second.close()


def test_interrupted_v6_migration_rolls_back_version_and_resumes(tmp_path, monkeypatch):
    path = tmp_path / "v4-interrupted.db"
    _create_v4_database(path, embedding_rows=8)

    original_v6 = Database._migrate_to_v6

    def boom(self):
        original_v6(self)
        raise RuntimeError("simulated migration interrupt")

    monkeypatch.setattr(Database, "_migrate_to_v6", boom)
    failed = Database(path)
    try:
        with pytest.raises(RuntimeError, match="simulated migration interrupt"):
            failed.open()
    finally:
        failed.close()

    # Version must not advance when the upgrade fails after bucket work.
    probe = sqlite3.connect(path)
    try:
        version = probe.execute("SELECT MAX(version) v FROM schema_version").fetchone()[0]
        assert version == 4
    finally:
        probe.close()

    monkeypatch.setattr(Database, "_migrate_to_v6", original_v6)
    recovered = Database(path)
    recovered.open()
    try:
        version = recovered.cx.execute("SELECT MAX(version) v FROM schema_version").fetchone()["v"]
        assert version == SCHEMA_VERSION
        assert recovered._has_required_columns(
            {
                "entity_embeddings": {"dimension", "bucket0", "bucket1", "bucket2", "bucket3"},
                "observation_embeddings": {
                    "dimension",
                    "bucket0",
                    "bucket1",
                    "bucket2",
                    "bucket3",
                },
            }
        )
    finally:
        recovered.close()


def test_migration_startup_uses_extended_busy_timeout(tmp_path, monkeypatch):
    path = tmp_path / "v4-busy.db"
    _create_v4_database(path, embedding_rows=3)
    seen: list[int] = []
    original = Database._set_busy_timeout

    def tracking(self, busy_timeout_ms: int) -> None:
        seen.append(busy_timeout_ms)
        return original(self, busy_timeout_ms)

    monkeypatch.setattr(Database, "_set_busy_timeout", tracking)
    db = Database(path)
    db.open()
    try:
        assert Database.MIGRATION_BUSY_TIMEOUT_MS in seen
        # Must not remain stuck on the old 100 ms fail-fast default.
        assert 100 not in seen or Database.BUSY_TIMEOUT_MS != 100
        assert db._busy_timeout_ms == Database.BUSY_TIMEOUT_MS
    finally:
        db.close()
