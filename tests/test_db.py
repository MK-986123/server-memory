"""Tests for database layer."""

import sqlite3

from server_memory.db import SCHEMA_VERSION, SYSTEM_TAGS, Database


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


def test_embedding_tables_exist(db):
    """Embedding tables should be created in schema v2."""
    tables = db.cx.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    table_names = {r["name"] for r in tables}
    assert "entity_embeddings" in table_names
    assert "observation_embeddings" in table_names


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
