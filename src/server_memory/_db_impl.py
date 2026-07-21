"""SQLite database layer with WAL, FTS5, and schema management."""

from __future__ import annotations

import base64
import logging
import os
import sqlite3
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DatabaseBusyError(sqlite3.OperationalError):
    """Retryable interactive write failure after one bounded deadline."""

    retryable = True

SCHEMA_VERSION = 6
SNAPSHOT_FORMAT = "server-memory-snapshot"
SNAPSHOT_VERSION = 1
SNAPSHOT_TABLES = (
    "entities",
    "observations",
    "observation_history",
    "relations",
    "tags",
    "entity_tags",
    "observation_tags",
    "relation_tags",
    "activity_log",
    "entity_embeddings",
    "observation_embeddings",
)

SYSTEM_TAGS = [
    ("pinned", "Permanently retained entities", "#FFD700", True, None),
    ("ephemeral", "Auto-expires after 24h", "#808080", True, 24),
    ("preference", "User preferences and settings", "#4CAF50", True, None),
    ("project", "Project-scoped knowledge", "#2196F3", True, None),
    ("debugging", "Debugging context, expires 48h", "#F44336", True, 48),
    ("architecture", "Architectural decisions", "#9C27B0", True, None),
    ("recent-change", "Recent code changes, expires 72h", "#FF9800", True, 72),
]

DDL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    entity_type TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    last_accessed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    deleted_at TEXT DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS observations (
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

CREATE TABLE IF NOT EXISTS observation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    observation_id INTEGER NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    version INTEGER NOT NULL,
    changed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS relations (
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

CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    color TEXT NOT NULL DEFAULT '',
    is_system INTEGER NOT NULL DEFAULT 0,
    auto_expire_hours INTEGER DEFAULT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS entity_tags (
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (entity_id, tag_id)
);

CREATE TABLE IF NOT EXISTS observation_tags (
    observation_id INTEGER NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
    tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (observation_id, tag_id)
);

CREATE TABLE IF NOT EXISTS relation_tags (
    relation_id INTEGER NOT NULL REFERENCES relations(id) ON DELETE CASCADE,
    tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (relation_id, tag_id)
);

CREATE TABLE IF NOT EXISTS activity_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL DEFAULT '',
    action TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    entity_ids_json TEXT NOT NULL DEFAULT '[]',
    tags_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- FTS5 virtual tables
CREATE VIRTUAL TABLE IF NOT EXISTS fts_entities USING fts5(
    name, entity_type,
    content='entities',
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_observations USING fts5(
    content,
    content='observations',
    content_rowid='id',
    tokenize='porter unicode61'
);

-- FTS sync triggers for entities
CREATE TRIGGER IF NOT EXISTS entities_ai AFTER INSERT ON entities BEGIN
    INSERT INTO fts_entities(rowid, name, entity_type)
    VALUES (new.id, new.name, new.entity_type);
END;

CREATE TRIGGER IF NOT EXISTS entities_ad AFTER DELETE ON entities BEGIN
    INSERT INTO fts_entities(fts_entities, rowid, name, entity_type)
    VALUES ('delete', old.id, old.name, old.entity_type);
END;

CREATE TRIGGER IF NOT EXISTS entities_au AFTER UPDATE ON entities BEGIN
    INSERT INTO fts_entities(fts_entities, rowid, name, entity_type)
    VALUES ('delete', old.id, old.name, old.entity_type);
    INSERT INTO fts_entities(rowid, name, entity_type)
    VALUES (new.id, new.name, new.entity_type);
END;

-- FTS sync triggers for observations
CREATE TRIGGER IF NOT EXISTS observations_ai AFTER INSERT ON observations BEGIN
    INSERT INTO fts_observations(rowid, content)
    VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS observations_ad AFTER DELETE ON observations BEGIN
    INSERT INTO fts_observations(fts_observations, rowid, content)
    VALUES ('delete', old.id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS observations_au AFTER UPDATE ON observations BEGIN
    INSERT INTO fts_observations(fts_observations, rowid, content)
    VALUES ('delete', old.id, old.content);
    INSERT INTO fts_observations(rowid, content)
    VALUES (new.id, new.content);
END;

-- Indexes
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_entities_deleted ON entities(deleted_at);
CREATE INDEX IF NOT EXISTS idx_observations_entity ON observations(entity_id)
    WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_relations_from ON relations(from_entity_id)
    WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_relations_to ON relations(to_entity_id)
    WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_activity_session ON activity_log(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_activity_action ON activity_log(action, created_at);

-- Embedding tables (schema v2 base). Columns dimension/buckets and their indexes
-- are added by migrations v5/v6 so existing DBs can upgrade without DDL failing
-- on CREATE INDEX against missing columns (CREATE TABLE IF NOT EXISTS is a no-op).
CREATE TABLE IF NOT EXISTS entity_embeddings (
    entity_id INTEGER PRIMARY KEY REFERENCES entities(id) ON DELETE CASCADE,
    embedding BLOB NOT NULL,
    model_name TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS observation_embeddings (
    observation_id INTEGER PRIMARY KEY REFERENCES observations(id) ON DELETE CASCADE,
    embedding BLOB NOT NULL,
    model_name TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
"""


class Database:
    """Manages SQLite connection with WAL mode, FK enforcement, and FTS5."""

    # Backward-compatible defaults aligned with public main. Operators may still
    # choose fail-fast interactive writes via MEMORY_WRITE_TIMEOUT_MS.
    CONNECT_TIMEOUT_SECONDS = 30.0
    BUSY_TIMEOUT_MS = 30_000
    WRITE_TIMEOUT_SECONDS = 30.0
    # Schema upgrades (especially v6 embedding buckets) may rewrite large tables.
    MIGRATION_BUSY_TIMEOUT_MS = 120_000
    LOCK_RETRY_DELAYS_SECONDS = (0.05, 0.1, 0.2, 0.4, 0.8)

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        *,
        write_timeout_seconds: float = WRITE_TIMEOUT_SECONDS,
    ):
        self.db_path = str(db_path)
        if write_timeout_seconds <= 0:
            raise ValueError("write_timeout_seconds must be positive")
        self.write_timeout_seconds = write_timeout_seconds
        self.conn: sqlite3.Connection | None = None
        self._busy_timeout_ms = self.BUSY_TIMEOUT_MS

    def open(self) -> None:
        try:
            self.conn = sqlite3.connect(
                self.db_path,
                timeout=self.CONNECT_TIMEOUT_SECONDS,
            )
            self.conn.row_factory = sqlite3.Row
            self._configure()
            self._init_schema()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None

    def _set_busy_timeout(self, busy_timeout_ms: int) -> None:
        assert self.conn is not None
        self._busy_timeout_ms = busy_timeout_ms
        self.conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")

    def _configure(self) -> None:
        assert self.conn is not None
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._set_busy_timeout(self.BUSY_TIMEOUT_MS)
        self.conn.execute("PRAGMA synchronous=NORMAL")

    def _init_schema(self) -> None:
        assert self.conn is not None

        required_columns = {
            "entities": {"last_accessed_at"},
            "observations": {"importance", "obs_type"},
            "entity_embeddings": {"dimension", "bucket0", "bucket1", "bucket2", "bucket3"},
            "observation_embeddings": {
                "dimension",
                "bucket0",
                "bucket1",
                "bucket2",
                "bucket3",
            },
        }

        # Fast path: if schema is already at current version and required columns
        # exist, skip all writes. This avoids write-lock contention when another
        # instance holds the DB.
        try:
            row = self.conn.execute("SELECT MAX(version) as v FROM schema_version").fetchone()
            current_version = row["v"] if row and row["v"] else 0
            if current_version >= SCHEMA_VERSION and self._has_required_columns(required_columns):
                return
        except sqlite3.OperationalError:
            # Table doesn't exist yet — proceed with full init
            current_version = 0

        # Use a migration-appropriate busy timeout while upgrading schema.
        # Keep migrations in one commit when possible; required-column detection
        # resumes incomplete upgrades if DDL auto-commit left a partial state.
        previous_busy_timeout = self._busy_timeout_ms
        self._set_busy_timeout(self.MIGRATION_BUSY_TIMEOUT_MS)
        try:
            # Full init with retry for lock contention
            for attempt in range(3):
                try:
                    self.conn.executescript(DDL)
                    # Seed system tags
                    for name, desc, color, is_sys, expire in SYSTEM_TAGS:
                        self.conn.execute(
                            "INSERT OR IGNORE INTO tags (name, description, color, "
                            "is_system, auto_expire_hours) VALUES (?, ?, ?, ?, ?)",
                            (name, desc, color, int(is_sys), expire),
                        )
                    # Run pending migrations
                    if current_version < 2:
                        self._migrate_to_v2()
                    if current_version < 3:
                        self._migrate_to_v3()
                    if current_version < 4:
                        self._migrate_to_v4()
                    if current_version < 5:
                        self._migrate_to_v5()
                    if current_version < 6:
                        self._migrate_to_v6()
                    # Record schema version
                    self.conn.execute(
                        "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                        (SCHEMA_VERSION,),
                    )
                    self.conn.commit()
                    return
                except sqlite3.OperationalError as e:
                    if self.conn.in_transaction:
                        self.conn.rollback()
                    if "locked" in str(e).lower() and attempt < 2:
                        delay = 2**attempt  # 1s, 2s
                        logger.warning(
                            "DB locked during init (attempt %d/3), retrying in %ds...",
                            attempt + 1,
                            delay,
                        )
                        time.sleep(delay)
                        continue
                    raise
                except Exception:
                    if self.conn.in_transaction:
                        self.conn.rollback()
                    raise
        finally:
            self._set_busy_timeout(previous_busy_timeout)

    def _migrate_to_v2(self) -> None:
        """Migrate from schema v1 to v2: add embedding tables."""
        assert self.conn is not None
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS entity_embeddings (
                entity_id INTEGER PRIMARY KEY REFERENCES entities(id) ON DELETE CASCADE,
                embedding BLOB NOT NULL,
                model_name TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS observation_embeddings (
                observation_id INTEGER PRIMARY KEY REFERENCES observations(id) ON DELETE CASCADE,
                embedding BLOB NOT NULL,
                model_name TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
        """)

    def _migrate_to_v3(self) -> None:
        """Migrate from schema v2 to v3: add last_accessed_at to entities."""
        assert self.conn is not None
        with suppress(sqlite3.OperationalError):
            self.conn.execute(
                "ALTER TABLE entities ADD COLUMN last_accessed_at TEXT NOT NULL "
                "DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"
            )

    def _migrate_to_v4(self) -> None:
        """Migrate to v4: add importance and obs_type to observations."""
        assert self.conn is not None
        for stmt in (
            "ALTER TABLE observations ADD COLUMN importance REAL NOT NULL DEFAULT 0.5",
            "ALTER TABLE observations ADD COLUMN obs_type TEXT NOT NULL DEFAULT ''",
        ):
            with suppress(sqlite3.OperationalError):
                self.conn.execute(stmt)

    def _migrate_to_v5(self) -> None:
        """Migrate to v5: record embedding dimensions for compatibility checks."""
        assert self.conn is not None
        for table in ("entity_embeddings", "observation_embeddings"):
            with suppress(sqlite3.OperationalError):
                self.conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN dimension INTEGER NOT NULL DEFAULT 0"
                )
            self.conn.execute(
                f"UPDATE {table} SET dimension = length(embedding) / 4 WHERE dimension = 0"
            )

    def _migrate_to_v6(self) -> None:
        """Add sign-projection buckets for recall-preserving bounded ANN lookup."""
        assert self.conn is not None
        from .embeddings import embedding_buckets

        for table, id_column in (
            ("entity_embeddings", "entity_id"),
            ("observation_embeddings", "observation_id"),
        ):
            for column in ("bucket0", "bucket1", "bucket2", "bucket3"):
                with suppress(sqlite3.OperationalError):
                    self.conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0"
                    )
            rows = self.conn.execute(
                f"SELECT {id_column}, embedding FROM {table}"
            ).fetchall()
            for row in rows:
                buckets = embedding_buckets(row["embedding"])
                self.conn.execute(
                    f"UPDATE {table} SET bucket0=?, bucket1=?, bucket2=?, bucket3=? "
                    f"WHERE {id_column}=?",
                    (*buckets, row[id_column]),
                )
            for bucket in range(4):
                self.conn.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{table}_bucket{bucket} "
                    f"ON {table}(model_name, dimension, bucket{bucket})"
                )

    def _has_required_columns(self, required_columns: dict[str, set[str]]) -> bool:
        """Check that the current schema matches the code's runtime expectations."""
        assert self.conn is not None
        for table, columns in required_columns.items():
            try:
                rows = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
            except sqlite3.OperationalError:
                return False
            existing = {row["name"] for row in rows}
            if not columns.issubset(existing):
                return False
        return True

    @property
    def cx(self) -> sqlite3.Connection:
        assert self.conn is not None, "Database not open"
        return self.conn

    @contextmanager
    def transaction(self):
        """Open a retryable write transaction and commit or roll back around the body.
        If already in a transaction, yield without starting/committing/rolling back."""
        cx = self.cx
        outer = cx.in_transaction
        if outer:
            # Already in a transaction, just yield (no BEGIN/commit/rollback)
            yield cx
            return
        # Not in a transaction: every retry shares one end-to-end deadline.
        deadline = time.monotonic() + self.write_timeout_seconds
        attempt = 0
        while True:
            attempt += 1
            try:
                cx.execute("BEGIN IMMEDIATE")
                break
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower():
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise DatabaseBusyError(
                        "memory database is locked/busy; retry this write shortly "
                        f"(deadline {self.write_timeout_seconds:.2f}s, attempts {attempt})"
                    ) from exc
                delay = min(
                    self.LOCK_RETRY_DELAYS_SECONDS[
                        min(attempt - 1, len(self.LOCK_RETRY_DELAYS_SECONDS) - 1)
                    ],
                    remaining,
                )
                logger.warning(
                    "DB locked acquiring write transaction (attempt %d), retrying in %.2fs...",
                    attempt,
                    delay,
                )
                time.sleep(delay)

        try:
            yield cx
            cx.commit()
        except Exception:
            if cx.in_transaction:
                cx.rollback()
            raise

    def cleanup_expired(self) -> int:
        """Soft-delete entities/observations/relations tagged with expired ephemeral tags.
        Returns count of items cleaned up."""
        assert self.conn is not None
        cur = self.conn.execute(
            """
            SELECT t.id, t.auto_expire_hours FROM tags t
            WHERE t.auto_expire_hours IS NOT NULL
            """
        )
        total = 0
        now_expr = "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"
        for row in cur.fetchall():
            tag_id, hours = row["id"], row["auto_expire_hours"]
            threshold = f"-{hours} hours"
            # Soft-delete entities with this tag whose tag was applied > hours ago
            res = self.conn.execute(
                f"""
                UPDATE entities SET deleted_at = {now_expr}
                WHERE deleted_at IS NULL AND id IN (
                    SELECT et.entity_id FROM entity_tags et
                    JOIN entities e ON e.id = et.entity_id
                    WHERE et.tag_id = ? AND e.created_at < strftime('%Y-%m-%dT%H:%M:%fZ', 'now', ?)
                )
                """,
                (tag_id, threshold),
            )
            total += res.rowcount
            # Soft-delete observations
            res = self.conn.execute(
                f"""
                UPDATE observations SET deleted_at = {now_expr}
                WHERE deleted_at IS NULL AND id IN (
                    SELECT ot.observation_id FROM observation_tags ot
                    JOIN observations o ON o.id = ot.observation_id
                    WHERE ot.tag_id = ? AND o.created_at < strftime('%Y-%m-%dT%H:%M:%fZ', 'now', ?)
                )
                """,
                (tag_id, threshold),
            )
            total += res.rowcount
        # Any entity expiry must cascade to its relations. This also repairs
        # dangling active relations left by versions that did not cascade.
        res = self.conn.execute(
            f"""
            UPDATE relations SET deleted_at = {now_expr}
            WHERE deleted_at IS NULL
              AND (
                from_entity_id IN (SELECT id FROM entities WHERE deleted_at IS NOT NULL)
                OR to_entity_id IN (SELECT id FROM entities WHERE deleted_at IS NOT NULL)
              )
            """
        )
        total += res.rowcount
        if total > 0:
            self.conn.commit()
        return total

    def cleanup_unused(self, days: int) -> int:
        """Soft-delete entities that have not been accessed in `days` days.
        Pinned entities are excluded from this cleanup."""
        assert self.conn is not None
        threshold = f"-{days} days"
        now_expr = "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"

        res = self.conn.execute(
            f"""
            UPDATE entities SET deleted_at = {now_expr}
            WHERE deleted_at IS NULL
            AND last_accessed_at < strftime('%Y-%m-%dT%H:%M:%fZ', 'now', ?)
            AND id NOT IN (
                SELECT et.entity_id FROM entity_tags et
                JOIN tags t ON t.id = et.tag_id
                WHERE t.name = 'pinned'
            )
            """,
            (threshold,),
        )
        count = res.rowcount
        if count > 0:
            self.conn.commit()
        return count

    def cleanup_empty_stale(self, days: int = 7) -> int:
        """Soft-delete entities with 0 observations that haven't been updated in `days` days.

        Uses updated_at (not last_accessed_at) so stale empty entities can age out
        independently from access-tracking policy. Pinned and project-tagged entities
        are excluded.
        """
        assert self.conn is not None
        threshold = f"-{days} days"
        now_expr = "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"

        res = self.conn.execute(
            f"""
            UPDATE entities SET deleted_at = {now_expr}
            WHERE deleted_at IS NULL
            AND updated_at < strftime('%Y-%m-%dT%H:%M:%fZ', 'now', ?)
            AND id NOT IN (
                SELECT o.entity_id FROM observations o
                WHERE o.entity_id = entities.id AND o.deleted_at IS NULL
            )
            AND id NOT IN (
                SELECT et.entity_id FROM entity_tags et
                JOIN tags t ON t.id = et.tag_id
                WHERE t.name IN ('pinned', 'project', 'architecture', 'preference')
            )
            """,
            (threshold,),
        )
        count = res.rowcount
        if count > 0:
            self.conn.commit()
        return count

    def export_snapshot(self) -> dict[str, Any]:
        """Return a versioned, lossless snapshot of all semantic tables."""
        tables: dict[str, list[dict[str, Any]]] = {}
        for table in SNAPSHOT_TABLES:
            rows = self.cx.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
            serialized: list[dict[str, Any]] = []
            for row in rows:
                item: dict[str, Any] = {}
                for key in row.keys():
                    value = row[key]
                    if isinstance(value, bytes):
                        item[key] = {"$base64": base64.b64encode(value).decode("ascii")}
                    else:
                        item[key] = value
                serialized.append(item)
            tables[table] = serialized
        return {
            "format": SNAPSHOT_FORMAT,
            "version": SNAPSHOT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "tables": tables,
        }

    def import_snapshot(self, snapshot: dict[str, Any], *, conflict: str = "fail") -> None:
        """Atomically restore a validated snapshot into an empty semantic store.

        The initial merge policy is deliberately explicit: populated stores are
        rejected rather than partially merged or silently overwritten.
        """
        if snapshot.get("format") != SNAPSHOT_FORMAT or snapshot.get("version") != SNAPSHOT_VERSION:
            raise ValueError("unsupported snapshot format or version")
        if conflict not in {"fail", "replace"}:
            raise ValueError("conflict policy must be 'fail' or 'replace'")
        tables = snapshot.get("tables")
        if not isinstance(tables, dict) or set(tables) != set(SNAPSHOT_TABLES):
            raise ValueError("snapshot must contain every semantic table exactly once")
        if conflict == "fail" and self.cx.execute(
            "SELECT EXISTS(SELECT 1 FROM entities) OR EXISTS(SELECT 1 FROM activity_log) AS used"
        ).fetchone()["used"]:
            raise ValueError("snapshot conflict: target store is not empty")

        columns_by_table = {
            table: [row["name"] for row in self.cx.execute(f"PRAGMA table_info({table})")]
            for table in SNAPSHOT_TABLES
        }
        decoded: dict[str, list[dict[str, Any]]] = {}
        for table, rows in tables.items():
            if not isinstance(rows, list):
                raise ValueError(f"snapshot table {table} must be a list")
            expected = set(columns_by_table[table])
            decoded_rows: list[dict[str, Any]] = []
            for row in rows:
                if not isinstance(row, dict) or set(row) != expected:
                    raise ValueError(f"snapshot row columns do not match table {table}")
                item: dict[str, Any] = {}
                for key, value in row.items():
                    if isinstance(value, dict) and set(value) == {"$base64"}:
                        try:
                            item[key] = base64.b64decode(value["$base64"], validate=True)
                        except (ValueError, TypeError) as exc:
                            raise ValueError(f"invalid base64 in {table}.{key}") from exc
                    else:
                        item[key] = value
                decoded_rows.append(item)
            decoded[table] = decoded_rows

        delete_order = (
            "relation_tags",
            "observation_tags",
            "entity_tags",
            "observation_history",
            "entity_embeddings",
            "observation_embeddings",
            "relations",
            "observations",
            "activity_log",
            "entities",
            "tags",
        )
        insert_order = (
            "entities",
            "observations",
            "observation_history",
            "relations",
            "tags",
            "entity_tags",
            "observation_tags",
            "relation_tags",
            "activity_log",
            "entity_embeddings",
            "observation_embeddings",
        )
        with self.transaction() as cx:
            for table in delete_order:
                cx.execute(f"DELETE FROM {table}")
            for table in insert_order:
                columns = columns_by_table[table]
                placeholders = ",".join("?" for _ in columns)
                sql = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
                for row in decoded[table]:
                    cx.execute(sql, [row[column] for column in columns])

    def backup(self, dest_path: str | Path) -> None:
        """Create a full no-clobber backup using an atomically reserved destination."""
        assert self.conn is not None
        path = Path(dest_path)
        flags = os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags, 0o600)
        except FileExistsError as exc:
            raise ValueError(f"backup destination already exists: {path}") from exc
        # /proc/self/fd pins SQLite to the exact inode reserved above, closing
        # the check-then-open and symlink-swap window on the supported Linux host.
        proc_fd_path = Path(f"/proc/self/fd/{fd}")
        sqlite_path = str(proc_fd_path if proc_fd_path.exists() else path)
        try:
            dest = sqlite3.connect(sqlite_path)
        except Exception:
            os.close(fd)
            raise
        try:
            self.conn.backup(dest)
        finally:
            dest.close()
            os.close(fd)
