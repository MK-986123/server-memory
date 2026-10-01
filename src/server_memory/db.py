"""Stable public database API with focused safety hardening."""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from typing import Any

from . import _db_impl as _impl

# Preserve the established module surface, including private helpers used by
# repository tests and downstream diagnostics. The public Database subclass
# adds narrowly scoped safety behavior without changing callers or schemas.
for _name, _value in vars(_impl).items():
    if not _name.startswith("__"):
        globals()[_name] = _value


class Database(_impl.Database):
    """Database implementation with bounded lock waits and strict snapshots."""

    @contextmanager
    def transaction(self):
        """Run a retryable write transaction within one end-to-end deadline."""
        cx = self.cx
        if cx.in_transaction:
            yield cx
            return

        deadline = time.monotonic() + self.write_timeout_seconds
        attempt = 0
        previous_busy_timeout = self._busy_timeout_ms
        try:
            while True:
                attempt += 1
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise DatabaseBusyError(
                        "memory database is locked/busy; retry this write shortly "
                        f"(deadline {self.write_timeout_seconds:.2f}s, attempts {attempt})"
                    )

                # SQLite's native wait must never exceed the configured Python
                # deadline. Restore the normal connection setting after BEGIN.
                self._set_busy_timeout(
                    max(1, min(previous_busy_timeout, int(remaining * 1000)))
                )
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
                        "DB locked acquiring write transaction (attempt %d), "
                        "retrying in %.2fs...",
                        attempt,
                        delay,
                    )
                    time.sleep(delay)
        finally:
            self._set_busy_timeout(previous_busy_timeout)

        try:
            yield cx
            cx.commit()
        except Exception:
            if cx.in_transaction:
                cx.rollback()
            raise

    def import_snapshot(self, snapshot: dict[str, Any], *, conflict: str = "fail") -> None:
        """Restore only current-schema snapshots into an actually empty store."""
        if (
            snapshot.get("format") == SNAPSHOT_FORMAT
            and snapshot.get("version") == SNAPSHOT_VERSION
            and snapshot.get("schema_version") != SCHEMA_VERSION
        ):
            raise ValueError(
                f"unsupported snapshot schema version; expected {SCHEMA_VERSION}"
            )

        tables = snapshot.get("tables")
        valid_shape = isinstance(tables, dict) and set(tables) == set(SNAPSHOT_TABLES)
        if conflict == "fail" and valid_shape:
            populated = any(
                self.cx.execute(
                    f"SELECT EXISTS(SELECT 1 FROM {table}) AS used"
                ).fetchone()["used"]
                for table in SNAPSHOT_TABLES
                if table != "tags"
            )
            seeded_tags = {
                (name, description, color, int(is_system), auto_expire_hours)
                for name, description, color, is_system, auto_expire_hours in SYSTEM_TAGS
            }
            actual_tags = {
                (
                    row["name"],
                    row["description"],
                    row["color"],
                    row["is_system"],
                    row["auto_expire_hours"],
                )
                for row in self.cx.execute(
                    "SELECT name, description, color, is_system, auto_expire_hours FROM tags"
                )
            }
            populated = populated or not actual_tags.issubset(seeded_tags)
            if populated:
                raise ValueError("snapshot conflict: target store is not empty")

        super().import_snapshot(snapshot, conflict=conflict)
