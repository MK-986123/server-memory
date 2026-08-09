"""Regression tests for private SQLite database and backup permissions."""

from __future__ import annotations

import os
import stat

import pytest

from server_memory.db import Database

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX file permissions only")


def _mode(path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_database_file_is_owner_only_under_permissive_umask(tmp_path):
    db_path = tmp_path / "memory.db"
    previous_umask = os.umask(0)
    db = Database(db_path)
    try:
        db.open()
    finally:
        os.umask(previous_umask)

    try:
        assert _mode(db_path) == 0o600

        db.cx.execute("INSERT INTO entities (name, entity_type) VALUES (?, ?)", ("secret", "test"))
        db.cx.commit()
        for suffix in ("-wal", "-shm"):
            sidecar = db_path.with_name(db_path.name + suffix)
            if sidecar.exists():
                assert _mode(sidecar) == 0o600
    finally:
        db.close()


def test_existing_database_permissions_are_tightened(tmp_path):
    db_path = tmp_path / "memory.db"
    db_path.touch(mode=0o644)
    assert _mode(db_path) == 0o644

    db = Database(db_path)
    try:
        db.open()
        assert _mode(db_path) == 0o600
    finally:
        db.close()


def test_backup_file_is_owner_only_under_permissive_umask(tmp_path):
    source = Database(":memory:")
    source.open()
    backup_path = tmp_path / "backup.db"
    previous_umask = os.umask(0)
    try:
        source.backup(backup_path)
    finally:
        os.umask(previous_umask)
        source.close()

    assert _mode(backup_path) == 0o600
