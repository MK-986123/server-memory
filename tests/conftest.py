"""Shared fixtures for tests."""

import pytest

import server_memory.graph as _graph_mod
from server_memory.db import Database
from server_memory.graph import KnowledgeGraphManager


@pytest.fixture(autouse=True)
def _reset_backfill_state():
    _graph_mod._active_embedding_backfills.clear()
    yield
    _graph_mod._active_embedding_backfills.clear()


@pytest.fixture
def db():
    """In-memory database for testing."""
    database = Database(":memory:")
    database.open()
    yield database
    database.close()


@pytest.fixture
def graph(db):
    """KnowledgeGraphManager with in-memory DB."""
    return KnowledgeGraphManager(db, session_id="test-session")
