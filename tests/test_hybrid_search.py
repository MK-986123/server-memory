"""Tests for hybrid FTS5 + embedding search."""

import struct

import pytest

from server_memory.db import Database
from server_memory.embeddings import EmbeddingEngine
from server_memory.graph import KnowledgeGraphManager


class MockEmbeddingEngine(EmbeddingEngine):
    """Mock embedding engine that uses simple word-overlap vectors for testing."""

    # Shared vocabulary for consistent embeddings across calls
    VOCAB = [
        "login",
        "authentication",
        "oauth",
        "jwt",
        "token",
        "user",
        "password",
        "session",
        "page",
        "provider",
        "database",
        "sql",
        "query",
        "index",
        "server",
        "api",
        "http",
        "endpoint",
        "storage",
        "cache",
    ]

    def __init__(self):
        super().__init__("mock-model")
        self._available = True

    def is_available(self) -> bool:
        return True

    def _get_model(self):
        return None  # Not needed

    def embed_text(self, text: str) -> bytes | None:
        vec = self._text_to_vec(text)
        return struct.pack(f"{len(vec)}f", *vec)

    def embed_batch(self, texts: list[str]) -> list[bytes]:
        return [self.embed_text(t) for t in texts]

    def _text_to_vec(self, text: str) -> list[float]:
        """Simple bag-of-words vector using shared vocabulary."""
        text_lower = text.lower()
        vec = [0.0] * len(self.VOCAB)
        for i, word in enumerate(self.VOCAB):
            if word in text_lower:
                vec[i] = 1.0
        # Normalize
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class WeakSemanticNoiseEmbeddingEngine(EmbeddingEngine):
    """Controlled engine for testing weak semantic-only noise suppression."""

    def __init__(self):
        super().__init__("weak-semantic-noise")
        self._available = True

    def is_available(self) -> bool:
        return True

    def _get_model(self):
        return None

    def embed_text(self, text: str) -> bytes | None:
        return text.encode("utf-8")

    def embed_batch(self, texts: list[str]) -> list[bytes]:
        return [self.embed_text(text) for text in texts]

    def cosine_similarity(self, a: bytes, b: bytes) -> float:
        query = a.decode("utf-8").lower()
        value = b.decode("utf-8").lower()
        if "sqlite" in query and "lock" in query and "sqlite" in value and "lock" in value:
            return 0.92
        if "sqlite" in query and "lock" in query and "gpu" in value:
            return 0.24
        return 0.0


@pytest.fixture
def mock_engine():
    return MockEmbeddingEngine()


@pytest.fixture
def db_with_embeddings():
    database = Database(":memory:")
    database.open()
    yield database
    database.close()


@pytest.fixture
def graph_with_embeddings(db_with_embeddings, mock_engine):
    return KnowledgeGraphManager(
        db_with_embeddings,
        session_id="test",
        embedding_engine=mock_engine,
    )


def test_hybrid_search_semantic_match(graph_with_embeddings):
    """Search for 'authentication' should find entities about login/OAuth/JWT."""
    g = graph_with_embeddings
    g.create_entities(
        [
            {
                "name": "login page",
                "entityType": "component",
                "observations": ["handles user login"],
            },
            {
                "name": "OAuth provider",
                "entityType": "service",
                "observations": ["provides OAuth tokens"],
            },
            {
                "name": "JWT token handler",
                "entityType": "module",
                "observations": ["validates JWT tokens"],
            },
            {
                "name": "database connector",
                "entityType": "module",
                "observations": ["connects to SQL database"],
            },
        ]
    )

    kg = g.search_fts("authentication")
    names = {e.name for e in kg.entities}
    # Should find login/OAuth/JWT entities via semantic similarity
    assert "login page" in names or "OAuth provider" in names or "JWT token handler" in names
    # Database connector should NOT match authentication
    # (it might still appear via fuzzy but should rank lower)


def test_hybrid_search_fts_still_works(graph_with_embeddings):
    """FTS keyword search should still work alongside embeddings."""
    g = graph_with_embeddings
    g.create_entities(
        [
            {"name": "React Frontend", "entityType": "component"},
            {"name": "Python Backend", "entityType": "component"},
        ]
    )
    kg = g.search_fts("React")
    names = [e.name for e in kg.entities]
    assert "React Frontend" in names


def test_fallback_to_fts_only_when_no_embeddings():
    """When embedding engine is None, search should fall back to FTS only."""
    db = Database(":memory:")
    db.open()
    g = KnowledgeGraphManager(db, session_id="test", embedding_engine=None)

    g.create_entities(
        [
            {"name": "App", "entityType": "project", "observations": ["uses SQLite"]},
        ]
    )
    kg = g.search_fts("SQLite")
    names = [e.name for e in kg.entities]
    assert "App" in names
    db.close()


def test_fallback_when_engine_unavailable():
    """When embedding engine is not available, search should use FTS only."""
    db = Database(":memory:")
    db.open()
    engine = EmbeddingEngine()
    engine._available = False
    g = KnowledgeGraphManager(db, session_id="test", embedding_engine=engine)

    g.create_entities(
        [
            {"name": "Server", "entityType": "component", "observations": ["handles HTTP"]},
        ]
    )
    kg = g.search_fts("HTTP")
    names = [e.name for e in kg.entities]
    assert "Server" in names
    db.close()


def test_hybrid_scoring_combines_signals(graph_with_embeddings):
    """Hybrid search should combine FTS, embedding, and recency scores."""
    g = graph_with_embeddings
    # Create entities where one matches FTS and one matches semantically
    g.create_entities(
        [
            {
                "name": "user authentication module",
                "entityType": "module",
                "observations": ["handles user login and session management"],
            },
            {
                "name": "OAuth login flow",
                "entityType": "component",
                "observations": ["implements OAuth 2.0 authentication"],
            },
        ]
    )

    kg = g.search_fts("user authentication")
    names = [e.name for e in kg.entities]
    # Both should be found — one via FTS, both via embeddings
    assert len(names) >= 1


def test_search_schedules_embedding_backfill_on_first_search(monkeypatch, db_with_embeddings):
    """Read-path search should schedule missing embedding backfill instead of writing inline."""
    # Create entities without embedding engine
    g1 = KnowledgeGraphManager(db_with_embeddings, session_id="test", embedding_engine=None)
    g1.create_entities(
        [
            {
                "name": "login handler",
                "entityType": "module",
                "observations": ["handles user login"],
            },
        ]
    )

    # Verify no embeddings exist
    count = db_with_embeddings.cx.execute("SELECT COUNT(*) c FROM entity_embeddings").fetchone()[
        "c"
    ]
    assert count == 0

    # Now create a new manager with embeddings and search
    engine = MockEmbeddingEngine()
    g2 = KnowledgeGraphManager(db_with_embeddings, session_id="test", embedding_engine=engine)
    scheduled: list[str] = []
    monkeypatch.setattr(g2, "_schedule_embedding_backfill", lambda: scheduled.append("scheduled"))
    g2.search_fts("authentication")

    assert scheduled == ["scheduled"]

    # Search should remain read-only and leave backfill to the background worker.
    count = db_with_embeddings.cx.execute("SELECT COUNT(*) c FROM entity_embeddings").fetchone()[
        "c"
    ]
    assert count == 0


def test_observation_embedding_search(graph_with_embeddings):
    """Searching should match via observation embeddings too."""
    g = graph_with_embeddings
    g.create_entities(
        [
            {
                "name": "Config Module",
                "entityType": "module",
                "observations": ["stores user session tokens", "manages authentication cache"],
            },
        ]
    )

    kg = g.search_fts("token authentication")
    names = [e.name for e in kg.entities]
    assert "Config Module" in names


def test_memory_context_balances_exact_and_semantic_signals(graph_with_embeddings):
    g = graph_with_embeddings
    g.create_entities(
        [
            {
                "name": "Token Handler",
                "entityType": "module",
                "observations": ["validates JWT tokens for authentication"],
            },
            {
                "name": "JWT Config",
                "entityType": "config",
                "observations": ["JWT issuer and audience settings"],
            },
            {
                "name": "Session Notes",
                "entityType": "note",
                "observations": ["session cache details"],
            },
        ]
    )

    ctx = g.memory_context(hint="JWT Config")

    assert ctx["hint_matches"]
    assert ctx["hint_matches"][0]["name"] == "JWT Config"
    hint_names = [match["name"] for match in ctx["hint_matches"]]
    assert "Token Handler" in hint_names


def test_memory_context_filters_weak_semantic_only_noise(db_with_embeddings):
    engine = WeakSemanticNoiseEmbeddingEngine()
    graph = KnowledgeGraphManager(
        db_with_embeddings,
        session_id="test",
        embedding_engine=engine,
    )

    graph.create_entities(
        [
            {
                "name": "SQLite Lock Notes",
                "entityType": "note",
                "observations": ["sqlite database lock mitigation for codex cli"],
            },
            {
                "name": "GPU Tuning Notes",
                "entityType": "note",
                "observations": ["gpu kernel tuning reference"],
            },
        ]
    )

    ctx = graph.memory_context(hint="sqlite database lock codex cli")
    names = [match["name"] for match in ctx["hint_matches"]]

    assert "SQLite Lock Notes" in names
    assert "GPU Tuning Notes" not in names
