"""Tests for embedding module (mocked, no real model required)."""

import math
import struct
import threading
import time
from unittest.mock import patch


def test_is_available_when_not_installed():
    """is_available() returns False when sentence-transformers is not installed."""
    from server_memory.embeddings import EmbeddingEngine

    engine = EmbeddingEngine()
    engine._available = None  # Reset cache
    with patch.dict("sys.modules", {"sentence_transformers": None}):
        # Force re-check by clearing import cache
        engine._available = None
        with patch("builtins.__import__", side_effect=ImportError):
            engine._available = None
            result = engine.is_available()
            assert result is False


def test_is_available_when_installed():
    """is_available() returns True when sentence-transformers is installed."""
    from server_memory.embeddings import EmbeddingEngine

    engine = EmbeddingEngine()
    engine._available = True
    assert engine.is_available() is True


def test_embed_text_returns_none_when_unavailable():
    """embed_text returns None when embeddings are unavailable."""
    from server_memory.embeddings import EmbeddingEngine

    engine = EmbeddingEngine()
    engine._available = False
    assert engine.embed_text("hello") is None


def test_embed_batch_returns_empty_when_unavailable():
    """embed_batch returns empty list when embeddings are unavailable."""
    from server_memory.embeddings import EmbeddingEngine

    engine = EmbeddingEngine()
    engine._available = False
    assert engine.embed_batch(["hello", "world"]) == []


def test_cosine_similarity_with_known_vectors():
    """Cosine similarity should compute correct values for known vectors."""
    from server_memory.embeddings import EmbeddingEngine

    # Identical normalized vectors → similarity = 1.0
    v1_bytes = struct.pack("3f", 1.0, 0.0, 0.0)
    sim = EmbeddingEngine.cosine_similarity(v1_bytes, v1_bytes)
    assert abs(sim - 1.0) < 1e-6

    # Orthogonal vectors → similarity = 0.0
    v2_bytes = struct.pack("3f", 0.0, 1.0, 0.0)
    sim = EmbeddingEngine.cosine_similarity(v1_bytes, v2_bytes)
    assert abs(sim - 0.0) < 1e-6

    # Opposite vectors → similarity = -1.0
    v3_bytes = struct.pack("3f", -1.0, 0.0, 0.0)
    sim = EmbeddingEngine.cosine_similarity(v1_bytes, v3_bytes)
    assert abs(sim - (-1.0)) < 1e-6


def test_cosine_similarity_with_partial_overlap():
    """Cosine similarity for partially overlapping normalized vectors."""
    from server_memory.embeddings import EmbeddingEngine

    # Normalized 45-degree angle vectors
    v1 = struct.pack("2f", 1.0, 0.0)
    v2 = struct.pack("2f", 1.0 / math.sqrt(2), 1.0 / math.sqrt(2))
    sim = EmbeddingEngine.cosine_similarity(v1, v2)
    assert abs(sim - (1.0 / math.sqrt(2))) < 1e-5


def test_bytes_to_floats():
    """bytes_to_floats should correctly convert blob to float list."""
    from server_memory.embeddings import EmbeddingEngine

    floats = [1.0, 2.0, 3.0]
    blob = struct.pack("3f", *floats)
    result = EmbeddingEngine.bytes_to_floats(blob)
    assert len(result) == 3
    for a, b in zip(result, floats):
        assert abs(a - b) < 1e-6


def test_timed_out_embedding_does_not_block_the_next_request(monkeypatch):
    from server_memory.embeddings import EmbeddingEngine

    engine = EmbeddingEngine()

    def embed(text: str):
        if text == "slow":
            time.sleep(0.2)
        return b"ok"

    monkeypatch.setattr(engine, "embed_text", embed)
    value, timed_out = engine.embed_text_with_timeout("slow", 0.01)
    started = time.monotonic()
    next_value, next_timed_out = engine.embed_text_with_timeout("fast", 0.1)

    assert value is None and timed_out
    assert next_value == b"ok" and not next_timed_out
    assert time.monotonic() - started < 0.08


def test_repeated_embedding_timeouts_use_at_most_two_workers(monkeypatch):
    from server_memory.embeddings import EmbeddingEngine

    engine = EmbeddingEngine()
    monkeypatch.setattr(engine, "embed_text", lambda _text: time.sleep(0.15))

    for index in range(8):
        value, timed_out = engine.embed_text_with_timeout(str(index), 0.005)
        assert value is None and timed_out

    workers = [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith("server-memory-embedding")
    ]
    assert len(workers) <= 2
