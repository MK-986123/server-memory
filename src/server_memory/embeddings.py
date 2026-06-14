"""Optional embedding engine for semantic search.

Gracefully degrades if sentence-transformers is not installed.
"""

from __future__ import annotations

import logging
import math
import struct
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_model_instance = None
_model_name_loaded: str = ""
_embedding_executor = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="server-memory-embeddings",
)


class EmbeddingEngine:
    """Lazy-loading embedding engine using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._available: bool | None = None

    def is_available(self) -> bool:
        """Check if sentence-transformers is installed."""
        if self._available is not None:
            return self._available
        try:
            import sentence_transformers  # noqa: F401

            self._available = True
        except ImportError:
            self._available = False
        return self._available

    def _get_model(self):
        """Lazy-load the model on first use."""
        global _model_instance, _model_name_loaded
        if _model_instance is not None and _model_name_loaded == self.model_name:
            return _model_instance
        if not self.is_available():
            return None
        from sentence_transformers import SentenceTransformer

        _model_instance = SentenceTransformer(self.model_name)
        _model_name_loaded = self.model_name
        return _model_instance

    def embed_text(self, text: str) -> bytes | None:
        """Embed a single text string. Returns raw float32 bytes or None."""
        model = self._get_model()
        if model is None:
            return None
        vec = model.encode(text, normalize_embeddings=True)
        return _pack_floats(vec)

    def embed_batch(self, texts: list[str]) -> list[bytes]:
        """Embed a batch of texts. Returns list of raw float32 bytes."""
        model = self._get_model()
        if model is None:
            return []
        vecs = model.encode(texts, normalize_embeddings=True, batch_size=64)
        return [_pack_floats(v) for v in vecs]

    def embed_text_with_timeout(
        self,
        text: str,
        timeout_seconds: float | None,
    ) -> tuple[bytes | None, bool]:
        """Embed one text with an optional timeout.

        Returns (embedding, timed_out). On timeout, returns (None, True).
        """
        if timeout_seconds is not None and timeout_seconds <= 0:
            return None, True
        if timeout_seconds is None:
            return self.embed_text(text), False

        future = _embedding_executor.submit(self.embed_text, text)
        try:
            return future.result(timeout=timeout_seconds), False
        except FutureTimeoutError:
            logger.warning(
                "Embedding text timed out after %.2fs; deferring embedding backfill",
                timeout_seconds,
            )
            return None, True

    def embed_batch_with_timeout(
        self,
        texts: list[str],
        timeout_seconds: float | None,
    ) -> tuple[list[bytes | None], bool]:
        """Embed a batch with an optional timeout.

        Returns (embeddings, timed_out). On timeout, returns a same-length list of None.
        """
        if timeout_seconds is not None and timeout_seconds <= 0:
            return [None for _ in texts], True
        if timeout_seconds is None:
            return self.embed_batch(texts), False

        future = _embedding_executor.submit(self.embed_batch, texts)
        try:
            return list(future.result(timeout=timeout_seconds)), False
        except FutureTimeoutError:
            logger.warning(
                "Embedding batch timed out after %.2fs; deferring embedding backfill",
                timeout_seconds,
            )
            return [None for _ in texts], True

    @staticmethod
    def cosine_similarity(a: bytes, b: bytes) -> float:
        """Compute cosine similarity between two embedding blobs.

        Since embeddings are L2-normalized, cosine similarity = dot product.
        """
        va = EmbeddingEngine.bytes_to_floats(a)
        vb = EmbeddingEngine.bytes_to_floats(b)
        if len(va) != len(vb):
            raise ValueError("Embedding vectors must have the same dimension")
        return sum(x * y for x, y in zip(va, vb, strict=True))

    @staticmethod
    def bytes_to_floats(blob: bytes) -> list[float]:
        """Convert embedding bytes to float list (for debugging)."""
        n = len(blob) // 4
        return list(struct.unpack(f"{n}f", blob))


def _pack_floats(values) -> bytes:
    """Pack a model output sequence into float32 bytes without requiring numpy."""
    floats = [float(v) for v in values]
    if not floats:
        return b""

    norm = math.sqrt(sum(v * v for v in floats))
    if norm and not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-6):
        floats = [v / norm for v in floats]
    return struct.pack(f"{len(floats)}f", *floats)
