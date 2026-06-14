"""Configuration from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

from .paths import default_auth_token_path, default_db_path, default_global_db_path


class CompressionLevel(IntEnum):
    NONE = 0
    LIGHT = 1
    MEDIUM = 2
    HEAVY = 3
    AUTO = 4  # Dynamically pick best level to fit budget


def env_path_or_default(name: str, default: Path) -> Path:
    """Resolve a path env var, treating blank values as unset."""
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return Path(value)


@dataclass(frozen=True)
class MemoryConfig:
    """Configuration resolved from environment variables and workspace context."""

    db_path: Path = field(default_factory=lambda: env_path_or_default("MEMORY_DB_PATH", default_db_path()))
    compression_level: CompressionLevel = field(
        default_factory=lambda: CompressionLevel(
            int(os.environ.get("MEMORY_COMPRESSION_LEVEL", "4"))  # Default AUTO
        )
    )
    token_budget: int = field(
        default_factory=lambda: int(os.environ.get("MEMORY_TOKEN_BUDGET", "2000"))
    )
    import_jsonl: str | None = field(default_factory=lambda: os.environ.get("MEMORY_IMPORT_JSONL"))
    session_id: str | None = field(default_factory=lambda: os.environ.get("MEMORY_SESSION_ID"))
    embedding_model: str = field(
        default_factory=lambda: os.environ.get("MEMORY_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    embedding_enabled: bool = field(
        default_factory=lambda: (
            os.environ.get("MEMORY_EMBEDDING_ENABLED", "true").lower() in ("true", "1", "yes")
        )
    )
    write_embedding_budget_ms: int = field(
        default_factory=lambda: int(os.environ.get("MEMORY_WRITE_EMBEDDING_BUDGET_MS", "10000"))
    )
    project: str = field(default_factory=lambda: os.environ.get("MEMORY_PROJECT", ""))
    dedup_threshold: float = field(
        default_factory=lambda: float(os.environ.get("MEMORY_DEDUP_THRESHOLD", "0.92"))
    )
    http_auth_enabled: bool = field(
        default_factory=lambda: (
            os.environ.get("MEMORY_HTTP_AUTH_ENABLED", "true").lower()
            in ("true", "1", "yes")
        )
    )
    auth_token_path: Path = field(
        default_factory=lambda: env_path_or_default(
            "MEMORY_AUTH_TOKEN_PATH",
            default_auth_token_path(),
        )
    )
    global_db_enabled: bool = field(
        default_factory=lambda: (
            os.environ.get("MEMORY_GLOBAL_DB_ENABLED", "true").lower() in ("true", "1", "yes")
        )
    )
    global_db_path: Path = field(
        default_factory=lambda: env_path_or_default(
            "MEMORY_GLOBAL_DB_PATH",
            default_global_db_path(),
        )
    )
    global_preference_routing_enabled: bool = field(
        default_factory=lambda: (
            os.environ.get("MEMORY_GLOBAL_PREFERENCE_ROUTING_ENABLED", "true").lower()
            in ("true", "1", "yes")
        )
    )

    def ensure_db_dir(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def ensure_auth_token_dir(self) -> None:
        self.auth_token_path.parent.mkdir(parents=True, exist_ok=True)

    def ensure_global_db_dir(self) -> None:
        self.global_db_path.parent.mkdir(parents=True, exist_ok=True)
