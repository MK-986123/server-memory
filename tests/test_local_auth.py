"""Tests for local daemon bearer token helpers."""

from __future__ import annotations

import os

import pytest

from server_memory.local_auth import (
    LocalTokenVerifier,
    compatible_auth_token_paths,
    ensure_local_auth_token,
    read_local_auth_token,
    read_local_auth_tokens,
)
from stdio_proxy import build_request_headers


def test_ensure_local_auth_token_creates_and_reads_token(tmp_path):
    token_path = tmp_path / "runtime" / "auth.token"

    token = ensure_local_auth_token(token_path)

    assert token
    assert read_local_auth_token(token_path) == token
    if os.name == "posix":
        assert oct(token_path.stat().st_mode & 0o777) == "0o600"


def test_default_auth_token_is_mirrored_to_legacy_path(monkeypatch, tmp_path):
    token_path = tmp_path / "runtime" / "auth.token"
    config_home = tmp_path / "config"
    legacy_path = config_home / "server-memory" / "runtime" / "auth.token"

    monkeypatch.delenv("MEMORY_AUTH_TOKEN_PATH", raising=False)
    monkeypatch.delenv("MEMORY_HTTP_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.setattr(
        "server_memory.local_auth.default_auth_token_path",
        lambda: token_path,
    )

    token = ensure_local_auth_token(token_path)

    assert compatible_auth_token_paths(token_path) == (token_path, legacy_path)
    assert read_local_auth_token(legacy_path) == token
    assert read_local_auth_tokens(token_path) == [token]
    if os.name == "posix":
        assert oct(legacy_path.stat().st_mode & 0o777) == "0o600"


def test_explicit_auth_token_path_does_not_use_legacy_path(monkeypatch, tmp_path):
    token_path = tmp_path / "custom" / "auth.token"
    legacy_path = tmp_path / "config" / "server-memory" / "runtime" / "auth.token"

    monkeypatch.setenv("MEMORY_AUTH_TOKEN_PATH", str(token_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_text("legacy-token\n", encoding="utf-8")

    token = ensure_local_auth_token(token_path)

    assert compatible_auth_token_paths(token_path) == (token_path,)
    assert read_local_auth_tokens(token_path) == [token]
    assert read_local_auth_token(legacy_path) == "legacy-token"


@pytest.mark.asyncio
async def test_local_token_verifier_accepts_matching_token():
    verifier = LocalTokenVerifier("expected-token")

    accepted = await verifier.verify_token("expected-token")
    rejected = await verifier.verify_token("wrong-token")

    assert accepted is not None
    assert accepted.client_id == "server-memory-local"
    assert rejected is None


def test_build_request_headers_include_session_and_auth():
    headers = build_request_headers(session_id="abc123", auth_token="secret")

    assert headers["Mcp-Session-Id"] == "abc123"
    assert headers["Authorization"] == "Bearer secret"
    assert headers["Accept"] == "application/json, text/event-stream"
