"""Local-only bearer token helpers for the shared HTTP daemon."""

from __future__ import annotations

import hmac
import os
import secrets
from contextlib import suppress
from pathlib import Path

from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings

from .paths import APP_NAME, default_auth_token_path

LOCAL_AUTH_CLIENT_ID = "server-memory-local"


def _format_base_url(host: str, port: int) -> str:
    """Build a localhost base URL that also handles IPv6 hosts."""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}"


def build_local_auth_settings(host: str, port: int) -> AuthSettings:
    """Create minimal auth settings required by FastMCP's token verifier flow."""
    base_url = _format_base_url(host, port)
    return AuthSettings(
        issuer_url=base_url,
        resource_server_url=base_url,
        required_scopes=[],
    )


def read_local_auth_token(path: Path) -> str | None:
    """Read an auth token from env override or token file if available."""
    override = os.environ.get("MEMORY_HTTP_AUTH_TOKEN", "").strip()
    if override:
        return override

    with suppress(FileNotFoundError):
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token

    return None


def _legacy_default_auth_token_path() -> Path:
    config_root = Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser()
    return config_root / APP_NAME / "runtime" / "auth.token"


def _is_default_auth_token_path(path: Path) -> bool:
    if os.environ.get("MEMORY_AUTH_TOKEN_PATH"):
        return False

    try:
        return path.expanduser() == default_auth_token_path().expanduser()
    except OSError:
        return False


def compatible_auth_token_paths(path: Path) -> tuple[Path, ...]:
    """Return primary plus legacy default token paths for local daemon compatibility."""
    primary = path.expanduser()
    paths = [primary]

    if _is_default_auth_token_path(primary):
        legacy = _legacy_default_auth_token_path()
        if legacy != primary:
            paths.append(legacy)

    return tuple(paths)


def read_local_auth_tokens(path: Path) -> list[str]:
    """Read candidate auth tokens in the order a local proxy should try them."""
    override = os.environ.get("MEMORY_HTTP_AUTH_TOKEN", "").strip()
    if override:
        return [override]

    tokens: list[str] = []
    seen: set[str] = set()
    for candidate_path in compatible_auth_token_paths(path):
        with suppress(FileNotFoundError):
            token = candidate_path.read_text(encoding="utf-8").strip()
            if token and token not in seen:
                tokens.append(token)
                seen.add(token)

    return tokens


def _write_private_token(path: Path, token: str) -> None:
    """Write the token with owner-only permissions where possible."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if os.name == "posix":
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(token)
        with suppress(OSError):
            path.chmod(0o600)
        return

    path.write_text(token, encoding="utf-8")
    with suppress(OSError):
        path.chmod(0o600)


def _sync_compatible_auth_token(path: Path, token: str) -> None:
    if os.environ.get("MEMORY_HTTP_AUTH_TOKEN"):
        return

    for compatible_path in compatible_auth_token_paths(path)[1:]:
        current = read_local_auth_token(compatible_path)
        if current == token:
            continue
        _write_private_token(compatible_path, f"{token}\n")


def ensure_local_auth_token(path: Path) -> str:
    """Return an existing local auth token or create a new one."""
    existing = read_local_auth_token(path)
    if existing:
        _sync_compatible_auth_token(path, existing)
        return existing

    token = secrets.token_urlsafe(32)
    _write_private_token(path, f"{token}\n")
    _sync_compatible_auth_token(path, token)
    return token


class LocalTokenVerifier(TokenVerifier):
    """Token verifier for the local shared daemon bearer token."""

    def __init__(self, expected_token: str, client_id: str = LOCAL_AUTH_CLIENT_ID):
        self.expected_token = expected_token
        self.client_id = client_id

    async def verify_token(self, token: str) -> AccessToken | None:
        if not hmac.compare_digest(token, self.expected_token):
            return None

        return AccessToken(
            token=token,
            client_id=self.client_id,
            scopes=[],
        )
