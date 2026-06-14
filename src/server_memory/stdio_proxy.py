"""Stdio-to-HTTP proxy for shared server-memory daemon mode."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

import httpx

from .config import MemoryConfig
from .local_auth import read_local_auth_token, read_local_auth_tokens

DEFAULT_URL = "http://127.0.0.1:8765/mcp"
AUTH_RETRYABLE_STATUS_CODES = {401, 403}

_stdout_lock = threading.Lock()


def build_request_headers(session_id: str | None, auth_token: str | None) -> dict[str, str]:
    """Build MCP HTTP headers for one proxied request."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if session_id:
        headers["Mcp-Session-Id"] = session_id
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    return headers


def load_auth_token(token_path: str | os.PathLike[str] | None = None) -> str | None:
    """Read the current daemon auth token from env override or token file."""
    cfg = MemoryConfig() if token_path is None else None
    resolved_path = cfg.auth_token_path if cfg is not None else Path(token_path)
    return read_local_auth_token(resolved_path)


def load_auth_tokens(token_path: str | os.PathLike[str] | None = None) -> list[str]:
    """Read candidate daemon auth tokens in retry order."""
    cfg = MemoryConfig() if token_path is None else None
    resolved_path = cfg.auth_token_path if cfg is not None else Path(token_path)
    return read_local_auth_tokens(resolved_path)


def _has_request_id(msg: dict[str, Any]) -> bool:
    return "id" in msg and msg.get("id") is not None


def _jsonrpc_error(msg: dict[str, Any], code: int, message: str) -> bytes | None:
    """Build a JSON-RPC error only for requests; notifications get no output."""
    if not _has_request_id(msg):
        return None
    return json.dumps(
        {
            "jsonrpc": "2.0",
            "id": msg.get("id"),
            "error": {"code": code, "message": message},
        },
        separators=(",", ":"),
    ).encode("utf-8")


def _write_stdout(data: bytes) -> None:
    """Write one complete validated JSON-RPC message to stdout atomically."""
    if not data or not data.strip():
        return
    payload = data + b"\n"
    fd = sys.stdout.buffer.fileno()
    with _stdout_lock:
        written = 0
        while written < len(payload):
            written += os.write(fd, payload[written:])


def _is_valid_single_jsonrpc_msg(item: Any) -> bool:
    """Validate a single JSON-RPC 2.0 item."""
    if not isinstance(item, dict):
        return False
    if item.get("jsonrpc") != "2.0":
        return False

    has_method = "method" in item
    has_result = "result" in item
    has_error = "error" in item

    if has_method:
        # Request or Notification
        if not isinstance(item["method"], str):
            return False
        if "id" in item:
            req_id = item["id"]
            if req_id is not None and not isinstance(req_id, (str, int, float)):
                return False
        return True

    # Response or Error
    if (has_result and has_error) or (not has_result and not has_error):
        return False

    if "id" not in item:
        return False
    resp_id = item["id"]
    if resp_id is not None and not isinstance(resp_id, (str, int, float)):
        return False

    if has_error:
        err = item["error"]
        if not isinstance(err, dict):
            return False
        if not isinstance(err.get("code"), int):
            return False
        if not isinstance(err.get("message"), str):
            return False

    return True


def _is_valid_jsonrpc_msg(msg: Any) -> bool:
    """Return True if msg matches a valid JSON-RPC 2.0 structure (single or batch)."""
    if isinstance(msg, list):
        if not msg:
            return False
        return all(_is_valid_single_jsonrpc_msg(item) for item in msg)
    return _is_valid_single_jsonrpc_msg(msg)


def _validated_json_payload(data: bytes | str) -> bytes | None:
    """Return canonical JSON bytes for valid JSON-RPC 2.0 messages, otherwise None."""
    if isinstance(data, bytes):
        stripped = data.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
    else:
        stripped_text = data.strip()
        if not stripped_text:
            return None
        try:
            parsed = json.loads(stripped_text)
        except json.JSONDecodeError:
            return None

    if not _is_valid_jsonrpc_msg(parsed):
        return None

    return json.dumps(parsed, separators=(",", ":")).encode("utf-8")


async def _emit_json_body_or_error(resp: httpx.Response, msg: dict[str, Any]) -> None:
    body = await resp.aread()
    payload = _validated_json_payload(body)
    if payload is not None:
        _write_stdout(payload)
        return
    error = _jsonrpc_error(msg, -32002, "server-memory daemon returned an invalid JSON response")
    if error is not None:
        _write_stdout(error)


async def _emit_sse_payloads(resp: httpx.Response) -> None:
    async for raw_line in resp.aiter_lines():
        line = raw_line.strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data:
            continue
        payload = _validated_json_payload(data)
        if payload is not None:
            _write_stdout(payload)


async def forward_request(
    client: httpx.AsyncClient,
    url: str,
    msg: dict[str, Any],
    session_id: str | None,
    token_path: str | os.PathLike[str] | None,
) -> str | None:
    """Send one JSON-RPC message to the HTTP daemon and write valid responses to stdout."""
    auth_tokens = load_auth_tokens(token_path)
    auth_token = auth_tokens[0] if auth_tokens else None
    tried_auth_tokens = {auth_token}
    headers = build_request_headers(session_id=session_id, auth_token=auth_token)
    new_session_id = session_id

    try:
        while True:
            async with client.stream("POST", url, json=msg, headers=headers) as resp:
                if "mcp-session-id" in resp.headers:
                    new_session_id = resp.headers["mcp-session-id"]

                if resp.status_code in AUTH_RETRYABLE_STATUS_CODES:
                    await resp.aread()
                    refreshed_auth_tokens = load_auth_tokens(token_path)
                    for refreshed_auth_token in refreshed_auth_tokens:
                        if refreshed_auth_token not in tried_auth_tokens:
                            auth_token = refreshed_auth_token
                            tried_auth_tokens.add(auth_token)
                            headers = build_request_headers(
                                session_id=new_session_id,
                                auth_token=auth_token,
                            )
                            break
                    else:
                        error = _jsonrpc_error(
                            msg,
                            -32001,
                            "server-memory daemon authentication failed",
                        )
                        if error is not None:
                            _write_stdout(error)
                        return new_session_id
                    continue

                if resp.status_code >= 400:
                    await resp.aread()
                    error = _jsonrpc_error(
                        msg,
                        -32000,
                        f"server-memory daemon returned HTTP {resp.status_code}",
                    )
                    if error is not None:
                        _write_stdout(error)
                    return new_session_id

                if resp.status_code in {202, 204}:
                    await resp.aread()
                    return new_session_id

                content_type = resp.headers.get("content-type", "")
                if "text/event-stream" in content_type:
                    await _emit_sse_payloads(resp)
                else:
                    await _emit_json_body_or_error(resp, msg)
                return new_session_id

    except httpx.RequestError:
        error = _jsonrpc_error(
            msg,
            -32000,
            "Cannot connect to server-memory daemon. Is the local daemon running?",
        )
        if error is not None:
            _write_stdout(error)

    return new_session_id


async def proxy(url: str) -> None:
    session_id: str | None = None
    token_path = MemoryConfig().auth_token_path
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin.buffer)

    async with httpx.AsyncClient(timeout=60) as client:
        buf = b""
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                break
            buf += chunk

            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue

                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(msg, dict):
                    session_id = await forward_request(client, url, msg, session_id, token_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="stdio proxy for a server-memory HTTP daemon")
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()
    asyncio.run(proxy(args.url))


if __name__ == "__main__":
    main()
