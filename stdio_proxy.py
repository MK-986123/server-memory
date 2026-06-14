#!/usr/bin/env python3
"""Stdio-to-HTTP proxy for MCP clients that only support stdio (e.g. Claude Desktop).

Reads JSON-RPC from stdin, forwards to the shared streamable-HTTP daemon,
and writes responses to stdout. This avoids SQLite locking conflicts from
multiple direct server instances.

Usage: python stdio_proxy.py [--url http://127.0.0.1:8765/mcp]
"""

import argparse
import asyncio
import json
import os
import sys
import threading
from pathlib import Path

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from server_memory.config import MemoryConfig
from server_memory.local_auth import read_local_auth_token, read_local_auth_tokens

# Lock to serialize all stdout writes — prevents interleaved chunks
# when multiple async tasks write concurrently.
_stdout_lock = threading.Lock()


DEFAULT_URL = "http://127.0.0.1:8765/mcp"
AUTH_RETRYABLE_STATUS_CODES = {401, 403}


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


def _write_stdout(data: bytes) -> None:
    """Write a complete message to stdout atomically.

    Uses a lock + os.write (single syscall) to avoid partial pipe reads
    that trigger 'Unexpected end of JSON input' in Claude Desktop's
    JSONRPC reader. Empty payloads are dropped — writing a bare newline
    causes the client to JSON.parse("") and crash the transport.
    """
    if not data or not data.strip():
        return
    payload = data + b"\n"
    fd = sys.stdout.buffer.fileno()
    with _stdout_lock:
        written = 0
        while written < len(payload):
            written += os.write(fd, payload[written:])


async def forward_request(
    client: httpx.AsyncClient,
    url: str,
    msg: dict,
    session_id: str | None,
    token_path: str | os.PathLike[str] | None,
) -> str | None:
    """Send one JSON-RPC message to the HTTP daemon and write responses to stdout."""
    auth_tokens = load_auth_tokens(token_path)
    auth_token = auth_tokens[0] if auth_tokens else None
    tried_auth_tokens = {auth_token}
    headers = build_request_headers(session_id=session_id, auth_token=auth_token)

    try:
        new_session_id = session_id
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
                        error = {
                            "jsonrpc": "2.0",
                            "id": msg.get("id"),
                            "error": {
                                "code": -32001,
                                "message": (
                                    "server-memory daemon authentication failed. "
                                    "The local auth token is missing, stale, or the stdio proxy/host "
                                    f"needs to reload {token_path}."
                                ),
                            },
                        }
                        _write_stdout(json.dumps(error).encode())
                        return new_session_id

                    continue

                ct = resp.headers.get("content-type", "")

                if "text/event-stream" in ct:
                    # Stream SSE events line by line
                    async for raw_line in resp.aiter_lines():
                        if raw_line.startswith("data: "):
                            data = raw_line[6:].strip()
                            if data:
                                _write_stdout(data.encode())
                else:
                    # JSON or other. Notifications return 202 with empty body —
                    # do not forward anything in that case.
                    if resp.status_code == 202:
                        await resp.aread()
                    else:
                        body = (await resp.aread()).strip()
                        if body:
                            _write_stdout(body)
                return new_session_id

    except httpx.ConnectError:
        err = {
            "jsonrpc": "2.0",
            "id": msg.get("id"),
            "error": {
                "code": -32000,
                "message": f"Cannot connect to server-memory daemon at {url}. "
                "Is the server-memory HTTP daemon running?",
            },
        }
        _write_stdout(json.dumps(err).encode())

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

                session_id = await forward_request(client, url, msg, session_id, token_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()
    asyncio.run(proxy(args.url))


if __name__ == "__main__":
    main()
