"""Tests for stdio proxy auth token refresh and error handling."""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

import stdio_proxy


class _FakeStreamResponse:
    def __init__(self, status_code: int, body: bytes, headers: dict[str, str] | None = None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {"content-type": "application/json"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aread(self) -> bytes:
        return self._body

    async def aiter_lines(self) -> Iterator[str]:
        for line in self._body.decode("utf-8").splitlines():
            yield line


class _FakeClient:
    def __init__(self, responses: list[_FakeStreamResponse]):
        self._responses = responses
        self.calls: list[dict[str, object]] = []

    def stream(self, method: str, url: str, json: dict, headers: dict[str, str]):
        self.calls.append({"method": method, "url": url, "json": json, "headers": dict(headers)})
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_forward_request_reloads_token_after_unauthorized(monkeypatch, tmp_path):
    outputs: list[bytes] = []
    monkeypatch.setattr(stdio_proxy, "_write_stdout", lambda data: outputs.append(data))

    token_values = iter([[], ["fresh-token"]])
    monkeypatch.setattr(stdio_proxy, "load_auth_tokens", lambda path: next(token_values))

    client = _FakeClient(
        [
            _FakeStreamResponse(status_code=401, body=b"unauthorized"),
            _FakeStreamResponse(
                status_code=200,
                body=b'{"jsonrpc":"2.0","id":1,"result":{"ok":true}}',
            ),
        ]
    )

    session_id = await stdio_proxy.forward_request(
        client=client,
        url="http://127.0.0.1:8765/mcp",
        msg={"jsonrpc": "2.0", "id": 1, "method": "tools/call"},
        session_id=None,
        token_path=tmp_path / "auth.token",
    )

    assert session_id is None
    assert "Authorization" not in client.calls[0]["headers"]
    assert client.calls[1]["headers"]["Authorization"] == "Bearer fresh-token"
    assert outputs == [b'{"jsonrpc":"2.0","id":1,"result":{"ok":true}}']


@pytest.mark.asyncio
async def test_forward_request_returns_jsonrpc_error_after_persistent_auth_failure(
    monkeypatch, tmp_path
):
    outputs: list[bytes] = []
    monkeypatch.setattr(stdio_proxy, "_write_stdout", lambda data: outputs.append(data))
    monkeypatch.setattr(stdio_proxy, "load_auth_tokens", lambda path: ["stale-token"])

    client = _FakeClient([_FakeStreamResponse(status_code=401, body=b"unauthorized")])

    await stdio_proxy.forward_request(
        client=client,
        url="http://127.0.0.1:8765/mcp",
        msg={"jsonrpc": "2.0", "id": 7, "method": "tools/call"},
        session_id="session-1",
        token_path=tmp_path / "auth.token",
    )

    assert len(outputs) == 1
    payload = json.loads(outputs[0].decode("utf-8"))
    assert payload["id"] == 7
    assert payload["error"]["code"] == -32001
    assert "auth token" in payload["error"]["message"].lower()


@pytest.mark.asyncio
async def test_forward_request_retries_alternate_candidate_token(monkeypatch, tmp_path):
    outputs: list[bytes] = []
    monkeypatch.setattr(stdio_proxy, "_write_stdout", lambda data: outputs.append(data))
    monkeypatch.setattr(stdio_proxy, "load_auth_tokens", lambda path: ["stale-token", "fresh-token"])

    client = _FakeClient(
        [
            _FakeStreamResponse(status_code=401, body=b"unauthorized"),
            _FakeStreamResponse(
                status_code=200,
                body=b'{"jsonrpc":"2.0","id":11,"result":{"ok":true}}',
            ),
        ]
    )

    await stdio_proxy.forward_request(
        client=client,
        url="http://127.0.0.1:8765/mcp",
        msg={"jsonrpc": "2.0", "id": 11, "method": "tools/call"},
        session_id=None,
        token_path=tmp_path / "auth.token",
    )

    assert client.calls[0]["headers"]["Authorization"] == "Bearer stale-token"
    assert client.calls[1]["headers"]["Authorization"] == "Bearer fresh-token"
    assert outputs == [b'{"jsonrpc":"2.0","id":11,"result":{"ok":true}}']
