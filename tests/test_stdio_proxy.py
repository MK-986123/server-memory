"""Tests for stdio proxy auth token refresh and error handling."""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx
import pytest

from server_memory import stdio_proxy


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


class _RaisingClient:
    def __init__(self, exc: httpx.RequestError):
        self.exc = exc

    def stream(self, method: str, url: str, json: dict, headers: dict[str, str]):
        raise self.exc


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
    assert "authentication failed" in payload["error"]["message"].lower()


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


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [404, 500])
async def test_forward_request_converts_http_errors_to_jsonrpc_errors(
    monkeypatch, tmp_path, status_code
):
    outputs: list[bytes] = []
    monkeypatch.setattr(stdio_proxy, "_write_stdout", lambda data: outputs.append(data))
    monkeypatch.setattr(stdio_proxy, "load_auth_tokens", lambda path: [])

    client = _FakeClient([_FakeStreamResponse(status_code=status_code, body=b"<html>no</html>")])

    await stdio_proxy.forward_request(
        client=client,
        url="http://127.0.0.1:8765/mcp",
        msg={"jsonrpc": "2.0", "id": 3, "method": "tools/call"},
        session_id=None,
        token_path=tmp_path / "auth.token",
    )

    payload = json.loads(outputs[0].decode("utf-8"))
    assert payload["id"] == 3
    assert payload["error"]["code"] == -32000
    assert f"HTTP {status_code}" in payload["error"]["message"]
    assert "html" not in payload["error"]["message"].lower()


@pytest.mark.asyncio
async def test_forward_request_emits_nothing_for_failed_notification(monkeypatch, tmp_path):
    outputs: list[bytes] = []
    monkeypatch.setattr(stdio_proxy, "_write_stdout", lambda data: outputs.append(data))
    monkeypatch.setattr(stdio_proxy, "load_auth_tokens", lambda path: [])

    client = _FakeClient([_FakeStreamResponse(status_code=500, body=b"server failure")])

    await stdio_proxy.forward_request(
        client=client,
        url="http://127.0.0.1:8765/mcp",
        msg={"jsonrpc": "2.0", "method": "notifications/initialized"},
        session_id=None,
        token_path=tmp_path / "auth.token",
    )

    assert outputs == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [
        httpx.TimeoutException("timed out"),
        httpx.RequestError("network failure"),
    ],
)
async def test_forward_request_catches_request_errors(monkeypatch, tmp_path, exc):
    outputs: list[bytes] = []
    monkeypatch.setattr(stdio_proxy, "_write_stdout", lambda data: outputs.append(data))
    monkeypatch.setattr(stdio_proxy, "load_auth_tokens", lambda path: [])

    await stdio_proxy.forward_request(
        client=_RaisingClient(exc),
        url="http://127.0.0.1:8765/mcp",
        msg={"jsonrpc": "2.0", "id": 4, "method": "tools/call"},
        session_id=None,
        token_path=tmp_path / "auth.token",
    )

    payload = json.loads(outputs[0].decode("utf-8"))
    assert payload["id"] == 4
    assert payload["error"]["code"] == -32000
    assert "local daemon" in payload["error"]["message"]
    assert "timed out" not in payload["error"]["message"]
    assert "network failure" not in payload["error"]["message"]


@pytest.mark.asyncio
async def test_forward_request_converts_malformed_json_body_to_jsonrpc_error(
    monkeypatch, tmp_path
):
    outputs: list[bytes] = []
    monkeypatch.setattr(stdio_proxy, "_write_stdout", lambda data: outputs.append(data))
    monkeypatch.setattr(stdio_proxy, "load_auth_tokens", lambda path: [])

    client = _FakeClient([_FakeStreamResponse(status_code=200, body=b"not json")])

    await stdio_proxy.forward_request(
        client=client,
        url="http://127.0.0.1:8765/mcp",
        msg={"jsonrpc": "2.0", "id": 5, "method": "tools/call"},
        session_id=None,
        token_path=tmp_path / "auth.token",
    )

    payload = json.loads(outputs[0].decode("utf-8"))
    assert payload["id"] == 5
    assert payload["error"]["code"] == -32002
    assert "not json" not in payload["error"]["message"].lower()


@pytest.mark.asyncio
async def test_forward_request_ignores_malformed_empty_comment_and_keepalive_sse(
    monkeypatch, tmp_path
):
    outputs: list[bytes] = []
    monkeypatch.setattr(stdio_proxy, "_write_stdout", lambda data: outputs.append(data))
    monkeypatch.setattr(stdio_proxy, "load_auth_tokens", lambda path: [])

    client = _FakeClient(
        [
            _FakeStreamResponse(
                status_code=200,
                body=(
                    b": keepalive\n"
                    b"\n"
                    b"event: ping\n"
                    b"data:\n"
                    b"data: not json\n"
                    b"data: {\"jsonrpc\":\"2.0\",\"id\":9,\"result\":{\"ok\":true}}\n"
                ),
                headers={"content-type": "text/event-stream"},
            )
        ]
    )

    await stdio_proxy.forward_request(
        client=client,
        url="http://127.0.0.1:8765/mcp",
        msg={"jsonrpc": "2.0", "id": 9, "method": "tools/call"},
        session_id=None,
        token_path=tmp_path / "auth.token",
    )

    assert outputs == [b'{"jsonrpc":"2.0","id":9,"result":{"ok":true}}']


@pytest.mark.asyncio
async def test_forward_request_forwards_valid_json_response(monkeypatch, tmp_path):
    outputs: list[bytes] = []
    monkeypatch.setattr(stdio_proxy, "_write_stdout", lambda data: outputs.append(data))
    monkeypatch.setattr(stdio_proxy, "load_auth_tokens", lambda path: [])

    client = _FakeClient(
        [_FakeStreamResponse(status_code=200, body=b'{"jsonrpc":"2.0","id":12,"result":{}}')]
    )

    await stdio_proxy.forward_request(
        client=client,
        url="http://127.0.0.1:8765/mcp",
        msg={"jsonrpc": "2.0", "id": 12, "method": "tools/call"},
        session_id=None,
        token_path=tmp_path / "auth.token",
    )

    assert outputs == [b'{"jsonrpc":"2.0","id":12,"result":{}}']


@pytest.mark.asyncio
async def test_forward_request_rejects_non_jsonrpc_valid_json(monkeypatch, tmp_path):
    outputs: list[bytes] = []
    monkeypatch.setattr(stdio_proxy, "_write_stdout", lambda data: outputs.append(data))
    monkeypatch.setattr(stdio_proxy, "load_auth_tokens", lambda path: [])

    client = _FakeClient(
        [
            _FakeStreamResponse(status_code=200, body=b'{"not_jsonrpc": true}'),
            _FakeStreamResponse(status_code=200, body=b'[{"jsonrpc":"2.0","result":{}}]'),
            _FakeStreamResponse(status_code=200, body=b'{"jsonrpc":"2.0","id":12,"result":{},"error":{}}'),
            _FakeStreamResponse(status_code=200, body=b'{"jsonrpc":"2.0","id":12}'),
            _FakeStreamResponse(status_code=200, body=b'{"jsonrpc":"2.0","id":12,"error":{"code":"not-int","message":"foo"}}'),
            _FakeStreamResponse(status_code=200, body=b'{"jsonrpc":"2.0","id":12,"error":{"code":-32000,"message":123}}'),
        ]
    )

    for _ in range(6):
        await stdio_proxy.forward_request(
            client=client,
            url="http://127.0.0.1:8765/mcp",
            msg={"jsonrpc": "2.0", "id": 12, "method": "tools/call"},
            session_id=None,
            token_path=tmp_path / "auth.token",
        )

    # 6 request errors should be emitted
    assert len(outputs) == 6
    for out in outputs:
        payload = json.loads(out.decode("utf-8"))
        assert payload["jsonrpc"] == "2.0"
        assert payload["id"] == 12
        assert payload["error"]["code"] == -32002

    # Now test with notifications (no id)
    outputs.clear()
    client_notif = _FakeClient(
        [
            _FakeStreamResponse(status_code=200, body=b'{"not_jsonrpc": true}'),
        ]
    )
    await stdio_proxy.forward_request(
        client=client_notif,
        url="http://127.0.0.1:8765/mcp",
        msg={"jsonrpc": "2.0", "method": "notifications/initialized"},
        session_id=None,
        token_path=tmp_path / "auth.token",
    )
    assert outputs == []


def test_strict_jsonrpc_validation():
    # ID validations
    assert stdio_proxy._is_valid_id(None) is True
    assert stdio_proxy._is_valid_id("id-123") is True
    assert stdio_proxy._is_valid_id(123) is True
    assert stdio_proxy._is_valid_id(12.3) is True
    assert stdio_proxy._is_valid_id(True) is False
    assert stdio_proxy._is_valid_id(False) is False

    # Single message validation
    # Params must be structured (dict or list)
    assert stdio_proxy._is_valid_single_jsonrpc_msg(
        {"jsonrpc": "2.0", "method": "foo", "params": "invalid_string"}
    ) is False
    assert stdio_proxy._is_valid_single_jsonrpc_msg(
        {"jsonrpc": "2.0", "method": "foo", "params": []}
    ) is True
    assert stdio_proxy._is_valid_single_jsonrpc_msg(
        {"jsonrpc": "2.0", "method": "foo", "params": {}}
    ) is True

    # Error code cannot be bool
    assert stdio_proxy._is_valid_single_jsonrpc_msg(
        {"jsonrpc": "2.0", "id": 123, "error": {"code": True, "message": "msg"}}
    ) is False
    assert stdio_proxy._is_valid_single_jsonrpc_msg(
        {"jsonrpc": "2.0", "id": 123, "error": {"code": -32600, "message": "msg"}}
    ) is True

    # ID in response cannot be bool
    assert stdio_proxy._is_valid_single_jsonrpc_msg(
        {"jsonrpc": "2.0", "id": True, "result": {}}
    ) is False

