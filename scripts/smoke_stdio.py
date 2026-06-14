#!/usr/bin/env python3
"""Smoke-test a server-memory MCP stdio entry point."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    command = sys.argv[1:] or ["server-memory"]
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "server-memory-smoke", "version": "1.0.0"},
        },
    }

    env = os.environ.copy()
    env["MEMORY_DB_PATH"] = str(Path(tempfile.mkdtemp()) / "memory.db")
    env["MEMORY_GLOBAL_DB_ENABLED"] = "false"
    env["PYTHONUNBUFFERED"] = "1"

    popen_kwargs: dict[str, object] = {
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "env": env,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(command, **popen_kwargs)
    try:
        stdout, stderr = proc.communicate(json.dumps(request) + "\n", timeout=30)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.terminate()
        stdout, stderr = proc.communicate(timeout=15)

    response = None
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"Non-JSON stdout line: {line!r}") from exc
        if message.get("id") == 1:
            response = message
            break

    assert response is not None, f"Missing initialize response. stderr={stderr!r}"
    assert response.get("jsonrpc") == "2.0", response
    assert isinstance(response.get("result"), dict), response
    print("stdio initialize smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
