#!/usr/bin/env python3
"""Run the memory MCP server as a shared network daemon.

Supports streamable-http (default) and sse transports.
Usage:
    python serve.py                    # streamable-http on :8765
    python serve.py --port 9000        # custom port
    python serve.py --transport sse    # legacy SSE mode
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MEMORY_HTTP_AUTH_ENABLED", "true")

from server_memory.config import MemoryConfig
from server_memory.server import create_server


def main():
    parser = argparse.ArgumentParser(description="server-memory MCP daemon")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--transport",
        choices=["streamable-http", "sse"],
        default="streamable-http",
    )
    args = parser.parse_args()

    config = MemoryConfig()
    server = create_server(
        config,
        host=args.host,
        port=args.port,
        enable_http_auth=config.http_auth_enabled,
    )
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
