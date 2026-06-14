"""Shared HTTP daemon entry point for server-memory."""

from __future__ import annotations

import argparse
import os

from .config import MemoryConfig
from .server import create_server

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def main() -> None:
    """Run the memory MCP server as a shared localhost daemon."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("MEMORY_HTTP_AUTH_ENABLED", "true")

    parser = argparse.ArgumentParser(description="server-memory MCP daemon")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
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
