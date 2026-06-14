"""Development entry point for `python -m server_memory`."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version

from .config import MemoryConfig
from .server import create_server


def _package_version() -> str:
    try:
        return version("server-memory")
    except PackageNotFoundError:
        return "0+local"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the server-memory MCP stdio server.")
    parser.add_argument("--version", action="version", version=f"server-memory {_package_version()}")
    parser.parse_args()

    config = MemoryConfig()
    server = create_server(config)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
