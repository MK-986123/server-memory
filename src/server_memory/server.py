"""Stable public MCP server API with bounded UTF-8 import validation."""

from __future__ import annotations

from functools import wraps
from typing import Any

from . import _server_impl as _impl

# Re-export the established server module surface, including private helpers
# used by repository tests and diagnostics.
for _name, _value in vars(_impl).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

IMPORT_PAYLOAD_LIMIT_BYTES = 50 * 1024 * 1024
IMPORT_SIZE_CHUNK_CHARS = 1024 * 1024


def _utf8_payload_exceeds_limit(data: str, limit: int) -> bool:
    """Count UTF-8 bytes incrementally without allocating one full encoded copy."""
    if limit < 0:
        raise ValueError("limit must be non-negative")
    total = 0
    for offset in range(0, len(data), IMPORT_SIZE_CHUNK_CHARS):
        total += len(data[offset : offset + IMPORT_SIZE_CHUNK_CHARS].encode("utf-8"))
        if total > limit:
            return True
    return False


@wraps(_impl.create_server)
def create_server(*args: Any, **kwargs: Any):
    """Create the MCP server and enforce the documented byte-sized import cap."""
    mcp = _impl.create_server(*args, **kwargs)
    tool = mcp._tool_manager._tools["import_graph"]
    original = tool.fn

    @wraps(original)
    def import_graph_with_byte_limit(*tool_args: Any, **tool_kwargs: Any):
        data = tool_kwargs.get("data")
        if data is None and len(tool_args) >= 2:
            data = tool_args[1]
        if isinstance(data, str) and _utf8_payload_exceeds_limit(
            data, IMPORT_PAYLOAD_LIMIT_BYTES
        ):
            raise ValueError("import payload exceeds 50 MiB UTF-8 limit")
        return original(*tool_args, **tool_kwargs)

    tool.fn = import_graph_with_byte_limit
    return mcp
