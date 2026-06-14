"""Development shim for running the src/ package without installation."""

from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC_PACKAGE = _HERE.parent / "src" / "server_memory"

if _SRC_PACKAGE.is_dir():
    __path__.append(str(_SRC_PACKAGE))

__version__ = "1.0.0"
