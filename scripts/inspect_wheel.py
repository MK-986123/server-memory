"""Validate built wheel contents for packaged entry-point modules."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

REQUIRED_MEMBERS = {
    "server_memory/serve.py",
    "server_memory/stdio_proxy.py",
}
REQUIRED_ENTRY_POINTS = {
    "server-memory = server_memory.__main__:main",
    "server-memory-serve = server_memory.serve:main",
    "server-memory-proxy = server_memory.stdio_proxy:main",
}


def _find_wheel(dist_dir: Path) -> Path:
    wheels = sorted(dist_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one wheel in {dist_dir}, found {len(wheels)}")
    return wheels[0]


def inspect_wheel(wheel_path: Path) -> None:
    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
        missing_members = sorted(REQUIRED_MEMBERS - names)
        if missing_members:
            raise SystemExit(f"wheel missing package files: {missing_members}")

        entry_point_files = [name for name in names if name.endswith(".dist-info/entry_points.txt")]
        if len(entry_point_files) != 1:
            raise SystemExit(f"expected one entry_points.txt, found {entry_point_files}")

        entry_points = wheel.read(entry_point_files[0]).decode("utf-8")
        missing_entry_points = sorted(
            entry_point for entry_point in REQUIRED_ENTRY_POINTS if entry_point not in entry_points
        )
        if missing_entry_points:
            raise SystemExit(f"wheel missing console scripts: {missing_entry_points}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist_dir", nargs="?", default="dist")
    args = parser.parse_args()

    wheel_path = _find_wheel(Path(args.dist_dir))
    inspect_wheel(wheel_path)
    print(f"wheel contents ok: {wheel_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
