"""Cross-platform application directories for server-memory runtime files."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

try:
    from platformdirs import PlatformDirs
except ImportError:  # pragma: no cover - exercised when dependency missing at runtime
    PlatformDirs = None


APP_NAME = "server-memory"
APP_AUTHOR = "server-memory"
WORKSPACES_DIRNAME = "workspaces"
GLOBAL_DIRNAME = "global"
PROJECT_MARKERS = (
    ".git",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "README.md",
)


@dataclass(frozen=True)
class AppPaths:
    """Resolved per-user directories for persistent/config/runtime data."""

    data_dir: Path
    config_dir: Path
    runtime_dir: Path


def _slugify_workspace_label(value: str) -> str:
    """Convert a project/workspace label into a filesystem-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "workspace"


def _find_workspace_marker(start: Path) -> Path | None:
    """Walk upward from a path to find the nearest likely project root."""
    candidate = start.resolve()
    if candidate.is_file():
        candidate = candidate.parent

    for path in (candidate, *candidate.parents):
        for marker in PROJECT_MARKERS:
            if (path / marker).exists():
                return path
    return None


def _resolve_workspace_root() -> Path | None:
    """Resolve a workspace root from env or the current working directory."""
    explicit_root = os.environ.get("MEMORY_WORKSPACE_ROOT", "").strip()
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()

    cwd_root = _find_workspace_marker(Path.cwd())
    if cwd_root is None:
        return None

    home = Path.home().resolve()
    return None if cwd_root == home else cwd_root


def _workspace_dir_name(workspace_root: Path | None, workspace_id: str, project: str) -> str | None:
    """Build a stable directory name for a workspace-specific DB path."""
    workspace_id = workspace_id.strip()
    project = project.strip()

    if workspace_id:
        label = workspace_id
        fingerprint_source = workspace_id
    elif workspace_root is not None:
        label = workspace_root.name
        fingerprint_source = str(workspace_root)
    elif project:
        label = project
        fingerprint_source = project
    else:
        return None

    suffix = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()[:12]
    return f"{_slugify_workspace_label(label)}_{suffix}"


def _legacy_app_paths() -> AppPaths:
    """Return sensible user paths when platformdirs is unavailable."""
    data_root = Path(os.environ.get("XDG_DATA_HOME", "~/.local/share")).expanduser()
    config_root = Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser()
    runtime_base = os.environ.get("XDG_RUNTIME_DIR")

    if runtime_base:
        runtime_dir = Path(runtime_base).expanduser() / APP_NAME
    else:
        runtime_dir = config_root / APP_NAME / "runtime"

    return AppPaths(
        data_dir=data_root / APP_NAME,
        config_dir=config_root / APP_NAME,
        runtime_dir=runtime_dir,
    )


def app_paths() -> AppPaths:
    """Return OS-native per-user directories for server-memory files."""
    if PlatformDirs is None:
        return _legacy_app_paths()

    dirs = PlatformDirs(APP_NAME, APP_AUTHOR)
    runtime_path = getattr(dirs, "user_runtime_path", None)
    if runtime_path is None:
        runtime_path = Path(dirs.user_config_path) / "runtime"

    return AppPaths(
        data_dir=Path(dirs.user_data_path),
        config_dir=Path(dirs.user_config_path),
        runtime_dir=Path(runtime_path),
    )


def default_db_path() -> Path:
    """Return the default central SQLite path for the user's memory database."""
    paths = app_paths()
    workspace_root = _resolve_workspace_root()
    workspace_id = os.environ.get("MEMORY_WORKSPACE_ID", "")
    project = os.environ.get("MEMORY_PROJECT", "")
    workspace_dir_name = _workspace_dir_name(workspace_root, workspace_id, project)

    if workspace_dir_name is None:
        return paths.data_dir / "memory.db"

    return paths.data_dir / WORKSPACES_DIRNAME / workspace_dir_name / "memory.db"


def default_auth_token_path() -> Path:
    """Return the default location for the local daemon bearer token."""
    return app_paths().runtime_dir / "auth.token"


def default_global_db_path() -> Path:
    """Return the default location for the shared global preferences database."""
    return app_paths().data_dir / GLOBAL_DIRNAME / "preferences.db"
