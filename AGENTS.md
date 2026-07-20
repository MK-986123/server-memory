# Repository instructions for coding agents

These instructions apply to AI coding agents and automated development tools working in this repository.

## Objective

Preserve `server-memory` as a local-first MCP memory server with explicit storage, bounded recall, workspace and global scope separation, and predictable stdio behavior.

Do not introduce hosted services, external credentials, telemetry, or mandatory network dependencies into the core package without explicit maintainer approval.

## Initial setup

1. Confirm Python 3.10 or newer.
2. Confirm SQLite FTS5 support.
3. Create an isolated virtual environment.
4. Install the editable project with development dependencies.

Linux and macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -c "import sqlite3; c=sqlite3.connect(':memory:'); c.execute('CREATE VIRTUAL TABLE t USING fts5(content)'); c.close(); print('FTS5 available')"
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -c "import sqlite3; c=sqlite3.connect(':memory:'); c.execute('CREATE VIRTUAL TABLE t USING fts5(content)'); c.close(); print('FTS5 available')"
```

Install optional embeddings only when the task requires them:

```bash
python -m pip install -e ".[dev,embeddings]"
```

## Safe test environment

Never use a real user memory database for tests or exploratory commands.

Set an isolated database path and disable global memory unless the task specifically tests global scope behavior:

```bash
export MEMORY_DB_PATH="$PWD/.tmp/agent-memory.db"
export MEMORY_GLOBAL_DB_ENABLED=false
```

PowerShell:

```powershell
$env:MEMORY_DB_PATH = "$PWD\.tmp\agent-memory.db"
$env:MEMORY_GLOBAL_DB_ENABLED = "false"
```

Do not commit `.tmp`, SQLite databases, exports, backups, token files, model caches, virtual environments, or build artifacts.

## Before editing

- Read `README.md`, `CONTRIBUTING.md`, and the files directly involved in the task.
- Inspect existing tests before changing behavior.
- Preserve public tool names, parameter shapes, scope semantics, and JSON-RPC behavior unless the task explicitly requires a breaking change.
- Keep stdio stdout protocol-clean. Send diagnostics to stderr or logging.
- Keep destructive operations explicit and scoped. Do not weaken rejection of `scope="all"` for destructive tools.
- Do not add performance claims without reproducible raw results.

## Required validation

Run after every code change:

```bash
python -m compileall -q src tests scripts
python -m ruff check .
python -m pytest -q
```

Run after packaging, entry-point, dependency, or release-related changes:

```bash
rm -rf dist build
python -m build
python -m twine check dist/*
python scripts/inspect_wheel.py dist
python -m pip_audit
python scripts/smoke_stdio.py server-memory
server-memory-serve --help
server-memory-proxy --help
```

PowerShell cleanup equivalent:

```powershell
Remove-Item -Recurse -Force dist, build -ErrorAction SilentlyContinue
```

Hosted GitHub Actions are not currently an authoritative validation signal for this repository. Local command output is required.

## Reporting

At completion, report:

- files changed
- behavior changed
- exact validation commands executed
- pass or fail status for each command
- Python version and operating system
- anything not tested and the reason

Never state that tests, scans, builds, or workflows passed unless they were actually executed and their output was observed.
