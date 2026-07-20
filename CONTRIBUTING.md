# Contributing

Thanks for considering a contribution.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Optional embeddings:

```bash
python -m pip install -e ".[dev,embeddings]"
```

Verify SQLite FTS5 support:

```bash
python -c "import sqlite3; c=sqlite3.connect(':memory:'); c.execute('CREATE VIRTUAL TABLE t USING fts5(content)'); c.close(); print('FTS5 available')"
```

## Validation

Hosted GitHub Actions are not currently treated as an active validation source. Run the relevant checks locally and report the exact commands and output in the pull request.

Required for every code change:

```bash
python -m compileall -q src tests scripts
python -m ruff check .
python -m pytest -q
```

Required for packaging, entry-point, dependency, or release changes:

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

On PowerShell:

```powershell
Remove-Item -Recurse -Force dist, build -ErrorAction SilentlyContinue
python -m build
python -m twine check dist/*
python scripts/inspect_wheel.py dist
python -m pip_audit
python scripts/smoke_stdio.py server-memory
server-memory-serve --help
server-memory-proxy --help
```

Do not claim a check passed unless it was actually executed. When a check cannot run, state the blocker and what remains unverified.

## Pull requests

- Keep changes scoped and reviewable.
- Update documentation when behavior changes.
- Add or update tests for behavior changes.
- Include the Python version and operating system used for validation.
- Include exact failing output when requesting help.
- Do not commit local databases, virtual environments, build artifacts, or editor-specific files.

## AI coding agents

Repository-aware coding agents should follow [`AGENTS.md`](AGENTS.md) before installing dependencies, modifying files, or running validation.

## Security-sensitive content

Never commit:

- secrets, tokens, cookies, or private keys
- real user data or personal notes
- live memory databases, exports, backups, or authentication-token files
- local machine paths in documentation or scripts when relative paths are sufficient
