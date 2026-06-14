# Contributing

Thanks for considering a contribution.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Optional extras:

```bash
python -m pip install -e '.[embeddings]'
python -m pip install -e '.[desktop-proxy]'
python -m pip install -e '.[benchmark]'
```

## Validation

Run these before opening a pull request:

```bash
python -m compileall -q src tests scripts
python -m ruff check .
python -m pytest
python -m build
python -m twine check dist/*
python -m pip_audit
```

## Pull requests

- Keep changes scoped and reviewable.
- Update documentation when behavior changes.
- Add or update tests for behavior changes.
- Do not commit local databases, benchmark outputs, virtual environments, or editor-specific files.

## Security-sensitive content

Never commit:

- secrets, tokens, cookies, or private keys
- real user data or personal notes
- local machine paths in docs or scripts when relative paths are sufficient
- generated benchmark outputs unless explicitly requested for review
