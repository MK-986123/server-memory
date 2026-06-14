# Public release checklist

## Before publishing

1. Choose and add a project license if the repository should grant open-source rights.
2. Initialize a fresh Git repository for the sanitized working tree, or otherwise confirm that the history you plan to publish contains no private metadata.
3. Run the full validation commands locally:

   ```bash
   python -m compileall -q src tests scripts
   python -m ruff check .
   python -m pytest
   python -m build
   python -m twine check dist/*
   python -m pip_audit
   python scripts/smoke_stdio.py server-memory
   ```

4. Review `git status` to confirm no local data files are staged.
5. Review the staged diff for paths, credentials, local config, and generated artifacts.
6. Push to GitHub and wait for CI, CodeQL, dependency review, and secret scanning to finish before changing visibility or announcing the repository.
