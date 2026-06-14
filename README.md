<h1 align="center">server-memory</h1>

<p align="center">
  <strong>Local-first durable memory for MCP agents, backed by SQLite and FTS5.</strong>
</p>

<p align="center">
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="MCP server" src="https://img.shields.io/badge/MCP-server-6f42c1">
  <img alt="SQLite and FTS5" src="https://img.shields.io/badge/storage-SQLite%20%2B%20FTS5-003B57?logo=sqlite&logoColor=white">
  <img alt="Local-first" src="https://img.shields.io/badge/design-local--first-2ea44f">
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#mcp-client-configuration">Client configuration</a> ·
  <a href="#tool-reference">Tools</a> ·
  <a href="#configuration">Configuration</a> ·
  <a href="#security-and-privacy">Security</a>
</p>

---

## Overview

`server-memory` is a local-first Model Context Protocol (MCP) server for durable agent memory: entities, observations, relations, tags, activity, and fast recall backed by SQLite and FTS5.

> It is intentionally boring where memory should be boring.

Data stays in local databases unless you export it. The default MCP transport is stdio, and the optional shared HTTP daemon is bound to localhost by default.

### At a glance

| Capability | Implementation |
| :--- | :--- |
| **Storage** | SQLite with WAL mode and FTS5 search |
| **Memory model** | Entities, observations, relations, tags, and activity |
| **MCP interface** | 20 tools; no resources or prompts |
| **Default transport** | stdio |
| **Shared mode** | Localhost HTTP daemon with a stdio proxy |
| **Memory scopes** | Workspace memory and optional global preference memory |
| **Retrieval** | FTS5 search with optional embedding assistance |
| **Data paths** | Platform-native user data and runtime directories through `platformdirs` |

### Highlights

- **Local knowledge graph:** Stores entities, observations, relations, tags, and activity in SQLite with WAL mode and FTS5 search.
- **MCP tools:** Exposes 20 tools for graph writes, recall, timeline queries, import/export, tagging, backup, and statistics.
- **Compact context:** Provides token-budgeted `memory_context` output for routine agent recall.
- **Optional embeddings:** Supports embedding-assisted retrieval through the `embeddings` extra.
- **Memory scopes:** Keeps workspace memory and global preference memory separate by default.
- **Shared mode:** Provides a localhost HTTP daemon and stdio proxy for clients that need one shared process.
- **Platform-native paths:** Uses `platformdirs` for per-user data and runtime directories.

---

## Architecture

### Default stdio mode

```text
┌────────────┐         stdio         ┌─────────────────────┐
│ MCP client │ ────────────────────> │    server-memory    │
└────────────┘                       │   FastMCP server     │
                                     ├─────────────────────┤
                                     │ Workspace SQLite DB │
                                     │                     │
                                     │ Global preferences  │
                                     │ DB, when enabled    │
                                     └─────────────────────┘
```

### Optional shared mode

```text
┌────────────┐      stdio       ┌─────────────────────┐
│ MCP client │ ───────────────> │ server-memory-proxy │
└────────────┘                  └──────────┬──────────┘
                                         │
                                         │ HTTP
                                         │ 127.0.0.1:8765/mcp
                                         ▼
                              ┌─────────────────────┐
                              │ server-memory-serve │
                              │ FastMCP daemon      │
                              └─────────────────────┘
```

---

## Requirements

- Python 3.10 or newer
- SQLite with FTS5 enabled
- macOS, Ubuntu, or Windows for the GitHub Actions support matrix

The CI workflow is configured to test:

- Python 3.10 through 3.14 on Ubuntu
- Python 3.10 and 3.14 on `ubuntu-latest`
- Python 3.10 and 3.14 on `windows-latest`
- Python 3.10 and 3.14 on `macos-latest`

---

## Installation

### Core installation

Install directly from this GitHub repository:

```bash
python -m pip install "server-memory @ git+https://github.com/MK-986123/server-memory.git"
```

### Installation with embeddings

```bash
python -m pip install "server-memory[embeddings] @ git+https://github.com/MK-986123/server-memory.git"
```

### Development checkout

```bash
git clone https://github.com/MK-986123/server-memory.git
cd server-memory

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install development and embedding dependencies together:

```bash
python -m pip install -e ".[dev,embeddings]"
```

---

## Quick start

### Run the stdio server

```bash
server-memory
```

Equivalent module form:

```bash
python -m server_memory
```

### Use a dedicated project database

```bash
MEMORY_DB_PATH=<PROJECT_ROOT>/memory.db server-memory
```

Use the equivalent environment-variable syntax for your shell when running on Windows.

### Run the shared localhost daemon

```bash
server-memory-serve \
  --host 127.0.0.1 \
  --port 8765 \
  --transport streamable-http
```

### Connect a stdio-only client to the daemon

```bash
server-memory-proxy --url http://127.0.0.1:8765/mcp
```

---

## MCP client configuration

### Direct stdio server

```json
{
  "mcpServers": {
    "server-memory": {
      "command": "server-memory",
      "env": {
        "MEMORY_PROJECT": "<PROJECT_NAME>"
      }
    }
  }
}
```

### Shared daemon proxy

Start `server-memory-serve` separately, then configure the MCP client to launch the proxy:

```json
{
  "mcpServers": {
    "server-memory": {
      "command": "server-memory-proxy",
      "args": [
        "--url",
        "http://127.0.0.1:8765/mcp"
      ]
    }
  }
}
```

---

## Tool reference

`server-memory` registers MCP tools only. It does not register resources or prompts.

### Scope behavior

Most tools accept one of three scopes:

| Scope | Behavior |
| :--- | :--- |
| `workspace` | Operates on the current workspace database and remains the default for ordinary project memory |
| `global` | Operates on the global preferences database |
| `all` | Combines supported workspace and global results with source labels |

Preference-tagged writes still auto-route to the global database when global preference routing is enabled.

> [!IMPORTANT]
> Destructive operations require an explicit `workspace` or `global` scope. They reject `scope="all"` to prevent accidental cross-database deletion, merging, or tag removal.

<details>
<summary><strong>View all 20 MCP tools</strong></summary>

| Tool | Purpose | Main inputs |
| :--- | :--- | :--- |
| `memory_context` | Compact scoped recall for ordinary agent context | `hint`, `project`, `limit`, `scope` |
| `memory_context_full` | Larger bootstrap context with pinned and recent items | `project`, `budget`, `scope` |
| `create_entities` | Add entities and optional initial observations | `entities`, `scope` |
| `add_observations` | Add observations to existing entities | `observations`, `scope` |
| `create_relations` | Connect existing entities | `relations`, `scope` |
| `read_graph` | Read graph data, compressed by default | `tags`, `entity_types`, `limit`, `include_deleted`, `compress`, `scope` |
| `search_nodes` | FTS5 search with filters | `query`, `tags`, `entity_types`, `time_range`, `limit`, `compress`, `scope` |
| `open_nodes` | Open named entities and optional neighbors | `names`, `depth`, `scope` |
| `log_activity` | Record a development or session event | `action`, `summary`, `entity_names`, `tags`, `metadata`, `scope` |
| `query_timeline` | Query activity history | `time_range`, `start`, `end`, `actions`, `entity_name`, `session_id`, `limit`, `scope` |
| `manage_tags` | List, create, delete, apply, remove, or clean tags | `action`, `name`, `entity_name`, `tag_name`, `scope` |
| `merge_entities` | Merge one entity into another | `source`, `target`, `strategy`, `scope` |
| `export_graph` | Export graph as JSON or JSONL | `format`, `scope` |
| `import_graph` | Import JSON or JSONL graph data | `data`, `scope` |
| `memory_stats` | Return counts and storage statistics | `scope` |
| `backup_memory` | Copy a SQLite database | `dest_path`, `scope` |
| `get_observation_history` | Show observation versions for an entity | `entity_name`, `content_prefix`, `scope` |
| `delete_entities` | Soft-delete or hard-delete entities | `entityNames`, `hard`, `scope` |
| `delete_observations` | Delete selected observations | `deletions`, `scope` |
| `delete_relations` | Delete selected relations | `relations`, `scope` |

</details>

Write tools modify the selected SQLite database. `backup_memory` writes a database backup. `export_graph` can print sensitive memory content, so review exports before sharing them.

---

## Configuration

Configuration is environment-driven. Empty path overrides in `.env.example` use the platform defaults.

<details>
<summary><strong>View all environment variables</strong></summary>

### Storage and scope

| Variable | Default | Meaning |
| :--- | :--- | :--- |
| `MEMORY_DB_PATH` | Platform user data directory, workspace-namespaced when a project root is detected | Workspace SQLite database |
| `MEMORY_PROJECT` | Empty | Default project scope |
| `MEMORY_GLOBAL_DB_ENABLED` | `true` | Enable the global preferences database |
| `MEMORY_GLOBAL_DB_PATH` | Platform user data directory | Global preferences SQLite database |
| `MEMORY_GLOBAL_PREFERENCE_ROUTING_ENABLED` | `true` | Route preference-tagged writes to global memory |
| `MEMORY_WORKSPACE_ROOT` | Unset | Explicit workspace root for default database placement |
| `MEMORY_WORKSPACE_ID` | Unset | Explicit workspace identifier for default database placement |

### Retrieval and compression

| Variable | Default | Meaning |
| :--- | :--- | :--- |
| `MEMORY_COMPRESSION_LEVEL` | `4` | Compression level from `0` through `4` |
| `MEMORY_TOKEN_BUDGET` | `2000` | Output token budget |
| `MEMORY_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Optional embedding model |
| `MEMORY_EMBEDDING_ENABLED` | `true` | Enable embedding search and backfill |
| `MEMORY_WRITE_EMBEDDING_BUDGET_MS` | `10000` | Write-path embedding budget |
| `MEMORY_DEDUP_THRESHOLD` | `0.92` | Semantic deduplication threshold |

### Runtime and shared daemon

| Variable | Default | Meaning |
| :--- | :--- | :--- |
| `MEMORY_IMPORT_JSONL` | Unset | Import JSONL on startup |
| `MEMORY_SESSION_ID` | Unset | Session identifier for activity logging |
| `MEMORY_HTTP_AUTH_ENABLED` | `true` | Require bearer authentication for the shared HTTP daemon |
| `MEMORY_AUTH_TOKEN_PATH` | Platform runtime directory | Local HTTP daemon token file |

</details>

---

## Development

Install the development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run the local validation sequence:

```bash
python -m compileall -q src tests scripts
python -m ruff check .
python -m pytest -q
python -m build
python -m twine check dist/*
python scripts/inspect_wheel.py dist
python -m pip_audit
```

Verify the installed entry points:

```bash
python scripts/smoke_stdio.py server-memory
server-memory-serve --help
server-memory-proxy --help
```

The stdio smoke test sends an MCP `initialize` request to the installed entry point and fails if stdout contains non-protocol output.

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidance.

---

## Security and privacy

> [!IMPORTANT]
> Memory databases, exports, backups, and activity logs can contain sensitive user data.

- The stdio server writes protocol data to stdout. Diagnostics should go to stderr or logs.
- The shared HTTP daemon defaults to `127.0.0.1` and local bearer-token authentication.
- The bearer token is generated locally and stored under a platform-native runtime directory unless `MEMORY_AUTH_TOKEN_PATH` is set.
- No external service credentials are required for the core server.
- Optional embeddings may load local or cached model files depending on the environment and installed extras.
- Review exported graph content before sharing it.
- Do not commit live memory databases, token files, or backups.

Report vulnerabilities through GitHub private vulnerability reporting when available. Do not include secrets or private memory exports in public issues.

See [SECURITY.md](SECURITY.md) for the project security policy.

---

## CI and supply chain

GitHub Actions are configured to run:

- syntax validation
- Ruff linting
- pytest across the supported Python and operating-system matrix
- wheel and source distribution builds
- wheel-content inspection
- clean installed-package checks outside the repository checkout
- MCP stdio and installed-command smoke tests
- `pip-audit`
- CodeQL
- Dependency Review

Dependabot is configured for Python dependencies and GitHub Actions.

The workflow uses GitHub-hosted `ubuntu-latest`, `windows-latest`, and `macos-latest` labels. These labels refer to GitHub's latest stable runner images and can temporarily lag the newest vendor operating-system release during image migrations.

---

## Troubleshooting

| Symptom | Check |
| :--- | :--- |
| `no such module: fts5` | Use a Python build linked against SQLite with FTS5 enabled. |
| MCP client hangs at startup | Run `python scripts/smoke_stdio.py server-memory` and inspect stderr. |
| Multiple clients lock the database | Run one `server-memory-serve` process and connect clients through `server-memory-proxy`. |
| Proxy returns an authentication failure | Restart the daemon and client so both read the same `MEMORY_AUTH_TOKEN_PATH`. |
| Memory is stored in an unexpected location | Set `MEMORY_DB_PATH`, `MEMORY_WORKSPACE_ROOT`, or `MEMORY_WORKSPACE_ID` explicitly. |

---

## License

> [!CAUTION]
> No open-source license has been selected. Public visibility does not grant permission to copy, modify, redistribute, or reuse the project.
