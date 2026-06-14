# server-memory

[![CI](https://github.com/MK-986123/server-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/MK-986123/server-memory/actions/workflows/ci.yml)

`server-memory` is a local-first Model Context Protocol server for durable agent memory: entities, observations, relations, tags, activity, and fast recall backed by SQLite and FTS5.

It is intentionally boring where memory should be boring. Data stays in local databases unless you export it, the default MCP transport is stdio, and the optional shared HTTP daemon is bound to localhost by default.

## What it does

- Stores a small knowledge graph in SQLite with WAL mode and FTS5 search.
- Exposes 20 MCP tools for graph writes, recall, timeline queries, import/export, tagging, backup, and statistics.
- Provides compact `memory_context` output for token-budgeted agent context.
- Supports optional embedding-assisted retrieval through the `embeddings` extra.
- Keeps workspace memory and global preference memory separate by default.
- Includes a localhost HTTP daemon plus stdio proxy for clients that need one shared process.
- Uses platform-native per-user data and runtime directories through `platformdirs`.

```text
MCP client
  |
  | stdio: server-memory
  v
FastMCP server
  |
  +-- workspace SQLite DB: entities, observations, relations, tags, activity
  |
  +-- optional global preferences DB

Optional shared mode:
MCP client -> server-memory-proxy -> http://127.0.0.1:8765/mcp -> server-memory-serve
```

## Requirements

- Python 3.10 or newer
- SQLite with FTS5 enabled
- macOS, Ubuntu, or Windows for the GitHub Actions support matrix

The CI workflow is configured to test Python 3.10 through 3.14 on Ubuntu and Python 3.10 plus 3.14 on `ubuntu-latest`, `windows-latest`, and `macos-latest`.

## Install

From this GitHub repository:

```bash
python -m pip install "server-memory @ git+https://github.com/MK-986123/server-memory.git"
```

For local development from a checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Optional extras:

```bash
python -m pip install -e ".[embeddings]"
```

## Quick start

Run the stdio MCP server:

```bash
server-memory
```

Equivalent module form:

```bash
python -m server_memory
```

Use a dedicated database for one project:

```bash
MEMORY_DB_PATH=<PROJECT_ROOT>/memory.db server-memory
```

Run the shared localhost daemon:

```bash
server-memory-serve --host 127.0.0.1 --port 8765 --transport streamable-http
```

Connect stdio-only clients to that daemon:

```bash
server-memory-proxy --url http://127.0.0.1:8765/mcp
```

## MCP client config

Stdio server:

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

Shared daemon proxy:

```json
{
  "mcpServers": {
    "server-memory": {
      "command": "server-memory-proxy",
      "args": ["--url", "http://127.0.0.1:8765/mcp"]
    }
  }
}
```

## Tools

`server-memory` registers tools only; it does not register MCP resources or prompts.

| Tool | Purpose | Main inputs |
| --- | --- | --- |
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
| `memory_stats` | Return counts and storage stats | `scope` |
| `backup_memory` | Copy a SQLite database | `dest_path`, `scope` |
| `get_observation_history` | Show observation versions for an entity | `entity_name`, `content_prefix`, `scope` |
| `delete_entities` | Soft-delete or hard-delete entities | `entityNames`, `hard`, `scope` |
| `delete_observations` | Delete selected observations | `deletions`, `scope` |
| `delete_relations` | Delete selected relations | `relations`, `scope` |

Write tools modify the configured SQLite database. `backup_memory` writes a database backup. `export_graph` can print sensitive memory content, so review exports before sharing.

Most tools accept `scope="workspace"`, `scope="global"`, or `scope="all"`. Workspace remains the default for ordinary project memory; preference-tagged writes still auto-route to the global database when global preference routing is enabled. `scope="all"` returns source-labeled results so same-named entities are not merged silently.

## Configuration

Configuration is environment-driven.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MEMORY_DB_PATH` | Platform user data dir, workspace-namespaced when a project root is detected | Workspace SQLite database |
| `MEMORY_COMPRESSION_LEVEL` | `4` | Compression level, `0` through `4` |
| `MEMORY_TOKEN_BUDGET` | `2000` | Output token budget |
| `MEMORY_IMPORT_JSONL` | unset | Import JSONL on startup |
| `MEMORY_SESSION_ID` | unset | Session ID for activity logging |
| `MEMORY_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Optional embedding model |
| `MEMORY_EMBEDDING_ENABLED` | `true` | Enable embedding search/backfill |
| `MEMORY_WRITE_EMBEDDING_BUDGET_MS` | `10000` | Write-path embedding budget |
| `MEMORY_PROJECT` | empty | Default project scope |
| `MEMORY_DEDUP_THRESHOLD` | `0.92` | Semantic dedup threshold |
| `MEMORY_HTTP_AUTH_ENABLED` | `true` | Require bearer auth for the shared HTTP daemon |
| `MEMORY_AUTH_TOKEN_PATH` | Platform runtime dir | Local HTTP daemon token file |
| `MEMORY_GLOBAL_DB_ENABLED` | `true` | Enable global preferences database |
| `MEMORY_GLOBAL_DB_PATH` | Platform user data dir | Global preferences SQLite database |
| `MEMORY_GLOBAL_PREFERENCE_ROUTING_ENABLED` | `true` | Route preference-tagged writes to global memory |
| `MEMORY_WORKSPACE_ROOT` | unset | Explicit workspace root for default DB placement |
| `MEMORY_WORKSPACE_ID` | unset | Explicit workspace ID for default DB placement |

`.env.example` contains a safe template with empty override values.

## Development

```bash
python -m pip install -e ".[dev]"
python -m compileall -q src tests scripts
python -m ruff check .
python -m pytest
python -m build
python -m twine check dist/*
python -m pip_audit
python scripts/smoke_stdio.py server-memory
server-memory-serve --help
server-memory-proxy --help
```

The smoke test sends an MCP `initialize` request to the installed stdio entry point and fails if stdout contains non-protocol text.

## Security and privacy

- Memory databases, exports, and backups can contain sensitive user data.
- The stdio server writes protocol data to stdout; diagnostics should go to stderr or logs.
- The shared HTTP daemon defaults to `127.0.0.1` and local bearer-token auth.
- The bearer token is generated locally and stored under a platform-native runtime directory unless `MEMORY_AUTH_TOKEN_PATH` is set.
- No external service credentials are required for the core server.
- Optional embeddings may load local or cached model files depending on your environment and installed extras.

Report vulnerabilities through GitHub private vulnerability reporting when available. Do not include secrets or private memory exports in public issues.

## CI and supply chain

GitHub Actions are configured to run syntax checks, Ruff, pytest, package build, wheel installation, MCP stdio smoke tests, `pip-audit`, CodeQL, and Dependency Review. Dependabot is configured for Python dependencies and GitHub Actions.

The workflow uses GitHub-hosted `ubuntu-latest`, `windows-latest`, and `macos-latest` labels. GitHub defines those as the latest stable runner images it provides, which can lag the newest vendor OS release during image migrations.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `no such module: fts5` | Use a Python build linked against SQLite with FTS5 enabled. |
| MCP client hangs at startup | Run `python scripts/smoke_stdio.py server-memory` and inspect stderr. |
| Multiple clients lock the database | Use `server-memory-serve` once and connect clients through `server-memory-proxy`. |
| Proxy returns auth failure | Restart the daemon and client so both read the same `MEMORY_AUTH_TOKEN_PATH`. |
| Unexpected memory location | Set `MEMORY_DB_PATH`, `MEMORY_WORKSPACE_ROOT`, or `MEMORY_WORKSPACE_ID` explicitly. |

## License

No open-source license has been selected. Public visibility does not grant reuse rights by itself.
