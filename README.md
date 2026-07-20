<h1 align="center">server-memory</h1>

<p align="center">
  <strong>Local-first durable memory for MCP agents, backed by SQLite and FTS5.</strong>
</p>

<p align="center">
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="MCP server" src="https://img.shields.io/badge/MCP-server-6f42c1">
  <img alt="SQLite and FTS5" src="https://img.shields.io/badge/storage-SQLite%20%2B%20FTS5-003B57?logo=sqlite&logoColor=white">
  <img alt="Local-first" src="https://img.shields.io/badge/design-local--first-2ea44f">
  <img alt="MIT license" src="https://img.shields.io/badge/license-MIT-blue">
</p>

<p align="center">
  <a href="#why-server-memory">Why</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#tool-reference">Tools</a> ·
  <a href="#evaluation">Evaluation</a> ·
  <a href="#security-and-privacy">Security</a>
</p>

---

## Overview

`server-memory` is an open-source Model Context Protocol server for durable AI-agent memory. It stores project facts, decisions, observations, relations, preferences, and activity in local SQLite databases, then returns compact, scoped context when an agent needs continuity across sessions.

It is designed for agents that repeatedly work on the same repositories, systems, incidents, or long-running tasks and need to remember what was already learned without replaying an entire conversation or loading the full knowledge graph every turn.

> It is intentionally boring where memory should be boring: local storage, explicit tools, inspectable data, bounded output, and predictable failure modes.

Data remains local unless it is explicitly exported. The default transport is stdio. An optional shared HTTP daemon binds to localhost and uses local bearer-token authentication by default.

## Why server-memory

LLM agents commonly lose useful state between sessions. The usual workarounds have real costs:

- repeating repository discovery and diagnostics
- pasting large handoff summaries into every new session
- consuming context with stale or irrelevant history
- forgetting accepted decisions, constraints, and unresolved work
- mixing user preferences with project-specific facts
- depending on hosted memory services for data that should stay local

`server-memory` addresses those problems with durable, queryable memory that can be read selectively instead of replayed wholesale.

The project is intended to improve:

- **Cross-session continuity:** retain facts and decisions after the original conversation ends.
- **Context efficiency:** return compact, relevant snippets instead of the entire stored graph.
- **Task completion:** help agents continue prior work without rediscovering established state.
- **Reduced repeated work:** preserve attempted commands, known failures, file locations, and next steps.
- **Safer scope separation:** keep workspace memory distinct from optional global preference memory.
- **Local control:** use inspectable SQLite databases without requiring external service credentials.

These are design goals, not performance claims. Verified results will be published only after controlled benchmark runs are complete.

## How it works

### Memory model

The server stores:

- **Entities:** projects, files, modules, services, people, configurations, incidents, and other named objects.
- **Observations:** durable facts, decisions, preferences, paths, dependencies, code snippets, and configuration details.
- **Relations:** typed links between entities.
- **Tags:** project scopes, pinned items, preferences, and workflow labels.
- **Activity:** decisions, changes, fixes, and other events worth carrying into later sessions.

### Retrieval path

Routine recall uses `memory_context`:

1. Scope the lookup to the active workspace and optional global preference database.
2. Collect candidates through FTS5, optional embeddings, activity links, and fallback matching.
3. Rank candidates using exact-name, lexical, semantic, importance, confidence, pinned, activity, access-recency, and staleness signals.
4. Suppress duplicate or low-value matches.
5. Return bounded snippets plus conflict and stale-state indicators.

The goal is to return the smallest useful memory slice for the current task, not to place the entire database in the model context.

### At a glance

| Capability | Implementation |
| :--- | :--- |
| **Storage** | SQLite with WAL mode and FTS5 search |
| **Memory model** | Entities, observations, relations, tags, and activity |
| **MCP interface** | 20 tools; no MCP resources or prompts |
| **Routine recall** | Compact `memory_context` output |
| **Broader recall** | `memory_context_full`, graph reads, node search, and timeline queries |
| **Retrieval** | FTS5, ranking signals, fuzzy fallback, and optional embeddings |
| **Scopes** | Workspace memory and optional global preference memory |
| **Default transport** | stdio |
| **Shared mode** | Localhost HTTP daemon with a stdio proxy |
| **Data paths** | Platform-native user data and runtime directories through `platformdirs` |
| **License** | MIT |

### Design principles

- **Local-first:** core operation requires no hosted database or external service credential.
- **Selective recall:** query relevant memory rather than replaying all stored history.
- **Bounded context:** token budgets and compact formatting limit retrieval output.
- **Explicit durability:** agents choose what to store through MCP tools.
- **Inspectable state:** memory remains readable, exportable, and testable.
- **Scope safety:** destructive operations reject `scope="all"`.
- **Graceful degradation:** lexical retrieval remains available without embeddings.

## Architecture

### Default stdio mode

```text
┌────────────┐         stdio         ┌─────────────────────┐
│ MCP client │ ────────────────────> │    server-memory    │
└────────────┘                       │    FastMCP server   │
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

Use shared mode when multiple local clients should access one database process rather than opening the same database independently.

## Requirements

- Python 3.10 or newer
- SQLite with FTS5 enabled
- macOS, Ubuntu, or Windows for the tested GitHub Actions support matrix

The CI workflow tests Python 3.10 through 3.14 on Ubuntu and selected Python versions on Ubuntu, Windows, and macOS.

## Installation

### Core installation

```bash
python -m pip install "server-memory @ git+https://github.com/MK-986123/server-memory.git"
```

### Installation with embeddings

```bash
python -m pip install "server-memory[embeddings] @ git+https://github.com/MK-986123/server-memory.git"
```

Embeddings are optional. Core storage and FTS5 retrieval work without them.

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

Use the equivalent environment-variable syntax for your shell on Windows.

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

## Recommended agent behavior

Use memory only when prior state may materially improve the task.

- Call `memory_context(hint="current topic", limit=3-5)` when earlier decisions, project facts, preferences, or unresolved work may matter.
- Skip memory lookup for one-off answers or tasks already fully grounded in the current context.
- Store durable facts and decisions, not routine conversation.
- Use `log_activity` after meaningful changes, fixes, or decisions.
- Tag only facts that must remain prominent as `pinned`.
- Use explicit workspace or global scope for destructive operations.

## Tool reference

`server-memory` registers MCP tools only. It does not register resources or prompts.

### Scope behavior

| Scope | Behavior |
| :--- | :--- |
| `workspace` | Operates on the current workspace database and is the default for project memory |
| `global` | Operates on the global preference database |
| `all` | Combines supported workspace and global results with source labels |

Preference-tagged writes can automatically route to the global database when global preference routing is enabled.

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
| `log_activity` | Record a durable development or session event | `action`, `summary`, `entity_names`, `tags`, `metadata`, `scope` |
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
| `delete_relations` | Delete relations | `relations`, `scope` |

</details>

Write tools modify the selected SQLite database. `backup_memory` writes a database backup. `export_graph` may expose sensitive memory content, so review exports before sharing them.

## Configuration

Configuration is environment-driven. Empty path overrides in `.env.example` use platform defaults.

<details>
<summary><strong>View environment variables</strong></summary>

### Storage and scope

| Variable | Default | Meaning |
| :--- | :--- | :--- |
| `MEMORY_DB_PATH` | Platform user-data directory, workspace-namespaced when detected | Workspace SQLite database |
| `MEMORY_PROJECT` | Empty | Default project scope |
| `MEMORY_GLOBAL_DB_ENABLED` | `true` | Enable the global preference database |
| `MEMORY_GLOBAL_DB_PATH` | Platform user-data directory | Global preference database |
| `MEMORY_GLOBAL_PREFERENCE_ROUTING_ENABLED` | `true` | Route preference-tagged writes to global memory |
| `MEMORY_WORKSPACE_ROOT` | Unset | Explicit workspace root for default database placement |
| `MEMORY_WORKSPACE_ID` | Unset | Explicit workspace identifier for default database placement |

### Retrieval and compression

| Variable | Default | Meaning |
| :--- | :--- | :--- |
| `MEMORY_COMPRESSION_LEVEL` | `4` | Compression level from `0` through `4`; `4` is automatic |
| `MEMORY_TOKEN_BUDGET` | `2000` | Maximum approximate token budget for compressed graph output |
| `MEMORY_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Optional embedding model |
| `MEMORY_EMBEDDING_ENABLED` | `true` | Enable embedding search and backfill when dependencies are available |
| `MEMORY_WRITE_EMBEDDING_BUDGET_MS` | `10000` | Write-path embedding time budget |
| `MEMORY_DEDUP_THRESHOLD` | `0.92` | Semantic deduplication threshold |

### Runtime and shared daemon

| Variable | Default | Meaning |
| :--- | :--- | :--- |
| `MEMORY_IMPORT_JSONL` | Unset | Import JSONL on startup |
| `MEMORY_SESSION_ID` | Unset | Session identifier for activity logging |
| `MEMORY_HTTP_AUTH_ENABLED` | `true` | Require bearer authentication for the shared HTTP daemon |
| `MEMORY_AUTH_TOKEN_PATH` | Platform runtime directory | Local HTTP daemon token file |

</details>

## Evaluation

The repository includes deterministic retrieval scenarios for `memory_context`, including exact-name lookup, importance ranking, pinned facts, access recency, activity links, file-path hints, stale-fact demotion, lexical fallback, and duplicate suppression.

Those tests validate expected ranking behavior, but they do not establish real-world improvements in agent completion rate or token use.

A separate controlled protocol is provided in [`docs/BENCHMARK_PROTOCOL.md`](docs/BENCHMARK_PROTOCOL.md). It compares:

1. fresh sessions with no memory
2. fresh sessions with a token-matched manual handoff summary
3. fresh sessions using `server-memory`

The protocol measures:

- task completion rate
- durable-fact recall and contradiction rate
- total tokens and tokens to first correct action
- repeated work
- tool-call efficiency
- hit@1, hit@3, and reciprocal rank
- memory latency and end-to-end duration
- stale-memory, leakage, duplicate, and incorrect-write failures

> [!NOTE]
> No performance numbers are claimed in this README yet. Verified results should include raw run records, exact model revisions, repository commits, configurations, task fixtures, evaluator rubrics, acceptance-test logs, and confidence intervals.

## Development

Install development dependencies:

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

Verify installed entry points:

```bash
python scripts/smoke_stdio.py server-memory
server-memory-serve --help
server-memory-proxy --help
```

The stdio smoke test sends an MCP `initialize` request to the installed entry point and fails if stdout contains non-protocol output.

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidance.

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

## CI and supply chain

GitHub Actions are configured to run:

- syntax validation
- Ruff linting
- pytest across the supported Python and operating-system matrix
- wheel and source-distribution builds
- wheel-content inspection
- clean installed-package checks outside the repository checkout
- MCP stdio and installed-command smoke tests
- `pip-audit`
- CodeQL
- Dependency Review

Dependabot is configured for Python dependencies and GitHub Actions.

The workflow uses GitHub-hosted `ubuntu-latest`, `windows-latest`, and `macos-latest` labels. These labels refer to GitHub's latest stable runner images and can temporarily lag the newest vendor operating-system release during image migrations.

## Troubleshooting

| Symptom | Check |
| :--- | :--- |
| `no such module: fts5` | Use a Python build linked against SQLite with FTS5 enabled. |
| MCP client hangs at startup | Run `python scripts/smoke_stdio.py server-memory` and inspect stderr. |
| Multiple clients lock the database | Run one `server-memory-serve` process and connect clients through `server-memory-proxy`. |
| Proxy returns an authentication failure | Restart the daemon and client so both read the same `MEMORY_AUTH_TOKEN_PATH`. |
| Memory is stored in an unexpected location | Set `MEMORY_DB_PATH`, `MEMORY_WORKSPACE_ROOT`, or `MEMORY_WORKSPACE_ID` explicitly. |

## License

Licensed under the [MIT License](LICENSE).
