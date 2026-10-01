"""FastMCP server with all 22 memory tools."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from typing_extensions import NotRequired, TypedDict

from .compression import _enforce_budget, compress_graph
from .config import MemoryConfig
from .db import Database
from .embeddings import EmbeddingEngine
from .graph import KnowledgeGraphManager
from .local_auth import LocalTokenVerifier, build_local_auth_settings, ensure_local_auth_token
from .models import KnowledgeGraph

logger = logging.getLogger(__name__)

_startup_cleanup_lock = threading.Lock()
_startup_cleanup_started = False

PREFERENCE_TAG = "preference"
WORKSPACE_MEMORY_SCORE_BONUS = 0.2

READ_ONLY_TOOL = ToolAnnotations(
    readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
)
WRITE_TOOL = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False
)
DESTRUCTIVE_TOOL = ToolAnnotations(
    readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False
)

Scope = Literal["workspace", "global", "all"]
NonEmptyText = Annotated[str, Field(min_length=1, max_length=16_384)]
Name = Annotated[str, Field(min_length=1, max_length=512)]

DESTRUCTIVE_SCOPE_ALL_ERROR = (
    "scope='all' is not supported for destructive operations to prevent unintentional "
    "data loss. Please specify 'workspace' or 'global'."
)
DESTRUCTIVE_TAG_SCOPE_ALL_ERROR = (
    "scope='all' is not supported for destructive tag action '{action}' to prevent "
    "unintentional data loss. Please specify 'workspace' or 'global'."
)
# manage_tags actions that must not fan out across scopes
DESTRUCTIVE_TAG_ACTIONS = frozenset({"delete", "untag", "cleanup"})


def _destructive_scope_all_error(*, action: str | None = None) -> str:
    """Return the public-main structured error for rejected scope='all' writes."""
    if action is not None:
        return json.dumps({"error": DESTRUCTIVE_TAG_SCOPE_ALL_ERROR.format(action=action)})
    return json.dumps({"error": DESTRUCTIVE_SCOPE_ALL_ERROR})


class StructuredToolResult(BaseModel):
    """Protocol result with exact legacy text plus machine-readable JSON when available."""

    model_config = ConfigDict(extra="forbid")

    text: str
    data: JsonValue | None = None

    @model_validator(mode="before")
    @classmethod
    def from_legacy_text(cls, value: Any) -> Any:
        # Tool functions intentionally keep returning their established strings.
        # FastMCP applies this adapter only at the protocol boundary, preserving
        # direct Python callers while advertising and emitting structuredContent.
        if not isinstance(value, str):
            return value
        try:
            data: JsonValue | None = json.loads(value)
        except json.JSONDecodeError:
            data = None
        return {"text": value, "data": data}


class EntityInput(TypedDict):
    name: Name
    entityType: str
    observations: NotRequired[Annotated[list[NonEmptyText], Field(max_length=500)]]
    tags: NotRequired[Annotated[list[Name], Field(max_length=100)]]
    metadata: NotRequired[dict[str, Any]]


EntityInput.__pydantic_config__ = ConfigDict(extra="forbid")


RelationInput = TypedDict(
    "RelationInput",
    {
        "from": Name,
        "to": Name,
        "relationType": Name,
        "weight": NotRequired[Annotated[float, Field(ge=0.0, le=1.0)]],
        "tags": NotRequired[Annotated[list[Name], Field(max_length=100)]],
    },
)
RelationInput.__pydantic_config__ = ConfigDict(extra="forbid")


class ObservationInput(TypedDict):
    entityName: Name
    contents: Annotated[list[NonEmptyText], Field(min_length=1, max_length=500)]
    source: NotRequired[Annotated[str, Field(max_length=2_048)]]
    confidence: NotRequired[Annotated[float, Field(ge=0.0, le=1.0)]]
    importance: NotRequired[Annotated[float, Field(ge=0.0, le=1.0)]]
    obs_type: NotRequired[Annotated[str, Field(max_length=128)]]
    tags: NotRequired[Annotated[list[Name], Field(max_length=100)]]


ObservationInput.__pydantic_config__ = ConfigDict(extra="forbid")


class ObservationDeletionInput(TypedDict):
    entityName: Name
    observations: Annotated[list[NonEmptyText], Field(min_length=1, max_length=500)]


ObservationDeletionInput.__pydantic_config__ = ConfigDict(extra="forbid")


RelationDeletionInput = TypedDict(
    "RelationDeletionInput",
    {"from": Name, "to": Name, "relationType": Name},
)
RelationDeletionInput.__pydantic_config__ = ConfigDict(extra="forbid")

INSTRUCTIONS = (
    "Memory server for persistent knowledge across conversations.\n"
    "Use memory_context(hint='current topic', limit=3-5) when prior sessions, stable project facts, or cross-session continuity may matter.\n"
    "Skip memory lookup for one-off answers, purely local inspection, or tasks already fully grounded in the current context.\n"
    "After durable changes or decisions, call log_activity(action='...', summary='...'); do not log routine chatter.\n"
    "Use search_nodes for targeted full-text search, create_entities/add_observations to store durable knowledge.\n"
    "Tag important items with 'pinned' only when they must remain visible in future scoped recall."
)


def _record_benchmark_event(event: str) -> None:
    """Record opt-in local treatment evidence without affecting normal service."""
    telemetry_path = os.environ.get("MEMORY_BENCHMARK_TELEMETRY_PATH", "")
    if not telemetry_path:
        return
    path = Path(telemetry_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"event": event, "timestamp": datetime.now(timezone.utc).isoformat()}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload) + "\n")


def _page_cursor(offset: int, fingerprint: str) -> str:
    payload = json.dumps({"offset": offset, "fingerprint": fingerprint}, separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")


def _cursor_offset(cursor: str, fingerprint: str) -> int:
    if not cursor:
        return 0
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
        offset = int(payload["offset"])
        if payload["fingerprint"] != fingerprint or offset < 0:
            raise ValueError
        return offset
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("invalid or mismatched pagination cursor") from exc


def _fingerprint_cursor(tool: str, arguments: dict[str, Any]) -> str:
    encoded = json.dumps([tool, arguments], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _resolve_backup_path(
    db_path: str | Path,
    dest_path: str,
    *,
    scope_name: str,
    multiple_scopes: bool,
    timestamp: str,
) -> Path:
    """Resolve a no-clobber backup path beneath a real, non-symlink backup root."""
    db_parent = Path(db_path).expanduser().resolve().parent
    configured_root = db_parent / "backups"
    if configured_root.is_symlink():
        raise ValueError("backup root must not be a symlink")
    allowed_root = configured_root.resolve(strict=False)

    if dest_path:
        requested = Path(dest_path).expanduser()
        candidate = (
            requested / f"{scope_name}-{timestamp}.db"
            if multiple_scopes or requested.suffix == ""
            else requested
        )
    else:
        candidate = allowed_root / f"{scope_name}-{timestamp}.db"
    backup_path = candidate.resolve(strict=False)

    if allowed_root != backup_path.parent and allowed_root not in backup_path.parents:
        raise ValueError(f"backup destination must be under {allowed_root}")
    if backup_path.exists() or backup_path.is_symlink():
        raise ValueError(f"backup destination already exists: {backup_path}")
    return backup_path


def _paginate_graph(
    graph: KnowledgeGraph,
    *,
    page_size: int,
    cursor: str,
    fingerprint: str,
) -> tuple[KnowledgeGraph, str]:
    """Return a deterministic page and an opaque cursor bound to its query."""
    offset = _cursor_offset(cursor, fingerprint)
    entities = graph.entities[offset : offset + page_size]
    names = {(entity.scope, entity.name) for entity in entities}
    relations = [
        relation
        for relation in graph.relations
        if (relation.scope, relation.from_name) in names
        and (relation.scope, relation.to_name) in names
    ]
    next_offset = offset + len(entities)
    next_cursor = (
        _page_cursor(next_offset, fingerprint) if next_offset < len(graph.entities) else ""
    )
    return KnowledgeGraph(entities=entities, relations=relations), next_cursor


def _run_startup_cleanup(cfg: MemoryConfig) -> None:
    """Expire only explicitly ephemeral records outside the MCP handshake path."""
    if not cfg.retention_cleanup_enabled:
        return
    cleanup_db = Database(cfg.db_path)
    try:
        cfg.ensure_db_dir()
        cleanup_db.open()

        cleaned = cleanup_db.cleanup_expired()
        if cleaned:
            logger.info("Cleaned up %d expired items", cleaned)

    except Exception as e:
        logger.warning("Background startup cleanup skipped: %s", e)
    finally:
        cleanup_db.close()


def _schedule_startup_cleanup(cfg: MemoryConfig) -> None:
    """Schedule startup cleanup once per process."""
    global _startup_cleanup_started

    with _startup_cleanup_lock:
        if _startup_cleanup_started:
            return
        _startup_cleanup_started = True

    cleanup_thread = threading.Thread(
        target=_run_startup_cleanup,
        args=(cfg,),
        name="server-memory-startup-cleanup",
        daemon=True,
    )
    cleanup_thread.start()


def _format_memory_context_result(result: dict[str, Any]) -> str:
    """Render compact memory_context output from graph-layer match metadata."""
    lines = []
    if result["pinned"]:
        pinned_str = ", ".join(
            f"{p['name']}[{p['type']}]"
            + (f"@{p['source']}" if p.get("source") else "")
            for p in result["pinned"]
        )
        lines.append(f"Pinned: {pinned_str}")
    if result["recent_activity"]:
        acts = [f"{a['action']}:{a['summary']}" for a in result["recent_activity"]]
        lines.append(f"Recent: {' | '.join(acts)}")
    if result["hint_matches"]:
        hint_parts = []
        for h in result["hint_matches"]:
            part = f"{h['name']}[{h['type']}]"
            if h.get("source"):
                part += f"@{h['source']}"
            snippets = h.get("snippets", [])
            if snippets:
                part += ': "' + '" | "'.join(snippets) + '"'
            if h.get("conflict"):
                part += " !conflict"
            if h.get("stale"):
                part += " !stale"
            hint_parts.append(part)
        lines.append(f"Relevant: {', '.join(hint_parts)}")
    s = result["stats"]
    lines.append(f"Graph: {s['entities']}E {s['observations']}O {s['relations']}R")

    return (
        "\n".join(lines)
        if lines
        else "Memory empty. Use create_entities to start building knowledge."
    )


def _is_preference_entity(entity: dict[str, Any]) -> bool:
    """Return True when an entity payload should be stored in the global DB."""
    tags = {str(tag).lower() for tag in entity.get("tags", [])}
    entity_type = str(entity.get("entityType", entity.get("entity_type", ""))).lower()
    return PREFERENCE_TAG in tags or entity_type == PREFERENCE_TAG


def _is_preference_observation(observation: dict[str, Any]) -> bool:
    """Return True when an observation payload should be stored in the global DB."""
    tags = {str(tag).lower() for tag in observation.get("tags", [])}
    obs_type = str(observation.get("obs_type", "")).lower()
    return PREFERENCE_TAG in tags or obs_type == PREFERENCE_TAG


def _merge_activity_entries(
    workspace_activity: list[dict[str, Any]],
    global_activity: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Merge activity rows while preserving order and removing duplicates."""
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    ordered = sorted(
        [
            *(dict(entry, source="workspace") for entry in workspace_activity),
            *(dict(entry, source="global") for entry in global_activity),
        ],
        key=lambda entry: str(entry.get("at", entry.get("created_at", ""))),
        reverse=True,
    )
    for entry in ordered:
        key = (entry.get("source", ""), entry.get("action", ""), entry.get("summary", ""))
        if key in seen:
            continue
        seen.add(key)
        merged.append(entry)
        if len(merged) >= limit:
            break

    return merged


def _merge_hint_matches(
    workspace_matches: list[dict[str, Any]],
    global_matches: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Merge hint matches with a slight bias toward workspace-local memory."""
    merged_by_identity: dict[tuple[str, str, str], dict[str, Any]] = {}

    for source, matches in (("workspace", workspace_matches), ("global", global_matches)):
        for match in matches:
            candidate = dict(match)
            candidate["source"] = source
            candidate["score"] = float(candidate.get("score", 0.0))
            if source == "workspace":
                candidate["score"] += WORKSPACE_MEMORY_SCORE_BONUS

            key = (source, candidate.get("name", ""), candidate.get("type", ""))
            current = merged_by_identity.get(key)
            if current is None or candidate["score"] > float(current.get("score", 0.0)):
                merged_by_identity[key] = candidate

    ranked = sorted(
        merged_by_identity.values(),
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )
    return ranked[:limit]


def merge_memory_context_results(
    workspace_result: dict[str, Any],
    global_result: dict[str, Any] | None,
    *,
    limit: int,
) -> dict[str, Any]:
    """Merge workspace-local and global preference context into one response."""
    if not global_result:
        return workspace_result

    pinned_by_identity: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in workspace_result.get("pinned", []):
        candidate = dict(item, source="workspace")
        pinned_by_identity[("workspace", item.get("name", ""), item.get("type", ""))] = candidate
    for item in global_result.get("pinned", []):
        candidate = dict(item, source="global")
        pinned_by_identity[("global", item.get("name", ""), item.get("type", ""))] = candidate

    workspace_stats = workspace_result.get("stats", {})
    global_stats = global_result.get("stats", {})

    return {
        "pinned": list(pinned_by_identity.values())[:limit],
        "recent_activity": _merge_activity_entries(
            workspace_result.get("recent_activity", []),
            global_result.get("recent_activity", []),
            limit=5,
        ),
        "hint_matches": _merge_hint_matches(
            workspace_result.get("hint_matches", []),
            global_result.get("hint_matches", []),
            limit=limit,
        ),
        "stats": {
            "entities": int(workspace_stats.get("entities", 0)) + int(global_stats.get("entities", 0)),
            "observations": int(workspace_stats.get("observations", 0))
            + int(global_stats.get("observations", 0)),
            "relations": int(workspace_stats.get("relations", 0)) + int(global_stats.get("relations", 0)),
            "workspace_entities": int(workspace_stats.get("entities", 0)),
            "global_entities": int(global_stats.get("entities", 0)),
        },
    }



def _safe_close_db(database: Database | None, has_active_exc: bool) -> None:
    if database is None:
        return
    try:
        database.close()
    except Exception as exc:
        if has_active_exc:
            logger.error(
                "Suppressing database close error because another exception is active: %s",
                exc,
            )
        else:
            raise


def _close_lifespan_databases(db: Database | None, global_db: Database | None) -> None:
    """Close lifespan databases safely, ensuring both are closed."""
    import sys

    has_active_exc = sys.exc_info()[1] is not None
    try:
        if global_db is not None:
            _safe_close_db(global_db, has_active_exc)
    finally:
        has_active_exc_now = has_active_exc or (sys.exc_info()[1] is not None)
        if db is not None:
            _safe_close_db(db, has_active_exc_now)


def create_server(
    config: MemoryConfig | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    enable_http_auth: bool = False,
) -> FastMCP:
    cfg = config or MemoryConfig()

    auth_settings = None
    token_verifier = None
    if enable_http_auth:
        cfg.ensure_auth_token_dir()
        token = ensure_local_auth_token(cfg.auth_token_path)
        auth_settings = build_local_auth_settings(host=host, port=port)
        token_verifier = LocalTokenVerifier(token)

    @asynccontextmanager
    async def lifespan(server: FastMCP):
        cfg.ensure_db_dir()
        db = Database(cfg.db_path, write_timeout_seconds=cfg.write_timeout_ms / 1000)
        db.open()

        global_db: Database | None = None
        global_graph_mgr: KnowledgeGraphManager | None = None
        use_global_db = cfg.global_db_enabled and cfg.global_db_path != cfg.db_path
        if use_global_db:
            cfg.ensure_global_db_dir()
            global_db = Database(
                cfg.global_db_path,
                write_timeout_seconds=cfg.write_timeout_ms / 1000,
            )
            global_db.open()

        # Create embedding engine if enabled
        embedding_engine: EmbeddingEngine | None = None
        if cfg.embedding_enabled:
            embedding_engine = EmbeddingEngine(model_name=cfg.embedding_model)

        graph_mgr = KnowledgeGraphManager(
            db,
            session_id=cfg.session_id or "",
            embedding_engine=embedding_engine,
            project=cfg.project,
            dedup_threshold=cfg.dedup_threshold,
            write_embedding_budget_ms=cfg.write_embedding_budget_ms,
        )

        if global_db is not None:
            global_graph_mgr = KnowledgeGraphManager(
                global_db,
                session_id=cfg.session_id or "",
                embedding_engine=embedding_engine,
                project="",
                dedup_threshold=cfg.dedup_threshold,
                write_embedding_budget_ms=cfg.write_embedding_budget_ms,
            )

        # Import from old server if requested
        if cfg.import_jsonl:
            try:
                with open(cfg.import_jsonl) as f:
                    data = f.read()
                counts = graph_mgr.import_graph(data)
                logger.info("Imported from JSONL: %s", counts)
            except Exception as e:
                logger.error("Failed to import JSONL: %s", e)

        _schedule_startup_cleanup(cfg)

        _record_benchmark_event("mcp_handshake")
        try:
            yield {
                "db": db,
                "graph": graph_mgr,
                "global_db": global_db,
                "global_graph": global_graph_mgr,
                "config": cfg,
            }
        finally:
            _close_lifespan_databases(db, global_db)

    mcp = FastMCP(
        "server-memory",
        instructions=INSTRUCTIONS,
        lifespan=lifespan,
        host=host,
        port=port,
        auth=auth_settings,
        token_verifier=token_verifier,
    )

    def _get_ctx(
        ctx: Context,
    ) -> tuple[KnowledgeGraphManager, MemoryConfig, KnowledgeGraphManager | None]:
        _record_benchmark_event("tool_call")
        lifespan_ctx = ctx.request_context.lifespan_context
        return lifespan_ctx["graph"], lifespan_ctx["config"], lifespan_ctx.get("global_graph")

    def _scope_graphs(
        workspace_graph: KnowledgeGraphManager,
        global_graph: KnowledgeGraphManager | None,
        scope: str,
    ) -> list[tuple[str, KnowledgeGraphManager]]:
        if scope not in {"workspace", "global", "all"}:
            raise ValueError("scope must be 'workspace', 'global', or 'all'")
        if scope == "workspace":
            return [("workspace", workspace_graph)]
        if global_graph is None:
            if scope == "global":
                raise ValueError("global memory scope is disabled")
            return [("workspace", workspace_graph)]
        if scope == "global":
            return [("global", global_graph)]
        return [("workspace", workspace_graph), ("global", global_graph)]

    def _entity_owner(
        workspace_graph: KnowledgeGraphManager,
        global_graph: KnowledgeGraphManager | None,
        name: str,
        *,
        scope: str = "all",
        include_deleted: bool = False,
    ) -> tuple[str, KnowledgeGraphManager] | None:
        matches: list[tuple[str, KnowledgeGraphManager]] = []
        for scope_name, candidate in _scope_graphs(workspace_graph, global_graph, scope):
            if include_deleted:
                row = candidate.db.cx.execute(
                    "SELECT 1 FROM entities WHERE name = ?", (name,)
                ).fetchone()
                if row:
                    matches.append((scope_name, candidate))
            elif candidate.get_entity_by_name(name) is not None:
                matches.append((scope_name, candidate))
        if len(matches) > 1:
            raise ValueError(
                f"Entity '{name}' exists in multiple scopes; specify scope='workspace' or 'global'"
            )
        return matches[0] if matches else None

    def _merge_graphs(graphs: list[KnowledgeGraph]) -> KnowledgeGraph:
        return KnowledgeGraph(
            entities=[entity for graph in graphs for entity in graph.entities],
            relations=[relation for graph in graphs for relation in graph.relations],
        )

    def _merge_scoped_graphs(
        graphs: list[tuple[str, KnowledgeGraph]],
    ) -> KnowledgeGraph:
        for scope_name, scoped_graph in graphs:
            for entity in scoped_graph.entities:
                entity.scope = scope_name
            for relation in scoped_graph.relations:
                relation.scope = scope_name
        return _merge_graphs([graph for _, graph in graphs])

    # ------------------------------------------------------------------
    # Tool 1: create_entities
    # ------------------------------------------------------------------
    @mcp.tool(annotations=WRITE_TOOL)
    def create_entities(
        ctx: Context,
        entities: Annotated[list[EntityInput], Field(min_length=1, max_length=500)],
        scope: Scope = "all",
    ) -> StructuredToolResult:
        """Create new entities in the knowledge graph.

        Each entity needs: name (str), entityType (str).
        Optional: observations (list[str]), tags (list[str]), metadata (dict).
        Skips duplicates by name.
        """
        graph, cfg, global_graph = _get_ctx(ctx)
        _scope_graphs(graph, global_graph, scope)
        workspace_entities = entities
        global_entities: list[dict[str, Any]] = []
        if scope == "global":
            workspace_entities = []
            global_entities = entities
        elif scope == "workspace":
            workspace_entities = entities
        elif cfg.global_preference_routing_enabled and global_graph is not None:
            workspace_entities = []
            for entity in entities:
                if _is_preference_entity(entity):
                    global_entities.append(entity)
                else:
                    workspace_entities.append(entity)

        created = []
        if workspace_entities:
            created.extend(graph.create_entities(workspace_entities))
        if global_entities and global_graph is not None:
            created.extend(global_graph.create_entities(global_entities))
        return json.dumps([e.to_dict() for e in created], indent=2)

    # ------------------------------------------------------------------
    # Tool 2: create_relations
    # ------------------------------------------------------------------
    @mcp.tool(annotations=WRITE_TOOL)
    def create_relations(
        ctx: Context,
        relations: Annotated[list[RelationInput], Field(min_length=1, max_length=500)],
        scope: Scope = "workspace",
    ) -> StructuredToolResult:
        """Create relations between entities.

        Each relation needs: from (str), to (str), relationType (str).
        Optional: weight (float), tags (list[str]).
        Both entities must exist. Use active voice for relationType.
        """
        graph, _, global_graph = _get_ctx(ctx)
        try:
            targets = _scope_graphs(graph, global_graph, scope)
            if scope == "all":
                grouped: dict[str, list[dict[str, Any]]] = {"workspace": [], "global": []}
                for relation in relations:
                    from_owner = _entity_owner(graph, global_graph, relation["from"], scope=scope)
                    to_owner = _entity_owner(graph, global_graph, relation["to"], scope=scope)
                    if from_owner is None or to_owner is None:
                        raise ValueError("relation endpoint not found")
                    if from_owner[0] != to_owner[0]:
                        raise ValueError("cross-scope relations are not supported")
                    grouped[from_owner[0]].append(relation)
                created = []
                for scope_name, target in targets:
                    if grouped[scope_name]:
                        created.extend(target.create_relations(grouped[scope_name]))
            else:
                created = targets[0][1].create_relations(relations)
            return json.dumps([r.to_dict() for r in created], indent=2)
        except ValueError as e:
            raise ValueError(str(e)) from e

    # ------------------------------------------------------------------
    # Tool 3: add_observations
    # ------------------------------------------------------------------
    @mcp.tool(annotations=WRITE_TOOL)
    def add_observations(
        ctx: Context,
        observations: Annotated[list[ObservationInput], Field(min_length=1, max_length=500)],
        scope: Scope = "all",
    ) -> StructuredToolResult:
        """Add observations to existing entities.

        Each item needs: entityName (str), contents (list[str]).
        Optional: source (str), confidence (float 0-1), tags (list[str]),
                  importance (float 0-1, default 0.5 — higher survives compression),
                  obs_type (str: fact, decision, preference, api_endpoint,
                           dependency, file_path, code_snippet, config, schema).
        Deduplicates by exact match and semantic similarity.
        Protected obs_types (api_endpoint, dependency, file_path, code_snippet,
        config, schema) always survive compression.
        """
        graph, _, global_graph = _get_ctx(ctx)
        try:
            workspace_observations: list[dict[str, Any]] = []
            global_observations: list[dict[str, Any]] = []
            for observation in observations:
                name = str(observation.get("entityName", ""))
                owner = _entity_owner(graph, global_graph, name, scope=scope)
                if owner is None:
                    raise ValueError(f"Entity not found: {name}")
                if owner[0] == "global":
                    global_observations.append(observation)
                else:
                    workspace_observations.append(observation)

            results = []
            if workspace_observations:
                results.extend(graph.add_observations(workspace_observations))
            if global_observations and global_graph is not None:
                results.extend(global_graph.add_observations(global_observations))
            return json.dumps(results, indent=2)
        except ValueError as e:
            raise ValueError(str(e)) from e

    # ------------------------------------------------------------------
    # Tool 4: delete_entities
    # ------------------------------------------------------------------
    @mcp.tool(annotations=DESTRUCTIVE_TOOL)
    def delete_entities(
        ctx: Context,
        entityNames: Annotated[list[Name], Field(min_length=1, max_length=500)],
        hard: bool = False,
        scope: Scope = "workspace",
    ) -> StructuredToolResult:
        """Delete entities (soft delete by default, cascades to relations).

        Set hard=true for permanent deletion.
        """
        if scope == "all":
            return _destructive_scope_all_error()
        graph, _, global_graph = _get_ctx(ctx)
        targets = _scope_graphs(graph, global_graph, scope)
        count = sum(target.delete_entities(entityNames, hard=hard) for _, target in targets)
        return json.dumps({"deleted": count})

    @mcp.tool(annotations=READ_ONLY_TOOL)
    def list_deleted_entities(
        ctx: Context,
        limit: Annotated[int, Field(ge=1, le=1000)] = 100,
        scope: Scope = "workspace",
    ) -> StructuredToolResult:
        """List soft-deleted workspace entities that can be restored.

        Results are ordered newest deletion first. Limit must be 1-1000.
        """
        graph, _, global_graph = _get_ctx(ctx)
        scoped_entities = [
            (scope_name, entity)
            for scope_name, target in _scope_graphs(graph, global_graph, scope)
            for entity in target.list_deleted_entities(limit=limit)
        ]
        scoped_entities.sort(key=lambda item: (item[1].deleted_at or "", item[1].id), reverse=True)
        return json.dumps(
            [
                {
                    **entity.to_dict(),
                    "scope": scope_name,
                    "deletedAt": entity.deleted_at,
                    "updatedAt": entity.updated_at,
                }
                for scope_name, entity in scoped_entities[:limit]
            ],
            indent=2,
        )

    @mcp.tool(annotations=WRITE_TOOL)
    def restore_entities(
        ctx: Context,
        entityNames: Annotated[list[Name], Field(min_length=1, max_length=500)],
        scope: Scope = "workspace",
    ) -> StructuredToolResult:
        """Restore soft-deleted entities and safe associated relations.

        Scope selects which database(s) to restore from: `workspace`, `global`,
        or `all` (both, when global memory is enabled).
        """
        graph, _, global_graph = _get_ctx(ctx)
        count = sum(
            target.restore_entities(entityNames)
            for _, target in _scope_graphs(graph, global_graph, scope)
        )
        return json.dumps({"restored": count})

    # ------------------------------------------------------------------
    # Tool 5: delete_observations
    # ------------------------------------------------------------------
    @mcp.tool(annotations=DESTRUCTIVE_TOOL)
    def delete_observations(
        ctx: Context,
        deletions: Annotated[list[ObservationDeletionInput], Field(min_length=1, max_length=500)],
        scope: Scope = "workspace",
    ) -> StructuredToolResult:
        """Delete specific observations from entities.

        Each item needs: entityName (str), observations (list[str] of content to delete).
        """
        if scope == "all":
            return _destructive_scope_all_error()
        graph, _, global_graph = _get_ctx(ctx)
        targets = _scope_graphs(graph, global_graph, scope)
        count = sum(target.delete_observations(deletions) for _, target in targets)
        return json.dumps({"deleted": count})

    # ------------------------------------------------------------------
    # Tool 6: delete_relations
    # ------------------------------------------------------------------
    @mcp.tool(annotations=DESTRUCTIVE_TOOL)
    def delete_relations(
        ctx: Context,
        relations: Annotated[list[RelationDeletionInput], Field(min_length=1, max_length=500)],
        scope: Scope = "workspace",
    ) -> StructuredToolResult:
        """Delete relations from the knowledge graph.

        Each item needs: from (str), to (str), relationType (str).
        """
        if scope == "all":
            return _destructive_scope_all_error()
        graph, _, global_graph = _get_ctx(ctx)
        targets = _scope_graphs(graph, global_graph, scope)
        count = sum(target.delete_relations(relations) for _, target in targets)
        return json.dumps({"deleted": count})

    # ------------------------------------------------------------------
    # Tool 7: read_graph
    # ------------------------------------------------------------------
    @mcp.tool(annotations=READ_ONLY_TOOL)
    def read_graph(
        ctx: Context,
        tags: list[str] | None = None,
        entity_types: list[str] | None = None,
        limit: Annotated[int, Field(ge=0, le=5000)] = 0,
        include_deleted: bool = False,
        compress: bool = True,
        scope: Scope = "workspace",
        page_size: Annotated[int, Field(ge=0, le=500)] = 0,
        cursor: Annotated[str, Field(max_length=1024)] = "",
    ) -> StructuredToolResult:
        """Read the knowledge graph (compressed by default).

        Filter by tags, entity_types. Set compress=false for full JSON.
        """
        graph_mgr, cfg, global_graph = _get_ctx(ctx)
        scoped = _scope_graphs(graph_mgr, global_graph, scope)
        kg = _merge_scoped_graphs(
            [
                (
                    scope_name,
                    target.read_graph(
                        tags=tags,
                        entity_types=entity_types,
                        limit=0 if page_size else limit,
                        include_deleted=include_deleted,
                    ),
                )
                for scope_name, target in scoped
            ]
        )
        if limit > 0 and not page_size:
            kg.entities = kg.entities[:limit]
        next_cursor = ""
        if page_size:
            fingerprint = _fingerprint_cursor(
                "read_graph",
                {
                    "tags": tags,
                    "entity_types": entity_types,
                    "include_deleted": include_deleted,
                    "scope": scope,
                    "page_size": page_size,
                },
            )
            kg, next_cursor = _paginate_graph(
                kg, page_size=page_size, cursor=cursor, fingerprint=fingerprint
            )
        if compress:
            pinned_ids = set().union(*(_get_pinned_ids(target) for _, target in scoped))
            rendered: Any = compress_graph(
                kg, cfg.compression_level, cfg.token_budget, pinned_ids
            )
        else:
            rendered = kg.to_dict()
        if page_size:
            return json.dumps({"result": rendered, "nextCursor": next_cursor}, indent=2)
        return rendered if isinstance(rendered, str) else json.dumps(rendered, indent=2)

    # ------------------------------------------------------------------
    # Tool 8: search_nodes
    # ------------------------------------------------------------------
    @mcp.tool(annotations=READ_ONLY_TOOL)
    def search_nodes(
        ctx: Context,
        query: NonEmptyText,
        tags: list[str] | None = None,
        entity_types: list[str] | None = None,
        time_range: list[str] | None = None,
        limit: Annotated[int, Field(ge=1, le=500)] = 20,
        compress: bool = True,
        scope: Scope = "workspace",
        page_size: Annotated[int, Field(ge=0, le=500)] = 0,
        cursor: Annotated[str, Field(max_length=1024)] = "",
    ) -> StructuredToolResult:
        """Full-text search with BM25 ranking.

        Supports prefix search, phrases ("exact match"), boolean (AND/OR/NOT).
        Filter by tags, entity_types, time_range ([start, end] ISO strings).
        """
        graph_mgr, cfg, global_graph = _get_ctx(ctx)
        tr = tuple(time_range) if time_range and len(time_range) == 2 else None
        searched: list[tuple[str, KnowledgeGraph]] = []
        diagnostics: list[str] = []
        for scope_name, target in _scope_graphs(graph_mgr, global_graph, scope):
            searched.append(
                (
                    scope_name,
                    target.search_fts(
                        query,
                        tags=tags,
                        entity_types=entity_types,
                        time_range=tr,
                        limit=0 if page_size else limit,
                    ),
                )
            )
            diagnostics.extend(
                f"{scope_name}:{diagnostic}"
                for diagnostic in getattr(target, "last_search_diagnostics", [])
            )
        kg = _merge_scoped_graphs(searched)
        kg.entities = kg.entities[:limit] if not page_size else kg.entities
        next_cursor = ""
        if page_size:
            fingerprint = _fingerprint_cursor(
                "search_nodes",
                {
                    "query": query,
                    "tags": tags,
                    "entity_types": entity_types,
                    "time_range": time_range,
                    "scope": scope,
                    "page_size": page_size,
                },
            )
            kg, next_cursor = _paginate_graph(
                kg, page_size=page_size, cursor=cursor, fingerprint=fingerprint
            )
        if compress:
            rendered = compress_graph(kg, cfg.compression_level, cfg.token_budget)
        else:
            rendered = kg.to_dict()
        if page_size:
            return json.dumps(
                {
                    "result": rendered,
                    "nextCursor": next_cursor,
                    "diagnostics": diagnostics,
                },
                indent=2,
            )
        if diagnostics:
            return json.dumps({"result": rendered, "diagnostics": diagnostics}, indent=2)
        return rendered if isinstance(rendered, str) else json.dumps(rendered, indent=2)

    # ------------------------------------------------------------------
    # Tool 9: open_nodes
    # ------------------------------------------------------------------
    @mcp.tool(annotations=READ_ONLY_TOOL)
    def open_nodes(
        ctx: Context,
        names: Annotated[list[Name], Field(min_length=1, max_length=500)],
        depth: Annotated[int, Field(ge=0, le=10)] = 0,
        scope: Scope = "workspace",
    ) -> StructuredToolResult:
        """Open specific entities by name.

        depth=0: exact entities only.
        depth=1: include direct neighbors via relations.
        depth=2+: BFS expansion.
        """
        graph_mgr, cfg, global_graph = _get_ctx(ctx)
        kg = _merge_scoped_graphs(
            [
                (scope_name, target.open_nodes(names, depth=depth))
                for scope_name, target in _scope_graphs(graph_mgr, global_graph, scope)
            ]
        )
        return json.dumps(kg.to_dict(), indent=2)

    # ------------------------------------------------------------------
    # Tool 10: memory_context (THE KEY TOOL)
    # ------------------------------------------------------------------
    @mcp.tool(annotations=READ_ONLY_TOOL)
    def memory_context(
        ctx: Context,
        hint: str = "",
        project: str = "",
        limit: Annotated[int, Field(ge=1, le=100)] = 10,
        scope: Scope = "all",
        budget: Annotated[int, Field(ge=100, le=100_000)] = 500,
    ) -> StructuredToolResult:
        """Lightweight context snapshot (~200-500 tokens) for scoped durable recall.

        Returns: pinned entities, recent activity, hint-matched entities, and graph stats.
        Call when prior sessions, stable project facts, or cross-session continuity may matter.
        Skip for one-off answers or tasks already fully grounded in the current context.
        Pass hint='current topic' to get relevant entities surfaced.
        Pass project='name' to scope results to a specific project tag.
        Pass limit to control how many hint matches are included; prefer 3-5 unless deeper recall is justified.
        """
        graph_mgr, cfg, global_graph_mgr = _get_ctx(ctx)
        targets = dict(_scope_graphs(graph_mgr, global_graph_mgr, scope))
        workspace_result = (
            targets["workspace"].memory_context(hint=hint, project=project, limit=limit)
            if "workspace" in targets
            else None
        )
        global_result = (
            targets["global"].memory_context(hint=hint, project="", limit=limit)
            if "global" in targets
            else None
        )
        if workspace_result is not None:
            result = merge_memory_context_results(workspace_result, global_result, limit=limit)
        elif global_result is not None:
            result = global_result
        else:  # pragma: no cover - _scope_graphs always returns or raises
            raise ValueError("no memory scope is available")
        return _enforce_budget(_format_memory_context_result(result), budget)

    # ------------------------------------------------------------------
    # Tool 11: log_activity
    # ------------------------------------------------------------------
    @mcp.tool(annotations=WRITE_TOOL)
    def log_activity(
        ctx: Context,
        action: str,
        summary: str = "",
        entity_names: list[str] | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StructuredToolResult:
        """Record what happened this turn. Auto-creates missing entities.

        Common actions: file_changed, decision_made, bug_fixed, feature_added,
        refactored, investigated, discussed, preference_set.
        """
        graph_mgr, cfg, global_graph_mgr = _get_ctx(ctx)
        target_graph = graph_mgr
        if cfg.global_preference_routing_enabled and global_graph_mgr is not None and action == "preference_set":
            target_graph = global_graph_mgr
        entry = target_graph.log_activity(
            action=action,
            summary=summary,
            entity_names=entity_names,
            tags=tags,
            metadata=metadata,
        )
        return json.dumps({"id": entry.id, "action": entry.action, "summary": entry.summary})

    # ------------------------------------------------------------------
    # Tool 12: query_timeline
    # ------------------------------------------------------------------
    @mcp.tool(annotations=READ_ONLY_TOOL)
    def query_timeline(
        ctx: Context,
        time_range: str | None = None,
        start: str | None = None,
        end: str | None = None,
        actions: list[str] | None = None,
        entity_name: str | None = None,
        session_id: str | None = None,
        limit: Annotated[int, Field(ge=1, le=1000)] = 50,
        scope: Scope = "workspace",
    ) -> StructuredToolResult:
        """Query activity timeline.

        time_range: relative like "2h", "7d", "30m".
        Or use start/end with ISO datetime strings.
        Filter by actions, entity_name, session_id.
        """
        graph_mgr, _, global_graph = _get_ctx(ctx)
        entries = [
            entry
            for _, target in _scope_graphs(graph_mgr, global_graph, scope)
            for entry in target.query_timeline(
                time_range=time_range,
                start=start,
                end=end,
                actions=actions,
                entity_name=entity_name,
                session_id=session_id,
                limit=limit,
            )
        ]
        entries.sort(key=lambda entry: entry.created_at, reverse=True)
        entries = entries[:limit]
        return json.dumps(
            [{"action": e.action, "summary": e.summary, "at": e.created_at} for e in entries],
            indent=2,
        )

    # ------------------------------------------------------------------
    # Tool 13: manage_tags
    # ------------------------------------------------------------------
    @mcp.tool(annotations=DESTRUCTIVE_TOOL)
    def manage_tags(
        ctx: Context,
        action: str = "list",
        name: str = "",
        description: str = "",
        color: str = "",
        auto_expire_hours: Annotated[int | None, Field(ge=1, le=87_600)] = None,
        entity_name: str = "",
        tag_name: str = "",
        scope: Scope = "workspace",
    ) -> StructuredToolResult:
        """Manage tags. Actions: list, create, delete, tag, untag, cleanup.

        list: show all tags.
        create: new tag (name required, optional description/color/auto_expire_hours).
        delete: remove a user tag (cannot delete system tags).
        tag: apply tag_name to entity_name.
        untag: remove tag_name from entity_name.
        cleanup: remove expired ephemeral items.
        """
        if scope == "all" and action in DESTRUCTIVE_TAG_ACTIONS:
            return _destructive_scope_all_error(action=action)
        graph_mgr, _, global_graph = _get_ctx(ctx)
        targets = _scope_graphs(graph_mgr, global_graph, scope)
        if action == "tag" and scope == "all" and entity_name:
            owner = _entity_owner(graph_mgr, global_graph, entity_name)
            if owner is None:
                raise ValueError(f"Entity not found: {entity_name}")
            targets = [owner]

        if action == "list":
            tags = [tag for _, target in targets for tag in target.list_tags()]
            return json.dumps(
                [
                    {
                        "name": t.name,
                        "description": t.description,
                        "system": t.is_system,
                        "auto_expire_hours": t.auto_expire_hours,
                    }
                    for t in tags
                ],
                indent=2,
            )

        elif action == "create":
            if not name:
                raise ValueError("name required")
            tag = targets[0][1].create_tag(name, description, color, auto_expire_hours)
            return json.dumps({"created": tag.name})

        elif action == "delete":
            if not name:
                raise ValueError("name required")
            try:
                ok = targets[0][1].delete_tag(name)
                return json.dumps({"deleted": ok})
            except ValueError as e:
                raise ValueError(str(e)) from e

        elif action == "tag":
            if not entity_name or not tag_name:
                raise ValueError("entity_name and tag_name required")
            ok = targets[0][1].tag_entity(entity_name, tag_name)
            return json.dumps({"tagged": ok})

        elif action == "untag":
            if not entity_name or not tag_name:
                raise ValueError("entity_name and tag_name required")
            ok = targets[0][1].untag_entity(entity_name, tag_name)
            return json.dumps({"untagged": ok})

        elif action == "cleanup":
            cleaned = sum(target.db.cleanup_expired() for _, target in targets)
            return json.dumps({"cleaned": cleaned})

        raise ValueError(f"Unknown action: {action}")

    # ------------------------------------------------------------------
    # Tool 14: merge_entities
    # ------------------------------------------------------------------
    @mcp.tool(annotations=DESTRUCTIVE_TOOL)
    def merge_entities(
        ctx: Context,
        source: str,
        target: str,
        strategy: str = "combine",
        scope: Scope = "workspace",
    ) -> StructuredToolResult:
        """Merge source entity into target. Source is soft-deleted.

        strategy: 'combine' (move all observations) or 'dedupe' (skip duplicates).
        Relations and tags are transferred. Self-relations are removed.
        """
        if scope == "all":
            return _destructive_scope_all_error()
        graph_mgr, _, global_graph = _get_ctx(ctx)
        targets = _scope_graphs(graph_mgr, global_graph, scope)
        result = targets[0][1].merge_entities(source, target, strategy)
        if result:
            return json.dumps(result.to_dict(), indent=2)
        raise ValueError("Source or target entity not found")

    # ------------------------------------------------------------------
    # Tool 15: export_graph / import_graph
    # ------------------------------------------------------------------
    @mcp.tool(annotations=READ_ONLY_TOOL)
    def export_graph(
        ctx: Context,
        format: str = "json",
        scope: Scope = "workspace",
    ) -> StructuredToolResult:
        """Export the full knowledge graph.

        format:
          - 'json' / 'jsonl': graph payload compatible with classic MCP memory JSONL
          - 'snapshot': lossless multi-scope snapshot
            (`format=server-memory-multiscope-snapshot`, `version=1`,
            `scopes` map of per-database snapshot tables)
        """
        graph_mgr, _, global_graph = _get_ctx(ctx)
        targets = _scope_graphs(graph_mgr, global_graph, scope)
        if format == "snapshot":
            return json.dumps(
                {
                    "format": "server-memory-multiscope-snapshot",
                    "version": 1,
                    "scopes": {
                        scope_name: target.db.export_snapshot()
                        for scope_name, target in targets
                    },
                },
                separators=(",", ":"),
            )
        if len(targets) == 1:
            return targets[0][1].export_graph(fmt=format)
        return json.dumps(
            {
                scope_name: json.loads(target.export_graph(fmt="json"))
                for scope_name, target in targets
            },
            indent=2,
        )

    @mcp.tool(annotations=DESTRUCTIVE_TOOL)
    def import_graph(
        ctx: Context,
        data: str,
        scope: Scope = "workspace",
    ) -> StructuredToolResult:
        """Import knowledge graph data. Auto-detects JSON or JSONL format.

        Compatible with old @modelcontextprotocol/server-memory JSONL files.
        Skips duplicate entities. Skips relations to missing entities.
        """
        # Character length is a cheap upper bound; UTF-8 can only be shorter or equal
        # for pure ASCII and at most 4x for worst-case code points.
        if len(data) > 50 * 1024 * 1024:
            raise ValueError("import payload exceeds 50 MiB limit")
        graph_mgr, _, global_graph = _get_ctx(ctx)
        parsed: Any = None
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            pass
        if isinstance(parsed, dict) and parsed.get("format") == "server-memory-multiscope-snapshot":
            if parsed.get("version") != 1 or not isinstance(parsed.get("scopes"), dict):
                raise ValueError("unsupported multi-scope snapshot")
            targets = dict(_scope_graphs(graph_mgr, global_graph, scope))
            requested = parsed["scopes"]
            if not set(requested).issubset(targets):
                raise ValueError("snapshot contains an unavailable or unrequested scope")
            # Validate every payload against disposable in-memory stores before
            # changing either persistent scope.
            for snapshot in requested.values():
                validator = Database(":memory:")
                validator.open()
                try:
                    validator.import_snapshot(snapshot)
                finally:
                    validator.close()
            prior = {
                scope_name: targets[scope_name].db.export_snapshot()
                for scope_name in requested
            }
            applied: list[str] = []
            try:
                for scope_name, snapshot in requested.items():
                    targets[scope_name].db.import_snapshot(snapshot)
                    applied.append(scope_name)
            except Exception:
                for scope_name in reversed(applied):
                    targets[scope_name].db.import_snapshot(
                        prior[scope_name], conflict="replace"
                    )
                raise
            return json.dumps({"imported_scopes": sorted(requested)})
        if scope == "all":
            return _destructive_scope_all_error()
        target = _scope_graphs(graph_mgr, global_graph, scope)[0][1]
        counts = target.import_graph(data)
        return json.dumps(counts)

    # ------------------------------------------------------------------
    # Tool 16: memory_stats
    # ------------------------------------------------------------------
    @mcp.tool(annotations=READ_ONLY_TOOL)
    def memory_stats(ctx: Context, scope: Scope = "workspace") -> StructuredToolResult:
        """Get memory statistics: entity/relation/observation counts,
        tag distribution, DB size, orphan entities, deleted items.
        """
        graph_mgr, _, global_graph = _get_ctx(ctx)
        scoped_stats = {
            scope_name: target.memory_stats()
            for scope_name, target in _scope_graphs(graph_mgr, global_graph, scope)
        }
        if len(scoped_stats) == 1:
            stats = next(iter(scoped_stats.values()))
        else:
            stats = {"scopes": scoped_stats}
        return json.dumps(stats, indent=2)

    # ------------------------------------------------------------------
    # Tool 17: backup_memory
    # ------------------------------------------------------------------
    @mcp.tool(annotations=WRITE_TOOL)
    def backup_memory(
        ctx: Context,
        dest_path: str = "",
        scope: Scope = "workspace",
    ) -> StructuredToolResult:
        """Create a backup of the memory database.

        Default destination: ~/.local/share/server-memory/backups/memory-YYYYMMDD-HHMMSS.db
        Provide dest_path to override the backup location.
        """
        graph_mgr, _, global_graph = _get_ctx(ctx)
        targets = _scope_graphs(graph_mgr, global_graph, scope)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        paths: dict[str, str] = {}
        for scope_name, target in targets:
            backup_path = _resolve_backup_path(
                target.db.db_path,
                dest_path,
                scope_name=scope_name,
                multiple_scopes=len(targets) > 1,
                timestamp=ts,
            )
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            target.db.backup(str(backup_path))
            paths[scope_name] = str(backup_path)
        result: dict[str, Any] = {"backup_paths": paths, "status": "ok"}
        if len(paths) == 1:
            result["backup_path"] = next(iter(paths.values()))
        return json.dumps(result)

    # ------------------------------------------------------------------
    # Tool 18: get_observation_history
    # ------------------------------------------------------------------
    @mcp.tool(annotations=READ_ONLY_TOOL)
    def get_observation_history(
        ctx: Context,
        entity_name: str,
        content_prefix: str = "",
        scope: Scope = "workspace",
    ) -> StructuredToolResult:
        """Get observation version history for an entity.

        Shows current content, version number, importance, obs_type,
        and all previous versions with timestamps.
        Use content_prefix to filter to specific observations.
        """
        graph_mgr, _, global_graph = _get_ctx(ctx)
        histories = [
            {
                "scope": scope_name,
                "history": target.get_observation_history(entity_name, content_prefix),
            }
            for scope_name, target in _scope_graphs(graph_mgr, global_graph, scope)
        ]
        history = histories[0]["history"] if len(histories) == 1 else histories
        if not history:
            raise ValueError(f"No observations found for '{entity_name}'")
        return json.dumps(history, indent=2)

    # ------------------------------------------------------------------
    # Tool 19: memory_context_full
    # ------------------------------------------------------------------
    @mcp.tool(annotations=READ_ONLY_TOOL)
    def memory_context_full(
        ctx: Context,
        project: str = "",
        budget: Annotated[int, Field(ge=100, le=100_000)] = 1000,
        scope: Scope = "all",
    ) -> StructuredToolResult:
        """Rich context snapshot for rare deep bootstrap (~500-1500 tokens).

        Returns all pinned entities with full observations, recent activity (last 10),
        and recently changed entities. Use only when compact recall is insufficient
        for a cross-session task. Prefer memory_context for ordinary scoped recall.
        """
        graph_mgr, cfg, global_graph_mgr = _get_ctx(ctx)
        active_project = project or cfg.project

        targets = _scope_graphs(graph_mgr, global_graph_mgr, scope)
        scoped_graphs: list[tuple[str, KnowledgeGraph]] = []
        for scope_name, target in targets:
            target_kg = target.read_graph(tags=["pinned"], limit=20)
            if scope_name == "workspace" and active_project:
                project_kg = target.read_graph(tags=[active_project], limit=20)
                known_ids = {entity.id for entity in target_kg.entities}
                target_kg.entities.extend(
                    entity for entity in project_kg.entities if entity.id not in known_ids
                )
                target_kg.relations.extend(project_kg.relations)
            elif scope_name == "global":
                preference_kg = target.read_graph(tags=[PREFERENCE_TAG], limit=20)
                known_ids = {entity.id for entity in target_kg.entities}
                target_kg.entities.extend(
                    entity for entity in preference_kg.entities if entity.id not in known_ids
                )
                target_kg.relations.extend(preference_kg.relations)
            scoped_graphs.append((scope_name, target_kg))
        kg = _merge_scoped_graphs(scoped_graphs)

        pinned_ids = set().union(*(_get_pinned_ids(target) for _, target in targets))
        output = compress_graph(kg, cfg.compression_level, budget, pinned_ids)

        # Append recent activity
        recent = [
            entry
            for _, target in targets
            for entry in target.query_timeline(time_range="24h", limit=10)
        ]
        recent.sort(key=lambda entry: entry.created_at, reverse=True)
        recent = recent[:10]
        if recent:
            acts = [f"{e.action}:{e.summary}" for e in recent]
            output += "\n---\nRecent(24h): " + " | ".join(acts)

        return _enforce_budget(output, budget)

    return mcp


def _get_pinned_ids(graph_mgr: KnowledgeGraphManager) -> set[int]:
    """Get IDs of pinned entities for compression priority."""
    cx = graph_mgr.db.cx
    rows = cx.execute(
        """
        SELECT e.id FROM entities e
        JOIN entity_tags et ON et.entity_id = e.id
        JOIN tags t ON t.id = et.tag_id
        WHERE t.name = 'pinned' AND e.deleted_at IS NULL
        """
    ).fetchall()
    return {r["id"] for r in rows}
