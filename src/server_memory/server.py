"""FastMCP server with all 20 memory tools."""

from __future__ import annotations

import json
import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from mcp.server.fastmcp import Context, FastMCP

from .compression import compress_graph
from .config import MemoryConfig
from .db import Database
from .embeddings import EmbeddingEngine
from .graph import KnowledgeGraphManager
from .local_auth import LocalTokenVerifier, build_local_auth_settings, ensure_local_auth_token

logger = logging.getLogger(__name__)

_startup_cleanup_lock = threading.Lock()
_startup_cleanup_started = False

PREFERENCE_TAG = "preference"
WORKSPACE_MEMORY_SCORE_BONUS = 0.2
Scope = Literal["workspace", "global", "all"]

INSTRUCTIONS = (
    "Memory server for persistent knowledge across conversations.\n"
    "Use memory_context(hint='current topic', limit=3-5) when prior sessions, stable project facts, or cross-session continuity may matter.\n"
    "Skip memory lookup for one-off answers, purely local inspection, or tasks already fully grounded in the current context.\n"
    "After durable changes or decisions, call log_activity(action='...', summary='...'); do not log routine chatter.\n"
    "Use search_nodes for targeted full-text search, create_entities/add_observations to store durable knowledge.\n"
    "Tag important items with 'pinned' only when they must remain visible in future scoped recall."
)


def _run_startup_cleanup(cfg: MemoryConfig) -> None:
    """Run non-critical cleanup outside the MCP handshake path."""
    cleanup_db = Database(cfg.db_path)
    try:
        cfg.ensure_db_dir()
        cleanup_db.open()

        cleaned = cleanup_db.cleanup_expired()
        if cleaned:
            logger.info("Cleaned up %d expired items", cleaned)

        unused_cleaned = cleanup_db.cleanup_unused(days=30)
        if unused_cleaned:
            logger.info("Cleaned up %d unused items", unused_cleaned)

        empty_cleaned = cleanup_db.cleanup_empty_stale(days=7)
        if empty_cleaned:
            logger.info("Cleaned up %d empty stale entities", empty_cleaned)
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
        pinned_parts = []
        for p in result["pinned"]:
            src = f" ({p['source']})" if p.get("source") else ""
            pinned_parts.append(f"{p['name']}[{p['type']}{src}]")
        pinned_str = ", ".join(pinned_parts)
        lines.append(f"Pinned: {pinned_str}")
    if result["recent_activity"]:
        acts = [f"{a['action']}:{a['summary']}" for a in result["recent_activity"]]
        lines.append(f"Recent: {' | '.join(acts)}")
    if result["hint_matches"]:
        hint_parts = []
        for h in result["hint_matches"]:
            src = f" ({h['source']})" if h.get("source") else ""
            part = f"{h['name']}[{h['type']}{src}]"
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
    seen: set[tuple[str, str]] = set()

    for entry in [*workspace_activity, *global_activity]:
        key = (entry.get("action", ""), entry.get("summary", ""))
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
        for item in workspace_result.get("pinned", []):
            item.setdefault("source", "workspace")
        for item in workspace_result.get("hint_matches", []):
            item.setdefault("source", "workspace")
        return workspace_result

    # Ensure source is labeled
    for item in workspace_result.get("pinned", []):
        item.setdefault("source", "workspace")
    for item in global_result.get("pinned", []):
        item.setdefault("source", "global")

    pinned_by_identity: dict[tuple[str, str, str], dict[str, Any]] = {
        (item.get("source", "workspace"), item.get("name", ""), item.get("type", "")): item
        for item in workspace_result.get("pinned", [])
    }
    for item in global_result.get("pinned", []):
        pinned_by_identity.setdefault(
            (item.get("source", "global"), item.get("name", ""), item.get("type", "")), item
        )

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
            "entities": int(workspace_stats.get("entities", 0))
            + int(global_stats.get("entities", 0)),
            "observations": int(workspace_stats.get("observations", 0))
            + int(global_stats.get("observations", 0)),
            "relations": int(workspace_stats.get("relations", 0))
            + int(global_stats.get("relations", 0)),
            "workspace_entities": int(workspace_stats.get("entities", 0)),
            "global_entities": int(global_stats.get("entities", 0)),
        },
    }


def _normalize_scope(scope: str) -> Scope:
    normalized = scope.strip().lower()
    if normalized in {"workspace", "global", "all"}:
        return normalized  # type: ignore[return-value]
    raise ValueError("scope must be one of: workspace, global, all")


def _scoped_graphs(
    graph_mgr: KnowledgeGraphManager,
    global_graph_mgr: KnowledgeGraphManager | None,
    scope: str,
) -> list[tuple[str, KnowledgeGraphManager]]:
    """Resolve a scope into deterministic graph targets."""
    normalized = _normalize_scope(scope)
    if normalized == "workspace":
        return [("workspace", graph_mgr)]
    if normalized == "global":
        return [("global", global_graph_mgr)] if global_graph_mgr is not None else []
    targets = [("workspace", graph_mgr)]
    if global_graph_mgr is not None:
        targets.append(("global", global_graph_mgr))
    return targets


def _scope_error(scope: str) -> str:
    return json.dumps({"error": f"No database available for scope '{scope}'"})


def _source_wrapped(source: str, payload: Any) -> dict[str, Any]:
    return {"source": source, "result": payload}


def _safe_close_db(database: Database | None, has_active_exc: bool) -> None:
    if database is None:
        return
    try:
        database.close()
    except Exception as exc:
        if has_active_exc:
            logger.error(
                "Suppressing database close error because another exception is active: %s", exc
            )
        else:
            raise


def _close_lifespan_databases(db: Database | None, global_db: Database | None) -> None:
    """Close lifespan databases safely, ensuring both are closed and exceptions are handled correctly."""
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
        db = Database(cfg.db_path)
        db.open()

        global_db: Database | None = None
        try:
            global_graph_mgr: KnowledgeGraphManager | None = None
            use_global_db = cfg.global_db_enabled and cfg.global_db_path != cfg.db_path
            if use_global_db:
                cfg.ensure_global_db_dir()
                g_db = Database(cfg.global_db_path)
                g_db.open()
                global_db = g_db

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
        lifespan_ctx = ctx.request_context.lifespan_context
        return lifespan_ctx["graph"], lifespan_ctx["config"], lifespan_ctx.get("global_graph")

    # ------------------------------------------------------------------
    # Tool 1: create_entities
    # ------------------------------------------------------------------
    @mcp.tool()
    def create_entities(
        ctx: Context,
        entities: list[dict[str, Any]],
        scope: str = "workspace",
    ) -> str:
        """Create new entities in the knowledge graph.

        Each entity needs: name (str), entityType (str).
        Optional: observations (list[str]), tags (list[str]), metadata (dict).
        Skips duplicates by name.
        """
        graph, cfg, global_graph = _get_ctx(ctx)
        normalized_scope = _normalize_scope(scope)
        if normalized_scope == "global":
            if global_graph is None:
                return _scope_error(scope)
            created = global_graph.create_entities(entities)
            return json.dumps([e.to_dict() for e in created], indent=2)
        if normalized_scope == "all":
            results = []
            for source, target_graph in _scoped_graphs(graph, global_graph, scope):
                created = target_graph.create_entities(entities)
                results.append(_source_wrapped(source, [e.to_dict() for e in created]))
            return json.dumps(results, indent=2)

        workspace_entities = entities
        global_entities: list[dict[str, Any]] = []
        if cfg.global_preference_routing_enabled and global_graph is not None:
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
    @mcp.tool()
    def create_relations(
        ctx: Context,
        relations: list[dict[str, Any]],
        scope: str = "workspace",
    ) -> str:
        """Create relations between entities.

        Each relation needs: from (str), to (str), relationType (str).
        Optional: weight (float), tags (list[str]).
        Both entities must exist. Use active voice for relationType.
        """
        graph, _, global_graph = _get_ctx(ctx)
        try:
            targets = _scoped_graphs(graph, global_graph, scope)
            if not targets:
                return _scope_error(scope)
            if len(targets) == 1:
                created = targets[0][1].create_relations(relations)
                return json.dumps([r.to_dict() for r in created], indent=2)
            results = []
            for source, target_graph in targets:
                created = target_graph.create_relations(relations)
                results.append(_source_wrapped(source, [r.to_dict() for r in created]))
            return json.dumps(results, indent=2)
        except ValueError as e:
            return json.dumps({"error": str(e)})

    # ------------------------------------------------------------------
    # Tool 3: add_observations
    # ------------------------------------------------------------------
    @mcp.tool()
    def add_observations(
        ctx: Context,
        observations: list[dict[str, Any]],
        scope: str = "workspace",
    ) -> str:
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
        graph, cfg, global_graph = _get_ctx(ctx)
        try:
            normalized_scope = _normalize_scope(scope)
            if normalized_scope == "global":
                if global_graph is None:
                    return _scope_error(scope)
                return json.dumps(global_graph.add_observations(observations), indent=2)
            if normalized_scope == "all":
                results = []
                for source, target_graph in _scoped_graphs(graph, global_graph, scope):
                    results.append(
                        _source_wrapped(source, target_graph.add_observations(observations))
                    )
                return json.dumps(results, indent=2)

            workspace_observations = observations
            global_observations: list[dict[str, Any]] = []
            if cfg.global_preference_routing_enabled and global_graph is not None:
                workspace_observations = []
                for observation in observations:
                    if _is_preference_observation(observation):
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
            return json.dumps({"error": str(e)})

    # ------------------------------------------------------------------
    # Tool 4: delete_entities
    # ------------------------------------------------------------------
    @mcp.tool()
    def delete_entities(
        ctx: Context,
        entityNames: list[str],
        hard: bool = False,
        scope: str = "workspace",
    ) -> str:
        """Delete entities (soft delete by default, cascades to relations).

        Set hard=true for permanent deletion.
        """
        graph, _, global_graph = _get_ctx(ctx)
        normalized_scope = _normalize_scope(scope)
        if normalized_scope == "all":
            return json.dumps(
                {
                    "error": "scope='all' is not supported for destructive operations to prevent unintentional data loss. Please specify 'workspace' or 'global'."
                }
            )
        targets = _scoped_graphs(graph, global_graph, scope)
        if not targets:
            return _scope_error(scope)
        if len(targets) == 1:
            count = targets[0][1].delete_entities(entityNames, hard=hard)
            return json.dumps({"deleted": count})
        results = [
            {"source": source, "deleted": target_graph.delete_entities(entityNames, hard=hard)}
            for source, target_graph in targets
        ]
        return json.dumps(results, indent=2)

    # ------------------------------------------------------------------
    # Tool 5: delete_observations
    # ------------------------------------------------------------------
    @mcp.tool()
    def delete_observations(
        ctx: Context,
        deletions: list[dict[str, Any]],
        scope: str = "workspace",
    ) -> str:
        """Delete specific observations from entities.

        Each item needs: entityName (str), observations (list[str] of content to delete).
        """
        graph, _, global_graph = _get_ctx(ctx)
        normalized_scope = _normalize_scope(scope)
        if normalized_scope == "all":
            return json.dumps(
                {
                    "error": "scope='all' is not supported for destructive operations to prevent unintentional data loss. Please specify 'workspace' or 'global'."
                }
            )
        targets = _scoped_graphs(graph, global_graph, scope)
        if not targets:
            return _scope_error(scope)
        if len(targets) == 1:
            count = targets[0][1].delete_observations(deletions)
            return json.dumps({"deleted": count})
        results = [
            {"source": source, "deleted": target_graph.delete_observations(deletions)}
            for source, target_graph in targets
        ]
        return json.dumps(results, indent=2)

    # ------------------------------------------------------------------
    # Tool 6: delete_relations
    # ------------------------------------------------------------------
    @mcp.tool()
    def delete_relations(
        ctx: Context,
        relations: list[dict[str, Any]],
        scope: str = "workspace",
    ) -> str:
        """Delete relations from the knowledge graph.

        Each item needs: from (str), to (str), relationType (str).
        """
        graph, _, global_graph = _get_ctx(ctx)
        normalized_scope = _normalize_scope(scope)
        if normalized_scope == "all":
            return json.dumps(
                {
                    "error": "scope='all' is not supported for destructive operations to prevent unintentional data loss. Please specify 'workspace' or 'global'."
                }
            )
        targets = _scoped_graphs(graph, global_graph, scope)
        if not targets:
            return _scope_error(scope)
        if len(targets) == 1:
            count = targets[0][1].delete_relations(relations)
            return json.dumps({"deleted": count})
        results = [
            {"source": source, "deleted": target_graph.delete_relations(relations)}
            for source, target_graph in targets
        ]
        return json.dumps(results, indent=2)

    # ------------------------------------------------------------------
    # Tool 7: read_graph
    # ------------------------------------------------------------------
    @mcp.tool()
    def read_graph(
        ctx: Context,
        tags: list[str] | None = None,
        entity_types: list[str] | None = None,
        limit: int = 0,
        include_deleted: bool = False,
        compress: bool = True,
        scope: str = "workspace",
    ) -> str:
        """Read the knowledge graph (compressed by default).

        Filter by tags, entity_types. Set compress=false for full JSON.
        """
        graph_mgr, cfg, global_graph_mgr = _get_ctx(ctx)
        targets = _scoped_graphs(graph_mgr, global_graph_mgr, scope)
        if not targets:
            return _scope_error(scope)
        if len(targets) > 1:
            results = []
            for source, target_graph in targets:
                kg = target_graph.read_graph(
                    tags=tags,
                    entity_types=entity_types,
                    limit=limit,
                    include_deleted=include_deleted,
                )
                payload: Any
                if compress:
                    pinned_ids = _get_pinned_ids(target_graph)
                    payload = compress_graph(
                        kg, cfg.compression_level, cfg.token_budget, pinned_ids
                    )
                else:
                    payload = kg.to_dict()
                results.append(_source_wrapped(source, payload))
            return json.dumps(results, indent=2)
        graph_mgr = targets[0][1]
        kg = graph_mgr.read_graph(
            tags=tags,
            entity_types=entity_types,
            limit=limit,
            include_deleted=include_deleted,
        )
        if compress:
            pinned_ids = _get_pinned_ids(graph_mgr)
            return compress_graph(kg, cfg.compression_level, cfg.token_budget, pinned_ids)
        return json.dumps(kg.to_dict(), indent=2)

    # ------------------------------------------------------------------
    # Tool 8: search_nodes
    # ------------------------------------------------------------------
    @mcp.tool()
    def search_nodes(
        ctx: Context,
        query: str,
        tags: list[str] | None = None,
        entity_types: list[str] | None = None,
        time_range: list[str] | None = None,
        limit: int = 20,
        compress: bool = True,
        scope: str = "workspace",
    ) -> str:
        """Full-text search with BM25 ranking.

        Supports prefix search, phrases ("exact match"), boolean (AND/OR/NOT).
        Filter by tags, entity_types, time_range ([start, end] ISO strings).
        """
        graph_mgr, cfg, global_graph_mgr = _get_ctx(ctx)
        targets = _scoped_graphs(graph_mgr, global_graph_mgr, scope)
        if not targets:
            return _scope_error(scope)
        tr = tuple(time_range) if time_range and len(time_range) == 2 else None
        if len(targets) > 1:
            results = []
            for source, target_graph in targets:
                kg = target_graph.search_fts(
                    query,
                    tags=tags,
                    entity_types=entity_types,
                    time_range=tr,
                    limit=limit,
                )
                payload = (
                    compress_graph(kg, cfg.compression_level, cfg.token_budget)
                    if compress
                    else kg.to_dict()
                )
                results.append(_source_wrapped(source, payload))
            return json.dumps(results, indent=2)
        graph_mgr = targets[0][1]
        kg = graph_mgr.search_fts(
            query,
            tags=tags,
            entity_types=entity_types,
            time_range=tr,
            limit=limit,
        )
        if compress:
            return compress_graph(kg, cfg.compression_level, cfg.token_budget)
        return json.dumps(kg.to_dict(), indent=2)

    # ------------------------------------------------------------------
    # Tool 9: open_nodes
    # ------------------------------------------------------------------
    @mcp.tool()
    def open_nodes(
        ctx: Context,
        names: list[str],
        depth: int = 0,
        scope: str = "workspace",
    ) -> str:
        """Open specific entities by name.

        depth=0: exact entities only.
        depth=1: include direct neighbors via relations.
        depth=2+: BFS expansion.
        """
        graph_mgr, _, global_graph_mgr = _get_ctx(ctx)
        targets = _scoped_graphs(graph_mgr, global_graph_mgr, scope)
        if not targets:
            return _scope_error(scope)
        if len(targets) == 1:
            kg = targets[0][1].open_nodes(names, depth=depth)
            return json.dumps(kg.to_dict(), indent=2)
        results = [
            _source_wrapped(source, target_graph.open_nodes(names, depth=depth).to_dict())
            for source, target_graph in targets
        ]
        return json.dumps(results, indent=2)

    # ------------------------------------------------------------------
    # Tool 10: memory_context (THE KEY TOOL)
    # ------------------------------------------------------------------
    @mcp.tool()
    def memory_context(
        ctx: Context,
        hint: str = "",
        project: str = "",
        limit: int = 10,
        scope: str = "all",
    ) -> str:
        """Lightweight context snapshot (~200-500 tokens) for scoped durable recall.

        Returns: pinned entities, recent activity, hint-matched entities, and graph stats.
        Call when prior sessions, stable project facts, or cross-session continuity may matter.
        Skip for one-off answers or tasks already fully grounded in the current context.
        Pass hint='current topic' to get relevant entities surfaced.
        Pass project='name' to scope results to a specific project tag.
        Pass limit to control how many hint matches are included; prefer 3-5 unless deeper recall is justified.
        """
        graph_mgr, cfg, global_graph_mgr = _get_ctx(ctx)
        normalized_scope = _normalize_scope(scope)
        if normalized_scope == "workspace":
            res = graph_mgr.memory_context(hint=hint, project=project, limit=limit)
            for item in res.get("pinned", []):
                item.setdefault("source", "workspace")
            for item in res.get("hint_matches", []):
                item.setdefault("source", "workspace")
            return _format_memory_context_result(res)
        if normalized_scope == "global":
            if global_graph_mgr is None:
                return _scope_error(scope)
            res = global_graph_mgr.memory_context(hint=hint, project="", limit=limit)
            for item in res.get("pinned", []):
                item.setdefault("source", "global")
            for item in res.get("hint_matches", []):
                item.setdefault("source", "global")
            return _format_memory_context_result(res)
        workspace_result = graph_mgr.memory_context(hint=hint, project=project, limit=limit)
        global_result = None
        if global_graph_mgr is not None:
            global_result = global_graph_mgr.memory_context(hint=hint, project="", limit=limit)
        result = merge_memory_context_results(workspace_result, global_result, limit=limit)
        return _format_memory_context_result(result)

    # ------------------------------------------------------------------
    # Tool 11: log_activity
    # ------------------------------------------------------------------
    @mcp.tool()
    def log_activity(
        ctx: Context,
        action: str,
        summary: str = "",
        entity_names: list[str] | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        scope: str = "workspace",
    ) -> str:
        """Record what happened this turn. Auto-creates missing entities.

        Common actions: file_changed, decision_made, bug_fixed, feature_added,
        refactored, investigated, discussed, preference_set.
        """
        graph_mgr, cfg, global_graph_mgr = _get_ctx(ctx)
        normalized_scope = _normalize_scope(scope)
        if normalized_scope == "global":
            if global_graph_mgr is None:
                return _scope_error(scope)
            target_graph = global_graph_mgr
        elif normalized_scope == "all":
            results = []
            for source, target_graph in _scoped_graphs(graph_mgr, global_graph_mgr, scope):
                entry = target_graph.log_activity(
                    action=action,
                    summary=summary,
                    entity_names=entity_names,
                    tags=tags,
                    metadata=metadata,
                )
                results.append(
                    {
                        "source": source,
                        "id": entry.id,
                        "action": entry.action,
                        "summary": entry.summary,
                    }
                )
            return json.dumps(results, indent=2)
        else:
            target_graph = graph_mgr
        if (
            normalized_scope == "workspace"
            and cfg.global_preference_routing_enabled
            and global_graph_mgr is not None
            and action == "preference_set"
        ):
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
    @mcp.tool()
    def query_timeline(
        ctx: Context,
        time_range: str | None = None,
        start: str | None = None,
        end: str | None = None,
        actions: list[str] | None = None,
        entity_name: str | None = None,
        session_id: str | None = None,
        limit: int = 50,
        scope: str = "workspace",
    ) -> str:
        """Query activity timeline.

        time_range: relative like "2h", "7d", "30m".
        Or use start/end with ISO datetime strings.
        Filter by actions, entity_name, session_id.
        """
        graph_mgr, _, global_graph_mgr = _get_ctx(ctx)
        targets = _scoped_graphs(graph_mgr, global_graph_mgr, scope)
        if not targets:
            return _scope_error(scope)

        def timeline_payload(target_graph: KnowledgeGraphManager) -> list[dict[str, Any]]:
            entries = target_graph.query_timeline(
                time_range=time_range,
                start=start,
                end=end,
                actions=actions,
                entity_name=entity_name,
                session_id=session_id,
                limit=limit,
            )
            return [{"action": e.action, "summary": e.summary, "at": e.created_at} for e in entries]

        if len(targets) == 1:
            return json.dumps(timeline_payload(targets[0][1]), indent=2)
        return json.dumps(
            [
                _source_wrapped(source, timeline_payload(target_graph))
                for source, target_graph in targets
            ],
            indent=2,
        )

    # ------------------------------------------------------------------
    # Tool 13: manage_tags
    # ------------------------------------------------------------------
    @mcp.tool()
    def manage_tags(
        ctx: Context,
        action: str = "list",
        name: str = "",
        description: str = "",
        color: str = "",
        auto_expire_hours: int | None = None,
        entity_name: str = "",
        tag_name: str = "",
        scope: str = "workspace",
    ) -> str:
        """Manage tags. Actions: list, create, delete, tag, untag, cleanup.

        list: show all tags.
        create: new tag (name required, optional description/color/auto_expire_hours).
        delete: remove a user tag (cannot delete system tags).
        tag: apply tag_name to entity_name.
        untag: remove tag_name from entity_name.
        cleanup: remove expired ephemeral items.
        """
        graph_mgr, _, global_graph_mgr = _get_ctx(ctx)
        normalized_scope = _normalize_scope(scope)
        if normalized_scope == "all" and action in {"delete", "untag"}:
            return json.dumps(
                {
                    "error": f"scope='all' is not supported for destructive tag action '{action}' to prevent unintentional data loss. Please specify 'workspace' or 'global'."
                }
            )
        targets = _scoped_graphs(graph_mgr, global_graph_mgr, scope)
        if not targets:
            return _scope_error(scope)

        def run_manage_tags(target_graph: KnowledgeGraphManager) -> Any:
            if action == "list":
                tags = target_graph.list_tags()
                return [
                    {
                        "name": t.name,
                        "description": t.description,
                        "system": t.is_system,
                        "auto_expire_hours": t.auto_expire_hours,
                    }
                    for t in tags
                ]

            if action == "create":
                if not name:
                    return {"error": "name required"}
                tag = target_graph.create_tag(name, description, color, auto_expire_hours)
                return {"created": tag.name}

            if action == "delete":
                if not name:
                    return {"error": "name required"}
                try:
                    ok = target_graph.delete_tag(name)
                    return {"deleted": ok}
                except ValueError as e:
                    return {"error": str(e)}

            if action == "tag":
                if not entity_name or not tag_name:
                    return {"error": "entity_name and tag_name required"}
                ok = target_graph.tag_entity(entity_name, tag_name)
                return {"tagged": ok}

            if action == "untag":
                if not entity_name or not tag_name:
                    return {"error": "entity_name and tag_name required"}
                ok = target_graph.untag_entity(entity_name, tag_name)
                return {"untagged": ok}

            if action == "cleanup":
                cleaned = target_graph.db.cleanup_expired()
                return {"cleaned": cleaned}

            return {"error": f"Unknown action: {action}"}

        if len(targets) == 1:
            return json.dumps(run_manage_tags(targets[0][1]), indent=2)
        return json.dumps(
            [
                _source_wrapped(source, run_manage_tags(target_graph))
                for source, target_graph in targets
            ],
            indent=2,
        )

    # ------------------------------------------------------------------
    # Tool 14: merge_entities
    # ------------------------------------------------------------------
    @mcp.tool()
    def merge_entities(
        ctx: Context,
        source: str,
        target: str,
        strategy: str = "combine",
        scope: str = "workspace",
    ) -> str:
        """Merge source entity into target. Source is soft-deleted.

        strategy: 'combine' (move all observations) or 'dedupe' (skip duplicates).
        Relations and tags are transferred. Self-relations are removed.
        """
        graph_mgr, _, global_graph_mgr = _get_ctx(ctx)
        normalized_scope = _normalize_scope(scope)
        if normalized_scope == "all":
            return json.dumps(
                {
                    "error": "scope='all' is not supported for destructive operations to prevent unintentional data loss. Please specify 'workspace' or 'global'."
                }
            )
        targets = _scoped_graphs(graph_mgr, global_graph_mgr, scope)
        if not targets:
            return _scope_error(scope)

        def merge_payload(target_graph: KnowledgeGraphManager) -> Any:
            result = target_graph.merge_entities(source, target, strategy)
            if result:
                return result.to_dict()
            return {"error": "Source or target entity not found"}

        if len(targets) == 1:
            return json.dumps(merge_payload(targets[0][1]), indent=2)
        return json.dumps(
            [
                _source_wrapped(source_name, merge_payload(target_graph))
                for source_name, target_graph in targets
            ],
            indent=2,
        )

    # ------------------------------------------------------------------
    # Tool 15: export_graph / import_graph
    # ------------------------------------------------------------------
    @mcp.tool()
    def export_graph(
        ctx: Context,
        format: str = "json",
        scope: str = "workspace",
    ) -> str:
        """Export the full knowledge graph.

        format: 'json' or 'jsonl' (compatible with old @modelcontextprotocol/server-memory).
        """
        graph_mgr, _, global_graph_mgr = _get_ctx(ctx)
        targets = _scoped_graphs(graph_mgr, global_graph_mgr, scope)
        if not targets:
            return _scope_error(scope)
        if len(targets) == 1:
            return targets[0][1].export_graph(fmt=format)
        if format == "jsonl":
            lines = []
            for source, target_graph in targets:
                for line in target_graph.export_graph(fmt="jsonl").splitlines():
                    if line.strip():
                        item = json.loads(line)
                        item["source"] = source
                        lines.append(json.dumps(item))
            return "\n".join(lines)
        return json.dumps(
            [
                _source_wrapped(source, json.loads(target_graph.export_graph(fmt="json")))
                for source, target_graph in targets
            ],
            indent=2,
        )

    @mcp.tool()
    def import_graph(
        ctx: Context,
        data: str,
        scope: str = "workspace",
    ) -> str:
        """Import knowledge graph data. Auto-detects JSON or JSONL format.

        Compatible with old @modelcontextprotocol/server-memory JSONL files.
        Skips duplicate entities. Skips relations to missing entities.
        """
        graph_mgr, _, global_graph_mgr = _get_ctx(ctx)
        targets = _scoped_graphs(graph_mgr, global_graph_mgr, scope)
        if not targets:
            return _scope_error(scope)
        if len(targets) == 1:
            counts = targets[0][1].import_graph(data)
            return json.dumps(counts)
        return json.dumps(
            [
                _source_wrapped(source, target_graph.import_graph(data))
                for source, target_graph in targets
            ],
            indent=2,
        )

    # ------------------------------------------------------------------
    # Tool 16: memory_stats
    # ------------------------------------------------------------------
    @mcp.tool()
    def memory_stats(ctx: Context, scope: str = "workspace") -> str:
        """Get memory statistics: entity/relation/observation counts,
        tag distribution, DB size, orphan entities, deleted items.
        """
        graph_mgr, _, global_graph_mgr = _get_ctx(ctx)
        targets = _scoped_graphs(graph_mgr, global_graph_mgr, scope)
        if not targets:
            return _scope_error(scope)
        if len(targets) == 1:
            return json.dumps(targets[0][1].memory_stats(), indent=2)
        return json.dumps(
            [
                _source_wrapped(source, target_graph.memory_stats())
                for source, target_graph in targets
            ],
            indent=2,
        )

    # ------------------------------------------------------------------
    # Tool 17: backup_memory
    # ------------------------------------------------------------------
    @mcp.tool()
    def backup_memory(
        ctx: Context,
        dest_path: str = "",
        scope: str = "workspace",
    ) -> str:
        """Create a backup of the memory database.

        Default destination: ~/.local/share/server-memory/backups/memory-YYYYMMDD-HHMMSS.db
        Provide dest_path to override the backup location.
        """
        graph_mgr, _, global_graph_mgr = _get_ctx(ctx)
        targets = _scoped_graphs(graph_mgr, global_graph_mgr, scope)
        if not targets:
            return _scope_error(scope)

        normalized_scope = _normalize_scope(scope)

        def backup_one(source: str, target_graph: KnowledgeGraphManager) -> dict[str, str]:
            db = target_graph.db
            if dest_path:
                requested_path = Path(dest_path)
                if len(targets) > 1:
                    if requested_path.suffix:
                        backup_path = (
                            requested_path.parent
                            / f"{requested_path.stem}-{source}{requested_path.suffix}"
                        )
                    else:
                        backup_path = requested_path / f"{source}.db"
                else:
                    backup_path = requested_path
            else:
                backup_dir = Path(db.db_path).parent / "backups"
                backup_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                backup_path = backup_dir / f"memory-{source}-{ts}.db"

            backup_path.parent.mkdir(parents=True, exist_ok=True)
            db.backup(str(backup_path))
            result = {"backup_path": str(backup_path), "status": "ok"}
            if normalized_scope != "workspace":
                result["source"] = source
            return result

        results = [backup_one(source, target_graph) for source, target_graph in targets]
        if len(results) == 1:
            return json.dumps(results[0])
        return json.dumps(results, indent=2)

    # ------------------------------------------------------------------
    # Tool 18: get_observation_history
    # ------------------------------------------------------------------
    @mcp.tool()
    def get_observation_history(
        ctx: Context,
        entity_name: str,
        content_prefix: str = "",
        scope: str = "workspace",
    ) -> str:
        """Get observation version history for an entity.

        Shows current content, version number, importance, obs_type,
        and all previous versions with timestamps.
        Use content_prefix to filter to specific observations.
        """
        graph_mgr, _, global_graph_mgr = _get_ctx(ctx)
        targets = _scoped_graphs(graph_mgr, global_graph_mgr, scope)
        if not targets:
            return _scope_error(scope)

        def history_payload(target_graph: KnowledgeGraphManager) -> Any:
            history = target_graph.get_observation_history(entity_name, content_prefix)
            if not history:
                return {"error": f"No observations found for '{entity_name}'"}
            return history

        if len(targets) == 1:
            return json.dumps(history_payload(targets[0][1]), indent=2)
        return json.dumps(
            [
                _source_wrapped(source, history_payload(target_graph))
                for source, target_graph in targets
            ],
            indent=2,
        )

    # ------------------------------------------------------------------
    # Tool 19: memory_context_full
    # ------------------------------------------------------------------
    @mcp.tool()
    def memory_context_full(
        ctx: Context,
        project: str = "",
        budget: int = 1000,
        scope: str = "all",
    ) -> str:
        """Rich context snapshot for rare deep bootstrap (~500-1500 tokens).

        Returns all pinned entities with full observations, recent activity (last 10),
        and recently changed entities. Use only when compact recall is insufficient
        for a cross-session task. Prefer memory_context for ordinary scoped recall.
        """
        graph_mgr, cfg, global_graph_mgr = _get_ctx(ctx)
        normalized_scope = _normalize_scope(scope)
        active_project = project or cfg.project

        # Read pinned entities with full data
        pinned_tags = ["pinned"]
        if active_project:
            pinned_tags.append(active_project)
        if normalized_scope == "global":
            if global_graph_mgr is None:
                return _scope_error(scope)
            graph_mgr = global_graph_mgr
            global_graph_mgr = None
            active_project = ""
        kg = graph_mgr.read_graph(tags=["pinned"], limit=20)

        # If project scoped, also get project-tagged entities
        if active_project:
            project_kg = graph_mgr.read_graph(tags=[active_project], limit=20)
            # Merge, avoiding duplicates
            existing_ids = {e.id for e in kg.entities}
            for e in project_kg.entities:
                if e.id not in existing_ids:
                    kg.entities.append(e)
                    existing_ids.add(e.id)
            for r in project_kg.relations:
                kg.relations.append(r)

        if normalized_scope == "all" and global_graph_mgr is not None:
            for entity in kg.entities:
                if "workspace" not in entity.tags:
                    entity.tags = list(entity.tags) + ["workspace"]

            global_kg = global_graph_mgr.read_graph(tags=["pinned", PREFERENCE_TAG], limit=20)
            existing_global_names = set()
            id_offset = 100_000_000
            for entity in global_kg.entities:
                if entity.name.lower() not in existing_global_names:
                    entity.id += id_offset
                    if "global" not in entity.tags:
                        entity.tags = list(entity.tags) + ["global"]
                    kg.entities.append(entity)
                    existing_global_names.add(entity.name.lower())
            kg.relations.extend(global_kg.relations)

        pinned_ids = _get_pinned_ids(graph_mgr)
        if normalized_scope == "all" and global_graph_mgr is not None:
            global_pinned = {pid + 100_000_000 for pid in _get_pinned_ids(global_graph_mgr)}
            pinned_ids |= global_pinned
        output = compress_graph(kg, cfg.compression_level, budget, pinned_ids)

        # Append recent activity
        recent = graph_mgr.query_timeline(time_range="24h", limit=10)
        if normalized_scope == "all" and global_graph_mgr is not None:
            recent.extend(global_graph_mgr.query_timeline(time_range="24h", limit=5))
        if recent:
            acts = [f"{e.action}:{e.summary}" for e in recent]
            output += "\n---\nRecent(24h): " + " | ".join(acts)

        return output

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
