"""FastMCP server with all 19 memory tools."""

from __future__ import annotations

import json
import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
        pinned_str = ", ".join(f"{p['name']}[{p['type']}]" for p in result["pinned"])
        lines.append(f"Pinned: {pinned_str}")
    if result["recent_activity"]:
        acts = [f"{a['action']}:{a['summary']}" for a in result["recent_activity"]]
        lines.append(f"Recent: {' | '.join(acts)}")
    if result["hint_matches"]:
        hint_parts = []
        for h in result["hint_matches"]:
            part = f"{h['name']}[{h['type']}]"
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
    merged_by_identity: dict[tuple[str, str], dict[str, Any]] = {}

    for source, matches in (("workspace", workspace_matches), ("global", global_matches)):
        for match in matches:
            candidate = dict(match)
            candidate["source"] = source
            candidate["score"] = float(candidate.get("score", 0.0))
            if source == "workspace":
                candidate["score"] += WORKSPACE_MEMORY_SCORE_BONUS

            key = (candidate.get("name", ""), candidate.get("type", ""))
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

    pinned_by_identity: dict[tuple[str, str], dict[str, Any]] = {
        (item.get("name", ""), item.get("type", "")): item
        for item in workspace_result.get("pinned", [])
    }
    for item in global_result.get("pinned", []):
        pinned_by_identity.setdefault((item.get("name", ""), item.get("type", "")), item)

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
        global_graph_mgr: KnowledgeGraphManager | None = None
        use_global_db = cfg.global_db_enabled and cfg.global_db_path != cfg.db_path
        if use_global_db:
            cfg.ensure_global_db_dir()
            global_db = Database(cfg.global_db_path)
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

        yield {
            "db": db,
            "graph": graph_mgr,
            "global_db": global_db,
            "global_graph": global_graph_mgr,
            "config": cfg,
        }
        db.close()
        if global_db is not None:
            global_db.close()

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
    ) -> str:
        """Create new entities in the knowledge graph.

        Each entity needs: name (str), entityType (str).
        Optional: observations (list[str]), tags (list[str]), metadata (dict).
        Skips duplicates by name.
        """
        graph, cfg, global_graph = _get_ctx(ctx)
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
    ) -> str:
        """Create relations between entities.

        Each relation needs: from (str), to (str), relationType (str).
        Optional: weight (float), tags (list[str]).
        Both entities must exist. Use active voice for relationType.
        """
        graph, _, _ = _get_ctx(ctx)
        try:
            created = graph.create_relations(relations)
            return json.dumps([r.to_dict() for r in created], indent=2)
        except ValueError as e:
            return json.dumps({"error": str(e)})

    # ------------------------------------------------------------------
    # Tool 3: add_observations
    # ------------------------------------------------------------------
    @mcp.tool()
    def add_observations(
        ctx: Context,
        observations: list[dict[str, Any]],
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
    ) -> str:
        """Delete entities (soft delete by default, cascades to relations).

        Set hard=true for permanent deletion.
        """
        graph, _, _ = _get_ctx(ctx)
        count = graph.delete_entities(entityNames, hard=hard)
        return json.dumps({"deleted": count})

    # ------------------------------------------------------------------
    # Tool 5: delete_observations
    # ------------------------------------------------------------------
    @mcp.tool()
    def delete_observations(
        ctx: Context,
        deletions: list[dict[str, Any]],
    ) -> str:
        """Delete specific observations from entities.

        Each item needs: entityName (str), observations (list[str] of content to delete).
        """
        graph, _, _ = _get_ctx(ctx)
        count = graph.delete_observations(deletions)
        return json.dumps({"deleted": count})

    # ------------------------------------------------------------------
    # Tool 6: delete_relations
    # ------------------------------------------------------------------
    @mcp.tool()
    def delete_relations(
        ctx: Context,
        relations: list[dict[str, Any]],
    ) -> str:
        """Delete relations from the knowledge graph.

        Each item needs: from (str), to (str), relationType (str).
        """
        graph, _, _ = _get_ctx(ctx)
        count = graph.delete_relations(relations)
        return json.dumps({"deleted": count})

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
    ) -> str:
        """Read the knowledge graph (compressed by default).

        Filter by tags, entity_types. Set compress=false for full JSON.
        """
        graph_mgr, cfg, _ = _get_ctx(ctx)
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
    ) -> str:
        """Full-text search with BM25 ranking.

        Supports prefix search, phrases ("exact match"), boolean (AND/OR/NOT).
        Filter by tags, entity_types, time_range ([start, end] ISO strings).
        """
        graph_mgr, cfg, _ = _get_ctx(ctx)
        tr = tuple(time_range) if time_range and len(time_range) == 2 else None
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
    ) -> str:
        """Open specific entities by name.

        depth=0: exact entities only.
        depth=1: include direct neighbors via relations.
        depth=2+: BFS expansion.
        """
        graph_mgr, cfg, _ = _get_ctx(ctx)
        kg = graph_mgr.open_nodes(names, depth=depth)
        return json.dumps(kg.to_dict(), indent=2)

    # ------------------------------------------------------------------
    # Tool 10: memory_context (THE KEY TOOL)
    # ------------------------------------------------------------------
    @mcp.tool()
    def memory_context(
        ctx: Context,
        hint: str = "",
        project: str = "",
        limit: int = 10,
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
    ) -> str:
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
    ) -> str:
        """Query activity timeline.

        time_range: relative like "2h", "7d", "30m".
        Or use start/end with ISO datetime strings.
        Filter by actions, entity_name, session_id.
        """
        graph_mgr, _, _ = _get_ctx(ctx)
        entries = graph_mgr.query_timeline(
            time_range=time_range,
            start=start,
            end=end,
            actions=actions,
            entity_name=entity_name,
            session_id=session_id,
            limit=limit,
        )
        return json.dumps(
            [{"action": e.action, "summary": e.summary, "at": e.created_at} for e in entries],
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
    ) -> str:
        """Manage tags. Actions: list, create, delete, tag, untag, cleanup.

        list: show all tags.
        create: new tag (name required, optional description/color/auto_expire_hours).
        delete: remove a user tag (cannot delete system tags).
        tag: apply tag_name to entity_name.
        untag: remove tag_name from entity_name.
        cleanup: remove expired ephemeral items.
        """
        graph_mgr, _, _ = _get_ctx(ctx)

        if action == "list":
            tags = graph_mgr.list_tags()
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
                return json.dumps({"error": "name required"})
            tag = graph_mgr.create_tag(name, description, color, auto_expire_hours)
            return json.dumps({"created": tag.name})

        elif action == "delete":
            if not name:
                return json.dumps({"error": "name required"})
            try:
                ok = graph_mgr.delete_tag(name)
                return json.dumps({"deleted": ok})
            except ValueError as e:
                return json.dumps({"error": str(e)})

        elif action == "tag":
            if not entity_name or not tag_name:
                return json.dumps({"error": "entity_name and tag_name required"})
            ok = graph_mgr.tag_entity(entity_name, tag_name)
            return json.dumps({"tagged": ok})

        elif action == "untag":
            if not entity_name or not tag_name:
                return json.dumps({"error": "entity_name and tag_name required"})
            ok = graph_mgr.untag_entity(entity_name, tag_name)
            return json.dumps({"untagged": ok})

        elif action == "cleanup":
            db = graph_mgr.db
            cleaned = db.cleanup_expired()
            return json.dumps({"cleaned": cleaned})

        return json.dumps({"error": f"Unknown action: {action}"})

    # ------------------------------------------------------------------
    # Tool 14: merge_entities
    # ------------------------------------------------------------------
    @mcp.tool()
    def merge_entities(
        ctx: Context,
        source: str,
        target: str,
        strategy: str = "combine",
    ) -> str:
        """Merge source entity into target. Source is soft-deleted.

        strategy: 'combine' (move all observations) or 'dedupe' (skip duplicates).
        Relations and tags are transferred. Self-relations are removed.
        """
        graph_mgr, _, _ = _get_ctx(ctx)
        result = graph_mgr.merge_entities(source, target, strategy)
        if result:
            return json.dumps(result.to_dict(), indent=2)
        return json.dumps({"error": "Source or target entity not found"})

    # ------------------------------------------------------------------
    # Tool 15: export_graph / import_graph
    # ------------------------------------------------------------------
    @mcp.tool()
    def export_graph(
        ctx: Context,
        format: str = "json",
    ) -> str:
        """Export the full knowledge graph.

        format: 'json' or 'jsonl' (compatible with old @modelcontextprotocol/server-memory).
        """
        graph_mgr, _, _ = _get_ctx(ctx)
        return graph_mgr.export_graph(fmt=format)

    @mcp.tool()
    def import_graph(
        ctx: Context,
        data: str,
    ) -> str:
        """Import knowledge graph data. Auto-detects JSON or JSONL format.

        Compatible with old @modelcontextprotocol/server-memory JSONL files.
        Skips duplicate entities. Skips relations to missing entities.
        """
        graph_mgr, _, _ = _get_ctx(ctx)
        counts = graph_mgr.import_graph(data)
        return json.dumps(counts)

    # ------------------------------------------------------------------
    # Tool 16: memory_stats
    # ------------------------------------------------------------------
    @mcp.tool()
    def memory_stats(ctx: Context) -> str:
        """Get memory statistics: entity/relation/observation counts,
        tag distribution, DB size, orphan entities, deleted items.
        """
        graph_mgr, _, _ = _get_ctx(ctx)
        stats = graph_mgr.memory_stats()
        return json.dumps(stats, indent=2)

    # ------------------------------------------------------------------
    # Tool 17: backup_memory
    # ------------------------------------------------------------------
    @mcp.tool()
    def backup_memory(
        ctx: Context,
        dest_path: str = "",
    ) -> str:
        """Create a backup of the memory database.

        Default destination: ~/.local/share/server-memory/backups/memory-YYYYMMDD-HHMMSS.db
        Provide dest_path to override the backup location.
        """
        graph_mgr, _, _ = _get_ctx(ctx)
        db = graph_mgr.db

        if dest_path:
            backup_path = Path(dest_path)
        else:
            backup_dir = Path(db.db_path).parent / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            backup_path = backup_dir / f"memory-{ts}.db"

        backup_path.parent.mkdir(parents=True, exist_ok=True)
        db.backup(str(backup_path))
        return json.dumps({"backup_path": str(backup_path), "status": "ok"})

    # ------------------------------------------------------------------
    # Tool 18: get_observation_history
    # ------------------------------------------------------------------
    @mcp.tool()
    def get_observation_history(
        ctx: Context,
        entity_name: str,
        content_prefix: str = "",
    ) -> str:
        """Get observation version history for an entity.

        Shows current content, version number, importance, obs_type,
        and all previous versions with timestamps.
        Use content_prefix to filter to specific observations.
        """
        graph_mgr, _, _ = _get_ctx(ctx)
        history = graph_mgr.get_observation_history(entity_name, content_prefix)
        if not history:
            return json.dumps({"error": f"No observations found for '{entity_name}'"})
        return json.dumps(history, indent=2)

    # ------------------------------------------------------------------
    # Tool 19: memory_context_full
    # ------------------------------------------------------------------
    @mcp.tool()
    def memory_context_full(
        ctx: Context,
        project: str = "",
        budget: int = 1000,
    ) -> str:
        """Rich context snapshot for rare deep bootstrap (~500-1500 tokens).

        Returns all pinned entities with full observations, recent activity (last 10),
        and recently changed entities. Use only when compact recall is insufficient
        for a cross-session task. Prefer memory_context for ordinary scoped recall.
        """
        graph_mgr, cfg, global_graph_mgr = _get_ctx(ctx)
        active_project = project or cfg.project

        # Read pinned entities with full data
        pinned_tags = ["pinned"]
        if active_project:
            pinned_tags.append(active_project)
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

        if global_graph_mgr is not None:
            global_kg = global_graph_mgr.read_graph(tags=["pinned", PREFERENCE_TAG], limit=20)
            existing_ids = {e.id for e in kg.entities}
            for entity in global_kg.entities:
                if entity.id not in existing_ids:
                    kg.entities.append(entity)
                    existing_ids.add(entity.id)
            kg.relations.extend(global_kg.relations)

        pinned_ids = _get_pinned_ids(graph_mgr)
        if global_graph_mgr is not None:
            pinned_ids |= _get_pinned_ids(global_graph_mgr)
        output = compress_graph(kg, cfg.compression_level, budget, pinned_ids)

        # Append recent activity
        recent = graph_mgr.query_timeline(time_range="24h", limit=10)
        if global_graph_mgr is not None:
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
