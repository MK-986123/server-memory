"""Knowledge graph manager: CRUD, search, traversal, activity tracking."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any

from .db import Database
from .embeddings import EmbeddingEngine, embedding_bucket_probes, embedding_buckets
from .models import (
    PROTECTED_OBS_TYPES,
    ActivityEntry,
    Entity,
    KnowledgeGraph,
    Observation,
    Relation,
    Tag,
)

logger = logging.getLogger(__name__)

_embedding_backfill_lock = threading.Lock()
_active_embedding_backfills: set[tuple[str, str]] = set()
MAX_ENTITY_EMBEDDING_CANDIDATES = 1_000
MAX_OBSERVATION_EMBEDDING_CANDIDATES = 4_000
USE_EMBEDDING_BUCKET_FILTER = True


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _embedding_bucket_filter(alias: str, embedding: bytes) -> tuple[str, list[int]]:
    """Build a bounded multi-probe LSH predicate and its parameters."""
    if not USE_EMBEDDING_BUCKET_FILTER:
        return "1", []
    clauses: list[str] = []
    params: list[int] = []
    for index, probes in enumerate(embedding_bucket_probes(embedding)):
        placeholders = ",".join("?" for _ in probes)
        clauses.append(f"{alias}.bucket{index} IN ({placeholders})")
        params.extend(probes)
    return "(" + " OR ".join(clauses) + ")", params


class KnowledgeGraphManager:
    """All graph operations backed by SQLite."""

    GENERIC_MEMORY_HINT_TOKENS = {
        "note",
        "notes",
        "project",
        "projects",
        "doc",
        "docs",
        "file",
        "files",
        "module",
        "modules",
        "config",
        "configs",
        "setting",
        "settings",
        "task",
        "tasks",
        "issue",
        "issues",
        "bug",
        "bugs",
    }

    MEMORY_SNIPPET_BUDGET_CHARS = 220
    MEMORY_SNIPPET_MIN_CHARS = 40
    MEMORY_MAX_SNIPPETS = 2
    MEMORY_ACTIVITY_LOOKBACK = 20
    MEMORY_ACCESS_STALE_DAYS = 30
    MEMORY_LOW_CONFIDENCE_STALE_DAYS = 30

    def __init__(
        self,
        db: Database,
        session_id: str = "",
        embedding_engine: EmbeddingEngine | None = None,
        project: str = "",
        dedup_threshold: float = 0.92,
        write_embedding_budget_ms: int = 10000,
    ):
        self.db = db
        self.session_id = session_id
        self.embedding_engine = embedding_engine
        self.project = project
        self.dedup_threshold = dedup_threshold
        self.write_embedding_budget_ms = max(int(write_embedding_budget_ms), 0)
        self._embeddings_synced = False

    ACCESS_TOUCH_INTERVAL_HOURS = 24

    # ------------------------------------------------------------------
    # Entity CRUD
    # ------------------------------------------------------------------

    def create_entities(
        self,
        entities: list[dict[str, Any]],
    ) -> list[Entity]:
        """Create entities, skipping duplicates. Returns newly created ones."""
        deadline = self._write_embedding_deadline()
        prepared_entities = []
        backfill_needed = False
        for e in entities:
            name = e["name"]
            etype = e.get("entityType", e.get("entity_type", ""))
            observations = list(e.get("observations", []))
            entity_embedding, entity_timed_out = self._prepare_embedding(
                f"{name} {etype}".strip(),
                deadline=deadline,
            )
            observation_embeddings, observation_timed_out = self._prepare_embeddings(
                observations,
                deadline=deadline,
            )
            backfill_needed = backfill_needed or entity_timed_out or observation_timed_out
            prepared_entities.append(
                {
                    "entity": e,
                    "entity_embedding": entity_embedding,
                    "observation_embeddings": observation_embeddings,
                }
            )

        created = []
        with self.db.transaction() as cx:
            for idx, prepared in enumerate(prepared_entities):
                e = prepared["entity"]
                savepoint = f"create_entity_{idx}"
                cx.execute(f"SAVEPOINT {savepoint}")
                name = e["name"]
                etype = e.get("entityType", e.get("entity_type", ""))
                metadata = json.dumps(e.get("metadata", {}))
                tags = e.get("tags", [])
                observations = e.get("observations", [])
                try:
                    cur = cx.execute(
                        "INSERT INTO entities (name, entity_type, metadata_json) VALUES (?, ?, ?)",
                        (name, etype, metadata),
                    )
                    eid = cur.lastrowid
                    assert eid is not None
                    # Apply tags
                    for tag_name in tags:
                        self._apply_tag_to_entity(eid, tag_name)
                    # Add observations inline
                    obs_ids = []
                    for obs_text, obs_embedding in zip(
                        observations,
                        prepared["observation_embeddings"],
                        strict=True,
                    ):
                        obs_cur = cx.execute(
                            "INSERT INTO observations (entity_id, content) VALUES (?, ?)",
                            (eid, obs_text),
                        )
                        obs_ids.append((obs_cur.lastrowid, obs_text, obs_embedding))
                    # Embed entity and observations
                    self._embed_entity(
                        eid,
                        name,
                        etype,
                        embedding=prepared["entity_embedding"],
                        allow_fallback=False,
                    )
                    for oid, obs_text, obs_embedding in obs_ids:
                        self._embed_observation(
                            oid,
                            obs_text,
                            embedding=obs_embedding,
                            allow_fallback=False,
                        )
                    ent = self._load_entity(eid)
                    if ent:
                        created.append(ent)
                    cx.execute(f"RELEASE SAVEPOINT {savepoint}")
                except sqlite3.IntegrityError:
                    cx.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                    cx.execute(f"RELEASE SAVEPOINT {savepoint}")
                    deleted = cx.execute(
                        "SELECT id FROM entities WHERE name = ? AND deleted_at IS NOT NULL",
                        (name,),
                    ).fetchone()
                    if deleted is not None:
                        raise ValueError(
                            f"Entity '{name}' exists but is soft-deleted; "
                            "restore it with restore_entities or hard-delete it before recreating"
                        ) from None
                    # Active duplicate name — skip (legacy create_entities behavior)
                    continue
                except Exception:
                    cx.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                    cx.execute(f"RELEASE SAVEPOINT {savepoint}")
                    raise
        if backfill_needed:
            self._embeddings_synced = False
            self._schedule_embedding_backfill()
        return created

    def get_entity_by_name(self, name: str) -> Entity | None:
        row = self.db.cx.execute(
            "SELECT * FROM entities WHERE name = ? AND deleted_at IS NULL", (name,)
        ).fetchone()
        if not row:
            return None
        return self._load_entity(row["id"])

    def delete_entities(self, names: list[str], hard: bool = False) -> int:
        """Soft-delete by default. Hard delete permanently removes."""
        count = 0
        with self.db.transaction() as cx:
            for name in names:
                row = cx.execute("SELECT id FROM entities WHERE name = ?", (name,)).fetchone()
                if not row:
                    continue
                eid = row["id"]
                if hard:
                    cx.execute("DELETE FROM entities WHERE id = ?", (eid,))
                else:
                    cx.execute(
                        "UPDATE entities SET deleted_at = ? WHERE id = ?",
                        (_now_iso(), eid),
                    )
                    # Cascade soft-delete to relations
                    cx.execute(
                        "UPDATE relations SET deleted_at = ? "
                        "WHERE (from_entity_id = ? OR to_entity_id = ?) AND deleted_at IS NULL",
                        (_now_iso(), eid, eid),
                    )
                count += 1
        return count

    def restore_entities(self, names: list[str]) -> int:
        """Restore soft-deleted entities."""
        count = 0
        with self.db.transaction() as cx:
            for name in names:
                res = cx.execute(
                    "UPDATE entities SET deleted_at = NULL, updated_at = ? "
                    "WHERE name = ? AND deleted_at IS NOT NULL",
                    (_now_iso(), name),
                )
                if res.rowcount:
                    count += res.rowcount
                    # Restore associated relations where both endpoints are alive
                    row = cx.execute("SELECT id FROM entities WHERE name = ?", (name,)).fetchone()
                    if row:
                        eid = row["id"]
                        cx.execute(
                            "UPDATE relations SET deleted_at = NULL "
                            "WHERE deleted_at IS NOT NULL "
                            "  AND (from_entity_id = ? OR to_entity_id = ?) "
                            "  AND from_entity_id IN (SELECT id FROM entities "
                            "WHERE deleted_at IS NULL) "
                            "  AND to_entity_id IN (SELECT id FROM entities "
                            "WHERE deleted_at IS NULL)",
                            (eid, eid),
                        )
        return count

    def list_deleted_entities(self, limit: int = 100) -> list[Entity]:
        """List recoverable soft-deleted entities, newest deletions first."""
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        rows = self.db.cx.execute(
            "SELECT id FROM entities WHERE deleted_at IS NOT NULL "
            "ORDER BY deleted_at DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [entity for row in rows if (entity := self._load_entity(row["id"])) is not None]

    # ------------------------------------------------------------------
    # Relation CRUD
    # ------------------------------------------------------------------

    def create_relations(self, relations: list[dict[str, Any]]) -> list[Relation]:
        """Create relations. Errors on missing entities."""
        created = []
        with self.db.transaction() as cx:
            for idx, r in enumerate(relations):
                savepoint = f"create_relation_{idx}"
                cx.execute(f"SAVEPOINT {savepoint}")
                from_name = r["from"]
                to_name = r["to"]
                rel_type = r["relationType"]
                weight = r.get("weight", 1.0)
                metadata = json.dumps(r.get("metadata", {}))
                tags = r.get("tags", [])
                from_row = cx.execute(
                    "SELECT id FROM entities WHERE name = ? AND deleted_at IS NULL",
                    (from_name,),
                ).fetchone()
                to_row = cx.execute(
                    "SELECT id FROM entities WHERE name = ? AND deleted_at IS NULL",
                    (to_name,),
                ).fetchone()
                if not from_row:
                    raise ValueError(f"Entity not found: {from_name}")
                if not to_row:
                    raise ValueError(f"Entity not found: {to_name}")
                try:
                    cur = cx.execute(
                        "INSERT INTO relations (from_entity_id, to_entity_id, "
                        "relation_type, weight, metadata_json) VALUES (?, ?, ?, ?, ?)",
                        (from_row["id"], to_row["id"], rel_type, weight, metadata),
                    )
                    rid = cur.lastrowid
                    assert rid is not None
                    for tag_name in tags:
                        self._apply_tag_to_relation(rid, tag_name)
                    rel = Relation(
                        id=rid,
                        from_entity_id=from_row["id"],
                        to_entity_id=to_row["id"],
                        relation_type=rel_type,
                        weight=weight,
                        from_name=from_name,
                        to_name=to_name,
                    )
                    created.append(rel)
                    cx.execute(f"RELEASE SAVEPOINT {savepoint}")
                except sqlite3.IntegrityError:
                    cx.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                    cx.execute(f"RELEASE SAVEPOINT {savepoint}")
                    # Duplicate — skip
                    continue
                except Exception:
                    cx.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                    cx.execute(f"RELEASE SAVEPOINT {savepoint}")
                    raise
        return created

    def delete_relations(self, relations: list[dict[str, Any]], hard: bool = False) -> int:
        count = 0
        with self.db.transaction() as cx:
            for r in relations:
                from_name = r["from"]
                to_name = r["to"]
                rel_type = r["relationType"]
                if hard:
                    res = cx.execute(
                        """
                        DELETE FROM relations WHERE
                            from_entity_id = (SELECT id FROM entities WHERE name = ?) AND
                            to_entity_id = (SELECT id FROM entities WHERE name = ?) AND
                            relation_type = ?
                        """,
                        (from_name, to_name, rel_type),
                    )
                else:
                    res = cx.execute(
                        """
                        UPDATE relations SET deleted_at = ?
                        WHERE deleted_at IS NULL AND
                            from_entity_id = (SELECT id FROM entities WHERE name = ?) AND
                            to_entity_id = (SELECT id FROM entities WHERE name = ?) AND
                            relation_type = ?
                        """,
                        (_now_iso(), from_name, to_name, rel_type),
                    )
                count += res.rowcount
        return count

    # ------------------------------------------------------------------
    # Observation CRUD
    # ------------------------------------------------------------------

    def add_observations(self, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add observations to existing entities. Returns list of {entityName, addedObservations}.

        Deduplicates by exact match AND semantic similarity (cosine > threshold).
        Supports importance (0.0-1.0) and obs_type fields.
        """
        deadline = self._write_embedding_deadline()
        prepared_batches = []
        backfill_needed = False
        for observation in observations:
            embeddings, timed_out = self._prepare_embeddings(
                observation["contents"],
                deadline=deadline,
            )
            prepared_batches.append(
                {
                    "payload": observation,
                    "embeddings": embeddings,
                }
            )
            backfill_needed = backfill_needed or timed_out

        results = []
        with self.db.transaction() as cx:
            for prepared in prepared_batches:
                o = prepared["payload"]
                entity_name = o["entityName"]
                contents = o["contents"]
                source = o.get("source", "")
                confidence = o.get("confidence", 1.0)
                importance = o.get("importance", 0.5)
                obs_type = o.get("obs_type", "")
                tags = o.get("tags", [])

                row = cx.execute(
                    "SELECT id FROM entities WHERE name = ? AND deleted_at IS NULL",
                    (entity_name,),
                ).fetchone()
                if not row:
                    raise ValueError(f"Entity not found: {entity_name}")
                eid = row["id"]

                # Existing observation contents for exact dedup
                existing = {
                    r["content"]
                    for r in cx.execute(
                        "SELECT content FROM observations WHERE entity_id = ? "
                        "AND deleted_at IS NULL",
                        (eid,),
                    ).fetchall()
                }

                # Load existing embeddings for semantic dedup
                existing_embeddings: list[bytes] = []
                if (
                    self.embedding_engine
                    and self.embedding_engine.is_available()
                    and self.dedup_threshold < 1.0
                ):
                    obs_rows = cx.execute(
                        "SELECT oe.embedding FROM observation_embeddings oe "
                        "JOIN observations o ON o.id = oe.observation_id "
                        "WHERE o.entity_id = ? AND o.deleted_at IS NULL "
                        "AND oe.model_name = ?",
                        (eid, self.embedding_engine.model_name),
                    ).fetchall()
                    existing_embeddings = [r["embedding"] for r in obs_rows]

                added = []
                skipped_semantic = 0
                semantic_budget_exhausted = False
                for content, new_emb in zip(
                    contents,
                    prepared["embeddings"],
                    strict=True,
                ):
                    if content in existing:
                        continue

                    # Semantic dedup: check cosine similarity against existing
                    if (
                        existing_embeddings
                        and self.embedding_engine
                        and new_emb
                        and not semantic_budget_exhausted
                    ):
                        if self._write_embedding_budget_exhausted(deadline):
                            semantic_budget_exhausted = True
                        else:
                            is_duplicate = False
                            for ex in existing_embeddings:
                                if self._write_embedding_budget_exhausted(deadline):
                                    semantic_budget_exhausted = True
                                    break
                                if (
                                    self.embedding_engine.cosine_similarity(new_emb, ex)
                                    >= self.dedup_threshold
                                ):
                                    is_duplicate = True
                                    break
                            if is_duplicate:
                                skipped_semantic += 1
                                continue

                    cur = cx.execute(
                        "INSERT INTO observations (entity_id, content, source, confidence, "
                        "importance, obs_type) VALUES (?, ?, ?, ?, ?, ?)",
                        (eid, content, source, confidence, importance, obs_type),
                    )
                    oid = cur.lastrowid
                    assert oid is not None
                    for tag_name in tags:
                        self._apply_tag_to_observation(oid, tag_name)
                    self._embed_observation(
                        oid, content, embedding=new_emb, allow_fallback=False
                    )
                    if new_emb:
                        existing_embeddings.append(new_emb)
                    added.append(content)

                # Touch entity updated_at
                cx.execute("UPDATE entities SET updated_at = ? WHERE id = ?", (_now_iso(), eid))
                result_entry: dict[str, Any] = {
                    "entityName": entity_name,
                    "addedObservations": added,
                }
                if skipped_semantic > 0:
                    result_entry["skippedSemantic"] = skipped_semantic
                results.append(result_entry)
        if backfill_needed:
            self._embeddings_synced = False
            self._schedule_embedding_backfill()
        return results

    def delete_observations(self, deletions: list[dict[str, Any]], hard: bool = False) -> int:
        count = 0
        with self.db.transaction() as cx:
            for d in deletions:
                entity_name = d["entityName"]
                obs_contents = d["observations"]
                row = cx.execute(
                    "SELECT id FROM entities WHERE name = ?", (entity_name,)
                ).fetchone()
                if not row:
                    continue
                eid = row["id"]
                for content in obs_contents:
                    if hard:
                        res = cx.execute(
                            "DELETE FROM observations WHERE entity_id = ? AND content = ?",
                            (eid, content),
                        )
                    else:
                        res = cx.execute(
                            "UPDATE observations SET deleted_at = ? "
                            "WHERE entity_id = ? AND content = ? AND deleted_at IS NULL",
                            (_now_iso(), eid, content),
                        )
                    count += res.rowcount
        return count

    def update_observation(self, entity_name: str, old_content: str, new_content: str) -> bool:
        """Update observation content, recording history."""
        with self.db.transaction() as cx:
            row = cx.execute(
                "SELECT id FROM entities WHERE name = ? AND deleted_at IS NULL",
                (entity_name,),
            ).fetchone()
            if not row:
                return False
            eid = row["id"]
            obs_row = cx.execute(
                "SELECT id, content, version FROM observations "
                "WHERE entity_id = ? AND content = ? AND deleted_at IS NULL",
                (eid, old_content),
            ).fetchone()
            if not obs_row:
                return False
            oid = obs_row["id"]
            old_ver = obs_row["version"]
            new_ver = old_ver + 1
            # Record history
            cx.execute(
                "INSERT INTO observation_history (observation_id, content, version) "
                "VALUES (?, ?, ?)",
                (oid, old_content, old_ver),
            )
            # Update
            cx.execute(
                "UPDATE observations SET content = ?, version = ?, updated_at = ? WHERE id = ?",
                (new_content, new_ver, _now_iso(), oid),
            )
        return True

    def get_observation_history(
        self, entity_name: str, content_prefix: str = ""
    ) -> list[dict[str, Any]]:
        """Get observation version history for an entity."""
        cx = self.db.cx
        row = cx.execute(
            "SELECT id FROM entities WHERE name = ? AND deleted_at IS NULL",
            (entity_name,),
        ).fetchone()
        if not row:
            return []
        eid = row["id"]

        if content_prefix:
            obs_rows = cx.execute(
                "SELECT id, content, version, importance, obs_type, updated_at FROM observations "
                "WHERE entity_id = ? AND deleted_at IS NULL AND content LIKE ?",
                (eid, f"{content_prefix}%"),
            ).fetchall()
        else:
            obs_rows = cx.execute(
                "SELECT id, content, version, importance, obs_type, updated_at FROM observations "
                "WHERE entity_id = ? AND deleted_at IS NULL",
                (eid,),
            ).fetchall()

        results = []
        for obs in obs_rows:
            history = cx.execute(
                "SELECT content, version, changed_at FROM observation_history "
                "WHERE observation_id = ? ORDER BY version DESC",
                (obs["id"],),
            ).fetchall()
            results.append(
                {
                    "content": obs["content"],
                    "version": obs["version"],
                    "importance": obs["importance"],
                    "obs_type": obs["obs_type"],
                    "updated_at": obs["updated_at"],
                    "history": [
                        {"content": h["content"], "version": h["version"], "at": h["changed_at"]}
                        for h in history
                    ],
                }
            )
        return results

    # ------------------------------------------------------------------
    # Read / Query
    # ------------------------------------------------------------------

    def read_graph(
        self,
        tags: list[str] | None = None,
        entity_types: list[str] | None = None,
        limit: int = 0,
        include_deleted: bool = False,
    ) -> KnowledgeGraph:
        """Read the full graph with optional filters."""
        cx = self.db.cx
        where = [] if include_deleted else ["e.deleted_at IS NULL"]
        params: list[Any] = []

        if entity_types:
            placeholders = ",".join("?" for _ in entity_types)
            where.append(f"e.entity_type IN ({placeholders})")
            params.extend(entity_types)

        if tags:
            placeholders = ",".join("?" for _ in tags)
            where.append(
                f"e.id IN (SELECT et.entity_id FROM entity_tags et JOIN tags t ON "
                f"t.id = et.tag_id WHERE t.name IN ({placeholders}))"
            )
            params.extend(tags)

        where_clause = " AND ".join(where) if where else "1"
        limit_clause = f"LIMIT {limit}" if limit > 0 else ""

        rows = cx.execute(
            f"SELECT e.* FROM entities e WHERE {where_clause} ORDER BY e.updated_at DESC "
            f"{limit_clause}",
            params,
        ).fetchall()

        entity_ids = {row["id"] for row in rows}
        entities = self._load_entities_batch(entity_ids, include_deleted=include_deleted)
        relations = self._load_relations_for(entity_ids, include_deleted)

        return KnowledgeGraph(entities=entities, relations=relations)

    def open_nodes(self, names: list[str], depth: int = 0) -> KnowledgeGraph:
        """Open specific nodes. depth=0 means exact names only, depth=1 includes neighbors, etc."""
        cx = self.db.cx
        # Start with named entities
        entity_ids: set[int] = set()
        for name in names:
            row = cx.execute(
                "SELECT id FROM entities WHERE name = ? AND deleted_at IS NULL",
                (name,),
            ).fetchone()
            if row:
                entity_ids.add(row["id"])

        # BFS for depth > 0
        if depth > 0:
            entity_ids = self._bfs_expand(entity_ids, depth)

        entities = self._load_entities_batch(entity_ids)
        relations = self._load_relations_for(entity_ids, include_deleted=False)
        return KnowledgeGraph(entities=entities, relations=relations)

    # ------------------------------------------------------------------
    # FTS5 Search
    # ------------------------------------------------------------------

    def search_fts(
        self,
        query: str,
        tags: list[str] | None = None,
        entity_types: list[str] | None = None,
        time_range: tuple[str, str] | None = None,
        limit: int = 20,
    ) -> KnowledgeGraph:
        """Hybrid search: FTS5 + optional embedding-based semantic search."""
        cx = self.db.cx
        sql_limit = limit if limit > 0 else -1
        self.last_search_diagnostics: list[str] = []
        query_norm = self._normalize_memory_text(query)

        # --- FTS5 path (keyword search) ---
        fts_scores: dict[int, float] = {}
        fts_query = self._sanitize_fts_query(query)
        exact_name_ids: set[int] = set()

        # Search entity names/types
        try:
            rows = cx.execute(
                """
                SELECT e.id, rank FROM fts_entities fe
                JOIN entities e ON e.id = fe.rowid
                WHERE fts_entities MATCH ? AND e.deleted_at IS NULL
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, sql_limit),
            ).fetchall()
            for row in rows:
                # BM25 rank is negative (lower=better), normalize later
                fts_scores[row["id"]] = row["rank"]
        except sqlite3.OperationalError as exc:
            self.last_search_diagnostics.append("entity_fts_fallback")
            logger.warning("Entity FTS unavailable; using fallback retrieval: %s", exc)

        if query_norm:
            exact_name_rows = cx.execute(
                "SELECT id FROM entities WHERE lower(name) = ? AND deleted_at IS NULL",
                (query.lower(),),
            ).fetchall()
            exact_name_ids = {row["id"] for row in exact_name_rows}

        # Search observation content (only for active parent entities)
        try:
            rows = cx.execute(
                """
                SELECT o.entity_id, rank FROM fts_observations fo
                JOIN observations o ON o.id = fo.rowid
                JOIN entities e ON e.id = o.entity_id
                WHERE fts_observations MATCH ?
                  AND o.deleted_at IS NULL
                  AND e.deleted_at IS NULL
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, sql_limit),
            ).fetchall()
            for row in rows:
                eid = row["entity_id"]
                # Keep best (most negative) rank per entity
                if eid not in fts_scores or row["rank"] < fts_scores[eid]:
                    fts_scores[eid] = row["rank"]
        except sqlite3.OperationalError as exc:
            self.last_search_diagnostics.append("observation_fts_fallback")
            logger.warning("Observation FTS unavailable; using fallback retrieval: %s", exc)

        # --- Embedding path (semantic search) ---
        embedding_scores: dict[int, float] = {}
        if self.embedding_engine and self.embedding_engine.is_available():
            self._ensure_embeddings()
            query_emb = self.embedding_engine.embed_text(query)
            if query_emb:
                entity_bucket_sql, entity_bucket_params = _embedding_bucket_filter(
                    "ee", query_emb
                )
                # Score entities by embedding similarity
                emb_rows = cx.execute(
                    f"""
                    SELECT ee.entity_id, ee.embedding FROM entity_embeddings ee
                    JOIN entities e ON e.id = ee.entity_id
                    WHERE e.deleted_at IS NULL
                      AND ee.model_name = ? AND ee.dimension = ?
                      AND {entity_bucket_sql}
                    ORDER BY ee.entity_id DESC
                    LIMIT ?
                    """,
                    (
                        self.embedding_engine.model_name,
                        len(query_emb) // 4,
                        *entity_bucket_params,
                        MAX_ENTITY_EMBEDDING_CANDIDATES,
                    ),
                ).fetchall()
                for er in emb_rows:
                    sim = self.embedding_engine.cosine_similarity(query_emb, er["embedding"])
                    embedding_scores[er["entity_id"]] = max(
                        embedding_scores.get(er["entity_id"], 0.0), sim
                    )
                # Also score via observation embeddings for active entities only
                obs_bucket_sql, obs_bucket_params = _embedding_bucket_filter("oe", query_emb)
                obs_emb_rows = cx.execute(
                    f"""
                    SELECT oe.observation_id, oe.embedding, o.entity_id
                    FROM observation_embeddings oe
                    JOIN observations o ON o.id = oe.observation_id
                    JOIN entities e ON e.id = o.entity_id
                    WHERE o.deleted_at IS NULL
                      AND e.deleted_at IS NULL
                      AND oe.model_name = ? AND oe.dimension = ?
                      AND {obs_bucket_sql}
                    ORDER BY oe.observation_id DESC
                    LIMIT ?
                    """,
                    (
                        self.embedding_engine.model_name,
                        len(query_emb) // 4,
                        *obs_bucket_params,
                        MAX_OBSERVATION_EMBEDDING_CANDIDATES,
                    ),
                ).fetchall()
                for oer in obs_emb_rows:
                    sim = self.embedding_engine.cosine_similarity(query_emb, oer["embedding"])
                    eid = oer["entity_id"]
                    embedding_scores[eid] = max(embedding_scores.get(eid, 0.0), sim)

        # --- Hybrid merge ---
        all_eids = set(fts_scores.keys()) | set(embedding_scores.keys())
        ordered_eids: list[int] = []

        # Fuzzy fallback if no results from either path
        if not all_eids:
            ordered_eids = self._fuzzy_search(query, sql_limit)
            all_eids = set(ordered_eids)

        if all_eids and (fts_scores or embedding_scores):
            # Normalize BM25 scores to 0-1 (rank is negative, more negative = better)
            if fts_scores:
                min_rank = min(fts_scores.values())
                max_rank = max(fts_scores.values())
                rank_range = max_rank - min_rank if max_rank != min_rank else 1.0
                norm_fts = {
                    eid: (max_rank - score) / rank_range for eid, score in fts_scores.items()
                }
            else:
                norm_fts = {}

            # Compute recency bonus
            recency_bonus: dict[int, float] = {}
            if all_eids:
                placeholders = ",".join("?" for _ in all_eids)
                rows = cx.execute(
                    f"SELECT id, updated_at FROM entities WHERE id IN ({placeholders}) "
                    "AND deleted_at IS NULL",
                    list(all_eids),
                ).fetchall()
                now = datetime.now(timezone.utc)
                for r in rows:
                    try:
                        updated = datetime.fromisoformat(r["updated_at"].replace("Z", "+00:00"))
                        days = max((now - updated).total_seconds() / 86400.0, 0)
                    except (ValueError, AttributeError):
                        days = 365.0
                    recency_bonus[r["id"]] = 1.0 / (1.0 + days)

            # Compute hybrid scores
            scored: list[tuple[int, float]] = []
            for eid in all_eids:
                fts_s = norm_fts.get(eid, 0.0)
                emb_s = embedding_scores.get(eid, 0.0)
                rec_s = recency_bonus.get(eid, 0.0)
                exact_s = 2.5 if eid in exact_name_ids else 0.0
                hybrid = exact_s + 0.4 * fts_s + 0.5 * emb_s + 0.1 * rec_s
                scored.append((eid, hybrid))

            scored.sort(key=lambda x: x[1], reverse=True)
            ordered_eids = [eid for eid, _ in (scored[:limit] if limit > 0 else scored)]
            all_eids = set(ordered_eids)
        elif all_eids and not ordered_eids:
            ordered_eids = list(all_eids)

        # Apply filters
        if entity_types and all_eids:
            placeholders = ",".join("?" for _ in entity_types)
            entity_placeholders = ",".join("?" for _ in all_eids)
            rows = cx.execute(
                f"SELECT id FROM entities WHERE id IN ({entity_placeholders}) "
                f"AND entity_type IN ({placeholders}) "
                "AND deleted_at IS NULL",
                [*all_eids, *entity_types],
            ).fetchall()
            all_eids = {row["id"] for row in rows}
            ordered_eids = [eid for eid in ordered_eids if eid in all_eids]

        if tags and all_eids:
            placeholders_tags = ",".join("?" for _ in tags)
            rows = cx.execute(
                f"""
                SELECT DISTINCT et.entity_id FROM entity_tags et
                JOIN tags t ON t.id = et.tag_id
                WHERE et.entity_id IN ({",".join("?" for _ in all_eids)})
                AND t.name IN ({placeholders_tags})
                """,
                [*all_eids, *tags],
            ).fetchall()
            all_eids = {row["entity_id"] for row in rows}
            ordered_eids = [eid for eid in ordered_eids if eid in all_eids]

        if time_range and all_eids:
            start, end = time_range
            entity_placeholders = ",".join("?" for _ in all_eids)
            rows = cx.execute(
                f"SELECT id FROM entities WHERE id IN ({entity_placeholders}) "
                "AND created_at BETWEEN ? AND ? AND deleted_at IS NULL",
                [*all_eids, start, end],
            ).fetchall()
            all_eids = {row["id"] for row in rows}
            ordered_eids = [eid for eid in ordered_eids if eid in all_eids]

        entities = self._load_entities_batch(ordered_eids or all_eids)
        relations = self._load_relations_for(all_eids, include_deleted=False)
        return KnowledgeGraph(entities=entities, relations=relations)

    # ------------------------------------------------------------------
    # Tag Management
    # ------------------------------------------------------------------

    def list_tags(self) -> list[Tag]:
        rows = self.db.cx.execute("SELECT * FROM tags ORDER BY name").fetchall()
        return [
            Tag(
                id=r["id"],
                name=r["name"],
                description=r["description"],
                color=r["color"],
                is_system=bool(r["is_system"]),
                auto_expire_hours=r["auto_expire_hours"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def create_tag(
        self,
        name: str,
        description: str = "",
        color: str = "",
        auto_expire_hours: int | None = None,
    ) -> Tag:
        with self.db.transaction() as cx:
            cur = cx.execute(
                "INSERT INTO tags (name, description, color, is_system, "
                "auto_expire_hours) VALUES (?, ?, ?, 0, ?)",
                (name, description, color, auto_expire_hours),
            )
            tid = cur.lastrowid
            assert tid is not None
        return Tag(
            id=tid,
            name=name,
            description=description,
            color=color,
            auto_expire_hours=auto_expire_hours,
        )

    def delete_tag(self, name: str) -> bool:
        with self.db.transaction() as cx:
            row = cx.execute("SELECT id, is_system FROM tags WHERE name = ?", (name,)).fetchone()
            if not row:
                return False
            if row["is_system"]:
                raise ValueError(f"Cannot delete system tag: {name}")
            cx.execute("DELETE FROM entity_tags WHERE tag_id = ?", (row["id"],))
            cx.execute("DELETE FROM observation_tags WHERE tag_id = ?", (row["id"],))
            cx.execute("DELETE FROM relation_tags WHERE tag_id = ?", (row["id"],))
            cx.execute("DELETE FROM tags WHERE id = ?", (row["id"],))
        return True

    def tag_entity(self, entity_name: str, tag_name: str) -> bool:
        with self.db.transaction() as cx:
            row = cx.execute(
                "SELECT id FROM entities WHERE name = ? AND deleted_at IS NULL",
                (entity_name,),
            ).fetchone()
            if not row:
                return False
            return self._apply_tag_to_entity(row["id"], tag_name)

    def untag_entity(self, entity_name: str, tag_name: str) -> bool:
        with self.db.transaction() as cx:
            row = cx.execute(
                "SELECT id FROM entities WHERE name = ? AND deleted_at IS NULL", (entity_name,)
            ).fetchone()
            if not row:
                return False
            tag_row = cx.execute("SELECT id FROM tags WHERE name = ?", (tag_name,)).fetchone()
            if not tag_row:
                return False
            res = cx.execute(
                "DELETE FROM entity_tags WHERE entity_id = ? AND tag_id = ?",
                (row["id"], tag_row["id"]),
            )
        return res.rowcount > 0

    # ------------------------------------------------------------------
    # Activity Logging
    # ------------------------------------------------------------------

    def log_activity(
        self,
        action: str,
        summary: str = "",
        entity_names: list[str] | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        auto_create_entities: bool = True,
    ) -> ActivityEntry:
        """Log an activity. Auto-creates missing entities if requested."""
        with self.db.transaction() as cx:
            entity_ids: list[int] = []

            if entity_names:
                for name in entity_names:
                    row = cx.execute(
                        "SELECT id FROM entities WHERE name = ? AND deleted_at IS NULL",
                        (name,),
                    ).fetchone()
                    if row:
                        entity_ids.append(row["id"])
                    elif auto_create_entities:
                        cur = cx.execute(
                            "INSERT INTO entities (name, entity_type) VALUES (?, ?)",
                            (name, "auto"),
                        )
                        eid = cur.lastrowid
                        assert eid is not None
                        entity_ids.append(eid)

            # Auto-tag by action type
            action_tag_map = {
                "file_changed": "recent-change",
                "bug_fixed": "debugging",
                "decision_made": "architecture",
            }
            auto_tags = list(tags or [])
            if action in action_tag_map:
                auto_tags.append(action_tag_map[action])

            cur = cx.execute(
                "INSERT INTO activity_log (session_id, action, summary, "
                "entity_ids_json, tags_json, metadata_json) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    self.session_id,
                    action,
                    summary,
                    json.dumps(entity_ids),
                    json.dumps(auto_tags),
                    json.dumps(metadata or {}),
                ),
            )
            aid = cur.lastrowid
            assert aid is not None
        return ActivityEntry(
            id=aid,
            session_id=self.session_id,
            action=action,
            summary=summary,
            entity_ids_json=json.dumps(entity_ids),
            tags_json=json.dumps(auto_tags),
            created_at=_now_iso(),
        )

    def query_timeline(
        self,
        time_range: str | None = None,
        start: str | None = None,
        end: str | None = None,
        actions: list[str] | None = None,
        entity_name: str | None = None,
        session_id: str | None = None,
        limit: int = 50,
    ) -> list[ActivityEntry]:
        """Query activity timeline with flexible time filters."""
        cx = self.db.cx
        where = []
        params: list[Any] = []

        if time_range:
            # Parse relative time like "2h", "7d"
            match = re.match(r"(\d+)([hHdDmM])", time_range)
            if match:
                amount = int(match.group(1))
                unit = match.group(2).lower()
                if unit == "h":
                    modifier = f"-{amount} hours"
                elif unit == "d":
                    modifier = f"-{amount} days"
                elif unit == "m":
                    modifier = f"-{amount} minutes"
                else:
                    modifier = f"-{amount} hours"
                where.append("created_at > strftime('%Y-%m-%dT%H:%M:%fZ', 'now', ?)")
                params.append(modifier)

        if start:
            where.append("created_at >= ?")
            params.append(start)
        if end:
            where.append("created_at <= ?")
            params.append(end)

        if actions:
            placeholders = ",".join("?" for _ in actions)
            where.append(f"action IN ({placeholders})")
            params.extend(actions)

        if session_id:
            where.append("session_id = ?")
            params.append(session_id)

        if entity_name:
            row = cx.execute("SELECT id FROM entities WHERE name = ?", (entity_name,)).fetchone()
            if row:
                eid = row["id"]
                where.append(
                    "EXISTS (SELECT 1 FROM json_each(activity_log.entity_ids_json) "
                    "WHERE json_each.value = ?)"
                )
                params.append(eid)

        where_clause = " AND ".join(where) if where else "1"
        rows = cx.execute(
            f"SELECT * FROM activity_log WHERE {where_clause} ORDER BY created_at DESC LIMIT ?",
            [*params, limit],
        ).fetchall()

        return [
            ActivityEntry(
                id=r["id"],
                session_id=r["session_id"],
                action=r["action"],
                summary=r["summary"],
                entity_ids_json=r["entity_ids_json"],
                tags_json=r["tags_json"],
                metadata_json=r["metadata_json"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Memory Context (lightweight aggregation)
    # ------------------------------------------------------------------

    def memory_context(self, hint: str = "", limit: int = 10, project: str = "") -> dict[str, Any]:
        """The key tool: lightweight context for every-turn use.

        If project is set (or self.project), scopes to entities tagged with that project.
        """
        cx = self.db.cx
        result: dict[str, Any] = {}
        active_project = project or self.project
        query_norm = self._normalize_memory_text(hint)
        query_tokens = self._tokenize_memory_text(hint)
        should_scope_global_context = bool(
            active_project or self._should_scope_memory_context_hint(hint, query_tokens)
        )
        hint_matches: list[dict[str, Any]] = []
        relevant_entity_ids: set[int] = set()

        # Project filter subquery
        if active_project:
            project_filter = (
                " AND e.id IN (SELECT et2.entity_id FROM entity_tags et2 "
                "JOIN tags t2 ON t2.id = et2.tag_id WHERE t2.name = ?)"
            )
            project_param = active_project
        else:
            project_filter = ""
            project_param = None

        # 1. Pinned entities (always first)
        pinned_sql = f"""
            SELECT e.id, e.name, e.entity_type FROM entities e
            JOIN entity_tags et ON et.entity_id = e.id
            JOIN tags t ON t.id = et.tag_id
            WHERE t.name = 'pinned' AND e.deleted_at IS NULL{project_filter}
            ORDER BY e.updated_at DESC
            LIMIT ?
        """
        pinned_params: list[Any] = []
        if project_param:
            pinned_params.append(project_param)
        pinned_params.append(limit)
        pinned_rows = cx.execute(pinned_sql, pinned_params).fetchall()

        # 2. Hint-matched entities (if hint provided) with observation snippets
        if hint:
            candidates = self._collect_memory_context_candidates(
                hint, limit=limit, project=active_project
            )
            ranked = self._score_memory_context_candidates(candidates, hint)
            deduped = self._dedup_memory_context_candidates(ranked)
            filtered = self._filter_memory_context_candidates(deduped)
            hint_matches = filtered[:limit]
            relevant_entity_ids = {candidate["entity"].id for candidate in hint_matches}

        if not should_scope_global_context:
            visible_pinned = pinned_rows
        else:
            visible_pinned = [row for row in pinned_rows if row["id"] in relevant_entity_ids]
        result["pinned"] = [
            {"name": r["name"], "type": r["entity_type"]} for r in visible_pinned
        ]

        # 3. Recent activity (scoped by project and/or current hint when available)
        result["recent_activity"] = self._recent_activity_for_context(
            limit=5,
            project=active_project,
            query_tokens=query_tokens if should_scope_global_context else None,
            query_norm=query_norm if should_scope_global_context else "",
            relevant_entity_ids=relevant_entity_ids if should_scope_global_context else None,
        )

        result["hint_matches"] = [
                {
                    "name": candidate["entity"].name,
                    "type": candidate["entity"].entity_type,
                    "snippets": candidate["snippets"],
                    "score": round(candidate["score"], 4),
                    "conflict": candidate["conflict"],
                    "stale": candidate["stale"],
                    "signals": candidate["signals"],
                }
                for candidate in hint_matches
            ]

        # 4. Stats summary
        total_entities = cx.execute(
            "SELECT COUNT(*) c FROM entities WHERE deleted_at IS NULL"
        ).fetchone()["c"]
        total_observations = cx.execute(
            "SELECT COUNT(*) c FROM observations WHERE deleted_at IS NULL"
        ).fetchone()["c"]
        total_relations = cx.execute(
            "SELECT COUNT(*) c FROM relations WHERE deleted_at IS NULL"
        ).fetchone()["c"]
        result["stats"] = {
            "entities": total_entities,
            "observations": total_observations,
            "relations": total_relations,
        }

        return result

    def _collect_memory_context_candidates(
        self, hint: str, limit: int, project: str = ""
    ) -> list[dict[str, Any]]:
        """Collect candidate entities and feature inputs for memory_context ranking."""
        cx = self.db.cx
        query_norm = self._normalize_memory_text(hint)
        query_tokens = self._tokenize_memory_text(hint)
        if not query_norm:
            return []
        hint_type_weights = self._hint_obs_type_weights(hint)

        project_clause = ""
        project_params: list[Any] = []
        if project:
            project_clause = (
                " AND e.id IN (SELECT et.entity_id FROM entity_tags et "
                "JOIN tags t ON t.id = et.tag_id WHERE t.name = ?)"
            )
            project_params.append(project)

        fts_query = self._sanitize_fts_query(hint)
        fts_scores: dict[int, float] = {}
        try:
            rows = cx.execute(
                f"""
                SELECT e.id, rank
                FROM fts_entities fe
                JOIN entities e ON e.id = fe.rowid
                WHERE fts_entities MATCH ? AND e.deleted_at IS NULL{project_clause}
                ORDER BY rank
                LIMIT ?
                """,
                [fts_query, *project_params, max(limit * 6, 20)],
            ).fetchall()
            for row in rows:
                fts_scores[row["id"]] = row["rank"]
        except sqlite3.OperationalError as exc:
            self.last_search_diagnostics = ["context_entity_fts_fallback"]
            logger.warning("Context entity FTS unavailable; using fallback retrieval: %s", exc)

        try:
            rows = cx.execute(
                f"""
                SELECT o.entity_id, rank
                FROM fts_observations fo
                JOIN observations o ON o.id = fo.rowid
                JOIN entities e ON e.id = o.entity_id
                                WHERE fts_observations MATCH ? AND o.deleted_at IS NULL
                                    AND e.deleted_at IS NULL{project_clause}
                ORDER BY rank
                LIMIT ?
                """,
                [fts_query, *project_params, max(limit * 8, 30)],
            ).fetchall()
            for row in rows:
                eid = row["entity_id"]
                if eid not in fts_scores or row["rank"] < fts_scores[eid]:
                    fts_scores[eid] = row["rank"]
        except sqlite3.OperationalError as exc:
            diagnostics = getattr(self, "last_search_diagnostics", [])
            diagnostics.append("context_observation_fts_fallback")
            self.last_search_diagnostics = diagnostics
            logger.warning("Context observation FTS unavailable; using fallback retrieval: %s", exc)

        activity_scores = self._activity_scores_for_hint(query_tokens, query_norm, project=project)

        semantic_scores: dict[int, float] = {}
        if self.embedding_engine and self.embedding_engine.is_available():
            self._ensure_embeddings()
            query_embedding = self.embedding_engine.embed_text(hint)
            if query_embedding:
                entity_bucket_sql, entity_bucket_params = _embedding_bucket_filter(
                    "ee", query_embedding
                )
                entity_rows = cx.execute(
                    f"""
                    SELECT ee.entity_id, ee.embedding
                    FROM entity_embeddings ee
                    JOIN entities e ON e.id = ee.entity_id
                    WHERE e.deleted_at IS NULL{project_clause}
                      AND ee.model_name = ? AND ee.dimension = ?
                      AND {entity_bucket_sql}
                    ORDER BY ee.entity_id DESC
                    LIMIT ?
                    """,
                    (
                        *project_params,
                        self.embedding_engine.model_name,
                        len(query_embedding) // 4,
                        *entity_bucket_params,
                        MAX_ENTITY_EMBEDDING_CANDIDATES,
                    ),
                ).fetchall()
                for row in entity_rows:
                    sim = self.embedding_engine.cosine_similarity(query_embedding, row["embedding"])
                    if sim > 0:
                        semantic_scores[row["entity_id"]] = max(
                            semantic_scores.get(row["entity_id"], 0.0), sim
                        )

                obs_bucket_sql, obs_bucket_params = _embedding_bucket_filter(
                    "oe", query_embedding
                )
                obs_rows = cx.execute(
                    f"""
                    SELECT o.entity_id, oe.embedding
                    FROM observation_embeddings oe
                    JOIN observations o ON o.id = oe.observation_id
                    JOIN entities e ON e.id = o.entity_id
                    WHERE o.deleted_at IS NULL AND e.deleted_at IS NULL{project_clause}
                      AND oe.model_name = ? AND oe.dimension = ?
                      AND {obs_bucket_sql}
                    ORDER BY oe.observation_id DESC
                    LIMIT ?
                    """,
                    (
                        *project_params,
                        self.embedding_engine.model_name,
                        len(query_embedding) // 4,
                        *obs_bucket_params,
                        MAX_OBSERVATION_EMBEDDING_CANDIDATES,
                    ),
                ).fetchall()
                for row in obs_rows:
                    sim = self.embedding_engine.cosine_similarity(query_embedding, row["embedding"])
                    if sim > 0:
                        semantic_scores[row["entity_id"]] = max(
                            semantic_scores.get(row["entity_id"], 0.0), sim
                        )

        ordered_candidate_ids: list[int] = []
        candidate_ids = set(fts_scores) | set(semantic_scores) | set(activity_scores)
        if not candidate_ids:
            ordered_candidate_ids = self._fuzzy_search(hint, max(limit * 4, 20))
            candidate_ids = set(ordered_candidate_ids)
            if project and candidate_ids:
                placeholders = ",".join("?" for _ in candidate_ids)
                rows = cx.execute(
                    f"""
                    SELECT DISTINCT et.entity_id
                    FROM entity_tags et
                    JOIN tags t ON t.id = et.tag_id
                    WHERE et.entity_id IN ({placeholders}) AND t.name = ?
                    """,
                    [*candidate_ids, project],
                ).fetchall()
                candidate_ids = {row["entity_id"] for row in rows}
                ordered_candidate_ids = [
                    eid for eid in ordered_candidate_ids if eid in candidate_ids
                ]

        if not candidate_ids:
            fallback_rows = cx.execute(
                f"""
                SELECT e.id
                FROM entities e
                WHERE e.deleted_at IS NULL{project_clause}
                ORDER BY e.updated_at DESC
                LIMIT ?
                """,
                [*project_params, max(limit * 2, 10)],
            ).fetchall()
            ordered_candidate_ids = [row["id"] for row in fallback_rows]
            candidate_ids = set(ordered_candidate_ids)
        if not candidate_ids:
            return []

        entities = self._load_entities_batch(ordered_candidate_ids or candidate_ids)
        if not entities:
            return []

        if fts_scores:
            min_rank = min(fts_scores.values())
            max_rank = max(fts_scores.values())
            rank_range = max_rank - min_rank if max_rank != min_rank else 1.0
            norm_fts = {eid: (max_rank - score) / rank_range for eid, score in fts_scores.items()}
        else:
            norm_fts = {}

        candidates: list[dict[str, Any]] = []
        for entity in entities:
            exact_phrase = query_norm in self._normalize_memory_text(entity.name)
            obs_signature_parts: list[str] = []
            for obs in entity.observations:
                obs_signature_parts.append(self._normalize_memory_text(obs.content))

            observation_matches = sorted(
                (
                    self._observation_match_features(
                        obs, query_tokens, query_norm, hint_type_weights
                    )
                    for obs in entity.observations
                ),
                key=lambda match: (
                    match["match_score"],
                    match["importance"],
                    match["confidence"],
                    match["observation"].updated_at,
                ),
                reverse=True,
            )
            if observation_matches and observation_matches[0]["exact_phrase"]:
                exact_phrase = True

            name_overlap = self._lexical_overlap(query_tokens, entity.name)
            exact_name = self._normalize_memory_text(entity.name) == query_norm
            best_match = observation_matches[0] if observation_matches else None
            second_match = observation_matches[1] if len(observation_matches) > 1 else None
            snippets = self._select_memory_snippets(observation_matches)
            stale_penalty = self._staleness_penalty(entity, best_match)
            candidates.append(
                {
                    "entity": entity,
                    "fts_score": norm_fts.get(entity.id, 0.0),
                    "semantic_score": semantic_scores.get(entity.id, 0.0),
                    "exact_name": exact_name,
                    "exact_phrase": exact_phrase,
                    "lexical_score": max(
                        name_overlap, best_match["lexical_overlap"] if best_match else 0.0
                    ),
                    "pinned": "pinned" in entity.tags,
                    "importance": best_match["importance"] if best_match else 0.0,
                    "confidence": best_match["confidence"] if best_match else 0.0,
                    "observation_score": best_match["match_score"] if best_match else 0.0,
                    "support_score": second_match["match_score"] if second_match else 0.0,
                    "obs_type_boost": best_match["obs_type_boost"] if best_match else 0.0,
                    "update_recency_score": self._bounded_recency_score(entity.updated_at),
                    "access_recency_score": self._bounded_recency_score(entity.last_accessed_at),
                    "activity_score": activity_scores.get(entity.id, 0.0),
                    "stale_penalty": stale_penalty,
                    "snippets": snippets,
                    "best_snippet": best_match["observation"].content if best_match else "",
                    "obs_signature": "|".join(sorted(part for part in obs_signature_parts if part)),
                    "conflict": self._has_conflicting_observation_matches(observation_matches),
                    "stale": stale_penalty >= 0.35,
                    "signals": [],
                }
            )
        return candidates

    def _score_memory_context_candidates(
        self, candidates: list[dict[str, Any]], hint: str
    ) -> list[dict[str, Any]]:
        """Score memory_context candidates across lexical, semantic, and durability signals."""
        query_norm = self._normalize_memory_text(hint)
        scored: list[dict[str, Any]] = []
        for candidate in candidates:
            exact_name_boost = 4.0 if candidate["exact_name"] else 0.0
            exact_phrase_boost = (
                1.3 if candidate["exact_phrase"] and not candidate["exact_name"] else 0.0
            )
            lexical_score = 1.4 * candidate["lexical_score"] + 0.55 * candidate["fts_score"]
            semantic_score = 0.9 * candidate["semantic_score"]
            pinned_boost = 0.9 if candidate["pinned"] else 0.0
            observation_score = (
                2.0 * candidate["observation_score"] + 0.25 * candidate["support_score"]
            )
            access_boost = 0.35 * candidate["access_recency_score"]
            activity_boost = 0.45 * candidate["activity_score"]
            update_recency_boost = 0.15 * candidate["update_recency_score"]
            stale_penalty = 0.45 * candidate["stale_penalty"]
            score = (
                exact_name_boost
                + exact_phrase_boost
                + lexical_score
                + semantic_score
                + pinned_boost
                + observation_score
                + access_boost
                + activity_boost
                + update_recency_boost
                - stale_penalty
            )
            candidate["score"] = score
            candidate["canonical_hint"] = query_norm
            candidate["signals"] = self._candidate_signals(candidate)
            scored.append(candidate)

        scored.sort(
            key=lambda candidate: (
                candidate["score"],
                candidate["pinned"],
                candidate["importance"],
                candidate["confidence"],
                candidate["access_recency_score"],
                candidate["semantic_score"],
                candidate["entity"].updated_at,
            ),
            reverse=True,
        )
        return scored

    def _dedup_memory_context_candidates(
        self, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Suppress near-identical matches without collapsing distinct memories."""
        deduped: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str]] = set()
        for candidate in candidates:
            entity = candidate["entity"]
            snippet_key = self._normalize_memory_text(candidate["best_snippet"])
            name_key = self._normalize_memory_text(entity.name)
            obs_signature = candidate.get("obs_signature", "")
            if snippet_key and obs_signature and len(snippet_key) >= 12:
                dedup_key = (entity.entity_type, obs_signature)
            else:
                dedup_key = (entity.entity_type, name_key)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            deduped.append(candidate)
        return deduped

    @staticmethod
    def _is_memory_context_candidate_relevant(candidate: dict[str, Any]) -> bool:
        """Keep only candidates with at least one strong relevance signal.

        This prevents weak embedding-only matches from polluting every-turn context,
        while preserving exact, lexical, FTS, and recent-activity driven hits.
        """
        if candidate["exact_name"] or candidate["exact_phrase"]:
            return True
        if candidate["fts_score"] > 0.0:
            return True
        if candidate["lexical_score"] >= 0.2:
            return True
        if candidate["observation_score"] >= 0.35:
            return True
        if candidate["activity_score"] >= 0.15:
            return True
        return candidate["semantic_score"] >= 0.45 and not candidate["stale"]

    def _filter_memory_context_candidates(
        self, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return [
            candidate
            for candidate in candidates
            if self._is_memory_context_candidate_relevant(candidate)
        ]

    def _recent_activity_for_context(
        self,
        limit: int,
        project: str = "",
        query_tokens: set[str] | None = None,
        query_norm: str = "",
        relevant_entity_ids: set[int] | None = None,
    ) -> list[dict[str, str]]:
        cx = self.db.cx
        if not project and not query_norm and not query_tokens and not relevant_entity_ids:
            rows = cx.execute(
                "SELECT action, summary, created_at FROM activity_log ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [
                {"action": row["action"], "summary": row["summary"], "at": row["created_at"]}
                for row in rows
            ]

        rows = cx.execute(
            "SELECT action, summary, created_at, entity_ids_json FROM activity_log "
            "ORDER BY created_at DESC LIMIT ?",
            (max(limit * 6, 20),),
        ).fetchall()
        tokens = query_tokens or set()
        relevant_ids = relevant_entity_ids or set()
        project_tag_cache: dict[int, list[str]] = {}
        filtered: list[dict[str, str]] = []

        for row in rows:
            try:
                entity_ids = json.loads(row["entity_ids_json"] or "[]")
            except json.JSONDecodeError:
                entity_ids = []

            if project:
                if not entity_ids:
                    continue
                project_matched = False
                for entity_id in entity_ids:
                    tags = project_tag_cache.setdefault(entity_id, self._entity_tags_for(entity_id))
                    if project in tags:
                        project_matched = True
                        break
                if not project_matched:
                    continue

            summary = row["summary"] or ""
            summary_norm = self._normalize_memory_text(summary)
            summary_overlap = self._lexical_overlap(tokens, summary) if tokens else 0.0
            linked_relevant = bool(relevant_ids and set(entity_ids) & relevant_ids)
            summary_matches = bool(query_norm and query_norm in summary_norm)

            if (query_norm or tokens or relevant_ids) and not (
                linked_relevant or summary_matches or summary_overlap > 0.0
            ):
                continue

            filtered.append(
                {"action": row["action"], "summary": summary, "at": row["created_at"]}
            )
            if len(filtered) >= limit:
                break

        return filtered

    def _activity_scores_for_hint(
        self, query_tokens: set[str], query_norm: str, project: str = ""
    ) -> dict[int, float]:
        if not query_tokens and not query_norm:
            return {}

        cx = self.db.cx
        rows = cx.execute(
            "SELECT entity_ids_json, summary, created_at FROM activity_log "
            "ORDER BY created_at DESC LIMIT ?",
            (self.MEMORY_ACTIVITY_LOOKBACK,),
        ).fetchall()
        scores: dict[int, float] = {}
        for row in rows:
            try:
                entity_ids = json.loads(row["entity_ids_json"] or "[]")
            except json.JSONDecodeError:
                entity_ids = []
            if not entity_ids:
                continue

            summary = row["summary"] or ""
            summary_overlap = self._lexical_overlap(query_tokens, summary)
            if query_norm and query_norm in self._normalize_memory_text(summary):
                summary_overlap = max(summary_overlap, 1.0)
            recency = self._bounded_recency_score(row["created_at"])
            base_score = (0.12 + 0.55 * summary_overlap) * (0.5 + 0.5 * recency)

            for eid in entity_ids:
                if project and project not in self._entity_tags_for(eid):
                    continue
                scores[eid] = max(scores.get(eid, 0.0), min(base_score, 1.0))
        return scores

    def _observation_match_features(
        self,
        observation: Observation,
        query_tokens: set[str],
        query_norm: str,
        hint_type_weights: dict[str, float],
    ) -> dict[str, Any]:
        normalized_content = self._normalize_memory_text(observation.content)
        lexical_overlap = self._lexical_overlap(query_tokens, observation.content)
        exact_phrase = bool(query_norm and query_norm in normalized_content)
        lexical_signal = max(lexical_overlap, 1.0 if exact_phrase else 0.0)
        obs_type_boost = self._observation_type_boost(observation.obs_type, hint_type_weights)
        confidence = min(max(observation.confidence, 0.0), 1.0)
        importance = min(max(observation.importance, 0.0), 1.0)
        match_score = (
            lexical_signal * (0.55 + 0.3 * importance + 0.15 * confidence) + obs_type_boost
        )
        return {
            "observation": observation,
            "normalized_content": normalized_content,
            "lexical_overlap": lexical_overlap,
            "exact_phrase": exact_phrase,
            "importance": importance,
            "confidence": confidence,
            "obs_type_boost": obs_type_boost,
            "match_score": match_score,
        }

    def _select_memory_snippets(self, observation_matches: list[dict[str, Any]]) -> list[str]:
        snippets: list[str] = []
        remaining = self.MEMORY_SNIPPET_BUDGET_CHARS
        seen: set[str] = set()

        for match in observation_matches:
            content = match["observation"].content.strip()
            normalized = match["normalized_content"]
            if not content or normalized in seen:
                continue
            snippet = self._snippet_for_budget(content, remaining)
            if not snippet:
                continue
            snippets.append(snippet)
            seen.add(normalized)
            remaining -= len(snippet)
            if (
                len(snippets) >= self.MEMORY_MAX_SNIPPETS
                or remaining < self.MEMORY_SNIPPET_MIN_CHARS
            ):
                break
        return snippets

    def _snippet_for_budget(self, text: str, remaining: int) -> str:
        if remaining < self.MEMORY_SNIPPET_MIN_CHARS:
            return ""
        if len(text) <= remaining:
            return text
        budget = min(remaining, 160)
        if budget <= 3:
            return ""
        return text[: budget - 3].rstrip() + "..."

    @staticmethod
    def _has_conflicting_observation_matches(observation_matches: list[dict[str, Any]]) -> bool:
        if len(observation_matches) < 2:
            return False

        primary = observation_matches[0]
        secondary = observation_matches[1]
        if primary["normalized_content"] == secondary["normalized_content"]:
            return False
        if primary["lexical_overlap"] < 0.5 or secondary["lexical_overlap"] < 0.5:
            return False
        if primary["match_score"] < 0.6 or secondary["match_score"] < 0.6:
            return False
        return abs(primary["match_score"] - secondary["match_score"]) <= 0.35

    @staticmethod
    def _candidate_signals(candidate: dict[str, Any]) -> list[str]:
        signals: list[str] = []
        if candidate["exact_name"]:
            signals.append("exact_name")
        elif candidate["exact_phrase"]:
            signals.append("exact_phrase")

        if candidate["pinned"]:
            signals.append("pinned")
        if candidate["semantic_score"] >= 0.2:
            signals.append("semantic")
        if candidate["activity_score"] >= 0.1:
            signals.append("recent_activity")
        if candidate["access_recency_score"] >= 0.4:
            signals.append("recently_accessed")
        if candidate["obs_type_boost"] > 0:
            signals.append("typed_observation")
        if candidate["stale"]:
            signals.append("stale")
        if candidate["conflict"]:
            signals.append("conflict")
        return signals

    @staticmethod
    def _hint_obs_type_weights(hint: str) -> dict[str, float]:
        weights: dict[str, float] = {}
        hint_lower = hint.lower()
        if "/" in hint or "\\" in hint or re.search(r"\.[a-z0-9]{1,8}\b", hint_lower):
            weights["file_path"] = 0.45
        if re.search(r"\b(get|post|put|patch|delete)\s+/", hint_lower) or hint_lower.startswith(
            "/api/"
        ):
            weights["api_endpoint"] = 0.35
        if any(token in hint for token in ("{", "}", "=>", "::", "()")) or re.search(
            r"\bdef\b|\bclass\b|[A-Za-z_][A-Za-z0-9_]*\(", hint
        ):
            weights["code_snippet"] = 0.35
        if "=" in hint or any(
            token in hint_lower for token in ("config", "setting", "issuer", "audience")
        ):
            weights["config"] = max(weights.get("config", 0.0), 0.25)
        if "schema" in hint_lower:
            weights["schema"] = 0.25
        return weights

    @staticmethod
    def _observation_type_boost(obs_type: str, hint_type_weights: dict[str, float]) -> float:
        if obs_type in hint_type_weights:
            return hint_type_weights[obs_type]
        if obs_type in PROTECTED_OBS_TYPES and hint_type_weights:
            return 0.08
        return 0.0

    def _staleness_penalty(self, entity: Entity, best_match: dict[str, Any] | None) -> float:
        if "pinned" in entity.tags:
            return 0.0

        penalty = 0.0
        access_days = self._days_since_iso(entity.last_accessed_at)
        if access_days > self.MEMORY_ACCESS_STALE_DAYS:
            penalty += min((access_days - self.MEMORY_ACCESS_STALE_DAYS) / 90.0, 0.35)

        if best_match:
            observation = best_match["observation"]
            if (
                best_match["confidence"] < 0.5
                and self._days_since_iso(observation.updated_at)
                > self.MEMORY_LOW_CONFIDENCE_STALE_DAYS
            ):
                penalty += 0.2

        return min(penalty, 0.5)

    @staticmethod
    def _days_since_iso(value: str) -> float:
        try:
            then = datetime.fromisoformat(value.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            return max((now - then).total_seconds() / 86400.0, 0.0)
        except (AttributeError, ValueError):
            return 365.0

    def _entity_tags_for(self, entity_id: int) -> list[str]:
        rows = self.db.cx.execute(
            "SELECT t.name FROM entity_tags et JOIN tags t ON t.id = et.tag_id "
            "WHERE et.entity_id = ?",
            (entity_id,),
        ).fetchall()
        return [row["name"] for row in rows]

    @staticmethod
    def _normalize_memory_text(text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", text.lower()))

    @staticmethod
    def _tokenize_memory_text(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def _should_scope_memory_context_hint(self, hint: str, query_tokens: set[str]) -> bool:
        if not hint.strip():
            return False
        if len(query_tokens) >= 2:
            return True
        if any(token in hint for token in ("/", "\\", ".", "::", "=>", "{")):
            return True
        token = next(iter(query_tokens), "")
        return bool(token and token not in self.GENERIC_MEMORY_HINT_TOKENS)

    def _lexical_overlap(self, query_tokens: set[str], text: str) -> float:
        if not query_tokens:
            return 0.0
        text_tokens = self._tokenize_memory_text(text)
        if not text_tokens:
            return 0.0
        return len(query_tokens & text_tokens) / len(query_tokens)

    @staticmethod
    def _bounded_recency_score(updated_at: str) -> float:
        try:
            updated = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            days = max((now - updated).total_seconds() / 86400.0, 0.0)
        except (AttributeError, ValueError):
            days = 365.0
        return min(1.0 / (1.0 + days), 1.0)

    # ------------------------------------------------------------------
    # Merge Entities
    # ------------------------------------------------------------------

    def merge_entities(
        self, source_name: str, target_name: str, strategy: str = "combine"
    ) -> Entity | None:
        """Merge source entity into target. Transfers observations, relations, tags."""
        with self.db.transaction() as cx:
            source = cx.execute(
                "SELECT id FROM entities WHERE name = ? AND deleted_at IS NULL",
                (source_name,),
            ).fetchone()
            target = cx.execute(
                "SELECT id FROM entities WHERE name = ? AND deleted_at IS NULL",
                (target_name,),
            ).fetchone()
            if not source or not target:
                return None
            sid, tid = source["id"], target["id"]

            # Transfer observations
            if strategy == "combine":
                cx.execute(
                    "UPDATE observations SET entity_id = ? "
                    "WHERE entity_id = ? AND deleted_at IS NULL",
                    (tid, sid),
                )
            elif strategy == "dedupe":
                # Only transfer observations not already on target
                existing = {
                    r["content"]
                    for r in cx.execute(
                        "SELECT content FROM observations WHERE entity_id = ? "
                        "AND deleted_at IS NULL",
                        (tid,),
                    ).fetchall()
                }
                rows = cx.execute(
                    "SELECT id, content FROM observations WHERE entity_id = ? "
                    "AND deleted_at IS NULL",
                    (sid,),
                ).fetchall()
                for r in rows:
                    if r["content"] not in existing:
                        cx.execute(
                            "UPDATE observations SET entity_id = ? WHERE id = ?", (tid, r["id"])
                        )
                    else:
                        cx.execute(
                            "UPDATE observations SET deleted_at = ? WHERE id = ?",
                            (_now_iso(), r["id"]),
                        )

            # Transfer relations — update from/to references, skip if would create duplicate
            for col in ("from_entity_id", "to_entity_id"):
                rows = cx.execute(
                    f"SELECT id, from_entity_id, to_entity_id, relation_type FROM relations "
                    f"WHERE {col} = ? AND deleted_at IS NULL",
                    (sid,),
                ).fetchall()
                for r in rows:
                    new_from = tid if r["from_entity_id"] == sid else r["from_entity_id"]
                    new_to = tid if r["to_entity_id"] == sid else r["to_entity_id"]
                    if new_from == new_to:
                        # Self-relation after merge — soft-delete
                        cx.execute(
                            "UPDATE relations SET deleted_at = ? WHERE id = ?",
                            (_now_iso(), r["id"]),
                        )
                        continue
                    try:
                        cx.execute(
                            "UPDATE relations SET from_entity_id = ?, to_entity_id = ? "
                            "WHERE id = ?",
                            (new_from, new_to, r["id"]),
                        )
                    except Exception:
                        # Unique constraint violation — soft-delete the duplicate
                        cx.execute(
                            "UPDATE relations SET deleted_at = ? WHERE id = ?",
                            (_now_iso(), r["id"]),
                        )

            # Transfer tags
            source_tags = cx.execute(
                "SELECT tag_id FROM entity_tags WHERE entity_id = ?", (sid,)
            ).fetchall()
            for t in source_tags:
                with suppress(Exception):
                    cx.execute(
                        "INSERT INTO entity_tags (entity_id, tag_id) VALUES (?, ?)",
                        (tid, t["tag_id"]),
                    )

            # Soft-delete source
            cx.execute("UPDATE entities SET deleted_at = ? WHERE id = ?", (_now_iso(), sid))
            cx.execute("UPDATE entities SET updated_at = ? WHERE id = ?", (_now_iso(), tid))
        # Re-embed target entity after merge (outside transaction — read-only + embed)
        cx = self.db.cx
        target_row = cx.execute(
            "SELECT name, entity_type FROM entities WHERE id = ?", (tid,)
        ).fetchone()
        if target_row:
            self._embed_entity(tid, target_row["name"], target_row["entity_type"])
        return self._load_entity(tid)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def memory_stats(self) -> dict[str, Any]:
        cx = self.db.cx
        stats: dict[str, Any] = {}

        stats["entities"] = cx.execute(
            "SELECT COUNT(*) c FROM entities WHERE deleted_at IS NULL"
        ).fetchone()["c"]
        stats["observations"] = cx.execute(
            "SELECT COUNT(*) c FROM observations WHERE deleted_at IS NULL"
        ).fetchone()["c"]
        stats["relations"] = cx.execute(
            "SELECT COUNT(*) c FROM relations WHERE deleted_at IS NULL"
        ).fetchone()["c"]
        stats["deleted_entities"] = cx.execute(
            "SELECT COUNT(*) c FROM entities WHERE deleted_at IS NOT NULL"
        ).fetchone()["c"]
        stats["activity_entries"] = cx.execute("SELECT COUNT(*) c FROM activity_log").fetchone()[
            "c"
        ]

        # Tag distribution
        tag_dist = cx.execute(
            """
            SELECT t.name, COUNT(et.entity_id) as count FROM tags t
            LEFT JOIN entity_tags et ON et.tag_id = t.id
            GROUP BY t.id ORDER BY count DESC
            """
        ).fetchall()
        stats["tag_distribution"] = {r["name"]: r["count"] for r in tag_dist}

        # DB size
        if self.db.db_path != ":memory:":
            try:
                import os

                stats["db_size_bytes"] = os.path.getsize(self.db.db_path)
            except OSError:
                stats["db_size_bytes"] = 0
        else:
            stats["db_size_bytes"] = 0

        # Orphan entities (no observations, no relations)
        stats["orphan_entities"] = cx.execute(
            """
            SELECT COUNT(*) c FROM entities e
            WHERE e.deleted_at IS NULL
                            AND NOT EXISTS (
                                    SELECT 1 FROM observations o
                                    WHERE o.entity_id = e.id AND o.deleted_at IS NULL
                            )
                            AND NOT EXISTS (
                                    SELECT 1 FROM relations r
                                    WHERE (r.from_entity_id = e.id OR r.to_entity_id = e.id)
                                        AND r.deleted_at IS NULL
                            )
            """
        ).fetchone()["c"]

        return stats

    # ------------------------------------------------------------------
    # Export / Import
    # ------------------------------------------------------------------

    def export_graph(self, fmt: str = "json") -> str:
        """Export entire graph as JSON or JSONL."""
        graph = self.read_graph(include_deleted=False)
        if fmt == "jsonl":
            lines = []
            for e in graph.entities:
                lines.append(
                    json.dumps(
                        {
                            "type": "entity",
                            "name": e.name,
                            "entityType": e.entity_type,
                            "observations": [o.content for o in e.observations],
                        }
                    )
                )
            for r in graph.relations:
                lines.append(
                    json.dumps(
                        {
                            "type": "relation",
                            "from": r.from_name,
                            "to": r.to_name,
                            "relationType": r.relation_type,
                        }
                    )
                )
            return "\n".join(lines)
        else:
            return json.dumps(graph.to_dict(), indent=2)

    def import_graph(self, data: str) -> dict[str, int]:
        """Import from JSON or JSONL (auto-detects format). Compatible with old TS server."""
        data = data.strip()
        counts = {"entities": 0, "relations": 0, "observations": 0}

        # Detect format: JSON (single object) vs JSONL (one JSON per line)
        # JSONL has multiple lines each starting with '{', JSON is a single object
        lines = [line.strip() for line in data.split("\n") if line.strip()]
        is_jsonl = len(lines) > 1 and all(line.startswith("{") for line in lines)

        if not is_jsonl and data.startswith("{"):
            # JSON format (single object with entities/relations arrays)
            obj = json.loads(data)
            entities = obj.get("entities", [])
            relations = obj.get("relations", [])
        else:
            # JSONL format (one JSON object per line)
            entities = []
            relations = []
            for line in lines:
                if not line:
                    continue
                item = json.loads(line)
                if item.get("type") == "entity":
                    entities.append(item)
                elif item.get("type") == "relation":
                    relations.append(item)

        # Import entities
        for e in entities:
            name = e.get("name", "")
            etype = e.get("entityType", e.get("entity_type", ""))
            observations = e.get("observations", [])
            created = self.create_entities(
                [{"name": name, "entityType": etype, "observations": observations}]
            )
            if created:
                counts["entities"] += 1
                counts["observations"] += len(observations)

        # Import relations
        for r in relations:
            try:
                created = self.create_relations(
                    [
                        {
                            "from": r.get("from", ""),
                            "to": r.get("to", ""),
                            "relationType": r.get("relationType", r.get("relation_type", "")),
                        }
                    ]
                )
                if created:
                    counts["relations"] += 1
            except ValueError:
                # Missing entity — skip
                continue

        return counts

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _load_entity(self, eid: int) -> Entity | None:
        cx = self.db.cx
        row = cx.execute("SELECT * FROM entities WHERE id = ?", (eid,)).fetchone()
        if not row:
            return None
        # Load observations
        obs_rows = cx.execute(
            "SELECT * FROM observations WHERE entity_id = ? AND deleted_at IS NULL "
            "ORDER BY created_at",
            (eid,),
        ).fetchall()
        observations = [
            Observation(
                id=r["id"],
                entity_id=eid,
                content=r["content"],
                source=r["source"],
                confidence=r["confidence"],
                importance=r["importance"] if "importance" in r else 0.5,
                obs_type=r["obs_type"] if "obs_type" in r else "",
                version=r["version"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
            )
            for r in obs_rows
        ]
        # Load tags
        tag_rows = cx.execute(
            "SELECT t.name FROM entity_tags et JOIN tags t ON t.id = et.tag_id "
            "WHERE et.entity_id = ?",
            (eid,),
        ).fetchall()
        tags = [r["name"] for r in tag_rows]

        return Entity(
            id=row["id"],
            name=row["name"],
            entity_type=row["entity_type"],
            metadata_json=row["metadata_json"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_accessed_at=row["last_accessed_at"],
            deleted_at=row["deleted_at"],
            observations=observations,
            tags=tags,
        )

    def _load_relations_for(self, entity_ids: set[int], include_deleted: bool) -> list[Relation]:
        if not entity_ids:
            return []
        cx = self.db.cx
        placeholders = ",".join("?" for _ in entity_ids)
        ids = list(entity_ids)
        deleted_clause = "" if include_deleted else "AND r.deleted_at IS NULL"
        rows = cx.execute(
            f"""
            SELECT r.*, ef.name as from_name, et.name as to_name
            FROM relations r
            JOIN entities ef ON ef.id = r.from_entity_id
            JOIN entities et ON et.id = r.to_entity_id
            WHERE r.from_entity_id IN ({placeholders})
              AND r.to_entity_id IN ({placeholders})
              {deleted_clause}
            """,
            ids + ids,
        ).fetchall()
        result = []
        for r in rows:
            tag_rows = cx.execute(
                "SELECT t.name FROM relation_tags rt JOIN tags t ON t.id = rt.tag_id "
                "WHERE rt.relation_id = ?",
                (r["id"],),
            ).fetchall()
            result.append(
                Relation(
                    id=r["id"],
                    from_entity_id=r["from_entity_id"],
                    to_entity_id=r["to_entity_id"],
                    relation_type=r["relation_type"],
                    weight=r["weight"],
                    from_name=r["from_name"],
                    to_name=r["to_name"],
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                    deleted_at=r["deleted_at"],
                    tags=[tr["name"] for tr in tag_rows],
                )
            )
        return result

    def _apply_tag_to_entity(self, entity_id: int, tag_name: str) -> bool:
        cx = self.db.cx
        tag_row = cx.execute("SELECT id FROM tags WHERE name = ?", (tag_name,)).fetchone()
        if not tag_row:
            # Auto-create tag
            cur = cx.execute("INSERT INTO tags (name) VALUES (?)", (tag_name,))
            tag_id = cur.lastrowid
        else:
            tag_id = tag_row["id"]
        try:
            cx.execute(
                "INSERT INTO entity_tags (entity_id, tag_id) VALUES (?, ?)",
                (entity_id, tag_id),
            )
            return True
        except Exception:
            return False

    def _apply_tag_to_observation(self, obs_id: int, tag_name: str) -> bool:
        cx = self.db.cx
        tag_row = cx.execute("SELECT id FROM tags WHERE name = ?", (tag_name,)).fetchone()
        if not tag_row:
            cur = cx.execute("INSERT INTO tags (name) VALUES (?)", (tag_name,))
            tag_id = cur.lastrowid
        else:
            tag_id = tag_row["id"]
        try:
            cx.execute(
                "INSERT INTO observation_tags (observation_id, tag_id) VALUES (?, ?)",
                (obs_id, tag_id),
            )
            return True
        except Exception:
            return False

    def _apply_tag_to_relation(self, rel_id: int, tag_name: str) -> bool:
        cx = self.db.cx
        tag_row = cx.execute("SELECT id FROM tags WHERE name = ?", (tag_name,)).fetchone()
        if not tag_row:
            cur = cx.execute("INSERT INTO tags (name) VALUES (?)", (tag_name,))
            tag_id = cur.lastrowid
        else:
            tag_id = tag_row["id"]
        try:
            cx.execute(
                "INSERT INTO relation_tags (relation_id, tag_id) VALUES (?, ?)",
                (rel_id, tag_id),
            )
            return True
        except Exception:
            return False

    def _sanitize_fts_query(self, query: str) -> str:
        """Sanitize user query for FTS5. Supports prefix search with *."""
        # Remove FTS5 special operators if user didn't intend them
        # Keep quoted phrases, AND, OR, NOT, *
        cleaned = query.strip()
        if not cleaned:
            return '""'
        # If it looks like a simple search term, wrap tokens for prefix matching
        if not any(op in cleaned for op in ['"', "AND", "OR", "NOT", "*", "NEAR"]):
            tokens = cleaned.split()
            if not tokens:
                return '""'
            # Use prefix matching on all tokens for better partial match recall
            parts = [f'"{t}"*' for t in tokens]
            return " AND ".join(parts)
        return cleaned

    def _fuzzy_search(self, query: str, limit: int) -> list[int]:
        """Fallback fuzzy search using trigram-like similarity on entity names and observation content."""
        cx = self.db.cx
        query_lower = query.lower()
        query_trigrams = self._trigrams(query_lower)
        if not query_trigrams:
            return []

        # Score entity names (Limit fallback scan to 10k most recent)
        rows = cx.execute(
            "SELECT id, name, entity_type FROM entities WHERE deleted_at IS NULL "
            "ORDER BY updated_at DESC LIMIT 10000"
        ).fetchall()

        best_scores: dict[int, float] = {}
        for row in rows:
            text = f"{row['name']} {row['entity_type']}".lower()
            text_trigrams = self._trigrams(text)
            if not text_trigrams:
                continue
            intersection = query_trigrams & text_trigrams
            union = query_trigrams | text_trigrams
            similarity = len(intersection) / len(union) if union else 0
            if similarity > 0.1:
                best_scores[row["id"]] = similarity

        # Also score observation content for active entities only (limit fallback scan)
        obs_rows = cx.execute(
            """
            SELECT o.entity_id, o.content
            FROM observations o
            JOIN entities e ON e.id = o.entity_id
            WHERE o.deleted_at IS NULL AND e.deleted_at IS NULL
            ORDER BY o.updated_at DESC
            LIMIT 10000
            """
        ).fetchall()
        for row in obs_rows:
            text = row["content"].lower()
            text_trigrams = self._trigrams(text)
            if not text_trigrams:
                continue
            intersection = query_trigrams & text_trigrams
            union = query_trigrams | text_trigrams
            similarity = len(intersection) / len(union) if union else 0
            if similarity > 0.1:
                eid = row["entity_id"]
                best_scores[eid] = max(best_scores.get(eid, 0.0), similarity)

        scored = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)
        return [eid for eid, _ in (scored[:limit] if limit > 0 else scored)]

    @staticmethod
    def _trigrams(text: str) -> set[str]:
        if len(text) < 3:
            return {text} if text else set()
        return {text[i : i + 3] for i in range(len(text) - 2)}

    def _load_entities_batch(
        self, entity_ids: list[int] | set[int], include_deleted: bool = False
    ) -> list[Entity]:
        """Load multiple entities in batch (avoids N+1 queries)."""
        if not entity_ids:
            return []
        cx = self.db.cx
        ids = list(dict.fromkeys(entity_ids))
        placeholders = ",".join("?" for _ in ids)

        # Batch fetch entities
        deleted_filter = "" if include_deleted else "AND deleted_at IS NULL"
        entity_rows = cx.execute(
            f"SELECT * FROM entities WHERE id IN ({placeholders}) {deleted_filter}",
            ids,
        ).fetchall()
        if not entity_rows:
            return []

        live_ids = [r["id"] for r in entity_rows]
        live_placeholders = ",".join("?" for _ in live_ids)

        # Batch fetch observations
        obs_rows = cx.execute(
            f"SELECT * FROM observations WHERE entity_id IN ({live_placeholders}) AND deleted_at IS NULL ORDER BY created_at",
            live_ids,
        ).fetchall()
        obs_by_entity: dict[int, list[Observation]] = {}
        for r in obs_rows:
            obs_by_entity.setdefault(r["entity_id"], []).append(
                Observation(
                    id=r["id"],
                    entity_id=r["entity_id"],
                    content=r["content"],
                    source=r["source"],
                    confidence=r["confidence"],
                    importance=r["importance"] if "importance" in r.keys() else 0.5,
                    obs_type=r["obs_type"] if "obs_type" in r.keys() else "",
                    version=r["version"],
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                )
            )

        # Batch fetch tags
        tag_rows = cx.execute(
            f"SELECT et.entity_id, t.name FROM entity_tags et JOIN tags t ON t.id = et.tag_id WHERE et.entity_id IN ({live_placeholders})",
            live_ids,
        ).fetchall()
        tags_by_entity: dict[int, list[str]] = {}
        for r in tag_rows:
            tags_by_entity.setdefault(r["entity_id"], []).append(r["name"])

        # Assemble
        entity_by_id: dict[int, Entity] = {}
        for row in entity_rows:
            eid = row["id"]
            entity_by_id[eid] = Entity(
                id=eid,
                name=row["name"],
                entity_type=row["entity_type"],
                metadata_json=row["metadata_json"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                last_accessed_at=row["last_accessed_at"],
                deleted_at=row["deleted_at"],
                observations=obs_by_entity.get(eid, []),
                tags=tags_by_entity.get(eid, []),
            )
        return [entity_by_id[eid] for eid in ids if eid in entity_by_id]

    def _prepare_embedding(
        self,
        text: str,
        deadline: float | None = None,
    ) -> tuple[bytes | None, bool]:
        """Compute an embedding outside a write transaction when embeddings are enabled."""
        if not text:
            return None, False
        if not self.embedding_engine or not self.embedding_engine.is_available():
            return None, False
        remaining = self._write_embedding_time_remaining(deadline)
        return self.embedding_engine.embed_text_with_timeout(text, remaining)

    def _prepare_embeddings(
        self,
        texts: list[str],
        deadline: float | None = None,
    ) -> tuple[list[bytes | None], bool]:
        """Batch-compute embeddings outside a write transaction when possible."""
        if not texts:
            return [], False
        if not self.embedding_engine or not self.embedding_engine.is_available():
            return [None for _ in texts], False

        remaining = self._write_embedding_time_remaining(deadline)
        embeddings, timed_out = self.embedding_engine.embed_batch_with_timeout(texts, remaining)
        if timed_out:
            return [None for _ in texts], True
        if len(embeddings) == len(texts):
            return list(embeddings), False

        fallback_embeddings: list[bytes | None] = []
        fallback_timed_out = False
        for text in texts:
            embedding, embedding_timed_out = self._prepare_embedding(text, deadline=deadline)
            fallback_embeddings.append(embedding)
            if embedding_timed_out:
                fallback_timed_out = True
                fallback_embeddings.extend(
                    [None for _ in range(len(texts) - len(fallback_embeddings))]
                )
                break
        return fallback_embeddings, fallback_timed_out

    def _embed_entity(
        self,
        entity_id: int,
        name: str,
        entity_type: str,
        embedding: bytes | None = None,
        allow_fallback: bool = True,
    ) -> None:
        """Embed an entity's name+type and store in entity_embeddings."""
        if not self.embedding_engine or not self.embedding_engine.is_available():
            return
        emb = embedding
        if emb is None and allow_fallback:
            text = f"{name} {entity_type}".strip()
            emb = self.embedding_engine.embed_text(text)
        if emb:
            buckets = embedding_buckets(emb)
            self.db.cx.execute(
                "INSERT OR REPLACE INTO entity_embeddings "
                "(entity_id, embedding, model_name, dimension, bucket0, bucket1, bucket2, bucket3) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (entity_id, emb, self.embedding_engine.model_name, len(emb) // 4, *buckets),
            )

    def _embed_observation(
        self,
        observation_id: int,
        content: str,
        embedding: bytes | None = None,
        allow_fallback: bool = True,
    ) -> None:
        """Embed an observation's content and store in observation_embeddings."""
        if not self.embedding_engine or not self.embedding_engine.is_available():
            return
        emb = embedding
        if emb is None and allow_fallback:
            emb = self.embedding_engine.embed_text(content)
        if emb:
            buckets = embedding_buckets(emb)
            self.db.cx.execute(
                "INSERT OR REPLACE INTO observation_embeddings "
                "(observation_id, embedding, model_name, dimension, "
                "bucket0, bucket1, bucket2, bucket3) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (observation_id, emb, self.embedding_engine.model_name, len(emb) // 4, *buckets),
            )

    def _ensure_embeddings(self) -> None:
        """Schedule lazy backfill without turning read paths into synchronous writes."""
        if self._embeddings_synced:
            return
        if not self.embedding_engine or not self.embedding_engine.is_available():
            return
        if self._embedding_backfill_active():
            return
        self._schedule_embedding_backfill()

    def _write_embedding_deadline(self) -> float | None:
        """Return a monotonic deadline for optional write-path embeddings."""
        if self.write_embedding_budget_ms <= 0:
            return None
        return time.monotonic() + (self.write_embedding_budget_ms / 1000.0)

    @staticmethod
    def _write_embedding_time_remaining(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return max(deadline - time.monotonic(), 0.0)

    @staticmethod
    def _write_embedding_budget_exhausted(deadline: float | None) -> bool:
        return deadline is not None and time.monotonic() >= deadline

    def _embedding_backfill_key(self) -> tuple[str, str] | None:
        if not self.embedding_engine or not self.embedding_engine.is_available():
            return None
        return (self.db.db_path, self.embedding_engine.model_name)

    def _embedding_backfill_active(self) -> bool:
        key = self._embedding_backfill_key()
        if key is None:
            return False
        with _embedding_backfill_lock:
            return key in _active_embedding_backfills

    def _schedule_embedding_backfill(self) -> None:
        """Backfill skipped embeddings in the background using a separate DB connection."""
        key = self._embedding_backfill_key()
        if key is None:
            return

        with _embedding_backfill_lock:
            if key in _active_embedding_backfills:
                return
            _active_embedding_backfills.add(key)

        def run_backfill() -> None:
            db = Database(key[0])
            try:
                db.open()
                backfill_graph = KnowledgeGraphManager(
                    db,
                    session_id=self.session_id,
                    embedding_engine=EmbeddingEngine(model_name=key[1]),
                    project=self.project,
                    dedup_threshold=self.dedup_threshold,
                    write_embedding_budget_ms=0,
                )
                backfill_graph._backfill_missing_embeddings()
            except Exception as exc:  # pragma: no cover - logging only
                logger.warning("Background embedding backfill skipped: %s", exc)
            finally:
                db.close()
                with _embedding_backfill_lock:
                    _active_embedding_backfills.discard(key)

        threading.Thread(
            target=run_backfill,
            name="server-memory-embedding-backfill",
            daemon=True,
        ).start()

    def _backfill_missing_embeddings(self, batch_size: int = 128) -> int:
        """Fill missing entity and observation embeddings in bounded batches."""
        if not self.embedding_engine or not self.embedding_engine.is_available():
            return 0

        total_inserted = 0
        while True:
            cx = self.db.cx
            missing_entities = cx.execute(
                """
                SELECT e.id, e.name, e.entity_type FROM entities e
                LEFT JOIN entity_embeddings ee
                  ON ee.entity_id = e.id AND ee.model_name = ?
                WHERE e.deleted_at IS NULL AND ee.entity_id IS NULL
                LIMIT ?
                """,
                (self.embedding_engine.model_name, batch_size),
            ).fetchall()
            missing_obs = cx.execute(
                """
                SELECT o.id, o.content FROM observations o
                LEFT JOIN observation_embeddings oe
                  ON oe.observation_id = o.id AND oe.model_name = ?
                WHERE o.deleted_at IS NULL AND oe.observation_id IS NULL
                LIMIT ?
                """,
                (self.embedding_engine.model_name, batch_size),
            ).fetchall()

            if not missing_entities and not missing_obs:
                self._embeddings_synced = True
                return total_inserted

            entity_embeddings: list[tuple[int, bytes]] = []
            if missing_entities:
                entity_texts = [
                    f"{row['name']} {row['entity_type']}".strip() for row in missing_entities
                ]
                prepared_entity_embeddings, _ = self._prepare_embeddings(entity_texts)
                entity_embeddings = [
                    (row["id"], emb)
                    for row, emb in zip(missing_entities, prepared_entity_embeddings)
                    if emb
                ]

            observation_embeddings: list[tuple[int, bytes]] = []
            if missing_obs:
                observation_texts = [row["content"] for row in missing_obs]
                prepared_observation_embeddings, _ = self._prepare_embeddings(observation_texts)
                observation_embeddings = [
                    (row["id"], emb)
                    for row, emb in zip(missing_obs, prepared_observation_embeddings)
                    if emb
                ]

            if not entity_embeddings and not observation_embeddings:
                self._embeddings_synced = False
                return total_inserted

            with self.db.transaction() as cx:
                for entity_id, emb in entity_embeddings:
                    buckets = embedding_buckets(emb)
                    cx.execute(
                        "INSERT OR REPLACE INTO entity_embeddings "
                        "(entity_id, embedding, model_name, dimension, "
                        "bucket0, bucket1, bucket2, bucket3) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (entity_id, emb, self.embedding_engine.model_name, len(emb) // 4, *buckets),
                    )
                for observation_id, emb in observation_embeddings:
                    buckets = embedding_buckets(emb)
                    cx.execute(
                        "INSERT OR REPLACE INTO observation_embeddings "
                        "(observation_id, embedding, model_name, dimension, "
                        "bucket0, bucket1, bucket2, bucket3) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            observation_id,
                            emb,
                            self.embedding_engine.model_name,
                            len(emb) // 4,
                            *buckets,
                        ),
                    )
            total_inserted += len(entity_embeddings) + len(observation_embeddings)

    def _bfs_expand(self, start_ids: set[int], depth: int) -> set[int]:
        """BFS expansion from start entities to given depth."""
        cx = self.db.cx
        visited = set(start_ids)
        frontier = set(start_ids)

        for _ in range(depth):
            if not frontier:
                break
            placeholders = ",".join("?" for _ in frontier)
            ids = list(frontier)
            rows = cx.execute(
                f"""
                SELECT DISTINCT from_entity_id, to_entity_id FROM relations
                WHERE deleted_at IS NULL AND (
                    from_entity_id IN ({placeholders}) OR
                    to_entity_id IN ({placeholders})
                )
                """,
                ids + ids,
            ).fetchall()
            next_frontier: set[int] = set()
            for r in rows:
                for eid in (r["from_entity_id"], r["to_entity_id"]):
                    if eid not in visited:
                        next_frontier.add(eid)
                        visited.add(eid)
            frontier = next_frontier

        return visited

    def _touch_entities_accessed(self, entity_ids: list[int]) -> None:
        """Best-effort access tracking that avoids turning read paths into lock hotspots."""
        if not entity_ids:
            return

        cx = self.db.cx
        placeholders = ",".join("?" for _ in entity_ids)
        try:
            res = cx.execute(
                f"""
                UPDATE entities
                SET last_accessed_at = ?
                WHERE id IN ({placeholders})
                  AND (
                    last_accessed_at IS NULL OR
                    last_accessed_at < strftime('%Y-%m-%dT%H:%M:%fZ', 'now', ?)
                  )
                """,
                (_now_iso(), *entity_ids, f"-{self.ACCESS_TOUCH_INTERVAL_HOURS} hours"),
            )
            if res.rowcount:
                cx.commit()
        except sqlite3.OperationalError:
            cx.rollback()
