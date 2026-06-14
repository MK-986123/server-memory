"""Tests for KnowledgeGraphManager CRUD operations."""

import sqlite3
import tempfile
import threading
import time
from pathlib import Path

from server_memory.db import Database
from server_memory.embeddings import EmbeddingEngine
from server_memory.graph import KnowledgeGraphManager


class SlowMockEmbeddingEngine(EmbeddingEngine):
    """Mock embedding engine that intentionally sleeps to simulate model latency."""

    def __init__(self, delay_seconds: float = 0.2):
        super().__init__("slow-mock")
        self._available = True
        self.delay_seconds = delay_seconds

    def is_available(self) -> bool:
        return True

    def _get_model(self):
        return None

    def embed_text(self, text: str) -> bytes | None:
        time.sleep(self.delay_seconds)
        return text.encode("utf-8") or b"x"

    def embed_batch(self, texts: list[str]) -> list[bytes]:
        return [self.embed_text(text) for text in texts]


class CountingSlowSimilarityEmbeddingEngine(EmbeddingEngine):
    """Cheap embeddings with intentionally slow cosine similarity checks."""

    def __init__(self, similarity_delay_seconds: float = 0.01):
        super().__init__("counting-slow")
        self._available = True
        self.similarity_delay_seconds = similarity_delay_seconds
        self.comparison_count = 0

    def is_available(self) -> bool:
        return True

    def _get_model(self):
        return None

    def embed_text(self, text: str) -> bytes | None:
        return text.encode("utf-8") or b"x"

    def embed_batch(self, texts: list[str]) -> list[bytes]:
        return [self.embed_text(text) for text in texts]

    def cosine_similarity(self, a: bytes, b: bytes) -> float:
        self.comparison_count += 1
        time.sleep(self.similarity_delay_seconds)
        return 0.0


def test_create_entities(graph):
    created = graph.create_entities(
        [
            {"name": "App", "entityType": "project", "observations": ["Built with Python"]},
            {"name": "DB", "entityType": "component"},
        ]
    )
    assert len(created) == 2
    assert created[0].name == "App"
    assert created[0].entity_type == "project"
    assert len(created[0].observations) == 1
    assert created[0].observations[0].content == "Built with Python"


def test_create_entities_skip_duplicates(graph):
    graph.create_entities([{"name": "App", "entityType": "project"}])
    created = graph.create_entities(
        [
            {"name": "App", "entityType": "project"},
            {"name": "New", "entityType": "feature"},
        ]
    )
    assert len(created) == 1
    assert created[0].name == "New"


def test_create_entities_skip_duplicates_releases_transaction_lock():
    db_path = Path(tempfile.mkdtemp()) / "memory.db"

    first_db = Database(db_path)
    first_db.open()
    second_db = Database(db_path)
    second_db.open()
    try:
        first_graph = KnowledgeGraphManager(first_db)
        second_graph = KnowledgeGraphManager(second_db)

        first_graph.create_entities([{"name": "App", "entityType": "project"}])
        created = first_graph.create_entities(
            [
                {"name": "App", "entityType": "project"},
                {"name": "New", "entityType": "feature"},
            ]
        )

        assert len(created) == 1
        assert created[0].name == "New"

        second_created = second_graph.create_entities(
            [
                {"name": "Other", "entityType": "feature"},
            ]
        )
        assert len(second_created) == 1
        assert second_created[0].name == "Other"
    finally:
        second_db.close()
        first_db.close()


def test_create_entities_raises_when_database_is_locked():
    db_path = Path(tempfile.mkdtemp()) / "memory.db"

    blocked_db = Database(db_path)
    blocked_db.open()
    blocked_db.cx.execute("PRAGMA busy_timeout=50")
    locker = sqlite3.connect(db_path, check_same_thread=False)
    locker.execute("BEGIN EXCLUSIVE")
    try:
        blocked_graph = KnowledgeGraphManager(blocked_db)
        try:
            blocked_graph.create_entities([{"name": "Blocked", "entityType": "note"}])
            assert False, "Expected create_entities to surface the lock error"
        except sqlite3.OperationalError as exc:
            assert "locked" in str(exc).lower()
    finally:
        blocked_db.close()
        locker.rollback()
        locker.close()


def test_create_entities_retries_until_lock_is_released():
    db_path = Path(tempfile.mkdtemp()) / "memory.db"

    blocked_db = Database(db_path)
    blocked_db.open()
    blocked_db.cx.execute("PRAGMA busy_timeout=50")
    locker = sqlite3.connect(db_path, check_same_thread=False)
    locker.execute("BEGIN EXCLUSIVE")

    def release_lock() -> None:
        time.sleep(0.2)
        locker.rollback()
        locker.close()

    releaser = threading.Thread(target=release_lock)
    releaser.start()

    try:
        blocked_graph = KnowledgeGraphManager(blocked_db)
        created = blocked_graph.create_entities(
            [
                {"name": "Recovered", "entityType": "note"},
            ]
        )
        assert len(created) == 1
        assert created[0].name == "Recovered"
    finally:
        releaser.join()
        blocked_db.close()


def test_memory_context_succeeds_while_another_client_holds_write_lock():
    db_path = Path(tempfile.mkdtemp()) / "memory.db"

    writer_db = Database(db_path)
    writer_db.open()
    writer_graph = KnowledgeGraphManager(writer_db)
    writer_graph.create_entities(
        [
            {
                "name": "Lock Notes",
                "entityType": "note",
                "observations": ["sqlite database lock mitigation for codex cli"],
            },
        ]
    )

    reader_db = Database(db_path)
    reader_db.open()
    reader_graph = KnowledgeGraphManager(reader_db)

    locker = sqlite3.connect(db_path, check_same_thread=False)
    locker.execute("BEGIN IMMEDIATE")
    try:
        ctx = reader_graph.memory_context(hint="sqlite database lock codex cli")
        assert ctx["hint_matches"]
        assert ctx["hint_matches"][0]["name"] == "Lock Notes"
    finally:
        locker.rollback()
        locker.close()
        reader_db.close()
        writer_db.close()


def test_memory_context_with_embeddings_missing_stays_read_only_under_write_lock(monkeypatch):
    db_path = Path(tempfile.mkdtemp()) / "memory.db"

    writer_db = Database(db_path)
    writer_db.open()
    writer_graph = KnowledgeGraphManager(writer_db)
    writer_graph.create_entities(
        [
            {
                "name": "Embedding Lock Notes",
                "entityType": "note",
                "observations": ["sqlite database lock mitigation for codex cli"],
            },
        ]
    )

    reader_db = Database(db_path)
    reader_db.open()
    reader_graph = KnowledgeGraphManager(
        reader_db,
        embedding_engine=SlowMockEmbeddingEngine(delay_seconds=0.01),
    )
    scheduled: list[str] = []
    monkeypatch.setattr(
        reader_graph,
        "_schedule_embedding_backfill",
        lambda: scheduled.append("scheduled"),
    )

    locker = sqlite3.connect(db_path, check_same_thread=False)
    locker.execute("BEGIN IMMEDIATE")
    try:
        ctx = reader_graph.memory_context(hint="sqlite database lock codex cli")
        assert scheduled == ["scheduled"]
        assert ctx["hint_matches"]
        assert ctx["hint_matches"][0]["name"] == "Embedding Lock Notes"
        assert writer_db.cx.execute("SELECT COUNT(*) AS c FROM entity_embeddings").fetchone()["c"] == 0
    finally:
        locker.rollback()
        locker.close()
        reader_db.close()
        writer_db.close()


def test_slow_embedding_work_does_not_hold_write_lock_for_other_clients():
    db_path = Path(tempfile.mkdtemp()) / "memory.db"

    fast_db = Database(db_path)
    fast_db.open()
    fast_db.cx.execute("PRAGMA busy_timeout=50")

    fast_graph = KnowledgeGraphManager(fast_db)

    slow_error: list[Exception] = []
    slow_started = threading.Event()

    def create_slow_entity() -> None:
        slow_db = Database(db_path)
        slow_db.open()
        try:
            slow_graph = KnowledgeGraphManager(
                slow_db,
                embedding_engine=SlowMockEmbeddingEngine(delay_seconds=0.2),
            )
            slow_started.set()
            slow_graph.create_entities(
                [
                    {
                        "name": "SlowEntity",
                        "entityType": "project",
                        "observations": ["embedding-heavy entity"],
                    }
                ]
            )
        except Exception as exc:  # pragma: no cover - assertion happens below
            slow_error.append(exc)
        finally:
            slow_db.close()

    worker = threading.Thread(target=create_slow_entity)
    worker.start()

    try:
        assert slow_started.wait(timeout=1)
        time.sleep(0.05)
        created = fast_graph.create_entities(
            [
                {"name": "FastEntity", "entityType": "project"},
            ]
        )

        worker.join()

        assert not slow_error
        assert len(created) == 1
        assert created[0].name == "FastEntity"
        names = {
            row["name"]
            for row in fast_db.cx.execute(
                "SELECT name FROM entities WHERE name IN ('SlowEntity', 'FastEntity')"
            ).fetchall()
        }
        assert names == {"SlowEntity", "FastEntity"}
    finally:
        worker.join()
        fast_db.close()


def test_create_entities_respects_write_embedding_budget(monkeypatch):
    db_path = Path(tempfile.mkdtemp()) / "memory.db"

    db = Database(db_path)
    db.open()
    try:
        graph = KnowledgeGraphManager(
            db,
            embedding_engine=SlowMockEmbeddingEngine(delay_seconds=0.05),
            write_embedding_budget_ms=10,
        )
        scheduled: list[str] = []
        monkeypatch.setattr(
            graph, "_schedule_embedding_backfill", lambda: scheduled.append("scheduled")
        )

        start = time.perf_counter()
        created = graph.create_entities(
            [
                {
                    "name": "BudgetedEntity",
                    "entityType": "note",
                    "observations": ["obs1", "obs2", "obs3"],
                }
            ]
        )
        elapsed = time.perf_counter() - start

        assert len(created) == 1
        assert created[0].name == "BudgetedEntity"
        assert elapsed < 0.15
        assert scheduled == ["scheduled"]
        assert db.cx.execute("SELECT COUNT(*) AS c FROM entity_embeddings").fetchone()["c"] == 0
        assert (
            db.cx.execute("SELECT COUNT(*) AS c FROM observation_embeddings").fetchone()["c"] == 0
        )
    finally:
        db.close()


def test_add_observations_limits_semantic_dedup_under_budget():
    db_path = Path(tempfile.mkdtemp()) / "memory.db"

    db = Database(db_path)
    db.open()
    try:
        engine = CountingSlowSimilarityEmbeddingEngine(similarity_delay_seconds=0.01)
        graph = KnowledgeGraphManager(db, embedding_engine=engine, write_embedding_budget_ms=0)
        graph.create_entities([{"name": "BudgetedObs", "entityType": "note"}])
        graph.add_observations(
            [
                {
                    "entityName": "BudgetedObs",
                    "contents": [f"existing-{index}" for index in range(20)],
                }
            ]
        )

        engine.comparison_count = 0
        graph.write_embedding_budget_ms = 20

        start = time.perf_counter()
        result = graph.add_observations(
            [
                {
                    "entityName": "BudgetedObs",
                    "contents": [f"new-{index}" for index in range(5)],
                }
            ]
        )
        elapsed = time.perf_counter() - start

        assert len(result) == 1
        assert result[0]["addedObservations"] == [f"new-{index}" for index in range(5)]
        assert elapsed < 0.15
        assert engine.comparison_count < 100
    finally:
        db.close()


def test_backfill_missing_embeddings_restores_budget_skipped_rows(monkeypatch):
    db_path = Path(tempfile.mkdtemp()) / "memory.db"

    db = Database(db_path)
    db.open()
    try:
        slow_graph = KnowledgeGraphManager(
            db,
            embedding_engine=SlowMockEmbeddingEngine(delay_seconds=0.05),
            write_embedding_budget_ms=10,
        )
        monkeypatch.setattr(slow_graph, "_schedule_embedding_backfill", lambda: None)

        slow_graph.create_entities(
            [
                {
                    "name": "NeedsBackfill",
                    "entityType": "note",
                    "observations": ["obs1", "obs2"],
                }
            ]
        )

        assert db.cx.execute("SELECT COUNT(*) AS c FROM entity_embeddings").fetchone()["c"] == 0
        assert (
            db.cx.execute("SELECT COUNT(*) AS c FROM observation_embeddings").fetchone()["c"] == 0
        )

        fast_graph = KnowledgeGraphManager(
            db,
            embedding_engine=CountingSlowSimilarityEmbeddingEngine(similarity_delay_seconds=0.0),
            write_embedding_budget_ms=0,
        )
        inserted = fast_graph._backfill_missing_embeddings()

        assert inserted == 3
        assert db.cx.execute("SELECT COUNT(*) AS c FROM entity_embeddings").fetchone()["c"] == 1
        assert (
            db.cx.execute("SELECT COUNT(*) AS c FROM observation_embeddings").fetchone()["c"] == 2
        )
    finally:
        db.close()


def test_create_entities_with_tags(graph):
    created = graph.create_entities(
        [
            {"name": "Important", "entityType": "note", "tags": ["pinned", "preference"]},
        ]
    )
    assert len(created) == 1
    assert "pinned" in created[0].tags
    assert "preference" in created[0].tags


def test_soft_delete_entity(graph):
    graph.create_entities([{"name": "ToDelete", "entityType": "test"}])
    count = graph.delete_entities(["ToDelete"])
    assert count == 1
    # Should not appear in normal reads
    kg = graph.read_graph()
    assert all(e.name != "ToDelete" for e in kg.entities)
    # Should appear with include_deleted
    kg = graph.read_graph(include_deleted=True)
    assert any(e.name == "ToDelete" for e in kg.entities)


def test_hard_delete_entity(graph):
    graph.create_entities([{"name": "Gone", "entityType": "test"}])
    count = graph.delete_entities(["Gone"], hard=True)
    assert count == 1
    kg = graph.read_graph(include_deleted=True)
    assert all(e.name != "Gone" for e in kg.entities)


def test_restore_entity(graph):
    graph.create_entities([{"name": "Restore", "entityType": "test"}])
    graph.delete_entities(["Restore"])
    count = graph.restore_entities(["Restore"])
    assert count == 1
    kg = graph.read_graph()
    assert any(e.name == "Restore" for e in kg.entities)


def test_create_relations(graph):
    graph.create_entities(
        [
            {"name": "A", "entityType": "node"},
            {"name": "B", "entityType": "node"},
        ]
    )
    created = graph.create_relations(
        [
            {"from": "A", "to": "B", "relationType": "depends_on"},
        ]
    )
    assert len(created) == 1
    assert created[0].from_name == "A"
    assert created[0].to_name == "B"
    assert created[0].relation_type == "depends_on"


def test_create_relation_missing_entity(graph):
    graph.create_entities([{"name": "A", "entityType": "node"}])
    try:
        graph.create_relations([{"from": "A", "to": "Missing", "relationType": "uses"}])
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "Missing" in str(e)


def test_delete_relation(graph):
    graph.create_entities(
        [
            {"name": "X", "entityType": "node"},
            {"name": "Y", "entityType": "node"},
        ]
    )
    graph.create_relations([{"from": "X", "to": "Y", "relationType": "calls"}])
    count = graph.delete_relations([{"from": "X", "to": "Y", "relationType": "calls"}])
    assert count == 1


def test_soft_delete_cascades_relations(graph):
    graph.create_entities(
        [
            {"name": "P", "entityType": "parent"},
            {"name": "C", "entityType": "child"},
        ]
    )
    graph.create_relations([{"from": "P", "to": "C", "relationType": "owns"}])
    graph.delete_entities(["P"])
    # Relations should also be soft-deleted
    kg = graph.read_graph()
    assert len(kg.relations) == 0


def test_add_observations(graph):
    graph.create_entities([{"name": "E", "entityType": "test"}])
    results = graph.add_observations(
        [
            {"entityName": "E", "contents": ["obs1", "obs2"]},
        ]
    )
    assert len(results) == 1
    assert len(results[0]["addedObservations"]) == 2


def test_add_observations_dedup(graph):
    graph.create_entities([{"name": "E", "entityType": "test", "observations": ["existing"]}])
    results = graph.add_observations(
        [
            {"entityName": "E", "contents": ["existing", "new"]},
        ]
    )
    assert results[0]["addedObservations"] == ["new"]


def test_add_observations_missing_entity(graph):
    try:
        graph.add_observations([{"entityName": "Ghost", "contents": ["data"]}])
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_add_observations_embeds_each_new_observation_once_even_with_multiple_tags(monkeypatch):
    db = Database(":memory:")
    db.open()
    try:
        graph = KnowledgeGraphManager(db, embedding_engine=CountingSlowSimilarityEmbeddingEngine())
        graph.create_entities([{"name": "E", "entityType": "test"}])

        embed_calls: list[tuple[int, str]] = []

        def record_embed(
            observation_id: int,
            content: str,
            embedding: bytes | None = None,
            allow_fallback: bool = True,
        ) -> None:
            embed_calls.append((observation_id, content))

        monkeypatch.setattr(graph, "_embed_observation", record_embed)

        results = graph.add_observations(
            [
                {
                    "entityName": "E",
                    "contents": ["obs1"],
                    "tags": ["alpha", "beta", "gamma"],
                }
            ]
        )

        assert results[0]["addedObservations"] == ["obs1"]
        assert len(embed_calls) == 1
        assert embed_calls[0][1] == "obs1"
    finally:
        db.close()


def test_delete_observations(graph):
    graph.create_entities([{"name": "E", "entityType": "test", "observations": ["a", "b", "c"]}])
    count = graph.delete_observations([{"entityName": "E", "observations": ["b"]}])
    assert count == 1
    entity = graph.get_entity_by_name("E")
    assert entity is not None
    contents = [o.content for o in entity.observations]
    assert "b" not in contents
    assert "a" in contents
    assert "c" in contents


def test_update_observation(graph):
    graph.create_entities([{"name": "E", "entityType": "test", "observations": ["old text"]}])
    ok = graph.update_observation("E", "old text", "new text")
    assert ok
    entity = graph.get_entity_by_name("E")
    assert entity is not None
    assert entity.observations[0].content == "new text"
    assert entity.observations[0].version == 2
    # Check history
    rows = graph.db.cx.execute("SELECT * FROM observation_history").fetchall()
    assert len(rows) == 1
    assert rows[0]["content"] == "old text"


def test_read_graph_filter_tags(graph):
    graph.create_entities(
        [
            {"name": "P1", "entityType": "project", "tags": ["pinned"]},
            {"name": "P2", "entityType": "project"},
        ]
    )
    kg = graph.read_graph(tags=["pinned"])
    assert len(kg.entities) == 1
    assert kg.entities[0].name == "P1"


def test_read_graph_filter_types(graph):
    graph.create_entities(
        [
            {"name": "A", "entityType": "project"},
            {"name": "B", "entityType": "person"},
        ]
    )
    kg = graph.read_graph(entity_types=["person"])
    assert len(kg.entities) == 1
    assert kg.entities[0].name == "B"


def test_tag_management(graph):
    tags = graph.list_tags()
    assert len(tags) > 0  # System tags exist

    tag = graph.create_tag("custom", "my tag", "#000")
    assert tag.name == "custom"

    graph.create_entities([{"name": "E", "entityType": "test"}])
    ok = graph.tag_entity("E", "custom")
    assert ok

    entity = graph.get_entity_by_name("E")
    assert "custom" in entity.tags

    ok = graph.untag_entity("E", "custom")
    assert ok
    entity = graph.get_entity_by_name("E")
    assert "custom" not in entity.tags

    ok = graph.delete_tag("custom")
    assert ok


def test_cannot_delete_system_tag(graph):
    try:
        graph.delete_tag("pinned")
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_merge_entities_combine(graph):
    graph.create_entities(
        [
            {"name": "Source", "entityType": "dup", "observations": ["obs1"]},
            {"name": "Target", "entityType": "main", "observations": ["obs2"]},
        ]
    )
    graph.create_relations([{"from": "Source", "to": "Target", "relationType": "related"}])
    result = graph.merge_entities("Source", "Target", strategy="combine")
    assert result is not None
    assert result.name == "Target"
    contents = [o.content for o in result.observations]
    assert "obs1" in contents
    assert "obs2" in contents
    # Source should be soft-deleted
    assert graph.get_entity_by_name("Source") is None


def test_merge_entities_dedupe(graph):
    graph.create_entities(
        [
            {"name": "S", "entityType": "dup", "observations": ["shared", "unique"]},
            {"name": "T", "entityType": "main", "observations": ["shared"]},
        ]
    )
    result = graph.merge_entities("S", "T", strategy="dedupe")
    assert result is not None
    contents = [o.content for o in result.observations]
    assert "unique" in contents
    assert contents.count("shared") == 1


def test_load_entities_batch(graph):
    """Batch loading should return the same entities as individual loading."""
    graph.create_entities(
        [
            {
                "name": "B1",
                "entityType": "test",
                "observations": ["obs1", "obs2"],
                "tags": ["pinned"],
            },
            {"name": "B2", "entityType": "test", "observations": ["obs3"]},
            {"name": "B3", "entityType": "test"},
        ]
    )
    # Get IDs
    ids = set()
    for name in ["B1", "B2", "B3"]:
        e = graph.get_entity_by_name(name)
        assert e is not None
        ids.add(e.id)

    entities = graph._load_entities_batch(ids)
    assert len(entities) == 3
    name_map = {e.name: e for e in entities}
    assert "B1" in name_map
    assert "B2" in name_map
    assert "B3" in name_map
    assert len(name_map["B1"].observations) == 2
    assert len(name_map["B2"].observations) == 1
    assert len(name_map["B3"].observations) == 0
    assert "pinned" in name_map["B1"].tags


def test_load_entities_batch_empty(graph):
    """Batch loading empty set should return empty list."""
    assert graph._load_entities_batch(set()) == []


def test_read_graph_uses_batch(graph):
    """read_graph should work correctly with batch loading."""
    graph.create_entities(
        [
            {"name": "R1", "entityType": "project", "observations": ["data1"]},
            {"name": "R2", "entityType": "project", "observations": ["data2"]},
        ]
    )
    graph.create_relations([{"from": "R1", "to": "R2", "relationType": "uses"}])
    kg = graph.read_graph()
    assert len(kg.entities) == 2
    assert len(kg.relations) == 1
