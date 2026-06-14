"""Tests for activity logging, timeline, memory context, import/export, stats, and backup."""

import json
import os
import tempfile
from types import SimpleNamespace

import pytest

from server_memory import server as server_module
from server_memory.config import MemoryConfig
from server_memory.db import Database
from server_memory.graph import KnowledgeGraphManager


def _tool_ctx(
    workspace_graph: KnowledgeGraphManager,
    config: MemoryConfig,
    global_graph: KnowledgeGraphManager | None = None,
):
    return SimpleNamespace(
        request_context=SimpleNamespace(
            lifespan_context={
                "graph": workspace_graph,
                "global_graph": global_graph,
                "config": config,
            }
        )
    )


def _tool_fn(name: str):
    server = server_module.create_server(MemoryConfig(global_db_enabled=False))
    return server._tool_manager._tools[name].fn


def test_log_activity(graph):
    graph.create_entities([{"name": "server.py", "entityType": "file"}])
    entry = graph.log_activity(
        action="file_changed",
        summary="Updated error handling",
        entity_names=["server.py"],
    )
    assert entry.action == "file_changed"
    assert entry.summary == "Updated error handling"
    # Auto-tag check
    assert "recent-change" in entry.tags


def test_log_activity_auto_create(graph):
    entry = graph.log_activity(
        action="discussed",
        summary="Talked about new feature",
        entity_names=["new-feature"],
        auto_create_entities=True,
    )
    assert entry.id > 0
    # Entity should have been auto-created
    entity = graph.get_entity_by_name("new-feature")
    assert entity is not None
    assert entity.entity_type == "auto"


def test_query_timeline(graph):
    graph.log_activity(action="file_changed", summary="change 1")
    graph.log_activity(action="bug_fixed", summary="fix 1")
    graph.log_activity(action="file_changed", summary="change 2")

    entries = graph.query_timeline(actions=["file_changed"])
    assert len(entries) == 2

    entries = graph.query_timeline(actions=["bug_fixed"])
    assert len(entries) == 1


def test_query_timeline_relative(graph):
    graph.log_activity(action="test", summary="recent")
    entries = graph.query_timeline(time_range="1h")
    assert len(entries) >= 1


def test_query_timeline_entity_filter_uses_json_membership_not_substring(graph):
    graph.create_entities(
        [
            {"name": f"Entity {index}", "entityType": "note"}
            for index in range(1, 11)
        ]
    )
    graph.log_activity(
        action="file_changed",
        summary="linked only to entity 10",
        entity_names=["Entity 10"],
    )

    entries = graph.query_timeline(entity_name="Entity 1")

    assert entries == []
    assert graph.query_timeline(entity_name="Entity 10")


def test_memory_context_empty(graph):
    ctx = graph.memory_context()
    assert ctx["stats"]["entities"] == 0
    assert ctx["pinned"] == []
    assert ctx["recent_activity"] == []


def test_memory_context_with_data(graph):
    graph.create_entities(
        [
            {
                "name": "Important",
                "entityType": "note",
                "tags": ["pinned"],
                "observations": ["key info"],
            },
            {"name": "Regular", "entityType": "note"},
        ]
    )
    graph.log_activity(action="test", summary="did something")

    ctx = graph.memory_context(hint="note")
    assert len(ctx["pinned"]) == 1
    assert ctx["pinned"][0]["name"] == "Important"
    assert len(ctx["recent_activity"]) == 1
    assert ctx["stats"]["entities"] == 2


def test_memory_context_hint_search(graph):
    graph.create_entities(
        [
            {"name": "Auth Module", "entityType": "module", "observations": ["handles JWT tokens"]},
            {"name": "DB Module", "entityType": "module", "observations": ["manages connections"]},
        ]
    )
    ctx = graph.memory_context(hint="JWT")
    hint_names = [h["name"] for h in ctx["hint_matches"]]
    assert "Auth Module" in hint_names


def test_memory_context_prefers_exact_hint_match(graph):
    graph.create_entities(
        [
            {
                "name": "JWT Config",
                "entityType": "config",
                "observations": ["token handling details"],
            },
            {
                "name": "Auth Notes",
                "entityType": "note",
                "observations": ["JWT tokens and settings"],
            },
        ]
    )

    ctx = graph.memory_context(hint="JWT Config")

    assert ctx["hint_matches"]
    assert ctx["hint_matches"][0]["name"] == "JWT Config"


def test_memory_context_prefers_pinned_fact_over_recent_noise(graph):
    graph.create_entities(
        [
            {
                "name": "Critical Config",
                "entityType": "config",
                "tags": ["pinned"],
                "observations": ["provider x requires strict validation"],
            },
            {
                "name": "Recent Chatter",
                "entityType": "note",
                "observations": ["provider x came up in passing"],
            },
        ]
    )
    graph.log_activity(action="discussed", summary="provider x came up again")
    graph.db.cx.execute(
        "UPDATE entities SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '+2 minutes') WHERE name = ?",
        ("Recent Chatter",),
    )
    graph.db.cx.execute(
        "UPDATE observations SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '+2 minutes') "
        "WHERE entity_id = (SELECT id FROM entities WHERE name = ?)",
        ("Recent Chatter",),
    )
    graph.db.cx.execute(
        "UPDATE entities SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-2 minutes') WHERE name = ?",
        ("Critical Config",),
    )
    graph.db.cx.execute(
        "UPDATE observations SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-2 minutes') "
        "WHERE entity_id = (SELECT id FROM entities WHERE name = ?)",
        ("Critical Config",),
    )
    graph.db.cx.commit()

    ctx = graph.memory_context(hint="provider x")

    assert ctx["hint_matches"]
    assert ctx["hint_matches"][0]["name"] == "Critical Config"


def test_memory_context_hint_filters_unrelated_global_pinned_and_recent_activity(graph):
    graph.create_entities(
        [
            {
                "name": "Global GPU Notes",
                "entityType": "note",
                "tags": ["pinned"],
                "observations": ["flux gpu dtype fix and kernel tuning notes"],
            },
            {
                "name": "SQLite Lock Runbook",
                "entityType": "doc",
                "observations": ["sqlite database lock mitigation for codex cli"],
            },
        ]
    )
    graph.log_activity(
        action="bug_fixed",
        summary="flux gpu dtype fix was validated",
        entity_names=["Global GPU Notes"],
    )
    graph.log_activity(
        action="investigated",
        summary="sqlite database lock mitigation for codex cli was reviewed",
        entity_names=["SQLite Lock Runbook"],
    )

    ctx = graph.memory_context(hint="sqlite database lock codex cli")

    assert ctx["pinned"] == []
    assert ctx["recent_activity"]
    assert all("sqlite" in item["summary"].lower() for item in ctx["recent_activity"])
    assert ctx["hint_matches"]
    assert ctx["hint_matches"][0]["name"] == "SQLite Lock Runbook"


def test_memory_context_deduplicates_near_identical_matches(graph):
    graph.create_entities(
        [
            {
                "name": "Auth Canonical",
                "entityType": "config",
                "observations": ["refresh token"],
            },
            {
                "name": "Auth Duplicate",
                "entityType": "config",
                "observations": ["refresh token"],
            },
        ]
    )

    ctx = graph.memory_context(hint="refresh token")
    hint_names = [h["name"] for h in ctx["hint_matches"]]

    assert len(hint_names) == 1
    assert hint_names[0] in {"Auth Canonical", "Auth Duplicate"}


def test_memory_context_uses_matching_observation_quality_not_entity_max_importance(graph):
    graph.create_entities(
        [
            {"name": "Rotation Policy", "entityType": "config"},
            {"name": "Noisy Notes", "entityType": "note"},
        ]
    )
    graph.add_observations(
        [
            {
                "entityName": "Rotation Policy",
                "contents": ["rotation window is 15 minutes for key rollover"],
                "importance": 0.9,
                "confidence": 0.95,
                "obs_type": "config",
            },
            {
                "entityName": "Noisy Notes",
                "contents": ["rotation window might be 30 minutes"],
                "importance": 0.1,
                "confidence": 0.2,
            },
            {
                "entityName": "Noisy Notes",
                "contents": ["critical launch checklist lives elsewhere"],
                "importance": 1.0,
                "confidence": 0.95,
            },
        ]
    )

    ctx = graph.memory_context(hint="rotation window")

    assert ctx["hint_matches"]
    assert ctx["hint_matches"][0]["name"] == "Rotation Policy"


def test_memory_context_uses_recent_access_as_tiebreaker(graph):
    graph.create_entities(
        [
            {
                "name": "Active Runbook",
                "entityType": "doc",
                "observations": ["provider x auth flow details"],
            },
            {
                "name": "Stale Runbook",
                "entityType": "doc",
                "observations": ["provider x auth flow details"],
            },
        ]
    )
    graph.db.cx.execute(
        "UPDATE entities SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-2 minutes'), "
        "last_accessed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE name = ?",
        ("Active Runbook",),
    )
    graph.db.cx.execute(
        "UPDATE entities SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), "
        "last_accessed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-30 days') WHERE name = ?",
        ("Stale Runbook",),
    )
    graph.db.cx.commit()

    ctx = graph.memory_context(hint="provider x auth flow")

    assert ctx["hint_matches"]
    assert ctx["hint_matches"][0]["name"] == "Active Runbook"


def test_memory_context_boosts_recent_activity_links(graph):
    graph.create_entities(
        [
            {
                "name": "Incident Runbook",
                "entityType": "doc",
                "observations": ["provider x rollback procedure"],
            },
            {
                "name": "Generic Notes",
                "entityType": "note",
                "observations": ["provider x rollback procedure"],
            },
        ]
    )
    graph.log_activity(
        action="decision_made",
        summary="provider x rollback was reviewed in the incident runbook",
        entity_names=["Incident Runbook"],
    )
    graph.db.cx.execute(
        "UPDATE entities SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '+1 minute') WHERE name = ?",
        ("Generic Notes",),
    )
    graph.db.cx.commit()

    ctx = graph.memory_context(hint="provider x rollback")

    assert ctx["hint_matches"]
    assert ctx["hint_matches"][0]["name"] == "Incident Runbook"


def test_memory_context_boosts_file_path_observations_for_path_hints(graph):
    graph.create_entities(
        [
            {"name": "Canonical Path", "entityType": "file"},
            {"name": "Discussion Notes", "entityType": "note"},
        ]
    )
    graph.add_observations(
        [
            {
                "entityName": "Canonical Path",
                "contents": ["/srv/app/config/auth.yaml"],
                "importance": 0.5,
                "confidence": 0.95,
                "obs_type": "file_path",
            },
            {
                "entityName": "Discussion Notes",
                "contents": ["/srv/app/config/auth.yaml was discussed during triage"],
                "importance": 0.9,
                "confidence": 0.95,
            },
        ]
    )

    ctx = graph.memory_context(hint="/srv/app/config/auth.yaml")

    assert ctx["hint_matches"]
    assert ctx["hint_matches"][0]["name"] == "Canonical Path"


def test_memory_context_honors_requested_hint_limit(graph):
    graph.create_entities(
        [
            {
                "name": f"Shared Match {index}",
                "entityType": "note",
                "observations": [f"shared-topic detail {index}"],
            }
            for index in range(7)
        ]
    )

    ctx = graph.memory_context(hint="shared-topic", limit=6)

    assert len(ctx["hint_matches"]) == 6


def test_memory_context_returns_formatter_metadata(graph):
    graph.create_entities(
        [
            {
                "name": "JWT Config",
                "entityType": "config",
                "tags": ["pinned"],
                "observations": ["JWT issuer is auth-prod"],
            },
        ]
    )

    ctx = graph.memory_context(hint="JWT Config")

    assert ctx["hint_matches"]
    match = ctx["hint_matches"][0]
    assert match["score"] > 0
    assert match["snippets"] == ["JWT issuer is auth-prod"]
    assert match["conflict"] is False
    assert match["stale"] is False
    assert "exact_name" in match["signals"]
    assert "pinned" in match["signals"]


def test_merge_memory_context_results_improves_recall_when_global_has_the_only_hit():
    workspace_result = {
        "pinned": [],
        "recent_activity": [],
        "hint_matches": [],
        "stats": {"entities": 1, "observations": 0, "relations": 0},
    }
    global_result = {
        "pinned": [],
        "recent_activity": [],
        "hint_matches": [
            {
                "name": "Editor Preference",
                "type": "preference",
                "snippets": ["prefer ruff over flake8"],
                "score": 0.7,
                "conflict": False,
                "stale": False,
                "signals": ["lexical"],
            }
        ],
        "stats": {"entities": 1, "observations": 1, "relations": 0},
    }

    merged = server_module.merge_memory_context_results(
        workspace_result,
        global_result,
        limit=5,
    )

    assert workspace_result["hint_matches"] == []
    assert merged["hint_matches"]
    assert merged["hint_matches"][0]["name"] == "Editor Preference"
    assert merged["hint_matches"][0]["source"] == "global"


def test_merge_memory_context_results_prefers_workspace_hit_over_equal_global_hit():
    workspace_result = {
        "pinned": [],
        "recent_activity": [],
        "hint_matches": [
            {
                "name": "Workspace JWT Config",
                "type": "config",
                "snippets": ["JWT issuer is workspace-prod"],
                "score": 0.6,
                "conflict": False,
                "stale": False,
                "signals": ["lexical"],
            }
        ],
        "stats": {"entities": 1, "observations": 1, "relations": 0},
    }
    global_result = {
        "pinned": [],
        "recent_activity": [],
        "hint_matches": [
            {
                "name": "Global JWT Preference",
                "type": "preference",
                "snippets": ["JWT issuer is global-prod"],
                "score": 0.6,
                "conflict": False,
                "stale": False,
                "signals": ["lexical"],
            }
        ],
        "stats": {"entities": 1, "observations": 1, "relations": 0},
    }

    merged = server_module.merge_memory_context_results(
        workspace_result,
        global_result,
        limit=5,
    )

    assert merged["hint_matches"]
    assert merged["hint_matches"][0]["name"] == "Workspace JWT Config"
    assert merged["hint_matches"][0]["source"] == "workspace"


def test_preference_routing_predicates_detect_global_preference_payloads():
    assert server_module._is_preference_entity(
        {"name": "Editor Preference", "entityType": "note", "tags": ["preference"]}
    )
    assert server_module._is_preference_observation(
        {"entityName": "Editor Preference", "contents": ["prefer ruff"], "obs_type": "preference"}
    )


def test_global_preference_graph_can_be_opened_separately_from_workspace_graph(tmp_path):
    workspace_db_path = tmp_path / "workspaces" / "workspace.db"
    global_db_path = tmp_path / "global" / "preferences.db"
    workspace_db_path.parent.mkdir(parents=True, exist_ok=True)
    global_db_path.parent.mkdir(parents=True, exist_ok=True)

    workspace_db = Database(workspace_db_path)
    global_db = Database(global_db_path)
    workspace_db.open()
    global_db.open()
    try:
        workspace_graph = KnowledgeGraphManager(workspace_db, session_id="workspace")
        global_graph = KnowledgeGraphManager(global_db, session_id="global")

        workspace_graph.create_entities(
            [{"name": "Workspace Config", "entityType": "config", "observations": ["JWT issuer"]}]
        )
        global_graph.create_entities(
            [
                {
                    "name": "Global Preference",
                    "entityType": "preference",
                    "tags": ["preference"],
                    "observations": ["prefer ruff over flake8"],
                }
            ]
        )

        assert workspace_graph.memory_context(hint="JWT issuer")["hint_matches"][0]["name"] == "Workspace Config"
        assert global_graph.memory_context(hint="prefer ruff")["hint_matches"][0]["name"] == "Global Preference"
    finally:
        global_db.close()
        workspace_db.close()


def test_mcp_global_preference_lifecycle_is_scoped_and_discoverable(tmp_path):
    workspace_db = Database(tmp_path / "workspace.db")
    global_db = Database(tmp_path / "global.db")
    workspace_db.open()
    global_db.open()
    try:
        workspace_graph = KnowledgeGraphManager(workspace_db, session_id="workspace")
        global_graph = KnowledgeGraphManager(global_db, session_id="global")
        config = MemoryConfig(
            db_path=tmp_path / "workspace.db",
            global_db_path=tmp_path / "global.db",
            embedding_enabled=False,
        )
        ctx = _tool_ctx(workspace_graph, config, global_graph)

        created = json.loads(
            _tool_fn("create_entities")(
                ctx,
                [
                    {
                        "name": "Editor Preference",
                        "entityType": "preference",
                        "tags": ["preference"],
                        "observations": ["prefer ruff over flake8"],
                    }
                ],
            )
        )
        assert created[0]["name"] == "Editor Preference"

        workspace_read = json.loads(_tool_fn("read_graph")(ctx, compress=False))
        global_read = json.loads(_tool_fn("read_graph")(ctx, compress=False, scope="global"))
        all_read = json.loads(_tool_fn("read_graph")(ctx, compress=False, scope="all"))
        assert workspace_read["entities"] == []
        assert global_read["entities"][0]["name"] == "Editor Preference"
        assert {item["source"] for item in all_read} == {"workspace", "global"}

        search = json.loads(
            _tool_fn("search_nodes")(
                ctx,
                query="ruff",
                compress=False,
                scope="global",
            )
        )
        assert search["entities"][0]["name"] == "Editor Preference"

        exported = json.loads(_tool_fn("export_graph")(ctx, scope="global"))
        assert exported["entities"][0]["name"] == "Editor Preference"

        stats = json.loads(_tool_fn("memory_stats")(ctx, scope="global"))
        assert stats["entities"] == 1
        assert stats["observations"] == 1

        backup = json.loads(
            _tool_fn("backup_memory")(
                ctx,
                dest_path=str(tmp_path / "backups" / "global-backup.db"),
                scope="global",
            )
        )
        assert backup["source"] == "global"
        assert os.path.exists(backup["backup_path"])

        deleted = json.loads(
            _tool_fn("delete_entities")(ctx, ["Editor Preference"], hard=True, scope="global")
        )
        assert deleted["deleted"] == 1

        absent_read = json.loads(_tool_fn("read_graph")(ctx, compress=False, scope="global"))
        absent_search = json.loads(
            _tool_fn("search_nodes")(ctx, query="ruff", compress=False, scope="global")
        )
        assert absent_read["entities"] == []
        assert absent_search["entities"] == []
    finally:
        global_db.close()
        workspace_db.close()


def test_mcp_scope_all_labels_sources_for_ambiguous_names(tmp_path):
    workspace_db = Database(tmp_path / "workspace.db")
    global_db = Database(tmp_path / "global.db")
    workspace_db.open()
    global_db.open()
    try:
        workspace_graph = KnowledgeGraphManager(workspace_db, session_id="workspace")
        global_graph = KnowledgeGraphManager(global_db, session_id="global")
        workspace_graph.create_entities([{"name": "Same Name", "entityType": "workspace"}])
        global_graph.create_entities([{"name": "Same Name", "entityType": "global"}])
        ctx = _tool_ctx(
            workspace_graph,
            MemoryConfig(
                db_path=tmp_path / "workspace.db",
                global_db_path=tmp_path / "global.db",
                embedding_enabled=False,
            ),
            global_graph,
        )

        opened = json.loads(
            _tool_fn("open_nodes")(ctx, ["Same Name"], scope="all")
        )

        assert [item["source"] for item in opened] == ["workspace", "global"]
        assert opened[0]["result"]["entities"][0]["entityType"] == "workspace"
        assert opened[1]["result"]["entities"][0]["entityType"] == "global"
    finally:
        global_db.close()
        workspace_db.close()


def test_memory_context_marks_conflicting_high_value_matches(graph):
    graph.create_entities(
        [
            {
                "name": "JWT Config",
                "entityType": "config",
                "observations": [
                    "JWT issuer is auth-prod",
                    "JWT issuer is auth-stage",
                ],
            },
        ]
    )

    ctx = graph.memory_context(hint="JWT issuer")

    assert ctx["hint_matches"]
    match = ctx["hint_matches"][0]
    assert match["conflict"] is True
    assert "conflict" in match["signals"]
    assert len(match["snippets"]) == 2


def test_memory_context_marks_stale_low_confidence_match(graph):
    graph.create_entities(
        [
            {"name": "Legacy Note", "entityType": "note"},
        ]
    )
    graph.add_observations(
        [
            {
                "entityName": "Legacy Note",
                "contents": ["legacy auth endpoint is /v1/login"],
                "confidence": 0.2,
                "importance": 0.4,
            },
        ]
    )
    graph.db.cx.execute(
        "UPDATE entities "
        "SET last_accessed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-120 days'), "
        "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-120 days') "
        "WHERE name = ?",
        ("Legacy Note",),
    )
    graph.db.cx.execute(
        "UPDATE observations "
        "SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-120 days') "
        "WHERE entity_id = (SELECT id FROM entities WHERE name = ?)",
        ("Legacy Note",),
    )
    graph.db.cx.commit()

    ctx = graph.memory_context(hint="legacy auth endpoint")

    assert ctx["hint_matches"]
    match = ctx["hint_matches"][0]
    assert match["stale"] is True
    assert "stale" in match["signals"]


def test_format_memory_context_marks_conflict_and_stale():
    rendered = server_module._format_memory_context_result(
        {
            "pinned": [],
            "recent_activity": [],
            "hint_matches": [
                {
                    "name": "JWT Config",
                    "type": "config",
                    "snippets": ["JWT issuer is auth-prod", "JWT issuer is auth-stage"],
                    "conflict": True,
                    "stale": True,
                }
            ],
            "stats": {"entities": 1, "observations": 2, "relations": 0},
        }
    )

    assert (
        "Relevant: JWT Config[config]: \"JWT issuer is auth-prod\" | "
        '"JWT issuer is auth-stage" !conflict !stale'
        in rendered
    )


def test_memory_stats(graph):
    graph.create_entities(
        [
            {"name": "A", "entityType": "test", "observations": ["obs1"]},
            {"name": "B", "entityType": "test"},
        ]
    )
    graph.create_relations([{"from": "A", "to": "B", "relationType": "knows"}])
    stats = graph.memory_stats()
    assert stats["entities"] == 2
    assert stats["observations"] == 1
    assert stats["relations"] == 1
    assert "tag_distribution" in stats
    assert "orphan_entities" in stats
    # B has no observations, but has a relation, so not orphan
    # Actually B has a relation, so it's not orphan
    assert stats["orphan_entities"] == 0


def test_export_json(graph):
    graph.create_entities([{"name": "E", "entityType": "t", "observations": ["obs"]}])
    exported = graph.export_graph(fmt="json")
    data = json.loads(exported)
    assert len(data["entities"]) == 1
    assert data["entities"][0]["name"] == "E"


def test_export_jsonl(graph):
    graph.create_entities(
        [
            {"name": "E1", "entityType": "t1", "observations": ["o1"]},
            {"name": "E2", "entityType": "t2"},
        ]
    )
    graph.create_relations([{"from": "E1", "to": "E2", "relationType": "uses"}])
    exported = graph.export_graph(fmt="jsonl")
    lines = [json.loads(line) for line in exported.strip().split("\n")]
    entity_lines = [line for line in lines if line["type"] == "entity"]
    relation_lines = [line for line in lines if line["type"] == "relation"]
    assert len(entity_lines) == 2
    assert len(relation_lines) == 1


def test_import_json(graph):
    data = json.dumps(
        {
            "entities": [
                {"name": "Imported", "entityType": "test", "observations": ["obs1", "obs2"]},
            ],
            "relations": [],
        }
    )
    counts = graph.import_graph(data)
    assert counts["entities"] == 1
    assert counts["observations"] == 2
    entity = graph.get_entity_by_name("Imported")
    assert entity is not None


def test_import_jsonl(graph):
    lines = [
        json.dumps(
            {"type": "entity", "name": "A", "entityType": "node", "observations": ["a-obs"]}
        ),
        json.dumps({"type": "entity", "name": "B", "entityType": "node", "observations": []}),
        json.dumps({"type": "relation", "from": "A", "to": "B", "relationType": "links"}),
    ]
    data = "\n".join(lines)
    counts = graph.import_graph(data)
    assert counts["entities"] == 2
    assert counts["relations"] == 1


def test_import_export_roundtrip(graph):
    graph.create_entities(
        [
            {"name": "R1", "entityType": "test", "observations": ["round trip data"]},
            {"name": "R2", "entityType": "test"},
        ]
    )
    graph.create_relations([{"from": "R1", "to": "R2", "relationType": "connects"}])

    exported = graph.export_graph(fmt="jsonl")

    # Import into fresh graph
    from server_memory.db import Database
    from server_memory.graph import KnowledgeGraphManager

    db2 = Database(":memory:")
    db2.open()
    graph2 = KnowledgeGraphManager(db2)

    counts = graph2.import_graph(exported)
    assert counts["entities"] == 2
    assert counts["relations"] == 1

    entity = graph2.get_entity_by_name("R1")
    assert entity is not None
    assert len(entity.observations) == 1

    db2.close()


def test_memory_context_hint_with_snippets(graph):
    """memory_context should include observation snippets in hint matches."""
    graph.create_entities(
        [
            {
                "name": "Auth Module",
                "entityType": "module",
                "observations": [
                    "handles JWT token validation and refresh for the production auth stack",
                    "supports OAuth 2.0 flow",
                ],
            },
        ]
    )
    ctx = graph.memory_context(hint="JWT")
    assert len(ctx["hint_matches"]) >= 1
    match = ctx["hint_matches"][0]
    assert "snippets" in match
    assert len(match["snippets"]) > 0
    assert "JWT token validation" in match["snippets"][0]
    assert len(match["snippets"][0]) > 53


def test_memory_context_snippet_budget_keeps_useful_full_observation(graph):
    """Useful observations should not be hard-truncated to a flat 50-char limit."""
    long_obs = (
        "JWT issuer and audience settings live in auth/config/jwt.yaml for production deploys"
    )
    graph.create_entities(
        [
            {"name": "LongObs", "entityType": "test", "observations": [long_obs]},
        ]
    )
    ctx = graph.memory_context(hint="jwt yaml")
    if ctx["hint_matches"]:
        match = ctx["hint_matches"][0]
        if match["snippets"]:
            assert match["snippets"][0] == long_obs


def test_backup_memory(graph):
    """Backup tool should create a valid SQLite backup."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backup_path = os.path.join(tmpdir, "test_backup.db")
        graph.db.backup(backup_path)

        assert os.path.exists(backup_path)
        assert os.path.getsize(backup_path) > 0

        # Verify backup is valid SQLite
        import sqlite3

        conn = sqlite3.connect(backup_path)
        conn.row_factory = sqlite3.Row
        tables = {
            r["name"]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert "entities" in tables
        assert "observations" in tables
        conn.close()


def test_backup_with_data(graph):
    """Backup should include data from the original database."""
    graph.create_entities(
        [
            {"name": "BackupTest", "entityType": "test", "observations": ["backup data"]},
        ]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        backup_path = os.path.join(tmpdir, "data_backup.db")
        graph.db.backup(backup_path)

        import sqlite3

        conn = sqlite3.connect(backup_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM entities WHERE name = 'BackupTest'").fetchone()
        assert row is not None
        conn.close()


def test_schedule_startup_cleanup_runs_once(monkeypatch, tmp_path):
    calls: list[tuple[str, object]] = []

    class FakeThread:
        def __init__(self, *, target, args, name, daemon):
            calls.append(("init", name, daemon, args[0]))
            self._target = target
            self._args = args

        def start(self):
            calls.append(("start", self._args[0]))

    monkeypatch.setattr(server_module.threading, "Thread", FakeThread)
    monkeypatch.setattr(server_module, "_startup_cleanup_started", False)

    config = MemoryConfig(db_path=tmp_path / "memory.db")

    server_module._schedule_startup_cleanup(config)
    server_module._schedule_startup_cleanup(config)

    assert calls == [
        ("init", "server-memory-startup-cleanup", True, config),
        ("start", config),
    ]


def test_lifespan_database_cleanup_closes_both_databases_when_one_close_fails():
    calls: list[str] = []

    class FakeDatabase:
        def __init__(self, name: str, *, fail: bool = False):
            self.name = name
            self.fail = fail

        def close(self):
            calls.append(self.name)
            if self.fail:
                raise RuntimeError(f"{self.name} close failed")

    # Workspace close fails
    workspace_db = FakeDatabase("workspace", fail=True)
    global_db = FakeDatabase("global")
    with pytest.raises(RuntimeError, match="workspace close failed"):
        server_module._close_lifespan_databases(workspace_db, global_db)
    assert calls == ["workspace", "global"]

    # Global close fails
    calls.clear()
    workspace_db2 = FakeDatabase("workspace")
    global_db2 = FakeDatabase("global", fail=True)
    with pytest.raises(RuntimeError, match="global close failed"):
        server_module._close_lifespan_databases(workspace_db2, global_db2)
    assert calls == ["workspace", "global"]

    # Both close fail
    calls.clear()
    workspace_db3 = FakeDatabase("workspace", fail=True)
    global_db3 = FakeDatabase("global", fail=True)
    with pytest.raises(RuntimeError, match="global close failed"):
        server_module._close_lifespan_databases(workspace_db3, global_db3)
    assert calls == ["workspace", "global"]


def test_destructive_operations_prevent_scope_all(tmp_path):
    workspace_db = Database(tmp_path / "workspace.db")
    global_db = Database(tmp_path / "global.db")
    workspace_db.open()
    global_db.open()
    try:
        workspace_graph = KnowledgeGraphManager(workspace_db, session_id="workspace")
        global_graph = KnowledgeGraphManager(global_db, session_id="global")
        config = MemoryConfig(
            db_path=tmp_path / "workspace.db",
            global_db_path=tmp_path / "global.db",
            embedding_enabled=False,
        )
        ctx = _tool_ctx(workspace_graph, config, global_graph)

        for tool_name in ["delete_entities", "delete_observations", "delete_relations", "merge_entities"]:
            # Setup dummy args
            if tool_name == "delete_entities":
                res = _tool_fn(tool_name)(ctx, entityNames=["foo"], scope="all")
            elif tool_name == "delete_observations":
                res = _tool_fn(tool_name)(ctx, deletions=[{"entityName": "foo", "observations": ["bar"]}], scope="all")
            elif tool_name == "delete_relations":
                res = _tool_fn(tool_name)(ctx, relations=[{"from": "foo", "to": "bar", "relationType": "knows"}], scope="all")
            elif tool_name == "merge_entities":
                res = _tool_fn(tool_name)(ctx, source="foo", target="bar", scope="all")
            
            res_dict = json.loads(res)
            assert "error" in res_dict
            assert "scope='all' is not supported for destructive operations" in res_dict["error"]

        # Test manage_tags action="delete" / "untag"
        res_delete = _tool_fn("manage_tags")(ctx, action="delete", name="foo", scope="all")
        assert "scope='all' is not supported for destructive tag action" in json.loads(res_delete)["error"]

        res_untag = _tool_fn("manage_tags")(ctx, action="untag", entity_name="foo", tag_name="bar", scope="all")
        assert "scope='all' is not supported for destructive tag action" in json.loads(res_untag)["error"]

    finally:
        global_db.close()
        workspace_db.close()


def test_memory_context_full_shifts_global_ids(tmp_path):
    workspace_db = Database(tmp_path / "workspace.db")
    global_db = Database(tmp_path / "global.db")
    workspace_db.open()
    global_db.open()
    try:
        workspace_graph = KnowledgeGraphManager(workspace_db, session_id="workspace")
        global_graph = KnowledgeGraphManager(global_db, session_id="global")
        
        # Create entity with ID 1 in both DBs (since they are new, their first entities will have ID 1)
        workspace_graph.create_entities([{"name": "Workspace Pinned", "entityType": "config", "tags": ["pinned"]}])
        global_graph.create_entities([{"name": "Global Pinned", "entityType": "preference", "tags": ["pinned"]}])
        
        config = MemoryConfig(
            db_path=tmp_path / "workspace.db",
            global_db_path=tmp_path / "global.db",
            embedding_enabled=False,
            compression_level=0,  # Level 0 is NONE, returns full JSON
        )
        ctx = _tool_ctx(workspace_graph, config, global_graph)
        
        # Query full context with scope="all"
        res = _tool_fn("memory_context_full")(ctx, scope="all", budget=10000)
        res_dict = json.loads(res)
        
        entity_names = {e["name"] for e in res_dict["entities"]}
        assert "Workspace Pinned" in entity_names
        # Both must be present! (The fix ensures the global pinned entity is not skipped due to ID collision)
        assert "Global Pinned" in entity_names
    finally:
        global_db.close()
        workspace_db.close()
