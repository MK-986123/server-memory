"""Tests for activity logging, timeline, memory context, import/export, stats, and backup."""

import asyncio
import json
import os
import tempfile
from types import SimpleNamespace

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


def test_mcp_tools_advertise_safety_and_bounded_contracts():
    tools = asyncio.run(server_module.create_server().list_tools())

    assert len(tools) == 22
    assert all(tool.annotations is not None for tool in tools)
    assert all(tool.annotations.openWorldHint is False for tool in tools)
    assert all(tool.outputSchema for tool in tools)

    by_name = {tool.name: tool for tool in tools}
    assert by_name["read_graph"].annotations.readOnlyHint is True
    assert by_name["delete_entities"].annotations.destructiveHint is True
    entity_schema = by_name["create_entities"].inputSchema
    assert entity_schema["properties"]["entities"]["maxItems"] == 500
    assert entity_schema["$defs"]["EntityInput"]["required"] == ["name", "entityType"]
    assert by_name["search_nodes"].inputSchema["properties"]["limit"]["maximum"] == 500
    assert by_name["search_nodes"].inputSchema["properties"]["page_size"]["maximum"] == 500
    assert "scope" in by_name["add_observations"].inputSchema["properties"]
    assert "scope" in by_name["memory_context"].inputSchema["properties"]
    assert "budget" in by_name["memory_context"].inputSchema["properties"]
    assert "scope" in by_name["memory_context_full"].inputSchema["properties"]
    assert "cursor" in by_name["read_graph"].inputSchema["properties"]
    assert by_name["open_nodes"].inputSchema["properties"]["depth"]["maximum"] == 10
    assert "scope" in by_name["restore_entities"].inputSchema["properties"]
    for tool in tools:
        assert tool.outputSchema["type"] == "object"
        assert set(tool.outputSchema["properties"]) == {"text", "data"}
        assert "result" not in tool.outputSchema["properties"]


def test_structured_tool_result_preserves_legacy_text_payload():
    """Text-oriented clients must still receive the prior JSON string payload."""
    legacy = json.dumps({"deleted": 2, "note": "legacy text clients"})
    adapted = server_module.StructuredToolResult.model_validate(legacy)
    assert adapted.text == legacy
    assert adapted.data == {"deleted": 2, "note": "legacy text clients"}

    plain = "not-json-but-still-text"
    adapted_plain = server_module.StructuredToolResult.model_validate(plain)
    assert adapted_plain.text == plain
    assert adapted_plain.data is None

    # Direct Python tool call path still returns the established string.
    workspace_db = Database(":memory:")
    workspace_db.open()
    try:
        graph = KnowledgeGraphManager(workspace_db)
        graph.create_entities([{"name": "Legacy", "entityType": "note"}])
        config = MemoryConfig(embedding_enabled=False, global_db_enabled=False)
        ctx = _tool_ctx(graph, config)
        raw = _tool_fn("delete_entities")(ctx, entityNames=["Legacy"], scope="workspace")
        assert isinstance(raw, str)
        assert json.loads(raw) == {"deleted": 1}
        structured = server_module.StructuredToolResult.model_validate(raw)
        assert structured.text == raw
        assert structured.data == {"deleted": 1}
    finally:
        workspace_db.close()


def test_graph_pagination_cursor_is_stable_and_query_bound(graph):
    graph.create_entities(
        [{"name": f"Entity {index}", "entityType": "note"} for index in range(5)]
    )
    full = graph.read_graph()
    fingerprint = server_module._fingerprint_cursor("read_graph", {"scope": "workspace"})

    first, cursor = server_module._paginate_graph(
        full, page_size=2, cursor="", fingerprint=fingerprint
    )
    second, next_cursor = server_module._paginate_graph(
        full, page_size=2, cursor=cursor, fingerprint=fingerprint
    )

    assert [entity.name for entity in first.entities] == ["Entity 0", "Entity 1"]
    assert [entity.name for entity in second.entities] == ["Entity 2", "Entity 3"]
    assert next_cursor
    try:
        server_module._paginate_graph(
            full, page_size=2, cursor=cursor, fingerprint="different-query"
        )
        assert False, "mismatched cursor should fail"
    except ValueError as exc:
        assert "cursor" in str(exc)


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


def test_merge_memory_context_preserves_same_name_in_both_scopes():
    base = {
        "pinned": [{"name": "Shared", "type": "note"}],
        "recent_activity": [],
        "hint_matches": [{"name": "Shared", "type": "note", "score": 0.8}],
        "stats": {"entities": 1, "observations": 1, "relations": 0},
    }

    merged = server_module.merge_memory_context_results(base, base, limit=10)

    assert {(item["source"], item["name"]) for item in merged["pinned"]} == {
        ("workspace", "Shared"),
        ("global", "Shared"),
    }
    assert {(item["source"], item["name"]) for item in merged["hint_matches"]} == {
        ("workspace", "Shared"),
        ("global", "Shared"),
    }


def test_merged_activity_is_globally_newest_first():
    merged = server_module._merge_activity_entries(
        [{"action": "old", "summary": "workspace", "at": "2025-01-01T00:00:00Z"}],
        [{"action": "new", "summary": "global", "at": "2026-01-01T00:00:00Z"}],
        limit=1,
    )

    assert merged == [
        {
            "action": "new",
            "summary": "global",
            "at": "2026-01-01T00:00:00Z",
            "source": "global",
        }
    ]


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


def test_backup_refuses_to_clobber_existing_file(graph, tmp_path):
    destination = tmp_path / "existing.db"
    destination.write_bytes(b"sentinel")

    try:
        graph.db.backup(destination)
        assert False, "existing destination should fail"
    except ValueError as exc:
        assert "already exists" in str(exc)

    assert destination.read_bytes() == b"sentinel"


def test_backup_path_is_bounded_and_no_clobber(tmp_path):
    db_path = tmp_path / "memory.db"
    backup_root = tmp_path / "backups"
    backup_root.mkdir()
    existing = backup_root / "existing.db"
    existing.write_text("sentinel", encoding="utf-8")

    default_path = server_module._resolve_backup_path(
        db_path,
        "",
        scope_name="workspace",
        multiple_scopes=False,
        timestamp="fixed",
    )
    assert default_path == backup_root / "workspace-fixed.db"

    try:
        server_module._resolve_backup_path(
            db_path,
            str(tmp_path / "outside.db"),
            scope_name="workspace",
            multiple_scopes=False,
            timestamp="fixed",
        )
        assert False, "outside destination should fail"
    except ValueError as exc:
        assert "under" in str(exc)

    try:
        server_module._resolve_backup_path(
            db_path,
            str(existing),
            scope_name="workspace",
            multiple_scopes=False,
            timestamp="fixed",
        )
        assert False, "existing destination should fail"
    except ValueError as exc:
        assert "already exists" in str(exc)


def test_backup_path_rejects_symlink_escape(tmp_path):
    db_parent = tmp_path / "database"
    db_parent.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    backup_root = db_parent / "backups"
    backup_root.mkdir()
    (backup_root / "escape").symlink_to(outside, target_is_directory=True)

    try:
        server_module._resolve_backup_path(
            db_parent / "memory.db",
            str(backup_root / "escape" / "backup.db"),
            scope_name="workspace",
            multiple_scopes=False,
            timestamp="fixed",
        )
        assert False, "symlink escape should fail"
    except ValueError as exc:
        assert "under" in str(exc)


def test_backup_path_rejects_symlink_root(tmp_path):
    db_parent = tmp_path / "database"
    db_parent.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (db_parent / "backups").symlink_to(outside, target_is_directory=True)

    try:
        server_module._resolve_backup_path(
            db_parent / "memory.db",
            "",
            scope_name="workspace",
            multiple_scopes=False,
            timestamp="fixed",
        )
        assert False, "symlink backup root should fail"
    except ValueError as exc:
        assert "symlink" in str(exc)


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


def test_startup_cleanup_preserves_old_durable_entities(tmp_path):
    db_path = tmp_path / "memory.db"
    database = Database(db_path)
    database.open()
    graph = KnowledgeGraphManager(database)
    graph.create_entities(
        [
            {"name": "Durable", "entityType": "decision"},
            {"name": "Temporary", "entityType": "scratch", "tags": ["ephemeral"]},
        ]
    )
    database.cx.execute(
        "UPDATE entities SET created_at = '2020-01-01T00:00:00.000Z', "
        "updated_at = '2020-01-01T00:00:00.000Z', "
        "last_accessed_at = '2020-01-01T00:00:00.000Z'"
    )
    database.cx.commit()
    database.close()

    server_module._run_startup_cleanup(MemoryConfig(db_path=db_path))

    check = Database(db_path)
    check.open()
    try:
        durable = check.cx.execute(
            "SELECT deleted_at FROM entities WHERE name = 'Durable'"
        ).fetchone()
        temporary = check.cx.execute(
            "SELECT deleted_at FROM entities WHERE name = 'Temporary'"
        ).fetchone()
        assert durable["deleted_at"] is None
        assert temporary["deleted_at"] is not None
    finally:
        check.close()


def test_server_exposes_deleted_entity_recovery_tools(tmp_path):
    server = server_module.create_server(
        MemoryConfig(
            db_path=tmp_path / "memory.db",
            global_db_enabled=False,
            embedding_enabled=False,
        )
    )

    tool_names = set(server._tool_manager._tools)

    assert "list_deleted_entities" in tool_names
    assert "restore_entities" in tool_names


def test_benchmark_telemetry_is_opt_in_and_local(monkeypatch, tmp_path):
    telemetry = tmp_path / "events.jsonl"
    monkeypatch.setenv("MEMORY_BENCHMARK_TELEMETRY_PATH", str(telemetry))

    server_module._record_benchmark_event("mcp_handshake")
    server_module._record_benchmark_event("tool_call")

    events = [json.loads(line)["event"] for line in telemetry.read_text().splitlines()]
    assert events == ["mcp_handshake", "tool_call"]


def _seed_entity(graph: KnowledgeGraphManager, name: str, *, obs: str = "note") -> None:
    graph.create_entities(
        [
            {
                "name": name,
                "entityType": "note",
                "observations": [obs],
            }
        ]
    )


def test_destructive_operations_prevent_scope_all_without_mutation(tmp_path):
    """scope='all' must return the public structured error and mutate nothing."""
    workspace_db = Database(tmp_path / "workspace.db")
    global_db = Database(tmp_path / "global.db")
    workspace_db.open()
    global_db.open()
    try:
        workspace_graph = KnowledgeGraphManager(workspace_db, session_id="workspace")
        global_graph = KnowledgeGraphManager(global_db, session_id="global")
        _seed_entity(workspace_graph, "W", obs="workspace-obs")
        _seed_entity(global_graph, "G", obs="global-obs")
        workspace_graph.create_entities([{"name": "W2", "entityType": "note"}])
        workspace_graph.create_relations(
            [{"from": "W", "to": "W2", "relationType": "related_to"}]
        )
        global_graph.create_entities([{"name": "G2", "entityType": "note"}])
        global_graph.create_relations(
            [{"from": "G", "to": "G2", "relationType": "related_to"}]
        )
        workspace_graph.create_tag("user-tag")
        global_graph.create_tag("user-tag")
        workspace_graph.tag_entity("W", "user-tag")
        global_graph.tag_entity("G", "user-tag")

        config = MemoryConfig(
            db_path=tmp_path / "workspace.db",
            global_db_path=tmp_path / "global.db",
            embedding_enabled=False,
        )
        ctx = _tool_ctx(workspace_graph, config, global_graph)

        def active_counts(graph: KnowledgeGraphManager) -> tuple[int, int, int]:
            cx = graph.db.cx
            entities = cx.execute(
                "SELECT COUNT(*) c FROM entities WHERE deleted_at IS NULL"
            ).fetchone()["c"]
            observations = cx.execute(
                "SELECT COUNT(*) c FROM observations WHERE deleted_at IS NULL"
            ).fetchone()["c"]
            relations = cx.execute(
                "SELECT COUNT(*) c FROM relations WHERE deleted_at IS NULL"
            ).fetchone()["c"]
            return entities, observations, relations

        before_w = active_counts(workspace_graph)
        before_g = active_counts(global_graph)

        cases = [
            ("delete_entities", {"entityNames": ["W", "G"], "hard": False}),
            ("delete_entities", {"entityNames": ["W", "G"], "hard": True}),
            (
                "delete_observations",
                {
                    "deletions": [
                        {"entityName": "W", "observations": ["workspace-obs"]},
                        {"entityName": "G", "observations": ["global-obs"]},
                    ]
                },
            ),
            (
                "delete_relations",
                {
                    "relations": [
                        {"from": "W", "to": "W2", "relationType": "related_to"},
                        {"from": "G", "to": "G2", "relationType": "related_to"},
                    ]
                },
            ),
            ("merge_entities", {"source": "W", "target": "W2"}),
            ("manage_tags", {"action": "delete", "name": "user-tag"}),
            ("manage_tags", {"action": "cleanup"}),
            ("manage_tags", {"action": "untag", "entity_name": "W", "tag_name": "user-tag"}),
            (
                "import_graph",
                {"data": json.dumps({"entities": [], "relations": []})},
            ),
        ]

        for tool_name, kwargs in cases:
            res = _tool_fn(tool_name)(ctx, scope="all", **kwargs)
            payload = json.loads(res)
            assert "error" in payload, tool_name
            if tool_name == "manage_tags":
                assert "scope='all' is not supported for destructive tag action" in payload["error"]
            else:
                assert "scope='all' is not supported for destructive operations" in payload["error"]
            assert active_counts(workspace_graph) == before_w
            assert active_counts(global_graph) == before_g
            assert workspace_graph.get_entity_by_name("W") is not None
            assert global_graph.get_entity_by_name("G") is not None
    finally:
        global_db.close()
        workspace_db.close()


def test_destructive_operations_succeed_for_workspace_and_global(tmp_path):
    workspace_db = Database(tmp_path / "workspace.db")
    global_db = Database(tmp_path / "global.db")
    workspace_db.open()
    global_db.open()
    try:
        workspace_graph = KnowledgeGraphManager(workspace_db, session_id="workspace")
        global_graph = KnowledgeGraphManager(global_db, session_id="global")
        for graph, prefix in ((workspace_graph, "W"), (global_graph, "G")):
            _seed_entity(graph, f"{prefix}-keep", obs=f"{prefix}-obs")
            _seed_entity(graph, f"{prefix}-soft", obs=f"{prefix}-soft-obs")
            _seed_entity(graph, f"{prefix}-hard", obs=f"{prefix}-hard-obs")
            graph.create_entities([{"name": f"{prefix}-rel", "entityType": "note"}])
            graph.create_relations(
                [
                    {
                        "from": f"{prefix}-keep",
                        "to": f"{prefix}-rel",
                        "relationType": "related_to",
                    }
                ]
            )

        config = MemoryConfig(
            db_path=tmp_path / "workspace.db",
            global_db_path=tmp_path / "global.db",
            embedding_enabled=False,
        )
        ctx = _tool_ctx(workspace_graph, config, global_graph)

        for scope, graph, prefix in (
            ("workspace", workspace_graph, "W"),
            ("global", global_graph, "G"),
        ):
            soft = json.loads(
                _tool_fn("delete_entities")(
                    ctx, entityNames=[f"{prefix}-soft"], hard=False, scope=scope
                )
            )
            assert soft["deleted"] == 1
            assert graph.get_entity_by_name(f"{prefix}-soft") is None
            hard = json.loads(
                _tool_fn("delete_entities")(
                    ctx, entityNames=[f"{prefix}-hard"], hard=True, scope=scope
                )
            )
            assert hard["deleted"] == 1
            row = graph.db.cx.execute(
                "SELECT 1 FROM entities WHERE name = ?", (f"{prefix}-hard",)
            ).fetchone()
            assert row is None

            obs = json.loads(
                _tool_fn("delete_observations")(
                    ctx,
                    deletions=[
                        {
                            "entityName": f"{prefix}-keep",
                            "observations": [f"{prefix}-obs"],
                        }
                    ],
                    scope=scope,
                )
            )
            assert obs["deleted"] == 1

            rel = json.loads(
                _tool_fn("delete_relations")(
                    ctx,
                    relations=[
                        {
                            "from": f"{prefix}-keep",
                            "to": f"{prefix}-rel",
                            "relationType": "related_to",
                        }
                    ],
                    scope=scope,
                )
            )
            assert rel["deleted"] == 1

            graph.create_entities(
                [
                    {"name": f"{prefix}-src", "entityType": "note"},
                    {"name": f"{prefix}-dst", "entityType": "note"},
                ]
            )
            merged = json.loads(
                _tool_fn("merge_entities")(
                    ctx,
                    source=f"{prefix}-src",
                    target=f"{prefix}-dst",
                    scope=scope,
                )
            )
            assert merged["name"] == f"{prefix}-dst"
            assert graph.get_entity_by_name(f"{prefix}-src") is None

            graph.create_tag(f"{prefix}-tag")
            deleted_tag = json.loads(
                _tool_fn("manage_tags")(
                    ctx, action="delete", name=f"{prefix}-tag", scope=scope
                )
            )
            assert deleted_tag["deleted"] is True

            cleaned = json.loads(
                _tool_fn("manage_tags")(ctx, action="cleanup", scope=scope)
            )
            assert "cleaned" in cleaned
    finally:
        global_db.close()
        workspace_db.close()
