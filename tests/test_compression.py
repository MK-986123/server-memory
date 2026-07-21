"""Tests for multi-level compression."""

from server_memory.compression import (
    CompressionLevel,
    _enforce_budget,
    _estimate_tokens,
    compress_graph,
    format_entity_heavy,
    format_entity_light,
    format_entity_medium,
    format_relation_heavy,
    format_relation_light,
    format_relation_medium,
)
from server_memory.models import Entity, KnowledgeGraph, Observation, Relation


def _make_entity(name="App", etype="project", obs=None, tags=None):
    observations = [Observation(id=i + 1, content=c) for i, c in enumerate(obs or [])]
    return Entity(
        id=1,
        name=name,
        entity_type=etype,
        observations=observations,
        tags=tags or [],
    )


def _make_relation(from_name="A", to_name="B", rel_type="uses"):
    return Relation(
        id=1,
        from_entity_id=1,
        to_entity_id=2,
        relation_type=rel_type,
        from_name=from_name,
        to_name=to_name,
    )


def test_format_entity_light():
    e = _make_entity("App", "project", ["Built with Python", "Uses SQLite"], ["pinned"])
    result = format_entity_light(e)
    assert "App [project]" in result
    assert "#pinned" in result
    assert "- Built with Python" in result
    assert "- Uses SQLite" in result


def test_format_entity_medium():
    e = _make_entity("App", "project", ["The app is built with Python"], ["pinned"])
    result = format_entity_medium(e)
    assert "- App (project)" in result
    assert "#pinned" in result
    # Filler words should be stripped
    assert "The " not in result
    assert "is " not in result


def test_format_entity_heavy():
    e = _make_entity("App", "project", ["observation " * 20])
    result = format_entity_heavy(e)
    assert result.startswith("App:project:")
    assert len(result) < 200  # Should be truncated


def test_format_relation_light():
    r = _make_relation("A", "B", "depends_on")
    assert format_relation_light(r) == "A --[depends_on]--> B"


def test_format_relation_medium():
    r = _make_relation("A", "B", "depends_on")
    assert format_relation_medium(r) == "A>depends_on>B"


def test_format_relation_heavy():
    r = _make_relation("A", "B", "depends_on")
    assert format_relation_heavy(r) == "A>B"


def test_text_formats_namespace_merged_scope_identity():
    workspace = _make_entity("Shared", "note", ["workspace fact"])
    workspace.scope = "workspace"
    global_entity = _make_entity("Shared", "note", ["global fact"])
    global_entity.scope = "global"
    relation = _make_relation("Shared", "Other", "uses")
    relation.scope = "global"

    graph = KnowledgeGraph(entities=[workspace, global_entity], relations=[relation])
    rendered = compress_graph(graph, CompressionLevel.MEDIUM, token_budget=2000)

    assert "Shared@workspace" in rendered
    assert "Shared@global" in rendered
    assert "Shared@global>uses>Other@global" in rendered


def test_compress_graph_medium():
    kg = KnowledgeGraph(
        entities=[
            _make_entity("App", "project", ["main app"]),
            _make_entity("DB", "database", ["SQLite storage"]),
        ],
        relations=[_make_relation("App", "DB", "uses")],
    )
    result = compress_graph(kg, CompressionLevel.MEDIUM, token_budget=2000)
    assert "- App (project)" in result
    assert "- DB (database)" in result
    assert "App>uses>DB" in result


def test_compress_graph_budget_truncation():
    # Create many entities to exceed budget
    entities = [
        _make_entity(f"Entity{i}", "test", [f"observation for entity {i}"]) for i in range(100)
    ]
    kg = KnowledgeGraph(entities=entities, relations=[])
    result = compress_graph(kg, CompressionLevel.MEDIUM, token_budget=200)
    assert "omitted" in result


def test_compress_graph_pinned_priority():
    pinned = _make_entity("Pinned", "important", ["critical info"])
    pinned.id = 1
    regular = _make_entity("Regular", "normal", ["less important"])
    regular.id = 2
    kg = KnowledgeGraph(entities=[regular, pinned], relations=[])
    result = compress_graph(kg, CompressionLevel.MEDIUM, token_budget=2000, pinned_entity_ids={1})
    # Pinned should appear before regular
    pinned_pos = result.find("Pinned")
    regular_pos = result.find("Regular")
    assert pinned_pos < regular_pos


def test_compress_graph_none_level():
    kg = KnowledgeGraph(
        entities=[_make_entity("A", "t", ["obs"])],
        relations=[],
    )
    result = compress_graph(kg, CompressionLevel.NONE, token_budget=2000)
    assert '"entities"' in result or '"name"' in result  # Should be JSON


def test_budget_includes_omission_footer_and_pinned_overflow():
    pinned = _make_entity("Pinned", "decision", ["critical detail " * 100], ["pinned"])
    pinned.id = 1
    graph = KnowledgeGraph(
        entities=[pinned, *[_make_entity(f"Other{i}", "note", ["fact " * 20]) for i in range(20)]],
        relations=[],
    )

    output = compress_graph(
        graph,
        CompressionLevel.AUTO,
        token_budget=40,
        pinned_entity_ids={1},
    )

    assert _estimate_tokens(output) <= 40
    assert "Pinned" in output


def test_auto_keeps_at_least_as_many_entities_as_best_fixed_layout():
    entities = [_make_entity(f"Entity{i}", "note", [f"important fact {i}"]) for i in range(20)]
    graph = KnowledgeGraph(entities=entities, relations=[])

    fixed = [
        compress_graph(graph, level, token_budget=60)
        for level in (CompressionLevel.LIGHT, CompressionLevel.MEDIUM, CompressionLevel.HEAVY)
    ]
    auto = compress_graph(graph, CompressionLevel.AUTO, token_budget=60)
    def retained(output: str) -> int:
        return sum(entity.name in output for entity in entities)

    assert _estimate_tokens(auto) <= 60
    assert retained(auto) >= max(retained(output) for output in fixed)


def test_unicode_cannot_bypass_token_budget_estimate():
    adversarial = "👩🏽‍💻🧠漢字" * 40

    output = _enforce_budget(adversarial, 40)

    assert _estimate_tokens(output) <= 40
    assert len(output) < len(adversarial) // 4


def test_punctuation_heavy_ascii_cannot_bypass_token_budget():
    adversarial = "! @ # $ % ^ & * ( ) " * 40

    output = _enforce_budget(adversarial, 25)

    assert _estimate_tokens(output) <= 25
    assert len(output) < len(adversarial) // 2
