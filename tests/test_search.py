"""Tests for FTS5 search and graph traversal."""


def test_fts_search_by_name(graph):
    graph.create_entities(
        [
            {"name": "React Frontend", "entityType": "component"},
            {"name": "Python Backend", "entityType": "component"},
        ]
    )
    kg = graph.search_fts("React")
    names = [e.name for e in kg.entities]
    assert "React Frontend" in names
    assert "Python Backend" not in names


def test_fts_search_by_observation(graph):
    graph.create_entities(
        [
            {"name": "App", "entityType": "project", "observations": ["uses SQLite for storage"]},
            {"name": "Other", "entityType": "project", "observations": ["uses Redis"]},
        ]
    )
    kg = graph.search_fts("SQLite")
    names = [e.name for e in kg.entities]
    assert "App" in names


def test_fts_search_by_type(graph):
    graph.create_entities(
        [
            {"name": "Auth", "entityType": "middleware"},
            {"name": "Logger", "entityType": "middleware"},
            {"name": "DB", "entityType": "database"},
        ]
    )
    kg = graph.search_fts("middleware")
    names = [e.name for e in kg.entities]
    assert "Auth" in names
    assert "Logger" in names
    assert "DB" not in names


def test_fuzzy_search_fallback(graph):
    graph.create_entities(
        [
            {"name": "authentication", "entityType": "module"},
        ]
    )
    # Misspelling that won't match FTS but should match fuzzy
    kg = graph.search_fts("authentcation")
    names = [e.name for e in kg.entities]
    assert "authentication" in names


def test_search_filter_by_tags(graph):
    graph.create_entities(
        [
            {
                "name": "P1",
                "entityType": "project",
                "tags": ["pinned"],
                "observations": ["main app"],
            },
            {"name": "P2", "entityType": "project", "observations": ["main service"]},
        ]
    )
    kg = graph.search_fts("main", tags=["pinned"])
    names = [e.name for e in kg.entities]
    assert "P1" in names
    assert "P2" not in names


def test_search_filter_by_type(graph):
    graph.create_entities(
        [
            {"name": "F1", "entityType": "feature", "observations": ["cool feature"]},
            {"name": "B1", "entityType": "bug", "observations": ["cool bug"]},
        ]
    )
    kg = graph.search_fts("cool", entity_types=["feature"])
    names = [e.name for e in kg.entities]
    assert "F1" in names
    assert "B1" not in names


def test_open_nodes_basic(graph):
    graph.create_entities(
        [
            {"name": "A", "entityType": "node"},
            {"name": "B", "entityType": "node"},
            {"name": "C", "entityType": "node"},
        ]
    )
    kg = graph.open_nodes(["A", "B"])
    names = [e.name for e in kg.entities]
    assert "A" in names
    assert "B" in names
    assert "C" not in names


def test_open_nodes_with_depth(graph):
    graph.create_entities(
        [
            {"name": "A", "entityType": "node"},
            {"name": "B", "entityType": "node"},
            {"name": "C", "entityType": "node"},
        ]
    )
    graph.create_relations(
        [
            {"from": "A", "to": "B", "relationType": "connected"},
            {"from": "B", "to": "C", "relationType": "connected"},
        ]
    )
    # depth=1 from A should include B
    kg = graph.open_nodes(["A"], depth=1)
    names = [e.name for e in kg.entities]
    assert "A" in names
    assert "B" in names
    assert "C" not in names

    # depth=2 from A should include B and C
    kg = graph.open_nodes(["A"], depth=2)
    names = [e.name for e in kg.entities]
    assert "A" in names
    assert "B" in names
    assert "C" in names


def test_search_includes_relations(graph):
    graph.create_entities(
        [
            {"name": "Server", "entityType": "component", "observations": ["handles HTTP"]},
            {
                "name": "Database",
                "entityType": "component",
                "observations": ["handles HTTP storage"],
            },
        ]
    )
    graph.create_relations([{"from": "Server", "to": "Database", "relationType": "uses"}])
    kg = graph.search_fts("HTTP")
    assert len(kg.relations) > 0


def test_search_fts_prefers_exact_entity_name_hit(graph):
    graph.create_entities(
        [
            {
                "name": "JWT Notes",
                "entityType": "note",
                "observations": ["JWT config mentioned in passing"],
            },
            {
                "name": "JWT Config",
                "entityType": "config",
                "observations": ["RS256 signing key rotation"],
            },
        ]
    )

    kg = graph.search_fts("JWT Config", limit=5)

    assert kg.entities
    assert kg.entities[0].name == "JWT Config"


def test_fuzzy_search_fallback_preserves_best_match_order(graph):
    graph.create_entities(
        [
            {"name": "auth", "entityType": "module"},
            {"name": "authentication", "entityType": "module"},
        ]
    )

    kg = graph.search_fts("authentcation", limit=5)

    assert kg.entities
    assert kg.entities[0].name == "authentication"
