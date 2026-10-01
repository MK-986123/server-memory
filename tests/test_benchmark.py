"""Focused tests for benchmark helper separation between retrieval, ranking, and compression."""

from __future__ import annotations

from benchmark.suite import (
    ENTITIES,
    RECALL_QUERIES,
    CompressionLevel,
    evaluate_compressed_fact_presence,
    evaluate_memory_context_recall,
    evaluate_search_recall,
    setup_db,
)
from server_memory.compression import compress_graph


def test_search_recall_reports_expected_shape():
    db, graph = setup_db()
    try:
        result = evaluate_search_recall(graph, RECALL_QUERIES[:3], limit=5)
    finally:
        db.close()

    assert result["total"] == 3
    assert 0 <= result["correct"] <= result["total"]
    assert isinstance(result["misses"], list)


def test_memory_context_recall_reports_hit_at_1_and_hit_at_3():
    db, graph = setup_db()
    try:
        result = evaluate_memory_context_recall(graph, RECALL_QUERIES[:5], limit=3)
    finally:
        db.close()

    assert result["total"] == 5
    assert 0 <= result["hit_at_1"] <= result["hit_at_3"] <= result["total"]
    assert isinstance(result["misses"], list)


def test_compressed_fact_presence_is_reported_per_level():
    db, graph = setup_db()
    try:
        kg = graph.read_graph()
        compressed_outputs = {
            level: compress_graph(kg, level=level, token_budget=0)
            for level in (CompressionLevel.NONE, CompressionLevel.MEDIUM, CompressionLevel.HEAVY)
        }
    finally:
        db.close()

    result = evaluate_compressed_fact_presence(compressed_outputs, RECALL_QUERIES[:4])

    assert set(result) == {CompressionLevel.NONE, CompressionLevel.MEDIUM, CompressionLevel.HEAVY}
    for level_result in result.values():
        assert level_result["total"] == 4
        assert 0 <= level_result["correct"] <= level_result["total"]
        assert isinstance(level_result["misses"], list)


def test_benchmark_fixture_dataset_is_non_empty():
    assert ENTITIES
    assert RECALL_QUERIES
