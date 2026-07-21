"""Tests for the fixed memory_context benchmark scenario table."""

from tests.memory_context_benchmarks import (
    BENCHMARK_SCENARIOS,
    run_memory_context_benchmark,
)


def test_memory_context_benchmark_scenarios_are_fixed_and_reportable():
    results = run_memory_context_benchmark()

    assert len(BENCHMARK_SCENARIOS) >= 6
    assert results["total"] == len(BENCHMARK_SCENARIOS)
    assert 0 <= results["hit_at_1"] <= results["hit_at_3"] <= results["total"]
    assert isinstance(results["misses"], list)
    assert results["hit_at_1"] >= results["total"] - 1
    assert results["hit_at_3"] == results["total"]
