"""Safety and treatment-activation tests for the external benchmark runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "server_memory_external_benchmark", Path(__file__).parents[1] / "benchmark" / "run_benchmark.py"
)
assert _SPEC is not None and _SPEC.loader is not None
run_benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_benchmark)


def test_trial_copy_cannot_modify_source_sentinels(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    tracked = source / "tracked.txt"
    untracked = source / "untracked.txt"
    tracked.write_bytes(b"tracked-local-edit\x00")
    untracked.write_bytes(b"untracked-local-file\xff")

    with run_benchmark.isolated_project_copy(source) as trial:
        (trial / "tracked.txt").write_bytes(b"agent rewrite")
        (trial / "untracked.txt").unlink()
        (trial / "new.txt").write_text("new trial artifact")

    assert tracked.read_bytes() == b"tracked-local-edit\x00"
    assert untracked.read_bytes() == b"untracked-local-file\xff"
    assert not (source / "new.txt").exists()


def test_treatment_activation_requires_handshake_and_tool_use(tmp_path):
    telemetry = tmp_path / "telemetry.jsonl"
    run_benchmark.append_telemetry_event(telemetry, "mcp_handshake")
    run_benchmark.append_telemetry_event(telemetry, "tool_call")

    evidence = run_benchmark.validate_treatment_activation(telemetry, enabled=True)

    assert evidence == {"mcp_handshakes": 1, "memory_tool_calls": 1}


def test_enabled_treatment_fails_closed_without_observed_tool_use(tmp_path):
    telemetry = tmp_path / "telemetry.jsonl"
    run_benchmark.append_telemetry_event(telemetry, "mcp_handshake")

    with pytest.raises(RuntimeError, match="tool call"):
        run_benchmark.validate_treatment_activation(telemetry, enabled=True)


def test_disabled_treatment_requires_zero_memory_events(tmp_path):
    telemetry = tmp_path / "telemetry.jsonl"

    assert run_benchmark.validate_treatment_activation(telemetry, enabled=False) == {
        "mcp_handshakes": 0,
        "memory_tool_calls": 0,
    }

    run_benchmark.append_telemetry_event(telemetry, "tool_call")
    with pytest.raises(RuntimeError, match="disabled"):
        run_benchmark.validate_treatment_activation(telemetry, enabled=False)
