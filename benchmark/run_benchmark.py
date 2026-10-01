#!/usr/bin/env python3
"""server-memory MCP Benchmark Runner.

Runs coding tasks with and without server-memory MCP, measuring:
- Accuracy (does the code work / tests pass)
- Speed (wall-clock seconds)
- Token usage (input + output)
- Error rate (tool failures)
- DB lock detection (concurrent stress test)
"""

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import yaml

BENCHMARK_DIR = Path(__file__).parent
PROJECT_DIR = BENCHMARK_DIR.parent
TASKS_FILE = BENCHMARK_DIR / "tasks.yaml"
RESULTS_DIR = BENCHMARK_DIR / "results"
RUNTIME_DIR = BENCHMARK_DIR / ".runtime"

MCP_CONFIG: Path | None = None
DB_PATH: Path | None = None
TELEMETRY_PATH: Path | None = None

# Lock stress test state
lock_errors: list[str] = []
stress_stop = threading.Event()


@contextmanager
def isolated_project_copy(source: Path):
    """Yield a disposable project copy and never mutate the source tree."""
    source = source.resolve(strict=True)
    if not source.is_dir() or source == Path(source.anchor):
        raise RuntimeError(f"Unsafe benchmark source root: {source}")
    with tempfile.TemporaryDirectory(prefix="server-memory-benchmark-") as temp_dir:
        trial = Path(temp_dir) / "project"
        shutil.copytree(
            source,
            trial,
            symlinks=True,
            ignore=shutil.ignore_patterns(
                ".git", ".venv", ".runtime", "__pycache__", ".pytest_cache", "results"
            ),
        )
        if source == trial or source in trial.parents or trial in source.parents:
            raise RuntimeError("Benchmark isolation could not be established")
        yield trial


def append_telemetry_event(path: Path, event: str) -> None:
    """Append one local benchmark-treatment event."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": event, "timestamp": time.time()}) + "\n")


def validate_treatment_activation(path: Path, *, enabled: bool) -> dict[str, int]:
    """Fail closed unless telemetry proves the requested benchmark arm."""
    events: list[str] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line).get("event")
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid treatment telemetry: {exc}") from exc
            if isinstance(event, str):
                events.append(event)
    evidence = {
        "mcp_handshakes": events.count("mcp_handshake"),
        "memory_tool_calls": events.count("tool_call"),
    }
    if enabled and evidence["mcp_handshakes"] < 1:
        raise RuntimeError("Enabled treatment recorded no MCP handshake")
    if enabled and evidence["memory_tool_calls"] < 1:
        raise RuntimeError("Enabled treatment recorded no memory tool call")
    if not enabled and any(evidence.values()):
        raise RuntimeError("Memory activity was observed in the disabled treatment")
    return evidence


def load_tasks() -> list[dict]:
    with open(TASKS_FILE) as f:
        data = yaml.safe_load(f)
    return data["tasks"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run server-memory benchmark tasks safely")
    parser.add_argument(
        "--mcp-config",
        type=Path,
        default=RUNTIME_DIR / "mcp-config.json",
        help="Path to the MCP config file the benchmark runner is allowed to rewrite.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=RUNTIME_DIR / "memory.db",
        help="Path to the SQLite database used by the optional lock stress thread.",
    )
    parser.add_argument(
        "--telemetry-path",
        type=Path,
        default=RUNTIME_DIR / "treatment-telemetry.jsonl",
        help="Local JSONL evidence for MCP initialization and memory tool calls.",
    )
    return parser.parse_args()


def configure_runtime(args: argparse.Namespace) -> None:
    global MCP_CONFIG, DB_PATH, TELEMETRY_PATH
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    MCP_CONFIG = args.mcp_config.expanduser().resolve()
    DB_PATH = args.db_path.expanduser().resolve()
    TELEMETRY_PATH = args.telemetry_path.expanduser().resolve()
    MCP_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def ensure_mcp_config_exists() -> None:
    if MCP_CONFIG is None:
        raise RuntimeError("Benchmark runtime was not configured")
    if not MCP_CONFIG.exists():
        with open(MCP_CONFIG, "w", encoding="utf-8") as f:
            json.dump({"mcpServers": {}}, f, indent=2)


def run_claude_task(
    prompt: str,
    timeout: int = 120,
    *,
    project_dir: Path = PROJECT_DIR,
    memory_enabled: bool = False,
) -> dict:
    """Run a single task via `claude -p` and capture metrics."""
    start = time.time()
    try:
        command = ["claude", "-p", "--output-format", "json", "--max-turns", "5"]
        if MCP_CONFIG is not None:
            command.extend(["--mcp-config", str(MCP_CONFIG), "--strict-mcp-config"])
        treatment_prompt = prompt
        if memory_enabled:
            treatment_prompt = (
                "Before editing, call the server-memory memory_context tool once with a concise "
                "hint for this task, then complete the task.\n\n" + prompt
            )
        result = subprocess.run(
            command,
            input=treatment_prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(project_dir),
        )
        elapsed = time.time() - start
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            return {
                "success": False,
                "elapsed": elapsed,
                "output": result.stdout[:2000],
                "error": f"JSON parse failed: {result.stderr[:500]}",
                "tokens_in": 0,
                "tokens_out": 0,
            }

        # Extract token usage from the JSON output
        tokens_in = data.get("usage", {}).get("input_tokens", 0) or 0
        tokens_out = data.get("usage", {}).get("output_tokens", 0) or 0
        text = data.get("result", data.get("text", result.stdout[:2000]))

        return {
            "success": result.returncode == 0,
            "elapsed": elapsed,
            "output": str(text)[:2000],
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "error": "",
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "elapsed": timeout,
            "output": "",
            "error": "timeout",
            "tokens_in": 0,
            "tokens_out": 0,
        }
    except Exception as e:
        return {
            "success": False,
            "elapsed": time.time() - start,
            "output": "",
            "error": str(e),
            "tokens_in": 0,
            "tokens_out": 0,
        }


def verify_task(task: dict, *, project_dir: Path = PROJECT_DIR) -> tuple[bool, str]:
    """Verify a task's output. Returns (passed, detail)."""
    method = task["verify"]

    if method == "grep":
        target = project_dir / task["verify_file"]
        if not target.exists():
            return False, f"File not found: {target}"
        content = target.read_text()
        pattern = task["verify_pattern"]
        if re.search(pattern, content):
            return True, f"Pattern '{pattern}' found"
        return False, f"Pattern '{pattern}' NOT found in {target.name}"

    elif method == "pytest":
        test_code = task["verify_test"]
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix="bench_test_",
            dir=tempfile.gettempdir(),
            delete=False,
        ) as f:
            f.write(test_code)
            test_path = f.name
        try:
            r = subprocess.run(
                [sys.executable, "-m", "pytest", test_path, "-x", "-q", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(project_dir),
                env={**os.environ, "PYTHONPATH": str(project_dir / "src")},
            )
            passed = r.returncode == 0
            detail = r.stdout[-500:] if r.stdout else r.stderr[-500:]
            return passed, detail.strip()
        except Exception as e:
            return False, str(e)
        finally:
            os.unlink(test_path)

    return False, f"Unknown verify method: {method}"


def db_stress_thread():
    """Concurrent DB writer to detect lock issues during benchmark."""
    global lock_errors
    if DB_PATH is None:
        return
    while not stress_stop.is_set():
        try:
            conn = sqlite3.connect(str(DB_PATH), timeout=3)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT OR IGNORE INTO entities (name, entity_type) VALUES (?, ?)",
                (f"_bench_stress_{time.time()}", "benchmark_probe"),
            )
            conn.commit()
            # Clean up probe rows
            conn.execute("DELETE FROM entities WHERE entity_type = 'benchmark_probe'")
            conn.commit()
            conn.close()
        except sqlite3.OperationalError as e:
            if "locked" in str(e):
                lock_errors.append(f"{datetime.now().isoformat()}: {e}")
        except Exception:
            pass
        time.sleep(0.5)


def toggle_mcp(enabled: bool):
    """Enable or disable server-memory MCP in an explicitly configured file only."""
    if MCP_CONFIG is None:
        raise RuntimeError("Benchmark runtime was not configured")

    ensure_mcp_config_exists()
    with open(MCP_CONFIG, encoding="utf-8") as f:
        config = json.load(f)

    servers = config.get("mcpServers", {})
    filtered = {k: v for k, v in servers.items() if "memory" not in k.lower()}

    if enabled:
        filtered["server-memory"] = {
            "command": sys.executable,
            "args": ["-m", "server_memory"],
            "env": {
                "MEMORY_DB_PATH": str(DB_PATH) if DB_PATH is not None else "",
                "MEMORY_BENCHMARK_TELEMETRY_PATH": (
                    str(TELEMETRY_PATH) if TELEMETRY_PATH is not None else ""
                ),
                "PYTHONPATH": str(PROJECT_DIR / "src"),
            },
        }

    config["mcpServers"] = filtered
    with open(MCP_CONFIG, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def run_suite(tasks: list[dict], mode: str) -> list[dict]:
    """Run all tasks in a given mode (with_memory / without_memory)."""
    results = []
    for i, task in enumerate(tasks):
        tid = task["id"]
        print(f"  [{mode}] {i + 1}/{len(tasks)}: {tid} ({task['tier']})...", end=" ", flush=True)

        if TELEMETRY_PATH is None:
            raise RuntimeError("Benchmark runtime was not configured")
        TELEMETRY_PATH.unlink(missing_ok=True)
        enabled = mode == "with_memory"
        with isolated_project_copy(PROJECT_DIR) as trial_dir:
            metrics = run_claude_task(
                task["prompt"],
                timeout=180,
                project_dir=trial_dir,
                memory_enabled=enabled,
            )
            passed, detail = verify_task(task, project_dir=trial_dir)
        treatment = validate_treatment_activation(TELEMETRY_PATH, enabled=enabled)

        # Score: 100 if passed, 0 if not
        score = 100 if passed else 0

        result = {
            "task_id": tid,
            "tier": task["tier"],
            "mode": mode,
            "score": score,
            "passed": passed,
            "verify_detail": detail,
            "elapsed_sec": round(metrics["elapsed"], 1),
            "tokens_in": metrics["tokens_in"],
            "tokens_out": metrics["tokens_out"],
            "tokens_total": metrics["tokens_in"] + metrics["tokens_out"],
            "error": metrics["error"],
            "treatment": treatment,
        }
        results.append(result)
        status = "PASS" if passed else "FAIL"
        print(f"{status} ({result['elapsed_sec']}s, {result['tokens_total']} tok)")

    return results


def print_summary(results: list[dict]):
    """Print a comparison table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    for mode in ["with_memory", "without_memory"]:
        mode_results = [r for r in results if r["mode"] == mode]
        if not mode_results:
            continue
        print(f"\n--- {mode.upper()} ---")
        for tier in ["low", "mid", "high"]:
            tier_results = [r for r in mode_results if r["tier"] == tier]
            if not tier_results:
                continue
            avg_score = sum(r["score"] for r in tier_results) / len(tier_results)
            avg_time = sum(r["elapsed_sec"] for r in tier_results) / len(tier_results)
            avg_tokens = sum(r["tokens_total"] for r in tier_results) / len(tier_results)
            pass_count = sum(1 for r in tier_results if r["passed"])
            print(
                f"  {tier.upper():>5}: {pass_count}/{len(tier_results)} passed | "
                f"avg {avg_time:.1f}s | avg {avg_tokens:.0f} tokens | "
                f"accuracy {avg_score:.0f}%"
            )

        total = mode_results
        total_pass = sum(1 for r in total if r["passed"])
        total_score = sum(r["score"] for r in total) / len(total)
        total_time = sum(r["elapsed_sec"] for r in total)
        total_tokens = sum(r["tokens_total"] for r in total)
        print(
            f"  TOTAL: {total_pass}/{len(total)} passed | "
            f"{total_time:.1f}s total | {total_tokens} total tokens | "
            f"accuracy {total_score:.0f}%"
        )

    # Comparison
    with_results = [r for r in results if r["mode"] == "with_memory"]
    without_results = [r for r in results if r["mode"] == "without_memory"]
    if with_results and without_results:
        print("\n--- COMPARISON (with_memory vs without_memory) ---")
        w_acc = sum(r["score"] for r in with_results) / len(with_results)
        wo_acc = sum(r["score"] for r in without_results) / len(without_results)
        w_time = sum(r["elapsed_sec"] for r in with_results)
        wo_time = sum(r["elapsed_sec"] for r in without_results)
        w_tok = sum(r["tokens_total"] for r in with_results)
        wo_tok = sum(r["tokens_total"] for r in without_results)
        print(f"  Accuracy:  {w_acc:.0f}% vs {wo_acc:.0f}% (delta: {w_acc - wo_acc:+.0f}%)")
        print(f"  Time:      {w_time:.0f}s vs {wo_time:.0f}s (delta: {w_time - wo_time:+.0f}s)")
        print(f"  Tokens:    {w_tok} vs {wo_tok} (delta: {w_tok - wo_tok:+d})")

    # Lock errors
    if lock_errors:
        print(f"\n--- DB LOCK ERRORS: {len(lock_errors)} ---")
        for e in lock_errors[:5]:
            print(f"  {e}")
    else:
        print("\n--- DB LOCK: No lock errors detected ---")


def main():
    args = parse_args()
    configure_runtime(args)
    RESULTS_DIR.mkdir(exist_ok=True)
    tasks = load_tasks()
    print(f"Loaded {len(tasks)} tasks")
    print(f"Using MCP config: {MCP_CONFIG}")
    print(f"Using benchmark DB: {DB_PATH}")

    all_results = []

    # Start DB stress thread
    print("Starting DB lock stress test thread...")
    stress_thread = threading.Thread(target=db_stress_thread, daemon=True)
    stress_thread.start()

    # Run WITH memory
    print("\n=== Phase 1: WITH server-memory ===")
    toggle_mcp(True)
    with_results = run_suite(tasks, "with_memory")
    all_results.extend(with_results)

    # Run WITHOUT memory
    print("\n=== Phase 2: WITHOUT server-memory ===")
    toggle_mcp(False)
    without_results = run_suite(tasks, "without_memory")
    all_results.extend(without_results)

    # Restore MCP config
    toggle_mcp(True)

    # Stop stress thread
    stress_stop.set()
    stress_thread.join(timeout=3)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS_DIR / f"benchmark-{timestamp}.json"
    output = {
        "timestamp": timestamp,
        "tasks_count": len(tasks),
        "lock_errors": lock_errors,
        "results": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    print_summary(all_results)


if __name__ == "__main__":
    main()
