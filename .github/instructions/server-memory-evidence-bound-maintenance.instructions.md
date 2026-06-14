---
description: 'Use when modifying server-memory Python code, tests, README, docs, or benchmarks. Enforces evidence-bound maintenance: keep implementation/tests/docs/benchmarks aligned, avoid runtime overclaims, preserve memory_context metadata paths, and report proof levels clearly.'
applyTo: 'src/server_memory/**/*.py,tests/**/*.py,README.md,docs/**/*.md,benchmark.py,serve.py,stdio_proxy.py'
---

# Server-Memory Evidence-Bound Maintenance

This file complements `python-mcp-server.instructions.md` with repository-specific maintenance and reporting rules for this workspace.

## Scope

Apply this instruction to substantive work that touches the files matched by `applyTo`, including implementation, tests, README/docs, benchmarks, and MCP entrypoints.

Treat this as an always-on repository rule for covered files, not only for explicit audit tasks.

## Before Changing Anything

- Read the relevant implementation, tests, and documentation before editing.
- Extract explicit requirements separately from inferred needs.
- Prefer the smallest correct patch set.
- Preserve existing public behavior unless the current behavior is the proven mismatch.

## Proof Levels

Every material conclusion in code-review, maintenance, benchmark, or handoff work must fit exactly one of these categories:

- `verified in code`
- `verified by executed test`
- `verified by benchmark run`
- `verified by runtime check`
- `inferred but not verified`

Do not blur these categories together.

## Operational Claim Rules

- Never state that a daemon, MCP server, proxy, CLI integration, restart, or network endpoint is healthy, working, fixed, resolved, or verified unless that was directly checked in the current session.
- If a live path was not exercised, say `Not verified in this session`.
- Do not fabricate before/after benchmark deltas. Only report a delta if both baseline and post-change runs were reproduced in the current session.

## Alignment Rules

Keep implementation, tests, docs, benchmarks, and summary language in strict alignment.

### Metadata Path Integrity

If graph-layer `memory_context` results feed server-layer formatting or tests, preserve formatter-needed fields end-to-end.

Do not silently drop these fields when available:

- `score`
- `snippets`
- `conflict`
- `stale`
- `signals`

### Formatter Integrity

- Keep `memory_context` rendering compact, deterministic, and easy to test.
- Surface `!conflict` and `!stale` markers when the corresponding metadata is present.
- If formatter behavior changes, update exact-output tests rather than relying on incidental coverage.

### Benchmark Integrity

- Preserve the fixed retrieval benchmark harness.
- Report current `hit_at_1` and `hit_at_3` honestly.
- Do not describe improvement unless both sides of the comparison were run in the current session.

### Documentation Integrity

- Keep README and docs evidence-neutral.
- Do not imply runtime proof that the repository itself does not establish.
- Prefer wording like `to verify`, `example`, `recommended`, or `not verified in this session` when proof is absent.

## Testing Expectations

- Add or update narrow behavior-focused tests for the exact behavior changed.
- Run the narrowest meaningful tests first.
- Expand to broader suites only as needed.
- When changing formatter behavior, test the exact surfaced markers and metadata path.
- When changing retrieval behavior, prefer tests that validate ranking, deduplication, snippet selection, and benchmark reporting directly.

## Reporting Expectations

Use the proof-level labels in every substantive completion summary for tasks touching covered files, especially when making claims about correctness, verification, benchmarks, docs, or runtime status.

For repo alignment or maintenance tasks, prefer this structure in the final report:

- `Files Changed`
- `What Is Now Verified`
- `What Is Improved But Not Runtime-Verified`
- `Exact Tests Run And Results`
- `Remaining Gaps`

For every verification claim, include the command, artifact, or file inspection that supports it.

## Repository-Specific Priorities

When touching retrieval work in this repository:

- Follow the repository's existing tooling and boot conventions unless a task explicitly asks to change them.
- Do not switch this repo to `uv`, alternate packaging, or different MCP runtime patterns just because a generic instruction mentions them.
- Ground environment, packaging, and startup guidance in the actual repository files such as `pyproject.toml`, `README.md`, and the current server entrypoints.
- Preserve `memory_context` as the primary workflow.
- Favor observation-level relevance, importance, confidence, recency, and activity signals over noisy volume.
- Keep benchmark, README, and tests synchronized with actual behavior.
- If a prior summary overstated certainty, correct it explicitly rather than softening it indirectly.
