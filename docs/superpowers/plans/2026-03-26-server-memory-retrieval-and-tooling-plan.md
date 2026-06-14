# Server-Memory Retrieval And Tooling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> Updated after review feedback. Some phase-1 ranking work may already exist; the tasks below define the full delta needed to satisfy the revised design rather than assuming the original draft is sufficient.

**Goal:** Improve `memory_context` so exact clues surface the right facts first, duplicates stop crowding out distinct memories, stored metadata is actually used by retrieval, snippet quality improves materially, and the current MCP workflow stays simple.

**Architecture:** Keep `memory_context` as the primary tool, and evolve the staged candidate pipeline in `graph.py` so it uses the full intended signal set. Rank candidates using exact, lexical, semantic, observation-level importance and confidence, pinned state, update recency, access recency, and recent activity signals; then let `server.py` render cleaner context lines from the improved result set under a token-aware snippet budget. Add benchmark and regression tests first and keep any helper-tool work optional until the ranking behavior is stable.

**Tech Stack:** Python 3.14, FastMCP, SQLite/FTS5, pytest

---

## File Map

- Modify: `src/server_memory/graph.py`
  - Own the candidate pipeline, weighted ranking, duplicate suppression, and `memory_context()` data assembly.
- Modify: `src/server_memory/server.py`
  - Keep the MCP tool contract stable while formatting richer `memory_context()` results.
- Modify: `tests/test_tools.py`
  - Add top-level `memory_context()` behavior tests.
- Modify: `tests/test_search.py`
  - Add lexical ranking and exact-match priority tests if the lower-level search path needs direct coverage.
- Modify: `tests/test_hybrid_search.py`
  - Add semantic-plus-lexical balance tests and embeddings-off fallback tests.
- Optional modify: `src/server_memory/server.py`
  - Only if needed, add one narrow diagnostic MCP tool such as `explain_memory_match`.
- Optional modify: `tests/`
  - Add a benchmark fixture or scenario table for `hit@1` and `hit@3` measurement.

## Constraints

- Do not redesign the main MCP workflow.
- Do not make embeddings mandatory.
- Keep outputs compact enough for turn-start use.
- Use existing stored metadata before introducing new schema fields.
- Preserve existing tests unless behavior is intentionally improved.
- Replace any fixed 50-character snippet assumptions with token-aware expectations where behavior is intentionally improved.
- Explicitly defer aliases, suppression flags, structured hints, caching, and relation-weight graph expansion unless this plan is revised again.
- This directory is not currently a git repo, so commit steps below should be skipped unless version control is initialized first.

### Task 0: Capture Baseline Retrieval Benchmarks

**Files:**
- Modify: `tests/test_tools.py`
- Optional add: `tests/test_memory_context_benchmarks.py`
- Test: `tests/test_tools.py`

- [ ] **Step 1: Define 10 to 20 benchmark scenarios**

Create a fixed scenario table of:

- hint
- fixture setup
- expected top result
- expected top-3 membership when ambiguity is intentional

Include at least:

- exact entity-name lookup
- high-importance observation beating noisy mentions
- pinned durable fact beating recent chatter
- recently accessed entity earning only a mild tie-break boost
- activity-linked entity being preferred when the activity is truly relevant
- technical hint shape preferring `file_path`, `config`, or `code_snippet`

- [ ] **Step 2: Record current baseline `hit@1` and `hit@3`**

Run the benchmark suite against the current implementation before the retrieval rewrite is considered complete.

- [ ] **Step 3: Preserve the benchmark suite for post-change comparison**

The same scenarios must be rerun after implementation and the final handoff must report the delta.

### Task 1: Baseline `memory_context` Regression Tests

**Files:**
- Modify: `tests/test_tools.py`
- Test: `tests/test_tools.py`

- [ ] **Step 1: Add a failing exact-priority test**

```python
def test_memory_context_prefers_exact_hint_match(graph):
    graph.create_entities([
        {
            "name": "JWT Config",
            "entityType": "config",
            "observations": ["RS256 signing key rotation"],
        },
        {
            "name": "Auth Notes",
            "entityType": "note",
            "observations": ["JWT is used in several places"],
        },
    ])

    ctx = graph.memory_context(hint="JWT Config")

    assert ctx["hint_matches"]
    assert ctx["hint_matches"][0]["name"] == "JWT Config"
```

- [ ] **Step 2: Add a failing pinned-vs-recency test**

```python
def test_memory_context_prefers_pinned_fact_over_recent_noise(graph):
    graph.create_entities([
        {
            "name": "Critical Config",
            "entityType": "config",
            "tags": ["pinned"],
            "observations": ["Use provider X for production auth"],
        },
        {
            "name": "Recent Chatter",
            "entityType": "note",
            "observations": ["provider x mentioned in passing"],
        },
    ])
    graph.log_activity(action="discussed", summary="provider x came up again")

    ctx = graph.memory_context(hint="provider x")

    assert ctx["hint_matches"]
    assert ctx["hint_matches"][0]["name"] == "Critical Config"
```

- [ ] **Step 3: Add a failing duplicate-suppression test**

```python
def test_memory_context_deduplicates_near_identical_matches(graph):
    graph.create_entities([
        {
            "name": "Auth Canonical",
            "entityType": "module",
            "observations": ["Handles JWT refresh token validation"],
        },
        {
            "name": "Auth Duplicate",
            "entityType": "module",
            "observations": ["Handles JWT refresh token validation"],
        },
    ])

    ctx = graph.memory_context(hint="refresh token")
    names = [match["name"] for match in ctx["hint_matches"]]

    assert len(names) == len(set(names))
    assert len(names) == 1
```

- [ ] **Step 4: Run the focused tests and verify they fail**

Run:

```bash
pytest tests/test_tools.py -k "memory_context and (prefers_exact_hint_match or prefers_pinned_fact_over_recent_noise or deduplicates_near_identical_matches)" -v
```

Expected: use these as a regression gate. On a branch that does not yet implement the revised ranking semantics, at least some of these tests should fail.

- [ ] **Step 5: Skip commit or commit if VCS exists**

```bash
git  status
```

Expected: currently not a git repo; skip commit unless that changes.

### Task 2: Add Lower-Level Search Coverage

**Files:**
- Modify: `tests/test_search.py`
- Modify: `tests/test_hybrid_search.py`
- Test: `tests/test_search.py`
- Test: `tests/test_hybrid_search.py`

- [ ] **Step 1: Add a failing lexical exact-match test in `test_search.py`**

```python
def test_search_fts_prefers_exact_entity_name_hit(graph):
    graph.create_entities([
        {"name": "JWT Config", "entityType": "config", "observations": ["RS256"]},
        {"name": "JWT Notes", "entityType": "note", "observations": ["JWT config mentioned"]},
    ])

    kg = graph.search_fts("JWT Config", limit=5)

    assert kg.entities
    assert kg.entities[0].name == "JWT Config"
```

- [ ] **Step 2: Add a failing hybrid-balance test in `test_hybrid_search.py`**

```python
def test_memory_context_balances_exact_and_semantic_signals(graph_with_embeddings):
    ctx = graph_with_embeddings.memory_context(hint="jwt rotation config")
    assert ctx["hint_matches"]
```

Note:
- Use the existing mock embedding pattern in `tests/test_hybrid_search.py`.
- The assertion should be made concrete to the fixture data you add there.

- [ ] **Step 3: Run the targeted search tests and verify failure**

Run:

```bash
pytest tests/test_search.py tests/test_hybrid_search.py -k "exact_entity_name_hit or balances_exact_and_semantic_signals" -v
```

Expected: use these as a regression gate. They should fail on an implementation that lacks the revised exact-match and hybrid-balance behavior.

- [ ] **Step 4: Skip commit or commit if VCS exists**

```bash
git  status
```

### Task 3: Extend Ranked Candidate Helpers In `graph.py`

**Files:**
- Modify: `src/server_memory/graph.py`
- Test: `tests/test_search.py`
- Test: `tests/test_hybrid_search.py`

- [ ] **Step 1: Add or extend focused helper methods for candidate collection**

Keep the candidate logic factored into small internal helpers rather than embedding all logic inside `memory_context()`:

```python
def _collect_memory_context_candidates(self, hint: str, limit: int, project: str) -> list[dict[str, Any]]:
    ...

def _score_memory_context_candidates(self, candidates: list[dict[str, Any]], hint: str) -> list[dict[str, Any]]:
    ...

def _dedup_memory_context_candidates(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ...
```

Keep them private and colocated near `memory_context()`.

- [ ] **Step 2: Implement exact, lexical, semantic, importance, confidence, access, activity, and recency features**

Minimum scoring shape:

```python
score = (
    exact_name_boost
    + exact_phrase_boost
    + lexical_score_weight * lexical_score
    + semantic_score_weight * semantic_score
    + pinned_boost
    + observation_evidence_score
    + obs_type_boost
    + update_recency_boost
    + access_recency_boost
    + activity_boost
    - stale_penalty
)
```

Implementation rules:
- exact strong clue should dominate
- observation evidence should favor one high-value relevant observation over many weak mentions
- `importance` and `confidence` must be derived from matching observations, not only an entity-level max
- access recency and activity boosts must be bounded and secondary
- protected or technical `obs_type` values only get a secondary boost when the hint shape suggests that type
- recency must be bounded
- embeddings must remain optional
- relation weights are explicitly out of scope for this task

- [ ] **Step 3: Add conservative dedup suppression**

Implement dedup so near-identical duplicate entities do not both survive final ranking:

```python
canonical_key = (
    normalized_name_or_best_snippet,
    entity_type,
)
```

Prefer:
- pinned entity over unpinned
- higher score over lower score
- newer entity only as a later tie-break

- [ ] **Step 4: Add sparse-graph and stale-candidate handling**

Rules:
- if the graph is very small, prefer broad useful coverage over overfitted ranking heuristics
- stale candidates should receive a mild penalty rather than disappearing
- unpinned, long-unaccessed entities should not outrank durable high-signal matches

- [ ] **Step 5: Run targeted failing tests until they pass**

Run:

```bash
pytest tests/test_tools.py -k "memory_context" -v
pytest tests/test_search.py tests/test_hybrid_search.py -k "exact_entity_name_hit or balances_exact_and_semantic_signals" -v
```

Expected: PASS for new ranking-related tests.

- [ ] **Step 6: Skip commit or commit if VCS exists**

```bash
git  status
```

### Task 4: Rework `memory_context()` Assembly

**Files:**
- Modify: `src/server_memory/graph.py`
- Modify: `src/server_memory/server.py`
- Test: `tests/test_tools.py`

- [ ] **Step 1: Update graph-layer `memory_context()` to return richer match metadata**

The returned `hint_matches` items should include enough structure for safe formatting:

```python
{
    "name": entity.name,
    "type": entity.entity_type,
    "snippets": snippets,
    "score": round(score, 4),
    "signals": ["exact_name", "pinned"],
    "conflict": False,
    "stale": False,
}
```

Note:
- keep extra fields additive so server formatting can choose what to expose
- do not force all metadata into the final string output

- [ ] **Step 2: Replace flat snippet truncation with token-aware snippet selection**

Rules:
- rank candidate observations by hint relevance first, then importance, confidence, and observation type
- prefer 1 to 2 strong observations over several chopped fragments
- use a small snippet budget per entity or per output section instead of a hard 50-character cutoff
- update or replace existing tests that assert the old 50-character behavior

- [ ] **Step 3: Update the MCP tool formatter in `server.py`**

Keep the current compact style, but format from the stronger structure:

```python
if result["hint_matches"]:
    hint_parts = []
    for h in result["hint_matches"]:
        part = f"{h['name']}[{h['type']}]"
        if h.get("snippets"):
            part += ': "' + '" | "'.join(h["snippets"]) + '"'
        if h.get("conflict"):
            part += " !conflict"
        if h.get("stale"):
            part += " !stale"
        hint_parts.append(part)
```

Do not expose raw numeric scores in the default user-facing output.

- [ ] **Step 4: Run the full `test_tools.py` suite**

Run:

```bash
pytest tests/test_tools.py -v
```

Expected: PASS with existing and new `memory_context()` tests.

- [ ] **Step 5: Skip commit or commit if VCS exists**

```bash
git  status
```

### Task 5: Add Conflict, Staleness, Activity, And Fallback Coverage

**Files:**
- Modify: `tests/test_tools.py`
- Modify: `src/server_memory/graph.py`
- Test: `tests/test_tools.py`

- [ ] **Step 1: Add failing tests for conflict and staleness markers**

```python
def test_memory_context_marks_conflicting_high_value_matches(graph):
    ...

def test_memory_context_marks_stale_low_confidence_match(graph):
    ...
```

Conflict is in scope for this revised plan, so at least one conflict-surfacing test should be present if the implementation exposes the marker.

- [ ] **Step 2: Add failing tests for activity and access-recency boosts**

Include cases where:

- a recently accessed entity earns a mild tie-break boost
- a recently discussed entity linked through `activity_log` outranks unrelated lexical noise
- the activity signal does not overpower a stronger exact or pinned candidate

- [ ] **Step 3: Add a failing embeddings-off fallback test**

```python
def test_memory_context_works_without_embeddings(graph):
    graph.create_entities([
        {"name": "Plain Lexical Match", "entityType": "note", "observations": ["JWT config lives here"]},
    ])

    ctx = graph.memory_context(hint="JWT config")
    assert ctx["hint_matches"][0]["name"] == "Plain Lexical Match"
```

- [ ] **Step 4: Implement the minimal fallback behavior**

Rules:
- lexical-only path must still yield useful ranking
- fuzzy fallback should only engage when stronger signals are absent
- duplicate suppression must still apply without embeddings

- [ ] **Step 5: Run focused tests**

Run:

```bash
pytest tests/test_tools.py -k "conflict or stale or activity or without_embeddings" -v
```

Expected: PASS for whichever tests were added.

- [ ] **Step 6: Skip commit or commit if VCS exists**

```bash
git  status
```

### Task 6: Optional Diagnostic MCP Tool

**Files:**
- Modify: `src/server_memory/server.py`
- Modify: `tests/test_tools.py`
- Test: `tests/test_tools.py`

- [ ] **Step 1: Decide whether the helper tool is still necessary**

Add the tool only if ranking behavior is hard to understand from tests and logs alone.

- [ ] **Step 2: If needed, add one narrow diagnostic tool**

Candidate:

```python
@mcp.tool()
def explain_memory_match(ctx: Context, hint: str, limit: int = 5) -> str:
    ...
```

Return:
- candidate names
- major ranking signals
- dedup decision summary

Do not make this a required part of normal usage.

- [ ] **Step 3: Add tool tests**

Run:

```bash
pytest tests/test_tools.py -k "explain_memory_match" -v
```

Expected: PASS if the tool is added; skip entirely otherwise.

- [ ] **Step 4: Skip commit or commit if VCS exists**

```bash
git  status
```

### Task 7: Final Verification

**Files:**
- Test: `tests/test_tools.py`
- Test: `tests/test_search.py`
- Test: `tests/test_hybrid_search.py`
- Test: benchmark suite from Task 0

- [ ] **Step 1: Run the targeted retrieval suites**

Run:

```bash
pytest tests/test_tools.py tests/test_search.py tests/test_hybrid_search.py -v
```

Expected: PASS

- [ ] **Step 2: Rerun the benchmark suite and compare with baseline**

Required output:

- baseline `hit@1`
- baseline `hit@3`
- post-change `hit@1`
- post-change `hit@3`
- brief note on any scenarios that still miss

- [ ] **Step 3: Run a quick import sanity check**

Run:

```bash
.venv/bin/python -c "from server_memory.graph import KnowledgeGraphManager; from server_memory.db import Database; db = Database(':memory:'); db.open(); graph = KnowledgeGraphManager(db); print(graph.memory_context(hint='test'))"
```

Expected: command succeeds and prints a valid context structure.

- [ ] **Step 4: Record final notes**

Document in the final handoff:
- whether a helper tool was added
- which ranking signals were implemented
- whether snippet selection now uses a budget instead of flat truncation
- benchmark deltas
- any intentionally deferred ideas from the spec

- [ ] **Step 5: Skip commit or commit if VCS exists**

```bash
git  status
```
