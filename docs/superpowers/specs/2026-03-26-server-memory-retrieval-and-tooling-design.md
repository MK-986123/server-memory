# Server-Memory Retrieval And Tooling Design

Date: 2026-03-26
Project: `server-memory`
Status: Approved design, updated after review

## Goal

Improve `server-memory` in two areas:

1. Retrieval quality, with priority on `memory_context`
2. MCP ergonomics, while keeping the existing primary workflow intact

The design target is a better top-of-turn memory experience: strong exact clues should surface the right fact first, contextual expansion should still help when hints are ambiguous, and noisy duplicates should stop crowding out distinct information.

This iteration should prefer wiring in metadata the system already stores before introducing new schema or broad API changes. Existing signals such as observation importance, observation confidence, `last_accessed_at`, activity links, protected observation types, and token budgets are part of the retrieval design, not future nice-to-haves.

## Scope

In scope:

- Improve `memory_context` retrieval and ranking
- Preserve `memory_context` as the main entrypoint
- Add at most one small helper or debug-oriented MCP tool if justified
- Add regression coverage for ranking, dedup, and snippet assembly
- Use existing stored metadata as ranking inputs before adding new schema
- Add a small benchmark set with baseline and post-change `hit@1` and `hit@3`
- Define conflict, staleness, sparse-graph, and token-budget behavior precisely

Out of scope:

- Large MCP surface redesign
- Breaking changes to normal `memory_context` usage
- Broad import/export redesign
- General-purpose database or schema work unrelated to retrieval quality
- New schema fields such as alias tables or suppression flags in this phase
- Structured multi-field query inputs in this phase
- Retrieval caching unless profiling later shows it is needed
- Relation-weight-based graph expansion in this phase; if not implemented, it must remain explicitly deferred rather than implied

## Requirements

### Functional

- Exact factual recall should win when the hint contains a strong lexical clue.
- Retrieval should remain balanced: exact matches first, then semantic and contextual expansion.
- Near-duplicate entities or overlapping observations should be clustered or suppressed.
- Pinned and higher-importance memories should beat generic recent noise.
- Conflicting or stale memories should be surfaced instead of silently blended.
- Observation-level `importance` and `confidence` must influence retrieval directly, not only compression.
- `last_accessed_at` must be treated as a distinct signal from `updated_at`.
- Recent activity must influence retrieval, not only be displayed.
- Snippet selection must choose the most relevant observations for the hint instead of the first observations on the entity.
- Snippet truncation must be token-budget-aware rather than fixed at 50 characters.
- Sparse graphs must degrade gracefully instead of pretending the full ranking pipeline is meaningful.

### Ergonomic

- `memory_context` should stay simple to call.
- Output should remain compact and readable.
- If a helper tool is added, it should be narrow and diagnostic rather than a new primary workflow.
- Technical hints should bias toward technical observations when the hint shape warrants it.

### Non-Functional

- Backward compatibility in spirit: same primary tool, better results.
- Deterministic enough behavior for reliable testing.
- No hidden reliance on embeddings; lexical and structural ranking must still work when embeddings are unavailable.
- Improvements must be measurable against a fixed benchmark set, not only described qualitatively.

## Recommended Approach

Use a hybrid of retrieval tuning and a small amount of tooling support:

1. Keep `memory_context` as the primary tool.
2. Improve internal ranking and snippet selection substantially.
3. Add one small helper tool only if needed to expose ranking decisions during development and debugging.

This yields the best near-term improvement with the lowest compatibility risk.

## Alternatives Considered

### 1. Retrieval Tuning Only

Improve ranking inside the existing `memory_context` flow without adding tools.

Pros:

- Lowest user-facing disruption
- Fastest to ship
- Keeps the MCP surface minimal

Cons:

- Harder to debug ranking failures
- Less visibility into why a memory was chosen

### 2. Retrieval Tuning Plus One Helper Tool

Keep `memory_context` as the main API and add one focused helper such as candidate preview or match explanation.

Pros:

- Preserves the main workflow
- Makes tuning and debugging observable
- Lower risk than a full redesign

Cons:

- Slightly larger MCP surface
- Requires clear tool boundaries

### 3. Rich Structured Recall Redesign

Redesign `memory_context` around structured buckets such as exact hits, related entities, recent activity, and conflicts.

Pros:

- Best long-term expressiveness
- Stronger introspection

Cons:

- Highest compatibility risk
- Larger implementation and test surface
- More likely to create client coupling

## Architecture

### Retrieval Model

Treat `memory_context` as a balanced ranker across several evidence channels:

- lexical/entity match
- lexical/observation match
- semantic similarity when embeddings are available
- observation importance and confidence
- update recency, access recency, and activity-derived session relevance
- importance, type, and pinning signals
- structural diversity, dedup, and token-budget controls

The pipeline should be staged:

1. Build a candidate pool from existing search paths.
2. Score each candidate using weighted features.
3. Cluster or suppress near-duplicates.
4. Apply explicit penalties for stale or conflicting evidence.
5. Promote exact and high-confidence candidates.
6. Assemble a compact final context from diverse, high-value items within a token budget.

### Candidate Features

Candidate scoring should consider:

- exact entity-name hit
- exact phrase hit in observations
- normalized lexical score
- semantic similarity score
- top relevant observation score, weighted by observation `importance` and `confidence`
- a small support bonus from the next-best relevant observation, not raw observation count
- pinned/system-tag boost
- observation-type boost when the hint shape suggests a technical artifact
- update-recency boost
- access-recency boost
- recent-activity/session-affinity boost
- project/session affinity
- conflict/staleness penalty
- duplicate-cluster penalty after canonical representative selection

Exact hits should not be unbeatable by default, but they should dominate when the lexical clue is strong and unambiguous.

Observation evidence must be aggregated in a way that rewards one highly relevant, high-importance observation more than many weak mentions. Entity ranking must not simply rise because an entity has many low-value observations.

### Observation-Level Signals

Observation-level scoring should be explicit:

- use the best-matching observation as the primary evidence for an entity
- apply `importance` as a strong durability and salience multiplier
- apply `confidence` as a trust multiplier and mild tie-break signal
- allow a small support contribution from the second-best observation when it adds distinct support
- do not sum across all observations in a way that rewards noisy volume

### Recency And Activity Signals

Treat these as separate features:

- `updated_at`: recently changed knowledge
- `last_accessed_at`: recently useful knowledge
- activity-log linkage: recently acted-on knowledge

`last_accessed_at` should be a mild boost, not a dominant ranking feature. Activity-derived boosts should come from both explicit `entity_ids` in recent log entries and lexical overlap between the hint and recent activity summaries.

### Observation-Type Awareness

When the hint shape suggests a technical lookup, ranking should prefer matching technical observations. Examples:

- path-like hints should boost `file_path`
- code-like hints should boost `code_snippet`
- config-like hints should boost `config`
- endpoint-like hints should boost `api_endpoint`

This should remain a secondary boost, not a replacement for lexical relevance.

### Diversity And Dedup

The final context should avoid crowding by:

- collapsing near-identical observations under the same entity
- preferring a canonical representative for duplicate clusters
- limiting repeated evidence from a single entity unless it materially adds new information
- ensuring that related context does not displace a stronger exact answer

### Snippet Assembly

Useful snippets are defined as the smallest set of observations that best explain why the entity matched the hint. That means:

- rank observations for snippet display by hint relevance first, then importance, confidence, and type priority
- prefer 1 to 2 strong observations over 3 or more aggressively truncated fragments
- use token-budget-aware truncation rather than a flat 50-character cutoff
- preserve enough text for the snippet to carry meaning on its own
- fill the final output greedily from ranked candidates until the display budget is exhausted

Ranking and output budgeting should be integrated: choose candidates, then choose snippets from those candidates to fit the context budget instead of ranking one way and compressing another.

### Conflict Handling

In this design, a conflict is any of the following:

- the same entity has two high-relevance observations for the same hint topic with materially different normalized content
- the same entity shows a recent observation-version change on the hinted topic
- the same entity has two similarly ranked candidate observations that cannot be safely merged into one summary

If a conflict is surfaced:

- prefer the stronger and more recent candidate for primary ranking
- mark the match as conflicted
- include both competing snippets when budget allows
- avoid flattening the competing claims into one statement

Full semantic contradiction detection across unrelated entities is deferred.

### Staleness Handling

Staleness should be defined rather than implied. Initial rules:

- an unpinned entity not accessed for a long interval should incur a mild stale penalty
- an older low-confidence observation should incur a stale penalty
- an observation superseded by a newer observation on the same topic should incur a stale penalty

Staleness should lower ranking, not silently remove a candidate.

### Graph Proximity

Relation weights and graph-proximity ranking are explicitly deferred for this phase. The spec does not assume BFS expansion or relation-weight scoring until that behavior is designed and tested. If added later, it should be a distinct phase with clear semantics for one-hop and weighted boosts.

### Sparse-Graph Behavior

When the graph is empty or very small, the system should degrade gracefully:

- skip heavyweight ranking heuristics when they do not add value
- return broad coverage rather than pretending there is meaningful precision
- surface a small hint that the graph is sparse when that explains limited recall quality

## MCP Tool Shape

### Primary Tool

Keep `memory_context` as the primary retrieval tool.

Expected changes:

- stronger ranking
- better snippet selection
- clearer prioritization of exact matches
- less duplication
- optional surfacing of conflict or staleness signals

### Optional Helper Tool

Add one helper tool only if implementation proves it useful. Candidate shape:

- `preview_memory_candidates`
- or `explain_memory_match`

The helper should:

- expose top candidate scores or reasons
- support tuning and debugging
- not become the normal user-facing entrypoint

If retrieval tuning is sufficiently observable from tests and logs, this tool can be skipped.

Structured hint inputs such as `entity_type`, `since`, or `project` filters beyond the current `project` argument are deferred to a later API pass rather than folded into this iteration.

## Implementation Slices

### Slice 1: Retrieval And Ranking

Target file: [src/server_memory/graph.py](../../../src/server_memory/graph.py)

Work:

- introduce staged candidate collection
- add weighted ranking that uses observation-level importance/confidence
- add duplicate clustering or suppression
- formalize exact-hit boosting
- add distinct update-recency, access-recency, and activity-derived boosts
- add type-aware secondary boosts for technical hints
- define and apply stale penalties
- improve fallback behavior when embeddings are unavailable

### Slice 2: Context Assembly

Target file: [src/server_memory/server.py](../../../src/server_memory/server.py)

Work:

- update `memory_context` to consume the stronger ranked results
- assemble more useful snippets from the most relevant observations
- budget snippet text rather than hard-cutting every observation at a fixed character limit
- keep output concise while preserving enough observation meaning
- surface conflict or staleness hints when they materially affect interpretation

### Slice 3: Verification

Target files:

- [tests/test_tools.py](../../../tests/test_tools.py)
- [tests/test_search.py](../../../tests/test_search.py)
- [tests/test_hybrid_search.py](../../../tests/test_hybrid_search.py)

Work:

- capture a baseline benchmark set before implementation changes
- add exact-hit priority tests
- add duplicate suppression tests
- add pinned, observation-importance, and confidence-weight tests
- add recency-balance tests
- add access-recency and recent-activity tests
- add snippet-selection and token-budget tests
- add lexical-plus-semantic mixed recall tests
- add conflict/staleness behavior tests as needed

### Slice 4: Explicitly Deferred Ideas

Track but do not implement in this phase:

- suppression or negative-feedback flags
- aliases or abbreviation-expansion schema changes
- broader structured query inputs
- relation-weight-based graph expansion
- caching and memoization

## Risks

### Overweighting Recency

Risk:

- recent noise displaces durable high-value facts

Mitigation:

- tie-break in favor of pinned and high-importance facts
- keep recency as a bounded boost

### Overweighting Exact Match

Risk:

- exact string overlap returns the wrong fact in ambiguous cases

Mitigation:

- combine exact boosts with entity type, confidence, and related-signal checks

### Overweighting Access Recency Or Activity

Risk:

- recently viewed or recently discussed noise displaces durable facts

Mitigation:

- keep access and activity boosts bounded and secondary
- let observation quality and exact relevance dominate

### Snippet Budget Waste

Risk:

- ranked entities win, but low-value truncation wastes the output budget

Mitigation:

- integrate snippet selection with ranking
- prefer fewer stronger snippets over more chopped fragments

### Hidden Duplicate Collapse

Risk:

- dedup removes legitimately distinct evidence

Mitigation:

- cluster conservatively
- prefer suppression only for near-identical content
- keep distinct observations when they add new information

### Tool-Surface Drift

Risk:

- a helper tool grows into a parallel retrieval workflow

Mitigation:

- keep helper tooling diagnostic-only
- preserve `memory_context` as the primary path

## Testing Strategy

Add tests that prove:

- exact clues surface the expected fact first
- duplicates no longer crowd out distinct facts
- pinned entities win close ranking ties
- observation importance and confidence affect ranking directly
- ambiguous hints still return contextual expansion
- embeddings-off mode still yields useful recall
- mixed lexical and semantic cases behave predictably
- recent access and recent activity produce only bounded boosts
- technical hint shapes bias toward the right observation types
- snippet selection prefers the most relevant observations
- sparse-graph behavior remains sensible

Before implementation, capture a fixed benchmark set of 10 to 20 scenarios expressed as hint-to-expected-top-result examples. Measure baseline `hit@1` and `hit@3`, then rerun after implementation to confirm improvement.

Prefer small, intention-revealing fixtures over broad integration scenarios.

## Success Criteria

The work is successful when:

- `memory_context` improves on the recorded baseline benchmark for both `hit@1` and `hit@3`
- `memory_context` reliably surfaces the expected fact for strong exact hints
- near-duplicate memories stop dominating the output
- pinned, higher-importance, and higher-confidence memories outrank generic recent noise
- contextual expansion remains helpful but secondary to exact recall
- recent access and activity influence ranking without overwhelming stronger evidence
- snippets explain the match without collapsing into unreadable fragments
- bad ranking behavior is explainable and testable without guesswork

## Next Step

Create an implementation plan for the approved design before changing code.
