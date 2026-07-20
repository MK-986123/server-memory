# Agent memory benchmark protocol

This protocol measures whether `server-memory` improves agent performance in realistic, multi-session software tasks. It is designed to produce publishable evidence without relying on synthetic retrieval scores alone.

No benchmark result should be published until all runs are complete and the raw outputs are retained.

## Questions answered

1. Does durable memory improve task completion across interrupted or multi-session work?
2. Does it improve recall of prior decisions, constraints, file locations, and unresolved work?
3. Does it reduce repeated discovery, duplicated tool calls, and prompt tokens?
4. Does it introduce measurable latency or incorrect-memory failures?

## Experimental conditions

Run the same task set under three conditions:

| Condition | Description |
| :--- | :--- |
| `no-memory` | Fresh agent session for every phase. No summary from earlier phases is supplied. |
| `manual-summary` | Fresh session receives a human-written handoff summary capped at the same token budget used by `memory_context`. |
| `server-memory` | Fresh session can call `memory_context`, `search_nodes`, and related tools. Durable facts and activity are written during earlier phases. |

The `manual-summary` condition is important. It separates the value of durable retrieval from the simpler benefit of giving the agent any summary at all.

## Task set

Use 12 to 20 tasks from at least four categories. Each task must require facts introduced in an earlier phase.

### Recommended categories

1. **Bug continuation**
   - Phase A: investigate a failure, identify likely root cause, and record relevant files and constraints.
   - Phase B: after a fresh session, implement and validate the fix.

2. **Feature continuation**
   - Phase A: inspect architecture and make a design decision.
   - Phase B: implement the feature while preserving that decision and existing constraints.

3. **Operational incident**
   - Phase A: diagnose an incident and establish a recovery sequence.
   - Phase B: continue from a fresh session and produce the correct next action or runbook update.

4. **Configuration migration**
   - Phase A: discover current configuration, compatibility limits, and protected values.
   - Phase B: perform or describe the migration without reverting prior decisions.

Each task should contain 5 to 12 durable facts, including a mix of:

- exact file paths
- accepted and rejected decisions
- dependency or version constraints
- commands already attempted
- test failures
- user preferences or nonfunctional requirements
- unresolved next steps

At least 20 percent of facts should be distractors or stale alternatives so the benchmark tests ranking and conflict handling rather than exact-name lookup alone.

## Controlling the runs

Keep these variables fixed across conditions:

- model and model revision
- system prompt
- tool set, except for memory tools
- repository revision
- temperature and reasoning settings
- maximum turns
- context-window limit
- task instructions
- evaluator rubric

Randomize condition order per task. Use at least three independent runs per task and condition. Five runs is preferable when model variance is high.

Do not let the evaluator know which condition produced an answer.

## Primary metrics

### 1. Task completion rate

A run passes only when all required acceptance checks pass.

```text
task_completion_rate = passed_runs / total_runs
```

For code tasks, use tests, build checks, linting, or a deterministic validation script. Avoid grading solely by prose quality.

### 2. Durable-fact recall

Score each required fact as:

- `1.0`: recalled and applied correctly
- `0.5`: recalled but applied incompletely
- `0.0`: omitted
- `-1.0`: contradicted or replaced with a stale fact

Report both raw recall and contradiction rate.

```text
fact_recall = earned_fact_points / maximum_fact_points
contradiction_rate = contradicted_facts / required_facts
```

### 3. Context efficiency

Record input tokens consumed before the first correct implementation action and across the full run.

```text
tokens_to_first_correct_action
 total_input_tokens
 total_output_tokens
 total_tokens
```

For the `server-memory` condition, include memory tool outputs in input-token accounting.

### 4. Repeated-work rate

Count actions that redo work completed in an earlier phase:

- reopening files already identified as irrelevant
- rerunning unchanged diagnostics
- rediscovering established paths or configuration
- restating or reversing accepted decisions

```text
repeated_work_rate = repeated_actions / total_tool_actions
```

### 5. Tool-call efficiency

Record:

```text
total_tool_calls
 repository_read_calls
 search_calls
 test_calls
 memory_read_calls
 memory_write_calls
```

Report tool calls per successful task, not just average calls across all runs.

## Secondary metrics

### Retrieval quality

For every memory query, label the minimum set of relevant entities before running the benchmark.

Report:

- hit@1
- hit@3
- mean reciprocal rank
- relevant facts returned per 1,000 context tokens
- stale or conflicting facts returned per query

### Latency

Measure:

- memory query median latency
- memory query p95 latency
- end-to-end task duration
- time to first correct implementation action

Warm and cold embedding runs must be reported separately when embeddings are enabled.

### Memory correctness and safety

Count:

- incorrect durable writes
- duplicate facts
- stale facts selected over current facts
- cross-workspace leakage
- global preference leakage into unrelated tasks
- destructive-operation scope violations

A memory-assisted run that succeeds by using incorrect or leaked context should not be counted as a clean pass.

## Token accounting

Use provider-reported token counts whenever available. If the provider does not expose counts, use one fixed tokenizer for every condition and disclose it.

Record tokens by source:

```json
{
  "system_prompt": 0,
  "task_prompt": 0,
  "manual_summary": 0,
  "memory_context": 0,
  "other_tool_outputs": 0,
  "assistant_output": 0
}
```

This prevents a misleading result where a compact memory response appears efficient only because other retrieved context was not counted.

## Result schema

Save one JSONL record per run:

```json
{
  "task_id": "bug-01",
  "condition": "server-memory",
  "run": 1,
  "model": "provider/model-revision",
  "repository_commit": "commit-sha",
  "passed": true,
  "fact_points": 8.5,
  "fact_points_max": 10,
  "contradicted_facts": 0,
  "input_tokens": 12400,
  "output_tokens": 2100,
  "tokens_to_first_correct_action": 5100,
  "tool_calls": 18,
  "repeated_actions": 2,
  "memory_calls": 3,
  "memory_latency_ms": [18.2, 15.7, 17.1],
  "duration_seconds": 402.3,
  "notes": ""
}
```

Retain raw transcripts, tool traces, memory database snapshots, evaluator outputs, task fixtures, and acceptance-test logs.

## Statistical reporting

For each condition, publish:

- number of tasks and runs
- mean and median
- standard deviation or interquartile range
- 95 percent confidence interval
- paired difference against `no-memory`
- paired difference against `manual-summary`

For pass/fail outcomes, use a paired binary comparison such as McNemar's test when runs are paired. For token counts and latency, use paired bootstrap confidence intervals or a paired nonparametric test when distributions are skewed.

Do not claim improvement from a single run or from retrieval hit rate alone.

## Minimum credible run

A practical minimum is:

```text
12 tasks × 3 conditions × 3 runs = 108 agent runs
```

A stronger release benchmark is:

```text
20 tasks × 3 conditions × 5 runs = 300 agent runs
```

## Publication checklist

Publish results only when all items are available:

- exact model and revision
- exact repository commit
- exact `server-memory` configuration
- task definitions and fixtures
- randomization method
- raw JSONL results
- evaluator rubric
- acceptance-test output
- exclusions and failed runs
- confidence intervals
- known limitations

The README should summarize verified results and link to the complete raw benchmark package. Keep unverified expectations clearly labeled as hypotheses.
