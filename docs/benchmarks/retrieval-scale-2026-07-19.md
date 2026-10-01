# Retrieval scale benchmark — 2026-07-19

Command:

```bash
./.venv/bin/python benchmark/retrieval_scale.py \
  --entities 10000 --observations 50000 --runs 3
```

Production retrieval path: `KnowledgeGraphManager.search_fts`, deterministic
`deterministic-sha256-v1` embeddings, dimension 32, Python 3.14.6, Linux
7.0.0-28-generic x86_64, 12th Gen Intel(R) Core(TM) i5-12600.

| Mode | p50 | p95 | recall@3 |
|---|---:|---:|---:|
| Historical unbounded scoring | 357.683 ms | 368.087 ms | 1.0 |
| Production indexed + bounded scoring | 56.560 ms | 82.487 ms | 1.0 |

- Measured p95 speedup: **4.46x**.
- Measured process peak RSS: **91,952 KiB**.
- Corpus: 10,000 entities and 50,000 observations.
- Candidate generation: four indexed locality-sensitive-hash buckets with
  exact and one-bit-neighbor probes.
- Scoring bounds after indexed generation: 1,000 entity embeddings and 4,000
  observation embeddings.

This is a deterministic local benchmark, not evidence for latency with every
sentence-transformer model or hardware configuration. Its purpose is to gate
the corpus-scaling behavior and recall of the production retrieval code path.
The baseline disables the bucket filter while retaining the same corpus,
queries, model identity, and scoring implementation.
