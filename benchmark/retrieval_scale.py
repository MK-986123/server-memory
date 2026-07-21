#!/usr/bin/env python3
"""Deterministic production-path retrieval scale benchmark.

Builds one in-memory 10k-entity/50k-observation corpus, then compares the
unbounded historical scoring path with the bounded production configuration.
No external model, network access, or persistent database is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import statistics
import struct
import time
from pathlib import Path

import server_memory.graph as graph_module
from server_memory.db import Database
from server_memory.embeddings import EmbeddingEngine, embedding_buckets
from server_memory.graph import KnowledgeGraphManager


class DeterministicEmbeddingEngine(EmbeddingEngine):
    DIMENSION = 32
    TARGETS = ("alpha", "beta", "gamma")

    def __init__(self) -> None:
        super().__init__("deterministic-sha256-v1")
        self._available = True

    def is_available(self) -> bool:
        return True

    def embed_text(self, text: str) -> bytes:
        lowered = text.lower()
        for index, target in enumerate(self.TARGETS):
            if target in lowered:
                values = [0.0] * self.DIMENSION
                values[index] = 1.0
                return struct.pack(f"{self.DIMENSION}f", *values)
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [((digest[index] / 255.0) * 2.0) - 1.0 for index in range(self.DIMENSION)]
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return struct.pack(f"{self.DIMENSION}f", *(value / norm for value in values))

    def embed_batch(self, texts: list[str]) -> list[bytes]:
        return [self.embed_text(text) for text in texts]


def build_corpus(entity_count: int, observation_count: int):
    db = Database(":memory:")
    db.open()
    engine = DeterministicEmbeddingEngine()
    target_count = len(engine.TARGETS)
    noise_entities = entity_count - target_count
    observations_per_entity, remainder = divmod(observation_count, entity_count)

    with db.transaction() as cx:
        for index in range(entity_count):
            if index >= noise_entities:
                target = engine.TARGETS[index - noise_entities]
                name = f"Canonical {target} runbook"
            else:
                name = f"Noise entity {index:05d}"
            entity_id = cx.execute(
                "INSERT INTO entities(name, entity_type) VALUES (?, 'note')", (name,)
            ).lastrowid
            entity_blob = engine.embed_text(name)
            entity_buckets = embedding_buckets(entity_blob)
            cx.execute(
                "INSERT INTO entity_embeddings(entity_id, embedding, model_name, dimension, "
                "bucket0, bucket1, bucket2, bucket3) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entity_id,
                    entity_blob,
                    engine.model_name,
                    engine.DIMENSION,
                    *entity_buckets,
                ),
            )
            count = observations_per_entity + (1 if index < remainder else 0)
            for obs_index in range(count):
                content = f"observation {index:05d}-{obs_index}"
                if index >= noise_entities:
                    content += f" canonical {engine.TARGETS[index - noise_entities]} procedure"
                observation_id = cx.execute(
                    "INSERT INTO observations(entity_id, content) VALUES (?, ?)",
                    (entity_id, content),
                ).lastrowid
                blob = engine.embed_text(content)
                buckets = embedding_buckets(blob)
                cx.execute(
                    "INSERT INTO observation_embeddings"
                    "(observation_id, embedding, model_name, dimension, "
                    "bucket0, bucket1, bucket2, bucket3) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        observation_id,
                        blob,
                        engine.model_name,
                        engine.DIMENSION,
                        *buckets,
                    ),
                )

    graph = KnowledgeGraphManager(db, embedding_engine=engine)
    graph._embeddings_synced = True
    return db, graph, engine


def measure(graph: KnowledgeGraphManager, engine: DeterministicEmbeddingEngine, *, runs: int):
    latencies: list[float] = []
    hits = 0
    total = 0
    for _ in range(runs):
        for target in engine.TARGETS:
            started = time.perf_counter()
            result = graph.search_fts(f"semantic query {target}", limit=3)
            latencies.append((time.perf_counter() - started) * 1000)
            total += 1
            if any(target in entity.name.lower() for entity in result.entities[:3]):
                hits += 1
    ordered = sorted(latencies)
    p95_index = max(math.ceil(len(ordered) * 0.95) - 1, 0)
    return {
        "queries": total,
        "recall_at_3": hits / total,
        "p50_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(ordered[p95_index], 3),
    }


def cpu_model() -> str:
    """Return a reportable CPU identity without optional dependencies."""
    reported = platform.processor().strip()
    if reported:
        return reported
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=int, default=10_000)
    parser.add_argument("--observations", type=int, default=50_000)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.entities < 3 or args.observations < args.entities:
        parser.error("entities must be >=3 and observations must be >= entities")

    db, graph, engine = build_corpus(args.entities, args.observations)
    try:
        graph_module.USE_EMBEDDING_BUCKET_FILTER = False
        graph_module.MAX_ENTITY_EMBEDDING_CANDIDATES = args.entities
        graph_module.MAX_OBSERVATION_EMBEDDING_CANDIDATES = args.observations
        baseline = measure(graph, engine, runs=args.runs)
        graph_module.USE_EMBEDDING_BUCKET_FILTER = True
        graph_module.MAX_ENTITY_EMBEDDING_CANDIDATES = 1_000
        graph_module.MAX_OBSERVATION_EMBEDDING_CANDIDATES = 4_000
        bounded = measure(graph, engine, runs=args.runs)
        speedup = baseline["p95_ms"] / bounded["p95_ms"] if bounded["p95_ms"] else None
        print(
            json.dumps(
                {
                    "corpus": {"entities": args.entities, "observations": args.observations},
                    "embedding": {"model": engine.model_name, "dimension": engine.DIMENSION},
                    "runtime": {
                        "python": platform.python_version(),
                        "platform": platform.platform(),
                        "cpu": cpu_model(),
                    },
                    "baseline_unbounded": baseline,
                    "production_bounded": bounded,
                    "p95_speedup": round(speedup, 2) if speedup is not None else None,
                    "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                },
                indent=2,
            )
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
