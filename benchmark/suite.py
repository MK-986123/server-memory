"""Comprehensive real-world benchmark: retrieval, ranking, compression, and query speed."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

# Add src to path
from server_memory.compression import (
    CompressionLevel,
    _estimate_tokens,
    compress_graph,
)
from server_memory.db import Database
from server_memory.graph import KnowledgeGraphManager
from tests.memory_context_benchmarks import run_memory_context_benchmark

# ── Realistic test data ─────────────────────────────────────────────────────

ENTITIES = [
    {
        "name": "AuthService",
        "entityType": "component",
        "observations": [
            "Handles JWT-based authentication with RS256 signing",
            "Supports OAuth2 flows for Google and GitHub providers",
            "Token refresh endpoint is POST /api/auth/refresh",
            "Rate limited to 10 login attempts per minute per IP",
            "Uses bcrypt with cost factor 12 for password hashing",
            "Session tokens expire after 24 hours by default",
        ],
    },
    {
        "name": "PostgresDatabase",
        "entityType": "infrastructure",
        "observations": [
            "Primary database running PostgreSQL 16.2 on port 5432",
            "Connection pool size is 20 with PgBouncer in transaction mode",
            "Read replicas deployed in us-east-1b and us-west-2a",
            "Automated backups run at 03:00 UTC with 30-day retention",
            "Uses pg_partman for time-series data partitioning",
            "Full-text search indexes on users.bio and posts.content columns",
        ],
    },
    {
        "name": "UserPreferences",
        "entityType": "preference",
        "observations": [
            "Prefers dark mode for all IDE and terminal interfaces",
            "Uses 4-space indentation, never tabs, in Python code",
            "Wants type hints on all public function signatures",
            "Prefers pytest over unittest for test frameworks",
            "Always use descriptive variable names, avoid abbreviations",
            "Commits should follow conventional commit format",
        ],
    },
    {
        "name": "PaymentProcessor",
        "entityType": "component",
        "observations": [
            "Integrates with Stripe API v2024-01-18 for payment processing",
            "Webhook endpoint at POST /api/webhooks/stripe handles 12 event types",
            "Implements idempotency keys using UUID v4 stored in Redis",
            "Supports USD, EUR, GBP, and JPY currencies",
            "PCI DSS Level 1 compliant — never stores raw card numbers",
            "Retry logic: exponential backoff with 3 attempts, max 30s delay",
        ],
    },
    {
        "name": "ReactFrontend",
        "entityType": "project",
        "observations": [
            "Built with React 18.3 and TypeScript 5.4 using Vite bundler",
            "State management uses Zustand with persist middleware",
            "UI component library is shadcn/ui built on Radix primitives",
            "E2E tests run with Playwright targeting Chrome and Firefox",
            "Bundle size budget is 250KB gzipped for initial load",
            "Deployed to Cloudflare Pages with automatic preview deployments",
        ],
    },
    {
        "name": "CacheLayer",
        "entityType": "infrastructure",
        "observations": [
            "Redis 7.2 cluster with 3 primary and 3 replica nodes",
            "Cache invalidation uses pub/sub pattern with channel per entity type",
            "TTL strategy: user sessions 24h, API responses 5min, feature flags 1h",
            "Memory limit set to 4GB with allkeys-lru eviction policy",
            "Lua scripts handle atomic read-modify-write operations",
            "Sentinel configuration for automatic failover with 5s timeout",
        ],
    },
    {
        "name": "DeploymentPipeline",
        "entityType": "devops",
        "observations": [
            "CI/CD runs on GitHub Actions with self-hosted ARM64 runners",
            "Docker images built with multi-stage Dockerfiles, final image ~85MB",
            "Kubernetes deployments use rolling update strategy with maxSurge=1",
            "Helm charts versioned independently from application code",
            "Canary deployments route 5% traffic before full rollout",
            "Infrastructure managed with Terraform 1.7 and state in S3",
        ],
    },
    {
        "name": "SearchService",
        "entityType": "component",
        "observations": [
            "Elasticsearch 8.12 cluster with 3 data nodes and 2 coordinating",
            "Custom analyzer with ICU tokenizer for multilingual support",
            "Reindexing pipeline processes 50,000 documents per minute",
            "Query DSL supports fuzzy matching, boosted fields, and geo filters",
            "Index aliases enable zero-downtime reindexing",
            "Search latency p99 target is under 200ms for all queries",
        ],
    },
    {
        "name": "MonitoringStack",
        "entityType": "infrastructure",
        "observations": [
            "Prometheus scrapes metrics every 15s from all service endpoints",
            "Grafana dashboards organized by team: platform, backend, frontend",
            "AlertManager routes critical alerts to PagerDuty, warnings to Slack",
            "Loki aggregates logs with 14-day retention and structured metadata",
            "Jaeger distributed tracing with 1% sampling rate in production",
            "SLO dashboard tracks 99.9% availability and p95 latency targets",
        ],
    },
    {
        "name": "APIGateway",
        "entityType": "component",
        "observations": [
            "Kong API Gateway running version 3.6 in DB-less declarative mode",
            "Rate limiting plugin configured at 1000 requests/min per API key",
            "Request/response transformation plugins handle versioned API contracts",
            "mTLS required for service-to-service communication on internal mesh",
            "OpenAPI spec auto-generated from route decorators, served at /docs",
            "Circuit breaker trips after 5 consecutive 5xx responses, 30s recovery",
        ],
    },
]

RELATIONS = [
    {"from": "ReactFrontend", "to": "APIGateway", "relationType": "calls"},
    {"from": "APIGateway", "to": "AuthService", "relationType": "authenticates_via"},
    {"from": "APIGateway", "to": "PaymentProcessor", "relationType": "routes_to"},
    {"from": "APIGateway", "to": "SearchService", "relationType": "routes_to"},
    {"from": "AuthService", "to": "PostgresDatabase", "relationType": "reads_from"},
    {"from": "AuthService", "to": "CacheLayer", "relationType": "caches_in"},
    {"from": "PaymentProcessor", "to": "PostgresDatabase", "relationType": "writes_to"},
    {"from": "PaymentProcessor", "to": "CacheLayer", "relationType": "uses"},
    {"from": "SearchService", "to": "PostgresDatabase", "relationType": "indexes_from"},
    {"from": "MonitoringStack", "to": "APIGateway", "relationType": "monitors"},
    {"from": "MonitoringStack", "to": "CacheLayer", "relationType": "monitors"},
    {"from": "MonitoringStack", "to": "PostgresDatabase", "relationType": "monitors"},
    {"from": "DeploymentPipeline", "to": "ReactFrontend", "relationType": "deploys"},
    {"from": "DeploymentPipeline", "to": "APIGateway", "relationType": "deploys"},
]


# ── Recall test: specific facts we expect to find ───────────────────────────

RECALL_QUERIES = [
    {
        "query": "JWT RS256",
        "expected_entity": "AuthService",
        "expected_fact": "RS256 signing",
        "category": "exact technical term",
    },
    {
        "query": "password hashing",
        "expected_entity": "AuthService",
        "expected_fact": "bcrypt",
        "category": "security detail",
    },
    {
        "query": "Stripe webhook",
        "expected_entity": "PaymentProcessor",
        "expected_fact": "webhook",
        "category": "integration endpoint",
    },
    {
        "query": "idempotency",
        "expected_entity": "PaymentProcessor",
        "expected_fact": "idempotency",
        "category": "design pattern",
    },
    {
        "query": "dark mode",
        "expected_entity": "UserPreferences",
        "expected_fact": "dark mode",
        "category": "user preference",
    },
    {
        "query": "bundle size",
        "expected_entity": "ReactFrontend",
        "expected_fact": "250KB",
        "category": "performance budget",
    },
    {
        "query": "Kubernetes rolling",
        "expected_entity": "DeploymentPipeline",
        "expected_fact": "rolling update",
        "category": "deployment strategy",
    },
    {
        "query": "Elasticsearch fuzzy",
        "expected_entity": "SearchService",
        "expected_fact": "fuzzy matching",
        "category": "search capability",
    },
    {
        "query": "Prometheus",
        "expected_entity": "MonitoringStack",
        "expected_fact": "Prometheus",
        "category": "monitoring tool",
    },
    {
        "query": "circuit breaker",
        "expected_entity": "APIGateway",
        "expected_fact": "circuit breaker",
        "category": "resilience pattern",
    },
    {
        "query": "rate limit login",
        "expected_entity": "AuthService",
        "expected_fact": "10 login attempts",
        "category": "rate limiting detail",
    },
    {
        "query": "Redis eviction",
        "expected_entity": "CacheLayer",
        "expected_fact": "allkeys-lru",
        "category": "cache policy",
    },
    {
        "query": "PCI DSS",
        "expected_entity": "PaymentProcessor",
        "expected_fact": "PCI DSS",
        "category": "compliance",
    },
    {
        "query": "Terraform state",
        "expected_entity": "DeploymentPipeline",
        "expected_fact": "Terraform",
        "category": "IaC tool",
    },
    {
        "query": "pytest unittest",
        "expected_entity": "UserPreferences",
        "expected_fact": "pytest",
        "category": "tooling preference",
    },
]


# ── Benchmark helpers ────────────────────────────────────────────────────────


@dataclass
class TimingResult:
    label: str
    times_ms: list[float]

    @property
    def mean(self) -> float:
        return statistics.mean(self.times_ms)

    @property
    def median(self) -> float:
        return statistics.median(self.times_ms)

    @property
    def p95(self) -> float:
        return sorted(self.times_ms)[int(len(self.times_ms) * 0.95)]

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.times_ms) if len(self.times_ms) > 1 else 0.0


def timeit(func, iterations=50):
    """Run func N times, return list of durations in ms."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return times, result


def setup_db():
    """Create in-memory DB with realistic test data."""
    db = Database(":memory:")
    db.open()
    mgr = KnowledgeGraphManager(db, session_id="bench")
    mgr.create_entities(ENTITIES)
    mgr.create_relations(RELATIONS)
    return db, mgr


def count_info_pieces(text: str, entity_data: list[dict]) -> dict:
    """Check how many specific facts survive compression."""
    # Key facts to look for across all entities
    key_facts = {
        "RS256": "AuthService signing algo",
        "bcrypt": "password hashing",
        "OAuth2": "auth protocol",
        "5432": "postgres port",
        "PgBouncer": "connection pooler",
        "pg_partman": "partitioning",
        "dark mode": "user preference",
        "4-space": "indentation style",
        "pytest": "test framework",
        "Stripe": "payment provider",
        "UUID v4": "idempotency key format",
        "PCI DSS": "compliance standard",
        "React 18": "frontend framework version",
        "Zustand": "state management",
        "shadcn": "UI library",
        "250KB": "bundle budget",
        "Redis 7.2": "cache version",
        "allkeys-lru": "eviction policy",
        "Lua scripts": "atomic operations",
        "GitHub Actions": "CI platform",
        "Helm charts": "k8s packaging",
        "Terraform 1.7": "IaC version",
        "Elasticsearch 8.12": "search engine",
        "ICU tokenizer": "multilingual search",
        "200ms": "latency target",
        "Prometheus": "metrics",
        "Grafana": "dashboards",
        "PagerDuty": "alerting",
        "Jaeger": "tracing",
        "Kong": "API gateway",
        "1000 requests": "rate limit",
        "mTLS": "service mesh security",
        "circuit breaker": "resilience",
    }

    found = {}
    missing = {}
    text_lower = text.lower()
    for fact, description in key_facts.items():
        if fact.lower() in text_lower:
            found[fact] = description
        else:
            missing[fact] = description

    return {"found": found, "missing": missing, "total": len(key_facts)}


def evaluate_search_recall(
    mgr: KnowledgeGraphManager,
    queries: list[dict[str, str]],
    limit: int = 5,
) -> dict[str, object]:
    """Measure raw DB retrieval recall via ``search_fts``.

    This isolates search quality from compression, because ``search_fts`` reads
    from the stored graph rather than any compressed output.
    """
    correct = 0
    misses: list[str] = []
    total = len(queries)

    for query in queries:
        results = mgr.search_fts(query["query"], limit=limit)
        entity_names = [entity.name for entity in results.entities]
        if query["expected_entity"] in entity_names:
            correct += 1
        else:
            misses.append(
                f"{query['query']} → expected {query['expected_entity']}, got {entity_names[:3]}"
            )

    return {
        "total": total,
        "correct": correct,
        "pct": (correct / total * 100) if total else 0.0,
        "misses": misses,
    }


def evaluate_memory_context_recall(
    mgr: KnowledgeGraphManager,
    queries: list[dict[str, str]],
    limit: int = 3,
) -> dict[str, object]:
    """Measure ``memory_context`` ranking quality on the realistic benchmark corpus."""
    hit_at_1 = 0
    hit_at_3 = 0
    misses: list[str] = []
    total = len(queries)

    for query in queries:
        ctx = mgr.memory_context(hint=query["query"], limit=limit)
        entity_names = [match["name"] for match in ctx["hint_matches"][:limit]]
        if entity_names[:1] == [query["expected_entity"]]:
            hit_at_1 += 1
        if query["expected_entity"] in entity_names[:limit]:
            hit_at_3 += 1
        else:
            misses.append(
                f"{query['query']} → expected {query['expected_entity']}, "
                f"got {entity_names[:limit]}"
            )

    return {
        "total": total,
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "hit_at_1_pct": (hit_at_1 / total * 100) if total else 0.0,
        "hit_at_3_pct": (hit_at_3 / total * 100) if total else 0.0,
        "misses": misses,
    }


def evaluate_compressed_fact_presence(
    compressed_outputs: dict[CompressionLevel, str],
    queries: list[dict[str, str]],
) -> dict[CompressionLevel, dict[str, object]]:
    """Measure fact readability in compressed text independently from retrieval."""
    results: dict[CompressionLevel, dict[str, object]] = {}
    total = len(queries)

    for level, output in compressed_outputs.items():
        correct = 0
        misses: list[str] = []
        text_lower = output.lower()
        for query in queries:
            if query["expected_fact"].lower() in text_lower:
                correct += 1
            else:
                misses.append(f"{query['query']} → expected readable fact {query['expected_fact']}")
        results[level] = {
            "total": total,
            "correct": correct,
            "pct": (correct / total * 100) if total else 0.0,
            "misses": misses,
        }

    return results


# ── Main benchmark ───────────────────────────────────────────────────────────


def main():
    print("=" * 78)
    print("SERVER-MEMORY COMPRESSION BENCHMARK")
    print("=" * 78)
    print()

    db, mgr = setup_db()

    # ── 1. Read graph raw (for baseline) ─────────────────────────────────
    kg = mgr.read_graph()
    total_entities = len(kg.entities)
    total_relations = len(kg.relations)
    total_observations = sum(len(e.observations) for e in kg.entities)
    print(
        f"Dataset: {total_entities} entities, {total_observations} observations, "
        f"{total_relations} relations"
    )
    print()

    # Get pinned IDs (none in this dataset, but exercise the code path)
    pinned_ids: set[int] = set()

    # ── 2. Compression output & info retention ───────────────────────────
    levels = [
        (CompressionLevel.NONE, "NONE (0) - Full JSON"),
        (CompressionLevel.LIGHT, "LIGHT (1) - Markdown"),
        (CompressionLevel.MEDIUM, "MEDIUM (2) - Pipe-delimited"),
        (CompressionLevel.HEAVY, "HEAVY (3) - Truncated"),
    ]

    print("-" * 78)
    print("SECTION 1: COMPRESSION RATIO & INFORMATION RETENTION")
    print("-" * 78)
    print()

    compressed_outputs = {}
    for level, label in levels:
        # Use large budget to avoid truncation (fair comparison)
        output = compress_graph(kg, level=level, token_budget=0, pinned_entity_ids=pinned_ids)
        compressed_outputs[level] = output

        raw_bytes = len(output.encode("utf-8"))
        est_tokens = _estimate_tokens(output)
        info = count_info_pieces(output, ENTITIES)
        pct_retained = len(info["found"]) / info["total"] * 100

        print(f"  {label}")
        print(f"    Size:      {raw_bytes:>6,} bytes  |  ~{est_tokens:>5,} tokens")
        print(
            f"    Facts:     {len(info['found']):>2}/{info['total']} retained ({pct_retained:.0f}%)"
        )
        if info["missing"]:
            missing_names = list(info["missing"].keys())[:5]
            suffix = f" +{len(info['missing']) - 5} more" if len(info["missing"]) > 5 else ""
            print(f"    Lost:      {', '.join(missing_names)}{suffix}")
        print()

    # Show compression ratios relative to NONE
    base_size = len(compressed_outputs[CompressionLevel.NONE].encode("utf-8"))
    print("  Compression ratios (vs NONE):")
    for level, label in levels:
        size = len(compressed_outputs[level].encode("utf-8"))
        ratio = size / base_size * 100
        savings = 100 - ratio
        print(
            f"    {label.split(' - ')[0]:>12}: {ratio:5.1f}% of original  ({savings:5.1f}% saved)"
        )
    print()

    # ── 3. Token budget enforcement ──────────────────────────────────────
    print("-" * 78)
    print("SECTION 2: TOKEN BUDGET ENFORCEMENT")
    print("-" * 78)
    print()

    budgets = [500, 1000, 2000, 5000]
    for budget in budgets:
        output = compress_graph(kg, level=CompressionLevel.MEDIUM, token_budget=budget)
        est_tokens = _estimate_tokens(output)
        info = count_info_pieces(output, ENTITIES)
        omitted = "...+" in output
        pct = len(info["found"]) / info["total"] * 100
        print(
            f"  Budget {budget:>5} tokens → {est_tokens:>5} tokens used | "
            f"Facts: {len(info['found']):>2}/{info['total']} ({pct:.0f}%) | "
            f"Truncated: {'YES' if omitted else 'no'}"
        )
    print()

    search_recall = evaluate_search_recall(mgr, RECALL_QUERIES, limit=5)
    memory_context_recall = evaluate_memory_context_recall(mgr, RECALL_QUERIES, limit=3)
    fixed_memory_context = run_memory_context_benchmark()

    # ── 4. Raw DB retrieval recall ───────────────────────────────────────
    print("-" * 78)
    print("SECTION 3: RAW DB RETRIEVAL RECALL")
    print("-" * 78)
    print()
    print("  Testing search_fts against the stored graph only.")
    print("  Compression does not participate in this section.")
    print()
    print(
        f"  search_fts(): {search_recall['correct']}/{search_recall['total']} queries found the "
        f"expected entity ({search_recall['pct']:.0f}%)"
    )
    if search_recall["misses"]:
        for failure in search_recall["misses"][:3]:
            print(f"               MISS: {failure}")
    print()

    # ── 5. memory_context ranking recall ────────────────────────────────
    print("-" * 78)
    print("SECTION 4: MEMORY_CONTEXT RANKING RECALL")
    print("-" * 78)
    print()
    print("  4A. Realistic-corpus ranking recall using RECALL_QUERIES")
    print(
        f"        hit@1: {memory_context_recall['hit_at_1']}/{memory_context_recall['total']} "
        f"({memory_context_recall['hit_at_1_pct']:.0f}%) | "
        f"hit@3: {memory_context_recall['hit_at_3']}/{memory_context_recall['total']} "
        f"({memory_context_recall['hit_at_3_pct']:.0f}%)"
    )
    if memory_context_recall["misses"]:
        for failure in memory_context_recall["misses"][:3]:
            print(f"               MISS: {failure}")
    print()
    print("  4B. Fixed scenario-table ranking benchmark")
    print(
        f"        hit@1: {fixed_memory_context['hit_at_1']}/{fixed_memory_context['total']} "
        f"({fixed_memory_context['hit_at_1_pct'] * 100:.0f}%) | "
        f"hit@3: {fixed_memory_context['hit_at_3']}/{fixed_memory_context['total']} "
        f"({fixed_memory_context['hit_at_3_pct'] * 100:.0f}%)"
    )
    if fixed_memory_context["misses"]:
        for miss in fixed_memory_context["misses"][:3]:
            print(
                "               MISS: "
                f"{miss['scenario']} → expected {miss['expected_top3']}, got {miss['actual']}"
            )
    print()

    compressed_fact_presence = evaluate_compressed_fact_presence(
        compressed_outputs,
        RECALL_QUERIES,
    )

    # ── 6. Compressed output fact verification ───────────────────────────
    print("-" * 78)
    print("SECTION 5: COMPRESSED FACT RETENTION")
    print("-" * 78)
    print()
    print("  Can key facts still be read from the compressed text?")
    print("  This measures lossy output readability, not retrieval quality.")
    print()

    for level, label in levels:
        fact_result = compressed_fact_presence[level]
        print(
            f"  {label.split(' - ')[0]:>12}: {fact_result['correct']}/{fact_result['total']} "
            f"query-targeted facts readable ({fact_result['pct']:.0f}%)"
        )
    print()

    # ── 7. Query performance benchmarks ──────────────────────────────────
    print("-" * 78)
    print("SECTION 6: QUERY PERFORMANCE (50 iterations each)")
    print("-" * 78)
    print()

    results_table: list[TimingResult] = []

    # read_graph
    times, _ = timeit(lambda: mgr.read_graph(), 50)
    results_table.append(TimingResult("read_graph()", times))

    # read_graph + compress at each level
    for level, label in levels:
        short = label.split(" - ")[0]
        times, _ = timeit(
            lambda lv=level: compress_graph(mgr.read_graph(), level=lv, token_budget=2000),
            50,
        )
        results_table.append(TimingResult(f"read+compress {short}", times))

    # search_fts (various queries)
    search_queries = [
        "JWT authentication",
        "Redis cache",
        "Kubernetes deploy",
        "rate limit",
        "database",
    ]
    for sq in search_queries:
        times, _ = timeit(lambda q=sq: mgr.search_fts(q, limit=10), 50)
        results_table.append(TimingResult(f'search_fts("{sq}")', times))

    # open_nodes at different depths
    for depth in [0, 1, 2]:
        times, _ = timeit(lambda d=depth: mgr.open_nodes(["APIGateway"], depth=d), 50)
        results_table.append(TimingResult(f"open_nodes(depth={depth})", times))

    # memory_context
    times, _ = timeit(lambda: mgr.memory_context(hint="payment"), 50)
    results_table.append(TimingResult('memory_context("payment")', times))

    # Print table
    print(f"  {'Operation':<35} {'Mean':>8} {'Median':>8} {'P95':>8} {'StDev':>8}")
    print(f"  {'-' * 35} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")
    for r in results_table:
        print(
            f"  {r.label:<35} {r.mean:>7.2f}ms {r.median:>7.2f}ms {r.p95:>7.2f}ms {r.stdev:>7.2f}ms"
        )
    print()

    # ── 8. Compression overhead isolation ────────────────────────────────
    print("-" * 78)
    print("SECTION 7: COMPRESSION OVERHEAD (compress_graph only, 200 iterations)")
    print("-" * 78)
    print()

    kg_snapshot = mgr.read_graph()  # read once, compress many times
    for level, label in levels:
        short = label.split(" - ")[0]
        times, _ = timeit(
            lambda lv=level: compress_graph(kg_snapshot, level=lv, token_budget=2000),
            200,
        )
        r = TimingResult(f"compress {short}", times)
        print(f"  {r.label:<25} mean={r.mean:.3f}ms  median={r.median:.3f}ms  p95={r.p95:.3f}ms")
    print()

    # ── 9. End-to-end: write → read → search cycle ──────────────────────
    print("-" * 78)
    print("SECTION 8: END-TO-END WRITE → READ → SEARCH CYCLE")
    print("-" * 78)
    print()

    def full_cycle():
        """Simulate a realistic usage cycle."""
        # Add an observation
        mgr.add_observations(
            [
                {
                    "entityName": "AuthService",
                    "contents": [f"Cycle test observation {time.monotonic_ns()}"],
                }
            ]
        )
        # Read full graph compressed
        kg = mgr.read_graph()
        output = compress_graph(kg, level=CompressionLevel.MEDIUM, token_budget=2000)
        # Search for something
        results = mgr.search_fts("authentication JWT", limit=5)
        return output, results

    times, _ = timeit(full_cycle, 30)
    r = TimingResult("Full write→read→search cycle", times)
    print(f"  {r.label}")
    print(
        f"    Mean: {r.mean:.2f}ms  Median: {r.median:.2f}ms  "
        f"P95: {r.p95:.2f}ms  StDev: {r.stdev:.2f}ms"
    )
    print()

    # ── 10. Sample output comparison ──────────────────────────────────────
    print("-" * 78)
    print("SECTION 9: SAMPLE OUTPUT AT EACH COMPRESSION LEVEL")
    print("-" * 78)
    print()
    print("  Showing AuthService entity at each level:")
    print()

    # Get just AuthService
    auth_kg = mgr.open_nodes(["AuthService"], depth=0)
    for level, label in levels:
        output = compress_graph(auth_kg, level=level, token_budget=0)
        print(f"  ── {label} ──")
        for line in output.split("\n"):
            print(f"    {line}")
        print()

    # ── 11. Activity logging benchmark ───────────────────────────────────
    print("-" * 78)
    print("SECTION 10: ACTIVITY LOGGING & TIMELINE QUERY")
    print("-" * 78)
    print()

    # Log some activities
    for i in range(20):
        mgr.log_activity(
            action="file_changed",
            summary=f"Updated module_{i}.py with new endpoint handler",
            entity_names=["AuthService"] if i % 3 == 0 else ["ReactFrontend"],
        )

    times, _ = timeit(
        lambda: mgr.log_activity(action="decision_made", summary="Switched to async handlers"),
        50,
    )
    r = TimingResult("log_activity()", times)
    print(f"  {r.label}: mean={r.mean:.2f}ms  median={r.median:.2f}ms  p95={r.p95:.2f}ms")

    times, _ = timeit(lambda: mgr.query_timeline(time_range="1h", limit=50), 50)
    r = TimingResult("query_timeline(1h)", times)
    print(f"  {r.label}: mean={r.mean:.2f}ms  median={r.median:.2f}ms  p95={r.p95:.2f}ms")

    times, _ = timeit(lambda: mgr.memory_context(hint="auth"), 50)
    r = TimingResult("memory_context(auth)", times)
    print(
        f"  {r.label}: mean={r.mean:.2f}ms  median={r.median:.2f}ms  "
        f"p95={r.p95:.2f}ms"
    )
    print()

    # ── Summary ──────────────────────────────────────────────────────────
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print()

    med_info = count_info_pieces(compressed_outputs[CompressionLevel.MEDIUM], ENTITIES)

    none_size = len(compressed_outputs[CompressionLevel.NONE].encode())
    med_size = len(compressed_outputs[CompressionLevel.MEDIUM].encode())

    print("  Level     │ Size (bytes) │ Tokens │ Facts Retained │ Compression")
    print("  ──────────┼──────────────┼────────┼────────────────┼────────────")
    for level, label in levels:
        out = compressed_outputs[level]
        sz = len(out.encode())
        tok = _estimate_tokens(out)
        info = count_info_pieces(out, ENTITIES)
        pct_facts = len(info["found"]) / info["total"] * 100
        pct_size = sz / none_size * 100
        short = label.split(" (")[0]
        print(
            f"  {short:<9} │ {sz:>12,} │ {tok:>6,} │ "
            f"{len(info['found']):>2}/{info['total']} ({pct_facts:>3.0f}%)   │ {pct_size:>5.1f}%"
        )

    print()
    print(
        "  Retrieval / ranking split: "
        f"search_fts={search_recall['correct']}/{search_recall['total']} | "
        f"memory_context realistic hit@1={memory_context_recall['hit_at_1']}/"
        f"{memory_context_recall['total']} "
        f"hit@3={memory_context_recall['hit_at_3']}/{memory_context_recall['total']} | "
        f"memory_context fixed hit@1={fixed_memory_context['hit_at_1']}/"
        f"{fixed_memory_context['total']} "
        f"hit@3={fixed_memory_context['hit_at_3']}/{fixed_memory_context['total']}"
    )
    print()
    print(
        "  Key takeaway: MEDIUM compression retains "
        f"{len(med_info['found'])}/{med_info['total']} facts "
        f"at {med_size / none_size * 100:.0f}% of original size."
    )
    if med_info["missing"]:
        print(f"  Lost at MEDIUM: {', '.join(list(med_info['missing'].keys()))}")
    print()

    db.close()


if __name__ == "__main__":
    main()
