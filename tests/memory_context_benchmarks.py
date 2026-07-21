"""Fixed memory_context benchmark scenarios for hit@1 and hit@3 reporting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from server_memory.db import Database
from server_memory.graph import KnowledgeGraphManager

SetupFn = Callable[[KnowledgeGraphManager], None]


@dataclass(frozen=True)
class MemoryContextScenario:
    name: str
    hint: str
    setup: SetupFn
    expected_top: tuple[str, ...]
    expected_top3: tuple[str, ...]


def _setup_exact_name(graph: KnowledgeGraphManager) -> None:
    graph.create_entities(
        [
            {
                "name": "JWT Config",
                "entityType": "config",
                "observations": ["JWT issuer and audience settings"],
            },
            {
                "name": "Auth Notes",
                "entityType": "note",
                "observations": ["JWT is used in several places"],
            },
        ]
    )


def _setup_high_importance(graph: KnowledgeGraphManager) -> None:
    graph.create_entities(
        [
            {"name": "Rotation Policy", "entityType": "config"},
            {"name": "Noisy Notes", "entityType": "note"},
        ]
    )
    graph.add_observations(
        [
            {
                "entityName": "Rotation Policy",
                "contents": ["rotation window is 15 minutes for key rollover"],
                "importance": 0.9,
                "confidence": 0.95,
                "obs_type": "config",
            },
            {
                "entityName": "Noisy Notes",
                "contents": ["rotation window might be 30 minutes"],
                "importance": 0.1,
                "confidence": 0.2,
            },
            {
                "entityName": "Noisy Notes",
                "contents": ["critical launch checklist lives elsewhere"],
                "importance": 1.0,
                "confidence": 0.95,
            },
        ]
    )


def _setup_pinned_vs_noise(graph: KnowledgeGraphManager) -> None:
    graph.create_entities(
        [
            {
                "name": "Critical Config",
                "entityType": "config",
                "tags": ["pinned"],
                "observations": ["provider x requires strict validation"],
            },
            {
                "name": "Recent Chatter",
                "entityType": "note",
                "observations": ["provider x came up in passing"],
            },
        ]
    )
    graph.log_activity(action="discussed", summary="provider x came up again")


def _setup_recent_access_tiebreak(graph: KnowledgeGraphManager) -> None:
    graph.create_entities(
        [
            {
                "name": "Active Runbook",
                "entityType": "doc",
                "observations": ["provider x auth flow details"],
            },
            {
                "name": "Stale Runbook",
                "entityType": "doc",
                "observations": ["provider x auth flow details"],
            },
        ]
    )
    graph.db.cx.execute(
        "UPDATE entities SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-2 minutes'), "
        "last_accessed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE name = ?",
        ("Active Runbook",),
    )
    graph.db.cx.execute(
        "UPDATE entities SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), "
        "last_accessed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-30 days') WHERE name = ?",
        ("Stale Runbook",),
    )
    graph.db.cx.commit()


def _setup_activity_boost(graph: KnowledgeGraphManager) -> None:
    graph.create_entities(
        [
            {
                "name": "Incident Runbook",
                "entityType": "doc",
                "observations": ["provider x rollback procedure"],
            },
            {
                "name": "Generic Notes",
                "entityType": "note",
                "observations": ["provider x rollback procedure"],
            },
        ]
    )
    graph.log_activity(
        action="decision_made",
        summary="provider x rollback was reviewed in the incident runbook",
        entity_names=["Incident Runbook"],
    )
    graph.db.cx.execute(
        "UPDATE entities SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '+1 minute') WHERE name = ?",
        ("Generic Notes",),
    )
    graph.db.cx.commit()


def _setup_file_path_hint(graph: KnowledgeGraphManager) -> None:
    graph.create_entities(
        [
            {"name": "Canonical Path", "entityType": "file"},
            {"name": "Discussion Notes", "entityType": "note"},
        ]
    )
    graph.add_observations(
        [
            {
                "entityName": "Canonical Path",
                "contents": ["/srv/app/config/auth.yaml"],
                "importance": 0.5,
                "confidence": 0.95,
                "obs_type": "file_path",
            },
            {
                "entityName": "Discussion Notes",
                "contents": ["/srv/app/config/auth.yaml was discussed during triage"],
                "importance": 0.9,
                "confidence": 0.95,
            },
        ]
    )


def _setup_exact_beats_recent_noise(graph: KnowledgeGraphManager) -> None:
    graph.create_entities(
        [
            {
                "name": "JWT Config",
                "entityType": "config",
                "observations": ["JWT issuer is auth-prod"],
            },
            {
                "name": "Recent JWT Discussion",
                "entityType": "note",
                "observations": ["JWT config was mentioned in triage"],
            },
        ]
    )
    graph.db.cx.execute(
        "UPDATE entities SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '+2 minutes') WHERE name = ?",
        ("Recent JWT Discussion",),
    )
    graph.db.cx.commit()


def _setup_lexical_only(graph: KnowledgeGraphManager) -> None:
    graph.create_entities(
        [
            {
                "name": "Plain Lexical Match",
                "entityType": "note",
                "observations": ["JWT config lives here"],
            },
        ]
    )


def _setup_stale_demoted(graph: KnowledgeGraphManager) -> None:
    graph.create_entities(
        [
            {"name": "Fresh Guide", "entityType": "doc"},
            {"name": "Legacy Guide", "entityType": "doc"},
        ]
    )
    graph.add_observations(
        [
            {
                "entityName": "Fresh Guide",
                "contents": ["auth endpoint moved to /v2/login"],
                "confidence": 0.95,
                "importance": 0.8,
            },
            {
                "entityName": "Legacy Guide",
                "contents": ["auth endpoint moved to /v1/login"],
                "confidence": 0.2,
                "importance": 0.4,
            },
        ]
    )
    graph.db.cx.execute(
        "UPDATE entities "
        "SET last_accessed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-120 days'), "
        "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-120 days') "
        "WHERE name = ?",
        ("Legacy Guide",),
    )
    graph.db.cx.execute(
        "UPDATE observations "
        "SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-120 days') "
        "WHERE entity_id = (SELECT id FROM entities WHERE name = ?)",
        ("Legacy Guide",),
    )
    graph.db.cx.commit()


def _setup_duplicate_suppression(graph: KnowledgeGraphManager) -> None:
    graph.create_entities(
        [
            {"name": "Auth Canonical", "entityType": "config", "observations": ["refresh token"]},
            {"name": "Auth Duplicate", "entityType": "config", "observations": ["refresh token"]},
        ]
    )


def _setup_contradiction_resolution(graph: KnowledgeGraphManager) -> None:
    graph.create_entities(
        [
            {"name": "Current Deployment Policy", "entityType": "config"},
            {"name": "Superseded Deployment Note", "entityType": "note"},
        ]
    )
    graph.add_observations(
        [
            {
                "entityName": "Current Deployment Policy",
                "contents": ["deployment approval requires two reviewers"],
                "confidence": 1.0,
                "importance": 1.0,
                "obs_type": "config",
            },
            {
                "entityName": "Superseded Deployment Note",
                "contents": ["deployment approval requires one reviewer"],
                "confidence": 0.1,
                "importance": 0.1,
            },
        ]
    )


def _setup_paraphrase(graph: KnowledgeGraphManager) -> None:
    graph.create_entities(
        [
            {
                "name": "Credential Rotation Runbook",
                "entityType": "doc",
                "observations": ["procedure for rotating service credentials safely"],
            },
            {
                "name": "Unrelated Credentials Note",
                "entityType": "note",
                "observations": ["credentials were mentioned during planning"],
            },
        ]
    )


def _setup_multilingual(graph: KnowledgeGraphManager) -> None:
    graph.create_entities(
        [
            {
                "name": "Guía de Rotación",
                "entityType": "doc",
                "observations": ["rotación de claves de producción cada noventa días"],
            },
            {
                "name": "Notas Generales",
                "entityType": "note",
                "observations": ["documentación general del equipo"],
            },
        ]
    )


def _setup_noisy_scale(graph: KnowledgeGraphManager) -> None:
    graph.create_entities(
        [
            {
                "name": f"Noise {index:04d}",
                "entityType": "note",
                "observations": [f"routine background record {index:04d}"],
            }
            for index in range(500)
        ]
        + [
            {
                "name": "Canonical Quasar Runbook",
                "entityType": "doc",
                "observations": ["quasar-needle recovery procedure"],
            }
        ]
    )


BENCHMARK_SCENARIOS: tuple[MemoryContextScenario, ...] = (
    MemoryContextScenario(
        name="exact_entity_name_lookup",
        hint="JWT Config",
        setup=_setup_exact_name,
        expected_top=("JWT Config",),
        expected_top3=("JWT Config",),
    ),
    MemoryContextScenario(
        name="high_importance_beats_noisy_mentions",
        hint="rotation window",
        setup=_setup_high_importance,
        expected_top=("Rotation Policy",),
        expected_top3=("Rotation Policy",),
    ),
    MemoryContextScenario(
        name="pinned_fact_beats_recent_chatter",
        hint="provider x",
        setup=_setup_pinned_vs_noise,
        expected_top=("Critical Config",),
        expected_top3=("Critical Config",),
    ),
    MemoryContextScenario(
        name="recent_access_is_tiebreaker",
        hint="provider x auth flow",
        setup=_setup_recent_access_tiebreak,
        expected_top=("Active Runbook",),
        expected_top3=("Active Runbook", "Stale Runbook"),
    ),
    MemoryContextScenario(
        name="activity_link_reorders_candidates",
        hint="provider x rollback",
        setup=_setup_activity_boost,
        expected_top=("Incident Runbook",),
        expected_top3=("Incident Runbook",),
    ),
    MemoryContextScenario(
        name="technical_hint_prefers_file_path",
        hint="/srv/app/config/auth.yaml",
        setup=_setup_file_path_hint,
        expected_top=("Canonical Path",),
        expected_top3=("Canonical Path",),
    ),
    MemoryContextScenario(
        name="exact_clue_beats_recent_noise",
        hint="JWT Config",
        setup=_setup_exact_beats_recent_noise,
        expected_top=("JWT Config",),
        expected_top3=("JWT Config",),
    ),
    MemoryContextScenario(
        name="lexical_only_fallback_still_hits",
        hint="JWT config",
        setup=_setup_lexical_only,
        expected_top=("Plain Lexical Match",),
        expected_top3=("Plain Lexical Match",),
    ),
    MemoryContextScenario(
        name="stale_candidate_is_demoted",
        hint="auth endpoint moved",
        setup=_setup_stale_demoted,
        expected_top=("Fresh Guide",),
        expected_top3=("Fresh Guide",),
    ),
    MemoryContextScenario(
        name="duplicate_matches_do_not_crowd_top3",
        hint="refresh token",
        setup=_setup_duplicate_suppression,
        expected_top=("Auth Canonical", "Auth Duplicate"),
        expected_top3=("Auth Canonical", "Auth Duplicate"),
    ),
    MemoryContextScenario(
        name="current_fact_beats_contradiction",
        hint="deployment approval reviewers",
        setup=_setup_contradiction_resolution,
        expected_top=("Current Deployment Policy",),
        expected_top3=("Current Deployment Policy",),
    ),
    MemoryContextScenario(
        name="paraphrased_procedure_is_retrieved",
        hint="how to rotate service credentials",
        setup=_setup_paraphrase,
        expected_top=("Credential Rotation Runbook",),
        expected_top3=("Credential Rotation Runbook",),
    ),
    MemoryContextScenario(
        name="multilingual_lexical_retrieval",
        hint="rotación claves producción",
        setup=_setup_multilingual,
        expected_top=("Guía de Rotación",),
        expected_top3=("Guía de Rotación",),
    ),
    MemoryContextScenario(
        name="target_survives_noisy_scale",
        hint="quasar-needle recovery",
        setup=_setup_noisy_scale,
        expected_top=("Canonical Quasar Runbook",),
        expected_top3=("Canonical Quasar Runbook",),
    ),
)


def run_memory_context_benchmark() -> dict[str, object]:
    """Execute the fixed scenario table and report hit@1/hit@3 counts."""
    hit_at_1 = 0
    hit_at_3 = 0
    misses: list[dict[str, object]] = []

    for scenario in BENCHMARK_SCENARIOS:
        db = Database(":memory:")
        db.open()
        try:
            graph = KnowledgeGraphManager(db, session_id=f"benchmark:{scenario.name}")
            scenario.setup(graph)
            ctx = graph.memory_context(hint=scenario.hint, limit=3)
            names = [match["name"] for match in ctx["hint_matches"][:3]]
            if names[:1] and names[0] in scenario.expected_top:
                hit_at_1 += 1
            if any(name in scenario.expected_top3 for name in names):
                hit_at_3 += 1
            else:
                misses.append(
                    {
                        "scenario": scenario.name,
                        "hint": scenario.hint,
                        "expected_top": scenario.expected_top,
                        "expected_top3": scenario.expected_top3,
                        "actual": names,
                    }
                )
        finally:
            db.close()

    total = len(BENCHMARK_SCENARIOS)
    return {
        "total": total,
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "hit_at_1_pct": hit_at_1 / total if total else 0.0,
        "hit_at_3_pct": hit_at_3 / total if total else 0.0,
        "misses": misses,
    }
