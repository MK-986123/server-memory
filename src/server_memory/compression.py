"""Multi-level compression for token-efficient graph output.

Levels:
    NONE (0)   — Full JSON
    LIGHT (1)  — Markdown format, no IDs/timestamps
    MEDIUM (2) — Pipe-delimited, strip filler
    HEAVY (3)  — Single-line, truncated
"""

from __future__ import annotations

import re
from typing import Any

from .config import CompressionLevel
from .models import PROTECTED_OBS_TYPES, Entity, KnowledgeGraph, Observation, Relation

# Filler words to strip at MEDIUM+ compression
_FILLER = re.compile(
    r"\b(the|a|an|is|are|was|were|has|have|had|that|this|it|its|for|of|in|on|to|and|or|but|with|from|by|as|at|be|been|being)\b",
    re.IGNORECASE,
)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def _strip_filler(text: str) -> str:
    """Remove common filler words for compression."""
    result = _FILLER.sub("", text)
    return re.sub(r"\s{2,}", " ", result).strip()


def _sort_observations(observations: list[Observation]) -> list[Observation]:
    """Sort observations: protected types first, then by importance desc."""
    return sorted(
        observations,
        key=lambda o: (
            0 if o.obs_type in PROTECTED_OBS_TYPES else 1,
            -o.importance,
        ),
    )


def format_entity_none(entity: Entity) -> dict[str, Any]:
    """Level 0: Full JSON representation."""
    return entity.to_dict()


def format_entity_light(entity: Entity) -> str:
    """Level 1: Markdown, no IDs/timestamps.
    Example: App [project] #android
    - observation 1
    - observation 2
    """
    tags_str = " ".join(f"#{t}" for t in entity.tags) if entity.tags else ""
    header = f"{entity.name} [{entity.entity_type}]"
    if tags_str:
        header += f" {tags_str}"
    lines = [header]
    for obs in entity.observations:
        lines.append(f"- {obs.content}")
    return "\n".join(lines)


def format_entity_medium(entity: Entity) -> str:
    """Level 2: Bullet lists, clear structure. Good for LLM attention.
    Example: - App (project) #android: obs1; obs2; obs3
    """
    tags_str = "".join(f" #{t}" for t in entity.tags) if entity.tags else ""
    sorted_obs = _sort_observations(entity.observations)
    obs_parts = [_strip_filler(obs.content) for obs in sorted_obs]
    obs_str = "; ".join(obs_parts) if obs_parts else ""
    result = f"- {entity.name} ({entity.entity_type}){tags_str}"
    if obs_str:
        result += f": {obs_str}"
    return result


def format_entity_heavy(entity: Entity, max_obs_chars: int = 80) -> str:
    """Level 3: Single-line, truncated. Protected obs types always included.
    Example: App:project: obs1 | obs2...
    """
    sorted_obs = _sort_observations(entity.observations)
    obs_parts = [_strip_filler(obs.content) for obs in sorted_obs[:3]]
    obs_str = " | ".join(obs_parts)
    if len(obs_str) > max_obs_chars:
        obs_str = obs_str[:max_obs_chars].rstrip() + "..."
    result = f"{entity.name}:{entity.entity_type}"
    if obs_str:
        result += f": {obs_str}"
    return result


def format_relation_none(rel: Relation) -> dict[str, Any]:
    return rel.to_dict()


def format_relation_light(rel: Relation) -> str:
    return f"{rel.from_name} --[{rel.relation_type}]--> {rel.to_name}"


def format_relation_medium(rel: Relation) -> str:
    return f"{rel.from_name}>{rel.relation_type}>{rel.to_name}"


def format_relation_heavy(rel: Relation) -> str:
    return f"{rel.from_name}>{rel.to_name}"


def compress_graph(
    graph: KnowledgeGraph,
    level: CompressionLevel = CompressionLevel.MEDIUM,
    token_budget: int = 2000,
    pinned_entity_ids: set[int] | None = None,
) -> str:
    """Compress a knowledge graph to fit within token budget.

    Priority order: pinned entities first, then by updated_at (most recent).
    Truncates with '...+N entities omitted' footer if over budget.

    AUTO mode: tries LIGHT → MEDIUM → HEAVY, pinned entities always LIGHT.
    """
    if level == CompressionLevel.AUTO:
        return _compress_auto(graph, token_budget, pinned_entity_ids)

    if level == CompressionLevel.NONE:
        return _compress_json(graph, token_budget)

    pinned_ids = pinned_entity_ids or set()

    # Sort: pinned first, then by updated_at descending
    entities = sorted(
        graph.entities,
        key=lambda e: (0 if e.id in pinned_ids else 1, e.updated_at or ""),
        reverse=False,  # pinned=0 sorts first; for second key we want desc
    )
    # Fix sort: pinned first (0 < 1), then most recent first
    pinned = [e for e in graph.entities if e.id in pinned_ids]
    unpinned = [e for e in graph.entities if e.id not in pinned_ids]
    unpinned.sort(key=lambda e: e.updated_at or "", reverse=True)
    entities = pinned + unpinned

    formatter = {
        CompressionLevel.LIGHT: format_entity_light,
        CompressionLevel.MEDIUM: format_entity_medium,
        CompressionLevel.HEAVY: format_entity_heavy,
    }[level]

    rel_formatter = {
        CompressionLevel.LIGHT: format_relation_light,
        CompressionLevel.MEDIUM: format_relation_medium,
        CompressionLevel.HEAVY: format_relation_heavy,
    }[level]

    parts: list[str] = []
    token_count = 0
    omitted = 0

    for included_entities, entity in enumerate(entities, start=1):
        line = formatter(entity)
        line_tokens = _estimate_tokens(line)
        if token_budget > 0 and token_count + line_tokens > token_budget:
            omitted = len(entities) - (included_entities - 1)
            break
        parts.append(line)
        token_count += line_tokens

    # Add relations if budget allows
    if graph.relations:
        sep = "\n---\n" if level == CompressionLevel.LIGHT else "\n"
        rel_lines = [rel_formatter(r) for r in graph.relations]
        rel_block = (
            sep.join(rel_lines) if level == CompressionLevel.LIGHT else " | ".join(rel_lines)
        )
        rel_tokens = _estimate_tokens(rel_block)
        if token_budget <= 0 or token_count + rel_tokens <= token_budget:
            parts.append(rel_block)
            token_count += rel_tokens

    if omitted > 0:
        parts.append(f"...+{omitted} entities omitted")

    sep = "\n\n" if level == CompressionLevel.LIGHT else "\n"
    return sep.join(parts)


def _compress_json(graph: KnowledgeGraph, token_budget: int) -> str:
    """Level 0: Full JSON, but truncate if over budget."""
    import json

    full = json.dumps(graph.to_dict(), separators=(",", ":"))
    if token_budget > 0 and _estimate_tokens(full) > token_budget:
        # Truncate entities
        d = graph.to_dict()
        while (
            d["entities"] and _estimate_tokens(json.dumps(d, separators=(",", ":"))) > token_budget
        ):
            d["entities"].pop()
        full = json.dumps(d, separators=(",", ":"))
    return full


def _compress_auto(
    graph: KnowledgeGraph,
    token_budget: int,
    pinned_entity_ids: set[int] | None = None,
) -> str:
    """AUTO mode: step through LIGHT → MEDIUM → HEAVY until output fits budget.

    Pinned entities are always formatted at LIGHT level for maximum clarity,
    even when unpinned entities are compressed harder.
    """
    if token_budget <= 0:
        # No budget constraint: use LIGHT for maximum readability
        return compress_graph(graph, CompressionLevel.LIGHT, 0, pinned_entity_ids)

    pinned_ids = pinned_entity_ids or set()

    # Try uniform levels first (fast path)
    for level in (CompressionLevel.LIGHT, CompressionLevel.MEDIUM):
        output = compress_graph(graph, level, token_budget, pinned_entity_ids)
        if _estimate_tokens(output) <= token_budget:
            return output

    # Mixed mode: pinned at LIGHT, unpinned at HEAVY
    pinned = [e for e in graph.entities if e.id in pinned_ids]
    unpinned = [e for e in graph.entities if e.id not in pinned_ids]
    unpinned.sort(key=lambda e: e.updated_at or "", reverse=True)

    parts: list[str] = []
    token_count = 0

    # Pinned entities always at LIGHT level
    for entity in pinned:
        line = format_entity_light(entity)
        parts.append(line)
        token_count += _estimate_tokens(line)

    # Fill remaining budget with unpinned at HEAVY
    omitted = 0
    for entity in unpinned:
        line = format_entity_heavy(entity)
        line_tokens = _estimate_tokens(line)
        if token_count + line_tokens > token_budget:
            omitted = len(unpinned) - (len(parts) - len(pinned))
            break
        parts.append(line)
        token_count += line_tokens

    # Add relations at HEAVY level if budget allows
    if graph.relations:
        rel_block = " | ".join(format_relation_heavy(r) for r in graph.relations)
        rel_tokens = _estimate_tokens(rel_block)
        if token_count + rel_tokens <= token_budget:
            parts.append(rel_block)

    if omitted > 0:
        parts.append(f"...+{omitted} entities omitted")

    return "\n".join(parts)
