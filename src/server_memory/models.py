"""Data models for the knowledge graph."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Entity:
    id: int = 0
    name: str = ""
    entity_type: str = ""
    metadata_json: str = "{}"
    created_at: str = ""
    updated_at: str = ""
    last_accessed_at: str = ""
    deleted_at: str | None = None
    scope: str = ""

    # Populated by joins, not stored in entities table
    observations: list[Observation] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @property
    def metadata(self) -> dict[str, Any]:
        return json.loads(self.metadata_json) if self.metadata_json else {}

    def __repr__(self) -> str:
        return f"Entity(name={self.name!r}, type={self.entity_type!r}, observations={len(self.observations)})"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "entityType": self.entity_type,
            "observations": [o.content for o in self.observations],
        }
        if self.tags:
            d["tags"] = self.tags
        if self.scope:
            d["scope"] = self.scope
        return d


# Observation types for type-aware compression.
# Protected types survive compression even under heavy budget pressure.
PROTECTED_OBS_TYPES = frozenset(
    {
        "api_endpoint",
        "dependency",
        "file_path",
        "code_snippet",
        "config",
        "credential_ref",
        "schema",
    }
)


@dataclass
class Observation:
    id: int = 0
    entity_id: int = 0
    content: str = ""
    source: str = ""
    confidence: float = 1.0
    importance: float = 0.5  # 0.0-1.0, higher = survives compression
    obs_type: str = ""  # fact, decision, preference, api_endpoint, dependency, etc.
    version: int = 1
    metadata_json: str = "{}"
    created_at: str = ""
    updated_at: str = ""
    deleted_at: str | None = None

    tags: list[str] = field(default_factory=list)


@dataclass
class Relation:
    id: int = 0
    from_entity_id: int = 0
    to_entity_id: int = 0
    relation_type: str = ""
    weight: float = 1.0
    metadata_json: str = "{}"
    created_at: str = ""
    updated_at: str = ""
    deleted_at: str | None = None
    scope: str = ""

    # Populated by joins
    from_name: str = ""
    to_name: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "from": self.from_name,
            "to": self.to_name,
            "relationType": self.relation_type,
        }
        if self.scope:
            result["scope"] = self.scope
        return result


@dataclass
class Tag:
    id: int = 0
    name: str = ""
    description: str = ""
    color: str = ""
    is_system: bool = False
    auto_expire_hours: int | None = None
    created_at: str = ""


@dataclass
class ActivityEntry:
    id: int = 0
    session_id: str = ""
    action: str = ""
    summary: str = ""
    entity_ids_json: str = "[]"
    tags_json: str = "[]"
    metadata_json: str = "{}"
    created_at: str = ""

    @property
    def entity_ids(self) -> list[int]:
        return json.loads(self.entity_ids_json) if self.entity_ids_json else []

    @property
    def tags(self) -> list[str]:
        return json.loads(self.tags_json) if self.tags_json else []


@dataclass
class KnowledgeGraph:
    """Container for graph query results."""

    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
        }
