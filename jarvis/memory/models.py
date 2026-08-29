"""Provider-neutral memory values and durable record models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping


class MemoryCategory(StrEnum):
    IDENTITY = "identity"
    PREFERENCE = "preference"
    PROJECT = "project"
    RELATIONSHIP = "relationship"
    ROUTINE = "routine"
    INSTRUCTION = "instruction"
    SETTING = "setting"
    GENERAL = "general"


class MemorySource(StrEnum):
    EXPLICIT_USER = "explicit_user"
    LLM_CANDIDATE = "llm_candidate"
    DEVELOPER_CLI = "developer_cli"
    MIGRATION = "migration"


class MemoryConfidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True, slots=True)
class MemoryCandidate:
    """A proposed fact, before the privacy and retention policy accepts it."""

    category: MemoryCategory | str
    key: str
    value: str
    summary: str = ""
    source: MemorySource | str = MemorySource.EXPLICIT_USER
    confidence: MemoryConfidence | str = MemoryConfidence.HIGH
    priority: int = 0

    def __post_init__(self) -> None:
        try:
            category = self.category.value if isinstance(self.category, MemoryCategory) else str(self.category).casefold()
            source = self.source.value if isinstance(self.source, MemorySource) else str(self.source).casefold()
            confidence = self.confidence.value if isinstance(self.confidence, MemoryConfidence) else str(self.confidence).casefold()
            object.__setattr__(self, "category", MemoryCategory(category))
            object.__setattr__(self, "source", MemorySource(source))
            object.__setattr__(self, "confidence", MemoryConfidence(confidence))
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid memory category, source, or confidence") from exc
        if not isinstance(self.key, str) or not isinstance(self.value, str):
            raise TypeError("memory key and value must be strings")
        if not isinstance(self.summary, str):
            raise TypeError("memory summary must be a string")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise TypeError("memory priority must be an integer")


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    id: int
    category: MemoryCategory
    key: str
    value: str
    summary: str
    source: MemorySource
    confidence: MemoryConfidence
    created_at: str
    updated_at: str
    last_used_at: str | None
    is_active: bool = True
    priority: int = 0
    supersedes_id: int | None = None

    def public_dict(self) -> Mapping[str, object]:
        """Return an inspectable representation with no provider-specific data."""

        return MappingProxyType(
            {
                "id": self.id,
                "category": self.category.value,
                "key": self.key,
                "value": self.value,
                "summary": self.summary,
                "source": self.source.value,
                "confidence": self.confidence.value,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "last_used_at": self.last_used_at,
                "is_active": self.is_active,
                "priority": self.priority,
                "supersedes_id": self.supersedes_id,
            }
        )
