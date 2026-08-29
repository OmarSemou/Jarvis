"""Conservative recognition and extraction for explicit memory requests.

This module is intentionally narrow.  It recognizes a strong user request to
persist or forget a fact, then extracts only a small set of relationship forms
that can be represented safely as a structured memory candidate.  Ordinary
conversation is never converted into memory here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import MemoryCandidate, MemoryCategory, MemoryConfidence, MemorySource


@dataclass(frozen=True, slots=True)
class ExplicitMemoryRequest:
    """One explicit persistence request recognized from the current turn."""

    kind: str
    candidate: MemoryCandidate | None = None
    query: str = ""

    def __post_init__(self) -> None:
        if self.kind not in {"remember", "forget"}:
            raise ValueError("memory request kind must be remember or forget")
        if self.kind == "remember" and self.query:
            raise ValueError("remember requests do not carry a forget query")
        if self.kind == "forget" and self.candidate is not None:
            raise ValueError("forget requests do not carry a candidate")


_REMEMBER_PREFIX = re.compile(
    r"^\s*(?:please\s+)?(?:can\s+you\s+)?"
    r"(?:remember|memorize|don't\s+forget|do\s+not\s+forget|"
    r"keep\s+in\s+mind|save\s+this|store\s+this)\b"
    r"(?:\s+(?:that|this(?:\s+for\s+(?:later|next\s+time))?))?"
    r"\s*:?[\s]*(?P<fact>.+?)\s*[.!?]*$",
    re.IGNORECASE,
)
_FORGET_PREFIX = re.compile(
    r"^\s*(?:please\s+)?(?:can\s+you\s+)?"
    r"(?:forget|remove\s+from\s+memory|delete\s+from\s+memory)\b"
    r"(?:\s+(?:that|this))?\s*:?[\s]*(?P<fact>.+?)\s*[.!?]*$",
    re.IGNORECASE,
)


def _clean_fact(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip())
    value = re.sub(r"^(?:that|this)\s+", "", value, flags=re.IGNORECASE)
    return value.strip(" .!?\t\r\n")


def _slug(value: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", value.casefold())
    return "_".join(words[:8])


def _candidate(fact: str) -> MemoryCandidate | None:
    """Extract a bounded structured fact from a clear relationship sentence."""

    fact = _clean_fact(fact)
    if not fact:
        return None

    # Stable, common preference forms.  The object is kept in the key while
    # the value contains only the preference itself (not the command text).
    match = re.fullmatch(
        r"my\s+(?:favorite|favourite)\s+(?P<object>[A-Za-z][A-Za-z0-9 -]{0,50}?)\s+"
        r"(?:is|=)\s+(?P<value>.+)",
        fact,
        flags=re.IGNORECASE,
    )
    if match:
        object_name = _slug(match.group("object"))
        value = _clean_fact(match.group("value"))
        if object_name and value:
            return MemoryCandidate(
                MemoryCategory.PREFERENCE,
                f"favorite_{object_name}",
                value,
                source=MemorySource.EXPLICIT_USER,
                confidence=MemoryConfidence.HIGH,
            )

    match = re.fullmatch(
        r"my\s+preferred\s+(?P<object>[A-Za-z][A-Za-z0-9 -]{0,50}?)\s+"
        r"(?:is|=)\s+(?P<value>.+)",
        fact,
        flags=re.IGNORECASE,
    )
    if match:
        object_name = _slug(match.group("object"))
        value = _clean_fact(match.group("value"))
        if object_name and value:
            return MemoryCandidate(
                MemoryCategory.PREFERENCE,
                f"preferred_{object_name}",
                value,
                source=MemorySource.EXPLICIT_USER,
                confidence=MemoryConfidence.HIGH,
            )

    # "I prefer pistachio ice cream" is normalized to the same logical key as
    # "my favorite ice cream is pistachio".  Keep this list deliberately small;
    # unknown object/value relationships fail closed instead of storing prose.
    match = re.fullmatch(
        r"i\s+(?:prefer|like|love|enjoy)\s+(?:the\s+)?(?P<value>.+?)\s+"
        r"(?P<object>ice\s+cream|voice|color|colour|music|language|tea|coffee)",
        fact,
        flags=re.IGNORECASE,
    )
    if match:
        object_name = _slug(match.group("object"))
        value = _clean_fact(match.group("value"))
        if object_name and value:
            return MemoryCandidate(
                MemoryCategory.PREFERENCE,
                f"favorite_{object_name}",
                value,
                source=MemorySource.EXPLICIT_USER,
                confidence=MemoryConfidence.HIGH,
            )

    match = re.fullmatch(
        r"my\s+(?P<key>[A-Za-z][A-Za-z0-9 -]{0,50}?)\s+(?:is|=)\s+(?P<value>.+)",
        fact,
        flags=re.IGNORECASE,
    )
    if match:
        raw_key = _slug(match.group("key"))
        value = _clean_fact(match.group("value"))
        if raw_key and value:
            if raw_key in {"name", "full_name", "preferred_name", "user_name"}:
                category = MemoryCategory.IDENTITY
                key = "user_name" if raw_key in {"name", "full_name"} else raw_key
            else:
                category = MemoryCategory.PREFERENCE
                key = raw_key
            return MemoryCandidate(
                category,
                key,
                value,
                source=MemorySource.EXPLICIT_USER,
                confidence=MemoryConfidence.HIGH,
            )

    # A small setting form is useful for explicit voice/UI preferences without
    # treating arbitrary "I ..." statements as durable memory.
    match = re.fullmatch(
        r"i\s+(?:use|choose|prefer)\s+(?:the\s+)?(?P<value>.+?)\s+"
        r"(?P<object>voice|language|theme|mode)",
        fact,
        flags=re.IGNORECASE,
    )
    if match:
        object_name = _slug(match.group("object"))
        value = _clean_fact(match.group("value"))
        if object_name and value:
            return MemoryCandidate(
                MemoryCategory.SETTING,
                f"preferred_{object_name}",
                value,
                source=MemorySource.EXPLICIT_USER,
                confidence=MemoryConfidence.HIGH,
            )
    return None


def recognize_explicit_memory_request(text: str) -> ExplicitMemoryRequest | None:
    """Recognize only strong, command-like remember/forget turns."""

    if not isinstance(text, str) or not text.strip():
        return None
    remember = _REMEMBER_PREFIX.match(text)
    if remember:
        return ExplicitMemoryRequest("remember", candidate=_candidate(remember.group("fact")))
    forget = _FORGET_PREFIX.match(text)
    if forget:
        fact = _clean_fact(forget.group("fact"))
        # Search terms are normalized enough to match canonical keys, while
        # preserving a user-provided phrase for unsupported forms.
        fact = re.sub(r"^my\s+", "", fact, flags=re.IGNORECASE)
        return ExplicitMemoryRequest("forget", query=fact)
    return None


def is_persistent_memory_query(text: str) -> bool:
    """Identify questions about durable memory, not ordinary recall."""

    if not isinstance(text, str):
        return False
    lowered = re.sub(r"\s+", " ", text.casefold()).strip()
    return any(
        phrase in lowered
        for phrase in (
            "what do you remember",
            "what is in your memory",
            "what's in your memory",
            "what is in persistent memory",
            "what's in persistent memory",
            "persistent memory",
            "in your memory",
            "stored memory",
            "remember my ",
            "do you remember",
            "is that in memory",
            "is that in your memory",
            "memory about me",
        )
    )

