"""Conservative privacy, size, and write-policy checks for BMO memory."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import MemoryCandidate, MemoryConfidence, MemorySource


class MemoryPolicyError(ValueError):
    """A candidate was rejected without being written."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


DEFAULT_MAX_KEY_CHARS = 80
DEFAULT_MAX_VALUE_CHARS = 500
DEFAULT_MAX_SUMMARY_CHARS = 240
DEFAULT_MAX_CONTEXT_ENTRIES = 8
DEFAULT_MAX_CONTEXT_CHARS = 3000
DEFAULT_MAX_RECORDS = 500

_SECRET_PATTERNS = (
    re.compile(r"\bpassword\b|\bpassphrase\b|\bapi[ _-]?key\b", re.I),
    re.compile(r"\b(auth|refresh|access)[ _-]?token\b|\bprivate[ _-]?key\b", re.I),
    re.compile(r"\brecovery[ _-]?code\b|\bsecurity answer\b|\bcredit[ -]?card\b", re.I),
    re.compile(r"\b(?:sk|pk)[_-][A-Za-z0-9]{12,}\b", re.I),
    re.compile(r"(?:^|\D)\d{13,19}(?:\D|$)"),
)
_SENSITIVE_PATTERNS = (
    re.compile(r"\bmedical|diagnos|medication|health condition|therapy\b", re.I),
    re.compile(r"\breligion|religious|church|mosque|synagogue\b", re.I),
    re.compile(r"\bpolitic|political party|voted for\b", re.I),
    re.compile(r"\bsexual|sex life|orientation|intimate\b", re.I),
    re.compile(r"\bcriminal record|arrested|conviction\b", re.I),
    re.compile(r"\bsocial security|\bssn\b|\bhome address\b|\bphone number\b|\bprecise location\b", re.I),
)
_TRANSIENT_PATTERNS = (
    re.compile(r"\bjust now\b|\bfor today\b|\btemporarily\b|\bthis session\b", re.I),
    re.compile(r"\bdebug(?:ging)?\b|\btraceback\b|\braw transcript\b|\btool log\b", re.I),
    re.compile(r"\b(?:hidden|system|developer) prompt\b|\bchain[ -]?of[ -]?thought\b|\bprivate reasoning\b", re.I),
)


@dataclass(frozen=True, slots=True)
class MemoryPolicy:
    max_key_chars: int = DEFAULT_MAX_KEY_CHARS
    max_value_chars: int = DEFAULT_MAX_VALUE_CHARS
    max_summary_chars: int = DEFAULT_MAX_SUMMARY_CHARS
    max_context_entries: int = DEFAULT_MAX_CONTEXT_ENTRIES
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS
    max_records: int = DEFAULT_MAX_RECORDS

    def __post_init__(self) -> None:
        for name in (
            "max_key_chars",
            "max_value_chars",
            "max_summary_chars",
            "max_context_entries",
            "max_context_chars",
            "max_records",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @staticmethod
    def _combined(candidate: MemoryCandidate) -> str:
        return " ".join((candidate.key, candidate.value, candidate.summary)).strip()

    @staticmethod
    def _contains(patterns: tuple[re.Pattern[str], ...], text: str) -> bool:
        return any(pattern.search(text) for pattern in patterns)

    def validate(self, candidate: MemoryCandidate, *, explicit: bool = False) -> MemoryCandidate:
        if not isinstance(candidate, MemoryCandidate):
            raise TypeError("memory policy accepts MemoryCandidate values")
        if len(candidate.key.strip()) == 0:
            raise MemoryPolicyError("empty_key", "memory key must not be empty")
        if len(candidate.key.strip()) > self.max_key_chars:
            raise MemoryPolicyError("key_too_long", "memory key exceeds the configured limit")
        if len(candidate.value.strip()) == 0:
            raise MemoryPolicyError("empty_value", "memory value must not be empty")
        if len(candidate.value) > self.max_value_chars:
            raise MemoryPolicyError("value_too_long", "memory value exceeds the configured limit")
        if len(candidate.value.split()) > 80 or "\n" in candidate.value:
            raise MemoryPolicyError("transcript_rejected", "transcript-sized text is not durable memory")
        if len(candidate.summary) > self.max_summary_chars:
            raise MemoryPolicyError("summary_too_long", "memory summary exceeds the configured limit")
        if candidate.priority < -100 or candidate.priority > 100:
            raise MemoryPolicyError("invalid_priority", "memory priority is outside the allowed range")
        combined = self._combined(candidate)
        if self._contains(_SECRET_PATTERNS, combined):
            raise MemoryPolicyError("secret_rejected", "secrets and authentication material are never stored")
        if self._contains(_SENSITIVE_PATTERNS, combined):
            raise MemoryPolicyError("sensitive_rejected", "sensitive personal data is not stored")
        if self._contains(_TRANSIENT_PATTERNS, combined):
            raise MemoryPolicyError("transient_rejected", "transient or diagnostic information is not durable memory")
        if candidate.category.value == "identity":
            allowed_user_keys = {"preferred_name", "user_name", "pronouns", "timezone"}
            if candidate.key.casefold() not in allowed_user_keys:
                raise MemoryPolicyError("identity_boundary", "identity memory is limited to user facts")
            if candidate.key.casefold() not in {"preferred_name", "user_name"} and re.search(
                r"\b(?:bmo|jarvis|robot|assistant)\b", candidate.value, re.I
            ):
                raise MemoryPolicyError("identity_boundary", "BMO identity is defined by the immutable profile")
        if candidate.source is MemorySource.LLM_CANDIDATE and candidate.confidence is MemoryConfidence.LOW:
            raise MemoryPolicyError("weak_candidate", "low-confidence model candidates require explicit confirmation")
        if not explicit and candidate.source is MemorySource.EXPLICIT_USER:
            # This is harmless for callers that construct an explicit candidate,
            # but explicit intent is still recorded by the source value.
            pass
        return MemoryCandidate(
            candidate.category,
            candidate.key.strip(),
            candidate.value.strip(),
            candidate.summary.strip(),
            candidate.source,
            candidate.confidence,
            candidate.priority,
        )
