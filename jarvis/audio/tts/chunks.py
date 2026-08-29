"""Deterministic, provider-neutral splitting of speech-safe assistant text."""

from __future__ import annotations

import re
from dataclasses import dataclass


_COMMON_ABBREVIATIONS = frozenset(
    {
        "co",
        "dept",
        "dr",
        "e.g",
        "etc",
        "fig",
        "i.e",
        "inc",
        "jr",
        "ltd",
        "mr",
        "mrs",
        "ms",
        "no",
        "prof",
        "sr",
        "st",
        "vs",
    }
)
_INITIALS = re.compile(r"(?:[A-Za-z]\.)*[A-Za-z]$")
_NUMBERED_ITEM = re.compile(r"^\s*\d+$")
_WHITESPACE = re.compile(r"\s+")
_CLOSERS = "\"')]}"


@dataclass(frozen=True, slots=True)
class SpeechChunk:
    """One ordered semantic unit ready for a local speech provider."""

    sequence: int
    text: str

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("speech chunk sequence cannot be negative")
        if not self.text.strip():
            raise ValueError("speech chunk text cannot be empty")


@dataclass(frozen=True, slots=True)
class SpeechChunkerSettings:
    """Small deterministic bounds; no NLP package or provider knowledge."""

    max_characters: int = 220

    def __post_init__(self) -> None:
        if not 40 <= self.max_characters <= 1_000:
            raise ValueError("speech chunk maximum must be from 40 to 1000 characters")


class SpeechChunker:
    """Split already-normalized speech text at natural, bounded boundaries."""

    def __init__(self, settings: SpeechChunkerSettings = SpeechChunkerSettings()) -> None:
        self.settings = settings

    @staticmethod
    def _period_is_protected(text: str, index: int, start: int) -> bool:
        previous = text[index - 1] if index else ""
        following = text[index + 1] if index + 1 < len(text) else ""
        if previous.isdigit() and following.isdigit():
            return True

        prefix = text[start:index].rstrip()
        token = prefix.rsplit(maxsplit=1)[-1] if prefix else ""
        lowered = token.casefold()
        if lowered in _COMMON_ABBREVIATIONS:
            return True
        if _NUMBERED_ITEM.fullmatch(prefix):
            return True
        # Protect initials ("J. R. R.") and compact initialisms ("U.S.").
        if _INITIALS.fullmatch(token) and len(token.replace(".", "")) <= 3:
            before_token = prefix[: -len(token)].rstrip()
            if "." in token or not before_token or before_token.endswith("."):
                return True
        return False

    @classmethod
    def _sentence_fragments(cls, paragraph: str) -> list[str]:
        fragments: list[str] = []
        start = 0
        index = 0
        while index < len(paragraph):
            character = paragraph[index]
            boundary = character in "!?;:"
            if character == ".":
                boundary = not cls._period_is_protected(paragraph, index, start)
            if not boundary:
                index += 1
                continue

            end = index + 1
            while end < len(paragraph) and paragraph[end] in _CLOSERS:
                end += 1
            if end < len(paragraph) and not paragraph[end].isspace():
                index += 1
                continue
            fragment = paragraph[start:end].strip()
            if fragment:
                fragments.append(fragment)
            start = end
            while start < len(paragraph) and paragraph[start].isspace():
                start += 1
            index = start

        remainder = paragraph[start:].strip()
        if remainder:
            fragments.append(remainder)
        return fragments

    def _bounded_parts(self, text: str) -> list[str]:
        maximum = self.settings.max_characters
        remaining = text.strip()
        parts: list[str] = []
        while len(remaining) > maximum:
            window = remaining[: maximum + 1]
            split_at = max(
                window.rfind(mark, 0, maximum) for mark in (", ", "; ", ": ")
            )
            if split_at >= max(20, maximum // 3):
                split_at += 1
            else:
                split_at = window.rfind(" ", 0, maximum + 1)
            if split_at < 1:
                split_at = maximum
            part = remaining[:split_at].strip()
            if part:
                parts.append(part)
            remaining = remaining[split_at:].strip()
        if remaining:
            parts.append(remaining)
        return parts

    def chunk(self, speech_text: str) -> tuple[SpeechChunk, ...]:
        if not isinstance(speech_text, str):
            raise TypeError("speech text must be a string")
        semantic_parts: list[str] = []
        for paragraph in re.split(r"\n+", speech_text):
            normalized = _WHITESPACE.sub(" ", paragraph).strip()
            if not normalized:
                continue
            fragments = self._sentence_fragments(normalized)
            combined: list[str] = []
            index = 0
            while index < len(fragments):
                fragment = fragments[index]
                while (
                    fragment.endswith((":", ";"))
                    and len(fragment) < 40
                    and index + 1 < len(fragments)
                    and len(fragment) + 1 + len(fragments[index + 1])
                    <= self.settings.max_characters
                ):
                    index += 1
                    fragment = f"{fragment} {fragments[index]}"
                combined.append(fragment)
                index += 1
            for fragment in combined:
                semantic_parts.extend(self._bounded_parts(fragment))
        return tuple(
            SpeechChunk(sequence, text)
            for sequence, text in enumerate(semantic_parts)
            if text
        )
