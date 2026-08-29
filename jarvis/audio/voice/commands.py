"""Tiny deterministic grammar for safety-oriented local voice commands."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class LocalVoiceCommand(StrEnum):
    """Commands that may bypass the probabilistic language model."""

    STOP = "stop"


@dataclass(frozen=True, slots=True)
class LocalVoiceCommandResult:
    success: bool
    message: str
    reason: str | None = None


class LocalVoiceCommandExecutor(Protocol):
    """Injected execution boundary; parsing itself has no robot authority."""

    def execute(self, command: LocalVoiceCommand) -> LocalVoiceCommandResult: ...


_STOP_GRAMMAR = frozenset(
    {
        ("stop",),
        ("stop", "now"),
        ("please", "stop"),
        ("please", "stop", "now"),
        ("jarvis", "stop"),
        ("jarvis", "stop", "now"),
        ("hey", "jarvis", "stop"),
        ("hey", "jarvis", "stop", "now"),
        ("hey", "jarvis", "please", "stop"),
        ("hey", "jarvis", "please", "stop", "now"),
        ("bmo", "stop"),
        ("bmo", "stop", "now"),
        ("hey", "bmo", "stop"),
        ("hey", "bmo", "stop", "now"),
        ("hey", "bmo", "please", "stop"),
        ("hey", "bmo", "please", "stop", "now"),
    }
)


def _normalized_tokens(text: str) -> tuple[str, ...]:
    if not isinstance(text, str):
        raise TypeError("voice command text must be a string")
    normalized = unicodedata.normalize("NFKC", text).casefold().strip()
    punctuation_spaced = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in normalized
    )
    return tuple(punctuation_spaced.split())


class LocalVoiceCommandRouter:
    """Match only explicit, anchored STOP utterances from an allowlist."""

    @staticmethod
    def match(text: str) -> LocalVoiceCommand | None:
        return (
            LocalVoiceCommand.STOP
            if _normalized_tokens(text) in _STOP_GRAMMAR
            else None
        )


_NO_SPEECH_MARKERS = frozenset(
    {
        "[blank_audio]",
        "[blank audio]",
        "[silence]",
        "[ silence ]",
        "(silence)",
        "<silence>",
        "[no speech]",
        "[no_speech]",
        "[inaudible]",
    }
)

_WAKE_ONLY_GRAMMAR = frozenset(
    {
        ("jarvis",),
        ("hey", "jarvis"),
        ("bmo",),
        ("hey", "bmo"),
    }
)


def is_no_speech_transcript(text: str) -> bool:
    """Recognize only empty text and known whole-output Whisper markers."""

    if not isinstance(text, str):
        raise TypeError("transcript must be a string")
    normalized = " ".join(text.casefold().strip().split()).strip(".,!?;:")
    return not normalized or normalized in _NO_SPEECH_MARKERS


def is_wake_only_transcript(text: str) -> bool:
    """Recognize an activation phrase with no following command."""

    return _normalized_tokens(text) in _WAKE_ONLY_GRAMMAR
