"""Explicit state machine for the voice interaction lifecycle."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum


class VoiceInteractionState(StrEnum):
    IDLE = "idle"
    WAKE_DETECTED = "wake_detected"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class VoiceStateTransitionError(RuntimeError):
    """Raised when application code requests an impossible transition."""


StateSink = Callable[[VoiceInteractionState], None]

_ALLOWED: dict[VoiceInteractionState, frozenset[VoiceInteractionState]] = {
    VoiceInteractionState.IDLE: frozenset(
        {
            VoiceInteractionState.WAKE_DETECTED,
            VoiceInteractionState.ERROR,
            VoiceInteractionState.SHUTDOWN,
        }
    ),
    VoiceInteractionState.WAKE_DETECTED: frozenset(
        {
            VoiceInteractionState.LISTENING,
            VoiceInteractionState.ERROR,
            VoiceInteractionState.SHUTDOWN,
        }
    ),
    VoiceInteractionState.LISTENING: frozenset(
        {
            VoiceInteractionState.PROCESSING,
            VoiceInteractionState.IDLE,
            VoiceInteractionState.ERROR,
            VoiceInteractionState.SHUTDOWN,
        }
    ),
    VoiceInteractionState.PROCESSING: frozenset(
        {
            VoiceInteractionState.SPEAKING,
            VoiceInteractionState.IDLE,
            VoiceInteractionState.ERROR,
            VoiceInteractionState.SHUTDOWN,
        }
    ),
    VoiceInteractionState.SPEAKING: frozenset(
        {
            VoiceInteractionState.IDLE,
            VoiceInteractionState.INTERRUPTED,
            VoiceInteractionState.ERROR,
            VoiceInteractionState.SHUTDOWN,
        }
    ),
    VoiceInteractionState.INTERRUPTED: frozenset(
        {
            VoiceInteractionState.LISTENING,
            VoiceInteractionState.ERROR,
            VoiceInteractionState.SHUTDOWN,
        }
    ),
    VoiceInteractionState.ERROR: frozenset(
        {VoiceInteractionState.IDLE, VoiceInteractionState.SHUTDOWN}
    ),
    VoiceInteractionState.SHUTDOWN: frozenset(),
}


@dataclass(slots=True)
class VoiceStateMachine:
    on_transition: StateSink | None = None
    current: VoiceInteractionState = VoiceInteractionState.IDLE
    history: list[VoiceInteractionState] = field(
        default_factory=lambda: [VoiceInteractionState.IDLE]
    )

    def transition(self, target: VoiceInteractionState) -> None:
        if target not in _ALLOWED[self.current]:
            raise VoiceStateTransitionError(
                f"Voice state cannot transition from {self.current.value} to {target.value}."
            )
        self.current = target
        self.history.append(target)
        if self.on_transition is not None:
            self.on_transition(target)

    def fail_to_idle(self) -> None:
        if self.current is VoiceInteractionState.SHUTDOWN:
            return
        if self.current is not VoiceInteractionState.ERROR:
            self.transition(VoiceInteractionState.ERROR)
        self.transition(VoiceInteractionState.IDLE)

    def shutdown(self) -> None:
        if self.current is not VoiceInteractionState.SHUTDOWN:
            self.transition(VoiceInteractionState.SHUTDOWN)
