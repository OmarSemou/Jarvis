"""Small immutable state types shared by face views and observers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FaceActivity(StrEnum):
    IDLE = "idle"
    WAKE_DETECTED = "wake_detected"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class FaceExpression(StrEnum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    AMUSED = "amused"
    CURIOUS = "curious"
    CONFUSED = "confused"
    CONCERNED = "concerned"
    EXCITED = "excited"
    SLEEPY = "sleepy"
    THINKING = "thinking"
    SURPRISED = "surprised"


class FaceGaze(StrEnum):
    CENTER = "center"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"


def _enum(value: object, enum_type: type[StrEnum], name: str) -> StrEnum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except (TypeError, ValueError):
        choices = ", ".join(item.value for item in enum_type)
        raise ValueError(f"{name} must be one of: {choices}") from None


@dataclass(frozen=True, slots=True)
class FaceState:
    """A privacy-preserving snapshot; it contains no transcript or audio."""

    activity: FaceActivity = FaceActivity.IDLE
    expression: FaceExpression = FaceExpression.NEUTRAL
    gaze: FaceGaze = FaceGaze.CENTER
    generation_id: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "activity", _enum(self.activity, FaceActivity, "activity"))
        object.__setattr__(self, "expression", _enum(self.expression, FaceExpression, "expression"))
        object.__setattr__(self, "gaze", _enum(self.gaze, FaceGaze, "gaze"))
        if self.generation_id is not None and (
            isinstance(self.generation_id, bool)
            or not isinstance(self.generation_id, int)
            or self.generation_id < 0
        ):
            raise ValueError("generation_id must be a non-negative integer or null")


FaceSnapshot = FaceState
