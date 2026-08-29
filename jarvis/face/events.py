"""Provider-neutral observation events for the face boundary."""

from __future__ import annotations

from dataclasses import dataclass

from .state import FaceState


@dataclass(frozen=True, slots=True)
class FaceStateEvent:
    state: FaceState


@dataclass(frozen=True, slots=True)
class PlaybackEvent:
    """A lifecycle marker with only an opaque generation identity."""

    kind: str
    generation_id: int

    def __post_init__(self) -> None:
        if self.kind not in {"started", "stopped", "cancelled"}:
            raise ValueError("playback event kind is invalid")
        if isinstance(self.generation_id, bool) or not isinstance(self.generation_id, int):
            raise TypeError("generation_id must be an integer")


FaceEvent = FaceStateEvent
FaceStateChanged = FaceStateEvent
VoiceStateEvent = FaceStateEvent
