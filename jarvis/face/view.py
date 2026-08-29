"""View protocol with no GUI toolkit dependency."""

from __future__ import annotations

from typing import Protocol

from .assets import FaceAsset
from .state import FaceState


class FaceView(Protocol):
    def render(self, state: FaceState, asset: FaceAsset) -> None: ...

    def close(self) -> None: ...
