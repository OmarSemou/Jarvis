"""Explicit developer demo for the animated face."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .assets import FaceAssetSet, default_bmo_asset_set
from .controller import FaceController
from .state import FaceActivity, FaceExpression
from .tkinter_view import TkinterFaceView


@dataclass(frozen=True, slots=True)
class FaceDemoSettings:
    fullscreen: bool = False
    gallery: bool = False


def run_face_demo(
    *,
    settings: FaceDemoSettings = FaceDemoSettings(),
    assets: FaceAssetSet | None = None,
    output_fn: Callable[[str], None] = print,
) -> int:
    asset_set = assets or default_bmo_asset_set()
    output_fn("Face: BMO prototype")
    output_fn("Assets are read-only prototype artwork; press Escape to close.")
    controller = FaceController(asset_set)
    sequence = (
        (FaceActivity.IDLE, FaceExpression.NEUTRAL),
        (FaceActivity.WAKE_DETECTED, FaceExpression.EXCITED),
        (FaceActivity.LISTENING, FaceExpression.CURIOUS),
        (FaceActivity.PROCESSING, FaceExpression.CONFUSED),
        (FaceActivity.SPEAKING, FaceExpression.HAPPY),
        (FaceActivity.ERROR, FaceExpression.CONCERNED),
    )

    def on_ready(view: TkinterFaceView) -> None:
        assert view.root is not None
        index = 0

        def advance() -> None:
            nonlocal index
            activity, expression = sequence[index % len(sequence)]
            controller.set_expression(expression)
            controller.set_activity(activity)
            index += 1
            if view.root is not None:
                view.root.after(1_200, advance)

        view.root.after(50, advance)

    view = TkinterFaceView(
        controller,
        asset_set,
        fullscreen=settings.fullscreen,
        gallery=settings.gallery,
        on_ready=on_ready,
    )
    view.run()
    return 0
