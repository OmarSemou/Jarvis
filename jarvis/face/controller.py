"""Thread-safe, observation-only face controller."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable

from .assets import FaceAssetSet, default_bmo_asset_set
from .state import FaceActivity, FaceExpression, FaceGaze, FaceState
from .view import FaceView


class FaceController:
    """Collect state from other subsystems and deliver it to a view on demand.

    Producers may be voice/audio/robot threads.  Only :meth:`pump` invokes a
    view, allowing a Tk view to remain on its owning main thread.
    """

    def __init__(
        self,
        assets: FaceAssetSet | None = None,
        view: FaceView | None = None,
    ) -> None:
        self.assets = assets or default_bmo_asset_set()
        self._view = view
        self._lock = threading.RLock()
        self._pending: queue.SimpleQueue[FaceState] = queue.SimpleQueue()
        self._state = FaceState()
        self._active_generation: int | None = None
        self._latest_generation = -1
        self._enqueue(self._state)

    def _enqueue(self, state: FaceState) -> FaceState:
        with self._lock:
            self._state = state
            self._pending.put(state)
        return state

    @property
    def snapshot(self) -> FaceState:
        with self._lock:
            return self._state

    @property
    def state(self) -> FaceState:
        return self.snapshot

    def bind_view(self, view: FaceView | None) -> None:
        with self._lock:
            self._view = view

    def set_activity(
        self,
        activity: FaceActivity | str,
        *,
        generation_id: int | None = None,
    ) -> FaceState:
        with self._lock:
            return self._enqueue(
                FaceState(activity, self._state.expression, self._state.gaze, generation_id)
            )

    def set_expression(self, expression: FaceExpression | str) -> FaceState:
        with self._lock:
            return self._enqueue(
                FaceState(
                    self._state.activity,
                    expression,
                    self._state.gaze,
                    self._state.generation_id,
                )
            )

    def set_gaze(self, gaze: FaceGaze | str) -> FaceState:
        with self._lock:
            return self._enqueue(
                FaceState(
                    self._state.activity,
                    self._state.expression,
                    gaze,
                    self._state.generation_id,
                )
            )

    def observe_voice_state(self, state: object) -> FaceState:
        value = getattr(state, "value", state)
        try:
            activity = FaceActivity(str(value))
        except ValueError:
            activity = FaceActivity.ERROR
        return self.set_activity(activity)

    def observe_robot_expression(self, expression: object) -> FaceState:
        value = getattr(expression, "value", expression)
        try:
            return self.set_expression(FaceExpression(str(value)))
        except ValueError:
            return self.set_expression(FaceExpression.NEUTRAL)

    def observe_robot_gaze(self, gaze: object) -> FaceState:
        value = getattr(gaze, "value", gaze)
        if str(value) == "user":
            value = FaceGaze.CENTER
        try:
            return self.set_gaze(FaceGaze(str(value)))
        except ValueError:
            return self.set_gaze(FaceGaze.CENTER)

    def observe_robot_state(self, state: object) -> FaceState:
        expression = getattr(state, "expression", None)
        head = getattr(state, "head", None)
        if expression is not None:
            self.observe_robot_expression(expression)
        if head is not None:
            self.observe_robot_gaze(head)
        return self.snapshot

    def on_playback_started(self, generation_id: int) -> FaceState | None:
        if (
            isinstance(generation_id, bool)
            or not isinstance(generation_id, int)
            or generation_id < 0
        ):
            raise ValueError("generation_id must be a non-negative integer")
        with self._lock:
            if generation_id < self._latest_generation:
                return None
            self._latest_generation = generation_id
            self._active_generation = generation_id
            return self._enqueue(
                FaceState(
                    FaceActivity.SPEAKING,
                    self._state.expression,
                    self._state.gaze,
                    generation_id,
                )
            )

    def _playback_end(self, kind: str, generation_id: int) -> FaceState | None:
        if (
            isinstance(generation_id, bool)
            or not isinstance(generation_id, int)
            or generation_id < 0
        ):
            raise ValueError("generation_id must be a non-negative integer")
        with self._lock:
            if generation_id != self._active_generation:
                return None
            self._active_generation = None
            target = FaceActivity.INTERRUPTED if kind == "cancelled" else FaceActivity.IDLE
            return self._enqueue(FaceState(target, self._state.expression, self._state.gaze, None))

    def on_playback_stopped(self, generation_id: int) -> FaceState | None:
        return self._playback_end("stopped", generation_id)

    def on_playback_cancelled(self, generation_id: int) -> FaceState | None:
        return self._playback_end("cancelled", generation_id)

    def observe_playback_event(self, kind: str, generation_id: int) -> FaceState | None:
        if kind == "started":
            return self.on_playback_started(generation_id)
        if kind == "stopped":
            return self.on_playback_stopped(generation_id)
        if kind == "cancelled":
            return self.on_playback_cancelled(generation_id)
        raise ValueError("unknown playback event")

    def pump(self, max_events: int | None = None) -> int:
        if max_events is not None and max_events < 1:
            raise ValueError("max_events must be positive or null")
        delivered = 0
        while max_events is None or delivered < max_events:
            try:
                state = self._pending.get_nowait()
            except queue.Empty:
                break
            with self._lock:
                view = self._view
            if view is not None:
                view.render(state, self.assets.asset_for(state))
            delivered += 1
        return delivered
