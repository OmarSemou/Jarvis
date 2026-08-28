"""Deterministic, hardware-free robot simulation downstream of safety."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from .intents import RobotAction, RobotExpression
from .interfaces import ApprovedRobotIntent


class MotionState(StrEnum):
    STOPPED = "stopped"
    FORWARD = "forward"
    BACKWARD = "backward"
    TURNING_LEFT = "turning_left"
    TURNING_RIGHT = "turning_right"
    FOLLOWING = "following"


class HeadDirection(StrEnum):
    CENTER = "center"
    LEFT = "left"
    RIGHT = "right"
    USER = "user"


@dataclass(frozen=True, slots=True)
class SimulatedRobotState:
    motion: MotionState
    following: bool
    head: HeadDirection
    last_gesture: RobotAction | None
    expression: RobotExpression
    emergency_stop_latched: bool
    event_log: tuple[str, ...]


EventSink = Callable[[str], None]


class SimulatedRobot:
    """Meaning-level state machine with no device, filesystem, or network access."""

    def __init__(self, *, event_sink: EventSink | None = None) -> None:
        self._lock = threading.RLock()
        self._event_sink = event_sink
        self._motion = MotionState.STOPPED
        self._following = False
        self._head = HeadDirection.CENTER
        self._last_gesture: RobotAction | None = None
        self._expression = RobotExpression.NEUTRAL
        self._emergency_stop_latched = False
        self._event_log: list[str] = []

    @property
    def state(self) -> SimulatedRobotState:
        with self._lock:
            return SimulatedRobotState(
                motion=self._motion,
                following=self._following,
                head=self._head,
                last_gesture=self._last_gesture,
                expression=self._expression,
                emergency_stop_latched=self._emergency_stop_latched,
                event_log=tuple(self._event_log),
            )

    def _record(self, event: str) -> None:
        self._event_log.append(event)
        if self._event_sink is not None:
            try:
                self._event_sink(f"[ROBOT] {event}")
            except Exception:
                # Developer logging is observational and cannot change execution.
                pass

    def sync_emergency_stop(self, latched: bool) -> None:
        """Mirror trusted safety state for simulator status reporting."""

        if not isinstance(latched, bool):
            raise TypeError("latched must be a boolean")
        with self._lock:
            if self._emergency_stop_latched != latched:
                self._emergency_stop_latched = latched
                self._record(f"estop={'latched' if latched else 'clear'}")

    def execute(self, approved: ApprovedRobotIntent) -> None:
        if not isinstance(approved, ApprovedRobotIntent):
            raise TypeError("simulator accepts only ApprovedRobotIntent values")
        intent = approved.intent
        if intent.is_physical_motion:
            if approved.lease is None or not approved.lease.is_valid(time.monotonic()):
                raise RuntimeError("approved physical intent has no valid movement lease")

        with self._lock:
            action = intent.action
            if action is RobotAction.STOP:
                self.stop()
            elif action is RobotAction.STOP_FOLLOWING:
                self._following = False
                self._motion = MotionState.STOPPED
                self._record("follow=inactive motion=stopped")
            elif action is RobotAction.MOVE_FORWARD:
                self._following = False
                self._motion = MotionState.FORWARD
                self._record("motion=forward")
            elif action is RobotAction.MOVE_BACKWARD:
                self._following = False
                self._motion = MotionState.BACKWARD
                self._record("motion=backward")
            elif action is RobotAction.TURN_LEFT:
                self._following = False
                self._motion = MotionState.TURNING_LEFT
                self._record("motion=turning_left")
            elif action is RobotAction.TURN_RIGHT:
                self._following = False
                self._motion = MotionState.TURNING_RIGHT
                self._record("motion=turning_right")
            elif action is RobotAction.FOLLOW_PERSON:
                self._following = True
                self._motion = MotionState.FOLLOWING
                self._record("follow=active")
            elif action is RobotAction.LOOK_LEFT:
                self._head = HeadDirection.LEFT
                self._record("head=left")
            elif action is RobotAction.LOOK_RIGHT:
                self._head = HeadDirection.RIGHT
                self._record("head=right")
            elif action is RobotAction.LOOK_AT_USER:
                self._head = HeadDirection.USER
                self._record("head=user")
            elif action in {
                RobotAction.NOD,
                RobotAction.WAVE,
                RobotAction.POINT_LEFT,
                RobotAction.POINT_RIGHT,
                RobotAction.SHRUG,
            }:
                self._last_gesture = action
                self._record(f"gesture={action.value}")
            elif action is RobotAction.SET_EXPRESSION:
                if intent.expression is None:
                    raise RuntimeError("approved expression intent is missing its expression")
                self._expression = intent.expression
                self._record(f"expression={intent.expression.value}")
            else:
                raise RuntimeError(f"simulator does not implement action '{action.value}'")

    def stop(self) -> None:
        with self._lock:
            self._motion = MotionState.STOPPED
            self._following = False
            self._record("motion=stopped follow=inactive")
