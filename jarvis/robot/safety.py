"""Fail-closed, LLM-independent robot safety contracts."""

from __future__ import annotations

import threading
import time
import math
from dataclasses import dataclass
from enum import StrEnum

from .intents import (
    BASE_MOTION_ACTIONS,
    FORWARD_RISK_ACTIONS,
    PHYSICAL_MOTION_ACTIONS,
    STOP_ACTIONS,
    RobotAction,
    RobotIntent,
)
from .interfaces import ApprovedRobotIntent, MovementLease


class SafetySignal(StrEnum):
    CLEAR = "clear"
    TRIGGERED = "triggered"
    UNKNOWN = "unknown"
    FAILED = "failed"


class HeartbeatState(StrEnum):
    FRESH = "fresh"
    STALE = "stale"
    UNKNOWN = "unknown"
    FAILED = "failed"


class ControllerState(StrEnum):
    READY = "ready"
    NOT_READY = "not_ready"
    UNKNOWN = "unknown"
    FAILED = "failed"


class SafetyAuthority(StrEnum):
    HARDWARE = "hardware"
    LOCAL_OPERATOR = "local_operator"
    SYSTEM = "system"
    TOOL = "tool"
    LLM = "llm"


class SafetyReason(StrEnum):
    ALLOWED = "allowed"
    STOP_ALWAYS_ALLOWED = "stop_always_allowed"
    NON_PHYSICAL_ACTION = "non_physical_action"
    EMERGENCY_STOP_LATCHED = "emergency_stop_latched"
    EMERGENCY_STOP_STATE_UNSAFE = "emergency_stop_state_unsafe"
    CONTROLLER_NOT_READY = "controller_not_ready"
    HEARTBEAT_STALE = "heartbeat_stale"
    HEARTBEAT_UNSAFE = "heartbeat_unsafe"
    SAFETY_STATE_UNKNOWN = "safety_state_unknown"
    OBSTACLE_AHEAD = "obstacle_ahead"
    CLIFF_OR_DROP = "cliff_or_drop"
    UNAUTHORIZED_STATE_UPDATE = "unauthorized_state_update"
    UNAUTHORIZED_ESTOP_RESET = "unauthorized_estop_reset"
    ESTOP_INPUT_NOT_CLEAR = "estop_input_not_clear"


@dataclass(frozen=True, slots=True)
class SafetySnapshot:
    """Latest deterministic safety inputs.

    Defaults are unknown so absence of sensor/control state fails closed.
    """

    emergency_stop: SafetySignal = SafetySignal.UNKNOWN
    heartbeat: HeartbeatState = HeartbeatState.UNKNOWN
    controller: ControllerState = ControllerState.UNKNOWN
    obstacle_ahead: SafetySignal = SafetySignal.UNKNOWN
    cliff_or_drop: SafetySignal = SafetySignal.UNKNOWN

    def __post_init__(self) -> None:
        expected = {
            "emergency_stop": (self.emergency_stop, SafetySignal),
            "heartbeat": (self.heartbeat, HeartbeatState),
            "controller": (self.controller, ControllerState),
            "obstacle_ahead": (self.obstacle_ahead, SafetySignal),
            "cliff_or_drop": (self.cliff_or_drop, SafetySignal),
        }
        for name, (value, enum_type) in expected.items():
            if not isinstance(value, enum_type):
                raise TypeError(f"{name} must be a {enum_type.__name__}")


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    allowed: bool
    reason: SafetyReason
    message: str
    approved: ApprovedRobotIntent | None = None


@dataclass(frozen=True, slots=True)
class SafetyTransition:
    accepted: bool
    reason: SafetyReason
    message: str
    emergency_stop_latched: bool


class SafetySupervisor:
    """Evaluates semantic intents using deterministic, fail-closed rules."""

    _SENSOR_AUTHORITIES = frozenset({SafetyAuthority.HARDWARE, SafetyAuthority.SYSTEM})
    _RESET_AUTHORITIES = frozenset({SafetyAuthority.HARDWARE, SafetyAuthority.LOCAL_OPERATOR})

    def __init__(self, snapshot: SafetySnapshot | None = None, *, lease_seconds: float = 0.5) -> None:
        if not isinstance(lease_seconds, (int, float)) or isinstance(lease_seconds, bool):
            raise TypeError("lease_seconds must be a number")
        if not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise ValueError("lease_seconds must be finite and positive")
        self._lock = threading.RLock()
        self._snapshot = snapshot or SafetySnapshot()
        self._emergency_stop_latched = self._snapshot.emergency_stop is SafetySignal.TRIGGERED
        self._lease_seconds = float(lease_seconds)

    @property
    def snapshot(self) -> SafetySnapshot:
        with self._lock:
            return self._snapshot

    @property
    def emergency_stop_latched(self) -> bool:
        with self._lock:
            return self._emergency_stop_latched

    def update_snapshot(self, snapshot: SafetySnapshot, *, authority: SafetyAuthority) -> SafetyTransition:
        """Accept sensor/control state only from deterministic system sources."""

        with self._lock:
            if not isinstance(authority, SafetyAuthority) or authority not in self._SENSOR_AUTHORITIES:
                authority_name = authority.value if isinstance(authority, SafetyAuthority) else "unknown"
                return SafetyTransition(
                    False,
                    SafetyReason.UNAUTHORIZED_STATE_UPDATE,
                    f"{authority_name} cannot update safety sensor state",
                    self._emergency_stop_latched,
                )
            self._snapshot = snapshot
            if snapshot.emergency_stop is SafetySignal.TRIGGERED:
                self._emergency_stop_latched = True
            return SafetyTransition(
                True,
                SafetyReason.ALLOWED,
                "safety state updated",
                self._emergency_stop_latched,
            )

    def latch_emergency_stop(self, *, authority: SafetyAuthority) -> SafetyTransition:
        """Any source may request a safer state; reset authority remains local."""

        with self._lock:
            self._emergency_stop_latched = True
            authority_name = authority.value if isinstance(authority, SafetyAuthority) else "unknown"
            return SafetyTransition(
                True,
                SafetyReason.ALLOWED,
                f"emergency stop latched by {authority_name}",
                True,
            )

    def request_emergency_stop_reset(self, *, authority: SafetyAuthority) -> SafetyTransition:
        """Reject LLM/tool resets and require a confirmed-clear physical input."""

        with self._lock:
            if not isinstance(authority, SafetyAuthority) or authority not in self._RESET_AUTHORITIES:
                authority_name = authority.value if isinstance(authority, SafetyAuthority) else "unknown"
                return SafetyTransition(
                    False,
                    SafetyReason.UNAUTHORIZED_ESTOP_RESET,
                    f"{authority_name} cannot reset the emergency stop",
                    self._emergency_stop_latched,
                )
            if self._snapshot.emergency_stop is not SafetySignal.CLEAR:
                return SafetyTransition(
                    False,
                    SafetyReason.ESTOP_INPUT_NOT_CLEAR,
                    "physical emergency-stop input is not confirmed clear",
                    self._emergency_stop_latched,
                )
            self._emergency_stop_latched = False
            return SafetyTransition(True, SafetyReason.ALLOWED, "emergency stop reset", False)

    def evaluate(self, intent: RobotIntent, *, now_monotonic: float | None = None) -> SafetyDecision:
        """Return an explicit approval or denial for a high-level intent."""

        with self._lock:
            if intent.action in STOP_ACTIONS:
                return SafetyDecision(
                    True,
                    SafetyReason.STOP_ALWAYS_ALLOWED,
                    "stop actions are always permitted",
                    ApprovedRobotIntent(intent, None),
                )
            if intent.action is RobotAction.SET_EXPRESSION:
                return SafetyDecision(
                    True,
                    SafetyReason.NON_PHYSICAL_ACTION,
                    "non-physical expression change permitted",
                    ApprovedRobotIntent(intent, None),
                )
            if intent.action not in PHYSICAL_MOTION_ACTIONS:
                return self._deny(SafetyReason.SAFETY_STATE_UNKNOWN, "unclassified action fails closed")
            if self._emergency_stop_latched:
                return self._deny(SafetyReason.EMERGENCY_STOP_LATCHED, "emergency stop is latched")

            state = self._snapshot
            if state.emergency_stop is not SafetySignal.CLEAR:
                return self._deny(
                    SafetyReason.EMERGENCY_STOP_STATE_UNSAFE,
                    f"emergency-stop input is {state.emergency_stop.value}",
                )
            if state.controller is not ControllerState.READY:
                return self._deny(
                    SafetyReason.CONTROLLER_NOT_READY,
                    f"controller state is {state.controller.value}",
                )
            if state.heartbeat is HeartbeatState.STALE:
                return self._deny(SafetyReason.HEARTBEAT_STALE, "control heartbeat is stale")
            if state.heartbeat is not HeartbeatState.FRESH:
                return self._deny(
                    SafetyReason.HEARTBEAT_UNSAFE,
                    f"control heartbeat is {state.heartbeat.value}",
                )
            if state.obstacle_ahead in {SafetySignal.UNKNOWN, SafetySignal.FAILED}:
                return self._deny(
                    SafetyReason.SAFETY_STATE_UNKNOWN,
                    f"obstacle state is {state.obstacle_ahead.value}",
                )
            if state.cliff_or_drop in {SafetySignal.UNKNOWN, SafetySignal.FAILED}:
                return self._deny(
                    SafetyReason.SAFETY_STATE_UNKNOWN,
                    f"cliff/drop state is {state.cliff_or_drop.value}",
                )
            if intent.action in FORWARD_RISK_ACTIONS and state.obstacle_ahead is SafetySignal.TRIGGERED:
                return self._deny(SafetyReason.OBSTACLE_AHEAD, "obstacle vetoed forward movement")
            if intent.action in BASE_MOTION_ACTIONS and state.cliff_or_drop is SafetySignal.TRIGGERED:
                return self._deny(SafetyReason.CLIFF_OR_DROP, "cliff/drop vetoed base movement")

            issued = time.monotonic() if now_monotonic is None else float(now_monotonic)
            if not math.isfinite(issued):
                return self._deny(SafetyReason.SAFETY_STATE_UNKNOWN, "movement lease clock is not finite")
            lease = MovementLease(issued, issued + self._lease_seconds)
            return SafetyDecision(
                True,
                SafetyReason.ALLOWED,
                "intent permitted by current deterministic safety state",
                ApprovedRobotIntent(intent, lease),
            )

    @staticmethod
    def _deny(reason: SafetyReason, message: str) -> SafetyDecision:
        return SafetyDecision(False, reason, message, None)
