"""Safety-gated controller for semantic simulated robot intents."""

from __future__ import annotations

from dataclasses import dataclass

from .intents import RobotAction, RobotIntent
from .safety import (
    ControllerState,
    HeartbeatState,
    SafetyAuthority,
    SafetySignal,
    SafetySnapshot,
    SafetySupervisor,
    SafetyTransition,
)
from .simulator import EventSink, SimulatedRobot, SimulatedRobotState


@dataclass(frozen=True, slots=True)
class RobotExecution:
    success: bool
    message: str
    reason: str | None = None


_SUCCESS_MESSAGES: dict[RobotAction, str] = {
    RobotAction.STOP: "Robot stopped.",
    RobotAction.WAVE: "Wave completed.",
    RobotAction.NOD: "Nod completed.",
    RobotAction.SHRUG: "Shrug completed.",
    RobotAction.POINT_LEFT: "Pointed left.",
    RobotAction.POINT_RIGHT: "Pointed right.",
    RobotAction.LOOK_LEFT: "Looked left.",
    RobotAction.LOOK_RIGHT: "Looked right.",
    RobotAction.LOOK_AT_USER: "Looked at the user.",
    RobotAction.FOLLOW_PERSON: "Following started.",
    RobotAction.STOP_FOLLOWING: "Following stopped.",
    RobotAction.MOVE_FORWARD: "Forward motion started.",
    RobotAction.MOVE_BACKWARD: "Backward motion started.",
    RobotAction.TURN_LEFT: "Left turn started.",
    RobotAction.TURN_RIGHT: "Right turn started.",
    RobotAction.SET_EXPRESSION: "Expression updated.",
}


def simulated_safe_snapshot() -> SafetySnapshot:
    """Explicit synthetic inputs for desktop simulation, never hardware claims."""

    return SafetySnapshot(
        emergency_stop=SafetySignal.CLEAR,
        heartbeat=HeartbeatState.FRESH,
        controller=ControllerState.READY,
        obstacle_ahead=SafetySignal.CLEAR,
        cliff_or_drop=SafetySignal.CLEAR,
    )


class SafeRobotController:
    """The sole tool-facing path into safety evaluation and the simulator."""

    def __init__(self, supervisor: SafetySupervisor, simulator: SimulatedRobot) -> None:
        self._supervisor = supervisor
        self._simulator = simulator
        self._simulator.sync_emergency_stop(supervisor.emergency_stop_latched)

    @property
    def safety(self) -> SafetySupervisor:
        return self._supervisor

    @property
    def state(self) -> SimulatedRobotState:
        self._simulator.sync_emergency_stop(self._supervisor.emergency_stop_latched)
        return self._simulator.state

    def execute_intent(self, intent: RobotIntent) -> RobotExecution:
        decision = self._supervisor.evaluate(intent)
        self._simulator.sync_emergency_stop(self._supervisor.emergency_stop_latched)
        if not decision.allowed or decision.approved is None:
            return RobotExecution(False, decision.message, decision.reason.value)
        try:
            self._simulator.execute(decision.approved)
        except Exception:
            self._simulator.stop()
            return RobotExecution(
                False,
                "simulator execution failed safely",
                "controller_execution_failed",
            )
        return RobotExecution(True, _SUCCESS_MESSAGES[intent.action])

    def latch_emergency_stop(self, *, authority: SafetyAuthority) -> SafetyTransition:
        transition = self._supervisor.latch_emergency_stop(authority=authority)
        self._simulator.stop()
        self._simulator.sync_emergency_stop(self._supervisor.emergency_stop_latched)
        return transition

    def reset_emergency_stop(self, *, authority: SafetyAuthority) -> SafetyTransition:
        transition = self._supervisor.request_emergency_stop_reset(authority=authority)
        self._simulator.sync_emergency_stop(self._supervisor.emergency_stop_latched)
        return transition


def create_simulated_controller(*, event_sink: EventSink | None = None) -> SafeRobotController:
    supervisor = SafetySupervisor(simulated_safe_snapshot())
    return SafeRobotController(supervisor, SimulatedRobot(event_sink=event_sink))
