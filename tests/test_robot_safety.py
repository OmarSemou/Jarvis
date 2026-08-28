from dataclasses import fields

import pytest

from jarvis.robot.intents import RobotAction, RobotExpression, RobotIntent
from jarvis.robot.safety import (
    ControllerState,
    HeartbeatState,
    SafetyAuthority,
    SafetyReason,
    SafetySignal,
    SafetySnapshot,
    SafetySupervisor,
)


def safe_snapshot(**changes):
    values = {
        "emergency_stop": SafetySignal.CLEAR,
        "heartbeat": HeartbeatState.FRESH,
        "controller": ControllerState.READY,
        "obstacle_ahead": SafetySignal.CLEAR,
        "cliff_or_drop": SafetySignal.CLEAR,
    }
    values.update(changes)
    return SafetySnapshot(**values)


def test_intents_contain_only_high_level_semantic_fields():
    field_names = {field.name for field in fields(RobotIntent)}
    forbidden = {"pwm", "voltage", "duty_cycle", "current", "wheel_speed", "servo_angle", "gpio", "serial", "transport"}

    assert field_names == {"action", "expression"}
    assert field_names.isdisjoint(forbidden)
    assert {action.value for action in RobotAction} >= {
        "stop",
        "move_forward",
        "follow_person",
        "look_at_user",
        "wave",
        "set_expression",
    }


def test_expression_parameter_is_semantic_and_constrained():
    intent = RobotIntent(RobotAction.SET_EXPRESSION, RobotExpression.CURIOUS)
    assert intent.expression is RobotExpression.CURIOUS

    with pytest.raises(ValueError, match="requires"):
        RobotIntent(RobotAction.SET_EXPRESSION)
    with pytest.raises(TypeError, match="RobotAction"):
        RobotIntent("move_forward")
    with pytest.raises(TypeError, match="RobotExpression"):
        RobotIntent(RobotAction.SET_EXPRESSION, "happy")


def test_stop_is_always_permitted_with_unknown_state():
    decision = SafetySupervisor().evaluate(RobotIntent(RobotAction.STOP))

    assert decision.allowed
    assert decision.reason is SafetyReason.STOP_ALWAYS_ALLOWED
    assert decision.approved is not None
    assert decision.approved.lease is None


def test_emergency_stop_prevents_motion():
    supervisor = SafetySupervisor(safe_snapshot(emergency_stop=SafetySignal.TRIGGERED))

    decision = supervisor.evaluate(RobotIntent(RobotAction.MOVE_FORWARD))

    assert not decision.allowed
    assert decision.reason is SafetyReason.EMERGENCY_STOP_LATCHED


@pytest.mark.parametrize("authority", [SafetyAuthority.LLM, SafetyAuthority.TOOL])
def test_ai_or_tool_cannot_unlatch_emergency_stop(authority):
    supervisor = SafetySupervisor(safe_snapshot())
    supervisor.latch_emergency_stop(authority=SafetyAuthority.HARDWARE)

    transition = supervisor.request_emergency_stop_reset(authority=authority)

    assert not transition.accepted
    assert transition.reason is SafetyReason.UNAUTHORIZED_ESTOP_RESET
    assert supervisor.emergency_stop_latched


def test_authorized_reset_requires_clear_physical_input():
    supervisor = SafetySupervisor(safe_snapshot(emergency_stop=SafetySignal.TRIGGERED))

    denied = supervisor.request_emergency_stop_reset(authority=SafetyAuthority.LOCAL_OPERATOR)
    assert not denied.accepted
    assert denied.reason is SafetyReason.ESTOP_INPUT_NOT_CLEAR

    supervisor.update_snapshot(safe_snapshot(), authority=SafetyAuthority.HARDWARE)
    accepted = supervisor.request_emergency_stop_reset(authority=SafetyAuthority.LOCAL_OPERATOR)
    assert accepted.accepted
    assert not supervisor.emergency_stop_latched


def test_stale_heartbeat_prevents_motion():
    supervisor = SafetySupervisor(safe_snapshot(heartbeat=HeartbeatState.STALE))

    decision = supervisor.evaluate(RobotIntent(RobotAction.TURN_LEFT))

    assert not decision.allowed
    assert decision.reason is SafetyReason.HEARTBEAT_STALE


def test_obstacle_prevents_forward_movement():
    supervisor = SafetySupervisor(safe_snapshot(obstacle_ahead=SafetySignal.TRIGGERED))

    forward = supervisor.evaluate(RobotIntent(RobotAction.MOVE_FORWARD))
    turn = supervisor.evaluate(RobotIntent(RobotAction.TURN_LEFT), now_monotonic=10.0)

    assert not forward.allowed
    assert forward.reason is SafetyReason.OBSTACLE_AHEAD
    assert turn.allowed


@pytest.mark.parametrize(
    "action",
    [
        RobotAction.MOVE_FORWARD,
        RobotAction.MOVE_BACKWARD,
        RobotAction.TURN_LEFT,
        RobotAction.TURN_RIGHT,
        RobotAction.FOLLOW_PERSON,
    ],
)
def test_cliff_prevents_base_movement(action):
    supervisor = SafetySupervisor(safe_snapshot(cliff_or_drop=SafetySignal.TRIGGERED))

    decision = supervisor.evaluate(RobotIntent(action))

    assert not decision.allowed
    assert decision.reason is SafetyReason.CLIFF_OR_DROP


@pytest.mark.parametrize("unsafe", [SafetySignal.UNKNOWN, SafetySignal.FAILED])
def test_unknown_or_failed_sensor_state_fails_closed(unsafe):
    supervisor = SafetySupervisor(safe_snapshot(obstacle_ahead=unsafe))

    decision = supervisor.evaluate(RobotIntent(RobotAction.WAVE))

    assert not decision.allowed
    assert decision.reason is SafetyReason.SAFETY_STATE_UNKNOWN
    assert decision.message


def test_missing_safety_state_fails_closed():
    decision = SafetySupervisor().evaluate(RobotIntent(RobotAction.MOVE_BACKWARD))

    assert not decision.allowed
    assert decision.reason is SafetyReason.EMERGENCY_STOP_STATE_UNSAFE


def test_malformed_safety_snapshot_is_rejected_before_use():
    with pytest.raises(TypeError, match="emergency_stop"):
        SafetySnapshot(emergency_stop="clear")


def test_llm_cannot_publish_fake_sensor_state():
    supervisor = SafetySupervisor()

    transition = supervisor.update_snapshot(safe_snapshot(), authority=SafetyAuthority.LLM)

    assert not transition.accepted
    assert transition.reason is SafetyReason.UNAUTHORIZED_STATE_UPDATE
    assert supervisor.snapshot == SafetySnapshot()


def test_approved_motion_has_a_short_renewable_lease():
    supervisor = SafetySupervisor(safe_snapshot(), lease_seconds=0.25)

    decision = supervisor.evaluate(RobotIntent(RobotAction.NOD), now_monotonic=100.0)

    assert decision.allowed
    assert decision.approved is not None
    assert decision.approved.lease is not None
    assert decision.approved.lease.is_valid(100.1)
    assert not decision.approved.lease.is_valid(100.25)
