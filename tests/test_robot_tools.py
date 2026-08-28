import ast
from pathlib import Path

import pytest

from jarvis.robot.controller import SafeRobotController, simulated_safe_snapshot
from jarvis.robot.intents import RobotAction, RobotExpression
from jarvis.robot.safety import (
    HeartbeatState,
    SafetyAuthority,
    SafetyReason,
    SafetySignal,
    SafetySnapshot,
    SafetySupervisor,
)
from jarvis.robot.simulator import HeadDirection, MotionState, SimulatedRobot
from jarvis.tools.policy import MAX_TOOL_CALLS_PER_BATCH, RobotToolPolicy
from jarvis.tools.registry import (
    LLM_EXPRESSION_VALUES,
    RobotToolRegistry,
    ToolValidationError,
)
from jarvis.tools.types import ToolCall, ToolResultStatus


ROOT = Path(__file__).resolve().parents[1]


def runtime(snapshot=None, *, events=None):
    sink = events.append if events is not None else None
    supervisor = SafetySupervisor(snapshot or simulated_safe_snapshot())
    controller = SafeRobotController(supervisor, SimulatedRobot(event_sink=sink))
    return supervisor, controller, RobotToolPolicy(RobotToolRegistry(), controller)


def test_registry_contains_only_the_explicit_robot_allowlist():
    names = {definition.name for definition in RobotToolRegistry().definitions}

    assert names == {
        "stop",
        "wave",
        "nod",
        "shrug",
        "point_left",
        "point_right",
        "look_left",
        "look_right",
        "look_at_user",
        "follow_person",
        "stop_following",
        "move_forward",
        "move_backward",
        "turn_left",
        "turn_right",
        "set_expression",
    }
    assert names.isdisjoint(
        {
            "clear_estop",
            "reset_estop",
            "disable_safety",
            "ignore_obstacle",
            "ignore_cliff",
            "set_heartbeat",
            "modify_safety_state",
        }
    )


def test_known_tool_resolves_to_high_level_intent():
    validated = RobotToolRegistry().validate(ToolCall("wave", {}))

    assert validated.intent.action is RobotAction.WAVE
    assert validated.intent.expression is None


@pytest.mark.parametrize(
    ("call", "reason"),
    [
        (ToolCall("not_registered", {}), "unknown_tool"),
        (ToolCall("wave", {"speed": 10}), "unexpected_arguments"),
        (ToolCall("set_expression", {}), "missing_arguments"),
        (ToolCall("set_expression", {"expression": "angry"}), "invalid_expression"),
        (
            ToolCall("set_expression", {"expression": "happy", "brightness": 100}),
            "unexpected_arguments",
        ),
    ],
)
def test_unknown_or_malformed_calls_are_rejected(call, reason):
    with pytest.raises(ToolValidationError) as raised:
        RobotToolRegistry().validate(call)

    assert raised.value.reason == reason


@pytest.mark.parametrize("expression", LLM_EXPRESSION_VALUES)
def test_expression_enum_is_constrained_and_validated(expression):
    validated = RobotToolRegistry().validate(
        ToolCall("set_expression", {"expression": expression})
    )

    assert validated.intent.expression is RobotExpression(expression)


def test_tool_schemas_expose_no_actuator_level_parameters():
    definitions = RobotToolRegistry().definitions
    parameters = {parameter.name for definition in definitions for parameter in definition.parameters}
    forbidden = {
        "distance",
        "speed",
        "wheel_speed",
        "pwm",
        "duty_cycle",
        "voltage",
        "gpio",
        "servo_angle",
        "duration",
        "serial",
    }

    assert parameters == {"expression"}
    assert parameters.isdisjoint(forbidden)


def test_multiple_non_conflicting_actions_execute_sequentially():
    _, controller, policy = runtime()

    results = policy.execute(
        (
            ToolCall("wave", {}),
            ToolCall("look_left", {}),
            ToolCall("follow_person", {}),
        )
    )

    state = controller.state
    assert all(result.success for result in results)
    assert state.last_gesture is RobotAction.WAVE
    assert state.head is HeadDirection.LEFT
    assert state.following is True
    assert state.motion is MotionState.FOLLOWING
    assert state.event_log == ("gesture=wave", "head=left", "follow=active")


def test_stop_executes_first_and_suppresses_all_other_physical_motion():
    _, controller, policy = runtime()

    results = policy.execute(
        (
            ToolCall("move_forward", {}),
            ToolCall("stop", {}),
            ToolCall("wave", {}),
        )
    )

    assert [result.status for result in results] == [
        ToolResultStatus.DENIED,
        ToolResultStatus.SUCCESS,
        ToolResultStatus.DENIED,
    ]
    assert results[0].denial_reason == "stop_precedence"
    assert results[2].denial_reason == "stop_precedence"
    assert controller.state.motion is MotionState.STOPPED
    assert controller.state.last_gesture is None
    assert controller.state.event_log == ("motion=stopped follow=inactive",)


def test_one_malformed_call_rejects_other_non_stop_actions_before_execution():
    _, controller, policy = runtime()

    results = policy.execute(
        (ToolCall("look_left", {}), ToolCall("wave", {"speed": 1}))
    )

    assert results[0].denial_reason == "batch_validation_failed"
    assert results[1].denial_reason == "unexpected_arguments"
    assert controller.state.head is HeadDirection.CENTER
    assert controller.state.event_log == ()


def test_oversized_tool_batch_executes_nothing():
    _, controller, policy = runtime()
    calls = tuple(ToolCall("wave", {}) for _ in range(MAX_TOOL_CALLS_PER_BATCH + 1))

    results = policy.execute(calls)

    assert len(results) == len(calls)
    assert all(result.denial_reason == "tool_batch_limit" for result in results)
    assert controller.state.event_log == ()


@pytest.mark.parametrize(
    ("snapshot", "tool", "reason"),
    [
        (
            SafetySnapshot(
                emergency_stop=SafetySignal.CLEAR,
                heartbeat=HeartbeatState.STALE,
                controller=simulated_safe_snapshot().controller,
                obstacle_ahead=SafetySignal.CLEAR,
                cliff_or_drop=SafetySignal.CLEAR,
            ),
            "turn_left",
            SafetyReason.HEARTBEAT_STALE.value,
        ),
        (
            SafetySnapshot(
                emergency_stop=SafetySignal.CLEAR,
                heartbeat=HeartbeatState.FRESH,
                controller=simulated_safe_snapshot().controller,
                obstacle_ahead=SafetySignal.TRIGGERED,
                cliff_or_drop=SafetySignal.CLEAR,
            ),
            "move_forward",
            SafetyReason.OBSTACLE_AHEAD.value,
        ),
        (
            SafetySnapshot(
                emergency_stop=SafetySignal.CLEAR,
                heartbeat=HeartbeatState.FRESH,
                controller=simulated_safe_snapshot().controller,
                obstacle_ahead=SafetySignal.CLEAR,
                cliff_or_drop=SafetySignal.TRIGGERED,
            ),
            "move_backward",
            SafetyReason.CLIFF_OR_DROP.value,
        ),
        (
            SafetySnapshot(
                emergency_stop=SafetySignal.CLEAR,
                heartbeat=HeartbeatState.FRESH,
                controller=simulated_safe_snapshot().controller,
                obstacle_ahead=SafetySignal.UNKNOWN,
                cliff_or_drop=SafetySignal.CLEAR,
            ),
            "wave",
            SafetyReason.SAFETY_STATE_UNKNOWN.value,
        ),
    ],
)
def test_safety_denials_are_structured_and_do_not_move(snapshot, tool, reason):
    _, controller, policy = runtime(snapshot)

    result = policy.execute((ToolCall(tool, {}),))[0]

    assert result.status is ToolResultStatus.DENIED
    assert result.denial_reason == reason
    assert controller.state.motion is MotionState.STOPPED
    assert controller.state.event_log == ()


def test_estop_denies_motion_but_stop_and_expression_remain_allowed():
    supervisor, controller, policy = runtime()
    controller.latch_emergency_stop(authority=SafetyAuthority.LOCAL_OPERATOR)

    denied = policy.execute((ToolCall("wave", {}),))[0]
    stopped = policy.execute((ToolCall("stop", {}),))[0]
    expression = policy.execute(
        (ToolCall("set_expression", {"expression": "concerned"}),)
    )[0]

    assert denied.denial_reason == SafetyReason.EMERGENCY_STOP_LATCHED.value
    assert stopped.success
    assert expression.success
    assert controller.state.last_gesture is None
    assert controller.state.expression is RobotExpression.CONCERNED
    assert supervisor.emergency_stop_latched


def test_stop_is_allowed_even_when_all_safety_state_is_unknown():
    _, controller, policy = runtime(SafetySnapshot())

    result = policy.execute((ToolCall("stop", {}),))[0]

    assert result.success
    assert controller.state.motion is MotionState.STOPPED


def test_denied_action_does_not_modify_existing_motion_state():
    supervisor, controller, policy = runtime()
    assert policy.execute((ToolCall("move_backward", {}),))[0].success
    unsafe = simulated_safe_snapshot()
    supervisor.update_snapshot(
        SafetySnapshot(
            emergency_stop=unsafe.emergency_stop,
            heartbeat=unsafe.heartbeat,
            controller=unsafe.controller,
            obstacle_ahead=SafetySignal.TRIGGERED,
            cliff_or_drop=unsafe.cliff_or_drop,
        ),
        authority=SafetyAuthority.SYSTEM,
    )

    denied = policy.execute((ToolCall("move_forward", {}),))[0]

    assert denied.denial_reason == SafetyReason.OBSTACLE_AHEAD.value
    assert controller.state.motion is MotionState.BACKWARD


def test_stop_clears_motion_and_following():
    _, controller, policy = runtime()
    policy.execute((ToolCall("follow_person", {}),))

    result = policy.execute((ToolCall("stop", {}),))[0]

    assert result.success
    assert controller.state.motion is MotionState.STOPPED
    assert controller.state.following is False


def test_stop_following_clears_follow_state():
    _, controller, policy = runtime()
    policy.execute((ToolCall("follow_person", {}),))

    result = policy.execute((ToolCall("stop_following", {}),))[0]

    assert result.success
    assert controller.state.motion is MotionState.STOPPED
    assert controller.state.following is False


def test_natural_language_tool_surface_cannot_reset_estop():
    supervisor, controller, policy = runtime()
    controller.latch_emergency_stop(authority=SafetyAuthority.LOCAL_OPERATOR)

    result = policy.execute((ToolCall("reset_estop", {}),))[0]

    assert result.status is ToolResultStatus.ERROR
    assert result.denial_reason == "unknown_tool"
    assert supervisor.emergency_stop_latched


def test_simulator_module_has_no_hardware_or_external_io_imports():
    path = ROOT / "jarvis" / "robot" / "simulator.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])

    assert imported.isdisjoint(
        {"socket", "subprocess", "serial", "gpiozero", "RPi", "cv2", "requests", "pathlib", "os"}
    )
