"""Explicit allowlist and strict argument validation for robot tools."""

from __future__ import annotations

from dataclasses import dataclass

from jarvis.robot.intents import RobotAction, RobotExpression, RobotIntent

from .types import ToolCall, ToolDefinition, ToolParameter, ToolParameterType


LLM_EXPRESSION_VALUES = (
    RobotExpression.NEUTRAL.value,
    RobotExpression.HAPPY.value,
    RobotExpression.AMUSED.value,
    RobotExpression.CURIOUS.value,
    RobotExpression.CONFUSED.value,
    RobotExpression.CONCERNED.value,
    RobotExpression.EXCITED.value,
    RobotExpression.SLEEPY.value,
)


_ACTION_BY_TOOL: dict[str, RobotAction] = {
    "stop": RobotAction.STOP,
    "wave": RobotAction.WAVE,
    "nod": RobotAction.NOD,
    "shrug": RobotAction.SHRUG,
    "point_left": RobotAction.POINT_LEFT,
    "point_right": RobotAction.POINT_RIGHT,
    "look_left": RobotAction.LOOK_LEFT,
    "look_right": RobotAction.LOOK_RIGHT,
    "look_at_user": RobotAction.LOOK_AT_USER,
    "follow_person": RobotAction.FOLLOW_PERSON,
    "stop_following": RobotAction.STOP_FOLLOWING,
    "move_forward": RobotAction.MOVE_FORWARD,
    "move_backward": RobotAction.MOVE_BACKWARD,
    "turn_left": RobotAction.TURN_LEFT,
    "turn_right": RobotAction.TURN_RIGHT,
    "set_expression": RobotAction.SET_EXPRESSION,
}


def _definition(name: str, description: str) -> ToolDefinition:
    return ToolDefinition(name, description)


ROBOT_TOOL_DEFINITIONS = (
    _definition("stop", "Immediately stop simulated robot motion and following."),
    _definition("wave", "Perform one simulated wave gesture."),
    _definition("nod", "Perform one simulated nod gesture."),
    _definition("shrug", "Perform one simulated shrug gesture."),
    _definition("point_left", "Point to the simulated robot's left."),
    _definition("point_right", "Point to the simulated robot's right."),
    _definition("look_left", "Turn the simulated robot's head to its left."),
    _definition("look_right", "Turn the simulated robot's head to its right."),
    _definition("look_at_user", "Aim the simulated robot's head toward the user."),
    _definition("follow_person", "Start safe simulated person-following motion."),
    _definition("stop_following", "Stop simulated person following."),
    _definition("move_forward", "Start safe semantic forward motion in simulation."),
    _definition("move_backward", "Start safe semantic backward motion in simulation."),
    _definition("turn_left", "Start a safe semantic left turn in simulation."),
    _definition("turn_right", "Start a safe semantic right turn in simulation."),
    ToolDefinition(
        "set_expression",
        "Set the simulated screen-face expression to one allowed value.",
        (
            ToolParameter(
                "expression",
                "The constrained expression name.",
                ToolParameterType.STRING,
                allowed_values=LLM_EXPRESSION_VALUES,
            ),
        ),
    ),
)


class ToolValidationError(ValueError):
    def __init__(self, call: ToolCall, reason: str, message: str) -> None:
        self.call = call
        self.reason = reason
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ValidatedRobotTool:
    call: ToolCall
    intent: RobotIntent


class RobotToolRegistry:
    """Resolve only statically declared names; never reflect over robot methods."""

    def __init__(self) -> None:
        self._definitions = ROBOT_TOOL_DEFINITIONS
        self._by_name = {definition.name: definition for definition in self._definitions}

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return self._definitions

    def validate(self, call: ToolCall) -> ValidatedRobotTool:
        if not isinstance(call, ToolCall):
            raise TypeError("registry accepts only ToolCall values")
        definition = self._by_name.get(call.name)
        action = _ACTION_BY_TOOL.get(call.name)
        if definition is None or action is None:
            raise ToolValidationError(call, "unknown_tool", f"Tool '{call.name}' is not registered.")

        expected = {parameter.name for parameter in definition.parameters}
        supplied = set(call.arguments)
        unexpected = supplied - expected
        missing = {
            parameter.name
            for parameter in definition.parameters
            if parameter.required and parameter.name not in supplied
        }
        if unexpected:
            names = ", ".join(sorted(str(name) for name in unexpected))
            raise ToolValidationError(
                call,
                "unexpected_arguments",
                f"Tool '{call.name}' received unexpected argument(s): {names}.",
            )
        if missing:
            names = ", ".join(sorted(missing))
            raise ToolValidationError(
                call,
                "missing_arguments",
                f"Tool '{call.name}' is missing required argument(s): {names}.",
            )

        if action is RobotAction.SET_EXPRESSION:
            value = call.arguments.get("expression")
            if not isinstance(value, str) or value not in LLM_EXPRESSION_VALUES:
                allowed = ", ".join(LLM_EXPRESSION_VALUES)
                raise ToolValidationError(
                    call,
                    "invalid_expression",
                    f"Expression must be one of: {allowed}.",
                )
            return ValidatedRobotTool(call, RobotIntent(action, RobotExpression(value)))
        return ValidatedRobotTool(call, RobotIntent(action))
