"""Meaning-level robot intents that may eventually be exposed as tools."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RobotAction(StrEnum):
    STOP = "stop"
    MOVE_FORWARD = "move_forward"
    MOVE_BACKWARD = "move_backward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    FOLLOW_PERSON = "follow_person"
    STOP_FOLLOWING = "stop_following"
    LOOK_LEFT = "look_left"
    LOOK_RIGHT = "look_right"
    LOOK_AT_USER = "look_at_user"
    NOD = "nod"
    WAVE = "wave"
    POINT_LEFT = "point_left"
    POINT_RIGHT = "point_right"
    SHRUG = "shrug"
    SET_EXPRESSION = "set_expression"


class RobotExpression(StrEnum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    AMUSED = "amused"
    CURIOUS = "curious"
    CONFUSED = "confused"
    THINKING = "thinking"
    CONCERNED = "concerned"
    EXCITED = "excited"
    SLEEPY = "sleepy"
    SURPRISED = "surprised"


STOP_ACTIONS = frozenset({RobotAction.STOP, RobotAction.STOP_FOLLOWING})
BASE_MOTION_ACTIONS = frozenset(
    {
        RobotAction.MOVE_FORWARD,
        RobotAction.MOVE_BACKWARD,
        RobotAction.TURN_LEFT,
        RobotAction.TURN_RIGHT,
        RobotAction.FOLLOW_PERSON,
    }
)
FORWARD_RISK_ACTIONS = frozenset({RobotAction.MOVE_FORWARD, RobotAction.FOLLOW_PERSON})
EXPRESSIVE_MOTION_ACTIONS = frozenset(
    {
        RobotAction.LOOK_LEFT,
        RobotAction.LOOK_RIGHT,
        RobotAction.LOOK_AT_USER,
        RobotAction.NOD,
        RobotAction.WAVE,
        RobotAction.POINT_LEFT,
        RobotAction.POINT_RIGHT,
        RobotAction.SHRUG,
    }
)
PHYSICAL_MOTION_ACTIONS = BASE_MOTION_ACTIONS | EXPRESSIVE_MOTION_ACTIONS


@dataclass(frozen=True, slots=True)
class RobotIntent:
    """A semantic request with no low-level actuator parameters."""

    action: RobotAction
    expression: RobotExpression | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action, RobotAction):
            raise TypeError("action must be a RobotAction")
        if self.expression is not None and not isinstance(self.expression, RobotExpression):
            raise TypeError("expression must be a RobotExpression or null")
        if self.action is RobotAction.SET_EXPRESSION and self.expression is None:
            raise ValueError("set_expression requires a named RobotExpression")
        if self.action is not RobotAction.SET_EXPRESSION and self.expression is not None:
            raise ValueError("expression is only valid for set_expression")

    @property
    def is_stop(self) -> bool:
        return self.action in STOP_ACTIONS

    @property
    def is_physical_motion(self) -> bool:
        return self.action in PHYSICAL_MOTION_ACTIONS

    @property
    def is_base_motion(self) -> bool:
        return self.action in BASE_MOTION_ACTIONS
