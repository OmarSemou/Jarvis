"""Bridge deterministic voice STOP to the safety-gated robot controller."""

from __future__ import annotations

from jarvis.audio.voice.commands import (
    LocalVoiceCommand,
    LocalVoiceCommandResult,
)
from jarvis.robot.controller import SafeRobotController
from jarvis.robot.intents import RobotAction, RobotIntent


class SafeLocalVoiceCommandExecutor:
    """Translate only local STOP into the existing semantic controller path."""

    def __init__(self, controller: SafeRobotController) -> None:
        self._controller = controller

    def execute(self, command: LocalVoiceCommand) -> LocalVoiceCommandResult:
        if command is not LocalVoiceCommand.STOP:
            raise ValueError("local voice execution permits only stop")
        execution = self._controller.execute_intent(RobotIntent(RobotAction.STOP))
        return LocalVoiceCommandResult(
            execution.success,
            execution.message,
            execution.reason,
        )
