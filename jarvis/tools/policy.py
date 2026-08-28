"""Batch policy joining tool validation to the safety-gated controller."""

from __future__ import annotations

from jarvis.robot.controller import SafeRobotController
from jarvis.robot.intents import RobotAction

from .registry import RobotToolRegistry, ToolValidationError, ValidatedRobotTool
from .types import ToolCall, ToolDefinition, ToolResult, ToolResultStatus


MAX_TOOL_CALLS_PER_BATCH = 8


class RobotToolPolicy:
    """Validate a whole batch, then execute allowed actions sequentially.

    A valid ``stop`` executes before every other call in its batch and suppresses
    all other physical motion. If any call is malformed, no non-stop call runs.
    """

    def __init__(self, registry: RobotToolRegistry, controller: SafeRobotController) -> None:
        self._registry = registry
        self._controller = controller

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return self._registry.definitions

    def _execute_validated(self, validated: ValidatedRobotTool) -> ToolResult:
        execution = self._controller.execute_intent(validated.intent)
        if execution.success:
            return ToolResult(validated.call, ToolResultStatus.SUCCESS, execution.message)
        status = (
            ToolResultStatus.ERROR
            if execution.reason == "controller_execution_failed"
            else ToolResultStatus.DENIED
        )
        return ToolResult(
            validated.call,
            status,
            execution.message,
            execution.reason or "execution_denied",
        )

    def execute(self, calls: tuple[ToolCall, ...]) -> tuple[ToolResult, ...]:
        calls = tuple(calls)
        if any(not isinstance(call, ToolCall) for call in calls):
            raise TypeError("tool batches must contain only ToolCall values")
        if not calls:
            return ()
        if len(calls) > MAX_TOOL_CALLS_PER_BATCH:
            return tuple(
                ToolResult(
                    call,
                    ToolResultStatus.DENIED,
                    f"Tool batch exceeds the limit of {MAX_TOOL_CALLS_PER_BATCH} calls.",
                    "tool_batch_limit",
                )
                for call in calls
            )

        validated: list[ValidatedRobotTool | None] = []
        results: list[ToolResult | None] = [None] * len(calls)
        validation_failed = False
        for index, call in enumerate(calls):
            try:
                validated.append(self._registry.validate(call))
            except ToolValidationError as exc:
                validated.append(None)
                validation_failed = True
                results[index] = ToolResult(
                    call,
                    ToolResultStatus.ERROR,
                    str(exc),
                    exc.reason,
                )

        stop_indices = [
            index
            for index, item in enumerate(validated)
            if item is not None and item.intent.action is RobotAction.STOP
        ]
        for index in stop_indices:
            item = validated[index]
            assert item is not None
            results[index] = self._execute_validated(item)

        if validation_failed:
            for index, item in enumerate(validated):
                if item is not None and results[index] is None:
                    results[index] = ToolResult(
                        item.call,
                        ToolResultStatus.DENIED,
                        "Batch rejected because another tool call failed validation.",
                        "batch_validation_failed",
                    )
            return tuple(result for result in results if result is not None)

        if stop_indices:
            for index, item in enumerate(validated):
                if item is None or results[index] is not None:
                    continue
                if item.intent.is_physical_motion:
                    results[index] = ToolResult(
                        item.call,
                        ToolResultStatus.DENIED,
                        "Action not executed because stop takes precedence in this batch.",
                        "stop_precedence",
                    )
                else:
                    results[index] = self._execute_validated(item)
            return tuple(result for result in results if result is not None)

        for index, item in enumerate(validated):
            assert item is not None
            results[index] = self._execute_validated(item)
        return tuple(result for result in results if result is not None)
