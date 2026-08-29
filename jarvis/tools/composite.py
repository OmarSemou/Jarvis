"""Explicit composition of independent allowlisted tool executors."""

from __future__ import annotations

from .types import ToolCall, ToolDefinition, ToolExecutor, ToolResult, ToolResultStatus


class CompositeToolExecutor:
    """Join registries without reflection; each call is routed by exact name."""

    def __init__(self, *executors: ToolExecutor) -> None:
        self._executors = tuple(executors)
        definitions: list[ToolDefinition] = []
        names: set[str] = set()
        for executor in self._executors:
            for definition in executor.definitions:
                if definition.name in names:
                    raise ValueError(f"duplicate tool definition: {definition.name}")
                names.add(definition.name)
                definitions.append(definition)
        self._definitions = tuple(definitions)
        self._by_name = {
            definition.name: executor
            for executor in self._executors
            for definition in executor.definitions
        }

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return self._definitions

    def set_user_text(self, text: str) -> None:
        for executor in self._executors:
            setter = getattr(executor, "set_user_text", None)
            if callable(setter):
                setter(text)

    def execute_explicit(self, text: str):
        """Delegate deterministic explicit-intent handling to the matching child."""

        for executor in self._executors:
            handler = getattr(executor, "execute_explicit", None)
            if callable(handler):
                outcome = handler(text)
                if outcome is not None:
                    return outcome
        return None

    def execute(self, calls: tuple[ToolCall, ...]) -> tuple[ToolResult, ...]:
        indexed: dict[int, tuple[ToolExecutor, list[tuple[int, ToolCall]]]] = {}
        results: list[ToolResult | None] = [None] * len(calls)
        for index, call in enumerate(calls):
            executor = self._by_name.get(call.name)
            if executor is None:
                results[index] = ToolResult(call, ToolResultStatus.ERROR, f"Tool '{call.name}' is not registered.", "unknown_tool")
            else:
                bucket = indexed.get(id(executor))
                if bucket is None:
                    bucket = (executor, [])
                    indexed[id(executor)] = bucket
                bucket[1].append((index, call))
        for executor, entries in indexed.values():
            try:
                batch = tuple(executor.execute(tuple(call for _, call in entries)))
            except Exception:
                batch = ()
            if len(batch) != len(entries) or any(result.call != call for result, (_, call) in zip(batch, entries, strict=True)):
                for index, call in entries:
                    results[index] = ToolResult(call, ToolResultStatus.ERROR, "Tool executor returned an invalid result.", "tool_execution_failure")
            else:
                for (index, _), result in zip(entries, batch, strict=True):
                    results[index] = result
        return tuple(result for result in results if result is not None)
