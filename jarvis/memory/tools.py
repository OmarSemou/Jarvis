"""Explicit model-facing memory tools; no SQL or arbitrary method dispatch."""

from __future__ import annotations

from .models import MemoryCandidate, MemoryCategory, MemoryConfidence, MemorySource
from .intent import recognize_explicit_memory_request
from .service import MemoryService
from jarvis.tools.types import ToolCall, ToolDefinition, ToolParameter, ToolParameterType, ToolResult, ToolResultStatus


MEMORY_TOOL_DEFINITIONS = (
    ToolDefinition(
        "remember_memory",
        "Store one durable user fact only when the user clearly asks you to remember it.",
        (
            ToolParameter("category", "Memory category.", ToolParameterType.STRING, allowed_values=tuple(item.value for item in MemoryCategory)),
            ToolParameter("key", "Short stable name for the fact.", ToolParameterType.STRING),
            ToolParameter("value", "The durable user fact, without secrets or transcript text.", ToolParameterType.STRING),
            ToolParameter("summary", "Optional concise summary.", ToolParameterType.STRING, required=False),
        ),
    ),
    ToolDefinition(
        "forget_memory",
        "Forget one memory by its numeric id, only when the user clearly asks.",
        (ToolParameter("memory_id", "Numeric memory id.", ToolParameterType.INTEGER),),
    ),
)


class MemoryToolRegistry:
    def __init__(self) -> None:
        self._by_name = {definition.name: definition for definition in MEMORY_TOOL_DEFINITIONS}

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return MEMORY_TOOL_DEFINITIONS

    def validate(self, call: ToolCall) -> None:
        definition = self._by_name.get(call.name)
        if definition is None:
            raise ValueError(f"Tool '{call.name}' is not registered.")
        expected = {parameter.name for parameter in definition.parameters}
        supplied = set(call.arguments)
        extra = supplied - expected
        missing = {parameter.name for parameter in definition.parameters if parameter.required and parameter.name not in supplied}
        if extra:
            raise ValueError("unexpected argument(s): " + ", ".join(sorted(extra)))
        if missing:
            raise ValueError("missing argument(s): " + ", ".join(sorted(missing)))
        for parameter in definition.parameters:
            if parameter.name not in call.arguments:
                continue
            value = call.arguments[parameter.name]
            if parameter.kind is ToolParameterType.STRING and not isinstance(value, str):
                raise ValueError(f"{parameter.name} must be a string")
            if parameter.kind is ToolParameterType.INTEGER and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise ValueError(f"{parameter.name} must be an integer")
        if call.name == "remember_memory":
            try:
                MemoryCategory(str(call.arguments["category"]).casefold())
            except ValueError:
                raise ValueError("invalid memory category") from None


class MemoryToolExecutor:
    def __init__(self, service: MemoryService) -> None:
        self.service = service
        self.registry = MemoryToolRegistry()
        self._intent_kind: str | None = None

    def set_user_text(self, text: str) -> None:
        """Supply current-turn intent without persisting the transcript."""

        request = recognize_explicit_memory_request(text)
        self._intent_kind = request.kind if request is not None else "none"

    def execute_explicit(self, text: str) -> tuple[ToolCall, ToolResult] | None:
        """Persist a strong explicit request before asking the model to reply.

        This is the reliability seam for phrases such as "remember that ...".
        The model still receives a normal provider-neutral tool interaction, but
        SQLite success no longer depends on Qwen choosing a tool voluntarily.
        """

        request = recognize_explicit_memory_request(text)
        if request is None:
            return None
        self.set_user_text(text)
        if request.kind == "remember":
            # Let the model perform structured extraction for explicit forms
            # outside the conservative deterministic patterns.  The common
            # high-confidence forms are handled above this boundary; unknown
            # prose is never silently stored by this executor.
            if request.candidate is None:
                return None
            call = ToolCall(
                "remember_memory",
                {
                    "category": request.candidate.category.value,
                    "key": request.candidate.key,
                    "value": request.candidate.value,
                },
            )
            result = self.execute((call,))[0]
            return call, result

        matches = self.service.search(request.query)
        if len(matches) == 1:
            call = ToolCall("forget_memory", {"memory_id": matches[0].id})
            result = self.execute((call,))[0]
            return call, result
        reason = "not_found" if not matches else "ambiguous"
        call = ToolCall("forget_memory", {"memory_id": 0})
        message = (
            "I found no active persistent memory matching that request; nothing was forgotten."
            if reason == "not_found"
            else "I found multiple matching memories; nothing was forgotten."
        )
        result = ToolResult(call, ToolResultStatus.DENIED, message, reason)
        self.service._log(f"rejected reason={reason}")
        return call, result

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return self.registry.definitions

    def execute(self, calls: tuple[ToolCall, ...]) -> tuple[ToolResult, ...]:
        calls = tuple(calls)
        validation_errors: dict[int, str] = {}
        for index, call in enumerate(calls):
            try:
                self.registry.validate(call)
            except (TypeError, ValueError) as exc:
                validation_errors[index] = str(exc)
        if validation_errors:
            return tuple(
                ToolResult(
                    call,
                    ToolResultStatus.ERROR if index in validation_errors else ToolResultStatus.DENIED,
                    (
                        f"Memory tool rejected safely: {validation_errors[index]}"
                        if index in validation_errors
                        else "Memory batch rejected because another call failed validation."
                    ),
                    "invalid_arguments" if index in validation_errors else "batch_validation_failed",
                )
                for index, call in enumerate(calls)
            )

        results: list[ToolResult] = []
        for call in calls:
            try:
                if self._intent_kind in {"none", "mixed"}:
                    results.append(
                        ToolResult(call, ToolResultStatus.DENIED, "Memory changes require a clear user memory request.", "explicit_intent_required")
                    )
                    continue
                expected_kind = "remember" if call.name == "remember_memory" else "forget"
                if self._intent_kind is not None and self._intent_kind != expected_kind:
                    results.append(
                        ToolResult(call, ToolResultStatus.DENIED, "Memory tool does not match the user's explicit memory intent.", "intent_mismatch")
                    )
                    continue
                if call.name == "remember_memory":
                    candidate = MemoryCandidate(
                        category=str(call.arguments["category"]),
                        key=str(call.arguments["key"]),
                        value=str(call.arguments["value"]),
                        summary=str(call.arguments.get("summary", "")),
                        source=(
                            MemorySource.EXPLICIT_USER
                            if self._intent_kind == "remember"
                            else MemorySource.LLM_CANDIDATE
                        ),
                        confidence=MemoryConfidence.HIGH,
                    )
                    outcome = self.service.remember(
                        candidate,
                        explicit=self._intent_kind == "remember",
                    )
                else:
                    raw_id = call.arguments["memory_id"]
                    if isinstance(raw_id, bool) or not isinstance(raw_id, int):
                        raise ValueError("memory_id must be an integer")
                    outcome = self.service.forget(raw_id)
                status = ToolResultStatus.SUCCESS if outcome.success else ToolResultStatus.DENIED
                results.append(ToolResult(call, status, outcome.message, None if outcome.success else outcome.reason))
            except (TypeError, ValueError) as exc:
                results.append(ToolResult(call, ToolResultStatus.ERROR, f"Memory tool rejected safely: {exc}", "invalid_arguments"))
        return tuple(results)
