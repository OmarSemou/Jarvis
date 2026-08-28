"""Provider-neutral structured tool definitions, calls, and results."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping, Protocol, runtime_checkable


class ToolParameterType(StrEnum):
    STRING = "string"


@dataclass(frozen=True, slots=True)
class ToolParameter:
    name: str
    description: str
    kind: ToolParameterType
    required: bool = True
    allowed_values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier() or self.name.startswith("_"):
            raise ValueError("tool parameter name must be a public identifier")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("tool parameter description must be non-empty")
        if not isinstance(self.kind, ToolParameterType):
            raise TypeError("tool parameter kind must be a ToolParameterType")
        if self.allowed_values and self.kind is not ToolParameterType.STRING:
            raise ValueError("allowed_values are supported only for string parameters")
        if any(not isinstance(value, str) or not value for value in self.allowed_values):
            raise ValueError("allowed_values must contain only non-empty strings")
        if len(self.allowed_values) != len(set(self.allowed_values)):
            raise ValueError("allowed_values must be unique")


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: tuple[ToolParameter, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier() or self.name.startswith("_"):
            raise ValueError("tool name must be a public identifier")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("tool description must be non-empty")
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("tool parameter names must be unique")


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments: Mapping[str, object]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name.strip()
            or not self.name.strip().isidentifier()
            or self.name.strip().startswith("_")
        ):
            raise ValueError("tool call name must be a public identifier")
        if not isinstance(self.arguments, Mapping):
            raise TypeError("tool call arguments must be a mapping")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))


class ToolResultStatus(StrEnum):
    SUCCESS = "success"
    DENIED = "denied"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ToolResult:
    call: ToolCall
    status: ToolResultStatus
    message: str
    denial_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.call, ToolCall):
            raise TypeError("tool result call must be a ToolCall")
        if not isinstance(self.status, ToolResultStatus):
            raise TypeError("tool result status must be a ToolResultStatus")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("tool result message must be non-empty")
        if self.denial_reason is not None and (
            not isinstance(self.denial_reason, str) or not self.denial_reason.strip()
        ):
            raise ValueError("denial_reason must be null or a non-empty string")
        if self.status is ToolResultStatus.SUCCESS and self.denial_reason is not None:
            raise ValueError("successful tool results cannot contain a denial reason")

    @property
    def success(self) -> bool:
        return self.status is ToolResultStatus.SUCCESS

    def model_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "tool": self.call.name,
            "status": self.status.value,
            "success": self.success,
            "message": self.message,
        }
        if self.denial_reason is not None:
            payload["denial_reason"] = self.denial_reason
        return payload


@runtime_checkable
class ToolExecutor(Protocol):
    @property
    def definitions(self) -> tuple[ToolDefinition, ...]: ...

    def execute(self, calls: tuple[ToolCall, ...]) -> tuple[ToolResult, ...]: ...
