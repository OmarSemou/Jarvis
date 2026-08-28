"""Interfaces located downstream of deterministic safety approval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .intents import RobotIntent


@dataclass(frozen=True, slots=True)
class MovementLease:
    """Short, renewable authority window for future physical movement."""

    issued_at_monotonic: float
    expires_at_monotonic: float

    def __post_init__(self) -> None:
        if self.expires_at_monotonic <= self.issued_at_monotonic:
            raise ValueError("movement lease must expire after it is issued")

    def is_valid(self, now_monotonic: float) -> bool:
        return self.issued_at_monotonic <= now_monotonic < self.expires_at_monotonic


@dataclass(frozen=True, slots=True)
class ApprovedRobotIntent:
    """An intent that has passed the deterministic safety supervisor."""

    intent: RobotIntent
    lease: MovementLease | None


@runtime_checkable
class RobotController(Protocol):
    """Future controller boundary; it accepts safety-approved intents only."""

    def execute(self, approved: ApprovedRobotIntent) -> None:
        """Execute an approved meaning-level action."""

    def stop(self) -> None:
        """Request an immediate deterministic stop."""

