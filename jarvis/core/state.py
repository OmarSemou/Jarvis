"""Application state values independent of Tkinter and robot hardware."""

from __future__ import annotations

from enum import StrEnum


class ApplicationState(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"
    CAPTURING = "capturing"
    WARMUP = "warmup"


class RobotConnectionState(StrEnum):
    """High-level connection state; not a hardware control interface."""

    UNAVAILABLE = "unavailable"
    DISCONNECTED = "disconnected"
    READY = "ready"
    FAULT = "fault"
    EMERGENCY_STOPPED = "emergency_stopped"


# Compatibility name used throughout the upstream ``agent.py``.
BotStates = ApplicationState

