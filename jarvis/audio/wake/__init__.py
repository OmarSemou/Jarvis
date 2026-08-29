"""Provider-neutral local wake-word detection."""

from .base import WakeWordDetection, WakeWordFailure, WakeWordProvider

__all__ = ["WakeWordDetection", "WakeWordFailure", "WakeWordProvider"]
