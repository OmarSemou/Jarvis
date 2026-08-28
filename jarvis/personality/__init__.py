"""Jarvis personality data and bounded system-prompt construction."""

from .profile import DEFAULT_JARVIS_PROFILE, PersonalityProfile
from .prompt import IMMUTABLE_SYSTEM_POLICY, build_system_prompt

__all__ = [
    "DEFAULT_JARVIS_PROFILE",
    "IMMUTABLE_SYSTEM_POLICY",
    "PersonalityProfile",
    "build_system_prompt",
]
