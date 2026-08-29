"""BMO personality data and bounded system-prompt construction."""

from .profile import (
    ACTIVE_ROBOT_NAME,
    DEFAULT_BMO_PROFILE,
    DEFAULT_JARVIS_PROFILE,
    PersonalityProfile,
)
from .prompt import IMMUTABLE_SYSTEM_POLICY, build_system_prompt

__all__ = [
    "ACTIVE_ROBOT_NAME",
    "DEFAULT_BMO_PROFILE",
    "DEFAULT_JARVIS_PROFILE",
    "IMMUTABLE_SYSTEM_POLICY",
    "PersonalityProfile",
    "build_system_prompt",
]
