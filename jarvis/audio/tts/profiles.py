"""Semantic, user-facing voice profiles and their provider mappings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VoiceProfile:
    id: str
    display_name: str
    provider: str
    provider_voice: str
    languages: tuple[str, ...] = ("en",)
    streaming: bool = False
    provenance: str = ""


FENRIR_PROFILE = VoiceProfile(
    "fenrir",
    "Fenrir",
    "kokoro",
    "am_fenrir",
    streaming=True,
    provenance="Kokoro-82M / kokoro-onnx local voice bundle",
)
BMO_PROFILE = VoiceProfile(
    "bmo",
    "BMO",
    "piper",
    "bmo",
    streaming=False,
    provenance=(
        "Original upstream Be More Agent custom Piper model; source and "
        "voice-dataset terms remain unresolved"
    ),
)

VOICE_PROFILES: dict[str, VoiceProfile] = {
    FENRIR_PROFILE.id: FENRIR_PROFILE,
    BMO_PROFILE.id: BMO_PROFILE,
}


def resolve_voice_profile(profile_id: str | None) -> VoiceProfile:
    normalized = "fenrir" if profile_id is None else profile_id.strip().casefold()
    try:
        return VOICE_PROFILES[normalized]
    except (KeyError, AttributeError):
        allowed = ", ".join(VOICE_PROFILES)
        raise ValueError(f"Voice profile must be one of: {allowed}; got '{profile_id}'.") from None


def profile_for_selection(provider: str, voice: str) -> VoiceProfile | None:
    for profile in VOICE_PROFILES.values():
        if profile.provider == provider and profile.provider_voice == voice:
            return profile
    return None
