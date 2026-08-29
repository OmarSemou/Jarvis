from types import SimpleNamespace

import pytest

from jarvis.audio.tts.base import SpeechSynthesisResult, SynthesizedAudio
from jarvis.audio.tts.profiles import (
    BMO_PROFILE,
    FENRIR_PROFILE,
    resolve_voice_profile,
)
from jarvis.audio.tts.playback import PlaybackResult
from jarvis.audio.tts.service import TTSService
from jarvis.cli import build_parser, create_tts_providers, run_chat
from jarvis.core.config import ConfigValidationError, parse_config
from jarvis.core.paths import JarvisPaths
from jarvis.core.preflight import CheckStatus, run_preflight
from jarvis.llm.base import LLMResponse


class Provider:
    def __init__(self, name, voices):
        self.name = name
        self.available_voices = tuple(voices)
        self.calls = []

    def readiness_error(self, _voice):
        return None

    def synthesize(self, text, *, voice, speed, language):
        self.calls.append((text, voice, speed, language))
        return SpeechSynthesisResult(
            True,
            self.name,
            voice,
            0.01,
            SynthesizedAudio(b"\x00\x00" * 20, 1_000),
        )


class Playback:
    def __init__(self):
        self.audio = []
        self.stopped = 0

    def play(self, audio):
        self.audio.append(audio)
        return PlaybackResult(True, "Mock speakers")

    def stop(self):
        self.stopped += 1


def _service(*, enabled=True):
    playback = Playback()
    kokoro = Provider("kokoro", ("am_fenrir",))
    piper = Provider("piper", ("en_US-joe-medium", "en_US-john-medium", "bmo"))
    service = TTSService(
        {"kokoro": kokoro, "piper": piper},
        playback,
        enabled=enabled,
        profile="fenrir",
    )
    return service, kokoro, piper, playback


def test_profiles_resolve_to_provider_neutral_mappings():
    assert resolve_voice_profile("FENRIR") == FENRIR_PROFILE
    assert resolve_voice_profile("bmo") == BMO_PROFILE
    assert FENRIR_PROFILE.provider == "kokoro"
    assert FENRIR_PROFILE.provider_voice == "am_fenrir"
    assert BMO_PROFILE.provider == "piper"
    assert BMO_PROFILE.provider_voice == "bmo"

    with pytest.raises(ValueError, match="one of: fenrir, bmo"):
        resolve_voice_profile("semaine")


def test_profile_configuration_is_optional_and_validated():
    assert parse_config({}).config.tts_profile is None
    assert parse_config({"tts_profile": "BMO"}).config.tts_profile == "bmo"
    assert parse_config(
        {"tts_profile": "bmo", "tts_provider": "piper", "tts_voice": "bmo"}
    ).config.tts_profile == "bmo"
    with pytest.raises(ConfigValidationError, match="'tts_profile' must be one of"):
        parse_config({"tts_profile": "voice-clone"})


def test_piper_provider_accepts_the_explicit_optional_bmo_voice(tmp_path):
    paths = JarvisPaths.from_repository_root(tmp_path.resolve())
    (tmp_path / "voices").mkdir()
    (tmp_path / "voices" / "bmo-custom.onnx").write_bytes(b"model")
    (tmp_path / "voices" / "bmo-custom.onnx.json").write_text("{}", encoding="utf-8")

    providers = create_tts_providers(SimpleNamespace(), paths)

    assert "bmo" in providers["piper"].available_voices
    assert providers["piper"].settings.voice_files["bmo"] == paths.legacy_bmo_voice_files


def test_preflight_reports_bmo_assets_without_loading_them(tmp_path):
    paths = JarvisPaths.from_repository_root(tmp_path.resolve())
    (tmp_path / "voices").mkdir()
    paths.legacy_bmo_voice_files[0].write_bytes(b"model")
    paths.legacy_bmo_voice_files[1].write_text("{}", encoding="utf-8")

    report = run_preflight(
        paths,
        which=lambda _name: None,
        module_available=lambda _name: False,
        version_info=(3, 13, 15),
    )

    checks = {check.name: check for check in report.checks}
    assert checks["Piper BMO model"].status is CheckStatus.AVAILABLE
    assert checks["Piper BMO config"].status is CheckStatus.AVAILABLE


def test_profile_switching_routes_markdown_safely_and_stops_playback():
    service, kokoro, piper, playback = _service()

    service.speak("Use **Power Stroke**.")
    assert kokoro.calls[-1][0] == "Use Power Stroke."
    assert service.profile_id == "fenrir"

    service.set_profile("bmo")
    assert service.profile_id == "bmo"
    assert service.provider == "piper"
    assert service.voice == "bmo"
    assert playback.stopped >= 1

    service.speak("The *same* text.")
    assert piper.calls[-1][0] == "The same text."

    service.set_profile("fenrir")
    assert service.profile_id == "fenrir"
    assert service.provider == "kokoro"
    assert service.voice == "am_fenrir"
    assert playback.stopped >= 2


def test_missing_original_bmo_model_fails_without_fallback():
    service, _, _, _ = _service()
    service.providers["piper"].available_voices = ("en_US-joe-medium",)

    with pytest.raises(ValueError, match="original BMO voice model is unavailable"):
        service.set_profile("bmo")
    assert service.profile_id == "fenrir"


def test_voice_parser_accepts_semantic_profiles():
    args = build_parser().parse_args(["voice", "--voice", "bmo"])
    assert args.voice_profile == "bmo"


def test_interactive_profile_switching_is_local_and_does_not_call_llm():
    service, _, _, _ = _service(enabled=False)

    class ConversationProvider:
        name = "mock"
        endpoint = "memory://conversation"
        requests = []

        def generate(self, request, *, cancellation=None):
            self.requests.append(request)
            return LLMResponse("unexpected", "qwen3:8b")

        def close(self):
            pass

    provider = ConversationProvider()
    from jarvis.core.conversation import ConversationService, ConversationSettings

    conversation = ConversationService(
        provider,
        ConversationSettings(model="qwen3:8b"),
        system_prompt="Test",
    )
    output = []
    values = iter(["/voice bmo", "/voice status", "/voice fenrir", "/quit"])

    result = run_chat(
        conversation,
        tts_runtime=service,
        input_fn=lambda _prompt: next(values),
        output_fn=output.append,
    )

    assert result == 0
    assert provider.requests == []
    assert service.profile_id == "fenrir"
    assert any("BMO / Piper bmo" in line for line in output)
    assert any("Fenrir / Kokoro am_fenrir" in line for line in output)
