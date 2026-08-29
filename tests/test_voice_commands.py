from pathlib import Path

import pytest

from jarvis.audio.voice.commands import (
    LocalVoiceCommand,
    LocalVoiceCommandRouter,
    is_no_speech_transcript,
    is_wake_only_transcript,
)
from jarvis.robot.controller import create_simulated_controller
from jarvis.robot.intents import RobotAction, RobotIntent
from jarvis.robot.safety import SafetyAuthority
from jarvis.robot.simulator import MotionState
from jarvis.integrations.voice_stop import SafeLocalVoiceCommandExecutor


@pytest.mark.parametrize(
    "text",
    (
        "stop",
        "stop now",
        "please stop",
        "jarvis stop",
        "hey jarvis stop",
        "  HEY JARVIS, STOP!  ",
        "Hey Jarvis, please stop now.",
        "bmo stop",
        "Hey BMO, please stop now.",
    ),
)
def test_anchored_local_stop_grammar_accepts_only_explicit_stop_utterances(text):
    assert LocalVoiceCommandRouter.match(text) is LocalVoiceCommand.STOP


@pytest.mark.parametrize(
    "text",
    (
        "don't stop",
        "what does stop mean?",
        "tell me about stop signs",
        "why did you stop?",
        "stop signs",
        "move forward",
        "follow me",
        "estop reset",
    ),
)
def test_local_stop_grammar_rejects_negation_questions_and_other_actions(text):
    assert LocalVoiceCommandRouter.match(text) is None


@pytest.mark.parametrize(
    "text",
    ("", "   ", "[BLANK_AUDIO]", "[ Silence ]", "(silence)", "[no speech]"),
)
def test_known_whisper_no_speech_outputs_are_exactly_rejected(text):
    assert is_no_speech_transcript(text)


def test_plain_word_silence_is_not_mistaken_for_a_whisper_marker():
    assert not is_no_speech_transcript("silence")


@pytest.mark.parametrize(
    "text",
    ("Jarvis", "Hey Jarvis", "  HEY, JARVIS! ", "BMO", "Hey BMO"),
)
def test_wake_only_filter_matches_only_the_activation_phrase(text):
    assert is_wake_only_transcript(text)


@pytest.mark.parametrize(
    "text",
    ("Hey Jarvis stop", "Hey Jarvis hello", "Hey BMO stop", "hello"),
)
def test_wake_only_filter_preserves_commands_and_conversation(text):
    assert not is_wake_only_transcript(text)


def test_local_stop_routes_through_safe_controller_and_stops_simulator():
    controller = create_simulated_controller()
    assert controller.execute_intent(RobotIntent(RobotAction.MOVE_FORWARD)).success
    executor = SafeLocalVoiceCommandExecutor(controller)

    result = executor.execute(LocalVoiceCommand.STOP)

    assert result.success
    assert controller.state.motion is MotionState.STOPPED
    assert controller.state.following is False
    assert controller.state.event_log[-1] == "motion=stopped follow=inactive"


def test_local_stop_never_clears_latched_emergency_stop():
    controller = create_simulated_controller()
    controller.latch_emergency_stop(authority=SafetyAuthority.LOCAL_OPERATOR)

    result = SafeLocalVoiceCommandExecutor(controller).execute(LocalVoiceCommand.STOP)

    assert result.success
    assert controller.state.emergency_stop_latched is True
    assert controller.safety.emergency_stop_latched is True


def test_local_router_and_executor_expose_no_low_level_or_general_motion_controls():
    assert tuple(LocalVoiceCommand) == (LocalVoiceCommand.STOP,)
    source = (
        Path(__file__).parents[1] / "jarvis" / "audio" / "voice" / "commands.py"
    ).read_text(encoding="utf-8").casefold()
    executor_source = (
        Path(__file__).parents[1] / "jarvis" / "integrations" / "voice_stop.py"
    ).read_text(encoding="utf-8").casefold()
    for forbidden in ("pwm", "gpio", "servo", "voltage", "motor speed", "shell"):
        assert forbidden not in source + executor_source
