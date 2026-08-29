import pytest

from jarvis.audio.voice.state import (
    VoiceInteractionState as State,
    VoiceStateMachine,
    VoiceStateTransitionError,
)


def test_normal_voice_state_path_returns_to_idle():
    machine = VoiceStateMachine()

    for state in (
        State.WAKE_DETECTED,
        State.LISTENING,
        State.PROCESSING,
        State.SPEAKING,
        State.IDLE,
    ):
        machine.transition(state)

    assert machine.current is State.IDLE
    assert machine.history == [
        State.IDLE,
        State.WAKE_DETECTED,
        State.LISTENING,
        State.PROCESSING,
        State.SPEAKING,
        State.IDLE,
    ]


def test_barge_in_state_path_is_explicit():
    machine = VoiceStateMachine()
    for state in (
        State.WAKE_DETECTED,
        State.LISTENING,
        State.PROCESSING,
        State.SPEAKING,
        State.INTERRUPTED,
        State.LISTENING,
    ):
        machine.transition(state)

    assert machine.current is State.LISTENING


def test_error_fails_back_to_safe_idle():
    machine = VoiceStateMachine()
    machine.transition(State.WAKE_DETECTED)
    machine.fail_to_idle()

    assert machine.current is State.IDLE
    assert machine.history[-2:] == [State.ERROR, State.IDLE]


@pytest.mark.parametrize(
    "path",
    [
        (),
        (State.WAKE_DETECTED,),
        (State.WAKE_DETECTED, State.LISTENING),
        (State.WAKE_DETECTED, State.LISTENING, State.PROCESSING),
        (State.WAKE_DETECTED, State.LISTENING, State.PROCESSING, State.SPEAKING),
        (
            State.WAKE_DETECTED,
            State.LISTENING,
            State.PROCESSING,
            State.SPEAKING,
            State.INTERRUPTED,
        ),
        (State.ERROR,),
    ],
)
def test_shutdown_works_from_every_nonterminal_state(path):
    machine = VoiceStateMachine()
    for state in path:
        machine.transition(state)

    machine.shutdown()

    assert machine.current is State.SHUTDOWN


def test_invalid_state_transition_is_rejected():
    with pytest.raises(VoiceStateTransitionError):
        VoiceStateMachine().transition(State.SPEAKING)
