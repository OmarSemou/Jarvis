from jarvis.core.state import ApplicationState, BotStates, RobotConnectionState


def test_legacy_state_name_is_alias():
    assert BotStates is ApplicationState
    assert BotStates.IDLE == "idle"
    assert BotStates.SPEAKING.upper() == "SPEAKING"


def test_application_states_cover_legacy_gui_states():
    assert {state.value for state in ApplicationState} == {
        "idle",
        "listening",
        "thinking",
        "speaking",
        "error",
        "capturing",
        "warmup",
    }


def test_robot_connection_state_is_hardware_independent():
    assert RobotConnectionState.READY.value == "ready"
    assert RobotConnectionState.EMERGENCY_STOPPED.value == "emergency_stopped"

