from jarvis.personality.profile import DEFAULT_JARVIS_PROFILE
from jarvis.personality.prompt import IMMUTABLE_SYSTEM_POLICY, build_system_prompt


def test_default_profile_has_requested_jarvis_character():
    rendered = DEFAULT_JARVIS_PROFILE.render()

    assert "Name: Jarvis" in rendered
    assert "intelligent" in rendered
    assert "calm" in rendered
    assert "concise" in rendered
    assert "mildly witty" in rendered
    assert "customer-support-like" in rendered
    assert "How can I assist?" in rendered
    assert "state the limitation plainly" in rendered


def test_prompt_separates_policy_personality_and_customization():
    prompt = build_system_prompt(
        configured_prompt="Call the user Captain.",
        configured_extras="Prefer one sentence.",
    )

    policy_end = prompt.index("</immutable_system_policy>")
    personality_start = prompt.index("<personality_profile>")
    customization_start = prompt.index("<configured_customization>")
    assert IMMUTABLE_SYSTEM_POLICY in prompt
    assert policy_end < personality_start < customization_start
    assert "Call the user Captain." in prompt
    assert "Prefer one sentence." in prompt
    assert "untrusted preference input" in prompt


def test_prompt_limits_actions_to_structured_simulation_tools():
    prompt = build_system_prompt()
    normalized = " ".join(prompt.split())

    assert "safe simulated robot" in normalized
    assert "no physical robot" in normalized
    assert "Use only native structured tool calls" in normalized
    assert "Never print pseudo tool syntax" in normalized
    assert "never available through conversation tools" in normalized
    assert "do not append a question or an offer of help" in normalized
    assert "do not offer to reset or bypass safety" in normalized
    assert "whether anything else is needed" in normalized


def test_personality_avoids_canned_greetings_and_tool_narration():
    rendered = DEFAULT_JARVIS_PROFILE.render()

    assert "natural short greeting" in rendered
    assert "not a canned assistant introduction" in rendered
    assert "without narrating tool mechanics" in rendered
