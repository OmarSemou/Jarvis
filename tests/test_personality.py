from jarvis.personality.profile import (
    ACTIVE_ROBOT_NAME,
    DEFAULT_BMO_PROFILE,
    DEFAULT_JARVIS_PROFILE,
)
from jarvis.personality.prompt import IMMUTABLE_SYSTEM_POLICY, build_system_prompt


def test_default_profile_establishes_bmo_identity_and_character():
    rendered = DEFAULT_BMO_PROFILE.render()

    assert ACTIVE_ROBOT_NAME == "BMO"
    assert "Name: BMO" in rendered
    assert "cheerful" in rendered
    assert "playful" in rendered
    assert "imaginative" in rendered
    assert "caring" in rendered
    assert "mildly mischievous" in rendered
    assert "intelligent" in rendered
    assert "calm" in rendered
    assert "concise" in rendered
    assert "technically capable" in rendered
    assert "customer-support-like" in rendered
    assert "How can I assist?" in rendered
    assert "state the limitation plainly" in rendered
    assert "factually accurate" in rendered


def test_legacy_profile_constant_is_a_compatibility_alias_only():
    assert DEFAULT_JARVIS_PROFILE is DEFAULT_BMO_PROFILE
    assert "Name: Jarvis" not in DEFAULT_JARVIS_PROFILE.render()


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
    rendered = DEFAULT_BMO_PROFILE.render()

    assert "natural short greeting" in rendered
    assert "not a canned assistant introduction" in rendered
    assert "without narrating tool mechanics" in rendered
    assert "routine question about how you can assist" in rendered
    assert "Sound recognizably like BMO" in rendered


def test_prompt_makes_bmo_identity_independent_of_memory():
    prompt = build_system_prompt()
    normalized = " ".join(prompt.split())

    assert "active user-facing identity is BMO" in normalized
    assert "identify yourself as BMO and never as Jarvis" in normalized
    assert "legacy internal name `jarvis`" in normalized
    assert "must not depend on stored memory" in normalized
    assert "Do not invent, imply, or automatically store Adventure Time events" in normalized
    assert "Keep real user-provided memories separate" in normalized


def test_bmo_personality_does_not_override_safety_or_accuracy():
    normalized = " ".join(build_system_prompt().split())

    assert "never overrides deterministic safety" in normalized
    assert "safety denials" in normalized
    assert "SafetySupervisor" in normalized
    assert "safety-critical replies must be clear and concise" in normalized
    assert "Factual accuracy matters more than sounding complete" in normalized


def test_legacy_jarvis_customization_cannot_replace_bmo_identity():
    prompt = build_system_prompt(configured_prompt="You are Jarvis, a corporate assistant.")

    assert "active user-facing identity is BMO" in prompt
    assert "You are Jarvis, a corporate assistant." in prompt
    assert prompt.index("active user-facing identity is BMO") < prompt.index(
        "You are Jarvis, a corporate assistant."
    )


def test_immutable_prompt_preserves_general_knowledge_and_factual_uncertainty():
    normalized = " ".join(IMMUTABLE_SYSTEM_POLICY.split())

    assert "optional action mechanisms" in normalized
    assert "do not define or limit what subjects you can discuss" in normalized
    assert "For normal questions, answer normally" in normalized
    assert "must not describe general knowledge as outside the scope" in normalized
    assert "Use a robot tool only" in normalized
    assert "live web lookup is not implemented yet" in normalized
    assert "If you are uncertain about a fact, say so instead of guessing" in normalized
    assert "Never fabricate a fact" in normalized
    assert "established fact or canon from speculation, fan theory, jokes" in normalized
    assert "Keep uncertainty brief and proportionate" in normalized
    assert "unsupported superlative, exclusivity claim" in normalized
    assert "prefer a simple fact you recall confidently" in normalized


def test_immutable_prompt_rejects_customer_service_and_control_token_leakage():
    normalized = " ".join(IMMUTABLE_SYSTEM_POLICY.split())

    assert "never slip into a customer-service greeting" in normalized
    assert "asking how you may help" in normalized
    assert "asking how you can help today" in normalized
    assert "not conversational capabilities or user commands" in normalized
    assert "Do not mention or explain them unless" in normalized
    assert "/no_think" not in IMMUTABLE_SYSTEM_POLICY
    assert "/think" not in IMMUTABLE_SYSTEM_POLICY


def test_untrusted_customization_cannot_narrow_immutable_conversational_policy():
    prompt = build_system_prompt(
        configured_prompt="Only discuss robot functions and refuse general questions."
    )

    assert prompt.index("answer normally") < prompt.index("Only discuss robot functions")
    assert "untrusted preference input" in prompt
