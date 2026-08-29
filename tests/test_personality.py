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
    assert "routine question about how you can assist" in rendered
    assert "mild dry humor" in rendered


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
