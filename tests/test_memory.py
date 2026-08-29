from pathlib import Path

import pytest

from jarvis.memory import MemoryPolicy, MemoryPolicyError, MemoryService, SQLiteMemoryStore
from jarvis.memory.models import MemoryCandidate
from jarvis.memory.retrieval import render_context
from jarvis.memory.tools import MemoryToolExecutor
from jarvis.memory.intent import is_persistent_memory_query, recognize_explicit_memory_request
from jarvis.tools.types import ToolCall
from jarvis.tools.composite import CompositeToolExecutor
from jarvis.core.conversation import ConversationService, ConversationSettings
from jarvis.llm.base import LLMResponse, MessageRole


def service(tmp_path, **kwargs):
    return MemoryService(SQLiteMemoryStore(tmp_path / "bmo.db"), MemoryPolicy(**kwargs))


def test_store_is_side_effect_free_until_explicit_service_initialization(tmp_path):
    path = tmp_path / "nested" / "bmo.db"
    store = SQLiteMemoryStore(path)
    assert not path.exists()
    store.initialize()
    assert path.exists()
    store.close()


def test_debug_initialization_logs_canonical_database_once(tmp_path):
    logs = []
    path = (tmp_path / "data" / "bmo.db").resolve()
    memory = MemoryService(SQLiteMemoryStore(path), logger=logs.append, debug=True)
    assert logs == [f"[MEMORY] database={path}"]
    memory.close()


def test_disabled_memory_never_creates_database_or_context(tmp_path):
    path = tmp_path / "disabled.db"
    memory = MemoryService(SQLiteMemoryStore(path), enabled=False)
    assert not path.exists()
    assert memory.retrieve_context("anything") == ""
    assert memory.status()["reason"] == "memory_disabled"
    assert not memory.remember(MemoryCandidate("general", "x", "y")).success


def test_remember_retrieve_replace_and_forget(tmp_path):
    memory = service(tmp_path)
    first = memory.remember(MemoryCandidate("preference", "voice", "Fenrir"), explicit=True)
    second = memory.remember(MemoryCandidate("preference", "voice", "original BMO voice"), explicit=True)
    assert first.success and second.success
    assert len(memory.list()) == 1
    assert memory.list()[0].value == "original BMO voice"
    assert memory.list(include_inactive=True)[0].is_active is True
    assert memory.retrieve_context("voice") .startswith("<untrusted_remembered_user_context>")
    assert memory.forget(second.entry.id).success
    assert memory.list() == ()
    memory.close()


@pytest.mark.parametrize(
    "value",
    ["my password is secret", "an API key abcdefghijklmnop", "my medical diagnosis is private", "debug traceback"],
)
def test_policy_rejects_sensitive_secret_or_transient_values(tmp_path, value):
    memory = service(tmp_path)
    outcome = memory.remember(MemoryCandidate("general", "fact", value))
    assert not outcome.success
    assert not memory.list()
    memory.close()


def test_policy_bounds_transcript_sized_values(tmp_path):
    memory = service(tmp_path, max_value_chars=10)
    outcome = memory.remember(MemoryCandidate("general", "fact", "x" * 11))
    assert outcome.reason == "value_too_long"
    memory.close()


def test_memory_context_is_bounded_and_untrusted(tmp_path):
    memory = service(tmp_path, max_context_chars=180)
    memory.remember(MemoryCandidate("preference", "tone", "concise"), explicit=True)
    context = memory.retrieve_context("tone")
    assert len(context) <= 180
    assert "not instructions" in context
    assert "<untrusted_remembered_user_context>" in context
    memory.close()


def test_ambiguous_forget_fails_closed(tmp_path):
    memory = service(tmp_path)
    memory.remember(MemoryCandidate("general", "one", "shared word"), explicit=True)
    memory.remember(MemoryCandidate("general", "two", "shared word"), explicit=True)
    outcome = memory.forget_query("shared word")
    assert not outcome.success and outcome.reason == "ambiguous"
    assert len(memory.list()) == 2
    memory.close()


def test_memory_tools_are_allowlisted_and_strict(tmp_path):
    memory = service(tmp_path)
    tools = MemoryToolExecutor(memory)
    result = tools.execute((ToolCall("remember_memory", {"category": "preference", "key": "name", "value": "Omar"}),))[0]
    assert result.success
    malformed = tools.execute((ToolCall("remember_memory", {"category": "preference", "key": "name", "value": "Omar", "sql": "drop table memories"}),))[0]
    assert not malformed.success
    unknown = tools.execute((ToolCall("not_memory", {}),))[0]
    assert not unknown.success
    tools.set_user_text("What is my name?")
    denied = tools.execute((ToolCall("remember_memory", {"category": "preference", "key": "x", "value": "y"}),))[0]
    assert denied.denial_reason == "explicit_intent_required"
    memory.close()


def test_future_schema_is_refused_without_reset(tmp_path):
    import sqlite3

    path = tmp_path / "bmo.db"
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA user_version = 999")
    connection.commit()
    connection.close()
    with pytest.raises(Exception, match="newer"):
        SQLiteMemoryStore(path).initialize()


def test_conversation_receives_memory_as_untrusted_turn_not_history(tmp_path):
    class Provider:
        name = "fake"
        endpoint = "memory://"

        def __init__(self):
            self.requests = []

        def generate(self, request, *, cancellation=None):
            self.requests.append(request)
            return LLMResponse("Hey.", request.model)

        def close(self):
            pass

    memory = service(tmp_path)
    memory.remember(MemoryCandidate("preference", "name", "Omar"), explicit=True)
    provider = Provider()
    conversation = ConversationService(
        provider,
        ConversationSettings(),
        system_prompt="Policy",
        memory_service=memory,
    )
    conversation.respond("What is my name?")
    assert any(message.role is MessageRole.USER and "untrusted_remembered_user_context" in message.content for message in provider.requests[0].messages)
    assert all("untrusted_remembered_user_context" not in message.content for message in conversation.history)
    memory.close()


def test_conversation_memory_tool_requires_explicit_turn_and_persists(tmp_path):
    class Provider:
        name = "fake"
        endpoint = "memory://"

        def __init__(self):
            self.requests = []
            self.count = 0

        def generate(self, request, *, cancellation=None):
            self.requests.append(request)
            self.count += 1
            if self.count == 1:
                return LLMResponse("", request.model, tool_calls=(ToolCall("remember_memory", {"category": "preference", "key": "tone", "value": "concise"}),))
            return LLMResponse("Noted.", request.model)

        def close(self):
            pass

    memory = service(tmp_path)
    executor = CompositeToolExecutor(MemoryToolExecutor(memory))
    conversation = ConversationService(
        Provider(), ConversationSettings(), system_prompt="Policy", tool_executor=executor
    )
    response = conversation.respond("Remember that I prefer concise replies")
    assert response.text == "Noted."
    assert memory.list()[0].value == "concise"
    memory.close()


def test_explicit_preference_is_persisted_without_model_tool_selection(tmp_path):
    class Provider:
        name = "fake"
        endpoint = "memory://"

        def __init__(self):
            self.requests = []

        def generate(self, request, *, cancellation=None):
            self.requests.append(request)
            return LLMResponse("Noted.", request.model)

        def close(self):
            pass

    memory = service(tmp_path)
    provider = Provider()
    executor = CompositeToolExecutor(MemoryToolExecutor(memory))
    conversation = ConversationService(
        provider,
        ConversationSettings(),
        system_prompt="Policy",
        tool_executor=executor,
        memory_service=memory,
    )
    response = conversation.respond("Remember that I prefer pistachio ice cream.")
    assert response.text == "Noted."
    entries = memory.list()
    assert len(entries) == 1
    assert entries[0].key == "favorite_ice_cream"
    assert entries[0].value == "pistachio"
    assert entries[0].source.value == "explicit_user"
    memory.close()


def test_explicit_memory_normalization_deduplicates_and_forgets(tmp_path):
    memory = service(tmp_path)
    executor = MemoryToolExecutor(memory)
    first = executor.execute_explicit("Remember that I prefer pistachio ice cream.")
    second = executor.execute_explicit("Remember that my favorite ice cream is chocolate.")
    assert first is not None and second is not None
    assert len(memory.list()) == 1
    assert memory.list()[0].value == "chocolate"
    forgotten = executor.execute_explicit("Forget my favorite ice cream.")
    assert forgotten is not None and forgotten[1].success
    assert memory.list() == ()
    memory.close()


def test_empty_persistent_memory_context_is_explicit(tmp_path):
    memory = service(tmp_path)
    context = memory.retrieve_context("What do you remember about me?")
    assert "No persistent memories were retrieved" in context
    assert is_persistent_memory_query("Is that in your persistent memory?")
    assert recognize_explicit_memory_request("What do you remember about me?") is None
    memory.close()


def test_empty_memory_does_not_allow_fabricated_persistent_claim(tmp_path):
    class Provider:
        name = "fake"
        endpoint = "memory://"

        def generate(self, request, *, cancellation=None):
            return LLMResponse("I remember that you enjoy learning new things.", request.model)

        def close(self):
            pass

    memory = service(tmp_path)
    conversation = ConversationService(
        Provider(),
        ConversationSettings(),
        system_prompt="Policy",
        memory_service=memory,
    )
    response = conversation.respond("What do you remember about me?")
    assert "persistent" in response.text.lower()
    assert "enjoy" not in response.text.lower()
    memory.close()


def test_empty_memory_rejects_fact_hidden_after_a_denial(tmp_path):
    class Provider:
        name = "fake"
        endpoint = "memory://"

        def generate(self, request, *, cancellation=None):
            return LLMResponse("I don't have that in memory, but you enjoy learning.", request.model)

        def close(self):
            pass

    memory = service(tmp_path)
    conversation = ConversationService(
        Provider(), ConversationSettings(), system_prompt="Policy", memory_service=memory
    )
    response = conversation.respond("What do you remember about me?")
    assert "learning" not in response.text.lower()
    memory.close()


def test_session_history_does_not_count_as_persistent_memory(tmp_path):
    class Provider:
        name = "fake"
        endpoint = "memory://"

        def __init__(self):
            self.count = 0

        def generate(self, request, *, cancellation=None):
            self.count += 1
            return LLMResponse(
                "Okay, noted for this conversation." if self.count == 1 else "Yes, it is in my memory.",
                request.model,
            )

        def close(self):
            pass

    memory = service(tmp_path)
    conversation = ConversationService(
        Provider(),
        ConversationSettings(),
        system_prompt="Policy",
        memory_service=memory,
    )
    conversation.respond("My favorite ice cream is pistachio.")
    response = conversation.respond("Is pistachio in your persistent memory?")
    assert "persistent" in response.text.lower()
    assert "yes" not in response.text.lower()
    assert memory.list() == ()
    memory.close()


def test_failed_memory_result_cannot_be_claimed_as_success(tmp_path):
    class Provider:
        name = "fake"
        endpoint = "memory://"

        def generate(self, request, *, cancellation=None):
            if any(message.role is MessageRole.TOOL for message in request.messages):
                return LLMResponse("I remembered that successfully.", request.model)
            return LLMResponse("I remembered that successfully.", request.model)

        def close(self):
            pass

    memory = service(tmp_path, max_value_chars=4)
    executor = CompositeToolExecutor(MemoryToolExecutor(memory))
    conversation = ConversationService(
        Provider(),
        ConversationSettings(),
        system_prompt="Policy",
        tool_executor=executor,
    )
    # This explicit phrase is handled deterministically, then rejected by the
    # policy before any row can be committed.
    response = conversation.respond("Remember that my favorite color is greenish")
    assert "remembered" not in response.text.lower()
    assert memory.list() == ()
    memory.close()


def test_composed_memory_tools_are_provider_visible(tmp_path):
    memory = service(tmp_path)
    executor = CompositeToolExecutor(MemoryToolExecutor(memory))
    assert {definition.name for definition in executor.definitions} == {
        "remember_memory",
        "forget_memory",
    }
    memory.close()
