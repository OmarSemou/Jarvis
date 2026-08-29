from types import SimpleNamespace

from jarvis.cli import build_parser, voice_command
from jarvis.core.config import JarvisConfig
from jarvis.core.paths import JarvisPaths


def test_voice_subcommand_and_latency_flag_are_registered():
    args = build_parser().parse_args(["voice", "--debug-latency"])
    assert args.command == "voice"
    assert args.debug_latency is True


def test_disabled_wake_word_refuses_continuous_mode_before_constructing_runtime(
    tmp_path, monkeypatch
):
    paths = JarvisPaths.from_repository_root(tmp_path.resolve())
    config = JarvisConfig(
        voice_mode_enabled=True,
        wakeword_enabled=False,
        tts_enabled=True,
    )
    monkeypatch.setattr("jarvis.cli.JarvisPaths.discover", lambda: paths)
    monkeypatch.setattr(
        "jarvis.cli.load_for_paths", lambda _paths: SimpleNamespace(config=config)
    )
    output = []

    assert voice_command(output_fn=output.append) == 2
    assert output == ["Voice mode unavailable: wake-word detection is disabled."]


def test_voice_command_uses_long_keep_alive_and_prints_concise_local_status(
    tmp_path, monkeypatch
):
    paths = JarvisPaths.from_repository_root(tmp_path.resolve())
    config = JarvisConfig(voice_mode_enabled=True, tts_enabled=True)
    monkeypatch.setattr("jarvis.cli.JarvisPaths.discover", lambda: paths)
    monkeypatch.setattr(
        "jarvis.cli.load_for_paths", lambda _paths: SimpleNamespace(config=config)
    )
    fake_controller = object()
    monkeypatch.setattr(
        "jarvis.cli.create_robot_runtime",
        lambda **_kwargs: SimpleNamespace(tools=object(), controller=fake_controller),
    )
    monkeypatch.setattr("jarvis.cli.create_voice_runtime", lambda *_args: object())
    fake_tts = SimpleNamespace(provider="kokoro", voice="am_fenrir")
    monkeypatch.setattr("jarvis.cli.create_tts_runtime", lambda *_args: fake_tts)
    created = {}

    class Service:
        def close(self):
            created["closed"] = True

    def conversation_factory(_config, *, tool_executor, keep_alive):
        created["tools"] = tool_executor
        created["keep_alive"] = keep_alive
        return Service()

    class Coordinator:
        def run(self):
            created["ran"] = True
            return 0

    monkeypatch.setattr("jarvis.cli.create_conversation", conversation_factory)
    monkeypatch.setattr(
        "jarvis.cli.create_voice_mode_coordinator",
        lambda *_args, **kwargs: created.update(
            debug=kwargs["debug_latency"],
            controller=kwargs["robot_controller"],
        )
        or Coordinator(),
    )
    output = []

    assert voice_command(debug_latency=True, output_fn=output.append) == 0
    assert created["keep_alive"] == "30m"
    assert created["debug"] is True
    assert created["controller"] is fake_controller
    assert created["ran"] and created["closed"]
    assert "BMO Voice" in output
    assert any("Hey Jarvis" in line for line in output)
    assert "Voice: kokoro / am_fenrir" in output
    assert "Barge-in: wakeword" in output
    assert "Preparing local voice models..." in output
