import pytest

from jarvis.audio.tts.text import prepare_text_for_speech


@pytest.mark.parametrize(
    ("display", "speech"),
    (
        ("**Power Stroke**", "Power Stroke"),
        ("__Power Stroke__", "Power Stroke"),
        ("*Star Wars*", "Star Wars"),
        ("_Star Wars_", "Star Wars"),
        ("***important***", "important"),
        ("___important___", "important"),
        ("## Combustion Engine", "Combustion Engine"),
        ("> Important", "Important"),
        ("`qwen3:8b`", "qwen3:8b"),
        ("[OpenAI](https://openai.com)", "OpenAI"),
        ("[https://openai.com](https://openai.com)", "https://openai.com"),
        ("![diagram](https://example.test/image.png)", "diagram"),
        ("* Intake\n* Compression\n* Power", "Intake.\nCompression.\nPower."),
        ("- Intake\n- Compression", "Intake.\nCompression."),
        ("1. **Intake**\n2. *Compression*", "1. Intake.\n2. Compression."),
        ("```text\nshort code\n```", "short code"),
        ("Use   the **Power Stroke** .", "Use the Power Stroke."),
        ("Plain text unchanged.", "Plain text unchanged."),
        ("  \n\n ", ""),
    ),
)
def test_prepare_text_for_speech_rules(display, speech):
    assert prepare_text_for_speech(display) == speech


def test_mixed_markdown_keeps_boundaries_without_formatting_punctuation():
    display = """# Four strokes

1. **Intake**
2. *Compression*

> Use the `Power Stroke` and read [the label](https://example.test).
"""

    speech = prepare_text_for_speech(display)

    assert speech == (
        "Four strokes\n\n"
        "1. Intake.\n"
        "2. Compression.\n\n"
        "Use the Power Stroke and read the label."
    )
    assert "*" not in speech


def test_formatter_does_not_fetch_execute_or_touch_files(tmp_path):
    before = list(tmp_path.iterdir())
    text = "[label](https://example.invalid)\n```python\nopen('x')\n```"

    assert prepare_text_for_speech(text) == "label\nopen('x')"
    assert list(tmp_path.iterdir()) == before
