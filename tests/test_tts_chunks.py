import re

from jarvis.audio.tts.chunks import SpeechChunker, SpeechChunkerSettings


def texts(value, *, maximum=220):
    return [
        chunk.text
        for chunk in SpeechChunker(
            SpeechChunkerSettings(max_characters=maximum)
        ).chunk(value)
    ]


def normalized(value):
    return re.sub(r"\s+", " ", value).strip()


def test_chunker_prefers_ordered_natural_sentence_boundaries():
    assert texts("Hello. This is another sentence. And a third.") == [
        "Hello.",
        "This is another sentence.",
        "And a third.",
    ]
    assert texts("Really? Yes! Calmly; then: done.") == [
        "Really?",
        "Yes!",
        "Calmly; then: done.",
    ]


def test_chunker_protects_decimals_abbreviations_initials_and_numbered_items():
    value = (
        "The estimate is 13.8 billion. Dr. Rivera agrees. "
        "J. R. R. Tolkien wrote books.\n1. Power Stroke: The gases expand."
    )

    result = texts(value)

    assert "13.8" in result[0]
    assert "Dr. Rivera agrees." in result
    assert "J. R. R. Tolkien wrote books." in result
    assert result[-1] == "1. Power Stroke: The gases expand."
    assert normalized(" ".join(result)) == normalized(value)


def test_short_acknowledgements_are_valid_standalone_chunks():
    assert texts("Hey.") == ["Hey."]
    assert texts("Stopped.") == ["Stopped."]


def test_long_sentence_fallback_is_bounded_and_lossless():
    value = (
        "This deliberately long sentence has several clauses, each carrying useful "
        "words, and it must be divided without losing or duplicating any of the "
        "speakable response text even when no terminal period appears until the end."
    )

    result = texts(value, maximum=70)

    assert len(result) > 2
    assert all(len(item) <= 70 for item in result)
    assert normalized(" ".join(result)) == normalized(value)


def test_empty_paragraphs_are_ignored_without_reordering_text():
    assert texts("First paragraph.\n\nSecond paragraph.") == [
        "First paragraph.",
        "Second paragraph.",
    ]
