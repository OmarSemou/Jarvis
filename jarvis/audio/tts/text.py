"""Deterministic display-Markdown to plain, speakable text normalization."""

from __future__ import annotations

import re


_FENCE = re.compile(r"^\s*(```+|~~~+)(?:\w[\w.+-]*)?\s*$")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")
_BLOCKQUOTE = re.compile(r"^\s{0,3}>\s?")
_BULLET = re.compile(r"^\s{0,3}[-+*]\s+")
_NUMBERED = re.compile(r"^\s*(\d+)[.)]\s+")
_IMAGE = re.compile(r"!\[([^\]]*)\]\((?:[^()\s]|\([^)]*\))+\)")
_LINK = re.compile(r"(?<!!)\[([^\]]+)\]\((?:[^()\s]|\([^)]*\))+\)")
_INLINE_CODE = re.compile(r"(`+)(.+?)\1")
_BOLD_ITALIC_ASTERISK = re.compile(r"\*{3}(.+?)\*{3}")
_BOLD_ITALIC_UNDERSCORE = re.compile(r"_{3}(.+?)_{3}")
_BOLD_ASTERISK = re.compile(r"\*{2}(.+?)\*{2}")
_BOLD_UNDERSCORE = re.compile(r"_{2}(.+?)_{2}")
_ITALIC_ASTERISK = re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)")
_ITALIC_UNDERSCORE = re.compile(r"(?<![\w_])_([^_\n]+?)_(?![\w_])")
_RESIDUAL_MARKDOWN_ASTERISK = re.compile(
    r"(?<!\w)\*{1,3}(?=\S)|(?<=\S)\*{1,3}(?!\w)"
)
_SPACE = re.compile(r"[ \t]+")
_PUNCTUATION_SPACE = re.compile(r"\s+([,.;:!?])")


def _normalize_inline(text: str) -> str:
    text = _IMAGE.sub(lambda match: match.group(1).strip(), text)
    text = _LINK.sub(lambda match: match.group(1).strip(), text)
    text = _INLINE_CODE.sub(lambda match: match.group(2).strip(), text)
    for pattern in (
        _BOLD_ITALIC_ASTERISK,
        _BOLD_ITALIC_UNDERSCORE,
        _BOLD_ASTERISK,
        _BOLD_UNDERSCORE,
        _ITALIC_ASTERISK,
        _ITALIC_UNDERSCORE,
    ):
        text = pattern.sub(lambda match: match.group(1).strip(), text)
    # Remove only unmatched emphasis-style delimiters. A mathematical asterisk
    # surrounded by spaces, as in ``2 * 3``, is not treated as Markdown.
    text = _RESIDUAL_MARKDOWN_ASTERISK.sub("", text)
    text = _PUNCTUATION_SPACE.sub(r"\1", _SPACE.sub(" ", text))
    return text.strip()


def _finish_list_item(text: str) -> str:
    return text if not text or text.endswith((".", "!", "?", ":", ";")) else f"{text}."


def prepare_text_for_speech(text: str) -> str:
    """Return speakable text without rendering, fetching, or interpreting it."""

    if not isinstance(text, str):
        raise TypeError("speech text must be a string")

    output: list[str] = []
    inside_fence = False
    for raw_line in text.splitlines():
        if _FENCE.match(raw_line):
            inside_fence = not inside_fence
            continue

        line = raw_line.strip() if inside_fence else raw_line
        line = _HEADING.sub("", line)
        line = _BLOCKQUOTE.sub("", line)

        numbered = _NUMBERED.match(line)
        bullet = _BULLET.match(line)
        if numbered is not None:
            item = _normalize_inline(line[numbered.end() :])
            line = f"{numbered.group(1)}. {_finish_list_item(item)}" if item else ""
        elif bullet is not None:
            line = _finish_list_item(_normalize_inline(line[bullet.end() :]))
        else:
            line = _normalize_inline(line)

        if line:
            output.append(line)
        elif output and output[-1] != "":
            output.append("")

    while output and output[-1] == "":
        output.pop()
    return "\n".join(output).strip()
