"""Split thinking from answer and pull the final code block out of a completion."""
from __future__ import annotations

import re

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n(.*?)```", re.DOTALL)
_CODE_TAG = re.compile(r"<code>(.*?)</code>", re.DOTALL)
_PY_LANGS = {"", "python", "py", "python3", "py3"}


def split_thinking(text: str, think_end: str = "</think>") -> tuple[str, str]:
    """Return (thinking, answer). With no end tag the whole text counts as unfinished thinking."""
    if not think_end:
        return "", text
    i = text.find(think_end)
    if i < 0:
        return text, ""
    return text[:i], text[i + len(think_end):]


def extract_code(answer: str) -> str | None:
    """Last Python (or untagged) fenced block; else the last <code> block; else None.
    A missing code block is scored as a failure by the caller, never repaired here."""
    blocks = [(lang.lower(), body) for lang, body in _FENCE.findall(answer)]
    py = [b for lang, b in blocks if lang in _PY_LANGS]
    if py:
        return py[-1].strip("\n") or None
    tags = _CODE_TAG.findall(answer)
    if tags:
        return tags[-1].strip("\n") or None
    return None
