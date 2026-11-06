from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from .repetition import compress_repetitions


_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class TextCleanResult:
    raw_text: str
    cleaned_text: str
    is_empty: bool
    has_repetition: bool
    flags: list[str]
    char_len: int
    word_like_count: int


def clean_text_value(value: Any) -> TextCleanResult:
    raw_text = "" if value is None else str(value)
    normalized = unicodedata.normalize("NFC", raw_text)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()

    repetition = compress_repetitions(normalized) if normalized else None
    cleaned_text = repetition.text if repetition is not None else normalized

    flags: list[str] = []
    has_repetition = False
    if repetition is not None and repetition.has_repetition:
        has_repetition = True
        flags.append("repetition_compressed" if repetition.compressed else "repetition_detected")

    is_empty = cleaned_text == ""
    if is_empty:
        flags.append("empty_text")

    return TextCleanResult(
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        is_empty=is_empty,
        has_repetition=has_repetition,
        flags=flags,
        char_len=len(cleaned_text),
        word_like_count=len([token for token in cleaned_text.split(" ") if token]),
    )
