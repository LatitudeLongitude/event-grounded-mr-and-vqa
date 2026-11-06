from __future__ import annotations

from dataclasses import dataclass
from math import ceil
import re


@dataclass(frozen=True)
class TextMetrics:
    char_len: int
    word_like_count: int
    token_estimate: int


def estimate_text_metrics(text: str) -> TextMetrics:
    normalized = text or ""
    char_len = len(normalized)
    word_like_count = len(re.findall(r"\S+", normalized))
    token_estimate = max(word_like_count, ceil(char_len / 4)) if char_len else 0
    return TextMetrics(char_len=char_len, word_like_count=word_like_count, token_estimate=token_estimate)
