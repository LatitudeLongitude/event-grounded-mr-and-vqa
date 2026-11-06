from __future__ import annotations

from dataclasses import dataclass

from .config import ChunkingConfig
from .sentence_split import split_sentences
from .token_estimate import estimate_text_metrics


@dataclass(frozen=True)
class SplitPart:
    text: str
    char_start: int
    char_end: int
    token_estimate: int
    split_index: int
    split_count: int


def split_long_text(text: str, config: ChunkingConfig) -> list[SplitPart]:
    stripped = text.strip()
    if not stripped:
        return []

    sentence_spans = _sentence_spans(text)
    if not sentence_spans:
        sentence_spans = [(0, len(text), stripped)]

    parts: list[tuple[str, int, int]] = []
    current_texts: list[str] = []
    current_start: int | None = None
    current_end: int | None = None

    for start, end, sentence in sentence_spans:
        if estimate_text_metrics(sentence).token_estimate > config.target_max_token_estimate:
            if current_texts:
                parts.append((" ".join(current_texts).strip(), current_start or 0, current_end or 0))
                current_texts = []
                current_start = None
                current_end = None
            parts.extend(_split_oversized_sentence(text, start, end, config))
            continue

        candidate = " ".join([*current_texts, sentence]).strip()
        candidate_tokens = estimate_text_metrics(candidate).token_estimate
        if current_texts and candidate_tokens > config.target_max_token_estimate:
            parts.append((" ".join(current_texts).strip(), current_start or 0, current_end or 0))
            current_texts = [sentence]
            current_start = start
            current_end = end
        else:
            if current_start is None:
                current_start = start
            current_texts.append(sentence)
            current_end = end

    if current_texts:
        parts.append((" ".join(current_texts).strip(), current_start or 0, current_end or 0))

    if not parts:
        parts.append((stripped, text.find(stripped), text.find(stripped) + len(stripped)))

    split_count = len(parts)
    return [
        SplitPart(
            text=part_text,
            char_start=char_start,
            char_end=char_end,
            token_estimate=estimate_text_metrics(part_text).token_estimate,
            split_index=index,
            split_count=split_count,
        )
        for index, (part_text, char_start, char_end) in enumerate(parts)
    ]


def _sentence_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    search_from = 0
    for sentence in split_sentences(text):
        start = text.find(sentence, search_from)
        if start == -1:
            start = search_from
        end = start + len(sentence)
        spans.append((start, end, sentence))
        search_from = end
    return spans


def _split_oversized_sentence(text: str, start: int, end: int, config: ChunkingConfig) -> list[tuple[str, int, int]]:
    segment = text[start:end]
    words = segment.split()
    if len(words) > 1:
        parts = []
        current_words: list[str] = []
        current_start = start
        cursor = start
        for word in words:
            candidate = " ".join([*current_words, word]).strip()
            if current_words and estimate_text_metrics(candidate).token_estimate > config.target_max_token_estimate:
                part_text = " ".join(current_words).strip()
                part_start = text.find(part_text, current_start)
                part_end = part_start + len(part_text)
                parts.append((part_text, part_start, part_end))
                current_words = [word]
                current_start = part_end
            else:
                current_words.append(word)
            cursor += len(word) + 1
        if current_words:
            part_text = " ".join(current_words).strip()
            part_start = text.find(part_text, current_start)
            part_end = part_start + len(part_text)
            parts.append((part_text, part_start, part_end))
        return parts

    max_chars = max(1, config.target_max_token_estimate * 4)
    parts = []
    for char_start in range(start, end, max_chars):
        char_end = min(end, char_start + max_chars)
        part_text = text[char_start:char_end].strip()
        if part_text:
            offset = text.find(part_text, char_start)
            parts.append((part_text, offset, offset + len(part_text)))
    return parts
