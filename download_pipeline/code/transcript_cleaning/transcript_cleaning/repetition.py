from __future__ import annotations

import re
from dataclasses import dataclass


_SENTENCE_RE = re.compile(r"[^.!?\u3002\uff01\uff1f]+[.!?\u3002\uff01\uff1f]?")


@dataclass(frozen=True)
class RepetitionResult:
    text: str
    has_repetition: bool
    compressed: bool


def compress_repetitions(text: str) -> RepetitionResult:
    after_sentences, sentence_repeated, sentence_compressed = _compress_sentence_runs(text)
    after_tokens, token_repeated, token_compressed = _compress_token_runs(after_sentences)
    return RepetitionResult(
        text=after_tokens,
        has_repetition=sentence_repeated or token_repeated,
        compressed=sentence_compressed or token_compressed,
    )


def _compress_sentence_runs(text: str) -> tuple[str, bool, bool]:
    parts = _split_sentences_with_spacing(text)
    if not parts:
        return text, False, False

    output: list[str] = []
    has_repetition = False
    compressed = False
    index = 0
    while index < len(parts):
        sentence, spacing = parts[index]
        key = _sentence_key(sentence)
        run_end = index + 1
        while run_end < len(parts) and _sentence_key(parts[run_end][0]) == key:
            run_end += 1

        run_length = run_end - index
        if key and run_length > 1:
            has_repetition = True
        if key and run_length > 3:
            output.append(sentence.strip())
            compressed = True
        else:
            for sent, space in parts[index:run_end]:
                output.append(sent.strip() + space)
        index = run_end

    rendered = " ".join(piece.strip() for piece in output if piece.strip())
    return rendered or text, has_repetition, compressed


def _split_sentences_with_spacing(text: str) -> list[tuple[str, str]]:
    matches = list(_SENTENCE_RE.finditer(text))
    if not matches:
        return []

    parts: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        sentence = match.group(0)
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        spacing = text[match.end() : next_start]
        if sentence.strip():
            parts.append((sentence, spacing))
    return parts


def _sentence_key(sentence: str) -> str:
    return re.sub(r"\s+", " ", sentence).strip().casefold()


def _compress_token_runs(text: str) -> tuple[str, bool, bool]:
    tokens = text.split(" ")
    if not tokens:
        return text, False, False

    output: list[str] = []
    has_repetition = False
    compressed = False
    index = 0
    while index < len(tokens):
        token = tokens[index]
        key = token.casefold()
        run_end = index + 1
        while run_end < len(tokens) and tokens[run_end].casefold() == key:
            run_end += 1

        run_length = run_end - index
        if key and run_length >= 3:
            has_repetition = True
        if key and run_length > 5:
            output.extend(tokens[index : index + 2])
            compressed = True
        else:
            output.extend(tokens[index:run_end])
        index = run_end

    return " ".join(output), has_repetition, compressed
