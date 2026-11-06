from __future__ import annotations


SENTENCE_ENDINGS = {".", "?", "!", "。", "？", "！", "\n"}


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    for index, char in enumerate(text):
        if char in SENTENCE_ENDINGS:
            piece = text[start : index + 1].strip()
            if piece:
                sentences.append(piece)
            start = index + 1
    remainder = text[start:].strip()
    if remainder:
        sentences.append(remainder)
    return sentences
