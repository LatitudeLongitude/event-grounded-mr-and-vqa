from __future__ import annotations

import re
from typing import Any

from .clean_text import TextCleanResult
from .clean_timestamps import TimestampCleanResult
from .discover import TranscriptFile


_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def make_chunk_id(item: TranscriptFile, raw_chunk_index: int) -> str:
    parts = [item.day, item.view, item.hour, f"{raw_chunk_index:06d}"]
    return "_".join(_safe_id_part(part) for part in parts)


def build_chunk_record(
    *,
    item: TranscriptFile,
    source_file: str,
    raw_chunk_index: int,
    raw_timestamp: Any,
    text_result: TextCleanResult,
    timestamp_result: TimestampCleanResult,
    extra_flags: list[str] | None = None,
) -> dict[str, Any]:
    flags = [
        *timestamp_result.flags,
        *text_result.flags,
        *(extra_flags or []),
    ]
    flags = _dedupe_flags(flags)
    return {
        "chunk_id": make_chunk_id(item, raw_chunk_index),
        "day": item.day,
        "view": item.view,
        "hour": item.hour,
        "source_file": source_file,
        "raw_chunk_index": raw_chunk_index,
        "raw_timestamp": raw_timestamp,
        "start_time": timestamp_result.start_time,
        "end_time": timestamp_result.end_time,
        "duration_sec": timestamp_result.duration_sec,
        "timestamp_valid": timestamp_result.timestamp_valid and not extra_flags,
        "raw_text": text_result.raw_text,
        "cleaned_text": text_result.cleaned_text,
        "is_empty": text_result.is_empty,
        "has_repetition": text_result.has_repetition,
        "quality_flags": flags,
        "char_len": text_result.char_len,
        "word_like_count": text_result.word_like_count,
    }


def invalid_chunk_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": record["chunk_id"],
        "day": record["day"],
        "view": record["view"],
        "hour": record["hour"],
        "source_file": record["source_file"],
        "raw_chunk_index": record["raw_chunk_index"],
        "raw_timestamp": record["raw_timestamp"],
        "raw_text": record["raw_text"],
        "quality_flags": record["quality_flags"],
    }


def _safe_id_part(value: str) -> str:
    cleaned = _SAFE_ID_RE.sub("_", value.strip())
    return cleaned.strip("_") or "unknown"


def _dedupe_flags(flags: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for flag in flags:
        if flag not in seen:
            seen.add(flag)
            deduped.append(flag)
    return deduped
