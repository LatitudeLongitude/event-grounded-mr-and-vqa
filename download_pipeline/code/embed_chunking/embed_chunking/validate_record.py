from __future__ import annotations

from collections import Counter
from typing import Any

from .schema import CleanedRecord, ValidationResult
from .token_estimate import estimate_text_metrics


REQUIRED_FIELDS = [
    "chunk_id",
    "day",
    "view",
    "hour",
    "source_file",
    "raw_chunk_index",
    "start_time",
    "end_time",
    "duration_sec",
    "timestamp_valid",
    "raw_text",
    "cleaned_text",
    "is_empty",
    "quality_flags",
]

OPTIONAL_FIELDS = ["raw_timestamp", "has_repetition", "char_len", "word_like_count"]


def validate_record(raw: Any, input_file: str, line_number: int) -> ValidationResult:
    if not isinstance(raw, dict):
        return ValidationResult(invalid=_invalid(input_file, line_number, "record_not_object", raw_record=raw))

    missing = [field for field in REQUIRED_FIELDS if field not in raw]
    if missing:
        return ValidationResult(
            invalid=_invalid(
                input_file,
                line_number,
                "missing_required_field",
                missing_fields=missing,
                raw_record=raw,
            )
        )

    missing_optional = [field for field in OPTIONAL_FIELDS if field not in raw or raw.get(field) is None]
    try:
        cleaned_text = _to_text(raw["cleaned_text"], "cleaned_text")
        quality_flags = _quality_flags(raw.get("quality_flags"))
        is_empty = _to_bool(raw["is_empty"], "is_empty")
        if _should_skip(cleaned_text, is_empty, quality_flags):
            return ValidationResult(skipped=_skipped_record(raw, cleaned_text, quality_flags))

        metrics = estimate_text_metrics(cleaned_text)
        char_len = _to_int(raw.get("char_len"), metrics.char_len, "char_len")
        word_like_count = _to_int(raw.get("word_like_count"), metrics.word_like_count, "word_like_count")
        token_estimate = max(word_like_count, (char_len + 3) // 4) if char_len else 0

        record = CleanedRecord(
            chunk_id=_to_text(raw["chunk_id"], "chunk_id"),
            day=_to_text(raw["day"], "day"),
            view=_to_text(raw["view"], "view"),
            hour=_to_text(raw["hour"], "hour"),
            source_file=_to_text(raw["source_file"], "source_file"),
            raw_chunk_index=_to_int(raw["raw_chunk_index"], None, "raw_chunk_index"),
            raw_timestamp=raw.get("raw_timestamp"),
            start_time=_to_optional_float(raw.get("start_time"), "start_time"),
            end_time=_to_optional_float(raw.get("end_time"), "end_time"),
            duration_sec=_to_optional_float(raw.get("duration_sec"), "duration_sec"),
            timestamp_valid=_to_bool(raw["timestamp_valid"], "timestamp_valid"),
            raw_text=_to_text(raw["raw_text"], "raw_text"),
            cleaned_text=cleaned_text,
            is_empty=is_empty,
            has_repetition=_to_bool(False if raw.get("has_repetition") is None else raw.get("has_repetition"), "has_repetition"),
            quality_flags=quality_flags,
            char_len=char_len,
            word_like_count=word_like_count,
            token_estimate=token_estimate,
            field_warnings=["source_field_missing"] if missing_optional else [],
        )
    except (TypeError, ValueError) as exc:
        return ValidationResult(
            invalid=_invalid(
                input_file,
                line_number,
                "field_type_error",
                error_message=str(exc),
                raw_record=raw,
            )
        )

    return ValidationResult(valid=record)


def count_missing_optional(raw: Any) -> Counter[str]:
    counts: Counter[str] = Counter()
    if isinstance(raw, dict):
        for field in OPTIONAL_FIELDS:
            if field not in raw or raw.get(field) is None:
                counts[field] += 1
    return counts


def _invalid(input_file: str, line_number: int, error_type: str, **extra: Any) -> dict[str, object]:
    payload: dict[str, object] = {
        "input_file": input_file,
        "line_number": line_number,
        "error_type": error_type,
    }
    payload.update(extra)
    return payload


def _skipped_record(raw: dict[str, Any], cleaned_text: str, quality_flags: list[str]) -> dict[str, object]:
    return {
        "chunk_id": raw.get("chunk_id"),
        "day": raw.get("day"),
        "view": raw.get("view"),
        "hour": raw.get("hour"),
        "source_file": raw.get("source_file"),
        "raw_chunk_index": raw.get("raw_chunk_index"),
        "skip_reason": "empty_cleaned_text",
        "cleaned_text": cleaned_text,
        "quality_flags": quality_flags,
    }


def _should_skip(cleaned_text: str, is_empty: bool, quality_flags: list[str]) -> bool:
    if is_empty or not cleaned_text.strip() or "empty_text" in quality_flags:
        return True
    if ("invalid_chunk_format" in quality_flags or "json_parse_error" in quality_flags) and not cleaned_text.strip():
        return True
    return False


def _to_text(value: Any, field: str) -> str:
    if value is None:
        raise TypeError(f"{field} is null")
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    raise TypeError(f"{field} must be string-compatible")


def _to_int(value: Any, default: int | None, field: str) -> int:
    if value is None:
        if default is None:
            raise TypeError(f"{field} is null")
        return default
    if isinstance(value, bool):
        raise TypeError(f"{field} must be int")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be int") from exc


def _to_optional_float(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{field} must be numeric")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be numeric") from exc


def _to_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    raise TypeError(f"{field} must be bool")


def _quality_flags(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError("quality_flags must be a list")
    return [str(item) for item in value]
