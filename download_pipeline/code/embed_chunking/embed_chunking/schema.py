from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CleanedRecord:
    chunk_id: str
    day: str
    view: str
    hour: str
    source_file: str
    raw_chunk_index: int
    raw_timestamp: Any
    start_time: float | None
    end_time: float | None
    duration_sec: float | None
    timestamp_valid: bool
    raw_text: str
    cleaned_text: str
    is_empty: bool
    has_repetition: bool
    quality_flags: list[str]
    char_len: int
    word_like_count: int
    token_estimate: int
    field_warnings: list[str] = field(default_factory=list)

    def source_mapping(self, source_text_span: dict[str, int] | None = None) -> dict[str, object]:
        mapping: dict[str, object] = {
            "chunk_id": self.chunk_id,
            "hour": self.hour,
            "source_file": self.source_file,
            "raw_chunk_index": self.raw_chunk_index,
            "raw_timestamp": self.raw_timestamp,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_sec": self.duration_sec,
            "timestamp_valid": self.timestamp_valid,
            "raw_text": self.raw_text,
            "cleaned_text": self.cleaned_text,
            "quality_flags": list(self.quality_flags),
        }
        if source_text_span is not None:
            mapping["source_text_span"] = source_text_span
        return mapping


@dataclass(frozen=True)
class ValidationResult:
    valid: CleanedRecord | None = None
    skipped: dict[str, object] | None = None
    invalid: dict[str, object] | None = None


@dataclass(frozen=True)
class LoadedFile:
    view: str
    input_file: str
    input_records: int
    records: list[CleanedRecord]
    skipped: list[dict[str, object]]
    invalid: list[dict[str, object]]
    missing_optional_fields: dict[str, int]
