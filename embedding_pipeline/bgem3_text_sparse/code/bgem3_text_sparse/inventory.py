from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .config import AppConfig


SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
REQUIRED_INPUT_FIELDS = {
    "embedding_chunk_id",
    "day",
    "view",
    "start_time",
    "end_time",
    "duration_sec",
    "start_hour",
    "end_hour",
    "hours",
    "text_for_embedding",
    "text_for_display",
    "char_len",
    "token_estimate",
    "word_like_count",
    "source_chunk_ids",
    "source_chunks",
    "source_files",
    "quality_flags",
    "contains_invalid_source_chunk",
    "contains_repetition_compressed_source",
    "chunking_method",
    "chunking_version",
}


@dataclass(frozen=True)
class ChunkRecord:
    input_row_number: int
    raw: dict
    embedding_chunk_id: str
    day: str
    view: str
    text_for_embedding: str


@dataclass(frozen=True)
class InvalidRecord:
    input_row_number: int
    reason: str
    embedding_chunk_id: str | None = None


@dataclass(frozen=True)
class ChunkShard:
    day: str
    view: str
    row_count: int
    first_input_row_number: int
    max_token_estimate: int

    @property
    def key(self) -> str:
        return f"{self.day}/{self.view}"


@dataclass(frozen=True)
class Inventory:
    chunks_path: Path
    meta_summary_path: Path
    chunks_sha256: str
    meta_summary_sha256: str | None
    total_rows: int
    valid_rows: int
    invalid: Tuple[InvalidRecord, ...]
    duplicate_ids: Tuple[str, ...]
    shards: Tuple[ChunkShard, ...]
    token_estimate: dict
    by_day: Dict[str, int]
    by_view: Dict[str, int]
    meta_summary: dict | None

    def as_dict(self) -> dict:
        avg_tokens = self.token_estimate.get("avg")
        estimated_nnz = max(1, int(avg_tokens or 256))
        return {
            "chunks_path": str(self.chunks_path),
            "meta_summary_path": str(self.meta_summary_path),
            "chunks_sha256": self.chunks_sha256,
            "meta_summary_sha256": self.meta_summary_sha256,
            "total_rows": self.total_rows,
            "valid_rows": self.valid_rows,
            "invalid_rows": len(self.invalid),
            "invalid": [item.__dict__ for item in self.invalid[:100]],
            "duplicate_ids": list(self.duplicate_ids[:100]),
            "candidate_shards": len(self.shards),
            "shard_granularity": "day/view",
            "shards": [item.__dict__ for item in self.shards],
            "token_estimate": self.token_estimate,
            "by_day": dict(sorted(self.by_day.items())),
            "by_view": dict(sorted(self.by_view.items())),
            "meta_summary": self.meta_summary,
            "estimated_sparse_nnz_per_row": estimated_nnz,
            "estimated_sparse_index_value_bytes": self.valid_rows * estimated_nnz * (4 + 2),
        }

    def selected_shards(
        self,
        *,
        day: Optional[str] = None,
        view: Optional[str] = None,
        max_chunks: Optional[int] = None,
    ) -> List[ChunkShard]:
        selected = [shard for shard in self.shards if (day is None or shard.day == day) and (view is None or shard.view == view)]
        if max_chunks is None:
            return selected
        remaining = max_chunks
        limited = []
        for shard in selected:
            if remaining <= 0:
                break
            rows = min(shard.row_count, remaining)
            limited.append(ChunkShard(shard.day, shard.view, rows, shard.first_input_row_number, shard.max_token_estimate))
            remaining -= rows
        return limited


def file_sha256(path: Path | str, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value) -> str:
    return json.dumps(value if value is not None else [], ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def safe_component(value: str, field: str) -> str:
    if not SAFE_COMPONENT.fullmatch(value):
        raise ValueError(f"Unsafe {field}: {value!r}")
    return value


def _percentile(values: Sequence[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def _read_meta_summary(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict | None, str | None]]:
    with path.open(encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            if not line.strip():
                yield row_number, None, "blank_line"
                continue
            try:
                yield row_number, json.loads(line), None
            except json.JSONDecodeError as exc:
                yield row_number, None, f"json_decode_error: {exc}"


def _record_from_raw(row_number: int, raw: dict) -> tuple[ChunkRecord | None, InvalidRecord | None]:
    missing = sorted(field for field in REQUIRED_INPUT_FIELDS if field not in raw)
    chunk_id = raw.get("embedding_chunk_id")
    if missing:
        return None, InvalidRecord(row_number, f"missing_fields:{','.join(missing)}", chunk_id)
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        return None, InvalidRecord(row_number, "missing_embedding_chunk_id", None)
    text = raw.get("text_for_embedding")
    if not isinstance(text, str) or not text.strip():
        return None, InvalidRecord(row_number, "missing_text_for_embedding", chunk_id)
    schema_error = _input_schema_error(raw)
    if schema_error is not None:
        return None, InvalidRecord(row_number, schema_error, chunk_id)
    day = str(raw.get("day") or "")
    view = str(raw.get("view") or "")
    try:
        safe_component(day, "day")
        safe_component(view, "view")
    except ValueError as exc:
        return None, InvalidRecord(row_number, str(exc), chunk_id)
    return ChunkRecord(row_number, raw, chunk_id, day, view, text), None


def _input_schema_error(raw: dict) -> str | None:
    string_fields = (
        "day",
        "view",
        "start_hour",
        "end_hour",
        "text_for_embedding",
        "text_for_display",
        "chunking_method",
        "chunking_version",
    )
    for field in string_fields:
        if not isinstance(raw.get(field), str):
            return f"invalid_type:{field}:expected_string"

    for field in ("start_time", "end_time", "duration_sec"):
        value = raw.get(field)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return f"invalid_type:{field}:expected_finite_number_or_null"

    for field in ("char_len", "token_estimate", "word_like_count"):
        value = raw.get(field)
        if type(value) is not int or value < 0:
            return f"invalid_type:{field}:expected_non_negative_integer"

    for field in ("contains_invalid_source_chunk", "contains_repetition_compressed_source"):
        if type(raw.get(field)) is not bool:
            return f"invalid_type:{field}:expected_boolean"

    string_list_fields = (
        "hours",
        "source_chunk_ids",
        "source_files",
        "quality_flags",
    )
    for field in string_list_fields:
        value = raw.get(field)
        if not isinstance(value, list):
            return f"invalid_type:{field}:expected_list"
        if any(not isinstance(item, str) for item in value):
            return f"invalid_type:{field}:expected_string_items"

    source_chunks = raw.get("source_chunks")
    if not isinstance(source_chunks, list):
        return "invalid_type:source_chunks:expected_list"
    if any(not isinstance(item, dict) for item in source_chunks):
        return "invalid_type:source_chunks:expected_object_items"
    return None


def load_chunk_records(
    config: AppConfig,
    *,
    day: Optional[str] = None,
    view: Optional[str] = None,
    max_chunks: Optional[int] = None,
) -> List[ChunkRecord]:
    records: List[ChunkRecord] = []
    seen = set()
    for row_number, raw, error in _iter_jsonl(config.paths.chunks_path):
        if error is not None:
            raise ValueError(f"Invalid input row {row_number}: {error}")
        assert raw is not None
        record, invalid = _record_from_raw(row_number, raw)
        if invalid is not None:
            raise ValueError(f"Invalid input row {row_number}: {invalid.reason}")
        assert record is not None
        if record.embedding_chunk_id in seen:
            raise ValueError(f"Duplicate embedding_chunk_id: {record.embedding_chunk_id}")
        seen.add(record.embedding_chunk_id)
        if (day is not None and record.day != day) or (view is not None and record.view != view):
            continue
        records.append(record)
        if max_chunks is not None and len(records) >= max_chunks:
            break
    return records


def group_records(records: Iterable[ChunkRecord]) -> List[tuple[ChunkShard, List[ChunkRecord]]]:
    grouped: Dict[tuple[str, str], List[ChunkRecord]] = {}
    for record in records:
        grouped.setdefault((record.day, record.view), []).append(record)
    result = []
    for (day, view), items in grouped.items():
        token_values = [int(item.raw.get("token_estimate") or 0) for item in items]
        result.append(
            (
                ChunkShard(day, view, len(items), min(item.input_row_number for item in items), max(token_values) if token_values else 0),
                sorted(items, key=lambda item: item.input_row_number),
            )
        )
    return sorted(result, key=lambda item: (item[0].day, item[0].view))


def discover_inventory(config: AppConfig) -> Inventory:
    chunks = config.paths.chunks_path
    summary = config.paths.meta_summary_path
    if not chunks.is_file():
        raise FileNotFoundError(f"chunks.jsonl is missing: {chunks}")
    invalid: List[InvalidRecord] = []
    valid_records: List[ChunkRecord] = []
    seen: Dict[str, int] = {}
    duplicates: List[str] = []
    token_values: List[int] = []
    by_day: Dict[str, int] = {}
    by_view: Dict[str, int] = {}
    for row_number, raw, error in _iter_jsonl(chunks):
        if error is not None:
            invalid.append(InvalidRecord(row_number, error))
            continue
        assert raw is not None
        record, invalid_item = _record_from_raw(row_number, raw)
        if invalid_item is not None:
            invalid.append(invalid_item)
            continue
        assert record is not None
        if record.embedding_chunk_id in seen:
            duplicates.append(record.embedding_chunk_id)
        seen[record.embedding_chunk_id] = row_number
        valid_records.append(record)
        token = int(raw.get("token_estimate") or 0)
        token_values.append(token)
        by_day[record.day] = by_day.get(record.day, 0) + 1
        by_view[record.view] = by_view.get(record.view, 0) + 1
    shards = [item[0] for item in group_records(valid_records)]
    meta_summary = _read_meta_summary(summary)
    return Inventory(
        chunks_path=chunks,
        meta_summary_path=summary,
        chunks_sha256=file_sha256(chunks),
        meta_summary_sha256=file_sha256(summary) if summary.is_file() else None,
        total_rows=len(valid_records) + len(invalid),
        valid_rows=len(valid_records),
        invalid=tuple(invalid),
        duplicate_ids=tuple(sorted(set(duplicates))),
        shards=tuple(shards),
        token_estimate={
            "min": min(token_values) if token_values else None,
            "max": max(token_values) if token_values else None,
            "avg": mean(token_values) if token_values else None,
            "p50": _percentile(token_values, 0.5),
            "p90": _percentile(token_values, 0.9),
        },
        by_day=by_day,
        by_view=by_view,
        meta_summary=meta_summary,
    )

