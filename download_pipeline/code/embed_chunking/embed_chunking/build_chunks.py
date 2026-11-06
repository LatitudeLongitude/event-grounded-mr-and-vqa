from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from math import ceil
import re

from .config import ChunkingConfig
from .schema import CleanedRecord
from .split_long import split_long_text
from .token_estimate import estimate_text_metrics


@dataclass(frozen=True)
class ChunkUnit:
    records: tuple[CleanedRecord, ...]
    text: str
    method: str
    source_text_span: dict[str, int] | None = None
    estimated_start_time: float | None = None
    estimated_end_time: float | None = None
    estimated_time_range: bool = False

    @property
    def token_estimate(self) -> int:
        return estimate_text_metrics(self.text).token_estimate

    @property
    def start_time(self) -> float | None:
        if self.estimated_start_time is not None:
            return self.estimated_start_time
        if all(record.timestamp_valid for record in self.records):
            return self.records[0].start_time
        return None

    @property
    def end_time(self) -> float | None:
        if self.estimated_end_time is not None:
            return self.estimated_end_time
        if all(record.timestamp_valid for record in self.records):
            return self.records[-1].end_time
        return None


def build_chunks(records: list[CleanedRecord], config: ChunkingConfig) -> list[dict[str, object]]:
    config.validate()
    grouped: dict[tuple[str, str], list[CleanedRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.view, record.day)].append(record)

    outputs: list[dict[str, object]] = []
    counters: dict[tuple[str, str], int] = defaultdict(int)

    for (view, day) in sorted(grouped, key=lambda item: (_natural_key(item[0][0]), _natural_key(item[0][1]))):
        group_records = sorted(grouped[(view, day)], key=_record_sort_key)
        pending: list[ChunkUnit] = []

        def flush_pending() -> None:
            nonlocal pending
            if not pending:
                return
            counters[(day, view)] += 1
            outputs.append(_build_output_chunk(pending, counters[(day, view)], config))
            pending = []

        for record in group_records:
            split_units = _units_for_record(record, config)
            if len(split_units) > 1 or split_units[0].method == "split_long_chunk":
                flush_pending()
                for unit in split_units:
                    counters[(day, view)] += 1
                    outputs.append(_build_output_chunk([unit], counters[(day, view)], config))
                continue

            unit = split_units[0]
            if pending and _would_exceed_hard_limit([*pending, unit], config):
                flush_pending()
            pending.append(unit)
            if _ready_to_flush(pending, config):
                flush_pending()

        flush_pending()

    return sorted(outputs, key=lambda item: (_natural_key(str(item["view"])), _natural_key(str(item["day"])), _none_last(item["start_time"]), item["embedding_chunk_id"]))


def _units_for_record(record: CleanedRecord, config: ChunkingConfig) -> list[ChunkUnit]:
    duration = record.duration_sec or 0.0
    should_split = record.token_estimate > config.hard_max_token_estimate or (
        record.timestamp_valid and duration > config.hard_max_duration_sec
    )
    if not should_split:
        return [ChunkUnit(records=(record,), text=record.cleaned_text, method="single_source_chunk")]

    if duration > config.hard_max_duration_sec and record.token_estimate <= config.target_max_token_estimate:
        desired_parts = max(2, ceil(duration / config.hard_max_duration_sec))
        target_max = max(1, ceil(max(record.token_estimate, 1) / desired_parts))
        split_config = replace(
            config,
            target_max_token_estimate=min(config.hard_max_token_estimate, max(target_max, 1)),
            hard_max_token_estimate=max(config.hard_max_token_estimate, target_max),
        )
        parts = split_long_text(record.cleaned_text, split_config)
    else:
        parts = split_long_text(record.cleaned_text, config)

    if len(parts) <= 1:
        return [ChunkUnit(records=(record,), text=record.cleaned_text, method="split_long_chunk")]

    units = []
    text_len = max(1, len(record.cleaned_text))
    valid_source_time = _valid_time(record.start_time, record.end_time)
    for part in parts:
        estimated_start = None
        estimated_end = None
        estimated = False
        if record.timestamp_valid and valid_source_time:
            span_start_ratio = part.char_start / text_len
            span_end_ratio = part.char_end / text_len
            source_duration = (record.end_time or 0.0) - (record.start_time or 0.0)
            estimated_start = (record.start_time or 0.0) + source_duration * span_start_ratio
            estimated_end = (record.start_time or 0.0) + source_duration * span_end_ratio
            estimated = True
        units.append(
            ChunkUnit(
                records=(record,),
                text=part.text,
                method="split_long_chunk",
                source_text_span={
                    "split_index": part.split_index,
                    "split_count": part.split_count,
                    "char_start": part.char_start,
                    "char_end": part.char_end,
                },
                estimated_start_time=estimated_start,
                estimated_end_time=estimated_end,
                estimated_time_range=estimated,
            )
        )
    return units


def _build_output_chunk(units: list[ChunkUnit], running_index: int, config: ChunkingConfig) -> dict[str, object]:
    records = [record for unit in units for record in unit.records]
    first = records[0]
    text = " ".join(unit.text.strip() for unit in units if unit.text.strip())
    metrics = estimate_text_metrics(text)
    start_time, end_time, duration_sec, time_flags = _time_range(units)
    hours = _unique_preserving_order(record.hour for record in records)
    source_files = _unique_preserving_order(record.source_file for record in records)
    quality_flags = _quality_flags(records, units, metrics.token_estimate, duration_sec, hours, time_flags, config)
    method = _chunking_method(units)

    return {
        "embedding_chunk_id": f"emb_{first.day}_{first.view}_{running_index:06d}",
        "day": first.day,
        "view": first.view,
        "start_time": _round_or_none(start_time),
        "end_time": _round_or_none(end_time),
        "duration_sec": _round_or_none(duration_sec),
        "start_hour": hours[0],
        "end_hour": hours[-1],
        "hours": hours,
        "text_for_embedding": text,
        "text_for_display": text,
        "char_len": metrics.char_len,
        "token_estimate": metrics.token_estimate,
        "word_like_count": metrics.word_like_count,
        "source_chunk_ids": [record.chunk_id for record in records],
        "source_chunks": _source_chunks(units),
        "source_files": source_files,
        "quality_flags": quality_flags,
        "contains_invalid_source_chunk": any(not record.timestamp_valid for record in records),
        "contains_repetition_compressed_source": any("repetition_compressed" in record.quality_flags for record in records),
        "chunking_method": method,
        "chunking_version": config.chunking_version,
    }


def _source_chunks(units: list[ChunkUnit]) -> list[dict[str, object]]:
    chunks = []
    for unit in units:
        for record in unit.records:
            chunks.append(record.source_mapping(unit.source_text_span))
    return chunks


def _ready_to_flush(units: list[ChunkUnit], config: ChunkingConfig) -> bool:
    token_count = _combined_tokens(units)
    duration = _combined_duration(units)
    if token_count >= config.target_max_token_estimate:
        return True
    if duration is not None and duration >= config.target_max_duration_sec:
        return True
    if token_count >= config.target_min_token_estimate and duration is not None and duration >= config.target_min_duration_sec:
        return True
    return False


def _would_exceed_hard_limit(units: list[ChunkUnit], config: ChunkingConfig) -> bool:
    token_count = _combined_tokens(units)
    duration = _combined_duration(units)
    if token_count > config.hard_max_token_estimate:
        return True
    return duration is not None and duration > config.hard_max_duration_sec


def _combined_tokens(units: list[ChunkUnit]) -> int:
    return estimate_text_metrics(" ".join(unit.text.strip() for unit in units if unit.text.strip())).token_estimate


def _combined_duration(units: list[ChunkUnit]) -> float | None:
    start, end, _, _ = _time_range(units)
    if start is None or end is None:
        return None
    return max(0.0, end - start)


def _time_range(units: list[ChunkUnit]) -> tuple[float | None, float | None, float | None, list[str]]:
    flags: list[str] = []
    if any(not record.timestamp_valid for unit in units for record in unit.records):
        flags.append("invalid_time_range")
    reliable_units = [unit for unit in units if unit.start_time is not None and unit.end_time is not None]
    if not reliable_units:
        return None, None, None, ["invalid_time_range"]
    start = reliable_units[0].start_time
    end = reliable_units[-1].end_time
    if start is None or end is None:
        return None, None, None, ["invalid_time_range"]
    if end < start:
        flags.append("invalid_time_range")
        return start, end, None, flags
    if any(unit.estimated_time_range for unit in units):
        flags.append("estimated_time_range")
    return start, end, end - start, flags


def _quality_flags(
    records: list[CleanedRecord],
    units: list[ChunkUnit],
    token_estimate: int,
    duration_sec: float | None,
    hours: list[str],
    time_flags: list[str],
    config: ChunkingConfig,
) -> list[str]:
    flags: list[str] = []
    for record in records:
        flags.extend(record.quality_flags)
        flags.extend(record.field_warnings)
    flags.extend(time_flags)
    if len(records) > 1:
        flags.append("merged_short_chunks")
    if any(unit.method == "split_long_chunk" for unit in units):
        flags.append("split_long_chunk")
    if len(hours) > 1:
        flags.append("cross_hour_chunk")
    if token_estimate < config.min_token_estimate:
        flags.append("text_too_short_but_kept")
    if token_estimate > config.hard_max_token_estimate:
        flags.append("text_too_long_after_split")
    if duration_sec is not None and duration_sec < config.min_duration_sec:
        flags.append("duration_too_short_but_kept")
    if duration_sec is not None and duration_sec > config.hard_max_duration_sec:
        flags.append("duration_too_long_after_split")
    return sorted(set(flags))


def _chunking_method(units: list[ChunkUnit]) -> str:
    split = any(unit.method == "split_long_chunk" for unit in units)
    source_count = sum(len(unit.records) for unit in units)
    if split and source_count > 1:
        return "merge_then_split"
    if split:
        return "split_long_chunk"
    if source_count > 1:
        return "merge_short_chunks"
    return "single_source_chunk"


def _record_sort_key(record: CleanedRecord) -> tuple[object, ...]:
    return (_natural_key(record.hour), _none_last(record.start_time), record.raw_chunk_index, record.chunk_id)


def _natural_key(value: str) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", value)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def _none_last(value: object) -> tuple[int, object]:
    if value is None:
        return (1, "")
    return (0, value)


def _unique_preserving_order(values) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _valid_time(start: float | None, end: float | None) -> bool:
    return start is not None and end is not None and end >= start


def _round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 3)
