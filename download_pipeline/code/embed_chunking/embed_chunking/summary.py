from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Iterable


def build_summary(
    *,
    input_root: str,
    output_root: str,
    main_output: str,
    meta_output: str,
    loaded_files,
    chunks: list[dict[str, object]],
) -> dict[str, object]:
    token_values = [int(chunk["token_estimate"]) for chunk in chunks]
    duration_values = [float(chunk["duration_sec"]) for chunk in chunks if chunk.get("duration_sec") is not None]
    views: dict[str, dict[str, int]] = defaultdict(lambda: {"source_records": 0, "output_chunks": 0, "skipped_records": 0})
    quality_flags: Counter[str] = Counter()
    missing_optional: Counter[str] = Counter()

    for loaded in loaded_files:
        views[loaded.view]["source_records"] += loaded.input_records
        views[loaded.view]["skipped_records"] += len(loaded.skipped)
        missing_optional.update(loaded.missing_optional_fields)
    for chunk in chunks:
        views[str(chunk["view"])]["output_chunks"] += 1
        quality_flags.update(chunk.get("quality_flags", []))

    return {
        "input_root": input_root,
        "output_root": output_root,
        "main_output": main_output,
        "meta_output": meta_output,
        "total_input_files": len(loaded_files),
        "success_input_files": len(loaded_files),
        "failed_input_files": 0,
        "total_source_records": sum(loaded.input_records for loaded in loaded_files),
        "valid_source_records": sum(len(loaded.records) for loaded in loaded_files),
        "skipped_source_records": sum(len(loaded.skipped) for loaded in loaded_files),
        "invalid_source_records": sum(len(loaded.invalid) for loaded in loaded_files),
        "output_chunks": len(chunks),
        "avg_token_estimate": _avg(token_values),
        "p50_token_estimate": _percentile(token_values, 50),
        "p90_token_estimate": _percentile(token_values, 90),
        "max_token_estimate": max(token_values) if token_values else 0,
        "avg_duration_sec": _avg(duration_values),
        "p50_duration_sec": _percentile(duration_values, 50),
        "p90_duration_sec": _percentile(duration_values, 90),
        "max_duration_sec": max(duration_values) if duration_values else 0,
        "views": dict(sorted(views.items())),
        "quality_flags": dict(sorted(quality_flags.items())),
        "missing_optional_fields": dict(sorted(missing_optional.items())),
    }


def manifest_rows(loaded_files, chunks: list[dict[str, object]]) -> list[dict[str, object]]:
    output_counts: Counter[str] = Counter(str(chunk["view"]) for chunk in chunks)
    rows = []
    for loaded in loaded_files:
        rows.append(
            {
                "view": loaded.view,
                "input_file": loaded.input_file,
                "parse_status": "success",
                "input_records": loaded.input_records,
                "valid_source_records": len(loaded.records),
                "skipped_records": len(loaded.skipped),
                "output_chunks": output_counts[loaded.view],
                "invalid_source_records": len(loaded.invalid),
            }
        )
    return rows


def _avg(values: Iterable[float | int]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return round(float(mean(values)), 3)


def _percentile(values: list[float | int], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    rank = (len(ordered) - 1) * percentile / 100
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    value = ordered[lower] * (1 - weight) + ordered[upper] * weight
    return round(value, 3)
