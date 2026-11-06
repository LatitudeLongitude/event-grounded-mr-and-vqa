from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TimestampCleanResult:
    raw_timestamp: Any
    start_time: float | None
    end_time: float | None
    duration_sec: float | None
    timestamp_valid: bool
    flags: list[str]


def clean_timestamp(
    raw_timestamp: Any,
    *,
    min_duration_sec: float = 1.0,
    max_duration_sec: float = 120.0,
) -> TimestampCleanResult:
    flags: list[str] = []

    if raw_timestamp is None:
        flags.extend(["missing_timestamp", "missing_start_time", "missing_end_time"])
        return TimestampCleanResult(raw_timestamp, None, None, None, False, flags)

    if not isinstance(raw_timestamp, (list, tuple)):
        flags.extend(["missing_timestamp", "missing_start_time", "missing_end_time"])
        return TimestampCleanResult(raw_timestamp, None, None, None, False, flags)

    start_time = _to_float(raw_timestamp[0]) if len(raw_timestamp) >= 1 else None
    end_time = _to_float(raw_timestamp[1]) if len(raw_timestamp) >= 2 else None

    if start_time is None:
        flags.append("missing_start_time")
    if end_time is None:
        flags.append("missing_end_time")

    duration_sec: float | None = None
    if start_time is not None and end_time is not None:
        if end_time < start_time:
            flags.append("end_before_start")
        else:
            duration_sec = round(end_time - start_time, 6)
            if duration_sec < min_duration_sec:
                flags.append("too_short_duration")
            if duration_sec > max_duration_sec:
                flags.append("too_long_duration")

    timestamp_valid = not flags
    return TimestampCleanResult(
        raw_timestamp=raw_timestamp,
        start_time=start_time,
        end_time=end_time,
        duration_sec=duration_sec,
        timestamp_valid=timestamp_valid,
        flags=flags,
    )


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
