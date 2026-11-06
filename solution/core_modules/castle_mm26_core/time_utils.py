from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Sequence


HMS_RE = re.compile(r"^(?P<h>[0-2][0-9]):(?P<m>[0-5][0-9]):(?P<s>[0-5][0-9])$")


@dataclass(frozen=True)
class Interval:
    start: float
    end: float

    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def parse_hms(value: str) -> int:
    match = HMS_RE.fullmatch(value)
    if not match:
        raise ValueError(f"invalid HH:MM:SS time: {value!r}")
    hour = int(match.group("h"))
    minute = int(match.group("m"))
    second = int(match.group("s"))
    if hour > 23:
        raise ValueError("official submission time cannot use 24:00:00")
    return hour * 3600 + minute * 60 + second


def format_submission_hms(seconds: float | int) -> str:
    if not math.isfinite(float(seconds)):
        raise ValueError(f"invalid seconds: {seconds!r}")
    value = int(seconds)
    if value < 0 or value >= 86400:
        raise ValueError(f"official CSV time outside [0, 86400): {seconds!r}")
    return f"{value // 3600:02d}:{(value % 3600) // 60:02d}:{value % 60:02d}"


def normalize_output_interval(start: float, end: float, *, coverage_end: float | None = None) -> Interval:
    out_start = math.floor(float(start))
    out_end = math.ceil(float(end))
    if out_end <= out_start:
        out_end = out_start + 1
    if coverage_end is not None:
        out_end = min(out_end, math.floor(float(coverage_end)))
    if out_end >= 86400:
        out_end = 86399
    if out_end <= out_start:
        raise ValueError(f"cannot create a positive official interval from {start}-{end}")
    return Interval(float(out_start), float(out_end))


def overlap_duration(left: Interval | tuple[float, float], right: Interval | tuple[float, float]) -> float:
    left_start, left_end = _pair(left)
    right_start, right_end = _pair(right)
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def interval_iou(left: Interval | tuple[float, float], right: Interval | tuple[float, float]) -> float:
    inter = overlap_duration(left, right)
    if inter <= 0:
        return 0.0
    left_start, left_end = _pair(left)
    right_start, right_end = _pair(right)
    union = max(left_end, right_end) - min(left_start, right_start)
    return inter / max(union, 1e-9)


def merge_seconds_to_intervals(seconds: Iterable[float], *, gap_threshold: float = 1.5) -> list[Interval]:
    ordered = sorted(float(value) for value in seconds if math.isfinite(float(value)))
    if not ordered:
        return []
    intervals: list[Interval] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value - previous > gap_threshold:
            intervals.append(Interval(start, previous + 1.0))
            start = value
        previous = value
    intervals.append(Interval(start, previous + 1.0))
    return intervals


def clip_to_coverage(candidate: Interval | tuple[float, float], coverage: Sequence[Interval]) -> list[Interval]:
    start, end = _pair(candidate)
    clipped: list[Interval] = []
    for interval in coverage:
        clipped_start = max(start, interval.start)
        clipped_end = min(end, interval.end)
        if clipped_end - clipped_start >= 1.0:
            clipped.append(Interval(clipped_start, clipped_end))
    return clipped


def nms_intervals(rows: Sequence[dict], *, iou_threshold: float, score_key: str = "final_score") -> list[dict]:
    ordered = sorted(rows, key=lambda row: (-float(row.get(score_key, 0.0)), row.get("source", ""), row.get("start_second", 0)))
    kept: list[dict] = []
    for row in ordered:
        current = (float(row["start_second"]), float(row["end_second"]))
        duplicate = False
        for selected in kept:
            if row.get("task_id") != selected.get("task_id") or row.get("day") != selected.get("day") or row.get("source") != selected.get("source"):
                continue
            if interval_iou(current, (float(selected["start_second"]), float(selected["end_second"]))) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(dict(row))
    return sorted(kept, key=lambda row: (row.get("task_id", ""), int(row.get("day", 0)), row.get("source", ""), float(row.get("start_second", 0))))


def _pair(value: Interval | tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, Interval):
        return value.start, value.end
    return float(value[0]), float(value[1])
