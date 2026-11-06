from __future__ import annotations

from collections.abc import Iterable


Interval = tuple[float, float]


def union_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    valid = sorted((float(a), float(b)) for a, b in intervals if float(a) < float(b))
    if not valid:
        return []
    merged: list[Interval] = [valid[0]]
    for start, end in valid[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def interval_duration(intervals: Iterable[Interval]) -> float:
    return float(sum(end - start for start, end in intervals))


def aligned_duration_sec(intervals: Iterable[Interval]) -> float:
    return interval_duration(union_intervals(intervals))


def overlap_duration(a: Interval, b: Interval) -> float:
    start = max(float(a[0]), float(b[0]))
    end = min(float(a[1]), float(b[1]))
    return max(0.0, end - start)
