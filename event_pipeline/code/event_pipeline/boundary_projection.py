from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import Optional


def triangular_kernel(score: Optional[float], distance: float, radius: float) -> Optional[float]:
    if score is None or radius <= 0:
        return None
    distance = abs(float(distance))
    if distance > radius:
        return None
    return float(score) * max(0.0, 1.0 - distance / radius)


def project_to_seconds(seconds: list[float], boundary_second: float, score: Optional[float], radius: float) -> dict[float, float]:
    if score is None:
        return {}
    lo = bisect_left(seconds, boundary_second - radius)
    hi = bisect_right(seconds, boundary_second + radius)
    result = {}
    for second in seconds[lo:hi]:
        value = triangular_kernel(score, second - boundary_second, radius)
        if value is not None:
            result[second] = value
    return result
