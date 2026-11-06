from __future__ import annotations

from typing import Optional

import numpy as np


def robust_z(values, *, clip: float = 6.0, epsilon: float = 1e-6) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    result = np.full(arr.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(arr)
    if not mask.any():
        return result
    finite = arr[mask]
    center = float(np.median(finite))
    mad = float(np.median(np.abs(finite - center)))
    scale = 1.4826 * mad
    if scale <= epsilon:
        q75, q25 = np.percentile(finite, [75, 25])
        scale = float((q75 - q25) / 1.349)
    if scale <= epsilon:
        scale = float(np.std(finite))
    if scale <= epsilon:
        scale = epsilon
    z = (finite - center) / scale
    result[mask] = np.clip(z, -clip, clip)
    return result


def weighted_mean_available(items: list[tuple[float, Optional[float]]]) -> Optional[float]:
    numerator = 0.0
    denominator = 0.0
    for weight, value in items:
        if value is None or not np.isfinite(float(value)):
            continue
        numerator += float(weight) * float(value)
        denominator += float(weight)
    if denominator == 0:
        return None
    return numerator / denominator
