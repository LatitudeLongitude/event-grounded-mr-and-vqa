from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np


def as_vector(value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError("vector must be one-dimensional")
    if arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError("vector must be finite and non-empty")
    if float(np.linalg.norm(arr)) == 0.0:
        raise ValueError("vector must be non-zero")
    return arr


def l2_normalize(value) -> np.ndarray:
    arr = as_vector(value)
    return arr / float(np.linalg.norm(arr))


def cosine(a, b) -> float:
    left = as_vector(a)
    right = as_vector(b)
    if left.shape != right.shape:
        raise ValueError("vectors must have matching shapes")
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0 or not math.isfinite(denom):
        raise ValueError("invalid vector norm")
    return float(np.dot(left, right) / denom)


def cosine_distance(a, b) -> float:
    return 1.0 - cosine(a, b)


def deterministic_uniform_sample(items: list, max_items: int) -> list:
    if max_items <= 0:
        raise ValueError("max_items must be positive")
    if len(items) <= max_items:
        return list(items)
    if max_items == 1:
        return [items[0]]
    indices = np.linspace(0, len(items) - 1, num=max_items)
    return [items[int(round(index))] for index in indices]


def weighted_jaccard(left: Mapping[int, float], right: Mapping[int, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    numerator = 0.0
    denominator = 0.0
    for key in keys:
        a = max(0.0, float(left.get(key, 0.0)))
        b = max(0.0, float(right.get(key, 0.0)))
        numerator += min(a, b)
        denominator += max(a, b)
    return 0.0 if denominator == 0 else numerator / denominator
