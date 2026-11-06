from __future__ import annotations

import numpy as np


def decode_sparse_binary(indices_bytes: bytes, values_bytes: bytes) -> dict[int, float]:
    indices = np.frombuffer(indices_bytes, dtype="<i4").astype(np.int32)
    values = np.frombuffer(values_bytes, dtype="<f2").astype(np.float32)
    return {int(index): float(value) for index, value in zip(indices, values)}
