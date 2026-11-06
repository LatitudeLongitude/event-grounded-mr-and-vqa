from __future__ import annotations

import numpy as np
import pyarrow as pa


EMBEDDING_DIM = 1536
SCHEMA_VERSION = "tipsv2_image_cls_v1"
PREPROCESS_ID = "tipsv2_rgb_resize448_bilinear_antialias_v1"
PARQUET_EMBEDDING_VALUE_TYPE = "float32"


def build_schema(metadata: dict | None = None) -> pa.Schema:
    schema = pa.schema(
        [
            pa.field("schema_version", pa.string(), nullable=False),
            pa.field("embedding_id", pa.string(), nullable=False),
            pa.field("day", pa.int8(), nullable=False),
            pa.field("source", pa.string(), nullable=False),
            pa.field("video_hour_stem", pa.string(), nullable=False),
            pa.field("frame_index", pa.int32(), nullable=False),
            pa.field("timestamp_second", pa.float32(), nullable=False),
            pa.field("absolute_second", pa.float64(), nullable=False),
            pa.field("timestamp_hms", pa.string(), nullable=False),
            pa.field("frame_rel_path", pa.string(), nullable=False),
            pa.field("embedding", pa.list_(pa.float32(), EMBEDDING_DIM), nullable=False),
            pa.field("embedding_norm_raw", pa.float32(), nullable=False),
            pa.field("model_id", pa.string(), nullable=False),
            pa.field("model_code_sha256", pa.string(), nullable=False),
            pa.field("preprocess_id", pa.string(), nullable=False),
        ]
    )
    encoded = {str(key).encode(): str(value).encode() for key, value in (metadata or {}).items()}
    return schema.with_metadata(encoded)


def validate_embedding_array(values, *, require_normalized: bool = True) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"Embeddings must have shape (N, {EMBEDDING_DIM})")
    if not np.isfinite(array).all():
        raise ValueError("Embeddings must contain only finite values")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms == 0):
        raise ValueError("Embeddings must not contain zero vectors")
    if require_normalized and not np.allclose(norms, 1.0, atol=1e-5, rtol=0):
        raise ValueError("Embeddings must be L2 normalized")
    return array
