from __future__ import annotations

import numpy as np
import pyarrow as pa


EMBEDDING_DIM = 4096
SCHEMA_VERSION = "qwen3_text_dense_v1"
PARQUET_EMBEDDING_VALUE_TYPE = "float16"
NORM_ATOL = 2e-3


def build_schema(metadata: dict | None = None) -> pa.Schema:
    schema = pa.schema(
        [
            pa.field("schema_version", pa.string(), nullable=False),
            pa.field("input_row_number", pa.int32(), nullable=False),
            pa.field("embedding_id", pa.string(), nullable=False),
            pa.field("embedding_chunk_id", pa.string(), nullable=False),
            pa.field("day", pa.string(), nullable=False),
            pa.field("view", pa.string(), nullable=False),
            pa.field("start_time", pa.float32(), nullable=True),
            pa.field("end_time", pa.float32(), nullable=True),
            pa.field("duration_sec", pa.float32(), nullable=True),
            pa.field("start_hour", pa.string(), nullable=False),
            pa.field("end_hour", pa.string(), nullable=False),
            pa.field("hours_json", pa.string(), nullable=False),
            pa.field("text_for_embedding", pa.string(), nullable=False),
            pa.field("text_for_display", pa.string(), nullable=False),
            pa.field("char_len", pa.int32(), nullable=False),
            pa.field("token_estimate", pa.int32(), nullable=False),
            pa.field("word_like_count", pa.int32(), nullable=False),
            pa.field("source_chunk_ids_json", pa.string(), nullable=False),
            pa.field("source_chunks_json", pa.string(), nullable=False),
            pa.field("source_files_json", pa.string(), nullable=False),
            pa.field("quality_flags_json", pa.string(), nullable=False),
            pa.field("contains_invalid_source_chunk", pa.bool_(), nullable=False),
            pa.field("contains_repetition_compressed_source", pa.bool_(), nullable=False),
            pa.field("chunking_method", pa.string(), nullable=False),
            pa.field("chunking_version", pa.string(), nullable=False),
            pa.field("embedding", pa.list_(pa.float16(), EMBEDDING_DIM), nullable=False),
            pa.field("embedding_norm_raw", pa.float32(), nullable=False),
            pa.field("model_id", pa.string(), nullable=False),
            pa.field("model_root", pa.string(), nullable=False),
            pa.field("model_fingerprint", pa.string(), nullable=False),
            pa.field("created_at_utc", pa.string(), nullable=False),
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
    if require_normalized and not np.allclose(norms, 1.0, atol=NORM_ATOL, rtol=0):
        raise ValueError("Embeddings must be L2 normalized")
    return array
