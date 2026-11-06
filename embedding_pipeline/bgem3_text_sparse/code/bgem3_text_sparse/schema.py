from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Mapping, Sequence

import numpy as np
import pyarrow as pa


SCHEMA_VERSION = "bgem3_text_sparse_v1"
SPARSE_INDEX_DTYPE = "int32"
SPARSE_VALUE_DTYPE = "float16"
SPARSE_ENCODING = "little_endian_int32_indices_float16_values"
PARQUET_EMBEDDING_VALUE_TYPE = SPARSE_VALUE_DTYPE


@dataclass(frozen=True)
class SparseVector:
    indices: np.ndarray
    values_f32: np.ndarray
    duplicate_index_merged_count: int = 0
    key_format: str = "vocab_id"
    raw_value_dtype: str = "float32"

    @property
    def nnz(self) -> int:
        return int(self.indices.shape[0])


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
            pa.field("text_sha256", pa.string(), nullable=False),
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
            pa.field("bge_sparse_indices_i32", pa.binary(), nullable=False),
            pa.field("bge_sparse_values_f16", pa.binary(), nullable=False),
            pa.field("sparse_nnz", pa.int32(), nullable=False),
            pa.field("sparse_index_dtype", pa.string(), nullable=False),
            pa.field("sparse_value_dtype", pa.string(), nullable=False),
            pa.field("sparse_encoding", pa.string(), nullable=False),
            pa.field("sparse_min_value_before_cast", pa.float32(), nullable=False),
            pa.field("sparse_max_value_before_cast", pa.float32(), nullable=False),
            pa.field("sparse_sum_values_before_cast", pa.float32(), nullable=False),
            pa.field("sparse_l1_norm_before_cast", pa.float32(), nullable=False),
            pa.field("duplicate_index_merged_count", pa.int32(), nullable=False),
            pa.field("model_id", pa.string(), nullable=False),
            pa.field("model_root", pa.string(), nullable=False),
            pa.field("model_fingerprint", pa.string(), nullable=False),
            pa.field("flagembedding_version", pa.string(), nullable=False),
            pa.field("transformers_version", pa.string(), nullable=False),
            pa.field("torch_version", pa.string(), nullable=False),
            pa.field("created_at_utc", pa.string(), nullable=False),
            pa.field("created_run_id", pa.string(), nullable=False),
        ]
    )
    encoded = {str(key).encode(): str(value).encode() for key, value in (metadata or {}).items()}
    return schema.with_metadata(encoded)


def _token_to_id(tokenizer, token: str) -> int:
    if tokenizer is None:
        raise ValueError("Tokenizer is required when lexical weight keys are token strings")
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        token_id = tokenizer.convert_tokens_to_ids(token)
    elif hasattr(tokenizer, "get_vocab"):
        token_id = tokenizer.get_vocab().get(token)
    else:
        raise ValueError("Tokenizer cannot convert token strings to vocabulary IDs")
    if token_id is None or int(token_id) < 0:
        raise ValueError(f"Cannot map token to a valid vocabulary ID: {token!r}")
    return int(token_id)


def parse_lexical_weights(weights: Mapping[object, object], *, tokenizer=None) -> SparseVector:
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError("Sparse lexical weights must be a non-empty mapping")

    parsed: dict[int, float] = {}
    duplicates = 0
    key_format = "vocab_id"
    raw_dtypes = set()
    for key, value in weights.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            key_format = "token"
            index = _token_to_id(tokenizer, str(key))
        if index < 0 or index > np.iinfo(np.int32).max:
            raise ValueError(f"Sparse index is outside int32 range: {index}")
        array_value = np.asarray(value)
        raw_dtypes.add(str(array_value.dtype))
        scalar = float(array_value.astype(np.float32))
        if index in parsed:
            duplicates += 1
            parsed[index] = max(parsed[index], scalar)
        else:
            parsed[index] = scalar

    indices = np.asarray(sorted(parsed), dtype=np.int32)
    values = np.asarray([parsed[int(index)] for index in indices], dtype=np.float32)
    validate_sparse_vector(indices, values)
    return SparseVector(
        indices=indices,
        values_f32=values,
        duplicate_index_merged_count=duplicates,
        key_format=key_format,
        raw_value_dtype=",".join(sorted(raw_dtypes)) if raw_dtypes else "unknown",
    )


def validate_sparse_vector(indices, values) -> tuple[np.ndarray, np.ndarray]:
    idx = np.asarray(indices, dtype=np.int32)
    val = np.asarray(values, dtype=np.float32)
    if idx.ndim != 1 or val.ndim != 1:
        raise ValueError("Sparse indices and values must be one-dimensional")
    if len(idx) != len(val):
        raise ValueError("Sparse indices and values must have the same length")
    if len(idx) == 0:
        raise ValueError("Sparse vector must be non-empty")
    if len(set(int(item) for item in idx.tolist())) != len(idx):
        raise ValueError("Sparse indices must be unique")
    if not np.isfinite(val).all():
        raise ValueError("Sparse values must be finite")
    if np.all(val == 0):
        raise ValueError("Sparse values must not be all zero")
    return idx, val


def encode_sparse_binary(vector: SparseVector) -> tuple[bytes, bytes]:
    indices, values = validate_sparse_vector(vector.indices, vector.values_f32)
    index_bytes = indices.astype("<i4", copy=False).tobytes()
    value_bytes = values.astype("<f2").tobytes()
    return index_bytes, value_bytes


def decode_sparse_binary(indices_bytes: bytes, values_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    if len(indices_bytes) % 4:
        raise ValueError("Sparse index binary length must be divisible by 4")
    if len(values_bytes) % 2:
        raise ValueError("Sparse value binary length must be divisible by 2")
    indices = np.frombuffer(indices_bytes, dtype="<i4").astype(np.int32)
    values = np.frombuffer(values_bytes, dtype="<f2").astype(np.float32)
    validate_sparse_vector(indices, values)
    return indices, values


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def sparse_value_stats(vectors: Sequence[SparseVector]) -> dict:
    nnz = [vector.nnz for vector in vectors]
    values_before = []
    values_after = []
    raw_dtypes = set()
    nan_count = inf_count = zero_count = empty_count = duplicate_count = 0
    nan_count_after = inf_count_after = zero_count_after = 0
    for vector in vectors:
        duplicate_count += vector.duplicate_index_merged_count
        raw_dtypes.add(vector.raw_value_dtype)
        if vector.nnz == 0:
            empty_count += 1
        arr = np.asarray(vector.values_f32, dtype=np.float32)
        cast_arr = arr.astype(np.float16).astype(np.float32)
        nan_count += int(np.isnan(arr).sum())
        inf_count += int(np.isinf(arr).sum())
        zero_count += int((arr == 0).sum())
        nan_count_after += int(np.isnan(cast_arr).sum())
        inf_count_after += int(np.isinf(cast_arr).sum())
        zero_count_after += int((cast_arr == 0).sum())
        values_before.extend(float(item) for item in arr[np.isfinite(arr)])
        values_after.extend(float(item) for item in cast_arr[np.isfinite(cast_arr)])
    return {
        "schema_version": SCHEMA_VERSION,
        "raw_sparse_value_dtypes": sorted(raw_dtypes),
        "nnz_min": min(nnz) if nnz else 0,
        "nnz_max": max(nnz) if nnz else 0,
        "nnz_mean": float(mean(nnz)) if nnz else 0.0,
        "nnz_p50": _percentile(nnz, 0.5),
        "nnz_p95": _percentile(nnz, 0.95),
        "value_min_before_cast": min(values_before) if values_before else None,
        "value_max_before_cast": max(values_before) if values_before else None,
        "value_mean_before_cast": float(mean(values_before)) if values_before else None,
        "value_p50_before_cast": _percentile(values_before, 0.5),
        "value_p95_before_cast": _percentile(values_before, 0.95),
        "value_p99_before_cast": _percentile(values_before, 0.99),
        "value_min_after_cast": min(values_after) if values_after else None,
        "value_max_after_cast": max(values_after) if values_after else None,
        "value_mean_after_cast": float(mean(values_after)) if values_after else None,
        "value_p50_after_cast": _percentile(values_after, 0.5),
        "value_p95_after_cast": _percentile(values_after, 0.95),
        "value_p99_after_cast": _percentile(values_after, 0.99),
        "empty_sparse_count": empty_count,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "zero_value_count": zero_count,
        "nan_count_after_cast": nan_count_after,
        "inf_count_after_cast": inf_count_after,
        "zero_value_count_after_cast": zero_count_after,
        "duplicate_index_merged_count": duplicate_count,
    }
