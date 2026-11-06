from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .schema import EMBEDDING_DIM, SCHEMA_VERSION, validate_embedding_array


@dataclass(frozen=True)
class ShardValidation:
    path: Path
    valid: bool
    row_count: int
    file_size: Optional[int] = None
    sha256: Optional[str] = None
    error: Optional[str] = None


def _embedding_numpy(table) -> np.ndarray:
    combined = table.column("embedding").combine_chunks()
    values = combined.values.to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=np.float32).reshape(len(combined), EMBEDDING_DIM)


def _file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_shard(
    path: Path | str,
    *,
    expected_rows: Optional[int] = None,
    expected_file_size: Optional[int] = None,
    expected_sha256: Optional[str] = None,
    compute_sha256: bool = False,
) -> ShardValidation:
    target = Path(path)
    row_count = 0
    file_size = None
    actual_sha256 = None
    try:
        file_size = target.stat().st_size
        if expected_file_size is not None and file_size != expected_file_size:
            raise ValueError(f"Expected file size {expected_file_size}, found {file_size}")
        parquet_file = pq.ParquetFile(target)
        schema = parquet_file.schema_arrow
        required = {
            "schema_version",
            "embedding_id",
            "day",
            "source",
            "video_hour_stem",
            "frame_index",
            "timestamp_second",
            "absolute_second",
            "timestamp_hms",
            "frame_rel_path",
            "embedding",
            "embedding_norm_raw",
            "model_id",
            "model_code_sha256",
            "preprocess_id",
        }
        if not required.issubset(schema.names):
            raise ValueError("Parquet schema is missing required columns")
        if schema.field("embedding").type.list_size != EMBEDDING_DIM:
            raise ValueError("Embedding field is not fixed length 1536")
        if schema.field("embedding").type.value_type != pa.float32():
            raise ValueError("Embedding field value type is not float32")
        row_count = parquet_file.metadata.num_rows
        if expected_rows is not None and row_count != expected_rows:
            raise ValueError(f"Expected {expected_rows} rows, found {row_count}")
        table = parquet_file.read(columns=["schema_version", "frame_index", "embedding", "embedding_norm_raw"])
        if any(value != SCHEMA_VERSION for value in table.column("schema_version").to_pylist()):
            raise ValueError("Unexpected schema_version")
        indices = table.column("frame_index").to_pylist()
        if indices != sorted(indices) or len(indices) != len(set(indices)):
            raise ValueError("frame_index is not unique and sorted")
        validate_embedding_array(_embedding_numpy(table), require_normalized=True)
        raw_norms = np.asarray(table.column("embedding_norm_raw").to_pylist(), dtype=np.float32)
        if not np.isfinite(raw_norms).all() or np.any(raw_norms <= 0):
            raise ValueError("Raw embedding norms must be finite and positive")
        if compute_sha256 or expected_sha256 is not None:
            actual_sha256 = _file_sha256(target)
        if expected_sha256 is not None and actual_sha256 != expected_sha256.lower():
            raise ValueError(f"Expected SHA256 {expected_sha256.lower()}, found {actual_sha256}")
        return ShardValidation(target, True, row_count, file_size, actual_sha256)
    except Exception as exc:
        return ShardValidation(
            target,
            False,
            row_count,
            file_size,
            actual_sha256,
            error=f"{type(exc).__name__}: {exc}",
        )
