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
            "input_row_number",
            "embedding_id",
            "embedding_chunk_id",
            "day",
            "view",
            "start_time",
            "end_time",
            "duration_sec",
            "start_hour",
            "end_hour",
            "hours_json",
            "text_for_embedding",
            "text_for_display",
            "char_len",
            "token_estimate",
            "word_like_count",
            "source_chunk_ids_json",
            "source_chunks_json",
            "source_files_json",
            "quality_flags_json",
            "contains_invalid_source_chunk",
            "contains_repetition_compressed_source",
            "chunking_method",
            "chunking_version",
            "embedding",
            "embedding_norm_raw",
            "model_id",
            "model_root",
            "model_fingerprint",
            "created_at_utc",
        }
        if not required.issubset(schema.names):
            raise ValueError("Parquet schema is missing required columns")
        embedding_type = schema.field("embedding").type
        if not pa.types.is_fixed_size_list(embedding_type) or embedding_type.list_size != EMBEDDING_DIM:
            raise ValueError("Embedding field is not fixed length 4096")
        if embedding_type.value_type != pa.float16():
            raise ValueError("Embedding field value type is not float16")
        row_count = parquet_file.metadata.num_rows
        if expected_rows is not None and row_count != expected_rows:
            raise ValueError(f"Expected {expected_rows} rows, found {row_count}")
        table = parquet_file.read(columns=["schema_version", "input_row_number", "embedding_chunk_id", "embedding", "embedding_norm_raw"])
        if any(value != SCHEMA_VERSION for value in table.column("schema_version").to_pylist()):
            raise ValueError("Unexpected schema_version")
        row_numbers = table.column("input_row_number").to_pylist()
        if row_numbers != sorted(row_numbers):
            raise ValueError("input_row_number is not sorted")
        ids = table.column("embedding_chunk_id").to_pylist()
        if len(ids) != len(set(ids)):
            raise ValueError("embedding_chunk_id is not unique")
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
