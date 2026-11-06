from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .schema import (
    SCHEMA_VERSION,
    SPARSE_ENCODING,
    SPARSE_INDEX_DTYPE,
    SPARSE_VALUE_DTYPE,
    build_schema,
    decode_sparse_binary,
)
from .inventory import ChunkRecord, stable_json


REQUIRED_FILE_METADATA = {
    "schema_version",
    "run_id",
    "invocation_id",
    "encoder_backend",
    "model_id",
    "model_root",
    "model_fingerprint",
    "chunks_sha256",
    "max_length",
    "use_fp16",
    "document_instruction",
    "output_type",
    "dense_saved",
    "colbert_saved",
    "sparse_encoding",
    "created_at_utc",
}

JSON_LIST_COLUMNS = (
    "hours_json",
    "source_chunk_ids_json",
    "source_chunks_json",
    "source_files_json",
    "quality_flags_json",
)

VALIDATION_COLUMNS = (
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
    "text_sha256",
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
    "bge_sparse_indices_i32",
    "bge_sparse_values_f16",
    "sparse_nnz",
    "sparse_index_dtype",
    "sparse_value_dtype",
    "sparse_encoding",
    "sparse_min_value_before_cast",
    "sparse_max_value_before_cast",
    "sparse_sum_values_before_cast",
    "sparse_l1_norm_before_cast",
    "duplicate_index_merged_count",
    "model_id",
    "model_root",
    "model_fingerprint",
    "created_run_id",
)


@dataclass(frozen=True)
class ShardValidation:
    path: Path
    valid: bool
    row_count: int
    file_size: Optional[int] = None
    sha256: Optional[str] = None
    error: Optional[str] = None


def _file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_metadata(metadata) -> dict[str, str]:
    if not metadata:
        raise ValueError("Parquet file metadata is missing")
    decoded = {}
    for key, value in metadata.items():
        decoded[key.decode("utf-8")] = value.decode("utf-8")
    missing = sorted(REQUIRED_FILE_METADATA - set(decoded))
    if missing:
        raise ValueError(f"Parquet file metadata is missing required keys: {', '.join(missing)}")
    return decoded


def _metadata_bool(metadata: dict[str, str], key: str) -> bool:
    value = metadata[key].strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(f"Parquet metadata {key} must be true or false")
    return value == "true"


def _require_close(name: str, stored, observed: float) -> None:
    stored_value = float(stored)
    if not np.isfinite(stored_value):
        raise ValueError(f"{name} must be finite")
    if not np.isclose(stored_value, observed, rtol=0.02, atol=5e-5):
        raise ValueError(f"{name} does not match decoded Float16 sparse values")


def _source_float32(value):
    if value is None:
        return None
    return float(np.float32(value))


def _expected_source_payload(record: ChunkRecord) -> dict:
    raw = record.raw
    return {
        "input_row_number": record.input_row_number,
        "embedding_id": record.embedding_chunk_id,
        "embedding_chunk_id": record.embedding_chunk_id,
        "day": record.day,
        "view": record.view,
        "start_time": _source_float32(raw["start_time"]),
        "end_time": _source_float32(raw["end_time"]),
        "duration_sec": _source_float32(raw["duration_sec"]),
        "start_hour": raw["start_hour"],
        "end_hour": raw["end_hour"],
        "hours_json": stable_json(raw["hours"]),
        "text_for_embedding": record.text_for_embedding,
        "text_for_display": raw["text_for_display"],
        "text_sha256": hashlib.sha256(record.text_for_embedding.encode("utf-8")).hexdigest(),
        "char_len": raw["char_len"],
        "token_estimate": raw["token_estimate"],
        "word_like_count": raw["word_like_count"],
        "source_chunk_ids_json": stable_json(raw["source_chunk_ids"]),
        "source_chunks_json": stable_json(raw["source_chunks"]),
        "source_files_json": stable_json(raw["source_files"]),
        "quality_flags_json": stable_json(raw["quality_flags"]),
        "contains_invalid_source_chunk": raw["contains_invalid_source_chunk"],
        "contains_repetition_compressed_source": raw["contains_repetition_compressed_source"],
        "chunking_method": raw["chunking_method"],
        "chunking_version": raw["chunking_version"],
    }


def _validate_source_payload(rows: list[dict], expected_records) -> None:
    expected_rows = [_expected_source_payload(record) for record in expected_records]
    if len(rows) != len(expected_rows):
        raise ValueError(
            f"Parquet payload row count does not match source records: {len(rows)} != {len(expected_rows)}"
        )
    source_fields = tuple(expected_rows[0]) if expected_rows else ()
    for row, expected in zip(rows, expected_rows):
        for field in source_fields:
            if row[field] != expected[field]:
                raise ValueError(
                    f"Parquet payload does not match source field {field} "
                    f"for input_row_number {expected['input_row_number']}"
                )


def validate_shard(
    path: Path | str,
    *,
    expected_rows: Optional[int] = None,
    expected_file_size: Optional[int] = None,
    expected_sha256: Optional[str] = None,
    compute_sha256: bool = False,
    expected_records=None,
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
        expected_schema = build_schema()
        missing = sorted(set(expected_schema.names) - set(schema.names))
        if missing:
            raise ValueError(f"Parquet schema is missing required columns: {', '.join(missing)}")
        for field in expected_schema:
            actual_type = schema.field(field.name).type
            if actual_type != field.type:
                raise ValueError(f"{field.name} must have type {field.type}, found {actual_type}")
        for name in ("bge_sparse_indices_i32", "bge_sparse_values_f16"):
            if schema.field(name).type != pa.binary():
                raise ValueError(f"{name} must be a binary column")

        metadata = _decode_metadata(schema.metadata)
        if metadata["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Parquet metadata has an unexpected schema_version")
        if metadata["output_type"] != "learned_sparse":
            raise ValueError("Parquet metadata output_type must be learned_sparse")
        if metadata["document_instruction"] != "not-used":
            raise ValueError("Parquet metadata document_instruction must be not-used")
        if _metadata_bool(metadata, "dense_saved"):
            raise ValueError("Parquet metadata indicates forbidden dense output")
        if _metadata_bool(metadata, "colbert_saved"):
            raise ValueError("Parquet metadata indicates forbidden ColBERT output")
        if metadata["sparse_encoding"] != SPARSE_ENCODING:
            raise ValueError("Parquet metadata has an unexpected sparse_encoding")

        row_count = parquet_file.metadata.num_rows
        if expected_rows is not None and row_count != expected_rows:
            raise ValueError(f"Expected {expected_rows} rows, found {row_count}")

        table = parquet_file.read(columns=list(VALIDATION_COLUMNS))
        row_numbers = table.column("input_row_number").to_pylist()
        if row_numbers != sorted(row_numbers):
            raise ValueError("input_row_number is not sorted")
        ids = table.column("embedding_chunk_id").to_pylist()
        if len(ids) != len(set(ids)):
            raise ValueError("embedding_chunk_id is not unique")

        rows = table.to_pylist()
        for row in rows:
            if row["schema_version"] != SCHEMA_VERSION:
                raise ValueError("Unexpected schema_version")
            if row["embedding_id"] != row["embedding_chunk_id"]:
                raise ValueError("embedding_id must equal embedding_chunk_id")
            expected_text_sha256 = hashlib.sha256(row["text_for_embedding"].encode("utf-8")).hexdigest()
            if row["text_sha256"] != expected_text_sha256:
                raise ValueError("text_sha256 does not match text_for_embedding")
            for name in JSON_LIST_COLUMNS:
                try:
                    parsed = json.loads(row[name])
                except (TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"{name} is not valid JSON") from exc
                if not isinstance(parsed, list):
                    raise ValueError(f"{name} must encode a JSON list")

            if row["sparse_index_dtype"] != SPARSE_INDEX_DTYPE:
                raise ValueError("Unexpected sparse_index_dtype")
            if row["sparse_value_dtype"] != SPARSE_VALUE_DTYPE:
                raise ValueError("Unexpected sparse_value_dtype")
            if row["sparse_encoding"] != SPARSE_ENCODING:
                raise ValueError("Unexpected sparse_encoding")
            indices, values = decode_sparse_binary(
                row["bge_sparse_indices_i32"],
                row["bge_sparse_values_f16"],
            )
            if len(indices) != int(row["sparse_nnz"]) or len(values) != int(row["sparse_nnz"]):
                raise ValueError("Sparse binary lengths do not match sparse_nnz")
            if int(row["duplicate_index_merged_count"]) < 0:
                raise ValueError("duplicate_index_merged_count cannot be negative")

            _require_close("sparse_min_value_before_cast", row["sparse_min_value_before_cast"], float(values.min()))
            _require_close("sparse_max_value_before_cast", row["sparse_max_value_before_cast"], float(values.max()))
            _require_close("sparse_sum_values_before_cast", row["sparse_sum_values_before_cast"], float(values.sum()))
            _require_close("sparse_l1_norm_before_cast", row["sparse_l1_norm_before_cast"], float(np.abs(values).sum()))

            for column, metadata_key in (
                ("model_id", "model_id"),
                ("model_root", "model_root"),
                ("model_fingerprint", "model_fingerprint"),
                ("created_run_id", "run_id"),
            ):
                if str(row[column]) != metadata[metadata_key]:
                    raise ValueError(f"{column} does not match Parquet file metadata")

        if expected_records is not None:
            _validate_source_payload(rows, expected_records)

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
