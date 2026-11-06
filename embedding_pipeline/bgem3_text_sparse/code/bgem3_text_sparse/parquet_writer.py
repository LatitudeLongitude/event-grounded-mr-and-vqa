from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from .schema import build_schema
from .validation import validate_shard


@dataclass(frozen=True)
class ShardWriteResult:
    path: Path
    row_count: int
    file_size: int
    sha256: str


def file_sha256(path: Path | str, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _table_from_rows(rows: Sequence[Mapping[str, object]], metadata: Mapping[str, object]) -> pa.Table:
    if not rows:
        raise ValueError("Cannot write an empty shard")
    schema = build_schema(dict(metadata))
    arrays = [pa.array([row[field.name] for row in rows], type=field.type) for field in schema]
    return pa.Table.from_arrays(arrays, schema=schema)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_embedding_shard(
    output_path: Path | str,
    rows: Iterable[Mapping[str, object]],
    *,
    run_id: str,
    metadata: Mapping[str, object],
    compression: str = "zstd",
    compression_level: int = 3,
    row_group_size: int = 512,
) -> ShardWriteResult:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    row_list = sorted(list(rows), key=lambda row: int(row["input_row_number"]))
    if len({row["embedding_chunk_id"] for row in row_list}) != len(row_list):
        raise ValueError("Duplicate embedding_chunk_id in shard")
    file_metadata = dict(metadata)
    file_metadata.setdefault("created_at_utc", datetime.now(timezone.utc).isoformat())
    table = _table_from_rows(row_list, file_metadata)
    temporary = output.with_name(f"{output.name}.tmp.{run_id}")
    if temporary.exists():
        raise FileExistsError(f"Temporary shard already exists: {temporary}")
    pq.write_table(
        table,
        temporary,
        compression=compression,
        compression_level=compression_level,
        row_group_size=row_group_size,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
    )
    validation = validate_shard(temporary, expected_rows=len(row_list))
    if not validation.valid:
        raise ValueError(f"Temporary shard validation failed: {validation.error}")
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    _fsync_directory(output.parent)
    size = output.stat().st_size
    return ShardWriteResult(output, len(row_list), size, file_sha256(output))
