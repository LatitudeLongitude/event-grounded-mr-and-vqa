from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Optional, Union

import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_PARQUET_BATCH_SIZE = 8192


def file_sha256(path: Union[Path, str], chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def canonical_json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def cleanup_tmp(path: Path) -> None:
    parent = path.parent
    if not parent.exists():
        return
    for candidate in parent.glob(f"{path.name}.tmp*"):
        if candidate.is_file():
            candidate.unlink()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cleanup_tmp(path)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _fsync_directory(path.parent)


def atomic_write_json(path: Path, data: Mapping) -> None:
    atomic_write_bytes(path, json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"))


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cleanup_tmp(path)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("wb") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8"))
            handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _fsync_directory(path.parent)


def rows_to_table(rows: Sequence[Mapping[str, object]], schema: Optional[pa.Schema] = None) -> pa.Table:
    if schema is None:
        return pa.Table.from_pylist(list(rows))
    if not rows:
        return pa.Table.from_arrays([pa.array([], type=field.type) for field in schema], schema=schema)
    normalized = []
    field_names = set(schema.names)
    for row in rows:
        normalized.append({name: row.get(name) for name in field_names})
    return pa.Table.from_pylist(normalized, schema=schema)


def write_table(path: Path, rows: Sequence[Mapping[str, object]], *, schema: Optional[pa.Schema] = None, metadata: Optional[Mapping[str, object]] = None, row_group_size: int = 8192) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    cleanup_tmp(path)
    tmp = path.with_name(f"{path.name}.tmp")
    table = rows_to_table(rows, schema)
    if metadata:
        encoded = {str(key).encode("utf-8"): str(value).encode("utf-8") for key, value in metadata.items()}
        table = table.replace_schema_metadata(encoded)
    pq.write_table(table, tmp, compression="zstd", compression_level=3, row_group_size=row_group_size, version="2.6")
    with tmp.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _fsync_directory(path.parent)
    return {"path": path.as_posix(), "row_count": table.num_rows, "file_size": path.stat().st_size, "sha256": file_sha256(path)}


class AtomicParquetWriter:
    def __init__(self, path: Path, schema: pa.Schema, *, metadata: Optional[Mapping[str, object]] = None, row_group_size: int = DEFAULT_PARQUET_BATCH_SIZE):
        self.path = path
        self.schema = schema
        self.metadata = metadata or {}
        self.row_group_size = row_group_size
        self.tmp = path.with_name(f"{path.name}.tmp")
        self.writer: Optional[pq.ParquetWriter] = None
        self.row_count = 0

    def __enter__(self) -> "AtomicParquetWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cleanup_tmp(self.path)
        schema = self.schema
        if self.metadata:
            encoded = {str(key).encode("utf-8"): str(value).encode("utf-8") for key, value in self.metadata.items()}
            schema = schema.with_metadata(encoded)
        self.writer = pq.ParquetWriter(self.tmp, schema, compression="zstd", compression_level=3, version="2.6")
        return self

    def write_rows(self, rows: Sequence[Mapping[str, object]]) -> None:
        if not rows:
            return
        if self.writer is None:
            raise RuntimeError("writer is not open")
        table = rows_to_table(rows, self.schema)
        self.writer.write_table(table, row_group_size=self.row_group_size)
        self.row_count += table.num_rows

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.writer is not None:
            self.writer.close()
        if exc_type is not None:
            if self.tmp.exists():
                self.tmp.unlink()
            return
        if self.row_count == 0:
            pq.write_table(rows_to_table([], self.schema), self.tmp, compression="zstd", compression_level=3, version="2.6")
        with self.tmp.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(self.tmp, self.path)
        _fsync_directory(self.path.parent)


def read_table(path: Path, columns: Optional[list[str]] = None) -> pa.Table:
    return pq.read_table(path, columns=columns)


def iter_pylist_batches(path: Path, columns: Optional[list[str]] = None, batch_size: int = DEFAULT_PARQUET_BATCH_SIZE) -> Iterable[list[dict]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pylist()


def iter_pylist_rows(path: Path, columns: Optional[list[str]] = None, batch_size: int = DEFAULT_PARQUET_BATCH_SIZE) -> Iterable[dict]:
    for batch in iter_pylist_batches(path, columns=columns, batch_size=batch_size):
        yield from batch


def read_pylist(path: Path, columns: Optional[list[str]] = None, batch_size: int = DEFAULT_PARQUET_BATCH_SIZE) -> list[dict]:
    rows: list[dict] = []
    for batch in iter_pylist_batches(path, columns=columns, batch_size=batch_size):
        rows.extend(batch)
    return rows


def read_column_pylist(path: Path, column: str, batch_size: int = DEFAULT_PARQUET_BATCH_SIZE) -> list:
    return [row[column] for row in iter_pylist_rows(path, columns=[column], batch_size=batch_size)]
