from __future__ import annotations

import csv
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path | str, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def canonical_json_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_text(payload)


def read_yaml(path: Path | str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def write_yaml(path: Path | str, data: object) -> None:
    text = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    atomic_write_text(Path(path), text)


def _fsync_dir(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _cleanup_tmp(path: Path) -> None:
    if not path.parent.exists():
        return
    for candidate in path.parent.glob(f"{path.name}.tmp*"):
        if candidate.is_file():
            try:
                candidate.unlink()
            except PermissionError:
                if os.name != "nt":
                    raise


def _tmp_path_for(target: Path) -> Path:
    return target.with_name(f"{target.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")


def _best_effort_unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except PermissionError:
        if os.name != "nt":
            raise


def atomic_write_bytes(path: Path | str, data: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _cleanup_tmp(target)
    tmp = _tmp_path_for(target)
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(tmp, target)
    except PermissionError:
        if os.name != "nt":
            raise
        # Some Windows sandboxes deny atomic replace even inside writable roots.
        # The official server path is Linux and still uses os.replace above.
        target.write_bytes(data)
        _best_effort_unlink(tmp)
    _fsync_dir(target.parent)


def atomic_write_text(path: Path | str, text: str) -> None:
    atomic_write_bytes(Path(path), text.encode("utf-8"))


def atomic_write_json(path: Path | str, data: object) -> None:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2)
    atomic_write_text(Path(path), payload + "\n")


def atomic_write_jsonl(path: Path | str, rows: Iterable[Mapping[str, object]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _cleanup_tmp(target)
    tmp = _tmp_path_for(target)
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(tmp, target)
    except PermissionError:
        if os.name != "nt":
            raise
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(tmp.read_text(encoding="utf-8"))
        _best_effort_unlink(tmp)
    _fsync_dir(target.parent)


def atomic_write_csv(path: Path | str, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _cleanup_tmp(target)
    tmp = _tmp_path_for(target)
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(tmp, target)
    except PermissionError:
        if os.name != "nt":
            raise
        with target.open("w", encoding="utf-8", newline="") as handle:
            handle.write(tmp.read_text(encoding="utf-8"))
        _best_effort_unlink(tmp)
    _fsync_dir(target.parent)


def file_record(path: Path | str) -> dict[str, object]:
    target = Path(path)
    return {
        "path": target.as_posix(),
        "exists": target.exists(),
        "size": target.stat().st_size if target.exists() else None,
        "sha256": sha256_file(target) if target.exists() and target.is_file() else None,
    }
