from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from .parquet_io import atomic_write_json, canonical_json_hash, file_sha256
from .paths import to_ref


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def params_hash(params: Mapping) -> str:
    return canonical_json_hash(params)


def file_record(path: Path, project_root: Path) -> dict:
    if not path.exists():
        return {"path": to_ref(path, project_root), "exists": False}
    return {
        "path": to_ref(path, project_root),
        "exists": True,
        "size": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def refresh_manifest_file_records(manifest_path: Path, *, project_root: Path, paths: list[Path]) -> None:
    if not manifest_path.exists():
        return
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {to_ref(path, project_root): file_record(path, project_root) for path in paths}
    records.update({path.resolve().as_posix(): file_record(path, project_root) for path in paths})
    changed = False
    for section in ("inputs", "outputs"):
        for item in data.get(section, []):
            replacement = records.get(str(item.get("path")))
            if replacement is None:
                continue
            schema = item.get("schema")
            item.update(replacement)
            if schema is not None:
                item["schema"] = schema
            changed = True
    if changed:
        atomic_write_json(manifest_path, data)


def write_stage_manifest(
    manifest_path: Path,
    *,
    stage: str,
    params: Mapping,
    inputs: list[dict],
    outputs: list[dict],
    schema_names: list[str],
    project_root: Path,
) -> dict:
    data = {
        "stage": stage,
        "created_at_utc": utc_now(),
        "params_hash": params_hash(params),
        "params": dict(params),
        "inputs": inputs,
        "outputs": outputs,
        "output_schemas": schema_names,
        "status": "complete",
    }
    text = str(data)
    if "retrieval_test" in text or "exploration" in text:
        raise ValueError("runtime manifest must not reference retrieval_test or exploration")
    atomic_write_json(manifest_path, data)
    return data
