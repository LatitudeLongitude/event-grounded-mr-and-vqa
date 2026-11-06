from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import pyarrow.parquet as pq

from .manifest import params_hash
from .parquet_io import atomic_write_json, cleanup_tmp, file_sha256


@dataclass(frozen=True)
class StagePaths:
    run_root: Path
    manifests_dir: Path

    def stage_manifest(self, stage: str) -> Path:
        return self.manifests_dir / f"{stage}_manifest.json"

    def state_path(self) -> Path:
        return self.manifests_dir / "run_state.json"


def stage_paths(run_root: Path) -> StagePaths:
    return StagePaths(run_root=run_root, manifests_dir=run_root / "manifests")


def output_schema_fingerprint(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        schema = pq.ParquetFile(path).schema_arrow
    except Exception:
        return None
    return normalize_schema_fingerprint(str(schema))


def normalize_schema_fingerprint(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return value.replace("element:", "item:").replace("child 0, element:", "child 0, item:")


def _resolve_manifest_record_path(raw_path: str, manifest_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    run_root = manifest_path.parent.parent.resolve()
    for base in (run_root, *run_root.parents):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (run_root / path).resolve()


def _record_matches_current_file(record: Mapping, manifest_path: Path) -> bool:
    raw_path = record.get("path")
    if not raw_path:
        return False
    path = _resolve_manifest_record_path(str(raw_path), manifest_path)
    expected_exists = record.get("exists", True)
    if expected_exists is False:
        return not path.exists()
    if not path.exists():
        return False
    expected_sha = record.get("sha256")
    if expected_sha and expected_sha != file_sha256(path):
        return False
    return True


def stage_complete(manifest_path: Path, *, params: Mapping, output_paths: list[Path]) -> bool:
    if not manifest_path.exists():
        return False
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if data.get("status") != "complete":
        return False
    if data.get("params_hash") != params_hash(params):
        return False
    for record in data.get("inputs", []):
        if not _record_matches_current_file(record, manifest_path):
            return False
    output_by_path = {Path(item["path"]).name: item for item in data.get("outputs", []) if item.get("exists", True)}
    for path in output_paths:
        cleanup_tmp(path)
        if not path.exists():
            return False
        record = output_by_path.get(path.name)
        if not record:
            return False
        if record.get("sha256") != file_sha256(path):
            return False
        if record.get("schema") and normalize_schema_fingerprint(record.get("schema")) != output_schema_fingerprint(path):
            return False
    return True


def update_run_state(run_root: Path, stage: str, manifest_ref: str) -> None:
    paths = stage_paths(run_root)
    paths.manifests_dir.mkdir(parents=True, exist_ok=True)
    state = {}
    if paths.state_path().exists():
        state = json.loads(paths.state_path().read_text(encoding="utf-8"))
    state[stage] = {"manifest": manifest_ref}
    atomic_write_json(paths.state_path(), state)
