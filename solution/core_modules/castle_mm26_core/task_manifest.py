from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import AppConfig
from .io_utils import atomic_write_json, atomic_write_text, canonical_json_hash, now_utc, sha256_file
from .models import TaskQuery


def write_task_manifest(
    config: AppConfig,
    query: TaskQuery,
    output_dir: Path,
    *,
    status: str,
    required_files: Sequence[str],
    stage_reports: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    for name in required_files:
        path = output_dir / name
        files[name] = _file_record(path)
    for stage_manifest in sorted(output_dir.glob("**/stage_manifest.json")):
        key = stage_manifest.relative_to(output_dir).as_posix()
        files[key] = _file_record(stage_manifest)
    manifest = {
        "schema_version": "castle_mm26_task_manifest_v1",
        "status": status,
        "task_id": query.task_id,
        "task_type": query.task_type,
        "written_at_utc": now_utc(),
        "input_hashes": {
            "config_path": config.path.as_posix(),
            "config_sha256": sha256_file(config.path),
            "task_query_sha256": canonical_json_hash(query.asdict()),
            "profile_registry_sha256": config.profile_registry_sha256,
            "resolved_profiles_sha256": canonical_json_hash(query.resolved_profiles),
        },
        "profile_refs": {
            "recall_profile": query.recall_profile,
            "scoring_profile": query.scoring_profile,
            "threshold_profile": query.threshold_profile,
            "output_profile": query.output_profile,
        },
        "required_files": list(required_files),
        "files": files,
        "stage_reports_sha256": canonical_json_hash(stage_reports or {}),
        "completed_marker_written_after_manifest": status == "completed",
    }
    if extra:
        manifest.update(dict(extra))
    atomic_write_json(output_dir / "task_manifest.json", manifest)
    return manifest


def write_completed_marker(output_dir: Path) -> None:
    atomic_write_text(output_dir / "completed.marker", "completed\n")


def read_task_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
        "size": path.stat().st_size if path.exists() and path.is_file() else None,
    }
