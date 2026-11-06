from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from .config import AppConfig
from .encoder import resolved_versions
from .fingerprint import model_fingerprint
from .gpu_guard import assert_gpus_idle, gpu_report, query_gpus, validate_gpu_policy
from .inventory import discover_inventory
from .paths import configure_runtime_environment


def _disk(path: Path) -> dict:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path.resolve()),
        "total": usage.total,
        "used": usage.used,
        "free": usage.free,
        "percent_used": round(usage.used / usage.total * 100, 2),
    }


def run_preflight(
    config: AppConfig,
    *,
    gpu_ids=None,
    require_idle_gpus: bool = True,
    hash_weights: bool = False,
) -> dict:
    config.validate()
    paths = config.paths
    selected_ids = tuple(config.runtime.gpu_ids if gpu_ids is None else gpu_ids)
    if require_idle_gpus:
        validate_gpu_policy(selected_ids)
    for path, description in (
        (paths.project_root, "project_root"),
        (paths.module_root, "module_root"),
        (paths.data_root, "data_root"),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{description} is missing: {path}")
    if not paths.chunks_path.is_file():
        raise FileNotFoundError(f"chunks.jsonl is missing: {paths.chunks_path}")
    if not paths.model_root.is_dir():
        raise FileNotFoundError(f"model_root is missing: {paths.model_root}")
    configure_runtime_environment(paths.output_root)
    paths.output_root.mkdir(parents=True, exist_ok=True)
    disks = [_disk(paths.output_root), _disk(paths.data_root), _disk(paths.model_root)]
    if os.name != "nt":
        disks.append(_disk(Path("/")))
    output_disk = disks[0]
    minimum_free = config.runtime.minimum_output_free_gib * 1024**3
    if output_disk["free"] < minimum_free:
        raise RuntimeError(f"Output filesystem has less than {config.runtime.minimum_output_free_gib} GiB free")
    inventory = discover_inventory(config)
    if inventory.invalid or inventory.duplicate_ids:
        raise RuntimeError("Inventory contains invalid rows or duplicate IDs")
    gpus = ()
    if require_idle_gpus:
        gpus = query_gpus()
        assert_gpus_idle(
            gpus,
            selected_ids,
            max_utilization=config.runtime.max_gpu_utilization,
            min_free_mib=config.runtime.min_free_memory_mib,
        )
    tmp_files = [str(path) for path in paths.output_root.rglob("*.tmp.*")]
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "paths": {name: str(value) for name, value in paths.__dict__.items()},
        "environment": configure_runtime_environment(paths.output_root),
        "disks": disks,
        "model": model_fingerprint(paths.model_root, hash_weights=hash_weights),
        "versions": resolved_versions(),
        "inventory": {
            "total_rows": inventory.total_rows,
            "valid_rows": inventory.valid_rows,
            "candidate_shards": len(inventory.shards),
            "chunks_sha256": inventory.chunks_sha256,
            "meta_summary_sha256": inventory.meta_summary_sha256,
        },
        "gpus": gpu_report(gpus),
        "selected_gpu_ids": list(selected_ids) if require_idle_gpus else [],
        "tmp_files": tmp_files,
    }
