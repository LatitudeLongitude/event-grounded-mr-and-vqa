from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from .config import AppConfig
from .encoder import COMPUTE_PRECISION_POLICY, resolved_versions
from .fingerprint import model_fingerprint
from .gpu_guard import assert_gpus_idle, gpu_report, query_gpus, validate_gpu_policy
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
    validate_gpu_policy(selected_ids, config.runtime.max_memory_fraction_per_gpu)
    for path, description in (
        (paths.project_root, "project_root"),
        (paths.data_root, "data_root"),
        (paths.model_root, "model_root"),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{description} is missing: {path}")
    if not config.summary_path or not config.summary_path.is_file():
        raise FileNotFoundError(f"Summary manifest is missing: {config.summary_path}")
    configure_runtime_environment(paths.output_root)
    paths.output_root.mkdir(parents=True, exist_ok=True)
    disks = [_disk(paths.output_root), _disk(paths.data_root), _disk(paths.model_root)]
    if os.name != "nt":
        disks.append(_disk(Path("/")))
    output_disk = disks[0]
    if output_disk["free"] < 20 * 1024**3:
        raise RuntimeError("Output filesystem has less than 20 GiB free")
    gpus = query_gpus()
    if require_idle_gpus:
        assert_gpus_idle(
            gpus,
            selected_ids,
            max_utilization=config.runtime.max_gpu_utilization,
            min_free_mib=config.runtime.min_free_memory_mib,
        )
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "paths": {name: str(value) for name, value in paths.__dict__.items()},
        "disks": disks,
        "model": model_fingerprint(paths.model_root, hash_weights=hash_weights),
        "versions": resolved_versions(),
        "compute_precision_policy": COMPUTE_PRECISION_POLICY,
        "gpus": gpu_report(gpus),
        "selected_gpu_ids": list(selected_ids),
    }
