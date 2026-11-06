from __future__ import annotations

import os
import shutil
from pathlib import Path


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__"}]
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    return total


def free_disk_bytes(path: Path) -> int:
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def human_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    number = float(value)
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if number < 1024 or unit == "PB":
            return f"{number:.2f} {unit}"
        number /= 1024


def ensure_enough_free_space(path: Path, min_free_gb: int) -> tuple[bool, int]:
    free = free_disk_bytes(path)
    return free >= min_free_gb * 1024**3, free
