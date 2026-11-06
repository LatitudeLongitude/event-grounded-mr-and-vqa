from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .hash_utils import read_sha256_file, sha256_file
from .hf_inventory import RemoteFile
from .logging_utils import ProgressReporter
from .manifests import write_non_video_manifest
from .time_utils import timestamp_for_filename, utc_now_iso


@dataclass(frozen=True)
class LocalCheck:
    ok: bool
    reason: str
    verified_by: str = "size"


def copy_file_atomic(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with source.open("rb") as src, tmp.open("wb") as dst:
        for chunk in iter(lambda: src.read(8 * 1024 * 1024), b""):
            dst.write(chunk)
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(tmp, target)


def sync_non_video_files(
    cfg: Any,
    inventory: dict[str, Any],
    downloader: Any,
    logger_or_stores: Any,
    maybe_logger: Any | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Sync non-video files using inventory + local filesystem facts only.

    The fourth/fifth parameters keep backward call compatibility with older
    code paths, but any legacy ProgressStore objects are deliberately ignored.
    Old progress/events files remain historical artifacts and are not updated
    by this lightweight non-video path.
    """

    logger = maybe_logger if maybe_logger is not None else logger_or_stores
    raw_files = [RemoteFile(**item) for item in inventory.get("non_video_files", [])]
    by_path = {item.path: item for item in raw_files}
    rows: list[dict[str, Any]] = []
    ordered = sorted(raw_files, key=lambda item: (not item.is_sha256, item.path))
    reporter = ProgressReporter(
        logger,
        "sync-non-video",
        cfg.logging.command_progress_every_items,
        cfg.logging.command_progress_interval_seconds,
    )
    reporter.start("syncing non-video files using local tree state", total=len(ordered))
    for index, item in enumerate(ordered, start=1):
        reporter.maybe(
            index,
            "checking non-video file",
            current_path=item.path,
            synced=len(rows),
            total=len(ordered),
        )
        target = cfg.processed_file_path(item.path)
        raw = cfg.local_raw_path(item.path)
        sha_path = _local_sidecar_path(cfg, item, by_path)

        processed_check = _validate_local_non_video(target, item.size, sha_path)
        if processed_check.ok:
            rows.append(_row(item, target, "already_verified", processed_check.verified_by))
            continue
        if processed_check.reason.startswith("corrupt"):
            _quarantine_file(cfg.paths.processed_data_dir, target, "processed", logger)

        raw_check = _validate_local_non_video(raw, item.size, sha_path)
        if raw_check.ok:
            if dry_run:
                logger.info("[dry-run] would copy non-video from raw cache %s", item.path)
                rows.append(_row(item, target, "dry_run_copy_from_raw", raw_check.verified_by))
                continue
            copy_file_atomic(raw, target)
            final_check = _validate_local_non_video(target, item.size, sha_path)
            if not final_check.ok:
                raise RuntimeError(f"processed copy failed validation for {item.path}: {final_check.reason}")
            rows.append(_row(item, target, "copied_from_raw", final_check.verified_by))
            continue
        if raw_check.reason.startswith("corrupt"):
            _quarantine_file(cfg.paths.raw_cache_dir, raw, "raw", logger)

        if dry_run:
            logger.info("[dry-run] would download non-video %s", item.path)
            rows.append(_row(item, target, "dry_run_download", "not_checked"))
            continue

        downloader.download_file(item.path, item.size, "non_video", sha256_path=sha_path, dry_run=False)
        downloaded_check = _validate_local_non_video(raw, item.size, sha_path)
        if not downloaded_check.ok:
            raise RuntimeError(f"downloaded non-video failed validation for {item.path}: {downloaded_check.reason}")
        copy_file_atomic(raw, target)
        final_check = _validate_local_non_video(target, item.size, sha_path)
        if not final_check.ok:
            raise RuntimeError(f"processed non-video failed validation for {item.path}: {final_check.reason}")
        rows.append(_row(item, target, "processed_verified", final_check.verified_by))
    write_non_video_manifest(cfg, rows)
    reporter.done(len(ordered), "non-video sync finished", synced=len(rows))
    return rows


def assess_non_video_files(cfg: Any, inventory: dict[str, Any]) -> dict[str, Any]:
    raw_files = [RemoteFile(**item) for item in inventory.get("non_video_files", [])]
    by_path = {item.path: item for item in raw_files}
    report: dict[str, Any] = {
        "total": len(raw_files),
        "processed_verified": 0,
        "raw_reusable": 0,
        "missing": [],
        "corrupt": [],
        "sha_checked": 0,
    }
    for item in raw_files:
        sha_path = _local_sidecar_path(cfg, item, by_path)
        processed = _validate_local_non_video(cfg.processed_file_path(item.path), item.size, sha_path)
        if processed.ok:
            report["processed_verified"] += 1
            if processed.verified_by == "sha256":
                report["sha_checked"] += 1
            continue
        raw = _validate_local_non_video(cfg.local_raw_path(item.path), item.size, sha_path)
        if raw.ok:
            report["raw_reusable"] += 1
            if raw.verified_by == "sha256":
                report["sha_checked"] += 1
            continue
        if processed.reason.startswith("corrupt") or raw.reason.startswith("corrupt"):
            report["corrupt"].append({"remote_path": item.path, "processed_reason": processed.reason, "raw_reason": raw.reason})
        else:
            report["missing"].append(item.path)
    report["passed"] = not report["missing"] and not report["corrupt"]
    return report


def _local_sidecar_path(cfg: Any, item: RemoteFile, by_path: dict[str, RemoteFile]) -> Path | None:
    if item.is_sha256:
        return None
    sidecar_remote = by_path.get(item.path + ".sha256")
    if sidecar_remote is None:
        return None
    processed_sidecar = cfg.processed_file_path(sidecar_remote.path)
    if processed_sidecar.exists():
        return processed_sidecar
    raw_sidecar = cfg.local_raw_path(sidecar_remote.path)
    if raw_sidecar.exists():
        return raw_sidecar
    return None


def _validate_local_non_video(path: Path, expected_size: int | None, sha_path: Path | None) -> LocalCheck:
    if not path.exists():
        return LocalCheck(False, "missing")
    try:
        actual_size = path.stat().st_size
    except OSError as exc:
        return LocalCheck(False, f"corrupt_stat_failed:{exc}")
    if expected_size is not None and actual_size != expected_size:
        return LocalCheck(False, f"corrupt_size_mismatch:local={actual_size}:expected={expected_size}")
    expected_sha = read_sha256_file(sha_path) if sha_path is not None else None
    if expected_sha:
        actual_sha = sha256_file(path)
        if actual_sha.lower() != expected_sha.lower():
            return LocalCheck(False, f"corrupt_sha256_mismatch:local={actual_sha}:expected={expected_sha}")
        return LocalCheck(True, "ok", "sha256")
    return LocalCheck(True, "ok", "size")


def _quarantine_file(root: Path, path: Path, label: str, logger: Any) -> Path | None:
    if not path.exists():
        return None
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = Path(path.name)
    quarantine = root / "_quarantine" / "non_video" / label / rel
    quarantine = quarantine.with_name(f"{quarantine.name}.{timestamp_for_filename()}.bad")
    quarantine.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(quarantine))
    logger.warning("quarantined corrupt non-video %s to %s", path, quarantine)
    return quarantine


def _row(item: RemoteFile, target: Path, status: str, verified_by: str) -> dict[str, Any]:
    return {
        "remote_path": item.path,
        "processed_path": str(target),
        "size": item.size,
        "is_root_file": item.is_root_file,
        "is_sha256": item.is_sha256,
        "is_sidecar": item.is_sidecar,
        "status": status,
        "verified_by": verified_by,
        "checked_at": utc_now_iso(),
    }
