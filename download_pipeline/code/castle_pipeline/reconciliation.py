from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .constants import (
    FILE_PROCESSED_VERIFIED,
    VIDEO_DONE,
    VIDEO_DOWNLOADED_VERIFIED,
    VIDEO_FAILED_NEED_MANUAL,
    VIDEO_FAILED_RETRYABLE,
    VIDEO_PROCESSED_VERIFIED,
    VIDEO_RAW_DELETED_VERIFIED,
)
from .ffmpeg_utils import probe_video
from .hf_inventory import RemoteFile, inventory_invalid_reason, inventory_remote_files
from .logging_utils import ProgressReporter
from .manifests import rewrite_final_video_artifacts, update_video_raw_deleted_metadata, video_artifacts_need_raw_deleted_at
from .time_utils import timestamp_for_filename, utc_now_iso
from .verifier import verify_processed_video_dir, write_verify_result


def load_inventory_if_exists(cfg: Any) -> dict[str, Any] | None:
    for path in [cfg.manifests_dir / "remote_inventory.json", cfg.paths.code_dir / "remote_inventory.json"]:
        if path.exists():
            inventory = json.loads(path.read_text(encoding="utf-8"))
            if inventory_invalid_reason(inventory):
                continue
            return inventory
    return None


def reconcile_progress(
    cfg: Any,
    stores: list[Any],
    inventory: dict[str, Any] | None,
    logger: Any,
    backup: bool = True,
    include_non_video: bool = True,
) -> dict[str, Any]:
    if backup:
        for store in stores:
            store.backup()
    inventory = inventory or load_inventory_if_exists(cfg)
    report = {"repaired": [], "manual": [], "checked": 0}
    if not inventory:
        logger.warning("no inventory available; reconciliation limited to progress files")
        return report
    files = inventory_remote_files(inventory)
    reporter = ProgressReporter(
        logger,
        "repair",
        cfg.logging.command_progress_every_items,
        cfg.logging.command_progress_interval_seconds,
    )
    reporter.start("reconciling progress files with filesystem", total=len(files))
    small_files_since_flush = 0
    batch_size = max(1, int(cfg.processing.non_video_small_file_progress_batch_size))
    try:
        for index, remote in enumerate(files, start=1):
            reporter.maybe(index, "checking remote file", current_path=remote.path, total=len(files))
            report["checked"] += 1
            if remote.is_video:
                if small_files_since_flush:
                    _flush_all(stores)
                    small_files_since_flush = 0
                _reconcile_video(cfg, stores, remote, report, logger)
            elif include_non_video:
                result = _reconcile_non_video(cfg, stores, remote, report)
                if result == "batched":
                    small_files_since_flush += 1
                    if small_files_since_flush >= batch_size:
                        _flush_all(stores)
                        small_files_since_flush = 0
                elif result == "immediate":
                    small_files_since_flush = 0
    finally:
        _flush_all(stores)
    reporter.done(report["checked"], "reconciliation finished", repaired=len(report["repaired"]), manual=len(report["manual"]))
    return report


def cleanup_processing_tmp_dirs(cfg: Any, logger: Any) -> list[str]:
    removed: list[str] = []
    for tmp in cfg.paths.processed_data_dir.rglob("*.__processing_tmp__"):
        if tmp.is_dir():
            quarantined = tmp.with_name(f"{tmp.name}.__incomplete_{timestamp_for_filename()}__")
            shutil.move(str(tmp), str(quarantined))
            removed.append(str(quarantined))
            logger.warning("moved processing tmp dir to quarantine: %s", quarantined)
    return removed


def _reconcile_video(cfg: Any, stores: list[Any], remote: RemoteFile, report: dict[str, Any], logger: Any) -> None:
    raw = cfg.local_raw_path(remote.path)
    final = cfg.processed_video_dir(remote.path)
    raw_verified = False
    if raw.exists() and (remote.size is None or raw.stat().st_size == remote.size):
        try:
            probe_video(cfg.tools.ffprobe, raw)
            raw_verified = True
        except Exception as exc:
            report["repaired"].append({"remote_path": remote.path, "status": VIDEO_FAILED_RETRYABLE, "reason": f"ffprobe failed: {exc}"})
    processed = verify_processed_video_dir(final, expected_frame_count=None, require_verify_result=True, sample_images=False)
    if final.exists() and not processed.passed:
        try:
            summary = rewrite_final_video_artifacts(final, remote.path, {"status": VIDEO_PROCESSED_VERIFIED})
            expected = summary.get("expected_frame_count")
            tolerance = int(summary.get("tolerance") or 2)
            repaired = verify_processed_video_dir(
                final,
                expected_frame_count=int(expected) if expected is not None else None,
                tolerance=tolerance,
                sample_images=False,
            )
            write_verify_result(final, repaired)
            processed = verify_processed_video_dir(
                final,
                expected_frame_count=int(expected) if expected is not None else None,
                tolerance=tolerance,
                require_verify_result=True,
                sample_images=False,
            )
            if processed.passed:
                report["repaired"].append({"remote_path": remote.path, "status": "artifact_paths_rewritten"})
        except Exception as exc:
            report["manual"].append({"remote_path": remote.path, "errors": [f"artifact repair failed: {exc}"]})
    if processed.passed and not raw.exists():
        if not _backfill_raw_deleted_at_if_needed(cfg, remote.path, final, report, logger):
            _set_all(stores, remote.path, "video", VIDEO_FAILED_NEED_MANUAL, "raw missing and processed verified, but raw deletion metadata repair failed", raw_path=str(raw), processed_path=str(final))
            return
        _set_all(stores, remote.path, "video", VIDEO_DONE, "reconciled from verified processed output and no raw", raw_path=str(raw), processed_path=str(final))
        report["repaired"].append({"remote_path": remote.path, "status": VIDEO_DONE})
    elif processed.passed and raw.exists():
        _set_all(stores, remote.path, "video", VIDEO_PROCESSED_VERIFIED, "reconciled verified processed output; raw still present", raw_path=str(raw), processed_path=str(final))
        report["repaired"].append({"remote_path": remote.path, "status": VIDEO_PROCESSED_VERIFIED})
    elif raw_verified:
        _set_all(stores, remote.path, "video", VIDEO_DOWNLOADED_VERIFIED, "reconciled verified raw video; processing needed", raw_path=str(raw), processed_path=str(final))
        report["repaired"].append({"remote_path": remote.path, "status": VIDEO_DOWNLOADED_VERIFIED})
    elif raw.exists() and not raw_verified:
        _set_all(stores, remote.path, "video", VIDEO_FAILED_RETRYABLE, "raw exists but size does not match inventory", raw_path=str(raw), processed_path=str(final))
        report["repaired"].append({"remote_path": remote.path, "status": VIDEO_FAILED_RETRYABLE})
    elif final.exists() and not processed.passed:
        _set_all(stores, remote.path, "video", VIDEO_FAILED_NEED_MANUAL, "processed output incomplete and raw missing", raw_path=str(raw), processed_path=str(final))
        report["manual"].append({"remote_path": remote.path, "errors": processed.errors})


def _reconcile_non_video(cfg: Any, stores: list[Any], remote: RemoteFile, report: dict[str, Any]) -> str:
    processed = cfg.processed_file_path(remote.path)
    raw = cfg.local_raw_path(remote.path)
    if processed.exists() and (remote.size is None or processed.stat().st_size == remote.size):
        if _stores_already_reconciled_non_video(cfg, stores, remote, raw, processed):
            return "skipped"
        is_small = _is_small_non_video_file(cfg, remote)
        _set_all(
            stores,
            remote.path,
            "non_video",
            FILE_PROCESSED_VERIFIED,
            "reconciled processed non-video",
            raw_path=str(raw),
            processed_path=str(processed),
            flush=not is_small,
        )
        report["repaired"].append({"remote_path": remote.path, "status": FILE_PROCESSED_VERIFIED})
        return "batched" if is_small else "immediate"
    return "none"


def _stores_already_reconciled_non_video(cfg: Any, stores: list[Any], remote: RemoteFile, raw: Path, processed: Path) -> bool:
    if len(stores) != 3:
        return False
    if not processed.exists():
        return False
    if remote.size is not None and processed.stat().st_size != remote.size:
        return False
    expected_raw_path = str(raw)
    expected_processed_path = str(processed)
    signatures: list[tuple[str | None, str | None, str | None, str | None, str | None]] = []
    for store in stores:
        files = getattr(store, "data", {}).get("files", {})
        entry = files.get(remote.path)
        if not isinstance(entry, dict):
            return False
        if entry.get("file_type") != "non_video":
            return False
        if entry.get("status") != FILE_PROCESSED_VERIFIED:
            return False
        if entry.get("remote_path") != remote.path:
            return False
        if entry.get("processed_path") != expected_processed_path:
            return False
        raw_path = entry.get("raw_path")
        if raw_path is not None and raw_path != expected_raw_path:
            return False
        signatures.append(
            (
                entry.get("file_type"),
                entry.get("status"),
                entry.get("remote_path"),
                entry.get("processed_path"),
                raw_path if "raw_path" in entry else None,
            )
        )
    return len(set(signatures)) == 1


def _is_small_non_video_file(cfg: Any, remote: RemoteFile) -> bool:
    return (
        remote.file_type == "non_video"
        and remote.size is not None
        and remote.size <= cfg.processing.non_video_small_file_threshold_bytes
    )


def _backfill_raw_deleted_at_if_needed(cfg: Any, remote_path: str, final: Path, report: dict[str, Any], logger: Any) -> bool:
    if not video_artifacts_need_raw_deleted_at(final):
        return True
    event_timestamp = _latest_raw_deleted_verified_event_timestamp(cfg, remote_path)
    raw_deleted_at = event_timestamp or utc_now_iso()
    source = "event_log" if event_timestamp else "repair_time"
    try:
        summary = update_video_raw_deleted_metadata(
            final,
            raw_deleted_at,
            status=VIDEO_RAW_DELETED_VERIFIED,
            raw_deleted_at_source=source,
        )
        expected = summary.get("expected_frame_count")
        tolerance = int(summary.get("tolerance") or 2)
        repaired = verify_processed_video_dir(
            final,
            expected_frame_count=int(expected) if expected is not None else None,
            tolerance=tolerance,
            require_verify_result=True,
            sample_images=False,
        )
        write_verify_result(final, repaired)
        if not repaired.passed:
            report["manual"].append({"remote_path": remote_path, "errors": [f"raw_deleted_at backfill verification failed: {err}" for err in repaired.errors]})
            logger.warning("raw_deleted_at backfill verification failed for %s: %s", remote_path, repaired.errors)
            return False
        report["repaired"].append(
            {
                "remote_path": remote_path,
                "status": "raw_deleted_at_backfilled",
                "raw_deleted_at": raw_deleted_at,
                "raw_deleted_at_source": source,
            }
        )
        return True
    except Exception as exc:
        report["manual"].append({"remote_path": remote_path, "errors": [f"raw_deleted_at backfill failed: {exc}"]})
        logger.warning("raw_deleted_at backfill failed for %s: %s", remote_path, exc)
        return False


def _latest_raw_deleted_verified_event_timestamp(cfg: Any, remote_path: str) -> str | None:
    timestamps: list[str] = []
    paths = [
        getattr(cfg, "video_pipeline_events_path", None),
        getattr(cfg, "video_download_events_path", None),
        getattr(cfg, "video_processing_events_path", None),
    ]
    for path in [item for item in paths if item is not None]:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("remote_path") != remote_path:
                    continue
                if event.get("new_status") != VIDEO_RAW_DELETED_VERIFIED:
                    continue
                timestamp = event.get("timestamp")
                if isinstance(timestamp, str) and timestamp:
                    timestamps.append(timestamp)
    return sorted(timestamps)[-1] if timestamps else None


def _flush_all(stores: list[Any]) -> None:
    for store in stores:
        flush = getattr(store, "flush", None)
        if callable(flush):
            flush()


def _set_all(stores: list[Any], remote_path: str, file_type: str, status: str, message: str, flush: bool = True, **kwargs: Any) -> None:
    for store in stores:
        store.set_file_status(remote_path, file_type, status, message, flush=flush, **kwargs)
