from __future__ import annotations

import json
from dataclasses import asdict, dataclass
import os
import shutil
import time
from pathlib import PurePosixPath
from typing import Any, Iterable

from .progress import atomic_write_json
from .time_utils import timestamp_for_filename, utc_now_iso


@dataclass(frozen=True)
class CastleVideoInfo:
    remote_path: str
    day: str
    source: str
    video_hour_stem: str


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int | None
    file_type: str
    extension: str
    is_video: bool
    is_root_file: bool
    is_sidecar: bool
    is_sha256: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_remote_path(path: str) -> str:
    return PurePosixPath(path.replace("\\", "/")).as_posix()


def parse_castle_video_path(remote_path: str) -> CastleVideoInfo:
    normalized = normalize_remote_path(remote_path)
    parts = PurePosixPath(normalized).parts
    day = ""
    source = ""
    if len(parts) >= 5 and parts[-2].lower() == "video":
        day = parts[-4]
        source = parts[-3]
    else:
        for index, part in enumerate(parts):
            if part.lower().startswith("day"):
                day = part
                source = parts[index + 1] if index + 1 < len(parts) else ""
                break
    return CastleVideoInfo(
        remote_path=normalized,
        day=day,
        source=source,
        video_hour_stem=PurePosixPath(normalized).stem,
    )


def classify_remote_file(path: str, size: int | None, video_extensions: Iterable[str]) -> RemoteFile:
    normalized = normalize_remote_path(path)
    posix = PurePosixPath(normalized)
    suffix = posix.suffix.lower()
    video_exts = {ext.lower() for ext in video_extensions}
    is_video = suffix in video_exts
    is_sha256 = normalized.lower().endswith(".sha256")
    is_root_file = len(posix.parts) == 1
    sidecar_markers = ("/metadata/", "/transcript/", "/imu/", "/gps/", "/audio/", "/bio")
    is_sidecar = is_sha256 or any(marker in f"/{normalized.lower()}" for marker in sidecar_markers)
    return RemoteFile(
        path=normalized,
        size=size,
        file_type="video" if is_video else "non_video",
        extension=suffix,
        is_video=is_video,
        is_root_file=is_root_file,
        is_sidecar=is_sidecar,
        is_sha256=is_sha256,
    )


def inventory_from_files(repo_id: str, files: list[RemoteFile]) -> dict[str, Any]:
    files = sorted({item.path: item for item in files}.values(), key=lambda item: item.path)
    videos = [item.to_dict() for item in files if item.is_video]
    non_videos = [item.to_dict() for item in files if not item.is_video]
    root_files = [item.to_dict() for item in files if item.is_root_file]
    sidecars = [item.to_dict() for item in files if item.is_sidecar]
    return {
        "repo_id": repo_id,
        "scan_time": utc_now_iso(),
        "total_files": len(files),
        "total_size_bytes": sum(item.size or 0 for item in files),
        "video_files": videos,
        "non_video_files": non_videos,
        "root_files": root_files,
        "sidecar_files": sidecars,
        "missing_or_unclassified": [],
    }


def inventory_invalid_reason(inventory: dict[str, Any]) -> str | None:
    for section in ["video_files", "non_video_files", "root_files", "sidecar_files"]:
        for item in inventory.get(section, []):
            if not isinstance(item, dict):
                return f"invalid inventory item in {section}"
            if item.get("size") is None and not item.get("is_video"):
                suffix = PurePosixPath(str(item.get("path", ""))).suffix
                if suffix == "":
                    return f"directory-like path stored as file in inventory: {item.get('path')}"
    return None


def scan_remote_inventory(cfg: Any, logger: Any | None = None) -> dict[str, Any]:
    cfg.apply_hf_environment()
    started = time.monotonic()
    if logger:
        logger.info(
            "starting remote inventory scan repo_id=%s repo_type=%s strategy=segmented_v1 expand=True",
            cfg.repo_id,
            cfg.repo_type,
        )
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        if logger:
            logger.error("remote inventory scan failed before request: huggingface_hub is missing")
        raise RuntimeError("huggingface_hub is required. Install with: pip install -r requirements.txt") from exc

    try:
        api = HfApi(token=os.getenv(cfg.huggingface.token_env))
        checkpoint = _load_or_create_scan_checkpoint(cfg, logger)
        if checkpoint.get("status") == "completed":
            if logger:
                logger.info(
                    "existing completed remote inventory checkpoint found files=%s checkpoint=%s",
                    len(checkpoint.get("files", {})),
                    _checkpoint_path(cfg),
                )
            inventory = _inventory_from_checkpoint(cfg, checkpoint)
            _write_inventory_outputs(cfg, inventory)
            return inventory
        _scan_root_and_build_segments(cfg, api, checkpoint, logger, started)
        segment_ids = _planned_segment_ids(checkpoint)
        if logger:
            logger.info(
                "remote inventory segmented scan plan segments=%s completed=%s checkpoint=%s",
                len(segment_ids),
                sum(1 for segment_id in segment_ids if checkpoint["segments"].get(segment_id, {}).get("status") == "completed"),
                _checkpoint_path(cfg),
            )
        for segment_id in segment_ids:
            segment = checkpoint["segments"][segment_id]
            if segment.get("status") == "completed":
                if logger:
                    logger.info("scan segment skipped already completed segment_id=%s path=%s", segment_id, segment.get("path"))
                continue
            _scan_segment_with_retries(cfg, api, checkpoint, segment_id, logger, started)
        checkpoint["status"] = "completed"
        checkpoint["completed_at"] = utc_now_iso()
        _refresh_checkpoint_summary(checkpoint)
        _write_scan_checkpoint(cfg, checkpoint)
        inventory = _inventory_from_checkpoint(cfg, checkpoint)
        _write_inventory_outputs(cfg, inventory)
    except KeyboardInterrupt:
        if logger:
            logger.warning("remote inventory scan interrupted by user after %.1f seconds", time.monotonic() - started)
        raise
    except Exception as exc:
        if logger:
            logger.exception("remote inventory scan failed after %.1f seconds: %s", time.monotonic() - started, exc)
        raise RuntimeError(f"remote inventory scan failed: {exc}") from exc
    if logger:
        logger.info(
            "remote inventory scan completed files=%s videos=%s non_videos=%s elapsed_seconds=%.1f",
            inventory["total_files"],
            len(inventory["video_files"]),
            len(inventory["non_video_files"]),
            time.monotonic() - started,
        )
    return inventory


def inventory_remote_files(inventory: dict[str, Any]) -> list[RemoteFile]:
    files: list[RemoteFile] = []
    for raw in inventory.get("video_files", []) + inventory.get("non_video_files", []):
        files.append(RemoteFile(**raw))
    return sorted(files, key=lambda item: item.path)


def _checkpoint_path(cfg: Any) -> Any:
    return cfg.manifests_dir / "remote_inventory.checkpoint.json"


def _code_checkpoint_path(cfg: Any) -> Any:
    return cfg.paths.code_dir / "remote_inventory.checkpoint.json"


def _load_or_create_scan_checkpoint(cfg: Any, logger: Any | None) -> dict[str, Any]:
    path = _checkpoint_path(cfg)
    fallback = _code_checkpoint_path(cfg)
    source = path if path.exists() else fallback
    if source.exists():
        try:
            checkpoint = json.loads(source.read_text(encoding="utf-8"))
            if checkpoint.get("repo_id") == cfg.repo_id and checkpoint.get("scan_strategy") == "segmented_v1":
                invalid_reason = _checkpoint_invalid_reason(checkpoint)
                if invalid_reason:
                    if logger:
                        logger.warning("ignoring invalid scan checkpoint %s: %s", source, invalid_reason)
                    _quarantine_invalid_checkpoint(cfg, source, logger)
                    if fallback.exists() and fallback != source:
                        _quarantine_invalid_checkpoint(cfg, fallback, logger)
                    _quarantine_inventory_outputs(cfg, logger)
                else:
                    if logger:
                        logger.info(
                            "resuming remote inventory scan from checkpoint status=%s files=%s checkpoint=%s",
                            checkpoint.get("status"),
                            len(checkpoint.get("files", {})),
                            source,
                        )
                    return checkpoint
            if logger:
                logger.warning("ignoring incompatible scan checkpoint: %s", source)
        except Exception as exc:
            if logger:
                logger.warning("ignoring unreadable scan checkpoint %s: %s", source, exc)
    checkpoint = {
        "schema_version": 1,
        "repo_id": cfg.repo_id,
        "repo_type": cfg.repo_type,
        "scan_strategy": "segmented_v1",
        "status": "in_progress",
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "completed_at": None,
        "segments": {},
        "planned_segments": [],
        "files": {},
        "summary": {
            "total_files": 0,
            "total_entries_seen": 0,
            "video_count": 0,
            "non_video_count": 0,
            "last_path": None,
            "last_success_at": None,
        },
    }
    _write_scan_checkpoint(cfg, checkpoint)
    return checkpoint


def _write_scan_checkpoint(cfg: Any, checkpoint: dict[str, Any]) -> None:
    checkpoint["updated_at"] = utc_now_iso()
    _refresh_checkpoint_summary(checkpoint)
    cfg.manifests_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.code_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(_checkpoint_path(cfg), checkpoint)
    atomic_write_json(_code_checkpoint_path(cfg), checkpoint)


def _checkpoint_invalid_reason(checkpoint: dict[str, Any]) -> str | None:
    files = checkpoint.get("files", {})
    if not isinstance(files, dict):
        return "files section is missing or invalid"
    for path, item in files.items():
        if not isinstance(item, dict):
            return f"file entry is invalid for {path}"
        if item.get("size") is None and not item.get("is_video"):
            suffix = PurePosixPath(str(path)).suffix
            if suffix == "":
                return f"directory-like path was stored as a file: {path}"
    if checkpoint.get("status") == "completed":
        planned = checkpoint.get("planned_segments", [])
        segments = checkpoint.get("segments", {})
        root = segments.get("children:__root__", {})
        if planned == ["children:__root__"] and root.get("entries_seen", 0) and not root.get("folders") and any(
            str(path) in {"main", "auxiliary"} for path in files
        ):
            return "completed checkpoint only contains root segment and misclassified top-level folders"
    return None


def _quarantine_invalid_checkpoint(cfg: Any, path: Any, logger: Any | None) -> None:
    if not path.exists():
        return
    invalid_path = path.with_name(f"{path.name}.invalid_{timestamp_for_filename()}")
    try:
        shutil.move(str(path), str(invalid_path))
        if logger:
            logger.warning("moved invalid scan checkpoint to %s", invalid_path)
    except OSError as exc:
        if logger:
            logger.warning("could not move invalid scan checkpoint %s: %s", path, exc)


def _quarantine_inventory_outputs(cfg: Any, logger: Any | None) -> None:
    for path in [cfg.manifests_dir / "remote_inventory.json", cfg.paths.code_dir / "remote_inventory.json"]:
        if path.exists():
            invalid_path = path.with_name(f"{path.name}.invalid_{timestamp_for_filename()}")
            try:
                shutil.move(str(path), str(invalid_path))
                if logger:
                    logger.warning("moved inventory generated from invalid checkpoint to %s", invalid_path)
            except OSError as exc:
                if logger:
                    logger.warning("could not move inventory generated from invalid checkpoint %s: %s", path, exc)


def _refresh_checkpoint_summary(checkpoint: dict[str, Any]) -> None:
    files = checkpoint.get("files", {})
    segments = checkpoint.get("segments", {})
    values = list(files.values())
    last_paths = [segment.get("last_path") for segment in segments.values() if segment.get("last_path")]
    checkpoint["summary"] = {
        "total_files": len(files),
        "total_entries_seen": sum(int(segment.get("entries_seen", 0) or 0) for segment in segments.values()),
        "video_count": sum(1 for item in values if item.get("is_video")),
        "non_video_count": sum(1 for item in values if not item.get("is_video")),
        "last_path": last_paths[-1] if last_paths else None,
        "last_success_at": checkpoint.get("summary", {}).get("last_success_at"),
    }


def _inventory_from_checkpoint(cfg: Any, checkpoint: dict[str, Any]) -> dict[str, Any]:
    files = [RemoteFile(**item) for item in checkpoint.get("files", {}).values()]
    return inventory_from_files(cfg.repo_id, files)


def _write_inventory_outputs(cfg: Any, inventory: dict[str, Any]) -> None:
    cfg.manifests_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(cfg.manifests_dir / "remote_inventory.json", inventory)
    atomic_write_json(cfg.paths.code_dir / "remote_inventory.json", inventory)


def _segment_id(path: str | None, recursive: bool) -> str:
    label = "__root__" if path in (None, "") else normalize_remote_path(path)
    mode = "recursive" if recursive else "children"
    return f"{mode}:{label}"


def _ensure_segment(checkpoint: dict[str, Any], path: str | None, recursive: bool, label: str | None = None) -> str:
    segment_id = _segment_id(path, recursive)
    segments = checkpoint.setdefault("segments", {})
    if segment_id not in segments:
        segments[segment_id] = {
            "segment_id": segment_id,
            "label": label or segment_id,
            "path": normalize_remote_path(path) if path else None,
            "recursive": recursive,
            "status": "pending",
            "attempts": 0,
            "entries_seen": 0,
            "files_count": 0,
            "folders": [],
            "last_path": None,
            "last_error": None,
            "started_at": None,
            "finished_at": None,
        }
    planned = checkpoint.setdefault("planned_segments", [])
    if segment_id not in planned:
        planned.append(segment_id)
    return segment_id


def _planned_segment_ids(checkpoint: dict[str, Any]) -> list[str]:
    planned = checkpoint.get("planned_segments", [])
    return [segment_id for segment_id in planned if segment_id in checkpoint.get("segments", {})]


def _scan_root_and_build_segments(cfg: Any, api: Any, checkpoint: dict[str, Any], logger: Any | None, started: float) -> None:
    root_id = _ensure_segment(checkpoint, None, False, "root")
    if checkpoint["segments"][root_id].get("status") != "completed":
        _scan_segment_with_retries(cfg, api, checkpoint, root_id, logger, started)
    root_folders = checkpoint["segments"][root_id].get("folders", [])
    for folder in sorted(root_folders):
        if folder == "main":
            main_children_id = _ensure_segment(checkpoint, "main", False, "main children")
            if checkpoint["segments"][main_children_id].get("status") != "completed":
                _scan_segment_with_retries(cfg, api, checkpoint, main_children_id, logger, started)
            for main_child in sorted(checkpoint["segments"][main_children_id].get("folders", [])):
                _ensure_segment(checkpoint, main_child, True, main_child)
        else:
            _ensure_segment(checkpoint, folder, True, folder)
    _write_scan_checkpoint(cfg, checkpoint)


def _scan_segment_with_retries(
    cfg: Any,
    api: Any,
    checkpoint: dict[str, Any],
    segment_id: str,
    logger: Any | None,
    started: float,
) -> None:
    segment = checkpoint["segments"][segment_id]
    max_attempts = max(1, int(cfg.logging.scan_max_attempts_per_segment))
    attempts_this_run = 0
    while attempts_this_run < max_attempts:
        attempts_this_run += 1
        segment["attempts"] = int(segment.get("attempts", 0) or 0) + 1
        try:
            _scan_segment_once(cfg, api, checkpoint, segment_id, logger, started)
            return
        except Exception as exc:
            segment["status"] = "failed_retryable" if _is_retryable_scan_error(exc) else "failed_need_manual"
            segment["last_error"] = str(exc)
            _write_scan_checkpoint(cfg, checkpoint)
            if not _is_retryable_scan_error(exc) or attempts_this_run >= max_attempts:
                if logger:
                    logger.error(
                        "scan segment failed segment_id=%s path=%s attempts_this_run=%s total_attempts=%s checkpoint=%s error=%s",
                        segment_id,
                        segment.get("path"),
                        attempts_this_run,
                        segment.get("attempts"),
                        _checkpoint_path(cfg),
                        exc,
                    )
                raise RuntimeError(
                    f"scan segment failed after {attempts_this_run} attempts; checkpoint kept at {_checkpoint_path(cfg)}"
                ) from exc
            wait_seconds = _retry_wait_seconds(cfg, attempts_this_run)
            if logger:
                logger.warning(
                    "retrying scan segment segment_id=%s path=%s retry=%s/%s wait_seconds=%s resume_checkpoint=%s reason=%s",
                    segment_id,
                    segment.get("path"),
                    attempts_this_run + 1,
                    max_attempts,
                    wait_seconds,
                    _checkpoint_path(cfg),
                    exc,
                )
            if wait_seconds > 0:
                time.sleep(wait_seconds)


def _scan_segment_once(
    cfg: Any,
    api: Any,
    checkpoint: dict[str, Any],
    segment_id: str,
    logger: Any | None,
    started: float,
) -> None:
    segment = checkpoint["segments"][segment_id]
    segment["status"] = "scanning"
    segment["started_at"] = segment.get("started_at") or utc_now_iso()
    segment["last_error"] = None
    segment["entries_seen"] = 0
    segment["files_count"] = 0
    segment["folders"] = []
    _write_scan_checkpoint(cfg, checkpoint)
    path = segment.get("path")
    recursive = bool(segment.get("recursive"))
    if logger:
        logger.info("scan segment started segment_id=%s path=%s recursive=%s", segment_id, path, recursive)
    last_report = time.monotonic()
    last_checkpoint_file_count = 0
    for entry in api.list_repo_tree(
        repo_id=cfg.repo_id,
        repo_type=cfg.repo_type,
        path_in_repo=path,
        recursive=recursive,
        expand=True,
    ):
        segment["entries_seen"] = int(segment.get("entries_seen", 0) or 0) + 1
        current_path = getattr(entry, "path", None) or getattr(entry, "rfilename", None) or ""
        segment["last_path"] = current_path or segment.get("last_path")
        if _is_directory_entry(entry):
            if current_path:
                folders = set(segment.get("folders", []))
                folders.add(normalize_remote_path(current_path))
                segment["folders"] = sorted(folders)
        elif current_path:
            remote = classify_remote_file(current_path, getattr(entry, "size", None), cfg.processing.video_extensions)
            checkpoint.setdefault("files", {})[remote.path] = remote.to_dict()
            segment["files_count"] = int(segment.get("files_count", 0) or 0) + 1
            checkpoint["summary"]["last_success_at"] = utc_now_iso()
        now = time.monotonic()
        files_total = len(checkpoint.get("files", {}))
        by_count = cfg.logging.scan_progress_every_files > 0 and files_total > 0 and files_total % cfg.logging.scan_progress_every_files == 0
        by_time = now - last_report >= cfg.logging.scan_progress_interval_seconds
        should_checkpoint = (
            cfg.logging.scan_checkpoint_every_files > 0
            and files_total - last_checkpoint_file_count >= cfg.logging.scan_checkpoint_every_files
        )
        if should_checkpoint or by_count or by_time:
            last_report = now
            last_checkpoint_file_count = files_total
            _write_scan_checkpoint(cfg, checkpoint)
            if logger and (by_count or by_time):
                summary = checkpoint.get("summary", {})
                logger.info(
                    "scan progress files=%s entries_seen=%s videos=%s non_videos=%s elapsed_seconds=%.1f current_path=%s segment_id=%s checkpoint=%s",
                    summary.get("total_files", files_total),
                    summary.get("total_entries_seen"),
                    summary.get("video_count"),
                    summary.get("non_video_count"),
                    now - started,
                    current_path,
                    segment_id,
                    _checkpoint_path(cfg),
                )
    segment["status"] = "completed"
    segment["finished_at"] = utc_now_iso()
    _write_scan_checkpoint(cfg, checkpoint)
    if logger:
        logger.info(
            "scan segment completed segment_id=%s path=%s recursive=%s files_in_segment=%s folders=%s total_files=%s checkpoint=%s",
            segment_id,
            path,
            recursive,
            segment.get("files_count"),
            len(segment.get("folders", [])),
            len(checkpoint.get("files", {})),
            _checkpoint_path(cfg),
        )


def _retry_wait_seconds(cfg: Any, attempt_number: int) -> int:
    waits = list(cfg.logging.scan_retry_backoff_seconds or [])
    if not waits:
        return 0
    return int(waits[min(max(0, attempt_number - 1), len(waits) - 1)])


def _is_retryable_scan_error(exc: BaseException) -> bool:
    text = f"{exc.__class__.__module__}.{exc.__class__.__name__}: {exc}".lower()
    retryable_markers = [
        "remoteprotocolerror",
        "connecterror",
        "readtimeout",
        "connecttimeout",
        "timeout",
        "temporarily unavailable",
        "server disconnected",
        "connection reset",
        "connection aborted",
        "proxy",
        "http 429",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
    ]
    return any(marker in text for marker in retryable_markers)


def _is_directory_entry(entry: Any) -> bool:
    entry_type = str(getattr(entry, "type", "") or "").lower()
    class_name = entry.__class__.__name__.lower()
    return entry_type in {"directory", "folder", "tree"} or "folder" in class_name
