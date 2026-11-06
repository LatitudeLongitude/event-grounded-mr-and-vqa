from __future__ import annotations

import csv
import json
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from .constants import DEFAULT_CSV_FIELDS, FRAME_RESIZE_CSV_FIELDS
from .hf_inventory import parse_castle_video_path
from .progress import atomic_write_json, atomic_write_text
from .time_utils import seconds_to_hms, utc_now_iso


_MANIFEST_THREAD_LOCK = threading.RLock()
_MANIFEST_LOCK_TIMEOUT_SECONDS = 300.0
_MANIFEST_LOCK_POLL_SECONDS = 0.2
_MANIFEST_REPLACE_BACKOFF_SECONDS = [0.1, 0.25, 0.5, 1.0, 2.0]


VIDEO_ARTIFACT_FILES = [
    "frame_manifest.csv",
    "frame_manifest.jsonl",
    "video_processing_summary.json",
    "ffmpeg_command.txt",
    "frame_resize_command.txt",
    "verify_result.json",
]


def frame_relative_path(remote_video_path: str, frame_name: str) -> str:
    base = PurePosixPath(remote_video_path).with_suffix("")
    return (base / "frames" / frame_name).as_posix()


def write_frame_manifests(video_dir: Path, remote_video_path: str, metadata: dict[str, Any]) -> None:
    frames_dir = video_dir / "frames"
    frames = sorted(frames_dir.glob("*.jpg"))
    info = parse_castle_video_path(remote_video_path)
    fps = int(metadata.get("fps") or 1)
    resize_fields_present = _include_resize_fields(metadata)
    fieldnames = list(DEFAULT_CSV_FIELDS)
    if resize_fields_present:
        fieldnames.extend(field for field in FRAME_RESIZE_CSV_FIELDS if field not in fieldnames)
    rows: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        timestamp_second = index / fps
        row = {
            "repo_id": metadata.get("repo_id", "CASTLE-Dataset/CASTLE2024"),
            "remote_video_path": remote_video_path,
            "original_video_filename": PurePosixPath(remote_video_path).name,
            "processed_video_dir": str(video_dir),
            "frame_rel_path": frame_relative_path(remote_video_path, frame.name),
            "frame_abs_path": str(frame.resolve()),
            "day": metadata.get("day") or info.day,
            "source": metadata.get("source") or info.source,
            "video_hour_stem": metadata.get("video_hour_stem") or info.video_hour_stem,
            "frame_index": index,
            "timestamp_second": int(timestamp_second) if float(timestamp_second).is_integer() else timestamp_second,
            "timestamp_hms": seconds_to_hms(timestamp_second),
            "fps": fps,
            "width": metadata.get("width"),
            "jpg_q": metadata.get("jpg_q"),
            "ffmpeg_command": metadata.get("ffmpeg_command", ""),
            "video_duration_seconds": metadata.get("video_duration_seconds"),
            "video_size_bytes": metadata.get("video_size_bytes"),
            "download_started_at": metadata.get("download_started_at", ""),
            "download_finished_at": metadata.get("download_finished_at", ""),
            "processing_started_at": metadata.get("processing_started_at", ""),
            "processing_finished_at": metadata.get("processing_finished_at", ""),
            "raw_deleted_at": metadata.get("raw_deleted_at", ""),
            "sha256_or_size_check": metadata.get("sha256_or_size_check", ""),
            "status": metadata.get("status", "processed_unverified"),
        }
        if resize_fields_present:
            for field in FRAME_RESIZE_CSV_FIELDS:
                row[field] = metadata.get(field, "")
        rows.append(row)
    csv_path = video_dir / "frame_manifest.csv"
    tmp_csv = csv_path.with_name(csv_path.name + ".tmp")
    with tmp_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_csv, csv_path)

    jsonl_path = video_dir / "frame_manifest.jsonl"
    atomic_write_text(jsonl_path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def write_video_summary(video_dir: Path, summary: dict[str, Any]) -> None:
    atomic_write_json(video_dir / "video_processing_summary.json", summary)


def video_artifacts_need_raw_deleted_at(video_dir: Path) -> bool:
    summary_path = video_dir / "video_processing_summary.json"
    csv_path = video_dir / "frame_manifest.csv"
    jsonl_path = video_dir / "frame_manifest.jsonl"
    try:
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if _is_blank(summary.get("raw_deleted_at")):
                return True
        if csv_path.exists():
            with csv_path.open("r", encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    if _is_blank(row.get("raw_deleted_at")):
                        return True
        if jsonl_path.exists():
            with jsonl_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    if _is_blank(payload.get("raw_deleted_at")):
                        return True
    except Exception:
        return True
    return False


def update_video_raw_deleted_metadata(
    video_dir: Path,
    raw_deleted_at: str,
    status: str | None = None,
    raw_deleted_at_source: str | None = None,
) -> dict[str, Any]:
    if _is_blank(raw_deleted_at):
        raise ValueError("raw_deleted_at must be a non-empty timestamp")
    summary_path = video_dir / "video_processing_summary.json"
    csv_path = video_dir / "frame_manifest.csv"
    jsonl_path = video_dir / "frame_manifest.jsonl"
    if not summary_path.exists():
        raise FileNotFoundError(f"video summary missing: {summary_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"frame manifest csv missing: {csv_path}")
    if not jsonl_path.exists():
        raise FileNotFoundError(f"frame manifest jsonl missing: {jsonl_path}")

    _update_csv_raw_deleted_at(csv_path, raw_deleted_at, status)
    _update_jsonl_raw_deleted_at(jsonl_path, raw_deleted_at, status)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError(f"video summary must be a JSON object: {summary_path}")
    summary["raw_deleted_at"] = raw_deleted_at
    if status is not None:
        summary["status"] = status
    if raw_deleted_at_source is not None:
        summary["raw_deleted_at_source"] = raw_deleted_at_source
    write_video_summary(video_dir, summary)
    return summary


def _update_csv_raw_deleted_at(path: Path, raw_deleted_at: str, status: str | None) -> None:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"csv manifest has no header: {path}")
        rows = list(reader)
    if "raw_deleted_at" not in fieldnames:
        fieldnames.append("raw_deleted_at")
    if status is not None and "status" not in fieldnames:
        fieldnames.append("status")
    for row in rows:
        if None in row:
            raise ValueError(f"csv manifest row has unexpected extra columns: {path}")
        row["raw_deleted_at"] = raw_deleted_at
        if status is not None:
            row["status"] = status
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _update_jsonl_raw_deleted_at(path: Path, raw_deleted_at: str, status: str | None) -> None:
    payloads: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"jsonl manifest row must be an object: {path}")
            payload["raw_deleted_at"] = raw_deleted_at
            if status is not None:
                payload["status"] = status
            payloads.append(payload)
    atomic_write_text(path, "".join(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n" for payload in payloads))


def _is_blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def sanitize_processing_tmp_paths(value: Any, final_dir: Path) -> Any:
    tmp_dir = final_dir.with_name(final_dir.name + ".__processing_tmp__")
    if isinstance(value, str):
        sanitized = value.replace(str(tmp_dir), str(final_dir))
        sanitized = sanitized.replace(tmp_dir.as_posix(), final_dir.as_posix())
        sanitized = sanitized.replace(".__processing_tmp__", "")
        return sanitized
    if isinstance(value, dict):
        return {key: sanitize_processing_tmp_paths(item, final_dir) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_processing_tmp_paths(item, final_dir) for item in value]
    return value


def artifact_tree_contains_processing_tmp(video_dir: Path) -> bool:
    for name in VIDEO_ARTIFACT_FILES:
        path = video_dir / name
        if path.exists() and ".__processing_tmp__" in path.read_text(encoding="utf-8", errors="replace"):
            return True
    return False


def rewrite_final_video_artifacts(
    video_dir: Path,
    remote_video_path: str,
    metadata_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary_path = video_dir / "video_processing_summary.json"
    metadata: dict[str, Any] = {}
    if summary_path.exists():
        try:
            metadata = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {}
    metadata = sanitize_processing_tmp_paths(metadata, video_dir)
    metadata.update(sanitize_processing_tmp_paths(metadata_overrides or {}, video_dir))

    command_text = str(metadata.get("ffmpeg_command") or "")
    command_path = video_dir / "ffmpeg_command.txt"
    if command_path.exists() and not command_text:
        command_text = command_path.read_text(encoding="utf-8", errors="replace").strip()
    command_text = sanitize_processing_tmp_paths(command_text, video_dir)

    resize_command_text = str(metadata.get("frame_resize_command") or "")
    resize_command_path = video_dir / "frame_resize_command.txt"
    if resize_command_path.exists() and not resize_command_text:
        resize_command_text = resize_command_path.read_text(encoding="utf-8", errors="replace").strip()
    resize_command_text = sanitize_processing_tmp_paths(resize_command_text, video_dir)

    info = parse_castle_video_path(remote_video_path)
    metadata.setdefault("repo_id", "CASTLE-Dataset/CASTLE2024")
    metadata["remote_video_path"] = remote_video_path
    metadata["original_video_filename"] = PurePosixPath(remote_video_path).name
    metadata["processed_video_dir"] = str(video_dir)
    metadata["day"] = metadata.get("day") or info.day
    metadata["source"] = metadata.get("source") or info.source
    metadata["video_hour_stem"] = metadata.get("video_hour_stem") or info.video_hour_stem
    metadata.setdefault("fps", 1)
    metadata.setdefault("width", 3840)
    metadata.setdefault("jpg_q", 2)
    metadata["ffmpeg_command"] = command_text
    if resize_command_text:
        metadata["frame_resize_command"] = resize_command_text
    metadata.setdefault("status", "processed_verified")

    atomic_write_text(command_path, command_text + "\n")
    if resize_command_text:
        atomic_write_text(resize_command_path, resize_command_text + "\n")
    write_frame_manifests(video_dir, remote_video_path, metadata)

    frames = sorted((video_dir / "frames").glob("*.jpg"))
    summary = {
        **metadata,
        "processed_video_dir": str(video_dir),
        "frame_count": len(frames),
        "final_artifacts_rewritten_at": utc_now_iso(),
    }
    write_video_summary(video_dir, summary)
    return summary


def append_global_manifests(cfg: Any, video_dir: Path, remote_video_path: str) -> None:
    manifests_dir = cfg.manifests_dir
    manifests_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    lock_path = manifests_dir / ".manifest_write.lock"
    with _manifest_write_lock(lock_path, logger):
        per_csv = video_dir / "frame_manifest.csv"
        per_jsonl = video_dir / "frame_manifest.jsonl"
        all_csv = manifests_dir / "all_frames.csv"
        all_jsonl = manifests_dir / "all_frames.jsonl"

        if per_csv.exists():
            with per_csv.open("r", encoding="utf-8", newline="") as source:
                reader = csv.DictReader(source)
                per_fields = list(reader.fieldnames or [])
                new_rows = list(reader)
            existing_rows: list[dict[str, Any]] = []
            existing_fields: list[str] = []
            if all_csv.exists():
                with all_csv.open("r", encoding="utf-8", newline="") as current:
                    reader = csv.DictReader(current)
                    existing_fields = list(reader.fieldnames or [])
                    for row in reader:
                        if row.get("remote_video_path") != remote_video_path:
                            existing_rows.append(row)
            include_resize_fields = _rows_or_headers_have_resize_fields(existing_fields, per_fields, existing_rows, new_rows)
            preferred_fields = DEFAULT_CSV_FIELDS + FRAME_RESIZE_CSV_FIELDS if include_resize_fields else DEFAULT_CSV_FIELDS
            fields = _ordered_csv_fields(preferred_fields, existing_fields, per_fields, existing_rows, new_rows)
            _atomic_write_csv_with_retry(all_csv, fields, existing_rows + new_rows, logger)
        if per_jsonl.exists():
            existing_payloads: list[dict[str, Any]] = []
            if all_jsonl.exists():
                with all_jsonl.open("r", encoding="utf-8") as current:
                    for line in current:
                        if line.strip():
                            try:
                                payload = json.loads(line)
                                if payload.get("remote_video_path") != remote_video_path:
                                    existing_payloads.append(payload)
                            except json.JSONDecodeError:
                                continue
            new_payloads: list[dict[str, Any]] = []
            with per_jsonl.open("r", encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    new_payloads.append(payload)
            _atomic_write_text_with_retry(
                all_jsonl,
                "".join(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n" for payload in existing_payloads + new_payloads),
                logger,
            )

        summary_path = video_dir / "video_processing_summary.json"
        if summary_path.exists():
            row = json.loads(summary_path.read_text(encoding="utf-8"))
            summary_csv = manifests_dir / "video_processing_summary.csv"
            existing_rows: list[dict[str, Any]] = []
            if summary_csv.exists():
                with summary_csv.open("r", encoding="utf-8", newline="") as current:
                    for item in csv.DictReader(current):
                        if item.get("remote_video_path") != remote_video_path:
                            existing_rows.append(item)
            fields = sorted({key for item in existing_rows + [row] for key in item.keys()})
            _atomic_write_csv_with_retry(summary_csv, fields, existing_rows + [row], logger)


def write_non_video_manifest(cfg: Any, rows: list[dict[str, Any]]) -> None:
    path = cfg.manifests_dir / "non_video_files_manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row.keys()})
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _ordered_csv_fields(
    preferred: list[str],
    existing_fields: list[str],
    new_fields: list[str],
    existing_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
) -> list[str]:
    fields: list[str] = []
    for source in [preferred, existing_fields, new_fields]:
        for field in source:
            if field and field not in fields:
                fields.append(field)
    for row in existing_rows + new_rows:
        for field in row.keys():
            if field and field not in fields:
                fields.append(field)
    return fields


def _rows_or_headers_have_resize_fields(
    existing_fields: list[str],
    new_fields: list[str],
    existing_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
) -> bool:
    resize = set(FRAME_RESIZE_CSV_FIELDS)
    if resize.intersection(existing_fields) or resize.intersection(new_fields):
        return True
    return any(resize.intersection(row.keys()) for row in existing_rows + new_rows)


def _include_resize_fields(metadata: dict[str, Any]) -> bool:
    if metadata.get("frame_resize_enabled") is False:
        return False
    if str(metadata.get("frame_resize_status") or "").strip().lower() == "disabled":
        return False
    return any(field in metadata for field in FRAME_RESIZE_CSV_FIELDS)


@contextmanager
def _manifest_write_lock(lock_path: Path, logger: logging.Logger) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + _MANIFEST_LOCK_TIMEOUT_SECONDS
    if not _MANIFEST_THREAD_LOCK.acquire(blocking=False):
        logger.info("waiting for manifest write thread lock path=%s", lock_path)
        remaining = max(deadline - time.monotonic(), 0.0)
        if not _MANIFEST_THREAD_LOCK.acquire(timeout=remaining):
            raise TimeoutError(f"timed out waiting for manifest write thread lock: {lock_path}")
    handle = None
    try:
        handle = lock_path.open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
            os.fsync(handle.fileno())
        waited_logged = False
        while True:
            try:
                _try_lock_file(handle)
                break
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for manifest write file lock: {lock_path}") from exc
                if not waited_logged:
                    logger.warning("waiting for manifest write file lock path=%s error=%s", lock_path, exc)
                    waited_logged = True
                time.sleep(_MANIFEST_LOCK_POLL_SECONDS)
        try:
            yield
        finally:
            _unlock_file(handle)
    finally:
        if handle is not None:
            handle.close()
        _MANIFEST_THREAD_LOCK.release()


def _try_lock_file(handle: Any) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(handle: Any) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_write_csv_with_retry(path: Path, fieldnames: list[str], rows: list[dict[str, Any]], logger: logging.Logger) -> None:
    tmp = _unique_tmp_path(path)
    try:
        with tmp.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
            target.flush()
            os.fsync(target.fileno())
        _replace_with_retry(tmp, path, logger)
    except Exception:
        if tmp.exists():
            logger.error("manifest atomic write failed; temporary file retained for inspection path=%s target=%s", tmp, path)
        raise


def _atomic_write_text_with_retry(path: Path, text: str, logger: logging.Logger) -> None:
    tmp = _unique_tmp_path(path)
    try:
        with tmp.open("w", encoding="utf-8", newline="") as target:
            target.write(text)
            target.flush()
            os.fsync(target.fileno())
        _replace_with_retry(tmp, path, logger)
    except Exception:
        if tmp.exists():
            logger.error("manifest atomic write failed; temporary file retained for inspection path=%s target=%s", tmp, path)
        raise


def _unique_tmp_path(path: Path) -> Path:
    thread_id = threading.get_ident()
    return path.with_name(f"{path.name}.tmp.{os.getpid()}.{thread_id}.{uuid.uuid4().hex}")


def _replace_with_retry(tmp: Path, target: Path, logger: logging.Logger) -> None:
    for attempt, delay in enumerate([0.0] + _MANIFEST_REPLACE_BACKOFF_SECONDS, start=1):
        if delay:
            time.sleep(delay)
        try:
            os.replace(tmp, target)
            return
        except OSError as exc:
            if not _is_transient_replace_error(exc) or attempt > len(_MANIFEST_REPLACE_BACKOFF_SECONDS):
                logger.error("manifest replace failed target=%s tmp=%s attempt=%d error=%s", target, tmp, attempt, exc)
                raise
            logger.warning(
                "manifest replace blocked; retrying target=%s tmp=%s attempt=%d error=%s",
                target,
                tmp,
                attempt,
                exc,
            )


def _is_transient_replace_error(exc: OSError) -> bool:
    if isinstance(exc, PermissionError):
        return True
    return getattr(exc, "winerror", None) in {5, 32}
