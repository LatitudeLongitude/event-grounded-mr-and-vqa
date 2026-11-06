from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .disk_utils import dir_size_bytes, free_disk_bytes
from .progress import append_jsonl
from .time_utils import utc_now_iso


_DIR_SIZE_CACHE: dict[str, tuple[int, float]] = {}


def setup_logging(logs_dir: Path, level: str = "INFO") -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("castle_pipeline")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    file_handler = None
    log_candidates = [
        logs_dir / "pipeline.log",
        logs_dir / f"pipeline_{os.getpid()}.log",
        logs_dir.parent / "pipeline.log",
        logs_dir.parent / f"pipeline_{os.getpid()}.log",
    ]
    last_error: Exception | None = None
    for log_path in log_candidates:
        try:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            break
        except OSError as exc:
            last_error = exc
    if file_handler is not None:
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        if file_handler.baseFilename != str(logs_dir / "pipeline.log"):
            logger.warning("primary log file unavailable; using fallback log file %s", file_handler.baseFilename)
    else:
        logger.warning("file logging unavailable; continuing with terminal logging only: %s", last_error)
    return logger


def log_to_file_only(logger: logging.Logger, level: int, message: str, *args: Any) -> None:
    record = logger.makeRecord(
        logger.name,
        level,
        fn="",
        lno=0,
        msg=message,
        args=args,
        exc_info=None,
        func=None,
        extra=None,
    )
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.handle(record)


def log_command_result_json(logger: logging.Logger, command: str, result_json: str) -> None:
    log_to_file_only(logger, logging.INFO, "command %s result:\n%s", command, result_json)


@dataclass
class ProgressReporter:
    logger: logging.Logger
    label: str
    every_items: int = 25
    every_seconds: int = 30
    started_at: float = field(default_factory=time.monotonic)
    last_report_at: float = field(default_factory=time.monotonic)

    def start(self, message: str, **fields: Any) -> None:
        self.logger.info("%s started: %s%s", self.label, message, _format_fields(fields))

    def maybe(self, count: int, message: str, force: bool = False, **fields: Any) -> None:
        now = time.monotonic()
        by_count = self.every_items > 0 and count > 0 and count % self.every_items == 0
        by_time = self.every_seconds > 0 and now - self.last_report_at >= self.every_seconds
        if not (force or by_count or by_time):
            return
        self.last_report_at = now
        elapsed = now - self.started_at
        self.logger.info(
            "%s progress: %s count=%s elapsed_seconds=%.1f%s",
            self.label,
            message,
            count,
            elapsed,
            _format_fields(fields),
        )

    def done(self, count: int, message: str, **fields: Any) -> None:
        elapsed = time.monotonic() - self.started_at
        self.logger.info(
            "%s completed: %s count=%s elapsed_seconds=%.1f%s",
            self.label,
            message,
            count,
            elapsed,
            _format_fields(fields),
        )


def _format_fields(fields: dict[str, Any]) -> str:
    if not fields:
        return ""
    parts = [f"{key}={value}" for key, value in fields.items()]
    return " " + " ".join(parts)


def _cached_dir_size_bytes(path: Path, refresh_interval_seconds: int | float) -> int:
    if refresh_interval_seconds <= 0:
        return dir_size_bytes(path)
    key = str(path.resolve() if path.exists() else path.absolute())
    now = time.monotonic()
    cached = _DIR_SIZE_CACHE.get(key)
    if cached is not None:
        size, checked_at = cached
        if now - checked_at < refresh_interval_seconds:
            return size
    size = dir_size_bytes(path)
    _DIR_SIZE_CACHE[key] = (size, now)
    return size


def heartbeat(
    heartbeat_path: Path,
    raw_cache_dir: Path,
    processed_data_dir: Path,
    current_remote_path: str | None,
    status: str,
    message: str,
    download_mode: str | None = None,
    attempt: int | None = None,
    downloaded_bytes: int | None = None,
    remote_size_bytes: int | None = None,
    frame_count: int | None = None,
    latest_frame_name: str | None = None,
    last_progress_seconds_ago: float | None = None,
    logger: logging.Logger | None = None,
    dir_size_refresh_interval_seconds: int | float = 300,
    include_dir_sizes: bool = True,
) -> dict[str, Any]:
    record = {
        "timestamp": utc_now_iso(),
        "current_remote_path": current_remote_path,
        "status": status,
        "download_mode": download_mode,
        "attempt": attempt,
        "downloaded_bytes": downloaded_bytes,
        "remote_size_bytes": remote_size_bytes,
        "frame_count": frame_count,
        "latest_frame_name": latest_frame_name,
        "raw_cache_size_bytes": _cached_dir_size_bytes(raw_cache_dir, dir_size_refresh_interval_seconds) if include_dir_sizes else None,
        "processed_data_size_bytes": _cached_dir_size_bytes(processed_data_dir, dir_size_refresh_interval_seconds) if include_dir_sizes else None,
        "free_disk_bytes": free_disk_bytes(raw_cache_dir),
        "last_progress_seconds_ago": last_progress_seconds_ago,
        "message": message,
    }
    append_jsonl(heartbeat_path, record)
    if logger:
        logger.info(
            "heartbeat path=%s status=%s mode=%s attempt=%s downloaded=%s remote=%s free=%s msg=%s",
            current_remote_path,
            status,
            download_mode,
            attempt,
            downloaded_bytes,
            remote_size_bytes,
            record["free_disk_bytes"],
            message,
        )
    return record
