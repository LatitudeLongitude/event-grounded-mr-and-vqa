from __future__ import annotations

import base64
import hashlib
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import (
    VIDEO_DOWNLOADED_UNVERIFIED,
    VIDEO_DOWNLOADED_VERIFIED,
    VIDEO_DOWNLOADING,
    VIDEO_FAILED_RETRYABLE,
)
from .disk_utils import ensure_enough_free_space, human_bytes
from .ffmpeg_utils import probe_video
from .hash_utils import read_sha256_file, sha256_file
from .logging_utils import heartbeat
from .time_utils import timestamp_for_filename, utc_now_iso


@dataclass
class DownloadResult:
    local_path: Path
    attempts: int
    final_mode: str
    started_at: str
    finished_at: str
    verified_by: str
    used_fallback_http: bool = False


@dataclass(frozen=True)
class DownloadObservation:
    downloaded_bytes: int | None
    final_bytes: int
    temporary_total_bytes: int
    temporary_largest_bytes: int
    temporary_files: tuple[tuple[str, int, int], ...]
    xet_activity_files: tuple[tuple[str, int, int], ...]
    unknown_incomplete_file_count: int
    source: str

    @property
    def temporary_file_count(self) -> int:
        return len(self.temporary_files)

    def advanced_from(self, previous: "DownloadObservation", mode: str = "http") -> bool:
        current_bytes = tuple((name, size) for name, size, _mtime_ns in self.temporary_files)
        previous_bytes = tuple((name, size) for name, size, _mtime_ns in previous.temporary_files)
        if self.final_bytes != previous.final_bytes:
            return True
        if mode == "xet":
            return self.temporary_files != previous.temporary_files or self.xet_activity_files != previous.xet_activity_files
        return current_bytes != previous_bytes

    def details(self) -> dict[str, Any]:
        return {
            "final_bytes": self.final_bytes,
            "temporary_total_bytes": self.temporary_total_bytes,
            "temporary_largest_bytes": self.temporary_largest_bytes,
            "temporary_file_count": self.temporary_file_count,
            "temporary_files": [
                {"name": name, "bytes": size, "mtime_ns": mtime_ns}
                for name, size, mtime_ns in self.temporary_files
            ],
            "xet_activity_files": [
                {"name": name, "bytes": size, "mtime_ns": mtime_ns}
                for name, size, mtime_ns in self.xet_activity_files
            ],
            "unknown_incomplete_file_count": self.unknown_incomplete_file_count,
        }


def _safe_log_stem(remote_path: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in remote_path).strip("_")[:180]


class DownloadManager:
    def __init__(self, cfg: Any, stores: Any, logger: Any) -> None:
        self.cfg = cfg
        self.stores = stores
        self.logger = logger

    def verify_downloaded_file(self, remote_path: str, expected_size: int | None, sha256_path: Path | None = None) -> str:
        local_path = self.cfg.local_raw_path(remote_path)
        if not local_path.exists():
            raise FileNotFoundError(f"downloaded file missing: {local_path}")
        if expected_size is not None and local_path.stat().st_size != expected_size:
            raise RuntimeError(
                f"size mismatch for {remote_path}: local={local_path.stat().st_size} remote={expected_size}"
            )
        expected_sha = read_sha256_file(sha256_path) if sha256_path else None
        if expected_sha:
            actual = sha256_file(local_path)
            if actual.lower() != expected_sha.lower():
                raise RuntimeError(f"sha256 mismatch for {remote_path}: local={actual} expected={expected_sha}")
            return "sha256"
        return "size"

    def verify_video_file(self, remote_path: str, expected_size: int | None, sha256_path: Path | None = None) -> str:
        verified_by = self.verify_downloaded_file(remote_path, expected_size, sha256_path)
        probe_video(self.cfg.tools.ffprobe, self.cfg.local_raw_path(remote_path))
        return verified_by

    def is_small_non_video_file(self, file_type: str, expected_size: int | None) -> bool:
        return (
            file_type == "non_video"
            and expected_size is not None
            and expected_size <= self.cfg.processing.non_video_small_file_threshold_bytes
        )

    def download_modes_for_file(self, file_type: str, expected_size: int | None) -> list[str]:
        if file_type == "video":
            return ["xet", "xet", "xet", "xet", "xet", "xet", "xet", "xet", "xet", "xet", "http"]
        if self.is_small_non_video_file(file_type, expected_size):
            return ["http"]
        modes = ["xet" if self.cfg.huggingface.prefer_xet else "http"]
        if modes[0] == "xet" and self.cfg.huggingface.fallback_to_http_on_stall:
            modes.append("http")
        return modes

    def download_file(
        self,
        remote_path: str,
        expected_size: int | None,
        file_type: str,
        sha256_path: Path | None = None,
        dry_run: bool = False,
        progress_flush: bool = True,
    ) -> DownloadResult:
        local_path = self.cfg.local_raw_path(remote_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        ok_space, free = ensure_enough_free_space(self.cfg.paths.raw_cache_dir, self.cfg.safety.min_free_disk_gb)
        if not ok_space:
            message = f"free disk below threshold before download: {human_bytes(free)}"
            self.logger.warning(message)
            heartbeat(
                self.cfg.heartbeat_log,
                self.cfg.paths.raw_cache_dir,
                self.cfg.paths.processed_data_dir,
                remote_path,
                "disk_warning",
                message,
                remote_size_bytes=expected_size,
                logger=self.logger,
                dir_size_refresh_interval_seconds=self.cfg.logging.heartbeat_dir_size_refresh_interval_seconds,
                include_dir_sizes=False,
            )
        if local_path.exists() and expected_size is not None and local_path.stat().st_size == expected_size:
            verified_by = self.verify_video_file(remote_path, expected_size, sha256_path) if file_type == "video" else self.verify_downloaded_file(remote_path, expected_size, sha256_path)
            self._set_status_all(
                remote_path,
                file_type,
                VIDEO_DOWNLOADED_VERIFIED,
                f"local raw file already verified by {verified_by}",
                raw_path=str(local_path),
                flush=progress_flush,
            )
            return DownloadResult(local_path, 0, "local", utc_now_iso(), utc_now_iso(), verified_by)
        if local_path.exists() and expected_size is not None and local_path.stat().st_size != expected_size:
            message = f"existing raw file size mismatch; local={local_path.stat().st_size} remote={expected_size}"
            self._set_status_all(remote_path, file_type, VIDEO_FAILED_RETRYABLE, message, raw_path=str(local_path))
            if self.cfg.safety.quarantine_size_mismatched_downloads:
                quarantine = self.cfg.paths.raw_cache_dir / "_quarantine" / f"{_safe_log_stem(remote_path)}_{timestamp_for_filename()}.bad"
                quarantine.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(local_path), str(quarantine))
                self.logger.error("%s; moved to %s", message, quarantine)
            else:
                raise RuntimeError(message)
        if dry_run:
            self.logger.info("[dry-run] would download %s to %s", remote_path, local_path)
            return DownloadResult(local_path, 0, "dry-run", utc_now_iso(), utc_now_iso(), "not_checked")

        modes = self.download_modes_for_file(file_type, expected_size)
        max_attempts = len(modes) if file_type == "video" else max(self.cfg.huggingface.max_attempts_per_file, len(modes))
        worker_script = Path(__file__).resolve().parent / "download_worker.py"
        started_at = utc_now_iso()
        last_error = ""
        used_fallback = False
        for attempt in range(1, max_attempts + 1):
            mode = modes[min(attempt - 1, len(modes) - 1)]
            used_fallback = used_fallback or ("xet" in modes[: attempt - 1] and mode == "http")
            log_path = self.cfg.download_logs_dir / f"{_safe_log_stem(remote_path)}_attempt_{attempt:03d}_{mode}.log"
            self._set_status_all(
                remote_path,
                file_type,
                VIDEO_DOWNLOADING,
                f"download attempt {attempt} mode={mode}",
                raw_path=str(local_path),
                attempt=attempt,
                flush=progress_flush,
            )
            with log_path.open("w", encoding="utf-8", errors="replace") as log_fh:
                cmd = [
                    sys.executable,
                    str(worker_script),
                    "--repo-id",
                    self.cfg.repo_id,
                    "--repo-type",
                    self.cfg.repo_type,
                    "--filename",
                    remote_path,
                    "--local-root",
                    str(self.cfg.paths.raw_cache_dir),
                    "--cache-dir",
                    str(self.cfg.paths.hf_cache),
                    "--hf-home",
                    str(self.cfg.paths.hf_home),
                    "--token-env",
                    self.cfg.huggingface.token_env,
                    "--mode",
                    mode,
                    "--download-timeout",
                    str(self.cfg.huggingface.worker_download_timeout_seconds),
                    "--etag-timeout",
                    str(self.cfg.huggingface.worker_etag_timeout_seconds),
                ]
                observation = self._observe_download_sources(local_path, remote_path, mode)
                observation_checked_at = time.monotonic()
                observation_refresh_interval = max(1.0, float(self.cfg.huggingface.heartbeat_interval_seconds))
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    env=self.cfg.download_subprocess_environment(force_http=mode == "http"),
                )
                xet_activity_paths: tuple[Path, ...] = ()
                last_progress = time.monotonic()
                monitoring_started_at = utc_now_iso()
                last_confirmed_activity_at: str | None = None
                last_heartbeat = time.monotonic()
                poll_interval = max(0.05, float(self.cfg.huggingface.worker_poll_interval_seconds))
                while True:
                    rc = proc.poll()
                    now = time.monotonic()
                    should_refresh_observation = (
                        rc is not None
                        or now - observation_checked_at >= observation_refresh_interval
                        or now - last_progress > self.cfg.huggingface.download_stall_timeout_seconds
                    )
                    if should_refresh_observation:
                        current_observation = self._observe_download_sources(
                            local_path,
                            remote_path,
                            mode,
                            xet_activity_paths=xet_activity_paths,
                        )
                        # The Xet runtime log can appear after its target .incomplete file.
                        # Retry discovery at observation cadence, then stat only known paths.
                        if mode == "xet" and not xet_activity_paths and current_observation.temporary_file_count:
                            discovered_paths = self._find_xet_activity_logs(getattr(proc, "pid", None))
                            if discovered_paths:
                                xet_activity_paths = discovered_paths
                                current_observation = self._observe_download_sources(
                                    local_path,
                                    remote_path,
                                    mode,
                                    xet_activity_paths=xet_activity_paths,
                                )
                        observation_checked_at = now
                        if current_observation.advanced_from(observation, mode):
                            last_progress = now
                            last_confirmed_activity_at = utc_now_iso()
                        observation = current_observation
                    should_heartbeat = rc is not None or now - last_heartbeat >= self.cfg.huggingface.heartbeat_interval_seconds
                    if should_heartbeat:
                        last_heartbeat = now
                        heartbeat(
                            self.cfg.heartbeat_log,
                            self.cfg.paths.raw_cache_dir,
                            self.cfg.paths.processed_data_dir,
                            remote_path,
                            VIDEO_DOWNLOADING,
                            "download still active" if rc is None else "download process exited",
                            download_mode=mode,
                            attempt=attempt,
                            downloaded_bytes=observation.downloaded_bytes,
                            remote_size_bytes=expected_size,
                            last_progress_seconds_ago=now - last_progress,
                            logger=self.logger,
                            dir_size_refresh_interval_seconds=self.cfg.logging.heartbeat_dir_size_refresh_interval_seconds,
                            include_dir_sizes=False,
                        )
                    if rc is not None:
                        break
                    stalled_seconds = now - last_progress
                    if self._should_terminate_for_stall(stalled_seconds):
                        proc.kill()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            pass
                        evidence = {
                            "reason": "no observed download activity changed before timeout",
                            "mode": mode,
                            "stall_timeout_seconds": self.cfg.huggingface.download_stall_timeout_seconds,
                            "last_progress_seconds_ago": stalled_seconds,
                            "monitoring_started_at": monitoring_started_at,
                            "last_confirmed_activity_at": last_confirmed_activity_at,
                            "watchdog_policy": self._watchdog_policy(mode, observation),
                            "observation_source": observation.source,
                            "temporary_files_found": observation.temporary_file_count > 0,
                            "sources": observation.details(),
                        }
                        if mode == "http" and observation.temporary_file_count:
                            last_error = (
                                f"download stalled after {self.cfg.huggingface.download_stall_timeout_seconds}s "
                                f"with reliable target temporary-file evidence"
                            )
                        else:
                            last_error = f"download stalled after {self.cfg.huggingface.download_stall_timeout_seconds}s with no observed {mode} activity"
                        self._set_status_all(
                            remote_path,
                            file_type,
                            VIDEO_FAILED_RETRYABLE,
                            last_error,
                            attempt=attempt,
                            flush=progress_flush,
                            extra={"stall_evidence": evidence},
                        )
                        break
                    time.sleep(poll_interval)
                if proc.returncode != 0:
                    last_error = f"download worker exited with code {proc.returncode}; see {log_path}"
                    self._set_status_all(remote_path, file_type, VIDEO_FAILED_RETRYABLE, last_error, attempt=attempt, flush=progress_flush)
                    continue
            try:
                self._set_status_all(remote_path, file_type, VIDEO_DOWNLOADED_UNVERIFIED, "download process finished", raw_path=str(local_path), attempt=attempt, flush=progress_flush)
                verified_by = self.verify_video_file(remote_path, expected_size, sha256_path) if file_type == "video" else self.verify_downloaded_file(remote_path, expected_size, sha256_path)
                finished_at = utc_now_iso()
                self._set_status_all(
                    remote_path,
                    file_type,
                    VIDEO_DOWNLOADED_VERIFIED,
                    f"download verified by {verified_by}",
                    raw_path=str(local_path),
                    attempt=attempt,
                    extra={"download_started_at": started_at, "download_finished_at": finished_at, "verified_by": verified_by},
                    flush=progress_flush,
                )
                return DownloadResult(local_path, attempt, mode, started_at, finished_at, verified_by, used_fallback)
            except Exception as exc:
                last_error = str(exc)
                self._set_status_all(remote_path, file_type, VIDEO_FAILED_RETRYABLE, last_error, raw_path=str(local_path), attempt=attempt, flush=progress_flush)
        raise RuntimeError(f"failed to download {remote_path}: {last_error}")

    def _observe_download_sources(
        self,
        local_path: Path,
        remote_path: str,
        mode: str,
        xet_activity_paths: tuple[Path, ...] = (),
    ) -> DownloadObservation:
        final_bytes = self._safe_file_size(local_path)
        temp_dir = self._local_dir_download_temp_dir(remote_path)
        prefix = self._local_dir_incomplete_prefix(local_path.name)
        temporary_files: list[tuple[str, int, int]] = []
        unknown_incomplete_file_count = 0
        if temp_dir.is_dir():
            try:
                for candidate in temp_dir.iterdir():
                    if not candidate.name.endswith(".incomplete"):
                        continue
                    if not candidate.name.startswith(prefix + "."):
                        unknown_incomplete_file_count += 1
                        continue
                    try:
                        stat = candidate.stat()
                    except OSError:
                        continue
                    temporary_files.append((candidate.name, stat.st_size, stat.st_mtime_ns))
            except OSError:
                pass
        temporary_files_tuple = tuple(sorted(temporary_files))
        temporary_sizes = [size for _name, size, _mtime in temporary_files_tuple]
        temporary_total_bytes = sum(temporary_sizes)
        temporary_largest_bytes = max(temporary_sizes, default=0)
        xet_activity_files: list[tuple[str, int, int]] = []
        for activity_path in xet_activity_paths:
            try:
                stat = activity_path.stat()
            except OSError:
                continue
            xet_activity_files.append((activity_path.name, stat.st_size, stat.st_mtime_ns))
        xet_activity_files_tuple = tuple(sorted(xet_activity_files))
        if final_bytes:
            downloaded_bytes: int | None = final_bytes
            source = "final"
        elif mode == "http" and temporary_files_tuple:
            downloaded_bytes = temporary_largest_bytes
            source = "hf_local_dir_incomplete"
        elif mode == "xet" and temporary_files_tuple:
            downloaded_bytes = None
            source = "hf_local_dir_incomplete_xet_activity"
        elif mode == "xet" and xet_activity_files_tuple:
            downloaded_bytes = None
            source = "xet_runtime_log_activity"
        else:
            downloaded_bytes = 0 if mode == "http" else None
            source = "none"
        return DownloadObservation(
            downloaded_bytes=downloaded_bytes,
            final_bytes=final_bytes,
            temporary_total_bytes=temporary_total_bytes,
            temporary_largest_bytes=temporary_largest_bytes,
            temporary_files=temporary_files_tuple,
            xet_activity_files=xet_activity_files_tuple,
            unknown_incomplete_file_count=unknown_incomplete_file_count,
            source=source,
        )

    def _local_dir_download_temp_dir(self, remote_path: str) -> Path:
        return self.cfg.paths.raw_cache_dir / ".cache" / "huggingface" / "download" / Path(remote_path).parent

    @staticmethod
    def _local_dir_incomplete_prefix(filename: str) -> str:
        # Matches Hugging Face local_dir temporary names. If this upstream
        # layout changes, byte progress may be unavailable, but timeout remains active.
        metadata_name = f"{filename}.metadata"
        digest = hashlib.sha1(metadata_name.encode("utf-8")).digest()
        return base64.urlsafe_b64encode(digest).decode("ascii")

    def _find_xet_activity_logs(self, worker_pid: int | None) -> tuple[Path, ...]:
        if worker_pid is None:
            return ()
        log_dir = self.cfg.paths.hf_home / "xet" / "logs"
        if not log_dir.is_dir():
            return ()
        try:
            return tuple(sorted(log_dir.glob(f"xet_*_{worker_pid}.log")))
        except OSError:
            return ()

    @staticmethod
    def _safe_file_size(path: Path) -> int:
        try:
            return path.stat().st_size
        except OSError:
            return 0

    def _should_terminate_for_stall(self, stalled_seconds: float) -> bool:
        return stalled_seconds > self.cfg.huggingface.download_stall_timeout_seconds

    @staticmethod
    def _watchdog_policy(mode: str, observation: DownloadObservation) -> str:
        if mode == "xet":
            return "xet_target_write_or_runtime_log_activity_timeout"
        if observation.temporary_file_count:
            return "http_target_local_dir_incomplete_bytes_timeout"
        return "attempt_timeout_without_target_temporary_progress"

    def _set_status_all(
        self,
        remote_path: str,
        file_type: str,
        status: str,
        message: str,
        flush: bool = True,
        **kwargs: Any,
    ) -> None:
        for store in self.stores:
            store.set_file_status(remote_path, file_type, status, message, flush=flush, **kwargs)
