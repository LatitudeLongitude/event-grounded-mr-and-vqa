from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from .constants import (
    VIDEO_DELETING_RAW,
    VIDEO_DONE,
    VIDEO_DOWNLOADED_VERIFIED,
    VIDEO_FAILED_NEED_MANUAL,
    VIDEO_FAILED_RETRYABLE,
    VIDEO_PROCESSED_UNVERIFIED,
    VIDEO_PROCESSED_VERIFIED,
    VIDEO_PROCESSING,
    VIDEO_RAW_DELETED_VERIFIED,
)
from .disk_utils import ensure_enough_free_space, human_bytes
from .ffmpeg_utils import build_ffmpeg_extract_command, expected_frames, probe_video, run_ffmpeg_with_heartbeat
from .frame_resize import expected_frame_width_for_video, frame_resize_enabled, processed_frames_need_resize, with_resize_pending_metadata
from .hf_inventory import RemoteFile, parse_castle_video_path
from .manifests import append_global_manifests, rewrite_final_video_artifacts, sanitize_processing_tmp_paths, update_video_raw_deleted_metadata, video_artifacts_need_raw_deleted_at, write_frame_manifests, write_video_summary
from .progress import atomic_write_text
from .time_utils import timestamp_for_filename, utc_now_iso
from .verifier import verify_processed_video_dir, write_verify_result


class VideoProcessor:
    def __init__(self, cfg: Any, stores: list[Any], downloader: Any, logger: Any) -> None:
        self.cfg = cfg
        self.stores = stores
        self.downloader = downloader
        self.logger = logger

    def process_video(self, remote: RemoteFile, dry_run: bool = False, allow_download: bool = True) -> None:
        remote_path = remote.path
        raw_path = self.cfg.local_raw_path(remote_path)
        final_dir = self.cfg.processed_video_dir(remote_path)
        tmp_dir = final_dir.with_name(final_dir.name + ".__processing_tmp__")
        info = parse_castle_video_path(remote_path)

        existing = verify_processed_video_dir(final_dir, expected_frame_count=None, require_verify_result=True, sample_images=False)
        if dry_run:
            self.logger.info("[dry-run] would process video %s", remote_path)
            return
        if final_dir.exists() and not existing.passed:
            repaired = self._repair_existing_final_artifacts(remote_path, final_dir)
            if repaired:
                existing = verify_processed_video_dir(final_dir, expected_frame_count=None, require_verify_result=True, sample_images=False)
        if existing.passed:
            deleted_raw = False
            if processed_frames_need_resize(self.cfg, final_dir):
                pending_metadata = with_resize_pending_metadata(self.cfg, {"status": VIDEO_PROCESSED_VERIFIED})
                rewrite_final_video_artifacts(final_dir, remote_path, pending_metadata)
                if raw_path.exists() and self.cfg.processing.delete_raw_video_after_verified:
                    self.delete_raw_video(
                        remote,
                        final_dir,
                        expected_frame_count=None,
                        expected_frame_width_override=self.cfg.processing.frame_width,
                    )
                append_global_manifests(self.cfg, final_dir, remote_path)
                self._set_all(
                    remote_path,
                    VIDEO_PROCESSED_VERIFIED,
                    "processed video already verified; frame resize pending",
                    raw_path=str(raw_path),
                    processed_path=str(final_dir),
                )
                return
            if raw_path.exists() and self.cfg.processing.delete_raw_video_after_verified:
                self.delete_raw_video(remote, final_dir)
                deleted_raw = True
            if deleted_raw:
                append_global_manifests(self.cfg, final_dir, remote_path)
            self._set_all(remote_path, VIDEO_DONE, "processed video already verified", raw_path=str(raw_path), processed_path=str(final_dir))
            return

        if not raw_path.exists():
            if not allow_download:
                self._set_all(remote_path, VIDEO_FAILED_RETRYABLE, "raw video missing; processing worker requested redownload", raw_path=str(raw_path), processed_path=str(final_dir))
                raise FileNotFoundError(f"raw video missing before processing: {remote_path}")
            self.downloader.download_file(remote_path, remote.size, "video", dry_run=False)
        else:
            self.downloader.verify_video_file(remote_path, remote.size)
            self._set_all(remote_path, VIDEO_DOWNLOADED_VERIFIED, "raw video already verified before processing", raw_path=str(raw_path))

        if tmp_dir.exists():
            broken_tmp = tmp_dir.with_name(f"{tmp_dir.name}.__incomplete_{timestamp_for_filename()}__")
            shutil.move(str(tmp_dir), str(broken_tmp))
            self.logger.warning("moved incomplete processing tmp dir to %s", broken_tmp)
        if final_dir.exists() and not existing.passed:
            if not raw_path.exists():
                self._set_all(remote_path, VIDEO_FAILED_NEED_MANUAL, "processed output incomplete and raw video missing", processed_path=str(final_dir))
                raise RuntimeError(f"cannot repair incomplete processed output without raw video: {remote_path}")
            broken = final_dir.with_name(final_dir.name + f".__broken_{utc_now_iso().replace(':', '').replace('-', '')}__")
            shutil.move(str(final_dir), str(broken))
            self.logger.warning("moved incomplete processed output to %s", broken)

        ok_space, free = ensure_enough_free_space(self.cfg.paths.processed_data_dir, self.cfg.safety.min_free_disk_gb)
        if not ok_space:
            self.logger.warning("free disk below threshold before ffmpeg: %s", human_bytes(free))

        self._set_all(remote_path, VIDEO_PROCESSING, "ffmpeg processing started", raw_path=str(raw_path), processed_path=str(tmp_dir))
        processing_started_at = utc_now_iso()
        tmp_frames = tmp_dir / "frames"
        tmp_frames.mkdir(parents=True, exist_ok=True)
        probe = probe_video(self.cfg.tools.ffprobe, raw_path)
        expected = expected_frames(probe.duration_seconds, self.cfg.processing.fps)
        tolerance = max(2, int(expected * 0.002)) if expected else 2
        cmd = build_ffmpeg_extract_command(
            self.cfg.tools.ffmpeg,
            raw_path,
            tmp_frames,
            self.cfg.processing.fps,
            self.cfg.processing.frame_width,
            self.cfg.processing.jpg_q,
        )
        ffmpeg_cmd_text = " ".join(cmd)
        def on_ffmpeg_progress(frame_count: int, latest_frame_name: str | None, last_progress_seconds_ago: float) -> None:
            self._set_all(
                remote_path,
                VIDEO_PROCESSING,
                "ffmpeg processing heartbeat",
                raw_path=str(raw_path),
                processed_path=str(tmp_dir),
                extra={
                    "tmp_dir": str(tmp_dir),
                    "frame_count": frame_count,
                    "latest_frame_name": latest_frame_name,
                    "last_progress_seconds_ago": last_progress_seconds_ago,
                    "last_progress_at": utc_now_iso(),
                    "processing_started_at": processing_started_at,
                },
            )

        run_ffmpeg_with_heartbeat(
            cmd,
            self.cfg.ffmpeg_logs_dir / f"{_safe_log_stem(remote_path)}.log",
            tmp_frames,
            self.cfg,
            remote_path,
            self.logger,
            progress_callback=on_ffmpeg_progress,
            heartbeat_include_dir_sizes=False,
        )
        extraction_finished_at = utc_now_iso()
        processing_finished_at = extraction_finished_at
        metadata = {
            "repo_id": self.cfg.repo_id,
            "remote_video_path": remote_path,
            "day": info.day,
            "source": info.source,
            "video_hour_stem": info.video_hour_stem,
            "fps": self.cfg.processing.fps,
            "width": self.cfg.processing.frame_width,
            "jpg_q": self.cfg.processing.jpg_q,
            "ffmpeg_command": ffmpeg_cmd_text,
            "video_duration_seconds": probe.duration_seconds,
            "video_size_bytes": raw_path.stat().st_size,
            "download_started_at": self._progress_field(remote_path, "download_started_at"),
            "download_finished_at": self._progress_field(remote_path, "download_finished_at"),
            "processing_started_at": processing_started_at,
            "processing_finished_at": processing_finished_at,
            "frame_extraction_finished_at": extraction_finished_at,
            "raw_deleted_at": "",
            "sha256_or_size_check": self._progress_field(remote_path, "verified_by") or "size",
            "status": VIDEO_PROCESSED_UNVERIFIED,
        }
        if frame_resize_enabled(self.cfg):
            metadata = with_resize_pending_metadata(self.cfg, metadata)

        processing_finished_at = extraction_finished_at
        metadata["processing_finished_at"] = processing_finished_at
        atomic_write_text(tmp_dir / "ffmpeg_command.txt", ffmpeg_cmd_text + "\n")
        write_frame_manifests(tmp_dir, remote_path, metadata)
        summary = {
            **metadata,
            "processed_video_dir": str(final_dir),
            "expected_frame_count": expected,
            "frame_count": len(list(tmp_frames.glob("*.jpg"))),
            "tolerance": tolerance,
            "ffprobe": {
                "duration_seconds": probe.duration_seconds,
                "width": probe.width,
                "height": probe.height,
                "avg_frame_rate": probe.avg_frame_rate,
                "codec_name": probe.codec_name,
            },
        }
        write_video_summary(tmp_dir, summary)
        self._set_all(remote_path, VIDEO_PROCESSED_UNVERIFIED, "ffmpeg output written, verifying", raw_path=str(raw_path), processed_path=str(tmp_dir))
        verify = verify_processed_video_dir(tmp_dir, expected_frame_count=expected, tolerance=tolerance, expected_frame_width=self.cfg.processing.frame_width)
        write_verify_result(tmp_dir, verify)
        verify = verify_processed_video_dir(tmp_dir, expected_frame_count=expected, tolerance=tolerance, require_verify_result=True, expected_frame_width=self.cfg.processing.frame_width)
        if not verify.passed:
            self._set_all(remote_path, VIDEO_FAILED_RETRYABLE, "processed verification failed: " + "; ".join(verify.errors), raw_path=str(raw_path), processed_path=str(tmp_dir))
            raise RuntimeError(f"processed verification failed for {remote_path}: {verify.errors}")
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp_dir, final_dir)
        final_metadata = {
            **metadata,
            "processed_video_dir": str(final_dir),
            "expected_frame_count": expected,
            "tolerance": tolerance,
            "status": VIDEO_PROCESSED_UNVERIFIED,
            "ffmpeg_command": sanitize_processing_tmp_paths(ffmpeg_cmd_text, final_dir),
        }
        rewrite_final_video_artifacts(final_dir, remote_path, final_metadata)
        final_verify = verify_processed_video_dir(final_dir, expected_frame_count=expected, tolerance=tolerance, expected_frame_width=self.cfg.processing.frame_width)
        write_verify_result(final_dir, final_verify)
        final_verify = verify_processed_video_dir(final_dir, expected_frame_count=expected, tolerance=tolerance, require_verify_result=True, expected_frame_width=self.cfg.processing.frame_width)
        if not final_verify.passed:
            self._set_all(remote_path, VIDEO_FAILED_RETRYABLE, "final output verification failed: " + "; ".join(final_verify.errors), raw_path=str(raw_path), processed_path=str(final_dir))
            raise RuntimeError(f"final output verification failed for {remote_path}: {final_verify.errors}")
        self._set_all(
            remote_path,
            VIDEO_PROCESSED_VERIFIED,
            "processed output verified",
            raw_path=str(raw_path),
            processed_path=str(final_dir),
            extra={
                "expected_frame_count": expected,
                "frame_count": final_verify.frame_count,
                "processing_started_at": processing_started_at,
                "processing_finished_at": processing_finished_at,
            },
        )
        if not frame_resize_enabled(self.cfg) and self.cfg.processing.delete_raw_video_after_verified:
            self.delete_raw_video(remote, final_dir, expected_frame_count=expected, tolerance=tolerance)
        if not frame_resize_enabled(self.cfg):
            append_global_manifests(self.cfg, final_dir, remote_path)
            self._set_all(remote_path, VIDEO_DONE, "video shard done", raw_path=str(raw_path), processed_path=str(final_dir))
        else:
            if self.cfg.processing.delete_raw_video_after_verified:
                self.delete_raw_video(
                    remote,
                    final_dir,
                    expected_frame_count=expected,
                    tolerance=tolerance,
                    expected_frame_width_override=self.cfg.processing.frame_width,
                )
            append_global_manifests(self.cfg, final_dir, remote_path)
            self._set_all(
                remote_path,
                VIDEO_PROCESSED_VERIFIED,
                "frame extraction verified; frame resize pending",
                raw_path=str(raw_path),
                processed_path=str(final_dir),
            )

    def delete_raw_video(
        self,
        remote: RemoteFile,
        final_dir: Path,
        expected_frame_count: int | None = None,
        tolerance: int = 2,
        expected_frame_width_override: int | None = None,
    ) -> None:
        remote_path = remote.path
        raw_path = self.cfg.local_raw_path(remote_path)
        verify = verify_processed_video_dir(
            final_dir,
            expected_frame_count,
            tolerance,
            require_verify_result=True,
            expected_frame_width=expected_frame_width_override if expected_frame_width_override is not None else expected_frame_width_for_video(self.cfg, final_dir),
        )
        if not verify.passed:
            self._set_all(remote_path, VIDEO_FAILED_NEED_MANUAL, "refusing raw delete because processed verification failed: " + "; ".join(verify.errors), raw_path=str(raw_path), processed_path=str(final_dir))
            raise RuntimeError(f"refusing to delete raw video; processed output is not verified: {remote_path}")
        if not raw_path.exists():
            deleted_at = ""
            if video_artifacts_need_raw_deleted_at(final_dir):
                deleted_at = utc_now_iso()
                self._write_raw_deleted_metadata(
                    remote_path,
                    final_dir,
                    raw_path,
                    deleted_at,
                    expected_frame_count,
                    tolerance,
                    expected_frame_width_override=expected_frame_width_override,
                    raw_deleted_at_source="raw_absent_verified_time",
                )
            self._set_all(
                remote_path,
                VIDEO_RAW_DELETED_VERIFIED,
                "raw video already absent after verification",
                raw_path=str(raw_path),
                processed_path=str(final_dir),
                extra={"raw_deleted_at": deleted_at} if deleted_at else None,
            )
            return
        self._set_all(remote_path, VIDEO_DELETING_RAW, "deleting verified raw video", raw_path=str(raw_path), processed_path=str(final_dir))
        raw_path.unlink()
        if raw_path.exists():
            self._set_all(remote_path, VIDEO_FAILED_NEED_MANUAL, "raw delete attempted but file still exists", raw_path=str(raw_path), processed_path=str(final_dir))
            raise RuntimeError(f"raw video still exists after delete attempt: {raw_path}")
        deleted_at = utc_now_iso()
        self._write_raw_deleted_metadata(
            remote_path,
            final_dir,
            raw_path,
            deleted_at,
            expected_frame_count,
            tolerance,
            expected_frame_width_override=expected_frame_width_override,
        )
        self._set_all(
            remote_path,
            VIDEO_RAW_DELETED_VERIFIED,
            "raw delete verified",
            raw_path=str(raw_path),
            processed_path=str(final_dir),
            extra={"raw_deleted_at": deleted_at},
        )

    def _write_raw_deleted_metadata(
        self,
        remote_path: str,
        final_dir: Path,
        raw_path: Path,
        raw_deleted_at: str,
        expected_frame_count: int | None,
        tolerance: int,
        expected_frame_width_override: int | None = None,
        raw_deleted_at_source: str | None = None,
    ) -> None:
        try:
            update_video_raw_deleted_metadata(
                final_dir,
                raw_deleted_at,
                status=VIDEO_RAW_DELETED_VERIFIED,
                raw_deleted_at_source=raw_deleted_at_source,
            )
            post = verify_processed_video_dir(
                final_dir,
                expected_frame_count,
                tolerance,
                require_verify_result=True,
                expected_frame_width=expected_frame_width_override if expected_frame_width_override is not None else expected_frame_width_for_video(self.cfg, final_dir),
            )
            write_verify_result(final_dir, post)
            if not post.passed:
                self._set_all(remote_path, VIDEO_FAILED_NEED_MANUAL, "processed verification failed after raw deletion metadata update: " + "; ".join(post.errors), raw_path=str(raw_path), processed_path=str(final_dir))
                raise RuntimeError(f"processed output failed after raw deletion metadata update: {remote_path}")
        except Exception as exc:
            self.logger.exception("raw deletion metadata update failed for %s: %s", remote_path, exc)
            self._set_all(remote_path, VIDEO_FAILED_NEED_MANUAL, f"raw deleted but metadata update failed: {exc}", raw_path=str(raw_path), processed_path=str(final_dir))
            raise

    def _progress_field(self, remote_path: str, field: str) -> Any:
        for store in self.stores:
            entry = store.data.get("files", {}).get(remote_path, {})
            if field in entry:
                return entry[field]
        return ""

    def _repair_existing_final_artifacts(self, remote_path: str, final_dir: Path) -> bool:
        try:
            summary = rewrite_final_video_artifacts(final_dir, remote_path, {"status": VIDEO_PROCESSED_UNVERIFIED})
            expected = summary.get("expected_frame_count")
            tolerance = int(summary.get("tolerance") or 2)
            verify = verify_processed_video_dir(
                final_dir,
                expected_frame_count=int(expected) if expected is not None else None,
                tolerance=tolerance,
            )
            write_verify_result(final_dir, verify)
            verify = verify_processed_video_dir(
                final_dir,
                expected_frame_count=int(expected) if expected is not None else None,
                tolerance=tolerance,
                require_verify_result=True,
                sample_images=False,
            )
            if verify.passed:
                self.logger.info("repaired stale final video artifacts for %s", remote_path)
                return True
            self.logger.warning("could not repair final video artifacts for %s: %s", remote_path, verify.errors)
            return False
        except Exception as exc:
            self.logger.warning("final video artifact repair failed for %s: %s", remote_path, exc)
            return False

    def _set_all(self, remote_path: str, status: str, message: str, **kwargs: Any) -> None:
        for store in self.stores:
            store.set_file_status(remote_path, "video", status, message, **kwargs)


def _safe_log_stem(remote_path: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in remote_path).strip("_")[:180]
