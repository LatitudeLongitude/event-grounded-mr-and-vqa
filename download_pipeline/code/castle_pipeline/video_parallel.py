from __future__ import annotations

import json
import queue
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from .constants import (
    VIDEO_DONE,
    VIDEO_DOWNLOADED_VERIFIED,
    VIDEO_FAILED_RETRYABLE,
    VIDEO_PENDING,
    VIDEO_PROCESSED_VERIFIED,
    VIDEO_RAW_DELETED_VERIFIED,
    VIDEO_RESIZING,
)
from .downloader import DownloadManager, DownloadResult
from .ffmpeg_utils import probe_video
from .frame_resize import cleanup_frame_resize_source, ensure_video_frames_resized, expected_frame_width, frame_resize_enabled, processed_frames_need_resize
from .hf_inventory import RemoteFile
from .logging_utils import heartbeat
from .manifests import append_global_manifests, rewrite_final_video_artifacts
from .processor import VideoProcessor
from .progress import ProgressStore
from .time_utils import timestamp_for_filename
from .verifier import verify_processed_video_dir, write_verify_result
from .video_queue import VideoRedownloadQueue


class VideoProcessingWorker:
    def __init__(
        self,
        cfg: Any,
        processing_store: ProgressStore,
        processor: Any,
        redownload_queue: VideoRedownloadQueue,
        resize_queue: "queue.Queue[RemoteFile | None]",
        logger: Any,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.cfg = cfg
        self.processing_store = processing_store
        self.processor = processor
        self.redownload_queue = redownload_queue
        self.resize_queue = resize_queue
        self.logger = logger
        self.stop_event = stop_event

    def validate_raw_or_request_redownload(self, remote: RemoteFile) -> bool:
        raw = self.cfg.local_raw_path(remote.path)
        local_size = raw.stat().st_size if raw.exists() else None
        reason = ""
        ffprobe_error = None
        if not raw.exists():
            reason = "raw_missing"
        elif remote.size is not None and local_size != remote.size:
            reason = "raw_size_mismatch"
        else:
            try:
                probe_video(self.cfg.tools.ffprobe, raw)
                return True
            except Exception as exc:
                reason = "ffprobe_failed"
                ffprobe_error = str(exc)
        self.redownload_queue.open_request(
            remote_path=remote.path,
            finder="processing-worker",
            reason=reason,
            local_size=local_size,
            expected_size=remote.size,
            ffprobe_error=ffprobe_error,
        )
        self.processing_store.set_file_status(
            remote.path,
            "video",
            VIDEO_FAILED_RETRYABLE,
            f"raw validation failed before processing; redownload requested: {reason}",
            raw_path=str(raw),
            processed_path=str(self.cfg.processed_video_dir(remote.path)),
            extra={"local_size": local_size, "expected_size": remote.size, "ffprobe_error": ffprobe_error},
        )
        return False

    def process(self, remote: RemoteFile, dry_run: bool = False) -> None:
        raw = self.cfg.local_raw_path(remote.path)
        final = self.cfg.processed_video_dir(remote.path)
        if processed_frames_need_resize(self.cfg, final):
            if self._should_stop():
                return
            self.processing_store.set_file_status(
                remote.path,
                "video",
                VIDEO_PROCESSED_VERIFIED,
                "processed output verified; frame resize queued",
                raw_path=str(raw),
                processed_path=str(final),
            )
            self._put_unless_stopped(self.resize_queue, remote)
            return
        if not self.validate_raw_or_request_redownload(remote):
            return
        self.processor.process_video(remote, dry_run=dry_run, allow_download=False)
        if frame_resize_enabled(self.cfg) and processed_frames_need_resize(self.cfg, final):
            if self._should_stop():
                return
            self.processing_store.set_file_status(
                remote.path,
                "video",
                VIDEO_PROCESSED_VERIFIED,
                "frame extraction verified; frame resize queued",
                raw_path=str(raw),
                processed_path=str(final),
            )
            self._put_unless_stopped(self.resize_queue, remote)

    def _should_stop(self) -> bool:
        return self.stop_event is not None and self.stop_event.is_set()

    def _put_unless_stopped(self, target: "queue.Queue[RemoteFile | None]", remote: RemoteFile) -> bool:
        while not self._should_stop():
            try:
                target.put(remote, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False


class VideoDownloadWorker:
    def __init__(
        self,
        cfg: Any,
        download_store: ProgressStore,
        downloader: DownloadManager,
        redownload_queue: VideoRedownloadQueue,
        ready_queue: "queue.Queue[RemoteFile | None]",
        resize_queue: "queue.Queue[RemoteFile | None]",
        logger: Any,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.cfg = cfg
        self.download_store = download_store
        self.downloader = downloader
        self.redownload_queue = redownload_queue
        self.ready_queue = ready_queue
        self.resize_queue = resize_queue
        self.logger = logger
        self.stop_event = stop_event

    def download_and_enqueue(self, remote: RemoteFile, force_redownload: bool = False, dry_run: bool = False) -> DownloadResult | None:
        final = self.cfg.processed_video_dir(remote.path)
        raw = self.cfg.local_raw_path(remote.path)
        if not raw.exists() and processed_frames_need_resize(self.cfg, final):
            if self._should_stop():
                return None
            self.download_store.set_file_status(
                remote.path,
                "video",
                VIDEO_DONE,
                "processed output exists and frame resize is queued without raw download",
                processed_path=str(final),
            )
            self._put_unless_stopped(self.resize_queue, remote)
            return None
        processed = verify_processed_video_dir(final, expected_frame_count=None, require_verify_result=True, sample_images=False)
        if processed.passed and not raw.exists():
            self.download_store.set_file_status(
                remote.path,
                "video",
                VIDEO_DONE,
                "processed output already verified and raw absent; download skipped",
                processed_path=str(final),
            )
            return None
        if force_redownload:
            self._quarantine_existing_raw(remote.path)
        if dry_run:
            self.logger.info("[dry-run] would download video %s", remote.path)
            self.download_store.set_file_status(remote.path, "video", VIDEO_PENDING, "dry-run download skipped")
            return None
        started = time.monotonic()
        try:
            result = self.downloader.download_file(remote.path, remote.size, "video", dry_run=False)
        except Exception:
            if force_redownload:
                raise
            if self.cfg.local_raw_path(remote.path).exists():
                self._quarantine_existing_raw(remote.path)
                result = self.downloader.download_file(remote.path, remote.size, "video", dry_run=False)
            else:
                raise
        elapsed = max(time.monotonic() - started, 0.001)
        if result.local_path.exists():
            speed = result.local_path.stat().st_size / elapsed
            self.logger.info(
                "video download verified path=%s mode=%s elapsed_seconds=%.1f approx_bytes_per_second=%.1f note=%s",
                remote.path,
                result.final_mode,
                elapsed,
                speed,
                "Xet may materialize small files in one step, so speed is approximate",
            )
        self._put_unless_stopped(self.ready_queue, remote)
        return result

    def handle_one_redownload_request(self, by_path: dict[str, RemoteFile], dry_run: bool = False) -> bool:
        request = self.redownload_queue.claim_open("download-worker")
        if request is None:
            return False
        remote = by_path.get(str(request["remote_path"]))
        if remote is None:
            return False
        self.download_and_enqueue(remote, force_redownload=True, dry_run=dry_run)
        self.redownload_queue.mark_redownloaded(remote.path, "download-worker")
        self.redownload_queue.mark_verified(remote.path, "download-worker")
        self.redownload_queue.close(remote.path, "download-worker")
        return True

    def _quarantine_existing_raw(self, remote_path: str) -> Path | None:
        raw = self.cfg.local_raw_path(remote_path)
        if not raw.exists():
            return None
        try:
            rel = raw.relative_to(self.cfg.paths.raw_cache_dir)
        except ValueError:
            rel = Path(raw.name)
        quarantine = self.cfg.paths.raw_cache_dir / "_quarantine" / "video_raw" / rel
        quarantine = quarantine.with_name(f"{quarantine.name}.{timestamp_for_filename()}.bad")
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(raw), str(quarantine))
        self.logger.warning("quarantined raw video before redownload %s -> %s", raw, quarantine)
        return quarantine

    def _should_stop(self) -> bool:
        return self.stop_event is not None and self.stop_event.is_set()

    def _put_unless_stopped(self, target: "queue.Queue[RemoteFile | None]", remote: RemoteFile) -> bool:
        while not self._should_stop():
            try:
                target.put(remote, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False


class FrameResizeWorker:
    def __init__(self, cfg: Any, resize_store: ProgressStore, logger: Any) -> None:
        self.cfg = cfg
        self.resize_store = resize_store
        self.logger = logger
        self.raw_deleter = VideoProcessor(cfg, [resize_store], downloader=object(), logger=logger)

    def process(self, remote: RemoteFile, dry_run: bool = False) -> None:
        if not frame_resize_enabled(self.cfg):
            return
        remote_path = remote.path
        final = self.cfg.processed_video_dir(remote_path)
        raw = self.cfg.local_raw_path(remote_path)
        if not processed_frames_need_resize(self.cfg, final):
            self.resize_store.set_file_status(
                remote_path,
                "video",
                VIDEO_DONE,
                "frame resize already satisfied",
                raw_path=str(raw),
                processed_path=str(final),
            )
            return
        if dry_run:
            self.logger.info("[dry-run] would resize extracted frames for %s", remote_path)
            return

        target_width = expected_frame_width(self.cfg)
        self.resize_store.set_file_status(
            remote_path,
            "video",
            VIDEO_RESIZING,
            "frame resize folder started",
            raw_path=str(raw),
            processed_path=str(final),
            extra={"frame_resize_workers": int(getattr(self.cfg.processing, "frame_resize_workers", 6) or 6)},
        )
        heartbeat(
            self.cfg.heartbeat_log,
            self.cfg.paths.raw_cache_dir,
            self.cfg.paths.processed_data_dir,
            remote_path,
            VIDEO_RESIZING,
            "frame resize folder started",
            logger=self.logger,
            dir_size_refresh_interval_seconds=self.cfg.logging.heartbeat_dir_size_refresh_interval_seconds,
            include_dir_sizes=False,
        )
        try:
            metadata = self._read_metadata(final)
            metadata = ensure_video_frames_resized(
                self.cfg,
                final,
                remote_path,
                metadata,
                self.logger,
                progress_callback=None,
            )
            summary = rewrite_final_video_artifacts(final, remote_path, metadata)
            expected = summary.get("expected_frame_count")
            tolerance = int(summary.get("tolerance") or 2)
            expected_count = int(expected) if expected is not None else None
            verify = verify_processed_video_dir(
                final,
                expected_frame_count=expected_count,
                tolerance=tolerance,
                expected_frame_width=target_width,
            )
            write_verify_result(final, verify)
            verify = verify_processed_video_dir(
                final,
                expected_frame_count=expected_count,
                tolerance=tolerance,
                require_verify_result=True,
                sample_images=False,
                expected_frame_width=target_width,
            )
            if not verify.passed:
                raise RuntimeError("frame resize verification failed: " + "; ".join(verify.errors))
            cleanup_frame_resize_source(self.cfg, final, self.logger)
            if self.cfg.processing.delete_raw_video_after_verified:
                self.raw_deleter.delete_raw_video(remote, final, expected_frame_count=expected_count, tolerance=tolerance)
            append_global_manifests(self.cfg, final, remote_path)
            self.resize_store.set_file_status(
                remote_path,
                "video",
                VIDEO_DONE,
                "frame resize folder completed",
                raw_path=str(raw),
                processed_path=str(final),
                extra={"frame_count": verify.frame_count, "frame_width_after_resize": target_width},
            )
            heartbeat(
                self.cfg.heartbeat_log,
                self.cfg.paths.raw_cache_dir,
                self.cfg.paths.processed_data_dir,
                remote_path,
                VIDEO_DONE,
                "frame resize folder completed",
                frame_count=verify.frame_count,
                logger=self.logger,
                dir_size_refresh_interval_seconds=self.cfg.logging.heartbeat_dir_size_refresh_interval_seconds,
                include_dir_sizes=False,
            )
        except Exception as exc:
            self.resize_store.set_file_status(
                remote_path,
                "video",
                VIDEO_FAILED_RETRYABLE,
                f"frame resize failed: {exc}",
                raw_path=str(raw),
                processed_path=str(final),
            )
            raise

    def _read_metadata(self, video_dir: Path) -> dict[str, Any]:
        summary_path = video_dir / "video_processing_summary.json"
        if not summary_path.exists():
            return {}
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}


class VideoPipelineCoordinator:
    def __init__(
        self,
        cfg: Any,
        pipeline_store: ProgressStore,
        download_store: ProgressStore,
        processing_store: ProgressStore,
        resize_store: ProgressStore,
        logger: Any,
    ) -> None:
        self.cfg = cfg
        self.pipeline_store = pipeline_store
        self.download_store = download_store
        self.processing_store = processing_store
        self.resize_store = resize_store
        self.logger = logger
        self.redownload_queue = VideoRedownloadQueue(cfg.video_redownload_queue_path)

    def run(self, videos: list[RemoteFile], dry_run: bool = False) -> dict[str, Any]:
        ready: queue.Queue[RemoteFile | None] = queue.Queue(maxsize=1)
        resize_ready: queue.Queue[RemoteFile | None] = queue.Queue(maxsize=1)
        errors: list[str] = []
        errors_lock = threading.Lock()
        stop_event = threading.Event()
        by_path = {remote.path: remote for remote in videos}
        downloader = DownloadManager(self.cfg, [self.download_store], self.logger)
        verifier_downloader = DownloadManager(self.cfg, [], self.logger)
        processor = VideoProcessor(self.cfg, [self.processing_store], verifier_downloader, self.logger)
        processing_active = threading.Event()
        resize_active = threading.Event()
        download_worker = VideoDownloadWorker(
            self.cfg,
            self.download_store,
            downloader,
            self.redownload_queue,
            ready,
            resize_ready,
            self.logger,
            stop_event,
        )
        processing_worker = VideoProcessingWorker(
            self.cfg,
            self.processing_store,
            processor,
            self.redownload_queue,
            resize_ready,
            self.logger,
            stop_event,
        )
        resize_worker = FrameResizeWorker(self.cfg, self.resize_store, self.logger)

        def record_error(worker_name: str, exc: Exception) -> None:
            with errors_lock:
                errors.append(f"{worker_name}: {exc}")
            stop_event.set()
            self.logger.exception("%s failed; stopping video pipeline after active tasks settle: %s", worker_name, exc)

        def drain_queue(target: "queue.Queue[RemoteFile | None]", queue_name: str) -> int:
            drained = 0
            while True:
                try:
                    target.get_nowait()
                except queue.Empty:
                    break
                else:
                    target.task_done()
                    drained += 1
            if drained:
                self.logger.warning("drained %d pending items from %s after worker error", drained, queue_name)
            return drained

        def put_sentinel(target: "queue.Queue[RemoteFile | None]", queue_name: str) -> None:
            while True:
                try:
                    target.put(None, timeout=0.2)
                    return
                except queue.Full:
                    if stop_event.is_set():
                        drain_queue(target, queue_name)

        def wait_for_queue(target: "queue.Queue[RemoteFile | None]", queue_name: str) -> None:
            while True:
                with target.all_tasks_done:
                    if target.unfinished_tasks == 0:
                        return
                if stop_event.is_set():
                    drain_queue(target, queue_name)
                time.sleep(0.1)

        def download_loop() -> None:
            try:
                for remote in videos:
                    if stop_event.is_set():
                        break
                    while not stop_event.is_set() and download_worker.handle_one_redownload_request(by_path, dry_run=dry_run):
                        pass
                    if stop_event.is_set():
                        break
                    download_worker.download_and_enqueue(remote, dry_run=dry_run)
                while True:
                    if stop_event.is_set():
                        break
                    handled = False
                    while not stop_event.is_set() and download_worker.handle_one_redownload_request(by_path, dry_run=dry_run):
                        handled = True
                    if not handled and ready.empty() and not processing_active.is_set():
                        break
                    time.sleep(1)
            except Exception as exc:
                record_error("download-worker", exc)
            finally:
                put_sentinel(ready, "ready")

        def process_loop() -> None:
            try:
                while True:
                    if stop_event.is_set():
                        return
                    try:
                        remote = ready.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    try:
                        if remote is None:
                            return
                        if stop_event.is_set():
                            return
                        processing_active.set()
                        processing_worker.process(remote, dry_run=dry_run)
                    finally:
                        processing_active.clear()
                        ready.task_done()
            except Exception as exc:
                record_error("processing-worker", exc)
            finally:
                put_sentinel(resize_ready, "resize_ready")

        def resize_loop() -> None:
            try:
                while True:
                    if stop_event.is_set():
                        return
                    try:
                        remote = resize_ready.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    try:
                        if remote is None:
                            return
                        if stop_event.is_set():
                            return
                        resize_active.set()
                        resize_worker.process(remote, dry_run=dry_run)
                    finally:
                        resize_active.clear()
                        resize_ready.task_done()
            except Exception as exc:
                record_error("resize-worker", exc)

        self.pipeline_store.set_current(None)
        download_thread = threading.Thread(target=download_loop, name="castle-video-download-worker", daemon=False)
        process_thread = threading.Thread(target=process_loop, name="castle-video-processing-worker", daemon=False)
        resize_thread = threading.Thread(target=resize_loop, name="castle-frame-resize-worker", daemon=False)
        resize_thread.start()
        process_thread.start()
        download_thread.start()
        download_thread.join()
        wait_for_queue(ready, "ready")
        process_thread.join()
        wait_for_queue(resize_ready, "resize_ready")
        resize_thread.join()
        self._write_pipeline_summary(videos)
        if errors:
            raise RuntimeError("; ".join(errors))
        return {"videos": len(videos), "exception_queue": self.redownload_queue.summary()}

    def _write_pipeline_summary(self, videos: list[RemoteFile]) -> None:
        for remote in videos:
            resize_status = self.resize_store.get_status(remote.path)
            processing_status = self.processing_store.get_status(remote.path)
            download_status = self.download_store.get_status(remote.path)
            status = resize_status or processing_status or download_status or VIDEO_PENDING
            message = "coordinator summarized video worker state"
            if resize_status in {VIDEO_DONE, VIDEO_RAW_DELETED_VERIFIED, VIDEO_PROCESSED_VERIFIED}:
                status = resize_status
            elif processing_status in {VIDEO_DONE, VIDEO_RAW_DELETED_VERIFIED, VIDEO_PROCESSED_VERIFIED}:
                status = processing_status
            elif download_status == VIDEO_DOWNLOADED_VERIFIED:
                status = download_status
            self.pipeline_store.set_file_status(
                remote.path,
                "video",
                status,
                message,
                raw_path=str(self.cfg.local_raw_path(remote.path)),
                processed_path=str(self.cfg.processed_video_dir(remote.path)),
            )
        self.pipeline_store.set_current(None)
