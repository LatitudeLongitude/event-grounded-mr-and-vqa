from __future__ import annotations

import json
import csv
from typing import Any

from .config import PipelineConfig
from .downloader import DownloadManager
from .frame_resize import expected_frame_width_for_video
from .hf_inventory import RemoteFile, inventory_remote_files, scan_remote_inventory
from .logging_utils import ProgressReporter, setup_logging
from .processor import VideoProcessor
from .progress import ProgressStore, atomic_write_json, atomic_write_text
from .reconciliation import cleanup_processing_tmp_dirs, load_inventory_if_exists, reconcile_progress
from .hash_utils import read_sha256_file, sha256_file
from .sync_non_video import assess_non_video_files, sync_non_video_files
from .validation import validate_environment
from .verifier import verify_processed_video_dir
from .video_parallel import VideoPipelineCoordinator
from .video_queue import VideoRedownloadQueue


class CastlePipeline:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.cfg.ensure_directories()
        self.logger = setup_logging(self.cfg.logs_dir, self.cfg.logging.level)
        self.pipeline_store = ProgressStore(self.cfg.video_pipeline_progress_path, "video_pipeline", self.cfg.video_pipeline_events_path)
        self.raw_store = ProgressStore(self.cfg.video_download_progress_path, "video_download", self.cfg.video_download_events_path)
        self.processed_store = ProgressStore(self.cfg.video_processing_progress_path, "video_processing", self.cfg.video_processing_events_path)
        self.resize_store = ProgressStore(self.cfg.video_resize_progress_path, "video_resize", self.cfg.video_resize_events_path)
        self.stores = [self.pipeline_store, self.raw_store, self.processed_store]
        for store in [*self.stores, self.resize_store]:
            store.load()
        self.downloader = DownloadManager(self.cfg, [], self.logger)
        self.processor = VideoProcessor(self.cfg, [self.processed_store], self.downloader, self.logger)

    def validate(self) -> dict[str, Any]:
        report = validate_environment(self.cfg)
        atomic_write_json(self.cfg.paths.code_dir / "validate_report.json", report.to_dict())
        if report.passed:
            self.logger.info("environment validation passed")
        else:
            self.logger.error("environment validation failed: %s", report.errors)
        for warning in report.warnings:
            self.logger.warning(warning)
        return report.to_dict()

    def scan(self) -> dict[str, Any]:
        return scan_remote_inventory(self.cfg, self.logger)

    def inventory(self) -> dict[str, Any]:
        inventory = load_inventory_if_exists(self.cfg)
        if inventory is None:
            inventory = self.scan()
        return inventory

    def repair(self) -> dict[str, Any]:
        cleanup_processing_tmp_dirs(self.cfg, self.logger)
        report = reconcile_progress(
            self.cfg,
            self.stores,
            load_inventory_if_exists(self.cfg),
            self.logger,
            backup=self.cfg.safety.backup_progress_before_repair,
            include_non_video=False,
        )
        atomic_write_json(self.cfg.paths.code_dir / "repair_report.json", report)
        return report

    def sync_non_video(self, dry_run: bool = False) -> list[dict[str, Any]]:
        inventory = self.inventory()
        return sync_non_video_files(self.cfg, inventory, self.downloader, self.logger, dry_run=dry_run)

    def run(self, smoke_test: bool = False, limit_videos: int | None = None, dry_run: bool = False, skip_non_video: bool = False) -> None:
        self.logger.info("run command started smoke_test=%s limit_videos=%s dry_run=%s skip_non_video=%s", smoke_test, limit_videos, dry_run, skip_non_video)
        validation = self.validate()
        if not validation.get("passed"):
            raise RuntimeError("environment validation failed; fix dependencies before running the pipeline")
        inventory = self.inventory()
        self.repair()
        if not skip_non_video and self.cfg.processing.keep_non_video_files:
            self.sync_non_video(dry_run=dry_run)
        videos = [item for item in inventory_remote_files(inventory) if item.is_video]
        if smoke_test and limit_videos is None:
            limit_videos = 1
        if limit_videos is not None:
            videos = videos[:limit_videos]
        coordinator = VideoPipelineCoordinator(self.cfg, self.pipeline_store, self.raw_store, self.processed_store, self.resize_store, self.logger)
        coordinator.run(videos, dry_run=dry_run)
        self.write_final_verify_report(inventory)

    def status(self) -> dict[str, Any]:
        inventory = load_inventory_if_exists(self.cfg)
        non_video_raw = assess_non_video_files(self.cfg, inventory) if inventory else {"inventory_available": False}
        non_video = dict(non_video_raw)
        if "missing" in non_video:
            missing = list(non_video.get("missing", []))
            non_video["missing_count"] = len(missing)
            non_video["missing_examples"] = missing[:10]
            non_video.pop("missing", None)
        if "corrupt" in non_video:
            corrupt = list(non_video.get("corrupt", []))
            non_video["corrupt_count"] = len(corrupt)
            non_video["corrupt_examples"] = corrupt[:10]
            non_video.pop("corrupt", None)
        queue_summary = VideoRedownloadQueue(self.cfg.video_redownload_queue_path).summary()
        result = {
            "non_video_local": non_video,
            "video_pipeline": self.pipeline_store.data.get("summary", {}),
            "video_download": self.raw_store.data.get("summary", {}),
            "video_processing": self.processed_store.data.get("summary", {}),
            "video_resize": self.resize_store.data.get("summary", {}),
            "video_exception_queue": {
                "path": queue_summary["path"],
                "by_status": queue_summary["by_status"],
                "open_examples": [item for item in queue_summary["requests"] if item.get("status") == "open"][:5],
            },
            "current_remote_path": self.pipeline_store.data.get("current_remote_path"),
        }
        self.logger.info(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    def verify(self, strict_non_video_sha256: bool = False) -> dict[str, Any]:
        inventory = self.inventory()
        report = self.write_final_verify_report(inventory)
        if strict_non_video_sha256:
            report["strict_non_video_sha256"] = self.write_strict_non_video_sha256_report(inventory)
            report["passed"] = bool(report["passed"] and report["strict_non_video_sha256"]["passed"])
        return report

    def clean_raw_cache(self, dry_run: bool = False) -> dict[str, Any]:
        inventory = self.inventory()
        result = {"deleted": [], "skipped": [], "errors": []}
        videos = [RemoteFile(**item) for item in inventory.get("video_files", [])]
        reporter = ProgressReporter(
            self.logger,
            "clean-raw-cache",
            self.cfg.logging.command_progress_every_items,
            self.cfg.logging.command_progress_interval_seconds,
        )
        reporter.start("checking raw videos eligible for safe deletion", total=len(videos), dry_run=dry_run)
        for index, remote in enumerate(videos, start=1):
            reporter.maybe(index, "checking raw video", current_path=remote.path, total=len(videos))
            final = self.cfg.processed_video_dir(remote.path)
            raw = self.cfg.local_raw_path(remote.path)
            if not raw.exists():
                result["skipped"].append({"remote_path": remote.path, "reason": "raw_absent"})
                continue
            verify = verify_processed_video_dir(
                final,
                expected_frame_count=None,
                require_verify_result=True,
                sample_images=False,
                expected_frame_width=expected_frame_width_for_video(self.cfg, final),
            )
            if not verify.passed:
                result["skipped"].append({"remote_path": remote.path, "reason": "processed_not_verified", "errors": verify.errors})
                continue
            if dry_run:
                result["deleted"].append({"remote_path": remote.path, "dry_run": True})
                continue
            self.processor.delete_raw_video(remote, final)
            result["deleted"].append({"remote_path": remote.path})
        reporter.done(len(videos), "raw cache cleanup finished", deleted=len(result["deleted"]), skipped=len(result["skipped"]))
        atomic_write_json(self.cfg.paths.code_dir / "clean_raw_cache_report.json", result)
        return result

    def dry_run(self) -> dict[str, Any]:
        inventory = load_inventory_if_exists(self.cfg)
        if inventory is None:
            result = {
                "repo_id": self.cfg.repo_id,
                "inventory_available": False,
                "message": "No remote_inventory.json found. Run `python run_pipeline.py scan` after installing dependencies.",
                "would_download": False,
                "would_delete": False,
            }
            self.logger.info(json.dumps(result, ensure_ascii=False, indent=2))
            return result
        videos = inventory.get("video_files", [])
        non_videos = inventory.get("non_video_files", [])
        result = {
            "repo_id": self.cfg.repo_id,
            "would_sync_non_video_files": len(non_videos),
            "would_process_video_files": len(videos),
            "first_video": videos[0]["path"] if videos else None,
            "raw_cache_dir": str(self.cfg.paths.raw_cache_dir),
            "processed_data_dir": str(self.cfg.paths.processed_data_dir),
        }
        self.logger.info(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    def write_final_verify_report(self, inventory: dict[str, Any]) -> dict[str, Any]:
        report = {
            "repo_id": self.cfg.repo_id,
            "missing_non_video": [],
            "corrupt_non_video": [],
            "video_errors": [],
            "raw_videos_remaining": [],
            "passed": False,
        }
        non_videos = inventory.get("non_video_files", [])
        videos = inventory.get("video_files", [])
        reporter = ProgressReporter(
            self.logger,
            "verify",
            self.cfg.logging.command_progress_every_items,
            self.cfg.logging.command_progress_interval_seconds,
        )
        reporter.start("verifying processed_data against remote inventory", non_videos=len(non_videos), videos=len(videos))
        non_video_report = assess_non_video_files(self.cfg, inventory)
        report["missing_non_video"] = non_video_report["missing"]
        report["corrupt_non_video"] = non_video_report["corrupt"]
        for index, item in enumerate(videos, start=1):
            reporter.maybe(index, "checking video output", current_path=item["path"], total=len(videos))
            remote = RemoteFile(**item)
            final = self.cfg.processed_video_dir(remote.path)
            verify = verify_processed_video_dir(
                final,
                expected_frame_count=None,
                require_verify_result=True,
                sample_images=False,
                expected_frame_width=expected_frame_width_for_video(self.cfg, final),
            )
            if not verify.passed:
                report["video_errors"].append({"remote_path": remote.path, "errors": verify.errors})
            raw = self.cfg.local_raw_path(remote.path)
            if raw.exists():
                report["raw_videos_remaining"].append(remote.path)
        report["passed"] = not report["missing_non_video"] and not report["corrupt_non_video"] and not report["video_errors"] and not report["raw_videos_remaining"]
        self.cfg.manifests_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.cfg.manifests_dir / "final_verify_report.json", report)
        md = [
            "# CASTLE Pipeline Final Verify Report",
            "",
            f"- repo_id: {self.cfg.repo_id}",
            f"- passed: {report['passed']}",
            f"- missing_non_video: {len(report['missing_non_video'])}",
            f"- corrupt_non_video: {len(report['corrupt_non_video'])}",
            f"- video_errors: {len(report['video_errors'])}",
            f"- raw_videos_remaining: {len(report['raw_videos_remaining'])}",
        ]
        atomic_write_text(self.cfg.manifests_dir / "final_verify_report.md", "\n".join(md) + "\n")
        reporter.done(
            len(non_videos) + len(videos),
            "verification report written",
            missing_non_video=len(report["missing_non_video"]),
            corrupt_non_video=len(report["corrupt_non_video"]),
            video_errors=len(report["video_errors"]),
            raw_videos_remaining=len(report["raw_videos_remaining"]),
        )
        return report

    def write_strict_non_video_sha256_report(self, inventory: dict[str, Any]) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        sidecars = {
            item["path"][:-7]: item["path"]
            for item in inventory.get("non_video_files", [])
            if str(item.get("path", "")).lower().endswith(".sha256")
        }
        reporter = ProgressReporter(
            self.logger,
            "strict-non-video-sha256",
            self.cfg.logging.command_progress_every_items,
            self.cfg.logging.command_progress_interval_seconds,
        )
        non_videos = inventory.get("non_video_files", [])
        reporter.start("strict sha256 verification for non-video files", total=len(non_videos))
        for index, item in enumerate(non_videos, start=1):
            remote_path = item["path"]
            reporter.maybe(index, "hashing non-video file", current_path=remote_path, total=len(non_videos))
            local = self.cfg.processed_file_path(remote_path)
            row = {
                "remote_path": remote_path,
                "processed_path": str(local),
                "size": item.get("size"),
                "exists": local.exists(),
                "sha256": "",
                "sidecar_sha256": "",
                "status": "missing",
            }
            if not local.exists():
                errors.append({"remote_path": remote_path, "reason": "missing"})
                rows.append(row)
                continue
            if item.get("size") is not None and local.stat().st_size != item.get("size"):
                row["status"] = "size_mismatch"
                errors.append({"remote_path": remote_path, "reason": "size_mismatch"})
                rows.append(row)
                continue
            actual_sha = sha256_file(local)
            row["sha256"] = actual_sha
            sidecar_path = sidecars.get(remote_path)
            if sidecar_path:
                expected = read_sha256_file(self.cfg.processed_file_path(sidecar_path))
                row["sidecar_sha256"] = expected or ""
                if expected and actual_sha.lower() != expected.lower():
                    row["status"] = "sha256_mismatch"
                    errors.append({"remote_path": remote_path, "reason": "sha256_mismatch"})
                    rows.append(row)
                    continue
            row["status"] = "verified"
            rows.append(row)
        self.cfg.manifests_dir.mkdir(parents=True, exist_ok=True)
        manifest_csv = self.cfg.manifests_dir / "non_video_sha256_manifest.csv"
        manifest_jsonl = self.cfg.manifests_dir / "non_video_sha256_manifest.jsonl"
        if rows:
            fields = sorted({key for row in rows for key in row.keys()})
            tmp = manifest_csv.with_name(manifest_csv.name + ".tmp")
            with tmp.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            tmp.replace(manifest_csv)
            atomic_write_text(manifest_jsonl, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
        report = {
            "passed": not errors,
            "checked": len(rows),
            "errors": errors,
            "manifest_csv": str(manifest_csv),
            "manifest_jsonl": str(manifest_jsonl),
        }
        atomic_write_json(self.cfg.manifests_dir / "strict_non_video_sha256_report.json", report)
        reporter.done(len(non_videos), "strict non-video sha256 verification finished", errors=len(errors))
        return report
