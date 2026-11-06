from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any


def _as_path(value: str | os.PathLike[str]) -> Path:
    return Path(str(value)).expanduser()


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _without_legacy_keys(values: dict[str, Any], *keys: str) -> dict[str, Any]:
    result = dict(values)
    for key in keys:
        result.pop(key, None)
    return result


def _default_download_environment() -> dict[str, str]:
    return {
        "HF_XET_HIGH_PERFORMANCE": "0",
        "HF_XET_CLIENT_ENABLE_ADAPTIVE_CONCURRENCY": "1",
        "HF_XET_CLIENT_AC_MIN_DOWNLOAD_CONCURRENCY": "4",
        "HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY": "16",
        "HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY": "64",
        "HTTP_PROXY": "http://127.0.0.1:7897",
        "HTTPS_PROXY": "http://127.0.0.1:7897",
        "ALL_PROXY": "http://127.0.0.1:7897",
        "NO_PROXY": "localhost,127.0.0.1,::1",
    }


def default_config_dict() -> dict[str, Any]:
    root = "E:/CASTLE_MM26_Challenge"
    pipeline = f"{root}/download_pipeline"
    return {
        "project": {
            "repo_id": "CASTLE-Dataset/CASTLE2024",
            "repo_type": "dataset",
        },
        "paths": {
            "download_pipeline_root": pipeline,
            "code_dir": f"{pipeline}/code",
            "raw_cache_dir": f"{pipeline}/raw_cache",
            "processed_data_dir": f"{pipeline}/processed_data",
            "hf_home": f"{pipeline}/raw_cache/hf_home",
            "hf_cache": f"{pipeline}/raw_cache/hf_cache",
        },
        "tools": {
            "ffmpeg": "ffmpeg",
            "ffprobe": "ffprobe",
        },
        "huggingface": {
            "token_env": "HF_TOKEN",
            "prefer_xet": True,
            "fallback_to_http_on_stall": True,
            "max_attempts_per_file": 5,
            "heartbeat_interval_seconds": 10,
            "worker_poll_interval_seconds": 0.25,
            "stall_timeout_seconds": 1800,
            "download_stall_timeout_seconds": 600,
            "worker_download_timeout_seconds": 300,
            "worker_etag_timeout_seconds": 30,
            "download_environment": _default_download_environment(),
        },
        "processing": {
            "video_extensions": [".mp4"],
            "fps": 1,
            "frame_width": 3840,
            "jpg_q": 2,
            "resize_extracted_frames": True,
            "resized_frame_width": 1920,
            "frame_resize_workers": 6,
            "delete_raw_video_after_verified": True,
            "keep_non_video_files": True,
            "non_video_small_file_threshold_bytes": 1024 * 1024,
            "non_video_small_file_progress_batch_size": 100,
        },
        "safety": {
            "min_free_disk_gb": 30,
            "backup_progress_before_repair": True,
            "quarantine_size_mismatched_downloads": True,
        },
        "logging": {
            "level": "INFO",
            "scan_progress_every_files": 100,
            "scan_progress_interval_seconds": 15,
            "scan_checkpoint_every_files": 100,
            "scan_max_attempts_per_segment": 5,
            "scan_retry_backoff_seconds": [10, 30, 60, 120],
            "command_progress_every_items": 25,
            "command_progress_interval_seconds": 30,
            "heartbeat_dir_size_refresh_interval_seconds": 300,
        },
    }


@dataclass(frozen=True)
class ProjectConfig:
    repo_id: str
    repo_type: str


@dataclass(frozen=True)
class PathsConfig:
    download_pipeline_root: Path
    code_dir: Path
    raw_cache_dir: Path
    processed_data_dir: Path
    hf_home: Path
    hf_cache: Path


@dataclass(frozen=True)
class ToolsConfig:
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"


@dataclass(frozen=True)
class HuggingFaceConfig:
    token_env: str = "HF_TOKEN"
    prefer_xet: bool = True
    fallback_to_http_on_stall: bool = True
    max_attempts_per_file: int = 5
    heartbeat_interval_seconds: int = 10
    worker_poll_interval_seconds: float = 0.25
    stall_timeout_seconds: int = 1800
    download_stall_timeout_seconds: int = 600
    worker_download_timeout_seconds: int = 300
    worker_etag_timeout_seconds: int = 30
    download_environment: dict[str, str] = field(default_factory=_default_download_environment)


@dataclass(frozen=True)
class ProcessingConfig:
    video_extensions: list[str] = field(default_factory=lambda: [".mp4"])
    fps: int = 1
    frame_width: int = 3840
    jpg_q: int = 2
    resize_extracted_frames: bool = True
    resized_frame_width: int = 1920
    frame_resize_workers: int = 6
    delete_raw_video_after_verified: bool = True
    keep_non_video_files: bool = True
    non_video_small_file_threshold_bytes: int = 1024 * 1024
    non_video_small_file_progress_batch_size: int = 100


@dataclass(frozen=True)
class SafetyConfig:
    min_free_disk_gb: int = 30
    backup_progress_before_repair: bool = True
    quarantine_size_mismatched_downloads: bool = True


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    scan_progress_every_files: int = 100
    scan_progress_interval_seconds: int = 15
    scan_checkpoint_every_files: int = 100
    scan_max_attempts_per_segment: int = 5
    scan_retry_backoff_seconds: list[int] = field(default_factory=lambda: [10, 30, 60, 120])
    command_progress_every_items: int = 25
    command_progress_interval_seconds: int = 30
    heartbeat_dir_size_refresh_interval_seconds: int = 300


@dataclass(frozen=True)
class PipelineConfig:
    project: ProjectConfig
    paths: PathsConfig
    tools: ToolsConfig
    huggingface: HuggingFaceConfig
    processing: ProcessingConfig
    safety: SafetyConfig
    logging: LoggingConfig

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str]) -> "PipelineConfig":
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required. Install with: pip install -r requirements.txt") from exc
        config_path = Path(path)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        return cls.from_dict(raw or {})

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PipelineConfig":
        data = _deep_update(default_config_dict(), raw)
        paths = data["paths"]
        return cls(
            project=ProjectConfig(**_without_legacy_keys(data["project"], "name")),
            paths=PathsConfig(
                download_pipeline_root=_as_path(paths["download_pipeline_root"]),
                code_dir=_as_path(paths["code_dir"]),
                raw_cache_dir=_as_path(paths["raw_cache_dir"]),
                processed_data_dir=_as_path(paths["processed_data_dir"]),
                hf_home=_as_path(paths["hf_home"]),
                hf_cache=_as_path(paths["hf_cache"]),
            ),
            tools=ToolsConfig(**data.get("tools", {})),
            huggingface=HuggingFaceConfig(**data["huggingface"]),
            processing=ProcessingConfig(**_without_legacy_keys(data["processing"], "shard_unit", "output_frame_pattern")),
            safety=SafetyConfig(
                **_without_legacy_keys(
                    data["safety"],
                    "never_delete_raw_unless_processed_verified",
                    "use_atomic_writes",
                    "require_manifest_before_raw_delete",
                    "require_verify_result_before_raw_delete",
                )
            ),
            logging=LoggingConfig(
                **_without_legacy_keys(
                    data["logging"],
                    "heartbeat_jsonl",
                    "terminal_progress",
                    "per_attempt_download_logs",
                    "ffmpeg_logs",
                )
            ),
        )

    def ensure_directories(self) -> None:
        for path in [
            self.paths.code_dir,
            self.paths.raw_cache_dir,
            self.paths.processed_data_dir,
            self.paths.hf_home,
            self.paths.hf_cache,
            self.logs_dir,
            self.download_logs_dir,
            self.ffmpeg_logs_dir,
            self.error_logs_dir,
            self.manifests_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def apply_hf_environment(self, force_http: bool = False) -> None:
        os.environ.update(self._hf_environment_overrides(force_http=force_http))

    def download_subprocess_environment(self, force_http: bool = False) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(self._hf_environment_overrides(force_http=force_http))
        if not force_http:
            environment.pop("HF_HUB_DISABLE_XET", None)
        return environment

    def _hf_environment_overrides(self, force_http: bool = False) -> dict[str, str]:
        environment = {
            "HF_HOME": str(self.paths.hf_home),
            "HF_HUB_CACHE": str(self.paths.hf_cache),
            "HF_HUB_DOWNLOAD_TIMEOUT": str(self.huggingface.worker_download_timeout_seconds),
            "HF_HUB_ETAG_TIMEOUT": str(self.huggingface.worker_etag_timeout_seconds),
        }
        environment.update({name: str(value) for name, value in self.huggingface.download_environment.items()})
        if force_http:
            environment["HF_HUB_DISABLE_XET"] = "1"
        return environment

    @property
    def repo_id(self) -> str:
        return self.project.repo_id

    @property
    def repo_type(self) -> str:
        return self.project.repo_type

    @property
    def logs_dir(self) -> Path:
        return self.paths.code_dir / "logs"

    @property
    def download_logs_dir(self) -> Path:
        return self.logs_dir / "downloads"

    @property
    def ffmpeg_logs_dir(self) -> Path:
        return self.logs_dir / "ffmpeg"

    @property
    def error_logs_dir(self) -> Path:
        return self.logs_dir / "errors"

    @property
    def heartbeat_log(self) -> Path:
        return self.logs_dir / "heartbeat.jsonl"

    @property
    def manifests_dir(self) -> Path:
        return self.paths.processed_data_dir / "_manifests"

    @property
    def pipeline_progress_path(self) -> Path:
        return self.paths.code_dir / "pipeline_progress.json"

    @property
    def pipeline_events_path(self) -> Path:
        return self.paths.code_dir / "pipeline_events.jsonl"

    @property
    def raw_progress_path(self) -> Path:
        return self.paths.raw_cache_dir / "raw_cache_progress.json"

    @property
    def raw_events_path(self) -> Path:
        return self.paths.raw_cache_dir / "raw_cache_events.jsonl"

    @property
    def processed_progress_path(self) -> Path:
        return self.paths.processed_data_dir / "processed_progress.json"

    @property
    def processed_events_path(self) -> Path:
        return self.paths.processed_data_dir / "processed_events.jsonl"

    @property
    def video_pipeline_progress_path(self) -> Path:
        return self.paths.code_dir / "video_pipeline_progress.json"

    @property
    def video_pipeline_events_path(self) -> Path:
        return self.paths.code_dir / "video_pipeline_events.jsonl"

    @property
    def video_download_progress_path(self) -> Path:
        return self.paths.raw_cache_dir / "video_download_progress.json"

    @property
    def video_download_events_path(self) -> Path:
        return self.paths.raw_cache_dir / "video_download_events.jsonl"

    @property
    def video_processing_progress_path(self) -> Path:
        return self.paths.processed_data_dir / "video_processing_progress.json"

    @property
    def video_processing_events_path(self) -> Path:
        return self.paths.processed_data_dir / "video_processing_events.jsonl"

    @property
    def video_resize_progress_path(self) -> Path:
        return self.paths.processed_data_dir / "video_resize_progress.json"

    @property
    def video_resize_events_path(self) -> Path:
        return self.paths.processed_data_dir / "video_resize_events.jsonl"

    @property
    def video_redownload_queue_path(self) -> Path:
        return self.paths.code_dir / "video_redownload_queue.sqlite"

    def local_raw_path(self, remote_path: str) -> Path:
        return self.paths.raw_cache_dir.joinpath(*PurePosixPath(remote_path).parts)

    def processed_file_path(self, remote_path: str) -> Path:
        return self.paths.processed_data_dir.joinpath(*PurePosixPath(remote_path).parts)

    def processed_video_dir(self, remote_video_path: str) -> Path:
        posix = PurePosixPath(remote_video_path)
        return self.paths.processed_data_dir.joinpath(*posix.with_suffix("").parts)

    def relative_to_processed(self, path: Path) -> str:
        return path.relative_to(self.paths.processed_data_dir).as_posix()
