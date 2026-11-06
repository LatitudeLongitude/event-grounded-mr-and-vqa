from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .paths import RuntimePaths


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = "google/tipsv2-g14"
    image_size: int = 448
    embedding_dim: int = 1536
    precision: str = "fp32"


@dataclass(frozen=True)
class RuntimeConfig:
    gpu_ids: List[int] = field(default_factory=lambda: [0, 1])
    batch_size: int = 32
    decode_workers: int = 4
    prefetch_batches_per_gpu: int = 2
    max_retries: int = 2
    max_gpu_utilization: int = 10
    min_free_memory_mib: int = 30000
    status_interval_seconds: int = 30
    max_memory_fraction_per_gpu: Optional[float] = 0.8
    writer_command_queue_size: int = 32
    writer_result_queue_size: int = 8
    writer_event_queue_size: int = 64
    writer_max_pending_hours: int = 2
    writer_max_pending_batches: int = 512
    writer_max_pending_bytes: int = 512 * 1024 * 1024


@dataclass(frozen=True)
class ParquetConfig:
    compression: str = "zstd"
    compression_level: int = 3
    row_group_size: int = 1024


@dataclass(frozen=True)
class AppConfig:
    paths: RuntimePaths
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    parquet: ParquetConfig = field(default_factory=ParquetConfig)
    summary_path: Optional[Path] = None

    def validate(self) -> None:
        if self.model.image_size <= 0:
            raise ValueError("model.image_size must be positive")
        if self.model.embedding_dim <= 0:
            raise ValueError("model.embedding_dim must be positive")
        if self.model.precision != "fp32":
            raise ValueError(
                "model.precision must be fp32; FP16/BF16/autocast mixed precision is forbidden"
            )
        if self.runtime.batch_size <= 0:
            raise ValueError("runtime.batch_size must be positive")
        if self.runtime.decode_workers not in {2, 4, 6}:
            raise ValueError("runtime.decode_workers must be one of 2, 4, or 6")
        if self.runtime.prefetch_batches_per_gpu <= 0:
            raise ValueError("runtime.prefetch_batches_per_gpu must be positive")
        if self.runtime.max_retries < 0:
            raise ValueError("runtime.max_retries cannot be negative")
        if not 0 <= self.runtime.max_gpu_utilization <= 100:
            raise ValueError("runtime.max_gpu_utilization must be between 0 and 100")
        if self.runtime.min_free_memory_mib < 0:
            raise ValueError("runtime.min_free_memory_mib cannot be negative")
        if self.runtime.status_interval_seconds <= 0:
            raise ValueError("runtime.status_interval_seconds must be positive")
        for name in (
            "writer_command_queue_size",
            "writer_result_queue_size",
            "writer_event_queue_size",
            "writer_max_pending_hours",
            "writer_max_pending_batches",
            "writer_max_pending_bytes",
        ):
            if getattr(self.runtime, name) <= 0:
                raise ValueError(f"runtime.{name} must be positive")
        if any(gpu_id < 0 for gpu_id in self.runtime.gpu_ids):
            raise ValueError("runtime.gpu_ids cannot contain negative values")
        if len(self.runtime.gpu_ids) != len(set(self.runtime.gpu_ids)):
            raise ValueError("runtime.gpu_ids must be unique")
        if self.parquet.row_group_size <= 0:
            raise ValueError("parquet.row_group_size must be positive")

    @classmethod
    def load(cls, path: Path | str) -> "AppConfig":
        config_path = Path(path).expanduser().resolve()
        with config_path.open(encoding="utf-8") as handle:
            raw: Dict[str, Any] = yaml.safe_load(handle) or {}
        path_values = dict(raw.get("paths") or {})
        required = ("project_root", "data_root", "model_root", "output_root")
        missing = [name for name in required if not path_values.get(name)]
        if missing:
            raise ValueError(f"Missing required paths: {', '.join(missing)}")
        project_root = Path(path_values["project_root"])
        module_root = Path(
            path_values.get(
                "module_root",
                project_root / "embedding_pipeline" / "tipsv2_image_cls",
            )
        )
        paths = RuntimePaths.create(
            project_root=project_root,
            module_root=module_root,
            data_root=Path(path_values["data_root"]),
            model_root=Path(path_values["model_root"]),
            output_root=Path(path_values["output_root"]),
        )
        summary = path_values.get("summary_path")
        config = cls(
            paths=paths,
            model=ModelConfig(**(raw.get("model") or {})),
            runtime=RuntimeConfig(**(raw.get("runtime") or {})),
            parquet=ParquetConfig(**(raw.get("parquet") or {})),
            summary_path=Path(summary).expanduser().resolve() if summary else paths.data_root / "_manifests" / "video_processing_summary.csv",
        )
        config.validate()
        return config
