from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .paths import RuntimePaths


DEFAULT_LOCAL_MODEL_ROOT = Path("E:/models/Qwen3-Embedding-8B")
DEFAULT_SERVER_MODEL_ROOT = Path("/mnt/data1/models/Qwen/Qwen3-Embedding-8B")


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = "Qwen/Qwen3-Embedding-8B"
    embedding_dim: int = 4096
    max_length: int = 32768
    inference_dtype: str = "float16"


@dataclass(frozen=True)
class RuntimeConfig:
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    batch_size: int = 4
    max_retries: int = 1
    max_gpu_utilization: int = 10
    min_free_memory_mib: int = 30000
    status_interval_seconds: int = 30
    minimum_output_free_gib: int = 10


@dataclass(frozen=True)
class ParquetConfig:
    compression: str = "zstd"
    compression_level: int = 3
    row_group_size: int = 512


@dataclass(frozen=True)
class AppConfig:
    paths: RuntimePaths
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    parquet: ParquetConfig = field(default_factory=ParquetConfig)

    def validate(self) -> None:
        if self.model.model_id != "Qwen/Qwen3-Embedding-8B":
            raise ValueError("model.model_id must be Qwen/Qwen3-Embedding-8B")
        if self.model.embedding_dim != 4096:
            raise ValueError("model.embedding_dim must be 4096")
        if self.model.max_length <= 0:
            raise ValueError("model.max_length must be positive")
        if self.model.inference_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("model.inference_dtype must be float16, bfloat16, or float32")
        if self.runtime.batch_size <= 0:
            raise ValueError("runtime.batch_size must be positive")
        if self.runtime.max_retries < 0:
            raise ValueError("runtime.max_retries cannot be negative")
        if not 0 <= self.runtime.max_gpu_utilization <= 100:
            raise ValueError("runtime.max_gpu_utilization must be between 0 and 100")
        if self.runtime.min_free_memory_mib < 0:
            raise ValueError("runtime.min_free_memory_mib cannot be negative")
        if self.runtime.status_interval_seconds <= 0:
            raise ValueError("runtime.status_interval_seconds must be positive")
        if self.runtime.minimum_output_free_gib < 0:
            raise ValueError("runtime.minimum_output_free_gib cannot be negative")
        if any(gpu_id < 0 for gpu_id in self.runtime.gpu_ids):
            raise ValueError("runtime.gpu_ids cannot contain negative values")
        if len(self.runtime.gpu_ids) != len(set(self.runtime.gpu_ids)):
            raise ValueError("runtime.gpu_ids must be unique")
        if len(self.runtime.gpu_ids) > 1:
            raise ValueError("Qwen3 dense pipeline supports one selected GPU per process")
        if self.parquet.row_group_size <= 0:
            raise ValueError("parquet.row_group_size must be positive")

    @classmethod
    def load(cls, path: Path | str) -> "AppConfig":
        config_path = Path(path).expanduser().resolve()
        with config_path.open(encoding="utf-8") as handle:
            raw: Dict[str, Any] = yaml.safe_load(handle) or {}
        config_dir = config_path.parent
        path_values = dict(raw.get("paths") or {})
        required = ("project_root", "module_root", "data_root", "chunks_path", "meta_summary_path", "model_root", "output_root")
        missing = [name for name in required if path_values.get(name) in {None, ""}]
        if missing:
            raise ValueError(f"Missing required paths: {', '.join(missing)}")
        paths = RuntimePaths.create(
            project_root=_resolve_config_path(path_values["project_root"], config_dir),
            module_root=_resolve_config_path(path_values["module_root"], config_dir),
            data_root=_resolve_config_path(path_values["data_root"], config_dir),
            chunks_path=_resolve_config_path(path_values["chunks_path"], config_dir),
            meta_summary_path=_resolve_config_path(path_values["meta_summary_path"], config_dir),
            model_root=_resolve_config_path(path_values["model_root"], config_dir, allow_auto_model=True),
            output_root=_resolve_config_path(path_values["output_root"], config_dir),
        )
        config = cls(
            paths=paths,
            model=ModelConfig(**(raw.get("model") or {})),
            runtime=RuntimeConfig(**(raw.get("runtime") or {})),
            parquet=ParquetConfig(**(raw.get("parquet") or {})),
        )
        config.validate()
        return config

def _expand_env_defaults(value: str) -> str:
    pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]+)\}")

    def replace(match: re.Match[str]) -> str:
        return os.environ.get(match.group(1), match.group(2))

    return os.path.expandvars(pattern.sub(replace, value))


def _resolve_config_path(value: Any, config_dir: Path, *, allow_auto_model: bool = False) -> Path:
    text = str(value)
    if allow_auto_model and text == "auto":
        env_value = os.environ.get("QWEN3_MODEL_ROOT")
        if env_value:
            return Path(env_value).expanduser().resolve()
        return (DEFAULT_LOCAL_MODEL_ROOT if os.name == "nt" else DEFAULT_SERVER_MODEL_ROOT).resolve()
    expanded = _expand_env_defaults(text)
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()
