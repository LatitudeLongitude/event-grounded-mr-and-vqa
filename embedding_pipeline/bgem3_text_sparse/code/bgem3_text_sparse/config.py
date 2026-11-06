from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .paths import RuntimePaths


DEFAULT_LOCAL_MODEL_ROOT = Path("E:/models/bge-m3")
DEFAULT_SERVER_MODEL_ROOT = Path("/mnt/data1/models/BAAI/bge-m3")


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = "BAAI/bge-m3"
    output_type: str = "learned_sparse"
    use_dense: bool = False
    use_sparse: bool = True
    use_colbert: bool = False
    use_late_interaction: bool = False
    local_files_only: bool = True
    sparse_value_dtype: str = "float16"
    sparse_index_dtype: str = "int32"
    sparse_encoding: str = "little_endian_int32_indices_float16_values"
    max_length: int = 8192
    use_fp16: bool = True


@dataclass(frozen=True)
class RuntimeConfig:
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    batch_size: int = 8
    max_retries: int = 1
    max_gpu_utilization: int = 10
    min_free_memory_mib: int = 30000
    status_interval_seconds: int = 30
    minimum_output_free_gib: int = 10
    records_per_shard: int = 10000


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
        if self.model.model_id != "BAAI/bge-m3":
            raise ValueError("model.model_id must be BAAI/bge-m3")
        if self.model.output_type != "learned_sparse":
            raise ValueError("model.output_type must be learned_sparse")
        if self.model.use_dense:
            raise ValueError("BGE-M3 dense output is forbidden")
        if not self.model.use_sparse:
            raise ValueError("BGE-M3 learned sparse output must be enabled")
        if self.model.use_colbert or self.model.use_late_interaction:
            raise ValueError("BGE-M3 ColBERT / late interaction output is forbidden")
        if self.model.sparse_value_dtype != "float16":
            raise ValueError("model.sparse_value_dtype must be float16")
        if self.model.sparse_index_dtype != "int32":
            raise ValueError("model.sparse_index_dtype must be int32")
        if self.model.sparse_encoding != "little_endian_int32_indices_float16_values":
            raise ValueError("Unsupported sparse encoding")
        if self.model.max_length <= 0:
            raise ValueError("model.max_length must be positive")
        if self.runtime.batch_size <= 0:
            raise ValueError("runtime.batch_size must be positive")
        if self.runtime.records_per_shard <= 0:
            raise ValueError("runtime.records_per_shard must be positive")
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
        if len(self.runtime.gpu_ids) > 3:
            raise ValueError("BGE-M3 sparse pipeline supports at most three selected GPUs")
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
        env_value = os.environ.get("BGEM3_MODEL_ROOT")
        if env_value:
            return Path(env_value).expanduser().resolve()
        return (DEFAULT_LOCAL_MODEL_ROOT if os.name == "nt" else DEFAULT_SERVER_MODEL_ROOT).resolve()
    expanded = _expand_env_defaults(text)
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()
