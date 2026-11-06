from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


class PathSafetyError(ValueError):
    pass


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _reject_symlink_components(path: Path, stop: Path) -> None:
    current = path
    while True:
        if current.exists() and current.is_symlink():
            raise PathSafetyError(f"Output path contains symlink: {current}")
        if current == stop or current.parent == current:
            return
        current = current.parent


@dataclass(frozen=True)
class RuntimePaths:
    project_root: Path
    module_root: Path
    data_root: Path
    chunks_path: Path
    meta_summary_path: Path
    model_root: Path
    output_root: Path

    @classmethod
    def create(
        cls,
        *,
        project_root: Path | str,
        module_root: Path | str,
        data_root: Path | str,
        chunks_path: Path | str,
        meta_summary_path: Path | str,
        model_root: Path | str,
        output_root: Path | str,
    ) -> "RuntimePaths":
        project = Path(project_root).expanduser().resolve()
        module = Path(module_root).expanduser().resolve()
        data = Path(data_root).expanduser().resolve()
        chunks = Path(chunks_path).expanduser().resolve()
        summary = Path(meta_summary_path).expanduser().resolve()
        model = Path(model_root).expanduser().resolve()
        output = Path(output_root).expanduser().resolve()
        if not is_within(module, project):
            raise PathSafetyError("module_root must be inside project_root")
        allowed_output = (module / "embedding_data").resolve()
        if output != allowed_output:
            raise PathSafetyError(f"output_root must equal {allowed_output}")
        if output == project or is_within(project, output):
            raise PathSafetyError("output_root cannot contain project_root")
        for protected in (data, model):
            if output == protected or is_within(output, protected) or is_within(protected, output):
                raise PathSafetyError(f"output_root overlaps protected path: {protected}")
        if not is_within(chunks, data):
            raise PathSafetyError("chunks_path must be inside data_root")
        if not is_within(summary, data):
            raise PathSafetyError("meta_summary_path must be inside data_root")
        _reject_symlink_components(output, module)
        return cls(project, module, data, chunks, summary, model, output)

    def require_input(self, path: Path | str) -> Path:
        resolved = Path(path).resolve()
        if not is_within(resolved, self.data_root):
            raise PathSafetyError(f"Input path is outside data_root: {resolved}")
        return resolved

    def require_model_file(self, path: Path | str) -> Path:
        resolved = Path(path).resolve()
        if not is_within(resolved, self.model_root):
            raise PathSafetyError(f"Model path is outside model_root: {resolved}")
        return resolved

    def require_output(self, path: Path | str) -> Path:
        candidate = Path(path)
        resolved_parent = candidate.parent.resolve()
        resolved = resolved_parent / candidate.name
        if not is_within(resolved, self.output_root):
            raise PathSafetyError(f"Write path is outside output_root: {resolved}")
        _reject_symlink_components(resolved_parent, self.output_root)
        return resolved


def configure_runtime_environment(output_root: Path | str) -> Dict[str, str]:
    root = Path(output_root).resolve()
    runtime = root / "_runtime_cache"
    values = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "USE_TF": "0",
        "USE_FLAX": "0",
        "HF_HOME": str(runtime / "huggingface"),
        "HF_MODULES_CACHE": str(runtime / "huggingface" / "modules"),
        "PIP_CACHE_DIR": str(runtime / "pip"),
        "TMPDIR": str(runtime / "tmp"),
        "TEMP": str(runtime / "tmp"),
        "TMP": str(runtime / "tmp"),
    }
    for key, value in values.items():
        os.environ[key] = value
    for key in ("HF_HOME", "HF_MODULES_CACHE", "PIP_CACHE_DIR", "TMPDIR"):
        Path(values[key]).mkdir(parents=True, exist_ok=True)
    return values
