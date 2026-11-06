from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

from .disk_utils import free_disk_bytes, human_bytes


@dataclass
class ValidationReport:
    passed: bool
    checks: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_environment(cfg: Any) -> ValidationReport:
    cfg.ensure_directories()
    cfg.apply_hf_environment()
    report = ValidationReport(passed=False)
    report.checks["python_version"] = sys.version
    if sys.version_info < (3, 10):
        report.errors.append("Python 3.10+ is required")
    required_modules = ["huggingface_hub", "yaml", "tqdm", "PIL"]
    if cfg.huggingface.prefer_xet:
        required_modules.append("hf_xet")
    for module in required_modules:
        report.checks[f"module_{module}"] = importlib.util.find_spec(module) is not None
        if not report.checks[f"module_{module}"]:
            report.errors.append(f"missing Python dependency: {module}")
    for tool_name, tool in [("ffmpeg", cfg.tools.ffmpeg), ("ffprobe", cfg.tools.ffprobe)]:
        path = shutil.which(tool)
        report.checks[tool_name] = path
        if not path:
            report.errors.append(f"{tool_name} not found on PATH: {tool}")
        else:
            try:
                result = subprocess.run([tool, "-version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=20)
                report.checks[f"{tool_name}_version_head"] = result.stdout.splitlines()[0] if result.stdout else ""
            except Exception as exc:
                report.errors.append(f"{tool_name} version check failed: {exc}")
    token = os.getenv(cfg.huggingface.token_env)
    report.checks["hf_token_env"] = cfg.huggingface.token_env
    report.checks["hf_token_present"] = bool(token)
    if not token:
        report.warnings.append(f"{cfg.huggingface.token_env} is not set; public files may work but gated access will fail")
    for path_name in ["code_dir", "raw_cache_dir", "processed_data_dir", "hf_home", "hf_cache"]:
        path = getattr(cfg.paths, path_name)
        report.checks[f"{path_name}_exists"] = path.exists()
        try:
            probe = path / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            report.checks[f"{path_name}_writable"] = True
        except Exception as exc:
            report.errors.append(f"{path_name} is not writable: {exc}")
    free = free_disk_bytes(cfg.paths.download_pipeline_root)
    report.checks["free_disk_bytes"] = free
    report.checks["free_disk_human"] = human_bytes(free)
    if free < cfg.safety.min_free_disk_gb * 1024**3:
        report.warnings.append(f"free disk is below configured threshold: {human_bytes(free)}")
    report.passed = not report.errors
    return report
