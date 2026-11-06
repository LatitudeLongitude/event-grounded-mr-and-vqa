from __future__ import annotations

import subprocess
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

from .config import AppConfig
from .encoder import COMPUTE_PRECISION_POLICY
from .fingerprint import model_fingerprint
from .gpu_guard import query_gpus
from .inventory import discover_tasks, filter_tasks, load_frame_manifest
from .pipeline import RUN_ID_PATTERN, RunOptions, is_cuda_oom_error, run_pipeline
from .reporting import write_report


@dataclass(frozen=True)
class BenchmarkOptions:
    run_id: str
    gpu_id: int
    day: str
    source: str
    hour: str
    max_frames: int = 128
    batch_sizes: Tuple[int, ...] = (2, 4, 8, 16, 24, 32)
    throughput_threshold: float = 0.95
    max_peak_memory_fraction: float = 0.90

    @staticmethod
    def default_batch_sizes() -> Tuple[int, ...]:
        return (2, 4, 8, 16, 24, 32)

    def validate(self) -> None:
        if not RUN_ID_PATTERN.fullmatch(self.run_id) or self.run_id in {".", ".."}:
            raise ValueError("benchmark run_id must be a single safe path component")
        if self.gpu_id < 0:
            raise ValueError("benchmark GPU ID cannot be negative")
        if self.max_frames < 128:
            raise ValueError("benchmark requires at least 128 consecutive frames")
        if not self.batch_sizes or any(value <= 0 for value in self.batch_sizes):
            raise ValueError("benchmark batch sizes must be positive")
        if len(self.batch_sizes) != len(set(self.batch_sizes)):
            raise ValueError("benchmark batch sizes must be unique")
        if tuple(sorted(self.batch_sizes)) != self.batch_sizes:
            raise ValueError("benchmark batch sizes must be strictly increasing")
        if not 0 < self.throughput_threshold <= 1:
            raise ValueError("benchmark throughput threshold must be in (0, 1]")
        if not 0 < self.max_peak_memory_fraction <= 1:
            raise ValueError("benchmark memory fraction must be in (0, 1]")


@dataclass(frozen=True)
class BenchmarkResult:
    run_id: str
    status: str
    candidates: Tuple[dict, ...]
    selected_batch_size: Optional[int]
    selection_reason: str
    report_json_path: Path
    report_markdown_path: Path
    exit_code: int

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "candidates": list(self.candidates),
            "selected_batch_size": self.selected_batch_size,
            "selection_reason": self.selection_reason,
            "report_json_path": str(self.report_json_path),
            "report_markdown_path": str(self.report_markdown_path),
            "exit_code": self.exit_code,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_code_commit(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _exception_details(exc: BaseException) -> tuple[str, str, bool]:
    parts = []
    current: Optional[BaseException] = exc
    oom = False
    while current is not None:
        error_type = type(current).__name__
        error_message = str(current)
        parts.append(f"{error_type}: {error_message}")
        oom = oom or is_cuda_oom_error(error_type, error_message)
        current = current.__cause__ or current.__context__
    return type(exc).__name__, " <- ".join(parts), oom


def _max_metric(mapping: object, gpu_id: int) -> Optional[float]:
    if not isinstance(mapping, dict):
        return None
    value = mapping.get(gpu_id, mapping.get(str(gpu_id)))
    return None if value is None else float(value)


def _candidate_from_summary(batch_size: int, summary, gpu_id: int, gpu_total_memory_mib: int) -> dict:
    metrics = dict(summary.metrics or {})
    monitor = metrics.get("gpu_monitor") or {}
    peak_memory_mib = _max_metric(monitor.get("peak_used_memory_mib"), gpu_id)
    peak_utilization = _max_metric(monitor.get("peak_utilization_percent"), gpu_id)
    processing_elapsed = float(metrics.get("processing_elapsed_seconds") or summary.elapsed_seconds)
    processing_ips = float(
        metrics.get("processing_images_per_second")
        or (summary.processed_frames / processing_elapsed if processing_elapsed else 0.0)
    )
    failures = list(summary.failures or ())
    failure = failures[0] if failures else {}
    error_type = str(failure.get("error_type") or "")
    error_message = str(failure.get("error_message") or "")
    status = "passed"
    if summary.failed:
        status = "oom" if is_cuda_oom_error(error_type, error_message) else "failed"
    elif summary.processed_frames != summary.selected_frames or summary.completed != 1:
        status = "failed"
        error_type = "IncompleteBenchmarkCandidate"
        error_message = (
            f"processed {summary.processed_frames} of {summary.selected_frames} frames; "
            f"completed shards={summary.completed}"
        )
    elif peak_memory_mib is None or peak_utilization is None:
        status = "failed"
        error_type = "MissingGPUMetrics"
        error_message = "GPU monitor did not record peak memory and utilization"
    return {
        "batch_size": batch_size,
        "status": status,
        "run_id": summary.run_id,
        "invocation_id": summary.invocation_id,
        "actual_images": int(summary.processed_frames),
        "total_elapsed_seconds": float(summary.elapsed_seconds),
        "processing_elapsed_seconds": processing_elapsed,
        "images_per_second": processing_ips,
        "gpu_peak_memory_mib": peak_memory_mib,
        "gpu_peak_memory_fraction": (
            peak_memory_mib / gpu_total_memory_mib if peak_memory_mib is not None else None
        ),
        "gpu_peak_utilization_percent": peak_utilization,
        "cpu_utilization_percent": float(metrics.get("coordinator_cpu_percent_of_host") or 0.0),
        "timings": {
            "decode_seconds": float(metrics.get("decode_seconds") or 0.0),
            "h2d_seconds": float(metrics.get("h2d_seconds") or 0.0),
            "forward_seconds": float(metrics.get("forward_seconds") or 0.0),
            "postprocess_seconds": float(metrics.get("postprocess_seconds") or 0.0),
            "writer_seconds": float(metrics.get("writer_seconds") or 0.0),
        },
        "error_type": error_type or None,
        "error_message": error_message or None,
    }


def _candidate_from_exception(batch_size: int, run_id: str, exc: BaseException) -> dict:
    error_type, error_message, oom = _exception_details(exc)
    return {
        "batch_size": batch_size,
        "status": "oom" if oom else "failed",
        "run_id": run_id,
        "invocation_id": None,
        "actual_images": 0,
        "total_elapsed_seconds": 0.0,
        "processing_elapsed_seconds": 0.0,
        "images_per_second": 0.0,
        "gpu_peak_memory_mib": None,
        "gpu_peak_memory_fraction": None,
        "gpu_peak_utilization_percent": None,
        "cpu_utilization_percent": 0.0,
        "timings": {
            "decode_seconds": 0.0,
            "h2d_seconds": 0.0,
            "forward_seconds": 0.0,
            "postprocess_seconds": 0.0,
            "writer_seconds": 0.0,
        },
        "error_type": error_type,
        "error_message": error_message,
    }


def select_batch_size(
    candidates: Sequence[dict],
    *,
    throughput_threshold: float,
    max_peak_memory_fraction: float,
) -> tuple[Optional[int], str]:
    eligible = [
        candidate
        for candidate in candidates
        if candidate["status"] == "passed"
        and candidate.get("gpu_peak_memory_fraction") is not None
        and candidate["gpu_peak_memory_fraction"] <= max_peak_memory_fraction
    ]
    if not eligible:
        return None, (
            "No passed candidate remained at or below the configured "
            f"{max_peak_memory_fraction:.0%} peak-memory limit."
        )
    highest_throughput = max(candidate["images_per_second"] for candidate in eligible)
    threshold_value = highest_throughput * throughput_threshold
    near_peak = [candidate for candidate in eligible if candidate["images_per_second"] >= threshold_value]
    selected = max(near_peak, key=lambda candidate: candidate["batch_size"])
    return int(selected["batch_size"]), (
        f"Selected the largest stable batch with throughput at least {throughput_threshold:.0%} "
        f"of the eligible peak ({highest_throughput:.3f} images/s) and peak memory at or below "
        f"{max_peak_memory_fraction:.0%}; selected batch {selected['batch_size']} at "
        f"{selected['images_per_second']:.3f} images/s."
    )


def _render_markdown(report: dict) -> str:
    lines = [
        f"# TIPSv2 Batch Benchmark {report['run_id']}",
        "",
        f"- **status**: `{report['status']}`",
        f"- **selected_batch_size**: `{report['selected_batch_size']}`",
        f"- **code_commit**: `{report['code_commit']}`",
        f"- **compute_precision_policy**: `{report['compute_precision_policy']}`",
        f"- **GPU**: `{report['gpu_id']}`",
        f"- **test_hour**: `{report['test_hour']}`",
        f"- **frames_per_candidate**: `{report['frames_per_candidate']}`",
        f"- **selection_reason**: {report['selection_reason']}",
        "",
        "## Selection Rule",
        "",
        "- Exclude OOM and failed candidates.",
        f"- Exclude candidates above {report['selection_rule']['max_peak_memory_fraction']:.0%} peak memory.",
        "- Find the highest eligible throughput.",
        f"- Select the largest stable batch reaching at least {report['selection_rule']['throughput_threshold']:.0%} of that peak.",
        "",
        "## Candidates",
        "",
        "| batch_size | status | images | total_s | processing_s | images/s | peak_mem_MiB | peak_mem_% | GPU_util_% | CPU_util_% | error |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for candidate in report["candidates"]:
        memory_fraction = candidate["gpu_peak_memory_fraction"]
        lines.append(
            "| {batch_size} | {status} | {actual_images} | {total:.3f} | {processing:.3f} | "
            "{ips:.3f} | {memory} | {memory_percent} | {gpu_util} | {cpu_util:.3f} | {error} |".format(
                batch_size=candidate["batch_size"],
                status=candidate["status"],
                actual_images=candidate["actual_images"],
                total=candidate["total_elapsed_seconds"],
                processing=candidate["processing_elapsed_seconds"],
                ips=candidate["images_per_second"],
                memory="" if candidate["gpu_peak_memory_mib"] is None else f"{candidate['gpu_peak_memory_mib']:.0f}",
                memory_percent="" if memory_fraction is None else f"{memory_fraction:.1%}",
                gpu_util="" if candidate["gpu_peak_utilization_percent"] is None else f"{candidate['gpu_peak_utilization_percent']:.0f}",
                cpu_util=candidate["cpu_utilization_percent"],
                error=(candidate["error_type"] or "") + (f": {candidate['error_message']}" if candidate["error_message"] else ""),
            )
        )
    lines.extend(["", "## Detailed Timings", ""])
    for candidate in report["candidates"]:
        lines.append(f"- Batch `{candidate['batch_size']}`: `{candidate['timings']}`")
    return "\n".join(lines) + "\n"


def execute_benchmark(
    config: AppConfig,
    options: BenchmarkOptions,
    *,
    runner: Callable = run_pipeline,
    gpu_total_memory_mib: Optional[int] = None,
    code_commit: Optional[str] = None,
    model_manifest: Optional[dict] = None,
) -> BenchmarkResult:
    options.validate()
    config.validate()
    report_json = config.paths.require_output(
        config.paths.output_root / "_reports" / "benchmark" / f"{options.run_id}.json"
    )
    report_markdown = config.paths.require_output(
        config.paths.output_root / "_reports" / "benchmark" / f"{options.run_id}.md"
    )
    if report_json.exists() or report_markdown.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark report: {report_json}")

    inventory = discover_tasks(config.paths.data_root, config.summary_path)
    tasks = filter_tasks(
        inventory.tasks,
        day=options.day,
        source=options.source,
        hour=options.hour,
        max_hours=2,
    )
    if len(tasks) != 1:
        raise ValueError("benchmark must select exactly one hour")
    task = tasks[0]
    if task.frame_count < options.max_frames:
        raise ValueError(
            f"benchmark hour has {task.frame_count} frames, fewer than requested {options.max_frames}"
        )
    frames = load_frame_manifest(task, config.paths.data_root)[: options.max_frames]
    for previous, current in zip(frames, frames[1:]):
        if current.frame_index != previous.frame_index + 1:
            raise ValueError("benchmark requires consecutive frame_index values")

    if gpu_total_memory_mib is None:
        matching = [gpu for gpu in query_gpus() if gpu.index == options.gpu_id]
        if len(matching) != 1:
            raise ValueError(f"benchmark GPU {options.gpu_id} was not returned exactly once")
        gpu_total_memory_mib = matching[0].total_mib
    if gpu_total_memory_mib <= 0:
        raise ValueError("benchmark GPU total memory must be positive")
    code_commit = code_commit or _resolve_code_commit(config.paths.project_root)
    model_manifest = model_manifest or model_fingerprint(config.paths.model_root, hash_weights=False)

    candidates = []
    infrastructure_failure = False
    for batch_size in options.batch_sizes:
        candidate_run_id = f"{options.run_id}-bs{batch_size}"
        candidate_config = replace(
            config,
            runtime=replace(config.runtime, batch_size=batch_size, gpu_ids=[options.gpu_id]),
        )
        try:
            summary = runner(
                candidate_config,
                RunOptions(
                    encoder_backend="tipsv2",
                    output_mode="benchmark",
                    run_id=candidate_run_id,
                    gpu_ids=(options.gpu_id,),
                    day=options.day,
                    source=options.source,
                    hour=options.hour,
                    max_hours=1,
                    max_frames_per_hour=options.max_frames,
                    resume=False,
                ),
            )
            candidate = _candidate_from_summary(
                batch_size, summary, options.gpu_id, gpu_total_memory_mib
            )
        except Exception as exc:
            candidate = _candidate_from_exception(batch_size, candidate_run_id, exc)
        candidates.append(candidate)
        if candidate["status"] == "oom":
            break
        if candidate["status"] == "failed":
            infrastructure_failure = True
            break

    selected_batch_size, selection_reason = select_batch_size(
        candidates,
        throughput_threshold=options.throughput_threshold,
        max_peak_memory_fraction=options.max_peak_memory_fraction,
    )
    if infrastructure_failure:
        selected_batch_size = None
        selection_reason = "Benchmark stopped because a non-OOM candidate failed; no formal value was selected."
    status = "completed" if selected_batch_size is not None and not infrastructure_failure else "failed"
    exit_code = 0 if status == "completed" else 1
    report = {
        "run_id": options.run_id,
        "created_at": _utc_now(),
        "status": status,
        "exit_code": exit_code,
        "code_commit": code_commit,
        "compute_precision_policy": COMPUTE_PRECISION_POLICY,
        "model_manifest": model_manifest,
        "gpu_id": options.gpu_id,
        "gpu_total_memory_mib": gpu_total_memory_mib,
        "test_hour": f"{options.day}/{options.source}/{task.hour}",
        "frames_per_candidate": options.max_frames,
        "batch_sizes": list(options.batch_sizes),
        "candidates": candidates,
        "selected_batch_size": selected_batch_size,
        "selection_rule": {
            "exclude_statuses": ["oom", "failed"],
            "throughput_threshold": options.throughput_threshold,
            "max_peak_memory_fraction": options.max_peak_memory_fraction,
            "tie_breaker": "largest stable batch meeting the throughput threshold",
        },
        "selection_reason": selection_reason,
        "formal_run_started": False,
    }
    write_report(
        report_json,
        report_markdown,
        report,
        markdown_content=_render_markdown(report),
    )
    return BenchmarkResult(
        run_id=options.run_id,
        status=status,
        candidates=tuple(candidates),
        selected_batch_size=selected_batch_size,
        selection_reason=selection_reason,
        report_json_path=report_json,
        report_markdown_path=report_markdown,
        exit_code=exit_code,
    )
