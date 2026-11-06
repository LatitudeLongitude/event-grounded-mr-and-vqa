from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

from .config import AppConfig
from .fingerprint import model_fingerprint
from .gpu_guard import query_gpus
from .pipeline import RUN_ID_PATTERN, RunOptions, is_cuda_oom_error, run_pipeline
from .reporting import write_report


@dataclass(frozen=True)
class BenchmarkOptions:
    run_id: str
    gpu_id: int
    day: Optional[str] = None
    view: Optional[str] = None
    max_chunks: int = 256
    batch_sizes: Tuple[int, ...] = (1, 2, 4, 8, 16)
    throughput_threshold: float = 0.95
    max_peak_memory_fraction: float = 0.90

    def validate(self) -> None:
        if not RUN_ID_PATTERN.fullmatch(self.run_id) or self.run_id in {".", ".."}:
            raise ValueError("benchmark run_id must be a single safe path component")
        if self.gpu_id < 0:
            raise ValueError("benchmark GPU ID cannot be negative")
        if self.max_chunks <= 0:
            raise ValueError("benchmark max_chunks must be positive")
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


def _candidate_from_summary(batch_size: int, summary, gpu_id: int, gpu_total_memory_mib: int) -> dict:
    metrics = dict(summary.metrics or {})
    monitor = metrics.get("gpu_monitor") or {}
    peak_memory = (monitor.get("peak_used_memory_mib") or {}).get(gpu_id)
    if peak_memory is None:
        peak_memory = (monitor.get("peak_used_memory_mib") or {}).get(str(gpu_id))
    peak_util = (monitor.get("peak_utilization_percent") or {}).get(gpu_id)
    if peak_util is None:
        peak_util = (monitor.get("peak_utilization_percent") or {}).get(str(gpu_id))
    failures = list(summary.failures or ())
    failure = failures[0] if failures else {}
    status = "passed"
    error_type = failure.get("error_type")
    error_message = failure.get("error_message")
    if summary.failed:
        status = "oom" if is_cuda_oom_error(str(error_type), str(error_message)) else "failed"
    return {
        "batch_size": batch_size,
        "status": status,
        "run_id": summary.run_id,
        "invocation_id": summary.invocation_id,
        "actual_chunks": int(summary.processed_chunks),
        "elapsed_seconds": float(summary.elapsed_seconds),
        "chunks_per_second": float(summary.chunks_per_second),
        "gpu_peak_memory_mib": peak_memory,
        "gpu_peak_memory_fraction": None if peak_memory is None else float(peak_memory) / gpu_total_memory_mib,
        "gpu_peak_utilization_percent": peak_util,
        "truncation_count": metrics.get("truncation_count", 0),
        "error_type": error_type,
        "error_message": error_message,
    }


def _candidate_from_exception(batch_size: int, run_id: str, exc: BaseException) -> dict:
    error_type = type(exc).__name__
    error_message = str(exc)
    return {
        "batch_size": batch_size,
        "status": "oom" if is_cuda_oom_error(error_type, error_message) else "failed",
        "run_id": run_id,
        "invocation_id": None,
        "actual_chunks": 0,
        "elapsed_seconds": 0.0,
        "chunks_per_second": 0.0,
        "gpu_peak_memory_mib": None,
        "gpu_peak_memory_fraction": None,
        "gpu_peak_utilization_percent": None,
        "truncation_count": 0,
        "error_type": error_type,
        "error_message": error_message,
    }


def select_batch_size(candidates: Sequence[dict], *, throughput_threshold: float, max_peak_memory_fraction: float) -> tuple[Optional[int], str]:
    eligible = [
        candidate
        for candidate in candidates
        if candidate["status"] == "passed"
        and candidate.get("gpu_peak_memory_fraction") is not None
        and candidate["gpu_peak_memory_fraction"] <= max_peak_memory_fraction
    ]
    if not eligible:
        return None, f"No passed candidate remained at or below {max_peak_memory_fraction:.0%} peak memory."
    highest = max(candidate["chunks_per_second"] for candidate in eligible)
    near_peak = [candidate for candidate in eligible if candidate["chunks_per_second"] >= highest * throughput_threshold]
    selected = max(near_peak, key=lambda item: item["batch_size"])
    return int(selected["batch_size"]), (
        f"Selected the largest stable batch within {throughput_threshold:.0%} of the eligible peak "
        f"({highest:.3f} chunks/s) and at or below {max_peak_memory_fraction:.0%} peak memory."
    )


def _render_markdown(report: dict) -> str:
    lines = [
        f"# Qwen3 Batch Benchmark {report['run_id']}",
        "",
        f"- status: `{report['status']}`",
        f"- selected_batch_size: `{report['selected_batch_size']}`",
        f"- gpu_id: `{report['gpu_id']}`",
        f"- max_chunks_per_candidate: `{report['max_chunks']}`",
        f"- selection_reason: {report['selection_reason']}",
        f"- formal_run_started: `{report['formal_run_started']}`",
        "",
        "| batch_size | status | chunks | elapsed_s | chunks/s | peak_mem_MiB | peak_mem_% | error |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for candidate in report["candidates"]:
        fraction = candidate["gpu_peak_memory_fraction"]
        lines.append(
            "| {batch_size} | {status} | {actual_chunks} | {elapsed:.3f} | {cps:.3f} | {mem} | {frac} | {error} |".format(
                batch_size=candidate["batch_size"],
                status=candidate["status"],
                actual_chunks=candidate["actual_chunks"],
                elapsed=candidate["elapsed_seconds"],
                cps=candidate["chunks_per_second"],
                mem="" if candidate["gpu_peak_memory_mib"] is None else f"{candidate['gpu_peak_memory_mib']:.0f}",
                frac="" if fraction is None else f"{fraction:.1%}",
                error=(candidate["error_type"] or "") + (f": {candidate['error_message']}" if candidate["error_message"] else ""),
            )
        )
    return "\n".join(lines) + "\n"


def execute_benchmark(
    config: AppConfig,
    options: BenchmarkOptions,
    *,
    runner: Callable = run_pipeline,
    gpu_total_memory_mib: Optional[int] = None,
) -> BenchmarkResult:
    options.validate()
    config.validate()
    report_json = config.paths.require_output(config.paths.output_root / "_reports" / "benchmark" / f"{options.run_id}.json")
    report_markdown = config.paths.require_output(config.paths.output_root / "_reports" / "benchmark" / f"{options.run_id}.md")
    if report_json.exists() or report_markdown.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark report: {report_json}")
    if gpu_total_memory_mib is None:
        matching = [gpu for gpu in query_gpus() if gpu.index == options.gpu_id]
        if len(matching) != 1:
            raise ValueError(f"benchmark GPU {options.gpu_id} was not returned exactly once")
        gpu_total_memory_mib = matching[0].total_mib
    candidates = []
    infrastructure_failure = False
    for batch_size in options.batch_sizes:
        candidate_run_id = f"{options.run_id}-bs{batch_size}"
        candidate_config = replace(config, runtime=replace(config.runtime, batch_size=batch_size, gpu_ids=[options.gpu_id]))
        try:
            summary = runner(
                candidate_config,
                RunOptions(
                    encoder_backend="qwen3",
                    output_mode="benchmark",
                    run_id=candidate_run_id,
                    gpu_ids=(options.gpu_id,),
                    day=options.day,
                    view=options.view,
                    max_chunks=options.max_chunks,
                    resume=False,
                ),
            )
            candidate = _candidate_from_summary(batch_size, summary, options.gpu_id, gpu_total_memory_mib)
        except Exception as exc:
            candidate = _candidate_from_exception(batch_size, candidate_run_id, exc)
        candidates.append(candidate)
        if candidate["status"] == "oom":
            break
        if candidate["status"] == "failed":
            infrastructure_failure = True
            break
    selected, reason = select_batch_size(
        candidates,
        throughput_threshold=options.throughput_threshold,
        max_peak_memory_fraction=options.max_peak_memory_fraction,
    )
    if infrastructure_failure:
        selected = None
        reason = "Benchmark stopped because a non-OOM candidate failed; no formal value was selected."
    status = "completed" if selected is not None and not infrastructure_failure else "failed"
    exit_code = 0 if status == "completed" else 1
    report = {
        "run_id": options.run_id,
        "created_at": _utc_now(),
        "status": status,
        "exit_code": exit_code,
        "model_manifest": model_fingerprint(config.paths.model_root, hash_weights=False),
        "gpu_id": options.gpu_id,
        "gpu_total_memory_mib": gpu_total_memory_mib,
        "day": options.day,
        "view": options.view,
        "max_chunks": options.max_chunks,
        "batch_sizes": list(options.batch_sizes),
        "candidates": candidates,
        "selected_batch_size": selected,
        "selection_reason": reason,
        "selection_rule": {
            "exclude_statuses": ["oom", "failed"],
            "throughput_threshold": options.throughput_threshold,
            "max_peak_memory_fraction": options.max_peak_memory_fraction,
            "tie_breaker": "largest stable batch meeting the throughput threshold",
        },
        "formal_run_started": False,
    }
    write_report(report_json, report_markdown, report, markdown_content=_render_markdown(report))
    return BenchmarkResult(options.run_id, status, tuple(candidates), selected, reason, report_json, report_markdown, exit_code)
