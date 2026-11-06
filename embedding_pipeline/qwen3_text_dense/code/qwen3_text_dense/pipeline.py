from __future__ import annotations

import json
import re
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import AppConfig
from .encoder import Qwen3DenseEncoder, resolved_versions
from .fingerprint import model_fingerprint
from .gpu_guard import GPUMonitor, assert_gpus_idle, gpu_report, query_gpus, validate_gpu_policy
from .inventory import ChunkRecord, ChunkShard, discover_inventory, group_records, load_chunk_records, stable_json
from .parquet_writer import write_embedding_shard
from .paths import PathSafetyError, configure_runtime_environment
from .progress import ProgressStore
from .reporting import append_event, write_report
from .schema import PARQUET_EMBEDDING_VALUE_TYPE, SCHEMA_VERSION
from .validation import validate_shard


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class RunOptions:
    encoder_backend: str = "qwen3"
    output_mode: str = "formal"
    run_id: Optional[str] = None
    gpu_ids: Optional[Tuple[int, ...]] = None
    day: Optional[str] = None
    view: Optional[str] = None
    max_chunks: Optional[int] = None
    resume: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    invocation_id: str
    namespace: str
    output_root: str
    selected_shards: int
    selected_chunks: int
    completed: int
    skipped: int
    failed: int
    processed_chunks: int
    elapsed_seconds: float
    chunks_per_second: float
    failures: Tuple[dict, ...]
    metrics: dict

    def as_dict(self) -> dict:
        return {**self.__dict__, "failures": list(self.failures)}


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


def _config_fingerprint(config: AppConfig, inventory_sha256: str) -> str:
    payload = {
        "model": config.model.__dict__,
        "runtime": config.runtime.__dict__,
        "parquet": config.parquet.__dict__,
        "chunks_sha256": inventory_sha256,
    }
    return __import__("hashlib").sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def validate_run_options(config: AppConfig, options: RunOptions) -> None:
    config.validate()
    if options.encoder_backend != "qwen3":
        raise ValueError("encoder_backend must be qwen3")
    if options.output_mode not in {"formal", "smoke", "benchmark"}:
        raise ValueError("output_mode must be formal, smoke, or benchmark")
    if options.max_chunks is not None and options.max_chunks <= 0:
        raise ValueError("max_chunks must be positive")
    if options.output_mode == "formal":
        if options.max_chunks is not None:
            raise ValueError("formal output must process all chunks; --max-chunks is forbidden")
    gpu_ids = tuple(config.runtime.gpu_ids if options.gpu_ids is None else options.gpu_ids)
    validate_gpu_policy(gpu_ids)


def resolve_run_layout(config: AppConfig, options: RunOptions) -> tuple[str, Path, Path]:
    run_id = options.run_id or time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    if not RUN_ID_PATTERN.fullmatch(run_id) or run_id in {".", ".."}:
        raise PathSafetyError("run_id must be a single safe path component")
    if options.output_mode == "formal":
        root = config.paths.output_root
        namespace = "main"
    elif options.output_mode == "smoke":
        root = config.paths.output_root / "smoke" / run_id
        namespace = f"smoke:{run_id}"
    elif options.output_mode == "benchmark":
        root = config.paths.output_root / "benchmark" / run_id
        namespace = f"benchmark:{run_id}"
    else:
        raise ValueError("Unsupported output_mode")
    config.paths.require_output(root / ".path-safety-check")
    state_path = config.paths.require_output(root / "_state" / "progress.sqlite3")
    return namespace, root, state_path


def _shard_path(run_root: Path, shard: ChunkShard) -> Path:
    path = run_root / "main" / shard.day / f"{shard.view}.parquet"
    path.resolve().relative_to(run_root.resolve())
    return path


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    return float(value)


def _int_value(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _emit_console_event(event_type: str, **payload: object) -> None:
    print(json.dumps({"event_type": event_type, **payload}, sort_keys=True, default=str), flush=True)


def _row_from_record(
    record: ChunkRecord,
    embedding: np.ndarray,
    raw_norm: float,
    *,
    model_id: str,
    model_root: str,
    model_fingerprint_text: str,
    created_at_utc: str,
) -> dict:
    raw = record.raw
    return {
        "schema_version": SCHEMA_VERSION,
        "input_row_number": record.input_row_number,
        "embedding_id": record.embedding_chunk_id,
        "embedding_chunk_id": record.embedding_chunk_id,
        "day": record.day,
        "view": record.view,
        "start_time": _float_or_none(raw.get("start_time")),
        "end_time": _float_or_none(raw.get("end_time")),
        "duration_sec": _float_or_none(raw.get("duration_sec")),
        "start_hour": str(raw.get("start_hour") or ""),
        "end_hour": str(raw.get("end_hour") or ""),
        "hours_json": stable_json(raw.get("hours")),
        "text_for_embedding": record.text_for_embedding,
        "text_for_display": str(raw.get("text_for_display") or ""),
        "char_len": _int_value(raw.get("char_len"), len(record.text_for_embedding)),
        "token_estimate": _int_value(raw.get("token_estimate")),
        "word_like_count": _int_value(raw.get("word_like_count")),
        "source_chunk_ids_json": stable_json(raw.get("source_chunk_ids")),
        "source_chunks_json": stable_json(raw.get("source_chunks")),
        "source_files_json": stable_json(raw.get("source_files")),
        "quality_flags_json": stable_json(raw.get("quality_flags")),
        "contains_invalid_source_chunk": bool(raw.get("contains_invalid_source_chunk")),
        "contains_repetition_compressed_source": bool(raw.get("contains_repetition_compressed_source")),
        "chunking_method": str(raw.get("chunking_method") or ""),
        "chunking_version": str(raw.get("chunking_version") or ""),
        "embedding": embedding.astype(np.float16),
        "embedding_norm_raw": float(raw_norm),
        "model_id": model_id,
        "model_root": model_root,
        "model_fingerprint": model_fingerprint_text,
        "created_at_utc": created_at_utc,
    }


def is_cuda_oom_error(error_type: str, error_message: str) -> bool:
    normalized_type = error_type.lower()
    normalized_message = error_message.lower()
    return (
        "outofmemory" in normalized_type
        or "cuda out of memory" in normalized_message
        or "cuda error: out of memory" in normalized_message
    )


def _render_report_markdown(report: dict) -> str:
    return "\n".join(
        [
            f"# Qwen3 Text Dense Run {report['run_id']}",
            "",
            f"- status: `{report['status']}`",
            f"- namespace: `{report['namespace']}`",
            f"- encoder_backend: `{report['arguments']['encoder_backend']}`",
            f"- output_mode: `{report['arguments']['output_mode']}`",
            f"- selected_chunks: `{report['selected_chunks']}`",
            f"- completed: `{report['completed']}`",
            f"- skipped: `{report['skipped']}`",
            f"- failed: `{report['failed']}`",
            f"- formal_full_run_started: `{report['formal_full_run_started']}`",
            f"- truncation_count: `{report['metrics'].get('truncation_count', 0)}`",
            "",
        ]
    )


def run_pipeline(
    config: AppConfig,
    options: RunOptions,
    *,
    writer_func=write_embedding_shard,
) -> RunSummary:
    started = time.monotonic()
    started_at_utc = _utc_now()
    validate_run_options(config, options)
    configure_runtime_environment(config.paths.output_root)
    run_id = options.run_id or time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    options = RunOptions(**{**asdict(options), "run_id": run_id})
    invocation_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex
    namespace, run_root, db_path = resolve_run_layout(config, options)
    inventory = discover_inventory(config)
    if inventory.invalid or inventory.duplicate_ids:
        raise ValueError("Inventory contains invalid rows or duplicate IDs")
    records = load_chunk_records(config, day=options.day, view=options.view, max_chunks=options.max_chunks)
    grouped = group_records(records)
    if not grouped:
        raise ValueError("No chunks matched the requested filters")
    selected_chunks = sum(shard.row_count for shard, _ in grouped)
    if options.dry_run:
        return RunSummary(run_id, invocation_id, namespace, str(run_root), len(grouped), selected_chunks, 0, 0, 0, 0, 0.0, 0.0, (), {})

    _emit_console_event(
        "qwen3_run_starting",
        run_id=run_id,
        invocation_id=invocation_id,
        namespace=namespace,
        output_mode=options.output_mode,
        encoder_backend=options.encoder_backend,
        selected_shards=len(grouped),
        selected_chunks=selected_chunks,
    )
    code_commit = _resolve_code_commit(config.paths.project_root)
    config_fingerprint = _config_fingerprint(config, inventory.chunks_sha256)
    gpu_ids = tuple(config.runtime.gpu_ids if options.gpu_ids is None else options.gpu_ids)
    gpu_snapshot = ()
    gpu_monitor = None
    _emit_console_event("qwen3_gpu_guard_start", run_id=run_id, gpu_ids=list(gpu_ids))
    gpu_snapshot = query_gpus()
    assert_gpus_idle(
        gpu_snapshot,
        gpu_ids,
        max_utilization=config.runtime.max_gpu_utilization,
        min_free_mib=config.runtime.min_free_memory_mib,
    )
    _emit_console_event("qwen3_gpu_guard_passed", run_id=run_id, gpu_ids=list(gpu_ids))
    _emit_console_event("qwen3_model_manifest_start", run_id=run_id, model_root=str(config.paths.model_root))
    model_manifest = model_fingerprint(config.paths.model_root, hash_weights=False)
    model_fingerprint_text = json.dumps(model_manifest, sort_keys=True)
    gpu_monitor = GPUMonitor(gpu_ids).start()
    _emit_console_event(
        "qwen3_model_load_start",
        run_id=run_id,
        model_root=str(config.paths.model_root),
        device=f"cuda:{gpu_ids[0]}",
        inference_dtype=config.model.inference_dtype,
    )
    encoder = Qwen3DenseEncoder(
        config.paths.model_root,
        device=f"cuda:{gpu_ids[0]}",
        model_id=config.model.model_id,
        max_length=config.model.max_length,
        inference_dtype=config.model.inference_dtype,
        model_fingerprint=model_fingerprint_text,
    ).load()
    _emit_console_event(
        "qwen3_model_load_done",
        run_id=run_id,
        model_id=encoder.model_id,
        max_length=config.model.max_length,
    )

    store = ProgressStore(db_path)
    store.initialize()
    invocation_config = {
        "arguments": asdict(options),
        "config_fingerprint": config_fingerprint,
        "chunks_sha256": inventory.chunks_sha256,
    }
    store.register_invocation(invocation_id, run_id, namespace, invocation_config, code_commit=code_commit)
    report_json_path = config.paths.require_output(config.paths.output_root / "_reports" / run_id / f"{invocation_id}.json")
    report_markdown_path = config.paths.require_output(config.paths.output_root / "_reports" / run_id / f"{invocation_id}.md")
    log_jsonl_path = config.paths.require_output(config.paths.output_root / "_logs" / f"{run_id}.jsonl")
    log_text_path = config.paths.require_output(config.paths.output_root / "_logs" / f"{run_id}.log")

    completed = skipped = failed = processed_chunks = 0
    failures: List[dict] = []
    metrics: Dict[str, object] = {
        "truncation_count": 0,
        "truncated_embedding_chunk_ids": [],
        "dtype_evidence": {},
    }
    last_status = time.monotonic()
    current_shard: tuple[str, str] | None = None

    def finish(status: str, error: BaseException | None = None) -> RunSummary:
        elapsed = time.monotonic() - started
        gpu_metrics = {}
        if gpu_monitor is not None:
            gpu_metrics.update(gpu_monitor.stop())
            if error is None:
                gpu_monitor.raise_if_failed()
        metrics["gpu_monitor"] = gpu_metrics
        report = {
            "run_id": run_id,
            "invocation_id": invocation_id,
            "code_commit": code_commit,
            "started_at_utc": started_at_utc,
            "finished_at_utc": _utc_now(),
            "arguments": asdict(options),
            "status": status,
            "namespace": namespace,
            "selected_shards": len(grouped),
            "selected_chunks": selected_chunks,
            "completed": completed,
            "skipped": skipped,
            "failed": failed,
            "processed_chunks": processed_chunks,
            "elapsed_seconds": elapsed,
            "chunks_per_second": processed_chunks / elapsed if elapsed else 0.0,
            "failures": failures,
            "inventory": inventory.as_dict(),
            "gpu_snapshot": gpu_report(gpu_snapshot),
            "model_manifest": model_manifest,
            "versions": resolved_versions(),
            "parquet_embedding_value_type": PARQUET_EMBEDDING_VALUE_TYPE,
            "metrics": metrics,
            "formal_full_run_started": options.output_mode == "formal",
        }
        if error is not None:
            report["error_type"] = type(error).__name__
            report["error_message"] = str(error)
        write_report(report_json_path, report_markdown_path, report, markdown_content=_render_report_markdown(report))
        store.finish_invocation(
            invocation_id,
            status=status,
            report=report,
            processed=processed_chunks,
            skipped=skipped,
            failed=failed,
            elapsed_seconds=elapsed,
            chunks_per_second=processed_chunks / elapsed if elapsed else 0.0,
            gpu_metrics=gpu_metrics,
            error_type=type(error).__name__ if error else None,
            error_message=str(error) if error else None,
            termination_reason=status,
        )
        append_event(
            log_jsonl_path,
            log_text_path,
            run_id=run_id,
            invocation_id=invocation_id,
            level="INFO" if error is None else "ERROR",
            event_type="invocation_finished",
            payload={"status": status, "completed": completed, "skipped": skipped, "failed": failed},
        )
        return RunSummary(
            run_id,
            invocation_id,
            namespace,
            str(run_root),
            len(grouped),
            selected_chunks,
            completed,
            skipped,
            failed,
            processed_chunks,
            elapsed,
            processed_chunks / elapsed if elapsed else 0.0,
            tuple(failures),
            metrics,
        )

    try:
        append_event(
            log_jsonl_path,
            log_text_path,
            run_id=run_id,
            invocation_id=invocation_id,
            level="INFO",
            event_type="invocation_started",
            payload={"namespace": namespace, "selected_chunks": selected_chunks, "selected_shards": len(grouped)},
        )
        for shard, shard_records in grouped:
            current_shard = (shard.day, shard.view)
            output_path = config.paths.require_output(_shard_path(run_root, shard))
            _emit_console_event(
                "qwen3_shard_start",
                run_id=run_id,
                shard=shard.key,
                rows=len(shard_records),
                output_path=str(output_path),
            )
            store.ensure_shard(namespace, shard.day, shard.view, expected_rows=len(shard_records), output_path=output_path)
            state = store.get_shard(namespace, shard.day, shard.view)
            if state and state["status"] == "completed":
                validation = validate_shard(
                    output_path,
                    expected_rows=int(state["expected_rows"]),
                    expected_file_size=state["file_size"],
                    expected_sha256=state["sha256"],
                    compute_sha256=True,
                )
                if validation.valid and options.resume:
                    skipped += 1
                    processed_chunks += int(state["expected_rows"])
                    store.add_event(run_id, "INFO", "shard_skipped", {"shard": shard.key}, invocation_id)
                    _emit_console_event(
                        "qwen3_shard_skipped",
                        run_id=run_id,
                        shard=shard.key,
                        rows=int(state["expected_rows"]),
                    )
                    continue
                if validation.valid:
                    raise FileExistsError(f"Completed shard already exists: {output_path}")
                raise ValueError(f"Completed shard is invalid: {validation.error}")
            if output_path.exists():
                raise FileExistsError(f"Refusing to overwrite uncommitted existing shard: {output_path}")
            store.start_shard(
                namespace,
                shard.day,
                shard.view,
                run_id=run_id,
                invocation_id=invocation_id,
                attempt=1,
                gpu_ids=",".join(map(str, gpu_ids)),
                batch_size=config.runtime.batch_size,
                config_fingerprint=config_fingerprint,
                model_fingerprint=model_fingerprint_text,
                input_sha256=inventory.chunks_sha256,
            )
            rows = []
            created_at_utc = _utc_now()
            for start in range(0, len(shard_records), config.runtime.batch_size):
                batch_records = shard_records[start : start + config.runtime.batch_size]
                result = encoder.encode_records(batch_records)
                metrics["dtype_evidence"] = result.dtype_evidence or {}
                if result.truncated_ids:
                    metrics["truncation_count"] = int(metrics["truncation_count"]) + len(result.truncated_ids)
                    metrics["truncated_embedding_chunk_ids"] = list(metrics["truncated_embedding_chunk_ids"]) + list(result.truncated_ids[:100])
                for record, embedding, raw_norm in zip(batch_records, result.embeddings, result.raw_norms):
                    rows.append(
                        _row_from_record(
                            record,
                            embedding,
                            raw_norm,
                            model_id=encoder.model_id,
                            model_root=str(config.paths.model_root),
                            model_fingerprint_text=model_fingerprint_text,
                            created_at_utc=created_at_utc,
                        )
                    )
                now = time.monotonic()
                if now - last_status >= config.runtime.status_interval_seconds:
                    print(json.dumps({"event_type": "pipeline_status", "run_id": run_id, "processed_chunks": processed_chunks + len(rows), "current_shard": shard.key}, sort_keys=True), flush=True)
                    last_status = now
            write_result = writer_func(
                output_path,
                rows,
                run_id=run_id,
                metadata={
                    "schema_version": SCHEMA_VERSION,
                    "run_id": run_id,
                    "invocation_id": invocation_id,
                    "encoder_backend": options.encoder_backend,
                    "model_id": encoder.model_id,
                    "model_root": str(config.paths.model_root),
                    "model_fingerprint": model_fingerprint_text,
                    "chunks_sha256": inventory.chunks_sha256,
                    "max_length": config.model.max_length,
                    "inference_dtype": config.model.inference_dtype,
                    "document_instruction": "not-used",
                    "pooling": "last-token",
                    "normalization": "fp32-l2-before-float16-storage",
                },
                compression=config.parquet.compression,
                compression_level=config.parquet.compression_level,
                row_group_size=config.parquet.row_group_size,
            )
            store.complete_shard(
                namespace,
                shard.day,
                shard.view,
                actual_rows=write_result.row_count,
                file_size=write_result.file_size,
                sha256=write_result.sha256,
            )
            completed += 1
            processed_chunks += write_result.row_count
            store.add_event(run_id, "INFO", "shard_completed", {"shard": shard.key, "rows": write_result.row_count}, invocation_id)
            _emit_console_event(
                "qwen3_shard_completed",
                run_id=run_id,
                shard=shard.key,
                rows=write_result.row_count,
                output_path=str(output_path),
                sha256=write_result.sha256,
            )
        return finish("completed" if failed == 0 else "completed_with_failures")
    except BaseException as exc:
        failed = max(failed, 1)
        failures.append({"error_type": type(exc).__name__, "error_message": str(exc), "shard": None if current_shard is None else "/".join(current_shard)})
        if current_shard is not None:
            try:
                store.fail_shard(namespace, current_shard[0], current_shard[1], error_type=type(exc).__name__, error_message=str(exc))
            except Exception:
                pass
        if is_cuda_oom_error(type(exc).__name__, str(exc)):
            failures[-1]["cuda_oom"] = True
        return finish("failed", error=exc)
    finally:
        if gpu_monitor is not None:
            gpu_monitor.stop()
