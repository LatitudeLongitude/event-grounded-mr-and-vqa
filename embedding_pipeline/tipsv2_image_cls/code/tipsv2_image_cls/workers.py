from __future__ import annotations

import os
import queue as queue_module
import time
from dataclasses import dataclass, field
from math import ceil
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .encoder import TIPSv2Encoder
from .inventory import FrameRecord, HourTask
from .parquet_writer import write_hour_shard
from .progress import ProgressStore
from .reporting import append_event, write_report
from .schema import EMBEDDING_DIM, PREPROCESS_ID, SCHEMA_VERSION


@dataclass(frozen=True)
class BatchJob:
    task_key: str
    batch_index: int
    pixel_values: np.ndarray
    attempt: int = 1


@dataclass(frozen=True)
class BatchResult:
    task_key: str
    batch_index: int
    embeddings: Optional[np.ndarray]
    raw_norms: Optional[np.ndarray]
    model_id: str
    model_code_sha256: str
    attempt: int = 1
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    timings: Optional[dict] = None
    peak_memory_mib: float = 0.0


@dataclass(frozen=True)
class SmokeCheckJob:
    pixel_values: np.ndarray
    text: str = "a photo of a person"


@dataclass(frozen=True)
class SmokeCheckResult:
    cosine_similarity: float
    text_embedding: Optional[np.ndarray]
    error_type: Optional[str]
    error_message: Optional[str]
    dtype_evidence: dict = field(default_factory=dict)


@dataclass(frozen=True)
class WorkerReady:
    backend: str
    gpu_id: Optional[int]


class WriterBackpressureError(RuntimeError):
    pass


@dataclass(frozen=True)
class HourWriteRequest:
    task: HourTask
    frames: Tuple[FrameRecord, ...]
    attempt: int
    invocation_id: str
    namespace: str
    output_path: Path
    expected_rows: int
    batch_size: int
    metadata: dict

    @property
    def expected_batches(self) -> int:
        return ceil(self.expected_rows / self.batch_size)

    @property
    def estimated_result_bytes(self) -> int:
        return self.expected_rows * (
            EMBEDDING_DIM * np.dtype(np.float32).itemsize + np.dtype(np.float32).itemsize
        )

    @property
    def key(self) -> Tuple[str, int]:
        return self.task.key, self.attempt


@dataclass
class WriterHourAssembly:
    request: HourWriteRequest
    results: Dict[int, BatchResult] = field(default_factory=dict)
    received_rows: int = 0
    received_bytes: int = 0

    @property
    def received_batches(self) -> int:
        return len(self.results)

    def build_rows(self) -> list[dict]:
        rows = []
        for batch_index in sorted(self.results):
            result = self.results[batch_index]
            if result.error_type:
                raise RuntimeError(f"{result.error_type}: {result.error_message}")
            if result.embeddings is None or result.raw_norms is None:
                raise RuntimeError("Encoder returned no embeddings")
            start = batch_index * self.request.batch_size
            batch_frames = self.request.frames[start : start + len(result.embeddings)]
            if len(batch_frames) != len(result.embeddings):
                raise ValueError("Batch result row count does not match frame metadata")
            for frame, embedding, raw_norm in zip(batch_frames, result.embeddings, result.raw_norms):
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "embedding_id": (
                            f"{self.request.task.day}/{self.request.task.source}/"
                            f"{self.request.task.hour}/{frame.frame_index}"
                        ),
                        "day": self.request.task.day_number,
                        "source": self.request.task.source,
                        "video_hour_stem": self.request.task.hour,
                        "frame_index": frame.frame_index,
                        "timestamp_second": np.float32(frame.timestamp_second),
                        "absolute_second": float(
                            self.request.task.numeric_hour * 3600 + frame.timestamp_second
                        ),
                        "timestamp_hms": frame.timestamp_hms,
                        "frame_rel_path": frame.frame_rel_path,
                        "embedding": np.asarray(embedding, dtype=np.float32),
                        "embedding_norm_raw": np.float32(raw_norm),
                        "model_id": result.model_id,
                        "model_code_sha256": result.model_code_sha256,
                        "preprocess_id": PREPROCESS_ID,
                    }
                )
        if len(rows) != self.request.expected_rows:
            raise ValueError(
                f"Writer row count {len(rows)} does not match expected {self.request.expected_rows}"
            )
        return rows


class WriterBatchAggregator:
    def __init__(self, *, max_hours: int, max_batches: int, max_bytes: int):
        self.max_hours = max_hours
        self.max_batches = max_batches
        self.max_bytes = max_bytes
        self._hours: Dict[Tuple[str, int], WriterHourAssembly] = {}
        self._reserved_batches = 0
        self._reserved_bytes = 0

    @property
    def pending_hours(self) -> int:
        return len(self._hours)

    @property
    def pending_batches(self) -> int:
        return self._reserved_batches

    @property
    def pending_bytes(self) -> int:
        return self._reserved_bytes

    def start(self, request: HourWriteRequest) -> None:
        if request.key in self._hours:
            raise ValueError(f"Writer hour is already registered: {request.key}")
        if self.pending_hours + 1 > self.max_hours:
            raise WriterBackpressureError("writer pending hour limit exceeded")
        if self.pending_batches + request.expected_batches > self.max_batches:
            raise WriterBackpressureError("writer pending batch limit exceeded")
        if self.pending_bytes + request.estimated_result_bytes > self.max_bytes:
            raise WriterBackpressureError("writer pending byte limit exceeded")
        self._hours[request.key] = WriterHourAssembly(request)
        self._reserved_batches += request.expected_batches
        self._reserved_bytes += request.estimated_result_bytes

    def add(self, result: BatchResult) -> WriterHourAssembly:
        key = (result.task_key, result.attempt)
        assembly = self._hours.get(key)
        if assembly is None:
            raise ValueError(f"Writer batch attempt is not registered: {key}")
        if result.batch_index in assembly.results:
            raise ValueError(f"Duplicate batch index for {key}: {result.batch_index}")
        if not 0 <= result.batch_index < assembly.request.expected_batches:
            raise ValueError(f"Batch index is outside expected range: {result.batch_index}")
        if result.error_type is None:
            if result.embeddings is None or result.raw_norms is None:
                raise ValueError("Successful batch result is missing embeddings or norms")
            expected_rows = min(
                assembly.request.batch_size,
                assembly.request.expected_rows - result.batch_index * assembly.request.batch_size,
            )
            if len(result.embeddings) != expected_rows or len(result.raw_norms) != expected_rows:
                raise ValueError(
                    f"Batch row count does not match expected {expected_rows}: "
                    f"embeddings={len(result.embeddings)}, norms={len(result.raw_norms)}"
                )
            assembly.received_rows += len(result.embeddings)
            assembly.received_bytes += result.embeddings.nbytes + result.raw_norms.nbytes
        assembly.results[result.batch_index] = result
        return assembly

    def seal(self, task_key: str, attempt: int) -> WriterHourAssembly:
        key = (task_key, attempt)
        assembly = self._hours.get(key)
        if assembly is None:
            raise ValueError(f"Writer hour is not registered: {key}")
        expected = set(range(assembly.request.expected_batches))
        missing = sorted(expected - set(assembly.results))
        if missing:
            raise ValueError(f"Writer end marker found missing batch indexes for {key}: {missing}")
        if assembly.received_rows != assembly.request.expected_rows and not any(
            result.error_type for result in assembly.results.values()
        ):
            raise ValueError(
                f"Writer received {assembly.received_rows} rows instead of {assembly.request.expected_rows}"
            )
        return assembly

    def get(self, task_key: str, attempt: int) -> WriterHourAssembly:
        key = (task_key, attempt)
        assembly = self._hours.get(key)
        if assembly is None:
            raise ValueError(f"Writer hour is not registered: {key}")
        return assembly

    def contains(self, task_key: str, attempt: int) -> bool:
        return (task_key, attempt) in self._hours

    def release(self, task_key: str, attempt: int) -> WriterHourAssembly:
        key = (task_key, attempt)
        assembly = self._hours.pop(key)
        self._reserved_batches -= assembly.request.expected_batches
        self._reserved_bytes -= assembly.request.estimated_result_bytes
        return assembly

    def abort(self, task_key: str, attempt: int) -> Optional[WriterHourAssembly]:
        if (task_key, attempt) not in self._hours:
            return None
        return self.release(task_key, attempt)


def encoder_worker_main(
    input_queue: Queue,
    result_queue: Queue,
    control_queue: Optional[Queue] = None,
    *,
    backend: str,
    model_path: str,
    gpu_id: Optional[int],
    precision: str,
    embedding_dim: int,
    max_memory_fraction: Optional[float],
) -> None:
    encoder = None
    control = control_queue if control_queue is not None else result_queue
    try:
        if backend != "tipsv2":
            raise ValueError(f"Unknown encoder backend: {backend}")
        if gpu_id is None:
            raise ValueError("TIPSv2 worker requires an explicit GPU ID")
        encoder = TIPSv2Encoder(
            model_path,
            device=f"cuda:{gpu_id}",
            precision=precision,
            embedding_dim=embedding_dim,
            max_memory_fraction=max_memory_fraction,
        ).load()
        control.put(WorkerReady(backend=backend, gpu_id=gpu_id))
        while True:
            job = input_queue.get()
            if job is None:
                return
            try:
                if isinstance(job, SmokeCheckJob):
                    cosine, text_embedding, dtype_evidence = encoder.smoke_check(job.pixel_values, job.text)
                    control.put(SmokeCheckResult(cosine, text_embedding, None, None, dtype_evidence))
                    continue
                embeddings, raw_norms = encoder.encode_images(job.pixel_values)
                result_queue.put(
                    BatchResult(
                        task_key=job.task_key,
                        batch_index=job.batch_index,
                        embeddings=embeddings,
                        raw_norms=raw_norms,
                        model_id=encoder.model_id,
                        model_code_sha256=encoder.model_code_sha256,
                        attempt=job.attempt,
                        timings=dict(encoder.last_timings),
                        peak_memory_mib=float(encoder.last_peak_memory_mib),
                    )
                )
            except Exception as exc:
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass
                if isinstance(job, SmokeCheckJob):
                    control.put(SmokeCheckResult(0.0, None, type(exc).__name__, str(exc), {}))
                    continue
                result_queue.put(
                    BatchResult(
                        task_key=job.task_key,
                        batch_index=job.batch_index,
                        embeddings=None,
                        raw_norms=None,
                        model_id=getattr(encoder, "model_id", backend),
                        model_code_sha256=getattr(encoder, "model_code_sha256", "unknown"),
                        attempt=job.attempt,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
    except Exception as exc:
        control.put(
            BatchResult(
                task_key="__worker_startup__",
                batch_index=-1,
                embeddings=None,
                raw_norms=None,
                model_id=backend,
                model_code_sha256="unknown",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        )


def writer_worker_main(
    command_queue: Queue,
    response_queue: Queue,
    result_queue: Optional[Queue] = None,
    *,
    db_path: str,
    parquet_config: dict,
    log_jsonl_path: str,
    log_text_path: str,
    report_json_path: str,
    report_markdown_path: str,
    max_pending_hours: int = 2,
    max_pending_batches: int = 512,
    max_pending_bytes: int = 512 * 1024 * 1024,
    finalize_delay_seconds: float = 0.0,
    crash_after_batches: Optional[int] = None,
    crash_after_publish: bool = False,
) -> None:
    store = ProgressStore(db_path)
    log_jsonl = Path(log_jsonl_path)
    log_text = Path(log_text_path)
    report_json = Path(report_json_path)
    report_markdown = Path(report_markdown_path)
    aggregator = WriterBatchAggregator(
        max_hours=max_pending_hours,
        max_batches=max_pending_batches,
        max_bytes=max_pending_bytes,
    )
    ignored_attempts = set()
    received_batches = 0

    def emit(event_type: str, **payload) -> None:
        response_queue.put({"kind": "event", "event_type": event_type, **payload})

    def respond(request_id: str, *, ok: bool, value: Any = None, error: Optional[BaseException] = None) -> None:
        message = {"kind": "response", "request_id": request_id, "ok": ok, "value": value}
        if error is not None:
            message.update({"error_type": type(error).__name__, "error_message": str(error)})
        response_queue.put(message)

    def fail_hour(request: HourWriteRequest, exc: BaseException) -> None:
        store.fail_shard_for_invocation(
            request.invocation_id,
            request.namespace,
            request.task.day_number,
            request.task.source,
            request.task.hour,
            attempt=request.attempt,
            expected_rows=request.expected_rows,
            output_path=request.output_path,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        payload = {
            "task_key": request.task.key,
            "attempt": request.attempt,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        run_id = str(request.metadata["run_id"])
        store.add_event(run_id, "ERROR", "hour_failed", payload, request.invocation_id)
        append_event(
            log_jsonl,
            log_text,
            run_id=run_id,
            invocation_id=request.invocation_id,
            level="ERROR",
            event_type="hour_failed",
            payload=payload,
        )
        emit("hour_failed", **payload)

    def finalize_hour(task_key: str, attempt: int) -> None:
        assembly = None
        try:
            assembly = aggregator.seal(task_key, attempt)
            request = assembly.request
            failed_result = next((result for result in assembly.results.values() if result.error_type), None)
            if failed_result is not None:
                raise RuntimeError(f"{failed_result.error_type}: {failed_result.error_message}")
            rows = assembly.build_rows()
            if finalize_delay_seconds:
                time.sleep(finalize_delay_seconds)
            writer_started = time.monotonic()
            result = write_hour_shard(
                request.output_path,
                rows,
                run_id=request.invocation_id,
                metadata=request.metadata,
                **parquet_config,
            )
            if crash_after_publish:
                os._exit(18)
            store.complete_shard_for_invocation(
                request.invocation_id,
                request.namespace,
                request.task.day_number,
                request.task.source,
                request.task.hour,
                attempt=request.attempt,
                expected_rows=request.expected_rows,
                actual_rows=result.row_count,
                output_path=request.output_path,
                file_size=result.file_size,
                sha256=result.sha256,
            )
            timing_totals = {"h2d_seconds": 0.0, "forward_seconds": 0.0, "postprocess_seconds": 0.0}
            peak_memory_mib = 0.0
            for batch in assembly.results.values():
                for name, value in (batch.timings or {}).items():
                    timing_totals[name] = timing_totals.get(name, 0.0) + float(value)
                peak_memory_mib = max(peak_memory_mib, float(batch.peak_memory_mib))
            payload = {
                "task_key": request.task.key,
                "attempt": request.attempt,
                "rows": result.row_count,
                "output_path": str(result.path),
                "file_size": result.file_size,
                "sha256": result.sha256,
                "writer_seconds": time.monotonic() - writer_started,
                "peak_memory_mib": peak_memory_mib,
                "timings": timing_totals,
                "completed_at": time.time(),
            }
            run_id = str(request.metadata["run_id"])
            store.add_event(run_id, "INFO", "hour_completed", payload, request.invocation_id)
            append_event(
                log_jsonl,
                log_text,
                run_id=run_id,
                invocation_id=request.invocation_id,
                level="INFO",
                event_type="hour_completed",
                payload=payload,
            )
            emit("hour_completed", **payload)
        except Exception as exc:
            if assembly is None:
                try:
                    assembly = aggregator.get(task_key, attempt)
                except Exception:
                    assembly = None
            if assembly is not None:
                fail_hour(assembly.request, exc)
            else:
                raise
        finally:
            released = aggregator.abort(task_key, attempt)
            if released is not None:
                ignored_attempts.add((task_key, attempt))
            emit(
                "writer_backlog",
                pending_hours=aggregator.pending_hours,
                pending_batches=aggregator.pending_batches,
                pending_bytes=aggregator.pending_bytes,
            )

    def handle_command(command: dict) -> bool:
        operation = command["operation"]
        request_id = command.get("request_id", "")
        reply = bool(command.get("reply", True))
        try:
            if operation == "stop":
                if reply:
                    respond(request_id, ok=True)
                return False
            if operation == "initialize_run":
                store.register_invocation(
                    command["invocation_id"],
                    command["run_id"],
                    command["namespace"],
                    command["config"],
                    code_commit=command["code_commit"],
                )
                recovered = store.recover_interrupted()
                for shard in command["shards"]:
                    store.ensure_shard(**shard)
                payload = {
                    "namespace": command["namespace"],
                    "selected_hours": len(command["shards"]),
                    "selected_frames": sum(shard["expected_rows"] for shard in command["shards"]),
                    "code_commit": command["code_commit"],
                }
                store.add_event(
                    command["run_id"],
                    "INFO",
                    "invocation_started",
                    payload,
                    command["invocation_id"],
                )
                append_event(
                    log_jsonl,
                    log_text,
                    run_id=command["run_id"],
                    invocation_id=command["invocation_id"],
                    level="INFO",
                    event_type="invocation_started",
                    payload=payload,
                )
                value: Any = {"recovered": recovered}
            elif operation == "start_hour":
                request = command["request"]
                aggregator.start(request)
                store.start_shard_for_invocation(
                    request.invocation_id,
                    request.namespace,
                    request.task.day_number,
                    request.task.source,
                    request.task.hour,
                    run_id=command["run_id"],
                    attempt=request.attempt,
                    expected_rows=request.expected_rows,
                    output_path=request.output_path,
                    gpu_ids=command.get("gpu_ids", ""),
                    batch_size=request.batch_size,
                    worker_config=command.get("worker_config", ""),
                )
                emit(
                    "writer_backlog",
                    pending_hours=aggregator.pending_hours,
                    pending_batches=aggregator.pending_batches,
                    pending_bytes=aggregator.pending_bytes,
                )
                value = None
            elif operation == "seal_hour":
                if reply:
                    respond(request_id, ok=True)
                    reply = False
                finalize_hour(command["task_key"], command["attempt"])
                value = None
            elif operation == "abort_hour":
                assembly = aggregator.abort(command["task_key"], command["attempt"])
                if assembly is not None:
                    ignored_attempts.add((command["task_key"], command["attempt"]))
                    fail_hour(assembly.request, RuntimeError(command.get("reason", "hour aborted")))
                value = None
            elif operation == "mark_failed":
                arguments = command["arguments"]
                store.fail_shard_for_invocation(
                    arguments["invocation_id"],
                    arguments["namespace"],
                    arguments["day"],
                    arguments["source"],
                    arguments["hour"],
                    attempt=arguments.get("attempt"),
                    expected_rows=arguments.get("expected_rows"),
                    output_path=arguments.get("output_path"),
                    error_type=arguments["error_type"],
                    error_message=arguments["error_message"],
                )
                value = None
            elif operation == "record_skipped":
                arguments = command["arguments"]
                store.record_invocation_shard(
                    arguments["invocation_id"], arguments["namespace"], arguments["day"],
                    arguments["source"], arguments["hour"], action="skipped", status="completed",
                    expected_rows=arguments["expected_rows"], actual_rows=arguments["actual_rows"],
                    output_path=arguments["output_path"],
                )
                value = None
            elif operation == "event":
                arguments = command["arguments"]
                store.add_event(**arguments)
                append_event(
                    log_jsonl, log_text, run_id=arguments["run_id"],
                    invocation_id=arguments["invocation_id"], level=arguments["level"],
                    event_type=arguments["event_type"], payload=arguments["payload"],
                )
                value = None
            elif operation == "finish_run":
                arguments = command["arguments"]
                write_report(report_json, report_markdown, arguments["report"])
                payload = {
                    "status": arguments["status"],
                    "termination_reason": arguments.get("termination_reason"),
                }
                append_event(
                    log_jsonl,
                    log_text,
                    run_id=arguments["report"]["run_id"],
                    invocation_id=arguments["invocation_id"],
                    level="INFO" if arguments["status"] == "completed" else "ERROR",
                    event_type="invocation_finished",
                    payload=payload,
                )
                store.add_event(
                    arguments["report"]["run_id"],
                    "INFO" if arguments["status"] == "completed" else "ERROR",
                    "invocation_finished",
                    payload,
                    arguments["invocation_id"],
                )
                store.finish_invocation(**arguments)
                value = None
            else:
                raise ValueError(f"Unknown writer operation: {operation}")
            if reply:
                respond(request_id, ok=True, value=value)
            return True
        except Exception as exc:
            if reply:
                respond(request_id, ok=False, error=exc)
                return True
            emit("writer_fatal", error_type=type(exc).__name__, error_message=str(exc))
            return False

    try:
        store.initialize()
        while True:
            try:
                command = command_queue.get_nowait()
            except queue_module.Empty:
                command = None
            if command is not None:
                if not handle_command(command):
                    return
                continue
            if result_queue is None:
                command = command_queue.get()
                if not handle_command(command):
                    return
                continue
            try:
                result = result_queue.get(timeout=0.1)
            except queue_module.Empty:
                continue
            if result is None:
                continue
            if not isinstance(result, BatchResult):
                emit("writer_fatal", error_type="TypeError", error_message="Writer result queue received invalid message")
                return
            key = (result.task_key, result.attempt)
            if key in ignored_attempts:
                continue
            if not aggregator.contains(*key):
                deadline = time.monotonic() + 10.0
                while not aggregator.contains(*key):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        emit(
                            "writer_fatal",
                            error_type="ValueError",
                            error_message=f"Writer batch attempt is not registered: {key}",
                        )
                        return
                    try:
                        command = command_queue.get(timeout=min(0.1, remaining))
                    except queue_module.Empty:
                        continue
                    if not handle_command(command):
                        return
                    if key in ignored_attempts:
                        break
                if key in ignored_attempts:
                    continue
            try:
                assembly = aggregator.add(result)
            except Exception as exc:
                emit("writer_fatal", error_type=type(exc).__name__, error_message=str(exc))
                return
            received_batches += 1
            emit(
                "batch_accepted",
                task_key=result.task_key,
                attempt=result.attempt,
                batch_index=result.batch_index,
                rows=0 if result.embeddings is None else len(result.embeddings),
                received_batches=assembly.received_batches,
                expected_batches=assembly.request.expected_batches,
                timings=result.timings or {},
                peak_memory_mib=float(result.peak_memory_mib),
                error_type=result.error_type,
                error_message=result.error_message,
            )
            if crash_after_batches is not None and received_batches >= crash_after_batches:
                os._exit(17)
    except Exception as exc:
        emit("writer_fatal", error_type=type(exc).__name__, error_message=str(exc))
