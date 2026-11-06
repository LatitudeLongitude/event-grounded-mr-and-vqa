from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import re
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import AppConfig
from .encoder import COMPUTE_PRECISION_POLICY, resolved_versions
from .fingerprint import ProtectedTreeChangedError, ReadOnlySnapshotAudit, model_fingerprint
from .gpu_guard import (
    GPUMonitor,
    assert_gpus_idle,
    assert_no_foreign_compute_processes,
    gpu_report,
    query_gpus,
    validate_gpu_policy,
)
from .image_loader import ImageBatchLoader, load_image_batch
from .inventory import FrameRecord, HourTask, discover_tasks, filter_tasks, load_frame_manifest
from .paths import PathSafetyError, configure_runtime_environment
from .progress import ProgressStore
from .reporting import append_event, write_report
from .schema import PARQUET_EMBEDDING_VALUE_TYPE, PREPROCESS_ID, SCHEMA_VERSION, validate_embedding_array
from .validation import validate_shard
from .workers import (
    BatchJob,
    BatchResult,
    HourWriteRequest,
    SmokeCheckJob,
    SmokeCheckResult,
    WorkerReady,
    encoder_worker_main,
    writer_worker_main,
)


@dataclass(frozen=True)
class RunOptions:
    encoder_backend: str = "tipsv2"
    output_mode: str = "formal"
    run_id: Optional[str] = None
    gpu_ids: Optional[Tuple[int, ...]] = None
    day: Optional[str] = None
    source: Optional[str] = None
    hour: Optional[str] = None
    max_hours: Optional[int] = None
    max_frames_per_hour: Optional[int] = None
    decode_workers: Optional[int] = None
    resume: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    invocation_id: str
    namespace: str
    output_root: str
    selected_hours: int
    selected_frames: int
    completed: int
    skipped: int
    failed: int
    processed_frames: int
    elapsed_seconds: float
    images_per_second: float
    failures: Tuple[dict, ...]
    metrics: dict


class RetryQueue:
    def __init__(self, tasks):
        self._items = deque((task, 1) for task in tasks)

    def __bool__(self) -> bool:
        return bool(self._items)

    def pop(self):
        return self._items.popleft()

    def retry(self, task, attempt: int) -> None:
        self._items.append((task, attempt + 1))

    def defer(self, task, attempt: int) -> None:
        self._items.appendleft((task, attempt))


@dataclass
class ActiveHour:
    request: HourWriteRequest
    accepted_batches: int = 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _queue_depth(work_queue) -> int | None:
    try:
        return int(work_queue.qsize())
    except (AttributeError, NotImplementedError, OSError):
        return None


class PeriodicStatusReporter:
    def __init__(self, interval_seconds: float, snapshot_provider):
        self.interval_seconds = interval_seconds
        self.snapshot_provider = snapshot_provider
        self._stop = threading.Event()
        self._thread = None
        self._last_time = time.monotonic()
        self._last_images = 0

    def start(self) -> "PeriodicStatusReporter":
        self._thread = threading.Thread(target=self._loop, name="pipeline-status", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.emit()

    def emit(self) -> None:
        try:
            payload = dict(self.snapshot_provider())
            now = time.monotonic()
            processed_images = int(payload.get("processed_images", 0))
            interval = now - self._last_time
            payload["recent_images_per_second"] = (
                (processed_images - self._last_images) / interval if interval > 0 else 0.0
            )
            self._last_time = now
            self._last_images = processed_images
            print(json.dumps({"event_type": "pipeline_status", **payload}, sort_keys=True), flush=True)
        except BaseException:
            # Status reporting is observational and must not destabilize the run.
            return

    def stop(self, *, emit_final: bool = True) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 2))
            self._thread = None
        if emit_final:
            self.emit()


def is_cuda_oom_error(error_type: str, error_message: str) -> bool:
    normalized_type = error_type.lower()
    normalized_message = error_message.lower()
    return (
        "outofmemory" in normalized_type
        or "cuda out of memory" in normalized_message
        or "cuda error: out of memory" in normalized_message
    )


def should_retry_shard_failure(
    error_type: str,
    error_message: str,
    *,
    attempt: int,
    max_retries: int,
) -> bool:
    return attempt <= max_retries and not is_cuda_oom_error(error_type, error_message)


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
    commit = result.stdout.strip()
    return commit if commit else "unknown"


def validate_run_options(config: AppConfig, options: RunOptions) -> None:
    config.validate()
    if options.encoder_backend != "tipsv2":
        raise ValueError("encoder_backend must be tipsv2")
    if options.output_mode not in {"formal", "smoke", "benchmark", "recompute"}:
        raise ValueError("output_mode must be formal, smoke, benchmark, or recompute")
    if options.max_hours is not None and options.max_hours <= 0:
        raise ValueError("max_hours must be positive")
    if options.max_frames_per_hour is not None and options.max_frames_per_hour <= 0:
        raise ValueError("max_frames_per_hour must be positive")
    if options.decode_workers is not None and options.decode_workers not in {2, 4, 6}:
        raise ValueError("decode_workers must be one of 2, 4, or 6")
    if options.output_mode in {"formal", "recompute"}:
        if options.max_frames_per_hour is not None:
            raise ValueError(f"{options.output_mode} output must process the complete hour")
    if options.output_mode == "recompute":
        if not (options.day and options.source and options.hour):
            raise ValueError("recompute requires one explicit day/source/hour shard")
        if options.max_hours != 1:
            raise ValueError("recompute must select exactly one complete hour")
    gpu_ids = tuple(config.runtime.gpu_ids if options.gpu_ids is None else options.gpu_ids)
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError("GPU IDs cannot be negative")
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("GPU IDs must be unique")
    if not gpu_ids:
        raise ValueError("TIPSv2 backend requires at least one GPU ID")
    validate_gpu_policy(gpu_ids, config.runtime.max_memory_fraction_per_gpu)


class WriterClient:
    def __init__(self, command_queue, response_queue, process):
        self.command_queue = command_queue
        self.response_queue = response_queue
        self.process = process
        self._events = deque()
        self._responses = {}

    def assert_alive(self) -> None:
        if not self.process.is_alive():
            raise RuntimeError(f"Writer process exited unexpectedly with code {self.process.exitcode}")

    def _put(self, command: dict) -> None:
        while True:
            self.assert_alive()
            try:
                self.command_queue.put(command, timeout=1)
                return
            except queue.Full:
                self.drain_events()

    def notify(self, operation: str, **payload) -> None:
        self._put({"operation": operation, "reply": False, **payload})

    def _receive(self, timeout: float):
        try:
            message = self.response_queue.get(timeout=timeout)
        except queue.Empty as exc:
            self.assert_alive()
            raise exc
        if message.get("kind") == "event":
            if message.get("event_type") == "writer_fatal":
                raise RuntimeError(
                    f"Writer fatal error: {message.get('error_type')}: {message.get('error_message')}"
                )
            self._events.append(message)
            return None
        request_id = message.get("request_id")
        if request_id:
            self._responses[request_id] = message
        return message

    def call(self, operation: str, **payload):
        request_id = uuid.uuid4().hex
        self._put({"operation": operation, "request_id": request_id, "reply": True, **payload})
        while True:
            response = self._responses.pop(request_id, None)
            if response is None:
                try:
                    self._receive(1.0)
                except queue.Empty:
                    continue
                continue
            if not response.get("ok"):
                raise RuntimeError(f"{response.get('error_type')}: {response.get('error_message')}")
            return response.get("value")

    def next_event(self, timeout: float = 0.0) -> Optional[dict]:
        if self._events:
            return self._events.popleft()
        if timeout <= 0:
            try:
                message = self.response_queue.get_nowait()
            except queue.Empty:
                self.assert_alive()
                return None
            if message.get("kind") == "event":
                if message.get("event_type") == "writer_fatal":
                    raise RuntimeError(
                        f"Writer fatal error: {message.get('error_type')}: {message.get('error_message')}"
                    )
                return message
            request_id = message.get("request_id")
            if request_id:
                self._responses[request_id] = message
            return None
        deadline = time.monotonic() + timeout
        while True:
            if self._events:
                return self._events.popleft()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.assert_alive()
                return None
            try:
                self._receive(min(1.0, remaining))
            except queue.Empty:
                continue

    def drain_events(self) -> list[dict]:
        events = []
        while True:
            event = self.next_event(0.0)
            if event is None:
                return events
            events.append(event)


def validate_smoke_result(result: SmokeCheckResult) -> None:
    if result.error_type:
        raise ValueError(f"Smoke verification failed: {result.error_type}: {result.error_message}")
    if result.cosine_similarity < 1.0 - 1e-5:
        raise ValueError(f"Repeated-image cosine similarity is too low: {result.cosine_similarity}")
    if result.text_embedding is None:
        raise ValueError("Smoke verification returned no text embedding")
    validate_embedding_array(result.text_embedding, require_normalized=True)
    expected_dtypes = {
        "cls_token_dtype": "torch.float32",
        "normalized_image_embedding_dtype": "torch.float32",
        "raw_text_embedding_dtype": "torch.float32",
        "normalized_text_embedding_dtype": "torch.float32",
        "similarity_dtype": "torch.float32",
        "numpy_embedding_before_save_dtype": "np.float32",
        "parquet_embedding_value_type": PARQUET_EMBEDDING_VALUE_TYPE,
        "compute_precision_policy": COMPUTE_PRECISION_POLICY,
    }
    evidence = result.dtype_evidence or {}
    missing = sorted(key for key in expected_dtypes if key not in evidence)
    if missing:
        raise ValueError(f"Smoke verification missing dtype evidence: {missing}")
    mismatches = {
        key: {"expected": expected, "actual": evidence.get(key)}
        for key, expected in expected_dtypes.items()
        if evidence.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Smoke verification dtype evidence mismatch: {mismatches}")
    if evidence.get("torch_autocast_cuda_enabled") is not False:
        raise ValueError("Smoke verification detected enabled CUDA autocast")
    if evidence.get("torch_autocast_cpu_enabled") is not False:
        raise ValueError("Smoke verification detected enabled CPU autocast")


def _run_smoke_check(first_frame: FrameRecord, input_queue, result_queue, worker_processes, image_size: int) -> SmokeCheckResult:
    pixel_values = load_image_batch([first_frame.path], image_size=image_size, decode_workers=1)[0]
    input_queue.put(SmokeCheckJob(pixel_values))
    while True:
        try:
            result = result_queue.get(timeout=60)
        except queue.Empty as exc:
            dead = [process.pid for process in worker_processes if not process.is_alive()]
            if dead:
                raise RuntimeError(f"Encoder worker exited during smoke verification: {dead}") from exc
            continue
        if not isinstance(result, SmokeCheckResult):
            raise RuntimeError("Unexpected encoder result during smoke verification")
        validate_smoke_result(result)
        return result


def _wait_for_workers(worker_processes, result_queue) -> None:
    ready = 0
    while ready < len(worker_processes):
        try:
            result = result_queue.get(timeout=60)
        except queue.Empty as exc:
            dead = [process.pid for process in worker_processes if not process.is_alive()]
            if dead:
                raise RuntimeError(f"Encoder worker exited during model loading: {dead}") from exc
            continue
        if isinstance(result, WorkerReady):
            ready += 1
            continue
        if isinstance(result, BatchResult) and result.task_key == "__worker_startup__":
            raise RuntimeError(f"Encoder worker startup failed: {result.error_type}: {result.error_message}")
        raise RuntimeError("Unexpected message while waiting for encoder workers")


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def resolve_run_layout(config: AppConfig, options: RunOptions) -> Tuple[str, Path, Path]:
    run_id = options.run_id if options.run_id is not None else time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    if not RUN_ID_PATTERN.fullmatch(run_id) or run_id in {".", ".."}:
        raise PathSafetyError("run_id must be a single safe path component")
    if options.output_mode == "smoke":
        root = config.paths.output_root / "smoke" / run_id
        namespace = f"smoke:{run_id}"
    elif options.output_mode == "benchmark":
        root = config.paths.output_root / "benchmark" / run_id
        namespace = f"benchmark:{run_id}"
    elif options.output_mode == "recompute":
        root = config.paths.output_root / "recompute" / run_id
        namespace = f"recompute:{run_id}"
    elif options.output_mode == "formal":
        root = config.paths.output_root
        namespace = "main"
    else:
        raise ValueError("output_mode must be formal, smoke, benchmark, or recompute")
    config.paths.require_output(root / ".path-safety-check")
    state_path = config.paths.require_output(root / "_state" / "progress.sqlite3")
    return namespace, root, state_path


def _shard_path(run_root: Path, task: HourTask) -> Path:
    path = run_root / "main" / task.day / task.source / f"{task.hour}.parquet"
    path.resolve().relative_to(run_root.resolve())
    return path


def _shutdown_compute_resources(image_loader, input_queues, workers) -> None:
    if image_loader is not None:
        image_loader.close()
    for input_queue in input_queues:
        try:
            input_queue.put_nowait(None)
        except queue.Full:
            pass
    for process in workers:
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)


def _capture_readonly_after(audit: ReadOnlySnapshotAudit | None) -> tuple[dict | None, BaseException | None]:
    if audit is None:
        return None, None
    try:
        return audit.capture_after_and_compare(), None
    except ProtectedTreeChangedError as exc:
        return exc.diff, exc
    except BaseException as exc:
        return None, exc


def run_pipeline(
    config: AppConfig,
    options: RunOptions,
    *,
    writer_target=writer_worker_main,
) -> RunSummary:
    started = time.monotonic()
    started_at_utc = _utc_now()
    cpu_started = time.process_time()
    validate_run_options(config, options)
    decode_workers = config.runtime.decode_workers if options.decode_workers is None else options.decode_workers
    run_id = options.run_id or time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    invocation_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex
    code_commit = _resolve_code_commit(config.paths.project_root)
    options = RunOptions(**{**asdict(options), "run_id": run_id})
    namespace, run_root, db_path = resolve_run_layout(config, options)
    inventory = discover_tasks(config.paths.data_root, config.summary_path)
    tasks = filter_tasks(
        inventory.tasks,
        day=options.day,
        source=options.source,
        hour=options.hour,
        max_hours=options.max_hours,
    )
    if not tasks:
        raise ValueError("No hour tasks matched the requested filters")
    selected_counts = [
        min(task.frame_count, options.max_frames_per_hour)
        if options.max_frames_per_hour
        else task.frame_count
        for task in tasks
    ]
    if options.dry_run:
        return RunSummary(
            run_id, invocation_id, namespace, str(run_root), len(tasks), sum(selected_counts),
            0, 0, 0, 0, 0.0, 0.0, (), {},
        )
    configure_runtime_environment(config.paths.output_root)

    gpu_ids = tuple(config.runtime.gpu_ids if options.gpu_ids is None else options.gpu_ids)
    gpu_snapshot = ()
    gpu_monitor = None
    readonly_audit = None
    readonly_diff = None
    readonly_before_captured = False
    gpu_snapshot = query_gpus()
    assert_gpus_idle(
        gpu_snapshot,
        gpu_ids,
        max_utilization=config.runtime.max_gpu_utilization,
        min_free_mib=config.runtime.min_free_memory_mib,
    )
    model_manifest = model_fingerprint(config.paths.model_root, hash_weights=False)
    if options.output_mode == "smoke":
        readonly_audit = ReadOnlySnapshotAudit.create(
            config.paths.output_root,
            run_id,
            {"model": config.paths.model_root, "processed_data": config.paths.data_root},
        )
        for path in (
            readonly_audit.before_path,
            readonly_audit.after_path,
            readonly_audit.diff_json_path,
            readonly_audit.diff_markdown_path,
        ):
            config.paths.require_output(path)

    worker_slots: Tuple[int, ...] = tuple(gpu_ids)
    context = mp.get_context("spawn")
    worker_control_queue = context.Queue(maxsize=max(8, len(worker_slots) * 4))
    writer_results = context.Queue(maxsize=config.runtime.writer_result_queue_size)
    input_queues = [context.Queue(maxsize=config.runtime.prefetch_batches_per_gpu) for _ in worker_slots]
    workers = []
    writer_commands = context.Queue(maxsize=config.runtime.writer_command_queue_size)
    writer_responses = context.Queue(maxsize=config.runtime.writer_event_queue_size)
    report_json_path = config.paths.require_output(
        config.paths.output_root / "_reports" / run_id / f"{invocation_id}.json"
    )
    report_markdown_path = config.paths.require_output(
        config.paths.output_root / "_reports" / run_id / f"{invocation_id}.md"
    )
    log_jsonl_path = config.paths.require_output(config.paths.output_root / "_logs" / f"{run_id}.jsonl")
    log_text_path = config.paths.require_output(config.paths.output_root / "_logs" / f"{run_id}.log")
    invocation_config = {
        "options": asdict(options),
        "runtime": asdict(config.runtime),
        "compute_precision_policy": COMPUTE_PRECISION_POLICY,
    }
    writer = context.Process(
        target=writer_target,
        kwargs={
            "command_queue": writer_commands,
            "response_queue": writer_responses,
            "result_queue": writer_results,
            "db_path": str(db_path),
            "parquet_config": {
                "compression": config.parquet.compression,
                "compression_level": config.parquet.compression_level,
                "row_group_size": config.parquet.row_group_size,
            },
            "log_jsonl_path": str(log_jsonl_path),
            "log_text_path": str(log_text_path),
            "report_json_path": str(report_json_path),
            "report_markdown_path": str(report_markdown_path),
            "max_pending_hours": config.runtime.writer_max_pending_hours,
            "max_pending_batches": config.runtime.writer_max_pending_batches,
            "max_pending_bytes": config.runtime.writer_max_pending_bytes,
        },
        name="tipsv2-writer",
    )
    image_loader = None
    client = None
    active: Dict[Tuple[str, int], ActiveHour] = {}
    completed = skipped = failed = processed_frames = 0
    failures: List[dict] = []
    aggregate_metrics = {
        "decode_seconds": 0.0,
        "h2d_seconds": 0.0,
        "forward_seconds": 0.0,
        "postprocess_seconds": 0.0,
        "writer_seconds": 0.0,
        "peak_memory_mib": 0.0,
        "backpressure_wait_count": 0,
        "max_writer_pending_hours": 0,
        "max_writer_pending_batches": 0,
        "max_writer_pending_bytes": 0,
        "hour_submission_started_at": {},
        "writer_completion_at": {},
        "compute_precision_policy": COMPUTE_PRECISION_POLICY,
        "parquet_embedding_value_type": PARQUET_EMBEDDING_VALUE_TYPE,
    }
    reserved_batches = 0
    reserved_bytes = 0
    processing_started = None
    skipped_frames = 0
    submitted_frames = 0
    retry_count = 0
    status_stage = "initializing"
    status_reporter = None

    def status_snapshot() -> dict:
        elapsed = time.monotonic() - started
        handled_images = processed_frames + skipped_frames
        remaining_images = max(0, sum(selected_counts) - handled_images)
        global_rate = processed_frames / elapsed if elapsed else 0.0
        timing_names = (
            "decode_seconds",
            "h2d_seconds",
            "forward_seconds",
            "postprocess_seconds",
            "writer_seconds",
        )
        timing_total = sum(float(aggregate_metrics.get(name, 0.0)) for name in timing_names)
        return {
            "run_id": run_id,
            "invocation_id": invocation_id,
            "stage": status_stage,
            "current_hours": sorted(hour.request.task.key for hour in active.values()),
            "hours_completed": completed,
            "hours_skipped": skipped,
            "hours_failed": failed,
            "hours_total": len(tasks),
            "processed_images": processed_frames,
            "submitted_images": submitted_frames,
            "skipped_images": skipped_frames,
            "handled_images": handled_images,
            "total_images": sum(selected_counts),
            "global_images_per_second": global_rate,
            "eta_seconds": remaining_images / global_rate if global_rate > 0 else None,
            "failed_count": failed,
            "retry_count": retry_count,
            "queue_depths": {
                "gpu_input": {
                    str(slot): _queue_depth(input_queue)
                    for slot, input_queue in zip(worker_slots, input_queues)
                },
                "writer_results": _queue_depth(writer_results),
                "writer_commands": _queue_depth(writer_commands),
                "writer_events": _queue_depth(writer_responses),
                "reserved_batches": reserved_batches,
                "reserved_bytes": reserved_bytes,
            },
            "gpu_metrics": gpu_monitor.snapshot() if gpu_monitor is not None else {},
            "timing_shares": {
                name: (float(aggregate_metrics.get(name, 0.0)) / timing_total if timing_total else 0.0)
                for name in timing_names
            },
        }

    def stop_gpu_monitor() -> tuple[dict, BaseException | None]:
        nonlocal gpu_monitor
        if gpu_monitor is None:
            return {}, None
        monitor = gpu_monitor
        gpu_monitor = None
        metrics = monitor.stop()
        try:
            monitor.raise_if_failed()
        except BaseException as monitor_exc:
            return metrics, monitor_exc
        return metrics, None

    def check_runtime() -> None:
        if client is not None:
            client.assert_alive()
        dead = [process.pid for process in workers if not process.is_alive()]
        if dead:
            raise RuntimeError(f"Encoder worker exited unexpectedly: {dead}")
        if gpu_monitor is not None:
            gpu_monitor.raise_if_failed()

    def handle_event(event: dict) -> None:
        nonlocal completed, failed, processed_frames, reserved_batches, reserved_bytes, retry_count
        event_type = event["event_type"]
        if event_type == "writer_backlog":
            aggregate_metrics["max_writer_pending_hours"] = max(
                aggregate_metrics["max_writer_pending_hours"], int(event["pending_hours"])
            )
            aggregate_metrics["max_writer_pending_batches"] = max(
                aggregate_metrics["max_writer_pending_batches"], int(event["pending_batches"])
            )
            aggregate_metrics["max_writer_pending_bytes"] = max(
                aggregate_metrics["max_writer_pending_bytes"], int(event["pending_bytes"])
            )
            return
        key = (event.get("task_key"), int(event.get("attempt", 0)))
        if event_type == "batch_accepted":
            hour = active.get(key)
            if hour is None:
                raise RuntimeError(f"Writer accepted a batch for an inactive hour: {key}")
            hour.accepted_batches += 1
            for name, value in (event.get("timings") or {}).items():
                aggregate_metrics[name] = aggregate_metrics.get(name, 0.0) + float(value)
            aggregate_metrics["peak_memory_mib"] = max(
                aggregate_metrics["peak_memory_mib"], float(event.get("peak_memory_mib", 0.0))
            )
            return
        if event_type not in {"hour_completed", "hour_failed"}:
            return
        hour = active.pop(key, None)
        if hour is None:
            raise RuntimeError(f"Writer completed an inactive hour: {key}")
        reserved_batches -= hour.request.expected_batches
        reserved_bytes -= hour.request.estimated_result_bytes
        if event_type == "hour_completed":
            completed += 1
            processed_frames += int(event["rows"])
            aggregate_metrics["writer_seconds"] += float(event.get("writer_seconds", 0.0))
            aggregate_metrics["writer_completion_at"][hour.request.task.key] = float(event["completed_at"])
        else:
            error = {
                "task": hour.request.task.key,
                "error_type": event.get("error_type", "WriterError"),
                "error_message": event.get("error_message", "writer failed"),
            }
            if should_retry_shard_failure(
                error["error_type"],
                error["error_message"],
                attempt=hour.request.attempt,
                max_retries=config.runtime.max_retries,
            ):
                pending.retry(hour.request.task, hour.request.attempt)
                retry_count += 1
            else:
                failed += 1
                failures.append(error)

    def pump_events(*, wait: float = 0.0) -> None:
        if client is None:
            return
        event = client.next_event(wait)
        if event is not None:
            handle_event(event)
        for queued in client.drain_events():
            handle_event(queued)

    def put_job(target_queue, job: BatchJob) -> None:
        while True:
            check_runtime()
            try:
                target_queue.put(job, timeout=1)
                return
            except queue.Full:
                aggregate_metrics["backpressure_wait_count"] += 1
                pump_events()

    try:
        status_reporter = PeriodicStatusReporter(
            config.runtime.status_interval_seconds, status_snapshot
        ).start()
        writer.start()
        client = WriterClient(writer_commands, writer_responses, writer)
        client.call(
            "initialize_run",
            invocation_id=invocation_id,
            run_id=run_id,
            namespace=namespace,
            config=invocation_config,
            code_commit=code_commit,
            shards=[
                {
                    "namespace": namespace,
                    "day": task.day_number,
                    "source": task.source,
                    "hour": task.hour,
                    "expected_rows": selected_counts[index],
                    "output_path": _shard_path(run_root, task),
                }
                for index, task in enumerate(tasks)
            ],
        )
        if readonly_audit is not None:
            status_stage = "readonly_snapshot_before"
            readonly_audit.capture_before()
            readonly_before_captured = True
        gpu_monitor = GPUMonitor(gpu_ids).start()

        for input_queue, gpu_id in zip(input_queues, worker_slots):
            status_stage = "loading_model"
            process = context.Process(
                target=encoder_worker_main,
                kwargs={
                    "input_queue": input_queue,
                    "result_queue": writer_results,
                    "control_queue": worker_control_queue,
                    "backend": options.encoder_backend,
                    "model_path": str(config.paths.model_root),
                    "gpu_id": gpu_id,
                    "precision": config.model.precision,
                    "embedding_dim": config.model.embedding_dim,
                    "max_memory_fraction": (
                        config.runtime.max_memory_fraction_per_gpu if len(gpu_ids) == 3 else None
                    ),
                },
                name=f"tipsv2-encoder-{gpu_id}",
            )
            process.start()
            workers.append(process)
        _wait_for_workers(workers, worker_control_queue)
        assert_no_foreign_compute_processes(
            query_gpus(), gpu_ids, [process.pid for process in workers if process.pid is not None]
        )
        image_loader = ImageBatchLoader(
            image_size=config.model.image_size,
            decode_workers=decode_workers,
        )
        processing_started = time.monotonic()
        status_stage = "processing"

        pending = RetryQueue(tasks)
        read_store = ProgressStore(db_path)
        while pending or active:
            pump_events()
            check_runtime()
            if not pending:
                pump_events(wait=1.0)
                continue
            task, attempt = pending.pop()
            expected_rows = (
                min(task.frame_count, options.max_frames_per_hour)
                if options.max_frames_per_hour
                else task.frame_count
            )
            expected_batches = (expected_rows + config.runtime.batch_size - 1) // config.runtime.batch_size
            estimated_bytes = expected_rows * (config.model.embedding_dim * 4 + 4)
            if expected_batches > config.runtime.writer_max_pending_batches:
                raise ValueError(
                    f"Hour {task.key} requires {expected_batches} writer batches, exceeding configured limit "
                    f"{config.runtime.writer_max_pending_batches}"
                )
            if estimated_bytes > config.runtime.writer_max_pending_bytes:
                raise ValueError(
                    f"Hour {task.key} requires {estimated_bytes} result bytes, exceeding configured limit "
                    f"{config.runtime.writer_max_pending_bytes}"
                )
            shard_path = _shard_path(run_root, task)
            state = read_store.get_shard(namespace, task.day_number, task.source, task.hour)
            if shard_path.exists() and (not state or state["status"] != "completed"):
                exc = FileExistsError(f"Refusing to overwrite uncommitted existing shard: {shard_path}")
                client.notify(
                    "mark_failed",
                    arguments={
                        "invocation_id": invocation_id,
                        "namespace": namespace,
                        "day": task.day_number,
                        "source": task.source,
                        "hour": task.hour,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "attempt": attempt,
                        "expected_rows": expected_rows,
                        "output_path": shard_path,
                    },
                )
                if attempt <= config.runtime.max_retries:
                    pending.retry(task, attempt)
                    retry_count += 1
                else:
                    failed += 1
                    failures.append(
                        {"task": task.key, "error_type": type(exc).__name__, "error_message": str(exc)}
                    )
                continue
            if state and state["status"] == "completed":
                validation = validate_shard(shard_path, expected_rows=expected_rows)
                if validation.valid and options.resume:
                    client.notify(
                        "record_skipped",
                        arguments={
                            "invocation_id": invocation_id,
                            "namespace": namespace,
                            "day": task.day_number,
                            "source": task.source,
                            "hour": task.hour,
                            "expected_rows": expected_rows,
                            "actual_rows": validation.row_count,
                            "output_path": shard_path,
                        },
                    )
                    skipped += 1
                    skipped_frames += expected_rows
                    continue
                if validation.valid:
                    raise FileExistsError(f"Completed shard already exists: {shard_path}")
                client.notify(
                    "mark_failed",
                    arguments={
                        "invocation_id": invocation_id,
                        "namespace": namespace,
                        "day": task.day_number,
                        "source": task.source,
                        "hour": task.hour,
                        "error_type": "InvalidCompletedShard",
                        "error_message": validation.error or "invalid shard",
                        "attempt": attempt,
                        "expected_rows": expected_rows,
                        "output_path": shard_path,
                    },
                )
            can_reserve = (
                len(active) + 1 <= config.runtime.writer_max_pending_hours
                and reserved_batches + expected_batches <= config.runtime.writer_max_pending_batches
                and reserved_bytes + estimated_bytes <= config.runtime.writer_max_pending_bytes
            )
            if not can_reserve:
                pending.defer(task, attempt)
                aggregate_metrics["backpressure_wait_count"] += 1
                pump_events(wait=1.0)
                continue

            all_frames = load_frame_manifest(task, config.paths.data_root)
            frames = all_frames[:expected_rows]
            smoke_check = None
            if options.output_mode == "smoke":
                smoke_check = _run_smoke_check(
                    frames[0], input_queues[0], worker_control_queue, workers, config.model.image_size
                )
                aggregate_metrics["smoke_dtype_evidence"] = dict(smoke_check.dtype_evidence)
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "invocation_id": invocation_id,
                "compute_precision_policy": COMPUTE_PRECISION_POLICY,
                "parquet_embedding_value_type": PARQUET_EMBEDDING_VALUE_TYPE,
                "model_id": config.model.model_id,
                "model_code_sha256": model_manifest["files"]["modeling_tips.py"]["sha256"],
                "model_fingerprint": json.dumps(model_manifest, sort_keys=True),
                "preprocess_id": PREPROCESS_ID,
                "preprocess_parameters": json.dumps(
                    {
                        "antialias": True,
                        "color_mode": "RGB",
                        "dtype": "float32",
                        "implementation": "Pillow.Image.resize",
                        "interpolation": "bilinear",
                        "normalization": "none",
                        "resize": [config.model.image_size, config.model.image_size],
                        "value_range": [0.0, 1.0],
                    },
                    sort_keys=True,
                ),
                "repeat_image_cosine": smoke_check.cosine_similarity if smoke_check else "not-run",
                "text_embedding_shape": list(smoke_check.text_embedding.shape) if smoke_check else "not-run",
            }
            if smoke_check is not None:
                for key, value in smoke_check.dtype_evidence.items():
                    metadata[f"smoke_{key}"] = value
            request = HourWriteRequest(
                task=task,
                frames=tuple(frames),
                attempt=attempt,
                invocation_id=invocation_id,
                namespace=namespace,
                output_path=shard_path,
                expected_rows=expected_rows,
                batch_size=config.runtime.batch_size,
                metadata=metadata,
            )
            key = request.key
            active[key] = ActiveHour(request)
            reserved_batches += request.expected_batches
            reserved_bytes += request.estimated_result_bytes
            aggregate_metrics["hour_submission_started_at"][task.key] = time.time()
            client.notify(
                "start_hour",
                request=request,
                run_id=run_id,
                gpu_ids=",".join(map(str, gpu_ids)),
                worker_config=json.dumps({"decode_workers": decode_workers}),
            )
            client.notify(
                "event",
                arguments={
                    "run_id": run_id,
                    "invocation_id": invocation_id,
                    "level": "INFO",
                    "event_type": "hour_started",
                    "payload": {"task": task.key, "attempt": attempt, "frames": len(frames)},
                },
            )
            for batch_index in range(request.expected_batches):
                batch_frames = frames[
                    batch_index * request.batch_size : (batch_index + 1) * request.batch_size
                ]
                decode_started = time.monotonic()
                pixel_values = image_loader.load([frame.path for frame in batch_frames])
                aggregate_metrics["decode_seconds"] += time.monotonic() - decode_started
                target = batch_index % len(input_queues)
                put_job(
                    input_queues[target],
                    BatchJob(task.key, batch_index, pixel_values, attempt=attempt),
                )
                submitted_frames += len(batch_frames)
            while key in active and active[key].accepted_batches < request.expected_batches:
                pump_events(wait=1.0)
                check_runtime()
            if key in active:
                client.notify("seal_hour", task_key=task.key, attempt=attempt)

        processing_elapsed = time.monotonic() - processing_started
        aggregate_metrics["processing_elapsed_seconds"] = processing_elapsed
        aggregate_metrics["processing_images_per_second"] = (
            processed_frames / processing_elapsed if processing_elapsed else 0.0
        )
        status_stage = "finalizing"
        _shutdown_compute_resources(image_loader, input_queues, workers)
        image_loader = None
        workers.clear()
        if status_reporter is not None:
            status_reporter.stop()
        gpu_metrics, monitor_failure = stop_gpu_monitor()
        readonly_diff, readonly_failure = _capture_readonly_after(
            readonly_audit if readonly_before_captured else None
        )
        if readonly_failure is not None:
            raise readonly_failure
        if monitor_failure is not None:
            raise monitor_failure
        elapsed = time.monotonic() - started
        cpu_seconds = time.process_time() - cpu_started
        aggregate_metrics["coordinator_cpu_seconds"] = cpu_seconds
        aggregate_metrics["coordinator_cpu_percent_of_host"] = (
            cpu_seconds / elapsed / max(1, os.cpu_count() or 1) * 100 if elapsed else 0.0
        )
        aggregate_metrics["gpu_monitor"] = gpu_metrics
        finished_at_utc = _utc_now()
        report = {
            "run_id": run_id,
            "invocation_id": invocation_id,
            "code_commit": code_commit,
            "started_at_utc": started_at_utc,
            "finished_at_utc": finished_at_utc,
            "arguments": asdict(options),
            "status": "completed" if failed == 0 else "completed_with_failures",
            "namespace": namespace,
            "completed": completed,
            "skipped": skipped,
            "failed": failed,
            "processed_frames": processed_frames,
            "elapsed_seconds": elapsed,
            "failures": failures,
            "images_per_second": processed_frames / elapsed if elapsed else 0.0,
            "inventory": inventory.as_dict(),
            "gpu_snapshot": gpu_report(gpu_snapshot),
            "model_manifest": model_manifest,
            "versions": resolved_versions(),
            "compute_precision_policy": COMPUTE_PRECISION_POLICY,
            "parquet_embedding_value_type": PARQUET_EMBEDDING_VALUE_TYPE,
            "dtype_evidence": aggregate_metrics.get("smoke_dtype_evidence", {}),
            "metrics": aggregate_metrics,
            "read_only_snapshot_diff": readonly_diff,
        }
        client.call(
            "finish_run",
            arguments={
                "invocation_id": invocation_id,
                "status": report["status"],
                "report": report,
                "processed": completed,
                "skipped": skipped,
                "failed": failed,
                "elapsed_seconds": elapsed,
                "images_per_second": report["images_per_second"],
                "gpu_metrics": gpu_metrics,
                "termination_reason": report["status"],
            },
        )
        return RunSummary(
            run_id,
            invocation_id,
            namespace,
            str(run_root),
            len(tasks),
            sum(selected_counts),
            completed,
            skipped,
            failed,
            processed_frames,
            elapsed,
            report["images_per_second"],
            tuple(failures),
            aggregate_metrics,
        )
    except BaseException as exc:
        status_stage = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        if processing_started is not None and "processing_elapsed_seconds" not in aggregate_metrics:
            processing_elapsed = time.monotonic() - processing_started
            aggregate_metrics["processing_elapsed_seconds"] = processing_elapsed
            aggregate_metrics["processing_images_per_second"] = (
                processed_frames / processing_elapsed if processing_elapsed else 0.0
            )
        _shutdown_compute_resources(image_loader, input_queues, workers)
        image_loader = None
        workers.clear()
        if status_reporter is not None:
            status_reporter.stop()
        gpu_metrics, monitor_failure = stop_gpu_monitor()
        aggregate_metrics["gpu_monitor"] = gpu_metrics
        readonly_failure = None
        if readonly_before_captured and readonly_diff is None:
            readonly_diff, readonly_failure = _capture_readonly_after(readonly_audit)
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        failure_items = [{"error_type": type(exc).__name__, "error_message": str(exc)}]
        if readonly_failure is not None and readonly_failure is not exc:
            failure_items.append(
                {"error_type": type(readonly_failure).__name__, "error_message": str(readonly_failure)}
            )
        if monitor_failure is not None and monitor_failure is not exc:
            failure_items.append(
                {"error_type": type(monitor_failure).__name__, "error_message": str(monitor_failure)}
            )
        finished_at_utc = _utc_now()
        failure_report = {
            "run_id": run_id,
            "invocation_id": invocation_id,
            "code_commit": code_commit,
            "started_at_utc": started_at_utc,
            "finished_at_utc": finished_at_utc,
            "arguments": asdict(options),
            "status": status,
            "namespace": namespace,
            "completed": completed,
            "skipped": skipped,
            "failed": max(1, failed),
            "processed_frames": processed_frames,
            "elapsed_seconds": time.monotonic() - started,
            "images_per_second": 0.0,
            "failures": failure_items,
            "inventory": inventory.as_dict(),
            "gpu_snapshot": gpu_report(gpu_snapshot),
            "model_manifest": model_manifest,
            "versions": resolved_versions(),
            "compute_precision_policy": COMPUTE_PRECISION_POLICY,
            "parquet_embedding_value_type": PARQUET_EMBEDDING_VALUE_TYPE,
            "dtype_evidence": aggregate_metrics.get("smoke_dtype_evidence", {}),
            "metrics": aggregate_metrics,
            "read_only_snapshot_diff": readonly_diff,
        }
        writer_finished = False
        if client is not None and writer.is_alive():
            try:
                for hour in list(active.values()):
                    client.notify(
                        "abort_hour",
                        task_key=hour.request.task.key,
                        attempt=hour.request.attempt,
                        reason=f"{status}: {type(exc).__name__}: {exc}",
                    )
                client.notify(
                    "event",
                    arguments={
                        "run_id": run_id,
                        "invocation_id": invocation_id,
                        "level": "ERROR",
                        "event_type": "invocation_failed",
                        "payload": failure_items[0],
                    },
                )
                client.call(
                    "finish_run",
                    arguments={
                        "invocation_id": invocation_id,
                        "status": status,
                        "report": failure_report,
                        "processed": completed,
                        "skipped": skipped,
                        "failed": max(1, failed),
                        "elapsed_seconds": failure_report["elapsed_seconds"],
                        "images_per_second": 0.0,
                        "gpu_metrics": gpu_metrics,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "termination_reason": status,
                    },
                )
                writer_finished = True
            except Exception:
                writer_finished = False
        if not writer_finished:
            if writer.is_alive():
                writer.terminate()
                writer.join(timeout=5)
            emergency_store = ProgressStore(db_path)
            emergency_store.initialize()
            invocation = emergency_store.get_invocation(invocation_id)
            if invocation is None:
                emergency_store.register_invocation(
                    invocation_id,
                    run_id,
                    namespace,
                    invocation_config,
                    code_commit=code_commit,
                )
                for index, task in enumerate(tasks):
                    emergency_store.ensure_shard(
                        namespace,
                        task.day_number,
                        task.source,
                        task.hour,
                        expected_rows=selected_counts[index],
                        output_path=_shard_path(run_root, task),
                    )
                invocation = emergency_store.get_invocation(invocation_id)
            if invocation is not None and invocation["finished_at"] is None:
                emergency_store.fail_open_invocation_shards(
                    invocation_id,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                if not report_json_path.exists() and not report_markdown_path.exists():
                    write_report(report_json_path, report_markdown_path, failure_report)
                event_payload = {
                    "status": status,
                    "termination_reason": status,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
                append_event(
                    log_jsonl_path,
                    log_text_path,
                    run_id=run_id,
                    invocation_id=invocation_id,
                    level="ERROR",
                    event_type="invocation_finished",
                    payload=event_payload,
                )
                emergency_store.add_event(
                    run_id,
                    "ERROR",
                    "invocation_finished",
                    event_payload,
                    invocation_id,
                )
                emergency_store.finish_invocation(
                    invocation_id,
                    status=status,
                    report=failure_report,
                    processed=completed,
                    skipped=skipped,
                    failed=max(1, failed),
                    elapsed_seconds=failure_report["elapsed_seconds"],
                    images_per_second=0.0,
                    gpu_metrics=gpu_metrics,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    termination_reason=status,
                )
        raise
    finally:
        if status_reporter is not None:
            status_reporter.stop(emit_final=False)
        if gpu_monitor is not None:
            gpu_monitor.stop()
        _shutdown_compute_resources(image_loader, input_queues, workers)
        if client is not None and writer.is_alive():
            try:
                client.call("stop")
            except Exception:
                pass
            writer.join(timeout=10)
        if writer.is_alive():
            writer.terminate()
            writer.join(timeout=5)
