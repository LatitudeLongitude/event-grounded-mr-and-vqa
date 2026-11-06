from __future__ import annotations

import csv
import io
import subprocess
import threading
from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple


class GPUProtectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class GPUProcess:
    pid: int
    name: str
    used_mib: int


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    total_mib: int
    used_mib: int
    free_mib: int
    utilization: int
    processes: Tuple[GPUProcess, ...]
    uuid: str = ""


def _run_nvidia_smi(arguments: Sequence[str]) -> str:
    result = subprocess.run(
        ["nvidia-smi", *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout


def _rows(text: str):
    return [tuple(value.strip() for value in row) for row in csv.reader(io.StringIO(text)) if any(value.strip() for value in row)]


def _query_text(label: str, arguments: Sequence[str]) -> str:
    try:
        return _run_nvidia_smi(arguments)
    except Exception as exc:
        raise GPUProtectionError(f"GPU state unknown: {label} query failed: {type(exc).__name__}: {exc}") from exc


def _parse_non_negative_int(value: str, field: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{field} cannot be negative")
    return parsed


def query_gpus() -> Tuple[GPUInfo, ...]:
    gpu_text = _query_text(
        "GPU-list",
        [
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
    )
    process_text = _query_text(
        "compute-process",
        [
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
    )
    try:
        gpu_rows = _rows(gpu_text)
        if not gpu_rows:
            raise ValueError("GPU list is empty")
        parsed_gpus = []
        seen_indices = set()
        seen_uuids = set()
        for row in gpu_rows:
            if len(row) != 7:
                raise ValueError(f"GPU row has {len(row)} fields instead of 7")
            index, gpu_uuid, name, total, used, free, utilization = row
            if not gpu_uuid or not name:
                raise ValueError("GPU UUID and name are required")
            parsed_index = _parse_non_negative_int(index, "GPU index")
            parsed_utilization = _parse_non_negative_int(utilization, "GPU utilization")
            if parsed_utilization > 100:
                raise ValueError("GPU utilization exceeds 100")
            if parsed_index in seen_indices or gpu_uuid in seen_uuids:
                raise ValueError("GPU index and UUID must be unique")
            seen_indices.add(parsed_index)
            seen_uuids.add(gpu_uuid)
            parsed_gpus.append(
                {
                    "index": parsed_index,
                    "uuid": gpu_uuid,
                    "name": name,
                    "total_mib": _parse_non_negative_int(total, "total memory"),
                    "used_mib": _parse_non_negative_int(used, "used memory"),
                    "free_mib": _parse_non_negative_int(free, "free memory"),
                    "utilization": parsed_utilization,
                }
            )
        processes: Dict[str, list[GPUProcess]] = {gpu_uuid: [] for gpu_uuid in seen_uuids}
        for row in _rows(process_text):
            if len(row) != 4:
                raise ValueError(f"Compute-process row has {len(row)} fields instead of 4")
            gpu_uuid, pid, name, used_mib = row
            if gpu_uuid not in seen_uuids:
                raise ValueError(f"Compute-process GPU UUID is not in GPU list: {gpu_uuid}")
            if not name:
                raise ValueError("Compute-process name is required")
            parsed_pid = int(pid)
            if parsed_pid <= 0:
                raise ValueError("Compute-process PID must be positive")
            processes[gpu_uuid].append(
                GPUProcess(parsed_pid, name, _parse_non_negative_int(used_mib, "compute-process memory"))
            )
        result = [
            GPUInfo(**gpu, processes=tuple(processes[gpu["uuid"]]))
            for gpu in parsed_gpus
        ]
        return tuple(sorted(result, key=lambda item: item.index))
    except Exception as exc:
        raise GPUProtectionError(f"GPU state unknown: nvidia-smi output is invalid: {type(exc).__name__}: {exc}") from exc


def assert_gpus_idle(
    gpus: Iterable[GPUInfo],
    target_ids: Sequence[int],
    *,
    max_utilization: int,
    min_free_mib: int,
) -> Tuple[GPUInfo, ...]:
    by_index = {gpu.index: gpu for gpu in gpus}
    selected = []
    errors = []
    for gpu_id in target_ids:
        gpu = by_index.get(gpu_id)
        if gpu is None:
            errors.append(f"GPU {gpu_id} is not present")
            continue
        if gpu.processes:
            details = ", ".join(f"PID {item.pid} {item.name} ({item.used_mib} MiB)" for item in gpu.processes)
            errors.append(f"GPU {gpu_id} has an existing compute process: {details}")
        if gpu.utilization > max_utilization:
            errors.append(f"GPU {gpu_id} utilization {gpu.utilization}% exceeds {max_utilization}%")
        if gpu.free_mib < min_free_mib:
            errors.append(f"GPU {gpu_id} free memory {gpu.free_mib} MiB is below {min_free_mib} MiB")
        selected.append(gpu)
    if errors:
        raise GPUProtectionError("; ".join(errors))
    return tuple(selected)


def validate_gpu_policy(target_ids: Sequence[int], max_memory_fraction_per_gpu: float | None) -> None:
    if not target_ids:
        raise GPUProtectionError("At least one GPU ID is required")
    if any(gpu_id < 0 for gpu_id in target_ids):
        raise GPUProtectionError("GPU IDs cannot be negative")
    if len(target_ids) != len(set(target_ids)):
        raise GPUProtectionError("GPU IDs must be unique")
    if len(target_ids) > 3:
        raise GPUProtectionError("At most three GPUs are supported")
    if len(target_ids) == 3:
        if max_memory_fraction_per_gpu is None or not 0 < max_memory_fraction_per_gpu <= 0.8:
            raise GPUProtectionError("Three-GPU mode requires max_memory_fraction_per_gpu in (0, 0.8]")


def assert_no_foreign_compute_processes(
    gpus: Iterable[GPUInfo], target_ids: Sequence[int], allowed_pids: Sequence[int]
) -> None:
    allowed = set(allowed_pids)
    targets = set(target_ids)
    foreign = []
    for gpu in gpus:
        if gpu.index not in targets:
            continue
        for process in gpu.processes:
            if process.pid not in allowed:
                foreign.append(f"GPU {gpu.index} PID {process.pid} {process.name} ({process.used_mib} MiB)")
    if foreign:
        raise GPUProtectionError("Foreign compute process appeared after model loading: " + "; ".join(foreign))


def gpu_report(gpus: Iterable[GPUInfo]) -> list[dict]:
    return [
        {
            "index": gpu.index,
            "uuid": gpu.uuid,
            "name": gpu.name,
            "total_mib": gpu.total_mib,
            "used_mib": gpu.used_mib,
            "free_mib": gpu.free_mib,
            "utilization": gpu.utilization,
            "processes": [process.__dict__ for process in gpu.processes],
        }
        for gpu in gpus
    ]


class GPUMonitor:
    def __init__(self, target_ids: Sequence[int], interval_seconds: float = 1.0):
        self.target_ids = set(target_ids)
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._samples = 0
        self._failure = None
        self._peak_utilization: Dict[int, int] = {gpu_id: 0 for gpu_id in target_ids}
        self._peak_used_mib: Dict[int, int] = {gpu_id: 0 for gpu_id in target_ids}
        self._min_free_mib: Dict[int, int] = {gpu_id: 2**31 - 1 for gpu_id in target_ids}
        self._current_utilization: Dict[int, int | None] = {gpu_id: None for gpu_id in target_ids}
        self._current_used_mib: Dict[int, int | None] = {gpu_id: None for gpu_id in target_ids}
        self._current_free_mib: Dict[int, int | None] = {gpu_id: None for gpu_id in target_ids}

    def start(self) -> "GPUMonitor":
        self._thread = threading.Thread(target=self._loop, name="nvidia-smi-monitor", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                gpus = query_gpus()
                with self._lock:
                    self._samples += 1
                    for gpu in gpus:
                        if gpu.index not in self.target_ids:
                            continue
                        self._peak_utilization[gpu.index] = max(self._peak_utilization[gpu.index], gpu.utilization)
                        self._peak_used_mib[gpu.index] = max(self._peak_used_mib[gpu.index], gpu.used_mib)
                        self._min_free_mib[gpu.index] = min(self._min_free_mib[gpu.index], gpu.free_mib)
                        self._current_utilization[gpu.index] = gpu.utilization
                        self._current_used_mib[gpu.index] = gpu.used_mib
                        self._current_free_mib[gpu.index] = gpu.free_mib
            except Exception as exc:
                with self._lock:
                    self._failure = exc
                self._stop.set()
                break
            self._stop.wait(self.interval_seconds)

    def raise_if_failed(self) -> None:
        with self._lock:
            failure = self._failure
        if failure is not None:
            if isinstance(failure, GPUProtectionError):
                raise failure
            raise GPUProtectionError(f"GPU state unknown: monitor query failed: {type(failure).__name__}: {failure}") from failure

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "samples": self._samples,
                "peak_utilization_percent": dict(self._peak_utilization),
                "peak_used_memory_mib": dict(self._peak_used_mib),
                "minimum_free_memory_mib": {
                    key: (None if value == 2**31 - 1 else value) for key, value in self._min_free_mib.items()
                },
                "current_utilization_percent": dict(self._current_utilization),
                "current_used_memory_mib": dict(self._current_used_mib),
                "current_free_memory_mib": dict(self._current_free_mib),
                "query_error": None if self._failure is None else f"{type(self._failure).__name__}: {self._failure}",
            }

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 2))
        return self.snapshot()
