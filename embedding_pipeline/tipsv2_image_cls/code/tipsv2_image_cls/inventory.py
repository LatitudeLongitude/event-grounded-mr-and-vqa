from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Sequence, Tuple


DAY_PATTERN = re.compile(r"^day([1-4])$")
VALID_STATUSES = {"raw_deleted_verified", "processed_verified"}


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class HourTask:
    day: str
    day_number: int
    source: str
    hour: str
    numeric_hour: int
    frame_count: int
    manifest_path: Path

    @property
    def key(self) -> str:
        return f"{self.day}/{self.source}/{self.hour}"


@dataclass(frozen=True)
class ExcludedTask:
    row_number: int
    reason: str
    day: str
    source: str
    hour: str


@dataclass(frozen=True)
class FrameRecord:
    frame_index: int
    timestamp_second: float
    timestamp_hms: str
    frame_rel_path: str
    path: Path


@dataclass(frozen=True)
class Inventory:
    tasks: Tuple[HourTask, ...]
    excluded: Tuple[ExcludedTask, ...]
    summary_rows: int
    main_tasks: int
    auxiliary_tasks: int
    expected_frames: int
    by_day_source: Dict[str, int]

    def as_dict(self) -> dict:
        return {
            "summary_rows": self.summary_rows,
            "main_tasks": self.main_tasks,
            "auxiliary_tasks": self.auxiliary_tasks,
            "expected_frames": self.expected_frames,
            "by_day_source": dict(sorted(self.by_day_source.items())),
            "excluded": [item.__dict__ for item in self.excluded],
        }


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def discover_tasks(data_root: Path | str, summary_path: Optional[Path | str] = None) -> Inventory:
    root = Path(data_root).resolve()
    summary = Path(summary_path).resolve() if summary_path else root / "_manifests" / "video_processing_summary.csv"
    tasks: List[HourTask] = []
    excluded: List[ExcludedTask] = []
    by_day_source: Dict[str, int] = {}
    auxiliary = 0
    with summary.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row_number, row in enumerate(rows, start=2):
        day = (row.get("day") or "").strip()
        source = (row.get("source") or "").strip()
        hour_raw = (row.get("video_hour_stem") or "").strip()
        day_match = DAY_PATTERN.match(day)
        if day_match is None:
            auxiliary += 1
            excluded.append(ExcludedTask(row_number, "auxiliary_or_missing_day", day, source, hour_raw))
            continue
        if not source:
            excluded.append(ExcludedTask(row_number, "missing_source", day, source, hour_raw))
            continue
        if source in {".", ".."} or "/" in source or "\\" in source:
            excluded.append(ExcludedTask(row_number, "unsafe_source", day, source, hour_raw))
            continue
        if not hour_raw.isdigit() or not 0 <= int(hour_raw) <= 23:
            excluded.append(ExcludedTask(row_number, "invalid_hour", day, source, hour_raw))
            continue
        if (row.get("status") or "").strip() not in VALID_STATUSES:
            excluded.append(ExcludedTask(row_number, "invalid_status", day, source, hour_raw))
            continue
        if _truthy(row.get("frame_resize_enabled") or "") and (row.get("frame_resize_status") or "").strip() != "completed":
            excluded.append(ExcludedTask(row_number, "resize_incomplete", day, source, hour_raw))
            continue
        hour = f"{int(hour_raw):02d}"
        manifest = root / "main" / day / source / "video" / hour / "frame_manifest.csv"
        resolved_manifest = manifest.resolve()
        try:
            resolved_manifest.relative_to(root)
        except ValueError:
            excluded.append(ExcludedTask(row_number, "unsafe_manifest", day, source, hour))
            continue
        if not resolved_manifest.is_file():
            excluded.append(ExcludedTask(row_number, "missing_manifest", day, source, hour))
            continue
        try:
            frame_count = int(row.get("frame_count") or "")
        except ValueError:
            excluded.append(ExcludedTask(row_number, "invalid_frame_count", day, source, hour))
            continue
        task = HourTask(day, int(day_match.group(1)), source, hour, int(hour), frame_count, resolved_manifest)
        tasks.append(task)
        key = f"{day}/{source}"
        by_day_source[key] = by_day_source.get(key, 0) + 1
    tasks.sort(key=lambda item: (item.day_number, item.source, item.numeric_hour))
    return Inventory(
        tasks=tuple(tasks),
        excluded=tuple(excluded),
        summary_rows=len(rows),
        main_tasks=len(tasks),
        auxiliary_tasks=auxiliary,
        expected_frames=sum(task.frame_count for task in tasks),
        by_day_source=by_day_source,
    )


def _portable_relative_path(value: str) -> PurePosixPath:
    normalized = value.strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise ManifestError(f"frame_rel_path is outside data_root: {value}")
    return path


def load_frame_manifest(task: HourTask, data_root: Path | str) -> Tuple[FrameRecord, ...]:
    root = Path(data_root).resolve()
    with task.manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != task.frame_count:
        raise ManifestError(f"Manifest row count {len(rows)} does not match expected {task.frame_count}")
    records: List[FrameRecord] = []
    previous: Optional[int] = None
    for row_number, row in enumerate(rows, start=2):
        try:
            index = int(row["frame_index"])
            timestamp = float(row["timestamp_second"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError(f"Invalid numeric value at manifest row {row_number}") from exc
        if previous is not None and index <= previous:
            raise ManifestError("frame_index must be unique and strictly increasing")
        previous = index
        relative = _portable_relative_path(row.get("frame_rel_path") or "")
        resolved = (root / Path(*relative.parts)).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ManifestError(f"frame_rel_path is outside data_root: {relative}") from exc
        if not resolved.is_file():
            raise ManifestError(f"Frame file does not exist: {relative}")
        records.append(
            FrameRecord(
                frame_index=index,
                timestamp_second=timestamp,
                timestamp_hms=(row.get("timestamp_hms") or "").strip(),
                frame_rel_path=relative.as_posix(),
                path=resolved,
            )
        )
    return tuple(records)


def filter_tasks(
    tasks: Sequence[HourTask],
    *,
    day: Optional[str] = None,
    source: Optional[str] = None,
    hour: Optional[str] = None,
    max_hours: Optional[int] = None,
) -> List[HourTask]:
    normalized_hour = f"{int(hour):02d}" if hour is not None and str(hour).isdigit() else hour
    selected = [
        task
        for task in tasks
        if (day is None or task.day == day)
        and (source is None or task.source == source)
        and (normalized_hour is None or task.hour == normalized_hour)
    ]
    return selected[:max_hours] if max_hours is not None else selected
