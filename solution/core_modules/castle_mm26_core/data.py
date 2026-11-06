from __future__ import annotations

import json
import os
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from .config import AppConfig
from .io_utils import atomic_write_json, atomic_write_jsonl, sha256_file
from .time_utils import Interval, clip_to_coverage, merge_seconds_to_intervals


ACCEPTED_SOURCES = [
    "Allie",
    "Bao",
    "Bjorn",
    "Cathal",
    "Florian",
    "Kitchen",
    "Klaus",
    "Living1",
    "Living2",
    "Luca",
    "Meeting",
    "Onanong",
    "Reading",
    "Stevan",
    "Tien",
    "Werner",
]


@dataclass(frozen=True)
class CoverageIndex:
    by_day_source: dict[tuple[int, str], list[Interval]]
    rows: list[dict]

    def clip(self, day: int, source: str, start: float, end: float) -> list[Interval]:
        return clip_to_coverage((start, end), self.by_day_source.get((int(day), source), []))


def discover_parquet_files(root: Path | str, *, max_files: int | None = None) -> list[Path]:
    target = Path(root)
    files = sorted(path.resolve() for path in target.rglob("*.parquet") if path.is_file())
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"no parquet files found under {target}")
    return files


def iter_pyarrow_batches(path: Path | str, *, columns: Sequence[str] | None = None, batch_size: int = 8192) -> Iterable[pa.RecordBatch]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=list(columns) if columns else None, use_threads=False):
        yield batch


def read_pylist(path: Path | str, columns: Sequence[str] | None = None) -> list[dict]:
    table = pq.read_table(path, columns=list(columns) if columns else None)
    return table.to_pylist()


def build_coverage_index(config: AppConfig, output_dir: Path) -> CoverageIndex:
    event_root = config.input_path("event_run_root")
    timeline_path = event_root / "atoms" / "canonical_frame_timeline.parquet"
    coverage_path = event_root / "inventory" / "source_hour_coverage.parquet"
    source_rows = read_pylist(coverage_path, columns=["day_num", "source", "source_role", "hour", "frames_available", "tipsv2_available", "text_available"])
    active_source_hours: dict[tuple[int, str, str], dict] = {}
    accepted_sources = set()
    for row in source_rows:
        source = row["source"]
        if source not in ACCEPTED_SOURCES:
            continue
        accepted_sources.add(source)
        key = (int(row["day_num"]), source, str(row.get("hour") or ""))
        active_source_hours[key] = row

    seconds_by_key: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for batch in iter_pyarrow_batches(timeline_path, columns=["day_num", "source", "canonical_second", "video_hour_stem"], batch_size=65536):
        for row in batch.to_pylist():
            source = row["source"]
            day = int(row["day_num"])
            hour = str(row.get("video_hour_stem") or "")
            source_hour = active_source_hours.get((day, source, hour))
            if source in ACCEPTED_SOURCES and source_hour and source_hour.get("frames_available") and source_hour.get("tipsv2_available"):
                seconds_by_key[(day, source, hour)].append(float(row["canonical_second"]))

    intervals_by_day_source: dict[tuple[int, str], list[Interval]] = defaultdict(list)
    segment_index_by_day_source: dict[tuple[int, str], int] = defaultdict(int)
    rows: list[dict] = []
    for key, seconds in sorted(seconds_by_key.items()):
        day, source, hour = key
        day_source_key = (day, source)
        unique_seconds = sorted(set(seconds))
        intervals = merge_seconds_to_intervals(unique_seconds, gap_threshold=1.5)
        source_hour = active_source_hours.get(key) or {}
        role = source_hour.get("source_role") or ""
        text_available = bool(source_hour.get("text_available"))
        for interval in intervals:
            intervals_by_day_source[day_source_key].append(interval)
            segment_index = segment_index_by_day_source[day_source_key]
            segment_index_by_day_source[day_source_key] += 1
            frame_count = sum(1 for second in unique_seconds if interval.start <= second < interval.end)
            rows.append(
                {
                    "day_num": day,
                    "source": source,
                    "coverage_start_second": interval.start,
                    "coverage_end_second": interval.end,
                    "hour": hour,
                    "frames_available": True,
                    "tipsv2_available": True,
                    "text_available": text_available,
                    "canonical_frame_count": int(frame_count),
                    "source_role": role,
                    "coverage_segment_index": segment_index,
                }
            )
    by_day_source = {key: sorted(intervals, key=lambda item: (item.start, item.end)) for key, intervals in intervals_by_day_source.items()}

    output_dir.mkdir(parents=True, exist_ok=True)
    list_path = output_dir / "accepted_source_list.json"
    jsonl_path = output_dir / "accepted_source_interval_index.jsonl"
    parquet_path = output_dir / "accepted_source_interval_index.parquet"
    atomic_write_json(list_path, sorted(accepted_sources))
    atomic_write_jsonl(jsonl_path, rows)
    _write_coverage_parquet(parquet_path, rows)
    manifest = {
        "accepted_source_list": list_path.as_posix(),
        "accepted_source_list_sha256": sha256_file(list_path),
        "accepted_source_interval_jsonl": jsonl_path.as_posix(),
        "accepted_source_interval_jsonl_sha256": sha256_file(jsonl_path),
        "accepted_source_interval_parquet": parquet_path.as_posix(),
        "accepted_source_interval_parquet_sha256": sha256_file(parquet_path),
        "source_count": len(accepted_sources),
        "interval_count": len(rows),
    }
    atomic_write_json(output_dir / "accepted_source_manifest.json", manifest)
    return CoverageIndex(by_day_source=by_day_source, rows=rows)


def _write_coverage_parquet(path: Path, rows: list[dict]) -> None:
    schema = pa.schema(
        [
            ("day_num", pa.int16()),
            ("source", pa.string()),
            ("coverage_start_second", pa.float64()),
            ("coverage_end_second", pa.float64()),
            ("hour", pa.string()),
            ("frames_available", pa.bool_()),
            ("tipsv2_available", pa.bool_()),
            ("text_available", pa.bool_()),
            ("canonical_frame_count", pa.int32()),
            ("source_role", pa.string()),
            ("coverage_segment_index", pa.int32()),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=schema) if rows else pa.Table.from_arrays([pa.array([], type=field.type) for field in schema], schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    pq.write_table(table, tmp, compression="zstd", compression_level=3)
    try:
        tmp.replace(path)
    except PermissionError:
        if os.name != "nt":
            raise
        path.write_bytes(tmp.read_bytes())
        try:
            tmp.unlink()
        except PermissionError:
            pass


class EventStore:
    def __init__(self, root: Path, *, processed_data_root: Path | None = None):
        self.root = Path(root)
        self.processed_data_root = Path(processed_data_root) if processed_data_root is not None else None
        self._frame_by_embedding: dict[str, dict] | None = None
        self._frame_by_id: dict[str, dict] | None = None
        self._frames_by_day_source: dict[tuple[int, str], list[dict]] | None = None
        self._text_by_chunk: dict[str, dict] | None = None
        self._text_by_id: dict[str, dict] | None = None
        self._text_atoms_by_day_source: dict[tuple[int, str], list[dict]] | None = None
        self._event_by_id: dict[str, dict] | None = None
        self._frame_links_by_frame: dict[str, list[dict]] | None = None
        self._text_links_by_text: dict[str, list[dict]] | None = None
        self._child_links_by_parent: dict[str, list[dict]] | None = None
        self._cross_source_links_by_event: dict[str, list[dict]] | None = None

    @classmethod
    def from_config(cls, config: AppConfig) -> "EventStore":
        return cls(config.input_path("event_run_root"), processed_data_root=config.input_path("processed_data_root"))

    @property
    def frame_by_embedding(self) -> dict[str, dict]:
        if self._frame_by_embedding is None:
            rows = read_pylist(self.root / "atoms" / "frame_atoms.parquet", columns=["frame_atom_id", "day_num", "source", "absolute_second", "frame_rel_path", "image_path_ref", "tipsv2_embedding_id"])
            self._frame_by_embedding = {row["tipsv2_embedding_id"]: row for row in rows if row.get("tipsv2_embedding_id")}
        return self._frame_by_embedding

    @property
    def frame_by_id(self) -> dict[str, dict]:
        if self._frame_by_id is None:
            self._frame_by_id = {row["frame_atom_id"]: row for row in self.frame_by_embedding.values() if row.get("frame_atom_id")}
        return self._frame_by_id

    @property
    def frames_by_day_source(self) -> dict[tuple[int, str], list[dict]]:
        if self._frames_by_day_source is None:
            grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
            for row in self.frame_by_embedding.values():
                if row.get("day_num") is None or not row.get("source"):
                    continue
                grouped[(int(row["day_num"]), str(row["source"]))].append(row)
            for rows in grouped.values():
                rows.sort(key=lambda item: (float(item.get("absolute_second") or 0.0), str(item.get("frame_atom_id") or "")))
            self._frames_by_day_source = grouped
        return self._frames_by_day_source

    @property
    def text_by_chunk(self) -> dict[str, dict]:
        if self._text_by_chunk is None:
            rows = read_pylist(self.root / "atoms" / "text_atoms.parquet", columns=["text_atom_id", "embedding_chunk_id", "day_num", "source", "absolute_start_second", "absolute_end_second", "union_time_intervals", "time_alignment_status", "text_preview", "quality_flags"])
            self._text_by_chunk = {row["embedding_chunk_id"]: row for row in rows if row.get("embedding_chunk_id")}
        return self._text_by_chunk

    @property
    def text_by_id(self) -> dict[str, dict]:
        if self._text_by_id is None:
            self._text_by_id = {row["text_atom_id"]: row for row in self.text_by_chunk.values() if row.get("text_atom_id")}
        return self._text_by_id

    @property
    def text_atoms_by_day_source(self) -> dict[tuple[int, str], list[dict]]:
        if self._text_atoms_by_day_source is None:
            grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
            for row in self.text_by_id.values():
                if row.get("day_num") is None or not row.get("source"):
                    continue
                grouped[(int(row["day_num"]), str(row["source"]))].append(row)
            for rows in grouped.values():
                rows.sort(key=lambda item: (float(item.get("absolute_start_second") or 0.0), float(item.get("absolute_end_second") or 0.0), str(item.get("text_atom_id") or "")))
            self._text_atoms_by_day_source = grouped
        return self._text_atoms_by_day_source

    def iter_cleaned_transcript_rows(self, *, sources: Sequence[str] | None = None) -> Iterable[dict]:
        if self.processed_data_root is None:
            return
        transcript_root = self.processed_data_root / "cleaned_transcripts"
        if not transcript_root.exists():
            return
        if sources:
            transcript_files = [transcript_root / f"{source}.jsonl" for source in dict.fromkeys(str(item) for item in sources if str(item).strip())]
            transcript_files = [path for path in transcript_files if path.is_file()]
        else:
            transcript_files = sorted(path for path in transcript_root.glob("*.jsonl") if path.is_file())
        for path in transcript_files:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    text = line.strip()
                    if not text:
                        continue
                    yield json.loads(text)

    @property
    def event_by_id(self) -> dict[str, dict]:
        if self._event_by_id is None:
            rows = []
            for rel in ("segments/l1_micro_events.parquet", "segments/l2_source_events.parquet", "segments/l3_global_events.parquet", "segments/l4_activity_episodes.parquet", "segments/l5_sessions.parquet"):
                rows.extend(read_pylist(self.root / rel, columns=None))
            self._event_by_id = {row["event_id"]: row for row in rows}
        return self._event_by_id

    @property
    def frame_links_by_frame(self) -> dict[str, list[dict]]:
        if self._frame_links_by_frame is None:
            grouped: dict[str, list[dict]] = defaultdict(list)
            rows = read_pylist(self.root / "links" / "event_frame_links.parquet", columns=["event_id", "frame_atom_id", "link_type", "link_role", "score", "selection_score"])
            for row in rows:
                grouped[row["frame_atom_id"]].append(row)
            self._frame_links_by_frame = grouped
        return self._frame_links_by_frame

    @property
    def text_links_by_text(self) -> dict[str, list[dict]]:
        if self._text_links_by_text is None:
            grouped: dict[str, list[dict]] = defaultdict(list)
            rows = read_pylist(self.root / "links" / "event_text_links.parquet", columns=["event_id", "text_atom_id", "link_type", "link_role", "score", "selection_score"])
            for row in rows:
                grouped[row["text_atom_id"]].append(row)
            self._text_links_by_text = grouped
        return self._text_links_by_text

    @property
    def child_links_by_parent(self) -> dict[str, list[dict]]:
        if self._child_links_by_parent is None:
            grouped: dict[str, list[dict]] = defaultdict(list)
            rows = read_pylist(self.root / "links" / "event_parent_child_links.parquet", columns=["parent_event_id", "child_event_id", "link_type", "link_role"])
            for row in rows:
                grouped[row["parent_event_id"]].append(row)
            self._child_links_by_parent = grouped
        return self._child_links_by_parent

    @property
    def cross_source_links_by_event(self) -> dict[str, list[dict]]:
        if self._cross_source_links_by_event is None:
            grouped: dict[str, list[dict]] = defaultdict(list)
            path = self.root / "links" / "cross_source_links.parquet"
            if path.exists():
                rows = read_pylist(path, columns=["event_id", "target_id", "link_type", "link_role", "score", "selection_score"])
                for row in rows:
                    event_id = str(row.get("event_id") or "")
                    target_id = str(row.get("target_id") or "")
                    if event_id:
                        grouped[event_id].append(row)
                    if target_id:
                        reverse = dict(row)
                        reverse["event_id"] = target_id
                        reverse["target_id"] = event_id
                        grouped[target_id].append(reverse)
            self._cross_source_links_by_event = grouped
        return self._cross_source_links_by_event

    def source_role(self, source: str) -> str:
        path = self.root / "inventory" / "source_hour_coverage.parquet"
        for row in read_pylist(path, columns=["source", "source_role"]):
            if row["source"] == source:
                return str(row.get("source_role") or "")
        return ""


def resolve_frame_path(config: AppConfig, frame_rel_path: str, image_path_ref: str | None = None) -> Path:
    processed_root = config.input_path("processed_data_root")
    root_text = processed_root.as_posix()
    rel = (image_path_ref or frame_rel_path).replace("\\", "/")
    if rel.startswith(root_text):
        return Path(rel)
    if rel.startswith("download_pipeline/processed_data/"):
        return config.repo_root / rel
    if rel.startswith("main/"):
        return processed_root / rel
    return processed_root / rel


def filter_text_atom_ids_for_source(text_atom_ids: Sequence[object], day: int, source: str, event_store: EventStore | None) -> list[str]:
    ids = [str(item) for item in text_atom_ids if str(item)]
    if event_store is None:
        return sorted(dict.fromkeys(ids))
    result: list[str] = []
    seen = set()
    for text_id in ids:
        if text_id in seen:
            continue
        text_atom = event_store.text_by_id.get(text_id)
        if text_atom is None:
            continue
        try:
            atom_day = int(text_atom.get("day_num") or -1)
        except (TypeError, ValueError):
            continue
        if atom_day != int(day) or str(text_atom.get("source") or "") != str(source):
            continue
        result.append(text_id)
        seen.add(text_id)
    return sorted(result)


def text_atom_ids_for_source_interval(day: int, source: str, start_second: float, end_second: float, event_store: EventStore | None, *, max_count: int = 8) -> list[str]:
    if event_store is None:
        return []
    grouped = getattr(event_store, "text_atoms_by_day_source", None)
    if isinstance(grouped, dict):
        atoms = grouped.get((int(day), str(source)), [])
    else:
        atoms = [
            row
            for row in event_store.text_by_id.values()
            if int(row.get("day_num") or -1) == int(day) and str(row.get("source") or "") == str(source)
        ]
    ranked = []
    start = float(start_second)
    end = float(end_second)
    for atom in atoms:
        if str(atom.get("time_alignment_status") or "").lower() == "invalid":
            continue
        atom_id = str(atom.get("text_atom_id") or "")
        if not atom_id:
            continue
        try:
            atom_start = float(atom.get("absolute_start_second"))
            atom_end = float(atom.get("absolute_end_second"))
        except (TypeError, ValueError):
            continue
        overlap = min(end, atom_end) - max(start, atom_start)
        if overlap <= 0:
            continue
        ranked.append((-overlap, atom_start, atom_end, atom_id))
    ranked.sort()
    return sorted(item[3] for item in ranked[:max_count])
