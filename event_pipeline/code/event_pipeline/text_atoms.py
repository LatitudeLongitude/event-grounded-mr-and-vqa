from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

from .config import AppConfig
from .filters import source_matches
from .ids import text_atom_id
from .intervals import aligned_duration_sec, union_intervals
from .manifest import file_record, params_hash, write_stage_manifest
from .parquet_io import AtomicParquetWriter, iter_pylist_rows
from .paths import to_ref
from .run_state import stage_complete, stage_paths, update_run_state
from .schema import TEXT_ATOM_FRAGMENTS, TEXT_ATOMS
from .time_utils import day_label, hour_to_second, normalize_day, normalize_hour

STAGE = "text_atoms"


def _safe_float(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _safe_hour(value) -> Optional[str]:
    try:
        return normalize_hour(value)
    except (TypeError, ValueError):
        return None


def _embedding_map(root: Path, config: AppConfig, day: Optional[str], source: Optional[str]) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for path in sorted(root.glob("day*/*.parquet")):
        found_day = day_label(path.parts[-2])
        found_source = path.stem
        if day and found_day != day_label(day):
            continue
        if not source_matches(found_source, source):
            continue
        for row in iter_pylist_rows(path, columns=["embedding_chunk_id", "embedding_id"], batch_size=config.run.batch_size):
            result[str(row["embedding_chunk_id"])] = (str(row["embedding_id"]), to_ref(path, config.project_root))
    return result


def _valid_fragment(chunk: dict) -> tuple[bool, Optional[float], Optional[float], list[str]]:
    flags = list(chunk.get("quality_flags") or [])
    start = chunk.get("start_time")
    end = chunk.get("end_time")
    hour = _safe_hour(chunk.get("hour"))
    if hour is None:
        return False, None, None, sorted(set(flags + ["missing_or_invalid_hour"]))
    start_f = _safe_float(start)
    end_f = _safe_float(end)
    if start_f is None or end_f is None:
        return False, None, None, sorted(set(flags + ["missing_or_nonfinite_time"]))
    if start_f >= end_f:
        return False, None, None, sorted(set(flags + ["start_time_gte_end_time"]))
    if start_f < 0:
        return False, None, None, sorted(set(flags + ["start_time_negative"]))
    if end_f > 3600:
        return False, None, None, sorted(set(flags + ["end_time_gt_3600"]))
    base = hour_to_second(hour)
    return True, base + start_f, base + end_f, flags


def _build_rows(obj: dict, qwen_map: dict[str, tuple[str, str]], bge_map: dict[str, tuple[str, str]], param_hash: str, input_manifest_ref: str) -> tuple[dict, list[dict]]:
    chunk_id = str(obj["embedding_chunk_id"])
    atom_id = text_atom_id(chunk_id)
    fragments = []
    fragment_rows = []
    valid_intervals: list[tuple[float, float]] = []
    any_invalid = False
    estimated_time = "estimated_time_range" in set(obj.get("quality_flags") or [])
    for chunk in obj.get("source_chunks") or []:
        ok, abs_start, abs_end, flags = _valid_fragment(chunk)
        any_invalid = any_invalid or not ok
        estimated_time = estimated_time or "estimated_time_range" in set(flags)
        if ok and abs_start is not None and abs_end is not None:
            valid_intervals.append((abs_start, abs_end))
        frag = {
            "source_chunk_id": chunk.get("chunk_id"),
            "hour": _safe_hour(chunk.get("hour")),
            "absolute_start_second": abs_start,
            "absolute_end_second": abs_end,
            "start_time": chunk.get("start_time"),
            "end_time": chunk.get("end_time"),
            "timestamp_valid": bool(chunk.get("timestamp_valid")),
            "is_valid_interval": ok,
            "quality_flags": flags,
        }
        fragments.append(frag)
        fragment_rows.append(
            {
                "text_atom_id": atom_id,
                "source_chunk_id": str(chunk.get("chunk_id")),
                "day": day_label(obj["day"]),
                "source": str(obj["view"]),
                "hour": frag["hour"],
                "absolute_start_second": abs_start,
                "absolute_end_second": abs_end,
                "start_time": _safe_float(chunk.get("start_time")),
                "end_time": _safe_float(chunk.get("end_time")),
                "timestamp_valid": bool(chunk.get("timestamp_valid")),
                "is_valid_interval": ok,
                "quality_flags": flags,
                "created_by": STAGE,
                "params_hash": param_hash,
                "input_manifest_ref": input_manifest_ref,
            }
        )
    unioned = union_intervals(valid_intervals)
    if not unioned:
        status = "invalid"
        abs_start = abs_end = duration = None
    else:
        status = "partial" if any_invalid or estimated_time else "exact"
        abs_start = unioned[0][0]
        abs_end = unioned[-1][1]
        duration = aligned_duration_sec(unioned)
    qwen_id, qwen_ref = qwen_map.get(chunk_id, (None, None))
    bge_id, bge_ref = bge_map.get(chunk_id, (None, None))
    quality_flags = sorted(set(obj.get("quality_flags") or []) | ({"missing_qwen3_embedding"} if qwen_id is None else set()) | ({"missing_bgem3_embedding"} if bge_id is None else set()))
    atom = {
        "text_atom_id": atom_id,
        "embedding_chunk_id": chunk_id,
        "day": day_label(obj["day"]),
        "day_num": normalize_day(obj["day"]),
        "source": str(obj["view"]),
        "start_hour": _safe_hour(obj.get("start_hour")),
        "end_hour": _safe_hour(obj.get("end_hour")),
        "hours": [hour for hour in (_safe_hour(hour) for hour in obj.get("hours") or []) if hour is not None],
        "raw_start_time": _safe_float(obj.get("start_time")),
        "raw_end_time": _safe_float(obj.get("end_time")),
        "time_fragments": json.dumps(fragments, ensure_ascii=False, separators=(",", ":")),
        "union_time_intervals": json.dumps(unioned, ensure_ascii=False, separators=(",", ":")),
        "absolute_start_second": abs_start,
        "absolute_end_second": abs_end,
        "aligned_duration_sec": duration,
        "time_alignment_status": status,
        "qwen3_embedding_id": qwen_id,
        "qwen3_parquet_ref": qwen_ref,
        "bgem3_embedding_id": bge_id,
        "bgem3_parquet_ref": bge_ref,
        "text_preview": str(obj.get("text_for_display") or "")[:500],
        "token_estimate": int(obj.get("token_estimate") or 0),
        "quality_flags": quality_flags,
        "created_by": STAGE,
        "params_hash": param_hash,
        "input_manifest_ref": input_manifest_ref,
    }
    return atom, fragment_rows


def build_text_atoms(config: AppConfig, run_root: Path, *, output_mode: str, run_id: Optional[str], day: Optional[str] = None, source: Optional[str] = None, resume: bool = False) -> dict:
    paths = stage_paths(run_root)
    atoms_path = run_root / "atoms" / "text_atoms.parquet"
    fragments_path = run_root / "atoms" / "text_atom_time_fragments.parquet"
    params = {"stage": STAGE, "output_mode": output_mode, "run_id": run_id, "day": day, "source": source}
    if resume and stage_complete(paths.stage_manifest(STAGE), params=params, output_paths=[atoms_path, fragments_path]):
        return {"stage": STAGE, "skipped": True}

    chunks_path = config.input_roots.embed_chunks_root / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(chunks_path)
    param_hash = params_hash(params)
    input_manifest_ref = to_ref(run_root / "manifests" / "input_data_manifest.json", config.project_root)
    qwen_map = _embedding_map(config.input_roots.qwen3_root, config, day, source)
    bge_map = _embedding_map(config.input_roots.bgem3_root, config, day, source)

    with AtomicParquetWriter(atoms_path, TEXT_ATOMS) as atom_writer, AtomicParquetWriter(fragments_path, TEXT_ATOM_FRAGMENTS) as fragment_writer:
        atom_batch: list[dict] = []
        fragment_batch: list[dict] = []
        with chunks_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                obj = json.loads(line)
                found_day = day_label(obj["day"])
                found_source = str(obj["view"])
                if day and found_day != day_label(day):
                    continue
                if not source_matches(found_source, source):
                    continue
                atom, fragments = _build_rows(obj, qwen_map, bge_map, param_hash, input_manifest_ref)
                atom_batch.append(atom)
                fragment_batch.extend(fragments)
                if len(atom_batch) >= 4096:
                    atom_writer.write_rows(atom_batch)
                    fragment_writer.write_rows(fragment_batch)
                    atom_batch.clear()
                    fragment_batch.clear()
        atom_writer.write_rows(atom_batch)
        fragment_writer.write_rows(fragment_batch)

    manifest = write_stage_manifest(
        paths.stage_manifest(STAGE),
        stage=STAGE,
        params=params,
        inputs=[file_record(chunks_path, config.project_root), file_record(run_root / "manifests" / "input_data_manifest.json", config.project_root)],
        outputs=[
            {**file_record(atoms_path, config.project_root), "schema": str(TEXT_ATOMS)},
            {**file_record(fragments_path, config.project_root), "schema": str(TEXT_ATOM_FRAGMENTS)},
        ],
        schema_names=["text_atoms", "text_atom_time_fragments"],
        project_root=config.project_root,
    )
    update_run_state(run_root, STAGE, to_ref(paths.stage_manifest(STAGE), config.project_root))
    return {"stage": STAGE, "manifest": manifest}
