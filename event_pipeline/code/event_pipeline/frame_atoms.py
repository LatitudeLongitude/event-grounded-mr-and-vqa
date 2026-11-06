from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .config import AppConfig
from .filters import source_matches
from .ids import frame_atom_id
from .manifest import file_record, params_hash, write_stage_manifest
from .parquet_io import AtomicParquetWriter, read_pylist
from .paths import to_ref
from .run_state import stage_complete, stage_paths, update_run_state
from .schema import CANONICAL_FRAME_TIMELINE, FRAME_ATOMS
from .time_utils import canonical_source_name, day_label, normalize_day, normalize_hour

STAGE = "frame_atoms"


def _norm_key(row: dict) -> tuple[int, str, str, int]:
    return (
        normalize_day(row["day"]),
        canonical_source_name(row["source"]),
        normalize_hour(row["video_hour_stem"]),
        int(row["frame_index"]),
    )


def _read_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if str(row.get("frame_rel_path", "")).replace("\\", "/").startswith("main/")]


def _read_tips(path: Path) -> list[dict]:
    return read_pylist(
        path,
        columns=[
            "embedding_id",
            "day",
            "source",
            "video_hour_stem",
            "frame_index",
            "timestamp_second",
            "absolute_second",
            "timestamp_hms",
            "frame_rel_path",
        ],
    )


def _discover(config: AppConfig, day: Optional[str], source: Optional[str]) -> list[tuple[str, str, str, Path, Path]]:
    found = []
    for manifest_path in sorted(config.input_roots.processed_data_root.glob("day*/*/video/*/frame_manifest.csv")):
        try:
            found_day = day_label(manifest_path.parts[-5])
            found_source = manifest_path.parts[-4]
            hour = normalize_hour(manifest_path.parts[-2])
        except Exception:
            continue
        if day and found_day != day_label(day):
            continue
        if not source_matches(found_source, source):
            continue
        tips_path = config.input_roots.tipsv2_root / found_day / found_source / f"{hour}.parquet"
        found.append((found_day, found_source, hour, manifest_path, tips_path))
    return found


def _joined_hour_rows(config: AppConfig, manifest_path: Path, tips_path: Path, param_hash: str, input_manifest_ref: str) -> list[dict]:
    if not tips_path.exists():
        manifest_rows = _read_manifest(manifest_path)
        if manifest_rows:
            raise FileNotFoundError(f"TIPSv2 shard missing for non-empty frame manifest: manifest={manifest_path} tips={tips_path}")
        return []
    manifest_by_key: dict[tuple[int, str, str, int], dict] = {}
    tips_by_key: dict[tuple[int, str, str, int], dict] = {}
    for row in _read_manifest(manifest_path):
        key = _norm_key(row)
        if key in manifest_by_key:
            raise ValueError(f"duplicate normalized manifest key {key} in {manifest_path}")
        manifest_by_key[key] = row
    for row in _read_tips(tips_path):
        key = _norm_key(row)
        if key in tips_by_key:
            raise ValueError(f"duplicate normalized TIPSv2 key {key} in {tips_path}")
        tips_by_key[key] = row
    missing = set(manifest_by_key) - set(tips_by_key)
    extra = set(tips_by_key) - set(manifest_by_key)
    if missing or extra:
        raise ValueError(f"frame join mismatch for {manifest_path}: missing={len(missing)} extra={len(extra)}")

    rows = []
    for key in sorted(manifest_by_key):
        day_num, source, hour, frame_index = key
        manifest = manifest_by_key[key]
        tips = tips_by_key[key]
        flags: list[str] = []
        timestamp_second = float(tips["timestamp_second"])
        absolute_second = float(tips["absolute_second"])
        raw_abs = int(hour) * 3600 + timestamp_second
        manifest_rel = str(manifest["frame_rel_path"]).replace("\\", "/")
        tips_rel = str(tips["frame_rel_path"]).replace("\\", "/")
        if manifest_rel != tips_rel:
            flags.append("frame_rel_path_mismatch")
        if abs(float(manifest["timestamp_second"]) - timestamp_second) > 1e-4:
            flags.append("timestamp_second_mismatch")
        if str(manifest["timestamp_hms"]) != str(tips["timestamp_hms"]):
            flags.append("timestamp_hms_mismatch")
        in_hour = 0.0 <= timestamp_second < 3600.0
        time_consistent = abs(raw_abs - absolute_second) <= 1e-6
        if not in_hour:
            flags.append("out_of_hour_timestamp")
        if not time_consistent:
            flags.append("absolute_second_mismatch")
        processed_parent = config.input_roots.processed_data_root.parent.resolve()
        image_path = (processed_parent / tips_rel).resolve()
        fatal_flags = [
            flag
            for flag in flags
            if flag in {"frame_rel_path_mismatch", "timestamp_second_mismatch", "timestamp_hms_mismatch"}
        ]
        if not tips_rel.startswith("main/") or not image_path.is_relative_to(processed_parent):
            fatal_flags.append("frame_rel_path_outside_processed_data_root")
        if not image_path.exists():
            fatal_flags.append("image_path_missing_at_build")
        if fatal_flags:
            raise ValueError(f"frame join quality failure for {manifest_path} key={key}: {sorted(set(fatal_flags))}")
        rows.append(
            {
                "frame_atom_id": frame_atom_id(day_num, source, hour, frame_index),
                "day": f"day{day_num}",
                "day_num": day_num,
                "source": source,
                "video_hour_stem": hour,
                "frame_index": frame_index,
                "timestamp_second": timestamp_second,
                "absolute_second": absolute_second,
                "timestamp_hms": str(tips["timestamp_hms"]),
                "frame_rel_path": tips_rel,
                "image_path_ref": tips_rel,
                "tipsv2_embedding_id": str(tips["embedding_id"]),
                "tipsv2_parquet_ref": to_ref(tips_path, config.project_root),
                "tipsv2_row_key": f"day{day_num}/{source}/{hour}/{frame_index}",
                "manifest_ref": to_ref(manifest_path, config.project_root),
                "path_exists_at_build": image_path.exists(),
                "embedding_exists_at_build": True,
                "is_canonical_frame": False,
                "canonical_second": absolute_second if in_hour and time_consistent else None,
                "canonical_exclusion_reason": None,
                "quality_flags": flags,
                "created_by": STAGE,
                "params_hash": param_hash,
                "input_manifest_ref": input_manifest_ref,
            }
        )
    return rows


def _mark_canonical(rows: list[dict]) -> None:
    groups: dict[tuple[str, str, float], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["day"], row["source"], float(row["absolute_second"]))].append(row)
    for group in groups.values():
        candidates = [
            row
            for row in group
            if "out_of_hour_timestamp" not in row["quality_flags"] and "absolute_second_mismatch" not in row["quality_flags"]
        ]
        winner = sorted(candidates, key=lambda row: (int(row["frame_index"]), row["tipsv2_embedding_id"]))[0] if candidates else None
        for row in group:
            if row is winner:
                row["is_canonical_frame"] = True
                row["canonical_second"] = row["absolute_second"]
                row["canonical_exclusion_reason"] = None
            else:
                reason = "duplicate_absolute_second_noncanonical" if winner else "invalid_frame_time"
                if "out_of_hour_timestamp" in row["quality_flags"]:
                    reason = "out_of_hour_timestamp"
                if "absolute_second_mismatch" in row["quality_flags"]:
                    reason = "absolute_second_mismatch"
                row["canonical_second"] = None
                row["canonical_exclusion_reason"] = reason
                if reason not in row["quality_flags"]:
                    row["quality_flags"].append(reason)


def _timeline_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "frame_atom_id": row["frame_atom_id"],
            "day": row["day"],
            "day_num": row["day_num"],
            "source": row["source"],
            "canonical_second": row["canonical_second"],
            "video_hour_stem": row["video_hour_stem"],
            "frame_index": row["frame_index"],
            "is_canonical_frame": row["is_canonical_frame"],
            "canonical_exclusion_reason": row["canonical_exclusion_reason"],
            "quality_flags": row["quality_flags"],
            "created_by": row["created_by"],
            "params_hash": row["params_hash"],
            "input_manifest_ref": row["input_manifest_ref"],
        }
        for row in rows
        if row["is_canonical_frame"]
    ]


def build_frame_atoms(config: AppConfig, run_root: Path, *, output_mode: str, run_id: Optional[str], day: Optional[str] = None, source: Optional[str] = None, resume: bool = False) -> dict:
    paths = stage_paths(run_root)
    atoms_path = run_root / "atoms" / "frame_atoms.parquet"
    timeline_path = run_root / "atoms" / "canonical_frame_timeline.parquet"
    params = {"stage": STAGE, "output_mode": output_mode, "run_id": run_id, "day": day, "source": source}
    if resume and stage_complete(paths.stage_manifest(STAGE), params=params, output_paths=[atoms_path, timeline_path]):
        return {"stage": STAGE, "skipped": True}

    discovered = _discover(config, day, source)
    if not discovered:
        raise FileNotFoundError("no frame/TIPSv2 shard pairs found")
    param_hash = params_hash(params)
    input_manifest_ref = to_ref(run_root / "manifests" / "input_data_manifest.json", config.project_root)

    grouped: dict[tuple[str, str], list[tuple[str, Path, Path]]] = defaultdict(list)
    for found_day, found_source, hour, manifest_path, tips_path in discovered:
        grouped[(found_day, found_source)].append((hour, manifest_path, tips_path))

    with AtomicParquetWriter(atoms_path, FRAME_ATOMS) as atoms_writer, AtomicParquetWriter(timeline_path, CANONICAL_FRAME_TIMELINE) as timeline_writer:
        for _, hour_items in sorted(grouped.items()):
            source_day_rows: list[dict] = []
            for _, manifest_path, tips_path in sorted(hour_items):
                source_day_rows.extend(_joined_hour_rows(config, manifest_path, tips_path, param_hash, input_manifest_ref))
            _mark_canonical(source_day_rows)
            atoms_writer.write_rows(source_day_rows)
            timeline_writer.write_rows(_timeline_rows(source_day_rows))

    manifest = write_stage_manifest(
        paths.stage_manifest(STAGE),
        stage=STAGE,
        params=params,
        inputs=[file_record(run_root / "manifests" / "input_data_manifest.json", config.project_root)],
        outputs=[
            {**file_record(atoms_path, config.project_root), "schema": str(FRAME_ATOMS)},
            {**file_record(timeline_path, config.project_root), "schema": str(CANONICAL_FRAME_TIMELINE)},
        ],
        schema_names=["frame_atoms", "canonical_frame_timeline"],
        project_root=config.project_root,
    )
    update_run_state(run_root, STAGE, to_ref(paths.stage_manifest(STAGE), config.project_root))
    return {"stage": STAGE, "manifest": manifest}
