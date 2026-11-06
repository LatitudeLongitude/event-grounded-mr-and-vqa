from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional

from .boundary_projection import project_to_seconds
from .config import AppConfig
from .filters import source_matches
from .manifest import file_record, params_hash, write_stage_manifest
from .normalization import weighted_mean_available
from .parquet_io import AtomicParquetWriter
from .parquet_io import read_pylist
from .paths import to_ref
from .run_state import stage_complete, stage_paths, update_run_state
from .schema import FUSED_BOUNDARIES

STAGE = "fusion"


def _group(rows: list[dict], key_fields=("day", "source")) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["day"], row["source"])].append(row)
    return grouped


def _project_modality(seconds: list[float], rows: list[dict], score_key: str, radius: float) -> tuple[dict[float, float], dict[float, list[str]]]:
    values: dict[float, float] = {}
    ids: dict[float, list[str]] = defaultdict(list)
    for row in rows:
        for second, score in project_to_seconds(seconds, float(row["boundary_second"]), row.get(score_key), radius).items():
            if second not in values or score > values[second]:
                values[second] = score
            ids[second].append(row["text_atom_id"])
    return values, ids


def _hard_segments(rows: list[dict]) -> list[tuple[int, int]]:
    if not rows:
        return []
    bounds: list[tuple[int, int]] = []
    start_index = 0
    previous_second = float(rows[0]["second"])
    for index in range(1, len(rows)):
        second = float(rows[index]["second"])
        if rows[index].get("hard_boundary") or second - previous_second > 1.5:
            bounds.append((start_index, index - 1))
            start_index = index
        previous_second = second
    bounds.append((start_index, len(rows) - 1))
    return bounds


def _project_modality_in_segments(group_rows: list[dict], rows: list[dict], score_key: str, radius: float) -> tuple[dict[float, float], dict[float, list[str]]]:
    values: dict[float, float] = {}
    ids: dict[float, list[str]] = defaultdict(list)
    if not group_rows:
        return values, ids
    for start_index, end_index in _hard_segments(group_rows):
        segment_rows = group_rows[start_index : end_index + 1]
        seconds = [float(row["second"]) for row in segment_rows]
        segment_start = seconds[0]
        segment_end = seconds[-1] + 1.0
        segment_feature_rows = [
            row
            for row in rows
            if row.get("boundary_second") is not None and segment_start <= float(row["boundary_second"]) < segment_end
        ]
        segment_values, segment_ids = _project_modality(seconds, segment_feature_rows, score_key, radius)
        for second, score in segment_values.items():
            if second not in values or score > values[second]:
                values[second] = score
        for second, projected_ids in segment_ids.items():
            ids[second].extend(projected_ids)
    return values, ids


def _segment_bounds(rows: list[dict]) -> list[tuple[float, float]]:
    if not rows:
        return []
    seconds = [float(row["second"]) for row in rows]
    bounds: list[tuple[int, int]] = []
    start_index = 0
    for index in range(1, len(rows)):
        if rows[index].get("hard_boundary") or seconds[index] - seconds[index - 1] > 1.5:
            bounds.append((start_index, index - 1))
            start_index = index
    bounds.append((start_index, len(rows) - 1))
    result = [(0.0, 0.0)] * len(rows)
    for start, end in bounds:
        segment_start = seconds[start]
        segment_end = seconds[end] + 1.0
        for index in range(start, end + 1):
            result[index] = (segment_start, segment_end)
    return result


def _temporal_structure(second: float, segment_start: float, segment_end: float, level_min: int, target_sec: int) -> float:
    left = float(second) - float(segment_start)
    right = float(segment_end) - float(second)
    if left < level_min or right < level_min:
        return 0.0
    segment_duration = max(float(segment_end) - float(segment_start), 1.0)
    if segment_duration <= target_sec:
        return 0.0
    balance = min(left, right) / max(segment_duration / 2.0, 1.0)
    overlong_pressure = min(1.0, (segment_duration - target_sec) / max(target_sec, 1))
    return max(0.0, min(1.0, overlong_pressure * balance))


def build_fused_boundaries(config: AppConfig, run_root: Path, *, output_mode: str, run_id: Optional[str], day: Optional[str] = None, source: Optional[str] = None, resume: bool = False) -> dict:
    paths = stage_paths(run_root)
    visual_path = run_root / "boundaries" / "visual_boundary_scores.parquet"
    text_path = run_root / "boundaries" / "text_boundary_scores.parquet"
    sparse_path = run_root / "features" / "sparse_keyword_features.parquet"
    output_path = run_root / "boundaries" / "fused_boundary_scores.parquet"
    params = {"stage": STAGE, "output_mode": output_mode, "run_id": run_id, "day": day, "source": source}
    if resume and stage_complete(paths.stage_manifest(STAGE), params=params, output_paths=[output_path]):
        return {"stage": STAGE, "skipped": True}

    visual_rows = read_pylist(visual_path, batch_size=config.run.batch_size)
    text_rows = read_pylist(text_path, batch_size=config.run.batch_size) if text_path.exists() else []
    sparse_rows = read_pylist(sparse_path, batch_size=config.run.batch_size) if sparse_path.exists() else []
    text_by_group = _group(text_rows)
    sparse_by_group = _group(sparse_rows)
    param_hash = params_hash(params)
    input_manifest_ref = to_ref(paths.stage_manifest("visual_features"), config.project_root)

    with AtomicParquetWriter(output_path, FUSED_BOUNDARIES) as writer:
        for key, group_rows in sorted(_group(visual_rows).items()):
            found_day, found_source = key
            if day and found_day != ("day" + str(day).replace("day", "")):
                continue
            if not source_matches(found_source, source):
                continue
            group_rows.sort(key=lambda row: float(row["second"]))
            bounds = _segment_bounds(group_rows)
            text_l1, text_ids_l1 = _project_modality_in_segments(group_rows, text_by_group.get(key, []), "text_semantic_boundary_score", 5)
            text_l2, text_ids_l2 = _project_modality_in_segments(group_rows, text_by_group.get(key, []), "text_semantic_boundary_score", 30)
            text_l4, text_ids_l4 = _project_modality_in_segments(group_rows, text_by_group.get(key, []), "text_semantic_boundary_score", 120)
            text_l5, text_ids_l5 = _project_modality_in_segments(group_rows, text_by_group.get(key, []), "text_semantic_boundary_score", 300)
            sparse_l1, sparse_ids_l1 = _project_modality_in_segments(group_rows, sparse_by_group.get(key, []), "sparse_keyword_shift_score", 5)
            sparse_l2, sparse_ids_l2 = _project_modality_in_segments(group_rows, sparse_by_group.get(key, []), "sparse_keyword_shift_score", 20)
            sparse_l4, sparse_ids_l4 = _project_modality_in_segments(group_rows, sparse_by_group.get(key, []), "sparse_keyword_shift_score", 120)
            sparse_l5, sparse_ids_l5 = _project_modality_in_segments(group_rows, sparse_by_group.get(key, []), "sparse_keyword_shift_score", 300)
            out = []
            for i, row in enumerate(group_rows):
                second = float(row["second"])
                visual = row.get("visual_boundary_score")
                segment_start, segment_end = bounds[i]
                temporal_l1 = _temporal_structure(second, segment_start, segment_end, 1, 15)
                temporal_l2 = _temporal_structure(second, segment_start, segment_end, 3, config.segmentation.l2_max_sec)
                temporal_l4 = _temporal_structure(second, segment_start, segment_end, 60, config.segmentation.l4_max_sec)
                temporal_l5 = _temporal_structure(second, segment_start, segment_end, 300, config.segmentation.l5_max_sec)
                if row.get("hard_boundary"):
                    scores = {"l1_boundary_score": None, "l2_boundary_score": None, "l4_boundary_score": None, "l5_boundary_score": None}
                else:
                    scores = {
                        "l1_boundary_score": weighted_mean_available([(0.85, visual), (0.03, text_l1.get(second)), (0.02, sparse_l1.get(second)), (0.10, temporal_l1)]),
                        "l2_boundary_score": weighted_mean_available([(0.68, visual), (0.13, text_l2.get(second)), (0.09, sparse_l2.get(second)), (0.10, temporal_l2)]),
                        "l4_boundary_score": weighted_mean_available([(0.50, visual), (0.25, text_l4.get(second)), (0.15, sparse_l4.get(second)), (0.10, temporal_l4)]),
                        "l5_boundary_score": weighted_mean_available([(0.35, visual), (0.32, text_l5.get(second)), (0.20, sparse_l5.get(second)), (0.13, temporal_l5)]),
                    }
                projected_text = sorted(set(text_ids_l1.get(second, []) + text_ids_l2.get(second, []) + text_ids_l4.get(second, []) + text_ids_l5.get(second, [])))
                projected_sparse = sorted(set(sparse_ids_l1.get(second, []) + sparse_ids_l2.get(second, []) + sparse_ids_l4.get(second, []) + sparse_ids_l5.get(second, [])))
                flags = list(row.get("quality_flags") or [])
                if not projected_text:
                    flags.append("missing_text_projection")
                if not projected_sparse:
                    flags.append("missing_sparse_projection")
                out.append(
                    {
                        "day": row["day"],
                        "day_num": row["day_num"],
                        "source": row["source"],
                        "second": second,
                        "visual_boundary_score": visual,
                        "text_semantic_boundary_score": text_l2.get(second),
                        "sparse_keyword_shift_score": sparse_l2.get(second),
                        "temporal_structure_score": temporal_l2,
                        **scores,
                        "projected_text_atom_ids": projected_text,
                        "projected_sparse_atom_ids": projected_sparse,
                        "hard_boundary": bool(row.get("hard_boundary")),
                        "quality_flags": sorted(set(flags)),
                        "created_by": STAGE,
                        "params_hash": param_hash,
                        "input_manifest_ref": input_manifest_ref,
                    }
                )
            writer.write_rows(out)

    manifest = write_stage_manifest(
        paths.stage_manifest(STAGE),
        stage=STAGE,
        params=params,
        inputs=[file_record(visual_path, config.project_root), file_record(text_path, config.project_root), file_record(sparse_path, config.project_root)],
        outputs=[{**file_record(output_path, config.project_root), "schema": str(FUSED_BOUNDARIES)}],
        schema_names=["fused_boundary_scores"],
        project_root=config.project_root,
    )
    update_run_state(run_root, STAGE, to_ref(paths.stage_manifest(STAGE), config.project_root))
    return {"stage": STAGE, "manifest": manifest}
