from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from .config import AppConfig
from .filters import source_matches
from .manifest import file_record, params_hash, write_stage_manifest
from .normalization import robust_z, weighted_mean_available
from .parquet_io import AtomicParquetWriter
from .parquet_io import iter_pylist_rows
from .paths import to_ref
from .run_state import stage_complete, stage_paths, update_run_state
from .schema import FEATURE_TABLE
from .vector_utils import cosine_distance

STAGE = "visual_features"


def _frame_groups(path: Path, batch_size: int) -> dict[tuple[str, str], list[dict]]:
    columns = ["frame_atom_id", "day", "day_num", "source", "canonical_second", "video_hour_stem", "frame_index", "is_canonical_frame"]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in iter_pylist_rows(path, columns=columns, batch_size=batch_size):
        if row["is_canonical_frame"]:
            groups[(row["day"], row["source"])].append(row)
    for group_rows in groups.values():
        group_rows.sort(key=lambda row: float(row["canonical_second"]))
    return groups


def _load_group_embeddings(config: AppConfig, group_rows: list[dict]) -> list[Optional[np.ndarray]]:
    by_hour: dict[str, list[dict]] = defaultdict(list)
    for row in group_rows:
        by_hour[row["video_hour_stem"]].append(row)
    embeddings_by_id: dict[str, np.ndarray] = {}
    for hour, rows in by_hour.items():
        sample = rows[0]
        path = config.input_roots.tipsv2_root / sample["day"] / sample["source"] / f"{hour}.parquet"
        ids_by_index: dict[int, list[str]] = defaultdict(list)
        for row in rows:
            ids_by_index[int(row["frame_index"])].append(row["frame_atom_id"])
        remaining = set(ids_by_index)
        for item in iter_pylist_rows(path, columns=["frame_index", "embedding"], batch_size=config.run.batch_size):
            frame_index = int(item["frame_index"])
            frame_ids = ids_by_index.get(frame_index)
            if not frame_ids:
                continue
            vector = np.asarray(item["embedding"], dtype=np.float32)
            for frame_id in frame_ids:
                embeddings_by_id[frame_id] = vector
            remaining.discard(frame_index)
            if not remaining:
                break
    return [embeddings_by_id.get(row["frame_atom_id"]) for row in group_rows]


def _mean(vectors: list[np.ndarray]) -> Optional[np.ndarray]:
    if not vectors:
        return None
    return np.mean(np.stack(vectors), axis=0)


def _window(vectors: list[Optional[np.ndarray]], seconds: list[float], start: int, end: int) -> list[np.ndarray]:
    if start < 0 or end >= len(vectors):
        return []
    for index in range(start + 1, end + 1):
        if seconds[index] - seconds[index - 1] > 1.5:
            return []
    return [vec for vec in vectors[start : end + 1] if vec is not None]


def _scores_for_group(config: AppConfig, rows: list[dict], param_hash: str, input_manifest_ref: str) -> list[dict]:
    embeddings = _load_group_embeddings(config, rows)
    n = len(rows)
    delta_1 = np.full(n, np.nan)
    delta_5 = np.full(n, np.nan)
    delta_15 = np.full(n, np.nan)
    hard = np.zeros(n, dtype=bool)
    seconds = [float(row["canonical_second"]) for row in rows]
    for i in range(n):
        if i > 0 and seconds[i] - seconds[i - 1] > 1.5:
            hard[i] = True
        if i > 0 and not hard[i] and embeddings[i] is not None and embeddings[i - 1] is not None:
            delta_1[i] = cosine_distance(embeddings[i], embeddings[i - 1])
        current = _window(embeddings, seconds, i - 2, i + 2)
        previous = _window(embeddings, seconds, i - 7, i - 3)
        if len(current) >= 3 and len(previous) >= 3:
            delta_5[i] = cosine_distance(_mean(current), _mean(previous))
        current = _window(embeddings, seconds, i - 7, i + 7)
        previous = _window(embeddings, seconds, i - 22, i - 8)
        if len(current) >= 9 and len(previous) >= 9:
            delta_15[i] = cosine_distance(_mean(current), _mean(previous))
    z1 = robust_z(delta_1)
    z5 = robust_z(delta_5)
    z15 = robust_z(delta_15)
    activity = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - 5)
        hi = min(n, i + 6)
        window = delta_1[lo:hi]
        finite = window[np.isfinite(window)]
        if finite.size:
            activity[i] = np.median(finite)
    z_activity = robust_z(activity)
    out = []
    for i, row in enumerate(rows):
        score = weighted_mean_available([(0.45, z1[i]), (0.35, z5[i]), (0.20, z15[i])])
        flags = []
        if hard[i]:
            flags.append("hard_boundary_gap")
        if embeddings[i] is None:
            flags.append("missing_visual_embedding")
        out.append(
            {
                "day": row["day"],
                "day_num": row["day_num"],
                "source": row["source"],
                "second": seconds[i],
                "anchor_id": row["frame_atom_id"],
                "delta_1": None if not np.isfinite(delta_1[i]) else float(delta_1[i]),
                "delta_5": None if not np.isfinite(delta_5[i]) else float(delta_5[i]),
                "delta_15": None if not np.isfinite(delta_15[i]) else float(delta_15[i]),
                "z_delta_1": None if not np.isfinite(z1[i]) else float(z1[i]),
                "z_delta_5": None if not np.isfinite(z5[i]) else float(z5[i]),
                "z_delta_15": None if not np.isfinite(z15[i]) else float(z15[i]),
                "activity_score": None if not np.isfinite(z_activity[i]) else float(z_activity[i]),
                "visual_boundary_score": score,
                "hard_boundary": bool(hard[i]),
                "quality_flags": flags,
                "created_by": STAGE,
                "params_hash": param_hash,
                "input_manifest_ref": input_manifest_ref,
            }
        )
    return out


def build_visual_features(config: AppConfig, run_root: Path, *, output_mode: str, run_id: Optional[str], day: Optional[str] = None, source: Optional[str] = None, resume: bool = False) -> dict:
    paths = stage_paths(run_root)
    timeline_path = run_root / "atoms" / "canonical_frame_timeline.parquet"
    boundary_path = run_root / "boundaries" / "visual_boundary_scores.parquet"
    motion_path = run_root / "features" / "visual_motion_features.parquet"
    params = {"stage": STAGE, "output_mode": output_mode, "run_id": run_id, "day": day, "source": source}
    if resume and stage_complete(paths.stage_manifest(STAGE), params=params, output_paths=[boundary_path, motion_path]):
        return {"stage": STAGE, "skipped": True}

    param_hash = params_hash(params)
    input_manifest_ref = to_ref(paths.stage_manifest("frame_atoms"), config.project_root)
    with AtomicParquetWriter(boundary_path, FEATURE_TABLE) as boundary_writer, AtomicParquetWriter(motion_path, FEATURE_TABLE) as motion_writer:
        for (found_day, found_source), rows in sorted(_frame_groups(timeline_path, config.run.batch_size).items()):
            if day and found_day != f"day{int(str(day).replace('day', ''))}":
                continue
            if not source_matches(found_source, source):
                continue
            out = _scores_for_group(config, rows, param_hash, input_manifest_ref)
            boundary_writer.write_rows(out)
            motion_writer.write_rows(out)

    manifest = write_stage_manifest(
        paths.stage_manifest(STAGE),
        stage=STAGE,
        params=params,
        inputs=[file_record(timeline_path, config.project_root)],
        outputs=[
            {**file_record(boundary_path, config.project_root), "schema": str(FEATURE_TABLE)},
            {**file_record(motion_path, config.project_root), "schema": str(FEATURE_TABLE)},
        ],
        schema_names=["visual_boundary_scores", "visual_motion_features"],
        project_root=config.project_root,
    )
    update_run_state(run_root, STAGE, to_ref(paths.stage_manifest(STAGE), config.project_root))
    return {"stage": STAGE, "manifest": manifest}
