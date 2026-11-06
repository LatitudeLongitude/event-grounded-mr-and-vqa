from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from .config import AppConfig
from .filters import source_matches
from .manifest import file_record, params_hash, write_stage_manifest
from .normalization import robust_z
from .parquet_io import AtomicParquetWriter
from .parquet_io import iter_pylist_rows, read_pylist
from .paths import to_ref
from .run_state import stage_complete, stage_paths, update_run_state
from .schema import TEXT_FEATURE_TABLE
from .vector_utils import cosine_distance

STAGE = "text_features"


def _intervals(row: dict) -> list[list[float]]:
    return json.loads(row.get("union_time_intervals") or "[]")


def _boundary_second(row: dict, previous: Optional[dict]) -> Optional[float]:
    intervals = _intervals(row)
    if not intervals:
        return row.get("absolute_start_second")
    current_start = float(intervals[0][0])
    if previous is None:
        return current_start
    previous_intervals = _intervals(previous)
    if previous_intervals:
        previous_end = float(previous_intervals[-1][1])
        if previous_end < current_start:
            return (previous_end + current_start) / 2.0
    return current_start


def _load_qwen_embeddings(config: AppConfig, rows: list[dict]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    by_path: dict[Path, set[str]] = defaultdict(set)
    for row in rows:
        if row.get("qwen3_embedding_id") and row.get("qwen3_parquet_ref"):
            by_path[config.project_root / row["qwen3_parquet_ref"]].add(row["qwen3_embedding_id"])
    for path, ids in by_path.items():
        remaining = set(ids)
        for item in iter_pylist_rows(path, columns=["embedding_id", "embedding"], batch_size=config.run.batch_size):
            embedding_id = item["embedding_id"]
            if embedding_id in remaining:
                result[embedding_id] = np.asarray(item["embedding"], dtype=np.float32)
                remaining.remove(embedding_id)
                if not remaining:
                    break
    return result


def build_text_features(config: AppConfig, run_root: Path, *, output_mode: str, run_id: Optional[str], day: Optional[str] = None, source: Optional[str] = None, resume: bool = False) -> dict:
    paths = stage_paths(run_root)
    atoms_path = run_root / "atoms" / "text_atoms.parquet"
    boundary_path = run_root / "boundaries" / "text_boundary_scores.parquet"
    feature_path = run_root / "features" / "text_topic_features.parquet"
    params = {"stage": STAGE, "output_mode": output_mode, "run_id": run_id, "day": day, "source": source}
    if resume and stage_complete(paths.stage_manifest(STAGE), params=params, output_paths=[boundary_path, feature_path]):
        return {"stage": STAGE, "skipped": True}

    all_rows = read_pylist(atoms_path, batch_size=config.run.batch_size)
    rows = [
        row
        for row in all_rows
        if row["time_alignment_status"] != "invalid"
        and row.get("qwen3_embedding_id")
        and (day is None or row["day"] == ("day" + str(day).replace("day", "")))
        and source_matches(row["source"], source)
    ]
    embeddings = _load_qwen_embeddings(config, rows)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["day"], row["source"])].append(row)

    param_hash = params_hash(params)
    input_manifest_ref = to_ref(paths.stage_manifest("text_atoms"), config.project_root)
    with AtomicParquetWriter(boundary_path, TEXT_FEATURE_TABLE) as boundary_writer, AtomicParquetWriter(feature_path, TEXT_FEATURE_TABLE) as feature_writer:
        for _, group in sorted(grouped.items()):
            group.sort(key=lambda item: (float(item["absolute_start_second"]), item["text_atom_id"]))
            dense_shift = np.full(len(group), np.nan)
            density = np.full(len(group), np.nan)
            density_change = np.full(len(group), np.nan)
            for i, row in enumerate(group):
                duration = max(float(row["aligned_duration_sec"] or 0.0), 1e-6)
                density[i] = float(row.get("token_estimate") or 0) / duration
                if i > 0:
                    prev = group[i - 1]
                    a = embeddings.get(row["qwen3_embedding_id"])
                    b = embeddings.get(prev["qwen3_embedding_id"])
                    if a is not None and b is not None:
                        dense_shift[i] = cosine_distance(a, b)
                    density_change[i] = abs(density[i] - density[i - 1])
            z_shift = robust_z(dense_shift)
            z_density = robust_z(density_change)
            out = []
            for i, row in enumerate(group):
                boundary_second = _boundary_second(row, group[i - 1] if i > 0 else None)
                score = None
                if np.isfinite(z_shift[i]) or np.isfinite(z_density[i]):
                    score = 0.80 * (0.0 if not np.isfinite(z_shift[i]) else float(z_shift[i])) + 0.20 * (0.0 if not np.isfinite(z_density[i]) else float(z_density[i]))
                out.append(
                    {
                        "day": row["day"],
                        "day_num": row["day_num"],
                        "source": row["source"],
                        "text_atom_id": row["text_atom_id"],
                        "boundary_second": boundary_second,
                        "dense_shift": None if not np.isfinite(dense_shift[i]) else float(dense_shift[i]),
                        "text_density": None if not np.isfinite(density[i]) else float(density[i]),
                        "text_density_change": None if not np.isfinite(density_change[i]) else float(density_change[i]),
                        "text_semantic_boundary_score": score,
                        "quality_flags": list(row.get("quality_flags") or []),
                        "created_by": STAGE,
                        "params_hash": param_hash,
                        "input_manifest_ref": input_manifest_ref,
                    }
                )
            boundary_writer.write_rows(out)
            feature_writer.write_rows(out)

    manifest = write_stage_manifest(
        paths.stage_manifest(STAGE),
        stage=STAGE,
        params=params,
        inputs=[file_record(atoms_path, config.project_root)],
        outputs=[
            {**file_record(boundary_path, config.project_root), "schema": str(TEXT_FEATURE_TABLE)},
            {**file_record(feature_path, config.project_root), "schema": str(TEXT_FEATURE_TABLE)},
        ],
        schema_names=["text_boundary_scores", "text_topic_features"],
        project_root=config.project_root,
    )
    update_run_state(run_root, STAGE, to_ref(paths.stage_manifest(STAGE), config.project_root))
    return {"stage": STAGE, "manifest": manifest}
