from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from .config import AppConfig
from .embedding_refs import decode_sparse_binary
from .filters import source_matches
from .manifest import file_record, params_hash, write_stage_manifest
from .normalization import robust_z
from .parquet_io import AtomicParquetWriter
from .parquet_io import iter_pylist_rows, read_pylist
from .paths import to_ref
from .run_state import stage_complete, stage_paths, update_run_state
from .schema import SPARSE_FEATURE_TABLE
from .vector_utils import weighted_jaccard

STAGE = "sparse_features"


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


def _load_sparse(config: AppConfig, rows: list[dict]) -> dict[str, dict[int, float]]:
    by_path: dict[Path, set[str]] = defaultdict(set)
    for row in rows:
        if row.get("bgem3_embedding_id") and row.get("bgem3_parquet_ref"):
            by_path[config.project_root / row["bgem3_parquet_ref"]].add(row["bgem3_embedding_id"])
    result = {}
    for path, ids in by_path.items():
        remaining = set(ids)
        for item in iter_pylist_rows(path, columns=["embedding_id", "bge_sparse_indices_i32", "bge_sparse_values_f16"], batch_size=config.run.batch_size):
            embedding_id = item["embedding_id"]
            if embedding_id in remaining:
                result[embedding_id] = decode_sparse_binary(item["bge_sparse_indices_i32"], item["bge_sparse_values_f16"])
                remaining.remove(embedding_id)
                if not remaining:
                    break
    return result


def _top_changed(left: dict[int, float], right: dict[int, float], limit: int = 16) -> list[str]:
    keys = set(left) | set(right)
    scored = sorted(((abs(float(left.get(k, 0.0)) - float(right.get(k, 0.0))), k) for k in keys), reverse=True)
    return [str(key) for _, key in scored[:limit]]


def build_sparse_features(config: AppConfig, run_root: Path, *, output_mode: str, run_id: Optional[str], day: Optional[str] = None, source: Optional[str] = None, resume: bool = False) -> dict:
    paths = stage_paths(run_root)
    atoms_path = run_root / "atoms" / "text_atoms.parquet"
    feature_path = run_root / "features" / "sparse_keyword_features.parquet"
    params = {"stage": STAGE, "output_mode": output_mode, "run_id": run_id, "day": day, "source": source}
    if resume and stage_complete(paths.stage_manifest(STAGE), params=params, output_paths=[feature_path]):
        return {"stage": STAGE, "skipped": True}

    all_rows = read_pylist(atoms_path, batch_size=config.run.batch_size)
    rows = [
        row
        for row in all_rows
        if row["time_alignment_status"] != "invalid"
        and row.get("bgem3_embedding_id")
        and (day is None or row["day"] == ("day" + str(day).replace("day", "")))
        and source_matches(row["source"], source)
    ]
    sparse = _load_sparse(config, rows)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["day"], row["source"])].append(row)

    param_hash = params_hash(params)
    input_manifest_ref = to_ref(paths.stage_manifest("text_atoms"), config.project_root)
    with AtomicParquetWriter(feature_path, SPARSE_FEATURE_TABLE) as writer:
        for _, group in sorted(grouped.items()):
            group.sort(key=lambda item: (float(item["absolute_start_second"]), item["text_atom_id"]))
            overlaps = np.full(len(group), np.nan)
            changed: list[list[str]] = [[] for _ in group]
            for i in range(1, len(group)):
                current = sparse.get(group[i]["bgem3_embedding_id"])
                previous = sparse.get(group[i - 1]["bgem3_embedding_id"])
                if current is not None and previous is not None:
                    overlaps[i] = weighted_jaccard(current, previous)
                    changed[i] = _top_changed(current, previous)
            shift = 1.0 - overlaps
            z_shift = robust_z(shift)
            out = []
            for i, row in enumerate(group):
                out.append(
                    {
                        "day": row["day"],
                        "day_num": row["day_num"],
                        "source": row["source"],
                        "text_atom_id": row["text_atom_id"],
                        "boundary_second": _boundary_second(row, group[i - 1] if i > 0 else None),
                        "sparse_overlap": None if not np.isfinite(overlaps[i]) else float(overlaps[i]),
                        "sparse_keyword_shift_score": None if not np.isfinite(z_shift[i]) else float(z_shift[i]),
                        "top_changed_sparse_vocab_ids": changed[i],
                        "readable_sparse_tokens": None,
                        "quality_flags": list(row.get("quality_flags") or []),
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
        inputs=[file_record(atoms_path, config.project_root)],
        outputs=[{**file_record(feature_path, config.project_root), "schema": str(SPARSE_FEATURE_TABLE)}],
        schema_names=["sparse_keyword_features"],
        project_root=config.project_root,
    )
    update_run_state(run_root, STAGE, to_ref(paths.stage_manifest(STAGE), config.project_root))
    return {"stage": STAGE, "manifest": manifest}
