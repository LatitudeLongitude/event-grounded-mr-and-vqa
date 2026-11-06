from __future__ import annotations

from pathlib import Path
from typing import Optional

from .config import AppConfig
from .manifest import file_record, params_hash, write_stage_manifest
from .parquet_io import atomic_write_jsonl, iter_pylist_rows, write_table
from .paths import to_ref
from .run_state import stage_complete, stage_paths, update_run_state
from .schema import CORPUS_INDEX

STAGE = "corpus_index"

SEGMENT_FILES = [
    "segments/l1_micro_events.parquet",
    "segments/l2_source_events.parquet",
    "segments/l3_global_events.parquet",
    "segments/l4_activity_episodes.parquet",
    "segments/l5_sessions.parquet",
]


def _index_row(event: dict, param_hash: str, input_manifest_ref: str) -> dict:
    return {
        "event_id": event["event_id"],
        "level": event["level"],
        "day": event["day"],
        "source": event.get("source"),
        "sources": event.get("sources") or [],
        "start_second": event["start_second"],
        "end_second": event["end_second"],
        "start_hms": event["start_hms"],
        "end_hms": event["end_hms"],
        "event_kind": event.get("event_kind"),
        "primary_parent_id": event.get("primary_parent_id"),
        "representative_frame_ids": event.get("representative_frame_ids") or [],
        "linked_text_atom_ids": event.get("linked_text_atom_ids") or [],
        "visual_centroid_ref": event.get("visual_centroid_ref"),
        "text_dense_centroid_ref": event.get("text_dense_centroid_ref"),
        "text_sparse_aggregate_ref": event.get("text_sparse_aggregate_ref"),
        "modality_support": event.get("modality_support") or [],
        "confidence": event.get("confidence"),
        "quality_flags": event.get("quality_flags") or [],
        "created_by": STAGE,
        "params_hash": param_hash,
        "input_manifest_ref": input_manifest_ref,
    }


def build_corpus_index(config: AppConfig, run_root: Path, *, output_mode: str, run_id: Optional[str], day: Optional[str] = None, source: Optional[str] = None, resume: bool = False) -> dict:
    paths = stage_paths(run_root)
    parquet_path = run_root / "indexes" / "event_corpus_index.parquet"
    jsonl_path = run_root / "indexes" / "event_corpus_index.jsonl"
    params = {"stage": STAGE, "output_mode": output_mode, "run_id": run_id, "day": day, "source": source}
    if resume and stage_complete(paths.stage_manifest(STAGE), params=params, output_paths=[parquet_path, jsonl_path]):
        return {"stage": STAGE, "skipped": True}
    param_hash = params_hash(params)
    input_manifest_ref = to_ref(paths.stage_manifest("links"), config.project_root)

    rows = []
    for rel in SEGMENT_FILES:
        path = run_root / rel
        if path.exists():
            rows.extend(_index_row(row, param_hash, input_manifest_ref) for row in iter_pylist_rows(path, batch_size=config.run.batch_size))
    rows.sort(key=lambda row: (row["day"], row["start_second"], row["level"], row["event_id"]))
    parquet_result = write_table(parquet_path, rows, schema=CORPUS_INDEX)
    atomic_write_jsonl(jsonl_path, rows)
    manifest = write_stage_manifest(
        paths.stage_manifest(STAGE),
        stage=STAGE,
        params=params,
        inputs=[
            *[file_record(run_root / rel, config.project_root) for rel in SEGMENT_FILES],
            file_record(run_root / "links" / "event_parent_child_links.parquet", config.project_root),
        ],
        outputs=[{**parquet_result, "schema": str(CORPUS_INDEX)}, file_record(jsonl_path, config.project_root)],
        schema_names=["event_corpus_index"],
        project_root=config.project_root,
    )
    update_run_state(run_root, STAGE, to_ref(paths.stage_manifest(STAGE), config.project_root))
    return {"stage": STAGE, "manifest": manifest}
