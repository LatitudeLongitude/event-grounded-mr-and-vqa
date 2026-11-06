from __future__ import annotations

import json
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .config import AppConfig
from .ids import stable_hash
from .manifest import file_record, params_hash, write_stage_manifest
from .parquet_io import iter_pylist_rows, write_table
from .paths import to_ref
from .run_state import stage_complete, stage_paths, update_run_state
from .schema import LINK_TABLE

STAGE = "links"

SEGMENT_FILES = [
    "segments/l1_micro_events.parquet",
    "segments/l2_source_events.parquet",
    "segments/l3_global_events.parquet",
    "segments/l4_activity_episodes.parquet",
    "segments/l5_sessions.parquet",
]


def _events(run_root: Path) -> list[dict]:
    rows = []
    for rel in SEGMENT_FILES:
        path = run_root / rel
        if path.exists():
            rows.extend(iter_pylist_rows(path))
    return rows


def _target_fields(link_type: str, event_id: str, target_id: str) -> dict:
    frame_atom = target_id if target_id.startswith("frame::") else None
    text_atom = target_id if target_id.startswith("text::") else None
    embedding_ref = target_id if link_type == "semantic_support" else None
    parent_event = event_id if link_type == "temporal_contains" else None
    child_event = target_id if link_type == "temporal_contains" else None
    return {
        "frame_atom_id": frame_atom,
        "text_atom_id": text_atom,
        "embedding_ref": embedding_ref,
        "parent_event_id": parent_event,
        "child_event_id": child_event,
    }


def _link(link_type: str, event_id: str, target_id: str, param_hash: str, input_manifest_ref: str, *, role: Optional[str] = None, score: Optional[float] = None, rank: int = 0, representative: bool = False, total: int = 1, selection_meta: Optional[dict] = None) -> dict:
    selection_meta = selection_meta or {}
    selection_score = score
    if representative and selection_score is None:
        selection_score = selection_meta.get("selection_score", max(0.0, 1.0 - (rank / max(total, 1))))
    selection_reason = selection_meta.get("selection_reason") or role
    selection_rank = int(selection_meta.get("selection_rank", rank))
    return {
        "link_id": f"{link_type}::" + stable_hash([event_id, target_id, role, selection_rank], 20),
        "event_id": event_id,
        "target_id": target_id,
        **_target_fields(link_type, event_id, target_id),
        "link_type": link_type,
        "link_role": role,
        "score": score,
        "selection_score": selection_score,
        "selection_reason": selection_reason,
        "selection_rank": selection_rank,
        "derived_from_event_id": selection_meta.get("derived_from_event_id"),
        "derived_from_frame_atom_id": selection_meta.get("derived_from_frame_atom_id") or (target_id if target_id.startswith("frame::") else None),
        "is_representative": representative,
        "quality_flags": [],
        "created_by": STAGE,
        "params_hash": param_hash,
        "input_manifest_ref": input_manifest_ref,
    }


def _canonical_frames_by_group(run_root: Path) -> dict[tuple[str, str], tuple[list[float], list[str]]]:
    path = run_root / "atoms" / "canonical_frame_timeline.parquet"
    if not path.exists():
        return {}
    grouped: dict[tuple[str, str], list[tuple[float, str]]] = defaultdict(list)
    for row in iter_pylist_rows(path, columns=["day", "source", "canonical_second", "frame_atom_id"]):
        if row.get("canonical_second") is None:
            continue
        grouped[(row["day"], row["source"])].append((float(row["canonical_second"]), row["frame_atom_id"]))
    result = {}
    for key, rows in grouped.items():
        rows.sort()
        result[key] = ([second for second, _ in rows], [frame_id for _, frame_id in rows])
    return result


def _event_sources(event: dict) -> list[str]:
    if event.get("source"):
        return [event["source"]]
    return list(event.get("sources") or [])


def _frames_for_event(event: dict, frames_by_group: dict[tuple[str, str], tuple[list[float], list[str]]]) -> list[str]:
    frame_ids = []
    start = float(event["start_second"])
    end = float(event["end_second"])
    for source in _event_sources(event):
        item = frames_by_group.get((event["day"], source))
        if item is None:
            continue
        seconds, ids = item
        left = bisect_left(seconds, start)
        right = bisect_left(seconds, end)
        frame_ids.extend(ids[left:right])
    return frame_ids


def build_links(config: AppConfig, run_root: Path, *, output_mode: str, run_id: Optional[str], day: Optional[str] = None, source: Optional[str] = None, resume: bool = False) -> dict:
    paths = stage_paths(run_root)
    frame_path = run_root / "links" / "event_frame_links.parquet"
    text_path = run_root / "links" / "event_text_links.parquet"
    embedding_path = run_root / "links" / "event_embedding_links.parquet"
    parent_path = run_root / "links" / "event_parent_child_links.parquet"
    params = {"stage": STAGE, "output_mode": output_mode, "run_id": run_id, "day": day, "source": source}
    if resume and stage_complete(paths.stage_manifest(STAGE), params=params, output_paths=[frame_path, text_path, embedding_path, parent_path]):
        return {"stage": STAGE, "skipped": True}

    rows = _events(run_root)
    representative_meta_path = run_root / "links" / "representative_frame_selection_meta.json"
    representative_meta = json.loads(representative_meta_path.read_text(encoding="utf-8")) if representative_meta_path.exists() else {}
    frames_by_group = _canonical_frames_by_group(run_root)
    param_hash = params_hash(params)
    input_manifest_ref = to_ref(paths.stage_manifest("event_embeddings"), config.project_root)
    frame_links = []
    text_links = []
    embedding_links = []
    parent_links_by_pair: dict[tuple[str, str], dict] = {}

    def add_parent_link(parent_id: str, child_id: str, role: str) -> None:
        key = (parent_id, child_id)
        row = _link("temporal_contains", parent_id, child_id, param_hash, input_manifest_ref, role=role)
        existing = parent_links_by_pair.get(key)
        if existing is None or (role == "primary_parent" and existing.get("link_role") != "primary_parent"):
            parent_links_by_pair[key] = row

    for event in rows:
        event_id = event["event_id"]
        frame_ids = event.get("representative_frame_ids") or []
        representative_ids = set(frame_ids)
        for rank, frame_id in enumerate(frame_ids):
            meta = (representative_meta.get(event_id) or {}).get(frame_id, {})
            role = meta.get("selection_reason") or "representative"
            frame_links.append(_link("frame_overlap_support", event_id, frame_id, param_hash, input_manifest_ref, role=role, rank=rank, representative=True, total=len(frame_ids), selection_meta=meta))
        if event.get("level") in {"L1", "L2"}:
            for rank, frame_id in enumerate(_frames_for_event(event, frames_by_group)):
                if frame_id in representative_ids:
                    continue
                frame_links.append(_link("frame_overlap_support", event_id, frame_id, param_hash, input_manifest_ref, role="frame_overlap_support", rank=rank, representative=False))
        for rank, text_id in enumerate(event.get("linked_text_atom_ids") or []):
            text_links.append(_link("text_overlap_support", event_id, text_id, param_hash, input_manifest_ref, role="overlap", rank=rank))
        for ref_name in ("visual_centroid_ref", "text_dense_centroid_ref", "text_sparse_aggregate_ref"):
            if event.get(ref_name):
                embedding_links.append(_link("semantic_support", event_id, event[ref_name], param_hash, input_manifest_ref, role=ref_name))
        if event.get("primary_parent_id"):
            add_parent_link(event["primary_parent_id"], event_id, "primary_parent")
        for child_id in event.get("child_event_ids") or event.get("child_source_event_ids") or []:
            add_parent_link(event_id, child_id, "child")

    write_table(frame_path, frame_links, schema=LINK_TABLE)
    write_table(text_path, text_links, schema=LINK_TABLE)
    write_table(embedding_path, embedding_links, schema=LINK_TABLE)
    write_table(parent_path, list(parent_links_by_pair.values()), schema=LINK_TABLE)
    manifest = write_stage_manifest(
        paths.stage_manifest(STAGE),
        stage=STAGE,
        params=params,
        inputs=[*[file_record(run_root / rel, config.project_root) for rel in SEGMENT_FILES], file_record(representative_meta_path, config.project_root)],
        outputs=[
            {**file_record(frame_path, config.project_root), "schema": str(LINK_TABLE)},
            {**file_record(text_path, config.project_root), "schema": str(LINK_TABLE)},
            {**file_record(embedding_path, config.project_root), "schema": str(LINK_TABLE)},
            {**file_record(parent_path, config.project_root), "schema": str(LINK_TABLE)},
        ],
        schema_names=["event_frame_links", "event_text_links", "event_embedding_links", "event_parent_child_links"],
        project_root=config.project_root,
    )
    update_run_state(run_root, STAGE, to_ref(paths.stage_manifest(STAGE), config.project_root))
    return {"stage": STAGE, "manifest": manifest}
