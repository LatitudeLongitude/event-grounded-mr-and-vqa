from __future__ import annotations

import json
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from .config import AppConfig
from .embedding_refs import decode_sparse_binary
from .intervals import overlap_duration
from .manifest import file_record, params_hash, refresh_manifest_file_records, write_stage_manifest
from .parquet_io import atomic_write_json, iter_pylist_rows, read_column_pylist, read_pylist, write_table
from .paths import to_ref
from .representative_frames import inherit_representative_frames, select_representative_frames
from .run_state import stage_complete, stage_paths, update_run_state
from .schema import EVENT_EMBEDDINGS, EVENT_TABLE, L3_EVENT_TABLE, L4_L5_EVENT_TABLE, SPARSE_AGGREGATES
from .vector_utils import deterministic_uniform_sample, l2_normalize

STAGE = "event_embeddings"


SEGMENT_FILES = [
    ("l1_micro_events", "segments/l1_micro_events.parquet", EVENT_TABLE),
    ("l2_source_events", "segments/l2_source_events.parquet", EVENT_TABLE),
    ("l3_global_events", "segments/l3_global_events.parquet", L3_EVENT_TABLE),
    ("l4_activity_episodes", "segments/l4_activity_episodes.parquet", L4_L5_EVENT_TABLE),
    ("l5_sessions", "segments/l5_sessions.parquet", L4_L5_EVENT_TABLE),
]


def _load_frame_meta(run_root: Path, batch_size: int) -> tuple[dict[str, dict], dict[tuple[str, str], tuple[list[float], list[dict]]]]:
    path = run_root / "atoms" / "frame_atoms.parquet"
    columns = [
        "frame_atom_id",
        "day",
        "day_num",
        "source",
        "canonical_second",
        "is_canonical_frame",
        "path_exists_at_build",
        "embedding_exists_at_build",
        "tipsv2_embedding_id",
        "tipsv2_parquet_ref",
        "quality_flags",
    ]
    rows = read_pylist(path, columns=columns, batch_size=batch_size)
    by_id = {row["frame_atom_id"]: row for row in rows}
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("is_canonical_frame") and row.get("canonical_second") is not None:
            grouped[(row["day"], row["source"])].append(row)
    indexed = {}
    for key, group_rows in grouped.items():
        group_rows.sort(key=lambda row: (float(row["canonical_second"]), row["frame_atom_id"]))
        indexed[key] = ([float(row["canonical_second"]) for row in group_rows], group_rows)
    return by_id, indexed


def _load_visual_scores(run_root: Path, batch_size: int) -> dict[str, dict]:
    path = run_root / "features" / "visual_motion_features.parquet"
    if not path.exists():
        return {}
    columns = ["anchor_id", "activity_score", "visual_boundary_score"]
    return {row["anchor_id"]: row for row in iter_pylist_rows(path, columns=columns, batch_size=batch_size)}


def _frames_between(indexed_frames: dict[tuple[str, str], tuple[list[float], list[dict]]], day: str, source: str, start: float, end: float) -> list[dict]:
    item = indexed_frames.get((day, source))
    if item is None:
        return []
    seconds, rows = item
    left = bisect_left(seconds, float(start))
    right = bisect_left(seconds, float(end))
    return rows[left:right]


def _event_sources(event: dict) -> list[str]:
    if event.get("source"):
        return [event["source"]]
    return list(event.get("sources") or [])


def _frames_for_event(event: dict, indexed_frames: dict[tuple[str, str], tuple[list[float], list[dict]]]) -> list[dict]:
    rows = []
    for source in _event_sources(event):
        rows.extend(_frames_between(indexed_frames, event["day"], source, event["start_second"], event["end_second"]))
    rows.sort(key=lambda row: (float(row["canonical_second"]), row["source"], row["frame_atom_id"]))
    return rows


def _load_embeddings_by_frame(config: AppConfig, frame_meta: dict[str, dict], frame_ids: set[str]) -> dict[str, np.ndarray]:
    by_path: dict[Path, set[str]] = defaultdict(set)
    id_to_frame = {}
    for frame_id in frame_ids:
        meta = frame_meta.get(frame_id)
        if not meta or not meta.get("tipsv2_embedding_id") or not meta.get("tipsv2_parquet_ref"):
            continue
        by_path[config.project_root / meta["tipsv2_parquet_ref"]].add(meta["tipsv2_embedding_id"])
        id_to_frame[meta["tipsv2_embedding_id"]] = frame_id
    result = {}
    for path, needed in by_path.items():
        remaining = set(needed)
        for row in iter_pylist_rows(path, columns=["embedding_id", "embedding"], batch_size=config.run.batch_size):
            embedding_id = row["embedding_id"]
            if embedding_id in remaining:
                result[id_to_frame[embedding_id]] = np.asarray(row["embedding"], dtype=np.float32)
                remaining.remove(embedding_id)
                if not remaining:
                    break
    return result


def _load_text_meta(run_root: Path, batch_size: int) -> dict[str, dict]:
    path = run_root / "atoms" / "text_atoms.parquet"
    if not path.exists():
        return {}
    return {row["text_atom_id"]: row for row in iter_pylist_rows(path, batch_size=batch_size)}


def _load_dense(config: AppConfig, text_meta: dict[str, dict], text_ids: set[str]) -> dict[str, np.ndarray]:
    by_path: dict[Path, set[str]] = defaultdict(set)
    id_to_text = {}
    for text_id in text_ids:
        meta = text_meta.get(text_id)
        if not meta or not meta.get("qwen3_embedding_id") or not meta.get("qwen3_parquet_ref"):
            continue
        by_path[config.project_root / meta["qwen3_parquet_ref"]].add(meta["qwen3_embedding_id"])
        id_to_text[meta["qwen3_embedding_id"]] = text_id
    result = {}
    for path, needed in by_path.items():
        remaining = set(needed)
        for row in iter_pylist_rows(path, columns=["embedding_id", "embedding"], batch_size=config.run.batch_size):
            embedding_id = row["embedding_id"]
            if embedding_id in remaining:
                result[id_to_text[embedding_id]] = np.asarray(row["embedding"], dtype=np.float32)
                remaining.remove(embedding_id)
                if not remaining:
                    break
    return result


def _load_sparse(config: AppConfig, text_meta: dict[str, dict], text_ids: set[str]) -> dict[str, dict[int, float]]:
    by_path: dict[Path, set[str]] = defaultdict(set)
    id_to_text = {}
    for text_id in text_ids:
        meta = text_meta.get(text_id)
        if not meta or not meta.get("bgem3_embedding_id") or not meta.get("bgem3_parquet_ref"):
            continue
        by_path[config.project_root / meta["bgem3_parquet_ref"]].add(meta["bgem3_embedding_id"])
        id_to_text[meta["bgem3_embedding_id"]] = text_id
    result = {}
    for path, needed in by_path.items():
        remaining = set(needed)
        for row in iter_pylist_rows(path, columns=["embedding_id", "bge_sparse_indices_i32", "bge_sparse_values_f16"], batch_size=config.run.batch_size):
            embedding_id = row["embedding_id"]
            if embedding_id in remaining:
                result[id_to_text[embedding_id]] = decode_sparse_binary(row["bge_sparse_indices_i32"], row["bge_sparse_values_f16"])
                remaining.remove(embedding_id)
                if not remaining:
                    break
    return result


def _centroid(vectors: list[np.ndarray], weights: Optional[list[float]] = None) -> Optional[list[float]]:
    if not vectors:
        return None
    if weights is None:
        pooled = np.mean(np.stack(vectors), axis=0)
    else:
        weight_array = np.asarray(weights, dtype=np.float64)
        if not np.isfinite(weight_array).any() or float(weight_array.sum()) <= 0:
            return None
        pooled = np.average(np.stack(vectors), axis=0, weights=weight_array)
    return l2_normalize(pooled).astype(float).tolist()


def _text_overlap_weights(event: dict, text_meta: dict[str, dict]) -> dict[str, float]:
    weights = {}
    event_interval = (float(event["start_second"]), float(event["end_second"]))
    for text_id in event.get("linked_text_atom_ids") or []:
        meta = text_meta.get(text_id)
        if not meta or meta.get("time_alignment_status") == "invalid":
            continue
        intervals = json.loads(meta.get("union_time_intervals") or "[]")
        overlap = sum(overlap_duration(event_interval, tuple(interval)) for interval in intervals)
        weights[text_id] = overlap if overlap > 0 else 1.0
    return weights


def _sparse_sum(vectors: list[tuple[dict[int, float], float]], limit: int) -> tuple[list[str], list[float]]:
    values: dict[int, float] = defaultdict(float)
    for vector, weight in vectors:
        for key, value in vector.items():
            values[key] += float(value) * float(weight)
    top = sorted(values.items(), key=lambda item: item[1], reverse=True)[:limit]
    return [str(key) for key, _ in top], [float(value) for _, value in top]


def _level_limit(level: str) -> int:
    return {"L1": 3, "L2": 8, "L3": 12, "L4": 24, "L5": 36}.get(level, 8)


def _current_segment_event_ids(run_root: Path) -> set[str]:
    event_ids = set()
    for _, rel, _ in SEGMENT_FILES:
        path = run_root / rel
        if path.exists():
            event_ids.update(read_column_pylist(path, "event_id"))
    return set(str(event_id) for event_id in event_ids)


def _embedding_outputs_cover(run_root: Path, event_ids: set[str]) -> bool:
    if not event_ids:
        return True
    paths = [
        run_root / "embeddings" / "event_visual_centroids.parquet",
        run_root / "embeddings" / "event_text_dense_centroids.parquet",
        run_root / "embeddings" / "event_text_sparse_aggregates.parquet",
    ]
    if not all(path.exists() for path in paths):
        return False
    try:
        seen_by_output = [
            set(str(item) for item in read_column_pylist(path, "event_id"))
            for path in paths
        ]
    except Exception:
        return False
    return all(event_ids <= seen for seen in seen_by_output)


def _propagate_high_level_representatives(events_by_id: dict[str, dict], representative_meta: dict[str, dict[str, dict]]) -> None:
    for level, field, limit, by_source in (
        ("L3", "child_source_event_ids", 12, True),
        ("L4", "child_event_ids", 24, False),
        ("L5", "child_event_ids", 36, False),
    ):
        level_events = [event for event in events_by_id.values() if event["level"] == level]
        for event in level_events:
            children = [events_by_id[child_id] for child_id in event.get(field) or [] if child_id in events_by_id]
            frames = inherit_representative_frames(children, limit, by_source=by_source)
            if frames:
                event["representative_frame_ids"] = frames
                event_meta = {}
                for rank, frame_id in enumerate(frames):
                    source_child = next((child for child in children if frame_id in (child.get("representative_frame_ids") or [])), None)
                    child_meta = (representative_meta.get(source_child["event_id"], {}) if source_child else {}).get(frame_id, {})
                    event_meta[frame_id] = {
                        "selection_score": child_meta.get("selection_score", max(0.0, 1.0 - rank / max(len(frames), 1))),
                        "selection_reason": "inherited_representative",
                        "selection_rank": rank,
                        "derived_from_event_id": source_child["event_id"] if source_child else None,
                        "derived_from_frame_atom_id": child_meta.get("derived_from_frame_atom_id") or frame_id,
                    }
                representative_meta[event["event_id"]] = event_meta


def build_event_embeddings(config: AppConfig, run_root: Path, *, output_mode: str, run_id: Optional[str], day: Optional[str] = None, source: Optional[str] = None, resume: bool = False) -> dict:
    paths = stage_paths(run_root)
    visual_path = run_root / "embeddings" / "event_visual_centroids.parquet"
    dense_path = run_root / "embeddings" / "event_text_dense_centroids.parquet"
    sparse_path = run_root / "embeddings" / "event_text_sparse_aggregates.parquet"
    representative_meta_path = run_root / "links" / "representative_frame_selection_meta.json"
    params = {"stage": STAGE, "output_mode": output_mode, "run_id": run_id, "day": day, "source": source}
    if (
        resume
        and stage_complete(paths.stage_manifest(STAGE), params=params, output_paths=[visual_path, dense_path, sparse_path, representative_meta_path])
        and _embedding_outputs_cover(run_root, _current_segment_event_ids(run_root))
    ):
        return {"stage": STAGE, "skipped": True}

    event_tables = []
    all_events = []
    for name, rel, schema in SEGMENT_FILES:
        path = run_root / rel
        rows = read_pylist(path, batch_size=config.run.batch_size) if path.exists() else []
        event_tables.append((path, schema, rows))
        all_events.extend(rows)

    frame_meta, indexed_frames = _load_frame_meta(run_root, config.run.batch_size)
    visual_scores = _load_visual_scores(run_root, config.run.batch_size)
    event_frame_rows: dict[str, list[dict]] = {}
    sampled_frame_ids: dict[str, list[str]] = {}
    representative_meta: dict[str, dict[str, dict]] = {}
    needed_frame_ids: set[str] = set()
    for event in all_events:
        rows = _frames_for_event(event, indexed_frames)
        for row in rows:
            row.update(visual_scores.get(row["frame_atom_id"], {}))
        event_frame_rows[event["event_id"]] = rows
        sample_ids = deterministic_uniform_sample([row["frame_atom_id"] for row in rows], config.run.max_event_frames) if rows else []
        sampled_frame_ids[event["event_id"]] = sample_ids
        needed_frame_ids.update(sample_ids)
        if event["level"] in {"L1", "L2"}:
            needed_frame_ids.update(row["frame_atom_id"] for row in rows)

    frame_embeddings = _load_embeddings_by_frame(config, frame_meta, needed_frame_ids)
    visual_centroids: dict[str, Optional[list[float]]] = {}
    for event in all_events:
        vectors = [frame_embeddings[frame_id] for frame_id in sampled_frame_ids[event["event_id"]] if frame_id in frame_embeddings]
        visual_centroids[event["event_id"]] = _centroid(vectors)

    for event in all_events:
        if event["level"] not in {"L1", "L2"}:
            continue
        centroid_list = visual_centroids.get(event["event_id"])
        centroid = np.asarray(centroid_list, dtype=np.float32) if centroid_list is not None else None
        selected, selected_meta = select_representative_frames(
            event_frame_rows[event["event_id"]],
            frame_embeddings,
            centroid,
            start_second=float(event["start_second"]),
            end_second=float(event["end_second"]),
            level=event["level"],
            max_count=_level_limit(event["level"]),
        )
        if selected:
            event["representative_frame_ids"] = selected
            representative_meta[event["event_id"]] = {
                frame_id: {
                    **selected_meta.get(frame_id, {}),
                    "derived_from_event_id": None,
                    "derived_from_frame_atom_id": frame_id,
                }
                for frame_id in selected
            }

    events_by_id = {event["event_id"]: event for event in all_events}
    _propagate_high_level_representatives(events_by_id, representative_meta)

    text_meta = _load_text_meta(run_root, config.run.batch_size)
    all_text_ids = {text_id for event in all_events for text_id in (event.get("linked_text_atom_ids") or [])}
    dense_embeddings = _load_dense(config, text_meta, all_text_ids)
    sparse_embeddings = _load_sparse(config, text_meta, all_text_ids)
    param_hash = params_hash(params)
    input_manifest_ref = to_ref(paths.stage_manifest("aggregate_l5"), config.project_root)

    visual_rows = []
    dense_rows = []
    sparse_rows = []
    refs = {}
    for event in all_events:
        event_id = event["event_id"]
        text_weights = _text_overlap_weights(event, text_meta)
        dense_vectors = []
        dense_weights = []
        sparse_vectors = []
        for text_id, weight in text_weights.items():
            if text_id in dense_embeddings:
                dense_vectors.append(dense_embeddings[text_id])
                dense_weights.append(weight)
            if text_id in sparse_embeddings:
                sparse_vectors.append((sparse_embeddings[text_id], weight))
        dense = _centroid(dense_vectors, dense_weights)
        sparse_ids, sparse_values = _sparse_sum(sparse_vectors, config.run.sparse_top_n)
        visual = visual_centroids.get(event_id)
        visual_ref = f"event_visual_centroids:{event_id}" if visual is not None else None
        dense_ref = f"event_text_dense_centroids:{event_id}" if dense is not None else None
        sparse_ref = f"event_text_sparse_aggregates:{event_id}" if sparse_ids else None
        refs[event_id] = (visual_ref, dense_ref, sparse_ref)
        visual_rows.append(
            {
                "event_id": event_id,
                "level": event["level"],
                "day": event["day"],
                "source": event.get("source"),
                "embedding_ref": visual_ref,
                "embedding": visual,
                "source_item_ids": sampled_frame_ids.get(event_id, []),
                "aggregation_method": f"canonical_frame_mean_l2_sampled_{len(sampled_frame_ids.get(event_id, []))}",
                "quality_flags": [] if visual is not None else ["missing_visual_embedding"],
                "created_by": STAGE,
                "params_hash": param_hash,
                "input_manifest_ref": input_manifest_ref,
            }
        )
        dense_rows.append(
            {
                "event_id": event_id,
                "level": event["level"],
                "day": event["day"],
                "source": event.get("source"),
                "embedding_ref": dense_ref,
                "embedding": dense,
                "source_item_ids": list(text_weights),
                "aggregation_method": "overlap_duration_weighted_text_mean_l2",
                "quality_flags": [] if dense is not None else ["missing_text_embedding"],
                "created_by": STAGE,
                "params_hash": param_hash,
                "input_manifest_ref": input_manifest_ref,
            }
        )
        sparse_rows.append(
            {
                "event_id": event_id,
                "level": event["level"],
                "day": event["day"],
                "source": event.get("source"),
                "embedding_ref": sparse_ref,
                "sparse_vocab_ids": sparse_ids,
                "sparse_values": sparse_values,
                "aggregation_method": "overlap_duration_weighted_sum_top_n",
                "quality_flags": [] if sparse_ids else ["missing_sparse_embedding"],
                "created_by": STAGE,
                "params_hash": param_hash,
                "input_manifest_ref": input_manifest_ref,
            }
        )

    for path, schema, rows in event_tables:
        for row in rows:
            updated = events_by_id[row["event_id"]]
            row.update(updated)
            visual_ref, dense_ref, sparse_ref = refs.get(row["event_id"], (None, None, None))
            row["visual_centroid_ref"] = visual_ref
            row["text_dense_centroid_ref"] = dense_ref
            row["text_sparse_aggregate_ref"] = sparse_ref
            support = set(row.get("modality_support") or [])
            if visual_ref:
                support.add("visual")
            if dense_ref or sparse_ref:
                support.add("text")
            row["modality_support"] = sorted(support)
        write_table(path, rows, schema=schema)

    write_table(visual_path, visual_rows, schema=EVENT_EMBEDDINGS)
    write_table(dense_path, dense_rows, schema=EVENT_EMBEDDINGS)
    write_table(sparse_path, sparse_rows, schema=SPARSE_AGGREGATES)
    atomic_write_json(representative_meta_path, representative_meta)
    manifest = write_stage_manifest(
        paths.stage_manifest(STAGE),
        stage=STAGE,
        params=params,
        inputs=[
            *[file_record(path, config.project_root) for path, _, _ in event_tables],
            file_record(run_root / "atoms" / "frame_atoms.parquet", config.project_root),
            file_record(run_root / "atoms" / "text_atoms.parquet", config.project_root),
            file_record(run_root / "features" / "visual_motion_features.parquet", config.project_root),
        ],
        outputs=[
            {**file_record(visual_path, config.project_root), "schema": str(EVENT_EMBEDDINGS)},
            {**file_record(dense_path, config.project_root), "schema": str(EVENT_EMBEDDINGS)},
            {**file_record(sparse_path, config.project_root), "schema": str(SPARSE_AGGREGATES)},
            file_record(representative_meta_path, config.project_root),
        ],
        schema_names=["event_visual_centroids", "event_text_dense_centroids", "event_text_sparse_aggregates"],
        project_root=config.project_root,
    )
    segment_paths = [path for path, _, _ in event_tables]
    refreshed_paths = [*segment_paths, visual_path, dense_path, sparse_path, representative_meta_path]
    for previous_stage in ("segments_l1_l2", "group_l3", "aggregate_l4", "aggregate_l5", "aggregate_l4_l5"):
        refresh_manifest_file_records(paths.stage_manifest(previous_stage), project_root=config.project_root, paths=refreshed_paths)
    update_run_state(run_root, STAGE, to_ref(paths.stage_manifest(STAGE), config.project_root))
    return {"stage": STAGE, "manifest": manifest}
