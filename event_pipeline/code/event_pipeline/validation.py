from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pyarrow.parquet as pq

from .config import AppConfig
from .intervals import overlap_duration, union_intervals
from .parquet_io import atomic_write_json, read_pylist
from .schema import schema_for
from .source_meta import source_roles

SEGMENT_FILES = [
    ("l1_micro_events", "segments/l1_micro_events.parquet"),
    ("l2_source_events", "segments/l2_source_events.parquet"),
    ("l3_global_events", "segments/l3_global_events.parquet"),
    ("l4_activity_episodes", "segments/l4_activity_episodes.parquet"),
    ("l5_sessions", "segments/l5_sessions.parquet"),
]

REQUIRED_OUTPUTS = [
    "manifests/input_data_manifest.json",
    "inventory/available_streams.parquet",
    "inventory/source_hour_coverage.parquet",
    "atoms/frame_atoms.parquet",
    "atoms/canonical_frame_timeline.parquet",
    "atoms/text_atoms.parquet",
    "atoms/text_atom_time_fragments.parquet",
    "boundaries/visual_boundary_scores.parquet",
    "boundaries/text_boundary_scores.parquet",
    "boundaries/fused_boundary_scores.parquet",
    "features/visual_motion_features.parquet",
    "features/text_topic_features.parquet",
    "features/sparse_keyword_features.parquet",
    "segments/l1_micro_events.parquet",
    "segments/l2_source_events.parquet",
    "segments/l3_global_events.parquet",
    "segments/l4_activity_episodes.parquet",
    "segments/l5_sessions.parquet",
    "segments/source_timeline_coverage.parquet",
    "embeddings/event_visual_centroids.parquet",
    "embeddings/event_text_dense_centroids.parquet",
    "embeddings/event_text_sparse_aggregates.parquet",
    "links/event_frame_links.parquet",
    "links/event_text_links.parquet",
    "links/event_embedding_links.parquet",
    "links/event_parent_child_links.parquet",
    "links/cross_source_links.parquet",
    "indexes/event_corpus_index.parquet",
    "indexes/event_corpus_index.jsonl",
]


def _read(path: Path) -> list[dict]:
    return read_pylist(path) if path.exists() else []


def _schema_name_for_rel(rel: str) -> Optional[str]:
    stem = Path(rel).stem
    return stem if stem != "event_corpus_index" or rel.endswith(".parquet") else None


def _check_schema(run_root: Path, rel: str, errors: list[str]) -> None:
    schema_name = _schema_name_for_rel(rel)
    if schema_name is None or not rel.endswith(".parquet"):
        return
    path = run_root / rel
    if not path.exists():
        return
    try:
        actual = pq.ParquetFile(path).schema_arrow
        expected = schema_for(schema_name)
    except Exception as exc:
        errors.append(f"schema unreadable: {rel}: {exc}")
        return
    if actual.names != expected.names:
        errors.append(f"schema fields mismatch: {rel}")
        return
    for name in expected.names:
        actual_type = str(actual.field(name).type).replace("element:", "item:")
        expected_type = str(expected.field(name).type).replace("element:", "item:")
        if actual_type != expected_type:
            errors.append(f"schema type mismatch: {rel}.{name}: {actual_type} != {expected_type}")


def _event_id_time_matches(event: dict) -> bool:
    source = "global" if event.get("source") is None else event.get("source")
    needle = f"/{source}/{int(round(float(event['start_second'])))}-{int(round(float(event['end_second'])))}-"
    return needle in event["event_id"]


def _event_interval(event: dict) -> tuple[float, float]:
    return float(event["start_second"]), float(event["end_second"])


def _load_embedding_refs(run_root: Path) -> set[str]:
    refs = set()
    for rel in (
        "embeddings/event_visual_centroids.parquet",
        "embeddings/event_text_dense_centroids.parquet",
        "embeddings/event_text_sparse_aggregates.parquet",
    ):
        for row in _read(run_root / rel):
            if row.get("embedding_ref"):
                refs.add(row["embedding_ref"])
    return refs


def _validate_l2_overlap(l2_rows: list[dict], errors: list[str]) -> None:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in l2_rows:
        grouped[(row["day"], row["source"])].append(row)
    for key, rows in grouped.items():
        rows.sort(key=lambda row: float(row["start_second"]))
        for left, right in zip(rows, rows[1:]):
            overlap = overlap_duration(_event_interval(left), _event_interval(right))
            if overlap > 1.0:
                errors.append(f"same-source L2 overlap too large: {key[0]}/{key[1]} {left['event_id']} {right['event_id']}")


def _validate_parent_chain(events: list[dict], errors: list[str], warnings: list[str]) -> None:
    by_id = {event["event_id"]: event for event in events}
    for event in events:
        if event["level"] == "L5":
            continue
        parent_id = event.get("primary_parent_id")
        if not parent_id:
            errors.append(f"event missing primary parent: {event['event_id']}")
            continue
        parent = by_id.get(parent_id)
        if parent is None:
            errors.append(f"primary parent does not resolve: {event['event_id']} -> {parent_id}")
            continue
        child_interval = _event_interval(event)
        parent_interval = _event_interval(parent)
        if overlap_duration(child_interval, parent_interval) <= 0:
            warnings.append(f"primary parent does not temporally overlap child: {event['event_id']} -> {parent_id}")
            continue
        if child_interval[0] < parent_interval[0] - 1e-6 or child_interval[1] > parent_interval[1] + 1e-6:
            errors.append(f"primary parent does not contain child: {event['event_id']} -> {parent_id}")


def _validate_representatives(config: AppConfig, run_root: Path, events: list[dict], errors: list[str]) -> None:
    frame_atoms = {row["frame_atom_id"]: row for row in _read(run_root / "atoms" / "frame_atoms.parquet")}
    for event in events:
        if event.get("event_kind") not in {"missing", "text_only_context"} and not event.get("representative_frame_ids") and not event.get("linked_text_atom_ids"):
            errors.append(f"event has no frame or text evidence: {event['event_id']}")
        for frame_id in event.get("representative_frame_ids") or []:
            frame = frame_atoms.get(frame_id)
            if frame is None:
                errors.append(f"representative frame does not resolve: {event['event_id']} -> {frame_id}")
                continue
            if not frame.get("is_canonical_frame"):
                errors.append(f"representative frame is not canonical: {frame_id}")
            if not frame.get("tipsv2_embedding_id") or not frame.get("tipsv2_parquet_ref"):
                errors.append(f"representative frame lacks embedding ref: {frame_id}")
            image_path = config.input_roots.processed_data_root.parent / str(frame.get("frame_rel_path", "")).replace("\\", "/")
            if not image_path.exists():
                errors.append(f"representative image path does not exist: {frame_id}")
            tips_path = config.project_root / frame["tipsv2_parquet_ref"] if frame.get("tipsv2_parquet_ref") else None
            if tips_path is None or not tips_path.exists():
                errors.append(f"representative embedding parquet does not exist: {frame_id}")


def _validate_frames(run_root: Path, errors: list[str]) -> None:
    frame_rows = _read(run_root / "atoms" / "frame_atoms.parquet")
    join_keys = set()
    canonical_keys = set()
    for row in frame_rows:
        key = row.get("tipsv2_row_key")
        if key in join_keys:
            errors.append(f"duplicate frame join key: {key}")
        join_keys.add(key)
        if row.get("is_canonical_frame"):
            ckey = (row["day"], row["source"], float(row["canonical_second"]))
            if ckey in canonical_keys:
                errors.append(f"duplicate canonical frame key: {ckey}")
            canonical_keys.add(ckey)
    timeline_rows = _read(run_root / "atoms" / "canonical_frame_timeline.parquet")
    last_by_group = {}
    for row in sorted(timeline_rows, key=lambda r: (r["day"], r["source"], float(r["canonical_second"]))):
        key = (row["day"], row["source"])
        previous = last_by_group.get(key)
        if previous is not None and float(row["canonical_second"]) <= previous:
            errors.append(f"canonical timeline is not strictly increasing: {key}")
        last_by_group[key] = float(row["canonical_second"])


def _validate_text_boundaries(run_root: Path, errors: list[str]) -> None:
    text_rows = {row["text_atom_id"]: row for row in _read(run_root / "atoms" / "text_atoms.parquet")}
    for rel in ("boundaries/text_boundary_scores.parquet", "features/sparse_keyword_features.parquet"):
        for row in _read(run_root / rel):
            text = text_rows.get(row.get("text_atom_id"))
            if text and text.get("time_alignment_status") == "invalid":
                errors.append(f"invalid text atom used in boundary scoring: {row['text_atom_id']}")
    for row in text_rows.values():
        intervals = json.loads(row.get("union_time_intervals") or "[]")
        duration = sum(end - start for start, end in union_intervals((float(a), float(b)) for a, b in intervals))
        if intervals and abs(duration - float(row.get("aligned_duration_sec") or 0.0)) > 1e-4:
            errors.append(f"text aligned_duration_sec is not union duration: {row['text_atom_id']}")


def _validate_links(run_root: Path, events: list[dict], errors: list[str]) -> None:
    event_ids = {event["event_id"] for event in events}
    events_by_id = {event["event_id"]: event for event in events}
    frame_ids = {row["frame_atom_id"] for row in _read(run_root / "atoms" / "frame_atoms.parquet")}
    text_ids = {row["text_atom_id"] for row in _read(run_root / "atoms" / "text_atoms.parquet")}
    embedding_refs = _load_embedding_refs(run_root)
    representative_links = set()
    for row in _read(run_root / "links" / "event_frame_links.parquet"):
        if row["event_id"] not in event_ids or row["target_id"] not in frame_ids:
            errors.append(f"event_frame_link does not resolve: {row['link_id']}")
        if row.get("frame_atom_id") != row.get("target_id"):
            errors.append(f"event_frame_link frame_atom_id mismatch: {row['link_id']}")
        if row.get("is_representative") and row.get("selection_score") is None:
            errors.append(f"representative frame link lacks selection_score: {row['link_id']}")
        if row.get("is_representative"):
            representative_links.add((row["event_id"], row["target_id"]))
            if not row.get("selection_reason"):
                errors.append(f"representative frame link lacks selection_reason: {row['link_id']}")
            event = events_by_id.get(row["event_id"])
            if event and event.get("level") in {"L3", "L4", "L5"} and not row.get("derived_from_event_id"):
                errors.append(f"inherited representative lacks derived_from_event_id: {row['link_id']}")
    for row in _read(run_root / "links" / "event_text_links.parquet"):
        if row["event_id"] not in event_ids or row["target_id"] not in text_ids:
            errors.append(f"event_text_link does not resolve: {row['link_id']}")
        if row.get("text_atom_id") != row.get("target_id"):
            errors.append(f"event_text_link text_atom_id mismatch: {row['link_id']}")
    for row in _read(run_root / "links" / "event_embedding_links.parquet"):
        if row["event_id"] not in event_ids or row["target_id"] not in embedding_refs:
            errors.append(f"event_embedding_link does not resolve: {row['link_id']}")
    parent_pairs = set()
    for row in _read(run_root / "links" / "event_parent_child_links.parquet"):
        if row["event_id"] not in event_ids or row["target_id"] not in event_ids:
            errors.append(f"event_parent_child_link does not resolve: {row['link_id']}")
        pair = (row.get("event_id"), row.get("target_id"))
        if pair in parent_pairs:
            errors.append(f"duplicate parent-child link pair: {pair[0]} -> {pair[1]}")
        parent_pairs.add(pair)
    for event in events:
        for frame_id in event.get("representative_frame_ids") or []:
            if (event["event_id"], frame_id) not in representative_links:
                errors.append(f"representative frame missing event_frame_link: {event['event_id']} -> {frame_id}")
    cross_links = _read(run_root / "links" / "cross_source_links.parquet")
    cross_by_id = {row["link_id"]: row for row in cross_links}
    for row in cross_links:
        left = events_by_id.get(row["event_id"])
        right = events_by_id.get(row["target_id"])
        if left is None or right is None:
            errors.append(f"cross_source_link does not resolve: {row['link_id']}")
            continue
        if left.get("source") == right.get("source"):
            errors.append(f"cross_source_link connects same source: {row['link_id']}")
        if row.get("link_role") == "strong" and float(row.get("score") or 0.0) < 0.70:
            errors.append(f"strong cross_source_link below threshold: {row['link_id']}")
        if row.get("link_role") == "strong" and row.get("quality_flags"):
            errors.append(f"strong cross_source_link has blocking quality flags: {row['link_id']}")
    for event in events:
        if event.get("level") != "L3":
            continue
        child_ids = set(event.get("child_source_event_ids") or [])
        for link_id in event.get("cross_source_link_ids") or []:
            link = cross_by_id.get(link_id)
            if link is None:
                errors.append(f"L3 cross_source_link_id does not resolve: {event['event_id']} -> {link_id}")
                continue
            endpoints = {link.get("event_id"), link.get("target_id")}
            if not endpoints <= child_ids:
                errors.append(f"L3 cross_source_link_id points outside cluster: {event['event_id']} -> {link_id}")


def _validate_embeddings(run_root: Path, events: list[dict], errors: list[str]) -> None:
    refs = _load_embedding_refs(run_root)
    for event in events:
        for field in ("visual_centroid_ref", "text_dense_centroid_ref", "text_sparse_aggregate_ref"):
            ref = event.get(field)
            if ref and ref not in refs:
                errors.append(f"event embedding ref does not resolve: {event['event_id']} {ref}")
        if not event.get("visual_centroid_ref") and "visual" in (event.get("modality_support") or []):
            errors.append(f"event claims visual support without visual centroid: {event['event_id']}")


def _input_manifest(run_root: Path) -> dict:
    path = run_root / "manifests" / "input_data_manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _validate_formal_scope(config: AppConfig, run_root: Path, event_rows_by_name: dict[str, list[dict]], inventory_coverage_rows: list[dict], l2_coverage_rows: list[dict], errors: list[str]) -> None:
    if not config.is_formal_run_root(run_root):
        return
    manifest = _input_manifest(run_root)
    filters = manifest.get("filters") or {}
    if filters.get("day") or filters.get("source"):
        errors.append("formal validation requires an unfiltered run; use smoke mode for day/source subsets")
        return
    days = {row["day"] for row in inventory_coverage_rows}
    missing_days = {f"day{index}" for index in range(1, 5)} - days
    if missing_days:
        errors.append(f"formal coverage missing required days: {sorted(missing_days)}")
    video_sources = {
        (row["day"], row["source"])
        for row in inventory_coverage_rows
        if row.get("frames_available") and row.get("tipsv2_available")
    }
    l2_coverage_sources = {(row["day"], row["source"]) for row in l2_coverage_rows}
    missing_l2_coverage = sorted(video_sources - l2_coverage_sources)
    if missing_l2_coverage:
        errors.append(f"formal video source/day missing L2 coverage rows: {missing_l2_coverage[:20]}")
    if len({source for _, source in video_sources}) < 2:
        errors.append("formal run must include multiple video sources")
    if video_sources and len({source for _, source in video_sources}) >= 2:
        if not event_rows_by_name.get("l3_global_events"):
            errors.append("formal multi-source run produced no L3 global events")
        cross_links_path = run_root / "links" / "cross_source_links.parquet"
        cross_count = len(_read(cross_links_path)) if cross_links_path.exists() else 0
        if cross_count == 0:
            errors.append("formal multi-source run produced no cross-source links")


def validate_run(config: AppConfig, run_root: Path) -> dict:
    errors: list[str] = []
    warnings: list[str] = []

    for rel in REQUIRED_OUTPUTS:
        if not (run_root / rel).exists():
            errors.append(f"missing output: {rel}")
        else:
            _check_schema(run_root, rel, errors)

    events = []
    event_rows_by_name = {}
    for name, rel in SEGMENT_FILES:
        rows = _read(run_root / rel)
        event_rows_by_name[name] = rows
        events.extend(rows)

    role_map = source_roles(config.input_roots.view_meta_path)
    for event in events:
        start = float(event["start_second"])
        end = float(event["end_second"])
        if not (0 <= start < end <= 86400):
            errors.append(f"invalid time range: {event['event_id']}")
        if event["level"] in {"L1", "L2"} and not _event_id_time_matches(event):
            errors.append(f"event_id time does not match event bounds: {event['event_id']}")
        if event["level"] == "L1" and end - start > 30:
            errors.append(f"L1 exceeds 30 seconds: {event['event_id']}")
        if event["level"] == "L2":
            role = event.get("source_role") or (role_map.get(event["source"]) or {}).get("source_role")
            min_sec = config.segmentation.fixed_room_l2_min_sec if role == "fixed_room" else config.segmentation.ego_person_l2_min_sec
            if end - start < min_sec and "below_min_duration_after_merge" not in (event.get("quality_flags") or []):
                errors.append(f"L2 shorter than role minimum: {event['event_id']}")
            if end - start > config.segmentation.l2_max_sec + 1e-6 and "above_max_duration_after_merge" not in (event.get("quality_flags") or []):
                errors.append(f"L2 exceeds configured max duration: {event['event_id']}")
        if event["level"] in {"L3", "L4", "L5"}:
            if event.get("source") is not None:
                errors.append(f"high-level event source must be null: {event['event_id']}")
            if not event.get("sources"):
                errors.append(f"high-level event sources must be non-empty: {event['event_id']}")
        if event["level"] == "L3":
            if len(set(event.get("sources") or [])) < 2:
                errors.append(f"L3 must contain multiple sources: {event['event_id']}")
            if end - start > config.segmentation.l3_max_sec:
                errors.append(f"L3 exceeds max span: {event['event_id']}")
        if event["level"] == "L4" and event.get("event_kind") != "text_only_context" and end - start < 60:
            if "short_activity_episode_context" not in (event.get("quality_flags") or []):
                warnings.append(f"L4 below target minimum without short-context flag: {event['event_id']}")
        if event["level"] == "L5" and end - start < 600:
            if "short_session_context" not in (event.get("quality_flags") or []):
                warnings.append(f"L5 below target minimum without short-context flag: {event['event_id']}")

    coverage_rows = _read(run_root / "segments" / "source_timeline_coverage.parquet")
    for row in coverage_rows:
        if float(row.get("coverage_ratio") or 0.0) < 0.95 and "coverage_below_95_percent" not in (row.get("quality_flags") or []):
            errors.append(f"L2 coverage below target without explanation: {row['day']}/{row['source']}")
    _validate_l2_overlap(event_rows_by_name.get("l2_source_events", []), errors)
    _validate_parent_chain(events, errors, warnings)
    _validate_representatives(config, run_root, events, errors)
    _validate_frames(run_root, errors)
    _validate_text_boundaries(run_root, errors)
    _validate_embeddings(run_root, events, errors)
    _validate_links(run_root, events, errors)
    inventory_coverage_rows = _read(run_root / "inventory" / "source_hour_coverage.parquet")
    _validate_formal_scope(config, run_root, event_rows_by_name, inventory_coverage_rows, coverage_rows, errors)

    manifest_dir = run_root / "manifests"
    manifest_text = ""
    if manifest_dir.exists():
        for path in manifest_dir.glob("*.json"):
            manifest_text += path.read_text(encoding="utf-8", errors="replace")
    lowered = manifest_text.lower()
    if "retrieval_test" in lowered or "exploration" in lowered:
        errors.append("runtime manifest references retrieval_test or exploration")
    if "qdrant" in lowered or "sqlite fts" in lowered:
        errors.append("runtime manifest references a forbidden DB service")

    atomic_write_json(
        run_root / "manifests" / "run_manifest.json",
        {
            "run_root": run_root.as_posix(),
            "event_count": len(events),
            "coverage_rows": len(coverage_rows),
            "segment_files": [rel for _, rel in SEGMENT_FILES],
            "quality_outputs": [
                "quality/schema_validation.json",
                "quality/timeline_coverage_report.json",
                "quality/hierarchy_consistency_report.json",
                "quality/cross_source_grouping_report.json",
            ],
        },
    )

    report = {
        "valid": not errors,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings[:200],
        "event_count": len(events),
        "coverage_rows": len(coverage_rows),
    }
    quality = run_root / "quality"
    atomic_write_json(quality / "schema_validation.json", report)
    atomic_write_json(quality / "timeline_coverage_report.json", {"rows": coverage_rows, "errors": [e for e in errors if "coverage" in e.lower() or "time" in e.lower() or "timeline" in e.lower()]})
    atomic_write_json(quality / "hierarchy_consistency_report.json", {"warnings": warnings, "errors": [e for e in errors if "parent" in e.lower() or "source" in e.lower() or "link" in e.lower()]})
    atomic_write_json(
        quality / "cross_source_grouping_report.json",
        {
            "l3_count": len(_read(run_root / "segments" / "l3_global_events.parquet")),
            "cross_source_links": len(_read(run_root / "links" / "cross_source_links.parquet")) if (run_root / "links" / "cross_source_links.parquet").exists() else 0,
            "errors": [e for e in errors if "cross" in e.lower() or "l3" in e.lower()],
        },
    )
    return report
