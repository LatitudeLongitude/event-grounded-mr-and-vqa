from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from .config import AppConfig
from .filters import source_matches
from .ids import event_id
from .intervals import overlap_duration, union_intervals
from .manifest import file_record, params_hash, write_stage_manifest
from .parquet_io import iter_pylist_rows, write_table
from .paths import to_ref
from .run_state import stage_complete, stage_paths, update_run_state
from .schema import EVENT_TABLE, SOURCE_TIMELINE_COVERAGE
from .source_meta import source_roles
from .time_utils import seconds_to_hms

STAGE = "segments_l1_l2"


def _empty_labels() -> dict:
    return {
        "reserved_label_primary": None,
        "reserved_label_secondary": None,
        "reserved_label_freeform": None,
        "reserved_label_confidence": None,
    }


def _event_row(level: str, row_day: str, day_num: int, source: str, role: Optional[str], start: float, end: float, *, kind: str, confidence: float, frame_ids: list[str], text_ids: list[str], flags: list[str], param_hash: str, input_manifest_ref: str) -> dict:
    return {
        "event_id": event_id(level, row_day, source, start, end, [frame_ids, text_ids, confidence]),
        "level": level,
        "day": row_day,
        "day_num": day_num,
        "source": source,
        "sources": [source],
        "source_role": role,
        "source_roles": [role] if role else [],
        "start_second": float(start),
        "end_second": float(end),
        "start_hms": seconds_to_hms(start),
        "end_hms": seconds_to_hms(end),
        "duration_sec": float(end - start),
        "event_kind": kind,
        "primary_parent_id": None,
        "global_event_id": None,
        "representative_frame_ids": frame_ids,
        "linked_text_atom_ids": text_ids,
        "visual_centroid_ref": None,
        "text_dense_centroid_ref": None,
        "text_sparse_aggregate_ref": None,
        "modality_support": sorted(set((["visual"] if frame_ids else []) + (["text"] if text_ids else []))),
        "confidence": float(confidence),
        "confidence_level": "high" if confidence >= 1.0 else ("medium" if confidence >= 0.3 else "low"),
        "quality_flags": flags,
        "created_by": STAGE,
        "params_hash": param_hash,
        "input_manifest_ref": input_manifest_ref,
        **_empty_labels(),
    }


def _text_overlaps(text_rows: list[dict], start: float, end: float) -> list[str]:
    ids = []
    for row in text_rows:
        if row["time_alignment_status"] == "invalid":
            continue
        intervals = json.loads(row["union_time_intervals"] or "[]")
        if any(overlap_duration((start, end), tuple(interval)) > 0 for interval in intervals):
            ids.append(row["text_atom_id"])
    return ids


def _frames_in(frames: list[dict], start: float, end: float) -> list[dict]:
    return [row for row in frames if start <= float(row["canonical_second"]) < end]


def _representatives(frames: list[dict], max_count: int) -> list[str]:
    if not frames:
        return []
    if len(frames) <= max_count:
        chosen = frames
    else:
        indices = np.linspace(0, len(frames) - 1, max_count)
        chosen = [frames[int(round(i))] for i in indices]
    return [row["frame_atom_id"] for row in chosen]


def _merge_short_segments(segments: list[tuple[float, float]], min_sec: int, max_sec: Optional[int] = None) -> list[tuple[float, float]]:
    if not segments:
        return []
    work = sorted(segments)

    def can_merge(left: tuple[float, float], right: tuple[float, float]) -> bool:
        if max_sec is None:
            return True
        return right[1] - left[0] <= float(max_sec) + 1e-6

    changed = True
    while changed and len(work) > 1:
        changed = False
        for index, (start, end) in enumerate(work):
            if end - start >= min_sec:
                continue
            if index == 0:
                if can_merge((start, end), work[1]):
                    work[1] = (start, work[1][1])
                    del work[0]
                else:
                    continue
            elif index == len(work) - 1:
                if can_merge(work[index - 1], (start, end)):
                    work[index - 1] = (work[index - 1][0], end)
                    del work[index]
                else:
                    continue
            else:
                left_len = work[index - 1][1] - work[index - 1][0]
                right_len = work[index + 1][1] - work[index + 1][0]
                left_ok = can_merge(work[index - 1], (start, end))
                right_ok = can_merge((start, end), work[index + 1])
                if left_ok and (left_len <= right_len or not right_ok):
                    work[index - 1] = (work[index - 1][0], end)
                    del work[index]
                elif right_ok:
                    work[index + 1] = (start, work[index + 1][1])
                    del work[index]
                else:
                    continue
            changed = True
            break
    return work


def _split_overlong_with_cuts(start: float, end: float, max_sec: int, candidate_cuts: list[float]) -> list[tuple[float, float]]:
    if end - start <= max_sec:
        return [(start, end)]
    cuts = sorted(cut for cut in candidate_cuts if start < cut < end)
    pieces = []
    current = start
    while end - current > max_sec:
        target = current + max_sec
        usable = [cut for cut in cuts if current + 1.0 < cut <= target]
        cut = usable[-1] if usable else target
        pieces.append((current, cut))
        current = cut
    if current < end:
        pieces.append((current, end))
    return pieces



def _continuous_frame_intervals(frames: list[dict]) -> list[tuple[float, float]]:
    if not frames:
        return []
    seconds = [float(row["canonical_second"]) for row in frames]
    intervals = []
    start = seconds[0]
    previous = seconds[0]
    for second in seconds[1:]:
        if second - previous > 1.5:
            intervals.append((start, previous + 1.0))
            start = second
        previous = second
    intervals.append((start, previous + 1.0))
    return intervals


def _best_l2_parent(start: float, end: float, l2_rows: list[dict]) -> Optional[dict]:
    best_parent = None
    best_overlap = 0.0
    best_duration = 0.0
    for row in l2_rows:
        overlap = overlap_duration((start, end), (row["start_second"], row["end_second"]))
        if overlap <= 0:
            continue
        duration = float(row["end_second"]) - float(row["start_second"])
        if overlap > best_overlap + 1e-9 or (abs(overlap - best_overlap) <= 1e-9 and duration > best_duration):
            best_parent = row
            best_overlap = overlap
            best_duration = duration
    return best_parent


def _clip_to_parent(start: float, end: float, parent: dict) -> Optional[tuple[float, float]]:
    clipped_start = max(float(start), float(parent["start_second"]))
    clipped_end = min(float(end), float(parent["end_second"]))
    if clipped_end - clipped_start < 1.0:
        return None
    return clipped_start, clipped_end


def _build_l2_for_group(config: AppConfig, fused: list[dict], frames: list[dict], texts: list[dict], role: Optional[str], param_hash: str, input_manifest_ref: str) -> tuple[list[dict], dict]:
    fused.sort(key=lambda row: float(row["second"]))
    frames.sort(key=lambda row: float(row["canonical_second"]))
    if not frames:
        return [], {}
    scores = np.asarray([row["l2_boundary_score"] if row.get("l2_boundary_score") is not None else np.nan for row in fused], dtype=float)
    finite = scores[np.isfinite(scores)]
    threshold = float(np.percentile(finite, 90 if role == "fixed_room" else 85)) if finite.size else float("inf")
    visual_values = [float(row["visual_boundary_score"]) for row in fused if row.get("visual_boundary_score") is not None and np.isfinite(float(row["visual_boundary_score"]))]
    visual_support_threshold = float(np.percentile(visual_values, 50)) if visual_values else float("inf")

    def cut_allowed(row: dict) -> bool:
        visual = row.get("visual_boundary_score")
        temporal = row.get("temporal_structure_score")
        has_visual_support = visual is not None and float(visual) >= visual_support_threshold
        has_temporal_support = temporal is not None and float(temporal) > 0.0
        return has_visual_support or has_temporal_support

    candidate_cuts = {
        float(row["second"])
        for row in fused
        if row.get("l2_boundary_score") is not None and float(row["l2_boundary_score"]) >= threshold and cut_allowed(row)
    }
    segments = []
    for interval_start, interval_end in _continuous_frame_intervals(frames):
        cuts = sorted(cut for cut in candidate_cuts if interval_start < cut < interval_end)
        prev = interval_start
        for cut in cuts:
            if cut - prev >= 1:
                segments.append((prev, cut))
            prev = cut
        if interval_end - prev >= 1:
            segments.append((prev, interval_end))
    min_sec = config.segmentation.fixed_room_l2_min_sec if role == "fixed_room" else config.segmentation.ego_person_l2_min_sec
    merged = _merge_short_segments(segments, min_sec, config.segmentation.l2_max_sec)
    high_cuts = sorted(
        float(row["second"])
        for row in fused
        if row.get("l2_boundary_score") is not None and float(row["l2_boundary_score"]) >= threshold and cut_allowed(row)
    )
    final_segments = [
        part
        for seg in merged
        for part in _split_overlong_with_cuts(seg[0], seg[1], config.segmentation.l2_max_sec, high_cuts)
    ]
    final_segments = _merge_short_segments(final_segments, min_sec, config.segmentation.l2_max_sec)
    events = []
    covered = []
    for start, end in final_segments:
        event_frames = _frames_in(frames, start, end)
        event_texts = _text_overlaps(texts, start, end)
        mean_score = float(np.nanmean([row.get("l2_boundary_score") for row in fused if start <= float(row["second"]) < end and row.get("l2_boundary_score") is not None] or [0.0]))
        flags = [] if event_frames else ["missing_visual"]
        kind = "activity_candidate" if mean_score >= 0.3 else "unknown_activity"
        if mean_score < 0.05:
            kind = "background_idle"
            flags.append("low_information")
        if end - start < min_sec:
            flags.append("below_min_duration_after_merge")
        events.append(_event_row("L2", frames[0]["day"], frames[0]["day_num"], frames[0]["source"], role, start, end, kind=kind, confidence=mean_score, frame_ids=_representatives(event_frames, 6), text_ids=event_texts, flags=sorted(set(flags)), param_hash=param_hash, input_manifest_ref=input_manifest_ref))
        covered.append((start, end))
    covered_union = union_intervals(covered)
    video_duration = sum(interval_end - interval_start for interval_start, interval_end in _continuous_frame_intervals(frames))
    covered_sec = sum(b - a for a, b in covered_union)
    coverage = {
        "day": frames[0]["day"],
        "day_num": frames[0]["day_num"],
        "source": frames[0]["source"],
        "covered_video_sec": video_duration,
        "l2_covered_sec": covered_sec,
        "coverage_ratio": covered_sec / video_duration if video_duration else 0.0,
        "uncovered_gaps": "[]",
        "quality_flags": [] if video_duration == 0 or covered_sec / video_duration >= 0.95 else ["coverage_below_95_percent"],
        "created_by": STAGE,
        "params_hash": param_hash,
        "input_manifest_ref": input_manifest_ref,
    }
    return events, coverage


def _choose_l1_window(second: float, frames: list[dict], fused: list[dict]) -> tuple[float, float]:
    frame_start = float(frames[0]["canonical_second"])
    frame_end = float(frames[-1]["canonical_second"]) + 1.0
    score_by_second = {float(row["second"]): row for row in fused}
    for left, right in ((2, 3), (5, 5), (10, 10)):
        start = max(frame_start, second - left)
        end = min(frame_end, second + right)
        local = [
            row
            for sec, row in score_by_second.items()
            if start <= sec < end and row.get("visual_boundary_score") is not None
        ]
        if len(local) <= 1:
            return start, end
        values = [float(row["visual_boundary_score"]) for row in local]
        if max(values) - min(values) <= max(0.5, abs(max(values)) * 0.75):
            return start, end
    return max(frame_start, second - 10), min(frame_end, second + 10)


def _build_l1_for_group(fused: list[dict], frames: list[dict], texts: list[dict], role: Optional[str], param_hash: str, input_manifest_ref: str, l2_rows: Optional[list[dict]] = None) -> list[dict]:
    if not frames:
        return []
    finite = [float(row["l1_boundary_score"]) for row in fused if row.get("l1_boundary_score") is not None and row.get("visual_boundary_score") is not None]
    if not finite:
        return []
    threshold = float(np.percentile(finite, 95))
    visual_values = [float(row["visual_boundary_score"]) for row in fused if row.get("visual_boundary_score") is not None]
    visual_threshold = float(np.percentile(visual_values, 90)) if visual_values else float("inf")
    candidates: list[tuple[float, float, float, Optional[str]]] = []
    for row in sorted(fused, key=lambda item: float(item["second"])):
        if row.get("visual_boundary_score") is None or row.get("l1_boundary_score") is None:
            continue
        if float(row["l1_boundary_score"]) < threshold:
            continue
        if float(row["visual_boundary_score"]) < visual_threshold:
            continue
        second = float(row["second"])
        start, end = _choose_l1_window(second, frames, fused)
        parent_id = None
        if l2_rows is not None:
            parent = _best_l2_parent(start, end, l2_rows)
            if parent is None:
                continue
            clipped = _clip_to_parent(start, end, parent)
            if clipped is None:
                continue
            start, end = clipped
            parent_id = parent["event_id"]
        candidates.append((start, end, float(row["l1_boundary_score"]), parent_id))

    merged: list[tuple[float, float, float, Optional[str]]] = []
    for start, end, score, parent_id in candidates:
        if merged and parent_id == merged[-1][3] and start <= merged[-1][1] + 3.0 and max(end, merged[-1][1]) - merged[-1][0] <= 30.0:
            previous_start, previous_end, previous_score, previous_parent_id = merged[-1]
            if abs(score - previous_score) <= max(abs(previous_score), abs(score), 1.0):
                merged[-1] = (previous_start, max(previous_end, end), max(previous_score, score), previous_parent_id)
                continue
        merged.append((start, min(end, start + 30.0), score, parent_id))

    events = []
    for start, end, score, parent_id in merged:
        event_frames = _frames_in(frames, start, end)
        event_texts = _text_overlaps(texts, start, end)
        row = _event_row("L1", frames[0]["day"], frames[0]["day_num"], frames[0]["source"], role, start, end, kind="activity_candidate", confidence=score, frame_ids=_representatives(event_frames, 3), text_ids=event_texts, flags=[], param_hash=param_hash, input_manifest_ref=input_manifest_ref)
        row["primary_parent_id"] = parent_id
        events.append(row)
    return events


def build_source_segments(config: AppConfig, run_root: Path, *, output_mode: str, run_id: Optional[str], day: Optional[str] = None, source: Optional[str] = None, resume: bool = False) -> dict:
    paths = stage_paths(run_root)
    fused_path = run_root / "boundaries" / "fused_boundary_scores.parquet"
    frames_path = run_root / "atoms" / "canonical_frame_timeline.parquet"
    text_path = run_root / "atoms" / "text_atoms.parquet"
    l1_path = run_root / "segments" / "l1_micro_events.parquet"
    l2_path = run_root / "segments" / "l2_source_events.parquet"
    coverage_path = run_root / "segments" / "source_timeline_coverage.parquet"
    params = {"stage": STAGE, "output_mode": output_mode, "run_id": run_id, "day": day, "source": source}
    if resume and stage_complete(paths.stage_manifest(STAGE), params=params, output_paths=[l1_path, l2_path, coverage_path]):
        return {"stage": STAGE, "skipped": True}

    role_map = source_roles(config.input_roots.view_meta_path)
    fused_by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in iter_pylist_rows(fused_path, batch_size=config.run.batch_size):
        fused_by_group[(row["day"], row["source"])].append(row)
    frames_by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in iter_pylist_rows(frames_path, batch_size=config.run.batch_size):
        frames_by_group[(row["day"], row["source"])].append(row)
    texts_by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in iter_pylist_rows(text_path, batch_size=config.run.batch_size):
        texts_by_group[(row["day"], row["source"])].append(row)

    param_hash = params_hash(params)
    input_manifest_ref = to_ref(paths.stage_manifest("fusion"), config.project_root)
    l1_rows: list[dict] = []
    l2_rows: list[dict] = []
    coverage_rows: list[dict] = []
    for key, fused_rows in sorted(fused_by_group.items()):
        found_day, found_source = key
        if day and found_day != ("day" + str(day).replace("day", "")):
            continue
        if not source_matches(found_source, source):
            continue
        frames = frames_by_group.get(key, [])
        texts = texts_by_group.get(key, [])
        role = (role_map.get(found_source) or {}).get("source_role")
        l2_group, coverage = _build_l2_for_group(config, fused_rows, frames, texts, role, param_hash, input_manifest_ref)
        l2_rows.extend(l2_group)
        l1_rows.extend(_build_l1_for_group(fused_rows, frames, texts, role, param_hash, input_manifest_ref, l2_group))
        if coverage:
            coverage_rows.append(coverage)


    write_table(l1_path, l1_rows, schema=EVENT_TABLE)
    write_table(l2_path, l2_rows, schema=EVENT_TABLE)
    write_table(coverage_path, coverage_rows, schema=SOURCE_TIMELINE_COVERAGE)
    manifest = write_stage_manifest(
        paths.stage_manifest(STAGE),
        stage=STAGE,
        params=params,
        inputs=[file_record(fused_path, config.project_root), file_record(frames_path, config.project_root), file_record(text_path, config.project_root)],
        outputs=[
            {**file_record(l1_path, config.project_root), "schema": str(EVENT_TABLE)},
            {**file_record(l2_path, config.project_root), "schema": str(EVENT_TABLE)},
            {**file_record(coverage_path, config.project_root), "schema": str(SOURCE_TIMELINE_COVERAGE)},
        ],
        schema_names=["l1_micro_events", "l2_source_events", "source_timeline_coverage"],
        project_root=config.project_root,
    )
    update_run_state(run_root, STAGE, to_ref(paths.stage_manifest(STAGE), config.project_root))
    return {"stage": STAGE, "manifest": manifest}
