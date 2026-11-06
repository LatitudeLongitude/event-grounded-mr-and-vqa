from __future__ import annotations

import json
import math
import heapq
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

import numpy as np
from .config import AppConfig
from .ids import event_id
from .manifest import file_record, params_hash, write_stage_manifest
from .normalization import weighted_mean_available
from .parquet_io import iter_pylist_rows, read_column_pylist, read_pylist, write_table
from .paths import to_ref
from .run_state import stage_complete, stage_paths, update_run_state
from .schema import EVENT_TABLE, L3_EVENT_TABLE, L4_L5_EVENT_TABLE
from .source_meta import location_hint_for_source, source_roles
from .time_utils import seconds_to_hms
from .vector_utils import cosine, weighted_jaccard

STAGE = "aggregate_l4_l5"


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
            set(read_column_pylist(path, "event_id"))
            for path in paths
        ]
    except Exception:
        return False
    return all(event_ids <= seen for seen in seen_by_output)


def _ensure_event_embeddings(config: AppConfig, run_root: Path, event_ids: set[str], *, output_mode: str, run_id: Optional[str], day: Optional[str], source: Optional[str]) -> None:
    if _embedding_outputs_cover(run_root, event_ids):
        return
    from .event_embeddings import build_event_embeddings

    build_event_embeddings(config, run_root, output_mode=output_mode, run_id=run_id, day=day, source=source, resume=False)


def _load_event_features_from_outputs(run_root: Path, event_ids: set[str]) -> dict[str, dict]:
    features = {event_id: {"visual": None, "text": None, "sparse": {}} for event_id in event_ids}
    visual_path = run_root / "embeddings" / "event_visual_centroids.parquet"
    dense_path = run_root / "embeddings" / "event_text_dense_centroids.parquet"
    sparse_path = run_root / "embeddings" / "event_text_sparse_aggregates.parquet"
    if visual_path.exists():
        remaining = set(event_ids)
        for row in iter_pylist_rows(visual_path, columns=["event_id", "embedding"]):
            if row["event_id"] in features and row.get("embedding") is not None:
                features[row["event_id"]]["visual"] = np.asarray(row["embedding"], dtype=np.float32)
                remaining.discard(row["event_id"])
                if not remaining:
                    break
    if dense_path.exists():
        remaining = set(event_ids)
        for row in iter_pylist_rows(dense_path, columns=["event_id", "embedding"]):
            if row["event_id"] in features and row.get("embedding") is not None:
                features[row["event_id"]]["text"] = np.asarray(row["embedding"], dtype=np.float32)
                remaining.discard(row["event_id"])
                if not remaining:
                    break
    if sparse_path.exists():
        remaining = set(event_ids)
        for row in iter_pylist_rows(sparse_path, columns=["event_id", "sparse_vocab_ids", "sparse_values"]):
            if row["event_id"] not in features:
                continue
            ids = row.get("sparse_vocab_ids") or []
            values = row.get("sparse_values") or []
            features[row["event_id"]]["sparse"] = {int(key): float(value) for key, value in zip(ids, values)}
            remaining.discard(row["event_id"])
            if not remaining:
                break
    return features


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    return len(left & right) / max(len(left | right), 1)


def _event_sources(row: dict) -> set[str]:
    return set(row.get("sources") or ([row["source"]] if row.get("source") else []))


def _event_roles(row: dict) -> set[str]:
    return set(row.get("source_roles") or ([row.get("source_role")] if row.get("source_role") else [])) - {None}


def _gap(a: dict, b: dict) -> float:
    return max(0.0, float(b["start_second"]) - float(a["end_second"]))


def _overlaps(a: dict, b: dict) -> bool:
    return float(a["start_second"]) < float(b["end_second"]) and float(b["start_second"]) < float(a["end_second"])


def _location_set(view_meta_path: Path, row: dict) -> set[str]:
    values = set()
    for source in _event_sources(row):
        hint = location_hint_for_source(view_meta_path, source)
        if hint:
            values.add(hint)
    return values


def _parallel_incompatible(view_meta_path: Path, left: dict, right: dict) -> bool:
    if not _overlaps(left, right):
        return False
    left_locations = _location_set(view_meta_path, left)
    right_locations = _location_set(view_meta_path, right)
    if left_locations and right_locations and left_locations.isdisjoint(right_locations):
        return True
    if _event_sources(left).isdisjoint(_event_sources(right)) and _event_roles(left) == {"fixed_room"} and _event_roles(right) == {"fixed_room"}:
        return True
    return False


def _safe_cosine(a, b) -> Optional[float]:
    if a is None or b is None:
        return None
    try:
        return max(0.0, min(1.0, (cosine(a, b) + 1.0) / 2.0))
    except ValueError:
        return None


def _mean_or_none(vectors: list[np.ndarray]) -> Optional[np.ndarray]:
    if not vectors:
        return None
    pooled = np.mean(np.stack(vectors), axis=0)
    norm = float(np.linalg.norm(pooled))
    return None if norm <= 0 else pooled / norm


def _feature_for_group(group: list[dict], features: Optional[dict[str, dict]]) -> dict:
    if not features:
        return {"visual": None, "text": None, "sparse": {}}
    visual = _mean_or_none([features[item["event_id"]]["visual"] for item in group if features.get(item["event_id"], {}).get("visual") is not None])
    text = _mean_or_none([features[item["event_id"]]["text"] for item in group if features.get(item["event_id"], {}).get("text") is not None])
    sparse: dict[int, float] = defaultdict(float)
    for item in group:
        for key, value in (features.get(item["event_id"], {}).get("sparse") or {}).items():
            sparse[int(key)] += float(value)
    return {"visual": visual, "text": text, "sparse": dict(sparse)}


def _merge_score(view_meta_path: Path, left_group: list[dict], right_group: Union[list[dict], dict], *, level: str, max_gap: float, features: Optional[dict[str, dict]] = None) -> float:
    if isinstance(right_group, dict):
        right_group = [right_group]
    left = _row_from_group(level, left_group, "score_only", "score_only")
    right = _row_from_group(level, right_group, "score_only", "score_only")
    gap = _gap(left, right)
    if gap > max_gap:
        return 0.0
    temporal_gap_score = math.exp(-gap / max(max_gap, 1.0))
    source_score = _jaccard(_event_sources(left), _event_sources(right))
    role_score = _jaccard(_event_roles(left), _event_roles(right))
    location_left = _location_set(view_meta_path, left)
    location_right = _location_set(view_meta_path, right)
    location_score = 0.0 if location_left and location_right and location_left.isdisjoint(location_right) else (1.0 if location_left & location_right else 0.5)
    source_role_continuity = max(source_score, 0.5 * role_score + 0.5 * location_score)
    left_features = _feature_for_group(left_group, features)
    right_features = _feature_for_group(right_group, features)
    text_topic_continuity = _safe_cosine(left_features.get("text"), right_features.get("text"))
    sparse_context_continuity = None
    if left_features.get("sparse") or right_features.get("sparse"):
        sparse_context_continuity = weighted_jaccard(left_features.get("sparse") or {}, right_features.get("sparse") or {})
    visual_continuity = _safe_cosine(left_features.get("visual"), right_features.get("visual"))
    if visual_continuity is None and not features:
        visual_continuity = max(source_score, 0.55 if _event_roles(left) & _event_roles(right) else 0.35)
    if level == "L4":
        return float(
            weighted_mean_available(
                [
                    (0.50, visual_continuity),
                    (0.20, text_topic_continuity),
                    (0.12, sparse_context_continuity),
                    (0.08, temporal_gap_score),
                    (0.10, source_role_continuity),
                ]
            )
            or 0.0
        )
    return float(
        weighted_mean_available(
            [
                (0.35, visual_continuity),
                (0.28, text_topic_continuity),
                (0.17, sparse_context_continuity),
                (0.10, temporal_gap_score),
                (0.10, source_role_continuity),
            ]
        )
        or 0.0
    )


def _merge_score_from_cached(view_meta_path: Path, left: dict, right: dict, left_features: dict, right_features: dict, *, level: str, max_gap: float, allow_visual_prior_fallback: bool = False) -> float:
    gap = _gap(left, right)
    if gap > max_gap:
        return 0.0
    temporal_gap_score = math.exp(-gap / max(max_gap, 1.0))
    source_score = _jaccard(_event_sources(left), _event_sources(right))
    role_score = _jaccard(_event_roles(left), _event_roles(right))
    location_left = _location_set(view_meta_path, left)
    location_right = _location_set(view_meta_path, right)
    location_score = 0.0 if location_left and location_right and location_left.isdisjoint(location_right) else (1.0 if location_left & location_right else 0.5)
    source_role_continuity = max(source_score, 0.5 * role_score + 0.5 * location_score)
    text_topic_continuity = _safe_cosine(left_features.get("text"), right_features.get("text"))
    sparse_context_continuity = None
    if left_features.get("sparse") or right_features.get("sparse"):
        sparse_context_continuity = weighted_jaccard(left_features.get("sparse") or {}, right_features.get("sparse") or {})
    visual_continuity = _safe_cosine(left_features.get("visual"), right_features.get("visual"))
    if visual_continuity is None and allow_visual_prior_fallback:
        visual_continuity = max(source_score, 0.55 if _event_roles(left) & _event_roles(right) else 0.35)
    if level == "L4":
        return float(
            weighted_mean_available(
                [
                    (0.50, visual_continuity),
                    (0.20, text_topic_continuity),
                    (0.12, sparse_context_continuity),
                    (0.08, temporal_gap_score),
                    (0.10, source_role_continuity),
                ]
            )
            or 0.0
        )
    return float(
        weighted_mean_available(
            [
                (0.35, visual_continuity),
                (0.28, text_topic_continuity),
                (0.17, sparse_context_continuity),
                (0.10, temporal_gap_score),
                (0.10, source_role_continuity),
            ]
        )
        or 0.0
    )


def _group_span(group: list[dict]) -> float:
    if not group:
        return 0.0
    return max(float(item["end_second"]) for item in group) - min(float(item["start_second"]) for item in group)


def _candidate_group_score(left_group: list[dict], right_group: list[dict], *, level: str, max_gap: float, max_span: float, view_meta_path: Path, features: Optional[dict[str, dict]]) -> Optional[float]:
    left = _row_from_group(level, left_group, "score_only", "score_only")
    right = _row_from_group(level, right_group, "score_only", "score_only")
    gap = max(0.0, float(right["start_second"]) - float(left["end_second"]))
    span = max(float(left["end_second"]), float(right["end_second"])) - min(float(left["start_second"]), float(right["start_second"]))
    if gap > max_gap or span > max_span:
        return None
    if level != "L5" and _parallel_incompatible(view_meta_path, left, right):
        return None
    return _merge_score(view_meta_path, left_group, right_group, level=level, max_gap=max_gap, features=features)


def _merge_short_context_groups(groups: list[list[dict]], *, level: str, max_gap: float, max_span: float, threshold: float, target_min_span: float, view_meta_path: Path, features: Optional[dict[str, dict]]) -> list[list[dict]]:
    if target_min_span <= 0 or len(groups) <= 1:
        return groups
    ordered = [list(group) for group in groups]
    while len(ordered) > 1:
        best: Optional[tuple[float, int]] = None
        for index in range(len(ordered) - 1):
            if _group_span(ordered[index]) >= target_min_span and _group_span(ordered[index + 1]) >= target_min_span:
                continue
            score = _candidate_group_score(
                ordered[index],
                ordered[index + 1],
                level=level,
                max_gap=max_gap,
                max_span=max_span,
                view_meta_path=view_meta_path,
                features=features,
            )
            if score is None or score < threshold:
                continue
            if best is None or score > best[0]:
                best = (score, index)
        if best is None:
            break
        _, index = best
        ordered[index].extend(ordered[index + 1])
        del ordered[index + 1]
    return ordered


def _merge_candidates(candidates: list[dict], *, level: str, max_gap: float, max_span: float, threshold: float, view_meta_path: Path, features: Optional[dict[str, dict]] = None, target_min_span: Optional[float] = None, short_merge_threshold: Optional[float] = None) -> list[list[dict]]:
    candidates.sort(key=lambda row: (float(row["start_second"]), float(row["end_second"])))
    if not candidates:
        return []
    groups: dict[int, list[dict]] = {index: [row] for index, row in enumerate(candidates)}
    previous: dict[int, Optional[int]] = {index: (index - 1 if index > 0 else None) for index in groups}
    next_id: dict[int, Optional[int]] = {index: (index + 1 if index < len(candidates) - 1 else None) for index in groups}
    versions: dict[int, int] = {index: 0 for index in groups}
    active = set(groups)
    heap: list[tuple[float, int, int, int, int]] = []
    summaries: dict[int, dict] = {index: _row_from_group(level, group, "score_only", "score_only") for index, group in groups.items()}
    group_features: dict[int, dict] = {index: _feature_for_group(group, features) for index, group in groups.items()}

    def candidate_score(left_id: int, right_id: int) -> Optional[float]:
        left = summaries[left_id]
        right = summaries[right_id]
        gap = max(0.0, float(right["start_second"]) - float(left["end_second"]))
        span = max(float(left["end_second"]), float(right["end_second"])) - min(float(left["start_second"]), float(right["start_second"]))
        if gap > max_gap or span > max_span:
            return None
        if level != "L5" and _parallel_incompatible(view_meta_path, left, right):
            return None
        score = _merge_score_from_cached(
            view_meta_path,
            left,
            right,
            group_features[left_id],
            group_features[right_id],
            level=level,
            max_gap=max_gap,
            allow_visual_prior_fallback=features is None,
        )
        return score if score >= threshold else None

    def push_pair(left_id: Optional[int], right_id: Optional[int]) -> None:
        if left_id is None or right_id is None or left_id not in active or right_id not in active:
            return
        score = candidate_score(left_id, right_id)
        if score is None:
            return
        heapq.heappush(heap, (-score, left_id, right_id, versions[left_id], versions[right_id]))

    for index in range(len(candidates) - 1):
        push_pair(index, index + 1)

    while heap:
        neg_score, left_id, right_id, left_version, right_version = heapq.heappop(heap)
        if left_id not in active or right_id not in active:
            continue
        if next_id.get(left_id) != right_id or previous.get(right_id) != left_id:
            continue
        if versions[left_id] != left_version or versions[right_id] != right_version:
            continue
        if -neg_score < threshold:
            continue
        groups[left_id].extend(groups[right_id])
        summaries[left_id] = _row_from_group(level, groups[left_id], "score_only", "score_only")
        group_features[left_id] = _feature_for_group(groups[left_id], features)
        versions[left_id] += 1
        right_next = next_id.get(right_id)
        next_id[left_id] = right_next
        if right_next is not None:
            previous[right_next] = left_id
        active.remove(right_id)
        del groups[right_id]
        summaries.pop(right_id, None)
        group_features.pop(right_id, None)
        push_pair(previous.get(left_id), left_id)
        push_pair(left_id, next_id.get(left_id))

    ordered = []
    current = next(index for index in range(len(candidates)) if index in active and previous.get(index) is None)
    while current is not None:
        ordered.append(groups[current])
        current = next_id.get(current)
    if target_min_span is not None and short_merge_threshold is not None:
        ordered = _merge_short_context_groups(
            ordered,
            level=level,
            max_gap=max_gap,
            max_span=max_span,
            threshold=short_merge_threshold,
            target_min_span=target_min_span,
            view_meta_path=view_meta_path,
            features=features,
        )
    return ordered


def _row_from_group(level: str, group: list[dict], param_hash: str, input_manifest_ref: str, index: int = 0) -> dict:
    start = min(float(item["start_second"]) for item in group)
    end = max(float(item["end_second"]) for item in group)
    day = group[0]["day"]
    day_num = group[0]["day_num"]
    sources = sorted({source for item in group for source in (item.get("sources") or ([item["source"]] if item.get("source") else []))})
    roles = sorted({role for item in group for role in (item.get("source_roles") or ([item.get("source_role")] if item.get("source_role") else [])) if role})
    representatives = list(dict.fromkeys(frame for item in group for frame in (item.get("representative_frame_ids") or [])))
    text_ids = list(dict.fromkeys(text for item in group for text in (item.get("linked_text_atom_ids") or [])))
    if level == "L4":
        child_ids = [item["event_id"] for item in group if item.get("event_kind") != "text_only_context"]
    else:
        child_ids = [item["event_id"] for item in group]
    eid = event_id(level, day, None, start, end, child_ids)
    contains_parallel = any(float(a["start_second"]) < float(b["end_second"]) and float(b["start_second"]) < float(a["end_second"]) for i, a in enumerate(group) for b in group[i + 1 :])
    child_flags = sorted(set(flag for item in group for flag in (item.get("quality_flags") or [])))
    if contains_parallel and "contains_parallel_activity" not in child_flags:
        child_flags.append("contains_parallel_activity")
    event_kind = "text_only_context" if all(item.get("event_kind") == "text_only_context" for item in group) else ("activity_episode" if level == "L4" else "session")
    return {
        "event_id": eid,
        "level": level,
        "day": day,
        "day_num": day_num,
        "source": None,
        "sources": sources,
        "source_role": None,
        "source_roles": roles,
        "start_second": start,
        "end_second": end,
        "start_hms": seconds_to_hms(start),
        "end_hms": seconds_to_hms(end),
        "duration_sec": end - start,
        "event_kind": event_kind,
        "primary_parent_id": None,
        "global_event_id": None,
        "representative_frame_ids": representatives[:24 if level == "L4" else 36],
        "linked_text_atom_ids": text_ids,
        "visual_centroid_ref": None,
        "text_dense_centroid_ref": None,
        "text_sparse_aggregate_ref": None,
        "modality_support": sorted(set(mod for item in group for mod in (item.get("modality_support") or []))),
        "confidence": sum(float(item.get("confidence") or 0.0) for item in group) / max(len(group), 1),
        "confidence_level": "medium",
        "quality_flags": child_flags,
        "reserved_label_primary": None,
        "reserved_label_secondary": None,
        "reserved_label_freeform": None,
        "reserved_label_confidence": None,
        "created_by": STAGE,
        "params_hash": param_hash,
        "input_manifest_ref": input_manifest_ref,
        "child_event_ids": child_ids,
        "dominant_sources": sources,
        "dominant_locations_reserved": [],
        "session_order_index": index,
        "contains_parallel_activity": contains_parallel,
    }


def _mark_short_context(row: dict, min_span: float) -> None:
    if row.get("event_kind") == "text_only_context":
        return
    if float(row.get("duration_sec") or 0.0) >= min_span:
        return
    flag = "short_activity_episode_context" if row.get("level") == "L4" else "short_session_context"
    flags = list(row.get("quality_flags") or [])
    if flag not in flags:
        flags.append(flag)
    row["quality_flags"] = sorted(set(flags))


def _intervals_overlap(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0])) > 0.0


def _canonical_video_intervals(run_root: Path) -> dict[tuple[str, str], list[tuple[float, float]]]:
    path = run_root / "atoms" / "canonical_frame_timeline.parquet"
    if not path.exists():
        return {}
    by_source_day: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in iter_pylist_rows(path, columns=["day", "source", "canonical_second"]):
        if row.get("canonical_second") is not None:
            by_source_day[(row["day"], row["source"])].append(float(row["canonical_second"]))
    intervals: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for key, seconds in by_source_day.items():
        if not seconds:
            continue
        seconds = sorted(set(seconds))
        start = seconds[0]
        previous = seconds[0]
        group_intervals = []
        for second in seconds[1:]:
            if second - previous > 1.5:
                group_intervals.append((start, previous + 1.0))
                start = second
            previous = second
        group_intervals.append((start, previous + 1.0))
        intervals[key] = group_intervals
    return intervals


def _text_only_candidates(config: AppConfig, run_root: Path, l2_rows: list[dict], param_hash: str, input_manifest_ref: str) -> list[dict]:
    text_path = run_root / "atoms" / "text_atoms.parquet"
    if not text_path.exists():
        return []
    role_map = source_roles(config.input_roots.view_meta_path)
    l2_by_source_day: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in l2_rows:
        l2_by_source_day[(row["day"], row["source"])].append((float(row["start_second"]), float(row["end_second"])))
    canonical_video = _canonical_video_intervals(run_root)
    result = []
    for text in iter_pylist_rows(text_path):
        if text.get("time_alignment_status") == "invalid":
            continue
        intervals = json.loads(text.get("union_time_intervals") or "[]")
        if not intervals:
            continue
        start = float(intervals[0][0])
        end = float(intervals[-1][1])
        existing = l2_by_source_day.get((text["day"], text["source"]), [])
        if any(_intervals_overlap((start, end), (a, b)) for a, b in existing):
            continue
        video_intervals = canonical_video.get((text["day"], text["source"]), [])
        if any(_intervals_overlap((start, end), interval) for interval in video_intervals):
            continue
        role = (role_map.get(text["source"]) or {}).get("source_role")
        row = {
            "event_id": event_id("L4TEXT", text["day"], text["source"], start, end, text["text_atom_id"]),
            "level": "L4",
            "day": text["day"],
            "day_num": text["day_num"],
            "source": None,
            "sources": [text["source"]],
            "source_role": None,
            "source_roles": [role] if role else [],
            "start_second": start,
            "end_second": end,
            "start_hms": seconds_to_hms(start),
            "end_hms": seconds_to_hms(end),
            "duration_sec": end - start,
            "event_kind": "text_only_context",
            "primary_parent_id": None,
            "global_event_id": None,
            "representative_frame_ids": [],
            "linked_text_atom_ids": [text["text_atom_id"]],
            "visual_centroid_ref": None,
            "text_dense_centroid_ref": None,
            "text_sparse_aggregate_ref": None,
            "modality_support": ["text"],
            "confidence": 0.25,
            "confidence_level": "low",
            "quality_flags": ["missing_visual", "text_only_context"],
            "reserved_label_primary": None,
            "reserved_label_secondary": None,
            "reserved_label_freeform": None,
            "reserved_label_confidence": None,
            "created_by": STAGE,
            "params_hash": param_hash,
            "input_manifest_ref": input_manifest_ref,
            "child_event_ids": [],
            "dominant_sources": [text["source"]],
            "dominant_locations_reserved": [],
            "session_order_index": 0,
            "contains_parallel_activity": False,
        }
        result.append(row)
    return result


def build_l4_episodes(config: AppConfig, run_root: Path, *, output_mode: str, run_id: Optional[str], day: Optional[str] = None, source: Optional[str] = None, resume: bool = False) -> dict:
    paths = stage_paths(run_root)
    l2_path = run_root / "segments" / "l2_source_events.parquet"
    l3_path = run_root / "segments" / "l3_global_events.parquet"
    l4_path = run_root / "segments" / "l4_activity_episodes.parquet"
    stage = "aggregate_l4"
    params = {"stage": stage, "output_mode": output_mode, "run_id": run_id, "day": day, "source": source}
    if resume and stage_complete(paths.stage_manifest(stage), params=params, output_paths=[l2_path, l3_path, l4_path]):
        return {"stage": stage, "skipped": True}

    l2_rows = read_pylist(l2_path, batch_size=config.run.batch_size) if l2_path.exists() else []
    l3_rows = read_pylist(l3_path, batch_size=config.run.batch_size) if l3_path.exists() else []
    l3_child_ids = {child for row in l3_rows for child in (row.get("child_source_event_ids") or [])}
    param_hash = params_hash(params)
    input_manifest_ref = to_ref(paths.stage_manifest("group_l3"), config.project_root)
    l4_inputs = list(l3_rows) + [row for row in l2_rows if row["event_id"] not in l3_child_ids]
    l4_inputs.extend(_text_only_candidates(config, run_root, l2_rows, param_hash, input_manifest_ref))
    l4_input_ids = {row["event_id"] for row in l4_inputs}
    materialized_l4_input_ids = {row["event_id"] for row in l4_inputs if row.get("event_kind") != "text_only_context"}
    _ensure_event_embeddings(config, run_root, materialized_l4_input_ids, output_mode=output_mode, run_id=run_id, day=day, source=source)
    l4_features = _load_event_features_from_outputs(run_root, l4_input_ids)

    by_day: dict[str, list[dict]] = defaultdict(list)
    for row in l4_inputs:
        by_day[row["day"]].append(row)
    l4_rows: list[dict] = []
    for rows in by_day.values():
        for group in _merge_candidates(
            rows,
            level="L4",
            max_gap=60,
            max_span=config.segmentation.l4_max_sec,
            threshold=0.68,
            view_meta_path=config.input_roots.view_meta_path,
            features=l4_features,
            target_min_span=60.0,
            short_merge_threshold=0.52,
        ):
            row = _row_from_group("L4", group, param_hash, input_manifest_ref)
            _mark_short_context(row, 60.0)
            l4_rows.append(row)

    for parent in l4_rows:
        for child_id in parent["child_event_ids"]:
            for row in l2_rows:
                if row["event_id"] == child_id and not row.get("primary_parent_id"):
                    row["primary_parent_id"] = parent["event_id"]
            for row in l3_rows:
                if row["event_id"] == child_id:
                    row["primary_parent_id"] = parent["event_id"]

    write_table(l2_path, l2_rows, schema=EVENT_TABLE)
    write_table(l3_path, l3_rows, schema=L3_EVENT_TABLE)
    write_table(l4_path, l4_rows, schema=L4_L5_EVENT_TABLE)
    manifest = write_stage_manifest(
        paths.stage_manifest(stage),
        stage=stage,
        params=params,
        inputs=[
            file_record(l2_path, config.project_root),
            file_record(l3_path, config.project_root),
            file_record(run_root / "atoms" / "text_atoms.parquet", config.project_root),
        ],
        outputs=[
            {**file_record(l2_path, config.project_root), "schema": str(EVENT_TABLE)},
            {**file_record(l3_path, config.project_root), "schema": str(L3_EVENT_TABLE)},
            {**file_record(l4_path, config.project_root), "schema": str(L4_L5_EVENT_TABLE)},
        ],
        schema_names=["l2_source_events", "l3_global_events", "l4_activity_episodes"],
        project_root=config.project_root,
    )
    update_run_state(run_root, stage, to_ref(paths.stage_manifest(stage), config.project_root))
    return {"stage": stage, "manifest": manifest}


def build_l5_sessions(config: AppConfig, run_root: Path, *, output_mode: str, run_id: Optional[str], day: Optional[str] = None, source: Optional[str] = None, resume: bool = False) -> dict:
    paths = stage_paths(run_root)
    l4_path = run_root / "segments" / "l4_activity_episodes.parquet"
    l5_path = run_root / "segments" / "l5_sessions.parquet"
    stage = "aggregate_l5"
    params = {"stage": stage, "output_mode": output_mode, "run_id": run_id, "day": day, "source": source}
    if resume and stage_complete(paths.stage_manifest(stage), params=params, output_paths=[l4_path, l5_path]):
        return {"stage": stage, "skipped": True}

    l4_rows = read_pylist(l4_path, batch_size=config.run.batch_size) if l4_path.exists() else []
    l4_ids = {row["event_id"] for row in l4_rows}
    materialized_l4_ids = {row["event_id"] for row in l4_rows if row.get("event_kind") != "text_only_context"}
    _ensure_event_embeddings(config, run_root, materialized_l4_ids, output_mode=output_mode, run_id=run_id, day=day, source=source)
    param_hash = params_hash(params)
    input_manifest_ref = to_ref(paths.stage_manifest("aggregate_l4"), config.project_root)
    l5_rows: list[dict] = []
    l5_features = _load_event_features_from_outputs(run_root, l4_ids)
    by_day_l4: dict[str, list[dict]] = defaultdict(list)
    for row in l4_rows:
        by_day_l4[row["day"]].append(row)
    for rows in by_day_l4.values():
        for index, group in enumerate(
            _merge_candidates(
                rows,
                level="L5",
                max_gap=300,
                max_span=config.segmentation.l5_max_sec,
                threshold=0.62,
                view_meta_path=config.input_roots.view_meta_path,
                features=l5_features,
                target_min_span=600.0,
                short_merge_threshold=0.45,
            )
        ):
            row = _row_from_group("L5", group, param_hash, input_manifest_ref, index=index)
            _mark_short_context(row, 600.0)
            l5_rows.append(row)
    for parent in l5_rows:
        for child_id in parent["child_event_ids"]:
            for row in l4_rows:
                if row["event_id"] == child_id:
                    row["primary_parent_id"] = parent["event_id"]

    write_table(l4_path, l4_rows, schema=L4_L5_EVENT_TABLE)
    write_table(l5_path, l5_rows, schema=L4_L5_EVENT_TABLE)
    manifest = write_stage_manifest(
        paths.stage_manifest(stage),
        stage=stage,
        params=params,
        inputs=[file_record(l4_path, config.project_root)],
        outputs=[
            {**file_record(l4_path, config.project_root), "schema": str(L4_L5_EVENT_TABLE)},
            {**file_record(l5_path, config.project_root), "schema": str(L4_L5_EVENT_TABLE)},
        ],
        schema_names=["l4_activity_episodes", "l5_sessions"],
        project_root=config.project_root,
    )
    update_run_state(run_root, stage, to_ref(paths.stage_manifest(stage), config.project_root))
    return {"stage": stage, "manifest": manifest}


def build_l4_l5(config: AppConfig, run_root: Path, *, output_mode: str, run_id: Optional[str], day: Optional[str] = None, source: Optional[str] = None, resume: bool = False) -> dict:
    l4_result = build_l4_episodes(config, run_root, output_mode=output_mode, run_id=run_id, day=day, source=source, resume=resume)
    l5_result = build_l5_sessions(config, run_root, output_mode=output_mode, run_id=run_id, day=day, source=source, resume=resume)
    return {"stage": STAGE, "results": [l4_result, l5_result]}
