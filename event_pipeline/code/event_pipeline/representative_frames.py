from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import numpy as np

from .vector_utils import cosine


def inherit_representative_frames(children: list[dict], limit: int, *, by_source: bool = False) -> list[str]:
    frames = []
    seen_sources = set()
    ordered = sorted(children, key=lambda row: (float(row["start_second"]), row["event_id"]))
    if by_source:
        for child in ordered:
            child_sources = child.get("sources") or ([child.get("source")] if child.get("source") else [])
            source_key = tuple(child_sources)
            if source_key in seen_sources:
                continue
            for frame_id in child.get("representative_frame_ids") or []:
                if frame_id not in frames:
                    frames.append(frame_id)
                    seen_sources.add(source_key)
                    break
            if len(frames) >= limit:
                return frames
    for child in ordered:
        for frame_id in child.get("representative_frame_ids") or []:
            if frame_id not in frames:
                frames.append(frame_id)
            if len(frames) >= limit:
                return frames
    return frames


def _finite_score(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _normalize_scores(values: Mapping[str, float]) -> dict[str, float]:
    finite = [float(v) for v in values.values() if np.isfinite(float(v))]
    if not finite:
        return {key: 0.0 for key in values}
    lo = min(finite)
    hi = max(finite)
    if hi - lo <= 1e-9:
        return {key: 1.0 for key in values}
    return {key: max(0.0, min(1.0, (float(value) - lo) / (hi - lo))) for key, value in values.items()}


def select_representative_frames(
    frame_rows: list[dict],
    frame_embeddings: Mapping[str, np.ndarray],
    visual_centroid: Optional[np.ndarray],
    *,
    start_second: float,
    end_second: float,
    level: str,
    max_count: int,
    min_spacing_sec: Optional[float] = None,
    similarity_dedupe_threshold: float = 0.98,
) -> tuple[list[str], dict[str, dict]]:
    """Score and dedupe L1/L2 representative frame candidates."""

    if not frame_rows or max_count <= 0:
        return [], {}
    if min_spacing_sec is None:
        min_spacing_sec = 2.0 if level == "L1" else 4.0
    duration = max(float(end_second) - float(start_second), 1e-6)
    mid = float(start_second) + duration / 2.0

    activity_raw = {
        row["frame_atom_id"]: _finite_score(row.get("activity_score") or row.get("visual_boundary_score"))
        for row in frame_rows
    }
    activity_norm = _normalize_scores(activity_raw)
    scored = []
    for row in frame_rows:
        frame_id = row["frame_atom_id"]
        second = _finite_score(row.get("canonical_second"))
        embedding = frame_embeddings.get(frame_id)
        if visual_centroid is not None and embedding is not None:
            centroid_nearest = max(0.0, min(1.0, (cosine(embedding, visual_centroid) + 1.0) / 2.0))
        else:
            centroid_nearest = 0.0
        boundary_distance = min(abs(second - float(start_second)), abs(second - float(end_second)))
        boundary_context = max(0.0, 1.0 - boundary_distance / max(3.0 if level == "L1" else 8.0, 1.0))
        temporal_coverage = max(0.0, 1.0 - abs(second - mid) / max(duration / 2.0, 1.0))
        flags = set(row.get("quality_flags") or [])
        quality = 1.0
        if row.get("path_exists_at_build") is False:
            quality -= 0.4
        if row.get("embedding_exists_at_build") is False or embedding is None:
            quality -= 0.4
        if flags & {"out_of_hour_timestamp", "duplicate_absolute_second_noncanonical", "invalid_frame_time"}:
            quality -= 0.6
        quality = max(0.0, quality)
        score = (
            0.35 * activity_norm.get(frame_id, 0.0)
            + 0.30 * centroid_nearest
            + 0.15 * boundary_context
            + 0.10 * temporal_coverage
            + 0.10 * quality
        )
        if boundary_context >= 0.75:
            reason = "start_context" if abs(second - float(start_second)) <= abs(second - float(end_second)) else "end_context"
        elif activity_norm.get(frame_id, 0.0) >= centroid_nearest:
            reason = "activity_peak"
        else:
            reason = "centroid_nearest"
        scored.append((score, second, frame_id, reason))

    selected: list[tuple[float, float, str, str]] = []
    for candidate in sorted(scored, reverse=True):
        score, second, frame_id, reason = candidate
        embedding = frame_embeddings.get(frame_id)
        too_close = any(abs(second - chosen_second) < min_spacing_sec for _, chosen_second, _, _ in selected)
        too_similar = False
        if embedding is not None:
            for _, _, chosen_id, _ in selected:
                chosen_embedding = frame_embeddings.get(chosen_id)
                if chosen_embedding is not None and cosine(embedding, chosen_embedding) > similarity_dedupe_threshold:
                    too_similar = True
                    break
        if too_close or too_similar:
            continue
        selected.append(candidate)
        if len(selected) >= max_count:
            break

    selected.sort(key=lambda item: item[1])
    metadata = {
        frame_id: {
            "selection_score": float(score),
            "selection_reason": reason,
            "selection_rank": rank,
        }
        for rank, (score, _, frame_id, reason) in enumerate(selected)
    }
    return [frame_id for _, _, frame_id, _ in selected], metadata
