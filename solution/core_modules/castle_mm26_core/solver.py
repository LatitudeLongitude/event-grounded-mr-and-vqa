from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from .data import CoverageIndex, EventStore, filter_text_atom_ids_for_source, text_atom_ids_for_source_interval
from .config import AppConfig
from .io_utils import atomic_write_json, atomic_write_jsonl
from .models import EventCandidate, EvidenceBundle, FrameHit, TaskQuery, TextHit
from .task_manifest import write_completed_marker, write_task_manifest
from .time_utils import Interval, format_submission_hms, nms_intervals, normalize_output_interval, overlap_duration


DIRECT_FRAME_ROLES = {"frame_overlap_support", "activity_peak", "start_context", "end_context"}
FIXED_ROOM_SOURCES = {"Kitchen", "Living1", "Living2", "Meeting", "Reading"}


INTERVAL_SCORE_SPECS: dict[str, dict[str, Any]] = {
    "MM26-EIS01": {"category": "EIS-V1", "visual_floor": 0.096, "text_floor": 0.35, "weights": {"visual_pos": 0.75, "text_pos": 0.25, "text_neg": 0.35}, "require_visual": True, "require_text": False},
    "MM26-EIS02": {"category": "EIS-V1", "visual_floor": 0.114, "text_floor": 0.25, "weights": {"visual_pos": 0.75, "text_pos": 0.25, "text_neg": 0.35}, "require_visual": True, "require_text": False},
    "MM26-EIS03": {"category": "EIS-ACT", "visual_floor": 0.105, "text_floor": 0.55, "weights": {"visual_pos": 0.55, "text_pos": 0.45, "text_neg": 0.45}, "require_visual": True, "require_text": False},
    "MM26-EIS04": {"category": "EIS-TG", "visual_floor": 0.098, "text_floor": 0.25, "weights": {"visual_pos": 0.35, "text_pos": 0.65, "text_neg": 0.50}, "require_visual": True, "require_text": False},
    "MM26-EIS05": {"category": "EIS-V1", "visual_floor": 0.092106, "text_floor": 0.45, "weights": {"visual_pos": 0.75, "text_pos": 0.25, "text_neg": 0.35}, "require_visual": True, "require_text": False},
    "MM26-EIS06": {"category": "EIS-V1", "visual_floor": 0.103, "text_floor": 0.20, "weights": {"visual_pos": 0.75, "text_pos": 0.25, "text_neg": 0.35}, "require_visual": True, "require_text": False},
    "MM26-EIS07": {"category": "EIS-V2", "visual_floor": 0.103, "text_floor": 0.20, "weights": {"visual_pos": 0.52, "text_pos": 0.48, "text_neg": 0.45}, "require_visual": True, "require_text": False},
    "MM26-EIS08": {"category": "EIS-V1", "visual_floor": 0.1015, "text_floor": 0.35, "weights": {"visual_pos": 0.75, "text_pos": 0.25, "text_neg": 0.35}, "require_visual": True, "require_text": False},
    "MM26-EIS09": {"category": "EIS-TG", "visual_floor": 0.095, "text_floor": 0.20, "weights": {"visual_pos": 0.35, "text_pos": 0.65, "text_neg": 0.50}, "require_visual": False, "require_text": True},
    "MM26-EIS10": {"category": "EIS-V2", "visual_floor": 0.100, "text_floor": 0.45, "weights": {"visual_pos": 0.70, "text_pos": 0.30, "text_neg": 0.40}, "require_visual": True, "require_text": False},
    "MM26-EIS11": {"category": "EIS-LONG", "visual_floor": 0.115, "text_floor": 0.50, "weights": {"visual_pos": 0.55, "text_pos": 0.45, "text_neg": 0.30}, "require_visual": True, "require_text": False},
    "MM26-EIS12": {"category": "EIS-TG", "visual_floor": 0.105, "text_floor": 0.40, "weights": {"visual_pos": 0.35, "text_pos": 0.65, "text_neg": 0.50}, "require_visual": False, "require_text": True},
    "MM26-OIS01": {"category": "OIS-TINY", "visual_floor": 0.112, "text_floor": 0.75, "weights": {"visual_pos": 0.90, "text_pos": 0.10, "text_neg": 0.40}, "require_visual": True, "require_text": False},
    "MM26-OIS02": {"category": "OIS-WEAK", "visual_floor": 0.0889, "text_floor": 0.60, "weights": {"visual_pos": 0.90, "text_pos": 0.10, "text_neg": 0.45}, "require_visual": True, "require_text": False},
    "MM26-OIS03": {"category": "OIS-TINY", "visual_floor": 0.098, "text_floor": 0.55, "weights": {"visual_pos": 0.90, "text_pos": 0.10, "text_neg": 0.40}, "require_visual": True, "require_text": False},
    "MM26-OIS04": {"category": "OIS-VIS", "visual_floor": 0.102, "text_floor": 0.50, "weights": {"visual_pos": 0.85, "text_pos": 0.15, "text_neg": 0.35}, "require_visual": True, "require_text": False},
    "MM26-OIS05": {"category": "OIS-TINY", "visual_floor": 0.096, "text_floor": 0.25, "weights": {"visual_pos": 0.90, "text_pos": 0.10, "text_neg": 0.40}, "require_visual": True, "require_text": False},
    "MM26-OIS06": {"category": "OIS-VIS", "visual_floor": 0.098, "text_floor": 0.40, "weights": {"visual_pos": 0.85, "text_pos": 0.15, "text_neg": 0.35}, "require_visual": True, "require_text": False},
    "MM26-OIS07": {"category": "OIS-VIS", "visual_floor": 0.097, "text_floor": 0.45, "weights": {"visual_pos": 0.85, "text_pos": 0.15, "text_neg": 0.35}, "require_visual": True, "require_text": False},
    "MM26-OIS08": {"category": "OIS-TINY", "visual_floor": 0.097, "text_floor": 0.45, "weights": {"visual_pos": 0.90, "text_pos": 0.10, "text_neg": 0.40}, "require_visual": True, "require_text": False},
    "MM26-OIS09": {"category": "OIS-EXCL", "visual_floor": 0.080, "text_floor": 0.18, "weights": {"visual_pos": 0.40, "text_pos": 0.60, "text_neg": 0.75}, "require_visual": True, "require_text": True},
    "MM26-OIS10": {"category": "OIS-VIS", "visual_floor": 0.094, "text_floor": 0.25, "weights": {"visual_pos": 0.85, "text_pos": 0.15, "text_neg": 0.35}, "require_visual": True, "require_text": False},
    "MM26-OIS11": {"category": "OIS-CLS", "visual_floor": 0.058, "text_floor": 0.60, "weights": {"visual_pos": 0.90, "text_pos": 0.10, "text_neg": 0.45}, "require_visual": True, "require_text": False},
    "MM26-OIS12": {"category": "OIS-VIS", "visual_floor": 0.095, "text_floor": 0.20, "weights": {"visual_pos": 0.85, "text_pos": 0.15, "text_neg": 0.35}, "require_visual": True, "require_text": False},
}


def build_candidates(
    query: TaskQuery,
    frame_hits: Sequence[FrameHit],
    text_hits: Sequence[TextHit],
    event_store: EventStore,
    *,
    max_frame_hits: int = 200,
    max_text_hits: int = 120,
) -> list[EventCandidate]:
    candidates: dict[str, EventCandidate] = {}
    selected_frame_hits = list(frame_hits[:max_frame_hits])
    for hit in selected_frame_hits:
        frame_atom = event_store.frame_by_embedding.get(hit.embedding_id)
        if frame_atom:
            hit.frame_atom_id = frame_atom["frame_atom_id"]
    selected_text_hits = list(text_hits[:max_text_hits])
    selected_text_hits = _prepend_lexical_marker_sweep_hits(query, selected_text_hits, event_store)
    for hit in selected_text_hits:
        text_atom = event_store.text_by_chunk.get(hit.embedding_chunk_id)
        if text_atom:
            hit.text_atom_id = text_atom["text_atom_id"]
    eis_l2_suppression = _build_eis_l2_suppression_plan(query, selected_frame_hits, event_store)
    support_hits_by_event = _support_hits_by_event(selected_frame_hits, event_store)
    calibrator = _build_interval_score_calibrator(query, selected_frame_hits, selected_text_hits)
    for hit in selected_frame_hits:
        event_rows = _events_for_frame(hit, event_store)
        event_rows = _eis_events_after_l2_suppression(event_rows, hit, eis_l2_suppression, event_store)
        for event in event_rows:
            if event.get("source") is None:
                continue
            event_id = str(event.get("event_id") or "")
            if not event_id:
                continue
            candidate_id = _event_candidate_id(query, event)
            event_support_hits = support_hits_by_event.get(str(event.get("event_id") or ""), [hit])
            current = candidates.get(candidate_id)
            if current is None:
                current = _new_event_candidate(query, event, hit.day, hit.source)
                current.provenance.update(_event_context_provenance(query, event, event_store, eis_l2_suppression))
                candidates[candidate_id] = current
            _merge_visual_into_event_candidate(current, query, event, hit, event_support_hits)
    for hit in selected_text_hits:
        event_rows = _events_for_text(hit, event_store)
        for event in event_rows:
            if event.get("source") is None:
                continue
            if _event_is_suppressed_eis_l2(query, event, eis_l2_suppression):
                continue
            event_id = str(event.get("event_id") or "")
            if not event_id:
                continue
            candidate_id = _event_candidate_id(query, event)
            current = candidates.get(candidate_id)
            if current is None:
                current = _new_event_candidate(query, event, hit.day, hit.source)
                current.provenance.update(_event_context_provenance(query, event, event_store, eis_l2_suppression))
                if query.evidence_policy.get("allow_transcript_marker_primary"):
                    text_frame_ids = _local_recovery_frame_ids(event_store, event, hit)
                    current.frame_atom_ids = sorted(set(current.frame_atom_ids + text_frame_ids))
                    if "transcript_marker_primary" not in current.evidence_modalities:
                        current.evidence_modalities.append("transcript_marker_primary")
                    if text_frame_ids and "local_visual_recovery" not in current.evidence_modalities:
                        current.evidence_modalities.append("local_visual_recovery")
                    current.provenance.update(_audio_sensitive_marker_provenance(query, hit, event, text_frame_ids))
                candidates[candidate_id] = current
            text_score_parts, text_scoring_audit = _text_score_parts(query, hit, _raw_text_score(hit), event=event)
            _merge_text_into_candidate(current, hit, text_score_parts, text_scoring_audit, query)
    if query.task_type == "object_instance_search":
        _apply_object_state_probes(query, candidates.values(), selected_text_hits)
    for candidate in candidates.values():
        score_event_candidate(candidate, query, calibrator)
    return sorted(candidates.values(), key=lambda item: (-item.final_score, item.candidate_id))


def _event_candidate_id(query: TaskQuery, event: dict) -> str:
    return f"{query.task_id}:{str(event.get('event_id') or '')}"


def _new_event_candidate(query: TaskQuery, event: dict, fallback_day: int, fallback_source: str) -> EventCandidate:
    start = float(event.get("start_second") or 0.0)
    end = float(event.get("end_second") or start + 1.0)
    event_id = str(event.get("event_id") or "")
    level = str(event.get("level") or "")
    return EventCandidate(
        task_id=query.task_id,
        candidate_id=_event_candidate_id(query, event),
        day=int(event.get("day_num") or fallback_day),
        source=str(event.get("source") or fallback_source),
        start_second=start,
        end_second=end,
        event_ids=[event_id] if event_id else [],
        frame_atom_ids=[],
        text_atom_ids=[],
        evidence_modalities=[],
        score_parts={
            "visual_pos_raw": 0.0,
            "text_pos_raw": 0.0,
            "text_neg_raw": 0.0,
            "visual_norm": 0.0,
            "text_pos_norm": 0.0,
            "text_neg_norm": 0.0,
        },
        final_score=0.0,
        provenance={
            "event_id": event_id,
            "level": level,
            "event_level": level,
            "best_visual_frame": None,
            "direct_text_atom_ids": [],
            "scoring": {"candidate_score_policy": "raw_event_visual_text_negative_v1"},
        },
    )


def _merge_visual_into_event_candidate(
    candidate: EventCandidate,
    query: TaskQuery,
    event: dict,
    hit: FrameHit,
    support_hits: Sequence[FrameHit],
) -> None:
    if "visual_direct_frame" not in candidate.evidence_modalities:
        candidate.evidence_modalities.append("visual_direct_frame")
    raw = _raw_visual_score(hit)
    best_raw = float(candidate.score_parts.get("visual_pos_raw", 0.0))
    if raw < best_raw:
        _merge_local_recovery_seed_text_into_candidate(candidate, query, hit)
        return
    candidate.score_parts["visual_pos_raw"] = raw
    candidate.score_parts["visual_cls_raw"] = float(hit.cls_score)
    candidate.score_parts["visual_patch_top5_raw"] = float(hit.patch_top5_mean if hit.patch_top5_mean is not None else raw)
    candidate.frame_atom_ids = [str(hit.frame_atom_id)] if hit.frame_atom_id else []
    candidate.provenance["frame_hit"] = hit.provenance
    candidate.provenance["direct_frame_atom_id"] = hit.frame_atom_id
    candidate.provenance["direct_frame_second"] = float(hit.absolute_second)
    candidate.provenance["best_visual_frame"] = {
        "frame_atom_id": hit.frame_atom_id,
        "embedding_id": hit.embedding_id,
        "absolute_second": float(hit.absolute_second),
        "frame_rel_path": hit.frame_rel_path,
        "image_path_ref": hit.image_path_ref,
        "patch_score": hit.patch_score,
        "cls_score": hit.cls_score,
        "patch_top5_mean": hit.patch_top5_mean,
        "rank": hit.rank,
        "score_source": "patch_score" if hit.patch_score is not None else "cls_score",
    }
    if query.task_type == "object_instance_search":
        start_second, end_second, audit = _ois_supported_interval(query, event, hit, support_hits)
        candidate.start_second = start_second
        candidate.end_second = end_second
        candidate.provenance["ois_boundary_refinement"] = audit
    elif event.get("start_second") is None or event.get("end_second") is None:
        candidate.start_second = max(0.0, float(hit.absolute_second) - 2.0)
        candidate.end_second = min(86399.0, float(hit.absolute_second) + 3.0)
    _merge_local_recovery_seed_text_into_candidate(candidate, query, hit)


def _merge_local_recovery_seed_text_into_candidate(candidate: EventCandidate, query: TaskQuery, hit: FrameHit) -> None:
    best_record = None
    best_score = -1.0
    for record in _local_recovery_seed_records(hit.provenance):
        score = _float_or_default(record.get("seed_text_score"), -1.0)
        if score > best_score:
            best_record = record
            best_score = score
    if not best_record or best_score < 0.0:
        return
    text = str(best_record.get("seed_text_excerpt") or "")
    adjusted_score, lexical_audit = _task_text_lexical_adjustment(query, text, best_score)
    negative_score = _float_or_default(best_record.get("seed_negative_text_score"), 0.0)
    candidate.score_parts["text_pos_raw"] = max(float(candidate.score_parts.get("text_pos_raw", 0.0)), float(adjusted_score))
    candidate.score_parts["text_neg_raw"] = max(float(candidate.score_parts.get("text_neg_raw", 0.0)), float(negative_score))
    text_atom_id = str(best_record.get("seed_text_atom_id") or "")
    if text_atom_id and text_atom_id not in candidate.text_atom_ids:
        candidate.text_atom_ids.append(text_atom_id)
    if "text_localized_visual_recovery" not in candidate.evidence_modalities:
        candidate.evidence_modalities.append("text_localized_visual_recovery")
    scoring = candidate.provenance.setdefault("scoring", {})
    scoring.setdefault("text_hits", []).append(
        {
            "policy": "text_localized_visual_recovery_seed_score_v1",
            "embedding_chunk_id": best_record.get("seed_embedding_chunk_id"),
            "text_atom_id": text_atom_id or None,
            "event_id": best_record.get("seed_event_id"),
            "text_excerpt": text,
            "text_pos_raw": float(adjusted_score),
            "text_neg_raw": float(negative_score),
            "positive_score_source": best_record.get("seed_text_score_source") or "local_recovery_seed_text_score",
            "negative_score_source": "negative_reranker_score" if best_record.get("seed_negative_text_score") is not None else "missing_negative_rerank",
            "lexical_adjustment": lexical_audit,
            "source": "local_visual_recovery_seed",
        }
    )
    _remember_local_recovery_seed_text(candidate, best_record, text, adjusted_score, negative_score)
    direct_text_atom_ids = candidate.provenance.setdefault("direct_text_atom_ids", [])
    if text_atom_id and text_atom_id not in direct_text_atom_ids:
        direct_text_atom_ids.append(text_atom_id)


def _remember_local_recovery_seed_text(
    candidate: EventCandidate,
    record: dict,
    text: str,
    adjusted_score: float,
    negative_score: float,
) -> None:
    if not text:
        return
    stored = candidate.provenance.setdefault("local_visual_recovery_seed_texts", [])
    if not isinstance(stored, list):
        stored = []
        candidate.provenance["local_visual_recovery_seed_texts"] = stored
    text_atom_id = str(record.get("seed_text_atom_id") or "")
    event_id = str(record.get("seed_event_id") or "")
    excerpt_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    for item in stored:
        if not isinstance(item, dict):
            continue
        if item.get("seed_text_atom_id") == text_atom_id and item.get("seed_event_id") == event_id and item.get("excerpt_hash") == excerpt_hash:
            return
    stored.append(
        {
            "seed_text_atom_id": text_atom_id or None,
            "seed_embedding_chunk_id": record.get("seed_embedding_chunk_id"),
            "seed_event_id": event_id or None,
            "seed_text_excerpt": text,
            "excerpt_hash": excerpt_hash,
            "text_pos_raw": float(adjusted_score),
            "text_neg_raw": float(negative_score),
            "source": "local_visual_recovery_seed",
        }
    )


def _local_recovery_seed_records(provenance: dict | None) -> list[dict]:
    if not isinstance(provenance, dict):
        return []
    records = []
    direct = provenance.get("local_visual_recovery")
    if isinstance(direct, dict):
        records.append(direct)
    for merged in provenance.get("merged_duplicate_sources") or []:
        if isinstance(merged, dict) and isinstance(merged.get("local_visual_recovery"), dict):
            records.append(merged["local_visual_recovery"])
    for merged in provenance.get("merged_candidate_records") or []:
        if not isinstance(merged, dict):
            continue
        snapshot = merged.get("provenance")
        if isinstance(snapshot, dict) and isinstance(snapshot.get("local_visual_recovery"), dict):
            records.append(snapshot["local_visual_recovery"])
    return records


def _raw_visual_score(hit: FrameHit) -> float:
    value = hit.patch_score if hit.patch_score is not None else hit.cls_score
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return score if math.isfinite(score) else 0.0


def _raw_text_score(hit: TextHit) -> float:
    for value in (hit.reranker_score, hit.dense_score, hit.sparse_score):
        if value is None:
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score):
            return score
    return 0.0


def _raw_negative_text_score(hit: TextHit) -> float:
    value = hit.negative_reranker_score
    if value is None:
        return 0.0
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return score if math.isfinite(score) else 0.0


def _build_interval_score_calibrator(query: TaskQuery, frame_hits: Sequence[FrameHit], text_hits: Sequence[TextHit]) -> dict:
    spec = _interval_score_spec(query)
    visual_values = [_raw_visual_score(hit) for hit in frame_hits]
    text_values = [_raw_text_score(hit) for hit in text_hits]
    negative_values = [_raw_negative_text_score(hit) for hit in text_hits if hit.negative_reranker_score is not None]
    return {
        "policy": "raw_floor_high_linear_v1",
        "task_id": query.task_id,
        "category": spec["category"],
        "visual_floor": float(spec["visual_floor"]),
        "text_floor": float(spec["text_floor"]),
        "negative_floor": 0.0,
        "visual_high": _raw_high_anchor(visual_values, float(spec["visual_floor"])),
        "text_high": _raw_high_anchor(text_values, float(spec["text_floor"])),
        "negative_high": _raw_high_anchor(negative_values, 0.0, default_high=1.0),
        "weights": dict(spec["weights"]),
        "require_visual": bool(spec.get("require_visual")),
        "require_text": bool(spec.get("require_text")),
    }


def _raw_high_anchor(values: Sequence[float], floor: float, *, default_high: float | None = None) -> float:
    finite_values = sorted((float(value) for value in values if math.isfinite(float(value))), reverse=True)
    if not finite_values:
        fallback = 1.0 if default_high is None else float(default_high)
        return max(float(floor) + 1e-6, fallback)
    top20_median = _median(finite_values[:20])
    p995 = _quantile(finite_values, 0.995)
    observed_high = max(top20_median, p995, finite_values[0])
    if default_high is not None:
        observed_high = max(observed_high, float(default_high))
    if not math.isfinite(observed_high) or observed_high <= float(floor):
        observed_high = max(finite_values[0], float(floor)) + 1e-6
    return float(observed_high)


def score_event_candidate(candidate: EventCandidate, query: TaskQuery, calibrator: dict) -> EventCandidate:
    weights = calibrator["weights"]
    visual_raw = _candidate_score_part(candidate, "visual_pos_raw", "patch", "visual")
    text_raw = _candidate_score_part(candidate, "text_pos_raw", "text", "text_context")
    neg_raw = _candidate_score_part(candidate, "text_neg_raw")
    visual_norm = _linear_norm(visual_raw, float(calibrator["visual_floor"]), float(calibrator["visual_high"]))
    text_norm = _linear_norm(text_raw, float(calibrator["text_floor"]), float(calibrator["text_high"])) if text_raw > 0.0 else 0.0
    neg_norm = _linear_norm(neg_raw, float(calibrator["negative_floor"]), float(calibrator["negative_high"])) if neg_raw > 0.0 else 0.0
    final = float(weights["visual_pos"]) * visual_norm + float(weights["text_pos"]) * text_norm - float(weights["text_neg"]) * neg_norm
    candidate.score_parts.update(
        {
            "visual_pos_raw": visual_raw,
            "text_pos_raw": text_raw,
            "text_neg_raw": neg_raw,
            "visual_norm": visual_norm,
            "text_pos_norm": text_norm,
            "text_neg_norm": neg_norm,
            "final_score_raw_weighted": final,
        }
    )
    candidate.final_score = final
    scoring = candidate.provenance.setdefault("scoring", {})
    scoring["score_definition"] = "final_score = wv*visual_norm + wt*text_pos_norm - wn*text_neg_norm"
    scoring["calibrator"] = {
        "policy": calibrator["policy"],
        "visual_floor": calibrator["visual_floor"],
        "visual_high": calibrator["visual_high"],
        "text_floor": calibrator["text_floor"],
        "text_high": calibrator["text_high"],
        "negative_floor": calibrator["negative_floor"],
        "negative_high": calibrator["negative_high"],
        "category": calibrator["category"],
    }
    scoring["weights"] = dict(weights)
    scoring.setdefault("threshold_decision", {"status": "pending_solver_gate"})
    return candidate


def _candidate_score_part(candidate: EventCandidate, key: str, *fallback_keys: str) -> float:
    for name in (key, *fallback_keys):
        value = candidate.score_parts.get(name)
        if value is None:
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score):
            return score
    return 0.0


def _linear_norm(value: float, floor: float, high: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    denom = max(float(high) - float(floor), 1e-6)
    return max(0.0, min(1.0, (float(value) - float(floor)) / denom))


def _interval_score_spec(query: TaskQuery) -> dict:
    spec = INTERVAL_SCORE_SPECS.get(query.task_id)
    if spec:
        return spec
    if query.task_type == "object_instance_search":
        return {"category": "OIS-FALLBACK", "visual_floor": 0.0, "text_floor": 0.0, "weights": {"visual_pos": 0.85, "text_pos": 0.15, "text_neg": 0.35}, "require_visual": True, "require_text": False}
    return {"category": "EIS-FALLBACK", "visual_floor": 0.0, "text_floor": 0.0, "weights": {"visual_pos": 0.75, "text_pos": 0.25, "text_neg": 0.35}, "require_visual": query.task_type == "event_instance_search", "require_text": False}


def _build_eis_l2_suppression_plan(query: TaskQuery, frame_hits: Sequence[FrameHit], store: EventStore) -> dict:
    if query.task_type != "event_instance_search":
        return {
            "suppressed_l2_ids": set(),
            "suppressed_l2_audit_by_parent": {},
            "unsuppressed_l2_audit_by_parent": {},
            "child_events_by_l2_frame": {},
        }
    l2_frames: dict[str, set[str]] = defaultdict(set)
    l2_frame_child_ids: dict[tuple[str, str], set[str]] = defaultdict(set)
    child_events_by_l2_frame: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for hit in frame_hits:
        if not hit.frame_atom_id:
            continue
        frame_atom_id = str(hit.frame_atom_id)
        events = _events_for_frame(hit, store)
        for event in events:
            if event.get("source") is None:
                continue
            level = str(event.get("level") or "")
            event_id = str(event.get("event_id") or "")
            if level == "L2":
                l2_frames[event_id].add(frame_atom_id)
                for child in _child_l1_events_for_l2_at_second(event, float(hit.absolute_second), store):
                    child_id = str(child.get("event_id") or "")
                    if not child_id:
                        continue
                    l2_frame_child_ids[(event_id, frame_atom_id)].add(child_id)
                    child_events_by_l2_frame[(event_id, frame_atom_id)][child_id] = child
            elif level == "L1":
                for parent_id in _parent_l2_event_ids_for_event(event, store):
                    l2_frame_child_ids[(parent_id, frame_atom_id)].add(event_id)
                    child_events_by_l2_frame[(parent_id, frame_atom_id)][event_id] = event

    suppressed_l2_ids: set[str] = set()
    suppressed_audit_by_parent: dict[str, dict] = {}
    unsuppressed_audit_by_parent: dict[str, dict] = {}
    policy = "l2_suppressed_when_all_visual_hits_covered_by_child_l1_v1"
    for l2_id, frame_ids in sorted(l2_frames.items()):
        frame_id_list = sorted(frame_ids)
        child_ids = sorted({child_id for frame_id in frame_id_list for child_id in l2_frame_child_ids.get((l2_id, frame_id), set())})
        uncovered = [frame_id for frame_id in frame_id_list if not l2_frame_child_ids.get((l2_id, frame_id))]
        if frame_id_list and not uncovered:
            audit = {
                "policy": policy,
                "parent_l2_event_id": l2_id,
                "child_l1_event_ids": child_ids,
                "recalled_frame_atom_ids": frame_id_list,
                "suppressed": True,
                "reason": "all_recalled_frames_covered_by_child_l1",
            }
            suppressed_l2_ids.add(l2_id)
            suppressed_audit_by_parent[l2_id] = audit
        else:
            unsuppressed_audit_by_parent[l2_id] = {
                "policy": policy,
                "parent_l2_event_id": l2_id,
                "child_l1_event_ids": child_ids,
                "recalled_frame_atom_ids": frame_id_list,
                "uncovered_frame_atom_ids": uncovered,
                "suppressed": False,
                "reason": "has_recalled_frames_outside_child_l1",
            }
    return {
        "suppressed_l2_ids": suppressed_l2_ids,
        "suppressed_l2_audit_by_parent": suppressed_audit_by_parent,
        "unsuppressed_l2_audit_by_parent": unsuppressed_audit_by_parent,
        "child_events_by_l2_frame": child_events_by_l2_frame,
    }


def _eis_events_after_l2_suppression(event_rows: Sequence[dict], hit: FrameHit, plan: dict, store: EventStore) -> list[dict]:
    suppressed_l2_ids = set(plan.get("suppressed_l2_ids") or set())
    if not suppressed_l2_ids or not hit.frame_atom_id:
        return list(event_rows)
    frame_atom_id = str(hit.frame_atom_id)
    result = []
    seen_event_ids = set()
    for event in event_rows:
        event_id = str(event.get("event_id") or "")
        if str(event.get("level") or "") == "L2" and event_id in suppressed_l2_ids:
            for child in (plan.get("child_events_by_l2_frame") or {}).get((event_id, frame_atom_id), {}).values():
                child_id = str(child.get("event_id") or "")
                if child_id and child_id not in seen_event_ids:
                    result.append(_event_with_graph_scores(child, store))
                    seen_event_ids.add(child_id)
            continue
        if event_id and event_id in seen_event_ids:
            continue
        result.append(event)
        if event_id:
            seen_event_ids.add(event_id)
    return result


def _event_is_suppressed_eis_l2(query: TaskQuery, event: dict, plan: dict) -> bool:
    if query.task_type != "event_instance_search":
        return False
    if str(event.get("level") or "") != "L2":
        return False
    return str(event.get("event_id") or "") in set(plan.get("suppressed_l2_ids") or set())


def _event_context_provenance(query: TaskQuery, event: dict, store: EventStore, plan: dict | None = None) -> dict:
    event_id = str(event.get("event_id") or "")
    parent_l2_ids = _parent_l2_event_ids_for_event(event, store)
    provenance: dict = {}
    if parent_l2_ids:
        provenance["parent_l2_event_ids"] = parent_l2_ids
        if len(parent_l2_ids) == 1:
            provenance["primary_parent_l2_event_id"] = parent_l2_ids[0]
    if query.task_type != "event_instance_search" or not plan:
        return provenance
    if str(event.get("level") or "") == "L1":
        audits = [
            (plan.get("suppressed_l2_audit_by_parent") or {}).get(parent_id)
            for parent_id in parent_l2_ids
            if (plan.get("suppressed_l2_audit_by_parent") or {}).get(parent_id)
        ]
        if audits:
            provenance["eis_l2_suppression"] = audits[0] if len(audits) == 1 else audits
    elif str(event.get("level") or "") == "L2":
        audit = (plan.get("unsuppressed_l2_audit_by_parent") or {}).get(event_id)
        if audit:
            provenance["eis_l2_suppression"] = audit
    return provenance


def _support_hits_by_event(frame_hits: Sequence[FrameHit], store: EventStore) -> dict[str, list[FrameHit]]:
    grouped: dict[str, list[FrameHit]] = defaultdict(list)
    for hit in frame_hits:
        if not hit.frame_atom_id:
            continue
        for event in _events_for_frame(hit, store):
            event_id = str(event.get("event_id") or "")
            if event_id:
                grouped[event_id].append(hit)
    for hits in grouped.values():
        hits.sort(key=lambda item: (float(item.absolute_second), item.embedding_id))
    return grouped


def _merge_text_into_candidate(
    candidate: EventCandidate,
    hit: TextHit,
    text_score_parts: dict[str, float],
    text_scoring_audit: dict,
    query: TaskQuery,
) -> None:
    if hit.text_atom_id and hit.text_atom_id not in candidate.text_atom_ids:
        candidate.text_atom_ids.append(hit.text_atom_id)
    if "text_overlap_support" not in candidate.evidence_modalities:
        candidate.evidence_modalities.append("text_overlap_support")
    for key, value in text_score_parts.items():
        candidate.score_parts[key] = max(candidate.score_parts.get(key, 0.0), value)
    scoring = candidate.provenance.setdefault("scoring", {})
    scoring.setdefault("text_hits", []).append(text_scoring_audit)
    direct_text_atom_ids = candidate.provenance.setdefault("direct_text_atom_ids", [])
    if hit.text_atom_id and hit.text_atom_id not in direct_text_atom_ids:
        direct_text_atom_ids.append(hit.text_atom_id)


def solve_interval_task(
    config: AppConfig,
    query: TaskQuery,
    candidates: Sequence[EventCandidate],
    coverage: CoverageIndex,
    output_dir: Path,
    *,
    stage_manifests: dict[str, dict] | None = None,
    event_store: EventStore | None = None,
) -> tuple[list[dict], list[EvidenceBundle], dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = query.resolved_profiles["output_profile"]["value"]
    normalized_candidates = sorted(list(candidates), key=lambda item: (-float(item.final_score), item.candidate_id))
    threshold = _calibrated_threshold(query, normalized_candidates)
    row_candidates = []
    rejected = []
    for candidate in normalized_candidates:
        threshold_decision = _candidate_threshold_decision(query, candidate, threshold)
        candidate.provenance.setdefault("scoring", {})["threshold_decision"] = threshold_decision
        candidate.provenance["threshold_decision"] = threshold_decision
        allowed_text_marker = "transcript_marker_primary" in candidate.evidence_modalities and query.evidence_policy.get("allow_transcript_marker_primary")
        needs_frame = query.task_type == "object_instance_search" or not allowed_text_marker
        if allowed_text_marker and "visual_direct_frame" not in candidate.evidence_modalities and not _candidate_has_lexical_marker_hit(candidate):
            candidate.rejected_reason = "missing_lexical_marker_hit"
            rejected.append(candidate)
            continue
        if needs_frame and "visual_direct_frame" not in candidate.evidence_modalities:
            candidate.rejected_reason = "direct_frame_evidence_required"
            rejected.append(candidate)
            continue
        if not threshold_decision["pass"]:
            candidate.rejected_reason = str(threshold_decision["reason"])
            rejected.append(candidate)
            continue
        if candidate.final_score < threshold["hard_reject_below_score"]:
            candidate.rejected_reason = "below_hard_reject"
            rejected.append(candidate)
            continue
        if candidate.final_score < threshold["submit_min_score"]:
            candidate.rejected_reason = "below_submit_threshold"
            rejected.append(candidate)
            continue
        clips = coverage.clip(candidate.day, candidate.source, candidate.start_second, candidate.end_second)
        if not clips:
            candidate.rejected_reason = "no_canonical_frame_coverage"
            rejected.append(candidate)
            continue
        for clip in clips:
            try:
                normalized = normalize_output_interval(clip.start, clip.end, coverage_end=clip.end)
            except ValueError:
                continue
            row = _candidate_row(candidate, normalized, event_store=event_store)
            if allowed_text_marker and _row_requires_audio_marker_text(row) and not row.get("text_atom_ids"):
                continue
            if abs(float(normalized.start) - float(candidate.start_second)) > 1e-6 or abs(float(normalized.end) - float(candidate.end_second)) > 1e-6:
                if needs_frame and "visual_direct_frame" in candidate.evidence_modalities and not row.get("frame_atom_ids"):
                    continue
                row["split_reason"] = row.get("split_reason") or "coverage_clip"
                provenance = row.setdefault("provenance", {})
                provenance["coverage_clip"] = {
                    "candidate_start_second": float(candidate.start_second),
                    "candidate_end_second": float(candidate.end_second),
                    "clip_start_second": float(normalized.start),
                    "clip_end_second": float(normalized.end),
                }
            row_candidates.append(row)
    rows = nms_intervals(row_candidates, iou_threshold=float(profile.get("nms_iou", 0.6)))
    rows = _merge_adjacent_rows(rows, merge_gap_sec=float(profile.get("merge_gap_sec", 0.0)), query=query)
    if _sort_interval_rows_by_final_score_enabled(config):
        rows = _sort_interval_rows_by_final_score(rows)
    hard_cap_hit = False
    evidence = [_evidence_from_row(row) for row in rows]
    calibration = dict(threshold)
    stage_audit = _solver_stage_audit(stage_manifests or {})
    calibration.update(
        {
            "task_id": query.task_id,
            "candidate_count": len(normalized_candidates),
            "profile": query.threshold_profile,
            "text_rerank_cap_hit": stage_audit["text_rerank"]["cap_hit"],
            "patch_full_recall_completed": stage_audit["visual_patch_rerank"]["full_recall_completed"],
            "stage_manifests": stage_audit,
        }
    )
    required_stage_status = _required_solver_stage_status(query, stage_audit)
    tail_audit = _build_tail_audit(query, normalized_candidates, rows, threshold, hard_cap_hit=hard_cap_hit)
    local_visual_recovery_audit = _local_visual_recovery_audit_records(query, rows, rejected)
    report = {
        "task_id": query.task_id,
        "candidate_count": len(normalized_candidates),
        "row_count": len(rows),
        "rejected_count": len(rejected),
        "threshold": threshold,
        "score_calibration": calibration,
        "tail_audit": tail_audit,
        "stage_manifests": stage_audit,
        "required_stage_status": required_stage_status,
        "hard_cap_hit": hard_cap_hit,
        "zero_output": len(rows) == 0,
        "local_visual_recovery_audit_count": len(local_visual_recovery_audit),
    }
    atomic_write_jsonl(output_dir / "result.jsonl", rows)
    atomic_write_jsonl(output_dir / "evidence.jsonl", [item.asdict() for item in evidence])
    atomic_write_json(output_dir / "score_calibration.json", calibration)
    atomic_write_json(output_dir / "recall_tail_audit.json", tail_audit)
    atomic_write_jsonl(output_dir / "rejected_candidates.jsonl", [_rejected_candidate_row(candidate) for candidate in rejected])
    required_files = ["result.jsonl", "evidence.jsonl", "solver_report.json", "score_calibration.json", "recall_tail_audit.json"]
    if query.evidence_policy.get("require_local_visual_recovery_audit"):
        atomic_write_jsonl(output_dir / "local_visual_recovery_audit.jsonl", local_visual_recovery_audit)
        required_files.append("local_visual_recovery_audit.jsonl")
    atomic_write_json(output_dir / "solver_report.json", report)
    completion_status = "completed" if required_stage_status["completed"] else "failed"
    write_task_manifest(
        config,
        query,
        output_dir,
        status=completion_status,
        required_files=tuple(required_files),
        stage_reports={"solver": report},
        extra={"row_count": len(rows), "evidence_count": len(evidence), "rejected_count": len(rejected), "required_stage_status": required_stage_status},
    )
    if completion_status == "completed":
        write_completed_marker(output_dir)
    else:
        marker = output_dir / "completed.marker"
        if marker.exists():
            marker.unlink()
    return rows, evidence, report


def _solver_stage_audit(stage_manifests: dict[str, dict]) -> dict:
    text_manifest = stage_manifests.get("text_rerank") or {}
    text_cap = text_manifest.get("cap_boundary_audit") if isinstance(text_manifest, dict) else {}
    patch_manifest = stage_manifests.get("visual_patch_rerank") or {}
    patch_input_count = int(patch_manifest.get("input_count") or 0) if isinstance(patch_manifest, dict) else 0
    patch_scored_count = int(patch_manifest.get("scored_count") or 0) if isinstance(patch_manifest, dict) else 0
    patch_failure_count = int(patch_manifest.get("failure_count") or 0) if isinstance(patch_manifest, dict) else 0
    local_manifest = stage_manifests.get("local_visual_recovery") or {}
    return {
        "text_rerank": {
            "status": text_manifest.get("status") if isinstance(text_manifest, dict) else None,
            "cap_hit": bool(isinstance(text_cap, dict) and text_cap.get("cap_hit") is True),
            "cap_boundary_audit": text_cap if isinstance(text_cap, dict) else {},
        },
        "visual_patch_rerank": {
            "status": patch_manifest.get("status") if isinstance(patch_manifest, dict) else None,
            "input_count": patch_input_count,
            "scored_count": patch_scored_count,
            "failure_count": patch_failure_count,
            "full_recall_completed": patch_manifest.get("status") == "completed" and patch_failure_count == 0 and patch_scored_count == patch_input_count,
        },
        "local_visual_recovery": {
            "status": local_manifest.get("status") if isinstance(local_manifest, dict) else None,
            "candidate_frame_count": int(local_manifest.get("candidate_frame_count") or 0) if isinstance(local_manifest, dict) else 0,
            "scored_frame_count": int(local_manifest.get("scored_frame_count") or 0) if isinstance(local_manifest, dict) else 0,
            "policy": local_manifest.get("policy") if isinstance(local_manifest, dict) else None,
        },
    }


def _required_solver_stage_status(query: TaskQuery, stage_audit: dict) -> dict:
    reasons = []
    text = stage_audit.get("text_rerank") or {}
    patch = stage_audit.get("visual_patch_rerank") or {}
    local = stage_audit.get("local_visual_recovery") or {}
    if text.get("status") != "completed":
        reasons.append("text_rerank_not_completed")
    if patch.get("full_recall_completed") is not True:
        reasons.append("visual_patch_rerank_full_recall_not_completed")
    if local.get("status") == "failed":
        reasons.append("local_visual_recovery_failed")
    if query.evidence_policy.get("allow_transcript_marker_primary") and local.get("status") not in {"completed", "skipped_no_text_events"}:
        reasons.append("audio_sensitive_local_visual_recovery_not_completed")
    return {"completed": not reasons, "block_reasons": reasons}


def _events_for_frame(hit: FrameHit, store: EventStore) -> list[dict]:
    if not hit.frame_atom_id:
        return []
    rows = []
    for link in store.frame_links_by_frame.get(hit.frame_atom_id, []):
        if link.get("link_type") != "frame_overlap_support" or link.get("link_role") not in DIRECT_FRAME_ROLES:
            continue
        event = store.event_by_id.get(link["event_id"])
        if not event:
            continue
        for source_event in _source_specific_frame_events(event, hit, store):
            linked_event = _event_with_link_scores(source_event, link, modality="frame")
            rows.append(_event_with_graph_scores(linked_event, store))
    return rows


def _source_specific_frame_events(event: dict, hit: FrameHit, store: EventStore) -> list[dict]:
    if event.get("source") is not None and event.get("level") in {"L1", "L2"}:
        return [event]
    descendants = []
    visited = {str(event.get("event_id") or "")}
    stack = [(str(event.get("event_id") or ""), 0)]
    frame_second = float(hit.absolute_second)
    while stack:
        parent_id, depth = stack.pop()
        if depth >= 4:
            continue
        for link in store.child_links_by_parent.get(parent_id, []):
            if link.get("link_type") != "temporal_contains" or link.get("link_role") != "primary_parent":
                continue
            child_id = str(link.get("child_event_id") or "")
            if not child_id or child_id in visited:
                continue
            visited.add(child_id)
            child = store.event_by_id.get(child_id)
            if not child:
                continue
            if not _event_contains_second(child, frame_second):
                continue
            if child.get("source") is not None and child.get("level") in {"L1", "L2"}:
                descendants.append(child)
            elif child.get("level") in {"L3", "L4", "L5"}:
                stack.append((child_id, depth + 1))
    if descendants:
        return descendants
    if event.get("source") is not None:
        return [event]
    return []


def _parent_l2_event_ids_for_event(event: dict, store: EventStore) -> list[str]:
    event_id = str(event.get("event_id") or "")
    level = str(event.get("level") or "")
    if level == "L2" and event_id:
        return [event_id]
    parent_ids = []
    primary_parent_id = str(event.get("primary_parent_id") or "")
    if primary_parent_id:
        parent = store.event_by_id.get(primary_parent_id)
        if (parent or {}).get("level") == "L2":
            parent_ids.append(primary_parent_id)
    if level == "L1" and event_id and not parent_ids:
        for parent_id, links in store.child_links_by_parent.items():
            for link in links:
                if link.get("link_type") != "temporal_contains" or link.get("link_role") != "primary_parent":
                    continue
                if str(link.get("child_event_id") or "") != event_id:
                    continue
                parent = store.event_by_id.get(str(parent_id))
                if (parent or {}).get("level") == "L2":
                    parent_ids.append(str(parent_id))
                    break
            if parent_ids:
                break
    return sorted(dict.fromkeys(parent_ids))


def _child_l1_events_for_l2_at_second(l2_event: dict, second: float, store: EventStore) -> list[dict]:
    l2_id = str(l2_event.get("event_id") or "")
    if not l2_id:
        return []
    children = []
    for link in store.child_links_by_parent.get(l2_id, []):
        if link.get("link_type") != "temporal_contains" or link.get("link_role") != "primary_parent":
            continue
        child = store.event_by_id.get(str(link.get("child_event_id") or ""))
        if not child or child.get("level") != "L1":
            continue
        if child.get("source") != l2_event.get("source"):
            continue
        if int(child.get("day_num") or l2_event.get("day_num") or 0) != int(l2_event.get("day_num") or 0):
            continue
        if _event_contains_second(child, second):
            children.append(child)
    if children:
        return children
    for child in store.event_by_id.values():
        if child.get("level") != "L1" or str(child.get("primary_parent_id") or "") != l2_id:
            continue
        if child.get("source") != l2_event.get("source"):
            continue
        if int(child.get("day_num") or l2_event.get("day_num") or 0) != int(l2_event.get("day_num") or 0):
            continue
        if _event_contains_second(child, second):
            children.append(child)
    return children


def _ois_supported_interval(query: TaskQuery, event: dict, hit: FrameHit, support_hits: Sequence[FrameHit] | None = None) -> tuple[float, float, dict]:
    profile = (query.resolved_profiles.get("output_profile") or {}).get("value") or {}
    configured_merge_gap = float(profile.get("merge_gap_sec", 5.0))
    merge_gap = _ois_effective_gap(configured_merge_gap, query=query, same_parent_l2=False)
    half_window = 3.0 if _is_static_fixed_ois_query(query) else 2.0
    raw_event_start = event.get("start_second")
    raw_event_end = event.get("end_second")
    event_start = float(raw_event_start) if raw_event_start is not None else max(0.0, float(hit.absolute_second) - half_window)
    event_end = float(raw_event_end) if raw_event_end is not None else min(86399.0, float(hit.absolute_second) + half_window + 1.0)
    scoped_hits = [
        item
        for item in (support_hits or [hit])
        if int(item.day) == int(event.get("day_num") or hit.day) and str(item.source) == str(event.get("source") or hit.source)
    ]
    if not scoped_hits:
        scoped_hits = [hit]
    best_support_score = max(_visual_support_score(item) for item in scoped_hits)
    seed_support_score = _visual_support_score(hit)
    continuation_floor = max(0.0, min(0.45, max(best_support_score, seed_support_score) * 0.60))
    floor_hits = [item for item in scoped_hits if _visual_support_score(item) >= continuation_floor]
    if not floor_hits:
        floor_hits = [hit]
    floor_hits.sort(key=lambda item: (float(item.absolute_second), item.embedding_id))
    support_segments: list[list[FrameHit]] = []
    for item in floor_hits:
        if not support_segments:
            support_segments.append([item])
            continue
        previous = support_segments[-1][-1]
        if float(item.absolute_second) - float(previous.absolute_second) <= merge_gap:
            support_segments[-1].append(item)
        else:
            support_segments.append([item])
    seed_second = float(hit.absolute_second)
    selected_segment = next(
        (
            segment
            for segment in support_segments
            if float(segment[0].absolute_second) <= seed_second <= float(segment[-1].absolute_second)
        ),
        None,
    )
    if selected_segment is None:
        selected_segment = max(
            support_segments,
            key=lambda segment: (
                sum(_visual_support_score(item) for item in segment),
                -min(abs(float(item.absolute_second) - seed_second) for item in segment),
            ),
        )
    support_seconds = [float(item.absolute_second) for item in scoped_hits]
    selected_seconds = [float(item.absolute_second) for item in selected_segment]
    supported_start = max(event_start, min(selected_seconds) - half_window)
    supported_end = min(event_end, max(selected_seconds) + half_window + 1.0)
    if supported_end <= supported_start:
        supported_start = max(event_start, float(hit.absolute_second))
        supported_end = min(event_end, supported_start + 1.0)
    audit = {
        "policy": "patch_continuity_supported_interval_v1",
        "support_frame_second": float(hit.absolute_second),
        "support_frame_seconds": support_seconds,
        "selected_support_frame_seconds": selected_seconds,
        "support_frame_atom_ids": [item.frame_atom_id for item in scoped_hits if item.frame_atom_id],
        "selected_support_frame_atom_ids": [item.frame_atom_id for item in selected_segment if item.frame_atom_id],
        "support_frame_count": len(scoped_hits),
        "selected_support_frame_count": len(selected_segment),
        "patch_scored_support_frame_count": sum(1 for item in scoped_hits if item.patch_score is not None),
        "continuation_floor": continuation_floor,
        "best_support_score": best_support_score,
        "seed_support_score": seed_support_score,
        "segment_count": len(support_segments),
        "segment_frame_seconds": [[float(item.absolute_second) for item in segment] for segment in support_segments],
        "event_start_second": event_start,
        "event_end_second": event_end,
        "supported_start_second": supported_start,
        "supported_end_second": supported_end,
        "merge_gap_sec": merge_gap,
        "configured_merge_gap_sec": configured_merge_gap,
        "half_window_sec": half_window,
        "patch_score": hit.patch_score,
        "cls_score": hit.cls_score,
        "rescan_scope": "visual_patch_rerank_candidate_set",
    }
    return supported_start, supported_end, audit


def _visual_support_score(hit: FrameHit) -> float:
    if hit.patch_score is not None:
        return _bounded(hit.patch_score)
    return _bounded(hit.cls_score)


def _events_for_text(hit: TextHit, store: EventStore) -> list[dict]:
    if not hit.text_atom_id:
        return []
    rows = []
    text_interval = Interval(float(hit.start_second or 0.0), float(hit.end_second or (hit.start_second or 0.0) + 1))
    for link in store.text_links_by_text.get(hit.text_atom_id, []):
        if link.get("link_type") != "text_overlap_support":
            continue
        event = store.event_by_id.get(link["event_id"])
        if not event:
            continue
        for source_event in _source_specific_text_events(event, text_interval, store):
            event_interval = Interval(float(source_event["start_second"]), float(source_event["end_second"]))
            denom = max(min(text_interval.duration(), event_interval.duration()), 1e-9)
            if overlap_duration(text_interval, event_interval) / denom >= 0.10:
                linked_event = _event_with_link_scores(source_event, link, modality="text")
                rows.append(_event_with_graph_scores(linked_event, store))
    return rows


def _event_with_link_scores(event: dict, link: dict, *, modality: str) -> dict:
    support_score = _link_support_score(link)
    if support_score is None:
        return event
    enriched = dict(event)
    if modality == "frame":
        enriched["visual_centroid_similarity"] = max(float(enriched.get("visual_centroid_similarity") or 0.0), support_score)
    elif modality == "text":
        enriched["text_dense_centroid_similarity"] = max(float(enriched.get("text_dense_centroid_similarity") or 0.0), support_score)
    enriched["event_embedding_score_source"] = {
        "modality": modality,
        "link_type": link.get("link_type"),
        "link_role": link.get("link_role"),
        "score_field": "selection_score" if link.get("selection_score") is not None else "score",
    }
    return enriched


def _link_support_score(link: dict) -> float | None:
    value = link.get("selection_score")
    if value is None:
        value = link.get("score")
    if value is None:
        return None
    return _bounded(float(value))


def _event_with_graph_scores(event: dict, store: EventStore) -> dict:
    event_id = str(event.get("event_id") or "")
    cross_links_by_event = getattr(store, "cross_source_links_by_event", {})
    cross_links = cross_links_by_event.get(event_id, [])
    if not cross_links:
        return event
    scores = []
    for link in cross_links:
        value = link.get("selection_score")
        if value is None:
            value = link.get("score")
        if value is not None:
            scores.append(_bounded(float(value)))
    if not scores:
        return event
    enriched = dict(event)
    enriched["cross_view_support_score"] = max(scores)
    enriched["cross_view_support_count"] = len(scores)
    return enriched


def _source_specific_text_events(event: dict, text_interval: Interval, store: EventStore) -> list[dict]:
    if event.get("source") is not None:
        return [event]
    descendants = []
    visited = {str(event.get("event_id") or "")}
    stack = [(str(event.get("event_id") or ""), 0)]
    while stack:
        parent_id, depth = stack.pop()
        if depth >= 4:
            continue
        for link in store.child_links_by_parent.get(parent_id, []):
            if link.get("link_type") != "temporal_contains" or link.get("link_role") != "primary_parent":
                continue
            child_id = str(link.get("child_event_id") or "")
            if not child_id or child_id in visited:
                continue
            visited.add(child_id)
            child = store.event_by_id.get(child_id)
            if not child:
                continue
            child_interval = Interval(float(child["start_second"]), float(child["end_second"]))
            if overlap_duration(text_interval, child_interval) <= 0:
                continue
            if child.get("source") is not None and child.get("level") in {"L1", "L2"}:
                descendants.append(child)
            elif child.get("level") in {"L3", "L4", "L5"}:
                stack.append((child_id, depth + 1))
    return descendants


def _event_contains_second(event: dict, second: float) -> bool:
    start = event.get("start_second")
    end = event.get("end_second")
    if start is None or end is None:
        return True
    return float(start) <= second < float(end)


def _local_recovery_frame_ids(store: EventStore, event: dict, hit: TextHit, *, max_frames: int = 8) -> list[str]:
    representative_ids = [str(frame_id) for frame_id in event.get("representative_frame_ids") or [] if str(frame_id)]
    start = float(event.get("start_second") or hit.start_second or 0.0)
    end = float(event.get("end_second") or hit.end_second or start + 1.0)
    day = int(event.get("day_num") or hit.day)
    source = str(event.get("source") or hit.source)
    local_ids = []
    for frame in store.frames_by_day_source.get((day, source), []):
        second = float(frame.get("absolute_second") or -1.0)
        if start <= second < end and frame.get("frame_atom_id"):
            local_ids.append(str(frame["frame_atom_id"]))
    merged = list(dict.fromkeys(representative_ids + local_ids))
    return merged[:max_frames]


def _audio_sensitive_marker_provenance(query: TaskQuery, hit: TextHit, event: dict, frame_atom_ids: Sequence[str]) -> dict:
    if not query.evidence_policy.get("require_local_visual_recovery_audit"):
        return {}
    sweep = query.raw.get("lexical_marker_sweep") or {}
    marker_terms = [str(item) for item in sweep.get("marker_terms") or []]
    text = hit.text_for_display or hit.text_for_embedding or ""
    marker_hits = [term for term in marker_terms if term.lower() in text.lower()]
    status = "completed" if frame_atom_ids else "completed_no_visual_confirmation"
    start = float(event.get("start_second") or hit.start_second or 0.0)
    end = float(event.get("end_second") or hit.end_second or start + 1.0)
    return {
        "lexical_marker_sweep_id": _lexical_marker_sweep_id(query, marker_terms),
        "accepted_source_interval_ref": _source_interval_ref(int(event.get("day_num") or hit.day), str(event.get("source") or hit.source), start, end),
        "local_visual_recovery_status": status,
        "local_visual_recovery_audit": {
            "audit_version": "local_visual_recovery_v1",
            "lexical_marker_terms": marker_terms,
            "lexical_marker_hits": marker_hits,
            "text_atom_id": hit.text_atom_id,
            "event_id": str(event.get("event_id") or ""),
            "event_level": event.get("level"),
            "representative_frame_ids": list(frame_atom_ids),
            "source_video_coverage_required": bool(query.evidence_policy.get("require_source_video_coverage")),
            "status": status,
        },
    }


def _prepend_lexical_marker_sweep_hits(query: TaskQuery, text_hits: Sequence[TextHit], event_store: EventStore) -> list[TextHit]:
    if not query.evidence_policy.get("allow_transcript_marker_primary"):
        return list(text_hits)
    sweep = query.raw.get("lexical_marker_sweep") or {}
    if not sweep.get("required"):
        return list(text_hits)
    marker_terms = [str(item) for item in sweep.get("marker_terms") or [] if str(item).strip()]
    if not marker_terms:
        return list(text_hits)
    scan_inputs = [str(item) for item in sweep.get("scan_inputs") or ["text_atoms"] if str(item).strip()]
    if not scan_inputs:
        scan_inputs = ["text_atoms"]
    requested_inputs = set(scan_inputs)
    seen_chunk_ids = {hit.embedding_chunk_id for hit in text_hits}
    sweep_hits: list[TextHit] = []
    if "text_atoms" in requested_inputs:
        for chunk_id, text_atom in event_store.text_by_chunk.items():
            hit = _lexical_marker_sweep_text_atom_hit(query, marker_terms, scan_inputs, chunk_id, text_atom, seen_chunk_ids)
            if hit is None:
                continue
            seen_chunk_ids.add(hit.embedding_chunk_id)
            sweep_hits.append(hit)
    if "cleaned_transcripts" in requested_inputs:
        iter_rows = getattr(event_store, "iter_cleaned_transcript_rows", None)
        if callable(iter_rows):
            for row in iter_rows():
                hit = _lexical_marker_sweep_cleaned_transcript_hit(query, marker_terms, scan_inputs, row, seen_chunk_ids)
                if hit is None:
                    continue
                seen_chunk_ids.add(hit.embedding_chunk_id)
                sweep_hits.append(hit)
    sweep_hits.sort(key=lambda hit: (hit.day, hit.source, float(hit.start_second or 0.0), hit.embedding_chunk_id))
    return sweep_hits + list(text_hits)


def _lexical_marker_sweep_text_atom_hit(
    query: TaskQuery,
    marker_terms: Sequence[str],
    scan_inputs: Sequence[str],
    chunk_id: str,
    text_atom: dict,
    seen_chunk_ids: set[str],
) -> TextHit | None:
    if chunk_id in seen_chunk_ids:
        return None
    text = str(text_atom.get("text_preview") or "")
    marker_hits = _lexical_marker_hits(marker_terms, text)
    if not marker_hits:
        return None
    start = text_atom.get("absolute_start_second")
    end = text_atom.get("absolute_end_second")
    if start is None or end is None:
        return None
    return TextHit(
        task_id=query.task_id,
        prompt="lexical_marker_sweep",
        embedding_chunk_id=str(chunk_id),
        day=int(text_atom.get("day_num") or 0),
        source=str(text_atom.get("source") or ""),
        start_second=float(start),
        end_second=float(end),
        text_for_embedding=text,
        text_for_display=text,
        dense_score=None,
        sparse_score=None,
        reranker_score=1.0,
        reranker_raw_score=1.0,
        text_atom_id=str(text_atom.get("text_atom_id") or ""),
        quality_flags=list(text_atom.get("quality_flags") or []),
        provenance={
            "lexical_marker_sweep": {
                "sweep_id": _lexical_marker_sweep_id(query, marker_terms),
                "scan_input": "text_atoms",
                "scan_inputs": list(scan_inputs),
                "full_text_atom_sweep": True,
                "marker_hits": marker_hits,
            }
        },
    )


def _lexical_marker_sweep_cleaned_transcript_hit(
    query: TaskQuery,
    marker_terms: Sequence[str],
    scan_inputs: Sequence[str],
    row: dict,
    seen_chunk_ids: set[str],
) -> TextHit | None:
    chunk_id = str(row.get("chunk_id") or row.get("embedding_chunk_id") or "").strip()
    if not chunk_id or chunk_id in seen_chunk_ids:
        return None
    text = str(row.get("cleaned_text") or row.get("text") or row.get("raw_text") or "")
    marker_hits = _lexical_marker_hits(marker_terms, text)
    if not marker_hits:
        return None
    start = row.get("start_time")
    end = row.get("end_time")
    if start is None or end is None:
        return None
    source = str(row.get("view") or row.get("source") or "")
    day = _cleaned_transcript_day_num(row.get("day"))
    quality_flags = [str(item) for item in row.get("quality_flags") or []]
    if row.get("timestamp_valid") is False and "time_alignment_status=invalid" not in quality_flags:
        quality_flags.append("time_alignment_status=invalid")
    return TextHit(
        task_id=query.task_id,
        prompt="lexical_marker_sweep",
        embedding_chunk_id=chunk_id,
        day=day,
        source=source,
        start_second=float(start),
        end_second=float(end),
        text_for_embedding=text,
        text_for_display=text,
        dense_score=None,
        sparse_score=None,
        reranker_score=1.0,
        reranker_raw_score=1.0,
        text_atom_id=str(row.get("text_atom_id") or ""),
        quality_flags=quality_flags,
        provenance={
            "lexical_marker_sweep": {
                "sweep_id": _lexical_marker_sweep_id(query, marker_terms),
                "scan_input": "cleaned_transcripts",
                "scan_inputs": list(scan_inputs),
                "full_cleaned_transcript_sweep": True,
                "marker_hits": marker_hits,
                "source_file": row.get("source_file"),
                "raw_chunk_index": row.get("raw_chunk_index"),
            }
        },
    )


def _cleaned_transcript_day_num(value: object) -> int:
    text = str(value or "").strip().lower()
    if text.startswith("day"):
        text = text[3:]
    try:
        return int(text)
    except ValueError:
        return 0


def _lexical_marker_hits(marker_terms: Sequence[str], text: str) -> list[str]:
    result = []
    for term in marker_terms:
        term_text = str(term).strip()
        if not term_text:
            continue
        if _lexical_marker_term_matches(term_text, text):
            result.append(term_text)
    return result


def _lexical_marker_term_matches(term: str, text: str) -> bool:
    if not text:
        return False
    if re.search(r"^[A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)*$", term):
        return re.search(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", text, flags=re.IGNORECASE) is not None
    return term.lower() in text.lower()


def _candidate_has_lexical_marker_hit(candidate: EventCandidate) -> bool:
    audit = (candidate.provenance or {}).get("local_visual_recovery_audit") or {}
    hits = audit.get("lexical_marker_hits")
    return isinstance(hits, list) and bool(hits)


def _lexical_marker_sweep_id(query: TaskQuery, marker_terms: Sequence[str]) -> str:
    payload = "\n".join([query.task_id, *marker_terms])
    return f"{query.task_id}:lexical-marker:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"


def _source_interval_ref(day: int, source: str, start: float, end: float) -> str:
    return f"{source} day {int(day)} {format_submission_hms(start)}-{format_submission_hms(end)}"


def _text_score_parts(query: TaskQuery, hit: TextHit, fallback_score: float, *, event: dict | None = None) -> tuple[dict[str, float], dict]:
    text_score = _raw_text_score(hit) if hit.reranker_score is not None or hit.dense_score is not None or hit.sparse_score is not None else float(fallback_score)
    negative_score = _raw_negative_text_score(hit)
    quality_multiplier, quality_audit = _text_quality_multiplier(query, hit)
    if quality_multiplier <= 0.0:
        text_score = 0.0
    text_score, lexical_audit = _task_text_lexical_adjustment(query, hit.text_for_display or hit.text_for_embedding, text_score)
    return {
        "text_pos_raw": float(text_score),
        "text_neg_raw": float(negative_score),
    }, {
        "policy": "direct_event_text_raw_reranker_v1",
        "embedding_chunk_id": hit.embedding_chunk_id,
        "text_atom_id": hit.text_atom_id,
        "event_id": str((event or {}).get("event_id") or ""),
        "event_level": (event or {}).get("level"),
        "text_pos_raw": float(text_score),
        "text_neg_raw": float(negative_score),
        "positive_score_source": "reranker_score" if hit.reranker_score is not None else "dense_or_sparse_score",
        "negative_score_source": "negative_reranker_score" if hit.negative_reranker_score is not None else "missing_negative_rerank",
        "quality_multiplier": quality_audit,
        "lexical_adjustment": lexical_audit,
    }


def _task_text_lexical_adjustment(query: TaskQuery, text: str, score: float) -> tuple[float, dict]:
    if query.task_id != "MM26-EIS11":
        return float(score), {"policy": "not_applicable"}
    normalized = _normalize_prompt_match_text(text)
    primary_terms = ("schnapsen", "schnappsen")
    rule_terms = ("trump", "ober", "unter", "65", "66", "points", "ace", "king", "queen", "jack")
    primary_hits = [term for term in primary_terms if _normalized_term_present(normalized, term)]
    rule_hits = [term for term in rule_terms if _normalized_term_present(normalized, term)]
    boost = 0.0
    if primary_hits:
        boost += 0.10
    if len(rule_hits) >= 2:
        boost += 0.05
    elif rule_hits:
        boost += 0.03
    adjusted = min(1.0, float(score) + boost)
    return adjusted, {
        "policy": "eis11_schnapsen_rule_terms_v1",
        "matched_primary_terms": primary_hits,
        "matched_rule_terms": rule_hits,
        "boost": boost,
        "original_score": float(score),
        "adjusted_score": adjusted,
    }


def _normalized_term_present(normalized_text: str, normalized_term: str) -> bool:
    term = _normalize_prompt_match_text(normalized_term)
    return bool(term and re.search(rf"(?<!\w){re.escape(term)}(?!\w)", normalized_text))


def _text_quality_multiplier(query: TaskQuery, hit: TextHit) -> tuple[float, dict]:
    flags = [str(item).strip().lower() for item in (hit.quality_flags or []) if str(item).strip()]
    flag_text = " ".join(flags)
    multiplier = 1.0
    reasons: list[str] = []
    invalid_time = (
        hit.start_second is None
        or hit.end_second is None
        or float(hit.end_second) <= float(hit.start_second)
        or "time_alignment_status=invalid" in flag_text
        or "time_alignment_status:invalid" in flag_text
        or "invalid_time" in flag_text
        or "invalid time" in flag_text
    )
    if invalid_time and query.task_type in {"event_instance_search", "object_instance_search"}:
        return 0.0, {"multiplier": 0.0, "flags": flags, "reasons": ["invalid_time_for_interval_task"]}
    if "contains_invalid_source_chunk" in flag_text or "invalid_source_chunk" in flag_text:
        multiplier = min(multiplier, 0.55)
        reasons.append("contains_invalid_source_chunk")
    if invalid_time:
        multiplier = min(multiplier, 0.75)
        reasons.append("invalid_time_range_recovered")
    if "too_long_duration" in flag_text or "too_long" in flag_text:
        multiplier = min(multiplier, 0.85)
        reasons.append("too_long_duration")
    if "repetition_detected" in flag_text or "repetition_compressed" in flag_text or "repetition" in flag_text:
        multiplier = min(multiplier, 0.90)
        reasons.append("repetition")
    return float(multiplier), {"multiplier": float(multiplier), "flags": flags, "reasons": reasons}


def _normalize_prompt_match_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", str(text).lower())).strip()


def _bounded(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    # Similarity values can be negative; map smoothly to [0, 1].
    return 1.0 / (1.0 + math.exp(-float(value) * 4.0))


def _apply_object_state_probes(query: TaskQuery, candidates: Sequence[EventCandidate], text_hits: Sequence[TextHit]) -> None:
    constraints = query.raw.get("object_state_constraints") or {}
    rules = constraints.get("state_fit_rules") or []
    if not rules:
        return
    excluded_contexts = [str(item).lower() for item in constraints.get("excluded_contexts") or []]
    for candidate in candidates:
        context = _text_context_for_candidate(candidate, text_hits)
        trace = []
        fit_scores = []
        for rule in rules:
            result = _object_state_probe_result(str(rule), candidate, context, excluded_contexts)
            trace.append(result)
            fit_scores.append(float(result["fit_score"]))
        candidate.provenance["object_state_fit_trace"] = trace
        candidate.provenance["object_state_fit_summary"] = {
            "policy": "audit_only_not_final_score_v1",
            "min_fit_score": min(fit_scores) if fit_scores else None,
            "rule_count": len(rules),
        }


def _text_context_for_candidate(candidate: EventCandidate, text_hits: Sequence[TextHit]) -> str:
    snippets = []
    candidate_interval = Interval(candidate.start_second, candidate.end_second)
    for hit in text_hits:
        if int(hit.day) != int(candidate.day) or str(hit.source) != str(candidate.source):
            continue
        if hit.start_second is None or hit.end_second is None:
            continue
        hit_interval = Interval(float(hit.start_second), float(hit.end_second))
        if overlap_duration(candidate_interval, hit_interval) <= 0:
            continue
        snippets.append(hit.text_for_display or hit.text_for_embedding)
        if len(snippets) >= 5:
            break
    for excerpt in _candidate_local_recovery_seed_excerpts(candidate):
        if excerpt not in snippets:
            snippets.append(excerpt)
        if len(snippets) >= 8:
            break
    return "\n".join(snippets).lower()


def _candidate_local_recovery_seed_excerpts(candidate: EventCandidate) -> list[str]:
    provenance = candidate.provenance or {}
    records: list[dict] = []
    stored = provenance.get("local_visual_recovery_seed_texts")
    if isinstance(stored, list):
        records.extend(item for item in stored if isinstance(item, dict))
    frame_hit = provenance.get("frame_hit")
    if isinstance(frame_hit, dict):
        records.extend(_local_recovery_seed_records(frame_hit))
    excerpts = []
    seen = set()
    for record in records:
        text = str(record.get("seed_text_excerpt") or "").strip()
        if not text:
            continue
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if text_hash in seen:
            continue
        seen.add(text_hash)
        excerpts.append(text)
    return excerpts


def _object_state_probe_result(rule: str, candidate: EventCandidate, context: str, excluded_contexts: Sequence[str]) -> dict:
    boundary = (candidate.provenance or {}).get("ois_boundary_refinement") or {}
    measured = {
        "source": candidate.source,
        "context_terms": [term for term in excluded_contexts if term and term in context],
        "support_frame_count": boundary.get("support_frame_count"),
        "patch_scored_support_frame_count": boundary.get("patch_scored_support_frame_count"),
    }
    if rule == "color_probe":
        color_features = _color_features_for_candidate(candidate)
        if color_features:
            measured["visual_color_features"] = color_features
            if "blue" in context and color_features.get("blue_pixel_ratio", 0.0) >= 0.18:
                return _probe(rule, "pass", 1.0, 1.0, measured)
            if "black" in context and color_features.get("dark_pixel_ratio", 0.0) >= 0.18:
                return _probe(rule, "pass", 1.0, 1.0, measured)
            if "blue" in context and color_features.get("blue_pixel_ratio", 0.0) < 0.06:
                return _probe(rule, "fail", 0.0, 0.85, measured, reject_reason="blue_color_not_supported")
            if "black" in context and color_features.get("dark_pixel_ratio", 0.0) < 0.06:
                return _probe(rule, "fail", 0.0, 0.85, measured, reject_reason="black_color_not_supported")
        if any(term in context for term in ("blue", "black")):
            return _probe(rule, "unknown", 0.85, 0.85, measured, note="color constraint present but visual color measurement unavailable")
        return _probe(rule, "unknown", 0.85, 0.85, measured, note="no matching color constraint in local context")
    if rule == "static_vs_in_use_probe":
        if any(term in context for term in ("active writing", "drawing with the pen", "drawing", "writing with")):
            return _probe(rule, "fail", 0.0, 0.45, measured, reject_reason="active_writing_context")
        support_count = int(boundary.get("selected_support_frame_count") or boundary.get("support_frame_count") or 0)
        seconds = [float(item) for item in boundary.get("selected_support_frame_seconds") or boundary.get("support_frame_seconds") or []]
        if support_count >= 3 and (not seconds or max(seconds) - min(seconds) >= 2.0):
            return _probe(rule, "pass", 1.0, 1.0, measured)
        if any(term in context for term in ("not in use", "unused", "lying", "on the table")):
            return _probe(rule, "pass", 1.0, 1.0, measured)
        return _probe(rule, "unknown", 0.80, 0.80, measured, note="no local object motion trace available")
    if rule == "cooking_context_probe":
        strong_non_cooking_tool_terms = (
            "screwdriver",
            "screw driver",
            "pliers",
            "cyber tool",
            "x-tool",
            "xtool",
            "multi tool",
            "multitool",
            "common bits",
            "bits",
            "drill",
            "hammer",
            "wrench",
            "spanner",
            "saw",
            "electronics",
            "repair",
        )
        ambiguous_tool_terms = (
            "scissors",
            "sharp knife",
            "knife",
        )
        explicit_non_cooking_usage = (
            "not using to cook",
            "not used to cook",
            "not being used to cook",
            "not using it to cook",
            "not used for cooking",
            "not used for cooking or eating",
            "not used to cook or eat",
            "not using to cook or eat",
        )
        cooking_terms = (
            "cooking utensil",
            "kitchen utensil",
            "food preparation",
            "cooking",
            "cook",
            "stove",
            "pan",
            "pot",
            "spatula",
            "whisk",
            "hand mixer",
            "knife",
            "bowl",
            "egg",
            "cookie",
        )
        hard_cooking_terms = (
            "cooking utensil",
            "kitchen utensil",
            "food preparation",
            "stove",
            "pan",
            "pot",
            "spatula",
            "whisk",
            "hand mixer",
            "bowl",
            "egg",
            "cookie",
        )
        measured["non_cooking_tool_terms"] = [term for term in strong_non_cooking_tool_terms if term in context]
        measured["ambiguous_tool_terms"] = [term for term in ambiguous_tool_terms if term in context]
        measured["explicit_non_cooking_usage"] = [term for term in explicit_non_cooking_usage if term in context]
        measured["cooking_terms"] = [term for term in cooking_terms if term in context]
        measured["hard_cooking_terms"] = [term for term in hard_cooking_terms if term in context]
        if measured["non_cooking_tool_terms"]:
            return _probe(rule, "pass", 1.0, 1.0, measured)
        if measured["ambiguous_tool_terms"] and measured["explicit_non_cooking_usage"] and not measured["hard_cooking_terms"]:
            measured["non_cooking_tool_terms"] = list(measured["ambiguous_tool_terms"])
            return _probe(rule, "pass", 0.90, 1.0, measured, note="ambiguous tool term has explicit non-cooking usage text")
        if measured["cooking_terms"]:
            return _probe(rule, "fail", 0.0, 0.40, measured, reject_reason="cooking_context")
        if measured["ambiguous_tool_terms"]:
            return _probe(rule, "unknown", 0.70, 0.80, measured, note="ambiguous tool term without explicit non-cooking usage")
        note = "kitchen source without cooking predicate" if candidate.source == "Kitchen" else "no cooking-context text evidence"
        return _probe(rule, "unknown", 0.80, 0.80, measured, note=note)
    if rule == "viewpoint_probe":
        patch = ((candidate.provenance or {}).get("frame_hit") or {}).get("patch") or {}
        image_path = patch.get("image_path")
        if candidate.source in FIXED_ROOM_SOURCES:
            return _probe(rule, "fail", 0.0, 0.55, measured, reject_reason="fixed_room_distant_view")
        if any(term in context for term in ("front", "screen", "ego", "phone", "smartphone")) and image_path:
            return _probe(rule, "pass", 1.0, 1.0, measured)
        return _probe(rule, "unknown", 0.75, 0.75, measured, note="no viewpoint saliency measurement available")
    if rule == "living_vs_toy_probe":
        if any(term in context for term in ("toy", "plush", "picture", "ornament")):
            return _probe(rule, "fail", 0.0, 0.35, measured, reject_reason="toy_or_picture_context")
        support_count = int(boundary.get("selected_support_frame_count") or boundary.get("support_frame_count") or 0)
        seconds = [float(item) for item in boundary.get("selected_support_frame_seconds") or boundary.get("support_frame_seconds") or []]
        if support_count >= 3 and (not seconds or max(seconds) - min(seconds) >= 2.0):
            return _probe(rule, "pass", 1.0, 1.0, measured)
        if any(term in context for term in ("animal", "pet", "living")):
            return _probe(rule, "pass", 1.0, 1.0, measured)
        return _probe(rule, "unknown", 0.70, 0.70, measured, note="no motion or living-context evidence")
    return _probe(rule, "unknown", 0.85, 0.85, measured, note="unknown probe rule")


def _color_features_for_candidate(candidate: EventCandidate) -> dict | None:
    patch = ((candidate.provenance or {}).get("frame_hit") or {}).get("patch") or {}
    image_path = patch.get("image_path")
    if not image_path:
        return None
    try:
        from PIL import Image

        with Image.open(str(image_path)) as image:
            rgb = image.convert("RGB")
            rgb.thumbnail((96, 96))
            pixels = list(rgb.getdata())
    except Exception:
        return None
    if not pixels:
        return None
    total = float(len(pixels))
    blue = sum(1 for red, green, blue_value in pixels if blue_value > red * 1.20 and blue_value > green * 1.10 and blue_value > 70)
    dark = sum(1 for red, green, blue_value in pixels if red + green + blue_value < 120)
    return {"blue_pixel_ratio": blue / total, "dark_pixel_ratio": dark / total, "pixel_sample_count": len(pixels)}


def _probe(rule: str, status: str, fit_score: float, multiplier: float, measured: dict, *, reject_reason: str | None = None, note: str | None = None) -> dict:
    return {
        "probe_name": rule,
        "rule_version": f"{rule}_v1",
        "status": status,
        "fit_score": fit_score,
        "applied_multiplier": multiplier,
        "evidence_refs": [],
        "measured_features": measured,
        "reject_reason": reject_reason,
        "note": note,
    }


def _sort_interval_rows_by_final_score_enabled(config: AppConfig | None) -> bool:
    if config is None:
        return True
    return bool(config.sort_interval_rows_by_final_score)


def _sort_interval_rows_by_final_score(rows: Sequence[dict]) -> list[dict]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            str(row.get("task_id") or row.get("task") or ""),
            -_float_or_default(row.get("final_score"), -1.0),
            int(row.get("day") or 0),
            str(row.get("source") or ""),
            _float_or_default(row.get("start_second"), -1.0),
            _float_or_default(row.get("end_second"), -1.0),
            str(row.get("csv_row_id") or ""),
        ),
    )


def _float_or_default(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _calibrated_threshold(query: TaskQuery, candidates: Sequence[EventCandidate]) -> dict:
    spec = _interval_score_spec(query)
    submit = 0.0
    hard = 0.0
    tail = -0.20
    return {
        "score_norm_version": "raw_floor_high_linear_v1",
        "threshold_source": "fixed_task_raw_floor_v1",
        "submit_min_score": submit,
        "hard_reject_below_score": hard,
        "tail_audit_min_score": tail,
        "knee_rank": None,
        "knee_threshold": None,
        "quantile_threshold": None,
        "visual_floor": float(spec["visual_floor"]),
        "text_floor": float(spec["text_floor"]),
        "require_visual": bool(spec.get("require_visual")),
        "require_text": bool(spec.get("require_text")),
        "weights": dict(spec["weights"]),
        "candidate_count": len(candidates),
    }


def _candidate_threshold_decision(query: TaskQuery, candidate: EventCandidate, threshold: dict) -> dict:
    visual_raw = _candidate_score_part(candidate, "visual_pos_raw", "patch", "visual")
    text_raw = _candidate_score_part(candidate, "text_pos_raw", "text", "text_context")
    neg_raw = _candidate_score_part(candidate, "text_neg_raw")
    final = float(candidate.final_score)
    has_marker = _candidate_has_lexical_marker_hit(candidate) or "transcript_marker_primary" in set(candidate.evidence_modalities or [])
    pass_visual = (not threshold.get("require_visual")) or visual_raw >= float(threshold.get("visual_floor", 0.0))
    pass_text = (not threshold.get("require_text")) or text_raw >= float(threshold.get("text_floor", 0.0)) or has_marker
    pass_final = final >= float(threshold.get("submit_min_score", 0.0))
    task_gate = _task_specific_threshold_gate(query, candidate)
    if not pass_visual:
        reason = "below_visual_floor"
    elif not pass_text:
        reason = "below_text_floor"
    elif not task_gate["pass"]:
        reason = str(task_gate["reason"])
    elif not pass_final:
        reason = "below_submit_threshold_after_negative_penalty"
    else:
        reason = "passed"
    return {
        "pass": bool(pass_visual and pass_text and task_gate["pass"] and pass_final),
        "reason": reason,
        "visual_pos_raw": visual_raw,
        "text_pos_raw": text_raw,
        "text_neg_raw": neg_raw,
        "final_score": final,
        "visual_floor": float(threshold.get("visual_floor", 0.0)),
        "text_floor": float(threshold.get("text_floor", 0.0)),
        "require_visual": bool(threshold.get("require_visual")),
        "require_text": bool(threshold.get("require_text")),
        "lexical_marker": bool(has_marker),
        "task_specific_gate": task_gate,
    }


def _task_specific_threshold_gate(query: TaskQuery, candidate: EventCandidate) -> dict:
    if query.task_id != "MM26-OIS09":
        return {"pass": True, "reason": "not_applicable"}
    trace = [
        item
        for item in ((candidate.provenance or {}).get("object_state_fit_trace") or [])
        if isinstance(item, dict) and item.get("probe_name") == "cooking_context_probe"
    ]
    if not trace:
        return {"pass": False, "reason": "missing_non_cooking_tool_semantic_support", "probe_status": "missing"}
    status = str(trace[0].get("status") or "unknown")
    if status == "pass":
        visual_raw = _candidate_score_part(candidate, "visual_pos_raw", "patch", "visual")
        fixed_room_min_visual = 0.080
        if candidate.source in FIXED_ROOM_SOURCES and visual_raw < fixed_room_min_visual:
            return {
                "pass": False,
                "reason": "weak_non_cooking_tool_visual_support",
                "probe_status": status,
                "matched_terms": ((trace[0].get("measured_features") or {}).get("non_cooking_tool_terms") or []),
                "visual_pos_raw": visual_raw,
                "fixed_room_min_visual": fixed_room_min_visual,
            }
        return {"pass": True, "reason": "passed", "probe_status": status, "matched_terms": ((trace[0].get("measured_features") or {}).get("non_cooking_tool_terms") or [])}
    if status == "fail":
        return {"pass": False, "reason": "object_state_cooking_context", "probe_status": status, "reject_reason": trace[0].get("reject_reason")}
    return {"pass": False, "reason": "missing_non_cooking_tool_semantic_support", "probe_status": status, "note": trace[0].get("note")}


def _build_tail_audit(
    query: TaskQuery,
    candidates: Sequence[EventCandidate],
    rows: Sequence[dict],
    threshold: dict,
    *,
    hard_cap_hit: bool,
) -> dict:
    submit = float(threshold["submit_min_score"])
    tail_min = float(threshold["tail_audit_min_score"])
    tail_candidates = [item for item in candidates if tail_min <= float(item.final_score) < submit]
    tail_clusters = []
    for index, candidate in enumerate(tail_candidates[:500], start=1):
        hour = int(float(candidate.start_second) // 3600)
        decision = "reject" if candidate.rejected_reason else "keep_diagnostic"
        if candidate.rejected_reason == "direct_frame_evidence_required":
            reason = "below_submit_threshold_text_only_for_visual_required_task"
        elif "visual_direct_frame" in candidate.evidence_modalities:
            reason = "below_submit_threshold_but_visual_patch_high"
        else:
            reason = "below_submit_threshold"
        tail_clusters.append(
            {
                "task_id": query.task_id,
                "cluster_id": f"tail::day{candidate.day}/{candidate.source}/{hour:02d}/{index:03d}",
                "day": candidate.day,
                "source": candidate.source,
                "start_second": candidate.start_second,
                "end_second": candidate.end_second,
                "max_score": candidate.final_score,
                "reason": reason,
                "decision": decision,
                "reject_reason": candidate.rejected_reason if decision == "reject" else None,
                "candidate_id": candidate.candidate_id,
            }
        )
    block_reasons = []
    if hard_cap_hit:
        block_reasons.append("output_hard_cap_hit")
    return {
        "task_id": query.task_id,
        "tail_audit_policy": "deterministic_cluster_audit_v1",
        "tail_cluster_count": len(tail_candidates),
        "tail_clusters": tail_clusters,
        "decision": "block_packaging" if block_reasons else "allow_packaging",
        "block_packaging": bool(block_reasons),
        "block_reasons": block_reasons,
    }


def _quantile(values: Sequence[float], q: float) -> float:
    rows = sorted(float(item) for item in values)
    if not rows:
        return 0.0
    q = max(0.0, min(1.0, float(q)))
    position = (len(rows) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return rows[lower]
    weight = position - lower
    return rows[lower] * (1.0 - weight) + rows[upper] * weight


def _median(values: Sequence[float]) -> float:
    return _quantile(values, 0.5)


def _rejected_candidate_row(candidate: EventCandidate) -> dict:
    return {
        "task_id": candidate.task_id,
        "candidate_id": candidate.candidate_id,
        "day": candidate.day,
        "source": candidate.source,
        "start_second": candidate.start_second,
        "end_second": candidate.end_second,
        "event_ids": candidate.event_ids,
        "frame_atom_ids": candidate.frame_atom_ids,
        "text_atom_ids": candidate.text_atom_ids,
        "evidence_modalities": candidate.evidence_modalities,
        "score_parts": candidate.score_parts,
        "final_score": candidate.final_score,
        "rejected_reason": candidate.rejected_reason,
        "provenance": candidate.provenance,
    }


def _candidate_row(candidate: EventCandidate, interval: Interval, *, event_store: EventStore | None = None) -> dict:
    row_hash = hashlib.sha256(candidate.candidate_id.encode("utf-8")).hexdigest()[:12]
    row_id = f"{candidate.task_id}:{candidate.day}:{candidate.source}:{int(interval.start)}-{int(interval.end)}:{row_hash}"
    provenance = _row_provenance(candidate, interval)
    frame_atom_ids = _row_frame_atom_ids(candidate, interval)
    text_atom_ids = filter_text_atom_ids_for_source(candidate.text_atom_ids, candidate.day, candidate.source, event_store)
    if not text_atom_ids and _candidate_requires_audio_marker_text(candidate):
        text_atom_ids = text_atom_ids_for_source_interval(candidate.day, candidate.source, interval.start, interval.end, event_store)
    return {
        "task_id": candidate.task_id,
        "task": candidate.task_id,
        "csv_row_id": row_id,
        "candidate_id": candidate.candidate_id,
        "day": candidate.day,
        "source": candidate.source,
        "start_second": interval.start,
        "end_second": interval.end,
        "starttime": format_submission_hms(interval.start),
        "endtime": format_submission_hms(interval.end),
        "event_ids": candidate.event_ids,
        "frame_atom_ids": frame_atom_ids,
        "text_atom_ids": text_atom_ids,
        "evidence_modalities": candidate.evidence_modalities,
        "score_parts": candidate.score_parts,
        "final_score": candidate.final_score,
        "split_reason": candidate.split_reason,
        "rejected_reason": candidate.rejected_reason,
        "provenance": provenance,
    }


def _candidate_requires_audio_marker_text(candidate: EventCandidate) -> bool:
    return candidate.task_id in {"MM26-EIS09", "MM26-EIS12"} and "transcript_marker_primary" in set(candidate.evidence_modalities or [])


def _row_requires_audio_marker_text(row: dict) -> bool:
    task = str(row.get("task_id") or row.get("task") or "")
    return task in {"MM26-EIS09", "MM26-EIS12"} and "transcript_marker_primary" in set(row.get("evidence_modalities") or [])


def _row_frame_atom_ids(candidate: EventCandidate, interval: Interval) -> list[str]:
    if "visual_direct_frame" not in set(candidate.evidence_modalities or []):
        return list(candidate.frame_atom_ids)
    provenance = candidate.provenance or {}
    refinement = provenance.get("ois_boundary_refinement") or {}
    support_ids = refinement.get("selected_support_frame_atom_ids") or refinement.get("support_frame_atom_ids") or []
    support_seconds = refinement.get("selected_support_frame_seconds") or refinement.get("support_frame_seconds") or []
    if support_ids and support_seconds and len(support_ids) == len(support_seconds):
        selected = []
        seen = set()
        for frame_id, second in zip(support_ids, support_seconds):
            frame_id = str(frame_id)
            if frame_id in seen:
                continue
            if float(interval.start) <= float(second) < float(interval.end):
                selected.append(frame_id)
                seen.add(frame_id)
        if selected:
            return selected
    direct_second = provenance.get("direct_frame_second")
    if direct_second is None:
        return list(candidate.frame_atom_ids)
    if float(interval.start) <= float(direct_second) < float(interval.end):
        return list(candidate.frame_atom_ids)
    return []


def _row_provenance(candidate: EventCandidate, interval: Interval) -> dict:
    provenance = dict(candidate.provenance)
    if "transcript_marker_primary" in candidate.evidence_modalities and provenance.get("local_visual_recovery_status"):
        interval_ref = _source_interval_ref(candidate.day, candidate.source, interval.start, interval.end)
        provenance["accepted_source_interval_ref"] = interval_ref
        audit = dict(provenance.get("local_visual_recovery_audit") or {})
        audit["accepted_source_interval_ref"] = interval_ref
        audit["row_interval_start_second"] = interval.start
        audit["row_interval_end_second"] = interval.end
        provenance["local_visual_recovery_audit"] = audit
    return provenance


def _local_visual_recovery_audit_records(query: TaskQuery, rows: Sequence[dict], rejected: Sequence[EventCandidate]) -> list[dict]:
    if not query.evidence_policy.get("require_local_visual_recovery_audit"):
        return []
    records = []
    for row in rows:
        provenance = row.get("provenance") or {}
        if "transcript_marker_primary" in set(row.get("evidence_modalities") or []):
            records.append(
                {
                    "task_id": query.task_id,
                    "candidate_id": row.get("candidate_id"),
                    "csv_row_id": row.get("csv_row_id"),
                    "action": "submitted",
                    "lexical_marker_sweep_id": provenance.get("lexical_marker_sweep_id"),
                    "accepted_source_interval_ref": provenance.get("accepted_source_interval_ref"),
                    "local_visual_recovery_status": provenance.get("local_visual_recovery_status"),
                    "audit": provenance.get("local_visual_recovery_audit") or {},
                }
            )
    for candidate in rejected:
        provenance = candidate.provenance or {}
        if "transcript_marker_primary" in set(candidate.evidence_modalities or []):
            records.append(
                {
                    "task_id": query.task_id,
                    "candidate_id": candidate.candidate_id,
                    "action": "rejected",
                    "rejected_reason": candidate.rejected_reason,
                    "lexical_marker_sweep_id": provenance.get("lexical_marker_sweep_id"),
                    "accepted_source_interval_ref": provenance.get("accepted_source_interval_ref"),
                    "local_visual_recovery_status": provenance.get("local_visual_recovery_status"),
                    "audit": provenance.get("local_visual_recovery_audit") or {},
                }
            )
    if not records:
        records.append(
            {
                "task_id": query.task_id,
                "action": "no_transcript_marker_candidates",
                "local_visual_recovery_status": "not_applicable_visual_direct_only",
                "audit": {
                    "audit_version": "local_visual_recovery_v1",
                    "reason": "no submitted or rejected transcript_marker_primary candidates",
                },
            }
        )
    return records


def _evidence_from_row(row: dict) -> EvidenceBundle:
    return EvidenceBundle(
        task_id=row["task_id"],
        csv_row_id=row["csv_row_id"],
        candidate_id=row["candidate_id"],
        day=int(row["day"]),
        source=row["source"],
        start_second=float(row["start_second"]),
        end_second=float(row["end_second"]),
        event_ids=list(row.get("event_ids") or []),
        frame_atom_ids=list(row.get("frame_atom_ids") or []),
        text_atom_ids=list(row.get("text_atom_ids") or []),
        evidence_modalities=list(row.get("evidence_modalities") or []),
        score_parts=dict(row.get("score_parts") or {}),
        final_score=float(row.get("final_score") or 0.0),
        split_reason=row.get("split_reason"),
        rejected_reason=row.get("rejected_reason"),
        provenance=dict(row.get("provenance") or {}),
    )


def _merge_adjacent_rows(rows: Sequence[dict], *, merge_gap_sec: float, query: TaskQuery | None = None) -> list[dict]:
    if not rows:
        return list(rows)
    ordered = sorted(rows, key=lambda row: (row.get("task_id", ""), int(row.get("day", 0)), row.get("source", ""), float(row.get("start_second", 0.0)), float(row.get("end_second", 0.0))))
    merged: list[dict] = []
    for row in ordered:
        current = dict(row)
        if not merged:
            merged.append(current)
            continue
        previous = merged[-1]
        decision = _row_merge_decision(previous, current, merge_gap_sec=merge_gap_sec, query=query)
        if decision.get("merge"):
            _merge_row_into_previous(previous, current, decision)
        else:
            merged.append(current)
    return merged


def _row_merge_decision(previous: dict, current: dict, *, merge_gap_sec: float, query: TaskQuery | None = None) -> dict:
    if not _same_row_group(previous, current):
        return {"merge": False, "reason": "different_task_day_source"}
    if previous.get("split_reason") == "coverage_clip" or current.get("split_reason") == "coverage_clip":
        return {"merge": False, "reason": "coverage_clip_split"}
    previous_start = float(previous["start_second"])
    previous_end = float(previous["end_second"])
    current_start = float(current["start_second"])
    current_end = float(current["end_second"])
    gap = current_start - previous_end
    overlap_sec = max(0.0, min(previous_end, current_end) - max(previous_start, current_start))
    task_type = _row_task_type(previous, query)
    if task_type == "event_instance_search":
        return _eis_row_merge_decision(previous, current, gap=gap, overlap_sec=overlap_sec)
    if task_type == "object_instance_search":
        return _ois_row_merge_decision(previous, current, gap=gap, overlap_sec=overlap_sec, merge_gap_sec=merge_gap_sec, query=query)
    return {"merge": False, "reason": "unsupported_interval_task_type"}


def _eis_row_merge_decision(previous: dict, current: dict, *, gap: float, overlap_sec: float) -> dict:
    previous_levels = _row_levels(previous)
    current_levels = _row_levels(current)
    if len(previous_levels) != 1 or len(current_levels) != 1:
        return {"merge": False, "reason": "eis_chain_merge_blocked"}
    level_pair = {previous_levels[0], current_levels[0]}
    if level_pair != {"L1", "L2"}:
        return {"merge": False, "reason": "eis_only_l1_l2_overlap"}
    if overlap_sec <= 0:
        return {"merge": False, "reason": "eis_no_adjacent_merge"}
    return {
        "merge": True,
        "merge_policy": "eis_l1_l2_overlap_merge_v1",
        "merge_type": "l1_l2_overlap",
        "gap_sec": gap,
        "overlap_sec": overlap_sec,
        "parent_l2_event_ids": _row_parent_l2_ids(previous) | _row_parent_l2_ids(current),
        "interval_resolution_policy": "overlap_event_union_after_eis_l2_suppression",
        "effective_merge_gap_sec": None,
        "configured_merge_gap_sec": None,
    }


def _ois_row_merge_decision(previous: dict, current: dict, *, gap: float, overlap_sec: float, merge_gap_sec: float, query: TaskQuery | None = None) -> dict:
    previous_levels = _row_levels(previous)
    current_levels = _row_levels(current)
    if not previous_levels or not current_levels:
        return {"merge": False, "reason": "missing_event_level"}
    if not set(previous_levels).issubset({"L1", "L2"}) or not set(current_levels).issubset({"L1", "L2"}):
        return {"merge": False, "reason": "ois_only_l1_l2"}
    same_parent_l2 = bool(_row_parent_l2_ids(previous) & _row_parent_l2_ids(current))
    effective_gap = _ois_effective_gap(merge_gap_sec, query=query, same_parent_l2=same_parent_l2)
    level_pair = _merge_level_pair(previous_levels, current_levels)
    if overlap_sec > 0:
        merge_type = "same_parent_l2_l1_group" if same_parent_l2 and level_pair == "l1_l1" else f"{level_pair}_overlap"
        return {
            "merge": True,
            "merge_policy": "ois_visual_object_track_merge_v1",
            "merge_type": merge_type,
            "gap_sec": gap,
            "overlap_sec": overlap_sec,
            "parent_l2_event_ids": _row_parent_l2_ids(previous) | _row_parent_l2_ids(current),
            "interval_resolution_policy": "union_for_ois_object_track",
            "effective_merge_gap_sec": effective_gap,
            "configured_merge_gap_sec": merge_gap_sec,
        }
    if gap < 0:
        return {"merge": False, "reason": "contained_non_overlap_ordering"}
    if "visual_direct_frame" not in set(previous.get("evidence_modalities") or []) or "visual_direct_frame" not in set(current.get("evidence_modalities") or []):
        return {"merge": False, "reason": "ois_adjacent_requires_visual_direct_frame"}
    if gap > effective_gap:
        return {"merge": False, "reason": "ois_adjacent_gap_exceeds_policy", "gap_sec": gap, "effective_merge_gap_sec": effective_gap}
    merge_type = "same_parent_l2_l1_group" if same_parent_l2 and level_pair == "l1_l1" else f"{level_pair}_adjacent"
    return {
        "merge": True,
        "merge_policy": "ois_visual_object_track_merge_v1",
        "merge_type": merge_type,
        "gap_sec": gap,
        "overlap_sec": overlap_sec,
        "parent_l2_event_ids": _row_parent_l2_ids(previous) | _row_parent_l2_ids(current),
        "interval_resolution_policy": "union_for_ois_object_track",
        "effective_merge_gap_sec": effective_gap,
        "configured_merge_gap_sec": merge_gap_sec,
    }


def _merge_row_into_previous(previous: dict, current: dict, decision: dict) -> None:
    previous_provenance = previous.setdefault("provenance", {})
    _ensure_merged_source_rows(previous)
    previous_provenance.setdefault("merged_source_rows", []).append(_compact_row_source(current))
    previous["start_second"] = min(float(previous["start_second"]), float(current["start_second"]))
    previous["end_second"] = max(float(previous["end_second"]), float(current["end_second"]))
    previous["starttime"] = format_submission_hms(previous["start_second"])
    previous["endtime"] = format_submission_hms(previous["end_second"])
    previous["final_score"] = max(float(previous.get("final_score", 0.0)), float(current.get("final_score", 0.0)))
    previous["text_atom_ids"] = sorted(set((previous.get("text_atom_ids") or []) + (current.get("text_atom_ids") or [])))
    previous["frame_atom_ids"] = sorted(set((previous.get("frame_atom_ids") or []) + (current.get("frame_atom_ids") or [])))
    previous["event_ids"] = sorted(set((previous.get("event_ids") or []) + (current.get("event_ids") or [])))
    previous["evidence_modalities"] = sorted(set((previous.get("evidence_modalities") or []) + (current.get("evidence_modalities") or [])))
    previous["split_reason"] = previous.get("split_reason") or current.get("split_reason")
    parent_l2_ids = sorted((_row_parent_l2_ids(previous) | _row_parent_l2_ids(current) | set(decision.get("parent_l2_event_ids") or [])))
    if parent_l2_ids:
        previous_provenance["parent_l2_event_ids"] = parent_l2_ids
    previous_provenance["merged_levels"] = sorted(set(_row_levels(previous) + _row_levels(current)))
    audit = {
        "merge_policy": decision.get("merge_policy"),
        "merge_type": decision.get("merge_type"),
        "gap_sec": decision.get("gap_sec"),
        "overlap_sec": decision.get("overlap_sec"),
        "selected_support_frame_seconds": _merged_support_frame_seconds(previous, current),
        "parent_l2_event_ids": parent_l2_ids,
        "merged_event_ids": list(previous["event_ids"]),
        "merged_frame_atom_ids": list(previous["frame_atom_ids"]),
        "interval_resolution_policy": decision.get("interval_resolution_policy"),
        "configured_merge_gap_sec": decision.get("configured_merge_gap_sec"),
        "effective_merge_gap_sec": decision.get("effective_merge_gap_sec"),
    }
    previous_provenance.setdefault("merge_audit", []).append(audit)
    if decision.get("merge_policy") == "eis_l1_l2_overlap_merge_v1":
        previous_provenance["eis_l1_l2_overlap_merge"] = audit
    elif decision.get("merge_policy") == "ois_visual_object_track_merge_v1":
        previous_provenance["ois_object_track_merge"] = audit
    previous["csv_row_id"] = _merged_row_id(previous)


def _same_row_group(previous: dict, current: dict) -> bool:
    return previous.get("task_id") == current.get("task_id") and int(previous.get("day", 0)) == int(current.get("day", 0)) and previous.get("source") == current.get("source")


def _row_task_type(row: dict, query: TaskQuery | None) -> str:
    if query is not None and query.task_id == str(row.get("task_id") or ""):
        return query.task_type
    task_id = str(row.get("task_id") or "")
    if task_id.startswith("MM26-EIS"):
        return "event_instance_search"
    if task_id.startswith("MM26-OIS"):
        return "object_instance_search"
    return ""


def _row_levels(row: dict) -> list[str]:
    provenance = row.get("provenance") or {}
    levels = [str(item) for item in provenance.get("merged_levels") or [] if str(item)]
    if not levels and provenance.get("level"):
        levels = [str(provenance["level"])]
    return sorted(dict.fromkeys(levels))


def _merge_level_pair(previous_levels: Sequence[str], current_levels: Sequence[str]) -> str:
    levels = set(previous_levels) | set(current_levels)
    if levels == {"L1"}:
        return "l1_l1"
    if levels == {"L2"}:
        return "l2_l2"
    return "l1_l2"


def _row_parent_l2_ids(row: dict) -> set[str]:
    provenance = row.get("provenance") or {}
    parent_ids = {str(item) for item in provenance.get("parent_l2_event_ids") or [] if str(item)}
    if provenance.get("primary_parent_l2_event_id"):
        parent_ids.add(str(provenance["primary_parent_l2_event_id"]))
    frame_hit = provenance.get("frame_hit") or {}
    pruning = frame_hit.get("post_patch_pruning") or {}
    parent_ids.update(str(item) for item in pruning.get("l2_event_ids") or [] if str(item))
    suppression = provenance.get("eis_l2_suppression") or {}
    if isinstance(suppression, dict) and suppression.get("parent_l2_event_id"):
        parent_ids.add(str(suppression["parent_l2_event_id"]))
    for audit in provenance.get("merge_audit") or []:
        parent_ids.update(str(item) for item in audit.get("parent_l2_event_ids") or [] if str(item))
    levels = _row_levels(row)
    if levels == ["L2"]:
        parent_ids.update(str(item) for item in row.get("event_ids") or [] if str(item))
    return parent_ids


def _ois_effective_gap(configured_merge_gap_sec: float, *, query: TaskQuery | None = None, same_parent_l2: bool = False) -> float:
    cap = 5.0 if same_parent_l2 or _is_static_fixed_ois_query(query) else 3.0
    if configured_merge_gap_sec <= 0:
        return 0.0
    return min(float(configured_merge_gap_sec), cap)


def _is_static_fixed_ois_query(query: TaskQuery | None) -> bool:
    if query is None or query.task_type != "object_instance_search":
        return False
    return query.scoring_profile in {"tiny_fixed_object", "location_object"} or query.output_profile == "ois_location_object"


def _ensure_merged_source_rows(row: dict) -> None:
    provenance = row.setdefault("provenance", {})
    if "merged_source_rows" not in provenance:
        provenance["merged_source_rows"] = [_compact_row_source(row)]


def _compact_row_source(row: dict) -> dict:
    provenance = row.get("provenance") or {}
    boundary = provenance.get("ois_boundary_refinement") or {}
    return {
        "candidate_id": row.get("candidate_id"),
        "event_ids": list(row.get("event_ids") or []),
        "frame_atom_ids": list(row.get("frame_atom_ids") or []),
        "text_atom_ids": list(row.get("text_atom_ids") or []),
        "level": provenance.get("level"),
        "parent_l2_event_ids": sorted(_row_parent_l2_ids(row)),
        "start_second": float(row.get("start_second", 0.0)),
        "end_second": float(row.get("end_second", 0.0)),
        "direct_frame_atom_id": provenance.get("direct_frame_atom_id"),
        "direct_frame_second": provenance.get("direct_frame_second"),
        "selected_support_frame_seconds": list(boundary.get("selected_support_frame_seconds") or []),
    }


def _merged_support_frame_seconds(previous: dict, current: dict) -> list[float]:
    seconds = []
    for row in (previous, current):
        provenance = row.get("provenance") or {}
        boundary = provenance.get("ois_boundary_refinement") or {}
        seconds.extend(float(item) for item in boundary.get("selected_support_frame_seconds") or boundary.get("support_frame_seconds") or [] if item is not None)
        if provenance.get("direct_frame_second") is not None:
            seconds.append(float(provenance["direct_frame_second"]))
    return sorted(set(seconds))


def _merged_row_id(row: dict) -> str:
    payload = f"{row.get('task_id')}|{row.get('day')}|{row.get('source')}|{row.get('start_second')}|{row.get('end_second')}|{','.join(row.get('event_ids') or [])}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{row.get('task_id')}:{row.get('day')}:{row.get('source')}:{int(float(row.get('start_second')))}-{int(float(row.get('end_second')))}:{digest}"
