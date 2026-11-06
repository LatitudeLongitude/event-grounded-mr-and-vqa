from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from .config import AppConfig
from .embedding_refs import decode_sparse_binary
from .filters import source_matches
from .ids import event_id, stable_hash
from .manifest import file_record, params_hash, write_stage_manifest
from .normalization import weighted_mean_available
from .parquet_io import iter_pylist_rows, read_pylist, write_table
from .paths import to_ref
from .run_state import stage_complete, stage_paths, update_run_state
from .schema import EVENT_TABLE, L3_EVENT_TABLE, LINK_TABLE
from .source_meta import l3_time_slack, source_roles
from .time_utils import seconds_to_hms
from .vector_utils import cosine, weighted_jaccard

STAGE = "group_l3"


def _time_iou(a: dict, b: dict) -> float:
    start = max(float(a["start_second"]), float(b["start_second"]))
    end = min(float(a["end_second"]), float(b["end_second"]))
    inter = max(0.0, end - start)
    union = max(float(a["end_second"]), float(b["end_second"])) - min(float(a["start_second"]), float(b["start_second"]))
    return inter / union if union > 0 else 0.0


def _gap(a: dict, b: dict) -> float:
    if float(a["end_second"]) < float(b["start_second"]):
        return float(b["start_second"]) - float(a["end_second"])
    if float(b["end_second"]) < float(a["start_second"]):
        return float(a["start_second"]) - float(b["end_second"])
    return 0.0


def _role_prior(role_a: Optional[str], role_b: Optional[str]) -> float:
    if role_a == role_b == "fixed_room":
        return 0.35
    if {role_a, role_b} == {"fixed_room", "ego_person"}:
        return 0.65
    if role_a == role_b == "ego_person":
        return 0.50
    return 0.20


def _source_pair(a: dict, b: dict) -> tuple[str, str]:
    return tuple(sorted((a["source"], b["source"])))


def _safe_cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    try:
        return max(0.0, min(1.0, (cosine(a, b) + 1.0) / 2.0))
    except ValueError:
        return None


def _load_frame_embeddings(config: AppConfig, run_root: Path, events: list[dict]) -> dict[str, np.ndarray]:
    frame_ids = {frame_id for event in events for frame_id in (event.get("representative_frame_ids") or [])}
    if not frame_ids:
        return {}
    frame_table = read_pylist(run_root / "atoms" / "frame_atoms.parquet", columns=["frame_atom_id", "tipsv2_embedding_id", "tipsv2_parquet_ref"], batch_size=config.run.batch_size)
    meta = {row["frame_atom_id"]: row for row in frame_table}
    by_path: dict[Path, set[str]] = defaultdict(set)
    id_to_frame = {}
    for frame_id in frame_ids:
        row = meta.get(frame_id)
        if row and row.get("tipsv2_embedding_id") and row.get("tipsv2_parquet_ref"):
            by_path[config.project_root / row["tipsv2_parquet_ref"]].add(row["tipsv2_embedding_id"])
            id_to_frame[row["tipsv2_embedding_id"]] = frame_id
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


def _load_text_embeddings(config: AppConfig, run_root: Path, events: list[dict]) -> tuple[dict[str, np.ndarray], dict[str, dict[int, float]]]:
    text_ids = {text_id for event in events for text_id in (event.get("linked_text_atom_ids") or [])}
    if not text_ids:
        return {}, {}
    text_rows = read_pylist(run_root / "atoms" / "text_atoms.parquet", batch_size=config.run.batch_size)
    meta = {row["text_atom_id"]: row for row in text_rows}
    dense_by_path: dict[Path, set[str]] = defaultdict(set)
    sparse_by_path: dict[Path, set[str]] = defaultdict(set)
    dense_id_to_text = {}
    sparse_id_to_text = {}
    for text_id in text_ids:
        row = meta.get(text_id)
        if not row:
            continue
        if row.get("qwen3_embedding_id") and row.get("qwen3_parquet_ref"):
            dense_by_path[config.project_root / row["qwen3_parquet_ref"]].add(row["qwen3_embedding_id"])
            dense_id_to_text[row["qwen3_embedding_id"]] = text_id
        if row.get("bgem3_embedding_id") and row.get("bgem3_parquet_ref"):
            sparse_by_path[config.project_root / row["bgem3_parquet_ref"]].add(row["bgem3_embedding_id"])
            sparse_id_to_text[row["bgem3_embedding_id"]] = text_id
    dense = {}
    for path, needed in dense_by_path.items():
        remaining = set(needed)
        for row in iter_pylist_rows(path, columns=["embedding_id", "embedding"], batch_size=config.run.batch_size):
            embedding_id = row["embedding_id"]
            if embedding_id in remaining:
                dense[dense_id_to_text[embedding_id]] = np.asarray(row["embedding"], dtype=np.float32)
                remaining.remove(embedding_id)
                if not remaining:
                    break
    sparse = {}
    for path, needed in sparse_by_path.items():
        remaining = set(needed)
        for row in iter_pylist_rows(path, columns=["embedding_id", "bge_sparse_indices_i32", "bge_sparse_values_f16"], batch_size=config.run.batch_size):
            embedding_id = row["embedding_id"]
            if embedding_id in remaining:
                sparse[sparse_id_to_text[embedding_id]] = decode_sparse_binary(row["bge_sparse_indices_i32"], row["bge_sparse_values_f16"])
                remaining.remove(embedding_id)
                if not remaining:
                    break
    return dense, sparse


def _mean_vector(vectors: list[np.ndarray]) -> Optional[np.ndarray]:
    if not vectors:
        return None
    arr = np.mean(np.stack(vectors), axis=0)
    norm = float(np.linalg.norm(arr))
    return None if norm <= 0 else arr / norm


def _event_features(config: AppConfig, run_root: Path, events: list[dict]) -> dict[str, dict]:
    frame_embeddings = _load_frame_embeddings(config, run_root, events)
    dense_embeddings, sparse_embeddings = _load_text_embeddings(config, run_root, events)
    output_features = _load_event_features_from_outputs(run_root, {event["event_id"] for event in events})
    visual_score_rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    visual_scores_path = run_root / "features" / "visual_motion_features.parquet"
    if visual_scores_path.exists():
        for row in iter_pylist_rows(visual_scores_path, columns=["day", "source", "second", "activity_score", "visual_boundary_score"], batch_size=config.run.batch_size):
            visual_score_rows[(row["day"], row["source"])].append(row)
        for rows in visual_score_rows.values():
            rows.sort(key=lambda item: float(item["second"]))
    features = {}
    for event in events:
        precomputed = output_features.get(event["event_id"], {})
        visual = precomputed.get("visual")
        if visual is None:
            visual = _mean_vector([frame_embeddings[fid] for fid in event.get("representative_frame_ids") or [] if fid in frame_embeddings])
        text = precomputed.get("text")
        if text is None:
            text = _mean_vector([dense_embeddings[tid] for tid in event.get("linked_text_atom_ids") or [] if tid in dense_embeddings])
        sparse: dict[int, float] = defaultdict(float)
        if precomputed.get("sparse"):
            sparse.update(precomputed["sparse"])
        else:
            for text_id in event.get("linked_text_atom_ids") or []:
                for key, value in sparse_embeddings.get(text_id, {}).items():
                    sparse[key] += float(value)
        motion_rows = [
            row
            for row in visual_score_rows.get((event["day"], event["source"]), [])
            if float(event["start_second"]) <= float(row["second"]) < float(event["end_second"])
        ]
        finite_activity = [float(row["activity_score"]) for row in motion_rows if row.get("activity_score") is not None and np.isfinite(float(row["activity_score"]))]
        finite_boundary = [row for row in motion_rows if row.get("visual_boundary_score") is not None and np.isfinite(float(row["visual_boundary_score"]))]
        peak_second = None
        if finite_boundary:
            peak_second = float(max(finite_boundary, key=lambda row: float(row["visual_boundary_score"]))["second"])
        features[event["event_id"]] = {
            "visual": visual,
            "text": text,
            "sparse": dict(sparse),
            "activity_mean": float(np.mean(finite_activity)) if finite_activity else None,
            "peak_second": peak_second,
        }
    return features


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


def _calibrate(values: list[Optional[float]]) -> list[Optional[float]]:
    finite = sorted(float(value) for value in values if value is not None and np.isfinite(float(value)))
    if len(finite) <= 1:
        return [None if value is None else float(value) for value in values]
    result = []
    denom = max(len(finite) - 1, 1)
    for value in values:
        if value is None or not np.isfinite(float(value)):
            result.append(None)
            continue
        rank = np.searchsorted(finite, float(value), side="right") - 1
        result.append(float(rank / denom))
    return result


def _raw_candidate(a: dict, b: dict, features: dict[str, dict], slack: float, location_by_source: dict[str, Optional[str]]) -> dict:
    fa = features.get(a["event_id"], {})
    fb = features.get(b["event_id"], {})
    centroid_similarity = _safe_cosine(fa.get("visual"), fb.get("visual"))
    activity_similarity = None
    if fa.get("activity_mean") is not None and fb.get("activity_mean") is not None:
        activity_similarity = max(0.0, min(1.0, 1.0 - abs(float(fa["activity_mean"]) - float(fb["activity_mean"])) / 6.0))
    peak_alignment = None
    if fa.get("peak_second") is not None and fb.get("peak_second") is not None:
        peak_alignment = max(0.0, 1.0 - abs(float(fa["peak_second"]) - float(fb["peak_second"])) / max(slack, 1.0))
    visual = weighted_mean_available(
        [
            (0.60, centroid_similarity),
            (0.20, activity_similarity),
            (0.20, peak_alignment),
        ]
    )
    text = _safe_cosine(fa.get("text"), fb.get("text"))
    sparse = weighted_jaccard(fa.get("sparse") or {}, fb.get("sparse") or {}) if (fa.get("sparse") or fb.get("sparse")) else None
    gap = _gap(a, b)
    location_a = location_by_source.get(a["source"])
    location_b = location_by_source.get(b["source"])
    roles = {a.get("source_role"), b.get("source_role")}
    return {
        "left": a,
        "right": b,
        "day": a["day"],
        "time_iou": _time_iou(a, b),
        "visual_similarity": visual,
        "visual_centroid_similarity": centroid_similarity,
        "visual_activity_similarity": activity_similarity,
        "visual_peak_alignment": peak_alignment,
        "text_similarity": text,
        "sparse_context_similarity": sparse,
        "source_role_prior": _role_prior(a.get("source_role"), b.get("source_role")),
        "temporal_alignment": max(0.0, 1.0 - gap / max(slack, 1.0)),
        "source_pair": _source_pair(a, b),
        "fixed_fixed_location_compatible": not (
            a.get("source_role") == b.get("source_role") == "fixed_room"
            and location_a
            and location_b
            and location_a != location_b
        ),
        "is_fixed_ego_pair": roles == {"fixed_room", "ego_person"},
        "fixed_ego_best_margin": None,
    }


def _score_candidate(candidate: dict) -> float:
    return float(
        weighted_mean_available(
            [
                (0.25, candidate["time_iou"]),
                (0.35, candidate["visual_similarity"]),
                (0.15, candidate["text_similarity"]),
                (0.10, candidate["sparse_context_similarity"]),
                (0.10, candidate["source_role_prior"]),
                (0.05, candidate["temporal_alignment"]),
            ]
        )
        or 0.0
    )


def _base_event_fields(day: str, day_num: int, start: float, end: float, sources: list[str], roles: list[str], children: list[dict], links: list[str], param_hash: str, input_manifest_ref: str) -> dict:
    representatives = list(dict.fromkeys(frame for child in children for frame in (child.get("representative_frame_ids") or [])))[:12]
    text_ids = list(dict.fromkeys(text for child in children for text in (child.get("linked_text_atom_ids") or [])))
    eid = event_id("L3", day, None, start, end, [child["event_id"] for child in children])
    return {
        "event_id": eid,
        "level": "L3",
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
        "event_kind": "activity_candidate",
        "primary_parent_id": None,
        "global_event_id": eid,
        "representative_frame_ids": representatives,
        "linked_text_atom_ids": text_ids,
        "visual_centroid_ref": None,
        "text_dense_centroid_ref": None,
        "text_sparse_aggregate_ref": None,
        "modality_support": sorted(set(mod for child in children for mod in (child.get("modality_support") or []))),
        "confidence": sum(float(child.get("confidence") or 0.0) for child in children) / max(len(children), 1),
        "confidence_level": "medium",
        "quality_flags": [],
        "reserved_label_primary": None,
        "reserved_label_secondary": None,
        "reserved_label_freeform": None,
        "reserved_label_confidence": None,
        "created_by": STAGE,
        "params_hash": param_hash,
        "input_manifest_ref": input_manifest_ref,
        "primary_sources": sources[:2],
        "supporting_sources": sources[2:],
        "child_source_event_ids": [child["event_id"] for child in children],
        "source_count": len(sources),
        "cross_source_link_ids": links,
    }


def _candidate_map(candidates: list[dict]) -> dict[frozenset[str], dict]:
    return {frozenset((item["left"]["event_id"], item["right"]["event_id"])): item for item in candidates}


def _internal_cluster_link_ids(links: list[dict], cluster: list[dict]) -> list[str]:
    cluster_ids = {item["event_id"] for item in cluster}
    return sorted(
        {
            link["link_id"]
            for link in links
            if link["event_id"] in cluster_ids and link["target_id"] in cluster_ids
        }
    )


def _complete_link_clusters(events: list[dict], candidates: list[dict], max_span: float) -> list[list[dict]]:
    by_pair = _candidate_map(candidates)
    strong = [item for item in candidates if _is_strong_candidate(item)]
    clusters: list[list[dict]] = []
    event_by_id = {event["event_id"]: event for event in events}
    for edge in sorted(strong, key=lambda item: item["link_score"], reverse=True):
        pair_events = [edge["left"], edge["right"]]
        placed = False
        for cluster in clusters:
            proposal = list({item["event_id"]: item for item in cluster + pair_events}.values())
            if len({item["source"] for item in proposal}) != len(proposal):
                continue
            start = min(float(item["start_second"]) for item in proposal)
            end = max(float(item["end_second"]) for item in proposal)
            if end - start > max_span:
                continue
            scores = []
            ok = True
            for i, left in enumerate(proposal):
                for right in proposal[i + 1 :]:
                    item = by_pair.get(frozenset((left["event_id"], right["event_id"])))
                    if item is None or item["link_score"] < 0.60:
                        ok = False
                        break
                    scores.append(item["link_score"])
                if not ok:
                    break
            if ok and scores and sum(scores) / len(scores) >= 0.70:
                cluster[:] = proposal
                placed = True
                break
        if not placed:
            start = min(float(item["start_second"]) for item in pair_events)
            end = max(float(item["end_second"]) for item in pair_events)
            if end - start <= max_span:
                clusters.append([event_by_id[edge["left"]["event_id"]], event_by_id[edge["right"]["event_id"]]])
    clusters.sort(key=lambda cluster: (min(float(item["start_second"]) for item in cluster), len(cluster)))
    used = set()
    result = []
    for cluster in clusters:
        if any(item["event_id"] in used for item in cluster):
            continue
        used.update(item["event_id"] for item in cluster)
        result.append(cluster)
    return result


def _fixed_ego_anchor(candidate: dict) -> Optional[str]:
    if not candidate.get("is_fixed_ego_pair"):
        return None
    left = candidate["left"]
    right = candidate["right"]
    if left.get("source_role") == "ego_person":
        return left["event_id"]
    if right.get("source_role") == "ego_person":
        return right["event_id"]
    return None


def _assign_fixed_ego_margins(candidates: list[dict]) -> None:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        anchor = _fixed_ego_anchor(candidate)
        if anchor:
            grouped[anchor].append(candidate)
    for group in grouped.values():
        ranked = sorted(group, key=lambda item: (float(item.get("non_prior_evidence") or 0.0), float(item.get("link_score") or 0.0)), reverse=True)
        best = ranked[0] if ranked else None
        second_score = float(ranked[1].get("non_prior_evidence") or 0.0) if len(ranked) > 1 else None
        for candidate in group:
            if candidate is best:
                if second_score is None:
                    candidate["fixed_ego_best_margin"] = 1.0
                else:
                    candidate["fixed_ego_best_margin"] = float(candidate.get("non_prior_evidence") or 0.0) - second_score
            else:
                candidate["fixed_ego_best_margin"] = 0.0


def _is_strong_candidate(candidate: dict) -> bool:
    if candidate.get("link_score", 0.0) < 0.70:
        return False
    if candidate.get("visual_similarity") is None or float(candidate.get("visual_similarity") or 0.0) < 0.35:
        return False
    if candidate.get("non_prior_evidence", 0.0) < 0.35:
        return False
    if not candidate.get("fixed_fixed_location_compatible", True):
        return False
    if candidate.get("is_fixed_ego_pair") and float(candidate.get("fixed_ego_best_margin") or 0.0) < 0.08:
        return False
    return True


def build_l3_groups(config: AppConfig, run_root: Path, *, output_mode: str, run_id: Optional[str], day: Optional[str] = None, source: Optional[str] = None, resume: bool = False) -> dict:
    paths = stage_paths(run_root)
    l2_path = run_root / "segments" / "l2_source_events.parquet"
    l3_path = run_root / "segments" / "l3_global_events.parquet"
    links_path = run_root / "links" / "cross_source_links.parquet"
    params = {"stage": STAGE, "output_mode": output_mode, "run_id": run_id, "day": day, "source": source}
    if resume and stage_complete(paths.stage_manifest(STAGE), params=params, output_paths=[l2_path, l3_path, links_path]):
        return {"stage": STAGE, "skipped": True}

    l2_rows = read_pylist(l2_path, batch_size=config.run.batch_size)
    role_map = source_roles(config.input_roots.view_meta_path)
    location_by_source = {source_name: meta.get("location_hint") for source_name, meta in role_map.items()}
    for row in l2_rows:
        row["source_role"] = row.get("source_role") or (role_map.get(row["source"]) or {}).get("source_role")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in l2_rows:
        if day and row["day"] != ("day" + str(day).replace("day", "")):
            continue
        if not source_matches(row["source"], source):
            continue
        grouped[row["day"]].append(row)
    slack = float(l3_time_slack(config.input_roots.view_meta_path))
    param_hash = params_hash(params)
    input_manifest_ref = to_ref(paths.stage_manifest("segments_l1_l2"), config.project_root)
    if l2_rows:
        from .event_embeddings import build_event_embeddings

        build_event_embeddings(config, run_root, output_mode=output_mode, run_id=run_id, day=day, source=source, resume=True)
    features = _event_features(config, run_root, l2_rows)

    all_candidates = []
    for rows in grouped.values():
        rows.sort(key=lambda row: float(row["start_second"]))
        active: list[dict] = []
        for row in rows:
            active = [item for item in active if float(item["end_second"]) + slack >= float(row["start_second"])]
            for other in active:
                if other["source"] == row["source"] or _gap(row, other) > slack:
                    continue
                all_candidates.append(_raw_candidate(row, other, features, slack, location_by_source))
            active.append(row)

    for pair in sorted({(candidate["day"], candidate["source_pair"]) for candidate in all_candidates}):
        group = [candidate for candidate in all_candidates if (candidate["day"], candidate["source_pair"]) == pair]
        for key in ("visual_similarity", "text_similarity", "sparse_context_similarity"):
            calibrated = _calibrate([candidate[key] for candidate in group])
            for candidate, value in zip(group, calibrated):
                candidate[key] = value
    for candidate in all_candidates:
        candidate["non_prior_evidence"] = max(
            candidate["visual_similarity"] or 0.0,
            candidate["text_similarity"] or 0.0,
            candidate["sparse_context_similarity"] or 0.0,
        )
        candidate["link_score"] = _score_candidate(candidate)
    _assign_fixed_ego_margins(all_candidates)

    links = []
    for candidate in all_candidates:
        score = candidate["link_score"]
        role = "strong" if _is_strong_candidate(candidate) else ("weak" if score >= 0.45 else None)
        if role is None:
            continue
        flags = []
        if candidate["non_prior_evidence"] < 0.35:
            flags.append("insufficient_non_prior_evidence")
        if candidate.get("visual_similarity") is None or float(candidate.get("visual_similarity") or 0.0) < 0.35:
            flags.append("insufficient_visual_evidence")
        if not candidate.get("fixed_fixed_location_compatible", True):
            flags.append("fixed_room_location_incompatible")
        if candidate.get("is_fixed_ego_pair") and float(candidate.get("fixed_ego_best_margin") or 0.0) < 0.08:
            flags.append("fixed_ego_best_margin_below_0_08")
        link_id = "cross::" + stable_hash([candidate["left"]["event_id"], candidate["right"]["event_id"]], 20)
        links.append(
            {
                "link_id": link_id,
                "event_id": candidate["left"]["event_id"],
                "target_id": candidate["right"]["event_id"],
                "frame_atom_id": None,
                "text_atom_id": None,
                "embedding_ref": None,
                "parent_event_id": None,
                "child_event_id": None,
                "link_type": "cross_source_member" if role == "strong" else "weak_related",
                "link_role": role,
                "score": score,
                "selection_score": score,
                "selection_reason": "sweep_line_multimodal_complete_link_candidate",
                "selection_rank": 0,
                "derived_from_event_id": None,
                "derived_from_frame_atom_id": None,
                "is_representative": False,
                "quality_flags": flags,
                "created_by": STAGE,
                "params_hash": param_hash,
                "input_manifest_ref": input_manifest_ref,
            }
        )

    l3_rows = []
    for day_value, rows in grouped.items():
        clusters = _complete_link_clusters(rows, [candidate for candidate in all_candidates if candidate["left"]["day"] == day_value], config.segmentation.l3_max_sec)
        for cluster in clusters:
            start = min(float(item["start_second"]) for item in cluster)
            end = max(float(item["end_second"]) for item in cluster)
            sources = sorted({item["source"] for item in cluster})
            roles = sorted({item.get("source_role") for item in cluster if item.get("source_role")})
            link_ids = _internal_cluster_link_ids(links, cluster)
            l3_rows.append(_base_event_fields(cluster[0]["day"], cluster[0]["day_num"], start, end, sources, roles, cluster, link_ids, param_hash, input_manifest_ref))

    parent_by_child = {child_id: row["event_id"] for row in l3_rows for child_id in row["child_source_event_ids"]}
    for row in l2_rows:
        if row["event_id"] in parent_by_child:
            row["primary_parent_id"] = parent_by_child[row["event_id"]]
            row["global_event_id"] = parent_by_child[row["event_id"]]

    write_table(l2_path, l2_rows, schema=EVENT_TABLE)
    write_table(l3_path, l3_rows, schema=L3_EVENT_TABLE)
    write_table(links_path, links, schema=LINK_TABLE)
    manifest = write_stage_manifest(
        paths.stage_manifest(STAGE),
        stage=STAGE,
        params=params,
        inputs=[
            file_record(l2_path, config.project_root),
            file_record(run_root / "atoms" / "frame_atoms.parquet", config.project_root),
            file_record(run_root / "atoms" / "text_atoms.parquet", config.project_root),
            file_record(run_root / "embeddings" / "event_visual_centroids.parquet", config.project_root),
            file_record(run_root / "embeddings" / "event_text_dense_centroids.parquet", config.project_root),
            file_record(run_root / "embeddings" / "event_text_sparse_aggregates.parquet", config.project_root),
        ],
        outputs=[
            {**file_record(l2_path, config.project_root), "schema": str(EVENT_TABLE)},
            {**file_record(l3_path, config.project_root), "schema": str(L3_EVENT_TABLE)},
            {**file_record(links_path, config.project_root), "schema": str(LINK_TABLE)},
        ],
        schema_names=["l2_source_events", "l3_global_events", "cross_source_links"],
        project_root=config.project_root,
    )
    update_run_state(run_root, STAGE, to_ref(paths.stage_manifest(STAGE), config.project_root))
    return {"stage": STAGE, "manifest": manifest}
