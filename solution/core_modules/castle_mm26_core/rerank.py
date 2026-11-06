from __future__ import annotations

import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .config import AppConfig
from .data import EventStore, resolve_frame_path
from .io_utils import atomic_write_json, atomic_write_jsonl, sha256_file
from .models import FrameHit, TaskQuery, TextHit
from .runtimes import Qwen3RerankerRuntime, TipsV2Runtime, release_runtime_state


def bypass_visual_patch_rerank_with_cls(
    query: TaskQuery,
    hits: Sequence[FrameHit],
    event_store: EventStore,
    output_dir: Path,
    *,
    max_frames: int | None = None,
) -> tuple[list[FrameHit], dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = list(hits[:max_frames] if max_frames is not None else hits)
    original_input_count = len(selected)
    selected, ois_rescan_manifest = _expand_ois_l2_support_hits(query, selected, event_store, enabled=max_frames is None)
    candidate_count_before_dedup = len(selected)
    selected, dedup_manifest = _deduplicate_patch_candidates(selected, event_store)
    for hit in selected:
        hit.patch_score = None
        hit.patch_top5_mean = None
        hit.provenance.setdefault("patch_rerank_bypass", {"policy": "task_policy_cls_visual_source_v1", "score_source": "cls_score"})
    selected.sort(key=lambda item: (-float(item.cls_score or 0.0), item.embedding_id))
    for index, hit in enumerate(selected, start=1):
        hit.rank = index
    rows = [
        {
            "task_id": hit.task_id,
            "embedding_id": hit.embedding_id,
            "candidate_record_id": hit.provenance.get("candidate_record_id"),
            "frame_atom_id": hit.frame_atom_id,
            "cls_score": hit.cls_score,
            "patch_score": None,
            "patch_top5_mean": None,
            "day": hit.day,
            "source": hit.source,
            "absolute_second": hit.absolute_second,
            "image_path": hit.frame_rel_path,
            "score_source": "cls_score",
            "bypass_policy": "task_policy_cls_visual_source_v1",
        }
        for hit in selected
    ]
    audit_rows = []
    linked = {id(hit): _linked_frame_events(hit, event_store) for hit in selected}
    for hit in selected:
        events = linked[id(hit)]
        audit = {
            "task_id": query.task_id,
            "embedding_id": hit.embedding_id,
            "candidate_record_id": hit.provenance.get("candidate_record_id"),
            "frame_atom_id": hit.frame_atom_id,
            "action": "kept_for_cls_event_scoring",
            "reason": "task_policy_cls_visual_source_bypasses_patch_rerank",
            "l2_event_ids": [event_id for event_id, level in events if level == "L2"],
            "protected_l1_event_ids": [event_id for event_id, level in events if level == "L1"],
            "patch_score": None,
            "cls_score": hit.cls_score,
            "rank": hit.rank,
        }
        hit.provenance["post_patch_pruning"] = audit
        audit_rows.append(audit)
    atomic_write_jsonl(output_dir / "patch_scores.jsonl", rows)
    atomic_write_jsonl(output_dir / "post_patch_pruning.jsonl", audit_rows)
    manifest = {
        "status": "completed",
        "task_id": query.task_id,
        "input_count": len(selected),
        "initial_input_count": original_input_count,
        "candidate_count_before_dedup": candidate_count_before_dedup,
        "candidate_count_after_dedup": dedup_manifest["candidate_count_after_dedup"],
        "deduped_candidate_count": dedup_manifest["deduped_candidate_count"],
        "dedup_key_policy": dedup_manifest["dedup_key_policy"],
        "merged_candidate_record_count": dedup_manifest["merged_candidate_record_count"],
        "max_frames": max_frames,
        "full_candidate_patch_rerank": max_frames is None,
        "ois_l2_rescan": ois_rescan_manifest,
        "scored_count": len(rows),
        "post_patch_kept_count": len(selected),
        "failure_count": 0,
        "failures": [],
        "mode": "task_policy_cls_visual_source_bypass",
        "score_source": "cls_score",
        "patch_runtime_version": "task_policy_cls_visual_source_v1",
        "unique_image_encode_count": 0,
        "tips_parallel_enabled": False,
        "tips_parallel_fallback_reason": None,
        "per_worker_encoded_image_count": {},
    }
    atomic_write_json(output_dir / "stage_manifest.json", manifest)
    return selected, manifest


def rerank_visual_patches(
    config: AppConfig,
    query: TaskQuery,
    hits: Sequence[FrameHit],
    runtime: TipsV2Runtime,
    event_store: EventStore,
    output_dir: Path,
    *,
    batch_size: int = 8,
    max_frames: int | None = None,
    cache_path: Path | None = None,
    patch_runtimes: Sequence[TipsV2Runtime] | None = None,
) -> tuple[list[FrameHit], dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = list(hits[:max_frames] if max_frames is not None else hits)
    original_input_count = len(selected)
    selected, ois_rescan_manifest = _expand_ois_l2_support_hits(query, selected, event_store, enabled=max_frames is None)
    candidate_count_before_dedup = len(selected)
    selected, dedup_manifest = _deduplicate_patch_candidates(selected, event_store)
    prompt_set = query.prompt_set("visual")
    prompt_set_hash = hashlib.sha256(json.dumps(prompt_set, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    model_fingerprint = runtime.fingerprint()
    model_fingerprint_hash = str(model_fingerprint.get("fingerprint_sha256") or hashlib.sha256(json.dumps(model_fingerprint, sort_keys=True).encode("utf-8")).hexdigest())
    patch_runtime_version = "tipsv2_native_patch_rerank_v1"
    text_embeddings = runtime.encode_text(prompt_set)
    rows = []
    failures = []
    patch_cache_path = cache_path or (output_dir / "patch_score_cache.jsonl")
    score_cache: dict[str, dict] = _load_patch_score_cache(patch_cache_path)
    disk_cache_entry_count = len(score_cache)
    cache_hits = 0
    encoded_image_count = 0
    workers = _patch_worker_list(runtime, patch_runtimes)
    disabled_worker_devices: set[str] = set()
    per_worker_encoded_image_count = {device: 0 for device in _worker_devices(workers)}
    tips_parallel_fallback_reasons: list[str] = []
    for start in range(0, len(selected), batch_size):
        batch = selected[start : start + batch_size]
        assignments = []
        pending_paths = []
        pending_keys = []
        pending_seen = set()
        pending_sha_by_key = {}
        for hit in batch:
            frame_atom = event_store.frame_by_embedding.get(hit.embedding_id)
            if frame_atom:
                hit.frame_atom_id = frame_atom["frame_atom_id"]
                image_ref = str(frame_atom.get("image_path_ref") or frame_atom.get("frame_rel_path") or hit.frame_rel_path)
            else:
                image_ref = hit.image_path_ref or hit.frame_rel_path
            image_path = resolve_frame_path(config, hit.frame_rel_path, image_ref)
            if not image_path.exists():
                failures.append({"embedding_id": hit.embedding_id, "reason": "image_missing", "image_path": image_path.as_posix()})
                continue
            frame_sha256 = sha256_file(image_path)
            frame_atom_key = hit.frame_atom_id or hit.embedding_id
            cache_key = _patch_cache_key(
                model_fingerprint_hash=model_fingerprint_hash,
                task_id=query.task_id,
                frame_atom_id=frame_atom_key,
                frame_file_sha256=frame_sha256,
                prompt_set_hash=prompt_set_hash,
                patch_runtime_version=patch_runtime_version,
            )
            assignments.append((hit, image_path, cache_key, frame_sha256))
            if cache_key in score_cache:
                cache_hits += 1
            elif cache_key not in pending_seen:
                pending_seen.add(cache_key)
                pending_keys.append(cache_key)
                pending_paths.append(image_path)
                pending_sha_by_key[cache_key] = frame_sha256
        if pending_paths:
            encoded, worker_report = _encode_image_with_patch_workers(workers, pending_paths, disabled_worker_devices)
            for device, count in worker_report["per_worker_encoded_image_count"].items():
                per_worker_encoded_image_count[device] = per_worker_encoded_image_count.get(device, 0) + int(count)
            tips_parallel_fallback_reasons.extend(worker_report["fallback_reasons"])
            patches = encoded["patch_tokens"]
            scores = np.max(patches @ text_embeddings.T, axis=2)
            encoded_image_count += len(pending_paths)
            for cache_key, image_path, patch_scores in zip(pending_keys, pending_paths, scores):
                sorted_scores = np.sort(patch_scores)[::-1]
                score_cache[cache_key] = {
                    "patch_score": float(sorted_scores[0]),
                    "patch_top5_mean": float(np.mean(sorted_scores[: min(5, len(sorted_scores))])),
                    "frame_file_sha256": pending_sha_by_key[cache_key],
                    "patch_token_count": int(patches.shape[1]),
                    "patch_dim": int(patches.shape[2]),
                    "model_fingerprint_sha256": model_fingerprint_hash,
                    "prompt_set_hash": prompt_set_hash,
                    "patch_runtime_version": patch_runtime_version,
                }
        if not assignments:
            continue
        for hit, image_path, cache_key, _frame_sha256 in assignments:
            cached = score_cache[cache_key]
            hit.patch_score = float(cached["patch_score"])
            hit.patch_top5_mean = float(cached["patch_top5_mean"])
            hit.provenance["patch"] = {
                "image_path": image_path.as_posix(),
                "frame_file_sha256": cached["frame_file_sha256"],
                "patch_token_count": int(cached["patch_token_count"]),
                "patch_dim": int(cached["patch_dim"]),
                "prompt": query.canonical_text_for_retrieval,
                "prompt_set": prompt_set,
                "prompt_set_hash": prompt_set_hash,
                "model_fingerprint_sha256": model_fingerprint_hash,
                "patch_runtime_version": patch_runtime_version,
                "cache_key": cache_key,
                "cache_duplicate_frame_encodes_only": True,
            }
            rows.append(
                {
                    "task_id": hit.task_id,
                    "embedding_id": hit.embedding_id,
                    "candidate_record_id": hit.provenance.get("candidate_record_id"),
                    "frame_atom_id": hit.frame_atom_id,
                    "cls_score": hit.cls_score,
                    "patch_score": hit.patch_score,
                    "patch_top5_mean": hit.patch_top5_mean,
                    "day": hit.day,
                    "source": hit.source,
                    "absolute_second": hit.absolute_second,
                    "image_path": image_path.as_posix(),
                }
            )
    atomic_write_jsonl(patch_cache_path, [{"cache_key": key, **value} for key, value in sorted(score_cache.items())])
    selected.sort(key=lambda item: (-(item.patch_score if item.patch_score is not None else -1e9), -item.cls_score, item.embedding_id))
    for index, hit in enumerate(selected, start=1):
        hit.rank = index
    pre_pruning_count = len(selected)
    selected, pruning_manifest = _write_post_patch_pruning_audit(query, selected, event_store, output_dir)
    atomic_write_jsonl(output_dir / "patch_scores.jsonl", rows)
    manifest = {
        "status": "completed" if not failures else "failed",
        "task_id": query.task_id,
        "input_count": pre_pruning_count,
        "initial_input_count": original_input_count,
        "candidate_count_before_dedup": candidate_count_before_dedup,
        "candidate_count_after_dedup": dedup_manifest["candidate_count_after_dedup"],
        "deduped_candidate_count": dedup_manifest["deduped_candidate_count"],
        "dedup_key_policy": dedup_manifest["dedup_key_policy"],
        "merged_candidate_record_count": dedup_manifest["merged_candidate_record_count"],
        "max_frames": max_frames,
        "full_candidate_patch_rerank": max_frames is None,
        "ois_l2_rescan": ois_rescan_manifest,
        "scored_count": len(rows),
        "post_patch_kept_count": len(selected),
        "failure_count": len(failures),
        "failures": failures[:20],
        "mode": "native_patch_tokens",
        "candidate_record_retention": "keep_all_records",
        "cache_duplicate_frame_encodes_only": True,
        "unique_image_encode_count": encoded_image_count,
        "duplicate_cache_hit_count": cache_hits,
        "disk_cache_entry_count_before_run": disk_cache_entry_count,
        "disk_cache_entry_count_after_run": len(score_cache),
        "patch_score_cache_path": patch_cache_path.as_posix(),
        "post_patch_pruning": pruning_manifest,
        "tips_worker_count": len([worker for worker in workers if _worker_device(worker) not in disabled_worker_devices]),
        "tips_worker_devices": [device for device in _worker_devices(workers) if device not in disabled_worker_devices],
        "tips_parallel_enabled": len([worker for worker in workers if _worker_device(worker) not in disabled_worker_devices]) > 1,
        "tips_parallel_fallback_reason": "; ".join(dict.fromkeys(tips_parallel_fallback_reasons)) or None,
        "per_worker_encoded_image_count": per_worker_encoded_image_count,
        "model_fingerprint_sha256": model_fingerprint_hash,
        "prompt_set_hash": prompt_set_hash,
        "patch_runtime_version": patch_runtime_version,
    }
    atomic_write_json(output_dir / "stage_manifest.json", manifest)
    return selected, manifest


def _worker_device(runtime: TipsV2Runtime) -> str:
    return str(getattr(runtime, "device", "unknown"))


def _worker_devices(workers: Sequence[TipsV2Runtime]) -> list[str]:
    return [_worker_device(worker) for worker in workers]


def _patch_worker_list(primary: TipsV2Runtime, patch_runtimes: Sequence[TipsV2Runtime] | None) -> list[TipsV2Runtime]:
    workers: list[TipsV2Runtime] = []
    seen = set()
    for worker in [primary, *(patch_runtimes or [])]:
        device = _worker_device(worker)
        if device in seen:
            continue
        seen.add(device)
        workers.append(worker)
    return workers or [primary]


def _deduplicate_patch_candidates(hits: Sequence[FrameHit], event_store: EventStore) -> tuple[list[FrameHit], dict]:
    groups: dict[str, list[tuple[int, FrameHit]]] = {}
    order: list[str] = []
    for index, hit in enumerate(hits):
        frame_atom = event_store.frame_by_embedding.get(hit.embedding_id)
        if frame_atom and not hit.frame_atom_id:
            hit.frame_atom_id = frame_atom.get("frame_atom_id")
        key = _patch_dedup_key(hit)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((index, hit))
    deduped: list[FrameHit] = []
    merged_record_count = 0
    for key in order:
        group = groups[key]
        best_index, best = sorted(group, key=lambda item: (-(item[1].cls_score if item[1].cls_score is not None else -1e9), item[0]))[0]
        if len(group) > 1:
            merged_record_count += len(group) - 1
            best.provenance["dedup_key"] = key
            best.provenance["merged_candidate_records"] = [
                {
                    "candidate_record_id": item.provenance.get("candidate_record_id"),
                    "embedding_id": item.embedding_id,
                    "frame_atom_id": item.frame_atom_id,
                    "cls_score": item.cls_score,
                    "provenance": _json_safe_provenance_snapshot(item.provenance),
                }
                for _index, item in group
            ]
            best.provenance["dedup_selected_record_index"] = best_index
        deduped.append(best)
    before = len(hits)
    after = len(deduped)
    return deduped, {
        "candidate_count_before_dedup": before,
        "candidate_count_after_dedup": after,
        "deduped_candidate_count": before - after,
        "dedup_key_policy": "frame_atom_id_else_embedding_id",
        "merged_candidate_record_count": merged_record_count,
    }


def _json_safe_provenance_snapshot(provenance: Mapping[str, object]) -> dict:
    snapshot = {str(key): value for key, value in provenance.items() if key != "merged_candidate_records"}
    try:
        return json.loads(json.dumps(snapshot, ensure_ascii=False, sort_keys=True, default=str))
    except (TypeError, ValueError) as exc:
        return {
            "snapshot_status": "failed",
            "snapshot_error": f"{type(exc).__name__}: {exc}",
            "source_keys": sorted(snapshot),
        }


def _patch_dedup_key(hit: FrameHit) -> str:
    if hit.frame_atom_id:
        return f"frame_atom_id:{hit.frame_atom_id}"
    return f"embedding_id:{hit.embedding_id}"


def _encode_image_with_patch_workers(
    workers: Sequence[TipsV2Runtime],
    image_paths: Sequence[Path],
    disabled_worker_devices: set[str],
) -> tuple[dict[str, np.ndarray], dict]:
    active_workers = [worker for worker in workers if _worker_device(worker) not in disabled_worker_devices]
    if not active_workers:
        active_workers = [workers[0]]
    active_workers = active_workers[: max(1, min(len(active_workers), len(image_paths)))]
    counts = {device: 0 for device in _worker_devices(workers)}
    fallback_reasons: list[str] = []
    active_workers, preload_report = _preload_patch_workers(workers, active_workers, disabled_worker_devices)
    fallback_reasons.extend(preload_report["fallback_reasons"])
    if not active_workers:
        active_workers = [workers[0]]
    if len(active_workers) == 1 or len(image_paths) <= 1:
        worker = active_workers[0]
        encoded = worker.encode_image_with_patches(image_paths)
        counts[_worker_device(worker)] = len(image_paths)
        return encoded, {"per_worker_encoded_image_count": counts, "fallback_reasons": fallback_reasons}

    chunks = _split_paths_for_workers(image_paths, len(active_workers))
    results: list[dict[str, np.ndarray] | None] = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        future_map = {
            executor.submit(worker.encode_image_with_patches, chunk): (index, worker, chunk)
            for index, (worker, chunk) in enumerate(zip(active_workers, chunks))
            if chunk
        }
        for future in as_completed(future_map):
            index, worker, chunk = future_map[future]
            device = _worker_device(worker)
            try:
                results[index] = future.result()
                counts[device] = counts.get(device, 0) + len(chunk)
            except Exception as exc:
                if worker is workers[0]:
                    raise
                disabled_worker_devices.add(device)
                fallback_reasons.append(f"{device}: {type(exc).__name__}: {exc}")
                release_runtime_state(worker)
                results[index] = workers[0].encode_image_with_patches(chunk)
                counts[_worker_device(workers[0])] = counts.get(_worker_device(workers[0]), 0) + len(chunk)
    encoded_results = [result for result in results if result is not None]
    combined = {"patch_tokens": np.concatenate([result["patch_tokens"] for result in encoded_results], axis=0)}
    if all("cls" in result for result in encoded_results):
        combined["cls"] = np.concatenate([result["cls"] for result in encoded_results], axis=0)
    return combined, {"per_worker_encoded_image_count": counts, "fallback_reasons": fallback_reasons}


def _preload_patch_workers(
    workers: Sequence[TipsV2Runtime],
    active_workers: Sequence[TipsV2Runtime],
    disabled_worker_devices: set[str],
) -> tuple[list[TipsV2Runtime], dict]:
    ready = []
    fallback_reasons = []
    for worker in active_workers:
        device = _worker_device(worker)
        try:
            _ensure_patch_worker_loaded(worker)
            ready.append(worker)
        except Exception as exc:
            if worker is workers[0]:
                raise
            disabled_worker_devices.add(device)
            fallback_reasons.append(f"{device}: {type(exc).__name__}: {exc}")
            release_runtime_state(worker)
    return ready, {"fallback_reasons": fallback_reasons}


def _ensure_patch_worker_loaded(worker: TipsV2Runtime) -> None:
    ensure_loaded = getattr(worker, "ensure_loaded", None)
    if callable(ensure_loaded):
        ensure_loaded()


def _split_paths_for_workers(image_paths: Sequence[Path], worker_count: int) -> list[list[Path]]:
    base = len(image_paths) // worker_count
    extra = len(image_paths) % worker_count
    chunks: list[list[Path]] = []
    start = 0
    for index in range(worker_count):
        size = base + (1 if index < extra else 0)
        chunks.append(list(image_paths[start : start + size]))
        start += size
    return chunks


def _expand_ois_l2_support_hits(query: TaskQuery, hits: Sequence[FrameHit], event_store: EventStore, *, enabled: bool) -> tuple[list[FrameHit], dict]:
    if query.task_type != "object_instance_search":
        return list(hits), {"status": "not_applicable", "reason": "task_is_not_ois", "added_frame_count": 0}
    if not enabled:
        return list(hits), {"status": "skipped", "reason": "smoke_or_explicit_max_frames", "added_frame_count": 0}
    selected = list(hits)
    seen_keys = set()
    for hit in selected:
        frame_atom = event_store.frame_by_embedding.get(hit.embedding_id)
        if frame_atom and not hit.frame_atom_id:
            hit.frame_atom_id = frame_atom.get("frame_atom_id")
        seen_keys.add(_rescan_frame_key({"frame_atom_id": hit.frame_atom_id, "tipsv2_embedding_id": hit.embedding_id}))
    events: dict[str, dict] = {}
    for hit in selected:
        frame_atom = event_store.frame_by_embedding.get(hit.embedding_id)
        if frame_atom:
            hit.frame_atom_id = frame_atom.get("frame_atom_id")
        for event_id, level in _linked_frame_events(hit, event_store):
            if level != "L2":
                continue
            event = event_store.event_by_id.get(event_id)
            if event and event.get("source") is not None:
                events[event_id] = event
    sampled_by_key: dict[str, dict] = {}
    sampled_before_dedup = 0
    theoretical_full_frame_count = 0
    added = []
    for event_id, event in sorted(events.items()):
        day = int(event.get("day_num") or 0)
        source = str(event.get("source") or "")
        start = float(event.get("start_second") or 0.0)
        end = float(event.get("end_second") or start)
        if not source or end <= start:
            continue
        interval_frames = _tips_frames_in_interval(event_store.frames_by_day_source.get((day, source), []), start, end)
        theoretical_full_frame_count += len(interval_frames)
        samples: list[tuple[str, dict]] = []
        if interval_frames:
            samples.append(("first_available_interval_frame", interval_frames[0]))
            samples.append(("last_available_interval_frame", interval_frames[-1]))
        representative = _first_tips_representative_frame(event, event_store)
        if representative:
            samples.append(("first_resolvable_representative_frame", representative))
        for sample_role, frame in samples:
            embedding_id = str(frame.get("tipsv2_embedding_id") or "")
            if not embedding_id:
                continue
            sampled_before_dedup += 1
            key = _rescan_frame_key(frame)
            existing = sampled_by_key.get(key)
            if existing:
                existing["sample_roles"].append(sample_role)
                existing["seed_l2_event_ids"].append(event_id)
                continue
            sampled_by_key[key] = {
                "frame": frame,
                "sample_roles": [sample_role],
                "seed_l2_event_ids": [event_id],
                "event_start_second": start,
                "event_end_second": end,
            }
    existing_candidate_overlap_count = 0
    for key, sample in sampled_by_key.items():
        frame = sample["frame"]
        embedding_id = str(frame.get("tipsv2_embedding_id") or "")
        if not embedding_id or key in seen_keys:
            existing_candidate_overlap_count += 1
            continue
        seen_keys.add(key)
        added.append(
            FrameHit(
                task_id=query.task_id,
                prompt=query.canonical_text_for_retrieval,
                embedding_id=embedding_id,
                day=int(frame.get("day_num") or 0),
                source=str(frame.get("source") or ""),
                absolute_second=float(frame.get("absolute_second") or 0.0),
                frame_rel_path=str(frame.get("frame_rel_path") or ""),
                image_path_ref=str(frame.get("image_path_ref") or frame.get("frame_rel_path") or ""),
                cls_score=0.0,
                frame_atom_id=str(frame.get("frame_atom_id") or "") or None,
                provenance={
                    "candidate_record_id": f"ois-l2-rescan:{','.join(dict.fromkeys(sample['seed_l2_event_ids']))}:{frame.get('frame_atom_id') or embedding_id}",
                    "ois_l2_rescan": {
                        "policy": "l2_keyframe_rescan_v1",
                        "sample_roles": list(dict.fromkeys(sample["sample_roles"])),
                        "seed_l2_event_ids": list(dict.fromkeys(sample["seed_l2_event_ids"])),
                        "event_start_second": sample["event_start_second"],
                        "event_end_second": sample["event_end_second"],
                        "dedup_key": key,
                    },
                },
            )
        )
    selected.extend(added)
    sampled_after_dedup = len(sampled_by_key)
    return selected, {
        "status": "completed",
        "policy": "l2_keyframe_rescan_v1",
        "sampling_strategy": "first_last_first_representative_tips_frame_per_seed_l2",
        "dedup_key_policy": "frame_atom_id_else_embedding_id",
        "seed_l2_event_count": len(events),
        "theoretical_full_frame_count": theoretical_full_frame_count,
        "initial_frame_count": len(hits),
        "sampled_frame_count_before_dedup": sampled_before_dedup,
        "sampled_frame_count_after_dedup": sampled_after_dedup,
        "duplicate_sampled_frame_count": sampled_before_dedup - sampled_after_dedup,
        "existing_candidate_overlap_count": existing_candidate_overlap_count,
        "added_frame_count": len(added),
        "final_frame_count": len(selected),
    }


def _tips_frames_in_interval(frames: Sequence[dict], start: float, end: float) -> list[dict]:
    rows = []
    for frame in frames:
        second = float(frame.get("absolute_second") or -1.0)
        if second >= end:
            break
        if (
            start <= second
            and frame.get("tipsv2_embedding_id")
            and frame.get("is_canonical_frame", True) is not False
        ):
            rows.append(frame)
    return rows


def _first_tips_representative_frame(event: dict, event_store: EventStore) -> dict | None:
    for frame_id in event.get("representative_frame_ids") or []:
        frame = event_store.frame_by_id.get(str(frame_id))
        if frame and frame.get("tipsv2_embedding_id"):
            return frame
    return None


def _rescan_frame_key(frame: Mapping[str, object]) -> str:
    frame_atom_id = str(frame.get("frame_atom_id") or "")
    if frame_atom_id:
        return f"frame_atom_id:{frame_atom_id}"
    return f"embedding_id:{frame.get('tipsv2_embedding_id') or ''}"


def _load_patch_score_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    cache: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            cache_key = str(row.get("cache_key") or "")
            if not cache_key:
                continue
            value = dict(row)
            value.pop("cache_key", None)
            required = {"patch_score", "patch_top5_mean", "frame_file_sha256", "patch_token_count", "patch_dim", "model_fingerprint_sha256", "prompt_set_hash", "patch_runtime_version"}
            if required <= set(value):
                cache[cache_key] = value
    return cache


def _patch_cache_key(
    *,
    model_fingerprint_hash: str,
    task_id: str,
    frame_atom_id: str,
    frame_file_sha256: str,
    prompt_set_hash: str,
    patch_runtime_version: str,
) -> str:
    payload = {
        "model_fingerprint_sha256": model_fingerprint_hash,
        "task_id": task_id,
        "frame_atom_id": frame_atom_id,
        "frame_file_sha256": frame_file_sha256,
        "prompt_set_hash": prompt_set_hash,
        "patch_runtime_version": patch_runtime_version,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"tipsv2-patch::{digest}"


def _write_post_patch_pruning_audit(query: TaskQuery, hits: Sequence[FrameHit], event_store: EventStore, output_dir: Path) -> tuple[list[FrameHit], dict]:
    audit_rows = []
    if query.task_type == "question_answering":
        atomic_write_jsonl(output_dir / "post_patch_pruning.jsonl", audit_rows)
        return list(hits), {"policy": "qa_visual_probe_cluster_topn", "kept_count": len(hits), "pruned_count": 0}
    linked = {id(hit): _linked_frame_events(hit, event_store) for hit in hits}
    for hit in hits:
        events = linked[id(hit)]
        l2_ids = [event_id for event_id, level in events if level == "L2"]
        l1_ids = [event_id for event_id, level in events if level == "L1"]
        audit = {
            "task_id": query.task_id,
            "embedding_id": hit.embedding_id,
            "candidate_record_id": hit.provenance.get("candidate_record_id"),
            "frame_atom_id": hit.frame_atom_id,
            "action": "kept_for_event_scoring",
            "reason": "event_level_best_patch_scoring_requires_full_candidate_set",
            "l2_event_ids": l2_ids,
            "protected_l1_event_ids": l1_ids,
            "patch_score": hit.patch_score,
            "rank": hit.rank,
        }
        hit.provenance["post_patch_pruning"] = audit
        audit_rows.append(audit)
    atomic_write_jsonl(output_dir / "post_patch_pruning.jsonl", audit_rows)
    return list(hits), {"policy": "event_level_best_patch_no_pruning_v1", "kept_count": len(hits), "pruned_count": 0}


def _linked_frame_events(hit: FrameHit, event_store: EventStore) -> list[tuple[str, str]]:
    if not hit.frame_atom_id:
        return []
    events = []
    for link in event_store.frame_links_by_frame.get(hit.frame_atom_id, []):
        if link.get("link_type") != "frame_overlap_support":
            continue
        event_id = str(link.get("event_id") or "")
        event = event_store.event_by_id.get(event_id)
        level = str((event or {}).get("level") or "")
        if event_id and level in {"L1", "L2"}:
            events.append((event_id, level))
    return events


def rerank_text_hits(
    query: TaskQuery,
    hits: Sequence[TextHit],
    runtime: Qwen3RerankerRuntime,
    output_dir: Path,
    *,
    batch_size: int = 4,
    cap: int | None = None,
    allow_cap_expansion: bool = True,
) -> tuple[list[TextHit], dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected, cap_audit = _select_text_rerank_candidates(query, hits, cap=cap, allow_cap_expansion=allow_cap_expansion)
    instruction = "Given an egocentric video search query, judge whether the transcript passage provides evidence that the described event, object, or question evidence appears."
    pairs = []
    documents_by_chunk: dict[str, str] = {}
    for hit in selected:
        document = (
            f"day={hit.day}; source={hit.source}; start={hit.start_second}; end={hit.end_second}\n"
            f"quality={','.join(hit.quality_flags)}\ntext={hit.text_for_embedding}"
        )
        documents_by_chunk[hit.embedding_chunk_id] = document
        pairs.append((hit.embedding_chunk_id, query.canonical_text_for_retrieval, document))
    results = runtime.score(pairs, instruction=instruction, batch_size=batch_size) if pairs else []
    negative_prompts = query.negative_prompt_set()
    negative_pairs = []
    for hit in selected:
        document = documents_by_chunk.get(hit.embedding_chunk_id)
        if document is None:
            continue
        for prompt_index, negative_prompt in enumerate(negative_prompts):
            pair_id = f"{hit.embedding_chunk_id}::negative::{prompt_index}"
            negative_pairs.append((pair_id, negative_prompt, document))
    negative_results = runtime.score(negative_pairs, instruction=instruction, batch_size=batch_size) if negative_pairs else []
    negative_by_chunk: dict[str, list[dict]] = {}
    for (pair_id, query_text, _document), result in zip(negative_pairs, negative_results):
        chunk_id, _, prompt_index_text = pair_id.partition("::negative::")
        try:
            prompt_index = int(prompt_index_text)
        except ValueError:
            prompt_index = -1
        negative_by_chunk.setdefault(chunk_id, []).append(
            {
                **result,
                "negative_prompt": query_text,
                "negative_prompt_index": prompt_index,
            }
        )
    rows = []
    negative_rows = []
    for hit, result in zip(selected, results):
        hit.reranker_score = float(result["reranker_score"])
        hit.reranker_raw_score = float(result["reranker_raw_score"])
        hit.provenance["qwen3_reranker"] = result
        negative_candidates = negative_by_chunk.get(hit.embedding_chunk_id) or []
        if negative_candidates:
            negative_result = max(
                negative_candidates,
                key=lambda item: (float(item.get("reranker_score") or 0.0), -int(item.get("negative_prompt_index") or 0)),
            )
            hit.negative_reranker_score = float(negative_result["reranker_score"])
            hit.negative_reranker_raw_score = float(negative_result["reranker_raw_score"])
            hit.provenance["qwen3_negative_reranker"] = {
                **negative_result,
                "negative_prompt_count": len(negative_prompts),
                "negative_prompt_scores": [
                    {
                        "negative_prompt": item.get("negative_prompt"),
                        "negative_prompt_index": item.get("negative_prompt_index"),
                        "reranker_score": item.get("reranker_score"),
                        "reranker_raw_score": item.get("reranker_raw_score"),
                        "token_count": item.get("token_count"),
                    }
                    for item in sorted(negative_candidates, key=lambda entry: int(entry.get("negative_prompt_index") or 0))
                ],
            }
            negative_rows.append(
                {
                    "task_id": hit.task_id,
                    "embedding_chunk_id": hit.embedding_chunk_id,
                    "day": hit.day,
                    "source": hit.source,
                    "negative_reranker_score": hit.negative_reranker_score,
                    "negative_reranker_raw_score": hit.negative_reranker_raw_score,
                    "negative_prompt": negative_result.get("negative_prompt"),
                    "negative_prompt_index": negative_result.get("negative_prompt_index"),
                    "negative_prompt_count": len(negative_prompts),
                    "negative_prompt_scores": hit.provenance["qwen3_negative_reranker"]["negative_prompt_scores"],
                    "token_count": negative_result.get("token_count"),
                }
            )
        rows.append(
            {
                "task_id": hit.task_id,
                "embedding_chunk_id": hit.embedding_chunk_id,
                "day": hit.day,
                "source": hit.source,
                "dense_score": hit.dense_score,
                "sparse_score": hit.sparse_score,
                "reranker_score": hit.reranker_score,
                "reranker_raw_score": hit.reranker_raw_score,
                "negative_reranker_score": hit.negative_reranker_score,
                "negative_reranker_raw_score": hit.negative_reranker_raw_score,
                "token_count": result.get("token_count"),
            }
        )
    selected.sort(key=lambda item: (-(item.reranker_score if item.reranker_score is not None else -1e9), item.embedding_chunk_id))
    for index, hit in enumerate(selected, start=1):
        hit.rank = index
    atomic_write_jsonl(output_dir / "reranked_text_hits.jsonl", rows)
    atomic_write_jsonl(output_dir / "negative_reranked_text_hits.jsonl", negative_rows)
    manifest = {
        "status": "completed",
        "task_id": query.task_id,
        "input_count": len(selected),
        "output_count": len(rows),
        "negative_output_count": len(negative_rows),
        "negative_prompt_count": len(negative_prompts),
        "cap": cap,
        "cap_boundary_audit": cap_audit,
        "runtime": runtime.evidence(),
    }
    atomic_write_json(output_dir / "stage_manifest.json", manifest)
    return selected, manifest


def _select_text_rerank_candidates(
    query: TaskQuery,
    hits: Sequence[TextHit],
    *,
    cap: int | None,
    allow_cap_expansion: bool,
) -> tuple[list[TextHit], dict]:
    rows = list(hits)
    if cap is None or cap <= 0 or len(rows) <= cap:
        return rows, {
            "cap_hit": False,
            "initial_cap": cap,
            "final_cap": len(rows),
            "candidate_count": len(rows),
            "input_candidate_ids_sha256": _text_hit_ids_hash(rows),
            "selected_candidate_ids_sha256": _text_hit_ids_hash(rows),
            "unresolved_count": 0,
            "decision": "full_candidate_set",
        }
    initial_cap = min(cap, len(rows))
    final_cap = initial_cap
    steps = [{"cap": initial_cap, "reason": "initial_cap"}]
    unresolved = _cap_boundary_unresolved(query, rows, final_cap)
    if allow_cap_expansion:
        for multiplier in (1.5, 2.0):
            if not unresolved or final_cap >= len(rows):
                break
            final_cap = min(len(rows), max(final_cap + 1, int(round(initial_cap * multiplier))))
            steps.append({"cap": final_cap, "reason": f"expand_{multiplier:g}x"})
            unresolved = _cap_boundary_unresolved(query, rows, final_cap)
        if unresolved and final_cap < len(rows):
            final_cap = len(rows)
            steps.append({"cap": final_cap, "reason": "full_candidate_set"})
            unresolved = _cap_boundary_unresolved(query, rows, final_cap)
    decision = "expanded" if final_cap > initial_cap else "accepted_cap_boundary"
    if unresolved and not allow_cap_expansion:
        decision = "smoke_cap_boundary_diagnostic_only"
    elif unresolved:
        decision = "full_candidate_set_unresolved_low_risk"
    return (
        rows[:final_cap],
        {
            "cap_hit": True,
            "initial_cap": initial_cap,
            "final_cap": final_cap,
            "candidate_count": len(rows),
            "input_candidate_ids_sha256": _text_hit_ids_hash(rows),
            "selected_candidate_ids_sha256": _text_hit_ids_hash(rows[:final_cap]),
            "unresolved_count": len(unresolved),
            "unresolved_boundary_ids": [item.embedding_chunk_id for item in unresolved[:20]],
            "unresolved_boundary_ids_sha256": _text_hit_ids_hash(unresolved),
            "expansion_steps": steps,
            "allow_cap_expansion": allow_cap_expansion,
            "decision": decision,
        },
    )


def _cap_boundary_unresolved(query: TaskQuery, hits: Sequence[TextHit], cap: int) -> list[TextHit]:
    if cap >= len(hits):
        return []
    hard_floor = float(query.resolved_profiles.get("threshold_profile", {}).get("value", {}).get("hard_floor", 0.0))
    denominator = max(len(hits) - 1, 1)
    boundary = hits[cap : min(len(hits), cap + 50)]
    unresolved = []
    for offset, hit in enumerate(boundary, start=cap + 1):
        rank_norm = 1.0 - (offset - 1) / denominator
        dual_channel = hit.dense_score is not None and hit.sparse_score is not None
        if rank_norm >= hard_floor or dual_channel:
            unresolved.append(hit)
    return unresolved


def _text_hit_ids_hash(hits: Sequence[TextHit]) -> str:
    ids = [str(hit.embedding_chunk_id) for hit in hits]
    return hashlib.sha256(json.dumps(ids, ensure_ascii=False).encode("utf-8")).hexdigest()
