from __future__ import annotations

import json
import platform
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

from .config import AppConfig, parse_devices, run_output_root
from .data import EventStore, build_coverage_index
from .io_utils import atomic_write_json, atomic_write_jsonl, atomic_write_text, canonical_json_hash, now_utc, sha256_file
from .models import AnswerResult, EvidenceBundle, FrameHit, TaskQuery, TextHit
from .official import save_official_snapshot
from .qa import answer_question, answer_question_from_packet, build_qa_evidence, write_qa_evidence_packet
from .registry import load_query_registry, write_normalized_registry
from .rerank import bypass_visual_patch_rerank_with_cls, rerank_text_hits, rerank_visual_patches
from .retrieval import merge_text_hits, retrieve_text_dense, retrieve_text_dense_batch, retrieve_text_sparse, retrieve_text_sparse_batch, retrieve_visual_cls, retrieve_visual_cls_batch
from .runtimes import BGEM3SparseRuntime, OpenRouterClient, Qwen3DenseRuntime, Qwen3RerankerRuntime, TipsV2Runtime, package_versions, release_runtime_state
from .solver import build_candidates, solve_interval_task
from .submission import OFFICIAL_QA_MODEL, write_diagnostic_smoke_bundle, write_openreview_package_checklist, write_submission_bundle
from .task_manifest import read_task_manifest


FULLFLOW_SMOKE_PROFILE_NAME = "fullflow_smoke_top100"
FULLFLOW_SMOKE_TASK_IDS = ("MM26-EIS01", "MM26-OIS01", "MM26-QA01")
FULLFLOW_SMOKE_INTERVAL_TASK_IDS = ("MM26-EIS01", "MM26-OIS01")
FULLFLOW_SMOKE_QA_TASK_IDS = ("MM26-QA01",)
FULLFLOW_SMOKE_ALL_INTERVAL_TASK_IDS = tuple(
    [f"MM26-EIS{index:02d}" for index in range(1, 13)]
    + [f"MM26-OIS{index:02d}" for index in range(1, 13)]
)
FULLFLOW_SMOKE_ALL_QA_TASK_IDS = tuple(f"MM26-QA{index:02d}" for index in range(1, 13))
FULLFLOW_SMOKE_ALL_TASK_IDS = FULLFLOW_SMOKE_ALL_INTERVAL_TASK_IDS + FULLFLOW_SMOKE_ALL_QA_TASK_IDS
CLS_VISUAL_SOURCE_BYPASS_TASK_IDS = {"MM26-OIS11"}


@dataclass(frozen=True)
class ExecutionProfile:
    name: str
    smoke_limit_files: int | None
    visual_top_k: int | None
    text_top_k_by_type: Mapping[str, int]
    text_rerank_cap_by_type: Mapping[str, int]
    qa_evidence_keep_top: int | None
    patch_max_frames: int | None
    local_recovery_patch_max_frames: int | None
    text_rerank_batch_size: int
    patch_batch_size: int
    allow_text_cap_expansion: bool
    visual_reserve_per_day_source: int = 0
    visual_reserve_per_source_hour: int = 0
    visual_probe_reserve_per_day_source: int = 0
    visual_probe_reserve_per_source_hour: int = 0
    text_reserve_per_source: int = 0
    local_recovery_candidate_cap: int | None = None
    local_recovery_max_frames_per_text_hit: int | None = None
    override_recall_profile: bool = False


@dataclass
class RetrievedTaskInputs:
    text_dense_hits: list[TextHit]
    dense_manifest: dict
    text_sparse_hits: list[TextHit]
    sparse_manifest: dict
    frame_hits: list[FrameHit]
    visual_manifest: dict


FULLFLOW_SMOKE_TOP100_PROFILE = ExecutionProfile(
    name=FULLFLOW_SMOKE_PROFILE_NAME,
    smoke_limit_files=None,
    visual_top_k=100,
    text_top_k_by_type={
        "event_instance_search": 20,
        "object_instance_search": 10,
        "question_answering": 150,
    },
    text_rerank_cap_by_type={
        "event_instance_search": 40,
        "object_instance_search": 20,
        "question_answering": 300,
    },
    qa_evidence_keep_top=20,
    patch_max_frames=None,
    local_recovery_patch_max_frames=None,
    text_rerank_batch_size=4,
    patch_batch_size=8,
    allow_text_cap_expansion=False,
    visual_reserve_per_day_source=1,
    visual_reserve_per_source_hour=1,
    visual_probe_reserve_per_day_source=1,
    visual_probe_reserve_per_source_hour=1,
    local_recovery_candidate_cap=50,
    local_recovery_max_frames_per_text_hit=8,
    override_recall_profile=True,
)


def preflight(config: AppConfig, queries_path: Path, run_id: str, *, fetch_official: bool = True, with_models: bool = False, devices: str = "0") -> dict:
    root = run_output_root(config, run_id)
    registry_dir = root / "registry"
    diagnostics_dir = root / "diagnostics"
    registry_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    official_queries = None
    official_snapshot = None
    if fetch_official:
        url = config.data["official_sources"]["challenge_url"]
        official_snapshot = save_official_snapshot(url, registry_dir)
        official_queries = official_snapshot["queries"]
    queries, registry_payload = load_query_registry(queries_path, config, official_queries=official_queries)
    write_normalized_registry(registry_dir / "normalized_queries.yaml", registry_payload["normalized"], registry_payload["report"])
    profile_report_path = registry_dir / "profile_resolution_report.json"
    atomic_write_json(
        profile_report_path,
        {
            "schema_version": "castle_mm26_profile_resolution_report_v1",
            "registry_validation": registry_payload["report"],
            "tasks": [
                {
                    "task_id": task["task_id"],
                    "task_type": task["task_type"],
                    "recall_profile": task.get("recall_profile"),
                    "scoring_profile": task.get("scoring_profile"),
                    "threshold_profile": task.get("threshold_profile"),
                    "output_profile": task.get("output_profile"),
                    "resolved_profiles": task.get("resolved_profiles"),
                }
                for task in registry_payload["normalized"]
            ],
        },
    )
    coverage = build_coverage_index(config, registry_dir)
    day_proof = _validate_day_proof(config)
    model_report = {}
    if with_models:
        model_report = model_preflight(config, devices=devices, output_dir=diagnostics_dir)
    official_manifest = official_snapshot["manifest"] if official_snapshot else {"valid": False, "reason": "not_fetched"}
    preflight_errors = _preflight_report_errors(
        config=config,
        registry_report=registry_payload["report"],
        official_manifest=official_manifest,
        day_proof=day_proof,
        with_models=with_models,
    )
    if with_models:
        preflight_errors.extend(_preflight_errors(model_report, require_chat_ping=False))
    report = {
        "valid": not preflight_errors,
        "run_id": run_id,
        "config_path": config.path.as_posix(),
        "queries_path": queries_path.resolve().as_posix(),
        "task_count": len(queries),
        "registry_validation": registry_payload["report"],
        "official_sources_manifest": official_manifest,
        "coverage_interval_count": len(coverage.rows),
        "day_date_proof": day_proof,
        "model_preflight": model_report,
        "errors": preflight_errors,
    }
    atomic_write_json(root / "preflight_report.json", report)
    manifest = base_manifest(config, run_id)
    manifest.update(
        {
            "preflight": report,
            "query_registry_path": queries_path.resolve().as_posix(),
            "query_registry_sha256": sha256_file(queries_path),
            "normalized_queries_path": (registry_dir / "normalized_queries.yaml").as_posix(),
            "normalized_queries_sha256": sha256_file(registry_dir / "normalized_queries.yaml"),
            "profile_resolution_report_path": profile_report_path.as_posix(),
            "profile_resolution_report_sha256": sha256_file(profile_report_path),
            "accepted_source_interval_index_path": (registry_dir / "accepted_source_interval_index.parquet").as_posix(),
        }
    )
    atomic_write_json(root / "run_manifest.json", manifest)
    if preflight_errors:
        raise RuntimeError(f"preflight failed: {preflight_errors}")
    return report


def model_preflight(config: AppConfig, *, devices: str, output_dir: Path) -> dict:
    device_list = parse_devices(devices)
    model_devices = assign_model_devices(device_list)
    openrouter = make_openrouter_client(config)
    openrouter_report = openrouter_preflight(config, openrouter)
    event_store = EventStore.from_config(config)
    first_frame = next(iter(event_store.frame_by_embedding.values()))
    from .data import resolve_frame_path

    image_path = resolve_frame_path(config, str(first_frame.get("frame_rel_path") or ""), str(first_frame.get("image_path_ref") or ""))
    cache = output_dir / "runtime_cache" / "model_preflight"
    tips = TipsV2Runtime(config.model_profile["tipsv2_model_root"], device=model_devices["tipsv2"], cache_dir=cache / "tipsv2")
    tips_report = tips.preflight(image_path, "a person pouring tea into a mug")
    qwen = Qwen3RerankerRuntime(config.model_profile["qwen3_reranker_model_root"], device=model_devices["qwen3_reranker"], cache_dir=cache / "qwen3_reranker", dtype="bfloat16")
    # Load tokenizer/model with one tiny pair to prove real model execution.
    qwen_scores = qwen.score([("preflight", "test query", "test document")], instruction="judge relevance", batch_size=1)
    report = {"device_assignment": model_devices, "tipsv2": tips_report, "qwen3_reranker": {"status": "completed", "scores": qwen_scores, "runtime": qwen.evidence()}, "openrouter": openrouter_report}
    atomic_write_json(output_dir / "model_preflight.json", report)
    return report


def openrouter_preflight(config: AppConfig, openrouter: OpenRouterClient) -> dict:
    report = {"api_key_source": f"env:{config.data['qa_mllm']['api_key_env']}", "api_key_present": openrouter.api_key_present(), "model": openrouter.model}
    try:
        metadata = openrouter.model_metadata()
        supported = metadata.get("supported_parameters") or []
        report.update({"model_found": True, "supported_parameters": supported, "required_supported": all(item in supported for item in config.data["qa_mllm"]["required_supported_parameters"])})
    except Exception as exc:
        report.update({"model_found": False, "required_supported": False, "error": f"{type(exc).__name__}: {exc}"})
    if report["api_key_present"]:
        try:
            ping = openrouter.answer(
                question="Schema ping. Return answer ok.",
                evidence=[{"evidence_id": "preflight:ev:001", "text_excerpt": "Synthetic preflight evidence only."}],
                schema=config.data["qa_mllm"]["response_format"],
                max_images=0,
            )
            report["chat_completion_ping"] = {"status": "completed", "valid_json": isinstance(ping, dict), "answer": ping.get("answer") if isinstance(ping, dict) else None}
        except Exception as exc:
            report["chat_completion_ping"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    else:
        report["chat_completion_ping"] = {"status": "skipped", "reason": f"{config.data['qa_mllm']['api_key_env']} is not set"}
    return report


def run_tasks(
    config: AppConfig,
    queries_path: Path,
    run_id: str,
    *,
    task_ids: Sequence[str],
    devices: str,
    smoke: bool,
    fetch_official: bool = True,
    resume: bool = False,
    task_family: str | None = None,
    execution_profile: ExecutionProfile | None = None,
) -> dict:
    root = run_output_root(config, run_id)
    _ensure_valid_preflight(
        config,
        queries_path,
        run_id,
        root,
        smoke=smoke,
        fetch_official=fetch_official,
        devices=devices,
        missing_message="official task execution requires an explicit preflight for this run_id before running tasks",
        invalid_message="official task execution requires a valid preflight_report.json before running tasks",
    )
    official_queries = None
    if fetch_official and (root / "registry" / "official_queries_extracted.json").exists():
        official_queries = json.loads((root / "registry" / "official_queries_extracted.json").read_text(encoding="utf-8"))
    queries, _ = load_query_registry(queries_path, config, official_queries=official_queries)
    family_type = _task_family_type(task_family) if task_family else None
    selected = [query for query in queries if (not task_ids or query.task_id in task_ids) and (family_type is None or query.task_type == family_type)]
    selected = [_query_for_execution_profile(query, execution_profile) for query in selected]
    if not selected:
        raise ValueError("no selected tasks")
    skipped_reports = []
    active = []
    for query in selected:
        task_dir = root / "tasks" / query.task_type / query.task_id
        if resume and _task_completed_for_query(config, query, task_dir):
            skipped_reports.append({"task_id": query.task_id, "task_type": query.task_type, "status": "skipped_completed_manifest_match", "task_dir": task_dir.as_posix()})
        else:
            active.append(query)
    if any(query.task_type == "question_answering" for query in active):
        openrouter_probe = make_openrouter_client(config)
        openrouter_report = openrouter_preflight(config, openrouter_probe)
        openrouter_errors = _preflight_errors({"openrouter": openrouter_report})
        if openrouter_errors:
            failure_manifest = _load_manifest(root)
            failure_manifest["stages"].append({"name": "run_tasks", "status": "failed", "finished_at_utc": now_utc(), "reason": "; ".join(openrouter_errors), "openrouter_preflight": openrouter_report, "task_count": len(selected), "smoke": smoke, "execution_profile": _execution_profile_name(execution_profile)})
            atomic_write_json(root / "run_manifest.json", failure_manifest)
            raise RuntimeError(f"OpenRouter readiness failed: {openrouter_errors}")
    if not active:
        task_report_path = root / "diagnostics" / "task_reports.json"
        task_report_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(task_report_path, skipped_reports)
        existing_manifest = _load_manifest(root)
        existing_manifest["stages"].append({"name": "run_tasks", "status": "completed", "finished_at_utc": now_utc(), "task_count": len(selected), "executed_task_count": 0, "skipped_task_count": len(skipped_reports), "smoke": smoke, "resume": resume, "task_family": task_family, "execution_profile": _execution_profile_name(execution_profile)})
        atomic_write_json(root / "run_manifest.json", existing_manifest)
        return {"task_reports": [], "evidence": [], "qa_results": []}
    coverage = build_coverage_index(config, root / "registry")
    event_store = EventStore.from_config(config)
    device_list = parse_devices(devices)
    model_devices = assign_model_devices(device_list)
    cache = root / "cache"
    shared_cache = config.output_root / ".runtime_cache"
    tips = TipsV2Runtime(config.model_profile["tipsv2_model_root"], device=model_devices["tipsv2"], cache_dir=cache / "tipsv2")
    tips_patch_workers = _build_tips_patch_workers(config, cache, model_devices, tips)
    dense = Qwen3DenseRuntime(config.model_profile["qwen3_embedding_model_root"], device=model_devices["qwen3_dense"], cache_dir=cache / "qwen3_dense")
    sparse = BGEM3SparseRuntime(config.model_profile["bgem3_model_root"], devices=model_devices["bgem3_sparse"], cache_dir=shared_cache / "bgem3")
    reranker = Qwen3RerankerRuntime(config.model_profile["qwen3_reranker_model_root"], device=model_devices["qwen3_reranker"], cache_dir=cache / "qwen3_reranker")
    openrouter = make_openrouter_client(config)
    retrieved = _batch_retrieve_task_inputs(config, active, tips, dense, sparse, smoke=smoke, execution_profile=execution_profile)
    all_evidence: list[EvidenceBundle] = []
    qa_results: list[AnswerResult] = []
    task_reports = []
    for query in active:
        task_dir = root / "tasks" / query.task_type / query.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        try:
            report = run_one_task(
                config,
                query,
                event_store,
                coverage,
                tips,
                dense,
                sparse,
                reranker,
                openrouter,
                task_dir,
                smoke=smoke,
                execution_profile=execution_profile,
                retrieved_inputs=retrieved[query.task_id],
                tips_patch_workers=tips_patch_workers,
            )
        except Exception as exc:
            report = _failed_task_report(query, task_dir, exc, execution_profile=execution_profile)
            task_reports.append(report)
            atomic_write_json(root / "diagnostics" / "task_reports.json", task_reports + skipped_reports)
            continue
        all_evidence.extend(report.pop("evidence_bundles", []))
        if "qa_result" in report:
            qa_results.append(report.pop("qa_result"))
        task_reports.append(report)
    atomic_write_json(root / "diagnostics" / "task_reports.json", task_reports + skipped_reports)
    failed_reports = [item for item in task_reports if item.get("status") == "failed"]
    existing_manifest = _load_manifest(root)
    existing_manifest["stages"].append(
        {
            "name": "run_tasks",
            "status": "failed" if failed_reports else "completed",
            "finished_at_utc": now_utc(),
            "task_count": len(selected),
            "executed_task_count": len([item for item in task_reports if item.get("status") != "failed"]),
            "failed_task_count": len(failed_reports),
            "skipped_task_count": len(skipped_reports),
            "smoke": smoke,
            "resume": resume,
            "task_family": task_family,
            "device_assignment": model_devices,
            "execution_profile": _execution_profile_name(execution_profile),
            "retrieval_batching": _batch_retrieval_manifest(retrieved),
        }
    )
    atomic_write_json(root / "run_manifest.json", existing_manifest)
    if failed_reports:
        failed_ids = ", ".join(item["task_id"] for item in failed_reports)
        raise RuntimeError(f"{len(failed_reports)} task(s) failed after other active tasks were attempted: {failed_ids}")
    return {"task_reports": task_reports, "evidence": all_evidence, "qa_results": qa_results}


def export_qa_packets(
    config: AppConfig,
    queries_path: Path,
    run_id: str,
    *,
    task_ids: Sequence[str],
    devices: str,
    smoke: bool,
    fetch_official: bool = True,
    resume: bool = False,
    execution_profile: ExecutionProfile | None = None,
) -> dict:
    root = run_output_root(config, run_id)
    _ensure_valid_preflight(
        config,
        queries_path,
        run_id,
        root,
        smoke=smoke,
        fetch_official=fetch_official,
        devices=devices,
        missing_message="official QA export requires an explicit preflight for this run_id before exporting evidence packets",
        invalid_message="official QA export requires a valid preflight_report.json before exporting evidence packets",
    )
    official_queries = None
    if fetch_official and (root / "registry" / "official_queries_extracted.json").exists():
        official_queries = json.loads((root / "registry" / "official_queries_extracted.json").read_text(encoding="utf-8"))
    queries, _ = load_query_registry(queries_path, config, official_queries=official_queries)
    selected = [query for query in queries if query.task_type == "question_answering" and (not task_ids or query.task_id in task_ids)]
    selected = [_query_for_execution_profile(query, execution_profile) for query in selected]
    if not selected:
        raise ValueError("no selected QA tasks")
    event_store = EventStore.from_config(config)
    device_list = parse_devices(devices)
    model_devices = assign_model_devices(device_list)
    cache = root / "cache"
    shared_cache = config.output_root / ".runtime_cache"
    tips = TipsV2Runtime(config.model_profile["tipsv2_model_root"], device=model_devices["tipsv2"], cache_dir=cache / "tipsv2")
    tips_patch_workers = _build_tips_patch_workers(config, cache, model_devices, tips)
    dense = Qwen3DenseRuntime(config.model_profile["qwen3_embedding_model_root"], device=model_devices["qwen3_dense"], cache_dir=cache / "qwen3_dense")
    sparse = BGEM3SparseRuntime(config.model_profile["bgem3_model_root"], devices=model_devices["bgem3_sparse"], cache_dir=shared_cache / "bgem3")
    reranker = Qwen3RerankerRuntime(config.model_profile["qwen3_reranker_model_root"], device=model_devices["qwen3_reranker"], cache_dir=cache / "qwen3_reranker")
    task_reports = []
    skipped_reports = []
    active = []
    for query in selected:
        task_dir = root / "tasks" / query.task_type / query.task_id
        packet_path = task_dir / "qa_evidence_packet.json"
        if resume and _qa_packet_complete_for_query(query, packet_path):
            skipped_reports.append({"task_id": query.task_id, "task_type": query.task_type, "status": "skipped_existing_packet_manifest_match", "packet_path": packet_path.as_posix()})
            continue
        active.append(query)
    retrieved = _batch_retrieve_task_inputs(config, active, tips, dense, sparse, smoke=smoke, execution_profile=execution_profile) if active else {}
    for query in active:
        task_dir = root / "tasks" / query.task_type / query.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        try:
            report = export_one_qa_packet(
                config,
                query,
                event_store,
                tips,
                dense,
                sparse,
                reranker,
                task_dir,
                smoke=smoke,
                execution_profile=execution_profile,
                retrieved_inputs=retrieved[query.task_id],
                tips_patch_workers=tips_patch_workers,
            )
        except Exception as exc:
            report = _failed_task_report(query, task_dir, exc, execution_profile=execution_profile)
            task_reports.append(report)
            atomic_write_json(root / "diagnostics" / "qa_export_reports.json", task_reports + skipped_reports)
            continue
        task_reports.append(report)
    atomic_write_json(root / "diagnostics" / "qa_export_reports.json", task_reports + skipped_reports)
    failed_reports = [item for item in task_reports if item.get("status") == "failed"]
    existing_manifest = _load_manifest(root)
    existing_manifest["stages"].append(
        {
            "name": "qa_export",
            "status": "failed" if failed_reports else "completed",
            "finished_at_utc": now_utc(),
            "task_count": len(selected),
            "executed_task_count": len([item for item in task_reports if item.get("status") != "failed"]),
            "failed_task_count": len(failed_reports),
            "skipped_task_count": len(skipped_reports),
            "resume": resume,
            "device_assignment": model_devices,
            "api_call_performed": False,
            "execution_profile": _execution_profile_name(execution_profile),
            "retrieval_batching": _batch_retrieval_manifest(retrieved),
        }
    )
    atomic_write_json(root / "run_manifest.json", existing_manifest)
    if failed_reports:
        failed_ids = ", ".join(item["task_id"] for item in failed_reports)
        raise RuntimeError(f"{len(failed_reports)} QA packet export task(s) failed after other active tasks were attempted: {failed_ids}")
    return {"task_reports": task_reports, "skipped_reports": skipped_reports}


def export_one_qa_packet(
    config: AppConfig,
    query: TaskQuery,
    event_store: EventStore,
    tips: TipsV2Runtime,
    dense: Qwen3DenseRuntime,
    sparse: BGEM3SparseRuntime,
    reranker: Qwen3RerankerRuntime,
    task_dir: Path,
    *,
    smoke: bool,
    execution_profile: ExecutionProfile | None = None,
    retrieved_inputs: RetrievedTaskInputs | None = None,
    tips_patch_workers: Sequence[TipsV2Runtime] | None = None,
) -> dict:
    query = _query_for_execution_profile(query, execution_profile)
    limits = _execution_limits(query, smoke=smoke, execution_profile=execution_profile)
    if retrieved_inputs:
        text_dense_hits, dense_manifest = retrieved_inputs.text_dense_hits, retrieved_inputs.dense_manifest
        text_sparse_hits, sparse_manifest = retrieved_inputs.text_sparse_hits, retrieved_inputs.sparse_manifest
        frame_hits, visual_manifest = retrieved_inputs.frame_hits, retrieved_inputs.visual_manifest
    else:
        text_dense_hits, dense_manifest = retrieve_text_dense(config, query, dense, smoke_limit_files=limits["smoke_limit_files"], smoke_top_k=limits["text_top_k"])
        text_sparse_hits, sparse_manifest = retrieve_text_sparse(config, query, sparse, smoke_limit_files=limits["smoke_limit_files"], smoke_top_k=limits["text_top_k"])
        frame_hits, visual_manifest = retrieve_visual_cls(config, query, tips, smoke_limit_files=limits["smoke_limit_files"], smoke_top_k=limits["visual_top_k"])
    text_hits = merge_text_hits(text_dense_hits, text_sparse_hits)
    text_hits, text_rerank_manifest = rerank_text_hits(query, text_hits, reranker, task_dir / "text_rerank", batch_size=limits["text_rerank_batch_size"], cap=limits["text_rerank_cap"], allow_cap_expansion=limits["allow_text_cap_expansion"])
    _release_text_runtimes_for_tips_patch(dense, sparse, reranker)
    frame_hits, patch_manifest = rerank_visual_patches(config, query, frame_hits, tips, event_store, task_dir / "visual_patch_rerank", batch_size=limits["patch_batch_size"], max_frames=limits["patch_max_frames"], patch_runtimes=tips_patch_workers)
    qa_evidence, qa_evidence_manifest = build_qa_evidence(config, query, text_hits, event_store, task_dir, keep_top=_qa_evidence_keep_top(query, smoke=smoke, execution_profile=execution_profile), visual_hits=frame_hits)
    packet_manifest = write_qa_evidence_packet(config, query, qa_evidence, task_dir)
    return {
        "task_id": query.task_id,
        "task_type": query.task_type,
        "execution_profile": _execution_profile_name(execution_profile),
        "dense_retrieval": dense_manifest,
        "sparse_retrieval": sparse_manifest,
        "text_rerank": text_rerank_manifest,
        "visual_retrieval": visual_manifest,
        "visual_patch_rerank": patch_manifest,
        "qa_evidence": qa_evidence_manifest,
        "qa_evidence_packet": packet_manifest,
    }


def answer_qa_packets(config: AppConfig, queries_path: Path, run_id: str, *, packet_paths: Sequence[Path], task_ids: Sequence[str] = ()) -> dict:
    root = run_output_root(config, run_id)
    packets = [Path(path) for path in packet_paths]
    if not packets:
        packets = sorted((root / "tasks" / "question_answering").glob("*/qa_evidence_packet.json"))
    selected_task_ids = set(task_ids)
    if selected_task_ids:
        packets = [packet for packet in packets if str(json.loads(packet.read_text(encoding="utf-8")).get("task_id") or "") in selected_task_ids]
    if not packets:
        raise ValueError("no QA evidence packets found for selected task ids" if selected_task_ids else "no QA evidence packets found")
    openrouter = make_openrouter_client(config)
    openrouter_report = openrouter_preflight(config, openrouter)
    openrouter_errors = _preflight_errors({"openrouter": openrouter_report})
    if openrouter_errors:
        failure_manifest = _load_manifest(root)
        failure_manifest["stages"].append(
            {
                "name": "qa_answer",
                "status": "failed",
                "finished_at_utc": now_utc(),
                "task_count": len(packets),
                "api_call_performed": False,
                "reason": "; ".join(openrouter_errors),
                "openrouter_preflight": openrouter_report,
            }
        )
        atomic_write_json(root / "run_manifest.json", failure_manifest)
        raise RuntimeError(f"OpenRouter readiness failed: {openrouter_errors}")
    reports = []
    reports_path = root / "diagnostics" / "qa_answer_reports.json"
    for packet in packets:
        task_id = str(json.loads(packet.read_text(encoding="utf-8"))["task_id"])
        task_dir = root / "tasks" / "question_answering" / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        try:
            _, manifest = answer_question_from_packet(config, packet, openrouter, output_dir=task_dir)
        except Exception as exc:
            failure = {"status": "failed", "task_id": task_id, "packet_path": packet.as_posix(), "error_type": type(exc).__name__, "error": str(exc)}
            reports.append(failure)
            atomic_write_json(reports_path, reports)
            existing_manifest = _load_manifest(root)
            existing_manifest["stages"].append(
                {
                    "name": "qa_answer",
                    "status": "failed",
                    "finished_at_utc": now_utc(),
                    "task_count": len(packets),
                    "completed_task_count": len([item for item in reports if item.get("status") == "completed"]),
                    "failed_task_count": len([item for item in reports if item.get("status") == "failed"]),
                    "api_call_performed": True,
                    "model": openrouter.model,
                    "reports_path": reports_path.as_posix(),
                }
            )
            atomic_write_json(root / "run_manifest.json", existing_manifest)
            raise
        reports.append(manifest)
    atomic_write_json(reports_path, reports)
    existing_manifest = _load_manifest(root)
    existing_manifest["stages"].append({"name": "qa_answer", "status": "completed", "finished_at_utc": now_utc(), "task_count": len(packets), "api_call_performed": True, "model": openrouter.model, "reports_path": reports_path.as_posix()})
    atomic_write_json(root / "run_manifest.json", existing_manifest)
    return {"qa_answer_reports": reports}


def _task_family_type(task_family: str | None) -> str | None:
    if task_family is None:
        return None
    mapping = {
        "eis": "event_instance_search",
        "ois": "object_instance_search",
        "qa": "question_answering",
    }
    if task_family not in mapping:
        raise ValueError(f"unknown task family: {task_family}")
    return mapping[task_family]


def _execution_profile_name(execution_profile: ExecutionProfile | None) -> str:
    return execution_profile.name if execution_profile else "official_or_legacy_smoke"


def _execution_limits(query: TaskQuery, *, smoke: bool, execution_profile: ExecutionProfile | None) -> dict[str, object]:
    if execution_profile:
        text_top_k = execution_profile.text_top_k_by_type.get(query.task_type)
        text_rerank_cap = execution_profile.text_rerank_cap_by_type.get(query.task_type)
        return {
            "smoke_limit_files": execution_profile.smoke_limit_files,
            "visual_top_k": execution_profile.visual_top_k,
            "text_top_k": text_top_k,
            "text_rerank_cap": text_rerank_cap,
            "patch_max_frames": execution_profile.patch_max_frames,
            "local_recovery_patch_max_frames": execution_profile.local_recovery_patch_max_frames,
            "text_rerank_batch_size": execution_profile.text_rerank_batch_size,
            "patch_batch_size": execution_profile.patch_batch_size,
            "allow_text_cap_expansion": execution_profile.allow_text_cap_expansion,
            "candidate_max_frame_hits": None,
            "candidate_max_text_hits": None,
        }
    if smoke:
        return {
            "smoke_limit_files": 1,
            "visual_top_k": 12,
            "text_top_k": 12,
            "text_rerank_cap": 12,
            "patch_max_frames": 8,
            "local_recovery_patch_max_frames": 32,
            "text_rerank_batch_size": 1,
            "patch_batch_size": 1,
            "allow_text_cap_expansion": False,
            "candidate_max_frame_hits": 50,
            "candidate_max_text_hits": 20,
        }
    profile = query.resolved_profiles["recall_profile"]["value"]
    return {
        "smoke_limit_files": None,
        "visual_top_k": None,
        "text_top_k": None,
        "text_rerank_cap": profile.get("text_rerank_input_cap"),
        "patch_max_frames": None,
        "local_recovery_patch_max_frames": None,
        "text_rerank_batch_size": 4,
        "patch_batch_size": 8,
        "allow_text_cap_expansion": True,
        "candidate_max_frame_hits": None,
        "candidate_max_text_hits": None,
    }


def _query_for_execution_profile(query: TaskQuery, execution_profile: ExecutionProfile | None) -> TaskQuery:
    if not execution_profile or not execution_profile.override_recall_profile:
        return query
    resolved_profiles = deepcopy(query.resolved_profiles)
    recall = resolved_profiles.setdefault("recall_profile", {}).setdefault("value", {})
    text_top_k = execution_profile.text_top_k_by_type.get(query.task_type)
    text_rerank_cap = execution_profile.text_rerank_cap_by_type.get(query.task_type)
    if execution_profile.visual_top_k is not None:
        recall["visual_top_k"] = int(execution_profile.visual_top_k)
        recall["visual_probe_top_k"] = int(execution_profile.visual_top_k)
    if text_top_k is not None:
        recall["text_dense_top_k"] = int(text_top_k)
        recall["text_sparse_top_k"] = int(text_top_k)
    if text_rerank_cap is not None:
        recall["text_rerank_input_cap"] = int(text_rerank_cap)
    if execution_profile.qa_evidence_keep_top is not None and query.task_type == "question_answering":
        recall["text_rerank_keep_top"] = int(execution_profile.qa_evidence_keep_top)
    recall["visual_reserve_per_day_source"] = int(execution_profile.visual_reserve_per_day_source)
    recall["visual_reserve_per_source_hour"] = int(execution_profile.visual_reserve_per_source_hour)
    recall["visual_probe_reserve_per_day_source"] = int(execution_profile.visual_probe_reserve_per_day_source)
    recall["visual_probe_reserve_per_source_hour"] = int(execution_profile.visual_probe_reserve_per_source_hour)
    recall["text_reserve_per_source"] = int(execution_profile.text_reserve_per_source)
    if execution_profile.local_recovery_candidate_cap is not None:
        recall["local_recovery_candidate_cap"] = int(execution_profile.local_recovery_candidate_cap)
    if execution_profile.local_recovery_max_frames_per_text_hit is not None:
        recall["local_recovery_max_frames_per_text_hit"] = int(execution_profile.local_recovery_max_frames_per_text_hit)
    resolved_profiles["execution_profile"] = _execution_profile_manifest(execution_profile, query)
    raw = deepcopy(query.raw)
    raw["_execution_profile"] = resolved_profiles["execution_profile"]
    return replace(query, raw=raw, resolved_profiles=resolved_profiles)


def _execution_profile_manifest(execution_profile: ExecutionProfile, query: TaskQuery) -> dict:
    return {
        "name": execution_profile.name,
        "diagnostic_only": True,
        "official_submission": False,
        "task_id": query.task_id,
        "task_type": query.task_type,
        "smoke_limit_files": execution_profile.smoke_limit_files,
        "visual_top_k": execution_profile.visual_top_k,
        "text_top_k": execution_profile.text_top_k_by_type.get(query.task_type),
        "text_rerank_cap": execution_profile.text_rerank_cap_by_type.get(query.task_type),
        "qa_evidence_keep_top": execution_profile.qa_evidence_keep_top if query.task_type == "question_answering" else None,
        "visual_reserves": {
            "reserve_per_day_source": execution_profile.visual_reserve_per_day_source,
            "reserve_per_source_hour": execution_profile.visual_reserve_per_source_hour,
            "probe_reserve_per_day_source": execution_profile.visual_probe_reserve_per_day_source,
            "probe_reserve_per_source_hour": execution_profile.visual_probe_reserve_per_source_hour,
        },
        "text_reserve_per_source": execution_profile.text_reserve_per_source,
        "local_recovery_candidate_cap": execution_profile.local_recovery_candidate_cap,
        "local_recovery_max_frames_per_text_hit": execution_profile.local_recovery_max_frames_per_text_hit,
        "patch_max_frames": execution_profile.patch_max_frames,
        "local_recovery_patch_max_frames": execution_profile.local_recovery_patch_max_frames,
        "ois_l2_rescan_expected": execution_profile.patch_max_frames is None,
    }


def _qa_evidence_keep_top(query: TaskQuery, *, smoke: bool, execution_profile: ExecutionProfile | None = None) -> int:
    if execution_profile and execution_profile.qa_evidence_keep_top is not None:
        return max(1, min(20, int(execution_profile.qa_evidence_keep_top)))
    if smoke:
        return 4
    profile = ((query.resolved_profiles or {}).get("recall_profile") or {}).get("value") or {}
    for key in ("text_rerank_keep_top", "text_rerank_input_cap"):
        value = profile.get(key)
        if value:
            return max(1, min(20, int(value)))
    return 12


def assign_model_devices(device_list: Sequence[str]) -> dict[str, str | list[str]]:
    if not device_list:
        raise ValueError("at least one device is required")
    dense_device = device_list[0]
    rerank_device = device_list[1] if len(device_list) >= 3 else device_list[0]
    tips_device = device_list[2] if len(device_list) >= 3 else device_list[0]
    if len(device_list) == 2:
        tips_device = device_list[1]
    sparse_devices = list(device_list) if len(device_list) >= 3 else [device_list[0]]
    return {
        "qwen3_dense": dense_device,
        "qwen3_reranker": rerank_device,
        "tipsv2": tips_device,
        "tipsv2_patch_workers": [tips_device] + [device for device in device_list if device != tips_device],
        "bgem3_sparse": sparse_devices,
    }


def _batch_retrieve_task_inputs(
    config: AppConfig,
    queries: Sequence[TaskQuery],
    tips: TipsV2Runtime,
    dense: Qwen3DenseRuntime,
    sparse: BGEM3SparseRuntime,
    *,
    smoke: bool,
    execution_profile: ExecutionProfile | None,
) -> dict[str, RetrievedTaskInputs]:
    if not queries:
        return {}
    limits_by_task = {query.task_id: _execution_limits(query, smoke=smoke, execution_profile=execution_profile) for query in queries}
    smoke_limit_files = _shared_smoke_limit(limits_by_task)
    visual_top_k_by_task = {task_id: limits["visual_top_k"] for task_id, limits in limits_by_task.items()}
    text_top_k_by_task = {task_id: limits["text_top_k"] for task_id, limits in limits_by_task.items()}
    dense_results = retrieve_text_dense_batch(config, queries, dense, smoke_limit_files=smoke_limit_files, smoke_top_k_by_task=text_top_k_by_task)
    sparse_results = retrieve_text_sparse_batch(config, queries, sparse, smoke_limit_files=smoke_limit_files, smoke_top_k_by_task=text_top_k_by_task)
    visual_results = retrieve_visual_cls_batch(config, queries, tips, smoke_limit_files=smoke_limit_files, smoke_top_k_by_task=visual_top_k_by_task)
    retrieved: dict[str, RetrievedTaskInputs] = {}
    for query in queries:
        dense_hits, dense_manifest = dense_results[query.task_id]
        sparse_hits, sparse_manifest = sparse_results[query.task_id]
        frame_hits, visual_manifest = visual_results[query.task_id]
        retrieved[query.task_id] = RetrievedTaskInputs(
            text_dense_hits=dense_hits,
            dense_manifest=dense_manifest,
            text_sparse_hits=sparse_hits,
            sparse_manifest=sparse_manifest,
            frame_hits=frame_hits,
            visual_manifest=visual_manifest,
        )
    return retrieved


def _shared_smoke_limit(limits_by_task: Mapping[str, Mapping[str, object]]) -> int | None:
    values = {limits["smoke_limit_files"] for limits in limits_by_task.values()}
    if len(values) != 1:
        raise ValueError(f"batch retrieval requires one shared smoke_limit_files value, got {sorted(str(value) for value in values)}")
    return next(iter(values))  # type: ignore[return-value]


def _batch_retrieval_manifest(retrieved: Mapping[str, RetrievedTaskInputs]) -> dict:
    if not retrieved:
        return {"enabled": True, "task_count": 0, "channels": {}}

    def channel_summary(manifest_getter) -> dict:
        manifests = [manifest_getter(item) for item in retrieved.values()]
        return {
            "batch_retrieval": all(bool(manifest.get("batch_retrieval")) for manifest in manifests),
            "batch_task_counts": sorted({int(manifest.get("batch_task_count") or 0) for manifest in manifests}),
            "top_k_scopes": sorted({str(manifest.get("top_k_scope") or "") for manifest in manifests}),
            "file_limits": sorted({str(manifest.get("file_limit")) for manifest in manifests}),
            "parquet_files": sorted({int(manifest.get("parquet_files") or 0) for manifest in manifests}),
            "task_count": len(manifests),
        }

    return {
        "enabled": True,
        "task_count": len(retrieved),
        "channels": {
            "text_dense": channel_summary(lambda item: item.dense_manifest),
            "text_sparse": channel_summary(lambda item: item.sparse_manifest),
            "visual_cls": channel_summary(lambda item: item.visual_manifest),
        },
    }


def _failed_task_report(query: TaskQuery, task_dir: Path, exc: Exception, *, execution_profile: ExecutionProfile | None) -> dict:
    return {
        "task_id": query.task_id,
        "task_type": query.task_type,
        "status": "failed",
        "execution_profile": _execution_profile_name(execution_profile),
        "task_dir": task_dir.as_posix(),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "completed_marker_written": False,
    }


def _build_tips_patch_workers(config: AppConfig, cache_dir: Path, model_devices: Mapping[str, object], primary: TipsV2Runtime) -> list[TipsV2Runtime]:
    primary_device = getattr(primary, "device", None)
    if primary_device is None:
        return [primary]
    devices = [str(device) for device in model_devices.get("tipsv2_patch_workers", [primary_device])]  # type: ignore[arg-type]
    workers: list[TipsV2Runtime] = []
    seen = set()
    for index, device in enumerate(devices):
        if not device or device in seen:
            continue
        seen.add(device)
        if device == primary_device:
            workers.append(primary)
        else:
            safe_device = device.replace(":", "_").replace("/", "_")
            workers.append(TipsV2Runtime(config.model_profile["tipsv2_model_root"], device=device, cache_dir=cache_dir / f"tipsv2_patch_{index}_{safe_device}"))
    return workers or [primary]


def _release_text_runtimes_for_tips_patch(*runtimes: object) -> None:
    for runtime in runtimes:
        release_runtime_state(runtime)


def _should_bypass_visual_patch_rerank(query: TaskQuery) -> bool:
    return query.task_id in CLS_VISUAL_SOURCE_BYPASS_TASK_IDS


def _rerank_visual_patches_for_task(
    config: AppConfig,
    query: TaskQuery,
    frame_hits: Sequence[FrameHit],
    tips: TipsV2Runtime,
    event_store: EventStore,
    output_dir: Path,
    *,
    batch_size: int = 8,
    max_frames: int | None = None,
    cache_path: Path | None = None,
    patch_runtimes: Sequence[TipsV2Runtime] | None = None,
) -> tuple[list[FrameHit], dict]:
    if _should_bypass_visual_patch_rerank(query):
        return bypass_visual_patch_rerank_with_cls(query, frame_hits, event_store, output_dir, max_frames=max_frames)
    return rerank_visual_patches(
        config,
        query,
        frame_hits,
        tips,
        event_store,
        output_dir,
        batch_size=batch_size,
        max_frames=max_frames,
        cache_path=cache_path,
        patch_runtimes=patch_runtimes,
    )


def _run_text_localized_visual_recovery(
    config: AppConfig,
    query: TaskQuery,
    text_hits: Sequence[TextHit],
    event_store: EventStore,
    tips: TipsV2Runtime,
    task_dir: Path,
    *,
    smoke: bool,
    max_frames: int | None = None,
    tips_patch_workers: Sequence[TipsV2Runtime] | None = None,
) -> tuple[list[FrameHit], dict]:
    if query.task_type not in {"event_instance_search", "object_instance_search"}:
        return [], {"status": "not_applicable", "reason": "task_family_without_interval_output", "candidate_frame_count": 0}
    candidates, candidate_manifest = _text_localized_visual_recovery_candidates(query, text_hits, event_store, smoke=smoke)
    output_dir = task_dir / "visual_patch_rerank" / "local_recovery"
    if not candidates:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "status": "skipped_no_text_events",
            "task_id": query.task_id,
            "policy": "text_localized_visual_recovery_v1",
            **candidate_manifest,
        }
        atomic_write_json(output_dir / "stage_manifest.json", manifest)
        return [], manifest
    effective_max_frames = max_frames
    if smoke and effective_max_frames is None:
        effective_max_frames = 32
    reranked, patch_manifest = _rerank_visual_patches_for_task(
        config,
        query,
        candidates,
        tips,
        event_store,
        output_dir,
        batch_size=1 if smoke else 8,
        max_frames=effective_max_frames,
        cache_path=task_dir / "visual_patch_rerank" / "patch_score_cache.jsonl",
        patch_runtimes=tips_patch_workers,
    )
    manifest = {
        "status": patch_manifest.get("status"),
        "task_id": query.task_id,
        "policy": "text_localized_visual_recovery_v1",
        "candidate_generation": candidate_manifest,
        "patch_rerank": patch_manifest,
        "candidate_frame_count": len(candidates),
        "scored_frame_count": int(patch_manifest.get("scored_count") or 0),
        "max_frames": effective_max_frames,
        "output_dir": output_dir.as_posix(),
    }
    atomic_write_json(output_dir / "local_recovery_manifest.json", manifest)
    return reranked, manifest


def _text_localized_visual_recovery_candidates(query: TaskQuery, text_hits: Sequence[TextHit], event_store: EventStore, *, smoke: bool) -> tuple[list[FrameHit], dict]:
    recall = ((query.resolved_profiles or {}).get("recall_profile") or {}).get("value") or {}
    max_text_hits = int(recall.get("local_recovery_max_text_hits") or 20)
    cap = int(recall.get("local_recovery_candidate_cap") or 1000)
    max_frames_per_text_hit = int(recall.get("local_recovery_max_frames_per_text_hit") or 0)
    selected_hits = sorted(
        text_hits,
        key=lambda hit: (-(hit.reranker_score if hit.reranker_score is not None else hit.dense_score if hit.dense_score is not None else 0.0), hit.embedding_chunk_id),
    )[:max_text_hits]
    candidates: list[FrameHit] = []
    seen_frame_keys = set()
    visited_l2_events = set()
    candidate_count_before_cap = 0
    deduped_within_l2_count = 0
    cap_applied = False
    per_text_cap_hit_count = 0
    for hit in selected_hits:
        text_frame_count = 0
        text_atom = event_store.text_by_chunk.get(hit.embedding_chunk_id)
        text_atom_id = str((text_atom or {}).get("text_atom_id") or hit.text_atom_id or "")
        if not text_atom_id:
            continue
        hit.text_atom_id = text_atom_id
        for link in event_store.text_links_by_text.get(text_atom_id, []):
            if link.get("link_type") != "text_overlap_support":
                continue
            event_id = str(link.get("event_id") or "")
            if not event_id.startswith("l2::") or event_id in visited_l2_events:
                continue
            event = event_store.event_by_id.get(event_id)
            if not event or event.get("source") is None:
                continue
            visited_l2_events.add(event_id)
            frames = _l2_key_frames(event, event_store)
            l2_seen_keys = set()
            for role, frame in frames:
                key = _frame_stable_key(frame)
                if not key:
                    continue
                if key in l2_seen_keys:
                    deduped_within_l2_count += 1
                    continue
                l2_seen_keys.add(key)
                if key in seen_frame_keys:
                    deduped_within_l2_count += 1
                    continue
                if max_frames_per_text_hit > 0 and text_frame_count >= max_frames_per_text_hit:
                    per_text_cap_hit_count += 1
                    break
                candidate_count_before_cap += 1
                if len(candidates) >= cap:
                    cap_applied = True
                    break
                seen_frame_keys.add(key)
                embedding_id = str(frame.get("tipsv2_embedding_id") or "")
                second = float(frame.get("absolute_second") or -1.0)
                candidates.append(
                    FrameHit(
                        task_id=query.task_id,
                        prompt=query.canonical_text_for_retrieval,
                        embedding_id=embedding_id,
                        day=int(frame.get("day_num") or event.get("day_num") or hit.day),
                        source=str(frame.get("source") or event.get("source") or hit.source),
                        absolute_second=second,
                        frame_rel_path=str(frame.get("frame_rel_path") or ""),
                        image_path_ref=str(frame.get("image_path_ref") or frame.get("frame_rel_path") or ""),
                        cls_score=0.0,
                        frame_atom_id=str(frame.get("frame_atom_id") or "") or None,
                        provenance={
                            "candidate_record_id": f"local-recovery:{event_id}:{frame.get('frame_atom_id') or embedding_id}",
                            "local_visual_recovery": {
                                "policy": "text_localized_visual_recovery_v1",
                                "seed_text_atom_id": text_atom_id,
                                "seed_embedding_chunk_id": hit.embedding_chunk_id,
                                "seed_text_score": _primary_text_hit_score(hit),
                                "seed_text_score_source": _primary_text_hit_score_source(hit),
                                "seed_negative_text_score": hit.negative_reranker_score,
                                "seed_text_excerpt": _text_hit_excerpt(hit),
                                "seed_event_id": event_id,
                                "key_frame_role": role,
                                "event_start_second": float(event.get("start_second") or 0.0),
                                "event_end_second": float(event.get("end_second") or 0.0),
                            },
                        },
                    )
                )
                text_frame_count += 1
            if cap_applied:
                break
            if max_frames_per_text_hit > 0 and text_frame_count >= max_frames_per_text_hit:
                break
        if cap_applied:
            break
    return candidates, {
        "policy": "text_localized_visual_recovery_v1",
        "selected_text_hit_count": len(selected_hits),
        "visited_text_event_count": len(visited_l2_events),
        "linked_l2_event_count": len(visited_l2_events),
        "candidate_frame_count_before_cap": candidate_count_before_cap,
        "candidate_frame_count": len(candidates),
        "frames_per_l2_policy": "first_canonical_last_canonical_first_representative",
        "deduped_within_l2_count": deduped_within_l2_count,
        "cap": cap,
        "cap_applied": cap_applied,
        "max_frames_per_text_hit": max_frames_per_text_hit or None,
        "per_text_cap_hit_count": per_text_cap_hit_count,
        "representative_frame_policy": "first_valid_event_representative_frame_id",
        "max_text_hits": max_text_hits,
    }


def _primary_text_hit_score(hit: TextHit) -> float | None:
    for value in (hit.reranker_score, hit.dense_score, hit.sparse_score):
        if value is not None:
            return float(value)
    return None


def _primary_text_hit_score_source(hit: TextHit) -> str | None:
    if hit.reranker_score is not None:
        return "reranker_score"
    if hit.dense_score is not None:
        return "dense_score"
    if hit.sparse_score is not None:
        return "sparse_score"
    return None


def _text_hit_excerpt(hit: TextHit, *, max_chars: int = 4000) -> str:
    text = str(hit.text_for_display or hit.text_for_embedding or "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _frame_stable_key(frame: Mapping[str, object]) -> str:
    frame_atom_id = str(frame.get("frame_atom_id") or "")
    if frame_atom_id:
        return f"frame_atom_id:{frame_atom_id}"
    embedding_id = str(frame.get("tipsv2_embedding_id") or "")
    return f"embedding_id:{embedding_id}" if embedding_id else ""


def _l2_key_frames(event: Mapping[str, object], event_store: EventStore) -> list[tuple[str, Mapping[str, object]]]:
    day = int(event.get("day_num") or 0)
    source = str(event.get("source") or "")
    start = float(event.get("start_second") or 0.0)
    end = float(event.get("end_second") or start)
    frames = [
        frame
        for frame in event_store.frames_by_day_source.get((day, source), [])
        if bool(frame.get("tipsv2_embedding_id"))
        and frame.get("is_canonical_frame", True) is not False
        and start <= float(frame.get("absolute_second") or -1.0) < end
    ]
    result: list[tuple[str, Mapping[str, object]]] = []
    if frames:
        result.append(("l2_first_canonical_tips_frame", frames[0]))
        result.append(("l2_last_canonical_tips_frame", frames[-1]))
    for frame_id in event.get("representative_frame_ids") or []:
        frame = event_store.frame_by_id.get(str(frame_id))
        if frame and frame.get("tipsv2_embedding_id"):
            result.append(("l2_representative_frame_id", frame))
            break
    return result


def _merge_frame_hits(primary: Sequence[FrameHit], recovery: Sequence[FrameHit]) -> list[FrameHit]:
    by_embedding: dict[str, FrameHit] = {}
    for hit in list(primary) + list(recovery):
        current = by_embedding.get(hit.embedding_id)
        if current is None:
            by_embedding[hit.embedding_id] = hit
            continue
        current_score = current.patch_score if current.patch_score is not None else -1e9
        hit_score = hit.patch_score if hit.patch_score is not None else -1e9
        if hit_score > current_score:
            by_embedding[hit.embedding_id] = hit
        else:
            current.provenance.setdefault("merged_duplicate_sources", []).append(hit.provenance)
    return sorted(by_embedding.values(), key=lambda item: (-(item.patch_score if item.patch_score is not None else item.cls_score), item.embedding_id))


def run_one_task(
    config: AppConfig,
    query: TaskQuery,
    event_store: EventStore,
    coverage,
    tips: TipsV2Runtime,
    dense: Qwen3DenseRuntime,
    sparse: BGEM3SparseRuntime,
    reranker: Qwen3RerankerRuntime,
    openrouter: OpenRouterClient,
    task_dir: Path,
    *,
    smoke: bool,
    execution_profile: ExecutionProfile | None = None,
    retrieved_inputs: RetrievedTaskInputs | None = None,
    tips_patch_workers: Sequence[TipsV2Runtime] | None = None,
) -> dict:
    query = _query_for_execution_profile(query, execution_profile)
    limits = _execution_limits(query, smoke=smoke, execution_profile=execution_profile)
    if retrieved_inputs:
        text_dense_hits, dense_manifest = retrieved_inputs.text_dense_hits, retrieved_inputs.dense_manifest
        text_sparse_hits, sparse_manifest = retrieved_inputs.text_sparse_hits, retrieved_inputs.sparse_manifest
        frame_hits, visual_manifest = retrieved_inputs.frame_hits, retrieved_inputs.visual_manifest
    else:
        text_dense_hits, dense_manifest = retrieve_text_dense(config, query, dense, smoke_limit_files=limits["smoke_limit_files"], smoke_top_k=limits["text_top_k"])
        text_sparse_hits, sparse_manifest = retrieve_text_sparse(config, query, sparse, smoke_limit_files=limits["smoke_limit_files"], smoke_top_k=limits["text_top_k"])
        frame_hits, visual_manifest = retrieve_visual_cls(config, query, tips, smoke_limit_files=limits["smoke_limit_files"], smoke_top_k=limits["visual_top_k"])
    text_hits = merge_text_hits(text_dense_hits, text_sparse_hits)
    text_hits, text_rerank_manifest = rerank_text_hits(query, text_hits, reranker, task_dir / "text_rerank", batch_size=limits["text_rerank_batch_size"], cap=limits["text_rerank_cap"], allow_cap_expansion=limits["allow_text_cap_expansion"])
    _release_text_runtimes_for_tips_patch(dense, sparse, reranker)
    report = {"task_id": query.task_id, "task_type": query.task_type, "execution_profile": _execution_profile_name(execution_profile), "dense_retrieval": dense_manifest, "sparse_retrieval": sparse_manifest, "text_rerank": text_rerank_manifest}
    if query.task_type == "question_answering":
        frame_hits, patch_manifest = rerank_visual_patches(config, query, frame_hits, tips, event_store, task_dir / "visual_patch_rerank", batch_size=limits["patch_batch_size"], max_frames=limits["patch_max_frames"], patch_runtimes=tips_patch_workers)
        qa_evidence, qa_evidence_manifest = build_qa_evidence(config, query, text_hits, event_store, task_dir, keep_top=_qa_evidence_keep_top(query, smoke=smoke, execution_profile=execution_profile), visual_hits=frame_hits)
        answer, qa_manifest = answer_question(config, query, qa_evidence, openrouter, task_dir)
        report.update({"visual_retrieval": visual_manifest, "visual_patch_rerank": patch_manifest, "qa_evidence": qa_evidence_manifest, "qa_mllm": qa_manifest, "qa_result": answer})
        return report
    frame_hits, patch_manifest = _rerank_visual_patches_for_task(config, query, frame_hits, tips, event_store, task_dir / "visual_patch_rerank", batch_size=limits["patch_batch_size"], max_frames=limits["patch_max_frames"], patch_runtimes=tips_patch_workers)
    local_recovery_hits, local_recovery_manifest = _run_text_localized_visual_recovery(config, query, text_hits, event_store, tips, task_dir, smoke=smoke, max_frames=limits["local_recovery_patch_max_frames"], tips_patch_workers=tips_patch_workers)
    frame_hits = _merge_frame_hits(frame_hits, local_recovery_hits)
    candidates = build_candidates(query, frame_hits, text_hits, event_store, max_frame_hits=limits["candidate_max_frame_hits"] or len(frame_hits), max_text_hits=limits["candidate_max_text_hits"] or len(text_hits))
    rows, evidence, solver_report = solve_interval_task(
        config,
        query,
        candidates,
        coverage,
        task_dir,
        stage_manifests={"text_rerank": text_rerank_manifest, "visual_patch_rerank": patch_manifest, "local_visual_recovery": local_recovery_manifest},
        event_store=event_store,
    )
    report.update({"visual_retrieval": visual_manifest, "patch_rerank": patch_manifest, "local_visual_recovery": local_recovery_manifest, "solver": solver_report, "row_count": len(rows), "evidence_bundles": evidence})
    return report


def full_smoke_export(
    config: AppConfig,
    queries_path: Path,
    run_id: str,
    *,
    devices: str,
    fetch_official: bool = True,
    resume: bool = False,
    all_queries: bool = False,
    task_ids: Sequence[str] = (),
) -> dict:
    selected_task_ids, interval_task_ids, qa_task_ids = _full_smoke_task_selection(all_queries=all_queries, task_ids=task_ids)
    interval = None
    if interval_task_ids:
        interval = run_tasks(
            config,
            queries_path,
            run_id,
            task_ids=interval_task_ids,
            devices=devices,
            smoke=False,
            fetch_official=fetch_official,
            resume=resume,
            execution_profile=FULLFLOW_SMOKE_TOP100_PROFILE,
        )
    qa = None
    if qa_task_ids:
        qa = export_qa_packets(
            config,
            queries_path,
            run_id,
            task_ids=qa_task_ids,
            devices=devices,
            smoke=False,
            fetch_official=fetch_official,
            resume=resume,
            execution_profile=FULLFLOW_SMOKE_TOP100_PROFILE,
        )
    root = run_output_root(config, run_id)
    manifest = _load_manifest(root)
    manifest["fullflow_smoke"] = {
        "profile": FULLFLOW_SMOKE_PROFILE_NAME,
        "official_submission": False,
        "selected_task_ids": list(selected_task_ids),
        "all_queries": _full_smoke_selection_is_all(selected_task_ids),
        "server_export_only": True,
        "qa_api_call_performed": False,
    }
    manifest["stages"].append(
        {
            "name": "full_smoke_export",
            "status": "completed",
            "finished_at_utc": now_utc(),
            "execution_profile": FULLFLOW_SMOKE_PROFILE_NAME,
            "selected_task_ids": list(selected_task_ids),
            "all_queries": _full_smoke_selection_is_all(selected_task_ids),
            "qa_api_call_performed": False,
        }
    )
    atomic_write_json(root / "run_manifest.json", manifest)
    return {"interval": interval, "qa_export": qa, "selected_task_ids": list(selected_task_ids), "all_queries": _full_smoke_selection_is_all(selected_task_ids)}


def full_smoke_package(config: AppConfig, queries_path: Path, run_id: str, *, all_queries: bool = False, task_ids: Sequence[str] = ()) -> dict:
    root = run_output_root(config, run_id)
    coverage = build_coverage_index(config, root / "registry")
    event_store = EventStore.from_config(config)
    loaded_manifest = _load_manifest(root)
    selected_task_ids = _full_smoke_selected_task_ids_from_manifest(loaded_manifest, all_queries=all_queries, task_ids=task_ids)
    selected_ids = set(selected_task_ids)
    eis_rows = [row for row in _read_task_rows(root / "tasks" / "event_instance_search") if str(row.get("task_id") or row.get("task")) in selected_ids]
    ois_rows = [row for row in _read_task_rows(root / "tasks" / "object_instance_search") if str(row.get("task_id") or row.get("task")) in selected_ids]
    evidence = [item for item in (_read_evidence(root / "tasks" / "event_instance_search") + _read_evidence(root / "tasks" / "object_instance_search")) if item.task_id in selected_ids]
    qa_results = [item for item in _read_qa_results(root / "tasks" / "question_answering") if item.task_id in selected_ids]
    qa_evidence_by_task = {task_id: rows for task_id, rows in _read_qa_evidence_by_task(root / "tasks" / "question_answering").items() if task_id in selected_ids}
    manifest = _without_previous_package_stages(loaded_manifest, names={"merge_and_package", "full_smoke_package"})
    diagnostics = _write_run_diagnostics(root)
    manifest["score_calibration"] = diagnostics["score_calibration"]
    manifest["tail_audit"] = diagnostics["tail_audit"]
    manifest["diagnostics"] = diagnostics["files"]
    manifest["task_artifacts"] = _collect_task_artifacts(root)
    manifest["fullflow_smoke"] = {
        "profile": FULLFLOW_SMOKE_PROFILE_NAME,
        "official_submission": False,
        "selected_task_ids": list(selected_task_ids),
        "all_queries": set(selected_task_ids) == set(FULLFLOW_SMOKE_ALL_TASK_IDS),
        "package_kind": "diagnostic_fullflow_smoke",
    }
    atomic_write_json(root / "run_manifest.json", manifest)
    validation = write_diagnostic_smoke_bundle(
        root,
        eis_rows=eis_rows,
        ois_rows=ois_rows,
        qa_results=qa_results,
        evidence=evidence,
        coverage=coverage,
        run_id=run_id,
        manifest=manifest,
        selected_task_ids=selected_task_ids,
        event_store=event_store,
        qa_evidence_by_task=qa_evidence_by_task,
        sort_interval_rows_by_final_score=config.sort_interval_rows_by_final_score,
    )
    manifest["fullflow_smoke"]["validation"] = {
        "valid": validation.get("valid"),
        "error_count": validation.get("error_count"),
        "zip": validation.get("zip"),
    }
    manifest["stages"].append(
        {
            "name": "full_smoke_package",
            "status": "completed" if validation["valid"] else "failed",
            "finished_at_utc": now_utc(),
            "execution_profile": FULLFLOW_SMOKE_PROFILE_NAME,
            "selected_task_ids": list(selected_task_ids),
            "all_queries": _full_smoke_selection_is_all(selected_task_ids),
            "package_kind": "diagnostic_fullflow_smoke",
            "official_submission": False,
        }
    )
    atomic_write_json(root / "run_manifest.json", manifest)
    return validation


def _full_smoke_task_selection(*, all_queries: bool, task_ids: Sequence[str] = ()) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if task_ids and all_queries:
        raise ValueError("--task-id cannot be combined with full-smoke all-queries selection")
    if task_ids:
        selected = _normalize_full_smoke_task_ids(task_ids)
        interval = tuple(task_id for task_id in selected if task_id in FULLFLOW_SMOKE_ALL_INTERVAL_TASK_IDS)
        qa = tuple(task_id for task_id in selected if task_id in FULLFLOW_SMOKE_ALL_QA_TASK_IDS)
        return selected, interval, qa
    if all_queries:
        return FULLFLOW_SMOKE_ALL_TASK_IDS, FULLFLOW_SMOKE_ALL_INTERVAL_TASK_IDS, FULLFLOW_SMOKE_ALL_QA_TASK_IDS
    return FULLFLOW_SMOKE_TASK_IDS, FULLFLOW_SMOKE_INTERVAL_TASK_IDS, FULLFLOW_SMOKE_QA_TASK_IDS


def _full_smoke_selected_task_ids_from_manifest(manifest: dict, *, all_queries: bool, task_ids: Sequence[str] = ()) -> tuple[str, ...]:
    if task_ids:
        return _full_smoke_task_selection(all_queries=False, task_ids=task_ids)[0]
    if all_queries:
        return FULLFLOW_SMOKE_ALL_TASK_IDS
    selected = ((manifest.get("fullflow_smoke") or {}).get("selected_task_ids") or []) if isinstance(manifest, dict) else []
    if selected:
        return _normalize_full_smoke_task_ids(selected)
    return FULLFLOW_SMOKE_TASK_IDS


def _normalize_full_smoke_task_ids(task_ids: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(str(task_id).strip() for task_id in task_ids if str(task_id).strip()))
    valid = set(FULLFLOW_SMOKE_ALL_TASK_IDS)
    unknown = [task_id for task_id in selected if task_id not in valid]
    if unknown:
        raise ValueError(f"unknown full-smoke query id(s): {', '.join(unknown)}")
    if not selected:
        raise ValueError("no full-smoke query ids selected")
    return selected


def _full_smoke_selection_is_all(task_ids: Sequence[str]) -> bool:
    return set(task_ids) == set(FULLFLOW_SMOKE_ALL_TASK_IDS)


def merge_and_package(config: AppConfig, queries_path: Path, run_id: str, *, write_zip: bool) -> dict:
    root = run_output_root(config, run_id)
    coverage = build_coverage_index(config, root / "registry")
    event_store = EventStore.from_config(config)
    eis_rows = _read_task_rows(root / "tasks" / "event_instance_search")
    ois_rows = _read_task_rows(root / "tasks" / "object_instance_search")
    evidence = _read_evidence(root / "tasks" / "event_instance_search") + _read_evidence(root / "tasks" / "object_instance_search")
    qa_results = _read_qa_results(root / "tasks" / "question_answering")
    qa_evidence_by_task = _read_qa_evidence_by_task(root / "tasks" / "question_answering")
    manifest = _without_previous_merge_stages(_load_manifest(root))
    diagnostics = _write_run_diagnostics(root)
    manifest["score_calibration"] = diagnostics["score_calibration"]
    manifest["tail_audit"] = diagnostics["tail_audit"]
    manifest["diagnostics"] = diagnostics["files"]
    manifest["task_artifacts"] = _collect_task_artifacts(root)
    atomic_write_json(root / "run_manifest.json", manifest)
    validation = write_submission_bundle(
        root,
        eis_rows=eis_rows,
        ois_rows=ois_rows,
        qa_results=qa_results,
        evidence=evidence,
        coverage=coverage,
        run_id=run_id,
        write_zip=write_zip,
        manifest=manifest,
        event_store=event_store,
        qa_evidence_by_task=qa_evidence_by_task,
        sort_interval_rows_by_final_score=config.sort_interval_rows_by_final_score,
    )
    _write_root_per_task_report(root, validation)
    manifest["outputs"] = validation.get("files", {})
    manifest["validation"] = validation
    manifest["stages"].append({"name": "merge_and_package", "status": "completed" if validation["valid"] else "failed", "finished_at_utc": now_utc()})
    atomic_write_json(root / "run_manifest.json", manifest)
    validation["openreview_package"] = write_openreview_package_checklist(root, run_id=run_id, validation=validation, manifest=manifest)
    atomic_write_json(root / "submissions" / "validation_report.json", validation)
    manifest["validation"] = validation
    atomic_write_json(root / "run_manifest.json", manifest)
    return validation


def make_openrouter_client(config: AppConfig) -> OpenRouterClient:
    qa = config.data["qa_mllm"]
    import os

    model = os.environ.get("OPENROUTER_MODEL_OVERRIDE") or qa["model"]
    return OpenRouterClient(endpoint=qa["endpoint"], model=model, api_key_env=qa["api_key_env"], referer_env=qa["http_referer_env"], title=qa["openrouter_title"], timeout_sec=int(qa.get("request_timeout_sec", 90)), max_retries=int(qa.get("max_retries", 2)))


def _preflight_errors(model_report: dict, *, require_chat_ping: bool = True) -> list[str]:
    errors: list[str] = []
    openrouter = model_report.get("openrouter") if isinstance(model_report, dict) else None
    if isinstance(openrouter, dict):
        if require_chat_ping and not openrouter.get("api_key_present"):
            errors.append("OPENROUTER_API_KEY is not set")
        if not openrouter.get("model_found"):
            errors.append("OpenRouter model metadata lookup failed")
        if not openrouter.get("required_supported"):
            errors.append("OpenRouter model does not support required structured-output parameters")
        ping = openrouter.get("chat_completion_ping")
        if require_chat_ping and isinstance(ping, dict) and ping.get("status") != "completed":
            errors.append("OpenRouter chat completion schema ping failed")
    return errors


def base_manifest(config: AppConfig, run_id: str) -> dict:
    import os

    configured_qa_model = config.data["qa_mllm"]["model"]
    override_qa_model = os.environ.get("OPENROUTER_MODEL_OVERRIDE")
    qa_model = override_qa_model or configured_qa_model
    runtime_manifest = _runtime_manifest(config)
    target_runtime = runtime_manifest["official_runtime_target"]
    actual_runtime = runtime_manifest["actual_runtime"]
    return {
        "run_id": run_id,
        "created_at_utc": now_utc(),
        "repo_root": config.repo_root.as_posix(),
        "config_path": config.path.as_posix(),
        "config_sha256": sha256_file(config.path),
        "profile_registry_sha256": config.profile_registry_sha256,
        "mode": config.mode,
        "active_environment_profile": config.active_environment_profile or config.data.get("active_environment_profile"),
        "active_model_profile": config.active_environment_profile or config.data.get("active_model_profile") or config.data.get("active_environment_profile"),
        "model_roots": _model_roots(config.model_profile),
        "allow_qdrant": False,
        "enable_patch_rerank": True,
        "enable_qwen3_text_rerank": True,
        "enable_openrouter_qa_mllm": True,
        "target_server_model_execution": target_runtime["server_model_execution"],
        "target_tipsv2_patch_runtime": target_runtime["tipsv2_patch_runtime"],
        "target_qwen3_reranker_runtime": target_runtime["qwen3_reranker_runtime"],
        "target_server_filesystem_model_api_inference": target_runtime["server_filesystem_model_api_inference"],
        "server_model_execution": actual_runtime["model_execution"],
        "tipsv2_patch_runtime": actual_runtime["tipsv2_patch_runtime"],
        "qwen3_reranker_runtime": actual_runtime["qwen3_reranker_runtime"],
        "server_filesystem_model_api_inference": actual_runtime["server_filesystem_model_api_inference"],
        "actual_model_execution": actual_runtime["model_execution"],
        "actual_runtime": actual_runtime,
        "official_runtime_target": target_runtime,
        "qa_mllm_api_inference_used": True,
        "qa_mllm_provider": "openrouter",
        "qa_mllm_model": qa_model,
        "qa_mllm_config_model": configured_qa_model,
        "qa_mllm_model_override_used": bool(override_qa_model and override_qa_model != configured_qa_model),
        "qa_mllm_official_required_model": OFFICIAL_QA_MODEL,
        "qa_mllm_api_key_source": "env:OPENROUTER_API_KEY",
        "official_sources": {
            "challenge_url": config.data["official_sources"]["challenge_url"],
            "dataset_card_url": config.data["official_sources"].get("dataset_card_url"),
            "openrouter_chat_completions_docs": config.data["official_sources"].get("openrouter_chat_completions_docs"),
            "openreview_group_url": "https://openreview.net/group?id=acmmm.org/ACMMM/2026/Workshop/CASTLE",
            "fully_automated_submission_paper_deadline": "2026-07-05",
        },
        "python": platform.python_version(),
        "package_versions": package_versions(),
        "stages": [],
    }


def _runtime_manifest(config: AppConfig) -> dict:
    target = {
        "active_environment_profile": "server_linux",
        "server_model_execution": "server_filesystem_weights",
        "tipsv2_patch_runtime": "server_native_patch_tokens",
        "qwen3_reranker_runtime": "server_filesystem_weights",
        "server_filesystem_model_api_inference": {"used": False, "field_scope": "server_filesystem_models_only"},
    }
    active = config.active_environment_profile or str(config.data.get("active_environment_profile") or "")
    if active == "server_linux":
        actual = {
            "active_environment_profile": active,
            "model_execution": "server_filesystem_weights",
            "tipsv2_patch_runtime": "server_native_patch_tokens",
            "qwen3_reranker_runtime": "server_filesystem_weights",
            "server_filesystem_model_api_inference": {"used": False, "field_scope": "server_filesystem_models_only"},
            "runtime_claim_scope": "server_profile_selected; model_preflight status is recorded under preflight.model_preflight",
        }
    else:
        actual = {
            "active_environment_profile": active,
            "model_execution": "not_run",
            "tipsv2_patch_runtime": "not_run",
            "qwen3_reranker_runtime": "not_run",
            "server_filesystem_model_api_inference": {"used": False, "field_scope": "not_applicable_local_preflight"},
            "runtime_claim_scope": "local_preflight_only; server filesystem model runtime not executed",
        }
    return {"official_runtime_target": target, "actual_runtime": actual}


def _validate_day_proof(config: AppConfig) -> dict:
    rows = {}
    for day, item in (config.data.get("day_date_map") or {}).items():
        proofs = item.get("proof") or []
        proof_rows = []
        for proof in proofs:
            path = config.repo_root / str(proof)
            date = str(item.get("date")).replace("-", "")
            proof_rows.append({"path": path.as_posix(), "exists": path.exists(), "date_in_filename": date in path.name})
        rows[str(day)] = {"date": str(item.get("date")), "weekday": item.get("weekday"), "proof": proof_rows, "valid": all(row["exists"] and row["date_in_filename"] for row in proof_rows)}
    return {"valid": all(row["valid"] for row in rows.values()), "days": rows}


def _preflight_report_errors(*, config: AppConfig, registry_report: dict, official_manifest: dict, day_proof: dict, with_models: bool) -> list[str]:
    errors: list[str] = []
    if registry_report.get("valid") is not True:
        errors.append("registry validation failed")
    if official_manifest.get("valid") is not True:
        reason = official_manifest.get("reason") or official_manifest.get("error") or "unknown"
        errors.append(f"official source snapshot invalid: {reason}")
    if day_proof.get("valid") is not True:
        errors.append("day/date proof validation failed")
    if _server_model_preflight_required(config) and not with_models:
        errors.append("server model preflight required but not run; rerun preflight with --with-models")
    return errors


def _server_model_preflight_required(config: AppConfig) -> bool:
    if config.active_environment_profile != "server_linux":
        return False
    preflight_cfg = config.data.get("preflight") or {}
    return any(
        bool(preflight_cfg.get(key))
        for key in (
            "verify_model_fingerprints",
            "require_openrouter_qa_mllm",
            "require_openrouter_model_parameter_preflight",
            "forbid_server_filesystem_model_api_inference_for_server_models",
        )
    )


def _ensure_valid_preflight(
    config: AppConfig,
    queries_path: Path,
    run_id: str,
    root: Path,
    *,
    smoke: bool,
    fetch_official: bool,
    devices: str,
    missing_message: str,
    invalid_message: str,
) -> dict:
    path = root / "preflight_report.json"
    if not path.exists():
        if not smoke:
            raise RuntimeError(missing_message)
        return preflight(config, queries_path, run_id, fetch_official=fetch_official, with_models=False, devices=devices)
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        if smoke:
            return preflight(config, queries_path, run_id, fetch_official=fetch_official, with_models=False, devices=devices)
        raise RuntimeError(f"{invalid_message}: invalid_json:{exc.msg}") from exc
    if report.get("valid") is True:
        return report
    if smoke:
        return preflight(config, queries_path, run_id, fetch_official=fetch_official, with_models=False, devices=devices)
    errors = report.get("errors") or ["preflight_report.valid is not true"]
    raise RuntimeError(f"{invalid_message}: {errors}")


def _load_manifest(root: Path) -> dict:
    path = root / "run_manifest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"run_id": root.name, "stages": []}


def _without_previous_merge_stages(manifest: dict) -> dict:
    return _without_previous_package_stages(manifest, names={"merge_and_package"})


def _without_previous_package_stages(manifest: dict, *, names: set[str]) -> dict:
    result = dict(manifest)
    result["stages"] = [stage for stage in manifest.get("stages", []) if not (isinstance(stage, dict) and stage.get("name") in names)]
    return result


def _write_run_diagnostics(root: Path) -> dict:
    diagnostics_dir = root / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    score_calibrations = _read_json_files(root / "tasks", "score_calibration.json")
    tail_audits = _read_json_files(root / "tasks", "recall_tail_audit.json")
    rejected_rows = _read_jsonl_files(root / "tasks", "rejected_candidates.jsonl")
    score_path = diagnostics_dir / "score_calibration.json"
    tail_path = diagnostics_dir / "recall_tail_audit.json"
    rejected_path = diagnostics_dir / "rejected_candidates.jsonl"
    atomic_write_json(
        score_path,
        {
            "score_norm_version": "raw_floor_high_linear_v1",
            "threshold_source": "fixed_task_raw_floor_v1",
            "task_count": len(score_calibrations),
            "tasks": score_calibrations,
        },
    )
    block_reasons = []
    for audit in tail_audits:
        if audit.get("block_packaging"):
            block_reasons.extend(f"{audit.get('task_id')}: {reason}" for reason in audit.get("block_reasons") or [])
    atomic_write_json(
        tail_path,
        {
            "tail_audit_policy": "deterministic_cluster_audit_v1",
            "task_count": len(tail_audits),
            "block_packaging": bool(block_reasons),
            "block_reasons": block_reasons,
            "tasks": tail_audits,
        },
    )
    atomic_write_jsonl(rejected_path, rejected_rows)
    return {
        "score_calibration": {
            "path": score_path.as_posix(),
            "task_count": len(score_calibrations),
            "threshold_source": "fixed_task_raw_floor_v1",
            "score_norm_version": "raw_floor_high_linear_v1",
        },
        "tail_audit": {
            "path": tail_path.as_posix(),
            "task_count": len(tail_audits),
            "tail_audit_policy": "deterministic_cluster_audit_v1",
            "block_packaging": bool(block_reasons),
            "block_reasons": block_reasons,
        },
        "files": {
            "score_calibration": {"path": score_path.as_posix(), "sha256": sha256_file(score_path)},
            "recall_tail_audit": {"path": tail_path.as_posix(), "sha256": sha256_file(tail_path)},
            "rejected_candidates": {"path": rejected_path.as_posix(), "sha256": sha256_file(rejected_path)},
        },
    }


def _collect_task_artifacts(root: Path) -> dict:
    task_root = root / "tasks"
    result = {}
    family_dirs = sorted(task_root.glob("*")) if task_root.exists() else []
    for family_dir in family_dirs:
        if not family_dir.is_dir():
            continue
        for task_dir in sorted(family_dir.glob("*")):
            if not task_dir.is_dir():
                continue
            task_id = task_dir.name
            files = {}
            for name in (
                "result.jsonl",
                "evidence.jsonl",
                "answer_result.json",
                "qa_evidence.jsonl",
                "qa_rejected_evidence.jsonl",
                "qa_evidence_packet.json",
                "qa_reducer_diagnostics.json",
                "qa_mllm_manifest.json",
                "task_manifest.json",
                "local_visual_recovery_audit.jsonl",
                "solver_report.json",
                "score_calibration.json",
                "recall_tail_audit.json",
                "rejected_candidates.jsonl",
            ):
                path = task_dir / name
                if path.exists():
                    files[name] = {"path": path.as_posix(), "sha256": sha256_file(path), "size": path.stat().st_size}
            for sub_path in sorted(task_dir.glob("**/stage_manifest.json")):
                key = sub_path.relative_to(task_dir).as_posix()
                files[key] = {"path": sub_path.as_posix(), "sha256": sha256_file(sub_path), "size": sub_path.stat().st_size}
            for sub_path in sorted(task_dir.glob("*/post_patch_pruning.jsonl")):
                key = sub_path.relative_to(task_dir).as_posix()
                files[key] = {"path": sub_path.as_posix(), "sha256": sha256_file(sub_path), "size": sub_path.stat().st_size}
            marker = task_dir / "completed.marker"
            result[task_id] = {
                "task_type": family_dir.name,
                "task_dir": task_dir.as_posix(),
                "completed_marker": marker.exists(),
                "completed_marker_sha256": sha256_file(marker) if marker.exists() else None,
                "task_manifest": _read_task_manifest_summary(task_dir / "task_manifest.json"),
                "files": files,
            }
    return result


def _write_root_per_task_report(root: Path, validation: dict) -> None:
    diagnostics_dir = root / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    rows = validation.get("row_counts") or {}
    lines = [
        f"# CASTLE MM26 Per-Task Diagnostics: {root.name}",
        "",
        "## Row Counts",
        "",
        f"- event_instance_search: {rows.get('event_instance_search', 0)}",
        f"- object_instance_search: {rows.get('object_instance_search', 0)}",
        f"- question_answering: {rows.get('question_answering', 0)}",
        "",
        "## Validation",
        "",
        f"- valid: {validation.get('valid')}",
        f"- error_count: {validation.get('error_count')}",
        f"- warning_count: {validation.get('warning_count')}",
        "",
        "## Errors",
        "",
    ]
    errors = validation.get("errors") or []
    lines.extend([f"- {error}" for error in errors] or ["- none"])
    lines.extend(["", "## Warnings", ""])
    warnings = validation.get("warnings") or []
    lines.extend([f"- {warning}" for warning in warnings] or ["- none"])
    lines.append("")
    atomic_write_text(diagnostics_dir / "per_task_report.md", "\n".join(lines))


def _read_task_manifest_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"status": "invalid_json", "error": str(exc), "path": path.as_posix()}
    return {
        "status": data.get("status"),
        "task_id": data.get("task_id"),
        "schema_version": data.get("schema_version"),
        "task_query_sha256": ((data.get("input_hashes") or {}).get("task_query_sha256")),
        "resolved_profiles_sha256": ((data.get("input_hashes") or {}).get("resolved_profiles_sha256")),
        "completed_marker_written_after_manifest": data.get("completed_marker_written_after_manifest"),
    }


def _task_completed_for_query(config: AppConfig, query: TaskQuery, task_dir: Path) -> bool:
    if not (task_dir / "completed.marker").exists():
        return False
    manifest = read_task_manifest(task_dir / "task_manifest.json")
    if not isinstance(manifest, dict):
        return False
    hashes = manifest.get("input_hashes") or {}
    return (
        manifest.get("status") == "completed"
        and manifest.get("task_id") == query.task_id
        and manifest.get("task_type") == query.task_type
        and manifest.get("completed_marker_written_after_manifest") is True
        and hashes.get("task_query_sha256") == canonical_json_hash(query.asdict())
        and hashes.get("resolved_profiles_sha256") == canonical_json_hash(query.resolved_profiles)
        and hashes.get("config_sha256") == sha256_file(config.path)
    )


def _qa_packet_complete_for_query(query: TaskQuery, packet_path: Path) -> bool:
    if not packet_path.exists():
        return False
    try:
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if packet.get("schema_version") != "castle_mm26_qa_evidence_packet_v1":
        return False
    if packet.get("task_id") != query.task_id:
        return False
    packet_query = packet.get("query")
    if not isinstance(packet_query, dict) or canonical_json_hash(packet_query) != canonical_json_hash(query.asdict()):
        return False
    stage_manifests = packet.get("stage_manifests")
    if not isinstance(stage_manifests, dict) or not {"text_rerank/stage_manifest.json", "visual_patch_rerank/stage_manifest.json"} <= set(stage_manifests):
        return False
    evidence = packet.get("evidence")
    return isinstance(evidence, list) and bool(evidence)


def _task_dir_has_completed_manifest(task_dir: Path) -> bool:
    if not (task_dir / "completed.marker").exists():
        return False
    manifest = read_task_manifest(task_dir / "task_manifest.json")
    return (
        isinstance(manifest, dict)
        and manifest.get("status") == "completed"
        and manifest.get("task_id") == task_dir.name
        and manifest.get("completed_marker_written_after_manifest") is True
    )


def _read_json_files(root: Path, name: str) -> list[dict]:
    rows = []
    if not root.exists():
        return rows
    for path in sorted(root.glob(f"*/*/{name}")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _read_jsonl_files(root: Path, name: str) -> list[dict]:
    rows = []
    if not root.exists():
        return rows
    for path in sorted(root.glob(f"*/*/{name}")):
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def _model_roots(model_profile: dict) -> dict:
    result = {}
    for key, value in model_profile.items():
        if key.endswith("_model_root"):
            result[key] = str(value)
    return result


def _read_task_rows(root: Path) -> list[dict]:
    rows = []
    if not root.exists():
        return rows
    for path in sorted(root.glob("*/result.jsonl")):
        if not _task_dir_has_completed_manifest(path.parent):
            continue
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def _read_evidence(root: Path) -> list[EvidenceBundle]:
    rows = []
    if not root.exists():
        return rows
    for path in sorted(root.glob("*/evidence.jsonl")):
        if not _task_dir_has_completed_manifest(path.parent):
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    data = json.loads(line)
                    rows.append(EvidenceBundle(**data))
    return rows


def _read_qa_results(root: Path) -> list[AnswerResult]:
    rows = []
    if not root.exists():
        return rows
    for path in sorted(root.glob("*/answer_result.json")):
        if not _task_dir_has_completed_manifest(path.parent):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(AnswerResult(**data))
    return rows


def _read_qa_evidence_by_task(root: Path) -> dict[str, list]:
    rows: dict[str, list] = {}
    if not root.exists():
        return rows
    from .models import QAEvidence

    for path in sorted(root.glob("*/qa_evidence.jsonl")):
        if not _task_dir_has_completed_manifest(path.parent):
            continue
        task_rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    task_rows.append(QAEvidence(**json.loads(line)))
        rows[path.parent.name] = task_rows
    return rows
