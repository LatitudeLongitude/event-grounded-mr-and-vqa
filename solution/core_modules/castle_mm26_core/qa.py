from __future__ import annotations

import json
import re
import shutil
import unicodedata
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import AppConfig
from .data import EventStore, resolve_frame_path
from .io_utils import atomic_write_json, atomic_write_jsonl, now_utc, sha256_file
from .models import AnswerResult, FrameHit, QAEvidence, TaskQuery, TextHit
from .runtimes import OpenRouterClient, prompt_image_attachment_summary
from .task_manifest import write_completed_marker, write_task_manifest
from .time_utils import format_submission_hms, normalize_output_interval


PERSON_NAMES = {"Allie", "Bao", "Björn", "Bjorn", "Cathal", "Florian", "Klaus", "Luca", "Onanong", "Stevan", "Tien", "Werner"}
FIXED_ROOM_SOURCES = {"Kitchen", "Living1", "Living2", "Meeting", "Reading"}
ACCEPTED_QA_SOURCES = PERSON_NAMES | FIXED_ROOM_SOURCES
PERSON_ALIAS_MAP = {
    "Björn": ("Björn", "Bjorn"),
    "Stevan": ("Stevan", "Stefan", "Steven"),
}
GLOBAL_ANSWER_DISPLAY_NORMALIZATIONS = (
    {
        "value": "Rumkugeln (rum balls)",
        "aliases": (
            "Rumkugel",
            "Rumkugeln",
            "rum ball",
            "rum balls",
        ),
    },
    {
        "value": "Mailandli (Mailänderli)",
        "aliases": (
            "Mailandli",
            "Mailanderli",
            "Mailänderli",
            "Meilanderli",
            "Mylanderli",
            "mylangole",
        ),
    },
)
ORDERING_ACTION_TERMS = (
    "did",
    "doing",
    "done",
    "has done",
    "was doing",
    "finished",
    "already finished",
    "completed",
    "performed",
)
ORDERING_NEGATIVE_TERMS = (
    "has not done",
    "hasn't done",
    "had not done",
    "not done",
    "not everybody has done",
    "need to do",
    "needs to do",
    "should do",
    "about to do",
    "without doing",
    "talking about reading without doing",
    "ordinary reading without the experiment",
)
NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "called",
    "did",
    "for",
    "from",
    "in",
    "of",
    "on",
    "the",
    "they",
    "to",
    "was",
    "were",
    "with",
}


QA_TEXT_FLOORS = {
    "MM26-QA01": 0.45,
    "MM26-QA02": 0.40,
    "MM26-QA03": 0.55,
    "MM26-QA04": 0.50,
    "MM26-QA05": 0.30,
    "MM26-QA06": 0.30,
    "MM26-QA07": 0.35,
    "MM26-QA08": 0.45,
    "MM26-QA09": 0.49,
    "MM26-QA10": 0.50,
    "MM26-QA11": 0.35,
    "MM26-QA12": 0.50,
}


QA_KEEP_TOP_LIMITS = {
    "MM26-QA01": 20,
    "MM26-QA02": 12,
    "MM26-QA03": 16,
    "MM26-QA04": 16,
    "MM26-QA05": 16,
    "MM26-QA06": 16,
    "MM26-QA07": 12,
    "MM26-QA08": 16,
    "MM26-QA09": 12,
    "MM26-QA10": 16,
    "MM26-QA11": 16,
    "MM26-QA12": 20,
}


@dataclass(frozen=True)
class _CountAssessment:
    supported_counts: frozenset[int]
    preferred_count: int | None
    supporting_evidence_ids: tuple[str, ...]
    strategy: str | None
    evidence_state: str

    def supports(self, count: int) -> bool:
        return count in self.supported_counts


CANONICAL_PERSON_COUNT_REPAIR_STATUSES = frozenset(
    {"count_dedupe_mismatch", "event_state_not_completed", "unsupported_count_answer"}
)


def build_qa_evidence(
    config: AppConfig,
    query: TaskQuery,
    text_hits: Sequence[TextHit],
    event_store: EventStore,
    output_dir: Path,
    *,
    keep_top: int = 12,
    visual_hits: Sequence[FrameHit] = (),
) -> tuple[list[QAEvidence], dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence: list[QAEvidence] = []
    rejected_evidence = []
    candidate_serial = 0
    visual_probe_index = _qa_visual_probe_index(visual_hits)
    text_floor = _qa_text_floor(query)
    effective_keep_top = _qa_effective_keep_top(query, keep_top)
    sorted_hits = sorted(text_hits, key=lambda hit: (-(hit.reranker_score if hit.reranker_score is not None else hit.dense_score if hit.dense_score is not None else 0.0), hit.embedding_chunk_id))
    supplement_hits = _qa_policy_lexical_supplement_hits(
        query,
        event_store,
        {hit.embedding_chunk_id for hit in text_hits},
        limit=max(12, effective_keep_top * 3),
    )
    for hit in [*sorted_hits, *supplement_hits]:
        text_atom = event_store.text_by_chunk.get(hit.embedding_chunk_id)
        if text_atom:
            hit.text_atom_id = text_atom["text_atom_id"]
            start = float(text_atom.get("absolute_start_second") or hit.start_second or 0.0)
            end = float(text_atom.get("absolute_end_second") or hit.end_second or start + 1)
        else:
            start = float(hit.start_second or 0.0)
            end = float(hit.end_second or start + 1)
        try:
            interval = normalize_output_interval(start, end)
        except ValueError:
            continue
        event_ids = []
        frame_ids = []
        image_refs = []
        if hit.text_atom_id:
            for link in event_store.text_links_by_text.get(hit.text_atom_id, [])[:5]:
                event = event_store.event_by_id.get(link["event_id"])
                if event:
                    event_ids.append(event["event_id"])
                    for frame_id in event.get("representative_frame_ids") or []:
                        frame_ids.append(frame_id)
                        image_ref = _image_ref_for_frame(config, event_store, frame_id)
                        if image_ref:
                            image_refs.append(image_ref)
        candidate_serial += 1
        evidence_id = f"{query.task_id}:ev:{candidate_serial:03d}"
        text = hit.text_for_display or hit.text_for_embedding
        entity_mentions = _entity_mentions(text)
        relation_candidates = _relation_candidates(query, text)
        time_filter_fit = _time_filter_fit(query, hit.day, interval.start, interval.end)
        image_refs = sorted(set(image_refs))[:3]
        visual_probe_score, visual_probe_audit = _qa_visual_probe_for_evidence(visual_probe_index, frame_ids, hit.day, hit.source, interval.start, interval.end)
        qa_score, qa_score_parts = _qa_score(
            query,
            hit,
            text,
            entity_mentions=entity_mentions,
            relation_candidates=relation_candidates,
            time_filter_fit=time_filter_fit,
            image_refs=image_refs,
            visual_probe_score=visual_probe_score,
        )
        if hit.provenance:
            qa_score_parts["hit_provenance"] = dict(hit.provenance)
        qa_score_parts["visual_probe_audit"] = visual_probe_audit
        gates = _qa_gate_audit(
            query,
            hit,
            text,
            entity_mentions=entity_mentions,
            relation_candidates=relation_candidates,
            time_filter_fit=time_filter_fit,
            text_floor=text_floor,
        )
        qa_score_parts.update(
            {
                "text_floor": text_floor,
                "text_gate": gates["text_gate"],
                "lexical_gate": gates["lexical_gate"],
                "time_gate": gates["time_gate"],
                "entity_gate": gates["entity_gate"],
                "relation_gate": gates["relation_gate"],
                "keep_decision": "candidate_pending_keep_top",
            }
        )
        if visual_probe_score is not None:
            visual_terms = [str(item) for item in (query.answer_policy or {}).get("visual_probe_terms") or []]
            if visual_terms:
                qa_score_parts["visual_probe_terms"] = visual_terms
                if visual_probe_audit.get("temporal_alignment") == "direct":
                    qa_score_parts["trusted_visual_labels"] = visual_terms
        candidate = QAEvidence(
            task_id=query.task_id,
            evidence_id=evidence_id,
            day_num=hit.day,
            source=hit.source,
            source_role=event_store.source_role(hit.source),
            start_second=interval.start,
            end_second=interval.end,
            text_atom_ids=[hit.text_atom_id] if hit.text_atom_id else [],
            event_ids=sorted(set(event_ids)),
            frame_atom_ids=sorted(set(frame_ids)),
            original_image_refs=image_refs,
            text_excerpt=text[:1200],
            entity_mentions=entity_mentions,
            relation_candidates=relation_candidates,
            time_filter_fit=time_filter_fit,
            reranker_score=hit.reranker_score,
            visual_probe_score=visual_probe_score,
            qa_evidence_score=qa_score,
            qa_score_parts=qa_score_parts,
        )
        if not gates["text_gate"]["pass"]:
            row = candidate.asdict()
            row["rejected_reason"] = "below_text_floor"
            row["gate_audit"] = gates
            rejected_evidence.append(row)
            continue
        if not gates["lexical_gate"]["pass"]:
            row = candidate.asdict()
            row["rejected_reason"] = "lexical_gate_failed"
            row["gate_audit"] = gates
            rejected_evidence.append(row)
            continue
        if _has_time_filter(query) and candidate.time_filter_fit <= 0.0:
            row = candidate.asdict()
            row["rejected_reason"] = "time_filter_zero_overlap"
            row["gate_audit"] = gates
            rejected_evidence.append(row)
            continue
        if not gates["entity_gate"]["pass"]:
            row = candidate.asdict()
            row["rejected_reason"] = "entity_gate_failed"
            row["gate_audit"] = gates
            rejected_evidence.append(row)
            continue
        if not gates["relation_gate"]["pass"]:
            row = candidate.asdict()
            row["rejected_reason"] = "relation_gate_failed"
            row["gate_audit"] = gates
            rejected_evidence.append(row)
            continue
        evidence.append(candidate)
    evidence, pruned_evidence = _qa_select_kept_evidence(query, evidence, keep_top=effective_keep_top)
    rejected_evidence.extend(pruned_evidence)
    evidence.sort(key=lambda item: (-item.qa_evidence_score, item.evidence_id))
    atomic_write_jsonl(output_dir / "qa_evidence.jsonl", [item.asdict() for item in evidence])
    atomic_write_jsonl(output_dir / "qa_rejected_evidence.jsonl", rejected_evidence)
    diagnostics = _qa_reducer_diagnostics(query, evidence, rejected_evidence=rejected_evidence)
    atomic_write_json(output_dir / "qa_reducer_diagnostics.json", diagnostics)
    manifest = {
        "status": "completed" if evidence else "failed",
        "task_id": query.task_id,
        "evidence_count": len(evidence),
        "rejected_evidence_count": len(rejected_evidence),
        "lexical_supplement_candidate_count": len(supplement_hits),
        "reducer": diagnostics["reducer"],
        "expected_answer_type": diagnostics["expected_answer_type"],
        "time_filter_pass_count": diagnostics["time_filter"]["fit_count"],
        "relation_candidate_count": diagnostics["relation_candidate_count"],
        "negative_keyword_evidence_count": diagnostics["keyword_frames"]["negative_evidence_count"],
        "visual_probe_evidence_count": sum(1 for item in evidence if item.visual_probe_score is not None),
        "score_weight_source": diagnostics["score_profile"]["weight_source"],
        "score_raw_weights": diagnostics["score_profile"]["raw_weights"],
    }
    atomic_write_json(output_dir / "qa_evidence_manifest.json", manifest)
    return evidence, manifest


def answer_question(
    config: AppConfig,
    query: TaskQuery,
    evidence: Sequence[QAEvidence],
    client: OpenRouterClient,
    output_dir: Path,
    *,
    write_completion: bool = True,
) -> tuple[AnswerResult, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not evidence:
        raise RuntimeError(f"{query.task_id} has no QA evidence")
    diagnostics_path = output_dir / "qa_reducer_diagnostics.json"
    if not diagnostics_path.exists():
        atomic_write_json(diagnostics_path, _qa_reducer_diagnostics(query, evidence))
    rejected_path = output_dir / "qa_rejected_evidence.jsonl"
    if not rejected_path.exists():
        atomic_write_jsonl(rejected_path, [])
    schema = config.data["qa_mllm"]["response_format"]
    prompt_evidence_limit = _qa_prompt_evidence_limit(config)
    max_prompt_images = _qa_prompt_image_limit(config)
    prompt_evidence = _qa_select_prompt_evidence(query, evidence, limit=prompt_evidence_limit)
    evidence_dicts = [item.asdict() for item in prompt_evidence]
    image_attachment_summary = prompt_image_attachment_summary(evidence_dicts, max_images=max_prompt_images)
    try:
        raw = client.answer(
            question=query.official_text,
            evidence=evidence_dicts,
            schema=schema,
            answer_policy=query.answer_policy,
            max_images=max_prompt_images,
        )
    except Exception as exc:
        manifest = {
            "status": "failed",
            "task_id": query.task_id,
            "qa_mllm_provider": "openrouter",
            "mllm_model": client.model,
            "api_key_source": f"env:{config.data['qa_mllm']['api_key_env']}",
            "api_key_present": client.api_key_present(),
            "available_evidence_count": len(evidence),
            "prompt_evidence_count": len(evidence_dicts),
            "prompt_image_attachment_summary": image_attachment_summary,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        atomic_write_json(output_dir / "qa_mllm_manifest.json", manifest)
        raise
    validation_evidence = _qa_validation_evidence(query, evidence, prompt_evidence)
    validation_scope = "prompt_plus_ordering_critical" if len(validation_evidence) != len(prompt_evidence) else "prompt_evidence"
    status = validate_mllm_answer(raw, validation_evidence, query=query)
    repair_manifest = None
    if status != "valid":
        repaired_raw, repair_manifest = _repair_invalid_mllm_answer(raw, validation_evidence, query, status)
        if repaired_raw is not None:
            repaired_status = validate_mllm_answer(repaired_raw, validation_evidence, query=query)
            repair_manifest["repaired_validation_status"] = repaired_status
            if repaired_status == "valid":
                raw = repaired_raw
                status = repaired_status
        if status != "valid":
            manifest = {
                "status": "failed",
                "task_id": query.task_id,
                "mllm_validation_status": status,
                "qa_mllm_provider": "openrouter",
                "mllm_model": client.model,
                "available_evidence_count": len(evidence),
                "prompt_evidence_count": len(evidence_dicts),
                "validation_evidence_count": len(validation_evidence),
                "validation_evidence_scope": validation_scope,
                "prompt_image_attachment_summary": image_attachment_summary,
                "mllm_raw_json": raw,
            }
            if repair_manifest is not None:
                manifest["mllm_answer_repair"] = repair_manifest
            atomic_write_json(output_dir / "qa_mllm_manifest.json", manifest)
            raise RuntimeError(f"OpenRouter answer validation failed for {query.task_id}: {status}")
    supporting = [str(item) for item in raw.get("supporting_evidence_ids") or []]
    supporting_evidence = [_evidence_by_id(evidence, evidence_id) for evidence_id in supporting]
    canonical_answer_type = str(query.answer_policy.get("expected_answer_type") or raw.get("answer_type") or "")
    direct_answer = _normalize_mllm_answer_text(str(raw.get("answer") or "").strip(), canonical_answer_type)
    direct_answer = _global_display_answer(direct_answer, supporting_evidence)
    if not direct_answer:
        raise RuntimeError(f"OpenRouter returned empty answer for {query.task_id}")
    refs = [render_source_time_ref(item) for item in supporting_evidence[:3]]
    answer_text = f"{direct_answer}. Evidence: {'; '.join(refs)}."
    result = AnswerResult(
        task_id=query.task_id,
        answer_value=direct_answer,
        answer_type=canonical_answer_type,
        confidence=float(raw.get("confidence") or 0.0),
        supporting_evidence_ids=supporting,
        mllm_raw_json=raw,
        mllm_validation_status=status,
        mllm_model=client.model,
        response_format_schema_name=schema["json_schema"]["name"],
        dedupe_trace={
            "supporting_evidence_count": len(supporting),
            "supporting_sources": sorted({item.source for item in supporting_evidence}),
            "supporting_time_refs": refs,
            "renderer": "render_source_time_ref_v1",
        },
        rejected_evidence_ids=[item.evidence_id for item in evidence if item.evidence_id not in supporting],
        answer_text=answer_text,
    )
    atomic_write_json(output_dir / "answer_result.json", result.asdict())
    manifest = {
        "status": "completed",
        "task_id": query.task_id,
        "mllm_validation_status": status,
        "supporting_evidence_count": len(supporting),
        "qa_mllm_provider": "openrouter",
        "mllm_model": client.model,
        "available_evidence_count": len(evidence),
        "prompt_evidence_count": len(evidence_dicts),
        "validation_evidence_count": len(validation_evidence),
        "validation_evidence_scope": validation_scope,
        "prompt_image_attachment_summary": image_attachment_summary,
    }
    if repair_manifest is not None:
        manifest["mllm_answer_repair"] = repair_manifest
    atomic_write_json(output_dir / "qa_mllm_manifest.json", manifest)
    if write_completion:
        write_task_manifest(
            config,
            query,
            output_dir,
            status="completed",
            required_files=("qa_evidence.jsonl", "qa_rejected_evidence.jsonl", "qa_reducer_diagnostics.json", "answer_result.json", "qa_mllm_manifest.json"),
            stage_reports={"qa_mllm": manifest},
            extra={"supporting_evidence_count": len(supporting), "answer_type": result.answer_type},
        )
        write_completed_marker(output_dir)
    return result, manifest


def write_qa_evidence_packet(config: AppConfig, query: TaskQuery, evidence: Sequence[QAEvidence], output_dir: Path) -> dict:
    packet_path = output_dir / "qa_evidence_packet.json"
    packet_evidence = [_packetize_qa_evidence_row(config, item.asdict()) for item in evidence]
    stage_manifests = _packet_stage_manifests(config, output_dir)
    rejected_packet_evidence = []
    rejected_path = output_dir / "qa_rejected_evidence.jsonl"
    if rejected_path.exists():
        for line in rejected_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rejected_packet_evidence.append(_packetize_qa_evidence_row(config, json.loads(line)))
    packet = {
        "schema_version": "castle_mm26_qa_evidence_packet_v1",
        "created_at_utc": now_utc(),
        "task_id": query.task_id,
        "query": query.asdict(),
        "qa_mllm_response_format": config.data["qa_mllm"]["response_format"],
        "image_ref_policy": "relative_to_config_repo_root",
        "evidence": packet_evidence,
        "rejected_evidence": rejected_packet_evidence,
        "stage_manifests": stage_manifests,
        "source_files": {
            "qa_evidence_jsonl": {
                "path": _repo_relative(config, (output_dir / "qa_evidence.jsonl").as_posix()),
                "sha256": sha256_file(output_dir / "qa_evidence.jsonl") if (output_dir / "qa_evidence.jsonl").exists() else None,
            },
            "qa_rejected_evidence_jsonl": {
                "path": _repo_relative(config, (output_dir / "qa_rejected_evidence.jsonl").as_posix()),
                "sha256": sha256_file(output_dir / "qa_rejected_evidence.jsonl") if (output_dir / "qa_rejected_evidence.jsonl").exists() else None,
            },
            "qa_reducer_diagnostics_json": {
                "path": _repo_relative(config, (output_dir / "qa_reducer_diagnostics.json").as_posix()),
                "sha256": sha256_file(output_dir / "qa_reducer_diagnostics.json") if (output_dir / "qa_reducer_diagnostics.json").exists() else None,
            },
            "text_rerank_stage_manifest_json": stage_manifests.get("text_rerank/stage_manifest.json", {}),
            "visual_patch_rerank_stage_manifest_json": stage_manifests.get("visual_patch_rerank/stage_manifest.json", {}),
        },
    }
    atomic_write_json(packet_path, packet)
    return {"status": "completed", "path": packet_path.as_posix(), "evidence_count": len(packet_evidence)}


def answer_question_from_packet(config: AppConfig, packet_path: Path, client: OpenRouterClient, output_dir: Path | None = None) -> tuple[AnswerResult, dict]:
    packet = json.loads(Path(packet_path).read_text(encoding="utf-8"))
    if packet.get("schema_version") != "castle_mm26_qa_evidence_packet_v1":
        raise ValueError(f"unsupported QA evidence packet: {packet_path}")
    query = TaskQuery(**packet["query"])
    evidence = []
    for item in packet.get("evidence") or []:
        evidence.append(QAEvidence(**_restore_packet_qa_evidence_row(config, item)))
    rejected_evidence = [_restore_packet_qa_evidence_row(config, item) for item in packet.get("rejected_evidence") or []]
    fallback_manifest = None
    if not evidence and rejected_evidence:
        evidence, fallback_manifest = _fallback_qa_evidence_from_rejected(query, rejected_evidence)
    elif rejected_evidence:
        evidence, fallback_manifest = _supplement_sparse_qa_evidence_from_rejected(
            query,
            evidence,
            rejected_evidence,
            minimum=4,
            limit=_qa_prompt_evidence_limit(config),
        )
    target_dir = Path(output_dir) if output_dir else Path(packet_path).resolve().parent
    target_dir.mkdir(parents=True, exist_ok=True)
    _restore_packet_stage_manifests(packet, target_dir)
    copied_packet_path = target_dir / "qa_evidence_packet.json"
    source_packet_path = Path(packet_path)
    if source_packet_path.resolve() != copied_packet_path.resolve():
        shutil.copy2(source_packet_path, copied_packet_path)
    atomic_write_jsonl(target_dir / "qa_evidence.jsonl", [item.asdict() for item in evidence])
    atomic_write_jsonl(target_dir / "qa_rejected_evidence.jsonl", rejected_evidence)
    atomic_write_json(target_dir / "qa_reducer_diagnostics.json", _qa_reducer_diagnostics(query, evidence, rejected_evidence=rejected_evidence))
    result, manifest = answer_question(config, query, evidence, client, target_dir, write_completion=False)
    if fallback_manifest:
        manifest["qa_evidence_fallback"] = fallback_manifest
    manifest["qa_evidence_packet_path"] = copied_packet_path.as_posix()
    manifest["qa_evidence_packet_sha256"] = sha256_file(copied_packet_path)
    manifest["qa_evidence_packet_source_path"] = source_packet_path.as_posix()
    manifest["qa_evidence_jsonl_path"] = (target_dir / "qa_evidence.jsonl").as_posix()
    manifest["qa_evidence_jsonl_sha256"] = sha256_file(target_dir / "qa_evidence.jsonl")
    atomic_write_json(target_dir / "qa_mllm_manifest.json", manifest)
    write_task_manifest(
        config,
        query,
        target_dir,
        status="completed",
        required_files=("qa_evidence.jsonl", "qa_rejected_evidence.jsonl", "qa_reducer_diagnostics.json", "qa_evidence_packet.json", "answer_result.json", "qa_mllm_manifest.json"),
        stage_reports={"qa_mllm": manifest},
        extra={"supporting_evidence_count": len(result.supporting_evidence_ids), "answer_type": result.answer_type},
    )
    write_completed_marker(target_dir)
    return result, manifest


def _qa_policy_lexical_supplement_hits(
    query: TaskQuery,
    event_store: EventStore,
    existing_chunk_ids: set[str],
    *,
    limit: int,
) -> list[TextHit]:
    policy = query.answer_policy or {}
    if not policy.get("activity_event"):
        return []
    text_by_chunk = getattr(event_store, "text_by_chunk", {})
    if not isinstance(text_by_chunk, Mapping) or not text_by_chunk:
        return []
    scored: list[tuple[tuple[float, float, int, float, str], TextHit]] = []
    for chunk_id, row in text_by_chunk.items():
        chunk_id_text = str(chunk_id or "")
        if not chunk_id_text or chunk_id_text in existing_chunk_ids:
            continue
        if not isinstance(row, Mapping):
            continue
        source = str(row.get("source") or "")
        if source not in ACCEPTED_QA_SOURCES:
            continue
        text = str(row.get("text_preview") or row.get("text_for_embedding") or row.get("cleaned_text") or "").strip()
        if not text:
            continue
        try:
            day = int(row.get("day_num") or str(row.get("day") or "").replace("day", "") or 0)
            start = float(row.get("absolute_start_second") if row.get("absolute_start_second") is not None else row.get("start_second") or 0.0)
            end = float(row.get("absolute_end_second") if row.get("absolute_end_second") is not None else row.get("end_second") or start + 1.0)
            interval = normalize_output_interval(start, end)
        except (TypeError, ValueError):
            continue
        time_fit = _time_filter_fit(query, day, interval.start, interval.end)
        if _has_time_filter(query) and time_fit <= 0.0:
            continue
        activity = _activity_event_audit(policy, text, source)
        activity_score = float(activity.get("score") or 0.0)
        support_score = activity_score
        activity_positive_hits = [str(item) for item in activity.get("positive_hits") or []]
        if activity_score < 0.80 or len(set(activity_positive_hits)) < 3:
            continue
        policy_hits: list[str] = []
        policy_hits.extend(activity_positive_hits)
        policy_hits.extend(str(item) for item in activity.get("people_hits") or [])
        if not policy_hits:
            continue
        rank_key = (
            support_score,
            float(time_fit),
            len(set(policy_hits)),
            -float(interval.end - interval.start),
            chunk_id_text,
        )
        scored.append(
            (
                rank_key,
                TextHit(
                    task_id=query.task_id,
                    prompt=query.canonical_text_for_retrieval,
                    embedding_chunk_id=chunk_id_text,
                    day=day,
                    source=source,
                    start_second=interval.start,
                    end_second=interval.end,
                    text_for_embedding=text,
                    text_for_display=text,
                    reranker_score=None,
                    rank=0,
                    text_atom_id=str(row.get("text_atom_id") or "") or None,
                    quality_flags=[str(item) for item in row.get("quality_flags") or []],
                    provenance={
                        "candidate_source": "lexical_answer_support_sweep_v1",
                        "activity_binding_score": activity_score,
                        "time_filter_fit": float(time_fit),
                        "policy_hits": sorted(set(policy_hits)),
                    },
                ),
            )
        )
    scored.sort(key=lambda item: item[0], reverse=True)
    return [hit for _rank_key, hit in scored[: max(0, int(limit))]]


def _fallback_qa_evidence_from_rejected(query: TaskQuery, rejected_rows: Sequence[dict[str, Any]], *, limit: int = 12) -> tuple[list[QAEvidence], dict[str, Any]]:
    scored_rows = []
    for row in rejected_rows:
        if str(row.get("task_id") or "") != query.task_id:
            continue
        if not row.get("original_image_refs"):
            continue
        gate = row.get("gate_audit") or {}
        parts = row.get("qa_score_parts") or {}
        text_pass = _gate_pass(gate, parts, "text_gate")
        lexical_pass = _gate_pass(gate, parts, "lexical_gate")
        time_pass = _gate_pass(gate, parts, "time_gate")
        relation_pass = _gate_pass(gate, parts, "relation_gate")
        people = _canonical_people_from_restored_row(row)
        score = float(row.get("qa_evidence_score") or 0.0)
        rejected_reason = str(row.get("rejected_reason") or "")
        priority = (
            int(bool(text_pass and lexical_pass and time_pass and relation_pass and people)),
            int(bool(text_pass and lexical_pass and time_pass and relation_pass)),
            int(bool(time_pass)),
            int(bool(lexical_pass)),
            int(bool(text_pass)),
            int(bool(people)),
            score,
        )
        scored_rows.append((priority, rejected_reason, row))
    scored_rows.sort(key=lambda item: item[0], reverse=True)
    selected = []
    for _priority, rejected_reason, row in scored_rows[: max(1, int(limit))]:
        restored = _qa_evidence_row_from_restored_packet_row(row)
        restored.qa_score_parts = dict(restored.qa_score_parts or {})
        restored.qa_score_parts["fallback_from_rejected_evidence"] = {
            "policy": "empty_packet_evidence_recover_top_rejected_v1",
            "original_rejected_reason": rejected_reason,
        }
        selected.append(restored)
    return selected, {
        "policy": "empty_packet_evidence_recover_top_rejected_v1",
        "selected_count": len(selected),
        "rejected_candidate_count": len(rejected_rows),
        "selected_evidence_ids": [item.evidence_id for item in selected],
    }


def _supplement_sparse_qa_evidence_from_rejected(
    query: TaskQuery,
    evidence: Sequence[QAEvidence],
    rejected_rows: Sequence[dict[str, Any]],
    *,
    minimum: int = 4,
    limit: int = 12,
) -> tuple[list[QAEvidence], dict[str, Any] | None]:
    current = list(evidence)
    if len(current) >= minimum or len(current) >= limit:
        return current, None
    existing_ids = {item.evidence_id for item in current}
    additions, manifest = _fallback_qa_evidence_from_rejected(query, rejected_rows, limit=max(1, limit - len(current)))
    selected = []
    for item in additions:
        if item.evidence_id in existing_ids:
            continue
        item.qa_score_parts = dict(item.qa_score_parts or {})
        item.qa_score_parts["fallback_from_rejected_evidence"] = {
            "policy": "sparse_packet_evidence_supplement_top_rejected_v1",
            "original_rejected_reason": (item.qa_score_parts.get("fallback_from_rejected_evidence") or {}).get("original_rejected_reason"),
        }
        selected.append(item)
        existing_ids.add(item.evidence_id)
        if len(current) + len(selected) >= limit:
            break
    if not selected:
        return current, None
    current.extend(selected)
    return current, {
        "policy": "sparse_packet_evidence_supplement_top_rejected_v1",
        "original_evidence_count": len(evidence),
        "supplemented_count": len(selected),
        "rejected_candidate_count": manifest["rejected_candidate_count"],
        "selected_evidence_ids": [item.evidence_id for item in selected],
    }


def _gate_pass(gate: Mapping[str, Any], parts: Mapping[str, Any], name: str) -> bool:
    value = gate.get(name)
    if value is None:
        value = parts.get(name)
    return bool(value.get("pass")) if isinstance(value, Mapping) else False


def _canonical_people_from_restored_row(row: Mapping[str, Any]) -> list[str]:
    evidence = _qa_evidence_row_from_restored_packet_row(row)
    return _canonical_people_from_evidence([evidence])


def _qa_evidence_row_from_restored_packet_row(row: Mapping[str, Any]) -> QAEvidence:
    fields = QAEvidence.__dataclass_fields__
    payload = {key: value for key, value in row.items() if key in fields}
    return QAEvidence(**payload)


def _qa_prompt_evidence_limit(config: AppConfig) -> int:
    qa_config = config.data.get("qa_mllm") or {}
    value = qa_config.get("max_prompt_evidence", 12)
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 12


def _qa_prompt_image_limit(config: AppConfig) -> int:
    qa_config = config.data.get("qa_mllm") or {}
    value = qa_config.get("max_prompt_images", qa_config.get("max_prompt_evidence", 12))
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 12


def _qa_select_prompt_evidence(query: TaskQuery, evidence: Sequence[QAEvidence], *, limit: int) -> list[QAEvidence]:
    limit = max(1, int(limit))
    if len(evidence) <= limit or not _qa_uses_person_ordering(query):
        return list(evidence[:limit])
    selected: list[QAEvidence] = []
    critical = sorted(
        (item for item in evidence if _qa_is_ordering_critical_evidence(query, item)),
        key=lambda item: _qa_ordering_priority_key(query, item),
    )
    for item in critical:
        if len(selected) >= limit:
            break
        if item not in selected:
            selected.append(item)
    for item in evidence:
        if len(selected) >= limit:
            break
        if item not in selected:
            selected.append(item)
    return selected


def _qa_validation_evidence(query: TaskQuery, evidence: Sequence[QAEvidence], prompt_evidence: Sequence[QAEvidence]) -> list[QAEvidence]:
    if not _qa_uses_person_ordering(query):
        return list(prompt_evidence)
    selected = list(prompt_evidence)
    selected_ids = {item.evidence_id for item in selected}
    candidates = sorted(
        (row for row in evidence if _qa_is_ordering_critical_evidence(query, row)),
        key=lambda row: _qa_ordering_priority_key(query, row),
    )
    for item in candidates:
        if item.evidence_id not in selected_ids:
            selected.append(item)
            selected_ids.add(item.evidence_id)
    return selected


def _packet_stage_manifests(config: AppConfig, output_dir: Path) -> dict:
    result = {}
    for rel in ("text_rerank/stage_manifest.json", "visual_patch_rerank/stage_manifest.json"):
        path = output_dir / rel
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        result[rel] = {
            "path": _repo_relative(config, path.as_posix()),
            "sha256": sha256_file(path),
            "payload": payload,
        }
    return result


def _restore_packet_stage_manifests(packet: dict, target_dir: Path) -> None:
    allowed = {"text_rerank/stage_manifest.json", "visual_patch_rerank/stage_manifest.json"}
    for rel, info in (packet.get("stage_manifests") or {}).items():
        if rel not in allowed or not isinstance(info, dict):
            continue
        payload = info.get("payload")
        if not isinstance(payload, dict):
            continue
        target_path = target_dir / rel
        target_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(target_path, payload)


def _repo_relative(config: AppConfig, path: str) -> str:
    target = Path(path)
    if not target.is_absolute():
        return target.as_posix()
    try:
        return target.relative_to(config.repo_root).as_posix()
    except ValueError:
        return target.as_posix()


def _packetize_qa_evidence_row(config: AppConfig, row: dict) -> dict:
    result = dict(row)
    result["original_image_rel_refs"] = [_repo_relative(config, path) for path in result.get("original_image_refs") or []]
    result.pop("original_image_refs", None)
    return result


def _restore_packet_qa_evidence_row(config: AppConfig, item: dict) -> dict:
    row = dict(item)
    rel_refs = row.pop("original_image_rel_refs", [])
    row["original_image_refs"] = [(config.repo_root / str(rel_ref)).as_posix() for rel_ref in rel_refs]
    return row


def render_source_time_ref(evidence: QAEvidence) -> str:
    return f"{evidence.source} day {evidence.day_num} {format_submission_hms(evidence.start_second)}-{format_submission_hms(evidence.end_second)}"


def validate_mllm_answer(raw: dict, evidence: Sequence[QAEvidence], *, query: TaskQuery | None = None) -> str:
    if not isinstance(raw, dict):
        return "invalid_schema"
    required = {"answer", "answer_type", "confidence", "supporting_evidence_ids", "rationale_short"}
    if not required <= set(raw):
        return "invalid_schema"
    answer_type = str(raw.get("answer_type") or "").strip()
    confidence = _parse_confidence(raw.get("confidence"))
    raw_answer = str(raw.get("answer") or "").strip()
    if not raw_answer or not answer_type or confidence is None:
        return "invalid_schema"
    expected_type = str((query.answer_policy or {}).get("expected_answer_type") or "").strip() if query else ""
    if expected_type and not _answer_type_matches(answer_type, expected_type):
        return "invalid_answer_type"
    normalized_type = expected_type or answer_type
    answer = _normalize_mllm_answer_text(raw_answer, normalized_type)
    if "Evidence:" in answer or "{" in answer or "}" in answer:
        return "invalid_answer_text"
    if _word_count(answer) > 24 or len(answer) > 180:
        return "answer_too_long"
    ids = {item.evidence_id for item in evidence}
    supporting = raw.get("supporting_evidence_ids")
    if not isinstance(supporting, list) or not supporting:
        return "invalid_evidence"
    supporting_ids = [str(item) for item in supporting]
    if any(item not in ids for item in supporting_ids):
        return "invalid_evidence"
    if "I could not determine" in answer:
        return "fallback_answer_forbidden"
    for room in FIXED_ROOM_SOURCES:
        if answer.strip() == room:
            return "fixed_room_as_actor"
    supporting_evidence = [_evidence_by_id(evidence, evidence_id) for evidence_id in supporting_ids]
    if query is not None and any(item.task_id != query.task_id for item in supporting_evidence):
        return "invalid_evidence_task"
    min_support = _required_support_count(query, confidence)
    if len({item.evidence_id for item in supporting_evidence}) < min_support:
        return "insufficient_supporting_evidence"
    evidence_status = _validate_supporting_evidence(supporting_evidence)
    if evidence_status != "valid":
        return evidence_status
    if normalized_type in {"person_name", "person", "actor"}:
        person_status = _validate_person_answer(answer, supporting_evidence)
        if person_status != "valid":
            return person_status
        actor_status = _validate_actor_resolution_answer(answer, supporting_evidence, query)
        if actor_status != "valid":
            return actor_status
        relation_status = _validate_relation_direction_answer(answer, supporting_evidence, query)
        if relation_status != "valid":
            return relation_status
        ordering_status = _validate_ordering_answer(answer, evidence, supporting_evidence, query)
        if ordering_status != "valid":
            return ordering_status
    elif normalized_type in {"count", "number", "integer"}:
        count = _parse_count_answer(answer)
        if count is None:
            return "invalid_count_answer"
        count_status = _validate_count_answer(count, supporting_evidence, query)
        if count_status != "valid":
            return count_status
    else:
        support_status = _validate_non_person_answer(answer, supporting_evidence, query)
        if support_status != "valid":
            return support_status
        extraction_status = _validate_extraction_context(supporting_evidence, query)
        if extraction_status != "valid":
            return extraction_status
        policy_answer_status = _validate_policy_answer_constraints(answer, supporting_evidence, query)
        if policy_answer_status != "valid":
            return policy_answer_status
    return "valid"


def _repair_invalid_mllm_answer(raw: dict, evidence: Sequence[QAEvidence], query: TaskQuery, status: str) -> tuple[dict | None, dict | None]:
    policy = query.answer_policy or {}
    expected_type = str(policy.get("expected_answer_type") or raw.get("answer_type") or "").strip().lower()
    if expected_type in {"person_name", "person", "actor"} and policy.get("actor_resolution_policy"):
        if status not in {"actor_resolution_not_supported", "unsupported_person_answer"}:
            return None, None
        inferred_actor = _deterministic_actor_from_evidence(evidence, policy)
        if inferred_actor is None:
            return None, None
        actor, supporting_ids, strategy = inferred_actor
        confidence = _parse_confidence(raw.get("confidence"))
        repaired = dict(raw)
        repaired["answer"] = actor
        repaired["answer_type"] = "person_name"
        repaired["confidence"] = confidence if confidence is not None and confidence >= 0.5 else 0.65
        repaired["supporting_evidence_ids"] = supporting_ids
        repaired["rationale_short"] = "Deterministic text repair: first-person action evidence resolves the actor."
        repaired["repair_applied"] = {
            "policy": "deterministic_actor_from_text_v1",
            "original_answer": raw.get("answer"),
            "original_validation_status": status,
            "strategy": strategy,
        }
        return repaired, {
            "policy": "deterministic_actor_from_text_v1",
            "original_validation_status": status,
            "original_answer": raw.get("answer"),
            "repaired_answer": actor,
            "supporting_evidence_ids": supporting_ids,
            "strategy": strategy,
        }
    if expected_type not in {"count", "number", "integer"}:
        return None, None
    count_unit = str(policy.get("count_unit") or "")
    dedupe_policy = str(policy.get("dedupe_policy") or "")
    if count_unit != "canonical_person" and "same_person_once" not in dedupe_policy:
        return None, None
    if status not in CANONICAL_PERSON_COUNT_REPAIR_STATUSES:
        return None, None
    assessment = _assess_canonical_person_count(evidence, query=query)
    if assessment.preferred_count is None or assessment.strategy is None:
        return None, None
    count = assessment.preferred_count
    supporting_ids = list(assessment.supporting_evidence_ids)
    strategy = assessment.strategy
    if not supporting_ids:
        return None, None
    confidence = _parse_confidence(raw.get("confidence"))
    repaired = dict(raw)
    repaired["answer"] = str(count)
    repaired["answer_type"] = "count"
    repaired["confidence"] = confidence if confidence is not None and confidence >= 0.5 else 0.65
    repaired["supporting_evidence_ids"] = supporting_ids
    repaired["rationale_short"] = "Deterministic evidence repair: event state and transcript participants support the inferred count."
    repaired["repair_applied"] = {
        "policy": "deterministic_count_from_text_v1",
        "original_answer": raw.get("answer"),
        "original_validation_status": status,
        "strategy": strategy,
    }
    manifest = {
        "policy": "deterministic_count_from_text_v1",
        "original_validation_status": status,
        "original_answer": raw.get("answer"),
        "repaired_answer": str(count),
        "supporting_evidence_ids": supporting_ids,
        "strategy": strategy,
    }
    return repaired, manifest


def _deterministic_actor_from_evidence(evidence: Sequence[QAEvidence], policy: dict[str, Any]) -> tuple[str, list[str], str] | None:
    actor_policy = policy.get("actor_resolution_policy") or {}
    if not actor_policy:
        return None
    buckets: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(evidence):
        text = item.text_excerpt or ""
        source_person = _canonical_person(item.source)
        first_person_hits = _keyword_hits(text, [str(term) for term in actor_policy.get("first_person_action_terms") or []])
        action_hits = _actor_action_hits(text, actor_policy)
        if source_person and item.source not in FIXED_ROOM_SOURCES and first_person_hits:
            _actor_bucket_add(buckets, source_person, item, index, score=3.0, strategy="source_first_person_action")
            continue
        if source_person and item.source not in FIXED_ROOM_SOURCES and action_hits and _trusted_visual_support_text(item):
            _actor_bucket_add(buckets, source_person, item, index, score=2.0, strategy="source_action_with_visual_support")
    if not buckets:
        return None
    ranked = sorted(
        buckets.items(),
        key=lambda pair: (
            -float(pair[1]["score"]),
            -len(pair[1]["evidence_ids"]),
            int(pair[1]["first_index"]),
            pair[0],
        ),
    )
    if len(ranked) > 1:
        top_score = float(ranked[0][1]["score"])
        second_score = float(ranked[1][1]["score"])
        top_count = len(ranked[0][1]["evidence_ids"])
        second_count = len(ranked[1][1]["evidence_ids"])
        if top_score == second_score and top_count == second_count:
            return None
    actor, bucket = ranked[0]
    return actor, list(bucket["evidence_ids"][:3]), str(bucket["strategy"])


def _actor_bucket_add(buckets: dict[str, dict[str, Any]], actor: str, item: QAEvidence, index: int, *, score: float, strategy: str) -> None:
    bucket = buckets.setdefault(actor, {"score": 0.0, "first_index": index, "evidence_ids": [], "strategy": strategy})
    bucket["score"] = float(bucket["score"]) + float(score)
    bucket["first_index"] = min(int(bucket["first_index"]), int(index))
    if item.evidence_id not in bucket["evidence_ids"]:
        bucket["evidence_ids"].append(item.evidence_id)
    if float(score) >= 3.0:
        bucket["strategy"] = strategy


def _assess_canonical_person_count(evidence: Sequence[QAEvidence], *, query: TaskQuery | None = None) -> _CountAssessment:
    policy = query.answer_policy if query else {}
    requires_completed = _event_state_requires_completed(policy)
    completed_evidence = [item for item in evidence if _event_state_supports_completed(policy, item.text_excerpt or "")]
    has_completed = bool(completed_evidence)
    has_future = any(_event_state_is_future_plan(policy, item.text_excerpt or "") for item in evidence)
    evidence_state = (
        {
            (False, False): "missing_completed_marker",
            (False, True): "future_only",
            (True, False): "completed",
            (True, True): "conflicting_completed_and_future",
        }[(has_completed, has_future)]
        if requires_completed
        else "unconstrained"
    )
    if requires_completed and evidence_state != "completed":
        if evidence and evidence_state in {"missing_completed_marker", "future_only"}:
            return _CountAssessment(
                supported_counts=frozenset({0}),
                preferred_count=0,
                supporting_evidence_ids=tuple(item.evidence_id for item in evidence[:3]),
                strategy="event_state_no_completed_zero_count_v1",
                evidence_state=evidence_state,
            )
        return _CountAssessment(
            supported_counts=frozenset(),
            preferred_count=None,
            supporting_evidence_ids=(),
            strategy=None,
            evidence_state=evidence_state,
        )

    count_evidence = completed_evidence if requires_completed else list(evidence)
    by_count: dict[int, dict[str, Any]] = {}
    supported_counts: set[int] = set()
    canonical_people = _canonical_people_from_evidence(evidence)
    if canonical_people:
        supported_counts.add(len(canonical_people))
    for index, item in enumerate(count_evidence):
        group = _explicit_person_group_from_text(item.text_excerpt or "", source=item.source)
        local_people = _canonical_people_from_evidence([item])
        candidate = group or (
            (1, local_people, "single_canonical_person_count_v1")
            if len(local_people) == 1
            else None
        )
        mentioned_people = {
            person
            for mention in _entity_mentions(item.text_excerpt or "")
            for person in [_canonical_person(str(mention.get("canonical") or mention.get("text") or ""))]
            if person
        }
        if mentioned_people:
            supported_counts.add(len(mentioned_people))
        if candidate is None:
            continue
        count, people, strategy = candidate
        supported_counts.add(count)
        bucket = by_count.setdefault(count, {"first_index": index, "evidence_ids": [], "people": set(), "strategy": strategy})
        bucket["evidence_ids"].append(item.evidence_id)
        bucket["people"].update(people)
    if by_count:
        count, bucket = sorted(by_count.items(), key=lambda pair: (-len(pair[1]["evidence_ids"]), pair[1]["first_index"], pair[0]))[0]
        return _CountAssessment(
            supported_counts=frozenset(supported_counts),
            preferred_count=count,
            supporting_evidence_ids=tuple(bucket["evidence_ids"][:3]),
            strategy=str(bucket["strategy"]),
            evidence_state=evidence_state,
        )
    return _CountAssessment(
        supported_counts=frozenset(supported_counts),
        preferred_count=None,
        supporting_evidence_ids=(),
        strategy=None,
        evidence_state=evidence_state,
    )


def _qa_reducer_diagnostics(query: TaskQuery, evidence: Sequence[QAEvidence], *, rejected_evidence: Sequence[dict] = ()) -> dict:
    policy = query.answer_policy or {}
    keyword_frames = policy.get("keyword_frames") or {}
    positives = [str(item) for item in keyword_frames.get("positive") or []]
    negatives = [str(item) for item in keyword_frames.get("negative") or []]
    rows = []
    positive_count = 0
    negative_count = 0
    relation_count = 0
    noun_phrase_count = 0
    for item in evidence:
        text = item.text_excerpt or ""
        positive_hits = _keyword_hits(text, positives)
        negative_hits = _keyword_hits(text, negatives)
        noun_phrases = _noun_phrase_candidates(query, text)
        positive_count += int(bool(positive_hits))
        negative_count += int(bool(negative_hits))
        relation_count += len(item.relation_candidates or [])
        noun_phrase_count += len(noun_phrases)
        rows.append(
            {
                "evidence_id": item.evidence_id,
                "qa_evidence_score": item.qa_evidence_score,
                "qa_score_parts": item.qa_score_parts,
                "time_filter_fit": item.time_filter_fit,
                "visual_probe_score": item.visual_probe_score,
                "ordering_critical": _qa_is_ordering_critical_evidence(query, item),
                "ordering_people": _ordering_people_from_evidence(item, policy) if _qa_uses_person_ordering(query) else [],
                "positive_keyword_hits": positive_hits,
                "negative_keyword_hits": negative_hits,
                "relation_candidate_count": len(item.relation_candidates or []),
                "noun_phrase_candidates": noun_phrases[:5],
                "entity_mentions": item.entity_mentions,
            }
        )
    scores = sorted((float(item.qa_evidence_score) for item in evidence), reverse=True)
    delta = scores[0] - scores[1] if len(scores) >= 2 else None
    score_parts = [item.qa_score_parts for item in evidence if item.qa_score_parts]
    return {
        "schema_version": "castle_mm26_qa_reducer_diagnostics_v1",
        "task_id": query.task_id,
        "reducer": policy.get("reducer"),
        "expected_answer_type": policy.get("expected_answer_type"),
        "score_profile": {
            "scoring_profile": query.scoring_profile,
            "weight_source": score_parts[0].get("weight_source") if score_parts else _qa_score_weights(query)[1],
            "raw_weights": score_parts[0].get("raw_weights") if score_parts else _qa_score_weights(query)[0],
            "used_weight_keys": sorted({key for parts in score_parts for key in (parts.get("used_weights") or {})}),
            "skipped_profile_keys": sorted({key for parts in score_parts for key in (parts.get("skipped_profile_keys") or [])}),
        },
        "time_filter": {
            "policy": policy.get("time_filters") or {},
            "fit_count": sum(1 for item in evidence if float(item.time_filter_fit) > 0.0),
            "zero_fit_count": sum(1 for item in evidence if float(item.time_filter_fit) <= 0.0),
            "rejected_zero_fit_count": sum(1 for item in rejected_evidence if item.get("rejected_reason") == "time_filter_zero_overlap"),
        },
        "keyword_frames": {
            "positive_terms": positives,
            "negative_terms": negatives,
            "positive_evidence_count": positive_count,
            "negative_evidence_count": negative_count,
        },
        "relation_candidate_count": relation_count,
        "noun_phrase_candidate_count": noun_phrase_count,
        "visual_probe": {
            "evidence_count": sum(1 for item in evidence if item.visual_probe_score is not None),
            "max_score": max((float(item.visual_probe_score) for item in evidence if item.visual_probe_score is not None), default=None),
        },
        "ordering": {
            "policy": policy.get("ordering") or {},
            "critical_evidence_ids": [item.evidence_id for item in evidence if _qa_is_ordering_critical_evidence(query, item)],
            "critical_evidence_count": sum(1 for item in evidence if _qa_is_ordering_critical_evidence(query, item)),
            "pruned_critical_evidence_ids": [
                str(item.get("evidence_id"))
                for item in rejected_evidence
                if item.get("rejected_reason") == "keep_top_pruned" and (item.get("qa_score_parts") or {}).get("ordering_critical")
            ],
        },
        "conflict": {
            "top_score": scores[0] if scores else None,
            "second_score": scores[1] if len(scores) >= 2 else None,
            "score_delta": delta,
            "low_margin_threshold": 0.08,
            "low_margin": bool(delta is not None and delta < 0.08),
        },
        "evidence": rows,
        "rejected_evidence_count": len(rejected_evidence),
    }


def _has_time_filter(query: TaskQuery) -> bool:
    return bool((query.answer_policy or {}).get("time_filters"))


def _event_state_requires_completed(policy: dict[str, Any]) -> bool:
    event_state = policy.get("event_state") or {}
    return str(event_state.get("required") or "").strip().lower() == "completed"


def _event_state_supports_completed(policy: dict[str, Any], text: str) -> bool:
    event_state = policy.get("event_state") or {}
    completed_hits = _keyword_hits(text, [str(item) for item in event_state.get("completed_markers") or []])
    future_hits = _keyword_hits(text, [str(item) for item in event_state.get("future_markers") or []])
    if bool(event_state.get("reject_future_plan")) and future_hits:
        return False
    return bool(completed_hits)


def _event_state_is_future_plan(policy: dict[str, Any], text: str) -> bool:
    event_state = policy.get("event_state") or {}
    if not bool(event_state.get("reject_future_plan")):
        return False
    future_hits = _keyword_hits(text, [str(item) for item in event_state.get("future_markers") or []])
    completed_hits = _keyword_hits(text, [str(item) for item in event_state.get("completed_markers") or []])
    return bool(future_hits and not completed_hits)


def _event_state_audit(policy: dict[str, Any], text: str) -> dict[str, Any]:
    event_state = policy.get("event_state") or {}
    if not event_state:
        return {"available": False, "score": 0.0, "completed_hits": [], "future_hits": [], "state": "not_configured"}
    completed_hits = _keyword_hits(text, [str(item) for item in event_state.get("completed_markers") or []])
    future_hits = _keyword_hits(text, [str(item) for item in event_state.get("future_markers") or []])
    if bool(event_state.get("reject_future_plan")) and future_hits:
        return {"available": True, "score": 0.0, "completed_hits": completed_hits, "future_hits": future_hits, "state": "future_plan"}
    if _event_state_requires_completed(policy) and completed_hits:
        return {"available": True, "score": 1.0, "completed_hits": completed_hits, "future_hits": future_hits, "state": "completed"}
    if _event_state_requires_completed(policy):
        return {"available": True, "score": 0.0, "completed_hits": completed_hits, "future_hits": future_hits, "state": "missing_completed_marker"}
    return {"available": True, "score": 1.0, "completed_hits": completed_hits, "future_hits": future_hits, "state": "unconstrained"}


def _activity_event_audit(policy: dict[str, Any], text: str, source: str) -> dict[str, Any]:
    activity = policy.get("activity_event") or {}
    if not activity:
        return {"available": False, "score": 0.0, "positive_hits": [], "negative_hits": [], "people_hits": []}
    positive_hits = _keyword_hits(text, [str(item) for item in activity.get("positive_terms") or []])
    negative_hits = _keyword_hits(text, [str(item) for item in activity.get("negative_terms") or []])
    required_people = [str(item) for item in activity.get("required_people") or []]
    people_hits = []
    for person in required_people:
        canonical = _canonical_person(person)
        if canonical and (_canonical_person(source) == canonical or any(re.search(_term_pattern(alias), text, flags=re.IGNORECASE) for alias in _person_aliases(canonical))):
            people_hits.append(canonical)
    positive_score = min(1.0, len(positive_hits) / 2.0) if positive_hits else 0.0
    people_score = min(1.0, len(set(people_hits)) / max(len(required_people), 1)) if required_people else 1.0
    score = (0.65 * positive_score) + (0.35 * people_score)
    if negative_hits:
        score *= 0.2
    return {"available": True, "score": _clip01(score), "positive_hits": positive_hits, "negative_hits": negative_hits, "people_hits": sorted(set(people_hits))}


def _actor_binding_audit(policy: dict[str, Any], text: str, source: str, trusted_visual_text: str) -> dict[str, Any]:
    actor_policy = policy.get("actor_resolution_policy") or {}
    if not actor_policy:
        return {"available": False, "score": 0.0, "first_person_hits": [], "action_hits": [], "negative_hits": [], "source_actor": None}
    first_person_hits = _keyword_hits(text, [str(item) for item in actor_policy.get("first_person_action_terms") or []])
    action_hits = _actor_action_hits(text, actor_policy)
    negative_hits = _keyword_hits(text, [str(item) for item in actor_policy.get("negative_actor_terms") or []])
    source_actor = _canonical_person(source) if source not in FIXED_ROOM_SOURCES else None
    score = 0.0
    if source_actor and first_person_hits:
        score = 1.0
    elif source_actor and action_hits and trusted_visual_text:
        score = 0.75
    elif action_hits:
        score = 0.45
    if negative_hits and not first_person_hits:
        score *= 0.2
    return {
        "available": True,
        "score": _clip01(score),
        "first_person_hits": first_person_hits,
        "action_hits": action_hits,
        "negative_hits": negative_hits,
        "source_actor": source_actor,
    }


def _keyword_hits(text: str, keywords: Sequence[str]) -> list[str]:
    lower = text.lower()
    return [keyword for keyword in keywords if keyword.lower() in lower]


def _keyword_hits_with_positions(text: str, keywords: Sequence[str]) -> list[tuple[str, int]]:
    hits: list[tuple[str, int]] = []
    for keyword in keywords:
        if not keyword:
            continue
        pattern = _term_pattern(str(keyword))
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            hits.append((str(keyword), match.start()))
    return hits


def _noun_phrase_candidates(query: TaskQuery, text: str) -> list[str]:
    policy = query.answer_policy or {}
    predicates = list((policy.get("keyword_frames") or {}).get("relation_predicates") or [])
    extraction = policy.get("extraction") or {}
    window = int(extraction.get("target_window_tokens_after_predicate") or extraction.get("target_window_tokens_after_called") or extraction.get("target_window_tokens_after_song_marker") or 12)
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+", text)
    lower_tokens = [token.lower() for token in tokens]
    candidates = []
    for predicate in predicates:
        predicate_tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+", str(predicate).lower())
        if not predicate_tokens:
            continue
        for index in range(0, max(0, len(tokens) - len(predicate_tokens) + 1)):
            if lower_tokens[index : index + len(predicate_tokens)] != predicate_tokens:
                continue
            phrase_tokens = []
            for token in tokens[index + len(predicate_tokens) : index + len(predicate_tokens) + window]:
                if token.lower() in STOPWORDS and not phrase_tokens:
                    continue
                phrase_tokens.append(token)
                if len(phrase_tokens) >= 6:
                    break
            phrase = " ".join(phrase_tokens).strip(" ,.;:-")
            if phrase and phrase.lower() not in STOPWORDS:
                candidates.append(phrase)
    seen = set()
    deduped = []
    for item in candidates:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def _parse_confidence(value: object) -> float | None:
    try:
        confidence = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not 0.0 <= confidence <= 1.0:
        return None
    return confidence


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9À-ÖØ-öø-ÿ]+", text))


def _answer_type_matches(actual: str, expected: str) -> bool:
    actual_norm = actual.strip().lower()
    expected_norm = expected.strip().lower()
    if actual_norm == expected_norm:
        return True
    aliases = {
        "person_name": {"person", "person_name"},
        "count": {"count", "number", "integer"},
        "object_name": {"object", "object_name", "entity"},
        "activity_phrase": {"activity", "activity_phrase"},
        "song_title": {"song", "song_title", "title"},
        "food_name": {"food", "food_name", "object_name", "entity"},
    }
    return actual_norm in aliases.get(expected_norm, set())


def _normalize_mllm_answer_text(answer: str, answer_type: str) -> str:
    answer = str(answer or "").strip()
    answer_type_norm = str(answer_type or "").strip().lower()
    if answer_type_norm in {"person_name", "person", "actor"}:
        person = _extract_single_person_answer(answer)
        if person:
            return person
    if answer_type_norm in {"count", "number", "integer"}:
        count = _parse_count_answer(answer)
        if count is not None:
            return str(count)
    return answer


def _extract_single_person_answer(answer: str) -> str | None:
    people: list[str] = []
    for name in _canonical_person_names():
        aliases = _person_aliases(name)
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", answer, flags=re.IGNORECASE):
                canonical = _canonical_person(alias)
                if canonical:
                    people.append(canonical)
    unique = sorted(set(people))
    if len(unique) == 1:
        return unique[0]
    return None


def _required_support_count(query: TaskQuery | None, confidence: float) -> int:
    if query is None:
        return 1
    policy = query.answer_policy or {}
    min_support = int(policy.get("min_supporting_evidence") or 1)
    low_confidence_min = int(policy.get("low_confidence_min_supporting_evidence") or min_support)
    if confidence < 0.5 and not policy.get("single_evidence_allowed_reason"):
        return max(min_support, low_confidence_min)
    return min_support


def _validate_supporting_evidence(evidence: Sequence[QAEvidence]) -> str:
    for item in evidence:
        if item.source not in ACCEPTED_QA_SOURCES:
            return "invalid_evidence_source"
        if not (1 <= int(item.day_num) <= 4):
            return "invalid_evidence_time"
        if float(item.start_second) >= float(item.end_second):
            return "invalid_evidence_time"
        if not item.text_atom_ids and not item.event_ids and not item.frame_atom_ids:
            return "invalid_evidence"
        if not item.original_image_refs:
            return "missing_evidence_image"
    return "valid"


def _validate_person_answer(answer: str, evidence: Sequence[QAEvidence]) -> str:
    canonical = _canonical_person(answer)
    if not canonical:
        return "invalid_person_answer"
    if canonical in FIXED_ROOM_SOURCES:
        return "fixed_room_as_actor"
    for item in evidence:
        if _evidence_mentions_person(item, canonical):
            return "valid"
        if _canonical_person(item.source) == canonical and item.source not in FIXED_ROOM_SOURCES:
            return "valid"
    return "unsupported_person_answer"


def _validate_relation_direction_answer(answer: str, evidence: Sequence[QAEvidence], query: TaskQuery | None) -> str:
    if query is None:
        return "valid"
    policy = query.answer_policy or {}
    relation_direction = policy.get("relation_direction") or {}
    if not relation_direction.get("require_direction"):
        return "valid"
    subject = _canonical_person(answer)
    if not subject:
        return "invalid_person_answer"
    object_aliases = _relation_object_aliases(policy, str(relation_direction.get("object") or ""))
    predicates = [str(item) for item in (policy.get("keyword_frames") or {}).get("relation_predicates") or []]
    if not object_aliases or not predicates:
        return "relation_direction_not_supported"
    for item in evidence:
        for relation in item.relation_candidates or []:
            if _relation_candidate_matches_direction(relation, subject, object_aliases, predicates):
                return "valid"
        if _text_supports_relation_direction(item.text_excerpt, subject, object_aliases, predicates):
            return "valid"
    if _combined_teaching_relation_supported(subject, evidence, object_aliases, predicates):
        return "valid"
    return "relation_direction_not_supported"


def _validate_count_answer(count: int, evidence: Sequence[QAEvidence], query: TaskQuery | None) -> str:
    if query is None:
        return "valid"
    policy = query.answer_policy or {}
    count_unit = str(policy.get("count_unit") or "")
    dedupe_policy = str(policy.get("dedupe_policy") or "")
    if count_unit != "canonical_person" and "same_person_once" not in dedupe_policy:
        return "valid"
    assessment = _assess_canonical_person_count(evidence, query=query)
    if assessment.supports(count):
        return "valid"
    if assessment.evidence_state != "completed" and _event_state_requires_completed(policy):
        return "event_state_not_completed"
    if not assessment.supported_counts:
        return "unsupported_count_answer"
    return "count_dedupe_mismatch"


def _explicit_person_group_from_text(text: str, *, source: str | None = None) -> tuple[int, list[str], str] | None:
    mentions = _person_group_mentions(text, source=source)
    if len(mentions) < 2:
        return None
    best: tuple[int, int, list[str]] | None = None
    for start_index, first in enumerate(mentions):
        current = [first]
        last_end = int(first["end"])
        for mention in mentions[start_index + 1 :]:
            gap = text[last_end : int(mention["start"])]
            if len(gap) > 90 or not _person_group_connector(gap):
                break
            current.append(mention)
            last_end = int(mention["end"])
            if last_end - int(first["start"]) > 260:
                break
        people = [str(item["canonical"]) for item in current]
        unique_people = list(dict.fromkeys(people))
        if len(unique_people) < 2:
            continue
        span = int(current[-1]["end"]) - int(first["start"])
        candidate = (len(unique_people), span, unique_people)
        if best is None or candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] < best[1]):
            best = candidate
    if best is None:
        return None
    return best[0], best[2], "explicit_person_group_count_v1"


def _person_group_mentions(text: str, *, source: str | None = None) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    seen_spans: set[tuple[int, int, str]] = set()
    for canonical in _canonical_person_names():
        for alias in _person_aliases(canonical):
            for match in re.finditer(_term_pattern(alias), text, flags=re.IGNORECASE):
                key = (match.start(), match.end(), canonical)
                if key in seen_spans:
                    continue
                mentions.append({"start": match.start(), "end": match.end(), "canonical": canonical})
                seen_spans.add(key)
    speaker = _canonical_person(str(source or "")) or "__speaker__"
    for match in re.finditer(r"(?<!\w)(?:myself|me|I)(?!\w)", text, flags=re.IGNORECASE):
        key = (match.start(), match.end(), speaker)
        if key in seen_spans:
            continue
        mentions.append({"start": match.start(), "end": match.end(), "canonical": speaker})
        seen_spans.add(key)
    return sorted(mentions, key=lambda item: (int(item["start"]), int(item["end"]), str(item["canonical"])))


def _person_group_connector(gap: str) -> bool:
    if "," in gap or "&" in gap:
        return True
    return re.search(r"\b(?:and|with|plus|along with|together with)\b", gap, flags=re.IGNORECASE) is not None


def _validate_ordering_answer(answer: str, all_evidence: Sequence[QAEvidence], supporting_evidence: Sequence[QAEvidence], query: TaskQuery | None) -> str:
    if query is None:
        return "valid"
    policy = query.answer_policy or {}
    ordering = policy.get("ordering") or {}
    if not ordering:
        return "valid"
    if str(policy.get("expected_answer_type") or "").lower() not in {"person_name", "person", "actor"}:
        return "valid"
    sort_by = str(ordering.get("sort_by") or "")
    if sort_by not in {"start_second", "end_second"}:
        return "valid"
    answer_person = _canonical_person(answer)
    if not answer_person:
        return "invalid_person_answer"
    candidates: list[tuple[tuple[int, float], QAEvidence, list[str]]] = []
    policy = query.answer_policy or {}
    for item in all_evidence:
        people = _ordering_people_from_evidence(item, policy)
        if ordering.get("require_actor") and not people:
            continue
        candidates.append((_ordering_value(item, sort_by), item, people))
    if not candidates:
        return "ordering_not_supported"
    direction = str(ordering.get("direction") or "ascending").lower()
    target_value = max(value for value, _, _ in candidates) if direction == "descending" else min(value for value, _, _ in candidates)
    target_people = {person for value, _, people in candidates if value == target_value for person in people}
    if answer_person in target_people:
        return "valid"
    for item in supporting_evidence:
        value = _ordering_value(item, sort_by)
        if value == target_value and answer_person in _canonical_people_from_evidence([item]):
            return "valid"
    return "ordering_not_supported"


def _qa_uses_person_ordering(query: TaskQuery) -> bool:
    policy = query.answer_policy or {}
    ordering = policy.get("ordering") or {}
    if not ordering:
        return False
    if str(policy.get("expected_answer_type") or "").lower() not in {"person_name", "person", "actor"}:
        return False
    return str(ordering.get("sort_by") or "") in {"start_second", "end_second"}


def _qa_is_ordering_critical_evidence(query: TaskQuery, item: QAEvidence) -> bool:
    if not _qa_uses_person_ordering(query):
        return False
    return bool(_ordering_people_from_evidence(item, query.answer_policy or {}))


def _qa_ordering_priority_key(query: TaskQuery, item: QAEvidence) -> tuple[float, float, float, str]:
    policy = query.answer_policy or {}
    ordering = policy.get("ordering") or {}
    sort_by = str(ordering.get("sort_by") or "end_second")
    day, value = _ordering_value(item, sort_by)
    score = float(item.qa_evidence_score)
    if str(ordering.get("direction") or "ascending").lower() == "descending":
        return (-float(day), -float(value), -score, item.evidence_id)
    return (float(day), float(value), -score, item.evidence_id)


def _ordering_value(item: QAEvidence, sort_by: str) -> tuple[int, float]:
    return (int(item.day_num), float(getattr(item, sort_by)))


def _ordering_people_from_evidence(item: QAEvidence, policy: dict[str, Any]) -> list[str]:
    people = _canonical_people_from_evidence([item])
    positives = [str(term) for term in (policy.get("keyword_frames") or {}).get("positive") or []]
    if not positives:
        return people
    text = str(item.text_excerpt or "")
    lower = text.lower()
    term_positions: list[int] = []
    local_terms = []
    for term in positives:
        term_lower = term.lower().strip()
        if term_lower:
            local_terms.append(term_lower)
        local_terms.extend(_meaningful_tokens(term))
    for term in dict.fromkeys(local_terms):
        if not term:
            continue
        term_positions.extend(match.start() for match in re.finditer(rf"\b{re.escape(term)}\b", lower))
    if not term_positions:
        return []
    action_positions = _ordering_action_positions(text, policy)
    negative_positions = _ordering_negative_positions(text, policy)
    supported = []
    max_distance = 140
    action_distance = 180
    for person in people:
        person_positions = [
            match.start()
            for alias in _person_aliases(person)
            for match in re.finditer(rf"\b{re.escape(alias.lower())}\b", lower)
        ]
        if not any(abs(person_pos - term_pos) <= max_distance for person_pos in person_positions for term_pos in term_positions):
            continue
        if action_positions and not any(
            abs(action_pos - person_pos) <= action_distance or abs(action_pos - term_pos) <= action_distance
            for action_pos in action_positions
            for person_pos in person_positions
            for term_pos in term_positions
        ):
            continue
        if any(
            abs(negative_pos - person_pos) <= action_distance or abs(negative_pos - term_pos) <= action_distance
            for negative_pos in negative_positions
            for person_pos in person_positions
            for term_pos in term_positions
        ):
            continue
        supported.append(person)
    return sorted(set(supported))


def _ordering_action_positions(text: str, policy: dict[str, Any]) -> list[int]:
    lower = text.lower()
    terms = list(ORDERING_ACTION_TERMS)
    terms.extend(str(item) for item in (policy.get("keyword_frames") or {}).get("relation_predicates") or [])
    positions: list[int] = []
    for term in dict.fromkeys(term.lower().strip() for term in terms if str(term).strip()):
        positions.extend(match.start() for match in re.finditer(_term_pattern(term), lower, flags=re.IGNORECASE))
    return positions


def _ordering_negative_positions(text: str, policy: dict[str, Any]) -> list[int]:
    lower = text.lower()
    terms = list(ORDERING_NEGATIVE_TERMS)
    terms.extend(str(item) for item in (policy.get("keyword_frames") or {}).get("negative") or [])
    positions: list[int] = []
    for term in dict.fromkeys(term.lower().strip() for term in terms if str(term).strip()):
        if not term:
            continue
        positions.extend(match.start() for match in re.finditer(_term_pattern(term), lower, flags=re.IGNORECASE))
    return positions


def _validate_extraction_context(evidence: Sequence[QAEvidence], query: TaskQuery | None) -> str:
    if query is None:
        return "valid"
    extraction = (query.answer_policy or {}).get("extraction") or {}
    terms = [str(item) for item in extraction.get("require_context_terms") or []]
    if not terms:
        return "valid"
    support_text = " ".join(_evidence_support_text(item) for item in evidence)
    support_text = f"{support_text} {' '.join(_trusted_visual_support_text(item) for item in evidence)}"
    hits = _keyword_hits(support_text, terms)
    min_hits = int(extraction.get("min_extraction_hits") or 1)
    if len(hits) < min_hits:
        return "extraction_context_not_supported"
    return "valid"


def _validate_non_person_answer(answer: str, evidence: Sequence[QAEvidence], query: TaskQuery | None = None) -> str:
    tokens = _meaningful_tokens(answer)
    if not tokens:
        return "invalid_answer_text"
    support_text = " ".join(_evidence_support_text(item) for item in evidence).lower()
    if _answer_tokens_supported_by_text(tokens, support_text):
        return "valid"
    visual_support_text = " ".join(_trusted_visual_support_text(item) for item in evidence).lower()
    if visual_support_text and _answer_tokens_supported_by_text(tokens, visual_support_text):
        return "valid"
    if _activity_answer_supported_by_policy(answer, evidence, query):
        return "valid"
    if _global_display_rule_for_answer(answer, evidence) is not None:
        return "valid"
    return "unsupported_answer"


def _activity_answer_supported_by_policy(answer: str, evidence: Sequence[QAEvidence], query: TaskQuery | None) -> bool:
    if query is None:
        return False
    policy = query.answer_policy or {}
    expected_type = str(policy.get("expected_answer_type") or "").strip().lower()
    activity = policy.get("activity_event") or {}
    if expected_type != "activity_phrase" or not activity:
        return False
    if _validate_policy_answer_constraints(answer, evidence, query) != "valid":
        return False
    generic_activity_verbs = {
        "check",
        "checked",
        "checking",
        "find",
        "found",
        "go",
        "going",
        "inspect",
        "inspected",
        "inspecting",
        "look",
        "looked",
        "looking",
        "search",
        "searched",
        "searching",
        "see",
        "seeing",
        "show",
        "showed",
        "showing",
        "switch",
        "switched",
        "switching",
        "turn",
        "turned",
        "turning",
        "went",
    }
    required = [token for token in _meaningful_tokens(answer) if token not in generic_activity_verbs]
    if not required:
        return False
    support_text = (
        " ".join(_evidence_support_text(item) for item in evidence)
        + " "
        + " ".join(_trusted_visual_support_text(item) for item in evidence)
    ).lower()
    return _answer_tokens_supported_by_text(required, support_text)


def _validate_policy_answer_constraints(answer: str, evidence: Sequence[QAEvidence], query: TaskQuery | None) -> str:
    if query is None:
        return "valid"
    policy = query.answer_policy or {}
    activity = policy.get("activity_event") or {}
    if activity:
        support_text = " ".join(_evidence_support_text(item) for item in evidence)
        positive_hits = _keyword_hits(support_text, [str(item) for item in activity.get("positive_terms") or []])
        negative_hits = _keyword_hits(support_text, [str(item) for item in activity.get("negative_terms") or []])
        if negative_hits and (activity.get("reject_negative_context") or not positive_hits):
            return "activity_event_negative_context"
        if activity.get("positive_terms") and not positive_hits:
            return "activity_event_not_supported"
    return "valid"


def _validate_actor_resolution_answer(answer: str, evidence: Sequence[QAEvidence], query: TaskQuery | None) -> str:
    if query is None:
        return "valid"
    policy = query.answer_policy or {}
    actor_policy = policy.get("actor_resolution_policy") or {}
    if not actor_policy:
        return "valid"
    answer_person = _canonical_person(answer)
    if not answer_person:
        return "invalid_person_answer"
    candidates = _actor_candidates_from_evidence(evidence, policy)
    if answer_person in candidates:
        return "valid"
    support_text = " ".join(_evidence_support_text(item) for item in evidence)
    negative_hits = _keyword_hits(support_text, [str(item) for item in actor_policy.get("negative_actor_terms") or []])
    action_hits = _actor_action_hits(support_text, actor_policy)
    if candidates or negative_hits or not action_hits:
        return "actor_resolution_not_supported"
    return "valid"


def _actor_candidates_from_evidence(evidence: Sequence[QAEvidence], policy: dict[str, Any]) -> list[str]:
    actor_policy = policy.get("actor_resolution_policy") or {}
    candidates: set[str] = set()
    for item in evidence:
        text = item.text_excerpt or ""
        source_person = _canonical_person(item.source)
        first_person_hits = _keyword_hits(text, [str(term) for term in actor_policy.get("first_person_action_terms") or []])
        action_hits = _actor_action_hits(text, actor_policy)
        if source_person and item.source not in FIXED_ROOM_SOURCES and first_person_hits:
            candidates.add(source_person)
            continue
        if source_person and item.source not in FIXED_ROOM_SOURCES and action_hits and _trusted_visual_support_text(item):
            candidates.add(source_person)
        action_positions = [position for _, position in _keyword_hits_with_positions(text, _actor_primary_action_terms(actor_policy))]
        if not action_positions:
            continue
        lower = text.lower()
        for person in _canonical_people_from_evidence([item]):
            positions = [
                match.start()
                for alias in _person_aliases(person)
                for match in re.finditer(_term_pattern(alias), lower, flags=re.IGNORECASE)
            ]
            if any(abs(person_pos - action_pos) <= 160 for person_pos in positions for action_pos in action_positions):
                candidates.add(person)
    return sorted(candidates)


def _actor_action_hits(text: str, actor_policy: dict[str, Any]) -> list[str]:
    return _keyword_hits(text, _actor_primary_action_terms(actor_policy))


def _actor_primary_action_terms(actor_policy: dict[str, Any]) -> list[str]:
    terms = [str(item) for item in actor_policy.get("primary_action_terms") or []]
    if terms:
        return terms
    action_terms = [str(item) for item in actor_policy.get("action_terms") or []]
    primary = [term for term in action_terms if "scrap" in term.lower()]
    return primary or action_terms


def _meaningful_tokens(answer: str) -> list[str]:
    return [token for token in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", answer.lower()) if token not in STOPWORDS]


def _answer_tokens_supported_by_text(tokens: Sequence[str], support_text: str) -> bool:
    optional_compound_modifiers = {"around", "fill", "filled", "flavor", "flavored", "flavour", "flavoured", "stuffed"}
    required = [token for token in tokens if token not in optional_compound_modifiers]
    if not required:
        return False
    if not all(_support_text_has_token(support_text, token) for token in required):
        return False
    if len(required) == 1 and len(required) != len(tokens):
        fill_context = {"inside", "middle", "with", "filled", "stuffed", "flavored", "flavoured"}
        return any(_support_text_has_token(support_text, token) for token in fill_context)
    return True


def _support_text_has_token(support_text: str, token: str) -> bool:
    if not token:
        return False
    variants = {token}
    semantic_variants = {
        "lit": {"light", "lighted"},
    }
    variants.update(semantic_variants.get(token, set()))
    if token.endswith("ies") and len(token) > 3:
        variants.add(f"{token[:-3]}y")
    if token.endswith("ing") and len(token) > 4:
        variants.add(token[:-3])
    if token.endswith("ied") and len(token) > 3:
        variants.add(f"{token[:-3]}y")
    if token.endswith("ed") and len(token) > 3:
        variants.add(token[:-2])
    if token.endswith("es") and len(token) > 2:
        variants.add(token[:-2])
    if token.endswith("s") and len(token) > 1:
        variants.add(token[:-1])
    support_variants = {support_text, _normalize_spelling_for_match(support_text)}
    expanded_variants = set(variants)
    expanded_variants.update(_normalize_spelling_for_match(variant) for variant in variants)
    return any(
        re.search(_term_pattern(variant), text, flags=re.IGNORECASE)
        for text in support_variants
        for variant in expanded_variants
        if variant
    )


def _normalize_spelling_for_match(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    without_marks = "".join(char for char in normalized if not unicodedata.combining(char))
    without_marks = without_marks.lower()
    without_marks = re.sub(r"['`´’]", "", without_marks)
    without_marks = re.sub(r"[-_/]+", " ", without_marks)
    return re.sub(r"\s+", " ", without_marks).strip()


def _global_display_answer(answer: str, evidence: Sequence[QAEvidence]) -> str:
    rule = _global_display_rule_for_answer(answer, evidence)
    return str(rule["value"]) if rule else answer


def _global_display_rule_for_answer(answer: str, evidence: Sequence[QAEvidence]) -> dict[str, Any] | None:
    answer_key = _display_match_key(answer)
    if not answer_key:
        return None
    support_key = _display_match_key(
        " ".join(_evidence_support_text(item) for item in evidence)
        + " "
        + " ".join(_trusted_visual_support_text(item) for item in evidence)
    )
    for rule in GLOBAL_ANSWER_DISPLAY_NORMALIZATIONS:
        aliases = [str(item) for item in rule.get("aliases", ())]
        alias_keys = [_display_match_key(alias) for alias in aliases]
        alias_keys = [key for key in alias_keys if key]
        value_key = _display_match_key(str(rule.get("value") or ""))
        if value_key:
            alias_keys.append(value_key)
        if not any(answer_key == key or answer_key in key or key in answer_key for key in alias_keys):
            continue
        if any(re.search(_term_pattern(key), support_key, flags=re.IGNORECASE) for key in alias_keys):
            return dict(rule)
    return None


def _display_match_key(text: str) -> str:
    return " ".join(_meaningful_tokens(_normalize_spelling_for_match(text)))


def _evidence_support_text(evidence: QAEvidence) -> str:
    parts = [evidence.text_excerpt]
    for mention in evidence.entity_mentions or []:
        parts.extend(str(mention.get(key) or "") for key in ("text", "canonical"))
    for relation in evidence.relation_candidates or []:
        parts.extend(str(relation.get(key) or "") for key in ("subject", "predicate", "object", "text_window"))
    return " ".join(parts)


def _trusted_visual_support_text(evidence: QAEvidence) -> str:
    parts: list[str] = []
    score_parts = evidence.qa_score_parts or {}
    for key in ("visual_answer_terms", "visual_probe_terms", "trusted_visual_labels"):
        value = score_parts.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif isinstance(value, str):
            parts.append(value)
    return " ".join(parts)


def _evidence_mentions_person(evidence: QAEvidence, canonical: str) -> bool:
    support = _evidence_support_text(evidence)
    for alias in _person_aliases(canonical):
        if re.search(rf"\b{re.escape(alias)}\b", support, re.IGNORECASE):
            return True
    return False


def _canonical_people_from_evidence(evidence: Sequence[QAEvidence]) -> list[str]:
    people: set[str] = set()
    for item in evidence:
        source_person = _canonical_person(item.source)
        if source_person and item.source not in FIXED_ROOM_SOURCES:
            people.add(source_person)
        for mention in item.entity_mentions or []:
            if str(mention.get("type") or "").lower() not in {"", "person"}:
                continue
            for key in ("canonical", "text"):
                person = _canonical_person(str(mention.get(key) or ""))
                if person:
                    people.add(person)
        for mention in _entity_mentions(item.text_excerpt):
            person = _canonical_person(str(mention.get("canonical") or mention.get("text") or ""))
            if person:
                people.add(person)
    return sorted(people)


def _canonical_person(value: str) -> str | None:
    normalized = str(value).strip().rstrip(".")
    if not normalized:
        return None
    alias_lookup = {
        alias.lower(): canonical
        for canonical in _canonical_person_names()
        for alias in _person_aliases(canonical)
    }
    mapped = alias_lookup.get(normalized.lower())
    if mapped:
        return mapped
    return None


def _person_aliases(canonical: str) -> list[str]:
    normalized = str(canonical).strip()
    mapped = PERSON_ALIAS_MAP.get(normalized)
    if mapped:
        return list(mapped)
    return [normalized] if normalized else []


def _canonical_person_names() -> list[str]:
    names = set(PERSON_NAMES)
    names.discard("Bjorn")
    return sorted(names)


def _parse_count_answer(answer: str) -> int | None:
    text = answer.strip().lower()
    match = re.search(r"\b\d+\b", text)
    if match:
        return int(match.group(0))
    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", text):
            return value
    return None


def _evidence_by_id(evidence: Sequence[QAEvidence], evidence_id: str) -> QAEvidence:
    for item in evidence:
        if item.evidence_id == evidence_id:
            return item
    raise KeyError(evidence_id)


def _entity_mentions(text: str) -> list[dict]:
    mentions = []
    seen = set()
    for canonical in _canonical_person_names():
        for alias in _person_aliases(canonical):
            if re.search(rf"\b{re.escape(alias)}\b", text, re.IGNORECASE):
                key = (canonical, alias.lower())
                if key not in seen:
                    mentions.append({"type": "person", "text": alias, "canonical": canonical})
                    seen.add(key)
    return mentions


def _relation_candidates(query: TaskQuery, text: str) -> list[dict]:
    policy = query.answer_policy or {}
    predicates = (policy.get("keyword_frames") or {}).get("relation_predicates") or []
    relation_direction = policy.get("relation_direction") or {}
    object_aliases = _relation_object_aliases(policy, str(relation_direction.get("object") or ""))
    result = []
    for predicate in predicates:
        for match in re.finditer(_term_pattern(str(predicate)), text, flags=re.IGNORECASE):
            window_start = max(0, match.start() - 180)
            window_end = min(len(text), match.end() + 220)
            candidate = {"predicate": predicate, "text_window": text[window_start:window_end]}
            subject = _nearest_person_before(text[: match.start()])
            if subject:
                candidate["subject"] = subject
            object_value = _first_matching_alias(text[match.end() : window_end], object_aliases)
            if object_value:
                candidate["object"] = _canonical_person(object_value) or object_value
            if subject and object_value:
                candidate["direction_supported"] = True
            result.append(candidate)
    return result


def _relation_object_aliases(policy: dict[str, Any], object_name: str) -> list[str]:
    aliases: set[str] = set()
    if object_name:
        aliases.add(object_name)
    entity_aliases = policy.get("entity_aliases") or {}
    object_canonical = _canonical_person(object_name)
    for key, values in entity_aliases.items():
        key_text = str(key)
        key_canonical = _canonical_person(key_text)
        if key_text == object_name or (object_canonical and key_canonical == object_canonical):
            aliases.add(key_text)
            if isinstance(values, SequenceABC) and not isinstance(values, (str, bytes, bytearray)):
                aliases.update(str(item) for item in values)
    if object_canonical == "Björn" or object_name in {"Björn", "Bjorn"}:
        aliases.update({"Björn", "Bjorn"})
    return sorted(alias for alias in aliases if alias)


def _relation_candidate_matches_direction(relation: dict[str, Any], subject: str, object_aliases: Sequence[str], predicates: Sequence[str]) -> bool:
    relation_subject = _canonical_person(str(relation.get("subject") or ""))
    if relation_subject != subject:
        return False
    relation_predicate = str(relation.get("predicate") or "")
    if not any(_predicate_matches(relation_predicate, predicate) for predicate in predicates):
        return False
    relation_object = str(relation.get("object") or "")
    if relation_object and any(_alias_matches(relation_object, alias) for alias in object_aliases):
        return True
    return _text_supports_relation_direction(str(relation.get("text_window") or ""), subject, object_aliases, predicates)


def _text_supports_relation_direction(text: str, subject: str, object_aliases: Sequence[str], predicates: Sequence[str]) -> bool:
    if not text:
        return False
    for subject_alias in _person_aliases(subject):
        for predicate in predicates:
            predicate_text = str(predicate)
            for object_alias in object_aliases:
                if _predicate_includes_object(predicate_text, object_alias):
                    if _ordered_terms_present(text, [subject_alias, predicate_text]):
                        return True
                elif _ordered_terms_present(text, [subject_alias, predicate_text, object_alias]):
                    return True
                if _passive_relation_present(text, subject_alias, predicate_text, object_alias):
                    return True
    return False


def _combined_teaching_relation_supported(subject: str, evidence: Sequence[QAEvidence], object_aliases: Sequence[str], predicates: Sequence[str]) -> bool:
    if not any(re.search(r"\b(?:teach|taught|show|showed|explain|explained)\b", str(predicate), flags=re.IGNORECASE) for predicate in predicates):
        return False
    object_people = {_canonical_person(alias) for alias in object_aliases}
    object_people.discard(None)
    if not object_people:
        return False
    reverse_object_aliases = _person_aliases(subject)
    for item in evidence:
        text = item.text_excerpt or ""
        if any(_text_supports_relation_direction(text, object_person, reverse_object_aliases, predicates) for object_person in object_people if object_person):
            return False
    subject_supported = any(_evidence_mentions_person(item, subject) or _canonical_person(item.source) == subject for item in evidence)
    object_supported = any(
        any(_evidence_mentions_person(item, object_person) or _canonical_person(item.source) == object_person for object_person in object_people if object_person)
        for item in evidence
    )
    if not subject_supported or not object_supported:
        return False
    support_text = " ".join(_evidence_support_text(item) for item in evidence)
    teaching_hits = _keyword_hits(
        support_text,
        ["card game", "Schnapsen", "Schnappschen", "schnappschen", "Rummy", "poker", "rule", "rules", "how do you play", "teach", "taught", "explained", "showed"],
    )
    return len(teaching_hits) >= 2


def _predicate_matches(actual: str, expected: str) -> bool:
    actual_norm = actual.strip().lower()
    expected_norm = expected.strip().lower()
    return bool(actual_norm and expected_norm and (actual_norm == expected_norm or actual_norm in expected_norm or expected_norm in actual_norm))


def _alias_matches(value: str, alias: str) -> bool:
    value_canonical = _canonical_person(value)
    alias_canonical = _canonical_person(alias)
    if value_canonical and alias_canonical:
        return value_canonical == alias_canonical
    return value.strip().lower() == alias.strip().lower()


def _predicate_includes_object(predicate: str, object_alias: str) -> bool:
    return re.search(_term_pattern(object_alias), predicate, flags=re.IGNORECASE) is not None


def _ordered_terms_present(text: str, terms: Sequence[str], *, max_gap: int = 180) -> bool:
    pattern = (rf".{{0,{max_gap}}}?").join(_term_pattern(term) for term in terms if term)
    return re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) is not None


def _passive_relation_present(text: str, subject_alias: str, predicate: str, object_alias: str) -> bool:
    if not re.search(r"\b(taught|teach|showed|shown|explained)\b", predicate, flags=re.IGNORECASE):
        return False
    pattern = (
        _term_pattern(object_alias)
        + r".{0,120}?\b(?:was|is|got|gets|being)?\s*(?:taught|shown|showed|explained)\b.{0,60}?\bby\b.{0,120}?"
        + _term_pattern(subject_alias)
    )
    return re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) is not None


def _nearest_person_before(text: str) -> str | None:
    best: tuple[int, str] | None = None
    for canonical in _canonical_person_names():
        for alias in _person_aliases(canonical):
            for match in re.finditer(_term_pattern(alias), text, flags=re.IGNORECASE):
                if best is None or match.start() > best[0]:
                    best = (match.start(), canonical)
    return best[1] if best else None


def _first_matching_alias(text: str, aliases: Sequence[str]) -> str | None:
    best: tuple[int, str] | None = None
    for alias in aliases:
        match = re.search(_term_pattern(alias), text, flags=re.IGNORECASE)
        if match and (best is None or match.start() < best[0]):
            best = (match.start(), alias)
    return best[1] if best else None


def _term_pattern(term: str) -> str:
    escaped = re.escape(term.strip())
    escaped = re.sub(r"\\\s+", r"\\s+", escaped)
    return rf"(?<!\w){escaped}(?!\w)"


def _time_filter_fit(query: TaskQuery, day: int, start: float, end: float) -> float:
    filters = (query.answer_policy or {}).get("time_filters") or {}
    if not filters:
        return 1.0
    if filters.get("day_filter") and int(filters["day_filter"]) != int(day):
        return 0.0
    window = filters.get("time_window")
    if not window:
        return 1.0
    from .time_utils import parse_hms, overlap_duration

    left = (float(start), float(end))
    right = (float(parse_hms(str(window[0]))), float(parse_hms(str(window[1]))))
    return min(1.0, overlap_duration(left, right) / max(end - start, 1.0))


def _qa_text_floor(query: TaskQuery) -> float:
    return float(QA_TEXT_FLOORS.get(query.task_id, 0.35))


def _qa_effective_keep_top(query: TaskQuery, keep_top: int) -> int:
    try:
        requested = int(keep_top)
    except (TypeError, ValueError):
        requested = 20
    task_limit = int(QA_KEEP_TOP_LIMITS.get(query.task_id, 12))
    return max(1, min(task_limit, requested))


def _qa_gate_audit(
    query: TaskQuery,
    hit: TextHit,
    text: str,
    *,
    entity_mentions: Sequence[dict[str, Any]],
    relation_candidates: Sequence[dict[str, Any]],
    time_filter_fit: float,
    text_floor: float,
) -> dict[str, Any]:
    policy = query.answer_policy or {}
    keyword_frames = policy.get("keyword_frames") or {}
    positives = [str(item) for item in keyword_frames.get("positive") or []]
    positive_hits = _keyword_hits(text, positives)
    reranker = float(hit.reranker_score if hit.reranker_score is not None else hit.dense_score if hit.dense_score is not None else hit.sparse_score if hit.sparse_score is not None else 0.0)
    lexical_pass = not positives or bool(positive_hits) or reranker >= min(0.80, float(text_floor) + 0.20)
    expected_type = str(policy.get("expected_answer_type") or "").lower()
    entity_terms = _entity_policy_terms(policy)
    entity_hits = _keyword_hits(text, entity_terms)
    entity_required = bool(entity_terms) or expected_type in {"person_name", "person", "actor", "count", "number", "integer"}
    source_is_person = _canonical_person(hit.source) is not None
    entity_pass = not entity_required or bool(entity_mentions) or bool(entity_hits) or source_is_person
    relation_direction = policy.get("relation_direction") or {}
    relation_required = bool(relation_direction.get("require_direction"))
    relation_pass = not relation_required or bool(relation_candidates)
    policy_floor_scores = {
        "event_state": float(_event_state_audit(policy, text).get("score") or 0.0),
        "activity_binding": float(_activity_event_audit(policy, text, hit.source).get("score") or 0.0),
        "actor_binding": float(_actor_binding_audit(policy, text, hit.source, "").get("score") or 0.0),
    }
    policy_floor_score = max(policy_floor_scores.values(), default=0.0)
    text_gate_pass = reranker >= float(text_floor) or policy_floor_score >= 0.80
    return {
        "text_gate": {
            "pass": bool(text_gate_pass),
            "reranker_score": reranker,
            "text_floor": float(text_floor),
            "policy_floor_score": policy_floor_score,
            "policy_floor_scores": policy_floor_scores,
        },
        "lexical_gate": {"pass": bool(lexical_pass), "positive_keyword_hits": positive_hits, "positive_keyword_count": len(positives)},
        "time_gate": {"pass": not _has_time_filter(query) or float(time_filter_fit) > 0.0, "time_filter_fit": float(time_filter_fit)},
        "entity_gate": {"pass": bool(entity_pass), "required": bool(entity_required), "entity_hits": entity_hits, "entity_mention_count": len(entity_mentions), "source_is_person": bool(source_is_person)},
        "relation_gate": {"pass": bool(relation_pass), "required": bool(relation_required), "relation_candidate_count": len(relation_candidates)},
    }


def _qa_select_kept_evidence(query: TaskQuery, evidence: Sequence[QAEvidence], *, keep_top: int) -> tuple[list[QAEvidence], list[dict]]:
    ordered = sorted(evidence, key=lambda item: (-float(item.qa_evidence_score), item.day_num, item.start_second, item.evidence_id))
    if len(ordered) <= keep_top:
        for item in ordered:
            item.qa_score_parts["keep_decision"] = "kept"
            item.qa_score_parts["reducer_cluster"] = _qa_reducer_cluster(item)
            item.qa_score_parts["ordering_critical"] = _qa_is_ordering_critical_evidence(query, item)
            item.qa_score_parts["ordering_people"] = _ordering_people_from_evidence(item, query.answer_policy or {}) if _qa_uses_person_ordering(query) else []
        return list(ordered), []
    selected: list[QAEvidence] = []
    if _qa_uses_person_ordering(query):
        selected.extend(sorted(
            (item for item in ordered if _qa_is_ordering_critical_evidence(query, item)),
            key=lambda item: _qa_ordering_priority_key(query, item),
        )[:keep_top])
    elif query.task_id in {"MM26-QA01", "MM26-QA12"}:
        by_cluster: dict[str, QAEvidence] = {}
        for item in ordered:
            cluster = _qa_reducer_cluster(item)
            previous = by_cluster.get(cluster)
            if previous is None or float(item.qa_evidence_score) > float(previous.qa_evidence_score):
                by_cluster[cluster] = item
        cluster_rows = sorted(by_cluster.values(), key=lambda item: (item.day_num, item.start_second, -float(item.qa_evidence_score), item.evidence_id))
        selected.extend(cluster_rows[:keep_top])
    for item in ordered:
        if len(selected) >= keep_top:
            break
        if item not in selected:
            selected.append(item)
    selected_ids = {item.evidence_id for item in selected}
    pruned = []
    for item in selected:
        item.qa_score_parts["keep_decision"] = "kept"
        item.qa_score_parts["reducer_cluster"] = _qa_reducer_cluster(item)
        item.qa_score_parts["ordering_critical"] = _qa_is_ordering_critical_evidence(query, item)
        item.qa_score_parts["ordering_people"] = _ordering_people_from_evidence(item, query.answer_policy or {}) if _qa_uses_person_ordering(query) else []
    for item in ordered:
        if item.evidence_id in selected_ids:
            continue
        item.qa_score_parts["keep_decision"] = "pruned_by_keep_top"
        item.qa_score_parts["reducer_cluster"] = _qa_reducer_cluster(item)
        item.qa_score_parts["ordering_critical"] = _qa_is_ordering_critical_evidence(query, item)
        item.qa_score_parts["ordering_people"] = _ordering_people_from_evidence(item, query.answer_policy or {}) if _qa_uses_person_ordering(query) else []
        row = item.asdict()
        row["rejected_reason"] = "keep_top_pruned"
        pruned.append(row)
    return selected, pruned


def _qa_reducer_cluster(item: QAEvidence) -> str:
    return f"day{int(item.day_num)}:{item.source}:hour{int(float(item.start_second) // 3600):02d}"


def _qa_score(
    query: TaskQuery,
    hit: TextHit,
    text: str,
    *,
    entity_mentions: Sequence[dict[str, Any]] = (),
    relation_candidates: Sequence[dict[str, Any]] = (),
    time_filter_fit: float = 1.0,
    image_refs: Sequence[str] = (),
    visual_probe_score: float | None = None,
) -> tuple[float, dict[str, Any]]:
    weights, weight_source = _qa_score_weights(query)
    components, available, debug = _qa_score_components(
        query,
        hit,
        text,
        entity_mentions=entity_mentions,
        relation_candidates=relation_candidates,
        time_filter_fit=time_filter_fit,
        image_refs=image_refs,
        visual_probe_score=visual_probe_score,
    )
    policy_weights = _qa_policy_score_weights(query)
    combined_weights = dict(weights)
    combined_weights.update(policy_weights)
    positive_component_keys = ("reranker", "visual_probe", "event_state", "activity_binding", "actor_binding")
    used_weights = {key: float(combined_weights[key]) for key in (*positive_component_keys, "text_neg") if key in combined_weights and key in components}
    skipped = [key for key in combined_weights if key not in used_weights]
    positive_denominator = max(sum(float(combined_weights.get(key, 0.0)) for key in positive_component_keys), 1e-6)
    positive_score = sum(components[key] * float(combined_weights.get(key, 0.0)) for key in positive_component_keys if key in components)
    score = (positive_score - components["text_neg"] * float(combined_weights.get("text_neg", 0.50))) / positive_denominator
    parts = {
        "score_definition": "reranker_first_with_policy_components_v2",
        "weight_source": weight_source,
        "raw_weights": combined_weights,
        "base_raw_weights": weights,
        "policy_raw_weights": policy_weights,
        "used_weights": used_weights,
        "skipped_profile_keys": skipped,
        "available_components": available,
        "components": components,
        "weight_sum": positive_denominator,
        "implemented_weight_sum": sum(used_weights.values()),
        "missing_components_treated_as_zero": False,
        "debug": debug,
    }
    return float(max(0.0, min(1.0, score))), parts


def _qa_score_weights(query: TaskQuery) -> tuple[dict[str, float], str]:
    profile = ((query.resolved_profiles or {}).get("scoring_profile") or {}).get("value") or {}
    weights = profile.get("evidence_weights") or {}
    if weights:
        return {str(key): float(value) for key, value in weights.items()}, ((query.resolved_profiles or {}).get("scoring_profile") or {}).get("source", "resolved_profiles.scoring_profile.value.evidence_weights")
    return {
        "reranker": 0.75,
        "visual_probe": 0.25,
        "text_neg": 0.50,
    }, "missing_profile_reranker_first_fallback"


def _qa_policy_score_weights(query: TaskQuery) -> dict[str, float]:
    policy = query.answer_policy or {}
    weights: dict[str, float] = {}
    if policy.get("event_state"):
        weights["event_state"] = 0.45
    if policy.get("activity_event"):
        weights["activity_binding"] = 0.65
    if policy.get("actor_resolution_policy"):
        weights["actor_binding"] = 0.60
    return weights


def _qa_score_components(
    query: TaskQuery,
    hit: TextHit,
    text: str,
    *,
    entity_mentions: Sequence[dict[str, Any]],
    relation_candidates: Sequence[dict[str, Any]],
    time_filter_fit: float,
    image_refs: Sequence[str],
    visual_probe_score: float | None,
) -> tuple[dict[str, float], dict[str, bool], dict[str, Any]]:
    policy = query.answer_policy or {}
    keyword_frames = policy.get("keyword_frames") or {}
    positives = [str(item) for item in keyword_frames.get("positive") or []]
    negatives = [str(item) for item in keyword_frames.get("negative") or []]
    relation_predicates = [str(item) for item in keyword_frames.get("relation_predicates") or []]
    positive_hits = _keyword_hits(text, positives)
    negative_hits = _keyword_hits(text, negatives)
    entity_terms = _entity_policy_terms(policy)
    entity_hits = _keyword_hits(text, entity_terms)
    expected_type = str(policy.get("expected_answer_type") or "").lower()
    entity_available = bool(entity_terms) or expected_type in {"person_name", "person", "actor", "count", "number", "integer"}
    person_entity_score = min(1.0, len(entity_mentions) / 2.0) if expected_type in {"person_name", "person", "actor", "count", "number", "integer"} else 0.0
    entity_policy_score = min(1.0, len(entity_hits) / max(len(entity_terms), 1)) if entity_terms else 0.0
    keyword_fallback_score = min(1.0, len(positive_hits) / max(len(positives), 1)) if positives and not entity_terms else 0.0
    entity_match = max(person_entity_score, entity_policy_score, keyword_fallback_score)
    relation_score = min(1.0, len(relation_candidates) / max(len(relation_predicates), 1)) if relation_predicates else 0.0
    positive_context_score = min(1.0, len(positive_hits) / max(len(positives), 1)) if positives else 0.0
    event_context = max(relation_score, positive_context_score)
    if negative_hits:
        event_context *= 0.5
    visual_probe_raw = visual_probe_score if visual_probe_score is not None else hit.provenance.get("visual_probe_score")
    visual_probe = _clip01(float(visual_probe_raw)) if visual_probe_raw is not None else 0.0
    trusted_visual_text = " ".join(str(item) for item in (policy.get("visual_probe_terms") or [])) if visual_probe_raw is not None else ""
    event_state = _event_state_audit(policy, text)
    activity_binding = _activity_event_audit(policy, text, hit.source)
    actor_binding = _actor_binding_audit(policy, text, hit.source, trusted_visual_text)
    policy_negative_hits = []
    if event_state.get("state") == "future_plan":
        policy_negative_hits.extend(event_state.get("future_hits") or [])
    policy_negative_hits.extend(activity_binding.get("negative_hits") or [])
    policy_negative_hits.extend(actor_binding.get("negative_hits") or [])
    components = {
        "reranker": _clip01(float(hit.reranker_score if hit.reranker_score is not None else 0.0)),
        "dense_sparse": _clip01(max(float(hit.dense_score or 0.0), float(hit.sparse_score or 0.0))),
        "entity_match": _clip01(entity_match),
        "time_filter": _clip01(float(time_filter_fit)),
        "event_context": _clip01(event_context),
        "visual_probe": _clip01(visual_probe),
        "event_state": _clip01(float(event_state.get("score") or 0.0)),
        "activity_binding": _clip01(float(activity_binding.get("score") or 0.0)),
        "actor_binding": _clip01(float(actor_binding.get("score") or 0.0)),
        "text_neg": 1.0 if negative_hits or policy_negative_hits else _clip01(float(hit.negative_reranker_score or 0.0)),
    }
    available = {
        "reranker": hit.reranker_score is not None,
        "dense_sparse": hit.dense_score is not None or hit.sparse_score is not None,
        "entity_match": entity_available,
        "time_filter": _has_time_filter(query),
        "event_context": bool(positives or relation_predicates or negatives),
        "visual_probe": visual_probe_raw is not None,
        "event_state": bool(event_state.get("available")),
        "activity_binding": bool(activity_binding.get("available")),
        "actor_binding": bool(actor_binding.get("available")),
        "text_neg": bool(negative_hits or policy_negative_hits) or hit.negative_reranker_score is not None,
    }
    debug = {
        "positive_keyword_hits": positive_hits,
        "negative_keyword_hits": negative_hits,
        "policy_negative_hits": policy_negative_hits,
        "entity_policy_terms": entity_terms,
        "entity_hits": entity_hits,
        "relation_candidate_count": len(relation_candidates),
        "image_ref_count": len(image_refs),
        "visual_probe_score": visual_probe_raw,
        "event_state": event_state,
        "activity_binding": activity_binding,
        "actor_binding": actor_binding,
    }
    return components, available, debug


def _qa_visual_probe_index(visual_hits: Sequence[FrameHit]) -> dict[str, Any]:
    by_frame: dict[str, list[dict[str, Any]]] = {}
    by_day_source: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for hit in visual_hits:
        score = _clip01(float(hit.patch_score if hit.patch_score is not None else hit.cls_score))
        row = {
            "embedding_id": hit.embedding_id,
            "frame_atom_id": hit.frame_atom_id,
            "day": int(hit.day),
            "source": str(hit.source),
            "absolute_second": float(hit.absolute_second),
            "score": score,
            "patch_score": hit.patch_score,
            "cls_score": hit.cls_score,
            "prompt": hit.prompt,
            "rank": hit.rank,
        }
        if hit.frame_atom_id:
            by_frame.setdefault(hit.frame_atom_id, []).append(row)
        by_day_source.setdefault((int(hit.day), str(hit.source)), []).append(row)
    return {"by_frame": by_frame, "by_day_source": by_day_source, "hit_count": len(visual_hits)}


def _qa_visual_probe_for_evidence(index: dict[str, Any], frame_ids: Sequence[str], day: int, source: str, start: float, end: float) -> tuple[float | None, dict[str, Any]]:
    matches = []
    by_frame = index.get("by_frame") or {}
    for frame_id in frame_ids:
        matches.extend(by_frame.get(str(frame_id), []))
    match_policy = "representative_frame_atom"
    if not matches:
        window_start = max(0.0, float(start) - 5.0)
        window_end = float(end) + 5.0
        candidates = (index.get("by_day_source") or {}).get((int(day), str(source)), [])
        matches = [item for item in candidates if window_start <= float(item["absolute_second"]) <= window_end]
        match_policy = "same_source_time_window"
    temporal_alignment = "direct"
    if not matches:
        window_start = max(0.0, float(start) - 300.0)
        window_end = float(end) + 300.0
        candidates = (index.get("by_day_source") or {}).get((int(day), str(source)), [])
        matches = [item for item in candidates if window_start <= float(item["absolute_second"]) <= window_end]
        match_policy = "same_source_nearby_5min"
        temporal_alignment = "weak"
    if not matches:
        return None, {"status": "no_match", "visual_probe_hit_count": int(index.get("hit_count") or 0), "match_policy": match_policy}
    best = max(matches, key=lambda item: (float(item["score"]), -int(item.get("rank") or 0), str(item.get("embedding_id") or "")))
    gap_seconds = 0.0
    if float(best["absolute_second"]) < float(start):
        gap_seconds = float(start) - float(best["absolute_second"])
    elif float(best["absolute_second"]) > float(end):
        gap_seconds = float(best["absolute_second"]) - float(end)
    return float(best["score"]), {
        "status": "matched",
        "match_policy": match_policy,
        "temporal_alignment": temporal_alignment,
        "time_gap_seconds": gap_seconds,
        "matched_count": len(matches),
        "best_embedding_id": best.get("embedding_id"),
        "best_frame_atom_id": best.get("frame_atom_id"),
        "best_absolute_second": best.get("absolute_second"),
        "best_prompt": best.get("prompt"),
        "best_rank": best.get("rank"),
        "score_source": "patch_score" if best.get("patch_score") is not None else "cls_score",
    }


def _entity_policy_terms(policy: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    aliases = policy.get("entity_aliases") or {}
    if isinstance(aliases, dict):
        for key, values in aliases.items():
            terms.append(str(key))
            terms.extend(str(value) for value in values or [])
    return [term for term in dict.fromkeys(term.strip() for term in terms) if term]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _image_ref_for_frame(config: AppConfig, store: EventStore, frame_atom_id: str) -> str | None:
    frame = getattr(store, "frame_by_id", {}).get(str(frame_atom_id))
    if frame:
        path = resolve_frame_path(config, str(frame.get("frame_rel_path") or ""), str(frame.get("image_path_ref") or ""))
        if path.exists():
            return path.as_posix()
    return None
