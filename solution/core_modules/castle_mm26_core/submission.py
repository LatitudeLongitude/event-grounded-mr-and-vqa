from __future__ import annotations

import json
import re
import shutil
import unicodedata
import zipfile
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data import ACCEPTED_SOURCES, CoverageIndex, EventStore, filter_text_atom_ids_for_source, text_atom_ids_for_source_interval
from .io_utils import atomic_write_csv, atomic_write_json, atomic_write_jsonl, atomic_write_text, now_utc, read_yaml, sha256_file
from .models import AnswerResult, EvidenceBundle, QAEvidence
from .qa import render_source_time_ref
from .registry import expected_task_ids
from .time_utils import parse_hms


EIS_FIELDS = ("task", "day", "starttime", "endtime", "source")
QA_FIELDS = ("task", "answer")
OFFICIAL_QA_MODEL = "openai/gpt-5.5"
QA_REF_RE = re.compile(
    r"(Allie|Bao|Bjorn|Cathal|Florian|Kitchen|Klaus|Living1|Living2|Luca|Meeting|Onanong|Reading|Stevan|Tien|Werner) day [1-4] (?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]-(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]"
)
PERSON_NAMES = {"Allie", "Bao", "Björn", "Bjorn", "Cathal", "Florian", "Klaus", "Luca", "Onanong", "Stevan", "Tien", "Werner"}
FIXED_ROOM_SOURCES = {"Kitchen", "Living1", "Living2", "Meeting", "Reading"}
NUMBER_WORD_VALUES = {
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
NUMBER_WORDS = set(NUMBER_WORD_VALUES)
ALLOWED_QA_ANSWER_TYPES = {"person_name", "count", "object_name", "activity_phrase", "song_title", "food_name"}
REQUIRED_MLLM_RAW_KEYS = {"answer", "answer_type", "confidence", "supporting_evidence_ids", "rationale_short"}


def write_submission_bundle(
    output_dir: Path,
    *,
    eis_rows: Sequence[dict],
    ois_rows: Sequence[dict],
    qa_results: Sequence[AnswerResult],
    evidence: Sequence[EvidenceBundle],
    coverage: CoverageIndex,
    run_id: str,
    write_zip: bool,
    manifest: dict,
    event_store: EventStore | None = None,
    qa_evidence_by_task: Mapping[str, Sequence[QAEvidence]] | None = None,
    sort_interval_rows_by_final_score: bool = True,
) -> dict:
    submissions = output_dir / "submissions"
    submissions.mkdir(parents=True, exist_ok=True)
    event_csv = submissions / "event_instance_search.csv"
    object_csv = submissions / "object_instance_search.csv"
    qa_csv = submissions / "question_answering.csv"
    submission_eis_rows = _drop_interval_rows_missing_required_text(
        _sanitize_interval_row_text_atoms(_submission_interval_rows(eis_rows, sort_by_final_score=sort_interval_rows_by_final_score), event_store)
    )
    submission_ois_rows = _sanitize_interval_row_text_atoms(_submission_interval_rows(ois_rows, sort_by_final_score=sort_interval_rows_by_final_score), event_store)
    submission_evidence = _sanitize_evidence_text_atoms(evidence, event_store)
    submitted_evidence_keys = {(str(row.get("task_id") or row.get("task") or ""), str(row.get("csv_row_id") or "")) for row in [*submission_eis_rows, *submission_ois_rows]}
    submission_evidence = [item for item in submission_evidence if (item.task_id, item.csv_row_id) in submitted_evidence_keys]
    atomic_write_csv(event_csv, [_csv_interval_row(row) for row in submission_eis_rows], EIS_FIELDS)
    atomic_write_csv(object_csv, [_csv_interval_row(row) for row in submission_ois_rows], EIS_FIELDS)
    atomic_write_csv(qa_csv, [{"task": result.task_id, "answer": result.answer_text} for result in sorted(qa_results, key=lambda item: item.task_id)], QA_FIELDS)
    validation = validate_submission(
        event_rows=submission_eis_rows,
        object_rows=submission_ois_rows,
        qa_results=qa_results,
        evidence=submission_evidence,
        coverage=coverage,
        manifest=manifest,
        event_store=event_store,
        qa_evidence_by_task=qa_evidence_by_task,
        run_root=output_dir,
    )
    validation["files"] = {
        "event_instance_search.csv": _file_info(event_csv),
        "object_instance_search.csv": _file_info(object_csv),
        "question_answering.csv": _file_info(qa_csv),
    }
    validation["interval_row_order"] = _interval_row_order_report(sort_interval_rows_by_final_score)
    validation["final_score_ranked_audit"] = write_final_score_ranked_audit(
        output_dir,
        eis_rows=submission_eis_rows,
        ois_rows=submission_ois_rows,
        qa_results=qa_results,
        qa_evidence_by_task=qa_evidence_by_task or {},
        task_ids=expected_task_ids(),
    )
    zip_path = submissions / f"castle_mm26_fully_automatic_{run_id}.zip"
    if validation["packaging_allowed"] and write_zip:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(event_csv, "event_instance_search.csv")
            archive.write(object_csv, "object_instance_search.csv")
            archive.write(qa_csv, "question_answering.csv")
        validation["zip"] = _file_info(zip_path)
    else:
        if zip_path.exists():
            zip_path.unlink()
        validation["zip"] = {"path": zip_path.as_posix(), "exists": False, "reason": "gate_failed_or_write_zip_false"}
    atomic_write_json(submissions / "validation_report.json", validation)
    validation["openreview_package"] = write_openreview_package_checklist(output_dir, run_id=run_id, validation=validation, manifest=manifest)
    atomic_write_json(submissions / "validation_report.json", validation)
    return validation


def write_diagnostic_smoke_bundle(
    output_dir: Path,
    *,
    eis_rows: Sequence[dict],
    ois_rows: Sequence[dict],
    qa_results: Sequence[AnswerResult],
    evidence: Sequence[EvidenceBundle],
    coverage: CoverageIndex,
    run_id: str,
    manifest: dict,
    selected_task_ids: Sequence[str],
    event_store: EventStore | None = None,
    qa_evidence_by_task: Mapping[str, Sequence[QAEvidence]] | None = None,
    sort_interval_rows_by_final_score: bool = True,
) -> dict:
    submissions = output_dir / "submissions" / "fullflow_smoke"
    submissions.mkdir(parents=True, exist_ok=True)
    event_csv = submissions / "event_instance_search.csv"
    object_csv = submissions / "object_instance_search.csv"
    qa_csv = submissions / "question_answering.csv"
    submission_eis_rows = _drop_interval_rows_missing_required_text(
        _sanitize_interval_row_text_atoms(_submission_interval_rows(eis_rows, sort_by_final_score=sort_interval_rows_by_final_score), event_store)
    )
    submission_ois_rows = _sanitize_interval_row_text_atoms(_submission_interval_rows(ois_rows, sort_by_final_score=sort_interval_rows_by_final_score), event_store)
    submission_evidence = _sanitize_evidence_text_atoms(evidence, event_store)
    submitted_evidence_keys = {(str(row.get("task_id") or row.get("task") or ""), str(row.get("csv_row_id") or "")) for row in [*submission_eis_rows, *submission_ois_rows]}
    submission_evidence = [item for item in submission_evidence if (item.task_id, item.csv_row_id) in submitted_evidence_keys]
    atomic_write_csv(event_csv, [_csv_interval_row(row) for row in submission_eis_rows], EIS_FIELDS)
    atomic_write_csv(object_csv, [_csv_interval_row(row) for row in submission_ois_rows], EIS_FIELDS)
    atomic_write_csv(qa_csv, [{"task": result.task_id, "answer": result.answer_text} for result in sorted(qa_results, key=lambda item: item.task_id)], QA_FIELDS)
    selected = list(selected_task_ids)
    validation = validate_submission(
        event_rows=submission_eis_rows,
        object_rows=submission_ois_rows,
        qa_results=qa_results,
        evidence=submission_evidence,
        coverage=coverage,
        manifest=manifest,
        event_store=event_store,
        qa_evidence_by_task=qa_evidence_by_task,
        expected_task_ids_override=selected,
        run_root=output_dir,
    )
    validation.update(
        {
            "package_kind": "diagnostic_fullflow_smoke",
            "official_submission": False,
            "selected_task_ids": selected,
            "expected_task_ids": selected,
            "files": {
                "event_instance_search.csv": _file_info(event_csv),
                "object_instance_search.csv": _file_info(object_csv),
                "question_answering.csv": _file_info(qa_csv),
            },
            "interval_row_order": _interval_row_order_report(sort_interval_rows_by_final_score),
        }
    )
    validation["final_score_ranked_audit"] = write_final_score_ranked_audit(
        output_dir,
        eis_rows=submission_eis_rows,
        ois_rows=submission_ois_rows,
        qa_results=qa_results,
        qa_evidence_by_task=qa_evidence_by_task or {},
        task_ids=selected,
    )
    zip_path = submissions / f"castle_mm26_fullflow_smoke_{run_id}.zip"
    if validation["valid"]:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(event_csv, "event_instance_search.csv")
            archive.write(object_csv, "object_instance_search.csv")
            archive.write(qa_csv, "question_answering.csv")
        validation["zip"] = _file_info(zip_path)
    else:
        if zip_path.exists():
            zip_path.unlink()
        validation["zip"] = {"path": zip_path.as_posix(), "exists": False, "reason": "diagnostic_gate_failed"}
    atomic_write_json(submissions / "fullflow_smoke_validation_report.json", validation)
    return validation


def _submission_interval_rows(rows: Sequence[dict], *, sort_by_final_score: bool) -> list[dict]:
    copied = [dict(row) for row in rows]
    if not sort_by_final_score:
        return copied
    return sorted(
        copied,
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


def _sanitize_interval_row_text_atoms(rows: Sequence[dict], event_store: EventStore | None) -> list[dict]:
    if event_store is None:
        return [dict(row) for row in rows]
    sanitized = []
    for row in rows:
        copied = dict(row)
        text_atom_ids = filter_text_atom_ids_for_source(
            copied.get("text_atom_ids") or [],
            int(copied.get("day") or 0),
            str(copied.get("source") or ""),
            event_store,
        )
        if not text_atom_ids and _requires_audio_marker_text(copied):
            text_atom_ids = text_atom_ids_for_source_interval(
                int(copied.get("day") or 0),
                str(copied.get("source") or ""),
                _float_or_default(copied.get("start_second"), 0.0),
                _float_or_default(copied.get("end_second"), 0.0),
                event_store,
            )
        copied["text_atom_ids"] = text_atom_ids
        sanitized.append(copied)
    return sanitized


def _sanitize_evidence_text_atoms(evidence: Sequence[EvidenceBundle], event_store: EventStore | None) -> list[EvidenceBundle]:
    if event_store is None:
        return list(evidence)
    sanitized = []
    for item in evidence:
        data = item.asdict()
        text_atom_ids = filter_text_atom_ids_for_source(item.text_atom_ids, int(item.day), str(item.source), event_store)
        if not text_atom_ids and _requires_audio_marker_text(data):
            text_atom_ids = text_atom_ids_for_source_interval(item.day, item.source, item.start_second, item.end_second, event_store)
        data["text_atom_ids"] = text_atom_ids
        sanitized.append(EvidenceBundle(**data))
    return sanitized


def _drop_interval_rows_missing_required_text(rows: Sequence[dict]) -> list[dict]:
    return [dict(row) for row in rows if not (_requires_audio_marker_text(row) and not row.get("text_atom_ids"))]


def _requires_audio_marker_text(row: Mapping[str, Any]) -> bool:
    task = str(row.get("task_id") or row.get("task") or "")
    modalities = set(row.get("evidence_modalities") or [])
    return task in {"MM26-EIS09", "MM26-EIS12"} and "transcript_marker_primary" in modalities


def _interval_row_order_report(sort_by_final_score: bool) -> dict:
    return {
        "sort_interval_rows_by_final_score": bool(sort_by_final_score),
        "policy": "task_id_then_final_score_desc" if sort_by_final_score else "input_order",
        "score_key": "final_score" if sort_by_final_score else None,
        "scope": "within_task",
    }


def write_final_score_ranked_audit(
    output_dir: Path,
    *,
    eis_rows: Sequence[dict],
    ois_rows: Sequence[dict],
    qa_results: Sequence[AnswerResult],
    qa_evidence_by_task: Mapping[str, Sequence[QAEvidence]],
    task_ids: Sequence[str],
) -> dict:
    root = output_dir / "diagnostics" / "final_score_ranked"
    task_ids = list(dict.fromkeys(str(task_id) for task_id in task_ids))
    supporting_by_task = {
        result.task_id: {
            "supporting": set(result.supporting_evidence_ids or []),
            "rejected": set(result.rejected_evidence_ids or []),
        }
        for result in qa_results
    }
    files: dict[str, dict] = {}
    for family, prefix, rows in (
        ("event_instance_search", "MM26-EIS", eis_rows),
        ("object_instance_search", "MM26-OIS", ois_rows),
    ):
        family_root = root / family
        by_task: dict[str, list[dict]] = {task_id: [] for task_id in task_ids if task_id.startswith(prefix)}
        for row in rows:
            task_id = str(row.get("task_id") or row.get("task") or "")
            if task_id in by_task:
                by_task[task_id].append(dict(row))
        for task_id in sorted(by_task):
            ranked = sorted(
                by_task[task_id],
                key=lambda row: (
                    -_float_or_default(row.get("final_score"), -1.0),
                    int(row.get("day") or 0),
                    str(row.get("source") or ""),
                    _float_or_default(row.get("start_second"), -1.0),
                    _float_or_default(row.get("end_second"), -1.0),
                    str(row.get("csv_row_id") or ""),
                ),
            )
            audit_rows = [_interval_audit_row(row, rank=index, task_type=family[:-1] if family.endswith("s") else family) for index, row in enumerate(ranked, start=1)]
            path = family_root / f"{task_id}.jsonl"
            atomic_write_jsonl(path, audit_rows)
            files[f"{family}/{task_id}.jsonl"] = _audit_file_info(path, len(audit_rows))
    qa_root = root / "question_answering"
    for task_id in sorted(task_id for task_id in task_ids if task_id.startswith("MM26-QA")):
        rows = list(qa_evidence_by_task.get(task_id, []))
        ranked = sorted(
            rows,
            key=lambda item: (
                -_float_or_default(_qa_field(item, "qa_evidence_score"), -1.0),
                -_float_or_default(_qa_field(item, "reranker_score"), -1.0),
                -_float_or_default(_qa_field(item, "visual_probe_score"), -1.0),
                int(_qa_field(item, "day_num") or 0),
                str(_qa_field(item, "source") or ""),
                _float_or_default(_qa_field(item, "start_second"), -1.0),
                str(_qa_field(item, "evidence_id") or ""),
            ),
        )
        status = supporting_by_task.get(task_id, {"supporting": set(), "rejected": set()})
        audit_rows = [_qa_audit_row(item, rank=index, supporting_status=status) for index, item in enumerate(ranked, start=1)]
        path = qa_root / f"{task_id}.jsonl"
        atomic_write_jsonl(path, audit_rows)
        files[f"question_answering/{task_id}.jsonl"] = _audit_file_info(path, len(audit_rows))
    return {
        "schema_version": "castle_mm26_final_score_ranked_audit_v1",
        "root": root.as_posix(),
        "purpose": "local_manual_review_only_not_official_submission",
        "zip_inclusion": "excluded_from_official_and_diagnostic_zip",
        "files": files,
    }


def write_openreview_package_checklist(output_dir: Path, *, run_id: str, validation: dict, manifest: dict) -> dict:
    package_root = output_dir / "submission_package"
    paper_dir = package_root / "paper"
    results_dir = package_root / "results"
    metadata_dir = package_root / "metadata"
    for path in (paper_dir, results_dir, metadata_dir):
        path.mkdir(parents=True, exist_ok=True)

    run_manifest_path = output_dir / "run_manifest.json"
    validation_report_path = output_dir / "submissions" / "validation_report.json"
    zip_info = validation.get("zip") or {}
    zip_path = Path(str(zip_info.get("path") or ""))
    packaged_zip = results_dir / f"castle_mm26_fully_automatic_{run_id}.zip"
    if zip_info.get("exists") and zip_path.exists():
        shutil.copy2(zip_path, packaged_zip)
    elif packaged_zip.exists():
        packaged_zip.unlink()

    if run_manifest_path.exists():
        shutil.copy2(run_manifest_path, metadata_dir / "run_manifest.json")
    if validation_report_path.exists():
        shutil.copy2(validation_report_path, metadata_dir / "validation_report.json")
    schema_source = Path(str(manifest.get("repo_root") or output_dir.parents[2])) / "event_pipeline" / "event_data" / "formal" / "castle-mm26-formal-20260623-v1" / "quality" / "schema_validation.json"
    if schema_source.exists():
        shutil.copy2(schema_source, metadata_dir / "schema_validation.json")

    per_task_path = metadata_dir / "per_task_report.md"
    atomic_write_text(per_task_path, _render_per_task_report(run_id, validation))
    manuscript_path = paper_dir / "manuscript.pdf"
    source_dir = paper_dir / "source"
    source_archive = paper_dir / "manuscript_source_archive.zip"
    metadata_path = metadata_dir / "openreview_submission_metadata.json"
    metadata = {
        "title": None,
        "authors": [],
        "contact_email": None,
        "run_id": run_id,
        "challenge_url": "https://castle-dataset.github.io/challenges/mm26/",
        "openreview_group_url": "https://openreview.net/group?id=acmmm.org/ACMMM/2026/Workshop/CASTLE",
        "fully_automated_submission_paper_deadline": "2026-07-05",
        "result_zip": _file_info(packaged_zip),
        "paper": _file_info(manuscript_path),
        "paper_source_exists": source_dir.exists() or source_archive.exists(),
        "run_manifest": _file_info(metadata_dir / "run_manifest.json"),
        "validation_report": _file_info(metadata_dir / "validation_report.json"),
        "schema_validation": _file_info(metadata_dir / "schema_validation.json"),
        "per_task_report": _file_info(per_task_path),
        "openreview_form_checked_at_utc": None,
        "openreview_form_url": None,
        "metadata_verified_at_utc": now_utc(),
    }
    missing = []
    if not validation.get("valid") or not zip_info.get("exists") or not packaged_zip.exists():
        missing.append("valid result ZIP")
    if not manuscript_path.exists() or manuscript_path.stat().st_size == 0:
        missing.append("paper/manuscript.pdf")
    if not metadata["paper_source_exists"]:
        missing.append("paper source archive or source directory")
    for key in ("run_manifest", "validation_report", "schema_validation", "per_task_report"):
        if not metadata[key].get("exists"):
            missing.append(key)
    for key in ("title", "authors", "contact_email", "openreview_form_checked_at_utc", "openreview_form_url"):
        if not metadata.get(key):
            missing.append(f"metadata.{key}")
    metadata["ready_for_openreview"] = not missing
    metadata["missing_items"] = missing
    atomic_write_json(metadata_path, metadata)
    return {"ready_for_openreview": metadata["ready_for_openreview"], "metadata_path": metadata_path.as_posix(), "missing_items": missing}


def validate_submission(
    *,
    event_rows: Sequence[dict],
    object_rows: Sequence[dict],
    qa_results: Sequence[AnswerResult],
    evidence: Sequence[EvidenceBundle],
    coverage: CoverageIndex,
    manifest: dict,
    event_store: EventStore | None = None,
    qa_evidence_by_task: Mapping[str, Sequence[QAEvidence]] | None = None,
    expected_task_ids_override: Sequence[str] | None = None,
    run_root: Path | None = None,
) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    expected_ids = list(expected_task_ids_override) if expected_task_ids_override is not None else expected_task_ids()
    evidence_by_key = {(item.task_id, item.csv_row_id): item for item in evidence}
    _validate_interval_family("MM26-EIS", event_rows, evidence_by_key, coverage, errors, event_store=event_store, expected_ids=expected_ids)
    _validate_interval_family("MM26-OIS", object_rows, evidence_by_key, coverage, errors, event_store=event_store, expected_ids=expected_ids)
    _validate_qa(qa_results, errors, qa_evidence_by_task=qa_evidence_by_task, expected_ids=expected_ids)
    if manifest.get("allow_qdrant") is not False:
        errors.append("manifest allow_qdrant must be false")
    if manifest.get("enable_patch_rerank") is not True:
        errors.append("manifest enable_patch_rerank must be true")
    if manifest.get("enable_qwen3_text_rerank") is not True:
        errors.append("manifest enable_qwen3_text_rerank must be true")
    if manifest.get("enable_openrouter_qa_mllm") is not True:
        errors.append("manifest enable_openrouter_qa_mllm must be true")
    if manifest.get("active_environment_profile") != "server_linux":
        errors.append("manifest active_environment_profile must be server_linux")
    _validate_official_qa_model_manifest(manifest, errors)
    preflight = manifest.get("preflight")
    if not isinstance(preflight, dict) or preflight.get("valid") is not True:
        errors.append("manifest preflight.valid must be true")
    elif not _preflight_gate_details_valid(preflight):
        errors.append("manifest preflight official registry/source/day proof gates must be valid")
    _validate_normalized_registry_manifest(manifest, errors, warnings, run_root=run_root)
    _validate_calibration_and_tail_audit(manifest, errors)
    _validate_task_artifacts(manifest, errors, expected_ids=expected_ids)
    _validate_manifest_stages(manifest, errors)
    _validate_manifest_forbidden_inputs(manifest, errors)
    if _manifest_contains_secret(manifest):
        errors.append("manifest appears to contain an OpenRouter secret")
    return {
        "valid": not errors,
        "packaging_allowed": not errors,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "row_counts": {
            "event_instance_search": len(event_rows),
            "object_instance_search": len(object_rows),
            "question_answering": len(qa_results),
        },
    }


def _validate_interval_family(
    prefix: str,
    rows: Sequence[dict],
    evidence_by_key: dict[tuple[str, str], EvidenceBundle],
    coverage: CoverageIndex,
    errors: list[str],
    *,
    event_store: EventStore | None = None,
    expected_ids: Sequence[str] | None = None,
) -> None:
    expected_source = list(expected_ids) if expected_ids is not None else expected_task_ids()
    expected = [task_id for task_id in expected_source if task_id.startswith(prefix)]
    by_task = {task_id: 0 for task_id in expected}
    seen = set()
    for row in rows:
        task = row.get("task") or row.get("task_id")
        if task not in by_task:
            errors.append(f"unknown task in {prefix}: {task}")
            continue
        by_task[task] += 1
        key = (task, int(row["day"]), row["source"], row["starttime"], row["endtime"])
        if key in seen:
            errors.append(f"duplicate interval row: {key}")
        seen.add(key)
        if row.get("source") not in ACCEPTED_SOURCES:
            errors.append(f"invalid source: {row.get('source')}")
        start = parse_hms(str(row["starttime"]))
        end = parse_hms(str(row["endtime"]))
        if start >= end:
            errors.append(f"starttime must be before endtime: {key}")
        clips = coverage.clip(int(row["day"]), str(row["source"]), float(start), float(end))
        if not clips or sum(clip.end - clip.start for clip in clips) < 1.0:
            errors.append(f"row lacks canonical frame coverage: {key}")
        if not _coverage_fully_contains(clips, float(start), float(end)):
            errors.append(f"row is not fully contained in one accepted coverage interval: {key}")
        evidence = evidence_by_key.get((task, row.get("csv_row_id")))
        if evidence is None:
            errors.append(f"row lacks EvidenceBundle: {task} {row.get('csv_row_id')}")
        else:
            _validate_interval_evidence(prefix, row, evidence, errors, event_store=event_store)
        modalities = set(row.get("evidence_modalities") or [])
        if prefix == "MM26-OIS" and "visual_direct_frame" not in modalities:
            errors.append(f"OIS row lacks visual direct frame: {task} {row.get('csv_row_id')}")
        if prefix == "MM26-EIS" and task not in {"MM26-EIS09", "MM26-EIS12"} and "visual_direct_frame" not in modalities:
            errors.append(f"EIS row lacks visual direct frame: {task} {row.get('csv_row_id')}")
        if prefix == "MM26-EIS" and task in {"MM26-EIS09", "MM26-EIS12"} and "visual_direct_frame" not in modalities:
            _validate_audio_sensitive_text_marker(row, task, errors)
    # The scoring rewrite intentionally allows zero EIS/OIS rows when every
    # candidate fails raw thresholds or evidence gates.


def _validate_qa(
    results: Sequence[AnswerResult],
    errors: list[str],
    *,
    qa_evidence_by_task: Mapping[str, Sequence[QAEvidence]] | None = None,
    expected_ids: Sequence[str] | None = None,
) -> None:
    expected_source = list(expected_ids) if expected_ids is not None else expected_task_ids()
    expected = [task_id for task_id in expected_source if task_id.startswith("MM26-QA")]
    counts = Counter(result.task_id for result in results)
    if len(results) != len(expected):
        errors.append(f"QA result row count must be exactly {len(expected)}: got {len(results)}")
    for task_id, count in sorted(counts.items()):
        if task_id not in expected:
            errors.append(f"unknown QA result task: {task_id}")
        elif count != 1:
            errors.append(f"QA result task must appear exactly once: {task_id} count={count}")
    by_task = {}
    for result in results:
        if result.task_id not in by_task:
            by_task[result.task_id] = result
    for task in expected:
        result = by_task.get(task)
        if result is None:
            errors.append(f"missing QA result: {task}")
            continue
        if result.mllm_validation_status != "valid":
            errors.append(f"QA result invalid MLLM status: {task} {result.mllm_validation_status}")
        if result.mllm_model != OFFICIAL_QA_MODEL:
            errors.append(f"QA result used non-official MLLM model: {task} {result.mllm_model}")
        if not result.supporting_evidence_ids:
            errors.append(f"QA result lacks supporting evidence ids: {task}")
        if qa_evidence_by_task is not None:
            task_evidence = {item.evidence_id: item for item in qa_evidence_by_task.get(task, [])}
            missing_ids = [evidence_id for evidence_id in result.supporting_evidence_ids if evidence_id not in task_evidence]
            if missing_ids:
                errors.append(f"QA result references missing evidence ids: {task} {missing_ids[:3]}")
            for evidence_id in result.supporting_evidence_ids:
                evidence = task_evidence.get(evidence_id)
                if evidence is None:
                    continue
                image_refs = list(evidence.original_image_refs or [])
                if not image_refs:
                    errors.append(f"QA supporting evidence lacks original images: {task} {evidence_id}")
                    continue
                missing_images = [path for path in image_refs if not Path(str(path)).exists()]
                if missing_images:
                    errors.append(f"QA supporting evidence image refs are not readable: {task} {evidence_id} {missing_images[:2]}")
            evidence_refs = {render_source_time_ref(item) for item in task_evidence.values()}
            supporting_refs = {render_source_time_ref(task_evidence[evidence_id]) for evidence_id in result.supporting_evidence_ids if evidence_id in task_evidence}
            rendered_refs = {match.group(0) for match in QA_REF_RE.finditer(result.answer_text.split("Evidence:", 1)[1] if "Evidence:" in result.answer_text else "")}
            if rendered_refs and not rendered_refs <= evidence_refs:
                errors.append(f"QA answer contains refs absent from qa_evidence.jsonl: {task}")
            if supporting_refs and rendered_refs and not rendered_refs <= supporting_refs:
                errors.append(f"QA answer refs are not limited to supporting evidence ids: {task}")
        if not isinstance(result.mllm_raw_json, dict) or not REQUIRED_MLLM_RAW_KEYS <= set(result.mllm_raw_json):
            errors.append(f"QA raw MLLM JSON missing strict schema keys: {task}")
        if "Evidence:" not in result.answer_text or not QA_REF_RE.search(result.answer_text.split("Evidence:", 1)[1]):
            errors.append(f"QA answer lacks parseable evidence refs: {task}")
        direct_answer = result.answer_text.split("Evidence:", 1)[0].strip().rstrip(".")
        if _qa_word_count(direct_answer) > 24 or len(direct_answer) > 180:
            errors.append(f"QA direct answer too long: {task}")
        if not _qa_answer_is_english_or_neutral(result.answer_value) or not _qa_answer_is_english_or_neutral(direct_answer):
            errors.append(f"QA answer is not English: {task}")
        if "I could not determine" in result.answer_text:
            errors.append(f"QA fallback answer forbidden: {task}")
        answer_type = str(result.answer_type or "").strip().lower()
        if answer_type not in ALLOWED_QA_ANSWER_TYPES:
            errors.append(f"QA answer type is not official schema type: {task} {result.answer_type}")
        if answer_type in {"count", "number", "integer"}:
            if _parse_count_answer(result.answer_value) is None:
                errors.append(f"QA count answer is not numeric: {task}")
            if not result.dedupe_trace:
                errors.append(f"QA count answer lacks dedupe trace: {task}")
        if answer_type in {"person", "person_name", "actor"}:
            canonical = _canonical_person(result.answer_value)
            if canonical is None:
                errors.append(f"QA person answer is not canonical: {task}")
            if result.answer_value.strip() in FIXED_ROOM_SOURCES:
                errors.append(f"QA fixed-room source used as person answer: {task}")


def _validate_audio_sensitive_text_marker(row: dict, task: str, errors: list[str]) -> None:
    modalities = set(row.get("evidence_modalities") or [])
    if "transcript_marker_primary" not in modalities:
        errors.append(f"audio-sensitive EIS row lacks transcript marker primary: {task} {row.get('csv_row_id')}")
        return
    if not row.get("text_atom_ids"):
        errors.append(f"audio-sensitive EIS row lacks text atoms: {task} {row.get('csv_row_id')}")
    provenance = row.get("provenance") or {}
    required = ("lexical_marker_sweep_id", "accepted_source_interval_ref", "local_visual_recovery_status")
    for key in required:
        if not provenance.get(key):
            errors.append(f"audio-sensitive EIS row lacks {key}: {task} {row.get('csv_row_id')}")
    if provenance.get("local_visual_recovery_status") not in {"completed", "completed_no_visual_confirmation"}:
        errors.append(f"audio-sensitive EIS row has invalid local visual recovery status: {task} {row.get('csv_row_id')}")
    audit = provenance.get("local_visual_recovery_audit") or {}
    lexical_hits = audit.get("lexical_marker_hits")
    if not isinstance(lexical_hits, list) or not lexical_hits:
        errors.append(f"audio-sensitive EIS row lacks actual lexical marker hit: {task} {row.get('csv_row_id')}")


def _preflight_gate_details_valid(preflight: dict) -> bool:
    registry = preflight.get("registry_validation")
    official = preflight.get("official_sources_manifest")
    day_proof = preflight.get("day_date_proof")
    model_preflight = preflight.get("model_preflight")
    return (
        isinstance(registry, dict)
        and registry.get("valid") is True
        and isinstance(official, dict)
        and official.get("valid") is True
        and isinstance(day_proof, dict)
        and day_proof.get("valid") is True
        and _model_preflight_gate_valid(model_preflight)
    )


def _validate_normalized_registry_manifest(manifest: dict, errors: list[str], warnings: list[str], *, run_root: Path | None = None) -> None:
    path_value = manifest.get("normalized_queries_path")
    if not path_value:
        warnings.append("manifest normalized_queries_path missing; normalized registry reload skipped")
        return
    path = _resolve_manifest_file_path(path_value, run_root=run_root, preferred_subdir="registry")
    if not path.is_absolute() and not path.exists():
        repo_root = manifest.get("repo_root")
        if repo_root:
            path = Path(str(repo_root)) / path
    if not path.exists() or not path.is_file():
        errors.append(f"manifest normalized_queries_path is not readable: {path}")
        return
    expected_sha = manifest.get("normalized_queries_sha256")
    if expected_sha and sha256_file(path) != expected_sha:
        errors.append("manifest normalized_queries_sha256 does not match file")
        return
    try:
        payload = read_yaml(path)
    except Exception as exc:
        errors.append(f"normalized query registry unreadable: {type(exc).__name__}")
        return
    if not isinstance(payload, dict) or payload.get("schema_version") != "castle_mm26_normalized_query_registry_v1":
        errors.append("normalized query registry schema_version invalid")
        return
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        errors.append("normalized query registry tasks must be a list")
        return
    actual_ids = [str(item.get("task_id") or "") for item in tasks if isinstance(item, dict)]
    if actual_ids != expected_task_ids():
        errors.append("normalized query registry task ids do not match official task order")
    registry_report = payload.get("registry_validation")
    if not isinstance(registry_report, dict) or registry_report.get("valid") is not True:
        errors.append("normalized query registry validation report is not valid")


def _resolve_manifest_file_path(path_value: object, *, run_root: Path | None = None, preferred_subdir: str | None = None) -> Path:
    raw = str(path_value)
    path = Path(raw)
    if path.exists():
        return path
    candidates: list[Path] = []
    if run_root is not None:
        current_root = Path(run_root)
        name = Path(raw.replace("\\", "/")).name
        if preferred_subdir and name:
            candidates.append(current_root / preferred_subdir / name)
        normalized = raw.replace("\\", "/")
        marker = f"/{preferred_subdir}/" if preferred_subdir else None
        if marker and marker in normalized:
            candidates.append(current_root / preferred_subdir / normalized.split(marker, 1)[1])
        if name:
            candidates.append(current_root / name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def _model_preflight_gate_valid(model_preflight: object) -> bool:
    if not isinstance(model_preflight, dict):
        return False
    openrouter = model_preflight.get("openrouter")
    tipsv2 = model_preflight.get("tipsv2")
    qwen3_reranker = model_preflight.get("qwen3_reranker")
    return (
        isinstance(openrouter, dict)
        and openrouter.get("model") == OFFICIAL_QA_MODEL
        and openrouter.get("model_found") is True
        and openrouter.get("required_supported") is True
        and isinstance(tipsv2, dict)
        and tipsv2.get("status") == "completed"
        and isinstance(qwen3_reranker, dict)
        and qwen3_reranker.get("status") == "completed"
    )


def _coverage_fully_contains(clips: Sequence, start: float, end: float) -> bool:
    eps = 1e-6
    return any(float(clip.start) <= start + eps and float(clip.end) >= end - eps for clip in clips)


def _validate_interval_evidence(prefix: str, row: dict, evidence: EvidenceBundle, errors: list[str], *, event_store: EventStore | None = None) -> None:
    task = row.get("task") or row.get("task_id")
    row_start = parse_hms(str(row["starttime"]))
    row_end = parse_hms(str(row["endtime"]))
    if evidence.task_id != task or evidence.csv_row_id != row.get("csv_row_id"):
        errors.append(f"EvidenceBundle key mismatch: {task} {row.get('csv_row_id')}")
    if int(evidence.day) != int(row["day"]) or evidence.source != row.get("source"):
        errors.append(f"EvidenceBundle day/source mismatch: {task} {row.get('csv_row_id')}")
    if float(evidence.start_second) > float(row_start) + 1e-6 or float(evidence.end_second) < float(row_end) - 1e-6:
        errors.append(f"EvidenceBundle does not cover CSV row interval: {task} {row.get('csv_row_id')}")
    if evidence.rejected_reason:
        errors.append(f"rejected EvidenceBundle appears in CSV: {task} {row.get('csv_row_id')}")
    if not evidence.event_ids:
        errors.append(f"EvidenceBundle lacks event ids: {task} {row.get('csv_row_id')}")
    modalities = set(row.get("evidence_modalities") or evidence.evidence_modalities or [])
    if "visual_direct_frame" in modalities and not evidence.frame_atom_ids:
        errors.append(f"visual EvidenceBundle lacks frame atoms: {task} {row.get('csv_row_id')}")
    if prefix == "MM26-OIS" and not evidence.frame_atom_ids:
        errors.append(f"OIS EvidenceBundle lacks frame atoms: {task} {row.get('csv_row_id')}")
    if prefix == "MM26-EIS" and task in {"MM26-EIS09", "MM26-EIS12"} and "transcript_marker_primary" in modalities and not evidence.text_atom_ids:
        errors.append(f"audio-sensitive EvidenceBundle lacks text atoms: {task} {row.get('csv_row_id')}")
    if event_store is not None:
        _validate_interval_evidence_backrefs(prefix, row, evidence, event_store, errors)


def _validate_interval_evidence_backrefs(prefix: str, row: dict, evidence: EvidenceBundle, event_store: EventStore, errors: list[str]) -> None:
    task = row.get("task") or row.get("task_id")
    row_start = float(parse_hms(str(row["starttime"])))
    row_end = float(parse_hms(str(row["endtime"])))
    for event_id in evidence.event_ids:
        event = event_store.event_by_id.get(str(event_id))
        if event is None:
            errors.append(f"EvidenceBundle references unknown event id: {task} {event_id}")
            continue
        if event.get("source") is not None and (int(event.get("day_num") or -1) != int(row["day"]) or str(event.get("source")) != str(row["source"])):
            errors.append(f"EvidenceBundle event id day/source mismatch: {task} {event_id}")
    for frame_id in evidence.frame_atom_ids:
        frame = event_store.frame_by_id.get(str(frame_id))
        if frame is None:
            errors.append(f"EvidenceBundle references unknown frame atom id: {task} {frame_id}")
            continue
        if int(frame.get("day_num") or -1) != int(row["day"]) or str(frame.get("source")) != str(row["source"]):
            errors.append(f"EvidenceBundle frame atom day/source mismatch: {task} {frame_id}")
        if "visual_direct_frame" in set(evidence.evidence_modalities or row.get("evidence_modalities") or []):
            second = float(frame.get("absolute_second") or -1.0)
            if not (row_start <= second < row_end):
                errors.append(f"EvidenceBundle direct frame atom outside row interval: {task} {frame_id}")
    for text_id in evidence.text_atom_ids:
        text_atom = event_store.text_by_id.get(str(text_id))
        if text_atom is None:
            errors.append(f"EvidenceBundle references unknown text atom id: {task} {text_id}")
            continue
        if int(text_atom.get("day_num") or -1) != int(row["day"]) or str(text_atom.get("source")) != str(row["source"]):
            errors.append(f"EvidenceBundle text atom day/source mismatch: {task} {text_id}")
        if prefix in {"MM26-EIS", "MM26-OIS"} and str(text_atom.get("time_alignment_status") or "").lower() == "invalid":
            errors.append(f"EvidenceBundle text atom has invalid time alignment: {task} {text_id}")


def _qa_word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9À-ÖØ-öø-ÿ]+", text))


def _qa_answer_is_english_or_neutral(text: str) -> bool:
    letters = [char for char in str(text) if char.isalpha()]
    if not letters:
        return True
    latin_letters = sum(1 for char in letters if "LATIN" in unicodedata.name(char, ""))
    return latin_letters / max(len(letters), 1) >= 0.8


def _parse_count_answer(answer: str) -> int | None:
    text = str(answer).strip().lower()
    match = re.search(r"\b\d+\b", text)
    if match:
        return int(match.group(0))
    for word, value in NUMBER_WORD_VALUES.items():
        if re.search(rf"\b{word}\b", text):
            return value
    return None


def _canonical_person(value: str) -> str | None:
    text = str(value).strip().rstrip(".")
    if text in {"Bjorn", "Björn"}:
        return "Björn"
    for name in PERSON_NAMES:
        if name in {"Bjorn", "Björn"}:
            continue
        if text.lower() == name.lower():
            return name
    return None


def _csv_interval_row(row: dict) -> dict:
    return {
        "task": row.get("task") or row.get("task_id"),
        "day": int(row["day"]),
        "starttime": row["starttime"],
        "endtime": row["endtime"],
        "source": row["source"],
    }


def _interval_audit_row(row: dict, *, rank: int, task_type: str) -> dict:
    return {
        "rank_by_final_score": rank,
        "task_id": row.get("task_id") or row.get("task"),
        "task_type": task_type,
        "csv_row_id": row.get("csv_row_id"),
        "candidate_id": row.get("candidate_id"),
        "day": row.get("day"),
        "source": row.get("source"),
        "start_second": row.get("start_second"),
        "end_second": row.get("end_second"),
        "starttime": row.get("starttime"),
        "endtime": row.get("endtime"),
        "final_score": row.get("final_score"),
        "score_parts": _sanitize_audit_value(row.get("score_parts") or {}),
        "evidence_modalities": list(row.get("evidence_modalities") or []),
        "event_ids": list(row.get("event_ids") or []),
        "frame_atom_ids": list(row.get("frame_atom_ids") or []),
        "text_atom_ids": list(row.get("text_atom_ids") or []),
        "split_reason": row.get("split_reason"),
        "evidence_backrefs": {
            "csv_row_id": row.get("csv_row_id"),
            "candidate_id": row.get("candidate_id"),
            "event_count": len(row.get("event_ids") or []),
            "frame_count": len(row.get("frame_atom_ids") or []),
            "text_count": len(row.get("text_atom_ids") or []),
        },
        "provenance": _sanitize_audit_value(row.get("provenance") or {}),
    }


def _qa_audit_row(item: QAEvidence | Mapping[str, Any], *, rank: int, supporting_status: Mapping[str, set[str]]) -> dict:
    evidence_id = str(_qa_field(item, "evidence_id") or "")
    if evidence_id in supporting_status.get("supporting", set()):
        status = "supporting_answer"
    elif evidence_id in supporting_status.get("rejected", set()):
        status = "rejected_by_answer_reducer"
    else:
        status = "retained_for_validator"
    text_excerpt = _truncate_text(_qa_field(item, "text_excerpt") or "", limit=500)
    return {
        "rank_by_qa_evidence_score": rank,
        "task_id": _qa_field(item, "task_id"),
        "evidence_id": evidence_id,
        "day_num": _qa_field(item, "day_num"),
        "source": _qa_field(item, "source"),
        "source_role": _qa_field(item, "source_role"),
        "start_second": _qa_field(item, "start_second"),
        "end_second": _qa_field(item, "end_second"),
        "qa_evidence_score": _qa_field(item, "qa_evidence_score"),
        "reranker_score": _qa_field(item, "reranker_score"),
        "visual_probe_score": _qa_field(item, "visual_probe_score"),
        "qa_score_parts": _sanitize_audit_value(_qa_field(item, "qa_score_parts") or {}),
        "entity_mentions": _sanitize_audit_value(_qa_field(item, "entity_mentions") or []),
        "relation_candidates": _sanitize_audit_value(_qa_field(item, "relation_candidates") or []),
        "event_ids": list(_qa_field(item, "event_ids") or []),
        "frame_atom_ids": list(_qa_field(item, "frame_atom_ids") or []),
        "text_atom_ids": list(_qa_field(item, "text_atom_ids") or []),
        "original_image_refs": list(_qa_field(item, "original_image_refs") or []),
        "text_excerpt": text_excerpt,
        "text_excerpt_truncated": len(str(_qa_field(item, "text_excerpt") or "")) > len(text_excerpt),
        "supporting_status": status,
        "rejected_reason": None if status != "rejected_by_answer_reducer" else "answer_reducer_rejected",
        "evidence_backrefs": {
            "evidence_id": evidence_id,
            "source_time_ref": render_source_time_ref(item) if isinstance(item, QAEvidence) else None,
            "event_count": len(_qa_field(item, "event_ids") or []),
            "frame_count": len(_qa_field(item, "frame_atom_ids") or []),
            "text_count": len(_qa_field(item, "text_atom_ids") or []),
            "image_ref_count": len(_qa_field(item, "original_image_refs") or []),
        },
    }


def _qa_field(item: QAEvidence | Mapping[str, Any], name: str) -> Any:
    if isinstance(item, QAEvidence):
        return getattr(item, name)
    return item.get(name)


def _float_or_default(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sanitize_audit_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 5:
        return "<truncated_depth>"
    if isinstance(value, str):
        return _truncate_text(value, limit=500)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, MappingABC):
        sanitized = {}
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            key_text = str(key)
            if _is_sensitive_audit_key(key_text):
                sanitized[key_text] = "<redacted>"
                continue
            sanitized[key_text] = _sanitize_audit_value(item, depth=depth + 1)
        return sanitized
    if isinstance(value, SequenceABC) and not isinstance(value, (bytes, bytearray)):
        rows = list(value)
        kept = [_sanitize_audit_value(item, depth=depth + 1) for item in rows[:50]]
        if len(rows) > 50:
            kept.append({"truncated_list_items": len(rows) - 50})
        return kept
    return str(value)


def _truncate_text(value: object, *, limit: int) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + f"...<truncated {len(text) - limit} chars>"


def _is_sensitive_audit_key(key: str) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in ("api_key", "authorization", "bearer", "openrouter_request", "request_body", "raw_request"))


def _audit_file_info(path: Path, line_count: int) -> dict:
    record = _file_info(path)
    record["line_count"] = line_count
    return record


def _file_info(path: Path) -> dict:
    return {"path": path.as_posix(), "exists": path.exists(), "sha256": sha256_file(path) if path.exists() else None, "size": path.stat().st_size if path.exists() else None}


def _render_per_task_report(run_id: str, validation: dict) -> str:
    rows = validation.get("row_counts") or {}
    lines = [
        f"# CASTLE MM26 Per-Task Report: {run_id}",
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
    return "\n".join(lines)


def _manifest_contains_secret(manifest: dict) -> bool:
    text = json.dumps(manifest, ensure_ascii=False)
    return "sk-or-" in text or "OPENROUTER_API_KEY=" in text


def _validate_manifest_stages(manifest: dict, errors: list[str]) -> None:
    stages = manifest.get("stages") or []
    if not isinstance(stages, list):
        errors.append("manifest stages must be a list")
        return
    qa_answer_completed = False
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            errors.append(f"manifest stage is not an object: index={index}")
            continue
        status = str(stage.get("status") or "")
        if status in {"failed", "skipped", "skipped_completed_marker"}:
            if not _stage_superseded_by_later_completion(stages, index):
                errors.append(f"manifest contains non-completed stage: {stage.get('name', index)} status={status}")
        if stage.get("name") == "qa_answer" and status == "completed" and stage.get("api_call_performed") is True and stage.get("model") == OFFICIAL_QA_MODEL:
            qa_answer_completed = True
    if not qa_answer_completed:
        errors.append("manifest must contain completed qa_answer stage with official model and api_call_performed=true")


def _stage_superseded_by_later_completion(stages: list, index: int) -> bool:
    current = stages[index]
    if not isinstance(current, dict):
        return False
    retry_key = _stage_retry_key(current)
    if retry_key is None:
        return False
    for later in stages[index + 1 :]:
        if isinstance(later, dict) and str(later.get("status") or "") == "completed" and _stage_retry_key(later) == retry_key:
            return True
    return False


def _stage_retry_key(stage: dict) -> tuple | None:
    name = stage.get("name")
    if not name:
        return None
    return (
        name,
        stage.get("task_family"),
        stage.get("execution_profile"),
        stage.get("package_kind"),
        stage.get("model"),
    )


def _validate_official_qa_model_manifest(manifest: dict, errors: list[str]) -> None:
    effective_model = manifest.get("qa_mllm_model")
    configured_model = manifest.get("qa_mllm_config_model")
    if effective_model != OFFICIAL_QA_MODEL:
        errors.append(f"manifest qa_mllm_model must be {OFFICIAL_QA_MODEL}")
    if configured_model != OFFICIAL_QA_MODEL:
        errors.append(f"manifest qa_mllm_config_model must be {OFFICIAL_QA_MODEL}")
    if manifest.get("qa_mllm_model_override_used"):
        errors.append("manifest qa_mllm_model_override_used must be false for official packaging")


def _validate_calibration_and_tail_audit(manifest: dict, errors: list[str]) -> None:
    calibration = manifest.get("score_calibration")
    if not isinstance(calibration, dict):
        errors.append("manifest score_calibration summary missing")
    else:
        if calibration.get("threshold_source") != "fixed_task_raw_floor_v1":
            errors.append("manifest score_calibration threshold_source invalid")
        if calibration.get("score_norm_version") != "raw_floor_high_linear_v1":
            errors.append("manifest score_calibration score_norm_version invalid")
    tail = manifest.get("tail_audit")
    if not isinstance(tail, dict):
        errors.append("manifest tail_audit summary missing")
    else:
        if tail.get("tail_audit_policy") != "deterministic_cluster_audit_v1":
            errors.append("manifest tail_audit policy invalid")
        if tail.get("block_packaging"):
            errors.append(f"manifest tail_audit blocks packaging: {tail.get('block_reasons')}")


def _validate_task_artifacts(manifest: dict, errors: list[str], *, expected_ids: Sequence[str] | None = None) -> None:
    artifacts = manifest.get("task_artifacts")
    if not isinstance(artifacts, dict):
        errors.append("manifest task_artifacts summary missing")
        return
    expected_artifact_ids = list(expected_ids) if expected_ids is not None else expected_task_ids()
    for task_id in expected_artifact_ids:
        item = artifacts.get(task_id)
        if not isinstance(item, dict):
            errors.append(f"missing task artifact summary: {task_id}")
            continue
        if item.get("completed_marker") is not True:
            errors.append(f"task lacks completed marker: {task_id}")
        task_manifest = item.get("task_manifest")
        if not isinstance(task_manifest, dict):
            errors.append(f"task manifest missing: {task_id}")
        else:
            if task_manifest.get("status") != "completed":
                errors.append(f"task manifest not completed: {task_id}")
            if task_manifest.get("task_id") != task_id:
                errors.append(f"task manifest task id mismatch: {task_id}")
            if not task_manifest.get("task_query_sha256") or not task_manifest.get("resolved_profiles_sha256"):
                errors.append(f"task manifest lacks input/profile hash: {task_id}")
            if task_manifest.get("completed_marker_written_after_manifest") is not True:
                errors.append(f"task manifest marker ordering missing: {task_id}")
        files = item.get("files")
        if not isinstance(files, dict):
            errors.append(f"task artifact files missing: {task_id}")
            continue
        if task_id.startswith(("MM26-EIS", "MM26-OIS")):
            required = [
                "result.jsonl",
                "evidence.jsonl",
                "solver_report.json",
                "score_calibration.json",
                "recall_tail_audit.json",
                "task_manifest.json",
                "text_rerank/stage_manifest.json",
                "visual_patch_rerank/stage_manifest.json",
            ]
            if task_id in {"MM26-EIS09", "MM26-EIS12"}:
                required.append("local_visual_recovery_audit.jsonl")
        else:
            required = (
                "answer_result.json",
                "qa_evidence.jsonl",
                "qa_rejected_evidence.jsonl",
                "qa_reducer_diagnostics.json",
                "qa_mllm_manifest.json",
                "task_manifest.json",
                "text_rerank/stage_manifest.json",
                "visual_patch_rerank/stage_manifest.json",
            )
        for name in required:
            info = files.get(name)
            if not isinstance(info, dict) or not info.get("sha256") or info.get("size") is None:
                errors.append(f"task artifact missing hash: {task_id} {name}")
            elif name.endswith("stage_manifest.json"):
                _validate_stage_manifest_file(task_id, name, info, errors)
            elif name == "qa_mllm_manifest.json":
                _validate_qa_mllm_manifest_file(task_id, info, errors)


def _validate_qa_mllm_manifest_file(task_id: str, info: dict, errors: list[str]) -> None:
    path = info.get("path")
    if not path:
        return
    target = Path(str(path))
    if not target.exists():
        return
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"QA MLLM manifest unreadable: {task_id} {type(exc).__name__}")
        return
    if payload.get("status") != "completed":
        errors.append(f"QA MLLM manifest not completed: {task_id} status={payload.get('status')}")
    if payload.get("mllm_model") != OFFICIAL_QA_MODEL:
        errors.append(f"QA MLLM manifest uses non-official model: {task_id} {payload.get('mllm_model')}")
    summary = payload.get("prompt_image_attachment_summary")
    if not isinstance(summary, dict):
        errors.append(f"QA MLLM manifest lacks image attachment summary: {task_id}")
        return
    if int(summary.get("missing_image_ref_count") or 0) != 0:
        errors.append(f"QA MLLM manifest has missing prompt image refs: {task_id}")
    if int(summary.get("attached_image_count") or 0) <= 0:
        errors.append(f"QA MLLM manifest attached no prompt images: {task_id}")
    if int(summary.get("evidence_with_attached_images") or 0) <= 0:
        errors.append(f"QA MLLM manifest has no evidence with attached images: {task_id}")


def _validate_stage_manifest_file(task_id: str, name: str, info: dict, errors: list[str]) -> None:
    path = info.get("path")
    if not path:
        return
    target = Path(str(path))
    if not target.exists():
        return
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"task stage manifest unreadable: {task_id} {name} {type(exc).__name__}")
        return
    if payload.get("status") != "completed":
        errors.append(f"task stage manifest not completed: {task_id} {name} status={payload.get('status')}")


def _validate_manifest_forbidden_inputs(manifest: dict, errors: list[str]) -> None:
    text = json.dumps(manifest, ensure_ascii=False).replace("\\", "/")
    forbidden = []
    if "retrieval_test" in text:
        forbidden.append("retrieval_test")
    if "/exploration/" in text or '"exploration/' in text or "exploration/outputs" in text:
        forbidden.append("exploration")
    if "Qdrant" in text or "qdrant://" in text or '"backend": "qdrant' in text:
        forbidden.append("Qdrant")
    for item in forbidden:
        errors.append(f"manifest contains forbidden official input/code path: {item}")
