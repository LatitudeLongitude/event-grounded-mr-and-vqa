from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .config import AppConfig
from .io_utils import canonical_json_hash, read_yaml, write_yaml
from .models import TaskQuery


TASK_TYPES = {
    "EIS": "event_instance_search",
    "OIS": "object_instance_search",
    "QA": "question_answering",
}


def expected_task_ids() -> list[str]:
    return [f"MM26-EIS{index:02d}" for index in range(1, 13)] + [f"MM26-OIS{index:02d}" for index in range(1, 13)] + [f"MM26-QA{index:02d}" for index in range(1, 13)]


def load_query_registry(path: Path | str, config: AppConfig, *, official_queries: dict[str, str] | None = None) -> tuple[list[TaskQuery], dict[str, Any]]:
    target = Path(path).resolve()
    if target.as_posix().replace("\\", "/").endswith("event_pipeline/future/task_queries.yaml"):
        raise ValueError("official registry must not point to event_pipeline/future/task_queries.yaml")
    raw = read_yaml(target)
    tasks_raw = raw.get("tasks")
    if not isinstance(tasks_raw, list):
        raise ValueError("queries.yaml must contain a tasks list")
    defaults = raw.get("type_defaults") or {}
    normalized_dicts: list[dict[str, Any]] = []
    for task in tasks_raw:
        if not isinstance(task, dict):
            raise ValueError("each task entry must be a mapping")
        task_type = task.get("task_type")
        merged = copy.deepcopy(defaults.get(task_type, {}))
        merged.update(copy.deepcopy(task))
        if task_type == "question_answering":
            answer_defaults = copy.deepcopy(defaults.get(task_type, {}).get("answer_policy_defaults", {}))
            answer_policy = copy.deepcopy(merged.get("answer_policy") or {})
            answer_defaults.update(answer_policy)
            merged["answer_policy"] = answer_defaults
        _validate_raw_task(merged)
        merged["resolved_profiles"] = _resolve_profiles(merged, config)
        normalized_dicts.append(merged)
    _validate_task_set(normalized_dicts, official_queries=official_queries)
    queries = [_task_from_dict(item) for item in normalized_dicts]
    registry_report = {
        "valid": True,
        "query_registry_path": target.as_posix(),
        "query_registry_sha256": canonical_json_hash(raw),
        "normalized_queries_sha256": canonical_json_hash(normalized_dicts),
        "profile_registry_sha256": config.profile_registry_sha256,
        "task_count": len(queries),
        "task_ids": [query.task_id for query in queries],
    }
    return queries, {"raw": raw, "normalized": normalized_dicts, "report": registry_report}


def write_normalized_registry(path: Path, normalized_tasks: list[dict[str, Any]], report: dict[str, Any]) -> None:
    payload = {
        "schema_version": "castle_mm26_normalized_query_registry_v1",
        "registry_validation": report,
        "tasks": normalized_tasks,
    }
    write_yaml(path, payload)


def _resolve_profiles(task: dict[str, Any], config: AppConfig) -> dict[str, Any]:
    resolved: dict[str, Any] = {"profile_registry_sha256": config.profile_registry_sha256}
    for key, block in (
        ("recall_profile", "recall_profiles"),
        ("scoring_profile", "scoring_profiles"),
        ("threshold_profile", "threshold_profiles"),
        ("output_profile", "output_profiles"),
    ):
        name = task.get(key)
        if not name:
            if key == "threshold_profile" and task["task_type"] == "question_answering":
                name = "qa_evidence_required"
                task[key] = name
            else:
                raise ValueError(f"{task.get('task_id')} missing {key}")
        resolved[key] = {"name": name, "source": f"config.yaml.profile_registry.{block}.{name}", "value": copy.deepcopy(config.resolve_profile(block, name))}
    scoring = resolved["scoring_profile"]["value"]
    component_profiles: dict[str, Any] = {}
    for component_key in ("visual_event_component_profile", "event_embedding_component_profile"):
        component_name = scoring.get(component_key)
        if component_name:
            component_profiles[component_key] = {
                "name": component_name,
                "source": f"config.yaml.profile_registry.scoring_component_profiles.{component_name}",
                "value": copy.deepcopy(config.resolve_profile("scoring_component_profiles", component_name)),
            }
    for component_name in ("prompt_positive_default", "negative_prompt_default", "patch_frame_default", "text_hit_default"):
        component_profiles[component_name] = {
            "name": component_name,
            "source": f"config.yaml.profile_registry.scoring_component_profiles.{component_name}",
            "value": copy.deepcopy(config.resolve_profile("scoring_component_profiles", component_name)),
        }
    if component_profiles:
        resolved["scoring_component_profiles"] = component_profiles
    return resolved


def _validate_raw_task(task: dict[str, Any]) -> None:
    task_id = str(task.get("task_id", ""))
    if task_id not in expected_task_ids():
        raise ValueError(f"unexpected task id: {task_id}")
    if task_id.startswith("MM26-EIS"):
        expected_type = "event_instance_search"
    elif task_id.startswith("MM26-OIS"):
        expected_type = "object_instance_search"
    elif task_id.startswith("MM26-QA"):
        expected_type = "question_answering"
    else:
        raise ValueError(f"unexpected task id prefix: {task_id}")
    if task.get("task_type") != expected_type:
        raise ValueError(f"task type mismatch for {task_id}")
    for key in ("official_text", "canonical_text_for_retrieval"):
        if not str(task.get(key, "")).strip():
            raise ValueError(f"{task_id} missing {key}")
    # Parameter-source caveat: the forbidden-key check below rejects only the
    # listed fields on the merged task mapping (type defaults plus task fields);
    # it does not define every runtime parameter source. In official runs, recall
    # Top-K values come from resolved recall profiles. Smoke paths and diagnostic
    # execution profiles may override those limits. Final EIS/OIS visual and
    # positive-text raw floors, plus interval-score weights, come from
    # solver.INTERVAL_SCORE_SPECS keyed by task_id, not from the resolved scoring
    # profile's final_weights/object_track_weights. Audit these sources when
    # changing parameters or reproducing results.
    forbidden = ("output_policy", "visual_top_k", "text_dense_top_k", "submit_min_score", "final_weights")
    for key in forbidden:
        if key in task:
            raise ValueError(f"{task_id} raw registry must not inline {key}")
    if task["task_type"] in {"event_instance_search", "object_instance_search"}:
        if not task.get("visual_prompts"):
            raise ValueError(f"{task_id} must define visual_prompts")
        policy = task.get("evidence_policy") or {}
        if task_id not in {"MM26-EIS09", "MM26-EIS12"} and policy.get("allow_transcript_marker_primary"):
            raise ValueError(f"{task_id} cannot enable transcript_marker_primary")
    if task["task_type"] == "question_answering":
        policy = task.get("answer_policy") or {}
        for key in ("reducer", "expected_answer_type", "min_supporting_evidence", "low_confidence_min_supporting_evidence", "keyword_frames", "speaker_attribution", "answer_template"):
            if key not in policy:
                raise ValueError(f"{task_id} answer_policy missing {key}")
        if int(policy.get("min_supporting_evidence") or 0) <= 0:
            raise ValueError(f"{task_id} answer_policy min_supporting_evidence must be positive")
        if int(policy.get("low_confidence_min_supporting_evidence") or 0) <= 0:
            raise ValueError(f"{task_id} answer_policy low_confidence_min_supporting_evidence must be positive")
        keyword_frames = policy.get("keyword_frames") or {}
        if not isinstance(keyword_frames, dict) or not {"positive", "negative", "relation_predicates"} <= set(keyword_frames):
            raise ValueError(f"{task_id} answer_policy keyword_frames must define positive/negative/relation_predicates")
        speaker = policy.get("speaker_attribution") or {}
        if not isinstance(speaker, dict) or speaker.get("fixed_room_is_actor") is not False:
            raise ValueError(f"{task_id} answer_policy speaker_attribution.fixed_room_is_actor must be false")
        if task_id == "MM26-QA02":
            filters = policy.get("time_filters") or {}
            if filters.get("day_filter") != 1:
                raise ValueError("MM26-QA02 day_filter must be 1 for Tuesday")


def _validate_task_set(tasks: list[dict[str, Any]], *, official_queries: dict[str, str] | None) -> None:
    ids = [task["task_id"] for task in tasks]
    expected = expected_task_ids()
    if sorted(ids) != sorted(expected) or len(ids) != len(set(ids)):
        raise ValueError("registry must cover exactly 36 unique MM26 tasks")
    if official_queries is not None:
        if sorted(official_queries) != sorted(expected):
            raise ValueError(f"official page query count/id mismatch: {len(official_queries)}")
        mismatches = []
        for task in tasks:
            page_text = official_queries.get(task["task_id"])
            if page_text != task["official_text"]:
                mismatches.append({"task_id": task["task_id"], "official": page_text, "registry": task["official_text"]})
        if mismatches:
            raise ValueError(f"official text mismatches: {mismatches[:3]}")


def _task_from_dict(task: dict[str, Any]) -> TaskQuery:
    return TaskQuery(
        task_id=task["task_id"],
        task_type=task["task_type"],
        official_text=task["official_text"],
        canonical_text_for_retrieval=task["canonical_text_for_retrieval"],
        visual_prompts=tuple(task.get("visual_prompts") or ()),
        text_prompts=tuple(task.get("text_prompts") or ()),
        negative_prompts=tuple(task.get("negative_prompts") or ()),
        recall_profile=task.get("recall_profile", ""),
        scoring_profile=task.get("scoring_profile", ""),
        threshold_profile=task.get("threshold_profile", ""),
        output_profile=task.get("output_profile", ""),
        expected_granularity=copy.deepcopy(task.get("expected_granularity") or {}),
        evidence_policy=copy.deepcopy(task.get("evidence_policy") or {}),
        answer_policy=copy.deepcopy(task.get("answer_policy") or {}),
        raw=copy.deepcopy(task),
        resolved_profiles=copy.deepcopy(task.get("resolved_profiles") or {}),
    )
