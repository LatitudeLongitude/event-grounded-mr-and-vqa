from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TaskQuery:
    task_id: str
    task_type: str
    official_text: str
    canonical_text_for_retrieval: str
    visual_prompts: tuple[str, ...] = ()
    text_prompts: tuple[str, ...] = ()
    negative_prompts: tuple[str, ...] = ()
    recall_profile: str = ""
    scoring_profile: str = ""
    threshold_profile: str = ""
    output_profile: str = ""
    expected_granularity: dict[str, Any] = field(default_factory=dict)
    evidence_policy: dict[str, Any] = field(default_factory=dict)
    answer_policy: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    resolved_profiles: dict[str, Any] = field(default_factory=dict)

    def prompt_set(self, modality: str) -> list[str]:
        base = [self.canonical_text_for_retrieval]
        if modality == "visual":
            base.extend(self.visual_prompts)
        else:
            base.extend(self.text_prompts)
        aliases = self.raw.get("aliases") or {}
        base.extend(aliases.get("positive") or [])
        multilingual = self.raw.get("multilingual_aliases") or self.answer_policy.get("multilingual_aliases") or {}
        if isinstance(multilingual, dict):
            base.extend(multilingual.get("positive") or [])
        elif isinstance(multilingual, list):
            base.extend(multilingual)
        seen = set()
        result = []
        for item in base:
            text = str(item).strip()
            if text and text not in seen:
                seen.add(text)
                result.append(text)
        return result

    def negative_prompt_set(self) -> list[str]:
        aliases = self.raw.get("aliases") or {}
        base = list(self.negative_prompts)
        base.extend(aliases.get("negative") or [])
        seen = set()
        result = []
        for item in base:
            text = str(item).strip()
            if text and text.lower() not in seen:
                seen.add(text.lower())
                result.append(text)
        return result

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FrameHit:
    task_id: str
    prompt: str
    embedding_id: str
    day: int
    source: str
    absolute_second: float
    frame_rel_path: str
    image_path_ref: str
    cls_score: float
    rank: int = 0
    frame_atom_id: str | None = None
    patch_score: float | None = None
    patch_top5_mean: float | None = None
    parquet_file: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class TextHit:
    task_id: str
    prompt: str
    embedding_chunk_id: str
    day: int
    source: str
    start_second: float | None
    end_second: float | None
    text_for_embedding: str
    text_for_display: str
    dense_score: float | None = None
    sparse_score: float | None = None
    reranker_score: float | None = None
    reranker_raw_score: float | None = None
    negative_reranker_score: float | None = None
    negative_reranker_raw_score: float | None = None
    rank: int = 0
    text_atom_id: str | None = None
    quality_flags: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventCandidate:
    task_id: str
    candidate_id: str
    day: int
    source: str
    start_second: float
    end_second: float
    event_ids: list[str]
    frame_atom_ids: list[str]
    text_atom_ids: list[str]
    evidence_modalities: list[str]
    score_parts: dict[str, float]
    final_score: float
    rejected_reason: str | None = None
    split_reason: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceBundle:
    task_id: str
    csv_row_id: str
    candidate_id: str
    day: int
    source: str
    start_second: float
    end_second: float
    event_ids: list[str]
    frame_atom_ids: list[str]
    text_atom_ids: list[str]
    evidence_modalities: list[str]
    score_parts: dict[str, float]
    final_score: float
    split_reason: str | None = None
    rejected_reason: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QAEvidence:
    task_id: str
    evidence_id: str
    day_num: int
    source: str
    source_role: str
    start_second: float
    end_second: float
    text_atom_ids: list[str]
    event_ids: list[str]
    frame_atom_ids: list[str]
    original_image_refs: list[str]
    text_excerpt: str
    entity_mentions: list[dict[str, Any]]
    relation_candidates: list[dict[str, Any]]
    time_filter_fit: float
    reranker_score: float | None
    visual_probe_score: float | None
    qa_evidence_score: float
    qa_score_parts: dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnswerResult:
    task_id: str
    answer_value: str
    answer_type: str
    confidence: float
    supporting_evidence_ids: list[str]
    mllm_raw_json: dict[str, Any]
    mllm_validation_status: str
    mllm_model: str
    response_format_schema_name: str
    dedupe_trace: dict[str, Any]
    rejected_evidence_ids: list[str]
    answer_text: str

    def asdict(self) -> dict[str, Any]:
        return asdict(self)
