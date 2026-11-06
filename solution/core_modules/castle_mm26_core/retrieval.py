from __future__ import annotations

import heapq
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .config import AppConfig
from .data import discover_parquet_files
from .models import FrameHit, TaskQuery, TextHit
from .runtimes import BGEM3SparseRuntime, Qwen3DenseRuntime, SparseVector, TipsV2Runtime, decode_sparse_binary


VISUAL_COLUMNS = [
    "embedding_id",
    "day",
    "source",
    "video_hour_stem",
    "frame_index",
    "absolute_second",
    "frame_rel_path",
    "embedding",
]

TEXT_DENSE_COLUMNS = [
    "embedding_id",
    "embedding_chunk_id",
    "day",
    "view",
    "start_time",
    "end_time",
    "duration_sec",
    "text_for_embedding",
    "text_for_display",
    "quality_flags_json",
    "contains_invalid_source_chunk",
    "contains_repetition_compressed_source",
    "embedding",
]

SPARSE_COLUMNS = [
    "embedding_chunk_id",
    "day",
    "view",
    "start_time",
    "end_time",
    "duration_sec",
    "text_for_embedding",
    "text_for_display",
    "quality_flags_json",
    "contains_invalid_source_chunk",
    "contains_repetition_compressed_source",
    "bge_sparse_indices_i32",
    "bge_sparse_values_f16",
    "sparse_nnz",
]


@dataclass(frozen=True)
class _BatchQuerySpec:
    query: TaskQuery
    prompts: list[str]
    prompt_start: int
    prompt_end: int
    top_k: int
    reserve_primary: int
    reserve_secondary: int = 0


def retrieve_visual_cls(
    config: AppConfig,
    query: TaskQuery,
    runtime: TipsV2Runtime,
    *,
    smoke_limit_files: int | None = None,
    smoke_top_k: int | None = None,
    rows_per_batch: int = 8192,
) -> tuple[list[FrameHit], dict]:
    return retrieve_visual_cls_batch(
        config,
        [query],
        runtime,
        smoke_limit_files=smoke_limit_files,
        smoke_top_k_by_task={query.task_id: smoke_top_k} if smoke_top_k is not None else None,
        rows_per_batch=rows_per_batch,
    )[query.task_id]


def retrieve_visual_cls_batch(
    config: AppConfig,
    queries: Sequence[TaskQuery],
    runtime: TipsV2Runtime,
    *,
    smoke_limit_files: int | None = None,
    smoke_top_k_by_task: Mapping[str, int | None] | None = None,
    rows_per_batch: int = 8192,
) -> dict[str, tuple[list[FrameHit], dict]]:
    specs: list[_BatchQuerySpec] = []
    all_prompts: list[str] = []
    for query in queries:
        spec = _visual_spec(query, smoke_top_k_by_task)
        start = len(all_prompts)
        all_prompts.extend(spec.prompts)
        specs.append(_BatchQuerySpec(spec.query, spec.prompts, start, len(all_prompts), spec.top_k, spec.reserve_primary, spec.reserve_secondary))
    if not specs:
        return {}
    prompt_embeddings = runtime.encode_text(all_prompts)
    files = discover_parquet_files(config.input_path("tipsv2_root"), max_files=smoke_limit_files)
    global_heaps: dict[str, list[tuple[float, int, dict]]] = {spec.query.task_id: [] for spec in specs}
    day_source_heaps: dict[str, dict[tuple[int, str], list[tuple[float, int, dict]]]] = {spec.query.task_id: defaultdict(list) for spec in specs}
    hour_heaps: dict[str, dict[tuple[int, str, str], list[tuple[float, int, dict]]]] = {spec.query.task_id: defaultdict(list) for spec in specs}
    row_serial = 0
    rows_scanned = 0
    for parquet_file in files:
        parquet = pq.ParquetFile(parquet_file)
        for batch in parquet.iter_batches(batch_size=rows_per_batch, columns=VISUAL_COLUMNS, use_threads=False):
            matrix = _fixed_list_to_numpy(batch.column(batch.schema.get_field_index("embedding")), expected_dim=prompt_embeddings.shape[1])
            scores = matrix @ prompt_embeddings.T
            rows = _batch_rows(batch, exclude={"embedding"})
            rows_scanned += len(rows)
            for row_index, row in enumerate(rows):
                row["parquet_file"] = parquet_file.as_posix()
                day = int(row["day"])
                source = str(row["source"])
                hour = str(row["video_hour_stem"])
                for spec in specs:
                    task_scores = scores[row_index, spec.prompt_start : spec.prompt_end]
                    prompt_offset = int(np.argmax(task_scores))
                    score = float(task_scores[prompt_offset])
                    payload = dict(row)
                    payload["prompt"] = spec.prompts[prompt_offset]
                    payload["prompt_index"] = prompt_offset
                    payload["cls_score"] = score
                    task_id = spec.query.task_id
                    _push_top(global_heaps[task_id], (score, row_serial, payload), spec.top_k)
                    if spec.reserve_primary > 0:
                        _push_top(day_source_heaps[task_id][(day, source)], (score, row_serial, payload), spec.reserve_primary)
                    if spec.reserve_secondary > 0:
                        _push_top(hour_heaps[task_id][(day, source, hour)], (score, row_serial, payload), spec.reserve_secondary)
                    row_serial += 1
    results: dict[str, tuple[list[FrameHit], dict]] = {}
    for spec in specs:
        task_id = spec.query.task_id
        records = []
        records.extend(item for _, _, item in global_heaps[task_id])
        for heap in day_source_heaps[task_id].values():
            records.extend(item for _, _, item in heap)
        for heap in hour_heaps[task_id].values():
            records.extend(item for _, _, item in heap)
        hits = _frame_hits_from_records(spec.query, records)
        manifest = {
            "task_id": task_id,
            "prompt_count": len(spec.prompts),
            "parquet_files": len(files),
            "file_limit": smoke_limit_files,
            "global_recall": smoke_limit_files is None,
            "rows_scanned": rows_scanned,
            "candidate_records": len(records),
            "deduped_frame_hits": len(hits),
            "top_k": spec.top_k,
            "top_k_scope": "task",
            "reserve_per_day_source": spec.reserve_primary,
            "reserve_per_source_hour": spec.reserve_secondary,
            "visual_probe_mode": spec.query.task_type == "question_answering",
            "batch_retrieval": True,
            "batch_task_count": len(specs),
        }
        results[task_id] = (hits, manifest)
    return results


def _visual_spec(query: TaskQuery, smoke_top_k_by_task: Mapping[str, int | None] | None) -> _BatchQuerySpec:
    profile = query.resolved_profiles["recall_profile"]["value"]
    if query.task_type == "question_answering":
        override = (smoke_top_k_by_task or {}).get(query.task_id)
        top_k = int(override or profile.get("visual_probe_top_k", profile.get("visual_top_k", 100)))
        reserve_day_source = int(profile.get("visual_probe_reserve_per_day_source", profile.get("visual_reserve_per_day_source", 0)))
        reserve_source_hour = int(profile.get("visual_probe_reserve_per_source_hour", profile.get("visual_reserve_per_source_hour", 0)))
    else:
        override = (smoke_top_k_by_task or {}).get(query.task_id)
        top_k = int(override or profile.get("visual_top_k", 100))
        reserve_day_source = int(profile.get("visual_reserve_per_day_source", 0))
        reserve_source_hour = int(profile.get("visual_reserve_per_source_hour", 0))
    return _BatchQuerySpec(query, query.prompt_set("visual"), 0, 0, top_k, reserve_day_source, reserve_source_hour)


def retrieve_text_dense(
    config: AppConfig,
    query: TaskQuery,
    runtime: Qwen3DenseRuntime,
    *,
    smoke_limit_files: int | None = None,
    smoke_top_k: int | None = None,
    rows_per_batch: int = 4096,
) -> tuple[list[TextHit], dict]:
    return retrieve_text_dense_batch(
        config,
        [query],
        runtime,
        smoke_limit_files=smoke_limit_files,
        smoke_top_k_by_task={query.task_id: smoke_top_k} if smoke_top_k is not None else None,
        rows_per_batch=rows_per_batch,
    )[query.task_id]


def retrieve_text_dense_batch(
    config: AppConfig,
    queries: Sequence[TaskQuery],
    runtime: Qwen3DenseRuntime,
    *,
    smoke_limit_files: int | None = None,
    smoke_top_k_by_task: Mapping[str, int | None] | None = None,
    rows_per_batch: int = 4096,
) -> dict[str, tuple[list[TextHit], dict]]:
    specs: list[_BatchQuerySpec] = []
    all_prompts: list[str] = []
    for query in queries:
        spec = _text_spec(query, smoke_top_k_by_task, dense=True)
        start = len(all_prompts)
        all_prompts.extend(spec.prompts)
        specs.append(_BatchQuerySpec(spec.query, spec.prompts, start, len(all_prompts), spec.top_k, spec.reserve_primary))
    if not specs:
        return {}
    query_embeddings = runtime.encode(all_prompts)
    files = discover_parquet_files(config.input_path("qwen3_dense_root"), max_files=smoke_limit_files)
    global_heaps: dict[str, list[tuple[float, int, dict]]] = {spec.query.task_id: [] for spec in specs}
    source_heaps: dict[str, dict[str, list[tuple[float, int, dict]]]] = {spec.query.task_id: defaultdict(list) for spec in specs}
    row_serial = 0
    rows_scanned = 0
    for parquet_file in files:
        parquet = pq.ParquetFile(parquet_file)
        for batch in parquet.iter_batches(batch_size=rows_per_batch, columns=TEXT_DENSE_COLUMNS, use_threads=False):
            matrix = _fixed_list_to_numpy(batch.column(batch.schema.get_field_index("embedding")), expected_dim=query_embeddings.shape[1])
            scores = matrix @ query_embeddings.T
            rows = _batch_rows(batch, exclude={"embedding"})
            rows_scanned += len(rows)
            for row_index, row in enumerate(rows):
                row["parquet_file"] = parquet_file.as_posix()
                source = str(row["view"])
                for spec in specs:
                    task_scores = scores[row_index, spec.prompt_start : spec.prompt_end]
                    prompt_offset = int(np.argmax(task_scores))
                    score = float(task_scores[prompt_offset])
                    payload = dict(row)
                    payload["prompt"] = spec.prompts[prompt_offset]
                    payload["prompt_index"] = prompt_offset
                    payload["dense_score"] = score
                    task_id = spec.query.task_id
                    _push_top(global_heaps[task_id], (score, row_serial, payload), spec.top_k)
                    if spec.reserve_primary > 0:
                        _push_top(source_heaps[task_id][source], (score, row_serial, payload), spec.reserve_primary)
                    row_serial += 1
    results: dict[str, tuple[list[TextHit], dict]] = {}
    for spec in specs:
        task_id = spec.query.task_id
        records = []
        records.extend(item for _, _, item in global_heaps[task_id])
        for heap in source_heaps[task_id].values():
            records.extend(item for _, _, item in heap)
        hits = _text_hits_from_dense_records(spec.query, records)
        manifest = {
            "task_id": task_id,
            "prompt_count": len(spec.prompts),
            "parquet_files": len(files),
            "file_limit": smoke_limit_files,
            "global_recall": smoke_limit_files is None,
            "rows_scanned": rows_scanned,
            "candidate_records": len(records),
            "deduped_text_hits": len(hits),
            "top_k": spec.top_k,
            "top_k_scope": "task",
            "reserve_per_source": spec.reserve_primary,
            "batch_retrieval": True,
            "batch_task_count": len(specs),
        }
        results[task_id] = (hits, manifest)
    return results


def _text_spec(query: TaskQuery, smoke_top_k_by_task: Mapping[str, int | None] | None, *, dense: bool) -> _BatchQuerySpec:
    profile = query.resolved_profiles["recall_profile"]["value"]
    override = (smoke_top_k_by_task or {}).get(query.task_id)
    key = "text_dense_top_k" if dense else "text_sparse_top_k"
    top_k = int(override or profile.get(key, profile.get("text_rerank_input_cap", 100)))
    reserve_source = int(profile.get("text_reserve_per_source", 0))
    return _BatchQuerySpec(query, query.prompt_set("text"), 0, 0, top_k, reserve_source)


def retrieve_text_sparse(
    config: AppConfig,
    query: TaskQuery,
    runtime: BGEM3SparseRuntime,
    *,
    smoke_limit_files: int | None = None,
    smoke_top_k: int | None = None,
    rows_per_batch: int = 2048,
) -> tuple[list[TextHit], dict]:
    return retrieve_text_sparse_batch(
        config,
        [query],
        runtime,
        smoke_limit_files=smoke_limit_files,
        smoke_top_k_by_task={query.task_id: smoke_top_k} if smoke_top_k is not None else None,
        rows_per_batch=rows_per_batch,
    )[query.task_id]


def retrieve_text_sparse_batch(
    config: AppConfig,
    queries: Sequence[TaskQuery],
    runtime: BGEM3SparseRuntime,
    *,
    smoke_limit_files: int | None = None,
    smoke_top_k_by_task: Mapping[str, int | None] | None = None,
    rows_per_batch: int = 2048,
) -> dict[str, tuple[list[TextHit], dict]]:
    specs: list[_BatchQuerySpec] = []
    all_prompts: list[str] = []
    for query in queries:
        spec = _text_spec(query, smoke_top_k_by_task, dense=False)
        start = len(all_prompts)
        all_prompts.extend(spec.prompts)
        specs.append(_BatchQuerySpec(spec.query, spec.prompts, start, len(all_prompts), spec.top_k, spec.reserve_primary))
    if not specs:
        return {}
    query_vectors = runtime.encode(all_prompts)
    postings = _query_postings(query_vectors)
    files = discover_parquet_files(config.input_path("bgem3_sparse_root"), max_files=smoke_limit_files)
    global_heaps: dict[str, list[tuple[float, int, dict]]] = {spec.query.task_id: [] for spec in specs}
    source_heaps: dict[str, dict[str, list[tuple[float, int, dict]]]] = {spec.query.task_id: defaultdict(list) for spec in specs}
    row_serial = 0
    rows_scanned = 0
    for parquet_file in files:
        parquet = pq.ParquetFile(parquet_file)
        for batch in parquet.iter_batches(batch_size=rows_per_batch, columns=SPARSE_COLUMNS, use_threads=False):
            rows = batch.to_pylist()
            rows_scanned += len(rows)
            for row in rows:
                vector = decode_sparse_binary(row["bge_sparse_indices_i32"], row["bge_sparse_values_f16"])
                sparse_nnz = int(row.get("sparse_nnz") or 0)
                if sparse_nnz != len(vector.indices) or sparse_nnz != len(vector.values):
                    raise ValueError(f"BGE sparse_nnz mismatch for {row.get('embedding_chunk_id')}: sparse_nnz={sparse_nnz}, decoded={len(vector.indices)}")
                scores = sparse_dot(vector, postings, len(all_prompts))
                row["parquet_file"] = parquet_file.as_posix()
                source = str(row["view"])
                for spec in specs:
                    task_scores = scores[spec.prompt_start : spec.prompt_end]
                    prompt_offset = int(np.argmax(task_scores))
                    score = float(task_scores[prompt_offset])
                    payload = dict(row)
                    payload["prompt"] = spec.prompts[prompt_offset]
                    payload["prompt_index"] = prompt_offset
                    payload["sparse_score"] = score
                    task_id = spec.query.task_id
                    _push_top(global_heaps[task_id], (score, row_serial, payload), spec.top_k)
                    if spec.reserve_primary > 0:
                        _push_top(source_heaps[task_id][source], (score, row_serial, payload), spec.reserve_primary)
                    row_serial += 1
    results: dict[str, tuple[list[TextHit], dict]] = {}
    for spec in specs:
        task_id = spec.query.task_id
        records = []
        records.extend(item for _, _, item in global_heaps[task_id])
        for heap in source_heaps[task_id].values():
            records.extend(item for _, _, item in heap)
        hits = _text_hits_from_sparse_records(spec.query, records)
        manifest = {
            "task_id": task_id,
            "prompt_count": len(spec.prompts),
            "parquet_files": len(files),
            "file_limit": smoke_limit_files,
            "global_recall": smoke_limit_files is None,
            "rows_scanned": rows_scanned,
            "candidate_records": len(records),
            "deduped_text_hits": len(hits),
            "top_k": spec.top_k,
            "top_k_scope": "task",
            "reserve_per_source": spec.reserve_primary,
            "batch_retrieval": True,
            "batch_task_count": len(specs),
        }
        results[task_id] = (hits, manifest)
    return results


def merge_text_hits(dense_hits: Sequence[TextHit], sparse_hits: Sequence[TextHit]) -> list[TextHit]:
    merged: dict[str, TextHit] = {}
    for hit in dense_hits:
        merged[hit.embedding_chunk_id] = hit
    for hit in sparse_hits:
        existing = merged.get(hit.embedding_chunk_id)
        if existing is None:
            merged[hit.embedding_chunk_id] = hit
        else:
            existing.sparse_score = hit.sparse_score
            existing.provenance["sparse"] = hit.provenance
    rows = list(merged.values())
    rows.sort(key=lambda hit: (-(hit.dense_score or -1e9), -(hit.sparse_score or -1e9), hit.embedding_chunk_id))
    for index, hit in enumerate(rows, start=1):
        hit.rank = index
    return rows


def _push_top(heap: list[tuple[float, int, dict]], item: tuple[float, int, dict], limit: int) -> None:
    if limit <= 0:
        return
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item[0] > heap[0][0]:
        heapq.heapreplace(heap, item)


def _fixed_list_to_numpy(array: pa.Array, *, expected_dim: int) -> np.ndarray:
    if not pa.types.is_fixed_size_list(array.type):
        raise ValueError(f"embedding column must be fixed_size_list, got {array.type}")
    if array.type.list_size != expected_dim:
        raise ValueError(f"embedding dimension must be {expected_dim}, got {array.type.list_size}")
    start = array.offset * expected_dim
    values = array.values.slice(start, len(array) * expected_dim).to_numpy(zero_copy_only=False)
    matrix = np.asarray(values, dtype=np.float32).reshape((len(array), expected_dim))
    if not np.isfinite(matrix).all():
        raise ValueError("embedding matrix contains non-finite values")
    return matrix


def _batch_rows(batch: pa.RecordBatch, *, exclude: set[str]) -> list[dict]:
    names = [name for name in batch.schema.names if name not in exclude]
    columns = {name: batch.column(batch.schema.get_field_index(name)).to_pylist() for name in names}
    rows = []
    for index in range(batch.num_rows):
        rows.append({name: values[index] for name, values in columns.items()})
    return rows


def _frame_hits_from_records(query: TaskQuery, records: Sequence[dict]) -> list[FrameHit]:
    hits: list[FrameHit] = []
    for record_index, row in enumerate(records):
        embedding_id = str(row["embedding_id"])
        score = float(row.get("score", row.get("similarity", 0.0))) if "score" in row else float(row.get("_score", 0.0))
        # heap item score is not inside payload, so recover it from provenance if available later.
        score = float(row.get("cls_score", row.get("similarity_score", row.get("score", 0.0))))
        if score == 0.0 and "_heap_score" in row:
            score = float(row["_heap_score"])
        prompt_index = row.get("prompt_index")
        candidate_record_id = f"{embedding_id}:prompt{prompt_index if prompt_index is not None else 'na'}:record{record_index:08d}"
        hit = FrameHit(
            task_id=query.task_id,
            prompt=str(row.get("prompt") or query.canonical_text_for_retrieval),
            embedding_id=embedding_id,
            day=int(row["day"]),
            source=str(row["source"]),
            absolute_second=float(row["absolute_second"]),
            frame_rel_path=str(row.get("frame_rel_path") or ""),
            image_path_ref=str(row.get("frame_rel_path") or ""),
            cls_score=score,
            parquet_file=str(row.get("parquet_file") or ""),
            provenance={
                "prompt_index": prompt_index,
                "candidate_record_id": candidate_record_id,
                "duplicate_frame_key": embedding_id,
                "pre_patch_candidate_retention": "kept_full_top_k_plus_reserve_record",
            },
        )
        hits.append(hit)
    hits = sorted(hits, key=lambda item: (-item.cls_score, item.embedding_id, str(item.provenance.get("candidate_record_id") or "")))
    for index, hit in enumerate(hits, start=1):
        hit.rank = index
    return hits


def _text_hits_from_dense_records(query: TaskQuery, records: Sequence[dict]) -> list[TextHit]:
    best: dict[str, TextHit] = {}
    for row in records:
        score = float(row.get("dense_score", row.get("score", row.get("_heap_score", 0.0))))
        hit = TextHit(
            task_id=query.task_id,
            prompt=str(row.get("prompt") or query.canonical_text_for_retrieval),
            embedding_chunk_id=str(row["embedding_chunk_id"]),
            day=_day_to_int(row["day"]),
            source=str(row["view"]),
            start_second=_optional_float(row.get("start_time")),
            end_second=_optional_float(row.get("end_time")),
            text_for_embedding=str(row.get("text_for_embedding") or ""),
            text_for_display=str(row.get("text_for_display") or ""),
            dense_score=score,
            quality_flags=_quality_flags(row),
            provenance={key: row.get(key) for key in ("parquet_file", "contains_invalid_source_chunk", "contains_repetition_compressed_source")},
        )
        current = best.get(hit.embedding_chunk_id)
        if current is None or (hit.dense_score or -1e9) > (current.dense_score or -1e9):
            best[hit.embedding_chunk_id] = hit
    hits = sorted(best.values(), key=lambda item: (-(item.dense_score or 0.0), item.embedding_chunk_id))
    for index, hit in enumerate(hits, start=1):
        hit.rank = index
    return hits


def _text_hits_from_sparse_records(query: TaskQuery, records: Sequence[dict]) -> list[TextHit]:
    best: dict[str, TextHit] = {}
    for row in records:
        score = float(row.get("sparse_score", row.get("score", row.get("_heap_score", 0.0))))
        hit = TextHit(
            task_id=query.task_id,
            prompt=str(row.get("prompt") or query.canonical_text_for_retrieval),
            embedding_chunk_id=str(row["embedding_chunk_id"]),
            day=_day_to_int(row["day"]),
            source=str(row["view"]),
            start_second=_optional_float(row.get("start_time")),
            end_second=_optional_float(row.get("end_time")),
            text_for_embedding=str(row.get("text_for_embedding") or ""),
            text_for_display=str(row.get("text_for_display") or ""),
            sparse_score=score,
            quality_flags=_quality_flags(row),
            provenance={key: row.get(key) for key in ("parquet_file", "contains_invalid_source_chunk", "contains_repetition_compressed_source", "sparse_nnz")},
        )
        current = best.get(hit.embedding_chunk_id)
        if current is None or (hit.sparse_score or -1e9) > (current.sparse_score or -1e9):
            best[hit.embedding_chunk_id] = hit
    hits = sorted(best.values(), key=lambda item: (-(item.sparse_score or 0.0), item.embedding_chunk_id))
    for index, hit in enumerate(hits, start=1):
        hit.rank = index
    return hits


def _query_postings(vectors: Sequence[SparseVector]) -> dict[int, list[tuple[int, float]]]:
    postings: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for query_index, vector in enumerate(vectors):
        for index, value in zip(vector.indices.tolist(), vector.values.tolist()):
            postings[int(index)].append((query_index, float(value)))
    return postings


def sparse_dot(document: SparseVector, postings: dict[int, list[tuple[int, float]]], query_count: int) -> np.ndarray:
    scores = np.zeros(query_count, dtype=np.float32)
    for index, doc_value in zip(document.indices.tolist(), document.values.tolist()):
        for query_index, query_value in postings.get(int(index), ()):
            scores[query_index] += np.float32(doc_value * query_value)
    return scores


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _day_to_int(value: object) -> int:
    text = str(value)
    if text.startswith("day"):
        return int(text[3:])
    return int(text)


def _quality_flags(row: dict) -> list[str]:
    raw = row.get("quality_flags_json")
    flags: list[str] = []
    if raw:
        try:
            parsed = json.loads(str(raw))
            if isinstance(parsed, list):
                flags.extend(str(item) for item in parsed)
        except json.JSONDecodeError:
            flags.append("quality_flags_parse_error")
    if row.get("contains_invalid_source_chunk"):
        flags.append("contains_invalid_source_chunk")
    if row.get("contains_repetition_compressed_source"):
        flags.append("contains_repetition_compressed_source")
    return flags
