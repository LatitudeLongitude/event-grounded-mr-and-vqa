from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pyarrow.parquet as pq

from .config import AppConfig
from .filters import source_filter_values, source_matches
from .manifest import file_record, params_hash, write_stage_manifest
from .parquet_io import atomic_write_json, write_table
from .parquet_io import iter_pylist_rows
from .paths import to_ref
from .run_state import stage_complete, stage_paths, update_run_state
from .schema import INVENTORY_STREAMS, SOURCE_HOUR_COVERAGE
from .source_meta import source_roles
from .time_utils import day_label, normalize_day, normalize_hour

STAGE = "inventory"


def _matches(day: str, source: str, day_filter: Optional[str], source_filter: Optional[str]) -> bool:
    return (day_filter is None or day == day_label(day_filter)) and source_matches(source, source_filter)


def _parquet_rows(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def _csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def _add_stream(rows: list[dict], config: AppConfig, role_map: dict, *, day: str, source: str, modality: str, hour: Optional[str], path: Path, row_count: int, exists: bool = True, is_seed: bool = False, flags: Optional[list[str]] = None, param_hash: str, input_manifest_ref: str) -> None:
    rows.append(
        {
            "day": day,
            "day_num": normalize_day(day),
            "source": source,
            "source_role": (role_map.get(source) or {}).get("source_role"),
            "modality": modality,
            "hour": hour,
            "path_ref": to_ref(path, config.project_root),
            "exists": exists,
            "row_count": int(row_count),
            "is_config_seed": is_seed,
            "quality_flags": flags or [],
            "created_by": STAGE,
            "params_hash": param_hash,
            "input_manifest_ref": input_manifest_ref,
        }
    )


def build_inventory(config: AppConfig, run_root: Path, *, output_mode: str, run_id: Optional[str], day: Optional[str] = None, source: Optional[str] = None, resume: bool = False) -> dict:
    paths = stage_paths(run_root)
    streams_path = run_root / "inventory" / "available_streams.parquet"
    coverage_path = run_root / "inventory" / "source_hour_coverage.parquet"
    input_manifest_path = run_root / "manifests" / "input_data_manifest.json"
    params = {"stage": STAGE, "output_mode": output_mode, "run_id": run_id, "day": day, "source": source}
    if resume and stage_complete(paths.stage_manifest(STAGE), params=params, output_paths=[streams_path, coverage_path, input_manifest_path]):
        return {"stage": STAGE, "skipped": True}
    param_hash = params_hash(params)
    input_manifest_ref = to_ref(input_manifest_path, config.project_root)

    role_map = source_roles(config.input_roots.view_meta_path)
    rows: list[dict] = []
    coverage: dict[tuple[str, str, str], dict[str, bool]] = defaultdict(dict)
    shard_records_by_path: dict[str, dict] = {}

    def record_shard(path: Path, *, modality: str, found_day: Optional[str], found_source: Optional[str], hour: Optional[str], row_count: int, hours: Optional[list[str]] = None) -> None:
        ref = to_ref(path, config.project_root)
        if ref in shard_records_by_path:
            return
        record = {
            **file_record(path, config.project_root),
            "modality": modality,
            "day": found_day,
            "source": found_source,
            "hour": hour,
            "hours": hours if hours is not None else ([] if hour is None else [hour]),
            "row_count": int(row_count),
        }
        shard_records_by_path[ref] = record

    for path in sorted(config.input_roots.processed_data_root.glob("day*/*/video/*/frame_manifest.csv")):
        try:
            found_day = day_label(path.parts[-5])
            found_source = path.parts[-4]
            hour = normalize_hour(path.parts[-2])
        except Exception:
            continue
        if not _matches(found_day, found_source, day, source):
            continue
        count = _csv_rows(path)
        record_shard(path, modality="frames_manifest", found_day=found_day, found_source=found_source, hour=hour, row_count=count)
        tips_path = config.input_roots.tipsv2_root / found_day / found_source / f"{hour}.parquet"
        tips_count = _parquet_rows(tips_path) if tips_path.exists() else 0
        record_shard(tips_path, modality="image_embedding", found_day=found_day, found_source=found_source, hour=hour, row_count=tips_count)
        coverage[(found_day, found_source, hour)]["frames_available"] = count > 0
        _add_stream(rows, config, role_map, day=found_day, source=found_source, modality="frames", hour=hour, path=path, row_count=count, param_hash=param_hash, input_manifest_ref=input_manifest_ref)

    for path in sorted(config.input_roots.tipsv2_root.glob("day*/*/*.parquet")):
        try:
            found_day = day_label(path.parts[-3])
            found_source = path.parts[-2]
            hour = normalize_hour(path.stem)
        except Exception:
            continue
        if not _matches(found_day, found_source, day, source):
            continue
        count = _parquet_rows(path)
        record_shard(path, modality="image_embedding", found_day=found_day, found_source=found_source, hour=hour, row_count=count)
        coverage[(found_day, found_source, hour)]["tipsv2_available"] = count > 0
        _add_stream(rows, config, role_map, day=found_day, source=found_source, modality="image_embedding", hour=hour, path=path, row_count=count, param_hash=param_hash, input_manifest_ref=input_manifest_ref)

    chunks_path = config.input_roots.embed_chunks_root / "chunks.jsonl"
    text_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    text_chunk_count = 0
    if chunks_path.exists():
        with chunks_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                obj = json.loads(line)
                found_day = day_label(obj["day"])
                found_source = obj["view"]
                if not _matches(found_day, found_source, day, source):
                    continue
                text_chunk_count += 1
                for hour_value in obj.get("hours") or [obj.get("start_hour")]:
                    if hour_value is None:
                        continue
                    hour = normalize_hour(hour_value)
                    text_counts[(found_day, found_source, hour)] += 1
                    coverage[(found_day, found_source, hour)]["text_available"] = True
    for (found_day, found_source, hour), count in sorted(text_counts.items()):
        _add_stream(rows, config, role_map, day=found_day, source=found_source, modality="transcript", hour=hour, path=chunks_path, row_count=count, param_hash=param_hash, input_manifest_ref=input_manifest_ref)
    if chunks_path.exists():
        record_shard(chunks_path, modality="transcript_chunks", found_day=day_label(day) if day else None, found_source=source, hour=None, row_count=text_chunk_count, hours=sorted({hour for _, _, hour in text_counts}))

    for modality, root, key_name in (
        ("qwen3_dense", config.input_roots.qwen3_root, "qwen3_available"),
        ("bgem3_sparse", config.input_roots.bgem3_root, "bgem3_available"),
    ):
        for path in sorted(root.glob("day*/*.parquet")):
            found_day = day_label(path.parts[-2])
            found_source = path.stem
            if not _matches(found_day, found_source, day, source):
                continue
            count = _parquet_rows(path)
            try:
                hours = sorted({normalize_hour(h) for row in iter_pylist_rows(path, columns=["hours_json"], batch_size=config.run.batch_size) for h in json.loads(row.get("hours_json") or "[]")})
            except Exception:
                hours = [None]
            record_shard(path, modality=modality, found_day=found_day, found_source=found_source, hour=None, row_count=count, hours=[hour for hour in hours if hour is not None])
            for hour in hours:
                if hour is not None:
                    coverage[(found_day, found_source, hour)][key_name] = True
                _add_stream(rows, config, role_map, day=found_day, source=found_source, modality=modality, hour=hour, path=path, row_count=count, param_hash=param_hash, input_manifest_ref=input_manifest_ref)

    scanned = {(row["day"], row["source"]) for row in rows}
    seed_days = sorted({item[0] for item in scanned} | ({day_label(day)} if day else {"day1", "day2", "day3", "day4"}))
    for seed_source, payload in sorted(role_map.items()):
        if not source_matches(seed_source, source):
            continue
        for seed_day in seed_days:
            if (seed_day, seed_source) not in scanned:
                _add_stream(rows, config, role_map, day=seed_day, source=seed_source, modality="config_seed", hour=None, path=config.input_roots.view_meta_path, row_count=0, exists=False, is_seed=True, flags=["missing_in_scanned_data"], param_hash=param_hash, input_manifest_ref=input_manifest_ref)

    expected_sources = sorted({row["source"] for row in rows} | set(role_map))
    source_values = source_filter_values(source)
    if source_values:
        expected_sources = sorted(source_values)
    expected_hours = [f"{hour:02d}" for hour in range(24)]
    for seed_day in seed_days:
        if day and seed_day != day_label(day):
            continue
        for expected_source in expected_sources:
            for expected_hour in expected_hours:
                coverage.setdefault((seed_day, expected_source, expected_hour), {})

    coverage_rows = []
    for found_day, found_source, hour in sorted(coverage):
        item = coverage[(found_day, found_source, hour)]
        missing = [
            label
            for key, label in (
                ("frames_available", "frames"),
                ("tipsv2_available", "image_embedding"),
                ("text_available", "transcript"),
                ("qwen3_available", "qwen3_dense"),
                ("bgem3_available", "bgem3_sparse"),
            )
            if not item.get(key)
        ]
        coverage_rows.append(
            {
                "day": found_day,
                "day_num": normalize_day(found_day),
                "source": found_source,
                "source_role": (role_map.get(found_source) or {}).get("source_role"),
                "hour": hour,
                "frames_available": bool(item.get("frames_available")),
                "tipsv2_available": bool(item.get("tipsv2_available")),
                "text_available": bool(item.get("text_available")),
                "qwen3_available": bool(item.get("qwen3_available")),
                "bgem3_available": bool(item.get("bgem3_available")),
                "missing_modalities": missing,
                "created_by": STAGE,
                "params_hash": param_hash,
                "input_manifest_ref": input_manifest_ref,
            }
        )

    stream_result = write_table(streams_path, rows, schema=INVENTORY_STREAMS)
    coverage_result = write_table(coverage_path, coverage_rows, schema=SOURCE_HOUR_COVERAGE)
    input_manifest = {
        "created_by": STAGE,
        "approved_input_roots": [to_ref(path, config.project_root) for path in config.input_roots.approved_roots],
        "filters": {"day": day, "source": source},
        "input_shards": sorted(shard_records_by_path.values(), key=lambda item: str(item.get("path"))),
        "shard_count": len(shard_records_by_path),
        "stream_count": len(rows),
        "coverage_count": len(coverage_rows),
        "sources": sorted({row["source"] for row in rows}),
    }
    atomic_write_json(input_manifest_path, input_manifest)

    manifest = write_stage_manifest(
        paths.stage_manifest(STAGE),
        stage=STAGE,
        params=params,
        inputs=[file_record(config.input_roots.view_meta_path, config.project_root), *sorted(shard_records_by_path.values(), key=lambda item: str(item.get("path")))],
        outputs=[
            {**stream_result, "schema": str(INVENTORY_STREAMS)},
            {**coverage_result, "schema": str(SOURCE_HOUR_COVERAGE)},
            file_record(input_manifest_path, config.project_root),
        ],
        schema_names=["available_streams", "source_hour_coverage"],
        project_root=config.project_root,
    )
    update_run_state(run_root, STAGE, to_ref(paths.stage_manifest(STAGE), config.project_root))
    return {"stage": STAGE, "manifest": manifest}
