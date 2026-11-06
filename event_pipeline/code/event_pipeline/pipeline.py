from __future__ import annotations

from pathlib import Path
from typing import Optional

from .aggregate_l4_l5 import build_l4_episodes, build_l4_l5, build_l5_sessions
from .config import AppConfig
from .corpus_index import build_corpus_index
from .event_embeddings import build_event_embeddings
from .frame_atoms import build_frame_atoms
from .fusion import build_fused_boundaries
from .group_l3 import build_l3_groups
from .inventory import build_inventory
from .links import build_links
from .reports import write_run_summary
from .segments_l1_l2 import build_source_segments
from .sparse_features import build_sparse_features
from .text_atoms import build_text_atoms
from .text_features import build_text_features
from .validation import validate_run
from .visual_features import build_visual_features


def run_root(config: AppConfig, output_mode: str, run_id: Optional[str]) -> Path:
    return config.run_output_root(output_mode, run_id)


def run_stage(config: AppConfig, stage: str, *, output_mode: str, run_id: Optional[str], day: Optional[str] = None, source: Optional[str] = None, resume: bool = False) -> list[dict]:
    root = run_root(config, output_mode, run_id)
    common = {"output_mode": output_mode, "run_id": run_id, "day": day, "source": source, "resume": resume}
    if stage == "inventory":
        return [build_inventory(config, root, **common)]
    if stage == "build-atoms":
        return [build_frame_atoms(config, root, **common), build_text_atoms(config, root, **common)]
    if stage == "visual-boundaries":
        return [build_visual_features(config, root, **common)]
    if stage == "text-boundaries":
        return [build_text_features(config, root, **common)]
    if stage == "sparse-keyword-features":
        return [build_sparse_features(config, root, **common)]
    if stage == "fuse-boundaries":
        return [build_fused_boundaries(config, root, **common)]
    if stage == "segment-sources":
        return [build_source_segments(config, root, **common)]
    if stage == "group-global-events":
        return [build_l3_groups(config, root, **common)]
    if stage == "build-episodes":
        return [build_l4_episodes(config, root, **common)]
    if stage == "build-sessions":
        return [build_l5_sessions(config, root, **common)]
    if stage == "build-l4-l5":
        return [build_l4_l5(config, root, **common)]
    if stage == "build-event-embeddings":
        return [build_event_embeddings(config, root, **common)]
    if stage == "build-corpus-index":
        return [build_links(config, root, **common), build_corpus_index(config, root, **common)]
    if stage == "export-event-corpus":
        return [build_corpus_index(config, root, **common)]
    if stage == "validate":
        report = validate_run(config, root)
        write_run_summary(root, report)
        if not report.get("valid"):
            raise SystemExit(1)
        return [{"stage": "validate", "report": report}]
    raise ValueError(f"unknown stage: {stage}")


RUN_ALL_STAGES = [
    "inventory",
    "build-atoms",
    "visual-boundaries",
    "text-boundaries",
    "sparse-keyword-features",
    "fuse-boundaries",
    "segment-sources",
    "group-global-events",
    "build-episodes",
    "build-sessions",
    "build-event-embeddings",
    "build-corpus-index",
    "validate",
]


def run_all(config: AppConfig, *, output_mode: str, run_id: Optional[str], day: Optional[str] = None, source: Optional[str] = None, resume: bool = False) -> list[dict]:
    results = []
    for stage in RUN_ALL_STAGES:
        results.extend(run_stage(config, stage, output_mode=output_mode, run_id=run_id, day=day, source=source, resume=resume))
    return results
