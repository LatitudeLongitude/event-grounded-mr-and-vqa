from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from .paths import ensure_under, reject_prohibited_path, resolve_path


@dataclass(frozen=True)
class InputRoots:
    processed_data_root: Path
    embed_chunks_root: Path
    tipsv2_root: Path
    qwen3_root: Path
    bgem3_root: Path
    view_meta_path: Path

    @property
    def approved_roots(self) -> tuple[Path, ...]:
        return (
            self.processed_data_root,
            self.embed_chunks_root,
            self.tipsv2_root,
            self.qwen3_root,
            self.bgem3_root,
            self.view_meta_path,
        )


@dataclass(frozen=True)
class RunConfig:
    batch_size: int = 8192
    # Accepted only for compatibility with archived YAML; writers use their fixed 8192 default.
    rows_per_parquet_group: int = 8192
    max_event_frames: int = 512
    sparse_top_n: int = 128


@dataclass(frozen=True)
class SegmentationConfig:
    fixed_room_l2_min_sec: int = 10
    ego_person_l2_min_sec: int = 3
    l2_max_sec: int = 180
    l3_max_sec: int = 300
    l4_max_sec: int = 1200
    l5_max_sec: int = 7200


@dataclass(frozen=True)
class AppConfig:
    config_path: Path
    project_root: Path
    module_root: Path
    input_roots: InputRoots
    output_root: Path
    run: RunConfig
    segmentation: SegmentationConfig
    raw: dict[str, Any]

    @staticmethod
    def _safe_run_id(output_mode: str, run_id: Optional[str]) -> str:
        if not run_id:
            raise ValueError(f"--run-id is required for {output_mode} mode")
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in run_id)
        if not safe:
            raise ValueError("run_id must contain at least one safe character")
        return safe

    @classmethod
    def load(cls, path: Union[str, Path] = "config.yaml") -> "AppConfig":
        config_path = Path(path)
        if not config_path.is_absolute():
            config_path = (Path.cwd() / config_path).resolve()
        else:
            config_path = config_path.resolve()
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        module_root = config_path.parent.resolve()
        project_root = module_root.parent.parent.resolve()
        paths = raw.get("paths") or {}

        input_roots = InputRoots(
            processed_data_root=resolve_path(paths["processed_data_root"], base=module_root),
            embed_chunks_root=resolve_path(paths["embed_chunks_root"], base=module_root),
            tipsv2_root=resolve_path(paths["tipsv2_root"], base=module_root),
            qwen3_root=resolve_path(paths["qwen3_root"], base=module_root),
            bgem3_root=resolve_path(paths["bgem3_root"], base=module_root),
            view_meta_path=resolve_path(paths["view_meta_path"], base=module_root),
        )
        for root in input_roots.approved_roots:
            reject_prohibited_path(root, role="approved input")

        output_root = resolve_path(paths["output_root"], base=module_root)
        event_data_root = project_root / "event_pipeline" / "event_data"
        ensure_under(output_root, event_data_root, role="output_root")

        return cls(
            config_path=config_path,
            project_root=project_root,
            module_root=module_root,
            input_roots=input_roots,
            output_root=output_root,
            run=RunConfig(**(raw.get("run") or {})),
            segmentation=SegmentationConfig(**(raw.get("segmentation") or {})),
            raw=raw,
        )

    def run_output_root(self, output_mode: str, run_id: Optional[str] = None) -> Path:
        if output_mode == "formal":
            return (self.output_root / "formal" / self._safe_run_id(output_mode, run_id)).resolve()
        if output_mode == "smoke":
            return (self.output_root / "smoke" / self._safe_run_id(output_mode, run_id)).resolve()
        raise ValueError(f"unsupported output mode: {output_mode}")

    def is_formal_run_root(self, run_root: Path) -> bool:
        formal_root = (self.output_root / "formal").resolve()
        resolved = run_root.resolve()
        return resolved.parent == formal_root

    def assert_can_read_input(self, path: Path) -> Path:
        reject_prohibited_path(path, role="input")
        resolved = path.resolve()
        if not any(resolved == root.resolve() or resolved.is_relative_to(root.resolve()) for root in self.input_roots.approved_roots):
            raise ValueError(f"input path is not approved: {resolved}")
        return resolved

    def assert_can_write_output(self, path: Path, output_mode: str, run_id: Optional[str]) -> Path:
        root = self.run_output_root(output_mode, run_id)
        return ensure_under(path, root, role="output path")
