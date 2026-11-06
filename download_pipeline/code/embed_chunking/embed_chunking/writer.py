from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .config import ChunkingConfig


META_FILES = {
    "manifest.jsonl",
    "summary.json",
    "skipped.jsonl",
    "invalid.jsonl",
    "config.json",
    "chunking.log",
}


def validate_output_root(output_root: str | Path, input_root: str | Path | None = None) -> Path:
    root = Path(output_root)
    if root.name != "embed_chunks":
        raise ValueError("output_root must be the embed_chunks directory, not processed_data or another directory")
    if input_root is not None and root.resolve() == Path(input_root).resolve():
        raise ValueError("output_root must not be the input_root")
    return root


def write_outputs(
    *,
    output_root: str | Path,
    chunks: list[dict[str, object]],
    manifest_rows: list[dict[str, object]],
    skipped_rows: list[dict[str, object]],
    invalid_rows: list[dict[str, object]],
    summary: dict[str, object],
    config: ChunkingConfig,
    log_lines: Iterable[str],
) -> None:
    root = validate_output_root(output_root)
    meta_root = root / "meta"
    root.mkdir(parents=True, exist_ok=True)
    meta_root.mkdir(parents=True, exist_ok=True)

    _write_jsonl(root / "chunks.jsonl", chunks)
    _write_jsonl(meta_root / "manifest.jsonl", manifest_rows)
    _write_jsonl(meta_root / "skipped.jsonl", skipped_rows)
    _write_jsonl(meta_root / "invalid.jsonl", invalid_rows)
    _write_json(meta_root / "summary.json", summary)
    _write_json(meta_root / "config.json", config.to_dict())
    (meta_root / "chunking.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
