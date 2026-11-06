from __future__ import annotations

import json
import os
import uuid
from pathlib import Path


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_report(json_path: Path | str, markdown_path: Path | str, report: dict, *, markdown_content: str | None = None) -> None:
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    if json_output.exists() or markdown_output.exists():
        raise FileExistsError(f"Refusing to overwrite report: {json_output}")
    _atomic_text(json_output, json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n")
    if markdown_content is None:
        markdown_content = "# BGE-M3 Learned Sparse Embedding Report\n\n```json\n" + json.dumps(report, indent=2, sort_keys=True) + "\n```\n"
    _atomic_text(markdown_output, markdown_content)


def write_json_report(json_path: Path | str, report: dict) -> None:
    json_output = Path(json_path)
    if json_output.exists():
        raise FileExistsError(f"Refusing to overwrite report: {json_output}")
    _atomic_text(json_output, json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n")


def append_event(
    jsonl_path: Path | str,
    text_path: Path | str,
    *,
    run_id: str,
    invocation_id: str,
    level: str,
    event_type: str,
    payload: dict,
) -> None:
    item = {
        "run_id": run_id,
        "invocation_id": invocation_id,
        "level": level,
        "event_type": event_type,
        "payload": payload,
    }
    jsonl = Path(jsonl_path)
    text = Path(text_path)
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    text.parent.mkdir(parents=True, exist_ok=True)
    with jsonl.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(item, sort_keys=True, ensure_ascii=True) + "\n")
    with text.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"[{level}] {event_type} {json.dumps(payload, sort_keys=True, ensure_ascii=True)}\n")

