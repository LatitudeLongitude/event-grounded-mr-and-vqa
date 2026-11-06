from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def append_event(
    jsonl_path: Path,
    text_path: Path,
    *,
    run_id: str = "",
    invocation_id: str = "",
    level: str,
    event_type: str,
    payload: dict,
) -> None:
    event = {
        "timestamp": utc_now(),
        **payload,
        "run_id": run_id,
        "invocation_id": invocation_id,
        "level": level,
        "event_type": event_type,
    }
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, sort_keys=True, ensure_ascii=True)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    with text_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{event['timestamp']} run_id={run_id} invocation_id={invocation_id} "
            f"{level} {event_type} {json.dumps(payload, sort_keys=True, ensure_ascii=True)}\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _publish_create_only(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite historical report: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"Refusing to overwrite historical report: {path}") from None
    finally:
        if temporary.exists():
            temporary.unlink()


def write_report(
    json_path: Path,
    markdown_path: Path,
    report: dict,
    *,
    markdown_content: str | None = None,
) -> None:
    if json_path.exists() or markdown_path.exists():
        existing = json_path if json_path.exists() else markdown_path
        raise FileExistsError(f"Refusing to overwrite historical report: {existing}")
    lines = [
        f"# TIPSv2 Run {report.get('run_id', '')}",
        "",
        f"- **invocation_id**: `{report.get('invocation_id', '')}`",
    ]
    for key in ("status", "namespace", "completed", "skipped", "failed", "processed_frames", "elapsed_seconds", "images_per_second"):
        if key in report:
            lines.append(f"- **{key}**: `{report[key]}`")
    failures = report.get("failures") or []
    lines.extend(["", "## Failures", ""])
    lines.extend([f"- `{item}`" for item in failures] or ["- None"])
    json_content = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    rendered_markdown = markdown_content if markdown_content is not None else "\n".join(lines) + "\n"
    created = []
    try:
        _publish_create_only(markdown_path, rendered_markdown)
        created.append(markdown_path)
        _publish_create_only(json_path, json_content)
        created.append(json_path)
        _fsync_directory(json_path.parent)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
