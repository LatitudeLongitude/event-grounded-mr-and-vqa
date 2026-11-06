from __future__ import annotations

from pathlib import Path

from .parquet_io import atomic_write_bytes


def write_run_summary(run_root: Path, validation_report: dict) -> Path:
    path = run_root / "reports" / "run_summary.md"
    lines = [
        "# Event Pipeline Run Summary",
        "",
        f"- valid: {validation_report.get('valid')}",
        f"- event_count: {validation_report.get('event_count')}",
        f"- error_count: {validation_report.get('error_count')}",
        f"- warning_count: {validation_report.get('warning_count')}",
        "",
        "## Errors",
        "",
    ]
    errors = validation_report.get("errors") or []
    lines.extend([f"- {item}" for item in errors] or ["- none"])
    lines.extend(["", "## Warnings", ""])
    warnings = validation_report.get("warnings") or []
    lines.extend([f"- {item}" for item in warnings[:50]] or ["- none"])
    atomic_write_bytes(path, ("\n".join(lines) + "\n").encode("utf-8"))
    return path
