from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml


def load_view_meta(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def source_roles(path: Path) -> dict[str, dict]:
    meta = load_view_meta(path)
    result: dict[str, dict] = {}
    for role, payload in (meta.get("source_roles") or {}).items():
        for source, source_payload in (payload.get("sources") or {}).items():
            result[source] = {"source_role": role, **(source_payload or {})}
    return result


def location_hint_for_source(path: Path, source: str) -> Optional[str]:
    return (source_roles(path).get(source) or {}).get("location_hint")


def l3_time_slack(path: Path) -> int:
    meta = load_view_meta(path)
    return int((((meta.get("fusion_defaults") or {}).get("cross_source_time_slack_sec") or {}).get("l3_global")) or 60)
