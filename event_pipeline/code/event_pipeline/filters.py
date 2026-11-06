from __future__ import annotations

from typing import Optional


def source_filter_values(source_filter: Optional[str]) -> Optional[set[str]]:
    if source_filter is None:
        return None
    values = {item.strip() for item in str(source_filter).split(",") if item.strip()}
    return values or None


def source_matches(source: str, source_filter: Optional[str]) -> bool:
    values = source_filter_values(source_filter)
    return values is None or source in values
