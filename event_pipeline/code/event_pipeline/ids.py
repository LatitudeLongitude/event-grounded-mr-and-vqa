from __future__ import annotations

import hashlib
import json
from typing import Optional, Union

from .time_utils import day_label, normalize_hour


def stable_hash(value: object, length: int = 16) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def frame_atom_id(day: Union[str, int], source: str, hour: Union[str, int], frame_index: int) -> str:
    return f"frame::{day_label(day)}/{source}/{normalize_hour(hour)}/{int(frame_index)}"


def text_atom_id(embedding_chunk_id: str) -> str:
    return f"text::{embedding_chunk_id}"


def event_id(level: str, day: Union[str, int], source: Optional[str], start: float, end: float, seed: object = None) -> str:
    prefix = f"{level.lower()}::{day_label(day)}/"
    source_part = "global" if source is None else source
    return prefix + f"{source_part}/{int(round(start))}-{int(round(end))}-{stable_hash(seed if seed is not None else [level, day, source, start, end], 10)}"
