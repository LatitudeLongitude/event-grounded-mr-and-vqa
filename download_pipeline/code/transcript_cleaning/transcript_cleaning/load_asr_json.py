from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ASRDocument:
    chunks: list[Any]
    raw: dict[str, Any]


class ASRLoadError(Exception):
    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message


def load_asr_json(path: Path) -> ASRDocument:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ASRLoadError("json_parse_error", str(exc)) from exc
    except OSError as exc:
        raise ASRLoadError("file_read_error", str(exc)) from exc

    if not isinstance(data, dict):
        raise ASRLoadError("invalid_json_root", "Top-level JSON value is not an object.")

    if "chunks" not in data:
        raise ASRLoadError("missing_chunks", "Missing required 'chunks' field.")

    chunks = data["chunks"]
    if not isinstance(chunks, list):
        raise ASRLoadError("missing_chunks", "'chunks' field is not a list.")

    return ASRDocument(chunks=chunks, raw=data)
