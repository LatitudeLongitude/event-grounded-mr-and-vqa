from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sha256_text(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        first = stripped.split()[0]
        if len(first) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in first):
            return first.lower()
    return None


def read_sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return parse_sha256_text(path.read_text(encoding="utf-8", errors="replace"))
