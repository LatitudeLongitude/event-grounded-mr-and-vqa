from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TextIO


class OutputWriter:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.manifest_root = output_root / "_manifests"
        self._view_handles: dict[str, TextIO] = {}
        self._manifest: TextIO | None = None
        self._failed: TextIO | None = None
        self._invalid: TextIO | None = None
        self._log: TextIO | None = None

    def __enter__(self) -> "OutputWriter":
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.manifest_root.mkdir(parents=True, exist_ok=True)
        self._remove_legacy_manifest_files()
        self._manifest = self._open_manifest("manifest.jsonl")
        self._failed = self._open_manifest("failed_files.jsonl")
        self._invalid = self._open_manifest("invalid_chunks.jsonl")
        self._log = self._open_manifest("cleaning.log")
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        for handle in self._view_handles.values():
            handle.close()
        for handle in [self._manifest, self._failed, self._invalid, self._log]:
            if handle is not None:
                handle.close()

    def write_chunk(self, view: str, record: dict[str, Any]) -> None:
        handle = self._view_handles.get(view)
        if handle is None:
            handle = self._open(f"{view}.jsonl")
            self._view_handles[view] = handle
        _write_jsonl(handle, record)

    def write_manifest(self, record: dict[str, Any]) -> None:
        _write_jsonl(_require(self._manifest), record)

    def write_failed(self, record: dict[str, Any]) -> None:
        _write_jsonl(_require(self._failed), record)

    def write_invalid(self, record: dict[str, Any]) -> None:
        _write_jsonl(_require(self._invalid), record)

    def log(self, message: str) -> None:
        _require(self._log).write(message.rstrip() + "\n")

    def write_summary(self, summary: dict[str, Any]) -> None:
        path = self.manifest_root / "run_summary.json"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    def _open(self, name: str) -> TextIO:
        return (self.output_root / name).open("w", encoding="utf-8", newline="\n")

    def _open_manifest(self, name: str) -> TextIO:
        return (self.manifest_root / name).open("w", encoding="utf-8", newline="\n")

    def _remove_legacy_manifest_files(self) -> None:
        for name in [
            "manifest.jsonl",
            "failed_files.jsonl",
            "invalid_chunks.jsonl",
            "run_summary.json",
            "cleaning.log",
        ]:
            path = self.output_root / name
            if path.exists() and path.is_file():
                path.unlink()


def _write_jsonl(handle: TextIO, record: dict[str, Any]) -> None:
    json.dump(record, handle, ensure_ascii=False, separators=(",", ":"))
    handle.write("\n")


def _require(handle: TextIO | None) -> TextIO:
    if handle is None:
        raise RuntimeError("OutputWriter is not open.")
    return handle
