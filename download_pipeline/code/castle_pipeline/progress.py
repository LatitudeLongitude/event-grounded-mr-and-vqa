from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from .constants import PROGRESS_SCHEMA_VERSION
from .time_utils import timestamp_for_filename, utc_now_iso


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def default_progress(kind: str) -> dict[str, Any]:
    return {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "kind": kind,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "repo_id": "CASTLE-Dataset/CASTLE2024",
        "current_remote_path": None,
        "files": {},
        "summary": {
            "total_files": 0,
            "video_done": 0,
            "non_video_done": 0,
            "failed_retryable": 0,
            "failed_need_manual": 0,
        },
        "repairs": [],
    }


# Concurrency contract for ProgressStore:
# - This JSON store is safe only when a single actor owns writes to a given
#   progress file. It rewrites the whole JSON document with os.replace.
# - Multiple workers writing the same file can overwrite each other's states,
#   interleave event ordering, stomp current_remote_path, and race whole-file
#   JSON writes.
# - The video pipeline therefore assigns one writer per progress file:
#   coordinator -> video_pipeline_progress.json, download worker ->
#   video_download_progress.json, processing worker ->
#   video_processing_progress.json, resize coordinator ->
#   video_resize_progress.json.
# - Workers communicate through SQLite/queues or append-only event streams; the
#   processing worker may read download state but must not edit download
#   progress, and the download worker must not edit processing progress.
class ProgressStore:
    def __init__(self, progress_path: Path, kind: str, events_path: Path | None = None) -> None:
        self.progress_path = progress_path
        self.kind = kind
        self.events_path = events_path or progress_path.with_name(progress_path.stem.replace("progress", "events") + ".jsonl")
        self.data: dict[str, Any] = default_progress(kind)
        self._dirty = False
        self._pending_events: list[dict[str, Any]] = []

    def load(self) -> dict[str, Any]:
        self._dirty = False
        self._pending_events = []
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.events_path.exists():
            try:
                self.events_path.touch()
            except OSError:
                pass
        if not self.progress_path.exists():
            self.data = default_progress(self.kind)
            self.save()
            return self.data
        try:
            self.data = json.loads(self.progress_path.read_text(encoding="utf-8"))
            if not isinstance(self.data, dict) or "files" not in self.data:
                raise ValueError("progress file has invalid structure")
        except Exception:
            corrupt = self.progress_path.with_name(
                f"{self.progress_path.name}.corrupt_{timestamp_for_filename()}"
            )
            shutil.move(str(self.progress_path), str(corrupt))
            self.data = default_progress(self.kind)
            self.data["repairs"].append(
                {
                    "timestamp": utc_now_iso(),
                    "message": f"Corrupt progress moved to {corrupt.name}",
                    "backup_path": str(corrupt),
                }
            )
            self.save()
        return self.data

    def backup(self) -> Path | None:
        if not self.progress_path.exists():
            return None
        backup_path = self.progress_path.with_name(f"{self.progress_path.name}.bak_{timestamp_for_filename()}")
        shutil.copy2(self.progress_path, backup_path)
        return backup_path

    def save(self) -> None:
        self.data["updated_at"] = utc_now_iso()
        self._refresh_summary()
        atomic_write_json(self.progress_path, self.data)

    def flush(self) -> None:
        if self._dirty:
            self.save()
            self._dirty = False
        pending_events = list(self._pending_events)
        self._pending_events.clear()
        for event in pending_events:
            self.append_event(event)

    def append_event(self, event: dict[str, Any]) -> None:
        enriched = {
            "event_id": str(uuid.uuid4()),
            "timestamp": utc_now_iso(),
            "repo_id": self.data.get("repo_id", "CASTLE-Dataset/CASTLE2024"),
            "actor": "pipeline",
            **event,
        }
        append_jsonl(self.events_path, enriched)

    def set_current(self, remote_path: str | None) -> None:
        self.data["current_remote_path"] = remote_path
        self.save()

    def get_status(self, remote_path: str) -> str | None:
        entry = self.data.get("files", {}).get(remote_path)
        return entry.get("status") if entry else None

    def set_file_status(
        self,
        remote_path: str,
        file_type: str,
        new_status: str,
        message: str = "",
        raw_path: str | None = None,
        processed_path: str | None = None,
        attempt: int | None = None,
        extra: dict[str, Any] | None = None,
        flush: bool = True,
    ) -> None:
        files = self.data.setdefault("files", {})
        entry = files.setdefault(
            remote_path,
            {
                "remote_path": remote_path,
                "file_type": file_type,
                "status": "pending",
                "created_at": utc_now_iso(),
                "history": [],
            },
        )
        old_status = entry.get("status")
        entry["file_type"] = file_type
        entry["status"] = new_status
        entry["updated_at"] = utc_now_iso()
        if raw_path is not None:
            entry["raw_path"] = raw_path
        if processed_path is not None:
            entry["processed_path"] = processed_path
        if attempt is not None:
            entry["last_attempt"] = attempt
        if extra:
            entry.update(extra)
        entry.setdefault("history", []).append(
            {
                "timestamp": utc_now_iso(),
                "old_status": old_status,
                "new_status": new_status,
                "message": message,
            }
        )
        event = {
            "remote_path": remote_path,
            "file_type": file_type,
            "old_status": old_status,
            "new_status": new_status,
            "message": message,
            "raw_path": raw_path,
            "processed_path": processed_path,
            "attempt": attempt,
        }
        if extra:
            event["extra"] = extra
        if flush:
            if self._dirty or self._pending_events:
                self.flush()
            self.save()
            self.append_event(event)
        else:
            self._dirty = True
            self._pending_events.append(event)

    def note_repair(self, message: str, remote_path: str | None = None, extra: dict[str, Any] | None = None) -> None:
        record = {"timestamp": utc_now_iso(), "message": message, "remote_path": remote_path}
        if extra:
            record.update(extra)
        self.data.setdefault("repairs", []).append(record)
        self.save()
        self.append_event({"remote_path": remote_path, "file_type": "repair", "message": message})

    def _refresh_summary(self) -> None:
        files = self.data.get("files", {})
        statuses = [entry.get("status") for entry in files.values()]
        self.data["summary"] = {
            "total_files": len(files),
            "video_done": sum(1 for entry in files.values() if entry.get("file_type") == "video" and entry.get("status") == "done"),
            "non_video_done": sum(
                1
                for entry in files.values()
                if entry.get("file_type") == "non_video" and entry.get("status") == "processed_verified"
            ),
            "failed_retryable": statuses.count("failed_retryable"),
            "failed_need_manual": statuses.count("failed_need_manual"),
        }
