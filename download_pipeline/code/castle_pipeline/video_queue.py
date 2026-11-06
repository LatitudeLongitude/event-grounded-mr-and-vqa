from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .time_utils import utc_now_iso


class VideoRedownloadQueue:
    """SQLite-backed shared queue for bad raw video redownload requests.

    The queue is intentionally a small single-file SQLite database from the
    Python standard library. It gives downloader/processor workers transactional
    coordination without asking them to co-write the same progress JSON.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def open_request(
        self,
        remote_path: str,
        finder: str,
        reason: str,
        local_size: int | None = None,
        expected_size: int | None = None,
        ffprobe_error: str | None = None,
    ) -> None:
        now = utc_now_iso()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO requests (
                    remote_path, status, finder, reason, local_size,
                    expected_size, ffprobe_error, retry_count,
                    first_seen_at, updated_at
                )
                VALUES (?, 'open', ?, ?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(remote_path) DO UPDATE SET
                    status = CASE
                        WHEN requests.status = 'closed' THEN 'open'
                        ELSE requests.status
                    END,
                    finder = excluded.finder,
                    reason = excluded.reason,
                    local_size = excluded.local_size,
                    expected_size = excluded.expected_size,
                    ffprobe_error = excluded.ffprobe_error,
                    retry_count = requests.retry_count + 1,
                    updated_at = excluded.updated_at
                """,
                (remote_path, finder, reason, local_size, expected_size, ffprobe_error, now, now),
            )
            self._append_event(conn, remote_path, "open", finder, reason, local_size, expected_size, ffprobe_error)

    def claim_open(self, actor: str) -> dict[str, Any] | None:
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT remote_path, status, finder, reason, local_size,
                       expected_size, ffprobe_error, retry_count,
                       first_seen_at, updated_at
                FROM requests
                WHERE status = 'open'
                ORDER BY updated_at, remote_path
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                conn.commit()
                return None
            remote_path = str(row["remote_path"])
            now = utc_now_iso()
            conn.execute(
                "UPDATE requests SET status = 'claimed_by_downloader', updated_at = ? WHERE remote_path = ?",
                (now, remote_path),
            )
            self._append_event(
                conn,
                remote_path,
                "claimed_by_downloader",
                actor,
                str(row["reason"] or ""),
                row["local_size"],
                row["expected_size"],
                row["ffprobe_error"],
            )
            conn.commit()
            payload = dict(row)
            payload["status"] = "claimed_by_downloader"
            payload["updated_at"] = now
            return payload

    def mark_redownloaded(self, remote_path: str, actor: str) -> None:
        self._set_status(remote_path, "redownloaded", actor)

    def mark_verified(self, remote_path: str, actor: str) -> None:
        self._set_status(remote_path, "verified", actor)

    def close(self, remote_path: str, actor: str) -> None:
        self._set_status(remote_path, "closed", actor)

    def has_open_requests(self) -> bool:
        with self._connection() as conn:
            row = conn.execute("SELECT 1 FROM requests WHERE status = 'open' LIMIT 1").fetchone()
            return row is not None

    def summary(self) -> dict[str, Any]:
        with self._connection() as conn:
            counts = {
                str(row["status"]): int(row["count"])
                for row in conn.execute("SELECT status, COUNT(*) AS count FROM requests GROUP BY status")
            }
            rows = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT remote_path, status, finder, reason, local_size,
                           expected_size, ffprobe_error, retry_count,
                           first_seen_at, updated_at
                    FROM requests
                    ORDER BY updated_at DESC, remote_path
                    """
                )
            ]
        return {"path": str(self.path), "by_status": counts, "requests": rows}

    def _set_status(self, remote_path: str, status: str, actor: str) -> None:
        now = utc_now_iso()
        with self._connection() as conn:
            row = conn.execute(
                "SELECT reason, local_size, expected_size, ffprobe_error FROM requests WHERE remote_path = ?",
                (remote_path,),
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO requests (
                        remote_path, status, finder, reason, retry_count,
                        first_seen_at, updated_at
                    )
                    VALUES (?, ?, ?, '', 0, ?, ?)
                    """,
                    (remote_path, status, actor, now, now),
                )
                reason = ""
                local_size = None
                expected_size = None
                ffprobe_error = None
            else:
                conn.execute(
                    "UPDATE requests SET status = ?, updated_at = ? WHERE remote_path = ?",
                    (status, now, remote_path),
                )
                reason = str(row["reason"] or "")
                local_size = row["local_size"]
                expected_size = row["expected_size"]
                ffprobe_error = row["ffprobe_error"]
            self._append_event(conn, remote_path, status, actor, reason, local_size, expected_size, ffprobe_error)

    def _init_schema(self) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS requests (
                    remote_path TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    finder TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    local_size INTEGER,
                    expected_size INTEGER,
                    ffprobe_error TEXT,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    first_seen_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    remote_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    reason TEXT,
                    local_size INTEGER,
                    expected_size INTEGER,
                    ffprobe_error TEXT
                )
                """
            )

    def _append_event(
        self,
        conn: sqlite3.Connection,
        remote_path: str,
        status: str,
        actor: str,
        reason: str | None,
        local_size: int | None,
        expected_size: int | None,
        ffprobe_error: str | None,
    ) -> None:
        conn.execute(
            """
            INSERT INTO events (
                timestamp, remote_path, status, actor, reason,
                local_size, expected_size, ffprobe_error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (utc_now_iso(), remote_path, status, actor, reason, local_size, expected_size, ffprobe_error),
        )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()
