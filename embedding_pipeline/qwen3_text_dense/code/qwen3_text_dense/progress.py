from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProgressStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path), timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    report_json TEXT
                );
                CREATE TABLE IF NOT EXISTS invocations (
                    invocation_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    code_commit TEXT NOT NULL,
                    report_json TEXT,
                    processed INTEGER NOT NULL DEFAULT 0,
                    skipped INTEGER NOT NULL DEFAULT 0,
                    failed INTEGER NOT NULL DEFAULT 0,
                    elapsed_seconds REAL,
                    chunks_per_second REAL,
                    gpu_metrics_json TEXT,
                    error_type TEXT,
                    error_message TEXT,
                    termination_reason TEXT
                );
                CREATE TABLE IF NOT EXISTS shards (
                    namespace TEXT NOT NULL,
                    day TEXT NOT NULL,
                    view TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    expected_rows INTEGER NOT NULL,
                    actual_rows INTEGER,
                    output_path TEXT NOT NULL,
                    file_size INTEGER,
                    sha256 TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    run_id TEXT,
                    invocation_id TEXT,
                    gpu_ids TEXT,
                    batch_size INTEGER,
                    config_fingerprint TEXT,
                    model_fingerprint TEXT,
                    input_sha256 TEXT,
                    error_type TEXT,
                    error_message TEXT,
                    PRIMARY KEY (namespace, day, view)
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    invocation_id TEXT,
                    created_at TEXT NOT NULL,
                    level TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )

    def register_invocation(self, invocation_id: str, run_id: str, namespace: str, config: dict, *, code_commit: str) -> None:
        started_at = utc_now()
        config_json = json.dumps(config, sort_keys=True)
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO runs(run_id, namespace, started_at, status, config_json)
                VALUES(?,?,?,?,?)
                ON CONFLICT(run_id) DO NOTHING
                """,
                (run_id, namespace, started_at, "active", config_json),
            )
            connection.execute(
                """
                INSERT INTO invocations(invocation_id, run_id, namespace, started_at, status, config_json, code_commit)
                VALUES(?,?,?,?,?,?,?)
                """,
                (invocation_id, run_id, namespace, started_at, "running", config_json, code_commit),
            )

    def finish_invocation(
        self,
        invocation_id: str,
        *,
        status: str,
        report: dict,
        processed: int,
        skipped: int,
        failed: int,
        elapsed_seconds: float,
        chunks_per_second: float,
        gpu_metrics: dict,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        termination_reason: Optional[str] = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE invocations SET finished_at=?, status=?, report_json=?, processed=?, skipped=?, failed=?,
                    elapsed_seconds=?, chunks_per_second=?, gpu_metrics_json=?, error_type=?, error_message=?,
                    termination_reason=?
                WHERE invocation_id=? AND finished_at IS NULL
                """,
                (
                    utc_now(),
                    status,
                    json.dumps(report, sort_keys=True),
                    processed,
                    skipped,
                    failed,
                    elapsed_seconds,
                    chunks_per_second,
                    json.dumps(gpu_metrics, sort_keys=True),
                    error_type[:200] if error_type else None,
                    error_message[:2000] if error_message else None,
                    termination_reason[:500] if termination_reason else None,
                    invocation_id,
                ),
            )

    def ensure_shard(self, namespace: str, day: str, view: str, *, expected_rows: int, output_path: Path | str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO shards(namespace, day, view, expected_rows, output_path)
                VALUES(?,?,?,?,?)
                ON CONFLICT(namespace, day, view) DO UPDATE SET
                    expected_rows=excluded.expected_rows,
                    output_path=excluded.output_path
                """,
                (namespace, day, view, expected_rows, str(output_path)),
            )

    def start_shard(
        self,
        namespace: str,
        day: str,
        view: str,
        *,
        run_id: str,
        invocation_id: str,
        attempt: int,
        gpu_ids: str,
        batch_size: int,
        config_fingerprint: str,
        model_fingerprint: str,
        input_sha256: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE shards SET status='running', attempt_count=?, started_at=?, finished_at=NULL,
                    run_id=?, invocation_id=?, gpu_ids=?, batch_size=?, config_fingerprint=?,
                    model_fingerprint=?, input_sha256=?, error_type=NULL, error_message=NULL
                WHERE namespace=? AND day=? AND view=?
                """,
                (
                    attempt,
                    utc_now(),
                    run_id,
                    invocation_id,
                    gpu_ids,
                    batch_size,
                    config_fingerprint,
                    model_fingerprint,
                    input_sha256,
                    namespace,
                    day,
                    view,
                ),
            )

    def complete_shard(
        self,
        namespace: str,
        day: str,
        view: str,
        *,
        actual_rows: int,
        file_size: int,
        sha256: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE shards SET status='completed', actual_rows=?, file_size=?, sha256=?, finished_at=?,
                    error_type=NULL, error_message=NULL
                WHERE namespace=? AND day=? AND view=?
                """,
                (actual_rows, file_size, sha256, utc_now(), namespace, day, view),
            )

    def fail_shard(self, namespace: str, day: str, view: str, *, error_type: str, error_message: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE shards SET status='failed', finished_at=?, error_type=?, error_message=?
                WHERE namespace=? AND day=? AND view=?
                """,
                (utc_now(), error_type[:200], error_message[:2000], namespace, day, view),
            )

    def add_event(self, run_id: str, level: str, event_type: str, payload: dict, invocation_id: Optional[str] = None) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO events(run_id, invocation_id, created_at, level, event_type, payload_json)
                VALUES(?,?,?,?,?,?)
                """,
                (run_id, invocation_id, utc_now(), level, event_type, json.dumps(payload, sort_keys=True)),
            )

    def get_shard(self, namespace: str, day: str, view: str) -> Optional[Dict[str, object]]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM shards WHERE namespace=? AND day=? AND view=?",
                (namespace, day, view),
            ).fetchone()
            return dict(row) if row is not None else None

    def list_shards(self, namespace: Optional[str] = None) -> List[Dict[str, object]]:
        with self.connect() as connection:
            if namespace is None:
                rows = connection.execute("SELECT * FROM shards ORDER BY day, view").fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM shards WHERE namespace=? ORDER BY day, view",
                    (namespace,),
                ).fetchall()
            return [dict(row) for row in rows]

    def list_invocations(self) -> List[Dict[str, object]]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM invocations ORDER BY started_at, invocation_id").fetchall()
            return [dict(row) for row in rows]
