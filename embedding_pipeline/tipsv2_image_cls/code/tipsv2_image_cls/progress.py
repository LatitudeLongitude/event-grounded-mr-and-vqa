from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional


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
                CREATE TABLE IF NOT EXISTS shards (
                    namespace TEXT NOT NULL,
                    day INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    hour TEXT NOT NULL,
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
                    gpu_ids TEXT,
                    batch_size INTEGER,
                    worker_config TEXT,
                    error_type TEXT,
                    error_message TEXT,
                    PRIMARY KEY (namespace, day, source, hour)
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
                    images_per_second REAL,
                    gpu_metrics_json TEXT,
                    error_type TEXT,
                    error_message TEXT,
                    termination_reason TEXT
                );
                CREATE INDEX IF NOT EXISTS invocations_run_id_idx
                    ON invocations(run_id, started_at, invocation_id);
                CREATE TABLE IF NOT EXISTS invocation_shards (
                    invocation_id TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    day INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    hour TEXT NOT NULL,
                    action TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt INTEGER,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    expected_rows INTEGER,
                    actual_rows INTEGER,
                    output_path TEXT,
                    error_type TEXT,
                    error_message TEXT,
                    PRIMARY KEY (invocation_id, namespace, day, source, hour),
                    FOREIGN KEY (invocation_id) REFERENCES invocations(invocation_id)
                );
                """
            )
            event_columns = {row["name"] for row in connection.execute("PRAGMA table_info(events)")}
            if "invocation_id" not in event_columns:
                connection.execute("ALTER TABLE events ADD COLUMN invocation_id TEXT")

    def register_invocation(
        self,
        invocation_id: str,
        run_id: str,
        namespace: str,
        config: dict,
        *,
        code_commit: str,
    ) -> None:
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
                INSERT INTO invocations(
                    invocation_id, run_id, namespace, started_at, status, config_json, code_commit
                ) VALUES(?,?,?,?,?,?,?)
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
        images_per_second: float,
        gpu_metrics: dict,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        termination_reason: Optional[str] = None,
    ) -> None:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE invocations SET finished_at=?, status=?, report_json=?, processed=?, skipped=?, failed=?,
                    elapsed_seconds=?, images_per_second=?, gpu_metrics_json=?, error_type=?, error_message=?,
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
                    images_per_second,
                    json.dumps(gpu_metrics, sort_keys=True),
                    error_type[:200] if error_type else None,
                    error_message[:2000] if error_message else None,
                    termination_reason[:500] if termination_reason else None,
                    invocation_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"Invocation is missing or already finished: {invocation_id}")

    def list_invocations(self, run_id: Optional[str] = None) -> List[Dict[str, object]]:
        with self.connect() as connection:
            if run_id is None:
                rows = connection.execute(
                    "SELECT * FROM invocations ORDER BY started_at, invocation_id"
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM invocations WHERE run_id=? ORDER BY started_at, invocation_id",
                    (run_id,),
                ).fetchall()
            return [dict(row) for row in rows]

    def record_invocation_shard(
        self,
        invocation_id: str,
        namespace: str,
        day: int,
        source: str,
        hour: str,
        *,
        action: str,
        status: str,
        attempt: Optional[int] = None,
        expected_rows: Optional[int] = None,
        actual_rows: Optional[int] = None,
        output_path: Optional[Path | str] = None,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        with self.connect() as connection:
            self._record_invocation_shard(
                connection,
                invocation_id,
                namespace,
                day,
                source,
                hour,
                action=action,
                status=status,
                attempt=attempt,
                expected_rows=expected_rows,
                actual_rows=actual_rows,
                output_path=output_path,
                error_type=error_type,
                error_message=error_message,
            )

    @staticmethod
    def _record_invocation_shard(
        connection: sqlite3.Connection,
        invocation_id: str,
        namespace: str,
        day: int,
        source: str,
        hour: str,
        *,
        action: str,
        status: str,
        attempt: Optional[int] = None,
        expected_rows: Optional[int] = None,
        actual_rows: Optional[int] = None,
        output_path: Optional[Path | str] = None,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        now = utc_now()
        finished_at = now if status in {"completed", "failed", "skipped"} else None
        connection.execute(
            """
            INSERT INTO invocation_shards(
                invocation_id, namespace, day, source, hour, action, status, attempt, started_at,
                finished_at, expected_rows, actual_rows, output_path, error_type, error_message
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(invocation_id, namespace, day, source, hour) DO UPDATE SET
                action=excluded.action,
                status=excluded.status,
                attempt=COALESCE(excluded.attempt, invocation_shards.attempt),
                finished_at=excluded.finished_at,
                expected_rows=COALESCE(excluded.expected_rows, invocation_shards.expected_rows),
                actual_rows=COALESCE(excluded.actual_rows, invocation_shards.actual_rows),
                output_path=COALESCE(excluded.output_path, invocation_shards.output_path),
                error_type=excluded.error_type,
                error_message=excluded.error_message
            """,
            (
                invocation_id,
                namespace,
                day,
                source,
                hour,
                action,
                status,
                attempt,
                now,
                finished_at,
                expected_rows,
                actual_rows,
                str(output_path) if output_path is not None else None,
                error_type[:200] if error_type else None,
                error_message[:2000] if error_message else None,
            ),
        )

    def start_shard_for_invocation(
        self,
        invocation_id: str,
        namespace: str,
        day: int,
        source: str,
        hour: str,
        *,
        run_id: str,
        attempt: int,
        expected_rows: int,
        output_path: Path | str,
        gpu_ids: str = "",
        batch_size: int = 0,
        worker_config: str = "",
    ) -> None:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE shards SET status='running', attempt_count=?, started_at=?, finished_at=NULL,
                    run_id=?, gpu_ids=?, batch_size=?, worker_config=?, error_type=NULL, error_message=NULL
                WHERE namespace=? AND day=? AND source=? AND hour=?
                """,
                (
                    attempt,
                    utc_now(),
                    run_id,
                    gpu_ids,
                    batch_size,
                    worker_config,
                    namespace,
                    day,
                    source,
                    hour,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"Shard is missing: {(namespace, day, source, hour)}")
            self._record_invocation_shard(
                connection,
                invocation_id,
                namespace,
                day,
                source,
                hour,
                action="computed",
                status="running",
                attempt=attempt,
                expected_rows=expected_rows,
                output_path=output_path,
            )

    def complete_shard_for_invocation(
        self,
        invocation_id: str,
        namespace: str,
        day: int,
        source: str,
        hour: str,
        *,
        attempt: int,
        expected_rows: int,
        actual_rows: int,
        output_path: Path | str,
        file_size: int,
        sha256: str,
    ) -> None:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE shards SET status='completed', actual_rows=?, file_size=?, sha256=?, finished_at=?,
                    error_type=NULL, error_message=NULL
                WHERE namespace=? AND day=? AND source=? AND hour=?
                """,
                (actual_rows, file_size, sha256, utc_now(), namespace, day, source, hour),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"Shard is missing: {(namespace, day, source, hour)}")
            self._record_invocation_shard(
                connection,
                invocation_id,
                namespace,
                day,
                source,
                hour,
                action="computed",
                status="completed",
                attempt=attempt,
                expected_rows=expected_rows,
                actual_rows=actual_rows,
                output_path=output_path,
            )

    def fail_shard_for_invocation(
        self,
        invocation_id: str,
        namespace: str,
        day: int,
        source: str,
        hour: str,
        *,
        attempt: Optional[int],
        expected_rows: Optional[int],
        output_path: Optional[Path | str],
        error_type: str,
        error_message: str,
    ) -> None:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE shards SET status='failed', finished_at=?, error_type=?, error_message=?
                WHERE namespace=? AND day=? AND source=? AND hour=?
                """,
                (
                    utc_now(),
                    error_type[:200],
                    error_message[:2000],
                    namespace,
                    day,
                    source,
                    hour,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"Shard is missing: {(namespace, day, source, hour)}")
            self._record_invocation_shard(
                connection,
                invocation_id,
                namespace,
                day,
                source,
                hour,
                action="computed",
                status="failed",
                attempt=attempt,
                expected_rows=expected_rows,
                output_path=output_path,
                error_type=error_type,
                error_message=error_message,
            )

    def list_invocation_shards(self, invocation_id: str) -> List[Dict[str, object]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM invocation_shards WHERE invocation_id=?
                ORDER BY day, source, CAST(hour AS INTEGER)
                """,
                (invocation_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_invocation(self, invocation_id: str) -> Optional[Dict[str, object]]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM invocations WHERE invocation_id=?", (invocation_id,)
            ).fetchone()
            return dict(row) if row is not None else None

    def fail_open_invocation_shards(
        self,
        invocation_id: str,
        *,
        error_type: str,
        error_message: str,
    ) -> int:
        now = utc_now()
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT namespace, day, source, hour FROM invocation_shards
                WHERE invocation_id=? AND status='running'
                """,
                (invocation_id,),
            ).fetchall()
            for row in rows:
                connection.execute(
                    """
                    UPDATE shards SET status='failed', finished_at=?, error_type=?, error_message=?
                    WHERE namespace=? AND day=? AND source=? AND hour=? AND status!='completed'
                    """,
                    (
                        now,
                        error_type[:200],
                        error_message[:2000],
                        row["namespace"],
                        row["day"],
                        row["source"],
                        row["hour"],
                    ),
                )
            cursor = connection.execute(
                """
                UPDATE invocation_shards SET status='failed', finished_at=?, error_type=?, error_message=?
                WHERE invocation_id=? AND status='running'
                """,
                (now, error_type[:200], error_message[:2000], invocation_id),
            )
            return cursor.rowcount

    def add_event(
        self,
        run_id: str,
        level: str,
        event_type: str,
        payload: dict,
        invocation_id: Optional[str] = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO events(run_id, invocation_id, created_at, level, event_type, payload_json)
                VALUES(?,?,?,?,?,?)
                """,
                (run_id, invocation_id, utc_now(), level, event_type, json.dumps(payload, sort_keys=True)),
            )

    def ensure_shard(
        self,
        namespace: str,
        day: int,
        source: str,
        hour: str,
        *,
        expected_rows: int,
        output_path: Path | str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO shards(namespace, day, source, hour, expected_rows, output_path)
                VALUES(?,?,?,?,?,?)
                ON CONFLICT(namespace, day, source, hour) DO UPDATE SET
                    expected_rows=excluded.expected_rows,
                    output_path=excluded.output_path
                """,
                (namespace, day, source, hour, expected_rows, str(output_path)),
            )

    def mark_running(
        self,
        namespace: str,
        day: int,
        source: str,
        hour: str,
        *,
        run_id: str,
        attempt: int,
        gpu_ids: str = "",
        batch_size: int = 0,
        worker_config: str = "",
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE shards SET status='running', attempt_count=?, started_at=?, finished_at=NULL,
                    run_id=?, gpu_ids=?, batch_size=?, worker_config=?, error_type=NULL, error_message=NULL
                WHERE namespace=? AND day=? AND source=? AND hour=?
                """,
                (attempt, utc_now(), run_id, gpu_ids, batch_size, worker_config, namespace, day, source, hour),
            )

    def mark_completed(
        self,
        namespace: str,
        day: int,
        source: str,
        hour: str,
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
                WHERE namespace=? AND day=? AND source=? AND hour=?
                """,
                (actual_rows, file_size, sha256, utc_now(), namespace, day, source, hour),
            )

    def mark_failed(
        self,
        namespace: str,
        day: int,
        source: str,
        hour: str,
        *,
        error_type: str,
        error_message: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE shards SET status='failed', finished_at=?, error_type=?, error_message=?
                WHERE namespace=? AND day=? AND source=? AND hour=?
                """,
                (utc_now(), error_type[:200], error_message[:2000], namespace, day, source, hour),
            )

    def recover_interrupted(self) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE shards SET status='pending', started_at=NULL, run_id=NULL WHERE status='running'"
            )
            return cursor.rowcount

    def reconcile_completed(self, validator: Callable[[Path], bool]) -> int:
        invalid = 0
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM shards WHERE status='completed'").fetchall()
            for row in rows:
                path = Path(row["output_path"])
                try:
                    valid = path.is_file() and validator(path)
                except Exception:
                    valid = False
                if not valid:
                    invalid += 1
                    connection.execute(
                        """
                        UPDATE shards SET status='failed', error_type='InvalidCompletedShard',
                            error_message='Completed shard is missing or invalid', finished_at=?
                        WHERE namespace=? AND day=? AND source=? AND hour=?
                        """,
                        (utc_now(), row["namespace"], row["day"], row["source"], row["hour"]),
                    )
        return invalid

    def get_shard(self, namespace: str, day: int, source: str, hour: str) -> Optional[Dict[str, object]]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM shards WHERE namespace=? AND day=? AND source=? AND hour=?",
                (namespace, day, source, hour),
            ).fetchone()
            return dict(row) if row is not None else None

    def list_shards(self, namespace: Optional[str] = None) -> List[Dict[str, object]]:
        with self.connect() as connection:
            if namespace is None:
                rows = connection.execute("SELECT * FROM shards ORDER BY day, source, CAST(hour AS INTEGER)").fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM shards WHERE namespace=? ORDER BY day, source, CAST(hour AS INTEGER)",
                    (namespace,),
                ).fetchall()
            return [dict(row) for row in rows]
