from __future__ import annotations

import html
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .io_utils import atomic_write_json, atomic_write_text, now_utc, sha256_text


TASK_LINE_RE = re.compile(r"(MM26-(?:EIS|OIS|QA)\d{2}):\s*([^<\n]+)")


def fetch_url(url: str, *, timeout_sec: int = 30, retries: int = 6, retry_sleep_sec: float = 2.0) -> tuple[int, str]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 castle-mm26-solution-preflight/1.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Connection": "close",
        },
    )
    last_error: BaseException | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout_sec) as response:
                status = int(getattr(response, "status", 200))
                charset = response.headers.get_content_charset() or "utf-8"
                return status, response.read().decode(charset, errors="replace")
        except (TimeoutError, ConnectionError, OSError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt >= max(1, retries):
                raise
            time.sleep(retry_sleep_sec * attempt)
    raise RuntimeError(f"unreachable fetch retry state: {last_error}")


def extract_official_queries(page_html: str) -> dict[str, str]:
    queries: dict[str, str] = {}
    for task_id, text in TASK_LINE_RE.findall(page_html):
        clean = html.unescape(text).strip()
        clean = re.sub(r"\s+", " ", clean)
        queries[task_id] = clean
    return dict(sorted(queries.items()))


def save_official_snapshot(url: str, output_dir: Path, *, timeout_sec: int = 30) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    status, body = fetch_url(url, timeout_sec=timeout_sec)
    queries = extract_official_queries(body)
    snapshot_path = output_dir / "official_page_snapshot.html"
    extracted_path = output_dir / "official_queries_extracted.json"
    manifest_path = output_dir / "official_sources_manifest.json"
    atomic_write_text(snapshot_path, body)
    atomic_write_json(extracted_path, queries)
    manifest = {
        "valid": status == 200 and len(queries) == 36,
        "challenge_url": url,
        "http_status": status,
        "fetch_time_utc": now_utc(),
        "page_sha256": sha256_text(body),
        "extracted_task_count": len(queries),
        "extracted_task_text_sha256": sha256_text("\n".join(f"{key}: {value}" for key, value in queries.items())),
        "snapshot_path": snapshot_path.as_posix(),
        "extracted_queries_path": extracted_path.as_posix(),
        "extraction_script_version": "castle_mm26_official_v1",
    }
    atomic_write_json(manifest_path, manifest)
    return {"html": body, "queries": queries, "manifest": manifest}
