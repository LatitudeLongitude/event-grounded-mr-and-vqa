from __future__ import annotations

import json
import os
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Iterator, Mapping, TextIO

from .encoder import sha256_file


HASHED_MODEL_FILES = (
    "config.json",
    "modeling_tips.py",
    "image_encoder.py",
    "text_encoder.py",
    "tokenizer.model",
)


def model_fingerprint(model_root: Path | str, *, hash_weights: bool = False) -> dict:
    root = Path(model_root).resolve()
    files: Dict[str, dict] = {}
    for name in HASHED_MODEL_FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Required model file is missing: {path}")
        stat = path.stat()
        files[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256_file(path)}
    weights = root / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"Required model weights are missing: {weights}")
    stat = weights.stat()
    files[weights.name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if hash_weights:
        files[weights.name]["sha256"] = sha256_file(weights)
    return {"model_root": str(root), "files": files}


class ProtectedTreeChangedError(RuntimeError):
    def __init__(self, diff: dict):
        self.diff = diff
        counts = diff["counts"]
        super().__init__(
            "Protected trees changed: "
            f"added={counts['added']}, removed={counts['removed']}, changed={counts['changed']}; "
            f"samples={json.dumps({'added': diff['added'], 'removed': diff['removed'], 'changed': diff['changed']}, sort_keys=True)}"
        )


def _file_type(mode: int) -> str:
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISLNK(mode):
        return "symlink"
    return "other"


def _record(root_name: str, relative_path: str, stat_result: os.stat_result) -> dict:
    return {
        "root": root_name,
        "path": relative_path,
        "type": _file_type(stat_result.st_mode),
        "size": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
    }


def _iter_directory(root_name: str, directory: Path, relative_directory: Path) -> Iterator[dict]:
    with os.scandir(directory) as entries:
        ordered = sorted(entries, key=lambda entry: entry.name)
    for entry in ordered:
        relative_path = relative_directory / entry.name
        stat_result = entry.stat(follow_symlinks=False)
        relative_text = relative_path.as_posix()
        yield _record(root_name, relative_text, stat_result)
        if stat.S_ISDIR(stat_result.st_mode):
            yield from _iter_directory(root_name, Path(entry.path), relative_path)


def iter_tree_snapshot_records(roots: Mapping[str, Path | str]) -> Iterator[dict]:
    for root_name in sorted(roots):
        root = Path(roots[root_name]).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Protected snapshot root is missing: {root}")
        yield _record(root_name, ".", root.stat())
        yield from _iter_directory(root_name, root, Path())


def write_tree_snapshot(roots: Mapping[str, Path | str], destination: Path | str) -> dict:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in iter_tree_snapshot_records(roots):
                handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(path.resolve()),
        "record_count": count,
        "roots": {name: str(Path(value).resolve()) for name, value in sorted(roots.items())},
    }


def _record_key(record: dict) -> tuple[str, tuple[str, ...]]:
    path_parts = () if record["path"] == "." else PurePosixPath(record["path"]).parts
    return record["root"], path_parts


def _read_record(
    handle: TextIO,
    previous_key: tuple[str, tuple[str, ...]] | None,
) -> tuple[dict | None, tuple[str, tuple[str, ...]] | None]:
    line = handle.readline()
    if not line:
        return None, previous_key
    record = json.loads(line)
    required = {"root", "path", "type", "size", "mtime_ns"}
    if set(record) != required:
        raise ValueError(f"Invalid snapshot record fields: {sorted(record)}")
    key = _record_key(record)
    if previous_key is not None and key <= previous_key:
        raise ValueError(f"Snapshot records are not strictly ordered at {key}")
    return record, key


def compare_tree_snapshots(before_path: Path | str, after_path: Path | str, *, sample_limit: int = 20) -> dict:
    counts = {"added": 0, "removed": 0, "changed": 0}
    samples = {"added": [], "removed": [], "changed": []}
    with Path(before_path).open(encoding="utf-8") as before_handle, Path(after_path).open(encoding="utf-8") as after_handle:
        before_previous = after_previous = None
        before, before_previous = _read_record(before_handle, before_previous)
        after, after_previous = _read_record(after_handle, after_previous)
        while before is not None or after is not None:
            before_key = _record_key(before) if before is not None else None
            after_key = _record_key(after) if after is not None else None
            if before is None or (after is not None and after_key < before_key):
                counts["added"] += 1
                if len(samples["added"]) < sample_limit:
                    samples["added"].append(after)
                after, after_previous = _read_record(after_handle, after_previous)
            elif after is None or before_key < after_key:
                counts["removed"] += 1
                if len(samples["removed"]) < sample_limit:
                    samples["removed"].append(before)
                before, before_previous = _read_record(before_handle, before_previous)
            else:
                if before != after:
                    counts["changed"] += 1
                    if len(samples["changed"]) < sample_limit:
                        samples["changed"].append(
                            {"root": before["root"], "path": before["path"], "before": before, "after": after}
                        )
                before, before_previous = _read_record(before_handle, before_previous)
                after, after_previous = _read_record(after_handle, after_previous)
    return {
        "identical": not any(counts.values()),
        "counts": counts,
        **samples,
        "before_path": str(Path(before_path).resolve()),
        "after_path": str(Path(after_path).resolve()),
    }


def write_snapshot_diff_report(diff: dict, json_path: Path | str, markdown_path: Path | str) -> None:
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(diff, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    counts = diff["counts"]
    lines = [
        "# Protected Tree Snapshot Diff",
        "",
        f"- Identical: `{diff['identical']}`",
        f"- Added: `{counts['added']}`",
        f"- Removed: `{counts['removed']}`",
        f"- Changed: `{counts['changed']}`",
        "",
        "## Difference Samples",
        "",
        "```json",
        json.dumps({name: diff[name] for name in ("added", "removed", "changed")}, indent=2, sort_keys=True),
        "```",
        "",
    ]
    markdown_output.write_text("\n".join(lines), encoding="utf-8")


@dataclass(frozen=True)
class ReadOnlySnapshotAudit:
    roots: Mapping[str, Path]
    before_path: Path
    after_path: Path
    diff_json_path: Path
    diff_markdown_path: Path

    @classmethod
    def create(cls, output_root: Path | str, run_id: str, roots: Mapping[str, Path | str]) -> "ReadOnlySnapshotAudit":
        report_root = Path(output_root).resolve() / "_reports" / "smoke-readonly" / run_id
        resolved_roots = {name: Path(value).resolve() for name, value in roots.items()}
        return cls(
            roots=resolved_roots,
            before_path=report_root / "before.jsonl",
            after_path=report_root / "after.jsonl",
            diff_json_path=report_root / "diff.json",
            diff_markdown_path=report_root / "diff.md",
        )

    def capture_before(self) -> dict:
        return write_tree_snapshot(self.roots, self.before_path)

    def capture_after_and_compare(self) -> dict:
        write_tree_snapshot(self.roots, self.after_path)
        diff = compare_tree_snapshots(self.before_path, self.after_path)
        write_snapshot_diff_report(diff, self.diff_json_path, self.diff_markdown_path)
        if not diff["identical"]:
            raise ProtectedTreeChangedError(diff)
        return diff
