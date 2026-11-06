from __future__ import annotations

from pathlib import Path
from typing import Union


PROHIBITED_PARTS = {
    "retrieval_test",
    "exploration",
}


def resolve_path(value: Union[str, Path], *, base: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def reject_prohibited_path(path: Path, *, role: str = "path") -> None:
    parts = {part.lower() for part in path.resolve().parts}
    if parts & PROHIBITED_PARTS:
        raise ValueError(f"{role} must not refer to retrieval_test or exploration: {path}")
    normalized = path.resolve().as_posix().lower()
    if normalized.endswith("event_pipeline/future/task_queries.yaml"):
        raise ValueError(f"{role} must not use official task queries as segmentation input: {path}")


def ensure_under(path: Path, root: Path, *, role: str) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if not is_relative_to(resolved, root_resolved):
        raise ValueError(f"{role} must stay under {root_resolved}: {resolved}")
    return resolved


def to_ref(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()
