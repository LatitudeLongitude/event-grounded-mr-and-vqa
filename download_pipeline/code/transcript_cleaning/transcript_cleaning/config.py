from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_MIN_DURATION_SEC = 1.0
DEFAULT_MAX_DURATION_SEC = 120.0


@dataclass(frozen=True)
class CleanConfig:
    input_root: Path
    output_root: Path
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC
    max_duration_sec: float = DEFAULT_MAX_DURATION_SEC


def validate_roots(input_root: Path, output_root: Path) -> tuple[Path, Path]:
    input_root = input_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()

    if not input_root.exists():
        raise ValueError(f"Input root does not exist: {input_root}")
    if not input_root.is_dir():
        raise ValueError(f"Input root is not a directory: {input_root}")

    if output_root.name != "cleaned_transcripts":
        raise ValueError(
            "Output root must be named 'cleaned_transcripts' to avoid writing "
            "to other processed_data locations."
        )

    if _is_relative_to(output_root, input_root):
        raise ValueError("Output root must not be inside the read-only input root.")

    processed_parent = _nearest_parent_named(output_root, "processed_data")
    if processed_parent is not None:
        expected = processed_parent / "cleaned_transcripts"
        if output_root != expected:
            raise ValueError(
                "When writing under processed_data, output root must be exactly "
                f"{expected}"
            )

    return input_root, output_root


def _nearest_parent_named(path: Path, name: str) -> Path | None:
    for parent in [path, *path.parents]:
        if parent.name == name:
            return parent
    return None


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False
