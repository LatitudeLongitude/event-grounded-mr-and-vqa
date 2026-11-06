from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True, order=True)
class TranscriptFile:
    day: str
    view: str
    hour: str
    source_file: Path


def discover_transcript_files(input_root: Path) -> Iterator[TranscriptFile]:
    """Yield day/view/transcript/*.json files without touching sha256 files."""
    day_dirs = sorted(
        (path for path in input_root.iterdir() if path.is_dir() and path.name.startswith("day")),
        key=lambda path: path.name,
    )
    for day_dir in day_dirs:
        view_dirs = sorted(
            (path for path in day_dir.iterdir() if path.is_dir()),
            key=lambda path: path.name.casefold(),
        )
        for view_dir in view_dirs:
            transcript_dir = view_dir / "transcript"
            if not transcript_dir.is_dir():
                continue
            json_files = sorted(
                (
                    path
                    for path in transcript_dir.iterdir()
                    if path.is_file()
                    and path.suffix.lower() == ".json"
                    and not path.name.lower().endswith(".json.sha256")
                ),
                key=lambda path: path.name,
            )
            for source_file in json_files:
                yield TranscriptFile(
                    day=day_dir.name,
                    view=view_dir.name,
                    hour=source_file.stem,
                    source_file=source_file,
                )


def display_path(path: Path, base_dir: Path | None = None) -> str:
    base_dir = (base_dir or Path.cwd()).resolve()
    resolved = path.resolve()
    try:
        rendered = resolved.relative_to(base_dir)
    except ValueError:
        rendered = resolved
    return rendered.as_posix()


def infer_display_base(input_root: Path) -> Path:
    """Prefer the repository root when input_root is under download_pipeline."""
    for parent in [input_root.resolve(), *input_root.resolve().parents]:
        if parent.name == "download_pipeline":
            return parent.parent
    return Path.cwd().resolve()
