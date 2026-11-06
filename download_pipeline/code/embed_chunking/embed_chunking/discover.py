from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InputFile:
    view: str
    path: Path


def discover_input_files(input_root: str | Path) -> list[InputFile]:
    root = Path(input_root)
    if not root.exists():
        raise FileNotFoundError(f"Input root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {root}")

    files = []
    for path in root.glob("*.jsonl"):
        if path.is_file():
            files.append(InputFile(view=path.stem, path=path))
    return sorted(files, key=lambda item: (item.view.lower(), item.view, item.path.name))
