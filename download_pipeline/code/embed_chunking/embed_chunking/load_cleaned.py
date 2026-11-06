from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from .schema import LoadedFile
from .validate_record import count_missing_optional, validate_record


def load_cleaned_file(path: str | Path, view: str) -> LoadedFile:
    input_path = Path(path)
    records = []
    skipped = []
    invalid = []
    missing_optional: Counter[str] = Counter()
    input_records = 0

    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            input_records += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid.append(
                    {
                        "input_file": _path_string(input_path),
                        "line_number": line_number,
                        "error_type": "json_parse_error",
                        "error_message": str(exc),
                        "raw_line": line.rstrip("\n"),
                    }
                )
                continue

            missing_optional.update(count_missing_optional(raw))
            result = validate_record(raw, _path_string(input_path), line_number)
            if result.valid is not None:
                records.append(result.valid)
            elif result.skipped is not None:
                skipped.append(result.skipped)
            elif result.invalid is not None:
                invalid.append(result.invalid)

    return LoadedFile(
        view=view,
        input_file=_path_string(input_path),
        input_records=input_records,
        records=records,
        skipped=skipped,
        invalid=invalid,
        missing_optional_fields=dict(missing_optional),
    )


def _path_string(path: Path) -> str:
    return path.as_posix()
