from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .build_chunks import build_chunks
from .config import ChunkingConfig
from .discover import discover_input_files
from .load_cleaned import load_cleaned_file
from .summary import build_summary, manifest_rows
from .writer import validate_output_root, write_outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate embedding-ready transcript chunks.")
    parser.add_argument("--input-root", required=True, help="Cleaned transcripts directory.")
    parser.add_argument("--output-root", required=True, help="Stage 2 embed_chunks output directory.")
    parser.add_argument("--target-min-token-estimate", type=int, default=ChunkingConfig.target_min_token_estimate)
    parser.add_argument("--target-max-token-estimate", type=int, default=ChunkingConfig.target_max_token_estimate)
    parser.add_argument("--hard-max-token-estimate", type=int, default=ChunkingConfig.hard_max_token_estimate)
    args = parser.parse_args(argv)

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    config = ChunkingConfig(
        target_min_token_estimate=args.target_min_token_estimate,
        target_max_token_estimate=args.target_max_token_estimate,
        hard_max_token_estimate=args.hard_max_token_estimate,
    )

    try:
        config.validate()
        validate_output_root(output_root, input_root)
        input_files = discover_input_files(input_root)
    except (OSError, ValueError) as exc:
        print(f"embed_chunking: {exc}", file=sys.stderr)
        return 2

    loaded_files = []
    all_records = []
    skipped_rows = []
    invalid_rows = []
    log_lines = [
        f"input_root={input_root.as_posix()}",
        f"output_root={output_root.as_posix()}",
        f"discovered_input_files={len(input_files)}",
    ]

    for input_file in input_files:
        loaded = load_cleaned_file(input_file.path, input_file.view)
        loaded_files.append(loaded)
        all_records.extend(loaded.records)
        skipped_rows.extend(loaded.skipped)
        invalid_rows.extend(loaded.invalid)
        log_lines.append(
            "processed "
            f"view={loaded.view} input_records={loaded.input_records} "
            f"valid={len(loaded.records)} skipped={len(loaded.skipped)} invalid={len(loaded.invalid)}"
        )

    chunks = build_chunks(all_records, config)
    summary = build_summary(
        input_root=input_root.as_posix(),
        output_root=output_root.as_posix(),
        main_output=(output_root / "chunks.jsonl").as_posix(),
        meta_output=(output_root / "meta").as_posix(),
        loaded_files=loaded_files,
        chunks=chunks,
    )
    log_lines.append(f"output_chunks={len(chunks)}")

    try:
        write_outputs(
            output_root=output_root,
            chunks=chunks,
            manifest_rows=manifest_rows(loaded_files, chunks),
            skipped_rows=skipped_rows,
            invalid_rows=invalid_rows,
            summary=summary,
            config=config,
            log_lines=log_lines,
        )
    except (OSError, ValueError) as exc:
        print(f"embed_chunking: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
