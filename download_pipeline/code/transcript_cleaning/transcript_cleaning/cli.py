from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from .clean_text import clean_text_value
from .clean_timestamps import clean_timestamp
from .config import CleanConfig, validate_roots
from .discover import TranscriptFile, discover_transcript_files, display_path, infer_display_base
from .load_asr_json import ASRLoadError, load_asr_json
from .schema import build_chunk_record, invalid_chunk_record
from .writer import OutputWriter


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    input_root, output_root = validate_roots(Path(args.input_root), Path(args.output_root))
    config = CleanConfig(input_root=input_root, output_root=output_root)
    run_cleaning(config)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean ASR transcript JSON files.")
    parser.add_argument(
        "--input-root",
        default="download_pipeline/processed_data/main",
        help="Read-only root containing day*/{view}/transcript/*.json files.",
    )
    parser.add_argument(
        "--output-root",
        default="download_pipeline/processed_data/cleaned_transcripts",
        help="Output directory. Must be named cleaned_transcripts.",
    )
    return parser


def run_cleaning(config: CleanConfig) -> dict[str, Any]:
    files = list(discover_transcript_files(config.input_root))
    base_dir = infer_display_base(config.input_root)
    stats = _empty_stats(config, total_files=len(files), base_dir=base_dir)

    with OutputWriter(config.output_root) as writer:
        writer.log(f"input_root={display_path(config.input_root, base_dir)}")
        writer.log(f"output_root={display_path(config.output_root, base_dir)}")
        writer.log(f"manifest_root={display_path(writer.manifest_root, base_dir)}")
        writer.log(f"discovered_files={len(files)}")

        for item in files:
            source_file = display_path(item.source_file, base_dir)
            _increment_view_file(stats, item.view)
            try:
                doc = load_asr_json(item.source_file)
            except ASRLoadError as exc:
                stats["failed_files"] += 1
                failed = _failed_file_record(item, source_file, exc.error_type, exc.message)
                writer.write_failed(failed)
                writer.write_manifest(_manifest_failure(item, source_file, exc.error_type))
                writer.log(f"failed {source_file} {exc.error_type}: {exc.message}")
                continue

            stats["success_files"] += 1
            file_counts = _process_file(config, writer, stats, item, source_file, doc.chunks)
            writer.write_manifest(_manifest_success(item, source_file, file_counts))
            writer.log(
                "success "
                f"{source_file} total_chunks={file_counts['total_chunks']} "
                f"written_chunks={file_counts['written_chunks']} "
                f"invalid_chunks={file_counts['invalid_chunks']}"
            )

        writer.write_summary(_finalize_stats(stats))

    return stats


def _process_file(
    config: CleanConfig,
    writer: OutputWriter,
    stats: dict[str, Any],
    item: TranscriptFile,
    source_file: str,
    chunks: list[Any],
) -> dict[str, int]:
    counts = {
        "total_chunks": len(chunks),
        "written_chunks": 0,
        "invalid_chunks": 0,
        "empty_chunks": 0,
        "repetition_chunks": 0,
    }
    stats["total_chunks"] += len(chunks)
    previous_end: float | None = None

    for raw_chunk_index, chunk in enumerate(chunks):
        raw_text, raw_timestamp, format_flags = _extract_chunk_fields(chunk)
        timestamp_result = clean_timestamp(
            raw_timestamp,
            min_duration_sec=config.min_duration_sec,
            max_duration_sec=config.max_duration_sec,
        )
        extra_flags = list(format_flags)
        if (
            previous_end is not None
            and timestamp_result.start_time is not None
            and timestamp_result.start_time < previous_end
        ):
            extra_flags.append("overlapping_timestamp")
        if timestamp_result.end_time is not None:
            previous_end = timestamp_result.end_time

        text_result = clean_text_value(raw_text)
        record = build_chunk_record(
            item=item,
            source_file=source_file,
            raw_chunk_index=raw_chunk_index,
            raw_timestamp=raw_timestamp,
            text_result=text_result,
            timestamp_result=timestamp_result,
            extra_flags=extra_flags,
        )
        writer.write_chunk(item.view, record)

        counts["written_chunks"] += 1
        stats["written_chunks"] += 1
        stats["views"][item.view]["chunks"] += 1

        if record["is_empty"]:
            counts["empty_chunks"] += 1
            stats["empty_chunks"] += 1
        if record["has_repetition"]:
            counts["repetition_chunks"] += 1
            stats["repetition_chunks"] += 1
        if record["quality_flags"]:
            counts["invalid_chunks"] += 1
            stats["invalid_chunks"] += 1
            writer.write_invalid(invalid_chunk_record(record))

    return counts


def _extract_chunk_fields(chunk: Any) -> tuple[Any, Any, list[str]]:
    if not isinstance(chunk, dict):
        return "", None, ["invalid_chunk_format"]
    return chunk.get("text", ""), chunk.get("timestamp"), []


def _empty_stats(config: CleanConfig, *, total_files: int, base_dir: Path) -> dict[str, Any]:
    return {
        "input_root": display_path(config.input_root, base_dir),
        "output_root": display_path(config.output_root, base_dir),
        "manifest_root": display_path(config.output_root / "_manifests", base_dir),
        "total_files": total_files,
        "success_files": 0,
        "failed_files": 0,
        "total_chunks": 0,
        "written_chunks": 0,
        "invalid_chunks": 0,
        "empty_chunks": 0,
        "repetition_chunks": 0,
        "views": defaultdict(lambda: {"files": 0, "chunks": 0}),
    }


def _finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    finalized = dict(stats)
    views = stats["views"]
    finalized["views"] = {
        view: views[view] for view in sorted(views.keys(), key=lambda value: value.casefold())
    }
    return finalized


def _increment_view_file(stats: dict[str, Any], view: str) -> None:
    stats["views"][view]["files"] += 1


def _manifest_success(
    item: TranscriptFile,
    source_file: str,
    counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "day": item.day,
        "view": item.view,
        "hour": item.hour,
        "source_file": source_file,
        "parse_status": "success",
        **counts,
    }


def _manifest_failure(
    item: TranscriptFile,
    source_file: str,
    error_type: str,
) -> dict[str, Any]:
    return {
        "day": item.day,
        "view": item.view,
        "hour": item.hour,
        "source_file": source_file,
        "parse_status": "failed",
        "error_type": error_type,
        "total_chunks": 0,
        "written_chunks": 0,
        "invalid_chunks": 0,
        "empty_chunks": 0,
        "repetition_chunks": 0,
    }


def _failed_file_record(
    item: TranscriptFile,
    source_file: str,
    error_type: str,
    error_message: str,
) -> dict[str, Any]:
    return {
        "day": item.day,
        "view": item.view,
        "hour": item.hour,
        "source_file": source_file,
        "error_type": error_type,
        "error_message": error_message,
    }


if __name__ == "__main__":
    raise SystemExit(main())
