from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from qwen3_text_dense.benchmark import BenchmarkOptions, execute_benchmark
from qwen3_text_dense.config import AppConfig
from qwen3_text_dense.inventory import discover_inventory
from qwen3_text_dense.pipeline import RunOptions, run_pipeline
from qwen3_text_dense.preflight import run_preflight
from qwen3_text_dense.progress import ProgressStore
from qwen3_text_dense.validation import validate_shard


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="YAML configuration path")


def _add_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--day", help="Optional day filter, for example day1")
    parser.add_argument("--view", help="Optional view/source filter, for example Allie")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3 document dense embedding pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="Inspect chunks.jsonl and shard candidates")
    _add_common(inventory)

    preflight = subparsers.add_parser("preflight", help="Validate paths, dependencies, model, disk, and GPU guard")
    _add_common(preflight)
    preflight.add_argument("--gpu-ids", help="Comma-separated GPU IDs")
    preflight.add_argument("--hash-weights", action="store_true", help="Hash all safetensors shards")
    preflight.add_argument("--skip-gpu-check", action="store_true", help="CPU-only preflight without GPU inspection")

    run = subparsers.add_parser("run", help="Run Qwen3 text dense embedding")
    _add_common(run)
    _add_filters(run)
    run.add_argument("--gpu-ids", help="Comma-separated GPU IDs")
    run.add_argument("--max-chunks", type=int, help="Limit chunks; forbidden for formal")
    run.add_argument("--output-mode", choices=("formal", "smoke"), default="formal")
    run.add_argument("--encoder-backend", choices=("qwen3",), default="qwen3")
    run.add_argument("--run-id")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--dry-run", action="store_true")

    validate = subparsers.add_parser("validate", help="Validate Parquet shards against SQLite")
    _add_common(validate)
    _add_filters(validate)
    validate.add_argument("--output-mode", choices=("formal", "smoke", "benchmark"), default="formal")
    validate.add_argument("--run-id")
    validate.add_argument("--max-chunks", type=int)

    verify = subparsers.add_parser("verify", help="Alias for validate")
    _add_common(verify)
    _add_filters(verify)
    verify.add_argument("--output-mode", choices=("formal", "smoke", "benchmark"), default="formal")
    verify.add_argument("--run-id")
    verify.add_argument("--max-chunks", type=int)

    status = subparsers.add_parser("status", help="Summarize SQLite shard states and temporary files")
    _add_common(status)
    status.add_argument("--output-mode", choices=("formal", "smoke", "benchmark"), default="formal")
    status.add_argument("--run-id")

    benchmark = subparsers.add_parser("benchmark", help="Benchmark candidate batch sizes on one safe idle GPU")
    _add_common(benchmark)
    _add_filters(benchmark)
    benchmark.add_argument("--gpu-ids", help="Exactly one GPU ID")
    benchmark.add_argument("--batch-sizes", default="1,2,4,8,16")
    benchmark.add_argument("--max-chunks", type=int, default=256)
    benchmark.add_argument("--throughput-threshold", type=float, default=0.95)
    benchmark.add_argument("--max-peak-memory-fraction", type=float, default=0.90)
    benchmark.add_argument("--run-id")

    return parser


def _gpu_ids(value: str | None, defaults) -> tuple[int, ...]:
    if value is None:
        return tuple(defaults)
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if any(gpu_id < 0 for gpu_id in result):
        raise ValueError("GPU IDs cannot be negative")
    if len(result) != len(set(result)):
        raise ValueError("GPU IDs must be unique")
    return result


def _run_root(config: AppConfig, output_mode: str, run_id: str | None) -> Path:
    if output_mode == "formal":
        return config.paths.output_root
    if not run_id:
        raise ValueError(f"--run-id is required for {output_mode}")
    return config.paths.output_root / output_mode / run_id


def command_inventory(config: AppConfig) -> int:
    inventory = discover_inventory(config)
    report = inventory.as_dict()
    report_path = config.paths.require_output(
        config.paths.output_root / "_reports" / f"inventory-{time.strftime('%Y%m%dT%H%M%S')}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "report_path": str(report_path)}, indent=2, sort_keys=True))
    return 1 if report["invalid_rows"] or report["duplicate_ids"] else 0


def command_run(config: AppConfig, args) -> int:
    summary = run_pipeline(
        config,
        RunOptions(
            encoder_backend=args.encoder_backend,
            output_mode=args.output_mode,
            run_id=args.run_id,
            gpu_ids=_gpu_ids(args.gpu_ids, config.runtime.gpu_ids),
            day=args.day,
            view=args.view,
            max_chunks=args.max_chunks,
            resume=args.resume,
            dry_run=args.dry_run,
        ),
    )
    print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))
    return 1 if summary.failed else 0


def command_validate(config: AppConfig, args) -> int:
    if args.output_mode == "formal" and args.max_chunks is not None:
        raise ValueError("--max-chunks is only valid for isolated smoke/benchmark validation")
    root = _run_root(config, args.output_mode, args.run_id)
    namespace = "main" if args.output_mode == "formal" else f"{args.output_mode}:{args.run_id}"
    store = ProgressStore(root / "_state" / "progress.sqlite3")
    if not store.path.is_file():
        print(json.dumps({"database": str(store.path), "exists": False}, indent=2))
        return 1
    inventory = discover_inventory(config)
    shards = inventory.selected_shards(day=args.day, view=args.view, max_chunks=args.max_chunks)
    results = []
    for shard in shards:
        state = store.get_shard(namespace, shard.day, shard.view)
        state_error = None
        if state is None:
            state_error = "StateMismatch: shard is missing from SQLite"
            path = root / "main" / shard.day / f"{shard.view}.parquet"
            expected_rows = shard.row_count
        else:
            path = Path(state["output_path"])
            expected_rows = int(state["expected_rows"])
            if state["status"] != "completed":
                state_error = f"StateMismatch: SQLite status is {state['status']!r}, expected 'completed'"
            elif state["file_size"] is None or not state["sha256"]:
                state_error = "StateMismatch: completed shard is missing file size or SHA256"
        validation = validate_shard(
            path,
            expected_rows=expected_rows,
            expected_file_size=state["file_size"] if state and state_error is None else None,
            expected_sha256=state["sha256"] if state and state_error is None else None,
            compute_sha256=True,
        )
        item = {
            "shard": shard.key,
            **validation.__dict__,
            "path": str(validation.path),
            "sqlite_status": None if state is None else state["status"],
            "sqlite_file_size": None if state is None else state["file_size"],
            "sqlite_sha256": None if state is None else state["sha256"],
        }
        if state_error is not None:
            item["valid"] = False
            item["error"] = item["error"] or state_error
        results.append(item)
    invalid = sum(not item["valid"] for item in results)
    print(json.dumps({"checked": len(results), "invalid": invalid, "results": results}, indent=2, sort_keys=True, default=str))
    return 1 if invalid else 0


def command_status(config: AppConfig, args) -> int:
    root = _run_root(config, args.output_mode, args.run_id)
    store = ProgressStore(root / "_state" / "progress.sqlite3")
    tmp_files = [str(path) for path in config.paths.output_root.rglob("*.tmp.*")]
    if not store.path.is_file():
        print(json.dumps({"database": str(store.path), "exists": False, "tmp_files": tmp_files}, indent=2))
        return 1
    rows = store.list_shards()
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(json.dumps({
        "database": str(store.path),
        "counts": counts,
        "shards": rows,
        "tmp_files": tmp_files,
        "invocations": store.list_invocations(),
    }, indent=2, sort_keys=True, default=str))
    return 0


def command_benchmark(config: AppConfig, args) -> int:
    gpu_ids = _gpu_ids(args.gpu_ids, config.runtime.gpu_ids[:1])
    if len(gpu_ids) != 1:
        raise ValueError("benchmark must use exactly one GPU")
    batch_sizes = tuple(int(value.strip()) for value in args.batch_sizes.split(",") if value.strip())
    result = execute_benchmark(
        config,
        BenchmarkOptions(
            run_id=args.run_id or time.strftime("qwen3-benchmark-%Y%m%dT%H%M%S"),
            gpu_id=gpu_ids[0],
            day=args.day,
            view=args.view,
            max_chunks=args.max_chunks,
            batch_sizes=batch_sizes,
            throughput_threshold=args.throughput_threshold,
            max_peak_memory_fraction=args.max_peak_memory_fraction,
        ),
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return result.exit_code


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = AppConfig.load(args.config)
        if args.command == "inventory":
            return command_inventory(config)
        if args.command == "preflight":
            report = run_preflight(
                config,
                gpu_ids=_gpu_ids(args.gpu_ids, config.runtime.gpu_ids),
                require_idle_gpus=not args.skip_gpu_check,
                hash_weights=args.hash_weights,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if args.command == "run":
            return command_run(config, args)
        if args.command in {"validate", "verify"}:
            return command_validate(config, args)
        if args.command == "status":
            return command_status(config, args)
        if args.command == "benchmark":
            return command_benchmark(config, args)
        raise ValueError(f"Unsupported command: {args.command}")
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
