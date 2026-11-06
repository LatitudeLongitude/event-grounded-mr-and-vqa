from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tipsv2_image_cls.benchmark import BenchmarkOptions, execute_benchmark
from tipsv2_image_cls.config import AppConfig
from tipsv2_image_cls.inventory import discover_tasks, filter_tasks
from tipsv2_image_cls.paths import configure_runtime_environment
from tipsv2_image_cls.pipeline import RunOptions, run_pipeline
from tipsv2_image_cls.preflight import run_preflight
from tipsv2_image_cls.progress import ProgressStore
from tipsv2_image_cls.validation import validate_shard


def _add_common(subparser):
    subparser.add_argument("--config", required=True, help="YAML configuration path")


def _add_filters(subparser):
    subparser.add_argument("--day")
    subparser.add_argument("--source")
    subparser.add_argument("--hour")
    subparser.add_argument("--max-hours", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TIPSv2 g/14 image CLS embedding pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="Discover hour tasks from the global video summary")
    _add_common(inventory)

    preflight = subparsers.add_parser("preflight", help="Validate paths, disks, model, dependencies, and GPU safety")
    _add_common(preflight)
    preflight.add_argument("--gpu-ids", help="Comma-separated GPU indices")
    preflight.add_argument("--hash-weights", action="store_true", help="Also hash the full model.safetensors file")

    run = subparsers.add_parser("run", help="Run TIPSv2 image embedding")
    _add_common(run)
    _add_filters(run)
    run.add_argument("--gpu-ids", help="Comma-separated GPU indices")
    run.add_argument("--decode-workers", type=int, choices=(2, 4, 6), help="Override image decode worker count")
    run.add_argument("--max-frames-per-hour", type=int)
    run.add_argument("--output-mode", choices=("formal", "smoke"), default="formal")
    run.add_argument("--encoder-backend", choices=("tipsv2",), default="tipsv2")
    run.add_argument("--run-id")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--recompute", help="Single shard key dayN/source/HH; writes to recompute/<run_id>")

    validate = subparsers.add_parser("validate", help="Validate existing Parquet shards")
    _add_common(validate)
    _add_filters(validate)
    validate.add_argument("--output-mode", choices=("formal", "smoke", "recompute"), default="formal")
    validate.add_argument("--run-id")
    validate.add_argument("--max-frames-per-hour", type=int)

    status = subparsers.add_parser("status", help="Summarize SQLite hour states")
    _add_common(status)
    status.add_argument("--output-mode", choices=("formal", "smoke", "recompute"), default="formal")
    status.add_argument("--run-id")

    benchmark = subparsers.add_parser("benchmark", help="Benchmark batch size on one safe idle GPU")
    _add_common(benchmark)
    _add_filters(benchmark)
    benchmark.add_argument("--gpu-ids", help="Exactly one GPU index")
    benchmark.add_argument("--batch-sizes", default="2,4,8,16,24,32")
    benchmark.add_argument("--max-frames-per-hour", type=int, default=128)
    benchmark.add_argument("--throughput-threshold", type=float, default=0.95)
    benchmark.add_argument("--max-peak-memory-fraction", type=float, default=0.90)
    benchmark.add_argument("--run-id")

    return parser


def _gpu_ids(value, defaults):
    if value is None:
        return tuple(defaults)
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if any(gpu_id < 0 for gpu_id in result):
        raise ValueError("GPU IDs cannot be negative")
    if len(result) != len(set(result)):
        raise ValueError("GPU IDs must be unique")
    return result


def _run_root(config: AppConfig, mode: str, run_id: str | None) -> Path:
    if mode == "formal":
        return config.paths.output_root
    if not run_id:
        raise ValueError(f"--run-id is required for {mode} status/validation")
    return config.paths.output_root / mode / run_id


def command_inventory(config: AppConfig) -> int:
    inventory = discover_tasks(config.paths.data_root, config.summary_path)
    configure_runtime_environment(config.paths.output_root)
    report = inventory.as_dict()
    report_path = config.paths.output_root / "_reports" / f"inventory-{time.strftime('%Y%m%dT%H%M%S')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "report_path": str(report_path)}, indent=2, sort_keys=True))
    return 0


def command_run(config: AppConfig, args) -> int:
    output_mode = args.output_mode
    day, source, hour = args.day, args.source, args.hour
    if args.recompute:
        parts = args.recompute.split("/")
        if len(parts) != 3 or not parts[2].isdigit():
            raise ValueError("--recompute must be dayN/source/HH")
        day, source, hour = parts[0], parts[1], f"{int(parts[2]):02d}"
        output_mode = "recompute"
    summary = run_pipeline(
        config,
        RunOptions(
            encoder_backend=args.encoder_backend,
            output_mode=output_mode,
            run_id=args.run_id,
            gpu_ids=_gpu_ids(args.gpu_ids, config.runtime.gpu_ids),
            day=day,
            source=source,
            hour=hour,
            max_hours=1 if args.recompute else args.max_hours,
            max_frames_per_hour=args.max_frames_per_hour,
            decode_workers=args.decode_workers,
            resume=args.resume,
            dry_run=args.dry_run,
        ),
    )
    print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
    return 1 if summary.failed else 0


def command_validate(config: AppConfig, args) -> int:
    if args.max_frames_per_hour is not None and args.output_mode != "smoke":
        raise ValueError("--max-frames-per-hour is only valid for isolated smoke validation")
    if args.max_frames_per_hour is not None and args.max_frames_per_hour <= 0:
        raise ValueError("--max-frames-per-hour must be positive")
    root = _run_root(config, args.output_mode, args.run_id)
    namespace = "main" if args.output_mode == "formal" else f"{args.output_mode}:{args.run_id}"
    store = ProgressStore(root / "_state" / "progress.sqlite3")
    database_exists = store.path.is_file()
    inventory = discover_tasks(config.paths.data_root, config.summary_path)
    tasks = filter_tasks(inventory.tasks, day=args.day, source=args.source, hour=args.hour, max_hours=args.max_hours)
    results = []
    for task in tasks:
        expected = min(task.frame_count, args.max_frames_per_hour) if args.max_frames_per_hour else task.frame_count
        path = root / "main" / task.day / task.source / f"{task.hour}.parquet"
        state = (
            store.get_shard(namespace, task.day_number, task.source, task.hour)
            if database_exists
            else None
        )
        state_error = None
        if state is None:
            state_error = "StateMismatch: shard is missing from SQLite"
        elif state["status"] != "completed":
            state_error = f"StateMismatch: SQLite status is {state['status']!r}, expected 'completed'"
        elif state["file_size"] is None or not state["sha256"]:
            state_error = "StateMismatch: completed shard is missing file size or SHA256"
        validation = validate_shard(
            path,
            expected_rows=expected,
            expected_file_size=state["file_size"] if state_error is None else None,
            expected_sha256=state["sha256"] if state_error is None else None,
            compute_sha256=True,
        )
        item = {
            "task": task.key,
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
    db = ProgressStore(root / "_state" / "progress.sqlite3")
    if not db.path.is_file():
        print(json.dumps({"database": str(db.path), "exists": False}, indent=2))
        return 1
    rows = db.list_shards()
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(json.dumps({"database": str(db.path), "counts": counts, "shards": rows}, indent=2, sort_keys=True, default=str))
    return 0


def command_benchmark(config: AppConfig, args) -> int:
    if not (args.day and args.source and args.hour):
        raise ValueError("benchmark requires --day, --source, and --hour")
    gpu_ids = _gpu_ids(args.gpu_ids, config.runtime.gpu_ids[:1])
    if len(gpu_ids) != 1:
        raise ValueError("benchmark must use exactly one GPU")
    try:
        batch_sizes = tuple(int(value.strip()) for value in args.batch_sizes.split(",") if value.strip())
    except ValueError as exc:
        raise ValueError("--batch-sizes must be a comma-separated list of integers") from exc
    base_run_id = args.run_id or time.strftime("benchmark-%Y%m%dT%H%M%S")
    result = execute_benchmark(
        config,
        BenchmarkOptions(
            run_id=base_run_id,
            gpu_id=gpu_ids[0],
            day=args.day,
            source=args.source,
            hour=args.hour,
            max_frames=args.max_frames_per_hour,
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
                require_idle_gpus=True,
                hash_weights=args.hash_weights,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if args.command == "run":
            return command_run(config, args)
        if args.command == "validate":
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
