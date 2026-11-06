from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .config import AppConfig
from .pipeline import RUN_ALL_STAGES, run_all, run_root, run_stage

COMMANDS = [
    "inventory",
    "build-atoms",
    "visual-boundaries",
    "text-boundaries",
    "sparse-keyword-features",
    "fuse-boundaries",
    "segment-sources",
    "group-global-events",
    "build-episodes",
    "build-sessions",
    "build-l4-l5",
    "build-event-embeddings",
    "build-corpus-index",
    "validate",
    "export-event-corpus",
    "run-all",
    "status",
]


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-mode", choices=["formal", "smoke"], default="formal")
    parser.add_argument("--run-id")
    parser.add_argument("--day")
    parser.add_argument("--source")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="event_pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        child = sub.add_parser(command)
        _add_common(child)
    return parser


def _status(root: Path) -> dict:
    manifests = sorted((root / "manifests").glob("*.json")) if (root / "manifests").exists() else []
    output_dirs = (
        "inventory",
        "atoms",
        "boundaries",
        "features",
        "segments",
        "embeddings",
        "links",
        "indexes",
        "manifests",
        "quality",
        "reports",
    )
    core_outputs = []
    for dirname in output_dirs:
        base = root / dirname
        if base.exists():
            core_outputs.extend(
                path.relative_to(root).as_posix()
                for path in base.rglob("*")
                if path.is_file() and path.suffix in {".parquet", ".json", ".jsonl", ".md"}
            )
    return {
        "run_root": root.as_posix(),
        "manifests": [path.name for path in manifests],
        "core_outputs": sorted(core_outputs),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = AppConfig.load(args.config)
    if args.output_mode in {"formal", "smoke"} and not args.run_id:
        parser.error(f"--run-id is required for {args.output_mode} mode")
    root = run_root(config, args.output_mode, args.run_id)
    if args.dry_run:
        plan = RUN_ALL_STAGES if args.command == "run-all" else [args.command]
        print(json.dumps({"run_root": root.as_posix(), "stages": plan}, indent=2, ensure_ascii=False))
        return 0
    if args.command == "status":
        print(json.dumps(_status(root), indent=2, ensure_ascii=False))
        return 0
    if args.command == "run-all":
        results = run_all(config, output_mode=args.output_mode, run_id=args.run_id, day=args.day, source=args.source, resume=args.resume)
    else:
        results = run_stage(config, args.command, output_mode=args.output_mode, run_id=args.run_id, day=args.day, source=args.source, resume=args.resume)
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
