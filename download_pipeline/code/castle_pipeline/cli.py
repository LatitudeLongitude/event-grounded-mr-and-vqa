from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .logging_utils import log_command_result_json
from .pipeline import CastlePipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CASTLE MM26 low-disk download/process pipeline")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.yaml"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    sub.add_parser("scan")
    sub.add_parser("sync-non-video")
    sub.add_parser("download-non-video")
    run = sub.add_parser("run")
    run.add_argument("--smoke-test", action="store_true")
    run.add_argument("--limit-videos", type=int, default=None)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--skip-non-video", action="store_true")
    run.add_argument("--no-resize-frames", action="store_true", help="Disable the post-extraction 1920px frame resize step for this run.")
    sub.add_parser("status")
    verify = sub.add_parser("verify")
    verify.add_argument("--strict-non-video-sha256", action="store_true")
    sub.add_parser("repair")
    clean = sub.add_parser("clean-raw-cache")
    clean.add_argument("--dry-run", action="store_true")
    sub.add_parser("dry-run")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command
    try:
        cfg = PipelineConfig.from_yaml(args.config)
        if command == "run" and args.no_resize_frames:
            cfg = replace(cfg, processing=replace(cfg.processing, resize_extracted_frames=False))
        pipeline = CastlePipeline(cfg)
        pipeline.logger.info("command started: %s", command)
        if command == "validate":
            result = pipeline.validate()
        elif command == "scan":
            result = pipeline.scan()
        elif command in {"sync-non-video", "download-non-video"}:
            result = pipeline.sync_non_video()
        elif command == "run":
            pipeline.run(
                smoke_test=args.smoke_test,
                limit_videos=args.limit_videos,
                dry_run=args.dry_run,
                skip_non_video=args.skip_non_video,
            )
            result = pipeline.status()
        elif command == "status":
            result = pipeline.status()
        elif command == "verify":
            result = pipeline.verify(strict_non_video_sha256=args.strict_non_video_sha256)
        elif command == "repair":
            result = pipeline.repair()
        elif command == "clean-raw-cache":
            result = pipeline.clean_raw_cache(dry_run=args.dry_run)
        elif command == "dry-run":
            result = pipeline.dry_run()
        else:
            raise AssertionError(command)
        result_json = json.dumps(result, ensure_ascii=False, indent=2)
        log_command_result_json(pipeline.logger, command, result_json)
        print(result_json)
        if command == "validate" and not result.get("passed"):
            return 2
        pipeline.logger.info("command completed: %s", command)
        return 0
    except KeyboardInterrupt:
        message = {"passed": False, "command": command, "error": "interrupted by user"}
        try:
            pipeline.logger.warning("command interrupted by user: %s", command)  # type: ignore[name-defined]
            log_command_result_json(pipeline.logger, command, json.dumps(message, ensure_ascii=False, indent=2))  # type: ignore[name-defined]
        except Exception:
            _write_emergency_cli_log(args.config, message)
        print(json.dumps(message, ensure_ascii=False, indent=2))
        return 130
    except Exception as exc:
        message = {"passed": False, "command": command, "error": str(exc)}
        try:
            pipeline.logger.exception("command failed: %s error=%s", command, exc)  # type: ignore[name-defined]
            log_command_result_json(pipeline.logger, command, json.dumps(message, ensure_ascii=False, indent=2))  # type: ignore[name-defined]
        except Exception:
            _write_emergency_cli_log(args.config, message)
        print(json.dumps(message, ensure_ascii=False, indent=2))
        return 1


def _write_emergency_cli_log(config_path: str, message: dict[str, Any]) -> None:
    try:
        path = Path(config_path).resolve().parent / "cli_error.log"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(message, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
