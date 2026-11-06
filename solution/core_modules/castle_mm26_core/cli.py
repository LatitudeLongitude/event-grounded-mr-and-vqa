from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

from .config import load_config
from .pipeline import full_smoke_export, full_smoke_package, merge_and_package, preflight, run_tasks


FULL_CONFIRM = "RUN_CASTLE_MM26_FULL"
SERVER_PROJECT_VENV_COMMANDS = {"smoke", "full-smoke-export", "run", "eis", "ois", "qa", "qa-export", "merge"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CASTLE MM26 fully-automatic solution runner")
    parser.add_argument("command", choices=["preflight", "smoke", "full-smoke-export", "full-smoke-package", "run", "eis", "ois", "qa", "qa-export", "qa-answer", "merge"], help="command to execute")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--environment", default=None, help="override active_environment_profile for diagnostics")
    parser.add_argument("--devices", default="0", help="comma-separated GPU ids, e.g. 0 or 0,1,2")
    parser.add_argument("--task-id", action="append", default=[], help="task id to run; may be repeated")
    parser.add_argument("--no-fetch-official", action="store_true")
    parser.add_argument("--with-models", action="store_true", help="preflight model runtimes")
    parser.add_argument("--write-zip", action="store_true", help="allow ZIP writing during merge")
    parser.add_argument("--confirm-official", default="", help=f"required for full run: {FULL_CONFIRM}")
    parser.add_argument("--resume", action="store_true", help="skip already completed per-query task directories")
    parser.add_argument("--env-file", default=None, type=Path, help="optional dotenv-style file for local runtime secrets")
    parser.add_argument("--qa-model", "--model", dest="qa_model", default=None, help="optional OpenRouter model override for smoke or local QA answer")
    parser.add_argument("--packet", action="append", default=[], type=Path, help="QA evidence packet for qa-answer; may be repeated")
    parser.add_argument("--smoke-mode", action="store_true", help="use small retrieval limits for qa-export diagnostics")
    parser.add_argument("--merge-after-run", action="store_true", help="run merge after smoke diagnostics; packaging is otherwise triggered by the merge command")
    parser.add_argument("--full-smoke-all-queries", action="store_true", help="run/package Top-100 full-smoke for all 36 queries instead of the default 3-query diagnostic set")
    sort_group = parser.add_mutually_exclusive_group()
    sort_group.add_argument("--sort-interval-rows-by-final-score", dest="sort_interval_rows_by_final_score", action="store_true", default=None, help="temporarily sort interval rows within each task by final_score descending")
    sort_group.add_argument("--no-sort-interval-rows-by-final-score", dest="sort_interval_rows_by_final_score", action="store_false", help="temporarily preserve interval row input order instead of sorting by final_score")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _load_runtime_env(args.env_file or _default_env_file(args.config))
    if args.qa_model:
        os.environ["OPENROUTER_MODEL_OVERRIDE"] = args.qa_model
    config = load_config(args.config, active_environment=args.environment)
    _apply_sort_interval_rows_override(config, args.sort_interval_rows_by_final_score)
    _assert_server_project_venv(config, args)
    fetch_official = not args.no_fetch_official
    if args.command == "preflight":
        preflight(config, args.queries, args.run_id, fetch_official=fetch_official, with_models=args.with_models, devices=args.devices)
        return 0
    if args.command == "smoke":
        task_ids = args.task_id or ["MM26-EIS01", "MM26-OIS12", "MM26-QA01"]
        run_tasks(config, args.queries, args.run_id, task_ids=task_ids, devices=args.devices, smoke=True, fetch_official=fetch_official, resume=args.resume)
        if args.merge_after_run:
            merge_and_package(config, args.queries, args.run_id, write_zip=False)
        return 0
    if args.command == "full-smoke-export":
        full_smoke_export(
            config,
            args.queries,
            args.run_id,
            devices=args.devices,
            fetch_official=fetch_official,
            resume=args.resume,
            all_queries=_resolve_full_smoke_all_queries(config, args),
            task_ids=args.task_id,
        )
        return 0
    if args.command == "full-smoke-package":
        full_smoke_package(config, args.queries, args.run_id, all_queries=_resolve_full_smoke_all_queries(config, args), task_ids=args.task_id)
        return 0
    if args.command == "run":
        if args.confirm_official != FULL_CONFIRM:
            raise SystemExit(f"full run requires --confirm-official {FULL_CONFIRM}")
        run_tasks(config, args.queries, args.run_id, task_ids=args.task_id, devices=args.devices, smoke=False, fetch_official=fetch_official, resume=args.resume)
        return 0
    if args.command in {"eis", "ois", "qa"}:
        run_tasks(config, args.queries, args.run_id, task_ids=args.task_id, devices=args.devices, smoke=False, fetch_official=fetch_official, resume=args.resume, task_family=args.command)
        return 0
    if args.command == "qa-export":
        from .pipeline import export_qa_packets

        export_qa_packets(config, args.queries, args.run_id, task_ids=args.task_id, devices=args.devices, smoke=args.smoke_mode, fetch_official=fetch_official, resume=args.resume)
        return 0
    if args.command == "qa-answer":
        from .pipeline import answer_qa_packets

        answer_qa_packets(config, args.queries, args.run_id, packet_paths=args.packet, task_ids=args.task_id)
        return 0
    if args.command == "merge":
        merge_and_package(config, args.queries, args.run_id, write_zip=args.write_zip)
        return 0
    raise SystemExit(f"unknown command: {args.command}")


def _assert_server_project_venv(config, args: argparse.Namespace) -> None:
    if config.active_environment_profile != "server_linux":
        return
    if args.command == "preflight" and not args.with_models:
        return
    if args.command not in SERVER_PROJECT_VENV_COMMANDS and args.command != "preflight":
        return
    expected_prefix = (Path(config.path).resolve().parent / ".venv").resolve()
    actual_prefix = Path(sys.prefix).resolve()
    if actual_prefix == expected_prefix:
        return
    expected_python = expected_prefix / "bin" / "python"
    raise SystemExit(
        "server_linux model/official commands must run with the project Python "
        f"{expected_python.as_posix()}; current sys.executable={sys.executable!r}, "
        f"sys.prefix={sys.prefix!r}. Do not use conda base for model preflight, "
        "retrieval export, smoke export, or official merge."
    )


def _full_smoke_all_queries_enabled(config, cli_enabled: bool) -> bool:
    return bool(cli_enabled or config.full_smoke_all_queries)


def _resolve_full_smoke_all_queries(config, args: argparse.Namespace) -> bool:
    if args.task_id and args.full_smoke_all_queries:
        raise SystemExit("--task-id cannot be combined with --full-smoke-all-queries")
    if args.task_id:
        return False
    return _full_smoke_all_queries_enabled(config, args.full_smoke_all_queries)


def _apply_sort_interval_rows_override(config, override: bool | None) -> None:
    if override is None:
        return
    submission = config.data.get("submission")
    if not isinstance(submission, dict):
        submission = {}
        config.data["submission"] = submission
    submission["sort_interval_rows_by_final_score"] = bool(override)


def _default_env_file(config_path: Path) -> Path:
    return Path(config_path).resolve().parent / ".secrets" / "openrouter.env"


def _load_runtime_env(path: Path) -> dict[str, str]:
    target = Path(path)
    loaded: dict[str, str] = {}
    if not target.exists():
        return loaded
    allowed = {"OPENROUTER_API_KEY", "OPENROUTER_HTTP_REFERER"}
    for line_no, raw in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            raise ValueError(f"invalid env file line {line_no}: {target}")
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in allowed:
            continue
        parsed = shlex.split(value, posix=True)
        if len(parsed) != 1:
            raise ValueError(f"invalid env value for {key} at line {line_no}: {target}")
        if not os.environ.get(key):
            os.environ[key] = parsed[0]
            loaded[key] = "loaded"
    return loaded


if __name__ == "__main__":
    raise SystemExit(main())
