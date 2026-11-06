from __future__ import annotations

from castle_pipeline.cli import main as cli_main


def main() -> int:
    return cli_main(["validate"])


if __name__ == "__main__":
    raise SystemExit(main())
