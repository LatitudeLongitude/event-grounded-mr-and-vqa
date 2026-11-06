from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap() -> None:
    here = Path(__file__).resolve()
    core_root = here.parents[1] / "core_modules"
    if str(core_root) not in sys.path:
        sys.path.insert(0, str(core_root))


def main(argv: list[str] | None = None) -> int:
    _bootstrap()
    from castle_mm26_core.cli import main as core_main

    return core_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
