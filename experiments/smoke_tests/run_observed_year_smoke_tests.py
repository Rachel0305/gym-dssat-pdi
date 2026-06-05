from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER = PROJECT_ROOT / "src" / "run_smoke_tests.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimal", action="store_true", help="Run only HLA 2007 null_zero.")
    args = parser.parse_args()
    cmd = [sys.executable, str(RUNNER)]
    if args.minimal:
        cmd.append("--minimal")
    raise SystemExit(subprocess.call(cmd, cwd=str(PROJECT_ROOT)))


if __name__ == "__main__":
    main()
