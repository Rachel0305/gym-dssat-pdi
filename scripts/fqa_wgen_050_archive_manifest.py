"""Build or verify the curated exact-byte archive for FQA checkpoint diagnostic 050."""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/fqa_wgen_050_file_manifest.csv"
BASE = ROOT / "results/fqa_047_checkpoint_full20_heldout_050"
FILES = [
    ROOT / "prompt_02/050_fqa_047_checkpoint_full20_heldout_evaluation.md",
    ROOT / "docs/fqa_047_checkpoint_full20_heldout_050_record.md",
    ROOT / "docs/fqa_wgen_archive_latest.md",
    ROOT / "scripts/fqa_wgen_050_archive_manifest.py",
    BASE / "run_checkpoint_eval.py",
    BASE / "audit_gate.py",
    BASE / "paired_checkpoint_full20.csv",
    BASE / "final_gate.json",
    BASE / "audit_attempt01_failed_gate.json",
]
for label in ("5k", "10k"):
    directory = BASE / label
    FILES.extend(directory / name for name in (
        "schedule.json",
        "preflight.json",
        "episode_manifest.csv",
        "endpoint_and_action_summary.csv",
        "daily_action_state_trace.csv",
        "resource_usage.csv",
        "result.json",
    ))
    FILES.extend(sorted((directory / "logs/FQA").glob("*.log")))
    FILES.extend(sorted((directory / "weather_daily").glob("*.csv")))
    FILES.extend(sorted((directory / "runtime_evidence").glob("*.json")))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def actual_rows() -> list[dict]:
    missing = [path for path in FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing archive input(s): " + ", ".join(str(p) for p in missing))
    return [
        {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)}
        for path in FILES
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    actual = actual_rows()
    if args.verify:
        with OUT.open(encoding="utf-8", newline="") as stream:
            expected = list(csv.DictReader(stream))
        ok = len(actual) == len(expected) and all(
            actual_row["path"] == expected_row["path"]
            and str(actual_row["bytes"]) == expected_row["bytes"]
            and actual_row["sha256"] == expected_row["sha256"]
            for actual_row, expected_row in zip(actual, expected)
        )
        print(f"archive_files={len(actual)} failed={0 if ok else 1}")
        if not ok:
            raise SystemExit(2)
    else:
        mode = "w" if args.rebuild else "x"
        with OUT.open(mode, encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["path", "bytes", "sha256"])
            writer.writeheader()
            writer.writerows(actual)
        print(f"archive_files={len(actual)} bytes={sum(row['bytes'] for row in actual)}")


if __name__ == "__main__":
    main()
