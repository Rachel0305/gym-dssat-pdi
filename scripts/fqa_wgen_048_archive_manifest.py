"""Build or verify a small, exact-byte FQA 048 Git archive."""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/fqa_wgen_048_file_manifest.csv"
BASE = ROOT / "results/fqa_wgen_10k_heldout_diagnostic_048"
FILES = [
    ROOT / "prompt_02/048_fqa_wgen_10k_heldout_pair_diagnostic.md",
    ROOT / "docs/fqa_wgen_10k_heldout_048_record.md",
    ROOT / "scripts/fqa_wgen_048_archive_manifest.py",
    BASE / "run_eval.py",
    BASE / "audit_gate.py",
    BASE / "plot_048_figures.py",
    BASE / "final_gate.json",
    BASE / "paired_endpoints.csv",
]
FILES.extend(sorted((BASE / "figures").glob("*")))
for label in ("2k", "10k"):
    directory = BASE / label
    FILES.extend(directory / name for name in (
        "schedule.json", "preflight.json", "episode_manifest.csv", "endpoint_metrics.csv",
        "resource_usage.csv", "result.json",
    ))
    FILES.extend(sorted((directory / "weather_daily").glob("*.csv")))
    FILES.extend(sorted((directory / "runtime_evidence").glob("*.json")))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def rows() -> list[dict]:
    return [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)} for p in FILES]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    verify = parser.parse_args().verify
    actual = rows()
    if verify:
        with OUT.open(encoding="utf-8", newline="") as stream:
            expected = list(csv.DictReader(stream))
        ok = len(actual) == len(expected) and all(a["path"] == e["path"] and str(a["bytes"]) == e["bytes"] and a["sha256"] == e["sha256"] for a, e in zip(actual, expected))
        print(f"archive_files={len(actual)} failed={0 if ok else 1}")
        if not ok:
            raise SystemExit(2)
    else:
        mode = "w" if parser.parse_args().rebuild else "x"
        with OUT.open(mode, encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["path", "bytes", "sha256"])
            writer.writeheader()
            writer.writerows(actual)
        print(f"archive_files={len(actual)} bytes={sum(x['bytes'] for x in actual)}")


if __name__ == "__main__":
    main()
