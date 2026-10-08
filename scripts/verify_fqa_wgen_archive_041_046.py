"""Verify the curated FQA 041-046 GitHub archive against its SHA-256 manifest."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/fqa_wgen_041_046_file_manifest.csv"


def main() -> None:
    with MANIFEST.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({row["path"] for row in rows}) != len(rows):
        raise RuntimeError("Archive manifest is empty or contains duplicate paths")
    failures = []
    for row in rows:
        path = (ROOT / row["path"]).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
            failures.append((row["path"], "missing_or_outside_project"))
            continue
        data = path.read_bytes()
        if len(data) != int(row["bytes"]) or hashlib.sha256(data).hexdigest().upper() != row["sha256"]:
            failures.append((row["path"], "size_or_sha256_mismatch"))
    print(f"archive_files={len(rows)} failed={len(failures)}")
    for path, reason in failures[:20]:
        print(f"{reason}: {path}")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
