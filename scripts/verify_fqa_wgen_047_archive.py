"""Verify the exact bytes of the curated FQA WGEN 047 GitHub archive."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/fqa_wgen_047_file_manifest.csv"


def main():
    with MANIFEST.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or len({r["path"] for r in rows}) != len(rows):
        raise RuntimeError("Missing or duplicate manifest rows")
    bad = []
    for row in rows:
        path = (ROOT / row["path"]).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
            bad.append((row["path"], "missing_or_outside_project"))
            continue
        data = path.read_bytes()
        if len(data) != int(row["bytes"]) or hashlib.sha256(data).hexdigest().upper() != row["sha256"]:
            bad.append((row["path"], "size_or_sha256_mismatch"))
    print(f"archive_files={len(rows)} failed={len(bad)}")
    for path, reason in bad[:20]:
        print(reason, path)
    if bad:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
