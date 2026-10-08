"""Build a SHA-256 inventory for the FQA WGEN 047 training archive."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/fqa_multiyear_wgen_ppo_047"
OUTPUT = ROOT / "docs/fqa_wgen_047_file_manifest.csv"
MODEL = "results/fqa_multiyear_wgen_ppo_047/attempt_01/models/fqa_multiyear_wgen_ppo_seed0_10k.zip"
EXCLUDED_DIRS = {"temp", "tmp", "__pycache__", "rendered_inputs", "runtime_templates", "tensorboard"}


def main():
    selected = [
        ROOT / "prompt_02/047_fqa_multiyear_wgen_ppo_10k_archive_gate.md",
        ROOT / "docs/fqa_wgen_multiyear_ppo_047_10k_record.md",
        ROOT / "docs/fqa_wgen_archive_latest.md",
        ROOT / "scripts/build_fqa_wgen_047_archive_manifest.py",
        ROOT / "scripts/verify_fqa_wgen_047_archive.py",
    ]
    for path in sorted(RESULT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if any(part in EXCLUDED_DIRS for part in path.relative_to(ROOT).parts):
            continue
        if "models" in path.parts and rel != MODEL:
            continue
        if path.suffix.lower() in {".pyc", ".wth", ".sol", ".out"}:
            continue
        if path.suffix.lower() == ".zip" and rel != MODEL:
            continue
        if path.stat().st_size > 1_000_000:
            raise RuntimeError(f"Archive file exceeds 1 MB review cap: {rel}")
        selected.append(path)
    if len(selected) != len(set(selected)) or any(not p.is_file() for p in selected):
        raise RuntimeError("Missing or duplicate archive file")
    rows = []
    for path in sorted(selected, key=lambda p: p.relative_to(ROOT).as_posix()):
        data = path.read_bytes()
        rows.append({"path": path.relative_to(ROOT).as_posix(), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest().upper()})
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    with OUTPUT.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"archive_files={len(rows)} total_bytes={sum(int(r['bytes']) for r in rows)}")


if __name__ == "__main__":
    main()
