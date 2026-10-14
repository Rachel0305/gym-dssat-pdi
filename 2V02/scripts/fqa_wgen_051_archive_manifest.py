"""Build the reproducibility manifest for FQA WGEN run 051.

The CSV lists every archived prompt, script, smoke artifact, and formal-run
file. DSSAT/SB3 temporary cache files under an attempt's ``temp`` directory
are deliberately excluded from the GitHub archive and manifest.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/fqa_multiyear_wgen_ppo_051"
OUTPUT = ROOT / "docs/fqa_wgen_051_file_manifest.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> None:
    explicit = [
        ROOT / ".gitattributes",
        ROOT / "prompt_02/051_fqa_multiyear_wgen_ppo_100k_with_checkpoints.md",
        BASE / "run_100k.py",
        BASE / "audit_100k.py",
        BASE / "smoke_console.log",
        BASE / "formal_console.log",
        ROOT / "docs/fqa_multiyear_wgen_ppo_051_100k_record.md",
        ROOT / "docs/fqa_wgen_archive_latest.md",
        Path(__file__).resolve(),
    ]
    files = {path.resolve() for path in explicit}
    for attempt_name in ("smoke_attempt_01", "attempt_01"):
        attempt = BASE / attempt_name
        for path in attempt.rglob("*"):
            if path.is_file() and "temp" not in path.relative_to(attempt).parts:
                files.add(path.resolve())

    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing manifest inputs: " + ", ".join(map(str, missing)))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("path", "bytes", "sha256"))
        for path in sorted(files, key=lambda item: item.relative_to(ROOT).as_posix()):
            writer.writerow((
                path.relative_to(ROOT).as_posix(),
                path.stat().st_size,
                sha256(path),
            ))
    print(f"wrote {len(files)} file hashes to {OUTPUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
