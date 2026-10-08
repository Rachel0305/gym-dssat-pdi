"""Create a GitHub-oriented SHA-256 manifest for the FQ eight-seed package.

Model/checkpoint archives and DSSAT/SB3 runtime caches are excluded. Per-episode
weather hashes are carried forward from the training manifests after the archive
audit verified the saved daily files against those hashes.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
FIGURES = BASE / "055_03_five_scenario/FQ"
OUTPUT = BASE / "github_sha256_manifest.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> None:
    entries: dict[str, tuple[str, str]] = {}

    def add_file(path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        entries[rel(path)] = (sha256(path), "recomputed_file_sha256")

    fixed = [
        ROOT / "prompt_02/052_fqa_wgen_ppo_100k_8seed_validation_figures.md",
        ROOT / "docs/fqa_multiyear_wgen_ppo_052_8seed_record.md",
        BASE / "final_gate.json",
        BASE / "seed_00_reuse.json",
        BASE / "run_seed_100k.py",
        BASE / "audit_seed.py",
        BASE / "evaluate_100k.py",
        BASE / "summarize_cohort.py",
        BASE / "run_seeds_02_07.ps1",
        BASE / "generate_sha256_manifest.py",
    ]
    for path in fixed:
        add_file(path)

    training_names = [
        "audit_gate.json", "effective_config.json", "episode_manifest.csv",
        "planned_episode_schedule.json", "preflight.json", "resource_usage.csv",
        "run_result.json", "training_episode_summary.csv",
    ]
    for seed in range(8):
        run = (ROOT / "results/fqa_multiyear_wgen_ppo_051/attempt_01" if seed == 0
               else BASE / f"seed_{seed:02d}/attempt_01")
        for name in training_names:
            path = run / name
            if path.is_file():
                add_file(path)
        manifest = pd.read_csv(run / "episode_manifest.csv", keep_default_na=False)
        for row in manifest.itertuples(index=False):
            weather_path = str(row.weather_path)
            weather_hash = str(row.weather_sha256).upper()
            if len(weather_hash) != 64:
                raise ValueError(f"Invalid weather SHA-256 in {run / 'episode_manifest.csv'}")
            entries[weather_path] = (weather_hash, "verified_episode_manifest_sha256")

        seed_dir = FIGURES / f"best_seed_seed{seed}"
        add_file(seed_dir / "README.md")
        for folder in (seed_dir / "figures", seed_dir / "tables"):
            for path in sorted(folder.iterdir()):
                if path.is_file() and path.suffix.lower() in {".png", ".csv", ".json"}:
                    add_file(path)

    cohort = FIGURES / "cohort_8seed_summary"
    for path in sorted(cohort.iterdir()):
        if path.is_file() and path.name != OUTPUT.name:
            add_file(path)

    for pattern in ("seed_*_formal*.log", "seed_*_eval*.log", "cohort_summary*.log"):
        for path in sorted(BASE.glob(pattern)):
            add_file(path)

    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["relative_path", "sha256", "hash_method"])
        for path, (digest, method) in sorted(entries.items()):
            writer.writerow([path, digest, method])
    print(f"manifest={rel(OUTPUT)} entries={len(entries)} weather_entries={sum(method == 'verified_episode_manifest_sha256' for _, method in entries.values())}")


if __name__ == "__main__":
    main()
