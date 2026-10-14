"""Verify and inventory the compact HLA 053 eight-seed GitHub backup."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
FIG = BASE / "055_03_five_scenario/HL"
MANIFEST = BASE / "github_sha256_manifest.csv"
GATE = BASE / "final_gate.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def rel(path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"Outside repository: {path}")
    return resolved.relative_to(ROOT).as_posix()


def main() -> None:
    if MANIFEST.exists() or GATE.exists():
        raise FileExistsError("Backup inventory already exists; preserve the frozen version")
    entries: dict[str, tuple[str, int]] = {}

    def add(path: Path) -> str:
        if not path.is_file():
            raise FileNotFoundError(path)
        key = rel(path)
        value = (digest(path), path.stat().st_size)
        if key in entries and entries[key] != value:
            raise ValueError(f"File changed during inventory: {path}")
        entries[key] = value
        return value[0]

    fixed = [
        ROOT / ".gitattributes",
        ROOT / "prompt_02/053_hla_wgen_8seed_resume.md",
        ROOT / "docs/hla_wgen_training_053_record.md",
        ROOT / "docs/hla_wgen_limited_candidates_053_record.md",
        ROOT / "docs/hla_wgen_100k_8seed_freeze_2026-10-09.md",
        ROOT / "src/hl_fq_8seed_cross_site/analyze_hl_fq_8seed.py",
        ROOT / "src/hl_fq_8seed_cross_site/plot_hl_fq_five_scenario_05503_style.py",
    ]
    for path in fixed:
        add(path)
    climate = ROOT / "results/hla_weather_enhancement_029"
    for path in sorted(climate.glob("*.py")):
        add(path)
    for path in sorted((climate / "weather_fitting/gpcc_raw").iterdir()):
        if path.is_file() and path.suffix.lower() in {".py", ".csv", ".json", ".cli"}:
            add(path)
    for path in sorted(BASE.iterdir()):
        if path.is_file() and path.suffix.lower() in {".py", ".ps1", ".json", ".jsonl", ".log"}:
            add(path)
    pool = BASE / "weather_pool"
    for path in sorted(pool.glob("*")):
        if path.is_file() and path.suffix.lower() in {".csv", ".json"}:
            add(path)
    for split in ("train", "heldout"):
        folder = pool / split
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in {".csv", ".json"}:
                add(path)
        for path in sorted((folder / "weather_daily").glob("*.csv")):
            add(path)

    weather_count = 0
    for seed in range(8):
        run = BASE / f"seed_{seed:02d}/attempt_01"
        result = json.loads((run / "run_result.json").read_text(encoding="utf-8"))
        gate = json.loads((run / "audit_gate.json").read_text(encoding="utf-8"))
        if result.get("status") != "PASS_100K_ARCHIVE_ONLY" or gate.get("status") != "PASS_100K_ARCHIVE_ONLY":
            raise ValueError(f"Seed {seed} training/audit gate failed")
        if result.get("actual_timesteps") != 100080:
            raise ValueError(f"Seed {seed} step count changed")
        for name in (
            "audit_gate.json", "effective_config.json", "episode_manifest.csv",
            "preflight.json", "resource_usage.csv", "run_result.json",
            "source_manifest.json", "training_episode_summary.csv",
        ):
            add(run / name)
        for step in (25000, 50000, 75000, 100000):
            path = run / "models" / f"checkpoint_{step}.zip"
            if add(path) != gate["checkpoint_files"][str(step)]["sha256"]:
                raise ValueError(f"Seed {seed} checkpoint hash mismatch: {step}")
        if add(run / "models/final_model_actual_100080.zip") != result["model_sha256"]:
            raise ValueError(f"Seed {seed} final model hash mismatch")
        with (run / "episode_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != result["episode_archives"]:
            raise ValueError(f"Seed {seed} episode count mismatch")
        for row in rows:
            weather = ROOT / row["weather_path"]
            if add(weather) != row["weather_sha256"].upper():
                raise ValueError(f"Seed {seed} weather hash mismatch: {weather}")
            evidence = ROOT / row["runtime_evidence_path"]
            add(evidence)
            weather_count += 1
        for year in range(2014, 2024):
            snapshot = BASE / f"validation/seed_{seed:02d}/snapshots/HLA/{year}/ppo_seed_{seed}"
            add(snapshot / "Summary.OUT")
            metadata = json.loads((snapshot / "evaluation_metadata.json").read_text(encoding="utf-8"))
            if metadata.get("seed") != seed or metadata.get("year") != year:
                raise ValueError(f"Seed {seed} validation metadata mismatch: {year}")
            add(snapshot / "evaluation_metadata.json")
        seed_dir = FIG / f"best_seed_seed{seed}"
        add(seed_dir / "README.md")
        figures = list((seed_dir / "figures").glob("*.png"))
        if len(figures) != 23:
            raise ValueError(f"Seed {seed} figure count: {len(figures)}")
        for folder in (seed_dir / "figures", seed_dir / "tables"):
            for path in sorted(folder.iterdir()):
                if path.is_file() and path.suffix.lower() in {".png", ".csv", ".json"}:
                    add(path)
        metrics = seed_dir / "tables" / f"hl_seed{seed}_yearly_five_scenario_metrics.csv"
        with metrics.open(encoding="utf-8-sig", newline="") as stream:
            if sum(1 for _ in csv.DictReader(stream)) != 50:
                raise ValueError(f"Seed {seed} five-scenario rows != 50")

    cohort = FIG / "cohort_8seed_summary"
    with (cohort / "all_seed_year_scenario_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        if sum(1 for _ in csv.DictReader(stream)) != 400:
            raise ValueError("Cohort table rows != 400")
    for path in sorted(cohort.iterdir()):
        if path.is_file():
            add(path)
    if weather_count != 5720:
        raise ValueError(f"Unexpected archived training weather count: {weather_count}")

    record = {
        "status": "PASS_8SEED_TRAINING_EVALUATION_BACKUP_INVENTORY",
        "training_seeds": list(range(8)),
        "validation_years": list(range(2014, 2024)),
        "actual_training_weather_csv_count": weather_count,
        "single_seed_figure_count": 8 * 23,
        "validation_summary_out_count": 80,
        "five_scenario_rows": 400,
        "inventory_file_count_before_manifest": len(entries),
        "inventory_bytes_before_manifest": sum(size for _, size in entries.values()),
        "cohort_table_sha256": add(cohort / "cohort_overall_mean_sd.csv"),
    }
    GATE.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    add(GATE)
    with MANIFEST.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["relative_path", "sha256", "bytes"])
        for path, (sha, size) in sorted(entries.items()):
            writer.writerow([path, sha, size])
    print(json.dumps({"status": record["status"], "files": len(entries), "weather_csv": weather_count,
                      "inventory_mb": round(sum(size for _, size in entries.values()) / 2**20, 2),
                      "manifest": rel(MANIFEST)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
