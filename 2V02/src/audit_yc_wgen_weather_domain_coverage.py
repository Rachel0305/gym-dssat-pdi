from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "results/yc_random_weather_ppo/004_05"
OUTPUT = ROOT / "results/yc_random_weather_ppo/weather_domain_coverage_audit"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    schedule_path = EXPERIMENT / "config/training_weather_schedule_random.csv"
    schedule = read_csv(schedule_path)
    schedule_counts = Counter(int(row["weather_seed"]) for row in schedule)
    schedule_rows = [
        {
            "weather_seed": seed,
            "schedule_occurrences": schedule_counts.get(seed, 0),
            "in_expected_training_pool": 1001 <= seed <= 1080,
        }
        for seed in range(1001, 1081)
    ]

    training_seed_rows: list[dict[str, object]] = []
    reference_sequence: list[tuple[str, str, str]] | None = None
    sequence_matches: dict[str, bool | None] = {}
    for seed in range(8):
        episode_path = EXPERIMENT / f"training/random_weather/ppo_seed_{seed}/training_episode_summary.csv"
        if not episode_path.exists():
            training_seed_rows.append({
                "ppo_seed": seed,
                "episode_log_status": "UNAVAILABLE",
                "completed_episodes": 0,
                "unique_wgen_seeds": 0,
                "min_episodes_per_wgen_seed": "",
                "max_episodes_per_wgen_seed": "",
                "unique_runtime_weather_hashes": 0,
                "sequence_matches_first_available_seed": "",
            })
            sequence_matches[str(seed)] = None
            continue

        episodes = read_csv(episode_path)
        seed_counts = Counter(int(row["training_weather_seed"]) for row in episodes if row["training_weather_seed"])
        sequence = [
            (row["historical_year"], row["training_weather_seed"], row["runtime_weather_sha256"])
            for row in episodes
        ]
        if reference_sequence is None:
            reference_sequence = sequence
            matches: bool | None = True
        else:
            matches = sequence == reference_sequence
        sequence_matches[str(seed)] = matches
        training_seed_rows.append({
            "ppo_seed": seed,
            "episode_log_status": "AVAILABLE",
            "completed_episodes": len(episodes),
            "unique_wgen_seeds": len(seed_counts),
            "min_episodes_per_wgen_seed": min(seed_counts.values()) if seed_counts else "",
            "max_episodes_per_wgen_seed": max(seed_counts.values()) if seed_counts else "",
            "unique_runtime_weather_hashes": len({row["runtime_weather_sha256"] for row in episodes if row["runtime_weather_sha256"]}),
            "sequence_matches_first_available_seed": matches,
        })

    seed5_train = EXPERIMENT / "training/random_weather/ppo_seed_5"
    seed5_episode_path = seed5_train / "training_episode_summary.csv"
    seed5_episodes = read_csv(seed5_episode_path) if seed5_episode_path.exists() else []
    train_wth = list(seed5_train.rglob("*.WTH"))
    observed_dir = EXPERIMENT / "evaluation/runs/observed_weather/random_weather/ppo_seed_5"
    observed_manifest = json.loads((observed_dir / "evaluation_manifest.json").read_text(encoding="utf-8"))
    observed_wth = list(observed_dir.rglob("*.WTH"))
    heldout_dir = EXPERIMENT / "evaluation/runs/heldout_wgen/random_weather/ppo_seed_5"
    heldout_manifest = json.loads((heldout_dir / "evaluation_manifest.json").read_text(encoding="utf-8"))
    heldout_episodes = read_csv(heldout_dir / "evaluation_episode_level.csv")
    heldout_wth = list(heldout_dir.rglob("*.WTH"))
    heldout_hashes = {row["runtime_weather_sha256"].upper() for row in heldout_episodes if row["runtime_weather_sha256"]}
    heldout_snapshot_matches = sum(sha256(path) in heldout_hashes for path in heldout_wth)

    qc_path = ROOT / "results/yc_wgen_cli_pilot/003_06_06/weather_qc/weather_qc_summary.json"
    qc = json.loads(qc_path.read_text(encoding="utf-8")) if qc_path.exists() else {}
    qc_scope = str(qc.get("scope", ""))

    completeness_rows = [
        {
            "source": "004_05 random WGEN training pool daily weather",
            "expected_count": 80,
            "available_daily_wth_count": len(train_wth),
            "status": "PARTIAL" if len(train_wth) < 80 else "AVAILABLE",
            "note": "Only context-year snapshots are retained under PPO seed 5; per-episode logs retain hashes, not daily values.",
        },
        {
            "source": "004_05 held-out WGEN daily weather",
            "expected_count": len(heldout_manifest.get("evaluation_weather_seeds", [])),
            "available_daily_wth_count": len(heldout_wth),
            "status": "PARTIAL" if len(heldout_wth) < len(heldout_manifest.get("evaluation_weather_seeds", [])) else "AVAILABLE",
            "note": f"Unique runtime weather hashes={len(heldout_hashes)}; retained WTH snapshots matching those hashes={heldout_snapshot_matches}.",
        },
        {
            "source": "004_05 observed weather daily WTH",
            "expected_count": len(observed_manifest.get("evaluation_years", [])),
            "available_daily_wth_count": len(observed_wth),
            "status": "AVAILABLE" if len(observed_wth) == len(observed_manifest.get("evaluation_years", [])) else "PARTIAL",
            "note": "Observed evaluation WTH files exist for seed 5; complete comparison still requires the full training WGEN daily series.",
        },
        {
            "source": "003_06_06 WGEN weather QC seeds",
            "expected_count": 80,
            "available_daily_wth_count": 0,
            "status": "NOT_COMPARABLE",
            "note": f"QC summary covers {qc.get('unique_seed_sequences', 0)} separate pilot seeds but retains no daily WTH files; scope is {qc_scope}.",
        },
    ]

    summary = {
        "audit_status": "INSUFFICIENT_FOR_FULL_WEATHER_DOMAIN_COVERAGE",
        "scope": "read-only inventory of existing 004_05 weather schedules, episode logs, WTH snapshots, manifests, and prior WGEN QC",
        "training_schedule": {
            "rows": len(schedule),
            "unique_weather_seeds": len(schedule_counts),
            "seed_min": min(schedule_counts) if schedule_counts else None,
            "seed_max": max(schedule_counts) if schedule_counts else None,
            "occurrences_per_seed_min": min(schedule_counts.values()) if schedule_counts else None,
            "occurrences_per_seed_max": max(schedule_counts.values()) if schedule_counts else None,
            "schedule_sha256": sha256(schedule_path),
        },
        "seed5_training_log": {
            "episodes": len(seed5_episodes),
            "unique_weather_seeds_seen": len({row["training_weather_seed"] for row in seed5_episodes}),
            "unique_runtime_weather_hashes": len({row["runtime_weather_sha256"] for row in seed5_episodes if row["runtime_weather_sha256"]}),
            "retained_daily_wth_files": len(train_wth),
            "daily_weather_pool_coverage_established": False,
        },
        "available_training_seed_logs": training_seed_rows,
        "training_weather_sequence_matches": sequence_matches,
        "heldout_wgen": {
            "manifest_seed_count": len(heldout_manifest.get("evaluation_weather_seeds", [])),
            "episode_rows": len(heldout_episodes),
            "unique_runtime_weather_hashes": len(heldout_hashes),
            "retained_daily_wth_files": len(heldout_wth),
            "retained_wth_hash_matches": heldout_snapshot_matches,
        },
        "observed_weather": {
            "manifest_years": len(observed_manifest.get("evaluation_years", [])),
            "retained_daily_wth_files": len(observed_wth),
        },
        "prior_wgen_qc": {
            "status": qc.get("status"),
            "unique_seed_sequences": qc.get("unique_seed_sequences"),
            "scope": qc_scope,
            "directly_covers_004_05_pool": False,
        },
        "conclusion": "Seed IDs, use frequency, and hashes establish schedule coverage and reproducibility, but not rainfall/temperature/radiation tails or stage-specific climate support.",
        "next_evidence_needed": "Daily weather series for the already-used 004_05 WGEN seeds 1001-1100, generated or recovered without running PPO or crop evaluation; preserve frozen CLI/parameter hashes and do not use held-out seeds for fitting.",
    }

    write_csv(OUTPUT / "training_weather_seed_coverage.csv", schedule_rows)
    write_csv(OUTPUT / "training_seed_log_inventory.csv", training_seed_rows)
    write_csv(OUTPUT / "weather_artifact_completeness.csv", completeness_rows)
    (OUTPUT / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "audit_status": summary["audit_status"],
        "schedule_rows": len(schedule),
        "schedule_unique_seeds": len(schedule_counts),
        "seed5_completed_episodes": len(seed5_episodes),
        "seed5_unique_wgen_seeds_seen": summary["seed5_training_log"]["unique_weather_seeds_seen"],
        "seed5_retained_training_wth": len(train_wth),
        "heldout_manifest_seeds": len(heldout_manifest.get("evaluation_weather_seeds", [])),
        "heldout_retained_wth": len(heldout_wth),
        "observed_retained_wth": len(observed_wth),
        "output_dir": str(OUTPUT),
    }, indent=2))


if __name__ == "__main__":
    main()
