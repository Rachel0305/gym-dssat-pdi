"""Audit one seed's HLA WGEN 100K training archive or smoke attempt."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
PROMPT = ROOT / "prompt_02/053_hla_wgen_8seed_resume.md"
SCRIPT = BASE / "run_seed_100k.py"
SOURCE_045 = ROOT / "results/hla_wgen_8seed_053/run_weather_gate.py"
PLAN_045 = ROOT / "results/hla_wgen_8seed_053/weather_pool_plan.json"
GATE_046 = ROOT / "results/hla_wgen_8seed_053/weather_pool/final_gate.json"
GATE_047 = ROOT / "results/hla_wgen_8seed_053/smoke_gpcc_raw_432_attempt02/run_result.json"
GATE_050 = ROOT / "results/hla_wgen_8seed_053/scenario_selection.json"
FROZEN_CONFIG = ROOT / "results/hla_wgen_8seed_053/smoke_gpcc_raw_432_attempt02/effective_config.json"
CHECKPOINTS = [25_000, 50_000, 75_000, 100_000]
RSS_LIMIT_MB = 1536
WALL_LIMIT_S = 7200


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def inside_root(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if path != ROOT.resolve() and ROOT.resolve() not in path.parents:
        raise ValueError(f"Path escapes workspace: {relative}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=range(8), required=True)
    parser.add_argument("--run", choices=("smoke", "formal"), required=True)
    args = parser.parse_args()
    expected_smoke = args.run == "smoke"
    run_name = "smoke_attempt_01" if expected_smoke else "attempt_01"
    out = BASE / f"seed_{args.seed:02d}" / run_name
    run = json.loads((out / "run_result.json").read_text(encoding="utf-8"))
    preflight = json.loads((out / "preflight.json").read_text(encoding="utf-8"))
    config = json.loads((out / "effective_config.json").read_text(encoding="utf-8"))
    schedule = json.loads((out / "planned_episode_schedule.json").read_text(encoding="utf-8"))
    manifest = read_csv(out / "episode_manifest.csv")
    resources = read_csv(out / "resource_usage.csv")
    expected_steps = 432 if expected_smoke else 100_000
    expected_status = "PASS_432_STEP_SMOKE_ARCHIVE_ONLY" if expected_smoke else "PASS_100K_ARCHIVE_ONLY"
    expected_checkpoints = [432] if expected_smoke else CHECKPOINTS

    checks: dict[str, bool] = {}
    source_manifest = json.loads((out / "source_manifest.json").read_text(encoding="utf-8"))
    checks["source_hashes"] = (
        source_manifest[PROMPT.relative_to(ROOT).as_posix()] == sha(PROMPT)
        and source_manifest[SCRIPT.relative_to(ROOT).as_posix()] == sha(SCRIPT)
        and source_manifest[SOURCE_045.relative_to(ROOT).as_posix()] == sha(SOURCE_045)
        and source_manifest[PLAN_045.relative_to(ROOT).as_posix()] == sha(PLAN_045)
        and source_manifest[GATE_046.relative_to(ROOT).as_posix()] == sha(GATE_046)
        and source_manifest[GATE_047.relative_to(ROOT).as_posix()] == sha(GATE_047)
        and source_manifest[GATE_050.relative_to(ROOT).as_posix()] == sha(GATE_050)
    )
    checks["frozen_contract"] = all(
        config[key] == json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))[key]
        for key in ("ppo", "reward", "discrete_actions", "action_safety")
    ) and int(config["seed"]) == args.seed
    checks["budget_and_status"] = (
        run["status"] == expected_status
        and int(run["ppo_seed"]) == args.seed
        and int(run["requested_timesteps"]) == expected_steps
        and int(run["actual_timesteps"]) >= expected_steps
    )
    checks["heldout_excluded_and_train_seeds"] = (
        run["training_seed_pool"] == [1001, 1080]
        and run["heldout_seed_pool_excluded"] == [1081, 1100]
        and all(1001 <= int(row["weather_seed"]) <= 1080 and row["pool"] == "train" for row in schedule)
    )
    frozen_train = json.loads(PLAN_045.read_text(encoding="utf-8"))["train"]
    checks["first_80_schedule_matches_045"] = schedule[:80] == frozen_train
    checks["run_schedule_and_episode_closure"] = (
        len(schedule) >= len(manifest)
        and int(run["episode_archives"]) == len(manifest)
        and int(run["archived_weather_days"]) == int(run["actual_timesteps"])
        and int(run["actual_timesteps"]) == sum(int(row["days"]) for row in manifest)
        and all(int(row["episode_index"]) == index for index, row in enumerate(manifest, 1))
    )
    weather_ok = True
    runtime_ok = True
    for row in manifest:
        seed = int(row["weather_seed"])
        sched = schedule[int(row["episode_index"]) - 1]
        weather_path = inside_root(row["weather_path"])
        weather = read_csv(weather_path)
        days = [date.fromisoformat(item["DATE"]) for item in weather]
        weather_ok &= (
            (row["physical_status"] == "PASS" or (row["status"].startswith("partial_") and row["physical_status"] == "FAIL_OR_INCOMPLETE"))
            and row["status"] in ("completed", "partial_at_stop", "partial_after_failure")
            and int(row["year"]) == int(sched["historical_year"])
            and seed == int(sched["weather_seed"]) == int(row["pdi_rseed1"])
            and len(weather) == int(row["days"]) == int(row["weather_rows"])
            and sha(weather_path) == row["weather_sha256"]
            and len(days) > 0
            and all(days[i] == days[i - 1] + timedelta(days=1) for i in range(1, len(days)))
            and list(weather[0]) == ["DATE", "DOY", "RAIN", "SRAD", "TMAX", "TMIN"]
            and all(float(x["RAIN"]) >= 0 and float(x["SRAD"]) > 0 and float(x["TMAX"]) >= float(x["TMIN"]) for x in weather)
        )
        proof = json.loads(inside_root(row["runtime_evidence_path"]).read_text(encoding="utf-8"))
        runtime_ok &= (
            proof["wther"] == "W"
            and proof["wsta_confirmed"]
            and proof["yaml_bootstrap_confirmed"]
            and proof["runtime_cli_sha256"] == proof["source_cli_sha256"]
            and int(proof["runtime_rseed1"]) == int(proof["scheduled_seed"]) == seed
        )
    checks["daily_weather_hash_dates_and_physical"] = bool(weather_ok)
    checks["runtime_wgen_cli_and_seed"] = bool(runtime_ok)
    checks["resources_within_limits"] = (
        bool(resources)
        and max(float(row["process_tree_rss_mb"]) for row in resources) < RSS_LIMIT_MB
        and float(run["peak_process_tree_rss_mb"]) < RSS_LIMIT_MB
        and float(run["elapsed_seconds"]) < WALL_LIMIT_S
    )

    checkpoint_steps = [int(x) for x in expected_checkpoints]
    checkpoint_actual = {str(k): int(v) for k, v in run["checkpoint_steps_actual"].items()}
    checkpoint_records = {}
    for step in checkpoint_steps:
        path = out / "models" / f"checkpoint_{step}.zip"
        checkpoint_records[str(step)] = {"exists": path.is_file(), "sha256": sha(path) if path.is_file() else ""}
    checks["all_checkpoint_files_and_steps"] = (
        [int(x) for x in run["checkpoint_steps_requested"]] == checkpoint_steps
        and all(checkpoint_records[str(step)]["exists"] for step in checkpoint_steps)
        and all(checkpoint_actual.get(str(step)) == step for step in checkpoint_steps)
    )
    final_model = inside_root(run["model_path"])
    checks["final_model_hash"] = final_model.is_file() and sha(final_model) == run["model_sha256"]
    training_summary_path = out / "training_episode_summary.csv"
    training_summary = read_csv(training_summary_path) if training_summary_path.is_file() else []
    checks["completed_episode_summary_closure"] = len(training_summary) == sum(row["status"] == "completed" for row in manifest)

    status = "PASS_432_STEP_SMOKE_ARCHIVE_ONLY" if expected_smoke else "PASS_100K_ARCHIVE_ONLY"
    if not all(checks.values()):
        status = "FAIL"
    report = {
        "status": status,
        "mode": "smoke" if expected_smoke else "formal_100k",
        "ppo_seed": args.seed,
        "checks": checks,
        "requested_timesteps": run["requested_timesteps"],
        "actual_timesteps": run["actual_timesteps"],
        "episode_archives": len(manifest),
        "completed_episodes": sum(row["status"] == "completed" for row in manifest),
        "partial_episodes": sum(row["status"] != "completed" for row in manifest),
        "archived_weather_days": sum(int(row["days"]) for row in manifest),
        "unique_weather_hashes": len({row["weather_sha256"] for row in manifest}),
        "checkpoint_files": checkpoint_records,
        "peak_process_tree_rss_mb": run["peak_process_tree_rss_mb"],
        "elapsed_seconds": run["elapsed_seconds"],
    }
    report_path = out / "audit_gate.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if status == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
