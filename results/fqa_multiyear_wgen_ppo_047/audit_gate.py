"""Reread the FQA 047 multi-year PPO 10K weather, runtime, and config gates."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent / "attempt_01"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    result = json.loads((RUN / "run_result.json").read_text(encoding="utf-8"))
    plan = json.loads((RUN / "planned_episode_schedule.json").read_text(encoding="utf-8"))
    frozen = json.loads((ROOT / "results/fqa_wgen_multiyear_heldout_gate_045/full_schedule_plan.json").read_text(encoding="utf-8"))
    rows = read_csv(RUN / "episode_manifest.csv")
    cfg = json.loads((RUN / "effective_config.json").read_text(encoding="utf-8"))
    cfg044 = json.loads((ROOT / "results/fqa_wgen_ppo_smoke_044/attempt_05/effective_config.json").read_text(encoding="utf-8"))
    checks = {}
    checks["run_status"] = result["status"] == "PASS_10K_ARCHIVE_ONLY"
    checks["frozen_ppo_reward_action_safety"] = all(cfg[k] == cfg044[k] for k in ("ppo", "reward", "discrete_actions", "action_safety"))
    checks["first_80_schedule_frozen"] = plan[:80] == frozen["train"]
    checks["all_years_and_train_pool_only"] = {int(r["year"]) for r in rows} == set(range(2005, 2014)) and all(1001 <= int(r["weather_seed"]) <= 1080 for r in rows)
    checks["manifest_schedule_closure"] = [(int(r["episode_index"]), int(r["year"]), int(r["weather_seed"])) for r in rows] == [(x["episode_index"], x["historical_year"], x["weather_seed"]) for x in plan[:len(rows)]]
    checks["episode_status"] = len(rows) == 97 and sum(r["status"] == "completed" for r in rows) == 96 and rows[-1]["status"] == "partial_at_stop"
    checks["step_weather_closure"] = sum(int(r["days"]) for r in rows) == result["actual_timesteps"] == 10080
    checks["all_realization_hashes_distinct"] = len({r["weather_sha256"] for r in rows}) == len(rows)
    weather_ok = True
    runtime_ok = True
    for row in rows:
        weather = ROOT / row["weather_path"]
        if not weather.is_file() or sha(weather) != row["weather_sha256"]:
            weather_ok = False
            continue
        daily = read_csv(weather)
        if len(daily) != int(row["days"]) or int(row["days"]) != int(row["weather_rows"]) or list(daily[0]) != ["DATE", "DOY", "RAIN", "SRAD", "TMAX", "TMIN"]:
            weather_ok = False
        dates = [date.fromisoformat(x["DATE"]) for x in daily]
        if any(d.year != int(row["year"]) for d in dates) or any(b-a != timedelta(days=1) for a,b in zip(dates,dates[1:])):
            weather_ok = False
        for day in daily:
            rain, srad, tmax, tmin = (float(day[k]) for k in ("RAIN", "SRAD", "TMAX", "TMIN"))
            if rain < 0 or srad < 0 or tmax < tmin:
                weather_ok = False
        if row["physical_status"] != "PASS":
            weather_ok = False
        proof = json.loads((ROOT / row["runtime_evidence_path"]).read_text(encoding="utf-8"))
        if not (proof["wther"] == "W" and proof["wsta_confirmed"] and proof["yaml_bootstrap_confirmed"] and proof["runtime_rseed1"] == int(row["weather_seed"]) and proof["runtime_cli_sha256"] == proof["source_cli_sha256"]):
            runtime_ok = False
    checks["all_weather_files_hash_rows_dates_physical"] = weather_ok
    checks["all_runtime_evidence"] = runtime_ok
    resources = read_csv(RUN / "resource_usage.csv")
    peak = max(float(r["process_tree_rss_mb"]) for r in resources)
    checks["rss_below_1536_mb"] = peak < 1536
    model = ROOT / result["model_path"]
    checks["model_hash_and_checkpoint_files"] = model.is_file() and sha(model) == result["model_sha256"] and all((RUN / "models" / f"checkpoint_{x}.zip").is_file() for x in (5000, 10000))
    checks["source_046_qc_pass"] = json.loads((ROOT / "results/fqa_wgen_full_pool_qc_046/final_gate.json").read_text(encoding="utf-8"))["status"] == "PASS_INPUT_QC_ONLY"
    output = {"status": "PASS_10K_ARCHIVE_ONLY" if all(checks.values()) else "FAIL", "checks": checks, "requested_steps": 10000, "actual_steps": 10080, "completed_episodes": 96, "partial_episodes": 1, "archived_weather_days": sum(int(r["days"]) for r in rows), "unique_weather_hashes": len({r["weather_sha256"] for r in rows}), "peak_process_tree_rss_mb": round(peak, 2), "elapsed_seconds": result["elapsed_seconds"], "limits": ["single PPO seed 0 and 10K steps only", "no heldout policy evaluation or crop outcome comparison", "not a 100K formal training result", "actual formal training weather must be archived again", "native FIELD coordinate warning remains unresolved"]}
    path = RUN / "final_gate.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "checks_passed": sum(checks.values()), "checks_total": len(checks), "actual_steps": output["actual_steps"], "archives": len(rows), "peak_rss_mb": output["peak_process_tree_rss_mb"]}, ensure_ascii=False))
    if output["status"] != "PASS_10K_ARCHIVE_ONLY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
