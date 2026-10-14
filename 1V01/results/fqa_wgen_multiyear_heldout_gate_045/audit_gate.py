"""Reread FQA 045 train/heldout archive evidence and issue a bounded gate."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    plan = json.loads((OUT / "full_schedule_plan.json").read_text(encoding="utf-8"))
    train = load_csv(OUT / "train/episode_manifest.csv")
    heldout = load_csv(OUT / "heldout/episode_manifest.csv")
    rows = train + heldout
    train_result = json.loads((OUT / "train/result.json").read_text(encoding="utf-8"))
    heldout_result = json.loads((OUT / "heldout/result.json").read_text(encoding="utf-8"))
    checks = {}
    checks["split_runs_pass"] = train_result["status"] == heldout_result["status"] == "PASS_ARCHIVE_GATE"
    checks["full_plan_pool_sizes_and_separation"] = len(plan["train"]) == 80 and len(plan["heldout"]) == 20 and {x["weather_seed"] for x in plan["train"]} == set(range(1001, 1081)) and {x["weather_seed"] for x in plan["heldout"]} == set(range(1081, 1101))
    checks["nine_training_years"] = len(train) == 9 and {int(r["year"]) for r in train} == set(range(2005, 2014))
    checks["actual_schedule_matches_plan"] = [(int(r["year"]), int(r["weather_seed"])) for r in train] == [(x["historical_year"], x["weather_seed"]) for x in plan["train"][:9]] and [(int(r["year"]), int(r["weather_seed"])) for r in heldout] == [(2007, 1081), (2007, 1100)]
    checks["actual_pools_disjoint"] = {int(r["weather_seed"]) for r in train}.isdisjoint({int(r["weather_seed"]) for r in heldout})
    checks["episode_sequences"] = [int(r["episode_index"]) for r in train] == list(range(1, 10)) and [int(r["episode_index"]) for r in heldout] == [1, 2]
    checks["all_runtime_seeds_and_physical_screen"] = all(int(r["weather_seed"]) == int(r["pdi_rseed1"]) and r["physical_status"] == "PASS" and r["status"] == "completed" for r in rows)
    weather_ok = True
    runtime_ok = True
    for row in rows:
        path = ROOT / row["weather_path"]
        if sha(path) != row["weather_sha256"]:
            weather_ok = False
        daily = load_csv(path)
        if len(daily) != int(row["days"]) or len(daily) != int(row["weather_rows"]) or list(daily[0]) != ["DATE", "DOY", "RAIN", "SRAD", "TMAX", "TMIN"]:
            weather_ok = False
        proof = json.loads((ROOT / row["runtime_evidence_path"]).read_text(encoding="utf-8"))
        if not (proof["wther"] == "W" and proof["wsta_confirmed"] and proof["yaml_bootstrap_confirmed"] and proof["runtime_rseed1"] == int(row["weather_seed"]) and proof["runtime_cli_sha256"] == proof["source_cli_sha256"]):
            runtime_ok = False
    checks["daily_weather_hash_and_row_reread"] = weather_ok
    checks["runtime_evidence_reread"] = runtime_ok
    checks["days_closed"] = sum(int(r["days"]) for r in train) == train_result["days_archived"] == 951 and sum(int(r["days"]) for r in heldout) == heldout_result["days_archived"] == 204
    resources = load_csv(OUT / "train/resource_usage.csv") + load_csv(OUT / "heldout/resource_usage.csv")
    peak = max(float(r["process_tree_rss_mb"]) for r in resources)
    checks["rss_below_limit"] = peak < 1536
    model = ROOT / "results/fqa_wgen_ppo_smoke_044/attempt_05/models/fqa_ppo_seed0_2k.zip"
    checks["model_frozen_hash"] = model.is_file() and all(json.loads((OUT / s / "preflight.json").read_text(encoding="utf-8"))["model_sha256"] == sha(model) for s in ("train", "heldout"))
    result = {"status": "PASS_ARCHIVE_GATE_ONLY" if all(checks.values()) else "FAIL", "checks": checks, "train_episodes": len(train), "heldout_episodes": len(heldout), "actual_weather_days": sum(int(r["days"]) for r in rows), "unique_realization_hashes": len({r["weather_sha256"] for r in rows}), "peak_process_tree_rss_mb": round(peak, 2), "scope": "044 checkpoint deterministic inference for nine FQA training years and two heldout weather seeds; no PPO learning", "limitations": ["80/20 plan separation is checked, but only 9 train and 2 heldout realizations were run", "full 80/20 weather climate distribution QA remains open", "native FIELD coordinate warning remains unresolved; no crop outcome inference from this gate"]}
    path = OUT / "final_gate.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if result["status"] == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
