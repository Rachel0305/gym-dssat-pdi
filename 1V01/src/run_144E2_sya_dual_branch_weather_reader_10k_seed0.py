"""144E2: authorized seed0 E2 extension to 10K, no longer horizons."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path: sys.path.insert(0, str(path))

import forecast_engineered_observation_056_057 as forecast
from run_141E1_sya_actionable_weather_encoding_smoke2k import patch_contract
from run_142E2_sya_dual_branch_weather_reader_smoke2k import patched_maskableppo

CONFIG = ROOT / "configs/144E2_sya_originIC_dual_branch_weather_reader_10k_seed0.json"
REFERENCE_CONFIG = ROOT / "configs/143E2_sya_originIC_dual_branch_weather_reader_5k.json"
PROMPT = ROOT / "prompts/2026-08-16_sya_E2_dual_branch_weather_reader_10k_seed0.md"
REFERENCE_VALIDATION = ROOT / "benchmark_results/143E2_sya_dual_branch_weather_reader_5k_validation/143E2_5k_validation_summary.json"


def gate() -> dict[str, Any]:
    cfg, ref = [json.loads(p.read_text(encoding="utf-8")) for p in [CONFIG, REFERENCE_CONFIG]]
    val = json.loads(REFERENCE_VALIDATION.read_text(encoding="utf-8")) if REFERENCE_VALIDATION.exists() else {}
    checks = {
        "143E2_validation_exists": REFERENCE_VALIDATION.exists(),
        "143E2_retain_true": val.get("retain_E2_for_independent_seed_validation") is True,
        "station_site_input_seed_unchanged": (cfg["station_code"],cfg["site"],cfg["input_profile"],cfg["seed"]) == (ref["station_code"],ref["site"],ref["input_profile"],ref["seed"]),
        "actions_unchanged": cfg["actions"] == ref["actions"],
        "weather_unchanged": cfg["forecast_features"] == ref["forecast_features"],
        "architecture_unchanged": cfg["policy_architecture"] == ref["policy_architecture"],
        "years_reward_safety_unchanged": cfg["scope"]["train_years"] == ref["scope"]["train_years"] and cfg["scope"]["validation_years"] == ref["scope"]["validation_years"] and cfg["scope"]["reward_and_safety"] == ref["scope"]["reward_and_safety"],
        "horizon_only_5k_to_10k": ref["training"] == {"total_timesteps":5000,"checkpoint_steps":[2000,5000]} and cfg["training"] == {"total_timesteps":10000,"checkpoint_steps":[2000,5000,10000]},
        "longer_horizons_forbidden": cfg["execution_policy"]["no_25k_50k_75k_100k"] is True,
    }
    checks["next_step_allowed"] = all(checks.values())
    return checks


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--dry-run",action="store_true"); parser.add_argument("--train10k",action="store_true"); args=parser.parse_args()
    if args.dry_run == args.train10k: raise ValueError("choose exactly one phase")
    auth=gate()
    if not auth["next_step_allowed"]: raise RuntimeError(json.dumps(auth,ensure_ascii=False))
    patch_contract(); original_verify=forecast.verify_completed_smoke; original_record=forecast.write_record
    forecast.verify_completed_smoke=lambda _cfg: auth
    def record(cfg,pf,copied,audit,action_gate,_phase,obs,calendar,smoke_verification=None): return original_record(cfg,pf,copied,audit,action_gate,"10K collapse check",obs,calendar,smoke_verification)
    forecast.write_record=record
    try:
        with patched_maskableppo(): result=forecast.run_forecast_experiment(CONFIG,PROMPT,"144E2",args.dry_run,False,True)
    finally:
        forecast.verify_completed_smoke=original_verify; forecast.write_record=original_record
    result["authorized_gate"]=auth
    if args.train10k:
        out=ROOT/result["output_root"]; (out/"144E2_10k_result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2)); return 0

if __name__ == "__main__": raise SystemExit(main())
