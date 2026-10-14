"""163LC: authorized 5K extension after the passed 162LC seed1 smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import forecast_engineered_observation_056_057 as forecast
import run_155LC_lca_lowIC_dual_branch_weather_reader_smoke2k as base

CONFIG = ROOT / "configs/163LC_lca_lowIC_dual_branch_weather_reader_5k_seed1.json"
SMOKE_OUT = ROOT / "benchmark_results/162LC_lca_lowIC_dual_branch_weather_reader_2k_seed1_smoke2k"
PROMPT = ROOT / "prompts/2026-08-16_lc_E2_independent_seed1_action_diversity_validation.md"
EXPECTED_TASK_ID = "163LC"


def isolation_audit() -> dict:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    smoke_cfg = json.loads((ROOT / "configs/162LC_lca_lowIC_dual_branch_weather_reader_2k_seed1.json").read_text(encoding="utf-8"))
    checks = {
        "station_site_profile_seed_unchanged": (cfg["station_code"], cfg["site"], cfg["input_profile"], cfg["seed"]) == (smoke_cfg["station_code"], smoke_cfg["site"], smoke_cfg["input_profile"], smoke_cfg["seed"]),
        "actions_weather_architecture_unchanged": cfg["actions"] == smoke_cfg["actions"] and cfg["forecast_features"] == smoke_cfg["forecast_features"] and cfg["policy_architecture"] == smoke_cfg["policy_architecture"],
        "years_reward_safety_unchanged": all(cfg["scope"][key] == smoke_cfg["scope"][key] for key in ["train_years", "validation_years", "reward_and_safety", "external_n", "native_dssat_automatic_irrigation"]),
        "only_training_horizon_extended": cfg["training"] == {"total_timesteps": 5000, "checkpoint_steps": [2000, 5000]},
        "longer_runs_forbidden": cfg["execution_policy"]["no_10k_25k_50k_75k_100k"] is True,
        "user_authorized_5k": cfg["execution_policy"]["user_authorized_5k"] is True,
    }
    return {"reference_config": "configs/162LC_lca_lowIC_dual_branch_weather_reader_2k_seed1.json", "changed_factor": "training_horizon_2k_to_5k_only_after_seed1_smoke", "checks": checks, "passed": all(checks.values())}


def verify_162_gate(_base_cfg=None) -> dict:
    result_path = SMOKE_OUT / "162LC_smoke_result.json"
    manifest_path = SMOKE_OUT / "162LC_run_manifest.json"
    checks = {"162LC_smoke_result_exists": result_path.exists(), "162LC_smoke_manifest_exists": manifest_path.exists()}
    if all(checks.values()):
        smoke = json.loads(result_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checks.update({
            "162LC_status_completed": manifest.get("status") == "completed_smoke",
            "162LC_smoke_gate_passed": smoke.get("smoke_gate", {}).get("next_step_allowed") is True,
            "162LC_isolation_passed": smoke.get("lc_e2_seed1_isolation_audit", {}).get("passed") is True,
            "station_profile_seed_match": (manifest.get("config", {}).get("station_code"), manifest.get("config", {}).get("site"), manifest.get("config", {}).get("input_profile"), manifest.get("config", {}).get("seed")) == ("LCA", "LC", "lowIC", 1),
        })
    checks["next_step_allowed"] = all(value for key, value in checks.items() if key != "next_step_allowed")
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train5k", action="store_true")
    args = parser.parse_args()
    if args.dry_run == args.train5k:
        raise ValueError("choose exactly one of --dry-run or --train5k")
    audit = isolation_audit()
    gate = verify_162_gate()
    if not audit["passed"] or not gate["next_step_allowed"]:
        raise RuntimeError(json.dumps({"isolation": audit, "162LC_gate": gate}, ensure_ascii=False))
    base.CONFIG = CONFIG
    base.patch_contract()
    original_verify = forecast.verify_completed_smoke
    original_write_record = forecast.write_record

    def write_5k_record(cfg, pf, copied, audit_info, action_gate, _phase, obs_smoke, calendar_audit, smoke_verification=None):
        return original_write_record(cfg, pf, copied, audit_info, action_gate, "5K seed1 extension", obs_smoke, calendar_audit, smoke_verification)

    forecast.verify_completed_smoke = verify_162_gate
    forecast.write_record = write_5k_record
    old_engine_seed = forecast.engine.base03222.SEED
    forecast.engine.base03222.SEED = 1
    try:
        with base.patched_maskableppo():
            result = forecast.run_forecast_experiment(CONFIG, PROMPT, EXPECTED_TASK_ID, args.dry_run, False, True)
    finally:
        forecast.engine.base03222.SEED = old_engine_seed
        forecast.verify_completed_smoke = original_verify
        forecast.write_record = original_write_record
    result["lc_e2_seed1_5k_isolation_audit"] = audit
    result["authorized_162LC_gate"] = gate
    if args.train5k:
        out = ROOT / result["output_root"]
        (out / "163LC_5k_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
