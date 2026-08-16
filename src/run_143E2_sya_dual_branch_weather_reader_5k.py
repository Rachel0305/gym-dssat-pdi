"""143E2: authorized fresh 5K extension of the passed 142E2 design."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import forecast_engineered_observation_056_057 as forecast
from run_141E1_sya_actionable_weather_encoding_smoke2k import patch_contract
from run_142E2_sya_dual_branch_weather_reader_smoke2k import patched_maskableppo


CONFIG = ROOT / "configs/143E2_sya_originIC_dual_branch_weather_reader_5k.json"
REFERENCE_CONFIG = ROOT / "configs/142E2_sya_originIC_dual_branch_weather_reader.json"
PROMPT = ROOT / "prompts/2026-08-16_sya_E2_dual_branch_weather_reader_5k.md"
SMOKE_OUT = ROOT / "benchmark_results/142E2_sya_originIC_dual_branch_weather_reader_smoke2k"
SMOKE_VALIDATION = ROOT / "benchmark_results/142E2_sya_dual_branch_weather_reader_validation/142E2_validation_summary.json"
EXPECTED_TASK_ID = "143E2"


def isolation_audit() -> dict[str, Any]:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    ref = json.loads(REFERENCE_CONFIG.read_text(encoding="utf-8"))
    checks = {
        "station_site_input_seed_unchanged": (cfg["station_code"], cfg["site"], cfg["input_profile"], cfg["seed"]) == (ref["station_code"], ref["site"], ref["input_profile"], ref["seed"]),
        "actions_unchanged": cfg["actions"] == ref["actions"],
        "weather_features_and_scales_unchanged": cfg["forecast_features"] == ref["forecast_features"],
        "observation_contract_core_unchanged": all(cfg["observation_contract"].get(k) == ref["observation_contract"].get(k) for k in ["base", "normalization_enabled", "weather_forecast_enabled", "weather_forecast_mode"]),
        "policy_architecture_unchanged": cfg["policy_architecture"] == ref["policy_architecture"],
        "years_reward_safety_unchanged": cfg["scope"]["train_years"] == ref["scope"]["train_years"] and cfg["scope"]["validation_years"] == ref["scope"]["validation_years"] and cfg["scope"]["reward_and_safety"] == ref["scope"]["reward_and_safety"],
        "automatic_controls_false": cfg["scope"]["external_n"] is False and cfg["scope"]["native_dssat_automatic_irrigation"] is False,
        "only_training_horizon_extended": ref["training"] == {"total_timesteps": 2000, "checkpoint_steps": [1000, 2000]} and cfg["training"] == {"total_timesteps": 5000, "checkpoint_steps": [2000, 5000]},
        "longer_runs_forbidden": cfg["execution_policy"]["no_10k_25k_50k_75k_100k"] is True,
    }
    return {"reference_config": REFERENCE_CONFIG.relative_to(ROOT).as_posix(), "changed_factor": "training_horizon_2k_to_5k_only", "checks": checks, "passed": all(checks.values())}


def verify_142e2_gate(_base_cfg: dict[str, Any]) -> dict[str, Any]:
    smoke_result_path = SMOKE_OUT / "142E2_smoke_result.json"
    smoke_manifest_path = SMOKE_OUT / "142E2_run_manifest.json"
    checks: dict[str, Any] = {
        "142E2_smoke_result_exists": smoke_result_path.exists(),
        "142E2_smoke_manifest_exists": smoke_manifest_path.exists(),
        "142E2_validation_exists": SMOKE_VALIDATION.exists(),
    }
    if all(checks.values()):
        smoke = json.loads(smoke_result_path.read_text(encoding="utf-8"))
        manifest = json.loads(smoke_manifest_path.read_text(encoding="utf-8"))
        validation = json.loads(SMOKE_VALIDATION.read_text(encoding="utf-8"))
        cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
        smoke_cfg = manifest.get("config", {})
        checks.update({
            "142E2_status_completed": manifest.get("status") == "completed_smoke",
            "142E2_smoke_gate_passed": smoke.get("smoke_gate", {}).get("next_step_allowed") is True,
            "142E2_architecture_isolation_passed": smoke.get("e2_isolation_audit", {}).get("passed") is True and smoke.get("e2_architecture_smoke", {}).get("passed") is True,
            "142E2_response_gate_passed": validation.get("counterfactual", {}).get("response_passed") is True,
            "142E2_five_k_candidate": validation.get("five_k_candidate") is True,
            "actions_match": smoke_cfg.get("actions") == cfg.get("actions"),
            "forecast_features_match": smoke_cfg.get("forecast_features") == cfg.get("forecast_features"),
            "input_station_seed_match": (smoke_cfg.get("input_profile"), smoke_cfg.get("station_code"), smoke_cfg.get("seed")) == (cfg.get("input_profile"), cfg.get("station_code"), cfg.get("seed")),
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
    iso = isolation_audit()
    gate = verify_142e2_gate({})
    if not iso["passed"] or not gate["next_step_allowed"]:
        raise RuntimeError(json.dumps({"isolation": iso, "142E2_gate": gate}, ensure_ascii=False))
    patch_contract()
    original_verify = forecast.verify_completed_smoke
    original_write_record = forecast.write_record

    def write_5k_record(cfg, pf, copied, audit, action_gate, _phase, obs_smoke, calendar_audit, smoke_verification=None):
        return original_write_record(cfg, pf, copied, audit, action_gate, "5K extension", obs_smoke, calendar_audit, smoke_verification)

    forecast.verify_completed_smoke = verify_142e2_gate
    forecast.write_record = write_5k_record
    try:
        with patched_maskableppo():
            result = forecast.run_forecast_experiment(CONFIG, PROMPT, EXPECTED_TASK_ID, args.dry_run, False, True)
    finally:
        forecast.verify_completed_smoke = original_verify
        forecast.write_record = original_write_record
    result["e2_5k_isolation_audit"] = iso
    result["authorized_142E2_gate"] = gate
    if args.train5k:
        out = ROOT / result["output_root"]
        (out / "143E2_5k_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
