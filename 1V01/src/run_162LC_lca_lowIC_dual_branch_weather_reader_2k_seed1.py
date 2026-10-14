"""162LC: independent seed1 2K smoke for the frozen LC-E2 contract."""

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

CONFIG = ROOT / "configs/162LC_lca_lowIC_dual_branch_weather_reader_2k_seed1.json"
REFERENCE_CONFIG = ROOT / "configs/155LC_lca_lowIC_dual_branch_weather_reader_smoke2k.json"
PROMPT = ROOT / "prompts/2026-08-16_lc_E2_independent_seed1_action_diversity_validation.md"
EXPECTED_TASK_ID = "162LC"


def isolation_audit() -> dict:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    ref = json.loads(REFERENCE_CONFIG.read_text(encoding="utf-8"))
    checks = {
        "station_site_profile_unchanged": (cfg["station_code"], cfg["site"], cfg["input_profile"]) == (ref["station_code"], ref["site"], ref["input_profile"]),
        "seed_is_independent_seed1": cfg["seed"] == 1 and cfg["seed"] != ref["seed"],
        "actions_unchanged": cfg["actions"] == ref["actions"],
        "weather_contract_unchanged": cfg["forecast_features"]["feature_names"] == ref["forecast_features"]["feature_names"] and cfg["forecast_features"]["raw_feature_names"] == ref["forecast_features"]["raw_feature_names"],
        "architecture_unchanged": cfg["policy_architecture"] == {**ref["policy_architecture"]},
        "years_reward_safety_unchanged": cfg["scope"]["train_years"] == ref["scope"]["train_years"] and cfg["scope"]["validation_years"] == ref["scope"]["validation_years"] and cfg["scope"]["reward_and_safety"] == ref["scope"]["reward_and_safety"],
        "automatic_controls_false": cfg["scope"]["external_n"] is False and cfg["scope"]["native_dssat_automatic_irrigation"] is False,
        "only_seed_changed": cfg["training"] == ref["training"],
    }
    return {"reference_config": REFERENCE_CONFIG.relative_to(ROOT).as_posix(), "changed_factor": "seed_0_to_seed_1_only", "checks": checks, "passed": all(checks.values())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.dry_run == args.smoke:
        raise ValueError("choose exactly one of --dry-run or --smoke")
    audit = isolation_audit()
    if not audit["passed"]:
        raise RuntimeError(json.dumps(audit, ensure_ascii=False))
    base.CONFIG = CONFIG
    base.patch_contract()
    # The inherited batch engine keeps a module-level SEED=0. Override it
    # explicitly so this is a real independent seed, then restore it.
    old_engine_seed = forecast.engine.base03222.SEED
    forecast.engine.base03222.SEED = 1
    try:
        with base.patched_maskableppo():
            result = forecast.run_forecast_experiment(CONFIG, PROMPT, EXPECTED_TASK_ID, args.dry_run, args.smoke, False)
    finally:
        forecast.engine.base03222.SEED = old_engine_seed
    result["lc_e2_seed1_isolation_audit"] = audit
    if args.smoke:
        out = ROOT / result["output_root"]
        result_path = out / "162LC_smoke_result.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        result["lc_e2_seed1_smoke_result_path"] = result_path.relative_to(ROOT).as_posix()
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
