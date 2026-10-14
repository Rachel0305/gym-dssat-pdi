"""167LC: LC-E2 seed1 matched no-forecast control (2K gate, then 5K)."""

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
import run_158LC_lca_lowIC_dual_branch_noforecast_control as base

CONFIG = ROOT / "configs/167LC_lca_lowIC_dual_branch_noforecast_control_5k_seed1.json"
REFERENCE_CONFIG = ROOT / "configs/163LC_lca_lowIC_dual_branch_weather_reader_5k_seed1.json"
PROMPT = ROOT / "prompts/2026-08-16_lc_seed1_noforecast_control_wp_et_comparison.md"
SMOKE_OUT = ROOT / "benchmark_results/167LC_lca_lowIC_dual_branch_noforecast_control_5k_seed1_smoke2k"
EXPECTED_TASK_ID = "167LC"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_seed1_zero_weather(cfg: dict, expected_task_id: str | None = None) -> dict:
    if expected_task_id is not None and str(cfg.get("task_id")) != expected_task_id:
        raise ValueError(f"Expected task_id={expected_task_id}, got {cfg.get('task_id')}")
    if (cfg.get("station_code"), cfg.get("site"), cfg.get("input_profile"), cfg.get("seed")) != ("LCA", "LC", "lowIC", 1):
        raise ValueError("167LC must remain LCA/LC lowIC seed1.")
    if cfg.get("actions", {}).get("irrigation_levels_mm") != [0.0, 15.0, 30.0, 45.0]:
        raise ValueError("167LC irrigation grid mismatch.")
    if cfg.get("actions", {}).get("nitrogen_levels_kg_ha") != [0.0, 40.0, 80.0, 120.0]:
        raise ValueError("167LC nitrogen grid mismatch.")
    obs = cfg.get("observation_contract", {})
    if obs.get("base") != "046_02_raw_observation" or bool(obs.get("normalization_enabled")):
        raise ValueError("167LC base observation contract mismatch.")
    if bool(obs.get("weather_forecast_enabled")) or obs.get("weather_forecast_mode") != "matched_architecture_zero_weather_control":
        raise ValueError("167LC must have no weather signal.")
    expected_arch = {"base_observation_dim": 26, "weather_observation_dim": 12, "base_latent_dim": 64, "weather_latent_dim": 32, "combined_features_dim": 96, "downstream_net_arch": [64, 64]}
    for key, value in expected_arch.items():
        if cfg.get("policy_architecture", {}).get(key) != value:
            raise ValueError(f"167LC architecture mismatch at {key}")
    if cfg.get("training") != {"total_timesteps": 5000, "checkpoint_steps": [2000, 5000]}:
        raise ValueError("167LC training contract mismatch.")
    isolation = base.control_isolation_audit(cfg)
    if not isolation["passed"]:
        raise ValueError(json.dumps(isolation, ensure_ascii=False))
    return cfg


def smoke_verification_seed1() -> dict:
    result_path = SMOKE_OUT / "167LC_smoke_result.json"
    manifest_path = SMOKE_OUT / "167LC_run_manifest.json"
    checks = {"smoke_result_exists": result_path.exists(), "smoke_manifest_exists": manifest_path.exists()}
    if all(checks.values()):
        result = read(result_path)
        manifest = read(manifest_path)
        gate = result.get("smoke_gate", result.get("action_forecast_gate", {}))
        checks.update({"status_completed_smoke": manifest.get("status") == "completed_smoke", "smoke_gate_passed": gate.get("next_step_allowed") is True, "control_audit_passed": result.get("lc_noforecast_control", {}).get("passed") is True})
    checks["next_step_allowed"] = all(bool(value) for key, value in checks.items() if key != "next_step_allowed")
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--smoke", action="store_true")
    phase.add_argument("--train5k", action="store_true")
    args = parser.parse_args()

    # Configure the imported 158 runner with the seed1-specific contract.
    base.CONFIG = CONFIG
    base.REFERENCE_CONFIG = REFERENCE_CONFIG
    base.PROMPT = PROMPT
    base.SMOKE_OUT = SMOKE_OUT
    base.EXPECTED_TASK_ID = EXPECTED_TASK_ID
    base.validate_zero_weather_config = validate_seed1_zero_weather
    base.smoke_verification = smoke_verification_seed1

    old_seed = forecast.engine.base03222.SEED
    forecast.engine.base03222.SEED = 1
    try:
        result = base.run_phase(smoke=args.smoke, formal=args.train5k)
    finally:
        forecast.engine.base03222.SEED = old_seed

    # The shared 158 helper writes legacy-named sidecars; retain explicit 167 names.
    out = ROOT / str(result["output_root"])
    if args.smoke:
        for source, target in [(out / "158LC_control_audit.json", out / "167LC_control_audit.json"), (out / "158LC_smoke_result.json", out / "167LC_smoke_result.json")]:
            if source.exists() and source != target:
                target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        source = out / "158LC_5k_result.json"
        target = out / "167LC_5k_result.json"
        if source.exists() and source != target:
            target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
