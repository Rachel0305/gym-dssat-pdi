"""Corrected SY rolling-7d forecast resource-cost half 2K smoke.

The override is applied after direct_ppo.build_env_config, where the actual
runtime reward is constructed. This avoids the configuration-layer mistake in
128.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import forecast_engineered_observation_056_057 as forecast
import run_111_112_sy_lc_rolling7d_perfect_forecast_smoke2k as rolling

CONFIG = ROOT / "configs/129_00_sya_originIC_rolling7d_forecast_resource_cost_half_corrected.json"
PROMPT = ROOT / "prompts/2026-08-16_sya_originIC_rolling7d_forecast_resource_cost_half_corrected_smoke2k.md"
TASK_ID = "129_00"
SCALE = 0.5


def prepare_2k(cfg: dict, smoke: bool) -> dict:
    planned = copy.deepcopy(cfg)
    planned["task_name"] = "sya_originIC_rolling7d_forecast_resource_cost_half_corrected_smoke2k"
    planned["training"] = {"total_timesteps": 2000, "checkpoint_steps": [1000, 2000]}
    return planned


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rolling.patch_contract()
    original_build_env_config = forecast.direct_ppo.build_env_config

    def scaled_build_env_config(config, selection):
        env_config = original_build_env_config(config, selection)
        reward = env_config.setdefault("reward", {})
        baseline_water = float(reward["water_cost"])
        baseline_nitrogen = float(reward["nitrogen_cost"])
        reward["water_cost"] = baseline_water * SCALE
        reward["nitrogen_cost"] = baseline_nitrogen * SCALE
        reward["resource_cost_scale_intervention"] = SCALE
        reward["resource_cost_baseline_water"] = baseline_water
        reward["resource_cost_baseline_nitrogen"] = baseline_nitrogen
        return env_config

    forecast.direct_ppo.build_env_config = scaled_build_env_config
    forecast.prepare_config = prepare_2k
    try:
        result = forecast.run_forecast_experiment(CONFIG, PROMPT, TASK_ID, args.dry_run, True, False)
        if args.dry_run:
            runtime = scaled_build_env_config(forecast.engine.load_config(), forecast.engine.base03222.build_selection(forecast.engine.base03222.load_split()))["reward"]
            result["effective_runtime_reward"] = {"water_cost": float(runtime["water_cost"]), "nitrogen_cost": float(runtime["nitrogen_cost"]), "resource_cost_scale_intervention": float(runtime["resource_cost_scale_intervention"])}
    finally:
        forecast.direct_ppo.build_env_config = original_build_env_config
    result["stage"] = "corrected_resource_cost_half_2k"
    result["reward_intervention"] = {"water_cost_scale": SCALE, "nitrogen_cost_scale": SCALE, "effective_water_cost": 0.05, "effective_nitrogen_cost": 0.125}
    if not args.dry_run:
        output = ROOT / result["output_root"]
        (output / "129_00_stage_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
