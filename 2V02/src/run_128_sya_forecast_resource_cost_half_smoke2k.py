"""SY rolling-7d forecast 2K smoke with a single reward cost-scale change."""

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

CONFIG = ROOT / "configs/128_00_sya_originIC_rolling7d_forecast_resource_cost_half.json"
PROMPT = ROOT / "prompts/2026-08-16_sya_originIC_rolling7d_forecast_resource_cost_half_smoke2k.md"
TASK_ID = "128_00"
SCALE = 0.5


def prepare_2k(cfg: dict, smoke: bool) -> dict:
    planned = copy.deepcopy(cfg)
    planned["task_name"] = "sya_originIC_rolling7d_forecast_resource_cost_half_smoke2k"
    planned["training"] = {"total_timesteps": 2000, "checkpoint_steps": [1000, 2000]}
    return planned


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rolling.patch_contract()
    original_load_config = forecast.engine.load_config

    def scaled_load_config():
        cfg = original_load_config()
        reward = cfg.setdefault("reward", {})
        reward["water_cost"] = float(reward["water_cost"]) * SCALE
        reward["nitrogen_cost"] = float(reward["nitrogen_cost"]) * SCALE
        reward["resource_cost_scale_intervention"] = SCALE
        return cfg

    forecast.engine.load_config = scaled_load_config
    forecast.prepare_config = prepare_2k
    try:
        result = forecast.run_forecast_experiment(CONFIG, PROMPT, TASK_ID, args.dry_run, True, False)
        if args.dry_run:
            effective = scaled_load_config()["reward"]
            result["effective_reward"] = {"water_cost": float(effective["water_cost"]), "nitrogen_cost": float(effective["nitrogen_cost"]), "resource_cost_scale_intervention": float(effective["resource_cost_scale_intervention"])}
    finally:
        forecast.engine.load_config = original_load_config
    result["stage"] = "resource_cost_half_2k"
    result["reward_intervention"] = {"water_cost_scale": SCALE, "nitrogen_cost_scale": SCALE, "effective_water_cost": 0.55, "effective_nitrogen_cost": 0.79}
    if not args.dry_run:
        output = ROOT / result["output_root"]
        (output / "128_00_stage_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
