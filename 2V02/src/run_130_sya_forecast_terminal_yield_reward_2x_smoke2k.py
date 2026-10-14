"""130_00 isolated 2K smoke: strengthen delayed terminal-yield credit only."""

from __future__ import annotations

import argparse
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

CONFIG = ROOT / "configs/130_00_sya_originIC_rolling7d_forecast_terminal_yield_reward_2x.json"
PROMPT = ROOT / "prompts/2026-08-16_sya_originIC_rolling7d_forecast_terminal_yield_reward_2x_smoke2k.md"
TASK_ID = "130_00"
YIELD_SCALE = 2.0


def scaled_load_config(original_load_config):
    baseline = original_load_config()
    cfg = copy.deepcopy(baseline)
    reward = cfg.setdefault("reward", {})
    baseline_yield = float(reward["yield_coef"])
    reward["yield_coef"] = baseline_yield * YIELD_SCALE
    reward["terminal_yield_credit_scale_intervention"] = YIELD_SCALE
    reward["terminal_yield_credit_baseline_yield_coef"] = baseline_yield
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    rolling.patch_contract()
    original_load_config = forecast.engine.load_config
    effective_loader = lambda: scaled_load_config(original_load_config)
    forecast.engine.load_config = effective_loader
    try:
        effective = scaled_load_config(original_load_config)
        reward = effective["reward"]
        result = forecast.run_forecast_experiment(CONFIG, PROMPT, TASK_ID, args.dry_run, args.smoke or not args.dry_run, False)
        result["reward_intervention"] = {
            "name": "terminal_yield_credit_scale",
            "yield_coef_scale": YIELD_SCALE,
            "baseline_yield_coef": float(reward["terminal_yield_credit_baseline_yield_coef"]),
            "effective_yield_coef": float(reward["yield_coef"]),
            "resource_costs_unchanged": True,
            "actual_runtime_reward_patch": "engine.load_config -> base03222.load_config -> StressAwareDiscreteWrapper.config",
        }
        if not args.dry_run:
            output = ROOT / result["output_root"]
            (output / "130_00_reward_intervention_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    finally:
        forecast.engine.load_config = original_load_config


if __name__ == "__main__":
    raise SystemExit(main())
