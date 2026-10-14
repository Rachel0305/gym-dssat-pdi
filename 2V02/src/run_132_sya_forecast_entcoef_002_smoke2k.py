"""132_00 isolated 2K smoke: entropy coefficient only."""

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

CONFIG = ROOT / "configs/132_00_sya_originIC_rolling7d_forecast_entcoef_002.json"
PROMPT = ROOT / "prompts/2026-08-16_sya_originIC_rolling7d_forecast_entcoef_002_smoke2k.md"
TASK_ID = "132_00"
ENTROPY_COEF = 0.02


def tuned_load_config(original_load_config):
    baseline = original_load_config()
    cfg = copy.deepcopy(baseline)
    ppo = cfg.setdefault("ppo", {})
    reward = cfg.setdefault("reward", {})
    baseline_ent = float(ppo["ent_coef"])
    baseline_yield = float(reward["yield_coef"])
    ppo["ent_coef"] = ENTROPY_COEF
    cfg["intervention"] = {
        "name": "entropy_coefficient_only",
        "baseline_ent_coef": baseline_ent,
        "effective_ent_coef": ENTROPY_COEF,
        "baseline_yield_coef": baseline_yield,
        "effective_yield_coef": float(reward["yield_coef"]),
    }
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    rolling.patch_contract()
    original_loader = forecast.engine.load_config
    forecast.engine.load_config = lambda: tuned_load_config(original_loader)
    try:
        effective = tuned_load_config(original_loader)
        result = forecast.run_forecast_experiment(CONFIG, PROMPT, TASK_ID, args.dry_run, args.smoke or not args.dry_run, False)
        result["ppo_intervention"] = {
            "name": "entropy_coefficient_only",
            "baseline_ent_coef": float(effective["intervention"]["baseline_ent_coef"]),
            "effective_ent_coef": float(effective["ppo"]["ent_coef"]),
            "baseline_yield_coef": float(effective["intervention"]["baseline_yield_coef"]),
            "effective_yield_coef": float(effective["reward"]["yield_coef"]),
            "reward_unchanged": True,
            "actual_runtime_patch": "engine.load_config -> base03222.load_config -> ppo_kwargs",
        }
        if not args.dry_run:
            output = ROOT / result["output_root"]
            (output / "132_00_ppo_intervention_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    finally:
        forecast.engine.load_config = original_loader


if __name__ == "__main__":
    raise SystemExit(main())
