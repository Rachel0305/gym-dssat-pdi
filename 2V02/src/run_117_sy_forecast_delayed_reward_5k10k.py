"""117_00 SY 10K run retaining 2K/5K/10K checkpoints."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import forecast_engineered_observation_056_057 as forecast
from run_115_116_sy_lc_delayed_reward_smoke2k import CONFIGS as OLD_CONFIGS, PROMPTS as OLD_PROMPTS, patch_contract

CONFIG = ROOT / "configs/117_00_sya_originIC_forecast_delayed_reward_5k10k.json"
PROMPT = ROOT / "prompts/2026-08-15_sy_forecast_delayed_reward_5k10k.md"


def patch_long_schedule() -> None:
    original_prepare = forecast.prepare_config

    def prepare(cfg: dict[str, Any], smoke: bool) -> dict[str, Any]:
        if smoke:
            planned = copy.deepcopy(cfg)
            planned["training"] = {"total_timesteps": 10000, "checkpoint_steps": [2000, 5000, 10000]}
            return planned
        return original_prepare(cfg, smoke)

    forecast.prepare_config = prepare


def main() -> int:
    patch_contract()
    patch_long_schedule()
    result = forecast.run_forecast_experiment(CONFIG, PROMPT, "117_00", False, True, False)
    output = ROOT / result["output_root"]
    summary_path = output / "evaluation/117_00_checkpoint_validation_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path, keep_default_na=False)
        result["checkpoint_rows"] = {str(step): int((pd.to_numeric(summary["checkpoint_step"], errors="coerce") == step).sum()) for step in [2000, 5000, 10000]}
        result["checkpoint_metrics_mean"] = {}
        for step in [2000, 5000, 10000]:
            frame = summary[pd.to_numeric(summary["checkpoint_step"], errors="coerce").eq(step)]
            cols = [c for c in ["final_grnwt", "total_irrigation", "total_n", "PFP_N"] if c in frame.columns]
            result["checkpoint_metrics_mean"][str(step)] = frame[cols].apply(pd.to_numeric, errors="coerce").mean().to_dict()
        (output / "117_00_checkpoint_metrics.json").write_text(json.dumps(result["checkpoint_metrics_mean"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
