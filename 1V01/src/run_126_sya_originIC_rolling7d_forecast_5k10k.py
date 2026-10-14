"""SY rolling-7d forecast 5K/10K run, isolated from the 111 runner."""

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

CONFIG = ROOT / "configs/126_00_sya_originIC_rolling7d_perfect_forecast_5k10k.json"
PROMPT = ROOT / "prompts/2026-08-16_sya_originIC_rolling7d_forecast_5k10k.md"
TASK_ID = "126_00"


def prepare_5k10k(cfg: dict, smoke: bool) -> dict:
    planned = copy.deepcopy(cfg)
    planned["training"] = {"total_timesteps": 10000, "checkpoint_steps": [2000, 5000, 10000]}
    return planned


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rolling.patch_contract()
    forecast.prepare_config = prepare_5k10k
    result = forecast.run_forecast_experiment(CONFIG, PROMPT, TASK_ID, args.dry_run, True, False)
    result["stage"] = "5k10k"
    if not args.dry_run:
        output = ROOT / result["output_root"]
        (output / "126_00_stage_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
