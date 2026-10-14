"""111/112 isolated rolling 7-day perfect-hindcast forecast smoke runner.

The shared 056/057 evaluator is reused, but its engineered summary feature
contract is replaced in-process by exactly seven future days x four weather
fields. No existing runner is modified on disk.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import forecast_engineered_observation_056_057 as forecast

CONFIGS = {
    "sy": ROOT / "configs/111_00_sya_originIC_rolling7d_perfect_forecast_smoke2k.json",
    "lc": ROOT / "configs/112_00_lca_lowIC_rolling7d_perfect_forecast_smoke2k.json",
}
PROMPTS = {
    "sy": ROOT / "prompts/2026-08-15_sy_rolling7d_perfect_forecast_smoke2k.md",
    "lc": ROOT / "prompts/2026-08-15_lc_rolling7d_perfect_forecast_smoke2k.md",
}
EXPECTED_IDS = {"sy": "111_00", "lc": "112_00"}
WEATHER_FIELDS = [("rain", "mm"), ("srad", "mj_m2"), ("tmax", "c"), ("tmin", "c")]
RAW_7D = [f"{field}_d{day}_{unit}" for day in range(1, 8) for field, unit in WEATHER_FIELDS]
NORM_7D = [f"{field}_d{day}_norm" for day in range(1, 8) for field, _ in WEATHER_FIELDS]


def raw_rolling_7d(weather: pd.DataFrame, date: pd.Timestamp, scales: dict[str, float]) -> dict[str, float]:
    values: dict[str, float] = {}
    for day in range(1, 8):
        target = date + pd.Timedelta(days=day)
        row = weather[weather["date"].eq(target)]
        for field, unit in WEATHER_FIELDS:
            key = f"{field}_d{day}_{unit}"
            if row.empty or pd.isna(row[field].iloc[0]):
                values[key] = 0.0
            else:
                values[key] = float(row[field].iloc[0])
    return values


def normalize_rolling_7d(raw: dict[str, float], scales: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for day in range(1, 8):
        out[f"rain_d{day}_norm"] = raw[f"rain_d{day}_mm"] / float(scales["rain_daily_mm"])
        out[f"srad_d{day}_norm"] = raw[f"srad_d{day}_mj_m2"] / float(scales["srad_daily_mj_m2"])
        out[f"tmax_d{day}_norm"] = (raw[f"tmax_d{day}_c"] - float(scales["tmax_center_c"])) / float(scales["tmax_scale_c"])
        out[f"tmin_d{day}_norm"] = (raw[f"tmin_d{day}_c"] - float(scales["tmin_center_c"])) / float(scales["tmin_scale_c"])
    return out


def patch_contract() -> None:
    forecast.FORECAST_FEATURE_NAMES = list(NORM_7D)
    forecast.RAW_FORECAST_FEATURE_NAMES = list(RAW_7D)
    forecast.raw_forecast_features_for_date = raw_rolling_7d
    forecast.normalize_forecast_features = normalize_rolling_7d
    original_validate = forecast.validate_forecast_config

    def validate_rolling(cfg: dict[str, Any], expected_task_id: str | None = None) -> dict[str, Any]:
        if cfg.get("observation_contract", {}).get("weather_forecast_mode") == "rolling_7d_perfect_hindcast":
            checked = copy.deepcopy(cfg)
            checked["observation_contract"]["weather_forecast_mode"] = "engineered_perfect_hindcast_lookahead"
            result = original_validate(checked, expected_task_id=expected_task_id)
            result["observation_contract"]["weather_forecast_mode"] = "rolling_7d_perfect_hindcast"
            return result
        return original_validate(cfg, expected_task_id=expected_task_id)

    forecast.validate_forecast_config = validate_rolling


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=["sy", "lc"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    patch_contract()
    site = args.site
    cfg_path = CONFIGS[site]
    prompt = PROMPTS[site]
    return_code = 0
    result = forecast.run_forecast_experiment(cfg_path, prompt, EXPECTED_IDS[site], args.dry_run, args.smoke, False)
    print(f"rolling_7d_feature_count={len(NORM_7D)} raw_feature_count={len(RAW_7D)}")
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
