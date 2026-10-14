"""113/114 compact rolling-7-day forecast smoke runner."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import forecast_engineered_observation_056_057 as forecast

CONFIGS = {"sy": ROOT / "configs/113_00_sya_originIC_compact_rolling7d_forecast.json", "lc": ROOT / "configs/114_00_lca_lowIC_compact_rolling7d_forecast.json"}
PROMPTS = {"sy": ROOT / "prompts/2026-08-15_sy_compact_rolling7d_forecast_smoke2k.md", "lc": ROOT / "prompts/2026-08-15_lc_compact_rolling7d_forecast_smoke2k.md"}
EXPECTED_IDS = {"sy": "113_00", "lc": "114_00"}
NORM_FEATURES = ["rain_next7_norm", "tmax_mean_next7_norm", "dry_spell_next7_flag"]
RAW_FEATURES = ["rain_next7_mm", "tmax_mean_next7_c", "dry_spell_next7_flag"]


def raw_compact(weather: pd.DataFrame, date: pd.Timestamp, scales: dict[str, float]) -> dict[str, float]:
    window = weather[(weather["date"] >= date + pd.Timedelta(days=1)) & (weather["date"] <= date + pd.Timedelta(days=7))].copy()
    rain = pd.to_numeric(window["rain"], errors="coerce") if not window.empty else pd.Series(dtype=float)
    tmax = pd.to_numeric(window["tmax"], errors="coerce") if not window.empty else pd.Series(dtype=float)
    rain_sum = float(rain.sum(skipna=True)) if not window.empty else 0.0
    tmax_mean = float(tmax.mean(skipna=True)) if not window.empty and tmax.notna().any() else 0.0
    heat_days = int((tmax >= float(scales["heat_days_threshold_c"])).sum()) if not window.empty else 0
    dry = float(rain_sum <= float(scales["dry_spell_next7_mm"]) and heat_days >= int(scales["heat_days_min"]))
    return {"rain_next7_mm": rain_sum, "tmax_mean_next7_c": tmax_mean, "dry_spell_next7_flag": dry}


def norm_compact(raw: dict[str, float], scales: dict[str, float]) -> dict[str, float]:
    return {"rain_next7_norm": raw["rain_next7_mm"] / float(scales["rain_next7_mm"]), "tmax_mean_next7_norm": (raw["tmax_mean_next7_c"] - float(scales["tmax_center_c"])) / float(scales["tmax_scale_c"]), "dry_spell_next7_flag": raw["dry_spell_next7_flag"]}


def patch_contract() -> None:
    forecast.FORECAST_FEATURE_NAMES = list(NORM_FEATURES)
    forecast.RAW_FORECAST_FEATURE_NAMES = list(RAW_FEATURES)
    forecast.raw_forecast_features_for_date = raw_compact
    forecast.normalize_forecast_features = norm_compact
    original_validate = forecast.validate_forecast_config

    def validate_compact(cfg: dict[str, Any], expected_task_id: str | None = None) -> dict[str, Any]:
        if cfg.get("observation_contract", {}).get("weather_forecast_mode") == "rolling_7d_compact_perfect_hindcast":
            checked = copy.deepcopy(cfg)
            checked["observation_contract"]["weather_forecast_mode"] = "engineered_perfect_hindcast_lookahead"
            result = original_validate(checked, expected_task_id=expected_task_id)
            result["observation_contract"]["weather_forecast_mode"] = "rolling_7d_compact_perfect_hindcast"
            return result
        return original_validate(cfg, expected_task_id=expected_task_id)

    forecast.validate_forecast_config = validate_compact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=["sy", "lc"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    patch_contract()
    result = forecast.run_forecast_experiment(CONFIGS[args.site], PROMPTS[args.site], EXPECTED_IDS[args.site], args.dry_run, args.smoke, False)
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
