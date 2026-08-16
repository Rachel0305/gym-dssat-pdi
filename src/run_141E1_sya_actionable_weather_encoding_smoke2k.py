"""141E1: isolated timing-aware weather-encoding smoke for SY.

Only the appended weather representation differs from 140F.  Reward, action
grid, action masks, PPO settings, input profile, years, and seed continue to be
provided by the same validated forecast engine.  This runner intentionally has
no 5K/formal CLI path; extension requires a separate user decision.
"""

from __future__ import annotations

import argparse
import copy
import json
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


CONFIG = ROOT / "configs/141E1_sya_originIC_actionable_weather_encoding.json"
REFERENCE_CONFIG = ROOT / "configs/140F_sya_originIC_paired_forecast.json"
PROMPT = ROOT / "prompts/2026-08-16_sya_E1_actionable_weather_encoding_smoke2k.md"
EXPECTED_TASK_ID = "141E1"

NORM_FEATURES = [
    "rain_next1_norm",
    "rain_next3_norm",
    "rain_next7_norm",
    "rain_max_next7_norm",
    "rain_days_next7_norm",
    "first_rain_lead_norm",
    "tmax_mean_next7_norm",
    "tmax_max_next7_norm",
    "tmin_mean_next7_norm",
    "srad_mean_next7_norm",
    "valid_days_next7_norm",
    "dry_spell_next7_flag",
]
RAW_FEATURES = [
    "rain_next1_mm",
    "rain_next3_mm",
    "rain_next7_mm",
    "rain_max_next7_mm",
    "rain_days_next7",
    "first_rain_lead_days",
    "tmax_mean_next7_c",
    "tmax_max_next7_c",
    "tmin_mean_next7_c",
    "srad_mean_next7_mj_m2",
    "valid_days_next7",
    "dry_spell_next7_flag",
]


def _numeric(window: pd.DataFrame, column: str) -> pd.Series:
    if window.empty or column not in window:
        return pd.Series(dtype=float)
    return pd.to_numeric(window[column], errors="coerce")


def _mean_or(series: pd.Series, default: float) -> float:
    return float(series.mean(skipna=True)) if series.notna().any() else float(default)


def _max_or(series: pd.Series, default: float) -> float:
    return float(series.max(skipna=True)) if series.notna().any() else float(default)


def raw_actionable_encoding(
    weather: pd.DataFrame, date: pd.Timestamp, scales: dict[str, float]
) -> dict[str, float]:
    start = date + pd.Timedelta(days=1)
    end = date + pd.Timedelta(days=7)
    window = weather[(weather["date"] >= start) & (weather["date"] <= end)].copy()
    window = window.sort_values("date")
    rain = _numeric(window, "rain").fillna(0.0).clip(lower=0.0)
    tmax = _numeric(window, "tmax")
    tmin = _numeric(window, "tmin")
    srad = _numeric(window, "srad")
    threshold = float(scales["rain_day_threshold_mm"])
    wet_positions = np.flatnonzero(rain.to_numpy() >= threshold)
    first_rain_lead = float(wet_positions[0] + 1) if len(wet_positions) else 7.0
    valid_mask = window[[c for c in ["rain", "tmax", "tmin", "srad"] if c in window]].notna().all(axis=1)
    rain_next7 = float(rain.iloc[:7].sum())
    return {
        "rain_next1_mm": float(rain.iloc[:1].sum()),
        "rain_next3_mm": float(rain.iloc[:3].sum()),
        "rain_next7_mm": rain_next7,
        "rain_max_next7_mm": _max_or(rain.iloc[:7], 0.0),
        "rain_days_next7": float((rain.iloc[:7] >= threshold).sum()),
        "first_rain_lead_days": first_rain_lead,
        "tmax_mean_next7_c": _mean_or(tmax.iloc[:7], float(scales["tmax_center_c"])),
        "tmax_max_next7_c": _max_or(tmax.iloc[:7], float(scales["tmax_center_c"])),
        "tmin_mean_next7_c": _mean_or(tmin.iloc[:7], float(scales["tmin_center_c"])),
        "srad_mean_next7_mj_m2": _mean_or(srad.iloc[:7], float(scales["srad_center_mj_m2"])),
        "valid_days_next7": float(valid_mask.iloc[:7].sum()),
        "dry_spell_next7_flag": float(rain_next7 <= float(scales["dry_spell_next7_mm"])),
    }


def normalize_actionable_encoding(
    raw: dict[str, float], scales: dict[str, float]
) -> dict[str, float]:
    clip_abs = float(scales["clip_abs"])

    def positive(value: float, scale: float) -> float:
        return float(np.clip(value / scale, 0.0, clip_abs))

    def centered(value: float, center: float, scale: float) -> float:
        return float(np.clip((value - center) / scale, -clip_abs, clip_abs))

    return {
        "rain_next1_norm": positive(raw["rain_next1_mm"], float(scales["rain_next1_mm"])),
        "rain_next3_norm": positive(raw["rain_next3_mm"], float(scales["rain_next3_mm"])),
        "rain_next7_norm": positive(raw["rain_next7_mm"], float(scales["rain_next7_mm"])),
        "rain_max_next7_norm": positive(raw["rain_max_next7_mm"], float(scales["rain_max_next7_mm"])),
        "rain_days_next7_norm": float(np.clip(raw["rain_days_next7"] / 7.0, 0.0, 1.0)),
        "first_rain_lead_norm": float(np.clip(raw["first_rain_lead_days"] / 7.0, 0.0, 1.0)),
        "tmax_mean_next7_norm": centered(raw["tmax_mean_next7_c"], float(scales["tmax_center_c"]), float(scales["tmax_scale_c"])),
        "tmax_max_next7_norm": centered(raw["tmax_max_next7_c"], float(scales["tmax_center_c"]), float(scales["tmax_scale_c"])),
        "tmin_mean_next7_norm": centered(raw["tmin_mean_next7_c"], float(scales["tmin_center_c"]), float(scales["tmin_scale_c"])),
        "srad_mean_next7_norm": centered(raw["srad_mean_next7_mj_m2"], float(scales["srad_center_mj_m2"]), float(scales["srad_scale_mj_m2"])),
        "valid_days_next7_norm": float(np.clip(raw["valid_days_next7"] / 7.0, 0.0, 1.0)),
        "dry_spell_next7_flag": float(raw["dry_spell_next7_flag"]),
    }


def patch_contract() -> None:
    forecast.FORECAST_FEATURE_NAMES = list(NORM_FEATURES)
    forecast.RAW_FORECAST_FEATURE_NAMES = list(RAW_FEATURES)
    forecast.raw_forecast_features_for_date = raw_actionable_encoding
    forecast.normalize_forecast_features = normalize_actionable_encoding
    original_validate = forecast.validate_forecast_config

    def validate_e1(
        cfg: dict[str, Any], expected_task_id: str | None = None
    ) -> dict[str, Any]:
        if cfg.get("observation_contract", {}).get("weather_forecast_mode") == "rolling_7d_actionable_prefix_encoding":
            checked = copy.deepcopy(cfg)
            checked["observation_contract"]["weather_forecast_mode"] = "engineered_perfect_hindcast_lookahead"
            result = original_validate(checked, expected_task_id=expected_task_id)
            result["observation_contract"]["weather_forecast_mode"] = "rolling_7d_actionable_prefix_encoding"
            return result
        return original_validate(cfg, expected_task_id=expected_task_id)

    forecast.validate_forecast_config = validate_e1


def isolation_audit() -> dict[str, Any]:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    ref = json.loads(REFERENCE_CONFIG.read_text(encoding="utf-8"))
    checks = {
        "input_profile_unchanged": cfg["input_profile"] == ref["input_profile"],
        "station_and_site_unchanged": (cfg["station_code"], cfg["site"]) == (ref["station_code"], ref["site"]),
        "seed_unchanged": cfg["seed"] == ref["seed"],
        "actions_unchanged": cfg["actions"] == ref["actions"],
        "years_unchanged": cfg["scope"]["train_years"] == ref["scope"]["train_years"] and cfg["scope"]["validation_years"] == ref["scope"]["validation_years"],
        "reward_and_safety_unchanged": cfg["scope"]["reward_and_safety"] == ref["scope"]["reward_and_safety"],
        "external_n_unchanged_false": cfg["scope"]["external_n"] is False and ref["scope"]["external_n"] is False,
        "native_auto_irrigation_unchanged_false": cfg["scope"]["native_dssat_automatic_irrigation"] is False and ref["scope"]["native_dssat_automatic_irrigation"] is False,
        "base_observation_unchanged": cfg["observation_contract"]["base"] == ref["observation_contract"]["base"],
        "base_normalization_unchanged": cfg["observation_contract"]["normalization_enabled"] == ref["observation_contract"]["normalization_enabled"],
        "weather_source_unchanged": cfg["forecast_features"]["source"] == ref["forecast_features"]["source"],
        "weather_window_today_excluded": cfg["forecast_features"]["include_today_in_future_windows"] is False,
        "smoke_is_2k_only": cfg["training"] == {"total_timesteps": 2000, "checkpoint_steps": [1000, 2000]},
    }
    return {
        "reference_config": REFERENCE_CONFIG.relative_to(ROOT).as_posix(),
        "changed_factor": "forecast_features_encoding_only",
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--dry-run", action="store_true")
    phase.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    audit = isolation_audit()
    if not audit["passed"]:
        raise RuntimeError("E1 isolation audit failed: " + json.dumps(audit, ensure_ascii=False))
    patch_contract()
    result = forecast.run_forecast_experiment(
        CONFIG, PROMPT, EXPECTED_TASK_ID, args.dry_run, args.smoke, False
    )
    result["e1_isolation_audit"] = audit
    if args.smoke:
        out = ROOT / result["output_root"]
        audit_path = out / "141E1_isolation_audit.json"
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        result["e1_isolation_audit_path"] = audit_path.relative_to(ROOT).as_posix()
        result_path = out / "141E1_smoke_result.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
