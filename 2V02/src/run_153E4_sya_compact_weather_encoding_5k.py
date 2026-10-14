"""153E4: compact six-feature weather encoding, 2K smoke then 5K seed0."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import forecast_engineered_observation_056_057 as forecast
from run_141E1_sya_actionable_weather_encoding_smoke2k import patch_contract
from run_142E2_sya_dual_branch_weather_reader_smoke2k import patched_maskableppo


CONFIG = ROOT / "configs/153E4_sya_originIC_compact_weather_encoding_5k_seed0.json"
PROMPT = ROOT / "prompts/2026-08-16_sya_E4_compact_weather_encoding_5k.md"
EXPECTED_TASK_ID = "153E4"

ACTIVE_NORM = {
    "rain_next1_norm",
    "rain_next3_norm",
    "rain_next7_norm",
    "first_rain_lead_norm",
    "tmax_max_next7_norm",
    "dry_spell_next7_flag",
}
INACTIVE_NORM = {
    "rain_max_next7_norm",
    "rain_days_next7_norm",
    "tmax_mean_next7_norm",
    "tmin_mean_next7_norm",
    "srad_mean_next7_norm",
    "valid_days_next7_norm",
}
ACTIVE_RAW = {
    "rain_next1_mm",
    "rain_next3_mm",
    "rain_next7_mm",
    "first_rain_lead_days",
    "tmax_max_next7_c",
    "dry_spell_next7_flag",
}
INACTIVE_RAW = {
    "rain_max_next7_mm",
    "rain_days_next7",
    "tmax_mean_next7_c",
    "tmin_mean_next7_c",
    "srad_mean_next7_mj_m2",
    "valid_days_next7",
}


class CompactWeatherObservationWrapper(forecast.EngineeredForecastObservationWrapper):
    """Keep E2's 12-slot branch but zero six selected redundant/noisy slots."""

    def _forecast_vector_for_date(self, date: pd.Timestamp) -> np.ndarray:
        super()._forecast_vector_for_date(date)
        for name in INACTIVE_RAW:
            self.last_raw_forecast_features[name] = 0.0
        for name in INACTIVE_NORM:
            self.last_norm_forecast_features[name] = 0.0
        return np.array([self.last_norm_forecast_features[name] for name in forecast.FORECAST_FEATURE_NAMES], dtype=np.float32)

    @property
    def last_action_info(self) -> dict[str, Any]:
        info = dict(super().last_action_info)
        info["forecast_mode_056_057"] = "compact_6_of_12_zero_fill"
        info["forecast_active_feature_count_153E4"] = 6
        info["forecast_inactive_feature_count_153E4"] = 6
        info["forecast_inactive_columns_all_zero_153E4"] = True
        return info


def make_compact_weather_factory(base_make_env, exp_cfg):
    def make_env(runtime_config, env_config, station, year, seed, run_tag, evaluation=False):
        env = base_make_env(runtime_config, env_config, station, year, seed, run_tag, evaluation=evaluation)
        return CompactWeatherObservationWrapper(env, runtime_config, env_config, exp_cfg, station, year)

    return make_env


def compact_observation_smoke(cfg: dict[str, Any], out: Path, make_env):
    original = compact_observation_smoke.original
    path = original(cfg, out, make_env)
    df = pd.read_csv(path, keep_default_na=False)
    # observation_smoke writes the feature dictionaries without the daily
    # CSV's ``forecast_obs_``/``forecast_raw_`` prefixes.
    inactive_cols = sorted(INACTIVE_NORM | INACTIVE_RAW)
    active_cols = sorted(ACTIVE_NORM | ACTIVE_RAW)
    missing = [c for c in inactive_cols + active_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"E4 observation smoke missing columns: {missing}")
    inactive_zero = all(np.isclose(pd.to_numeric(df[c], errors="coerce").fillna(0.0), 0.0, atol=1e-9).all() for c in inactive_cols)
    active_finite = all(np.isfinite(pd.to_numeric(df[c], errors="coerce")).all() for c in active_cols)
    if not inactive_zero or not active_finite:
        raise RuntimeError(f"E4 observation smoke failed: inactive_zero={inactive_zero}, active_finite={active_finite}")
    check = {
        "task": cfg["task_id"],
        "observation_dim_all_37": bool((df["obs_dim"] == 37).all()),
        "inactive_columns_all_zero": inactive_zero,
        "active_columns_finite": active_finite,
        "active_feature_count": 6,
        "inactive_feature_count": 6,
        "next_step_allowed": bool((df["obs_dim"] == 37).all() and inactive_zero and active_finite),
    }
    (out / "audits" / f"{cfg['task_id']}_compact_observation_smoke_check.json").write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def compact_audit(out: Path, cfg: dict[str, Any]):
    audit, gate = compact_audit.original(out, cfg)
    summary_path = out / "evaluation" / f"{forecast.task_prefix(cfg)}_checkpoint_validation_summary.csv"
    inactive_cols = [f"forecast_obs_{n}" for n in INACTIVE_NORM] + [f"forecast_raw_{n}" for n in INACTIVE_RAW]
    active_cols = [f"forecast_obs_{n}" for n in ACTIVE_NORM] + [f"forecast_raw_{n}" for n in ACTIVE_RAW]
    checks = []
    summary = pd.read_csv(summary_path, keep_default_na=False)
    for rec in summary.itertuples(index=False):
        daily_path = ROOT / str(getattr(rec, "daily_csv_path"))
        daily = pd.read_csv(daily_path, keep_default_na=False) if daily_path.exists() else pd.DataFrame()
        inactive_zero = bool(daily_path.exists() and all(col in daily.columns and np.isclose(pd.to_numeric(daily[col], errors="coerce").fillna(0.0), 0.0, atol=1e-9).all() for col in inactive_cols))
        active_present = int(sum(col in daily.columns for col in active_cols)) if daily_path.exists() else 0
        checks.append({"checkpoint_step": int(rec.checkpoint_step), "year": int(rec.year), "inactive_weather_columns_all_zero": inactive_zero, "active_weather_columns_present": active_present})
    compact_df = pd.DataFrame(checks)
    audit = audit.merge(compact_df, on=["checkpoint_step", "year"], how="left", validate="one_to_one")
    final = compact_df[compact_df["checkpoint_step"].eq(int(max(cfg["training"]["checkpoint_steps"])))].copy()
    gate["inactive_weather_columns_all_zero"] = bool(len(final) == len(cfg["scope"]["validation_years"]) and final["inactive_weather_columns_all_zero"].all())
    gate["active_weather_columns_present"] = bool(len(final) == len(cfg["scope"]["validation_years"]) and (final["active_weather_columns_present"] == 12).all())
    gate["next_step_allowed"] = bool(gate.get("next_step_allowed") and gate["inactive_weather_columns_all_zero"] and gate["active_weather_columns_present"])
    return audit, gate


def compact_calendar_audit(cfg: dict[str, Any], out: Path):
    path = compact_calendar_audit.original(cfg, out)
    df = pd.read_csv(path, keep_default_na=False)
    for name in INACTIVE_RAW | INACTIVE_NORM:
        if name in df.columns:
            df[name] = 0.0
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def validate_e4(cfg: dict[str, Any], expected_task_id: str | None = None):
    checked = validate_e4.original(cfg, expected_task_id=expected_task_id)
    ff = checked["forecast_features"]
    if set(ff.get("active_feature_names", [])) != ACTIVE_NORM or set(ff.get("inactive_feature_names", [])) != INACTIVE_NORM:
        raise ValueError("E4 active/inactive feature contract mismatch.")
    if checked["policy_architecture"].get("weather_observation_dim") != 12 or checked["policy_architecture"].get("combined_features_dim") != 96:
        raise ValueError("E4 must retain the E2 25+12 -> 96 architecture.")
    return checked


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--smoke", action="store_true")
    phase.add_argument("--train5k", action="store_true")
    args = parser.parse_args()
    if args.dry_run == (args.smoke or args.train5k) or (args.smoke and args.train5k):
        raise ValueError("choose exactly one of --dry-run, --smoke, or --train5k")

    patch_contract()
    validate_e4.original = forecast.validate_forecast_config
    forecast.validate_forecast_config = validate_e4
    compact_observation_smoke.original = forecast.observation_smoke
    forecast.observation_smoke = compact_observation_smoke
    compact_audit.original = forecast.audit_actions_and_forecast_columns
    forecast.audit_actions_and_forecast_columns = compact_audit
    compact_calendar_audit.original = forecast.forecast_calendar_audit
    forecast.forecast_calendar_audit = compact_calendar_audit
    original_factory = forecast.make_forecast_env_factory
    forecast.make_forecast_env_factory = make_compact_weather_factory
    try:
        with patched_maskableppo():
            result = forecast.run_forecast_experiment(CONFIG, PROMPT, EXPECTED_TASK_ID, args.dry_run, args.smoke, args.train5k)
    finally:
        forecast.make_forecast_env_factory = original_factory
        forecast.observation_smoke = compact_observation_smoke.original
        forecast.audit_actions_and_forecast_columns = compact_audit.original
        forecast.forecast_calendar_audit = compact_calendar_audit.original
    result["e4_compact_encoding"] = {
        "active_features": sorted(ACTIVE_NORM),
        "inactive_features": sorted(INACTIVE_NORM),
        "reward_changed": False,
        "action_evaluation_changed": False,
        "architecture_changed": False,
    }
    if args.smoke:
        out = ROOT / result["output_root"]
        (out / "153E4_smoke_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.train5k:
        out = ROOT / result["output_root"]
        (out / "153E4_5k_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
