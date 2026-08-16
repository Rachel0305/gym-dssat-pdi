"""Shared engineered weather-forecast observation utilities for 056/057.

The purpose of this module is deliberately narrow:

1. keep the accepted 046_10/053_00 PPO environment unchanged;
2. append normalized, agronomically interpretable future-weather summaries;
3. write an audit trail proving that the appended values come from dates after
   the current DSSAT DAP, not from another hidden simulator state.

This is a lightweight forecast-informed RL branch.  It is not a nested DSSAT
model-predictive-control implementation; that would require daily inner
rollouts and is intentionally left out to avoid training-time I/O explosion.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine


SMOKE_TIMESTEPS = 2_000
SMOKE_CHECKPOINTS = [1_000, 2_000]
EXPECTED_IRRIGATION_LEVELS = [0.0, 15.0, 30.0, 45.0]
EXPECTED_NITROGEN_LEVELS = [0.0, 40.0, 80.0, 120.0]
INPUT_PROFILES = {
    "originIC": ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013",
    "lowIC": ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual",
}
FORECAST_FEATURE_NAMES = [
    "rain_next3_norm",
    "rain_next7_norm",
    "rain_next14_norm",
    "rain_past7_norm",
    "tmean_next7_norm",
    "tmax_mean_next7_norm",
    "heat_days_next7_norm",
    "srad_next7_norm",
    "heavy_rain_next3_flag",
    "dry_spell_next7_flag",
    "n_leach_risk_next3_flag",
]
RAW_FORECAST_FEATURE_NAMES = [
    "rain_next3_mm",
    "rain_next7_mm",
    "rain_next14_mm",
    "rain_past7_mm",
    "tmean_next7_c",
    "tmax_mean_next7_c",
    "heat_days_next7",
    "srad_next7_mj_m2",
    "heavy_rain_next3_flag",
    "dry_spell_next7_flag",
    "n_leach_risk_next3_flag",
]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def output_root(cfg: dict[str, Any]) -> Path:
    return ROOT / "benchmark_results" / f"{cfg['task_id']}_{cfg['task_name']}"


def task_prefix(cfg: dict[str, Any]) -> str:
    return str(cfg["task_id"])


def prepare_config(cfg: dict[str, Any], smoke: bool) -> dict[str, Any]:
    planned = json.loads(json.dumps(cfg))
    if smoke:
        planned["task_name"] = f"{planned['task_name']}_smoke2k"
        planned["training"] = {"total_timesteps": SMOKE_TIMESTEPS, "checkpoint_steps": SMOKE_CHECKPOINTS}
    return planned


def validate_forecast_config(cfg: dict[str, Any], expected_task_id: str | None = None) -> dict[str, Any]:
    if expected_task_id is not None and str(cfg.get("task_id")) != expected_task_id:
        raise ValueError(f"Expected task_id={expected_task_id}, got {cfg.get('task_id')}.")
    if cfg.get("input_profile") not in INPUT_PROFILES:
        raise ValueError(f"input_profile must be one of {list(INPUT_PROFILES)}.")
    actions = cfg.get("actions", {})
    irrigation = list(map(float, actions.get("irrigation_levels_mm", [])))
    nitrogen = list(map(float, actions.get("nitrogen_levels_kg_ha", [])))
    if irrigation != EXPECTED_IRRIGATION_LEVELS or nitrogen != EXPECTED_NITROGEN_LEVELS:
        raise ValueError("Forecast branch must keep the accepted 16-action grid.")
    obs = cfg.get("observation_contract", {})
    if obs.get("base") != "046_02_raw_observation":
        raise ValueError("Forecast branch must inherit the 046_02 raw observation base.")
    if bool(obs.get("normalization_enabled", True)):
        raise ValueError("Base observation normalization remains disabled; only appended forecast features are normalized.")
    if not bool(obs.get("weather_forecast_enabled", False)):
        raise ValueError("weather_forecast_enabled must be true.")
    if obs.get("weather_forecast_mode") != "engineered_perfect_hindcast_lookahead":
        raise ValueError("Only engineered_perfect_hindcast_lookahead is implemented in 056/057.")
    ff = cfg.get("forecast_features", {})
    if not bool(ff.get("enabled", False)):
        raise ValueError("forecast_features.enabled must be true.")
    if list(ff.get("feature_names", [])) != FORECAST_FEATURE_NAMES:
        raise ValueError("forecast_features.feature_names do not match the code contract.")
    if list(ff.get("raw_feature_names", [])) != RAW_FORECAST_FEATURE_NAMES:
        raise ValueError("forecast_features.raw_feature_names do not match the code contract.")
    return cfg


def _weather_table(config: dict[str, Any], station: str) -> pd.DataFrame:
    weather = direct_ppo.weather_for_daily(config)
    weather = weather[weather["station_code"].astype(str).eq(str(station))].copy()
    weather["date"] = pd.to_datetime(weather["date"])
    for col in ["rain", "tmin", "tmax", "srad"]:
        weather[col] = pd.to_numeric(weather.get(col), errors="coerce")
    return weather.sort_values("date").reset_index(drop=True)


def _window(weather: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return weather[(weather["date"] >= start) & (weather["date"] <= end)].copy()


def _safe_sum(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").sum(skipna=True))


def _safe_mean(series: pd.Series, default: float = 0.0) -> float:
    values = pd.to_numeric(series, errors="coerce")
    return float(values.mean(skipna=True)) if values.notna().any() else float(default)


def raw_forecast_features_for_date(weather: pd.DataFrame, date: pd.Timestamp, scales: dict[str, float]) -> dict[str, float]:
    next3 = _window(weather, date + pd.Timedelta(days=1), date + pd.Timedelta(days=3))
    next7 = _window(weather, date + pd.Timedelta(days=1), date + pd.Timedelta(days=7))
    next14 = _window(weather, date + pd.Timedelta(days=1), date + pd.Timedelta(days=14))
    past7 = _window(weather, date - pd.Timedelta(days=7), date - pd.Timedelta(days=1))

    rain_next3 = _safe_sum(next3["rain"]) if not next3.empty else 0.0
    rain_next7 = _safe_sum(next7["rain"]) if not next7.empty else 0.0
    rain_next14 = _safe_sum(next14["rain"]) if not next14.empty else 0.0
    rain_past7 = _safe_sum(past7["rain"]) if not past7.empty else 0.0
    tmean_next7 = _safe_mean((next7["tmax"] + next7["tmin"]) / 2.0) if not next7.empty else 0.0
    tmax_mean_next7 = _safe_mean(next7["tmax"]) if not next7.empty else 0.0
    heat_days_next7 = float((pd.to_numeric(next7["tmax"], errors="coerce") >= 32.0).sum()) if not next7.empty else 0.0
    srad_next7 = _safe_sum(next7["srad"]) if not next7.empty else 0.0

    heavy_rain_next3 = float(rain_next3 >= float(scales["heavy_rain_next3_mm"]))
    dry_spell_next7 = float(rain_next7 <= float(scales["dry_spell_next7_mm"]) and heat_days_next7 >= 2.0)
    n_leach_risk_next3 = float(rain_next3 >= float(scales["n_leach_risk_next3_mm"]))

    return {
        "rain_next3_mm": rain_next3,
        "rain_next7_mm": rain_next7,
        "rain_next14_mm": rain_next14,
        "rain_past7_mm": rain_past7,
        "tmean_next7_c": tmean_next7,
        "tmax_mean_next7_c": tmax_mean_next7,
        "heat_days_next7": heat_days_next7,
        "srad_next7_mj_m2": srad_next7,
        "heavy_rain_next3_flag": heavy_rain_next3,
        "dry_spell_next7_flag": dry_spell_next7,
        "n_leach_risk_next3_flag": n_leach_risk_next3,
    }


def normalize_forecast_features(raw: dict[str, float], scales: dict[str, float]) -> dict[str, float]:
    return {
        "rain_next3_norm": raw["rain_next3_mm"] / float(scales["rain_next3_mm"]),
        "rain_next7_norm": raw["rain_next7_mm"] / float(scales["rain_next7_mm"]),
        "rain_next14_norm": raw["rain_next14_mm"] / float(scales["rain_next14_mm"]),
        "rain_past7_norm": raw["rain_past7_mm"] / float(scales["rain_past7_mm"]),
        "tmean_next7_norm": (raw["tmean_next7_c"] - float(scales["tmean_center_c"])) / float(scales["tmean_scale_c"]),
        "tmax_mean_next7_norm": (raw["tmax_mean_next7_c"] - float(scales["tmax_center_c"])) / float(scales["tmax_scale_c"]),
        "heat_days_next7_norm": raw["heat_days_next7"] / float(scales["heat_days_next7"]),
        "srad_next7_norm": raw["srad_next7_mj_m2"] / float(scales["srad_next7_mj_m2"]),
        "heavy_rain_next3_flag": raw["heavy_rain_next3_flag"],
        "dry_spell_next7_flag": raw["dry_spell_next7_flag"],
        "n_leach_risk_next3_flag": raw["n_leach_risk_next3_flag"],
    }


class EngineeredForecastObservationWrapper(gym.Env):
    """Append normalized future-weather summaries to the PPO observation."""

    def __init__(self, env: gym.Env, runtime_config: dict[str, Any], env_config: dict[str, Any], exp_cfg: dict[str, Any], station: str, year: int):
        super().__init__()
        self.env = env
        self.runtime_config = runtime_config
        self.env_config = env_config
        self.exp_cfg = exp_cfg
        self.station = str(station)
        self.year = int(year)
        self.scales = dict(exp_cfg["forecast_features"]["scales"])
        self.weather = _weather_table(runtime_config, self.station)
        self.planting = pd.Timestamp(direct_ppo.find_year(env_config, self.station, self.year)["planting_date"])
        self.action_space = env.action_space
        self.metadata = getattr(env, "metadata", {})
        base_space = env.observation_space
        base_low = np.asarray(base_space.low, dtype=np.float32).reshape(-1)
        base_high = np.asarray(base_space.high, dtype=np.float32).reshape(-1)
        forecast_low = np.full(len(FORECAST_FEATURE_NAMES), -5.0, dtype=np.float32)
        forecast_high = np.full(len(FORECAST_FEATURE_NAMES), 5.0, dtype=np.float32)
        self.base_observation_dim = int(len(base_low))
        self.enhanced_observation_dim = self.base_observation_dim + len(FORECAST_FEATURE_NAMES)
        self.observation_space = spaces.Box(
            low=np.concatenate([base_low, forecast_low]).astype(np.float32),
            high=np.concatenate([base_high, forecast_high]).astype(np.float32),
            dtype=np.float32,
        )
        self.last_raw_forecast_features: dict[str, float] = {}
        self.last_norm_forecast_features: dict[str, float] = {}
        self.last_observation_date: pd.Timestamp | None = None
        self.last_raw_enhanced_obs: np.ndarray | None = None

    def _date_from_obs(self, obs: np.ndarray, info: dict | None = None) -> pd.Timestamp:
        latest = direct_ppo.latest_observation_dict(self.env, obs, info or {})
        dap_raw = direct_ppo.scalar(latest.get("dap", np.nan), np.nan)
        if np.isfinite(dap_raw) and dap_raw > 0:
            return self.planting + pd.Timedelta(days=int(round(float(dap_raw))) - 1)
        return self.planting

    def _forecast_vector_for_date(self, date: pd.Timestamp) -> np.ndarray:
        raw = raw_forecast_features_for_date(self.weather, date, self.scales)
        norm = normalize_forecast_features(raw, self.scales)
        self.last_observation_date = date
        self.last_raw_forecast_features = raw
        self.last_norm_forecast_features = norm
        return np.array([norm[name] for name in FORECAST_FEATURE_NAMES], dtype=np.float32)

    def _augment(self, obs: np.ndarray, info: dict | None = None) -> np.ndarray:
        flat = np.asarray(obs, dtype=np.float32).reshape(-1)
        if len(flat) != self.base_observation_dim:
            raise RuntimeError(f"Expected base obs len {self.base_observation_dim}, got {len(flat)}")
        out = np.concatenate([flat, self._forecast_vector_for_date(self._date_from_obs(flat, info))]).astype(np.float32)
        if not np.isfinite(out).all():
            raise RuntimeError("Forecast-enhanced observation contains NaN or Inf.")
        self.last_raw_enhanced_obs = out
        return out

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        return self._augment(obs, info), dict(info or {})

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(obs, info), reward, terminated, truncated, dict(info or {})

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()

    @property
    def last_action_info(self) -> dict[str, Any]:
        base_info = getattr(self.env, "last_action_info", {})
        info = dict(base_info) if isinstance(base_info, dict) else {}
        for key, value in self.last_raw_forecast_features.items():
            info[f"forecast_raw_{key}"] = float(value)
        for key, value in self.last_norm_forecast_features.items():
            info[f"forecast_obs_{key}"] = float(value)
        info["forecast_observation_enabled_056_057"] = True
        info["forecast_mode_056_057"] = "engineered_perfect_hindcast_lookahead"
        info["forecast_base_observation_dim_056_057"] = int(self.base_observation_dim)
        info["forecast_enhanced_observation_dim_056_057"] = int(self.enhanced_observation_dim)
        if self.last_observation_date is not None:
            info["forecast_observation_date"] = self.last_observation_date.strftime("%Y-%m-%d")
        return info

    def close(self):
        return self.env.close()

    def render(self):
        return self.env.render()

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.env, name)


def split_for_station(station: str) -> pd.DataFrame:
    split = engine.base03222.load_split().copy()
    split["year"] = pd.to_numeric(split["year"], errors="coerce").astype(int)
    return split[split["station_code"].astype(str).eq(str(station))].sort_values("year").reset_index(drop=True)


def preflight(cfg: dict[str, Any], prompt: Path) -> dict[str, Any]:
    station = str(cfg["station_code"])
    site = str(cfg.get("site", station))
    input_root = INPUT_PROFILES[str(cfg["input_profile"])]
    source_mzx_name = "CNSY1201.MZX" if site == "SY" else f"CN{site}0801.MZX"
    source_mzx = input_root / site / source_mzx_name
    split = split_for_station(station)
    train_actual = sorted(split.loc[split["split"].eq("train"), "year"].astype(int).tolist())
    validation_actual = sorted(split.loc[split["split"].eq("validation"), "year"].astype(int).tolist())
    expected_train = list(map(int, cfg["scope"]["train_years"]))
    expected_validation = list(map(int, cfg["scope"]["validation_years"]))
    issues: list[str] = []
    if not input_root.exists():
        issues.append(f"input root missing: {rel(input_root)}")
    if not source_mzx.exists():
        issues.append(f"source MZX missing: {rel(source_mzx)}")
    if train_actual != expected_train:
        issues.append(f"train years mismatch: config={expected_train}, engine={train_actual}")
    if validation_actual != expected_validation:
        issues.append(f"validation years mismatch: config={expected_validation}, engine={validation_actual}")
    if not prompt.exists():
        issues.append(f"prompt missing: {rel(prompt)}")
    return {
        "task": f"{cfg['task_id']}_{cfg['task_name']}",
        "station_code": station,
        "site": site,
        "input_profile": cfg["input_profile"],
        "resolved_input_root": rel(input_root),
        "source_mzx": rel(source_mzx),
        "config_train_years": expected_train,
        "config_validation_years": expected_validation,
        "engine_train_years": train_actual,
        "engine_validation_years": validation_actual,
        "reference_run": cfg["reference_run"],
        "observation_contract": cfg["observation_contract"],
        "forecast_features": cfg["forecast_features"]["feature_names"],
        "irrigation_levels_mm": cfg["actions"]["irrigation_levels_mm"],
        "nitrogen_levels_kg_ha": cfg["actions"]["nitrogen_levels_kg_ha"],
        "combined_action_count": len(cfg["actions"]["irrigation_levels_mm"]) * len(cfg["actions"]["nitrogen_levels_kg_ha"]),
        "issues": issues,
        "next_step_allowed": not issues,
    }


def make_forecast_env_factory(
    base_make_env: Callable[..., gym.Env],
    exp_cfg: dict[str, Any],
) -> Callable[..., EngineeredForecastObservationWrapper]:
    def make_env(runtime_config: dict[str, Any], env_config: dict[str, Any], station: str, year: int, seed: int, run_tag: str, evaluation: bool = False):
        env = base_make_env(runtime_config, env_config, station, year, seed, run_tag, evaluation=evaluation)
        return EngineeredForecastObservationWrapper(env, runtime_config, env_config, exp_cfg, station, year)

    return make_env


def observation_smoke(cfg: dict[str, Any], out: Path, make_env: Callable[..., EngineeredForecastObservationWrapper]) -> Path:
    runtime_config = engine.load_config()
    selection = engine.base03222.build_selection(engine.base03222.load_split())
    env_config = direct_ppo.build_env_config(runtime_config, selection)
    station = str(cfg["station_code"])
    years = [int(cfg["scope"]["train_years"][0]), int(cfg["scope"]["validation_years"][0]), int(cfg["scope"]["validation_years"][-1])]
    rows: list[dict[str, Any]] = []
    for year in years:
        env = make_env(runtime_config, env_config, station, year, int(cfg.get("seed", 0)), f"{cfg['task_id']}_forecast_obs_smoke_{year}", evaluation=True)
        try:
            obs, _info = env.reset()
            row = {
                "year": year,
                "station_code": station,
                "obs_dim": int(np.asarray(obs).reshape(-1).shape[0]),
                "base_observation_dim": int(env.base_observation_dim),
                "forecast_feature_count": len(FORECAST_FEATURE_NAMES),
                "expected_dim": int(env.enhanced_observation_dim),
                "date": env.last_observation_date.strftime("%Y-%m-%d") if env.last_observation_date is not None else "",
            }
            row.update(env.last_raw_forecast_features)
            row.update(env.last_norm_forecast_features)
            rows.append(row)
        finally:
            env.close()
    path = out / "audits" / f"{task_prefix(cfg)}_forecast_observation_smoke.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def forecast_calendar_audit(cfg: dict[str, Any], out: Path) -> Path:
    runtime_config = engine.load_config()
    station = str(cfg["station_code"])
    weather = _weather_table(runtime_config, station)
    scales = dict(cfg["forecast_features"]["scales"])
    rows: list[dict[str, Any]] = []
    for year in list(map(int, cfg["scope"]["validation_years"][:2])):
        selection = engine.base03222.build_selection(engine.base03222.load_split())
        env_config = direct_ppo.build_env_config(runtime_config, selection)
        planting = pd.Timestamp(direct_ppo.find_year(env_config, station, year)["planting_date"])
        for dap in [1, 15, 30, 60, 90]:
            date = planting + pd.Timedelta(days=dap - 1)
            raw = raw_forecast_features_for_date(weather, date, scales)
            norm = normalize_forecast_features(raw, scales)
            rows.append(
                {
                    "year": year,
                    "dap": dap,
                    "current_date": date.strftime("%Y-%m-%d"),
                    "future_window_starts": (date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    "future_window_excludes_today": True,
                    **raw,
                    **norm,
                }
            )
    path = out / "audits" / f"{task_prefix(cfg)}_forecast_calendar_audit.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def copy_clean_names(out: Path, prefix: str) -> dict[str, str]:
    mapping = {
        "evaluation/042_10_training_checkpoint_inventory.csv": f"evaluation/{prefix}_training_checkpoint_inventory.csv",
        "evaluation/042_10_checkpoint_validation_summary.csv": f"evaluation/{prefix}_checkpoint_validation_summary.csv",
        "evaluation/042_10_validation_summary_by_station_checkpoint.csv": f"evaluation/{prefix}_validation_summary_by_station_checkpoint.csv",
        "042_10_result.json": f"{prefix}_engine_result.json",
    }
    copied: dict[str, str] = {}
    for source_rel, target_rel in mapping.items():
        source, target = out / source_rel, out / target_rel
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied[target_rel] = source_rel
    return copied


def _on_grid(values: pd.Series, allowed: list[float]) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    return values.map(lambda value: bool(np.isclose(float(value), allowed, atol=1e-6).any()))


def audit_actions_and_forecast_columns(out: Path, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    prefix = task_prefix(cfg)
    summary_path = out / "evaluation" / f"{prefix}_checkpoint_validation_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing validation summary: {summary_path}")
    summary = pd.read_csv(summary_path, keep_default_na=False)
    rows: list[dict[str, Any]] = []
    required_forecast_cols = [f"forecast_obs_{name}" for name in FORECAST_FEATURE_NAMES] + [f"forecast_raw_{name}" for name in RAW_FORECAST_FEATURE_NAMES]
    for record in summary.itertuples(index=False):
        daily_path = ROOT / str(getattr(record, "daily_csv_path"))
        if not daily_path.exists():
            rows.append({"checkpoint_step": int(record.checkpoint_step), "year": int(record.year), "daily_exists": False})
            continue
        daily = pd.read_csv(daily_path)
        irrigation = pd.to_numeric(daily.get("safe_action_amir"), errors="coerce").fillna(0.0)
        nitrogen = pd.to_numeric(daily.get("safe_action_anfer"), errors="coerce").fillna(0.0)
        daps = pd.to_numeric(daily.get("dap"), errors="coerce")
        positive = (irrigation > 1e-9) | (nitrogen > 1e-9)
        present_forecast_cols = [col for col in required_forecast_cols if col in daily.columns]
        forecast_nonconstant_cols = 0
        for col in present_forecast_cols:
            values = pd.to_numeric(daily[col], errors="coerce")
            if values.notna().sum() >= 2 and float(values.max() - values.min()) > 1e-9:
                forecast_nonconstant_cols += 1
        rows.append(
            {
                "checkpoint_step": int(record.checkpoint_step),
                "year": int(record.year),
                "daily_exists": True,
                "row_count": int(len(daily)),
                "off_grid_irrigation_rows": int((~_on_grid(irrigation, EXPECTED_IRRIGATION_LEVELS)).sum()),
                "off_grid_nitrogen_rows": int((~_on_grid(nitrogen, EXPECTED_NITROGEN_LEVELS)).sum()),
                "positive_action_rows": int(positive.sum()),
                "positive_action_rows_after_dap1": int((positive & (daps > 1)).sum()),
                "forecast_required_column_count": len(required_forecast_cols),
                "forecast_present_column_count": len(present_forecast_cols),
                "forecast_nonconstant_column_count": int(forecast_nonconstant_cols),
                "forecast_missing_columns": ";".join(sorted(set(required_forecast_cols) - set(present_forecast_cols))),
            }
        )
    audit = pd.DataFrame(rows).sort_values(["checkpoint_step", "year"]).reset_index(drop=True)
    final_checkpoint = int(max(cfg["training"]["checkpoint_steps"]))
    final = audit[audit["checkpoint_step"].eq(final_checkpoint)].copy()
    gate = {
        "final_checkpoint": final_checkpoint,
        "expected_validation_rows": len(cfg["scope"]["validation_years"]),
        "observed_validation_rows": int(len(final)),
        "all_daily_files_exist": bool(len(final) and final["daily_exists"].all()),
        "all_actions_on_declared_grid": bool(len(final) and final["off_grid_irrigation_rows"].sum() == 0 and final["off_grid_nitrogen_rows"].sum() == 0),
        "positive_actions_present": bool(len(final) and final["positive_action_rows"].sum() > 0),
        "not_all_positive_actions_at_dap1": bool(len(final) and final["positive_action_rows_after_dap1"].sum() > 0),
        "forecast_columns_present": bool(len(final) and final["forecast_present_column_count"].min() == len(required_forecast_cols)),
        "forecast_columns_vary_within_seasons": bool(len(final) and final["forecast_nonconstant_column_count"].min() >= 4),
    }
    gate["next_step_allowed"] = bool(
        gate["observed_validation_rows"] == gate["expected_validation_rows"]
        and all(value for key, value in gate.items() if key not in {"final_checkpoint", "expected_validation_rows", "observed_validation_rows", "next_step_allowed"})
    )
    return audit, gate


def verify_completed_smoke(base_cfg: dict[str, Any]) -> dict[str, Any]:
    smoke_cfg = prepare_config(base_cfg, smoke=True)
    smoke_out = output_root(smoke_cfg)
    result_path = smoke_out / f"{task_prefix(smoke_cfg)}_smoke_result.json"
    manifest_path = smoke_out / f"{task_prefix(smoke_cfg)}_run_manifest.json"
    checks: dict[str, Any] = {
        "smoke_result_exists": result_path.exists(),
        "smoke_manifest_exists": manifest_path.exists(),
    }
    if not all(checks.values()):
        checks["next_step_allowed"] = False
        return checks
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    smoke_config = manifest.get("config", {})
    gate = result.get("smoke_gate", {})
    checks.update(
        {
            "smoke_status_completed": manifest.get("status") == "completed_smoke",
            "smoke_gate_passed": gate.get("next_step_allowed") is True,
            "smoke_timesteps_match": smoke_config.get("training", {}).get("total_timesteps") == SMOKE_TIMESTEPS,
            "smoke_checkpoints_match": smoke_config.get("training", {}).get("checkpoint_steps") == SMOKE_CHECKPOINTS,
            "smoke_input_profile_match": smoke_config.get("input_profile") == base_cfg.get("input_profile"),
            "smoke_station_match": smoke_config.get("station_code") == base_cfg.get("station_code"),
            "smoke_actions_match": smoke_config.get("actions") == base_cfg.get("actions"),
            "smoke_forecast_features_match": smoke_config.get("forecast_features", {}).get("feature_names") == base_cfg.get("forecast_features", {}).get("feature_names"),
            "smoke_output_root": rel(smoke_out),
        }
    )
    checks["next_step_allowed"] = all(value for key, value in checks.items() if key not in {"smoke_output_root", "next_step_allowed"})
    return checks


def write_manifest(out: Path, cfg_path: Path, prompt: Path, cfg: dict[str, Any], pf: dict[str, Any], status: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    config_dir = out / "configs"
    config_dir.mkdir(exist_ok=True)
    shutil.copy2(cfg_path, config_dir / cfg_path.name)
    if prompt.exists():
        shutil.copy2(prompt, config_dir / prompt.name)
    path = out / f"{task_prefix(cfg)}_run_manifest.json"
    path.write_text(json.dumps({"status": status, "config": cfg, "preflight": pf}, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_record(
    cfg: dict[str, Any],
    pf: dict[str, Any],
    copied: dict[str, str],
    audit: pd.DataFrame,
    gate: dict[str, Any],
    phase: str,
    obs_smoke: Path,
    calendar_audit: Path,
    smoke_verification: dict[str, Any] | None = None,
) -> Path:
    doc = ROOT / "docs" / f"{cfg['task_id']}_{cfg['task_name']}_record.md"
    lines = [
        f"# {cfg['task_id']} {cfg['station_code']} engineered forecast MaskablePPO record",
        "",
        "## Design",
        "",
        f"- Reference run: `{cfg['reference_run']}`.",
        "- Controlled change: append normalized engineered short-term weather-forecast summaries to the PPO observation.",
        "- Unchanged: base DSSAT observation, reward/safety logic, action grid, train/validation years, seed, and input profile.",
        f"- Forecast mode: `{cfg['observation_contract']['weather_forecast_mode']}`.",
        f"- Forecast source: `{cfg['forecast_features']['source']}`.",
        f"- Forecast features: `{cfg['forecast_features']['feature_names']}`.",
        f"- Observation smoke audit: `{rel(obs_smoke)}`.",
        f"- Forecast calendar audit: `{rel(calendar_audit)}`.",
        f"- Output root: `{rel(output_root(cfg))}`.",
        "",
        f"## {phase} gate",
        "",
    ]
    for key, value in gate.items():
        lines.append(f"- `{key}`: `{value}`")
    if smoke_verification is not None:
        lines.extend(["", "## Formal smoke verification", ""])
        for key, value in smoke_verification.items():
            lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Year-level action and forecast-column audit", "", audit.to_markdown(index=False), "", "## Normalized outputs", ""])
    for target, source in copied.items():
        lines.append(f"- `{target}` <- `{source}`")
    lines.append("")
    doc.write_text("\n".join(lines), encoding="utf-8")
    return doc


def run_forecast_experiment(
    cfg_path: Path,
    prompt: Path,
    expected_task_id: str,
    dry_run: bool,
    smoke: bool,
    formal: bool,
) -> dict[str, Any]:
    base_cfg = validate_forecast_config(read_json(cfg_path), expected_task_id=expected_task_id)
    cfg = prepare_config(base_cfg, smoke)
    pf = preflight(cfg, prompt)
    out = output_root(cfg)
    smoke_verification: dict[str, Any] | None = None
    if formal:
        smoke_verification = verify_completed_smoke(base_cfg)
        if not smoke_verification["next_step_allowed"]:
            pf["issues"].append(f"{expected_task_id} completed smoke verification did not pass")
            pf["next_step_allowed"] = False
    if dry_run:
        result = {**pf, "mode": "dry_run", "output_root": rel(out)}
        if smoke_verification is not None:
            result["smoke_verification"] = smoke_verification
        return result
    if not smoke and not formal:
        raise ValueError("Use --smoke for 2K validation or --formal for authorized 100K training.")
    if not pf["next_step_allowed"]:
        raise RuntimeError("Preflight failed: " + "; ".join(pf["issues"]))
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment output: {rel(out)}")

    old = {key: getattr(engine, key) for key in ["TASK_ID", "TASK_NAME", "BASE_OUT", "BASE_DOC", "PROMPT", "LOWIC_INPUT_ROOT", "STATION", "SITES", "BINARY_IRRIGATION_LEVELS", "BINARY_NITROGEN_LEVELS"]}
    old_site_names = dict(engine.base03222.SITE_NAMES)
    old_make_env = engine.base03222.base.make_env
    old_input_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    make_env = make_forecast_env_factory(old_make_env, cfg)
    try:
        engine.TASK_ID = str(cfg["task_id"])
        engine.TASK_NAME = str(cfg["task_name"])
        engine.BASE_OUT = out
        engine.BASE_DOC = ROOT / "docs" / f"{cfg['task_id']}_{cfg['task_name']}_record.md"
        engine.PROMPT = prompt
        engine.LOWIC_INPUT_ROOT = INPUT_PROFILES[str(cfg["input_profile"])]
        engine.STATION = str(cfg["station_code"])
        engine.SITES = [str(cfg["station_code"])]
        engine.BINARY_IRRIGATION_LEVELS = list(map(float, cfg["actions"]["irrigation_levels_mm"]))
        engine.BINARY_NITROGEN_LEVELS = list(map(float, cfg["actions"]["nitrogen_levels_kg_ha"]))
        engine.base03222.SITE_NAMES[str(cfg["station_code"])] = str(cfg.get("site", cfg["station_code"]))
        engine.base03222.base.make_env = make_env
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = INPUT_PROFILES[str(cfg["input_profile"])]
        obs_smoke = observation_smoke(cfg, out, make_env)
        calendar_audit = forecast_calendar_audit(cfg, out)
        engine.run_training(int(cfg["training"]["total_timesteps"]), list(map(int, cfg["training"]["checkpoint_steps"])), suffix="")
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root
        engine.base03222.base.make_env = old_make_env
        for key, value in old.items():
            setattr(engine, key, value)
        engine.base03222.SITE_NAMES.clear()
        engine.base03222.SITE_NAMES.update(old_site_names)

    copied = copy_clean_names(out, task_prefix(cfg))
    audit, gate = audit_actions_and_forecast_columns(out, cfg)
    phase = "smoke" if smoke else "formal"
    audit_path = out / "audits" / f"{task_prefix(cfg)}_{phase}_action_forecast_audit.csv"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    manifest = write_manifest(out, cfg_path, prompt, cfg, pf, f"completed_{phase}")
    record = write_record(cfg, pf, copied, audit, gate, "2K smoke" if smoke else "100K formal", obs_smoke, calendar_audit, smoke_verification)
    result = {
        **pf,
        "mode": f"{phase}_train_and_validate",
        "output_root": rel(out),
        "manifest": rel(manifest),
        "record_md": rel(record),
        "observation_smoke_csv": rel(obs_smoke),
        "forecast_calendar_audit_csv": rel(calendar_audit),
        "action_forecast_audit_path": rel(audit_path),
        "action_forecast_gate": gate,
    }
    if smoke:
        result["smoke_gate"] = gate
        result["next_step_allowed"] = gate["next_step_allowed"]
        result_path = out / f"{task_prefix(cfg)}_smoke_result.json"
    else:
        result["smoke_verification"] = smoke_verification
        result_path = out / f"{task_prefix(cfg)}_formal_result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def cli(default_config: Path, default_prompt: Path, expected_task_id: str) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--dry-run", action="store_true")
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--smoke", action="store_true")
    phase.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    cfg_path = args.config if args.config.is_absolute() else (Path.cwd() / args.config).resolve()
    print(json.dumps(run_forecast_experiment(cfg_path, default_prompt, expected_task_id, args.dry_run, args.smoke, args.formal), indent=2, ensure_ascii=False))


__all__ = [
    "EngineeredForecastObservationWrapper",
    "FORECAST_FEATURE_NAMES",
    "RAW_FORECAST_FEATURE_NAMES",
    "cli",
    "run_forecast_experiment",
]
