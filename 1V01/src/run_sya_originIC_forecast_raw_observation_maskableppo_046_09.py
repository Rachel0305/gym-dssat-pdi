"""046_09: originIC PPO with raw perfect-forecast observation features.

This is deliberately not a continuation of 046_07.  It does not normalize the
existing observation vector.  It only appends five raw weather/forecast
features to the 046_02 binary-timing PPO setup.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine
import run_sya_ppo_configured_046_02 as base04602


DEFAULT_CONFIG = ROOT / "configs" / "046_09_sya_originIC_binary_timing_forecast_raw_observation_ppo.json"
PROMPT = ROOT / "prompts" / "046_09_sya_originIC_forecast_raw_observation_ppo.md"
WEATHER_FEATURE_NAMES = ["rain_today_mm", "tmin_today_c", "rain_past7_mm", "rain_future7_mm", "tmean_future7_c"]
BASE_OBSERVATION_DIM = 25
ENHANCED_OBSERVATION_DIM = BASE_OBSERVATION_DIM + len(WEATHER_FEATURE_NAMES)
ORIGINAL_BASE_MAKE_ENV = engine.base03222.base.make_env


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_config(path: Path) -> dict[str, Any]:
    cfg = base04602.read_config(path)
    wf = cfg.get("weather_forecast_observation_04609", {})
    if not bool(wf.get("enabled", False)):
        raise ValueError("046_09 requires weather_forecast_observation_04609.enabled=true")
    if bool(wf.get("normalization_enabled", True)):
        raise ValueError("046_09 is the no-normalization forecast branch; normalization_enabled must be false")
    if list(wf.get("feature_names", [])) != WEATHER_FEATURE_NAMES:
        raise ValueError("046_09 weather feature names do not match the code contract")
    return cfg


def output_root(cfg: dict[str, Any]) -> Path:
    return ROOT / "benchmark_results" / f"{cfg['task_id']}_{cfg['task_name']}"


def _weather_table(config: dict[str, Any], station: str) -> pd.DataFrame:
    weather = direct_ppo.weather_for_daily(config)
    weather = weather[weather["station_code"].astype(str).eq(str(station))].copy()
    weather["date"] = pd.to_datetime(weather["date"])
    for col in ["rain", "tmin", "tmax"]:
        weather[col] = pd.to_numeric(weather[col], errors="coerce")
    return weather.sort_values("date").reset_index(drop=True)


class RawForecastObservationWrapper(gym.Env):
    """Append raw weather and perfect-forecast features without normalization."""

    def __init__(self, env: gym.Env, config: dict[str, Any], env_config: dict[str, Any], station: str, year: int):
        super().__init__()
        self.env = env
        self.config = config
        self.env_config = env_config
        self.station = str(station)
        self.year = int(year)
        self.weather = _weather_table(config, self.station)
        self.planting = pd.Timestamp(direct_ppo.find_year(env_config, self.station, self.year)["planting_date"])
        self.action_space = env.action_space
        self.metadata = getattr(env, "metadata", {})
        base_space = env.observation_space
        base_low = np.asarray(base_space.low, dtype=np.float32).reshape(-1)
        base_high = np.asarray(base_space.high, dtype=np.float32).reshape(-1)
        if len(base_low) != BASE_OBSERVATION_DIM:
            raise RuntimeError(f"Expected base obs dim {BASE_OBSERVATION_DIM}, got {len(base_low)}")
        weather_low = np.array([0.0, -40.0, 0.0, 0.0, -40.0], dtype=np.float32)
        weather_high = np.array([300.0, 60.0, 500.0, 500.0, 60.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.concatenate([base_low, weather_low]).astype(np.float32),
            high=np.concatenate([base_high, weather_high]).astype(np.float32),
            dtype=np.float32,
        )
        self.last_weather_features: dict[str, float] = {}
        self.last_raw_enhanced_obs: np.ndarray | None = None

    def _date_from_obs(self, obs: np.ndarray, info: dict | None = None) -> pd.Timestamp:
        latest = direct_ppo.latest_observation_dict(self.env, obs, info or {})
        dap_raw = direct_ppo.scalar(latest.get("dap", np.nan), np.nan)
        if np.isfinite(dap_raw) and dap_raw > 0:
            return self.planting + pd.Timedelta(days=int(round(float(dap_raw))) - 1)
        return self.planting

    def _features_for_date(self, date: pd.Timestamp) -> np.ndarray:
        cur = self.weather[self.weather["date"].eq(date)]
        rain_today = float(cur["rain"].iloc[0]) if not cur.empty and pd.notna(cur["rain"].iloc[0]) else 0.0
        tmin_today = float(cur["tmin"].iloc[0]) if not cur.empty and pd.notna(cur["tmin"].iloc[0]) else 0.0
        past = self.weather[(self.weather["date"] >= date - pd.Timedelta(days=6)) & (self.weather["date"] <= date)]
        future = self.weather[(self.weather["date"] >= date) & (self.weather["date"] <= date + pd.Timedelta(days=6))]
        rain_past7 = float(past["rain"].sum(skipna=True)) if not past.empty else 0.0
        rain_future7 = float(future["rain"].sum(skipna=True)) if not future.empty else 0.0
        if not future.empty:
            tmean = (future["tmax"] + future["tmin"]) / 2.0
            tmean_future7 = float(tmean.mean(skipna=True)) if tmean.notna().any() else 0.0
        else:
            tmean_future7 = 0.0
        values = np.array([rain_today, tmin_today, rain_past7, rain_future7, tmean_future7], dtype=np.float32)
        self.last_weather_features = {name: float(value) for name, value in zip(WEATHER_FEATURE_NAMES, values)}
        return values

    def _augment(self, obs: np.ndarray, info: dict | None = None) -> np.ndarray:
        flat = np.asarray(obs, dtype=np.float32).reshape(-1)
        if len(flat) != BASE_OBSERVATION_DIM:
            raise RuntimeError(f"Expected raw obs len {BASE_OBSERVATION_DIM}, got {len(flat)}")
        raw = np.concatenate([flat, self._features_for_date(self._date_from_obs(flat, info))]).astype(np.float32)
        if not np.isfinite(raw).all():
            raise RuntimeError("046_09 enhanced observation contains NaN or Inf")
        self.last_raw_enhanced_obs = raw
        return raw

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        info = dict(info or {})
        return self._augment(obs, info), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(obs, info), reward, terminated, truncated, dict(info or {})

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()

    @property
    def last_action_info(self) -> dict[str, Any]:
        base_info = getattr(self.env, "last_action_info", {})
        info = dict(base_info) if isinstance(base_info, dict) else {}
        for key, value in self.last_weather_features.items():
            info[f"obs_{key}"] = float(value)
        info["enhanced_observation_dim_04609"] = ENHANCED_OBSERVATION_DIM
        info["weather_forecast_raw_04609_enabled"] = True
        info["observation_normalization_04609_enabled"] = False
        return info

    def close(self):
        return self.env.close()

    def render(self):
        return self.env.render()

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.env, name)


def make_env_04609(config: dict[str, Any], env_config: dict[str, Any], station: str, year: int, seed: int, run_tag: str, evaluation: bool = False):
    env = ORIGINAL_BASE_MAKE_ENV(config, env_config, station, year, seed, run_tag, evaluation=evaluation)
    return RawForecastObservationWrapper(env, config, env_config, station, year)


def copy_clean_names(out: Path) -> dict[str, str]:
    mapping = {
        "evaluation/042_10_training_checkpoint_inventory.csv": "evaluation/046_09_training_checkpoint_inventory.csv",
        "evaluation/042_10_checkpoint_validation_summary.csv": "evaluation/046_09_checkpoint_validation_summary.csv",
        "evaluation/042_10_validation_summary_by_station_checkpoint.csv": "evaluation/046_09_validation_summary_by_station_checkpoint.csv",
        "042_10_result.json": "046_09_engine_result.json",
    }
    done: dict[str, str] = {}
    for source_rel, target_rel in mapping.items():
        source, target = out / source_rel, out / target_rel
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            done[target_rel] = source_rel
    return done


def observation_smoke(cfg: dict[str, Any], out: Path) -> Path:
    config = engine.load_config()
    selection = engine.base03222.build_selection(engine.base03222.load_split())
    env_config = direct_ppo.build_env_config(config, selection)
    rows: list[dict[str, Any]] = []
    for year in [2005, 2014, 2023]:
        env = make_env_04609(config, env_config, "SYA", year, int(cfg.get("seed", 0)), f"046_09_obs_smoke_{year}", evaluation=True)
        try:
            obs, _info = env.reset()
            rows.append({
                "year": year,
                "obs_dim": int(np.asarray(obs).reshape(-1).shape[0]),
                "expected_dim": ENHANCED_OBSERVATION_DIM,
                **env.last_weather_features,
            })
        finally:
            env.close()
    path = out / "audits" / "046_09_raw_forecast_observation_smoke.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def run(cfg_path: Path, dry_run: bool = False, smoke: bool = False) -> dict[str, Any]:
    cfg = read_config(cfg_path)
    if smoke:
        cfg = json.loads(json.dumps(cfg))
        cfg["task_name"] = f"{cfg['task_name']}_smoke2k"
        cfg["training"] = {"total_timesteps": 2000, "checkpoint_steps": [1000, 2000]}
    pf = base04602.preflight(cfg)
    out = output_root(cfg)
    out.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return {**pf, "mode": "dry_run", "output_root": rel(out), "enhanced_observation_dim": ENHANCED_OBSERVATION_DIM}
    if not pf["next_step_allowed"]:
        raise RuntimeError("preflight failed: " + "; ".join(pf["issues"]))

    input_root = base04602.INPUT_PROFILES[str(cfg["input_profile"])]
    doc = ROOT / "docs" / f"{cfg['task_id']}_{cfg['task_name']}_record.md"
    checkpoints = list(map(int, cfg["training"]["checkpoint_steps"]))
    total = int(cfg["training"]["total_timesteps"])
    old_values = {name: getattr(engine, name) for name in ["TASK_ID", "TASK_NAME", "BASE_OUT", "BASE_DOC", "PROMPT", "LOWIC_INPUT_ROOT", "STATION", "SITES", "BINARY_IRRIGATION_LEVELS", "BINARY_NITROGEN_LEVELS"]}
    old_make_env = engine.base03222.base.make_env
    old_input_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    try:
        engine.TASK_ID = str(cfg["task_id"])
        engine.TASK_NAME = str(cfg["task_name"])
        engine.BASE_OUT = out
        engine.BASE_DOC = doc
        engine.PROMPT = PROMPT
        engine.LOWIC_INPUT_ROOT = input_root
        engine.STATION = str(cfg["station_code"])
        engine.SITES = [str(cfg["station_code"])]
        engine.BINARY_IRRIGATION_LEVELS = list(map(float, cfg["actions"]["irrigation_levels_mm"]))
        engine.BINARY_NITROGEN_LEVELS = list(map(float, cfg["actions"]["nitrogen_levels_kg_ha"]))
        engine.patch_base_module(out, doc, total, checkpoints)
        engine.base03222.base.make_env = make_env_04609
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = input_root
        obs_smoke = observation_smoke(cfg, out)
        engine.base03222.main()
        copied = copy_clean_names(out)
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root
        engine.base03222.base.make_env = old_make_env
        for name, value in old_values.items():
            setattr(engine, name, value)
    shutil.copy2(cfg_path, out / "configs" / cfg_path.name)
    if PROMPT.exists():
        shutil.copy2(PROMPT, out / "configs" / PROMPT.name)
    record = ROOT / "docs" / f"{cfg['task_id']}_{cfg['task_name']}_record.md"
    record.write_text("\n".join([
        f"# {cfg['task_id']} SYA originIC raw forecast observation PPO 记录",
        "",
        "- 与 046_02 相比，唯一方法改动是 observation 追加 5 个未归一化天气/完美预报变量。",
        "- 不使用 046_07 的 train-stat normalization。",
        f"- observation smoke: `{rel(obs_smoke)}`",
        f"- 训练步数：`{total}`；checkpoint：`{checkpoints}`。",
        f"- 输出目录：`{rel(out)}`。",
    ]) + "\n", encoding="utf-8")
    return {
        **pf,
        "mode": "train_and_validate",
        "output_root": rel(out),
        "record_md": rel(record),
        "observation_smoke_csv": rel(obs_smoke),
        "clean_outputs": copied,
        "enhanced_observation_dim": ENHANCED_OBSERVATION_DIM,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg_path = args.config if args.config.is_absolute() else (Path.cwd() / args.config).resolve()
    print(json.dumps(run(cfg_path, args.dry_run, args.smoke), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
