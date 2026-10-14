"""118 SY forecast-conditioned potential shaping, gated 2K then 5K/10K."""

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
from run_113_114_sy_lc_compact_forecast_smoke2k import patch_contract as patch_compact

CONFIG = ROOT / "configs/118_00_sya_originIC_forecast_potential_shaping_5k10k.json"
PROMPT = ROOT / "prompts/2026-08-15_sy_forecast_potential_shaping_5k10k.md"
BASE_TASK = "118_00"
LONG_MODE = False
SMOKE_TASK_NAME = "sya_originIC_forecast_potential_shaping_smoke2k_rerun1"
LONG_TASK_NAME = "sya_originIC_forecast_potential_shaping_5k10k_rerun1"


class ForecastPotentialShapingWrapper(gym.Env):
    """Add a bounded potential difference using forecast risk and current stress."""

    def __init__(self, env: gym.Env, cfg: dict[str, Any]):
        super().__init__()
        self.env = env
        intervention = cfg["reward_intervention"]
        self.coef = float(intervention["coef"])
        self.gamma = float(intervention["gamma"])
        self.weather_weights = dict(intervention["weather_risk_weights"])
        self.stress_weights = dict(intervention["stress_weights"])
        self.baseline = float(intervention["potential_baseline"])
        self.prev_potential = 0.0
        self.last_potential_before = 0.0
        self.last_potential_after = 0.0
        self.last_shaping_reward = 0.0
        self.last_weather_risk = 0.0
        self.last_crop_stress = 0.0
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = getattr(env, "metadata", {})

    @staticmethod
    def _scalar(value: Any, default: float = 0.0) -> float:
        try:
            arr = np.asarray(value, dtype=float).reshape(-1)
            return float(arr[0]) if len(arr) else default
        except Exception:
            return default

    def _risk_and_stress(self, obs: Any, info: dict | None) -> tuple[float, float]:
        latest = forecast.direct_ppo.latest_observation_dict(self.env, obs, info or {})
        features = dict(getattr(self.env, "last_norm_forecast_features", {}) or {})
        rain = float(np.clip(features.get("rain_next7_norm", 0.0), 0.0, 1.0))
        heat = float(np.clip((features.get("tmax_mean_next7_norm", 0.0) + 1.0) / 2.0, 0.0, 1.0))
        dry = float(np.clip(features.get("dry_spell_next7_flag", 0.0), 0.0, 1.0))
        weather_risk = float(np.clip(
            self.weather_weights["rain_deficit"] * (1.0 - rain)
            + self.weather_weights["heat"] * heat
            + self.weather_weights["dry_spell"] * dry,
            0.0,
            1.0,
        ))
        swfac = float(np.clip(self._scalar(latest.get("swfac", 0.0), 0.0), 0.0, 1.0))
        nstres = float(np.clip(self._scalar(latest.get("nstres", 0.0), 0.0), 0.0, 1.0))
        crop_stress = float(np.clip(self.stress_weights["swfac"] * swfac + self.stress_weights["nstres"] * nstres, 0.0, 1.0))
        return weather_risk, crop_stress

    def _potential(self, obs: Any, info: dict | None) -> float:
        weather_risk, crop_stress = self._risk_and_stress(obs, info)
        self.last_weather_risk = weather_risk
        self.last_crop_stress = crop_stress
        value = -((self.baseline + crop_stress) * weather_risk)
        return float(np.clip(value, -2.0, 0.0))

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.prev_potential = self._potential(obs, info)
        self.last_potential_before = self.prev_potential
        self.last_potential_after = self.prev_potential
        self.last_shaping_reward = 0.0
        return obs, dict(info or {})

    def step(self, action):
        before = float(self.prev_potential)
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        after = 0.0 if done else self._potential(obs, info)
        shaping = float(self.coef * (self.gamma * after - before))
        self.prev_potential = after
        self.last_potential_before = before
        self.last_potential_after = after
        self.last_shaping_reward = shaping
        info = dict(info or {})
        info["forecast_potential_before"] = before
        info["forecast_potential_after"] = after
        info["forecast_potential_shaping_reward"] = shaping
        info["forecast_potential_weather_risk"] = float(self.last_weather_risk)
        info["forecast_potential_crop_stress"] = float(self.last_crop_stress)
        return obs, float(reward) + shaping, terminated, truncated, info

    def action_masks(self):
        return self.env.action_masks()

    @property
    def last_action_info(self):
        info = dict(getattr(self.env, "last_action_info", {}) or {})
        info.update(
            {
                "forecast_potential_before": float(self.last_potential_before),
                "forecast_potential_after": float(self.last_potential_after),
                "forecast_potential_shaping_reward": float(self.last_shaping_reward),
                "forecast_potential_weather_risk": float(self.last_weather_risk),
                "forecast_potential_crop_stress": float(self.last_crop_stress),
            }
        )
        return info

    def close(self):
        return self.env.close()

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.env, name)


def patch_potential_contract() -> None:
    patch_compact()
    original_factory = forecast.make_forecast_env_factory

    def potential_factory(base_make_env, exp_cfg):
        base_factory = original_factory(base_make_env, exp_cfg)

        def make_env(runtime_config, env_config, station, year, seed, run_tag, evaluation=False):
            base = base_factory(runtime_config, env_config, station, year, seed, run_tag, evaluation=evaluation)
            return ForecastPotentialShapingWrapper(base, exp_cfg)

        return make_env

    forecast.make_forecast_env_factory = potential_factory


def patch_stage_schedule(long_mode: bool) -> None:
    global LONG_MODE
    LONG_MODE = bool(long_mode)

    def prepare(cfg: dict[str, Any], smoke: bool) -> dict[str, Any]:
        planned = copy.deepcopy(cfg)
        if smoke:
            if LONG_MODE:
                planned["task_name"] = LONG_TASK_NAME
                planned["training"] = {"total_timesteps": 10000, "checkpoint_steps": [2000, 5000, 10000]}
            else:
                planned["task_name"] = SMOKE_TASK_NAME
                planned["training"] = {"total_timesteps": 2000, "checkpoint_steps": [1000, 2000]}
            return planned
        return planned

    forecast.prepare_config = prepare


def add_checkpoint_metrics(result: dict[str, Any]) -> dict[str, Any]:
    output = ROOT / result["output_root"]
    prefix = BASE_TASK
    summary_path = output / "evaluation" / f"{prefix}_checkpoint_validation_summary.csv"
    if not summary_path.exists():
        return result
    summary = pd.read_csv(summary_path, keep_default_na=False)
    steps = sorted(pd.to_numeric(summary["checkpoint_step"], errors="coerce").dropna().astype(int).unique().tolist())
    metrics: dict[str, Any] = {}
    for step in steps:
        frame = summary[pd.to_numeric(summary["checkpoint_step"], errors="coerce").eq(step)]
        cols = [c for c in ["final_grnwt", "total_irrigation", "total_n", "PFP_N"] if c in frame.columns]
        metrics[str(step)] = frame[cols].apply(pd.to_numeric, errors="coerce").mean().to_dict()
    result["checkpoint_rows"] = {str(step): int((pd.to_numeric(summary["checkpoint_step"], errors="coerce") == step).sum()) for step in steps}
    result["checkpoint_metrics_mean"] = metrics
    (output / f"{BASE_TASK}_checkpoint_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--smoke2k", action="store_true")
    phase.add_argument("--long", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    long_mode = bool(args.long)
    patch_potential_contract()
    patch_stage_schedule(long_mode)
    result = forecast.run_forecast_experiment(CONFIG, PROMPT, BASE_TASK, args.dry_run, True, False)
    result["stage"] = "5k10k" if long_mode else "smoke2k"
    result = add_checkpoint_metrics(result)
    if not args.dry_run:
        output = ROOT / result["output_root"]
        output.mkdir(parents=True, exist_ok=True)
        (output / f"{BASE_TASK}_stage_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
