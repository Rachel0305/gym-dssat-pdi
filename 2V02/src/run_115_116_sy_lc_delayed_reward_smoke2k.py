"""115/116 compact forecast plus a small seven-step delayed stress reward."""

from __future__ import annotations

import argparse
import copy
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
from run_113_114_sy_lc_compact_forecast_smoke2k import norm_compact, raw_compact

CONFIGS = {"sy": ROOT / "configs/115_00_sya_originIC_forecast_delayed_reward_smoke2k.json", "lc": ROOT / "configs/116_00_lca_lowIC_forecast_delayed_reward_smoke2k.json"}
PROMPTS = {"sy": ROOT / "prompts/2026-08-15_sy_forecast_delayed_reward_smoke2k.md", "lc": ROOT / "prompts/2026-08-15_lc_forecast_delayed_reward_smoke2k.md"}
EXPECTED_IDS = {"sy": "115_00", "lc": "116_00"}


class SevenDayDelayedStressRewardWrapper(gym.Env):
    """Adds one delayed, forecast-conditioned stress-improvement term."""

    def __init__(self, env: gym.Env, cfg: dict[str, Any]):
        super().__init__()
        self.env = env
        self.cfg = cfg
        intervention = cfg["reward_intervention"]
        self.coef = float(intervention["coef"])
        self.delay_steps = int(intervention["delay_steps"])
        self.pending: list[dict[str, Any]] = []
        self.last_obs: Any | None = None
        self.last_delayed_bonus = 0.0
        self.last_delayed_records = 0
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = getattr(env, "metadata", {})

    def _scalar(self, value: Any, default: float = 0.0) -> float:
        try:
            value = np.asarray(value, dtype=float).reshape(-1)
            return float(value[0]) if len(value) else default
        except Exception:
            return default

    def _current_swfac(self) -> float:
        latest = forecast.direct_ppo.latest_observation_dict(self.env, self.last_obs, {})
        return self._scalar(latest.get("swfac", 0.0), 0.0)

    def _dry_risk(self) -> float:
        features = dict(getattr(self.env, "last_norm_forecast_features", {}) or {})
        rain = float(np.clip(features.get("rain_next7_norm", 0.0), 0.0, 1.0))
        heat = float(np.clip((features.get("tmax_mean_next7_norm", 0.0) + 1.0) / 2.0, 0.0, 1.0))
        dry_flag = float(np.clip(features.get("dry_spell_next7_flag", 0.0), 0.0, 1.0))
        return float(np.clip(0.6 * (1.0 - rain) + 0.2 * heat + 0.2 * dry_flag, 0.0, 1.0))

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.pending = []
        self.last_obs = obs
        self.last_delayed_bonus = 0.0
        self.last_delayed_records = 0
        return obs, dict(info or {})

    def step(self, action):
        before_swfac = self._current_swfac()
        dry_risk = self._dry_risk()
        inner_obs, reward, terminated, truncated, info = self.env.step(action)
        after_swfac = self._current_swfac() if inner_obs is None else self._scalar(forecast.direct_ppo.latest_observation_dict(self.env, inner_obs, info or {}).get("swfac", 0.0), 0.0)
        action_info = dict(getattr(self.env, "last_action_info", {}) or {})
        safe_i = self._scalar(action_info.get("safe_action_amir", 0.0), 0.0)
        bonus = 0.0
        completed = 0
        for record in self.pending:
            record["age"] += 1
            record["future_swfac"].append(after_swfac)
        still_pending: list[dict[str, Any]] = []
        for record in self.pending:
            if record["age"] >= self.delay_steps:
                improvement = max(float(np.mean(record["future_swfac"])) - record["before_swfac"], 0.0)
                if record["safe_i"] > 0.0 and record["dry_risk"] > 0.0:
                    bonus += self.coef * record["dry_risk"] * improvement
                completed += 1
            else:
                still_pending.append(record)
        self.pending = still_pending
        if safe_i > 0.0:
            self.pending.append({"age": 0, "before_swfac": before_swfac, "dry_risk": dry_risk, "safe_i": safe_i, "future_swfac": []})
        self.last_delayed_bonus = float(bonus)
        self.last_delayed_records = int(completed)
        info = dict(info or {})
        info["forecast_delayed_reward_bonus"] = float(bonus)
        info["forecast_delayed_reward_completed_records"] = int(completed)
        info["forecast_delayed_reward_coef"] = float(self.coef)
        self.last_obs = inner_obs
        return inner_obs, float(reward) + float(bonus), terminated, truncated, info

    def action_masks(self):
        return self.env.action_masks()

    @property
    def last_action_info(self):
        info = dict(getattr(self.env, "last_action_info", {}) or {})
        info["forecast_delayed_reward_bonus"] = float(self.last_delayed_bonus)
        info["forecast_delayed_reward_completed_records"] = int(self.last_delayed_records)
        return info

    def close(self):
        return self.env.close()

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.env, name)


def patch_contract() -> None:
    forecast.FORECAST_FEATURE_NAMES = ["rain_next7_norm", "tmax_mean_next7_norm", "dry_spell_next7_flag"]
    forecast.RAW_FORECAST_FEATURE_NAMES = ["rain_next7_mm", "tmax_mean_next7_c", "dry_spell_next7_flag"]
    forecast.raw_forecast_features_for_date = raw_compact
    forecast.normalize_forecast_features = norm_compact
    original_validate = forecast.validate_forecast_config

    def validate_delayed(cfg: dict[str, Any], expected_task_id: str | None = None) -> dict[str, Any]:
        if cfg.get("observation_contract", {}).get("weather_forecast_mode") == "rolling_7d_compact_perfect_hindcast":
            checked = copy.deepcopy(cfg)
            checked["observation_contract"]["weather_forecast_mode"] = "engineered_perfect_hindcast_lookahead"
            result = original_validate(checked, expected_task_id=expected_task_id)
            result["observation_contract"]["weather_forecast_mode"] = "rolling_7d_compact_perfect_hindcast"
            return result
        return original_validate(cfg, expected_task_id=expected_task_id)

    forecast.validate_forecast_config = validate_delayed
    original_factory = forecast.make_forecast_env_factory

    def delayed_factory(base_make_env, exp_cfg):
        base_factory = original_factory(base_make_env, exp_cfg)

        def make_env(runtime_config, env_config, station, year, seed, run_tag, evaluation=False):
            return SevenDayDelayedStressRewardWrapper(base_factory(runtime_config, env_config, station, year, seed, run_tag, evaluation=evaluation), exp_cfg)

        return make_env

    forecast.make_forecast_env_factory = delayed_factory


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
