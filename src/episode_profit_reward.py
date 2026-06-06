from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from ppo_action_safety import denormalize_action


@dataclass(frozen=True)
class ProfitRewardCandidate:
    name: str
    family: str
    grain_value_coef: float
    season_water_cost: float
    season_n_cost: float
    daily_water_cost: float
    daily_n_cost: float
    baseline_passthrough: bool = False


PROFIT_REWARD_CANDIDATES: dict[str, ProfitRewardCandidate] = {
    "P0_current_reward_baseline": ProfitRewardCandidate(
        name="P0_current_reward_baseline",
        family="P0_baseline",
        grain_value_coef=0.0,
        season_water_cost=0.0,
        season_n_cost=0.0,
        daily_water_cost=0.0,
        daily_n_cost=0.0,
        baseline_passthrough=True,
    ),
    "P1_terminal_profit_weak_cost": ProfitRewardCandidate(
        name="P1_terminal_profit_weak_cost",
        family="P_terminal_profit",
        grain_value_coef=0.01,
        season_water_cost=0.1,
        season_n_cost=0.05,
        daily_water_cost=0.0,
        daily_n_cost=0.0,
    ),
    "P2_terminal_profit_medium_cost": ProfitRewardCandidate(
        name="P2_terminal_profit_medium_cost",
        family="P_terminal_profit",
        grain_value_coef=0.01,
        season_water_cost=0.5,
        season_n_cost=0.25,
        daily_water_cost=0.0,
        daily_n_cost=0.0,
    ),
    "P3_terminal_profit_strong_cost": ProfitRewardCandidate(
        name="P3_terminal_profit_strong_cost",
        family="P_terminal_profit",
        grain_value_coef=0.01,
        season_water_cost=1.0,
        season_n_cost=0.5,
        daily_water_cost=0.0,
        daily_n_cost=0.0,
    ),
    "P4_terminal_profit_with_daily_cost": ProfitRewardCandidate(
        name="P4_terminal_profit_with_daily_cost",
        family="P_terminal_plus_daily_cost",
        grain_value_coef=0.01,
        season_water_cost=0.5,
        season_n_cost=0.25,
        daily_water_cost=0.05,
        daily_n_cost=0.02,
    ),
    "P5_lower_grain_value_medium_cost": ProfitRewardCandidate(
        name="P5_lower_grain_value_medium_cost",
        family="P_terminal_plus_daily_cost",
        grain_value_coef=0.005,
        season_water_cost=0.5,
        season_n_cost=0.25,
        daily_water_cost=0.05,
        daily_n_cost=0.02,
    ),
    "P6_high_economic_pressure": ProfitRewardCandidate(
        name="P6_high_economic_pressure",
        family="P_terminal_plus_daily_cost",
        grain_value_coef=0.005,
        season_water_cost=1.0,
        season_n_cost=0.5,
        daily_water_cost=0.1,
        daily_n_cost=0.05,
    ),
}


def candidate_table() -> list[dict[str, Any]]:
    return [asdict(candidate) for candidate in PROFIT_REWARD_CANDIDATES.values()]


def scalar(value: Any, default: float = np.nan) -> float:
    try:
        arr = np.asarray(value).flatten()
        if len(arr) == 0:
            return default
        return float(arr[0])
    except Exception:
        return default


class EpisodeProfitRewardWrapper(gym.Env):
    """Replace step-wise crop reward with a season profit objective.

    The wrapper is intentionally outside the original gym-DSSAT reward function.
    The original callback does not expose a done flag, while this layer sees
    terminated/truncated and can safely add a terminal reward on the last step.
    """

    def __init__(self, env: gym.Env, reward_version: str):
        super().__init__()
        if reward_version not in PROFIT_REWARD_CANDIDATES:
            raise KeyError(f"Unknown profit reward version: {reward_version}")
        self.env = env
        self.candidate = PROFIT_REWARD_CANDIDATES[reward_version]
        self.reward_version = reward_version
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = getattr(env, "metadata", {})
        self.total_irrigation = 0.0
        self.total_n = 0.0
        self.daily_cost_total = 0.0
        self.last_profit_info: dict[str, Any] = {}

    def reset(self, *args, **kwargs):
        self.total_irrigation = 0.0
        self.total_n = 0.0
        self.daily_cost_total = 0.0
        self.last_profit_info = {}
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        amir, anfer = self._last_safe_action(action)
        self.total_irrigation += amir
        self.total_n += anfer

        daily_cost = self.candidate.daily_water_cost * amir + self.candidate.daily_n_cost * anfer
        self.daily_cost_total += daily_cost
        terminal_reward = 0.0
        final_grnwt = scalar(self._latest_observation_dict(obs, info).get("grnwt"), 0.0)
        grain_value_component = 0.0
        water_cost_component = 0.0
        n_cost_component = 0.0

        if self.candidate.baseline_passthrough:
            reward = float(original_reward)
        else:
            reward = -float(daily_cost)
            if done:
                grain_value_component = self.candidate.grain_value_coef * final_grnwt
                water_cost_component = self.candidate.season_water_cost * self.total_irrigation
                n_cost_component = self.candidate.season_n_cost * self.total_n
                terminal_reward = grain_value_component - water_cost_component - n_cost_component
                reward += terminal_reward

        season_profit_score = (
            self.candidate.grain_value_coef * final_grnwt
            - self.candidate.season_water_cost * self.total_irrigation
            - self.candidate.season_n_cost * self.total_n
            - self.daily_cost_total
        )
        self.last_profit_info = {
            "reward_version": self.reward_version,
            "reward_family": self.candidate.family,
            "original_env_reward": float(original_reward) if original_reward is not None else 0.0,
            "profit_reward": float(reward),
            "daily_cost": float(daily_cost),
            "daily_cost_total": float(self.daily_cost_total),
            "terminal_reward": float(terminal_reward),
            "season_profit_score": float(season_profit_score),
            "profit_final_grnwt": float(final_grnwt),
            "profit_total_irrigation": float(self.total_irrigation),
            "profit_total_n": float(self.total_n),
            "grain_value_component": float(grain_value_component),
            "water_cost_component": float(water_cost_component),
            "n_cost_component": float(n_cost_component),
        }
        out_info = dict(info or {})
        out_info.update(self.last_profit_info)
        return obs, float(reward), terminated, truncated, out_info

    def _last_safe_action(self, action) -> tuple[float, float]:
        safety_result = getattr(self.env, "last_safety_result", None)
        if safety_result is not None:
            safe = safety_result.safe_real_action
            return float(safe.get("amir", 0.0)), float(safe.get("anfer", 0.0))
        action_names = self.env.formator.action_names
        action_space_dict = self.env.formator.action_space_dict
        real = denormalize_action(action_names, action_space_dict, action)
        return float(real.get("amir", 0.0)), float(real.get("anfer", 0.0))

    def _latest_observation_dict(self, obs_array, info) -> dict[str, Any]:
        obs_vars = list(getattr(self.unwrapped, "observation_variables", []))
        obs_dict = {name: scalar(value) for name, value in zip(obs_vars, np.asarray(obs_array).flatten())}
        history = getattr(self.unwrapped, "history", {})
        if isinstance(history, dict):
            observations = history.get("observation", [])
            if observations and isinstance(observations[-1], dict):
                obs_dict.update(observations[-1])
        if isinstance(info, dict):
            obs_dict.update(info)
        return obs_dict

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name):
        return getattr(self.env, name)
