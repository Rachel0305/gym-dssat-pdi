from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

from ppo_action_safety import denormalize_action, normalize_action


@dataclass
class PriorActionTable:
    path: Path

    def __post_init__(self):
        self.table = pd.read_csv(self.path)
        self.table["station"] = self.table["station"].astype(str)
        self.table["sim_day"] = self.table["sim_day"].astype(int)

    def action(self, station: str, sim_day: int) -> dict[str, float]:
        match = self.table[(self.table["station"].eq(str(station))) & (self.table["sim_day"].eq(int(sim_day)))]
        if len(match):
            return {
                "amir": float(match["real_action_amir"].iloc[0]),
                "anfer": float(match["real_action_anfer"].iloc[0]),
            }
        return {"amir": 0.0, "anfer": 0.0}


class PriorReplayPolicy:
    def __init__(self, env, prior: PriorActionTable, station: str):
        self.env = env
        self.prior = prior
        self.station = station
        self.step_count = 0

    def reset(self):
        self.step_count = 0

    def predict(self, obs, deterministic: bool = True):
        self.step_count += 1
        real = self.prior.action(self.station, self.step_count)
        norm = normalize_action(self.env.formator.action_names, self.env.formator.action_space_dict, real)
        return norm, None


class ImitationPenaltyRewardWrapper(gym.Env):
    def __init__(self, env, prior: PriorActionTable, station: str, lambda_bc: float):
        super().__init__()
        self.env = env
        self.prior = prior
        self.station = station
        self.lambda_bc = float(lambda_bc)
        self.step_count = 0
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = getattr(env, "metadata", {})
        self.last_bc_info: dict[str, Any] = {}

    def reset(self, *args, **kwargs):
        self.step_count = 0
        self.last_bc_info = {}
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        self.step_count += 1
        prior_action = self.prior.action(self.station, self.step_count)
        obs, reward, terminated, truncated, info = self.env.step(action)
        safety_result = getattr(self.env, "last_safety_result", None)
        if safety_result is not None:
            actual = safety_result.safe_real_action
        else:
            actual = denormalize_action(self.env.formator.action_names, self.env.formator.action_space_dict, action)
        diff_i = (float(actual.get("amir", 0.0)) - float(prior_action.get("amir", 0.0))) / 100.0
        diff_n = (float(actual.get("anfer", 0.0)) - float(prior_action.get("anfer", 0.0))) / 150.0
        penalty = self.lambda_bc * (diff_i * diff_i + diff_n * diff_n)
        self.last_bc_info = {
            "bc_prior_amir": float(prior_action.get("amir", 0.0)),
            "bc_prior_anfer": float(prior_action.get("anfer", 0.0)),
            "bc_action_deviation": float((diff_i * diff_i + diff_n * diff_n) ** 0.5),
            "bc_action_penalty": float(penalty),
        }
        out_info = dict(info or {})
        out_info.update(self.last_bc_info)
        return obs, float(reward) - float(penalty), terminated, truncated, out_info

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name):
        return getattr(self.env, name)


class ResidualPriorActionWrapper(gym.Env):
    def __init__(self, env, prior: PriorActionTable, station: str, irrigation_range: float, n_range: float):
        super().__init__()
        self.env = env
        self.prior = prior
        self.station = station
        self.irrigation_range = float(irrigation_range)
        self.n_range = float(n_range)
        self.step_count = 0
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = getattr(env, "metadata", {})
        self.last_residual_info: dict[str, Any] = {}

    def reset(self, *args, **kwargs):
        self.step_count = 0
        self.last_residual_info = {}
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        self.step_count += 1
        arr = np.asarray(action, dtype=float).flatten()
        prior = self.prior.action(self.station, self.step_count)
        full = {
            "amir": max(0.0, float(prior.get("amir", 0.0)) + float(arr[0]) * self.irrigation_range),
            "anfer": max(0.0, float(prior.get("anfer", 0.0)) + float(arr[1]) * self.n_range),
        }
        full_norm = normalize_action(self.env.formator.action_names, self.env.formator.action_space_dict, full)
        obs, reward, terminated, truncated, info = self.env.step(full_norm)
        self.last_residual_info = {
            "bc_prior_amir": float(prior.get("amir", 0.0)),
            "bc_prior_anfer": float(prior.get("anfer", 0.0)),
            "residual_norm_amir": float(arr[0]),
            "residual_norm_anfer": float(arr[1]),
            "residual_real_amir": float(arr[0]) * self.irrigation_range,
            "residual_real_anfer": float(arr[1]) * self.n_range,
        }
        out_info = dict(info or {})
        out_info.update(self.last_residual_info)
        return obs, reward, terminated, truncated, out_info

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name):
        return getattr(self.env, name)
