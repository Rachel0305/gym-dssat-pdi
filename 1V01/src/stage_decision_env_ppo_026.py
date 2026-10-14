from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from run_sy2014_stage_mc_dqn_seed1_short_022_02 import FixedObservationScaler, make_env
from stage_based_dqn_core_022 import (
    ACTION_TABLE_9,
    EXECUTABLE_STAGE_DAPS,
    FEASIBILITY_BONUS,
    IRRIGATION_BUDGET,
    NITROGEN_BUDGET,
    NULL_YIELD,
    execute_stage_action,
    valid_action_indices,
)
from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar


class StageDecisionEnv026(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, run_dir: Path, scaler_path: Path, seed: int = 0) -> None:
        super().__init__()
        self.raw_env = make_env(run_dir, seed=seed)
        self.scaler = FixedObservationScaler(scaler_path)
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(25,), dtype=np.float32)
        self.obs: Any = None
        self.state: dict[str, Any] = {}
        self.stage_index = 0
        self.used_i = 0.0
        self.used_n = 0.0
        self.executed = []
        self.stage_rows: list[dict[str, Any]] = []
        self.last_result: dict[str, Any] | None = None

    def _dap(self) -> int:
        value = scalar(self.state.get("dap"))
        if value is None:
            raise RuntimeError("DSSAT state has no DAP")
        return int(round(float(value)))

    def _noop(self) -> np.ndarray:
        return normalize_action(
            self.raw_env.formator.action_names,
            self.raw_env.formator.action_space_dict,
            {"amir": 0.0, "anfer": 0.0},
        )

    def _raw_action(self, irrigation: float, nitrogen: float) -> np.ndarray:
        return normalize_action(
            self.raw_env.formator.action_names,
            self.raw_env.formator.action_space_dict,
            {"amir": irrigation, "anfer": nitrogen},
        )

    def _scaled_obs(self) -> np.ndarray:
        return self.scaler.transform(self.obs, self.state).astype(np.float32)

    def action_masks(self) -> np.ndarray:
        dap = EXECUTABLE_STAGE_DAPS[self.stage_index]
        remaining_i = IRRIGATION_BUDGET - self.used_i
        remaining_n = NITROGEN_BUDGET - self.used_n
        mask = np.zeros(9, dtype=bool)
        for index in valid_action_indices(dap):
            request = ACTION_TABLE_9[index]
            if request["amir"] <= remaining_i + 1e-9 and request["anfer"] <= remaining_n + 1e-9:
                mask[index] = True
        if not mask[0]:
            raise RuntimeError("No-op must always remain valid")
        return mask

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.obs, info = self.raw_env.reset()
        self.state = latest_observation_dict(self.raw_env, self.obs, info)
        while self._dap() == 0:
            self.obs, _, terminated, truncated, info = self.raw_env.step(self._noop())
            if terminated or truncated:
                raise RuntimeError("Season ended during DAP0 initialization")
            self.state = latest_observation_dict(self.raw_env, self.obs, info)
        if self._dap() != EXECUTABLE_STAGE_DAPS[0]:
            raise RuntimeError(f"Expected DAP1, got {self._dap()}")
        self.stage_index = 0
        self.used_i = self.used_n = 0.0
        self.executed = []
        self.stage_rows = []
        self.last_result = None
        return self._scaled_obs(), {"dap": self._dap(), "action_mask": self.action_masks().copy()}

    def step(self, action_index: int):
        action_index = int(action_index)
        mask = self.action_masks()
        if action_index < 0 or action_index >= 9 or not mask[action_index]:
            raise ValueError(f"Masked/invalid action {action_index} at DAP {self._dap()}")
        stage_dap = EXECUTABLE_STAGE_DAPS[self.stage_index]
        if self._dap() != stage_dap:
            raise RuntimeError(f"Expected DAP{stage_dap}, got {self._dap()}")
        action = execute_stage_action(action_index, stage_dap, self.used_i, self.used_n)
        self.executed.append(action)
        self.used_i += action.executed_irrigation
        self.used_n += action.executed_nitrogen
        self.stage_rows.append({"stage_index": self.stage_index, "dap": stage_dap, **asdict(action)})
        reward = -(action.executed_irrigation + 5.0 * action.executed_nitrogen) / 1000.0
        self.obs, _, terminated, truncated, info = self.raw_env.step(
            self._raw_action(action.executed_irrigation, action.executed_nitrogen)
        )
        self.state = latest_observation_dict(self.raw_env, self.obs, info)
        next_stage = EXECUTABLE_STAGE_DAPS[self.stage_index + 1] if self.stage_index + 1 < 6 else None
        while not (terminated or truncated) and next_stage is not None and self._dap() < next_stage:
            self.obs, _, terminated, truncated, info = self.raw_env.step(self._noop())
            self.state = latest_observation_dict(self.raw_env, self.obs, info)
        if next_stage is not None and not (terminated or truncated) and self._dap() != next_stage:
            raise RuntimeError(f"DSSAT skipped DAP{next_stage}; current={self._dap()}")
        self.stage_index += 1
        done = bool(terminated or truncated)
        if self.stage_index == 6 and not done:
            while not (terminated or truncated):
                self.obs, _, terminated, truncated, info = self.raw_env.step(self._noop())
                self.state = latest_observation_dict(self.raw_env, self.obs, info)
            done = True
        if done:
            if self.stage_index != 6:
                raise RuntimeError("Season ended before six stage decisions")
            final_yield = float(scalar(self.state.get("grnwt")) or 0.0)
            final_biomass = float(scalar(self.state.get("topwt")) or 0.0)
            terminal = max(0.0, final_yield - NULL_YIELD)
            if final_yield >= 11077.0:
                terminal += FEASIBILITY_BONUS
            reward += terminal / 1000.0
            self.last_result = {
                "final_yield": final_yield,
                "final_biomass": final_biomass,
                "irrigation_total": self.used_i,
                "nitrogen_total": self.used_n,
                "stage_rows": self.stage_rows,
                "executed_actions": self.executed,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
            obs_out = np.zeros(25, dtype=np.float32)
            info_out = {**self.last_result, "terminal_observation_available": False}
        else:
            obs_out = self._scaled_obs()
            info_out = {"dap": self._dap(), "action_mask": self.action_masks().copy()}
        return obs_out, float(reward), bool(terminated), bool(truncated), info_out

    def close(self) -> None:
        self.raw_env.close()

