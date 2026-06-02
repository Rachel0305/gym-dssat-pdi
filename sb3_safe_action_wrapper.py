from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sb3_wrapper import GymDssatWrapper


@dataclass(frozen=True)
class SafeActionCaps:
    anfer: float | None = 60.0
    amir: float | None = 20.0

    def as_dict(self) -> dict[str, float]:
        caps: dict[str, float] = {}
        if self.anfer is not None:
            caps["anfer"] = float(self.anfer)
        if self.amir is not None:
            caps["amir"] = float(self.amir)
        return caps


class SafeActionGymDssatWrapper(GymDssatWrapper):
    """Gym wrapper that clips real DSSAT actions without changing the DSSAT environment."""

    def __init__(self, env, action_caps: SafeActionCaps | None = None):
        self.action_caps = action_caps or SafeActionCaps()
        super().__init__(env)
        self._safe_highs = self._build_safe_highs()
        self._last_raw_action = None
        self._last_safe_action = None

    def _build_safe_highs(self) -> np.ndarray:
        caps = self.action_caps.as_dict()
        highs = []
        for name in self.formator.action_names:
            low, high = self.formator._get_action_bounds(name)
            cap = caps.get(name)
            if cap is None:
                highs.append(high)
            else:
                highs.append(min(float(cap), high))
            if highs[-1] < low:
                raise ValueError(f"Safe cap for {name}={highs[-1]} is below action lower bound {low}.")
        return np.asarray(highs, dtype=np.float32)

    def step(self, action):
        safe_action = self._safe_scale_action(action)
        formatted_action = self.formator.format_actions(safe_action)
        self._last_safe_action = formatted_action.copy()

        return self._step_with_formatted_action(action, formatted_action)

    def _safe_scale_action(self, action) -> np.ndarray:
        lower_bounds = []
        for name in self.formator.action_names:
            low, _ = self.formator._get_action_bounds(name)
            lower_bounds.append(low)
        lower_bounds_array = np.asarray(lower_bounds, dtype=np.float32)
        action_array = np.asarray(self.formator._check_array_actions(action), dtype=np.float32)
        safe_action = lower_bounds_array + 0.5 * (action_array + 1.0) * (self._safe_highs - lower_bounds_array)
        return np.clip(safe_action, lower_bounds_array, self._safe_highs)

    def _step_with_formatted_action(self, action, formatted_action: dict):
        action_array = np.asarray(self.formator._check_array_actions(action), dtype=np.float32)
        self._last_raw_action = dict(zip(self.formator.action_names, action_array.tolist()))

        result = self.env.step(formatted_action)

        if result is None or (isinstance(result, tuple) and result[0] is None):
            return self._last_obs, 0.0, True, False, self._last_info

        if len(result) == 4:
            obs, reward, done, info = result
            truncated = False
        else:
            obs, reward, done, truncated, info = result

        obs = np.asarray(self.formator.format_observation(obs), dtype=np.float32)
        self._last_obs = obs
        self._last_info = info if info is not None else {}
        self._last_info["raw_action"] = self._last_raw_action
        self._last_info["safe_action"] = self._last_safe_action
        reward = float(reward) if reward is not None else 0.0

        return obs, reward, bool(done), bool(truncated), self._last_info


@dataclass(frozen=True)
class SeasonalBudgetCaps:
    anfer: float | None = 220.0
    amir: float | None = 100.0

    def as_dict(self) -> dict[str, float]:
        budgets: dict[str, float] = {}
        if self.anfer is not None:
            budgets["anfer"] = float(self.anfer)
        if self.amir is not None:
            budgets["amir"] = float(self.amir)
        return budgets


class BudgetedSafeActionGymDssatWrapper(SafeActionGymDssatWrapper):
    """Safe daily caps plus seasonal action budgets."""

    def __init__(
        self,
        env,
        action_caps: SafeActionCaps | None = None,
        seasonal_budgets: SeasonalBudgetCaps | None = None,
    ):
        self.seasonal_budgets = seasonal_budgets or SeasonalBudgetCaps()
        self._budget_used: dict[str, float] = {}
        super().__init__(env, action_caps)
        self._budget_limits = self.seasonal_budgets.as_dict()
        self._reset_budget()

    def _reset_budget(self) -> None:
        self._budget_used = {name: 0.0 for name in self.formator.action_names}

    def reset(self, *, seed=None, options=None):
        self._reset_budget()
        return super().reset(seed=seed, options=options)

    def _apply_budget(self, safe_action: np.ndarray) -> np.ndarray:
        budgeted_action = safe_action.copy()
        for index, name in enumerate(self.formator.action_names):
            limit = self._budget_limits.get(name)
            if limit is None:
                continue
            remaining = max(0.0, limit - self._budget_used.get(name, 0.0))
            budgeted_action[index] = min(float(budgeted_action[index]), remaining)
        return budgeted_action

    def step(self, action):
        safe_action = self._safe_scale_action(action)
        budgeted_action = self._apply_budget(safe_action)
        formatted_action = self.formator.format_actions(budgeted_action)
        for name, value in formatted_action.items():
            if name in self._budget_limits:
                self._budget_used[name] = self._budget_used.get(name, 0.0) + float(value)
        self._last_safe_action = formatted_action.copy()

        obs, reward, done, truncated, info = self._step_with_formatted_action(action, formatted_action)
        info["seasonal_budget_used"] = self._budget_used.copy()
        info["seasonal_budget_limits"] = self._budget_limits.copy()
        return obs, reward, done, truncated, info
