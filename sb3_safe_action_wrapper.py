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
        lower_bounds = []
        for name in self.formator.action_names:
            low, _ = self.formator._get_action_bounds(name)
            lower_bounds.append(low)
        lower_bounds_array = np.asarray(lower_bounds, dtype=np.float32)
        action_array = np.asarray(self.formator._check_array_actions(action), dtype=np.float32)
        safe_action = lower_bounds_array + 0.5 * (action_array + 1.0) * (self._safe_highs - lower_bounds_array)
        safe_action = np.clip(safe_action, lower_bounds_array, self._safe_highs)
        formatted_action = self.formator.format_actions(safe_action)
        self._last_raw_action = dict(zip(self.formator.action_names, action_array.tolist()))
        self._last_safe_action = formatted_action.copy()

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
