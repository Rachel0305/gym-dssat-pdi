from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import gymnasium as gym


@dataclass
class ActionSafetyState:
    cumulative_irrigation: float = 0.0
    cumulative_n: float = 0.0
    last_irrigation_dap: int | None = None
    last_fertilization_dap: int | None = None


@dataclass
class ActionSafetyResult:
    raw_real_action: dict[str, float]
    safe_real_action: dict[str, float]
    action_clipped_amir: bool
    action_clipped_anfer: bool
    season_irrigation_so_far: float
    season_n_so_far: float
    safety_rule_triggered: str


def _range_tuple(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, str):
        left, right = value.replace(" ", "").split("-")
        return int(left), int(right)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    return default


def denormalize_action(action_names: list[str], action_space_dict: Any, normalized_action) -> dict[str, float]:
    spaces = getattr(action_space_dict, "spaces", action_space_dict)
    arr = np.asarray(normalized_action, dtype=float).flatten()
    result: dict[str, float] = {}
    for name, value in zip(action_names, arr):
        space = spaces[name]
        low = float(np.asarray(space.low).flatten()[0])
        high = float(np.asarray(space.high).flatten()[0])
        result[name] = low + 0.5 * (float(value) + 1.0) * (high - low)
    return result


def normalize_action(action_names: list[str], action_space_dict: Any, real_action: dict[str, float]) -> np.ndarray:
    spaces = getattr(action_space_dict, "spaces", action_space_dict)
    values: list[float] = []
    for name in action_names:
        space = spaces[name]
        low = float(np.asarray(space.low).flatten()[0])
        high = float(np.asarray(space.high).flatten()[0])
        value = float(real_action.get(name, 0.0))
        values.append(2.0 * ((value - low) / (high - low)) - 1.0)
    return np.asarray(values, dtype=np.float32)


def apply_action_safety(
    raw_real_action: dict[str, float],
    dap: int,
    state: ActionSafetyState,
    safety_config: dict,
) -> ActionSafetyResult:
    enabled = bool(safety_config.get("enabled", False))
    if not enabled:
        return ActionSafetyResult(
            raw_real_action=dict(raw_real_action),
            safe_real_action=dict(raw_real_action),
            action_clipped_amir=False,
            action_clipped_anfer=False,
            season_irrigation_so_far=state.cumulative_irrigation,
            season_n_so_far=state.cumulative_n,
            safety_rule_triggered="",
        )

    safe = {name: max(0.0, float(value)) for name, value in raw_real_action.items()}
    triggers: list[str] = []

    daily_irrigation_max = float(safety_config.get("daily_irrigation_max", 40.0))
    daily_n_max = float(safety_config.get("daily_n_max", 80.0))
    season_irrigation_soft_limit = float(safety_config.get("season_irrigation_soft_limit", 200.0))
    season_n_soft_limit = float(safety_config.get("season_n_soft_limit", 300.0))
    min_days_between_irrigation = int(safety_config.get("min_days_between_irrigation", 7))
    min_days_between_fertilization = int(safety_config.get("min_days_between_fertilization", 10))
    irrigation_range = _range_tuple(safety_config.get("irrigation_allowed_dap_range"), (1, 120))
    fertilization_range = _range_tuple(safety_config.get("fertilization_allowed_dap_range"), (1, 90))

    raw_amir = safe.get("amir", 0.0)
    raw_anfer = safe.get("anfer", 0.0)

    if not (irrigation_range[0] <= dap <= irrigation_range[1]):
        if raw_amir > 0:
            triggers.append("amir_dap_range")
        safe["amir"] = 0.0
    if not (fertilization_range[0] <= dap <= fertilization_range[1]):
        if raw_anfer > 0:
            triggers.append("anfer_dap_range")
        safe["anfer"] = 0.0

    if state.last_irrigation_dap is not None and dap - state.last_irrigation_dap < min_days_between_irrigation:
        if safe.get("amir", 0.0) > 0:
            triggers.append("amir_min_interval")
        safe["amir"] = 0.0
    if state.last_fertilization_dap is not None and dap - state.last_fertilization_dap < min_days_between_fertilization:
        if safe.get("anfer", 0.0) > 0:
            triggers.append("anfer_min_interval")
        safe["anfer"] = 0.0

    if state.cumulative_irrigation >= season_irrigation_soft_limit:
        if safe.get("amir", 0.0) > 0:
            triggers.append("amir_season_limit")
        safe["amir"] = 0.0
    else:
        allowed = max(0.0, season_irrigation_soft_limit - state.cumulative_irrigation)
        if safe.get("amir", 0.0) > allowed:
            triggers.append("amir_season_clip")
        safe["amir"] = min(safe.get("amir", 0.0), allowed)

    if state.cumulative_n >= season_n_soft_limit:
        if safe.get("anfer", 0.0) > 0:
            triggers.append("anfer_season_limit")
        safe["anfer"] = 0.0
    else:
        allowed = max(0.0, season_n_soft_limit - state.cumulative_n)
        if safe.get("anfer", 0.0) > allowed:
            triggers.append("anfer_season_clip")
        safe["anfer"] = min(safe.get("anfer", 0.0), allowed)

    if safe.get("amir", 0.0) > daily_irrigation_max:
        triggers.append("amir_daily_max")
        safe["amir"] = daily_irrigation_max
    if safe.get("anfer", 0.0) > daily_n_max:
        triggers.append("anfer_daily_max")
        safe["anfer"] = daily_n_max

    action_clipped_amir = abs(safe.get("amir", 0.0) - raw_real_action.get("amir", 0.0)) > 1e-9
    action_clipped_anfer = abs(safe.get("anfer", 0.0) - raw_real_action.get("anfer", 0.0)) > 1e-9

    return ActionSafetyResult(
        raw_real_action=dict(raw_real_action),
        safe_real_action=safe,
        action_clipped_amir=action_clipped_amir,
        action_clipped_anfer=action_clipped_anfer,
        season_irrigation_so_far=state.cumulative_irrigation,
        season_n_so_far=state.cumulative_n,
        safety_rule_triggered=";".join(dict.fromkeys(triggers)),
    )


def update_action_safety_state(state: ActionSafetyState, safe_real_action: dict[str, float], dap: int) -> None:
    amir = float(safe_real_action.get("amir", 0.0))
    anfer = float(safe_real_action.get("anfer", 0.0))
    if amir > 0:
        state.last_irrigation_dap = int(dap)
    if anfer > 0:
        state.last_fertilization_dap = int(dap)
    state.cumulative_irrigation += amir
    state.cumulative_n += anfer


class SafeActionWrapper(gym.Env):
    """A light wrapper that clips PPO actions before they reach DSSAT."""

    def __init__(self, env, safety_config: dict):
        super().__init__()
        self.env = env
        self.safety_config = safety_config
        self.state = ActionSafetyState()
        self.last_safety_result: ActionSafetyResult | None = None
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = getattr(env, "metadata", {})

    def reset(self, *args, **kwargs):
        self.state = ActionSafetyState()
        self.last_safety_result = None
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        latest = self.latest_observation_dict()
        dap = int(round(float(latest.get("dap", 0) or 0)))
        raw_real = denormalize_action(self.env.formator.action_names, self.env.formator.action_space_dict, action)
        result = apply_action_safety(raw_real, dap, self.state, self.safety_config)
        safe_norm = normalize_action(self.env.formator.action_names, self.env.formator.action_space_dict, result.safe_real_action)
        obs, reward, terminated, truncated, info = self.env.step(safe_norm)
        update_action_safety_state(self.state, result.safe_real_action, dap)
        self.last_safety_result = result
        return obs, reward, terminated, truncated, info

    def latest_observation_dict(self) -> dict:
        history = getattr(self.env.unwrapped, "history", {})
        if isinstance(history, dict):
            observations = history.get("observation", [])
            if observations and isinstance(observations[-1], dict):
                return observations[-1]
        return {}

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name):
        return getattr(self.env, name)
