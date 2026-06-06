from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from ppo_action_safety import denormalize_action, normalize_action


DEFAULT_FERTILIZATION_WINDOWS = [(1, 7), (25, 40), (55, 70)]
DEFAULT_IRRIGATION_WINDOWS = [(20, 35), (45, 65), (70, 95)]


@dataclass
class ActionDesignResult:
    raw_real_action: dict[str, float]
    design_filtered_action: dict[str, float]
    action_design_rule_triggered: str
    is_decision_day: bool
    is_in_fertilization_window: bool
    is_in_irrigation_window: bool


def _range_tuple(value: Any) -> tuple[int, int]:
    if isinstance(value, str):
        left, right = value.replace(" ", "").split("-")
        return int(left), int(right)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"Invalid DAP range: {value!r}")


def _window_list(value: Any, default: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if value is None:
        return default
    return [_range_tuple(item) for item in value]


def _in_windows(dap: int, windows: list[tuple[int, int]]) -> bool:
    return any(left <= dap <= right for left, right in windows)


class ScheduledActionDesignWrapper(gym.Env):
    """Filter PPO actions before action safety and DSSAT execution."""

    def __init__(
        self,
        env: gym.Env,
        *,
        decision_interval_days: int | None = None,
        fertilization_windows: list[tuple[int, int]] | None = None,
        irrigation_windows: list[tuple[int, int]] | None = None,
        use_phenology_windows: bool = False,
    ):
        super().__init__()
        self.env = env
        self.decision_interval_days = int(decision_interval_days) if decision_interval_days else None
        self.use_phenology_windows = bool(use_phenology_windows)
        self.fertilization_windows = _window_list(fertilization_windows, DEFAULT_FERTILIZATION_WINDOWS)
        self.irrigation_windows = _window_list(irrigation_windows, DEFAULT_IRRIGATION_WINDOWS)
        self.last_design_result: ActionDesignResult | None = None
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = getattr(env, "metadata", {})

    def reset(self, *args, **kwargs):
        self.last_design_result = None
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        dap = self._current_dap()
        action_names = self.env.formator.action_names
        action_space_dict = self.env.formator.action_space_dict
        raw_real = denormalize_action(action_names, action_space_dict, action)
        filtered = {name: max(0.0, float(value)) for name, value in raw_real.items()}
        triggers: list[str] = []

        is_decision_day = True
        if self.decision_interval_days:
            is_decision_day = dap <= 1 or ((dap - 1) % self.decision_interval_days == 0)
            if not is_decision_day:
                if filtered.get("amir", 0.0) > 0:
                    triggers.append("not_decision_day_amir")
                if filtered.get("anfer", 0.0) > 0:
                    triggers.append("not_decision_day_anfer")
                filtered["amir"] = 0.0
                filtered["anfer"] = 0.0

        is_in_irrigation_window = True
        is_in_fertilization_window = True
        if self.use_phenology_windows:
            is_in_irrigation_window = _in_windows(dap, self.irrigation_windows)
            is_in_fertilization_window = _in_windows(dap, self.fertilization_windows)
            if not is_in_irrigation_window and filtered.get("amir", 0.0) > 0:
                triggers.append("outside_irrigation_window")
                filtered["amir"] = 0.0
            if not is_in_fertilization_window and filtered.get("anfer", 0.0) > 0:
                triggers.append("outside_fertilization_window")
                filtered["anfer"] = 0.0

        self.last_design_result = ActionDesignResult(
            raw_real_action=dict(raw_real),
            design_filtered_action=dict(filtered),
            action_design_rule_triggered=";".join(dict.fromkeys(triggers)),
            is_decision_day=bool(is_decision_day),
            is_in_fertilization_window=bool(is_in_fertilization_window),
            is_in_irrigation_window=bool(is_in_irrigation_window),
        )
        filtered_action = normalize_action(action_names, action_space_dict, filtered)
        return self.env.step(filtered_action)

    def _current_dap(self) -> int:
        history = getattr(self.unwrapped, "history", {})
        if isinstance(history, dict):
            observations = history.get("observation", [])
            if observations and isinstance(observations[-1], dict):
                try:
                    return int(round(float(observations[-1].get("dap", 1) or 1)))
                except Exception:
                    return 1
        return 1

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name):
        return getattr(self.env, name)


class DecisionIntervalActionWrapper(ScheduledActionDesignWrapper):
    def __init__(self, env: gym.Env, decision_interval_days: int):
        super().__init__(env, decision_interval_days=decision_interval_days, use_phenology_windows=False)


class PhenologyWindowActionWrapper(ScheduledActionDesignWrapper):
    def __init__(
        self,
        env: gym.Env,
        fertilization_windows: list[tuple[int, int]] | None = None,
        irrigation_windows: list[tuple[int, int]] | None = None,
    ):
        super().__init__(
            env,
            fertilization_windows=fertilization_windows,
            irrigation_windows=irrigation_windows,
            use_phenology_windows=True,
        )
