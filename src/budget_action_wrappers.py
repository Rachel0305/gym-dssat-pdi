from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from ppo_action_safety import normalize_action


@dataclass
class BudgetActionResult:
    raw_policy_action: list[float]
    budget_action_irrigation: float
    budget_action_n: float
    scheduled_event_irrigation: float
    scheduled_event_n: float
    executed_irrigation: float
    executed_n: float
    budget_remaining_irrigation: float
    budget_remaining_n: float
    stage_name: str
    event_name: str
    is_budget_decision_day: bool
    is_scheduled_event_day: bool
    budget_rule_triggered: str


def _scale(value: float, low: float, high: float) -> float:
    clipped = max(-1.0, min(1.0, float(value)))
    return low + 0.5 * (clipped + 1.0) * (high - low)


def _current_dap(env) -> int:
    history = getattr(env.unwrapped, "history", {})
    if isinstance(history, dict):
        observations = history.get("observation", [])
        if observations and isinstance(observations[-1], dict):
            try:
                return int(round(float(observations[-1].get("dap", 1) or 1)))
            except Exception:
                return 1
    return 1


class BudgetActionBaseWrapper(gym.Env):
    def __init__(self, env: gym.Env):
        super().__init__()
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = getattr(env, "metadata", {})
        self.last_budget_result: BudgetActionResult | None = None
        self.remaining_irrigation = 0.0
        self.remaining_n = 0.0

    def reset(self, *args, **kwargs):
        self.last_budget_result = None
        return self.env.reset(*args, **kwargs)

    def _step_real_action(self, action, real_action: dict[str, float], result: BudgetActionResult):
        clipped = self._clip_to_env_bounds(real_action)
        result.executed_irrigation = float(clipped.get("amir", 0.0))
        result.executed_n = float(clipped.get("anfer", 0.0))
        self.last_budget_result = result
        normalized = normalize_action(self.env.formator.action_names, self.env.formator.action_space_dict, clipped)
        return self.env.step(normalized)

    def _clip_to_env_bounds(self, real_action: dict[str, float]) -> dict[str, float]:
        spaces = getattr(self.env.formator.action_space_dict, "spaces", self.env.formator.action_space_dict)
        clipped: dict[str, float] = {}
        for name in self.env.formator.action_names:
            space = spaces[name]
            low = float(np.asarray(space.low).flatten()[0])
            high = float(np.asarray(space.high).flatten()[0])
            clipped[name] = max(low, min(high, float(real_action.get(name, 0.0))))
        return clipped

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name):
        return getattr(self.env, name)


class SeasonalBudgetActionWrapper(BudgetActionBaseWrapper):
    """Choose season budgets on DAP 1, then release them at fixed events."""

    def __init__(
        self,
        env: gym.Env,
        *,
        irrigation_cap: float = 300.0,
        n_cap: float = 450.0,
        irrigation_schedule: list[dict[str, Any]] | None = None,
        nitrogen_schedule: list[dict[str, Any]] | None = None,
    ):
        super().__init__(env)
        self.irrigation_cap = float(irrigation_cap)
        self.n_cap = float(n_cap)
        self.irrigation_schedule = irrigation_schedule or [
            {"dap": 25, "fraction": 0.2, "name": "early_growth"},
            {"dap": 50, "fraction": 0.4, "name": "mid_growth"},
            {"dap": 75, "fraction": 0.4, "name": "silking_grainfill"},
        ]
        self.nitrogen_schedule = nitrogen_schedule or [
            {"dap": 1, "fraction": 0.3, "name": "basal"},
            {"dap": 30, "fraction": 0.3, "name": "early_growth"},
            {"dap": 60, "fraction": 0.4, "name": "pre_tasseling"},
        ]
        self.season_irrigation_budget = 0.0
        self.season_n_budget = 0.0

    def reset(self, *args, **kwargs):
        self.season_irrigation_budget = 0.0
        self.season_n_budget = 0.0
        self.remaining_irrigation = 0.0
        self.remaining_n = 0.0
        return super().reset(*args, **kwargs)

    def step(self, action):
        arr = np.asarray(action, dtype=float).flatten()
        dap = _current_dap(self.env)
        triggers: list[str] = []
        is_budget_decision_day = dap <= 1
        if is_budget_decision_day:
            self.season_irrigation_budget = _scale(arr[0], 0.0, self.irrigation_cap)
            self.season_n_budget = _scale(arr[1] if len(arr) > 1 else arr[0], 0.0, self.n_cap)
            self.remaining_irrigation = self.season_irrigation_budget
            self.remaining_n = self.season_n_budget
            triggers.append("season_budget_decision")

        irrigation_event = next((item for item in self.irrigation_schedule if int(item["dap"]) == dap), None)
        nitrogen_event = next((item for item in self.nitrogen_schedule if int(item["dap"]) == dap), None)
        event_name_parts: list[str] = []
        irrig = 0.0
        n = 0.0
        if irrigation_event:
            event_name_parts.append(f"irrigation:{irrigation_event['name']}")
            irrig = min(self.remaining_irrigation, self.season_irrigation_budget * float(irrigation_event["fraction"]))
        if nitrogen_event:
            event_name_parts.append(f"nitrogen:{nitrogen_event['name']}")
            n = min(self.remaining_n, self.season_n_budget * float(nitrogen_event["fraction"]))
        self.remaining_irrigation = max(0.0, self.remaining_irrigation - irrig)
        self.remaining_n = max(0.0, self.remaining_n - n)
        if irrig or n:
            triggers.append("scheduled_budget_release")

        result = BudgetActionResult(
            raw_policy_action=arr.tolist(),
            budget_action_irrigation=self.season_irrigation_budget,
            budget_action_n=self.season_n_budget,
            scheduled_event_irrigation=irrig,
            scheduled_event_n=n,
            executed_irrigation=irrig,
            executed_n=n,
            budget_remaining_irrigation=self.remaining_irrigation,
            budget_remaining_n=self.remaining_n,
            stage_name="season",
            event_name=";".join(event_name_parts),
            is_budget_decision_day=is_budget_decision_day,
            is_scheduled_event_day=bool(irrigation_event or nitrogen_event),
            budget_rule_triggered=";".join(triggers),
        )
        return self._step_real_action(action, {"amir": irrig, "anfer": n}, result)


class ScheduledEventActionWrapper(BudgetActionBaseWrapper):
    """Let PPO decide event amounts only on fixed event days."""

    def __init__(
        self,
        env: gym.Env,
        *,
        irrigation_events: list[dict[str, Any]] | None = None,
        nitrogen_events: list[dict[str, Any]] | None = None,
        single_irrigation_event_max: float = 100.0,
        single_n_event_max: float = 150.0,
        irrigation_cap: float = 300.0,
        n_cap: float = 450.0,
    ):
        super().__init__(env)
        self.irrigation_events = irrigation_events or [
            {"dap": 25, "name": "early_growth"},
            {"dap": 50, "name": "mid_growth"},
            {"dap": 75, "name": "silking_grainfill"},
        ]
        self.nitrogen_events = nitrogen_events or [
            {"dap": 1, "name": "basal"},
            {"dap": 30, "name": "early_growth"},
            {"dap": 60, "name": "pre_tasseling"},
        ]
        self.single_irrigation_event_max = float(single_irrigation_event_max)
        self.single_n_event_max = float(single_n_event_max)
        self.irrigation_cap = float(irrigation_cap)
        self.n_cap = float(n_cap)

    def reset(self, *args, **kwargs):
        self.remaining_irrigation = self.irrigation_cap
        self.remaining_n = self.n_cap
        return super().reset(*args, **kwargs)

    def step(self, action):
        arr = np.asarray(action, dtype=float).flatten()
        dap = _current_dap(self.env)
        irrigation_event = next((item for item in self.irrigation_events if int(item["dap"]) == dap), None)
        nitrogen_event = next((item for item in self.nitrogen_events if int(item["dap"]) == dap), None)
        irrig = 0.0
        n = 0.0
        event_name_parts: list[str] = []
        triggers: list[str] = []
        if irrigation_event:
            event_name_parts.append(f"irrigation:{irrigation_event['name']}")
            irrig = min(self.remaining_irrigation, _scale(arr[0], 0.0, self.single_irrigation_event_max))
            triggers.append("irrigation_event_action")
        if nitrogen_event:
            event_name_parts.append(f"nitrogen:{nitrogen_event['name']}")
            n = min(self.remaining_n, _scale(arr[1] if len(arr) > 1 else arr[0], 0.0, self.single_n_event_max))
            triggers.append("nitrogen_event_action")
        self.remaining_irrigation = max(0.0, self.remaining_irrigation - irrig)
        self.remaining_n = max(0.0, self.remaining_n - n)
        result = BudgetActionResult(
            raw_policy_action=arr.tolist(),
            budget_action_irrigation=0.0,
            budget_action_n=0.0,
            scheduled_event_irrigation=irrig,
            scheduled_event_n=n,
            executed_irrigation=irrig,
            executed_n=n,
            budget_remaining_irrigation=self.remaining_irrigation,
            budget_remaining_n=self.remaining_n,
            stage_name="event",
            event_name=";".join(event_name_parts),
            is_budget_decision_day=False,
            is_scheduled_event_day=bool(irrigation_event or nitrogen_event),
            budget_rule_triggered=";".join(triggers),
        )
        return self._step_real_action(action, {"amir": irrig, "anfer": n}, result)


class StageBudgetActionWrapper(BudgetActionBaseWrapper):
    """At stage starts, PPO selects a stage budget released on that day."""

    def __init__(
        self,
        env: gym.Env,
        *,
        stages: list[dict[str, Any]] | None = None,
        irrigation_cap: float = 300.0,
        n_cap: float = 450.0,
    ):
        super().__init__(env)
        self.stages = stages or [
            {"name": "stage_1", "start": 1, "end": 30, "irrigation_max": 75.0, "n_max": 150.0},
            {"name": "stage_2", "start": 31, "end": 60, "irrigation_max": 75.0, "n_max": 150.0},
            {"name": "stage_3", "start": 61, "end": 95, "irrigation_max": 100.0, "n_max": 150.0},
            {"name": "stage_4", "start": 96, "end": 120, "irrigation_max": 50.0, "n_max": 0.0},
        ]
        self.remaining_irrigation = float(irrigation_cap)
        self.remaining_n = float(n_cap)

    def reset(self, *args, **kwargs):
        self.remaining_irrigation = sum(float(item.get("irrigation_max", 0.0)) for item in self.stages)
        self.remaining_n = sum(float(item.get("n_max", 0.0)) for item in self.stages)
        return super().reset(*args, **kwargs)

    def step(self, action):
        arr = np.asarray(action, dtype=float).flatten()
        dap = _current_dap(self.env)
        stage = next((item for item in self.stages if int(item["start"]) <= dap <= int(item["end"])), None)
        is_start = stage is not None and int(stage["start"]) == dap
        irrig = 0.0
        n = 0.0
        triggers: list[str] = []
        if stage and is_start:
            irrig = min(self.remaining_irrigation, _scale(arr[0], 0.0, float(stage.get("irrigation_max", 0.0))))
            n = min(self.remaining_n, _scale(arr[1] if len(arr) > 1 else arr[0], 0.0, float(stage.get("n_max", 0.0))))
            triggers.append("stage_budget_decision")
        self.remaining_irrigation = max(0.0, self.remaining_irrigation - irrig)
        self.remaining_n = max(0.0, self.remaining_n - n)
        result = BudgetActionResult(
            raw_policy_action=arr.tolist(),
            budget_action_irrigation=irrig,
            budget_action_n=n,
            scheduled_event_irrigation=irrig,
            scheduled_event_n=n,
            executed_irrigation=irrig,
            executed_n=n,
            budget_remaining_irrigation=self.remaining_irrigation,
            budget_remaining_n=self.remaining_n,
            stage_name=str(stage["name"]) if stage else "",
            event_name=str(stage["name"]) if stage and is_start else "",
            is_budget_decision_day=bool(is_start),
            is_scheduled_event_day=bool(is_start),
            budget_rule_triggered=";".join(triggers),
        )
        return self._step_real_action(action, {"amir": irrig, "anfer": n}, result)
