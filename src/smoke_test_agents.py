from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class FixedPolicy:
    name: str
    nitrogen_by_dap: Dict[int, float] = field(default_factory=dict)
    irrigation_by_dap: Dict[int, float] = field(default_factory=dict)

    def action_for_dap(self, dap: int, action_names: list[str]) -> dict[str, float]:
        action = {name: 0.0 for name in action_names}
        if "anfer" in action:
            action["anfer"] = float(self.nitrogen_by_dap.get(dap, 0.0))
        if "amir" in action:
            action["amir"] = float(self.irrigation_by_dap.get(dap, 0.0))
        return action


POLICIES: dict[str, FixedPolicy] = {
    "null_zero": FixedPolicy("null_zero"),
    "fixed_low_input": FixedPolicy(
        "fixed_low_input",
        nitrogen_by_dap={1: 30.0, 30: 20.0},
    ),
    "fixed_medium_input": FixedPolicy(
        "fixed_medium_input",
        nitrogen_by_dap={1: 50.0, 30: 50.0, 60: 50.0},
        irrigation_by_dap={30: 30.0, 60: 30.0},
    ),
    "fixed_high_input": FixedPolicy(
        "fixed_high_input",
        nitrogen_by_dap={1: 80.0, 30: 80.0, 60: 90.0},
        irrigation_by_dap={30: 40.0, 60: 40.0, 90: 40.0},
    ),
}


def clip_real_action(
    action_by_name: dict[str, float],
    action_space_dict: dict,
) -> tuple[dict[str, float], list[str]]:
    spaces = getattr(action_space_dict, "spaces", action_space_dict)
    clipped: dict[str, float] = {}
    notes: list[str] = []
    for name, value in action_by_name.items():
        space = spaces.get(name) if hasattr(spaces, "get") else spaces[name]
        if space is None:
            continue
        low = float(np.asarray(space.low).flatten()[0])
        high = float(np.asarray(space.high).flatten()[0])
        clipped_value = min(max(float(value), low), high)
        if clipped_value != float(value):
            notes.append(f"{name}_clipped_{value}_to_{clipped_value}")
        clipped[name] = clipped_value
    return clipped, notes


def normalize_real_action(
    action_by_name: dict[str, float],
    action_names: list[str],
    action_space_dict: dict,
) -> np.ndarray:
    spaces = getattr(action_space_dict, "spaces", action_space_dict)
    normalized: list[float] = []
    for name in action_names:
        space = spaces[name]
        low = float(np.asarray(space.low).flatten()[0])
        high = float(np.asarray(space.high).flatten()[0])
        value = float(action_by_name.get(name, 0.0))
        normalized.append(2.0 * ((value - low) / (high - low)) - 1.0)
    return np.asarray(normalized, dtype=np.float32)
