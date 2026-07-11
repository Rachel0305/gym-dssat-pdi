"""Configuration-driven discrete water-nitrogen action tables."""

from __future__ import annotations

from itertools import product
from typing import Any


def build_action_table(config: dict[str, Any]) -> dict[int, dict[str, float]]:
    """Build the Cartesian irrigation-nitrogen action table.

    Irrigation varies fastest to preserve the validated 9-action ordering in
    ``src/frozen_nstep_dqn_config_020_11.py``.
    """

    action_config = config.get("action_space", config)
    irrigation = [float(value) for value in action_config["irrigation_levels"]]
    nitrogen = [float(value) for value in action_config["nitrogen_levels"]]
    table: dict[int, dict[str, float]] = {}
    index = 0
    for nitrogen_value, irrigation_value in product(nitrogen, irrigation):
        table[index] = {"amir": irrigation_value, "anfer": nitrogen_value}
        index += 1
    return table


def action_count(config: dict[str, Any]) -> int:
    """Return the number of discrete action combinations."""

    return len(build_action_table(config))

