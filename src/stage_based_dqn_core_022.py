from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


ACTION_TABLE_9 = {
    0: {"amir": 0.0, "anfer": 0.0},
    1: {"amir": 15.0, "anfer": 0.0},
    2: {"amir": 30.0, "anfer": 0.0},
    3: {"amir": 0.0, "anfer": 50.0},
    4: {"amir": 15.0, "anfer": 50.0},
    5: {"amir": 30.0, "anfer": 50.0},
    6: {"amir": 0.0, "anfer": 100.0},
    7: {"amir": 15.0, "anfer": 100.0},
    8: {"amir": 30.0, "anfer": 100.0},
}

SOURCE_STAGE_DAPS = (0, 30, 50, 65, 85, 110)
EXECUTABLE_STAGE_DAPS = (1, 30, 50, 65, 85, 110)
ORACLE_AUDIT_ONLY_DAPS = (22, 29, 42, 56, 79)
IRRIGATION_BUDGET = 120.0
NITROGEN_BUDGET = 300.0
WATER_COST = 1.0
NITROGEN_COST = 5.0
NULL_YIELD = 5408.0
EXPERT_GATE_YIELD = 11077.0
FEASIBILITY_BONUS = 1620.0


@dataclass(frozen=True)
class ExecutedAction:
    action_index: int
    requested_irrigation: float
    requested_nitrogen: float
    executed_irrigation: float
    executed_nitrogen: float


def valid_action_indices(dap: int) -> tuple[int, ...]:
    if dap not in EXECUTABLE_STAGE_DAPS:
        return (0,)
    if dap >= 90:
        return (0, 1, 2)
    return tuple(ACTION_TABLE_9)


def execute_stage_action(
    action_index: int,
    dap: int,
    used_irrigation: float,
    used_nitrogen: float,
) -> ExecutedAction:
    if action_index not in valid_action_indices(dap):
        raise ValueError(f"Action {action_index} is invalid at DAP {dap}; valid={valid_action_indices(dap)}")
    request = ACTION_TABLE_9[action_index]
    executed_i = min(request["amir"], max(0.0, IRRIGATION_BUDGET - used_irrigation))
    executed_n = min(request["anfer"], max(0.0, NITROGEN_BUDGET - used_nitrogen))
    return ExecutedAction(
        action_index=action_index,
        requested_irrigation=request["amir"],
        requested_nitrogen=request["anfer"],
        executed_irrigation=executed_i,
        executed_nitrogen=executed_n,
    )


def terminal_value(final_yield: float) -> float:
    yield_gain = max(0.0, float(final_yield) - NULL_YIELD)
    gate = FEASIBILITY_BONUS if final_yield >= EXPERT_GATE_YIELD else 0.0
    return yield_gain + gate


def terminal_complete_returns(
    executed_actions: Iterable[ExecutedAction],
    final_yield: float,
) -> list[float]:
    actions = list(executed_actions)
    rewards = [
        -WATER_COST * action.executed_irrigation - NITROGEN_COST * action.executed_nitrogen
        for action in actions
    ]
    rewards[-1] += terminal_value(final_yield)
    returns = [0.0] * len(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + running  # gamma=1 for finite complete season
        returns[index] = running
    return returns


def nearest_stage_distance(dap: int) -> int:
    return min(abs(int(dap) - stage) for stage in EXECUTABLE_STAGE_DAPS)

