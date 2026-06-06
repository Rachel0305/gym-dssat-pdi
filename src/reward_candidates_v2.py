from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


RewardFunc = Callable[[dict | None, dict | None, dict, str], float | None]


@dataclass(frozen=True)
class RewardCandidateV2:
    name: str
    family: str
    batch: str
    kind: str
    description: str
    irrigation_cost_coef: float = 0.0
    nitrogen_cost_coef: float = 0.0
    cumulative_irrigation_coef: float = 0.0
    cumulative_n_coef: float = 0.0
    target_irrigation_upper: float = 220.0
    target_n_upper: float = 320.0
    target_irrigation_penalty_coef: float = 0.0
    target_n_penalty_coef: float = 0.0
    grain_value: float = 0.0
    water_cost: float = 0.0
    n_cost: float = 0.0


REWARD_CANDIDATES_V2: dict[str, RewardCandidateV2] = {
    "candidate_D1": RewardCandidateV2("candidate_D1", "candidate_D_strong_linear_cost", "batch_1_D", "base_minus_cost", "Strong linear cost D1.", 0.5, 0.25),
    "candidate_D2": RewardCandidateV2("candidate_D2", "candidate_D_strong_linear_cost", "batch_1_D", "base_minus_cost", "Strong linear cost D2.", 1.0, 0.5),
    "candidate_D3": RewardCandidateV2("candidate_D3", "candidate_D_strong_linear_cost", "batch_1_D", "base_minus_cost", "Strong linear cost D3.", 2.0, 1.0),
    "candidate_D4": RewardCandidateV2("candidate_D4", "candidate_D_strong_linear_cost", "batch_1_D", "base_minus_cost", "Strong linear cost D4.", 5.0, 2.5),
    "candidate_E1": RewardCandidateV2("candidate_E1", "candidate_E_cumulative_quadratic_cost", "batch_2_E", "base_minus_cumulative", "Cumulative quadratic cost E1.", 0.5, 0.25, 0.0005, 0.0002),
    "candidate_E2": RewardCandidateV2("candidate_E2", "candidate_E_cumulative_quadratic_cost", "batch_2_E", "base_minus_cumulative", "Cumulative quadratic cost E2.", 0.5, 0.25, 0.001, 0.0005),
    "candidate_E3": RewardCandidateV2("candidate_E3", "candidate_E_cumulative_quadratic_cost", "batch_2_E", "base_minus_cumulative", "Cumulative quadratic cost E3.", 0.5, 0.25, 0.005, 0.002),
    "candidate_E4": RewardCandidateV2("candidate_E4", "candidate_E_cumulative_quadratic_cost", "batch_2_E", "base_minus_cumulative", "Cumulative quadratic cost E4.", 0.5, 0.25, 0.01, 0.005),
    "candidate_F1": RewardCandidateV2("candidate_F1", "candidate_F_target_interval_penalty", "batch_3_F", "base_minus_target_penalty", "Weak target interval upper-bound penalty.", 0.5, 0.25, target_irrigation_penalty_coef=0.02, target_n_penalty_coef=0.01),
    "candidate_F2": RewardCandidateV2("candidate_F2", "candidate_F_target_interval_penalty", "batch_3_F", "base_minus_target_penalty", "Medium target interval upper-bound penalty.", 0.5, 0.25, target_irrigation_penalty_coef=0.10, target_n_penalty_coef=0.05),
    "candidate_F3": RewardCandidateV2("candidate_F3", "candidate_F_target_interval_penalty", "batch_3_F", "base_minus_target_penalty", "Strong target interval upper-bound penalty.", 0.5, 0.25, target_irrigation_penalty_coef=0.50, target_n_penalty_coef=0.25),
    "candidate_G1": RewardCandidateV2("candidate_G1", "candidate_G_economic_proxy", "batch_4_G", "economic_proxy", "Economic proxy G1.", grain_value=0.01, water_cost=0.2, n_cost=0.1),
    "candidate_G2": RewardCandidateV2("candidate_G2", "candidate_G_economic_proxy", "batch_4_G", "economic_proxy", "Economic proxy G2.", grain_value=0.01, water_cost=0.5, n_cost=0.25),
    "candidate_G3": RewardCandidateV2("candidate_G3", "candidate_G_economic_proxy", "batch_4_G", "economic_proxy", "Economic proxy G3.", grain_value=0.005, water_cost=0.5, n_cost=0.25),
    "candidate_G4": RewardCandidateV2("candidate_G4", "candidate_G_economic_proxy", "batch_4_G", "economic_proxy", "Economic proxy G4.", grain_value=0.005, water_cost=1.0, n_cost=0.5),
}


BATCHES: list[list[str]] = [
    ["candidate_D1", "candidate_D2", "candidate_D3", "candidate_D4"],
    ["candidate_E1", "candidate_E2", "candidate_E3", "candidate_E4"],
    ["candidate_F1", "candidate_F2", "candidate_F3"],
    ["candidate_G1", "candidate_G2", "candidate_G3", "candidate_G4"],
]

_PATCH_STATE: dict[str, object] = {}


def _float_state(state: dict | None, key: str, default: float = 0.0) -> float:
    if not isinstance(state, dict):
        return default
    try:
        return float(state.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _last_action(history: dict) -> dict:
    actions = history.get("action", []) if isinstance(history, dict) else []
    if not actions:
        return {}
    item = actions[-1]
    return item if isinstance(item, dict) else {}


def _total_action(history: dict, key: str) -> float:
    actions = history.get("action", []) if isinstance(history, dict) else []
    total = 0.0
    for action in actions:
        if isinstance(action, dict):
            try:
                total += float(action.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
    return total


def _action_values(history: dict) -> tuple[float, float, float, float]:
    last = _last_action(history)
    last_amir = float(last.get("amir", 0.0) or 0.0)
    last_anfer = float(last.get("anfer", 0.0) or 0.0)
    total_amir = _total_action(history, "amir")
    total_anfer = _total_action(history, "anfer")
    return last_amir, last_anfer, total_amir, total_anfer


def _cost(candidate: RewardCandidateV2, history: dict) -> float:
    last_amir, last_anfer, total_amir, total_anfer = _action_values(history)
    target_irrig_excess = max(0.0, total_amir - candidate.target_irrigation_upper)
    target_n_excess = max(0.0, total_anfer - candidate.target_n_upper)
    return (
        candidate.irrigation_cost_coef * last_amir
        + candidate.nitrogen_cost_coef * last_anfer
        + candidate.cumulative_irrigation_coef * total_amir * total_amir
        + candidate.cumulative_n_coef * total_anfer * total_anfer
        + candidate.target_irrigation_penalty_coef * target_irrig_excess * target_irrig_excess
        + candidate.target_n_penalty_coef * target_n_excess * target_n_excess
    )


def make_candidate_reward(candidate_name: str, baseline_reward: RewardFunc) -> RewardFunc:
    if candidate_name not in REWARD_CANDIDATES_V2:
        known = ", ".join(sorted(REWARD_CANDIDATES_V2))
        raise KeyError(f"Unknown reward candidate '{candidate_name}'. Known candidates: {known}")
    candidate = REWARD_CANDIDATES_V2[candidate_name]

    def reward(previous_state, next_state, history, cultivar):
        if next_state is None:
            return None
        if candidate.kind == "economic_proxy":
            previous_grain = _float_state(previous_state, "grnwt", 0.0)
            next_grain = _float_state(next_state, "grnwt", 0.0)
            grain_gain = max(0.0, next_grain - previous_grain)
            last_amir, last_anfer, _total_amir, _total_anfer = _action_values(history)
            return candidate.grain_value * grain_gain - candidate.water_cost * last_amir - candidate.n_cost * last_anfer
        base = baseline_reward(previous_state, next_state, history, cultivar)
        if base is None:
            return None
        return float(base) - _cost(candidate, history)

    return reward


def patch_reward_candidate_v2(candidate_name: str) -> RewardCandidateV2:
    from gym_dssat_pdi.envs.configs import rewards

    if "original_all_reward" not in _PATCH_STATE:
        _PATCH_STATE["original_all_reward"] = rewards.all_reward
        _PATCH_STATE["original_get_reward_function"] = rewards.get_reward_function

    original_all_reward = _PATCH_STATE["original_all_reward"]
    patched = make_candidate_reward(candidate_name, original_all_reward)

    def get_reward_function(mode: str):
        if mode == "all":
            return patched
        return _PATCH_STATE["original_get_reward_function"](mode)

    rewards.all_reward = patched
    rewards.get_reward_function = get_reward_function
    _PATCH_STATE["active_candidate"] = candidate_name
    return REWARD_CANDIDATES_V2[candidate_name]


def unpatch_reward_candidate_v2() -> None:
    if "original_all_reward" not in _PATCH_STATE:
        return
    from gym_dssat_pdi.envs.configs import rewards

    rewards.all_reward = _PATCH_STATE["original_all_reward"]
    rewards.get_reward_function = _PATCH_STATE["original_get_reward_function"]
    _PATCH_STATE.pop("active_candidate", None)


def candidate_table() -> list[dict]:
    return [
        {
            "reward_version": item.name,
            "reward_family": item.family,
            "batch": item.batch,
            "kind": item.kind,
            "description": item.description,
            "irrigation_cost_coef": item.irrigation_cost_coef,
            "nitrogen_cost_coef": item.nitrogen_cost_coef,
            "cumulative_irrigation_coef": item.cumulative_irrigation_coef,
            "cumulative_n_coef": item.cumulative_n_coef,
            "target_irrigation_upper": item.target_irrigation_upper,
            "target_n_upper": item.target_n_upper,
            "target_irrigation_penalty_coef": item.target_irrigation_penalty_coef,
            "target_n_penalty_coef": item.target_n_penalty_coef,
            "grain_value": item.grain_value,
            "water_cost": item.water_cost,
            "n_cost": item.n_cost,
        }
        for item in REWARD_CANDIDATES_V2.values()
    ]
