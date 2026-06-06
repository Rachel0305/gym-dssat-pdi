from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


RewardFunc = Callable[[dict | None, dict | None, dict, str], float | None]


@dataclass(frozen=True)
class RewardCandidate:
    name: str
    family: str
    description: str
    irrigation_cost_coef: float = 0.0
    nitrogen_cost_coef: float = 0.0
    target_irrigation: float = 200.0
    target_n: float = 300.0
    excess_irrigation_coef: float = 0.0
    excess_n_coef: float = 0.0
    yield_value_coef: float = 0.0
    use_incremental_yield_proxy: bool = False


REWARD_CANDIDATES: dict[str, RewardCandidate] = {
    "current_reward_baseline": RewardCandidate(
        name="current_reward_baseline",
        family="baseline",
        description="Original gym-DSSAT all_reward used as the baseline.",
    ),
    "candidate_A1": RewardCandidate(
        name="candidate_A1",
        family="candidate_A_linear_cost",
        description="Original all_reward plus weak linear irrigation and nitrogen action costs.",
        irrigation_cost_coef=0.05,
        nitrogen_cost_coef=0.02,
    ),
    "candidate_A2": RewardCandidate(
        name="candidate_A2",
        family="candidate_A_linear_cost",
        description="Original all_reward plus medium linear irrigation and nitrogen action costs.",
        irrigation_cost_coef=0.10,
        nitrogen_cost_coef=0.05,
    ),
    "candidate_A3": RewardCandidate(
        name="candidate_A3",
        family="candidate_A_linear_cost",
        description="Original all_reward plus stronger linear irrigation and nitrogen action costs.",
        irrigation_cost_coef=0.20,
        nitrogen_cost_coef=0.10,
    ),
    "candidate_B1": RewardCandidate(
        name="candidate_B1",
        family="candidate_B_linear_plus_excess_penalty",
        description="Linear costs plus weak quadratic penalty above 200 mm irrigation and 300 kg/ha N.",
        irrigation_cost_coef=0.10,
        nitrogen_cost_coef=0.05,
        excess_irrigation_coef=0.005,
        excess_n_coef=0.002,
    ),
    "candidate_B2": RewardCandidate(
        name="candidate_B2",
        family="candidate_B_linear_plus_excess_penalty",
        description="Linear costs plus medium quadratic penalty above 200 mm irrigation and 300 kg/ha N.",
        irrigation_cost_coef=0.20,
        nitrogen_cost_coef=0.10,
        excess_irrigation_coef=0.030,
        excess_n_coef=0.012,
    ),
    "candidate_B3": RewardCandidate(
        name="candidate_B3",
        family="candidate_B_linear_plus_excess_penalty",
        description="Linear costs plus strong quadratic penalty above 200 mm irrigation and 300 kg/ha N.",
        irrigation_cost_coef=0.50,
        nitrogen_cost_coef=0.25,
        excess_irrigation_coef=0.100,
        excess_n_coef=0.040,
    ),
    "candidate_C1": RewardCandidate(
        name="candidate_C1",
        family="candidate_C_incremental_yield_minus_total_cost",
        description=(
            "Incremental grain-yield proxy minus daily irrigation and nitrogen costs. "
            "The gym reward callback has no explicit done flag, so final-yield reward is approximated "
            "by positive daily grain-yield increments."
        ),
        irrigation_cost_coef=0.50,
        nitrogen_cost_coef=0.20,
        yield_value_coef=0.08,
        use_incremental_yield_proxy=True,
    ),
}

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


def _linear_and_excess_cost(candidate: RewardCandidate, history: dict) -> float:
    last = _last_action(history)
    last_amir = float(last.get("amir", 0.0) or 0.0)
    last_anfer = float(last.get("anfer", 0.0) or 0.0)
    total_amir = _total_action(history, "amir")
    total_anfer = _total_action(history, "anfer")
    excess_amir = max(0.0, total_amir - candidate.target_irrigation)
    excess_anfer = max(0.0, total_anfer - candidate.target_n)
    return (
        candidate.irrigation_cost_coef * last_amir
        + candidate.nitrogen_cost_coef * last_anfer
        + candidate.excess_irrigation_coef * excess_amir * excess_amir
        + candidate.excess_n_coef * excess_anfer * excess_anfer
    )


def make_candidate_reward(candidate_name: str, baseline_reward: RewardFunc) -> RewardFunc:
    if candidate_name not in REWARD_CANDIDATES:
        known = ", ".join(sorted(REWARD_CANDIDATES))
        raise KeyError(f"Unknown reward candidate '{candidate_name}'. Known candidates: {known}")
    candidate = REWARD_CANDIDATES[candidate_name]

    def reward(previous_state, next_state, history, cultivar):
        if next_state is None:
            return None
        if candidate.name == "current_reward_baseline":
            return baseline_reward(previous_state, next_state, history, cultivar)
        if candidate.use_incremental_yield_proxy:
            previous_grain = _float_state(previous_state, "grnwt", 0.0)
            next_grain = _float_state(next_state, "grnwt", 0.0)
            yield_gain = max(0.0, next_grain - previous_grain)
            return candidate.yield_value_coef * yield_gain - _linear_and_excess_cost(candidate, history)
        base = baseline_reward(previous_state, next_state, history, cultivar)
        if base is None:
            return None
        return float(base) - _linear_and_excess_cost(candidate, history)

    return reward


def patch_reward_candidate(candidate_name: str) -> RewardCandidate:
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
    return REWARD_CANDIDATES[candidate_name]


def unpatch_reward_candidate() -> None:
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
            "description": item.description,
            "irrigation_cost_coef": item.irrigation_cost_coef,
            "nitrogen_cost_coef": item.nitrogen_cost_coef,
            "target_irrigation": item.target_irrigation,
            "target_n": item.target_n,
            "excess_irrigation_coef": item.excess_irrigation_coef,
            "excess_n_coef": item.excess_n_coef,
            "yield_value_coef": item.yield_value_coef,
            "use_incremental_yield_proxy": item.use_incremental_yield_proxy,
        }
        for item in REWARD_CANDIDATES.values()
    ]
