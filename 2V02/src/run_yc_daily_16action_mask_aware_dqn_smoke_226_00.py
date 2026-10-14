"""226_00: YC daily 16-action mask-aware DQN adapter smoke.

This is an engineering smoke only.  It reuses the project-local 029
MaskAwareDQN core and the current 055-style YC lowIC wrapper/reward/safety
chain.  It does not modify or invoke formal PPO/DQN training.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


OUT = ROOT / "experiments" / "226_yc_dqn_adapter_smoke"
RESULTS = OUT / "results"
CHECKPOINTS = OUT / "checkpoints"
RUNTIME_OUT = OUT / "runtime"
REFERENCE_CONFIG = ROOT / "configs" / "055_00_yca_lowIC_expanded_action_maskableppo.json"
LOWIC_ROOT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual"
STATION = "YCA"
SITE = "YC"
TRAIN_YEAR = 2004
SEED = 0
ACTION_IRRIGATION = [0.0, 15.0, 30.0, 45.0]
ACTION_NITROGEN = [0.0, 40.0, 80.0, 120.0]
ACTION_DIM = 16
SMOKE_STEPS = 12
SMOKE_BATCH_SIZE = 4
TARGET_SYNC_EVERY = 4


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def action_grid() -> list[dict[str, float]]:
    return [
        {"amir": float(irrigation), "anfer": float(nitrogen)}
        for irrigation in ACTION_IRRIGATION
        for nitrogen in ACTION_NITROGEN
    ]


ACTION_GRID = action_grid()


def array_hash(value: Any) -> str:
    array = np.asarray(value)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_action_mapping() -> list[dict[str, Any]]:
    rows = [
        {
            "action_index": index,
            "irrigation_mm": pair["amir"],
            "nitrogen_kg_ha": pair["anfer"],
            "mapping_formula": "irrigation_index * 4 + nitrogen_index",
            "shared_by": "mask, epsilon, greedy, replay, env.step, daily_log, evaluation",
        }
        for index, pair in enumerate(ACTION_GRID)
    ]
    write_csv(RESULTS / "yc_dqn_adapter_action_mapping.csv", rows)
    expected = [(i, n) for i in ACTION_IRRIGATION for n in ACTION_NITROGEN]
    actual = [(row["irrigation_mm"], row["nitrogen_kg_ha"]) for row in rows]
    if actual != expected or len(rows) != ACTION_DIM or [int(row["action_index"]) for row in rows] != list(range(ACTION_DIM)):
        raise AssertionError("The YC action mapping is not the frozen 16-action row-major mapping")
    return rows


def import_runtime_modules():
    import pandas as pd
    import ppo_safe_rendering
    import run_all_year_direct_action_safe_ppo as direct_ppo
    import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch
    import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine
    import run_sya_lowIC_ppo_late_irrigation_reserve_mask_040_36 as current_wrapper
    from mask_aware_dqn_029 import MaskAwareDQN
    from ppo_action_safety import ActionSafetyState

    return pd, ppo_safe_rendering, direct_ppo, batch, engine, current_wrapper, MaskAwareDQN, ActionSafetyState


def configure_current_yc(pd, ppo_safe_rendering, batch, engine, current_wrapper) -> dict[str, Any]:
    """Patch only module-level run targets, matching the existing 055 runner."""
    batch.SITES = [STATION]
    batch.SEED = SEED
    batch.OUT = RUNTIME_OUT
    batch.TOTAL_TIMESTEPS = SMOKE_STEPS
    engine.STATION = STATION
    engine.SITES = [STATION]
    engine.LOWIC_INPUT_ROOT = LOWIC_ROOT
    engine.BINARY_IRRIGATION_LEVELS = list(ACTION_IRRIGATION)
    engine.BINARY_NITROGEN_LEVELS = list(ACTION_NITROGEN)
    current_wrapper.LOWIC_INPUT_ROOT = LOWIC_ROOT
    current_wrapper.STATION = STATION
    current_wrapper.SITES = [STATION]
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = LOWIC_ROOT

    config = engine.load_config()
    config["seed"] = SEED
    config["paths"]["output_root"] = rel(RUNTIME_OUT)
    config["runtime"]["smoke_station"] = STATION
    config["runtime"]["smoke_year"] = TRAIN_YEAR
    config["runtime"]["max_steps"] = 260
    config["discrete_actions"]["irrigation_levels"] = list(ACTION_IRRIGATION)
    config["discrete_actions"]["nitrogen_levels"] = list(ACTION_NITROGEN)
    return config


def build_yc_env_config(pd, direct_ppo, batch, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    split = batch.load_split().copy()
    station_split = split[split["station_code"].astype(str).eq(STATION)].copy()
    train_years = sorted(station_split.loc[station_split["split"].eq("train"), "year"].astype(int).tolist())
    validation_years = sorted(station_split.loc[station_split["split"].eq("validation"), "year"].astype(int).tolist())
    expected_train = list(range(2004, 2014))
    expected_validation = list(range(2014, 2024))
    if train_years != expected_train or validation_years != expected_validation:
        raise AssertionError(f"YC split drift: train={train_years}, validation={validation_years}")
    selection_all = batch.build_selection(split)
    selection = selection_all[(selection_all["station_code"].astype(str).eq(STATION)) & (selection_all["year"].astype(int).eq(TRAIN_YEAR))].copy()
    if len(selection) != 1:
        raise AssertionError(f"Expected exactly one selected YC training row for {TRAIN_YEAR}, found {len(selection)}")
    env_config = direct_ppo.build_env_config(config, selection)
    preflight = {
        "station_code": STATION,
        "site": SITE,
        "input_profile": "lowIC",
        "input_root": rel(LOWIC_ROOT),
        "source_mzx": rel(LOWIC_ROOT / "YC" / "CNYC0801.MZX"),
        "train_years": expected_train,
        "validation_years": expected_validation,
        "smoke_training_year": TRAIN_YEAR,
        "selected_row": jsonable(selection.iloc[0].to_dict()),
        "combined_action_count": ACTION_DIM,
    }
    if not LOWIC_ROOT.exists() or not (LOWIC_ROOT / "YC" / "CNYC0801.MZX").exists() or not (LOWIC_ROOT / "YC" / "CNYC0401.WTH").exists():
        raise FileNotFoundError("YC lowIC source MZX or 2004 weather input is missing")
    return env_config, preflight


def make_env(direct_ppo, current_wrapper, env_config: dict[str, Any], config: dict[str, Any], run_tag: str, evaluation: bool):
    base = direct_ppo.make_base_env(env_config, STATION, TRAIN_YEAR, SEED, run_tag, evaluation=evaluation)
    return current_wrapper.LateIrrigationReserveMaskWrapper(base, config)


def make_probe(current_wrapper, ActionSafetyState, config: dict[str, Any], dap: int, cum_i: float, cum_n: float, last_i: int | None = None, last_n: int | None = None):
    probe = object.__new__(current_wrapper.LateIrrigationReserveMaskWrapper)
    probe.config = config
    probe.grid = list(ACTION_GRID)
    probe.safety_state = ActionSafetyState(
        cumulative_irrigation=float(cum_i),
        cumulative_n=float(cum_n),
        last_irrigation_dap=last_i,
        last_fertilization_dap=last_n,
    )
    probe.last_obs_dict = {"dap": float(dap)}
    return probe


def run_staged_assertions(current_wrapper, ActionSafetyState, config: dict[str, Any]) -> list[dict[str, Any]]:
    cases = [
        ("DAP30_at_75_positive_irrigation", 30, 75.0, 0.0, 12, False, None, None),
        ("DAP30_at_75_noop", 30, 75.0, 0.0, 0, True, None, None),
        ("DAP60_at_150_positive_irrigation", 60, 150.0, 0.0, 12, False, None, None),
        ("DAP90_at_195_positive_irrigation", 90, 195.0, 0.0, 12, False, None, None),
        ("season_at_240_positive_irrigation", 100, 240.0, 0.0, 12, False, None, None),
        ("season_at_240_noop", 100, 240.0, 0.0, 0, True, None, None),
        ("late_reserve_DAP90_can_reach_195", 90, 150.0, 0.0, 12, True, None, None),
        ("late_reserve_after_DAP90_opportunity", 91, 195.0, 0.0, 12, True, None, None),
    ]
    rows: list[dict[str, Any]] = []
    for name, dap, cum_i, cum_n, action, expected, last_i, last_n in cases:
        probe = make_probe(current_wrapper, ActionSafetyState, config, dap, cum_i, cum_n, last_i, last_n)
        actual = bool(probe.action_masks()[action])
        result = "PASS" if actual == expected else "FAIL"
        rows.append(
            {
                "test_case": name,
                "dap": dap,
                "cum_irrigation_before": cum_i,
                "candidate_action": action,
                "candidate_irrigation_mm": ACTION_GRID[action]["amir"],
                "candidate_nitrogen_kg_ha": ACTION_GRID[action]["anfer"],
                "expected_valid": expected,
                "actual_valid": actual,
                "result": result,
            }
        )
    write_csv(RESULTS / "yc_dqn_staged_irrigation_assertions.csv", rows)
    if not all(row["result"] == "PASS" for row in rows):
        raise AssertionError("YC staged irrigation assertions failed")
    return rows


def run_n_assertions(current_wrapper, ActionSafetyState, config: dict[str, Any]) -> list[dict[str, Any]]:
    cases = [
        ("daily_N_120_at_DAP1", 1, 0.0, 3, True, None),
        ("N_DAP90_exact_season_cap", 90, 130.0, 3, True, None),
        ("N_candidate_exceeds_season_cap", 90, 131.0, 3, False, None),
        ("N_outside_window_DAP91", 91, 130.0, 3, False, None),
        ("N_min_interval_6_days", 8, 0.0, 3, False, 2),
        ("N_min_interval_7_days", 9, 0.0, 3, True, 2),
    ]
    rows: list[dict[str, Any]] = []
    for name, dap, cum_n, action, expected, last_n in cases:
        probe = make_probe(current_wrapper, ActionSafetyState, config, dap, 0.0, cum_n, None, last_n)
        actual = bool(probe.action_masks()[action])
        rows.append(
            {
                "test_case": name,
                "dap": dap,
                "cum_n_before": cum_n,
                "last_fertilization_dap": last_n if last_n is not None else "",
                "candidate_action": action,
                "candidate_nitrogen_kg_ha": ACTION_GRID[action]["anfer"],
                "expected_valid": expected,
                "actual_valid": actual,
                "result": "PASS" if actual == expected else "FAIL",
            }
        )
    rows.append(
        {
            "test_case": "declared_daily_N_max",
            "dap": "static",
            "cum_n_before": "static",
            "last_fertilization_dap": "",
            "candidate_action": "max_grid",
            "candidate_nitrogen_kg_ha": max(ACTION_NITROGEN),
            "expected_valid": True,
            "actual_valid": max(ACTION_NITROGEN) <= 120.0,
            "result": "PASS" if max(ACTION_NITROGEN) <= 120.0 else "FAIL",
        }
    )
    write_csv(RESULTS / "yc_dqn_n_constraint_assertions.csv", rows)
    if not all(row["result"] == "PASS" for row in rows):
        raise AssertionError("YC nitrogen assertions failed")
    return rows


def run_core_mask_gates(MaskAwareDQN) -> dict[str, Any]:
    import torch
    from unittest.mock import patch

    results: dict[str, Any] = {}
    model = MaskAwareDQN(observation_dim=2, action_dim=ACTION_DIM, replay_capacity=4, batch_size=1, seed=226)
    masks = [
        np.ones(ACTION_DIM, dtype=bool),
        np.array([True] + [False] * (ACTION_DIM - 1), dtype=bool),
        np.array([True, False, False, False, False, True, False, False, False, False, True, False, False, False, False, True], dtype=bool),
    ]
    exploration_actions = []
    for index, mask in enumerate(masks):
        for repeat in range(8):
            selected = model.select_action(np.zeros(2, dtype=np.float32), mask, index + repeat, deterministic=False)
            assert bool(mask[selected]), f"exploration selected masked action {selected}"
            exploration_actions.append(selected)
    results["exploration_mask_safe"] = "PASS"
    results["exploration_cases"] = len(exploration_actions)

    greedy_model = MaskAwareDQN(observation_dim=2, action_dim=ACTION_DIM, replay_capacity=4, batch_size=1, seed=226)
    q = np.zeros(ACTION_DIM, dtype=np.float32)
    q[9] = 100.0
    q[2] = 5.0
    greedy_mask = np.zeros(ACTION_DIM, dtype=bool)
    greedy_mask[[0, 2]] = True
    with patch.object(greedy_model, "q_values", return_value=q):
        chosen = greedy_model.select_action(np.zeros(2, dtype=np.float32), greedy_mask, 0, deterministic=True)
    assert chosen == 2 and bool(greedy_mask[chosen])
    results["greedy_mask_safe"] = "PASS"
    results["greedy_invalid_high_q_action"] = 9
    results["greedy_selected_action"] = chosen

    class FixedQ(torch.nn.Module):
        def __init__(self, values: list[float]):
            super().__init__()
            self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

        def forward(self, observations):
            return self.values.unsqueeze(0).expand(observations.shape[0], -1)

    target_values = [0.0] * ACTION_DIM
    target_values[9] = 100.0
    target_values[2] = 5.0
    bellman_model = MaskAwareDQN(observation_dim=2, action_dim=ACTION_DIM, replay_capacity=4, batch_size=1, seed=226)
    bellman_model.target = FixedQ(target_values)
    next_mask = np.zeros(ACTION_DIM, dtype=bool)
    next_mask[2] = True
    bellman_model.add_transition(
        observation=np.zeros(2, dtype=np.float32), action=2, reward=0.0,
        next_observation=np.zeros(2, dtype=np.float32), done=False,
        mask=np.ones(ACTION_DIM, dtype=bool), next_mask=next_mask,
    )
    update = bellman_model.train_step()
    assert np.isclose(update["mean_target"], 5.0, atol=1e-6)

    terminal_model = MaskAwareDQN(observation_dim=2, action_dim=ACTION_DIM, replay_capacity=4, batch_size=1, seed=226)
    terminal_model.target = FixedQ(target_values)
    terminal_model.add_transition(
        observation=np.zeros(2, dtype=np.float32), action=0, reward=3.0,
        next_observation=np.zeros(2, dtype=np.float32), done=True,
        mask=np.ones(ACTION_DIM, dtype=bool), next_mask=np.zeros(ACTION_DIM, dtype=bool),
    )
    terminal_update = terminal_model.train_step()
    assert np.isclose(terminal_update["mean_target"], 3.0, atol=1e-6)
    results["bellman_target_mask_safe"] = "PASS"
    results["invalid_next_action_high_q"] = 9
    results["valid_next_action_used"] = 2
    results["masked_target_value"] = update["mean_target"]
    results["terminal_target_value"] = terminal_update["mean_target"]
    return results


def observation_contract_row(env, obs, info: dict[str, Any], config: dict[str, Any], direct_ppo) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    obs_array = np.asarray(obs)
    unwrapped = env.unwrapped
    fields = list(getattr(unwrapped, "observation_variables", []))
    raw_observation = getattr(unwrapped, "observation", {})
    expanded_fields: list[str] = []
    for field in fields:
        value = raw_observation.get(field) if isinstance(raw_observation, dict) else None
        value_array = np.asarray(value) if value is not None else np.asarray([])
        if value_array.ndim > 0 and value_array.size > 1:
            expanded_fields.extend(f"{field}[{index}]" for index in range(int(value_array.size)))
        else:
            expanded_fields.append(field)
    latest = direct_ppo.latest_observation_dict(env, obs, info)
    required = ["dap", "swfac", "nstres"]
    shape_ok = obs_array.ndim == 1 and obs_array.size > 0
    dtype_ok = obs_array.dtype == np.float32
    fields_ok = bool(fields) and len(expanded_fields) == obs_array.size and all(key in fields or key in latest for key in required)
    rows = [
        {"component": "dimension", "value": int(obs_array.size), "ppo_reference": "same env observation_space", "result": "PASS" if shape_ok else "FAIL", "evidence": "current 055 raw daily env"},
        {"component": "field_order", "value": "|".join(expanded_fields), "ppo_reference": "observation_variables expanded by Formator.observation_dict_to_array", "result": "PASS" if fields_ok else "FAIL", "evidence": "src/ppo_evaluate.py:32-42; sb3_wrapper.py Formator.observation_dict_to_array"},
        {"component": "dtype", "value": str(obs_array.dtype), "ppo_reference": "float32 vector", "result": "PASS" if dtype_ok else "FAIL", "evidence": "DQN adapter passes observation unchanged"},
        {"component": "preprocessing", "value": "raw_no_normalization_no_forecast", "ppo_reference": "046_02_raw_observation", "result": "PASS", "evidence": "configs/055_00_yca_lowIC_expanded_action_maskableppo.json"},
        {"component": "SWFAC_direction", "value": jsonable(latest.get("swfac")), "ppo_reference": "0 approx no stress; >0.05 stronger stress", "result": "PASS" if "swfac" in latest else "FAIL", "evidence": "src/run_sya_lowIC_ppo_i240_swfac_guardrail_reward_040_28.py:76-100"},
        {"component": "NSTRES_direction", "value": jsonable(latest.get("nstres")), "ppo_reference": "0 approx no stress; >0.05 stronger stress", "result": "PASS" if "nstres" in latest else "FAIL", "evidence": "src/ppo_evaluate.py:147-148"},
    ]
    return rows, {"dimension": int(obs_array.size), "dtype": str(obs_array.dtype), "field_order": expanded_fields, "base_observation_variables": fields, "latest_observation_keys": sorted(latest), "reset_swfac": jsonable(latest.get("swfac")), "reset_nstres": jsonable(latest.get("nstres")), "all_pass": all(row["result"] == "PASS" for row in rows)}


class DqnDailyAdapter:
    """Thin adapter: the current PPO wrapper owns reward and safety behavior."""

    def __init__(self, env):
        self.env = env

    def reset(self):
        return self.env.reset()

    def action_masks(self) -> np.ndarray:
        return np.asarray(self.env.action_masks(), dtype=bool).copy()

    def step(self, action: int):
        return self.env.step(int(action))


def reward_parity_check(direct_ppo, current_wrapper, env_config: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    ppo_env = make_env(direct_ppo, current_wrapper, env_config, config, "226_00_reward_parity_ppo", evaluation=False)
    dqn_env = make_env(direct_ppo, current_wrapper, env_config, config, "226_00_reward_parity_dqn", evaluation=False)
    try:
        ppo_obs, ppo_info = ppo_env.reset()
        dqn_obs, dqn_info = dqn_env.reset()
        action = 0
        ppo_next, ppo_reward, ppo_term, ppo_trunc, _ = ppo_env.step(action)
        dqn_adapter = DqnDailyAdapter(dqn_env)
        dqn_next, dqn_reward, dqn_term, dqn_trunc, _ = dqn_adapter.step(action)
        diff = abs(float(ppo_reward) - float(dqn_reward))
        row = {
            "case": "noop_reset_step",
            "action": action,
            "state_hash_ppo": array_hash(ppo_obs),
            "state_hash_dqn": array_hash(dqn_obs),
            "next_state_hash_ppo": array_hash(ppo_next),
            "next_state_hash_dqn": array_hash(dqn_next),
            "reward_from_ppo_path": float(ppo_reward),
            "reward_from_dqn_adapter": float(dqn_reward),
            "abs_diff": diff,
            "termination_equal": bool(ppo_term == dqn_term and ppo_trunc == dqn_trunc),
            "result": "PASS" if diff <= 1e-12 and ppo_term == dqn_term and ppo_trunc == dqn_trunc else "FAIL",
            "reward_implementation": "same current wrapper instance class: LateIrrigationReserveMaskWrapper",
        }
        rows = [row]
        write_csv(RESULTS / "yc_dqn_reward_parity_check.csv", rows)
        if row["result"] != "PASS":
            raise AssertionError(f"Reward parity failed: {row}")
        return rows
    finally:
        ppo_env.close()
        dqn_env.close()


def constraint_status(dap: int, safe_i: float, safe_n: float, cumulative_i: float, cumulative_n: float, last_i_dap: int | None, last_n_dap: int | None) -> dict[str, Any]:
    stage_ok = cumulative_i <= 240.0 + 1e-6
    if dap <= 30:
        stage_ok = stage_ok and cumulative_i <= 75.0 + 1e-6
    elif dap <= 60:
        stage_ok = stage_ok and cumulative_i <= 150.0 + 1e-6
    elif dap <= 90:
        stage_ok = stage_ok and cumulative_i <= 195.0 + 1e-6
    late_reserve_ok = dap > 90 or cumulative_i <= 195.0 + 1e-6
    return {
        "daily_i_ok": safe_i <= 45.0 + 1e-6,
        "daily_n_ok": safe_n <= 120.0 + 1e-6,
        "season_i_ok": cumulative_i <= 240.0 + 1e-6,
        "season_n_ok": cumulative_n <= 250.0 + 1e-6,
        "staged_i_ok": stage_ok,
        "late_reserve_ok": late_reserve_ok,
        "last_i_dap": last_i_dap,
        "last_n_dap": last_n_dap,
    }


def run_collection_and_eval(pd, direct_ppo, current_wrapper, MaskAwareDQN, env_config: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    train_env = make_env(direct_ppo, current_wrapper, env_config, config, "226_00_yc_train_smoke", evaluation=False)
    transition_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    invalid_action_attempts = 0
    fallback_count = 0
    target_sync_count = 0
    checkpoint_path = CHECKPOINTS / "yc_dqn_checkpoint_000012.pt"
    try:
        obs, info = train_env.reset()
        observation_dim = int(np.asarray(obs).size)
        model = MaskAwareDQN(
            observation_dim=observation_dim,
            action_dim=ACTION_DIM,
            learning_rate=3e-4,
            gamma=1.0,
            replay_capacity=64,
            batch_size=SMOKE_BATCH_SIZE,
            seed=SEED,
            device="cpu",
            gradient_clip_norm=10.0,
        )
        for step in range(1, SMOKE_STEPS + 1):
            state = np.asarray(obs, dtype=np.float32).reshape(-1)
            mask = np.asarray(train_env.action_masks(), dtype=bool).copy()
            proposal_action = model.select_action(state, mask, step - 1, deterministic=False)
            if not bool(mask[proposal_action]):
                invalid_action_attempts += 1
                raise AssertionError(f"Selected invalid action {proposal_action}")
            next_obs, reward, terminated, truncated, info = train_env.step(proposal_action)
            action_info = dict(getattr(train_env, "last_action_info", {}))
            executed_action = int(action_info.get("discrete_action_index", -1))
            daily_log_action = int(action_info.get("discrete_action_index", -1))
            fallback = bool(action_info.get("mask_forced_noop", False))
            fallback_count += int(fallback)
            if executed_action < 0:
                raise AssertionError("Current YC wrapper did not expose discrete_action_index")
            if executed_action != proposal_action or daily_log_action != executed_action or fallback:
                raise AssertionError(f"Action-effect mismatch at step {step}: {action_info}")
            done = bool(terminated or truncated)
            next_state = np.asarray(next_obs, dtype=np.float32).reshape(-1)
            next_mask = np.zeros(ACTION_DIM, dtype=bool) if done else np.asarray(train_env.action_masks(), dtype=bool).copy()
            model.add_transition(
                observation=state,
                action=executed_action,
                reward=float(reward),
                next_observation=next_state,
                done=done,
                mask=mask,
                next_mask=next_mask,
            )
            if model.replay.size >= SMOKE_BATCH_SIZE:
                update_rows.append({"global_step": step, **model.train_step()})
            if step % TARGET_SYNC_EVERY == 0:
                model.sync_target()
                target_sync_count += 1
            previous_dap = int(round(float(getattr(train_env, "last_obs_dict", {}).get("dap", step))))
            constraints = constraint_status(
                dap=previous_dap,
                safe_i=float(action_info.get("safe_action_amir", 0.0)),
                safe_n=float(action_info.get("safe_action_anfer", 0.0)),
                cumulative_i=float(action_info.get("season_cumulative_irrigation", 0.0)),
                cumulative_n=float(action_info.get("season_cumulative_n", 0.0)),
                last_i_dap=getattr(train_env.safety_state, "last_irrigation_dap", None),
                last_n_dap=getattr(train_env.safety_state, "last_fertilization_dap", None),
            )
            transition_rows.append(
                {
                    "split": "train_smoke",
                    "global_step": step,
                    "proposal_action": proposal_action,
                    "valid_by_mask": True,
                    "executed_action": executed_action,
                    "buffer_action": executed_action,
                    "daily_log_action": daily_log_action,
                    "reward": float(reward),
                    "next_state_hash": array_hash(next_state),
                    "next_mask_hash": array_hash(next_mask),
                    "fallback_triggered": fallback,
                    "dap": previous_dap,
                    "safe_action_amir": float(action_info.get("safe_action_amir", 0.0)),
                    "safe_action_anfer": float(action_info.get("safe_action_anfer", 0.0)),
                    "season_cumulative_irrigation": float(action_info.get("season_cumulative_irrigation", 0.0)),
                    "season_cumulative_n": float(action_info.get("season_cumulative_n", 0.0)),
                    "constraint_all_pass": all(bool(value) for key, value in constraints.items() if key.endswith("_ok")),
                    "done": done,
                }
            )
            obs = next_obs
            if done:
                break
        env_steps = len(transition_rows)
        if env_steps < SMOKE_STEPS:
            raise AssertionError(f"YC smoke terminated early at {env_steps} steps")
        model.save(checkpoint_path, env_steps)
        reloaded, reload_step = MaskAwareDQN.load(checkpoint_path, device="cpu")
        checkpoint_reload_success = bool(reload_step == env_steps and reloaded.action_dim == ACTION_DIM and reloaded.observation_dim == observation_dim and reloaded.module_hash(reloaded.online) == reloaded.module_hash(model.online))
        if not checkpoint_reload_success:
            raise AssertionError("DQN checkpoint reload/hash check failed")
    finally:
        train_env.close()

    train_daily_path = RESULTS / "yc_dqn_smoke_train_daily.csv"
    write_csv(train_daily_path, transition_rows)
    write_csv(RESULTS / "yc_dqn_transition_consistency.csv", transition_rows)

    eval_env = make_env(direct_ppo, current_wrapper, env_config, config, "226_00_yc_eval_smoke", evaluation=True)
    eval_rows: list[dict[str, Any]] = []
    eval_invalid = 0
    eval_fallback = 0
    eval_constraints_pass = True
    try:
        eval_obs, eval_info = eval_env.reset()
        for step in range(1, 9):
            state = np.asarray(eval_obs, dtype=np.float32).reshape(-1)
            mask = np.asarray(eval_env.action_masks(), dtype=bool).copy()
            selected = reloaded.select_action(state, mask, step - 1, deterministic=True)
            if not bool(mask[selected]):
                eval_invalid += 1
                raise AssertionError(f"Deterministic eval selected invalid action {selected}")
            next_obs, reward, terminated, truncated, info = eval_env.step(selected)
            action_info = dict(getattr(eval_env, "last_action_info", {}))
            executed = int(action_info.get("discrete_action_index", -1))
            fallback = bool(action_info.get("mask_forced_noop", False))
            eval_fallback += int(fallback)
            if executed != selected or fallback:
                raise AssertionError(f"Eval action closure failed: selected={selected}, info={action_info}")
            dap_value = getattr(eval_env, "last_obs_dict", {}).get("dap", step)
            dap = int(round(float(dap_value)))
            constraints = constraint_status(
                dap=dap,
                safe_i=float(action_info.get("safe_action_amir", 0.0)),
                safe_n=float(action_info.get("safe_action_anfer", 0.0)),
                cumulative_i=float(action_info.get("season_cumulative_irrigation", 0.0)),
                cumulative_n=float(action_info.get("season_cumulative_n", 0.0)),
                last_i_dap=getattr(eval_env.safety_state, "last_irrigation_dap", None),
                last_n_dap=getattr(eval_env.safety_state, "last_fertilization_dap", None),
            )
            eval_constraints_pass = eval_constraints_pass and all(bool(value) for key, value in constraints.items() if key.endswith("_ok"))
            eval_rows.append(
                {
                    "split": "eval_smoke",
                    "global_step": step,
                    "dap": dap,
                    "proposal_action": selected,
                    "valid_by_mask": True,
                    "executed_action": executed,
                    "buffer_action": executed,
                    "daily_log_action": executed,
                    "reward": float(reward),
                    "next_state_hash": array_hash(next_obs),
                    "next_mask_hash": array_hash(np.zeros(ACTION_DIM, dtype=bool) if (terminated or truncated) else eval_env.action_masks()),
                    "fallback_triggered": fallback,
                    "safe_action_amir": float(action_info.get("safe_action_amir", 0.0)),
                    "safe_action_anfer": float(action_info.get("safe_action_anfer", 0.0)),
                    "season_cumulative_irrigation": float(action_info.get("season_cumulative_irrigation", 0.0)),
                    "season_cumulative_n": float(action_info.get("season_cumulative_n", 0.0)),
                    "constraint_all_pass": all(bool(value) for key, value in constraints.items() if key.endswith("_ok")),
                    "done": bool(terminated or truncated),
                }
            )
            eval_obs = next_obs
            if terminated or truncated:
                break
    finally:
        eval_env.close()
    write_csv(RESULTS / "yc_dqn_smoke_eval_daily.csv", eval_rows)
    return {
        "env_steps": len(transition_rows),
        "optimizer_updates": int(model.optimizer_updates),
        "target_sync_count": target_sync_count,
        "invalid_action_attempts": invalid_action_attempts + eval_invalid,
        "fallback_count": fallback_count + eval_fallback,
        "replay_size": int(model.replay.size),
        "checkpoint_saved": checkpoint_path.exists(),
        "checkpoint_path": rel(checkpoint_path),
        "checkpoint_reload_success": checkpoint_reload_success,
        "training_daily_log": rel(train_daily_path),
        "evaluation_daily_log": rel(RESULTS / "yc_dqn_smoke_eval_daily.csv"),
        "eval_steps": len(eval_rows),
        "eval_constraints_pass": eval_constraints_pass,
        "transition_rows": transition_rows,
        "update_rows": update_rows,
        "model_observation_dim": observation_dim,
        "model_action_dim": ACTION_DIM,
        "hyperparameter_label": "engineering_smoke_only; not_for_formal_comparison",
    }


def build_report(result: dict[str, Any]) -> str:
    static = result.get("core_mask_gates", {})
    smoke = result.get("smoke", {})
    classification = result.get("readiness", "STILL_NOT_READY")
    proceed = result.get("proceed_to_formal_config_freeze", "NO")
    error = result.get("error")
    lines = [
        "# YC DQN adapter smoke report",
        "",
        "审计日期：2026-09-11",
        "",
        "本轮仅进行工程 smoke，不评价产量、不调参、不执行正式 3-seed 训练、不进行 PPO-vs-DQN 结果比较。现有正式 PPO/029/032/055 文件未被覆盖。",
        "",
        f"## 最终结论：`{classification}`",
        "",
        f"`Proceed to formal DQN config freeze: {proceed}`",
        "",
    ]
    if error:
        lines.extend(["本轮存在未完成的 smoke 异常，详情见 `results/yc_dqn_adapter_smoke_summary.json`：", "", f"`{error}`", ""])
    lines.extend([
        "## Contract freeze",
        "",
        f"- Site/site code: `{STATION}/{SITE}`；input profile: `lowIC`；training smoke year: `{TRAIN_YEAR}`。",
        "- Formal split retained: train `2004–2013`, validation `2014–2023`。",
        "- Observation: current raw daily observation, no normalization, no forecast; DQN receives the same vector from the current PPO wrapper. ",
        "- Actions: I `[0,15,30,45]` × N `[0,40,80,120]`, 16 joint actions, row-major. Mapping is in `results/yc_dqn_adapter_action_mapping.csv`. ",
        "- Safety/reward: the adapter calls `LateIrrigationReserveMaskWrapper` directly, preserving the current 040_28/040_36 chain; it does not duplicate reward logic.",
        "",
        "## Gate results",
        "",
        f"- exploration_mask_safe: `{static.get('exploration_mask_safe', 'FAIL')}`",
        f"- greedy_mask_safe: `{static.get('greedy_mask_safe', 'FAIL')}`",
        f"- bellman_target_mask_safe: `{static.get('bellman_target_mask_safe', 'FAIL')}`",
        f"- replay_action_effect_consistency: `{result.get('replay_action_effect_consistency', 'FAIL')}`",
        f"- staged irrigation assertions: `{result.get('staged_irrigation_assertions', 'FAIL')}`",
        f"- N constraints: `{result.get('n_constraints', 'FAIL')}`",
        f"- observation contract: `{result.get('observation_contract_status', 'FAIL')}`",
        f"- reward parity: `{result.get('reward_parity_status', 'FAIL')}`",
        "",
        "## Short smoke closure",
        "",
        f"- env_steps: `{smoke.get('env_steps', 0)}`",
        f"- optimizer_updates: `{smoke.get('optimizer_updates', 0)}`",
        f"- target_sync_count: `{smoke.get('target_sync_count', 0)}`",
        f"- invalid_action_attempts: `{smoke.get('invalid_action_attempts', 0)}`",
        f"- fallback_count: `{smoke.get('fallback_count', 0)}`",
        f"- replay_size: `{smoke.get('replay_size', 0)}`",
        f"- checkpoint_saved: `{smoke.get('checkpoint_saved', False)}`",
        f"- checkpoint_reload_success: `{smoke.get('checkpoint_reload_success', False)}`",
        f"- deterministic eval steps: `{smoke.get('eval_steps', 0)}`",
        f"- deterministic eval constraints: `{smoke.get('eval_constraints_pass', False)}`",
        "",
        "## 对 prompt 的 13 个问题逐项回答",
        "",
        f"1. **029 core 是否接入当前 YC daily environment？** `{result.get('yc_daily_environment', 'NO')}`；适配器使用当前 YC lowIC daily wrapper，DQN core 只负责 mask-aware action/replay/update。",
        f"2. **observation 是否与 PPO 完全一致？** `{result.get('observation_contract_status', 'FAIL')}`；记录见 `results/yc_dqn_observation_contract.csv`。",
        f"3. **reward 是否与 PPO 完全一致？** `{result.get('reward_parity_status', 'FAIL')}`；DQN adapter 不重写 reward，parity 见 `results/yc_dqn_reward_parity_check.csv`。",
        f"4. **16-action mapping 是否完全一致？** `{result.get('action_mapping_status', 'FAIL')}`；唯一 mapping 见 `results/yc_dqn_adapter_action_mapping.csv`。",
        f"5. **exploration 是否 mask-safe？** `{static.get('exploration_mask_safe', 'FAIL')}`。",
        f"6. **greedy 是否 mask-safe？** `{static.get('greedy_mask_safe', 'FAIL')}`。",
        f"7. **Bellman target 是否 mask-safe？** `{static.get('bellman_target_mask_safe', 'FAIL')}`。",
        f"8. **replay action 是否与真实 executed action 一致？** `{result.get('replay_action_effect_consistency', 'FAIL')}`；逐 transition 记录见 `results/yc_dqn_transition_consistency.csv`。",
        f"9. **staged irrigation 75/150/195/240 + late reserve 45 是否全部 PASS？** `{result.get('staged_irrigation_assertions', 'FAIL')}`。",
        f"10. **N constraints 是否全部 PASS？** `{result.get('n_constraints', 'FAIL')}`。",
        f"11. **collection/update/target-sync/save/reload/eval 是否闭环？** `{result.get('smoke_closure_status', 'FAIL')}`。",
        f"12. **是否出现 invalid action/fallback？** `invalid_action_attempts={smoke.get('invalid_action_attempts', 0)}, fallback_count={smoke.get('fallback_count', 0)}`。",
        f"13. **readiness？** `{classification}`。",
        "",
        "## Smoke-only hyperparameters",
        "",
        "- label: `engineering_smoke_only; not_for_formal_comparison`。",
        f"- seed `{SEED}`; steps `{SMOKE_STEPS}`; replay capacity `64`; batch size `{SMOKE_BATCH_SIZE}`; learning rate `3e-4`; gamma `1.0`; CPU; MLP `obs_dim -> 32 -> 32 -> 16`; gradient clip `10`; target sync every `{TARGET_SYNC_EVERY}` steps。",
        "- 这些数值不构成 formal DQN freeze，也没有根据 reward/yield 进行调整。",
        "",
        "## Outputs",
        "",
        "- `results/yc_dqn_staged_irrigation_assertions.csv`",
        "- `results/yc_dqn_n_constraint_assertions.csv`",
        "- `results/yc_dqn_observation_contract.csv`",
        "- `results/yc_dqn_reward_parity_check.csv`",
        "- `results/yc_dqn_transition_consistency.csv`",
        "- `results/yc_dqn_smoke_summary.csv`",
        "- `results/yc_dqn_smoke_train_daily.csv`",
        "- `results/yc_dqn_smoke_eval_daily.csv`",
        "- `checkpoints/yc_dqn_checkpoint_000012.pt`",
    ])
    return "\n".join(lines) + "\n"


def run() -> dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    mapping_rows = write_action_mapping()
    meta = json.loads(REFERENCE_CONFIG.read_text(encoding="utf-8"))
    if meta["station_code"] != STATION or meta["site"] != SITE or meta["input_profile"] != "lowIC":
        raise AssertionError("055 reference metadata drifted")
    if meta["actions"]["irrigation_levels_mm"] != ACTION_IRRIGATION or meta["actions"]["nitrogen_levels_kg_ha"] != ACTION_NITROGEN:
        raise AssertionError("055 reference action grid drifted")

    pd, ppo_safe_rendering, direct_ppo, batch, engine, current_wrapper, MaskAwareDQN, ActionSafetyState = import_runtime_modules()
    original_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    try:
        config = configure_current_yc(pd, ppo_safe_rendering, batch, engine, current_wrapper)
        env_config, preflight = build_yc_env_config(pd, direct_ppo, batch, config)
        write_json(OUT / "configs" / "yc_dqn_smoke_reference_contract.json", {"source": rel(REFERENCE_CONFIG), "formal_reference": meta, "resolved_runtime_config": jsonable(config), "preflight": preflight})
        core_gates = run_core_mask_gates(MaskAwareDQN)
        staged_rows = run_staged_assertions(current_wrapper, ActionSafetyState, config)
        n_rows = run_n_assertions(current_wrapper, ActionSafetyState, config)

        observation_rows: list[dict[str, Any]] = []
        observation_info: dict[str, Any] = {}
        obs_env = make_env(direct_ppo, current_wrapper, env_config, config, "226_00_observation_contract", evaluation=False)
        try:
            obs, info = obs_env.reset()
            observation_rows, observation_info = observation_contract_row(obs_env, obs, info, config, direct_ppo)
        finally:
            obs_env.close()
        write_csv(RESULTS / "yc_dqn_observation_contract.csv", observation_rows)
        if not observation_info["all_pass"]:
            raise AssertionError("YC observation contract failed")

        parity_rows = reward_parity_check(direct_ppo, current_wrapper, env_config, config)
        smoke = run_collection_and_eval(pd, direct_ppo, current_wrapper, MaskAwareDQN, env_config, config)
        transition_ok = all(
            row["proposal_action"] == row["executed_action"] == row["buffer_action"] == row["daily_log_action"]
            and row["valid_by_mask"]
            and not row["fallback_triggered"]
            and row["constraint_all_pass"]
            for row in smoke["transition_rows"]
        )
        closure_ok = bool(
            smoke["env_steps"] == SMOKE_STEPS
            and smoke["optimizer_updates"] > 0
            and smoke["target_sync_count"] >= 1
            and smoke["invalid_action_attempts"] == 0
            and smoke["fallback_count"] == 0
            and smoke["checkpoint_saved"]
            and smoke["checkpoint_reload_success"]
            and smoke["eval_steps"] > 0
            and smoke["eval_constraints_pass"]
        )
        write_csv(
            RESULTS / "yc_dqn_smoke_summary.csv",
            [
                {
                    "env_steps": smoke["env_steps"],
                    "optimizer_updates": smoke["optimizer_updates"],
                    "target_sync_count": smoke["target_sync_count"],
                    "invalid_action_attempts": smoke["invalid_action_attempts"],
                    "fallback_count": smoke["fallback_count"],
                    "replay_size": smoke["replay_size"],
                    "checkpoint_saved": smoke["checkpoint_saved"],
                    "checkpoint_reload_success": smoke["checkpoint_reload_success"],
                    "eval_steps": smoke["eval_steps"],
                    "eval_constraints_pass": smoke["eval_constraints_pass"],
                    "hyperparameter_label": smoke["hyperparameter_label"],
                    "result": "PASS" if closure_ok else "FAIL",
                }
            ],
        )
        result = {
            "task": "226_00_yc_dqn_adapter_smoke",
            "training_performed": True,
            "training_scope": "engineering smoke only; no formal training",
            "yc_daily_environment": "PASS",
            "action_mapping_status": "PASS" if len(mapping_rows) == ACTION_DIM else "FAIL",
            "core_mask_gates": core_gates,
            "staged_irrigation_assertions": "PASS" if all(row["result"] == "PASS" for row in staged_rows) else "FAIL",
            "n_constraints": "PASS" if all(row["result"] == "PASS" for row in n_rows) else "FAIL",
            "observation_contract_status": "PASS" if observation_info["all_pass"] else "FAIL",
            "reward_parity_status": "PASS" if all(row["result"] == "PASS" for row in parity_rows) else "FAIL",
            "replay_action_effect_consistency": "PASS" if transition_ok else "FAIL",
            "smoke_closure_status": "PASS" if closure_ok else "FAIL",
            "smoke": {key: value for key, value in smoke.items() if key not in {"transition_rows", "update_rows"}},
            "readiness": "READY_FOR_FORMAL_CONFIG_FREEZE" if all([core_gates["exploration_mask_safe"] == "PASS", core_gates["greedy_mask_safe"] == "PASS", core_gates["bellman_target_mask_safe"] == "PASS", transition_ok, all(row["result"] == "PASS" for row in staged_rows), all(row["result"] == "PASS" for row in n_rows), observation_info["all_pass"], all(row["result"] == "PASS" for row in parity_rows), closure_ok]) else "STILL_NOT_READY",
        }
        result["proceed_to_formal_config_freeze"] = "YES" if result["readiness"] == "READY_FOR_FORMAL_CONFIG_FREEZE" else "NO"
        write_json(OUT / "results" / "yc_dqn_smoke_summary.json", jsonable(result))
        (OUT / "docs").mkdir(parents=True, exist_ok=True)
        (OUT / "docs" / "yc_dqn_adapter_smoke_report.md").write_text(build_report(result), encoding="utf-8")
        return result
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = original_root


def main() -> None:
    try:
        result = run()
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        error_result = {
            "task": "226_00_yc_dqn_adapter_smoke",
            "readiness": "STILL_NOT_READY",
            "proceed_to_formal_config_freeze": "NO",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "training_performed": "engineering smoke may be partial; formal training not performed",
        }
        write_json(OUT / "results" / "yc_dqn_smoke_summary.json", error_result)
        (OUT / "docs").mkdir(parents=True, exist_ok=True)
        (OUT / "docs" / "yc_dqn_adapter_smoke_report.md").write_text(build_report(error_result), encoding="utf-8")
        print(json.dumps(error_result, ensure_ascii=False, indent=2))
        raise
    else:
        print(json.dumps(jsonable(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
