"""Task 181: transparent late-stage 7/14-day action counterfactual.

The core experiment is intentionally written as a linear, printable workflow
inside ``run()``.  Each block corresponds to a step that can be shown to a
supervisor or stopped in a debugger:

1. build the task-177 runtime;
2. replay PPO-2000 to DAP 96/110;
3. list the actual valid actions;
4. restart DSSAT, replay the identical prefix, and apply one candidate action;
5. apply no-op for the remaining 13 days;
6. save daily, 7-day, 14-day, ranking, and comparison tables.

There is no PPO training and no hidden whole-season optimizer in this file.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import run_178_sya_weather_signal_pathway_audit as task178


TASK_ID = "181"
TASK_NAME = "sya_late_stage_short_horizon_counterfactual"
DEFAULT_CONFIG = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / "2026-08-20_sya_late_stage_short_horizon_counterfactual.md"
MODEL = (
    ROOT
    / "benchmark_results"
    / "177_sya_dense_yield_credit_smoke2k"
    / "models"
    / "SYA"
    / "SYA_half_split_stress_aware_maskableppo_seed0_ckpt2000.zip"
)

STATION = "SYA"
SEED = 0
CHECKPOINT = 2000
TARGET_DAPS = [96, 110]
HORIZONS = [7, 14]
MAX_HORIZON = 14
NO_OP_ACTION = 0
SCENARIOS = {
    2005: "original",
    2006: "early_dry",
    2007: "mid_dry",
    2008: "late_dry",
    2009: "early_wet",
    2010: "mid_wet",
}


def number(value: Any, default: float = np.nan) -> float:
    """Convert DSSAT/gym scalar-like values to a finite float."""
    try:
        values = np.asarray(value, dtype=float).reshape(-1)
        result = float(values[0]) if values.size else default
        return result if np.isfinite(result) else default
    except Exception:
        return default


def one_action(action: Any) -> int:
    return int(np.asarray(action).reshape(-1)[0])


def current_state(env, obs: Any, fallback_dap: int = -1) -> dict[str, float]:
    """Small, explicit state snapshot used in CSVs and replay checks."""
    observed = dict(getattr(env, "last_obs_dict", {}) or {})
    obs_array = np.asarray(obs, dtype=np.float32).reshape(-1)
    safety = getattr(env, "safety_state", None)
    return {
        "dap": int(round(number(observed.get("dap"), fallback_dap))),
        "rain": number(obs_array[-1], 0.0),
        "swfac": number(observed.get("swfac")),
        "nstres": number(observed.get("nstres")),
        "grnwt": number(observed.get("grnwt"), 0.0),
        "topwt": number(observed.get("topwt")),
        "xlai": number(observed.get("xlai")),
        "cumulative_irrigation": number(getattr(safety, "cumulative_irrigation", np.nan)),
        "cumulative_nitrogen": number(getattr(safety, "cumulative_nitrogen", np.nan)),
    }


def mask_text(mask: np.ndarray) -> str:
    return "".join("1" if bool(item) else "0" for item in np.asarray(mask, dtype=bool).reshape(-1))


def grid_action(env, action_index: int) -> dict[str, float]:
    raw = dict(list(getattr(env, "grid"))[int(action_index)])
    return {
        "irrigation_mm": number(raw.get("amir"), 0.0),
        "nitrogen_kg_ha": number(raw.get("anfer"), 0.0),
    }


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def tie_aware_action_signature(group: pd.DataFrame, value_column: str) -> str:
    """Show equal action values as ties, for example ``0=4=8=12``."""
    ordered = group.sort_values([value_column, "candidate_action_index"], ascending=[False, True])
    levels: list[list[int]] = []
    level_values: list[float] = []
    for row in ordered.itertuples(index=False):
        value = float(getattr(row, value_column))
        action = int(row.candidate_action_index)
        if not level_values or not np.isclose(value, level_values[-1], rtol=0.0, atol=1e-9):
            level_values.append(value)
            levels.append([action])
        else:
            levels[-1].append(action)
    return ">".join("=".join(str(action) for action in level) for level in levels)


def run(config_path: Path) -> dict[str, Any]:
    # ------------------------------------------------------------------
    # Step 0. Read and verify the experiment contract.
    # ------------------------------------------------------------------
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg.get("training_executed") is not False:
        raise ValueError("Task 181 is read-only and must not train PPO.")
    if [int(x) for x in cfg["target_daps"]] != TARGET_DAPS:
        raise ValueError(f"Target DAP drift: {cfg['target_daps']} != {TARGET_DAPS}")
    if [int(x) for x in cfg["horizons_days"]] != HORIZONS:
        raise ValueError(f"Horizon drift: {cfg['horizons_days']} != {HORIZONS}")
    if int(cfg["continuation_action_index"]) != NO_OP_ACTION:
        raise ValueError("Task 181 requires action 0 as the fixed continuation action.")
    if not MODEL.exists():
        raise FileNotFoundError(MODEL)
    if OUT.exists():
        raise FileExistsError(f"Refuse to overwrite existing output: {relative(OUT)}")

    task168 = task178.task177.task174.task173.task172.task168
    gate = task168.verify_gate(cfg)
    manifest = task168.read_manifest(cfg)
    split = task168.load_split(manifest)

    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "configs").mkdir()
    (OUT / "debug").mkdir()
    (OUT / "evaluation").mkdir()
    (OUT / "configs" / config_path.name).write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (OUT / "configs" / "181_scenario_split.csv").write_text(split.to_csv(index=False), encoding="utf-8-sig")
    (OUT / "configs" / "167_physical_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"[Step 0/4] Contract checked: no training; target DAPs {TARGET_DAPS}; horizons {HORIZONS} days.",
        flush=True,
    )

    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks

    # ------------------------------------------------------------------
    # Step 1. Patch only the already-audited task-177 runtime.
    # The assignments are kept here instead of hidden in another runner.
    # ------------------------------------------------------------------
    saved_task178_globals = {
        "TASK_ID": task178.TASK_ID,
        "TASK_NAME": task178.TASK_NAME,
        "OUT": task178.OUT,
        "DOC": task178.DOC,
        "PROMPT": task178.PROMPT,
    }
    task178.TASK_ID = TASK_ID
    task178.TASK_NAME = TASK_NAME
    task178.OUT = OUT
    task178.DOC = DOC
    task178.PROMPT = PROMPT
    saved_runtime = task178.patch_runtime_178(manifest)

    prefix_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []

    try:
        engine = task168.engine
        base03222 = task168.base03222
        runtime_config = engine.load_config()
        env_config = base03222.direct_ppo.build_env_config(runtime_config, split)
        model = MaskablePPO.load(str(MODEL), device="cpu")

        # --------------------------------------------------------------
        # Step 2. Loop through six weather scenarios and two target DAPs.
        # First make one reference replay and save every PPO prefix action.
        # --------------------------------------------------------------
        branch_counter = 0
        expected_branch_count = len(SCENARIOS) * len(TARGET_DAPS) * 4
        for scenario_key, weather_variant in SCENARIOS.items():
            for target_dap in TARGET_DAPS:
                print(
                    f"[Step 1/4] Build prefix: scenario={weather_variant}, target_DAP={target_dap}",
                    flush=True,
                )

                reference_env = base03222.base.make_env(
                    runtime_config,
                    env_config,
                    STATION,
                    scenario_key,
                    SEED,
                    f"SYA_2005_{weather_variant}_181_reference_dap{target_dap}",
                    evaluation=True,
                )
                prefix_actions: list[int] = []
                try:
                    reference_obs, reference_info = reference_env.reset()
                    reference_step = 0
                    while reference_step < int(runtime_config["runtime"]["max_steps"]):
                        state_before = current_state(reference_env, reference_obs, reference_step)
                        if int(state_before["dap"]) == target_dap:
                            target_mask = np.asarray(get_action_masks(reference_env), dtype=bool).reshape(-1)
                            target_state = dict(state_before)
                            target_observation = np.asarray(reference_obs, dtype=np.float32).reshape(-1).copy()
                            ppo_action_raw, _ = model.predict(
                                target_observation,
                                action_masks=target_mask,
                                deterministic=True,
                            )
                            ppo_target_action = one_action(ppo_action_raw)
                            break

                        prefix_mask = np.asarray(get_action_masks(reference_env), dtype=bool).reshape(-1)
                        ppo_action_raw, _ = model.predict(
                            np.asarray(reference_obs, dtype=np.float32).reshape(-1),
                            action_masks=prefix_mask,
                            deterministic=True,
                        )
                        ppo_action = one_action(ppo_action_raw)
                        if not bool(prefix_mask[ppo_action]):
                            raise RuntimeError(
                                f"Invalid PPO prefix action {ppo_action} at DAP {int(state_before['dap'])}"
                            )
                        raw_prefix_action = grid_action(reference_env, ppo_action)
                        prefix_rows.append(
                            {
                                "scenario_key": scenario_key,
                                "weather_variant": weather_variant,
                                "target_dap": target_dap,
                                "prefix_step": reference_step,
                                "dap": int(state_before["dap"]),
                                "rain": state_before["rain"],
                                "mask": mask_text(prefix_mask),
                                "ppo_action_index": ppo_action,
                                **raw_prefix_action,
                            }
                        )
                        prefix_actions.append(ppo_action)
                        reference_obs, _, terminated, truncated, reference_info = reference_env.step(ppo_action)
                        reference_step += 1
                        if bool(terminated or truncated):
                            raise RuntimeError(
                                f"Reference episode ended before DAP {target_dap}: {weather_variant}"
                            )
                    else:
                        raise RuntimeError(f"Target DAP {target_dap} not reached: {weather_variant}")

                    valid_actions = np.flatnonzero(target_mask).astype(int).tolist()
                    reference_grid = {idx: grid_action(reference_env, idx) for idx in valid_actions}
                    if len(valid_actions) != 4:
                        raise RuntimeError(
                            f"Expected 4 valid actions at DAP {target_dap}, got {valid_actions}"
                        )
                    if NO_OP_ACTION not in valid_actions or reference_grid[NO_OP_ACTION] != {
                        "irrigation_mm": 0.0,
                        "nitrogen_kg_ha": 0.0,
                    }:
                        raise RuntimeError("Action 0 is not the expected no-op continuation action.")
                finally:
                    reference_env.close()

                # Save a partial prefix trace after every target state.  If a
                # later branch fails, the exact replay history remains visible.
                pd.DataFrame(prefix_rows).to_csv(
                    OUT / "debug" / "181_prefix_actions_partial.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

                print(
                    f"[Step 2/4] DAP {target_dap} valid actions: "
                    + ", ".join(
                        f"a{idx}=I{reference_grid[idx]['irrigation_mm']:.0f}/N{reference_grid[idx]['nitrogen_kg_ha']:.0f}"
                        for idx in valid_actions
                    ),
                    flush=True,
                )

                # ------------------------------------------------------
                # Step 3. For each legal action, restart DSSAT and replay
                # exactly the same prefix. Apply the candidate once, then
                # no-op through day 14. Day 7 is read from the same branch.
                # ------------------------------------------------------
                for candidate_action in valid_actions:
                    branch_counter += 1
                    candidate = reference_grid[candidate_action]
                    print(
                        f"[Step 3/4] Branch {branch_counter}/{expected_branch_count}: "
                        f"{weather_variant}, DAP {target_dap}, action {candidate_action} "
                        f"(I={candidate['irrigation_mm']:.0f}, N={candidate['nitrogen_kg_ha']:.0f})",
                        flush=True,
                    )

                    branch_env = base03222.base.make_env(
                        runtime_config,
                        env_config,
                        STATION,
                        scenario_key,
                        SEED,
                        f"SYA_2005_{weather_variant}_181_dap{target_dap}_action{candidate_action}",
                        evaluation=True,
                    )
                    try:
                        branch_obs, branch_info = branch_env.reset()

                        # Explicit deterministic prefix replay.
                        for prefix_step, prefix_action in enumerate(prefix_actions):
                            replay_mask = np.asarray(get_action_masks(branch_env), dtype=bool).reshape(-1)
                            if not bool(replay_mask[prefix_action]):
                                raise RuntimeError(
                                    f"Prefix action {prefix_action} invalid during replay step {prefix_step}"
                                )
                            branch_obs, _, terminated, truncated, branch_info = branch_env.step(prefix_action)
                            if bool(terminated or truncated):
                                raise RuntimeError("Branch ended during prefix replay")

                        branch_start = current_state(branch_env, branch_obs, len(prefix_actions))
                        replay_check_fields = [
                            "rain",
                            "swfac",
                            "nstres",
                            "grnwt",
                            "cumulative_irrigation",
                            "cumulative_nitrogen",
                        ]
                        replay_errors = []
                        for field in replay_check_fields:
                            branch_value = number(branch_start[field])
                            target_value = number(target_state[field])
                            if np.isfinite(branch_value) and np.isfinite(target_value):
                                replay_errors.append(abs(branch_value - target_value))
                        replay_max_error = max(replay_errors) if replay_errors else np.nan
                        if int(branch_start["dap"]) != target_dap or (
                            np.isfinite(replay_max_error) and replay_max_error > 1e-6
                        ):
                            raise RuntimeError(
                                f"Prefix replay mismatch: dap={branch_start['dap']}, error={replay_max_error}"
                            )

                        cumulative_reward = 0.0
                        cumulative_reward_unscaled = 0.0
                        cumulative_yield_component = 0.0
                        cumulative_resource_cost = 0.0
                        cumulative_stress_bonus = 0.0
                        cumulative_rain = 0.0

                        for rollout_day in range(1, MAX_HORIZON + 1):
                            state_before = current_state(branch_env, branch_obs, target_dap + rollout_day - 1)
                            action_to_apply = candidate_action if rollout_day == 1 else NO_OP_ACTION
                            current_mask = np.asarray(get_action_masks(branch_env), dtype=bool).reshape(-1)
                            if not bool(current_mask[action_to_apply]):
                                raise RuntimeError(
                                    f"Action {action_to_apply} invalid on rollout day {rollout_day}; "
                                    f"mask={mask_text(current_mask)}"
                                )

                            applied = grid_action(branch_env, action_to_apply)
                            branch_obs_after, reward, terminated, truncated, branch_info = branch_env.step(
                                action_to_apply
                            )
                            state_after = current_state(
                                branch_env,
                                branch_obs_after,
                                target_dap + rollout_day,
                            )
                            action_info = dict(getattr(branch_env, "last_action_info", {}) or {})

                            reward_unscaled = number(action_info.get("reward_unscaled"), reward)
                            yield_component = number(action_info.get("yield_component"), 0.0)
                            resource_cost = number(action_info.get("resource_cost"), 0.0)
                            stress_bonus = number(action_info.get("stress_relief_bonus"), 0.0)
                            cumulative_reward += float(reward)
                            cumulative_reward_unscaled += reward_unscaled
                            cumulative_yield_component += yield_component
                            cumulative_resource_cost += resource_cost
                            cumulative_stress_bonus += stress_bonus
                            cumulative_rain += state_before["rain"]

                            daily_rows.append(
                                {
                                    "checkpoint_step": CHECKPOINT,
                                    "scenario_key": scenario_key,
                                    "weather_variant": weather_variant,
                                    "target_dap": target_dap,
                                    "candidate_action_index": candidate_action,
                                    "candidate_irrigation_mm": candidate["irrigation_mm"],
                                    "candidate_nitrogen_kg_ha": candidate["nitrogen_kg_ha"],
                                    "ppo_action_index_at_target": ppo_target_action,
                                    "rollout_day": rollout_day,
                                    "action_index_applied": action_to_apply,
                                    "action_irrigation_mm_applied": applied["irrigation_mm"],
                                    "action_nitrogen_kg_ha_applied": applied["nitrogen_kg_ha"],
                                    "dap_before": int(state_before["dap"]),
                                    "dap_after": int(state_after["dap"]),
                                    "rain_before": state_before["rain"],
                                    "swfac_before": state_before["swfac"],
                                    "swfac_after": state_after["swfac"],
                                    "nstres_before": state_before["nstres"],
                                    "nstres_after": state_after["nstres"],
                                    "grnwt_before": state_before["grnwt"],
                                    "grnwt_after": state_after["grnwt"],
                                    "grnwt_increment": number(action_info.get("grnwt_increment"), 0.0),
                                    "reward": float(reward),
                                    "reward_unscaled": reward_unscaled,
                                    "yield_component": yield_component,
                                    "resource_cost": resource_cost,
                                    "stress_relief_bonus": stress_bonus,
                                    "cumulative_reward": cumulative_reward,
                                    "cumulative_reward_unscaled": cumulative_reward_unscaled,
                                    "cumulative_yield_component": cumulative_yield_component,
                                    "cumulative_resource_cost": cumulative_resource_cost,
                                    "cumulative_stress_relief_bonus": cumulative_stress_bonus,
                                    "prefix_replay_max_abs_error": replay_max_error,
                                    "terminated": bool(terminated),
                                    "truncated": bool(truncated),
                                }
                            )

                            if rollout_day in HORIZONS:
                                horizon_rows.append(
                                    {
                                        "checkpoint_step": CHECKPOINT,
                                        "scenario_key": scenario_key,
                                        "weather_variant": weather_variant,
                                        "target_dap": target_dap,
                                        "horizon_days": rollout_day,
                                        "candidate_action_index": candidate_action,
                                        "candidate_irrigation_mm": candidate["irrigation_mm"],
                                        "candidate_nitrogen_kg_ha": candidate["nitrogen_kg_ha"],
                                        "ppo_action_index_at_target": ppo_target_action,
                                        "start_grnwt": branch_start["grnwt"],
                                        "end_grnwt": state_after["grnwt"],
                                        "grnwt_gain": state_after["grnwt"] - branch_start["grnwt"],
                                        "end_swfac": state_after["swfac"],
                                        "end_nstres": state_after["nstres"],
                                        "cumulative_rain": cumulative_rain,
                                        "cumulative_reward": cumulative_reward,
                                        "cumulative_reward_unscaled": cumulative_reward_unscaled,
                                        "cumulative_yield_component": cumulative_yield_component,
                                        "cumulative_resource_cost": cumulative_resource_cost,
                                        "cumulative_stress_relief_bonus": cumulative_stress_bonus,
                                        "prefix_replay_max_abs_error": replay_max_error,
                                    }
                                )

                            branch_obs = branch_obs_after
                            if bool(terminated or truncated):
                                if rollout_day < MAX_HORIZON:
                                    raise RuntimeError(
                                        f"Branch ended at rollout day {rollout_day}, before day {MAX_HORIZON}"
                                    )
                                break
                    finally:
                        branch_env.close()

                    progress_rows.append(
                        {
                            "branch_number": branch_counter,
                            "weather_variant": weather_variant,
                            "target_dap": target_dap,
                            "candidate_action_index": candidate_action,
                            "candidate_irrigation_mm": candidate["irrigation_mm"],
                            "status": "completed",
                        }
                    )

                    # Partial CSVs are deliberately refreshed after every
                    # branch for live debugging and crash recovery.
                    pd.DataFrame(daily_rows).to_csv(
                        OUT / "debug" / "181_rollout_daily_partial.csv",
                        index=False,
                        encoding="utf-8-sig",
                    )
                    pd.DataFrame(horizon_rows).to_csv(
                        OUT / "debug" / "181_horizon_partial.csv",
                        index=False,
                        encoding="utf-8-sig",
                    )
                    pd.DataFrame(progress_rows).to_csv(
                        OUT / "debug" / "181_progress.csv",
                        index=False,
                        encoding="utf-8-sig",
                    )
    except Exception:
        (OUT / "181_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        task178.restore_runtime_178(saved_runtime)
        for key, value in saved_task178_globals.items():
            setattr(task178, key, value)

    # ------------------------------------------------------------------
    # Step 4. Build explicit candidate and weather-ranking tables.
    # ------------------------------------------------------------------
    prefix_frame = pd.DataFrame(prefix_rows)
    daily_frame = pd.DataFrame(daily_rows)
    horizon_frame = pd.DataFrame(horizon_rows)

    noop = horizon_frame[horizon_frame["candidate_action_index"].eq(NO_OP_ACTION)][
        [
            "scenario_key",
            "weather_variant",
            "target_dap",
            "horizon_days",
            "grnwt_gain",
            "cumulative_reward_unscaled",
            "end_swfac",
            "end_nstres",
        ]
    ].rename(
        columns={
            "grnwt_gain": "noop_grnwt_gain",
            "cumulative_reward_unscaled": "noop_reward_unscaled",
            "end_swfac": "noop_end_swfac",
            "end_nstres": "noop_end_nstres",
        }
    )
    candidate_frame = horizon_frame.merge(
        noop,
        on=["scenario_key", "weather_variant", "target_dap", "horizon_days"],
        how="left",
        validate="many_to_one",
    )
    candidate_frame["marginal_grnwt_vs_noop"] = (
        candidate_frame["grnwt_gain"] - candidate_frame["noop_grnwt_gain"]
    )
    candidate_frame["marginal_reward_unscaled_vs_noop"] = (
        candidate_frame["cumulative_reward_unscaled"] - candidate_frame["noop_reward_unscaled"]
    )
    candidate_frame["end_swfac_delta_vs_noop"] = (
        candidate_frame["end_swfac"] - candidate_frame["noop_end_swfac"]
    )
    candidate_frame["end_nstres_delta_vs_noop"] = (
        candidate_frame["end_nstres"] - candidate_frame["noop_end_nstres"]
    )
    candidate_frame["reward_rank"] = candidate_frame.groupby(
        ["weather_variant", "target_dap", "horizon_days"]
    )["cumulative_reward_unscaled"].rank(method="first", ascending=False)
    candidate_frame["grnwt_rank"] = candidate_frame.groupby(
        ["weather_variant", "target_dap", "horizon_days"]
    )["grnwt_gain"].rank(method="first", ascending=False)

    ranking_rows: list[dict[str, Any]] = []
    for (scenario_key, variant, target_dap, horizon), group in candidate_frame.groupby(
        ["scenario_key", "weather_variant", "target_dap", "horizon_days"], sort=True
    ):
        by_reward = group.sort_values(
            ["cumulative_reward_unscaled", "candidate_action_index"], ascending=[False, True]
        )
        by_grnwt = group.sort_values(["grnwt_gain", "candidate_action_index"], ascending=[False, True])
        reward_top = by_reward.iloc[0]
        grnwt_top = by_grnwt.iloc[0]
        ranking_rows.append(
            {
                "scenario_key": int(scenario_key),
                "weather_variant": str(variant),
                "target_dap": int(target_dap),
                "horizon_days": int(horizon),
                "ppo_action_index_at_target": int(group["ppo_action_index_at_target"].iloc[0]),
                "best_reward_action_index": int(reward_top["candidate_action_index"]),
                "best_reward_irrigation_mm": float(reward_top["candidate_irrigation_mm"]),
                "best_reward_unscaled": float(reward_top["cumulative_reward_unscaled"]),
                "best_grnwt_action_index": int(grnwt_top["candidate_action_index"]),
                "best_grnwt_irrigation_mm": float(grnwt_top["candidate_irrigation_mm"]),
                "best_grnwt_gain": float(grnwt_top["grnwt_gain"]),
                "reward_spread_unscaled": float(
                    group["cumulative_reward_unscaled"].max() - group["cumulative_reward_unscaled"].min()
                ),
                "grnwt_gain_spread": float(group["grnwt_gain"].max() - group["grnwt_gain"].min()),
                "reward_ranking_signature": tie_aware_action_signature(
                    group, "cumulative_reward_unscaled"
                ),
                "grnwt_ranking_signature": tie_aware_action_signature(group, "grnwt_gain"),
                "max_abs_marginal_grnwt_vs_noop": float(group["marginal_grnwt_vs_noop"].abs().max()),
                "max_abs_stress_delta_vs_noop": float(
                    max(
                        group["end_swfac_delta_vs_noop"].abs().max(),
                        group["end_nstres_delta_vs_noop"].abs().max(),
                    )
                ),
                "all_prefix_replays_exact": bool(group["prefix_replay_max_abs_error"].max() <= 1e-6),
            }
        )
    ranking_frame = pd.DataFrame(ranking_rows)

    original = ranking_frame[ranking_frame["weather_variant"].eq("original")][
        [
            "target_dap",
            "horizon_days",
            "best_reward_action_index",
            "best_grnwt_action_index",
            "reward_ranking_signature",
            "grnwt_ranking_signature",
        ]
    ].rename(
        columns={
            "best_reward_action_index": "original_best_reward_action_index",
            "best_grnwt_action_index": "original_best_grnwt_action_index",
            "reward_ranking_signature": "original_reward_ranking_signature",
            "grnwt_ranking_signature": "original_grnwt_ranking_signature",
        }
    )
    comparison_frame = ranking_frame[ranking_frame["weather_variant"].ne("original")].merge(
        original,
        on=["target_dap", "horizon_days"],
        how="left",
        validate="many_to_one",
    )
    comparison_frame["best_reward_action_changed_vs_original"] = comparison_frame[
        "best_reward_action_index"
    ].ne(comparison_frame["original_best_reward_action_index"])
    comparison_frame["best_grnwt_action_changed_vs_original"] = comparison_frame[
        "best_grnwt_action_index"
    ].ne(comparison_frame["original_best_grnwt_action_index"])
    comparison_frame["reward_ranking_changed_vs_original"] = comparison_frame[
        "reward_ranking_signature"
    ].ne(comparison_frame["original_reward_ranking_signature"])
    comparison_frame["grnwt_ranking_changed_vs_original"] = comparison_frame[
        "grnwt_ranking_signature"
    ].ne(comparison_frame["original_grnwt_ranking_signature"])

    reward_top_changes = int(comparison_frame["best_reward_action_changed_vs_original"].sum())
    grnwt_top_changes = int(comparison_frame["best_grnwt_action_changed_vs_original"].sum())
    reward_rank_changes = int(comparison_frame["reward_ranking_changed_vs_original"].sum())
    grnwt_rank_changes = int(comparison_frame["grnwt_ranking_changed_vs_original"].sum())
    max_marginal_grnwt = float(candidate_frame["marginal_grnwt_vs_noop"].abs().max())
    max_stress_delta = float(
        max(
            candidate_frame["end_swfac_delta_vs_noop"].abs().max(),
            candidate_frame["end_nstres_delta_vs_noop"].abs().max(),
        )
    )

    if reward_top_changes > 0 or grnwt_top_changes > 0:
        diagnosis = "weather_changes_short_horizon_best_action"
    elif reward_rank_changes > 0 or grnwt_rank_changes > 0:
        diagnosis = "weather_changes_short_horizon_action_ranking_but_not_top_action"
    elif max_marginal_grnwt > 1e-6:
        diagnosis = "candidate_actions_have_delayed_effect_but_weather_does_not_change_ordering"
    elif max_stress_delta > 1e-6:
        diagnosis = "candidate_actions_change_stress_only_but_weather_does_not_change_ordering"
    else:
        diagnosis = "no_detectable_14day_action_effect_under_tested_states"

    prefix_frame.to_csv(OUT / "debug" / "181_prefix_actions.csv", index=False, encoding="utf-8-sig")
    daily_frame.to_csv(OUT / "evaluation" / "181_rollout_daily.csv", index=False, encoding="utf-8-sig")
    candidate_frame.to_csv(OUT / "evaluation" / "181_candidate_horizon_metrics.csv", index=False, encoding="utf-8-sig")
    ranking_frame.to_csv(OUT / "evaluation" / "181_scenario_action_rankings.csv", index=False, encoding="utf-8-sig")
    comparison_frame.to_csv(
        OUT / "evaluation" / "181_weather_ranking_comparison.csv", index=False, encoding="utf-8-sig"
    )

    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "status": "completed",
        "mode": "read_only_short_horizon_counterfactual",
        "training_executed": False,
        "source_task": "177_sya_dense_yield_credit_smoke2k",
        "checkpoint_step": CHECKPOINT,
        "station": STATION,
        "scenario_count": len(SCENARIOS),
        "target_daps": TARGET_DAPS,
        "horizons_days": HORIZONS,
        "continuation_policy": "no_op_after_target_action",
        "candidate_branch_count": int(len(progress_rows)),
        "daily_rows": int(len(daily_frame)),
        "physical_gate_passed": bool(gate.get("physical_response_passed")),
        "diagnosis": {
            "label": diagnosis,
            "best_reward_action_changes_vs_original": reward_top_changes,
            "best_grnwt_action_changes_vs_original": grnwt_top_changes,
            "reward_ranking_changes_vs_original": reward_rank_changes,
            "grnwt_ranking_changes_vs_original": grnwt_rank_changes,
            "max_abs_marginal_grnwt_vs_noop": max_marginal_grnwt,
            "max_abs_stress_delta_vs_noop": max_stress_delta,
            "interpretation": "One target action followed by fixed no-op continuation; not a whole-season optimum.",
        },
        "outputs": {
            "prefix_actions": relative(OUT / "debug" / "181_prefix_actions.csv"),
            "progress": relative(OUT / "debug" / "181_progress.csv"),
            "daily_rollouts": relative(OUT / "evaluation" / "181_rollout_daily.csv"),
            "candidate_metrics": relative(OUT / "evaluation" / "181_candidate_horizon_metrics.csv"),
            "scenario_rankings": relative(OUT / "evaluation" / "181_scenario_action_rankings.csv"),
            "weather_comparison": relative(OUT / "evaluation" / "181_weather_ranking_comparison.csv"),
        },
    }
    (OUT / "181_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    DOC.write_text(
        "\n".join(
            [
                "# Task 181：SYA 后期 7/14 天动作反事实 rollout",
                "",
                "> PPO-2000 只用于生成目标 DAP 前的确定性动作前缀；本实验没有训练。",
                "",
                f"- 情景：`{len(SCENARIOS)}`；目标 DAP：`{TARGET_DAPS}`；horizon：`{HORIZONS}`。",
                f"- 候选分支：`{len(progress_rows)}`；每日记录：`{len(daily_frame)}`。",
                "- 目标日后统一 no-op，候选分支之间唯一改变的是目标日动作。",
                f"- diagnosis：`{diagnosis}`。",
                f"- reward top action 相对 original 改变：`{reward_top_changes}`。",
                f"- GRNWT top action 相对 original 改变：`{grnwt_top_changes}`。",
                f"- 最大相对 no-op GRNWT 差异：`{max_marginal_grnwt}`。",
                "",
                "## 解释边界",
                "",
                "这是 7/14 天固定后续 no-op 的局部边际作用审计，不是整季最优策略搜索。",
                "",
                "## 输出",
                "",
                f"- `{relative(OUT / '181_result.json')}`",
                f"- `{relative(OUT / 'evaluation' / '181_candidate_horizon_metrics.csv')}`",
                f"- `{relative(OUT / 'evaluation' / '181_scenario_action_rankings.csv')}`",
                f"- `{relative(OUT / 'evaluation' / '181_weather_ranking_comparison.csv')}`",
                f"- `{relative(OUT / 'debug' / '181_prefix_actions.csv')}`",
                f"- `{relative(OUT / 'debug' / '181_progress.csv')}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[Step 4/4] Completed. diagnosis={diagnosis}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    print(json.dumps(run(config_path.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
