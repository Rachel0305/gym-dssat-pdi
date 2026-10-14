"""Task 184: critic/one-step TD audit for late PPO irrigation choices.

This read-only diagnostic asks whether the critic locally supports the actor's
DAP97/104 I45 choices.  PPO has a state-value critic, not a Q-function, so the
action score used here is a one-step proxy:

    TD(a) = reward(s, a) + gamma * V(s')

The script keeps the main workflow explicit and linear for demonstration:
load runtime, replay to target state, branch each legal action once, compute
TD proxies, save tables, and summarize actor/critic alignment.
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


TASK_ID = "184"
TASK_NAME = "sya_ppo_critic_td_action_value_audit"
DEFAULT_CONFIG = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / "2026-08-20_sya_ppo_critic_td_action_value_audit.md"
MODEL_ROOT = ROOT / "benchmark_results" / "177_sya_dense_yield_credit_smoke2k" / "models" / "SYA"

STATION = "SYA"
SEED = 0
CHECKPOINTS = [1000, 2000]
TARGET_DAPS = [97, 104]
SCENARIOS = {
    2005: "original",
    2006: "early_dry",
    2007: "mid_dry",
    2008: "late_dry",
    2009: "early_wet",
    2010: "mid_wet",
}
KEY_ACTIONS = {"noop": 0, "i15_n0": 4, "i30_n0": 8, "i45_n0": 12}


def model_path(checkpoint: int) -> Path:
    return MODEL_ROOT / f"SYA_half_split_stress_aware_maskableppo_seed0_ckpt{int(checkpoint)}.zip"


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def number(value: Any, default: float = np.nan) -> float:
    try:
        values = np.asarray(value, dtype=float).reshape(-1)
        result = float(values[0]) if values.size else default
        return result if np.isfinite(result) else default
    except Exception:
        return default


def one_action(action: Any) -> int:
    return int(np.asarray(action).reshape(-1)[0])


def mask_text(mask: np.ndarray) -> str:
    return "".join("1" if bool(item) else "0" for item in np.asarray(mask, dtype=bool).reshape(-1))


def entropy(values: np.ndarray) -> float:
    probabilities = values[np.isfinite(values) & (values > 0)]
    if probabilities.size == 0:
        return np.nan
    return float(-(probabilities * np.log(probabilities)).sum())


def current_state(env, obs: Any, fallback_dap: int = -1) -> dict[str, float]:
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


def grid_action(env, action_index: int) -> dict[str, float]:
    raw = dict(list(getattr(env, "grid"))[int(action_index)])
    return {
        "irrigation_mm": number(raw.get("amir"), 0.0),
        "nitrogen_kg_ha": number(raw.get("anfer"), 0.0),
    }


def policy_distribution(model: Any, obs: Any, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        dist = model.policy.get_distribution(
            obs_tensor,
            action_masks=np.asarray(mask, dtype=bool).reshape(1, -1),
        )
        raw_distribution = getattr(dist, "distribution", dist)
        raw_probs = getattr(raw_distribution, "probs", None)
        raw_logits = getattr(raw_distribution, "logits", None)
        probs = (
            raw_probs.detach().cpu().numpy().reshape(-1).astype(float)
            if raw_probs is not None
            else np.full(len(mask), np.nan)
        )
        logits = (
            raw_logits.detach().cpu().numpy().reshape(-1).astype(float)
            if raw_logits is not None
            else np.full(len(mask), np.nan)
        )
        return probs, logits
    except Exception:
        return np.full(len(mask), np.nan), np.full(len(mask), np.nan)


def state_value(model: Any, obs: Any) -> float:
    import torch

    try:
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            value = model.policy.predict_values(obs_tensor)
        return number(value.detach().cpu().numpy(), np.nan)
    except Exception:
        return np.nan


def probability_rank(probs: np.ndarray, mask: np.ndarray, action_index: int) -> float:
    if action_index >= len(probs) or not bool(mask[action_index]) or not np.isfinite(probs[action_index]):
        return np.nan
    legal_probs = probs[np.asarray(mask, dtype=bool)]
    return float(1 + np.sum(legal_probs > probs[action_index] + 1e-12))


def action_signature(rows: list[int], env) -> str:
    out: list[str] = []
    for step, action_index in enumerate(rows):
        grid = grid_action(env, action_index)
        if grid["irrigation_mm"] > 1e-9 or grid["nitrogen_kg_ha"] > 1e-9:
            out.append(f"step{step}:a{action_index}=I{grid['irrigation_mm']:g}/N{grid['nitrogen_kg_ha']:g}")
    return "; ".join(out)


def run(config_path: Path) -> dict[str, Any]:
    # ------------------------------------------------------------------
    # Step 0. Verify the read-only contract.
    # ------------------------------------------------------------------
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg.get("training_executed") is not False:
        raise ValueError("Task 184 is read-only and must not train PPO.")
    if [int(x) for x in cfg["checkpoint_steps"]] != CHECKPOINTS:
        raise ValueError("Task 184 expects checkpoints [1000, 2000].")
    if [int(x) for x in cfg["target_daps"]] != TARGET_DAPS:
        raise ValueError("Task 184 expects target DAPs [97, 104].")
    if OUT.exists():
        raise FileExistsError(f"Refuse to overwrite existing output: {relative(OUT)}")
    for checkpoint in CHECKPOINTS:
        if not model_path(checkpoint).exists():
            raise FileNotFoundError(model_path(checkpoint))

    task168 = task178.task177.task174.task173.task172.task168
    gate = task168.verify_gate(cfg)
    manifest = task168.read_manifest(cfg)
    split = task168.load_split(manifest)

    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "configs").mkdir()
    (OUT / "debug").mkdir()
    (OUT / "evaluation").mkdir()
    (OUT / "logs").mkdir()
    (OUT / "configs" / config_path.name).write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (OUT / "configs" / "184_scenario_split.csv").write_text(split.to_csv(index=False), encoding="utf-8-sig")
    (OUT / "configs" / "167_physical_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[Step 0/5] Contract checked: read-only critic one-step TD audit.", flush=True)

    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks

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

    target_state_rows: list[dict[str, Any]] = []
    td_rows: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []

    try:
        engine = task168.engine
        base03222 = task168.base03222
        runtime_config = engine.load_config()
        env_config = base03222.direct_ppo.build_env_config(runtime_config, split)
        max_steps = int(runtime_config["runtime"]["max_steps"])

        # --------------------------------------------------------------
        # Step 1. Loop through checkpoints/scenarios/target DAPs.
        # Each target first gets one reference replay to collect prefix actions.
        # --------------------------------------------------------------
        for checkpoint in CHECKPOINTS:
            model = MaskablePPO.load(str(model_path(checkpoint)), device="cpu")
            gamma = float(model.gamma)
            print(f"[Step 1/5] Loaded checkpoint {checkpoint}; gamma={gamma}", flush=True)

            for scenario_key, weather_variant in SCENARIOS.items():
                for target_dap in TARGET_DAPS:
                    print(
                        f"[Step 2/5] Reference replay: ckpt={checkpoint}, scenario={weather_variant}, DAP={target_dap}",
                        flush=True,
                    )
                    reference_env = base03222.base.make_env(
                        runtime_config,
                        env_config,
                        STATION,
                        scenario_key,
                        SEED,
                        f"SYA_2005_{weather_variant}_184_ckpt{checkpoint}_ref_dap{target_dap}",
                        evaluation=True,
                    )
                    prefix_actions: list[int] = []
                    try:
                        obs, info = reference_env.reset()
                        for step_index in range(max_steps):
                            state = current_state(reference_env, obs, step_index)
                            mask = np.asarray(get_action_masks(reference_env), dtype=bool).reshape(-1)
                            probs, logits = policy_distribution(model, obs, mask)
                            selected_action_raw, _ = model.predict(obs, action_masks=mask, deterministic=True)
                            selected_action = one_action(selected_action_raw)
                            if int(state["dap"]) == target_dap:
                                value_before = state_value(model, obs)
                                selected_grid = grid_action(reference_env, selected_action)
                                legal_count = int(mask.sum())
                                legal_entropy = entropy(probs[mask])
                                normalized_entropy = (
                                    float(legal_entropy / np.log(legal_count))
                                    if legal_count > 1 and np.isfinite(legal_entropy)
                                    else np.nan
                                )
                                target_state = dict(state)
                                target_obs = np.asarray(obs, dtype=np.float32).reshape(-1).copy()
                                target_mask = mask.copy()
                                target_probs = probs.copy()
                                target_logits = logits.copy()
                                target_selected_action = selected_action
                                target_selected_grid = selected_grid
                                target_value_before = value_before
                                target_entropy = legal_entropy
                                target_entropy_norm = normalized_entropy
                                break
                            if not bool(mask[selected_action]):
                                raise RuntimeError(
                                    f"Invalid reference action {selected_action} at DAP {state['dap']}"
                                )
                            prefix_actions.append(selected_action)
                            obs, reward, terminated, truncated, info = reference_env.step(selected_action)
                            if bool(terminated or truncated):
                                raise RuntimeError(f"Episode ended before target DAP {target_dap}")
                        else:
                            raise RuntimeError(f"Target DAP {target_dap} not reached")
                    finally:
                        reference_env.close()

                    candidate_actions = np.flatnonzero(target_mask).astype(int).tolist()
                    target_state_rows.append(
                        {
                            "checkpoint_step": checkpoint,
                            "scenario_key": scenario_key,
                            "weather_variant": weather_variant,
                            "target_dap": target_dap,
                            "prefix_length": len(prefix_actions),
                            "rain": target_state["rain"],
                            "swfac": target_state["swfac"],
                            "nstres": target_state["nstres"],
                            "grnwt": target_state["grnwt"],
                            "topwt": target_state["topwt"],
                            "xlai": target_state["xlai"],
                            "cumulative_irrigation": target_state["cumulative_irrigation"],
                            "cumulative_nitrogen": target_state["cumulative_nitrogen"],
                            "mask": mask_text(target_mask),
                            "legal_action_count": int(target_mask.sum()),
                            "noop_legal": bool(target_mask[KEY_ACTIONS["noop"]]),
                            "i45_n0_legal": bool(target_mask[KEY_ACTIONS["i45_n0"]]),
                            "actor_selected_action": target_selected_action,
                            "actor_selected_irrigation_mm": target_selected_grid["irrigation_mm"],
                            "actor_selected_nitrogen_kg_ha": target_selected_grid["nitrogen_kg_ha"],
                            "actor_selected_probability": float(target_probs[target_selected_action]),
                            "actor_selected_rank": probability_rank(target_probs, target_mask, target_selected_action),
                            "critic_value_before": target_value_before,
                            "legal_entropy": target_entropy,
                            "legal_entropy_normalized": target_entropy_norm,
                            "prob_noop": float(target_probs[KEY_ACTIONS["noop"]]),
                            "prob_i45_n0": float(target_probs[KEY_ACTIONS["i45_n0"]]),
                            "i45_minus_noop_prob": float(target_probs[KEY_ACTIONS["i45_n0"]] - target_probs[KEY_ACTIONS["noop"]]),
                        }
                    )

                    # --------------------------------------------------
                    # Step 2. For every legal candidate action, restart
                    # DSSAT, replay the same prefix, apply the candidate
                    # once, and compute reward + gamma * V(next).
                    # --------------------------------------------------
                    for candidate_action in candidate_actions:
                        candidate_grid = None
                        branch_env = base03222.base.make_env(
                            runtime_config,
                            env_config,
                            STATION,
                            scenario_key,
                            SEED,
                            f"SYA_2005_{weather_variant}_184_ckpt{checkpoint}_dap{target_dap}_a{candidate_action}",
                            evaluation=True,
                        )
                        try:
                            branch_obs, branch_info = branch_env.reset()
                            for prefix_step, prefix_action in enumerate(prefix_actions):
                                replay_mask = np.asarray(get_action_masks(branch_env), dtype=bool).reshape(-1)
                                if not bool(replay_mask[prefix_action]):
                                    raise RuntimeError(
                                        f"Prefix action {prefix_action} invalid during replay step {prefix_step}"
                                    )
                                branch_obs, reward, terminated, truncated, branch_info = branch_env.step(prefix_action)
                                if bool(terminated or truncated):
                                    raise RuntimeError("Branch ended during prefix replay")

                            branch_state = current_state(branch_env, branch_obs, len(prefix_actions))
                            replay_error = max(
                                abs(number(branch_state[field]) - number(target_state[field]))
                                for field in [
                                    "rain",
                                    "swfac",
                                    "nstres",
                                    "grnwt",
                                    "cumulative_irrigation",
                                    "cumulative_nitrogen",
                                ]
                            )
                            if int(branch_state["dap"]) != target_dap or replay_error > 1e-6:
                                raise RuntimeError(
                                    f"Prefix replay mismatch at DAP {target_dap}: dap={branch_state['dap']}, error={replay_error}"
                                )

                            branch_mask = np.asarray(get_action_masks(branch_env), dtype=bool).reshape(-1)
                            if not bool(branch_mask[candidate_action]):
                                raise RuntimeError(
                                    f"Candidate action {candidate_action} invalid at target; mask={mask_text(branch_mask)}"
                                )

                            candidate_grid = grid_action(branch_env, candidate_action)
                            obs_after, reward, terminated, truncated, info = branch_env.step(candidate_action)
                            state_after = current_state(branch_env, obs_after, target_dap + 1)
                            action_info = dict(getattr(branch_env, "last_action_info", {}) or {})
                            value_after = 0.0 if bool(terminated or truncated) else state_value(model, obs_after)
                            td_scaled = float(reward) + gamma * value_after
                            reward_unscaled = number(action_info.get("reward_unscaled"), np.nan)
                            yield_component = number(action_info.get("yield_component"), 0.0)
                            resource_cost = number(action_info.get("resource_cost"), 0.0)
                            stress_bonus = number(action_info.get("stress_relief_bonus"), 0.0)
                            grnwt_increment = number(
                                action_info.get("grnwt_increment"),
                                state_after["grnwt"] - branch_state["grnwt"],
                            )
                            td_rows.append(
                                {
                                    "checkpoint_step": checkpoint,
                                    "scenario_key": scenario_key,
                                    "weather_variant": weather_variant,
                                    "target_dap": target_dap,
                                    "candidate_action_index": candidate_action,
                                    "candidate_irrigation_mm": candidate_grid["irrigation_mm"],
                                    "candidate_nitrogen_kg_ha": candidate_grid["nitrogen_kg_ha"],
                                    "candidate_is_actor_selected": bool(candidate_action == target_selected_action),
                                    "actor_selected_action": target_selected_action,
                                    "actor_selected_irrigation_mm": target_selected_grid["irrigation_mm"],
                                    "actor_selected_nitrogen_kg_ha": target_selected_grid["nitrogen_kg_ha"],
                                    "actor_probability_for_candidate": float(target_probs[candidate_action]),
                                    "actor_logit_for_candidate": float(target_logits[candidate_action]),
                                    "actor_rank_for_candidate": probability_rank(target_probs, target_mask, candidate_action),
                                    "critic_value_before": target_value_before,
                                    "reward_scaled": float(reward),
                                    "reward_unscaled": reward_unscaled,
                                    "yield_component_unscaled": yield_component,
                                    "resource_cost_unscaled": resource_cost,
                                    "stress_relief_bonus_unscaled": stress_bonus,
                                    "grnwt_increment": grnwt_increment,
                                    "critic_value_after": value_after,
                                    "gamma": gamma,
                                    "td_one_step_scaled": td_scaled,
                                    "swfac_before": branch_state["swfac"],
                                    "swfac_after": state_after["swfac"],
                                    "nstres_before": branch_state["nstres"],
                                    "nstres_after": state_after["nstres"],
                                    "grnwt_before": branch_state["grnwt"],
                                    "grnwt_after": state_after["grnwt"],
                                    "cumulative_irrigation_before": branch_state["cumulative_irrigation"],
                                    "cumulative_irrigation_after": state_after["cumulative_irrigation"],
                                    "prefix_replay_max_abs_error": replay_error,
                                    "terminated": bool(terminated),
                                    "truncated": bool(truncated),
                                }
                            )
                        finally:
                            branch_env.close()

                    progress_rows.append(
                        {
                            "checkpoint_step": checkpoint,
                            "weather_variant": weather_variant,
                            "target_dap": target_dap,
                            "legal_candidate_count": len(candidate_actions),
                            "status": "completed",
                        }
                    )
                    pd.DataFrame(target_state_rows).to_csv(
                        OUT / "debug" / "184_target_state_values_partial.csv",
                        index=False,
                        encoding="utf-8-sig",
                    )
                    pd.DataFrame(td_rows).to_csv(
                        OUT / "debug" / "184_action_one_step_td_partial.csv",
                        index=False,
                        encoding="utf-8-sig",
                    )
                    pd.DataFrame(progress_rows).to_csv(
                        OUT / "debug" / "184_progress.csv",
                        index=False,
                        encoding="utf-8-sig",
                    )
    except Exception:
        (OUT / "184_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        task178.restore_runtime_178(saved_runtime)
        for key, value in saved_task178_globals.items():
            setattr(task178, key, value)

    # ------------------------------------------------------------------
    # Step 3. Build actor/critic alignment summaries.
    # ------------------------------------------------------------------
    target_frame = pd.DataFrame(target_state_rows)
    td_frame = pd.DataFrame(td_rows)

    noop = td_frame[td_frame["candidate_action_index"].eq(KEY_ACTIONS["noop"])][
        ["checkpoint_step", "scenario_key", "weather_variant", "target_dap", "td_one_step_scaled", "reward_unscaled", "critic_value_after"]
    ].rename(
        columns={
            "td_one_step_scaled": "noop_td_one_step_scaled",
            "reward_unscaled": "noop_reward_unscaled",
            "critic_value_after": "noop_critic_value_after",
        }
    )
    td_frame = td_frame.merge(
        noop,
        on=["checkpoint_step", "scenario_key", "weather_variant", "target_dap"],
        how="left",
        validate="many_to_one",
    )
    td_frame["td_delta_vs_noop_scaled"] = td_frame["td_one_step_scaled"] - td_frame["noop_td_one_step_scaled"]
    td_frame["reward_unscaled_delta_vs_noop"] = td_frame["reward_unscaled"] - td_frame["noop_reward_unscaled"]
    td_frame["critic_value_after_delta_vs_noop"] = td_frame["critic_value_after"] - td_frame["noop_critic_value_after"]

    alignment_rows: list[dict[str, Any]] = []
    for (checkpoint, scenario_key, weather_variant, target_dap), group in td_frame.groupby(
        ["checkpoint_step", "scenario_key", "weather_variant", "target_dap"],
        sort=True,
    ):
        by_td = group.sort_values(["td_one_step_scaled", "candidate_action_index"], ascending=[False, True])
        by_actor = group.sort_values(["actor_probability_for_candidate", "candidate_action_index"], ascending=[False, True])
        td_top = by_td.iloc[0]
        actor_top = by_actor.iloc[0]
        selected = group[group["candidate_is_actor_selected"].astype(bool)].iloc[0]
        i45 = group[group["candidate_action_index"].eq(KEY_ACTIONS["i45_n0"])].iloc[0]
        noop_row = group[group["candidate_action_index"].eq(KEY_ACTIONS["noop"])].iloc[0]
        alignment_rows.append(
            {
                "checkpoint_step": int(checkpoint),
                "scenario_key": int(scenario_key),
                "weather_variant": str(weather_variant),
                "target_dap": int(target_dap),
                "actor_selected_action": int(selected["candidate_action_index"]),
                "actor_selected_pair": f"I{float(selected['candidate_irrigation_mm']):g}/N{float(selected['candidate_nitrogen_kg_ha']):g}",
                "actor_top_action": int(actor_top["candidate_action_index"]),
                "actor_top_pair": f"I{float(actor_top['candidate_irrigation_mm']):g}/N{float(actor_top['candidate_nitrogen_kg_ha']):g}",
                "td_top_action": int(td_top["candidate_action_index"]),
                "td_top_pair": f"I{float(td_top['candidate_irrigation_mm']):g}/N{float(td_top['candidate_nitrogen_kg_ha']):g}",
                "actor_selected_is_td_top": bool(int(selected["candidate_action_index"]) == int(td_top["candidate_action_index"])),
                "td_top_is_noop": bool(int(td_top["candidate_action_index"]) == KEY_ACTIONS["noop"]),
                "td_top_is_i45": bool(int(td_top["candidate_action_index"]) == KEY_ACTIONS["i45_n0"]),
                "selected_td_one_step_scaled": float(selected["td_one_step_scaled"]),
                "td_top_one_step_scaled": float(td_top["td_one_step_scaled"]),
                "noop_td_one_step_scaled": float(noop_row["td_one_step_scaled"]),
                "i45_td_one_step_scaled": float(i45["td_one_step_scaled"]),
                "i45_td_delta_vs_noop_scaled": float(i45["td_delta_vs_noop_scaled"]),
                "i45_reward_unscaled_delta_vs_noop": float(i45["reward_unscaled_delta_vs_noop"]),
                "i45_critic_value_after_delta_vs_noop": float(i45["critic_value_after_delta_vs_noop"]),
                "actor_i45_minus_noop_prob": float(i45["actor_probability_for_candidate"] - noop_row["actor_probability_for_candidate"]),
            }
        )
    alignment_frame = pd.DataFrame(alignment_rows)

    group_summary = (
        alignment_frame.groupby(["checkpoint_step", "target_dap"], as_index=False)
        .agg(
            scenario_count=("weather_variant", "count"),
            actor_selected_is_td_top_count=("actor_selected_is_td_top", lambda s: int(pd.Series(s).astype(bool).sum())),
            td_top_noop_count=("td_top_is_noop", lambda s: int(pd.Series(s).astype(bool).sum())),
            td_top_i45_count=("td_top_is_i45", lambda s: int(pd.Series(s).astype(bool).sum())),
            mean_i45_td_delta_vs_noop_scaled=("i45_td_delta_vs_noop_scaled", "mean"),
            min_i45_td_delta_vs_noop_scaled=("i45_td_delta_vs_noop_scaled", "min"),
            max_i45_td_delta_vs_noop_scaled=("i45_td_delta_vs_noop_scaled", "max"),
            mean_i45_reward_unscaled_delta_vs_noop=("i45_reward_unscaled_delta_vs_noop", "mean"),
            mean_i45_critic_value_after_delta_vs_noop=("i45_critic_value_after_delta_vs_noop", "mean"),
            mean_actor_i45_minus_noop_prob=("actor_i45_minus_noop_prob", "mean"),
        )
    )

    # ------------------------------------------------------------------
    # Step 4. Save outputs and classify the evidence.
    # ------------------------------------------------------------------
    target_frame.to_csv(OUT / "evaluation" / "184_target_state_values.csv", index=False, encoding="utf-8-sig")
    td_frame.to_csv(OUT / "evaluation" / "184_action_one_step_td.csv", index=False, encoding="utf-8-sig")
    alignment_frame.to_csv(OUT / "evaluation" / "184_actor_critic_alignment.csv", index=False, encoding="utf-8-sig")
    group_summary.to_csv(OUT / "evaluation" / "184_actor_critic_group_summary.csv", index=False, encoding="utf-8-sig")

    ckpt2000 = alignment_frame[alignment_frame["checkpoint_step"].eq(2000)].copy()
    ckpt1000 = alignment_frame[alignment_frame["checkpoint_step"].eq(1000)].copy()
    ckpt2000_actor_td_aligned = int(ckpt2000["actor_selected_is_td_top"].astype(bool).sum())
    ckpt2000_td_top_i45 = int(ckpt2000["td_top_is_i45"].astype(bool).sum())
    ckpt2000_td_top_noop = int(ckpt2000["td_top_is_noop"].astype(bool).sum())
    ckpt2000_i45_td_positive = int((ckpt2000["i45_td_delta_vs_noop_scaled"] > 1e-12).sum())
    ckpt2000_i45_td_negative = int((ckpt2000["i45_td_delta_vs_noop_scaled"] < -1e-12).sum())
    ckpt1000_actor_td_aligned = int(ckpt1000["actor_selected_is_td_top"].astype(bool).sum())

    if ckpt2000_td_top_i45 == len(ckpt2000):
        primary_label = "critic_one_step_td_supports_i45_actor_choice"
    elif ckpt2000_td_top_noop == len(ckpt2000):
        primary_label = "critic_one_step_td_prefers_noop_against_i45_actor_choice"
    elif ckpt2000_actor_td_aligned == 0:
        primary_label = "critic_one_step_td_disagrees_with_actor_choice"
    else:
        primary_label = "critic_one_step_td_mixed_alignment"

    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "status": "completed",
        "mode": "read_only_critic_one_step_td_audit",
        "training_executed": False,
        "source_task": "177_sya_dense_yield_credit_smoke2k",
        "station": STATION,
        "checkpoint_steps": CHECKPOINTS,
        "target_daps": TARGET_DAPS,
        "scenario_count": len(SCENARIOS),
        "target_state_rows": int(len(target_frame)),
        "candidate_td_rows": int(len(td_frame)),
        "physical_gate_passed": bool(gate.get("physical_response_passed")),
        "rootcause_summary": {
            "primary_label": primary_label,
            "ckpt2000_actor_td_aligned_rows": ckpt2000_actor_td_aligned,
            "ckpt2000_td_top_i45_rows": ckpt2000_td_top_i45,
            "ckpt2000_td_top_noop_rows": ckpt2000_td_top_noop,
            "ckpt2000_i45_td_delta_positive_rows": ckpt2000_i45_td_positive,
            "ckpt2000_i45_td_delta_negative_rows": ckpt2000_i45_td_negative,
            "ckpt1000_actor_td_aligned_rows": ckpt1000_actor_td_aligned,
            "interpretation": (
                "TD(a)=reward_scaled+gamma*V(next) is a local critic proxy. "
                "Use task 182 for full-season event payoff."
            ),
        },
        "outputs": {
            "target_state_values": relative(OUT / "evaluation" / "184_target_state_values.csv"),
            "action_one_step_td": relative(OUT / "evaluation" / "184_action_one_step_td.csv"),
            "actor_critic_alignment": relative(OUT / "evaluation" / "184_actor_critic_alignment.csv"),
            "group_summary": relative(OUT / "evaluation" / "184_actor_critic_group_summary.csv"),
            "progress": relative(OUT / "debug" / "184_progress.csv"),
        },
    }
    (OUT / "184_rootcause_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    DOC.write_text(
        "\n".join(
            [
                "# Task 184：SYA PPO critic/one-step TD 动作价值审计记录",
                "",
                "> 自动简要记录；详细解释运行后按模板补充。",
                "",
                f"- 状态：completed；target states：{len(target_frame)}；candidate TD rows：{len(td_frame)}。",
                f"- checkpoint：{CHECKPOINTS}；目标 DAP：{TARGET_DAPS}；情景数：{len(SCENARIOS)}。",
                f"- primary label：`{primary_label}`。",
                f"- ckpt2000 actor/TD aligned rows：`{ckpt2000_actor_td_aligned}`。",
                f"- ckpt2000 TD top I45 rows：`{ckpt2000_td_top_i45}`。",
                f"- ckpt2000 TD top no-op rows：`{ckpt2000_td_top_noop}`。",
                "",
                "## 输出",
                "",
                f"- `{relative(OUT / '184_rootcause_summary.json')}`",
                f"- `{relative(OUT / 'evaluation' / '184_target_state_values.csv')}`",
                f"- `{relative(OUT / 'evaluation' / '184_action_one_step_td.csv')}`",
                f"- `{relative(OUT / 'evaluation' / '184_actor_critic_alignment.csv')}`",
                f"- `{relative(OUT / 'evaluation' / '184_actor_critic_group_summary.csv')}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[Step 5/5] Completed. primary_label={primary_label}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    print(json.dumps(run(config_path.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
