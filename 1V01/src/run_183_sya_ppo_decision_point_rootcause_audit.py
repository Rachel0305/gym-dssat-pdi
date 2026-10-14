"""Task 183: decision-point root-cause audit for fixed late PPO irrigation.

This is a read-only diagnostic.  It replays the task-177 PPO checkpoints to
DAP 97 and DAP 104, records the pre-action state, action mask, and masked
policy probabilities, and links checkpoint-2000 decisions to task-182 ablation
effects.
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


TASK_ID = "183"
TASK_NAME = "sya_ppo_decision_point_rootcause_audit"
DEFAULT_CONFIG = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / "2026-08-20_sya_ppo_decision_point_rootcause_audit.md"
MODEL_ROOT = ROOT / "benchmark_results" / "177_sya_dense_yield_credit_smoke2k" / "models" / "SYA"
TASK182_EFFECTS = (
    ROOT
    / "benchmark_results"
    / "182_sya_fullseason_ppo_action_ablation"
    / "evaluation"
    / "182_ablation_effects_vs_full_ppo.csv"
)

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
KEY_ACTIONS = {
    "noop": 0,
    "i15_n0": 4,
    "i30_n0": 8,
    "i45_n0": 12,
}


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


def action_signature(frame: pd.DataFrame) -> str:
    out: list[str] = []
    if frame.empty:
        return ""
    for row in frame.itertuples(index=False):
        irrigation = float(row.irrigation_mm)
        nitrogen = float(row.nitrogen_kg_ha)
        if irrigation > 1e-9 or nitrogen > 1e-9:
            out.append(f"DAP{int(row.dap)} I{irrigation:g}/N{nitrogen:g}")
    return "; ".join(out)


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


def probability_rank(probs: np.ndarray, mask: np.ndarray, action_index: int) -> float:
    if action_index >= len(probs) or not bool(mask[action_index]) or not np.isfinite(probs[action_index]):
        return np.nan
    legal_probs = probs[np.asarray(mask, dtype=bool)]
    return float(1 + np.sum(legal_probs > probs[action_index] + 1e-12))


def rank_signature(prob_rows: pd.DataFrame) -> str:
    legal = prob_rows[prob_rows["legal"].astype(bool)].copy()
    if legal.empty:
        return ""
    ordered = legal.sort_values(["probability", "action_index"], ascending=[False, True])
    return ">".join(
        f"{int(row.action_index)}(I{float(row.irrigation_mm):g}/N{float(row.nitrogen_kg_ha):g})"
        for row in ordered.itertuples(index=False)
    )


def load_task182_reference() -> dict[tuple[str, int], dict[str, float]]:
    if not TASK182_EFFECTS.exists():
        return {}
    effects = pd.read_csv(TASK182_EFFECTS)
    mapping = {
        97: "drop_dap97",
        104: "drop_dap104",
    }
    out: dict[tuple[str, int], dict[str, float]] = {}
    for dap, treatment in mapping.items():
        subset = effects[effects["treatment"].eq(treatment)].copy()
        for row in subset.itertuples(index=False):
            out[(str(row.weather_variant), int(dap))] = {
                "task182_treatment": treatment,
                "task182_delta_final_grnwt_when_event_removed": float(row.delta_final_grnwt_vs_full),
                "task182_delta_profit_when_event_removed": float(row.delta_profit_simple_vs_full),
                "task182_delta_reward_unscaled_when_event_removed": float(row.delta_reward_unscaled_sum_vs_full),
            }
    return out


def run(config_path: Path) -> dict[str, Any]:
    # ------------------------------------------------------------------
    # Step 0. Verify the read-only experiment contract.
    # ------------------------------------------------------------------
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg.get("training_executed") is not False:
        raise ValueError("Task 183 is read-only and must not train PPO.")
    if [int(x) for x in cfg["checkpoint_steps"]] != CHECKPOINTS:
        raise ValueError("Task 183 expects checkpoints [1000, 2000].")
    if [int(x) for x in cfg["target_daps"]] != TARGET_DAPS:
        raise ValueError("Task 183 expects target DAPs [97, 104].")
    if OUT.exists():
        raise FileExistsError(f"Refuse to overwrite existing output: {relative(OUT)}")
    for checkpoint in CHECKPOINTS:
        if not model_path(checkpoint).exists():
            raise FileNotFoundError(model_path(checkpoint))

    task168 = task178.task177.task174.task173.task172.task168
    gate = task168.verify_gate(cfg)
    manifest = task168.read_manifest(cfg)
    split = task168.load_split(manifest)
    task182_reference = load_task182_reference()

    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "configs").mkdir()
    (OUT / "debug").mkdir()
    (OUT / "evaluation").mkdir()
    (OUT / "logs").mkdir()
    (OUT / "configs" / config_path.name).write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (OUT / "configs" / "183_scenario_split.csv").write_text(split.to_csv(index=False), encoding="utf-8-sig")
    (OUT / "configs" / "167_physical_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[Step 0/4] Contract checked: read-only policy audit for DAP97/104.", flush=True)

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

    decision_rows: list[dict[str, Any]] = []
    action_prob_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []

    try:
        engine = task168.engine
        base03222 = task168.base03222
        runtime_config = engine.load_config()
        env_config = base03222.direct_ppo.build_env_config(runtime_config, split)
        max_steps = int(runtime_config["runtime"]["max_steps"])

        # --------------------------------------------------------------
        # Step 1. Replay each checkpoint/scenario to the two target DAPs.
        # --------------------------------------------------------------
        for checkpoint in CHECKPOINTS:
            print(f"[Step 1/4] Load PPO checkpoint {checkpoint}", flush=True)
            model = MaskablePPO.load(str(model_path(checkpoint)), device="cpu")
            for scenario_key, weather_variant in SCENARIOS.items():
                print(
                    f"[Step 2/4] Replay checkpoint {checkpoint}, scenario={weather_variant}",
                    flush=True,
                )
                env = base03222.base.make_env(
                    runtime_config,
                    env_config,
                    STATION,
                    scenario_key,
                    SEED,
                    f"SYA_2005_{weather_variant}_183_ckpt{checkpoint}",
                    evaluation=True,
                )
                captured: set[int] = set()
                local_trajectory: list[dict[str, Any]] = []
                try:
                    obs, info = env.reset()
                    for step_index in range(max_steps):
                        state = current_state(env, obs, step_index)
                        mask = np.asarray(get_action_masks(env), dtype=bool).reshape(-1)
                        probs, logits = policy_distribution(model, obs, mask)
                        action_raw, _ = model.predict(obs, action_masks=mask, deterministic=True)
                        selected_action = one_action(action_raw)
                        selected_grid = grid_action(env, selected_action)
                        if not bool(mask[selected_action]):
                            raise RuntimeError(
                                f"Invalid selected action {selected_action} at ckpt {checkpoint}, "
                                f"{weather_variant}, DAP {state['dap']}"
                            )

                        local_trajectory.append(
                            {
                                "checkpoint_step": checkpoint,
                                "scenario_key": scenario_key,
                                "weather_variant": weather_variant,
                                "step_index": step_index,
                                "dap": int(state["dap"]),
                                "selected_action": selected_action,
                                "irrigation_mm": selected_grid["irrigation_mm"],
                                "nitrogen_kg_ha": selected_grid["nitrogen_kg_ha"],
                                "mask": mask_text(mask),
                                "legal_action_count": int(mask.sum()),
                            }
                        )

                        if int(state["dap"]) in TARGET_DAPS and int(state["dap"]) not in captured:
                            captured.add(int(state["dap"]))
                            obs_flat = np.asarray(obs, dtype=np.float32).reshape(-1)
                            legal_probs = probs[mask] if len(probs) == len(mask) else np.array([])
                            legal_entropy = entropy(legal_probs)
                            legal_count = int(mask.sum())
                            normalized_entropy = (
                                float(legal_entropy / np.log(legal_count))
                                if legal_count > 1 and np.isfinite(legal_entropy)
                                else np.nan
                            )
                            top_action = int(np.nanargmax(probs)) if np.isfinite(probs).any() else -1
                            top_grid = grid_action(env, top_action) if top_action >= 0 else {"irrigation_mm": np.nan, "nitrogen_kg_ha": np.nan}
                            target_ref = task182_reference.get((weather_variant, int(state["dap"])), {})

                            decision_row = {
                                "checkpoint_step": checkpoint,
                                "scenario_key": scenario_key,
                                "weather_variant": weather_variant,
                                "target_dap": int(state["dap"]),
                                "step_index": step_index,
                                "obs_dim": int(obs_flat.size),
                                "rain": state["rain"],
                                "swfac": state["swfac"],
                                "nstres": state["nstres"],
                                "grnwt": state["grnwt"],
                                "topwt": state["topwt"],
                                "xlai": state["xlai"],
                                "cumulative_irrigation": state["cumulative_irrigation"],
                                "cumulative_nitrogen": state["cumulative_nitrogen"],
                                "mask": mask_text(mask),
                                "legal_action_count": legal_count,
                                "noop_legal": bool(mask[KEY_ACTIONS["noop"]]),
                                "i15_n0_legal": bool(mask[KEY_ACTIONS["i15_n0"]]),
                                "i30_n0_legal": bool(mask[KEY_ACTIONS["i30_n0"]]),
                                "i45_n0_legal": bool(mask[KEY_ACTIONS["i45_n0"]]),
                                "selected_action": selected_action,
                                "selected_irrigation_mm": selected_grid["irrigation_mm"],
                                "selected_nitrogen_kg_ha": selected_grid["nitrogen_kg_ha"],
                                "selected_action_prob": float(probs[selected_action]) if np.isfinite(probs[selected_action]) else np.nan,
                                "selected_action_rank": probability_rank(probs, mask, selected_action),
                                "top_action": top_action,
                                "top_irrigation_mm": top_grid["irrigation_mm"],
                                "top_nitrogen_kg_ha": top_grid["nitrogen_kg_ha"],
                                "top_probability": float(np.nanmax(probs)) if np.isfinite(probs).any() else np.nan,
                                "legal_policy_entropy": legal_entropy,
                                "legal_policy_entropy_normalized": normalized_entropy,
                                "prob_action0_noop": float(probs[KEY_ACTIONS["noop"]]) if np.isfinite(probs[KEY_ACTIONS["noop"]]) else np.nan,
                                "prob_action4_i15_n0": float(probs[KEY_ACTIONS["i15_n0"]]) if np.isfinite(probs[KEY_ACTIONS["i15_n0"]]) else np.nan,
                                "prob_action8_i30_n0": float(probs[KEY_ACTIONS["i30_n0"]]) if np.isfinite(probs[KEY_ACTIONS["i30_n0"]]) else np.nan,
                                "prob_action12_i45_n0": float(probs[KEY_ACTIONS["i45_n0"]]) if np.isfinite(probs[KEY_ACTIONS["i45_n0"]]) else np.nan,
                                "rank_action0_noop": probability_rank(probs, mask, KEY_ACTIONS["noop"]),
                                "rank_action4_i15_n0": probability_rank(probs, mask, KEY_ACTIONS["i15_n0"]),
                                "rank_action8_i30_n0": probability_rank(probs, mask, KEY_ACTIONS["i30_n0"]),
                                "rank_action12_i45_n0": probability_rank(probs, mask, KEY_ACTIONS["i45_n0"]),
                                **target_ref,
                            }
                            decision_rows.append(decision_row)

                            prob_frame_rows: list[dict[str, Any]] = []
                            for action_index in range(len(mask)):
                                action_grid = grid_action(env, action_index)
                                prob_row = {
                                    "checkpoint_step": checkpoint,
                                    "scenario_key": scenario_key,
                                    "weather_variant": weather_variant,
                                    "target_dap": int(state["dap"]),
                                    "action_index": action_index,
                                    "legal": bool(mask[action_index]),
                                    "irrigation_mm": action_grid["irrigation_mm"],
                                    "nitrogen_kg_ha": action_grid["nitrogen_kg_ha"],
                                    "probability": float(probs[action_index]) if np.isfinite(probs[action_index]) else np.nan,
                                    "logit": float(logits[action_index]) if np.isfinite(logits[action_index]) else np.nan,
                                    "rank_among_legal": probability_rank(probs, mask, action_index),
                                    "selected": bool(action_index == selected_action),
                                    "top_action": bool(action_index == top_action),
                                }
                                action_prob_rows.append(prob_row)
                                prob_frame_rows.append(prob_row)
                            decision_rows[-1]["legal_rank_signature"] = rank_signature(pd.DataFrame(prob_frame_rows))

                            for obs_index, obs_value in enumerate(obs_flat.tolist()):
                                observation_rows.append(
                                    {
                                        "checkpoint_step": checkpoint,
                                        "scenario_key": scenario_key,
                                        "weather_variant": weather_variant,
                                        "target_dap": int(state["dap"]),
                                        "obs_index": obs_index,
                                        "obs_value": float(obs_value),
                                    }
                                )

                        obs, reward, terminated, truncated, info = env.step(selected_action)
                        if bool(terminated or truncated):
                            break
                finally:
                    env.close()

                trajectory_rows.extend(local_trajectory)
                if captured != set(TARGET_DAPS):
                    raise RuntimeError(
                        f"Missing target DAPs for checkpoint {checkpoint}, {weather_variant}: "
                        f"captured={sorted(captured)}"
                    )

                pd.DataFrame(decision_rows).to_csv(
                    OUT / "debug" / "183_decision_states_partial.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                pd.DataFrame(action_prob_rows).to_csv(
                    OUT / "debug" / "183_action_probabilities_partial.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
    except Exception:
        (OUT / "183_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        task178.restore_runtime_178(saved_runtime)
        for key, value in saved_task178_globals.items():
            setattr(task178, key, value)

    # ------------------------------------------------------------------
    # Step 3. Summarize mask forcing, concentration, weather sensitivity,
    # and checkpoint changes.
    # ------------------------------------------------------------------
    decision_frame = pd.DataFrame(decision_rows)
    action_prob_frame = pd.DataFrame(action_prob_rows)
    observation_frame = pd.DataFrame(observation_rows)
    trajectory_frame = pd.DataFrame(trajectory_rows)

    policy_summary_rows: list[dict[str, Any]] = []
    for (checkpoint, target_dap), group in decision_frame.groupby(["checkpoint_step", "target_dap"], sort=True):
        selected_actions = sorted(set(group["selected_action"].astype(int)))
        selected_pairs = sorted(
            set(
                f"I{float(row.selected_irrigation_mm):g}/N{float(row.selected_nitrogen_kg_ha):g}"
                for row in group.itertuples(index=False)
            )
        )
        rank_signatures = sorted(set(group["legal_rank_signature"].astype(str)))
        rain_range = float(group["rain"].max() - group["rain"].min())
        swfac_range = float(group["swfac"].max() - group["swfac"].min())
        nstres_range = float(group["nstres"].max() - group["nstres"].min())
        grnwt_range = float(group["grnwt"].max() - group["grnwt"].min())
        prob_i45_range = float(group["prob_action12_i45_n0"].max() - group["prob_action12_i45_n0"].min())
        prob_noop_range = float(group["prob_action0_noop"].max() - group["prob_action0_noop"].min())
        policy_summary_rows.append(
            {
                "checkpoint_step": int(checkpoint),
                "target_dap": int(target_dap),
                "scenario_count": int(len(group)),
                "selected_action_unique_count": int(len(selected_actions)),
                "selected_actions": ",".join(str(x) for x in selected_actions),
                "selected_pairs": ";".join(selected_pairs),
                "legal_rank_signature_unique_count": int(len(rank_signatures)),
                "rank_signatures": " | ".join(rank_signatures),
                "all_noop_legal": bool(group["noop_legal"].all()),
                "all_i45_n0_legal": bool(group["i45_n0_legal"].all()),
                "selected_i45_rows": int(group["selected_action"].eq(KEY_ACTIONS["i45_n0"]).sum()),
                "selected_i45_and_noop_legal_rows": int(
                    (group["selected_action"].eq(KEY_ACTIONS["i45_n0"]) & group["noop_legal"].astype(bool)).sum()
                ),
                "mean_top_probability": float(group["top_probability"].mean()),
                "min_top_probability": float(group["top_probability"].min()),
                "max_top_probability": float(group["top_probability"].max()),
                "mean_legal_entropy_normalized": float(group["legal_policy_entropy_normalized"].mean()),
                "rain_range": rain_range,
                "swfac_range": swfac_range,
                "nstres_range": nstres_range,
                "grnwt_range": grnwt_range,
                "prob_i45_range": prob_i45_range,
                "prob_noop_range": prob_noop_range,
                "weather_or_state_varied": bool(
                    max(abs(rain_range), abs(swfac_range), abs(nstres_range), abs(grnwt_range)) > 1e-9
                ),
                "action_invariant_across_weather": bool(len(selected_actions) == 1 and len(rank_signatures) == 1),
            }
        )
    policy_summary_frame = pd.DataFrame(policy_summary_rows)

    comparison_rows: list[dict[str, Any]] = []
    for (scenario_key, weather_variant, target_dap), group in decision_frame.groupby(
        ["scenario_key", "weather_variant", "target_dap"],
        sort=True,
    ):
        if set(group["checkpoint_step"].astype(int)) != set(CHECKPOINTS):
            continue
        ckpt1000 = group[group["checkpoint_step"].eq(1000)].iloc[0]
        ckpt2000 = group[group["checkpoint_step"].eq(2000)].iloc[0]
        comparison_rows.append(
            {
                "scenario_key": int(scenario_key),
                "weather_variant": weather_variant,
                "target_dap": int(target_dap),
                "ckpt1000_selected_action": int(ckpt1000["selected_action"]),
                "ckpt1000_selected_pair": f"I{float(ckpt1000['selected_irrigation_mm']):g}/N{float(ckpt1000['selected_nitrogen_kg_ha']):g}",
                "ckpt1000_top_probability": float(ckpt1000["top_probability"]),
                "ckpt1000_i45_probability": float(ckpt1000["prob_action12_i45_n0"]),
                "ckpt1000_noop_probability": float(ckpt1000["prob_action0_noop"]),
                "ckpt2000_selected_action": int(ckpt2000["selected_action"]),
                "ckpt2000_selected_pair": f"I{float(ckpt2000['selected_irrigation_mm']):g}/N{float(ckpt2000['selected_nitrogen_kg_ha']):g}",
                "ckpt2000_top_probability": float(ckpt2000["top_probability"]),
                "ckpt2000_i45_probability": float(ckpt2000["prob_action12_i45_n0"]),
                "ckpt2000_noop_probability": float(ckpt2000["prob_action0_noop"]),
                "selected_action_changed": bool(int(ckpt1000["selected_action"]) != int(ckpt2000["selected_action"])),
                "i45_probability_delta_2000_minus_1000": float(
                    ckpt2000["prob_action12_i45_n0"] - ckpt1000["prob_action12_i45_n0"]
                ),
                "noop_probability_delta_2000_minus_1000": float(
                    ckpt2000["prob_action0_noop"] - ckpt1000["prob_action0_noop"]
                ),
            }
        )
    checkpoint_comparison_frame = pd.DataFrame(comparison_rows)

    l1_rows: list[dict[str, Any]] = []
    for (checkpoint, target_dap), group in action_prob_frame.groupby(["checkpoint_step", "target_dap"], sort=True):
        original = group[group["weather_variant"].eq("original")].sort_values("action_index")
        if original.empty:
            continue
        original_probs = original["probability"].to_numpy(dtype=float)
        original_mask = original["legal"].to_numpy(dtype=bool)
        for variant, variant_group in group.groupby("weather_variant", sort=True):
            variant_sorted = variant_group.sort_values("action_index")
            variant_probs = variant_sorted["probability"].to_numpy(dtype=float)
            variant_mask = variant_sorted["legal"].to_numpy(dtype=bool)
            union_mask = original_mask | variant_mask
            if len(original_probs) != len(variant_probs):
                continue
            l1_rows.append(
                {
                    "checkpoint_step": int(checkpoint),
                    "target_dap": int(target_dap),
                    "weather_variant": variant,
                    "probability_l1_vs_original": float(np.nansum(np.abs(variant_probs[union_mask] - original_probs[union_mask]))),
                    "mask_same_as_original": bool(np.array_equal(variant_mask, original_mask)),
                }
            )
    weather_sensitivity_frame = pd.DataFrame(l1_rows)

    trajectory_summary = (
        trajectory_frame.groupby(["checkpoint_step", "scenario_key", "weather_variant"], as_index=False)
        .apply(lambda g: pd.Series({
            "episode_steps_observed": int(len(g)),
            "action_signature_to_targets": action_signature(g),
            "nonzero_action_count_to_targets": int(
                ((g["irrigation_mm"].astype(float) > 1e-9) | (g["nitrogen_kg_ha"].astype(float) > 1e-9)).sum()
            ),
        }))
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # Step 4. Save tables and machine-readable root-cause summary.
    # ------------------------------------------------------------------
    decision_frame.to_csv(OUT / "evaluation" / "183_decision_state_observations.csv", index=False, encoding="utf-8-sig")
    action_prob_frame.to_csv(OUT / "evaluation" / "183_action_mask_policy_probs.csv", index=False, encoding="utf-8-sig")
    observation_frame.to_csv(OUT / "evaluation" / "183_observation_vector_values.csv", index=False, encoding="utf-8-sig")
    trajectory_frame.to_csv(OUT / "debug" / "183_replay_trajectories_to_targets.csv", index=False, encoding="utf-8-sig")
    policy_summary_frame.to_csv(OUT / "evaluation" / "183_policy_concentration_summary.csv", index=False, encoding="utf-8-sig")
    checkpoint_comparison_frame.to_csv(OUT / "evaluation" / "183_checkpoint_comparison.csv", index=False, encoding="utf-8-sig")
    weather_sensitivity_frame.to_csv(OUT / "evaluation" / "183_weather_scenario_probability_l1.csv", index=False, encoding="utf-8-sig")
    trajectory_summary.to_csv(OUT / "evaluation" / "183_replay_signature_summary.csv", index=False, encoding="utf-8-sig")

    ckpt2000 = decision_frame[decision_frame["checkpoint_step"].eq(2000)].copy()
    ckpt1000 = decision_frame[decision_frame["checkpoint_step"].eq(1000)].copy()
    top_threshold = float(cfg["policy_concentration_top_prob_threshold"])
    mask_forced_i45_rows = int((ckpt2000["selected_action"].eq(KEY_ACTIONS["i45_n0"]) & ~ckpt2000["noop_legal"].astype(bool)).sum())
    selected_i45_noop_legal_rows = int((ckpt2000["selected_action"].eq(KEY_ACTIONS["i45_n0"]) & ckpt2000["noop_legal"].astype(bool)).sum())
    high_concentration_rows = int((ckpt2000["top_probability"] >= top_threshold).sum())
    invariant_groups_ckpt2000 = int(
        policy_summary_frame[
            policy_summary_frame["checkpoint_step"].eq(2000)
            & policy_summary_frame["weather_or_state_varied"].astype(bool)
            & policy_summary_frame["action_invariant_across_weather"].astype(bool)
        ].shape[0]
    )
    changed_checkpoint_rows = int(checkpoint_comparison_frame["selected_action_changed"].sum()) if not checkpoint_comparison_frame.empty else 0
    drop_profit_positive_rows = int((ckpt2000.get("task182_delta_profit_when_event_removed", pd.Series(dtype=float)) > 1e-6).sum())

    if mask_forced_i45_rows > 0:
        primary_label = "mask_constraint_contributes_to_i45_selection"
    elif selected_i45_noop_legal_rows > 0 and high_concentration_rows > 0:
        primary_label = "policy_prefers_i45_without_mask_force"
    elif selected_i45_noop_legal_rows > 0:
        primary_label = "policy_selects_i45_without_mask_force_but_not_highly_concentrated"
    else:
        primary_label = "target_i45_not_selected_at_checkpoint2000"

    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "status": "completed",
        "mode": "read_only_decision_point_policy_audit",
        "training_executed": False,
        "source_task": "177_sya_dense_yield_credit_smoke2k",
        "ablation_reference_task": "182_sya_fullseason_ppo_action_ablation",
        "station": STATION,
        "checkpoint_steps": CHECKPOINTS,
        "target_daps": TARGET_DAPS,
        "scenario_count": len(SCENARIOS),
        "decision_rows": int(len(decision_frame)),
        "action_probability_rows": int(len(action_prob_frame)),
        "physical_gate_passed": bool(gate.get("physical_response_passed")),
        "rootcause_summary": {
            "primary_label": primary_label,
            "ckpt2000_selected_i45_noop_legal_rows": selected_i45_noop_legal_rows,
            "ckpt2000_mask_forced_i45_rows": mask_forced_i45_rows,
            "ckpt2000_high_concentration_rows_top_prob_ge_threshold": high_concentration_rows,
            "top_probability_threshold": top_threshold,
            "ckpt2000_weather_or_state_varied_but_action_invariant_groups": invariant_groups_ckpt2000,
            "checkpoint_selected_action_changed_rows": changed_checkpoint_rows,
            "ckpt2000_task182_drop_event_profit_positive_rows": drop_profit_positive_rows,
            "interpretation": (
                "Read together with task 182: if I45 is selected while no-op is legal and deleting "
                "the event improves profit/reward, the issue is not mask forcing but learned policy preference."
            ),
        },
        "outputs": {
            "decision_states": relative(OUT / "evaluation" / "183_decision_state_observations.csv"),
            "action_policy_probs": relative(OUT / "evaluation" / "183_action_mask_policy_probs.csv"),
            "observation_vectors": relative(OUT / "evaluation" / "183_observation_vector_values.csv"),
            "policy_summary": relative(OUT / "evaluation" / "183_policy_concentration_summary.csv"),
            "checkpoint_comparison": relative(OUT / "evaluation" / "183_checkpoint_comparison.csv"),
            "weather_probability_l1": relative(OUT / "evaluation" / "183_weather_scenario_probability_l1.csv"),
            "replay_signatures": relative(OUT / "evaluation" / "183_replay_signature_summary.csv"),
        },
    }
    (OUT / "183_rootcause_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    DOC.write_text(
        "\n".join(
            [
                "# Task 183：SYA PPO DAP97/104 决策点根因审计记录",
                "",
                "> 自动简要记录；详细解释运行后按模板补充。",
                "",
                f"- 状态：completed；decision rows：{len(decision_frame)}；action probability rows：{len(action_prob_frame)}。",
                f"- checkpoint：{CHECKPOINTS}；目标 DAP：{TARGET_DAPS}；情景数：{len(SCENARIOS)}。",
                f"- primary label：`{primary_label}`。",
                f"- ckpt2000 I45 且 no-op 合法行数：`{selected_i45_noop_legal_rows}`。",
                f"- ckpt2000 mask-forced I45 行数：`{mask_forced_i45_rows}`。",
                f"- ckpt2000 高集中度行数：`{high_concentration_rows}`。",
                "",
                "## 输出",
                "",
                f"- `{relative(OUT / '183_rootcause_summary.json')}`",
                f"- `{relative(OUT / 'evaluation' / '183_decision_state_observations.csv')}`",
                f"- `{relative(OUT / 'evaluation' / '183_action_mask_policy_probs.csv')}`",
                f"- `{relative(OUT / 'evaluation' / '183_policy_concentration_summary.csv')}`",
                f"- `{relative(OUT / 'evaluation' / '183_checkpoint_comparison.csv')}`",
                f"- `{relative(OUT / 'evaluation' / '183_weather_scenario_probability_l1.csv')}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[Step 4/4] Completed. primary_label={primary_label}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    print(json.dumps(run(config_path.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
