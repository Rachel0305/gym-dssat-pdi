from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from imitation_policy_models import as_feature_row, predict_real_action
from ppo_action_safety import ActionSafetyState, apply_action_safety, normalize_action, update_action_safety_state
from ppo_evaluate import latest_observation_dict, make_env, scalar
from ppo_experiment_plan import find_year
from ppo_plot_results import plot_episode
from ppo_safe_rendering import PROJECT_ROOT


def profit_score(final_grnwt: float, irrigation: float, nitrogen: float, config: dict) -> float:
    econ = config.get("economics", {})
    return (
        float(econ.get("grain_value_coef", 0.01)) * float(final_grnwt)
        - float(econ.get("water_cost", 0.5)) * float(irrigation)
        - float(econ.get("n_cost", 0.25)) * float(nitrogen)
    )


def normalized_action_dict(env, normalized_action) -> dict[str, float]:
    arr = np.asarray(normalized_action).flatten()
    return {name: float(value) for name, value in zip(env.formator.action_names, arr)}


def evaluate_imitation_policy(
    policy,
    config: dict,
    station: str,
    eval_year: int,
    policy_name: str,
    model_type: str,
    expert_lookup: pd.DataFrame,
    ppo_reference_yield: float,
) -> dict[str, Any]:
    seed = int(config.get("seed", 0))
    output_root = PROJECT_ROOT / config["paths"]["output_root"]
    env = make_env(config, station, int(eval_year), seed, run_tag=f"{policy_name}_{station}_{eval_year}", evaluation=True, action_safety_enabled=False)
    records: list[dict[str, Any]] = []
    safety_state = ActionSafetyState()
    safety_config = {**config.get("action_safety", {}), "enabled": True}
    post = config.get("action_postprocess", {})
    threshold = float(post.get("event_threshold", 25.0))
    cumulative_irrigation = 0.0
    cumulative_n = 0.0
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        year_info = find_year(config, station, int(eval_year))
        while not done and step_count < int(config.get("runtime", {}).get("max_steps", 260)):
            latest = latest_observation_dict(env, obs, info)
            sim_day = step_count + 1
            planting = pd.Timestamp(year_info["planting_date"])
            date = planting + pd.Timedelta(days=step_count)
            features = as_feature_row(station, sim_day, int(date.dayofyear), latest)
            raw_action = predict_real_action(policy, features, postprocess=True, threshold=threshold)
            safety_result = apply_action_safety(raw_action, sim_day, safety_state, safety_config)
            safe_norm = normalize_action(env.formator.action_names, env.formator.action_space_dict, safety_result.safe_real_action)
            norm_action = normalized_action_dict(env, safe_norm)
            obs, reward, terminated, truncated, info = env.step(safe_norm)
            done = bool(terminated or truncated)
            latest = latest_observation_dict(env, obs, info)
            update_action_safety_state(safety_state, safety_result.safe_real_action, sim_day)
            real_amir = float(safety_result.safe_real_action.get("amir", 0.0))
            real_anfer = float(safety_result.safe_real_action.get("anfer", 0.0))
            cumulative_irrigation += real_amir
            cumulative_n += real_anfer
            totir_raw = scalar(latest.get("totir"))
            tofer_raw = scalar(latest.get("tofer"))
            records.append(
                {
                    "station": station,
                    "policy_name": policy_name,
                    "model_type": model_type,
                    "eval_year": int(eval_year),
                    "date": date.strftime("%Y-%m-%d"),
                    "year": int(date.year),
                    "doy": int(date.dayofyear),
                    "sim_day": int(sim_day),
                    "dap": scalar(latest.get("dap", sim_day)),
                    "topwt": scalar(latest.get("topwt")),
                    "grnwt": scalar(latest.get("grnwt")),
                    "xlai": scalar(latest.get("xlai")),
                    "totir_raw": totir_raw,
                    "totir": totir_raw if not np.isnan(totir_raw) else cumulative_irrigation,
                    "tofer_raw": tofer_raw,
                    "tofer": tofer_raw if not np.isnan(tofer_raw) else cumulative_n,
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "reward": float(reward),
                    "real_action_amir": real_amir,
                    "real_action_anfer": real_anfer,
                    "normalized_action_amir": norm_action.get("amir", np.nan),
                    "normalized_action_anfer": norm_action.get("anfer", np.nan),
                    "raw_real_action_amir": float(safety_result.raw_real_action.get("amir", 0.0)),
                    "raw_real_action_anfer": float(safety_result.raw_real_action.get("anfer", 0.0)),
                    "safe_real_action_amir": real_amir,
                    "safe_real_action_anfer": real_anfer,
                    "safety_rule_triggered": safety_result.safety_rule_triggered,
                    "done": done,
                    "info": json.dumps(info, ensure_ascii=False, default=str),
                }
            )
            step_count += 1
    finally:
        try:
            env.close()
        except Exception:
            pass

    daily = pd.DataFrame(records)
    daily_dir = output_root / "daily_outputs" / station
    daily_dir.mkdir(parents=True, exist_ok=True)
    daily_csv = daily_dir / f"{station}_{policy_name}_eval{eval_year}_daily.csv"
    daily.to_csv(daily_csv, index=False, encoding="utf-8-sig")
    fig_dir = output_root / "figures" / station / policy_name / f"eval_{eval_year}"
    plot_episode(daily, fig_dir)

    episode_completed = bool(records and records[-1]["done"])
    final_grnwt = float(daily["grnwt"].iloc[-1]) if len(daily) else np.nan
    total_irrigation = float(daily["real_action_amir"].sum()) if len(daily) else 0.0
    total_n = float(daily["real_action_anfer"].sum()) if len(daily) else 0.0
    expert = expert_lookup[(expert_lookup["station"].eq(station)) & (expert_lookup["eval_year"].astype(int).eq(int(eval_year)))]
    expert_yield = float(expert["final_grnwt"].iloc[0]) if len(expert) else np.nan
    expert_irrigation = float(expert["total_irrigation"].iloc[0]) if len(expert) else np.nan
    expert_n = float(expert["total_n_fertilizer"].iloc[0]) if len(expert) else np.nan
    return {
        "station": station,
        "policy_name": policy_name,
        "model_type": model_type,
        "eval_year": int(eval_year),
        "run_status": "ok" if episode_completed else "failed",
        "episode_completed": episode_completed,
        "episode_length": int(len(daily)),
        "final_grnwt": final_grnwt,
        "total_irrigation": total_irrigation,
        "total_n_fertilizer": total_n,
        "mean_swfac": float(pd.to_numeric(daily["swfac"], errors="coerce").mean()) if len(daily) else np.nan,
        "mean_nstres": float(pd.to_numeric(daily["nstres"], errors="coerce").mean()) if len(daily) else np.nan,
        "profit_score": profit_score(final_grnwt, total_irrigation, total_n, config) if not np.isnan(final_grnwt) else np.nan,
        "yield_loss_vs_expert_schedule": (expert_yield - final_grnwt) / expert_yield if expert_yield and not np.isnan(expert_yield) else np.nan,
        "yield_loss_vs_ppo_baseline": (ppo_reference_yield - final_grnwt) / ppo_reference_yield if ppo_reference_yield else np.nan,
        "irrigation_diff_vs_expert": total_irrigation - expert_irrigation if not np.isnan(expert_irrigation) else np.nan,
        "n_diff_vs_expert": total_n - expert_n if not np.isnan(expert_n) else np.nan,
        "daily_csv_path": str(daily_csv.relative_to(PROJECT_ROOT)),
        "figure_dir": str(fig_dir.relative_to(PROJECT_ROOT)),
        "notes": "action_safe_bc_eval",
    }
