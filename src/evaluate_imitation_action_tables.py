from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ppo_action_safety import ActionSafetyState, apply_action_safety, normalize_action, update_action_safety_state
from ppo_evaluate import latest_observation_dict, make_env, scalar
from ppo_experiment_plan import find_year
from ppo_plot_results import plot_episode
from ppo_safe_rendering import PROJECT_ROOT, load_yaml


CONFIG_PATH = PROJECT_ROOT / "experiments" / "ppo_observed_years" / "config_imitation_learning_prior.yaml"


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


def load_action_tables(output_root: Path) -> dict[str, pd.DataFrame]:
    table_dir = output_root / "evaluation" / "policy_action_tables"
    tables: dict[str, pd.DataFrame] = {}
    for path in sorted(table_dir.glob("*_action_table.csv")):
        df = pd.read_csv(path)
        tables[str(df["policy_name"].iloc[0])] = df
    return tables


def action_for(table: pd.DataFrame, station: str, sim_day: int) -> dict[str, float]:
    match = table[(table["station"].astype(str).eq(station)) & (table["sim_day"].astype(int).eq(int(sim_day)))]
    if len(match):
        return {
            "amir": float(match["real_action_amir"].iloc[0]),
            "anfer": float(match["real_action_anfer"].iloc[0]),
        }
    return {"amir": 0.0, "anfer": 0.0}


def evaluate_table_policy(config: dict, table: pd.DataFrame, station: str, eval_year: int) -> dict:
    output_root = PROJECT_ROOT / config["paths"]["output_root"]
    policy_name = str(table["policy_name"].iloc[0])
    model_type = str(table["model_type"].iloc[0])
    seed = int(config.get("seed", 0))
    env = make_env(config, station, int(eval_year), seed, run_tag=f"{policy_name}_{station}_{eval_year}", evaluation=True, action_safety_enabled=False)
    records: list[dict] = []
    safety_state = ActionSafetyState()
    safety_config = {**config.get("action_safety", {}), "enabled": True}
    cumulative_irrigation = 0.0
    cumulative_n = 0.0
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        year_info = find_year(config, station, int(eval_year))
        planting = pd.Timestamp(year_info["planting_date"])
        while not done and step_count < int(config.get("runtime", {}).get("max_steps", 260)):
            sim_day = step_count + 1
            date = planting + pd.Timedelta(days=step_count)
            raw_action = action_for(table, station, sim_day)
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
        "daily_csv_path": str(daily_csv.relative_to(PROJECT_ROOT)),
        "figure_dir": str(fig_dir.relative_to(PROJECT_ROOT)),
        "notes": "action_table_replay_from_bc_model",
    }


def add_expert_and_ppo_deltas(config: dict, summary: pd.DataFrame) -> pd.DataFrame:
    output_root = PROJECT_ROOT / config["paths"]["output_root"]
    expert = pd.read_csv(output_root / "evaluation" / "expert_schedule_lookup_for_bc.csv")
    ref_yield = float(config.get("ppo_reference", {}).get("mean_yield", np.nan))
    rows = []
    for _, row in summary.iterrows():
        match = expert[(expert["station"].eq(row["station"])) & (expert["eval_year"].astype(int).eq(int(row["eval_year"])))]
        expert_yield = float(match["final_grnwt"].iloc[0]) if len(match) else np.nan
        expert_i = float(match["total_irrigation"].iloc[0]) if len(match) else np.nan
        expert_n = float(match["total_n_fertilizer"].iloc[0]) if len(match) else np.nan
        item = dict(row)
        item["yield_loss_vs_expert_schedule"] = (expert_yield - row["final_grnwt"]) / expert_yield if expert_yield and not np.isnan(expert_yield) and row["run_status"] == "ok" else np.nan
        item["yield_loss_vs_ppo_baseline"] = (ref_yield - row["final_grnwt"]) / ref_yield if ref_yield and row["run_status"] == "ok" else np.nan
        item["irrigation_diff_vs_expert"] = row["total_irrigation"] - expert_i if not np.isnan(expert_i) and row["run_status"] == "ok" else np.nan
        item["n_diff_vs_expert"] = row["total_n_fertilizer"] - expert_n if not np.isnan(expert_n) and row["run_status"] == "ok" else np.nan
        rows.append(item)
    return pd.DataFrame(rows)


def main() -> None:
    config = load_yaml(CONFIG_PATH)
    output_root = PROJECT_ROOT / config["paths"]["output_root"]
    tables = load_action_tables(output_root)
    rows: list[dict] = []
    for name, table in tables.items():
        for station, years in config["imitation"]["evaluation_years"].items():
            for year in years:
                print(f"[imitation-table] DSSAT eval {name} {station} {year}", flush=True)
                try:
                    rows.append(evaluate_table_policy(config, table, station, int(year)))
                except Exception as exc:
                    rows.append(
                        {
                            "station": station,
                            "policy_name": name,
                            "model_type": str(table["model_type"].iloc[0]),
                            "eval_year": int(year),
                            "run_status": "failed",
                            "episode_completed": False,
                            "error_message": f"{type(exc).__name__}: {exc}",
                        }
                    )
    summary = add_expert_and_ppo_deltas(config, pd.DataFrame(rows))
    out = output_root / "evaluation" / "imitation_policy_dssat_evaluation_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
