from __future__ import annotations

import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constrained_ppo_wrappers import ImitationPenaltyRewardWrapper, PriorActionTable, PriorReplayPolicy, ResidualPriorActionWrapper
from episode_profit_reward import EpisodeProfitRewardWrapper
from ppo_action_safety import ActionSafetyState, SafeActionWrapper, apply_action_safety, normalize_action, update_action_safety_state
from ppo_evaluate import latest_observation_dict, make_env, scalar
from ppo_experiment_plan import find_year
from ppo_plot_results import plot_episode
from ppo_safe_rendering import PROJECT_ROOT, load_yaml


CONFIG_PATH = PROJECT_ROOT / "experiments" / "ppo_observed_years" / "config_constrained_ppo_finetuning.yaml"
OUTPUT_ROOT = PROJECT_ROOT / "Leave_One_experiments" / "constrained_ppo_finetuning"
DOC_MD = PROJECT_ROOT / "docs" / "2026-06-06_constrained_ppo_finetuning_report.md"
DOC_PPT = PROJECT_ROOT / "docs" / "2026-06-06_constrained_ppo_finetuning_report.pptx"


def ensure_dirs() -> None:
    for sub in ["configs", "prior_policies", "models", "logs", "tensorboard", "daily_outputs", "evaluation", "figures", "reports"]:
        (OUTPUT_ROOT / sub).mkdir(parents=True, exist_ok=True)
    for st in ["HLA", "SYA", "LCA"]:
        (OUTPUT_ROOT / "models" / st).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "daily_outputs" / st).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "figures" / st).mkdir(parents=True, exist_ok=True)


def df_to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return ""
    work = df.copy()
    if max_rows is not None:
        work = work.head(max_rows)
    for col in work.select_dtypes(include=["float", "int"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
    for col in work.columns:
        work[col] = work[col].map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(work.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in work.astype(str).values.tolist()]
    return "\n".join([header, separator, *rows])


def profit_score(final_grnwt: float, irrigation: float, nitrogen: float, config: dict) -> float:
    econ = config.get("economics", {})
    return (
        float(econ.get("grain_value_coef", 0.01)) * float(final_grnwt)
        - float(econ.get("water_cost", 0.5)) * float(irrigation)
        - float(econ.get("n_cost", 0.25)) * float(nitrogen)
    )


def ppo_kwargs(config: dict) -> dict:
    allowed = ["learning_rate", "gamma", "n_steps", "batch_size", "ent_coef", "clip_range"]
    return {key: value for key, value in config.get("ppo", {}).items() if key in allowed and value is not None}


def method_safety_config(config: dict, method: dict) -> dict:
    safety = dict(config.get("action_safety", {}))
    safety["enabled"] = True
    safety["season_irrigation_soft_limit"] = float(method.get("guardrail_irrigation", safety.get("season_irrigation_soft_limit", 300.0)))
    safety["season_n_soft_limit"] = float(method.get("guardrail_n", safety.get("season_n_soft_limit", 450.0)))
    return safety


def prior_for_method(config: dict, method: dict) -> PriorActionTable:
    policy = str(method.get("prior_policy", config["primary_prior"]["policy_name"]))
    if policy == config["primary_prior"]["policy_name"]:
        path = PROJECT_ROOT / config["primary_prior"]["action_table"]
    elif policy == config["backup_prior"]["policy_name"]:
        path = PROJECT_ROOT / config["backup_prior"]["action_table"]
    else:
        raise KeyError(f"Unknown prior policy: {policy}")
    return PriorActionTable(path)


def build_train_env(config: dict, station: str, year: int, method: dict):
    base = make_env(config, station, year, int(config.get("seed", 0)), run_tag=f"{method['name']}_{station}_{year}_train", evaluation=False, action_safety_enabled=False)
    safe = SafeActionWrapper(base, method_safety_config(config, method))
    profit = EpisodeProfitRewardWrapper(safe, "P2_terminal_profit_medium_cost")
    prior = prior_for_method(config, method)
    if method["kind"] == "imitation_penalty":
        return ImitationPenaltyRewardWrapper(profit, prior, station, float(method.get("lambda_bc", 0.5)))
    if method["kind"] == "residual":
        return ResidualPriorActionWrapper(
            profit,
            prior,
            station,
            float(method.get("irrigation_residual_range", 20.0)),
            float(method.get("n_residual_range", 30.0)),
        )
    return profit


def build_eval_env(config: dict, station: str, year: int, method: dict, run_tag: str):
    base = make_env(config, station, year, int(config.get("seed", 0)), run_tag=run_tag, evaluation=True, action_safety_enabled=False)
    safe = SafeActionWrapper(base, method_safety_config(config, method))
    profit = EpisodeProfitRewardWrapper(safe, "P2_terminal_profit_medium_cost")
    prior = prior_for_method(config, method)
    if method["kind"] == "residual":
        return ResidualPriorActionWrapper(
            profit,
            prior,
            station,
            float(method.get("irrigation_residual_range", 20.0)),
            float(method.get("n_residual_range", 30.0)),
        )
    if method["kind"] == "imitation_penalty":
        return ImitationPenaltyRewardWrapper(profit, prior, station, float(method.get("lambda_bc", 0.5)))
    return profit


def latest_wrapper_info(env) -> dict:
    out: dict[str, Any] = {}
    for attr in ["last_bc_info", "last_residual_info", "last_profit_info"]:
        value = getattr(env, attr, None)
        if isinstance(value, dict):
            out.update(value)
    safety_result = getattr(env, "last_safety_result", None)
    if safety_result is not None:
        out["safety_rule_triggered"] = safety_result.safety_rule_triggered
        out["real_action_amir"] = float(safety_result.safe_real_action.get("amir", 0.0))
        out["real_action_anfer"] = float(safety_result.safe_real_action.get("anfer", 0.0))
        out["raw_real_action_amir"] = float(safety_result.raw_real_action.get("amir", 0.0))
        out["raw_real_action_anfer"] = float(safety_result.raw_real_action.get("anfer", 0.0))
    return out


def normalized_action_dict(env, normalized_action) -> dict[str, float]:
    arr = np.asarray(normalized_action).flatten()
    return {name: float(value) for name, value in zip(env.formator.action_names, arr)}


def evaluate_policy(config: dict, method: dict, station: str, train_year: int, eval_year: int, model, model_path: Path | None) -> dict:
    seed = int(config.get("seed", 0))
    policy_name = f"{method['name']}_{station}_train{train_year}_seed{seed}"
    env = build_eval_env(config, station, eval_year, method, run_tag=f"{policy_name}_eval{eval_year}")
    prior = prior_for_method(config, method)
    prior_policy = PriorReplayPolicy(env, prior, station)
    if method["kind"] == "prior_replay":
        prior_policy.reset()
    records: list[dict[str, Any]] = []
    cumulative_irrigation = 0.0
    cumulative_n = 0.0
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        year_info = find_year(config, station, int(eval_year))
        while not done and step_count < int(config.get("runtime", {}).get("max_steps", 260)):
            sim_day = step_count + 1
            if method["kind"] == "prior_replay":
                action, _ = prior_policy.predict(obs, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32).flatten()
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest = latest_observation_dict(env, obs, info)
            wrapper_info = latest_wrapper_info(env)
            real_amir = float(wrapper_info.get("real_action_amir", 0.0))
            real_anfer = float(wrapper_info.get("real_action_anfer", 0.0))
            cumulative_irrigation += real_amir
            cumulative_n += real_anfer
            date = pd.Timestamp(year_info["planting_date"]) + pd.Timedelta(days=step_count)
            totir_raw = scalar(latest.get("totir"))
            tofer_raw = scalar(latest.get("tofer"))
            norm = normalized_action_dict(env, action)
            prior_action = prior.action(station, sim_day)
            bc_dev = ((real_amir - prior_action["amir"]) / 100.0) ** 2 + ((real_anfer - prior_action["anfer"]) / 150.0) ** 2
            rec = {
                "station": station,
                "policy_name": policy_name,
                "finetune_method": method["name"],
                "prior_policy": method.get("prior_policy", config["primary_prior"]["policy_name"]),
                "train_year": int(train_year),
                "eval_year": int(eval_year),
                "seed": seed,
                "date": date.strftime("%Y-%m-%d"),
                "year": int(date.year),
                "doy": int(date.dayofyear),
                "sim_day": sim_day,
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
                "normalized_action_amir": norm.get("amir", np.nan),
                "normalized_action_anfer": norm.get("anfer", np.nan),
                "bc_prior_amir": float(prior_action["amir"]),
                "bc_prior_anfer": float(prior_action["anfer"]),
                "bc_action_deviation": float(bc_dev ** 0.5),
                "done": done,
                "info": json.dumps(info, ensure_ascii=False, default=str),
            }
            rec.update(wrapper_info)
            records.append(rec)
            step_count += 1
    finally:
        try:
            env.close()
        except Exception:
            pass
    daily = pd.DataFrame(records)
    daily_dir = OUTPUT_ROOT / "daily_outputs" / station
    daily_dir.mkdir(parents=True, exist_ok=True)
    daily_csv = daily_dir / f"{policy_name}_eval{eval_year}_daily.csv"
    daily.to_csv(daily_csv, index=False, encoding="utf-8-sig")
    fig_dir = OUTPUT_ROOT / "figures" / station / method["name"] / f"eval_{eval_year}"
    plot_episode(daily, fig_dir)
    episode_completed = bool(records and records[-1]["done"])
    final_grnwt = float(daily["grnwt"].iloc[-1]) if len(daily) else np.nan
    total_irrig = float(daily["real_action_amir"].sum()) if len(daily) else 0.0
    total_n = float(daily["real_action_anfer"].sum()) if len(daily) else 0.0
    return {
        "station": station,
        "policy_name": policy_name,
        "finetune_method": method["name"],
        "prior_policy": method.get("prior_policy", config["primary_prior"]["policy_name"]),
        "train_year": int(train_year),
        "eval_year": int(eval_year),
        "seed": seed,
        "model_path": str(model_path.relative_to(PROJECT_ROOT)) if model_path else "",
        "run_status": "ok" if episode_completed else "failed",
        "episode_completed": episode_completed,
        "episode_length": int(len(daily)),
        "final_grnwt": final_grnwt,
        "total_irrigation": total_irrig,
        "total_n_fertilizer": total_n,
        "mean_swfac": float(pd.to_numeric(daily["swfac"], errors="coerce").mean()) if len(daily) else np.nan,
        "mean_nstres": float(pd.to_numeric(daily["nstres"], errors="coerce").mean()) if len(daily) else np.nan,
        "profit_score": profit_score(final_grnwt, total_irrig, total_n, config) if not np.isnan(final_grnwt) else np.nan,
        "bc_action_deviation_mean": float(pd.to_numeric(daily["bc_action_deviation"], errors="coerce").mean()) if len(daily) else np.nan,
        "bc_action_deviation_max": float(pd.to_numeric(daily["bc_action_deviation"], errors="coerce").max()) if len(daily) else np.nan,
        "irrigation_saturation_ratio_300_450": total_irrig / 300.0,
        "n_saturation_ratio_300_450": total_n / 450.0,
        "irrigation_saturation_ratio_guardrail": total_irrig / float(method.get("guardrail_irrigation", 300.0)),
        "n_saturation_ratio_guardrail": total_n / float(method.get("guardrail_n", 450.0)),
        "daily_csv_path": str(daily_csv.relative_to(PROJECT_ROOT)),
        "figure_dir": str(fig_dir.relative_to(PROJECT_ROOT)),
        "notes": "constrained_ppo_eval",
    }


def run_null_or_fixed_smoke(config: dict, station: str, year: int, method_name: str) -> list[dict]:
    rows: list[dict] = []
    smoke_root = OUTPUT_ROOT / "smoke_checks"
    smoke_root.mkdir(parents=True, exist_ok=True)
    for policy in ["null_zero", "fixed_low_input"]:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "src" / "run_smoke_tests.py"),
            "--single",
            "--station",
            station,
            "--year",
            str(year),
            "--policy",
            policy,
        ]
        env = None
        try:
            import os

            env = os.environ.copy()
            env["SMOKE_TEST_OUTPUT_ROOT"] = str(smoke_root)
            completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, timeout=int(config["runtime"].get("episode_timeout_seconds", 300)), text=True, capture_output=True)
            single_path = smoke_root / "evaluation" / "single" / f"{station}_{year}_{policy}.json"
            if completed.returncode == 0 and single_path.exists():
                data = pd.read_json(single_path, typ="series").to_dict()
                rows.append({"station": station, "train_year": year, "finetune_method": method_name, "policy_name": policy, "run_status": data.get("run_status", "unknown"), "episode_completed": data.get("episode_completed", False), "notes": "pretrain_smoke_check"})
            else:
                rows.append({"station": station, "train_year": year, "finetune_method": method_name, "policy_name": policy, "run_status": "failed", "episode_completed": False, "error_message": (completed.stderr or completed.stdout)[-800:], "notes": "pretrain_smoke_check"})
        except Exception as exc:
            rows.append({"station": station, "train_year": year, "finetune_method": method_name, "policy_name": policy, "run_status": "failed", "episode_completed": False, "error_message": f"{type(exc).__name__}: {exc}", "notes": "pretrain_smoke_check"})
    return rows


def run_prior_smoke(config: dict, station: str, year: int, method: dict) -> dict:
    try:
        row = evaluate_policy(config, {**method, "kind": "prior_replay", "name": f"{method['name']}_prior_smoke"}, station, year, year, None, None)
        return {"station": station, "train_year": year, "finetune_method": method["name"], "policy_name": "prior_replay", "run_status": row["run_status"], "episode_completed": row["episode_completed"], "notes": "pretrain_smoke_check"}
    except Exception as exc:
        return {"station": station, "train_year": year, "finetune_method": method["name"], "policy_name": "prior_replay", "run_status": "failed", "episode_completed": False, "error_message": f"{type(exc).__name__}: {exc}", "notes": "pretrain_smoke_check"}


def run_training(config: dict, method: dict, station: str, train_year: int):
    from stable_baselines3 import PPO

    env = build_train_env(config, station, train_year, method)
    seed = int(config.get("seed", 0))
    policy_name = f"{method['name']}_{station}_train{train_year}_seed{seed}"
    model_dir = OUTPUT_ROOT / "models" / station
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / policy_name
    try:
        model = PPO("MlpPolicy", env, verbose=0, seed=seed, tensorboard_log=str(OUTPUT_ROOT / "tensorboard" / station), **ppo_kwargs(config))
        model.learn(total_timesteps=int(config["runtime"].get("total_timesteps", 5000)), progress_bar=False)
        model.save(str(model_path))
    finally:
        try:
            env.close()
        except Exception:
            pass
    return model, model_path.with_suffix(".zip")


def add_reference_deltas(config: dict, summary: pd.DataFrame) -> pd.DataFrame:
    expert = pd.read_csv(PROJECT_ROOT / "Leave_One_experiments" / "imitation_learning_prior" / "evaluation" / "policy_comparison_expert_bc_ppo.csv")
    prior = pd.read_csv(PROJECT_ROOT / "Leave_One_experiments" / "imitation_learning_prior" / "evaluation" / "imitation_policy_dssat_evaluation_summary.csv")
    ppo_yield = float(config["ppo_reference"]["mean_yield"])
    rows = []
    for _, row in summary.iterrows():
        item = dict(row)
        e = expert[(expert["policy_type"].eq("expert_schedule")) & (expert["station"].eq(row["station"])) & (expert["year"].astype(int).eq(int(row["eval_year"])))]
        b = prior[(prior["policy_name"].eq("BC_two_stage_classifier_regressor")) & (prior["station"].eq(row["station"])) & (prior["eval_year"].astype(int).eq(int(row["eval_year"])))]
        expert_y = float(e["final_grnwt"].iloc[0]) if len(e) else np.nan
        prior_y = float(b["final_grnwt"].iloc[0]) if len(b) else np.nan
        item["yield_loss_vs_prior"] = (prior_y - row["final_grnwt"]) / prior_y if prior_y and not np.isnan(prior_y) else np.nan
        item["yield_loss_vs_expert"] = (expert_y - row["final_grnwt"]) / expert_y if expert_y and not np.isnan(expert_y) else np.nan
        item["yield_loss_vs_ppo_baseline"] = (ppo_yield - row["final_grnwt"]) / ppo_yield if ppo_yield and not np.isnan(row["final_grnwt"]) else np.nan
        item["input_reduction_vs_ppo_baseline"] = 1.0 - ((row["total_irrigation"] / 300.0 + row["total_n_fertilizer"] / 450.0) / 2.0)
        rows.append(item)
    return pd.DataFrame(rows)


def write_prior_selection(config: dict) -> None:
    im = pd.read_csv(PROJECT_ROOT / "Leave_One_experiments" / "imitation_learning_prior" / "evaluation" / "imitation_policy_dssat_evaluation_summary.csv")
    grouped = im.groupby("policy_name").agg(mean_yield=("final_grnwt", "mean"), mean_irrigation=("total_irrigation", "mean"), mean_n=("total_n_fertilizer", "mean"), mean_yield_loss_vs_ppo=("yield_loss_vs_ppo_baseline", "mean"), ok=("run_status", lambda s: int((s == "ok").sum()))).reset_index()
    text = f"""# Prior policy selection

Generated at: 2026-06-06

`BC_constant_schedule_baseline` is not selected as the main learned prior because it is a deterministic expert replay baseline rather than a learned generalizable policy. It is useful as a reference but not as the primary PPO initialization target.

`BC_two_stage_classifier_regressor` is selected as the main prior because it handles sparse expert actions explicitly and passed the 006_09 gate: low irrigation, about 154 kg/ha N, and acceptable yield. `BC_random_forest_regressor` is kept as a backup because it also avoided cap saturation but had lower yield than the two-stage prior. `BC_mlp_regressor` is not recommended because it under-applied N and lost too much yield.

Prior limitation: selected expert schedules have zero irrigation and sparse nitrogen event actions, so constrained PPO is mainly learning around a nitrogen schedule prior rather than a full irrigation optimization prior.

{df_to_markdown(grouped)}
"""
    (OUTPUT_ROOT / "evaluation" / "prior_policy_selection.md").write_text(text, encoding="utf-8")


def make_policy_comparison(config: dict, summary: pd.DataFrame) -> pd.DataFrame:
    old = pd.read_csv(PROJECT_ROOT / "Leave_One_experiments" / "imitation_learning_prior" / "evaluation" / "policy_comparison_expert_bc_ppo.csv")
    rows = []
    for _, row in old.iterrows():
        if row["policy_type"] in ["expert_schedule", "ppo_cap_saturated_baseline"] or row["policy_name"] in ["BC_two_stage_classifier_regressor", "BC_random_forest_regressor"]:
            rows.append(dict(row))
    for _, row in summary.iterrows():
        if row["run_status"] == "ok":
            rows.append(
                {
                    "station": row["station"],
                    "year": int(row["eval_year"]),
                    "policy_type": "constrained_ppo_finetune" if not row["finetune_method"].startswith("FT0") else "prior_replay",
                    "policy_name": row["finetune_method"],
                    "final_grnwt": row["final_grnwt"],
                    "total_irrigation": row["total_irrigation"],
                    "total_n_fertilizer": row["total_n_fertilizer"],
                    "profit_score": row["profit_score"],
                    "mean_swfac": row["mean_swfac"],
                    "mean_nstres": row["mean_nstres"],
                    "yield_loss_vs_ppo": row["yield_loss_vs_ppo_baseline"],
                    "input_reduction_vs_ppo": row["input_reduction_vs_ppo_baseline"],
                    "notes": row["notes"],
                }
            )
    comp = pd.DataFrame(rows)
    comp.to_csv(OUTPUT_ROOT / "evaluation" / "policy_comparison_expert_bc_finetune_ppo.csv", index=False, encoding="utf-8-sig")
    return comp


def make_figures(summary: pd.DataFrame, comparison: pd.DataFrame) -> None:
    fig_dir = OUTPUT_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ok = summary[summary["run_status"].eq("ok")].copy()
    plt.figure(figsize=(9, 5))
    for name, part in ok.groupby("finetune_method"):
        plt.scatter(part["total_n_fertilizer"], part["final_grnwt"], label=name)
    plt.xlabel("Total N fertilizer (kg/ha)")
    plt.ylabel("Final grain yield")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(fig_dir / "constrained_ppo_yield_vs_input.png", dpi=180)
    plt.close()

    grouped = ok.groupby("finetune_method").agg(mean_profit=("profit_score", "mean"), mean_irrigation=("total_irrigation", "mean"), mean_n=("total_n_fertilizer", "mean"), mean_yield=("final_grnwt", "mean")).reset_index()
    plt.figure(figsize=(9, 5))
    plt.barh(grouped["finetune_method"], grouped["mean_profit"], color="#4F7ECF")
    plt.xlabel("Mean profit score")
    plt.tight_layout()
    plt.savefig(fig_dir / "constrained_ppo_profit_comparison.png", dpi=180)
    plt.close()

    x = np.arange(len(grouped))
    plt.figure(figsize=(9, 5))
    plt.bar(x - 0.2, grouped["mean_irrigation"], width=0.4, label="irrigation")
    plt.bar(x + 0.2, grouped["mean_n"], width=0.4, label="N")
    plt.xticks(x, grouped["finetune_method"], rotation=25, ha="right")
    plt.ylabel("Seasonal input")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "constrained_ppo_inputs_by_policy.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.hist(ok["bc_action_deviation_mean"], bins=20, color="#4F7ECF")
    plt.xlabel("Mean BC action deviation")
    plt.ylabel("Evaluations")
    plt.tight_layout()
    plt.savefig(fig_dir / "bc_deviation_distribution.png", dpi=180)
    plt.close()

    for fname, col in [("site_level_policy_comparison.png", "final_grnwt")]:
        plt.figure(figsize=(9, 5))
        site = ok.groupby(["station", "finetune_method"])[col].mean().reset_index()
        for station, part in site.groupby("station"):
            plt.plot(part["finetune_method"], part[col], marker="o", label=station)
        plt.xticks(rotation=25, ha="right")
        plt.ylabel(col)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / fname, dpi=180)
        plt.close()

    # Daily examples for the recommended low-input method if present.
    for out_name, cumulative in [("expert_bc_finetune_daily_actions.png", False), ("expert_bc_finetune_cumulative_inputs.png", True)]:
        plt.figure(figsize=(9, 5))
        for rel in ok.sort_values("finetune_method")["daily_csv_path"].dropna().head(4):
            daily = pd.read_csv(PROJECT_ROOT / rel)
            y = daily["real_action_anfer"].cumsum() if cumulative else daily["real_action_anfer"]
            label = f"{daily['station'].iloc[0]} {daily['finetune_method'].iloc[0]}"
            plt.plot(daily["sim_day"], y, label=label)
        plt.xlabel("Simulation day")
        plt.ylabel("Cumulative N" if cumulative else "Daily N action")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(fig_dir / out_name, dpi=180)
        plt.close()


def report_and_ppt(config: dict, summary: pd.DataFrame, comparison: pd.DataFrame) -> None:
    ok = summary[summary["run_status"].eq("ok")]
    method_summary = ok.groupby("finetune_method").agg(eval_count=("eval_year", "count"), mean_yield=("final_grnwt", "mean"), mean_profit=("profit_score", "mean"), mean_irrigation=("total_irrigation", "mean"), mean_n=("total_n_fertilizer", "mean"), mean_yield_loss_vs_ppo=("yield_loss_vs_ppo_baseline", "mean"), mean_bc_deviation=("bc_action_deviation_mean", "mean"), max_i_sat=("irrigation_saturation_ratio_300_450", "max"), max_n_sat=("n_saturation_ratio_300_450", "max")).reset_index()
    bc_prior = pd.read_csv(PROJECT_ROOT / "Leave_One_experiments" / "imitation_learning_prior" / "evaluation" / "imitation_policy_dssat_evaluation_summary.csv")
    bc_two_stage_profit = float(
        bc_prior.loc[bc_prior["policy_name"].eq("BC_two_stage_classifier_regressor"), "profit_score"].mean()
    )
    method_summary["profit_gap_vs_bc_two_stage"] = method_summary["mean_profit"] - bc_two_stage_profit
    passed = method_summary[
        (method_summary["eval_count"] >= 10)
        & (method_summary["mean_irrigation"] <= 100)
        & (method_summary["mean_n"] <= 200)
        & (method_summary["mean_yield_loss_vs_ppo"] <= 0.15)
        & (method_summary["mean_profit"] >= 0.9 * bc_two_stage_profit)
        & (method_summary["max_i_sat"] < 0.99)
        & (method_summary["max_n_sat"] < 0.99)
    ]
    recommended = passed.sort_values(["mean_profit", "mean_yield"], ascending=[False, False]).head(1)
    text = f"""# Constrained PPO fine-tuning report

Generated at: 2026-06-06

## Goal

This stage tests constrained PPO fine-tuning from imitation priors. It does not train unrestricted PPO, does not run multi-seed PPO, does not enter rainfall scaling, and does not modify my_data or previous 006_03-006_09 outputs.

## Prior selection

Main prior: `BC_two_stage_classifier_regressor`. Backup prior: `BC_random_forest_regressor`. `BC_constant_schedule_baseline` is only an expert replay reference, not the main learned prior.

## Methods tested

- FT0_BC_two_stage_replay: prior replay, no PPO training.
- FT1_bc_penalty_lambda_0p5: PPO with action deviation penalty around BC prior.
- FT2_residual_bc_prior: PPO residual action around BC prior.
- FT3_budget_guardrail_100_200: PPO with imitation penalty and strict 100/200 seasonal guardrail.

## Method summary

{df_to_markdown(method_summary)}

## Recommended method

{df_to_markdown(recommended) if len(recommended) else 'No fine-tuned PPO method passed all gates.'}

## Interpretation

Methods are judged by run success, irrigation <= 100 mm, N <= 200 kg/ha, yield loss vs PPO baseline <= 15%, mean profit at least 90% of the BC_two_stage prior profit, and no 300/450 saturation. If a fine-tuned method increases inputs substantially without meaningful yield/profit gain over the BC prior, it is not recommended.

## Next step

If a recommended fine-tuning method exists, run constrained PPO multi-seed only for that method. If no fine-tuned PPO method passes all gates, do not run multi-seed yet; return to expert dataset augmentation, stronger profit/cost calibration, or a stricter residual/guardrail formulation. Rainfall-scaling should remain later, after the constrained method is stable.
"""
    DOC_MD.write_text(text, encoding="utf-8")
    shutil.copy2(DOC_MD, OUTPUT_ROOT / "reports" / DOC_MD.name)

    try:
        from pptx import Presentation
        from pptx.dml.color import RGBColor
        from pptx.enum.text import PP_ALIGN
        from pptx.util import Inches, Pt
    except ModuleNotFoundError:
        return

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    def title(slide, t):
        box = slide.shapes.add_textbox(Inches(0.45), Inches(0.25), Inches(12.4), Inches(0.45))
        p = box.text_frame.paragraphs[0]
        p.text = t
        p.font.name = "Microsoft YaHei"
        p.font.size = Pt(22)
        p.font.bold = True
        p.font.color.rgb = RGBColor(0, 0, 0)

    def bullets(slide, lines):
        box = slide.shapes.add_textbox(Inches(0.7), Inches(1.0), Inches(12), Inches(5.8))
        tf = box.text_frame
        tf.clear()
        for i, line in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = line
            p.font.name = "Microsoft YaHei"
            p.font.size = Pt(15)
            p.font.color.rgb = RGBColor(0, 0, 0)

    def table(slide, df):
        data = df.head(8).copy()
        shp = slide.shapes.add_table(len(data) + 1, len(data.columns), Inches(0.35), Inches(1.0), Inches(12.6), Inches(5.8))
        tbl = shp.table
        for j, col in enumerate(data.columns):
            cell = tbl.cell(0, j)
            cell.text = str(col)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(79, 126, 207)
            for p in cell.text_frame.paragraphs:
                p.font.name = "Microsoft YaHei"
                p.font.size = Pt(8)
                p.font.bold = True
                p.font.color.rgb = RGBColor(255, 255, 255)
                p.alignment = PP_ALIGN.CENTER
        for i, (_, row) in enumerate(data.iterrows(), start=1):
            for j, col in enumerate(data.columns):
                value = row[col]
                if isinstance(value, float):
                    value = round(value, 4)
                cell = tbl.cell(i, j)
                cell.text = "" if pd.isna(value) else str(value)
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(235, 240, 250) if i % 2 == 0 else RGBColor(255, 255, 255)
                for p in cell.text_frame.paragraphs:
                    p.font.name = "Microsoft YaHei"
                    p.font.size = Pt(7)
                    p.font.color.rgb = RGBColor(0, 0, 0)

    slide = prs.slides.add_slide(blank)
    title(slide, "006_10 Constrained PPO fine-tuning")
    bullets(slide, ["主 prior: BC_two_stage_classifier_regressor。", "测试 FT0/FT1/FT2/FT3，seed=0，只做 HLA/SYA/LCA。", "目标：避免 300/450 饱和，同时保持 yield/profit。"])
    slide = prs.slides.add_slide(blank)
    title(slide, "Method summary")
    table(slide, method_summary.round(4))
    slide = prs.slides.add_slide(blank)
    title(slide, "Recommended method")
    table(slide, recommended.round(4) if len(recommended) else pd.DataFrame([{"result": "No method passed all gates"}]))
    slide = prs.slides.add_slide(blank)
    title(slide, "Next step")
    bullets(slide, ["若没有 fine-tuned PPO 同时通过投入、产量和 profit gate，不进入 multi-seed。", "rainfall-scaling budget scenario 仍应等 constrained 方法稳定之后再进入。", "不要把本阶段结果写成最终 RL 策略。"])
    prs.save(DOC_PPT)
    shutil.copy2(DOC_PPT, OUTPUT_ROOT / "reports" / DOC_PPT.name)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    config = load_yaml(CONFIG_PATH)
    shutil.copy2(CONFIG_PATH, OUTPUT_ROOT / "configs" / CONFIG_PATH.name)
    if args.report_only:
        summary = pd.read_csv(OUTPUT_ROOT / "evaluation" / "constrained_ppo_evaluation_summary.csv")
        comparison = pd.read_csv(OUTPUT_ROOT / "evaluation" / "policy_comparison_expert_bc_finetune_ppo.csv")
        make_figures(summary, comparison)
        report_and_ppt(config, summary, comparison)
        return
    shutil.copy2(PROJECT_ROOT / config["primary_prior"]["action_table"], OUTPUT_ROOT / "prior_policies" / Path(config["primary_prior"]["action_table"]).name)
    shutil.copy2(PROJECT_ROOT / config["backup_prior"]["action_table"], OUTPUT_ROOT / "prior_policies" / Path(config["backup_prior"]["action_table"]).name)
    write_prior_selection(config)
    smoke_rows: list[dict] = []
    eval_rows: list[dict] = []
    train_rows: list[dict] = []
    for method in config["finetune_methods"]:
        for station, site in config["sites"].items():
            train_year = int(site["train_year"])
            print(f"[constrained-ft] smoke {method['name']} {station} {train_year}", flush=True)
            smoke_rows.extend(run_null_or_fixed_smoke(config, station, train_year, method["name"]))
            smoke_rows.append(run_prior_smoke(config, station, train_year, method))
            latest_smoke = smoke_rows[-3:]
            if not all(row.get("run_status") == "ok" and bool(row.get("episode_completed")) for row in latest_smoke):
                train_rows.append({"station": station, "train_year": train_year, "finetune_method": method["name"], "run_status": "skipped", "notes": "pretrain_smoke_failed"})
                continue
            model = None
            model_path = None
            if bool(method.get("train", False)):
                print(f"[constrained-ft] train {method['name']} {station} {train_year}", flush=True)
                try:
                    model, model_path = run_training(config, method, station, train_year)
                    train_rows.append({"station": station, "train_year": train_year, "finetune_method": method["name"], "run_status": "ok", "model_path": str(model_path.relative_to(PROJECT_ROOT))})
                except Exception as exc:
                    train_rows.append({"station": station, "train_year": train_year, "finetune_method": method["name"], "run_status": "failed", "error_message": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-2000:]})
                    continue
            for eval_year in site["eval_years"]:
                print(f"[constrained-ft] eval {method['name']} {station} {eval_year}", flush=True)
                try:
                    eval_rows.append(evaluate_policy(config, method, station, train_year, int(eval_year), model, model_path))
                except Exception as exc:
                    eval_rows.append({"station": station, "finetune_method": method["name"], "train_year": train_year, "eval_year": int(eval_year), "run_status": "failed", "episode_completed": False, "error_message": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-2000:]})

    smoke = pd.DataFrame(smoke_rows)
    smoke.to_csv(OUTPUT_ROOT / "evaluation" / "pretrain_smoke_check_summary.csv", index=False, encoding="utf-8-sig")
    train = pd.DataFrame(train_rows)
    train.to_csv(OUTPUT_ROOT / "evaluation" / "constrained_ppo_training_summary.csv", index=False, encoding="utf-8-sig")
    summary = add_reference_deltas(config, pd.DataFrame(eval_rows))
    summary.to_csv(OUTPUT_ROOT / "evaluation" / "constrained_ppo_evaluation_summary.csv", index=False, encoding="utf-8-sig")
    comparison = make_policy_comparison(config, summary)
    make_figures(summary, comparison)
    report_and_ppt(config, summary, comparison)


if __name__ == "__main__":
    main()
