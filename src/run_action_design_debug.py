from __future__ import annotations

import shutil
import traceback
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from action_design_wrappers import ScheduledActionDesignWrapper
from episode_profit_reward import EpisodeProfitRewardWrapper, PROFIT_REWARD_CANDIDATES
from ppo_action_safety import SafeActionWrapper
from ppo_evaluate import latest_observation_dict, make_env, normalized_action_dict
from ppo_experiment_plan import find_year
from ppo_plot_results import plot_episode
from ppo_safe_rendering import PROJECT_ROOT, load_yaml
from ppo_train import ppo_kwargs, run_pretrain_smoke_check


CONFIG_PATH = PROJECT_ROOT / "experiments" / "ppo_observed_years" / "config_action_design_debug.yaml"
OUTPUT_ROOT = PROJECT_ROOT / "Leave_One_experiments" / "action_design_debug"
DOC_MD = PROJECT_ROOT / "docs" / "2026-06-06_action_design_debug_report.md"
DOC_PPT = PROJECT_ROOT / "docs" / "2026-06-06_action_design_debug_report.pptx"


def ensure_dirs() -> None:
    for sub in [
        "configs",
        "wrappers",
        "smoke_checks",
        "rendered_inputs",
        "models/HLA",
        "logs/HLA",
        "tensorboard/HLA",
        "daily_outputs/HLA",
        "evaluation",
        "figures/summary",
        "reports",
    ]:
        (OUTPUT_ROOT / sub).mkdir(parents=True, exist_ok=True)


def df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    work = df.copy()
    for col in work.columns:
        work[col] = work[col].map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(work.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in work.astype(str).values.tolist()]
    return "\n".join([header, separator, *rows])


def design_by_name(config: dict, name: str) -> dict:
    for item in config["action_designs"]:
        if item["name"] == name:
            return dict(item)
    raise KeyError(name)


def row_metadata(config: dict, design: dict, station: str) -> dict:
    reward_version = config["reward"]["version"]
    return {
        "action_design": design["name"],
        "action_design_type": design["type"],
        "reward_version": reward_version,
        "reward_family": PROFIT_REWARD_CANDIDATES[reward_version].family,
        "cap_name": config["cap"]["cap_name"],
        "season_irrigation_cap": float(config["cap"]["season_irrigation_cap"]),
        "season_n_cap": float(config["cap"]["season_n_cap"]),
        "cv_type": config["training"]["cv_types"].get(station, ""),
    }


def wrap_env_with_action_design(env, config: dict, design: dict):
    if design["type"] == "baseline":
        return env
    windows = config.get("phenology_windows", {})
    return ScheduledActionDesignWrapper(
        env,
        decision_interval_days=design.get("decision_interval_days"),
        fertilization_windows=windows.get("fertilization_windows"),
        irrigation_windows=windows.get("irrigation_windows"),
        use_phenology_windows=design["type"] in {"phenology_window", "decision_interval_plus_window"},
    )


def make_action_design_env(config: dict, station: str, year: int, seed: int, run_tag: str, evaluation: bool, design: dict):
    base = make_env(config, station, year, seed, run_tag=run_tag, evaluation=evaluation, action_safety_enabled=False)
    designed = wrap_env_with_action_design(base, config, design)
    safe = SafeActionWrapper(designed, {**config.get("action_safety", {}), "enabled": True})
    return EpisodeProfitRewardWrapper(safe, config["reward"]["version"])


def design_result_from_env(env) -> Any:
    # EpisodeProfitRewardWrapper -> SafeActionWrapper -> optional ScheduledActionDesignWrapper -> base env.
    safe = getattr(env, "env", None)
    inner = getattr(safe, "env", None)
    return getattr(inner, "last_design_result", None)


def safety_result_from_env(env) -> Any:
    safe = getattr(env, "env", None)
    return getattr(safe, "last_safety_result", None)


def write_review_and_plan(config: dict) -> None:
    review = """# Daily continuous action design failure review

Generated at: 2026-06-06

## Diagnosis

The previous reward experiments show that daily continuous actions make cap saturation easy. The agent receives a chance to output irrigation and fertilizer every simulated day. Even when action safety clips daily amounts, intervals, windows, and seasonal caps, the learned behavior can become "keep asking for water and nitrogen until safety stops it".

Action safety is therefore acting as the real manager, not just a safety guard. It prevents physical over-application but also creates a simple attractor: repeatedly request positive actions and let the wrapper spend the entire seasonal cap.

Reward cost tuning and terminal profit reward did change reward scale, but did not change the action opportunity structure. As long as the agent can ask every day, saturation remains an easy policy.

Real field management is lower-frequency and stage-based. Irrigation and fertilization are normally decided in a few operational windows, not every day. Therefore this stage tests low-frequency and phenology-window action designs.

## Minimal-intrusion wrappers

- DecisionIntervalActionWrapper / ScheduledActionDesignWrapper: only allow PPO actions every N days.
- PhenologyWindowActionWrapper behavior: only allow irrigation and N in agronomic DAP windows.
- Wrapper order: GymDssatWrapper -> ActionDesignWrapper -> SafeActionWrapper -> EpisodeProfitRewardWrapper.
"""
    (OUTPUT_ROOT / "evaluation" / "daily_action_design_failure_review.md").write_text(review, encoding="utf-8")

    plan_rows = []
    for design in config["action_designs"]:
        plan_rows.append(
            {
                "action_design": design["name"],
                "type": design["type"],
                "decision_interval_days": design.get("decision_interval_days", ""),
                "fertilization_windows": str(config["phenology_windows"]["fertilization_windows"]) if "window" in design["type"] else "",
                "irrigation_windows": str(config["phenology_windows"]["irrigation_windows"]) if "window" in design["type"] else "",
                "reward_version": config["reward"]["version"],
                "cap": "300 mm irrigation / 450 kg ha-1 N",
            }
        )
    plan = pd.DataFrame(plan_rows)
    plan.to_csv(OUTPUT_ROOT / "configs" / "action_design_candidate_plan.csv", index=False, encoding="utf-8-sig")
    text = "# Action design candidate plan\n\n" + df_to_markdown(plan)
    text += "\n\nDesign C, explicit seasonal budget allocation, is documented as the next option if A/B wrappers still fail."
    (OUTPUT_ROOT / "evaluation" / "action_design_candidate_plan.md").write_text(text, encoding="utf-8")


def write_design_files(config: dict) -> None:
    shutil.copy2(CONFIG_PATH, OUTPUT_ROOT / "configs" / CONFIG_PATH.name)
    shutil.copy2(PROJECT_ROOT / "src" / "action_design_wrappers.py", OUTPUT_ROOT / "wrappers" / "action_design_wrappers.py")
    write_review_and_plan(config)


def train_action_design_policy(config: dict, design: dict, station: str, train_year: int):
    from stable_baselines3 import PPO

    seed = int(config["seed"])
    meta = row_metadata(config, design, station)
    smoke_ok, smoke_summary = run_pretrain_smoke_check(config, station, train_year, seed, row_metadata=meta)
    if not smoke_ok:
        raise RuntimeError(f"pretrain smoke check failed: {smoke_summary}")

    policy_name = f"{station}_train{train_year}_{design['name']}_seed{seed}_action_design_profit"
    env = make_action_design_env(config, station, train_year, seed, f"{policy_name}_train", False, design)
    model_dir = OUTPUT_ROOT / "models" / station
    tensorboard_dir = OUTPUT_ROOT / "tensorboard" / station
    model_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / policy_name
    try:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=seed,
            tensorboard_log=str(tensorboard_dir),
            **ppo_kwargs(config, debug=False),
        )
        model.learn(total_timesteps=int(config["training"]["total_timesteps"]), progress_bar=False)
        model.save(str(model_path))
    finally:
        try:
            env.close()
        except Exception:
            pass
    return model, model_path.with_suffix(".zip"), policy_name


def plot_action_design_episode(daily: pd.DataFrame, figure_dir: Path) -> None:
    plot_episode(daily, figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    x = pd.to_numeric(daily["dap"], errors="coerce")

    if {
        "raw_real_action_amir",
        "design_filtered_action_amir",
        "safe_real_action_amir",
        "raw_real_action_anfer",
        "design_filtered_action_anfer",
        "safe_real_action_anfer",
    }.issubset(daily.columns):
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.0), sharex=True)
        axes[0].plot(x, daily["raw_real_action_amir"], label="raw")
        axes[0].plot(x, daily["design_filtered_action_amir"], label="filtered")
        axes[0].plot(x, daily["safe_real_action_amir"], label="safe")
        axes[0].set_ylabel("Irrigation")
        axes[0].grid(alpha=0.25)
        axes[0].legend(frameon=False)
        axes[1].plot(x, daily["raw_real_action_anfer"], label="raw")
        axes[1].plot(x, daily["design_filtered_action_anfer"], label="filtered")
        axes[1].plot(x, daily["safe_real_action_anfer"], label="safe")
        axes[1].set_ylabel("N fertilizer")
        axes[1].set_xlabel("DAP")
        axes[1].grid(alpha=0.25)
        axes[1].legend(frameon=False)
        fig.suptitle("Raw, design-filtered, and safe actions")
        fig.tight_layout()
        fig.savefig(figure_dir / "daily_actions_raw_filtered_safe.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for col in ["is_decision_day", "is_in_irrigation_window", "is_in_fertilization_window"]:
        if col in daily.columns:
            ax.plot(x, pd.to_numeric(daily[col], errors="coerce"), label=col)
    if "action_design_rule_triggered" in daily.columns:
        triggered = daily["action_design_rule_triggered"].fillna("").astype(str).str.len().gt(0).astype(int)
        ax.plot(x, triggered, label="any_design_filter", alpha=0.7)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Action design triggers")
    ax.set_xlabel("DAP")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "action_design_triggers.png", dpi=160)
    plt.close(fig)


def evaluate_action_design_model(
    model,
    config: dict,
    design: dict,
    station: str,
    train_year: int,
    eval_year: int,
    model_path: Path,
    policy_name: str,
) -> dict:
    seed = int(config["seed"])
    train_info = find_year(config, station, train_year)
    eval_info = find_year(config, station, eval_year)
    env = make_action_design_env(config, station, eval_year, seed, f"{policy_name}_eval{eval_year}", True, design)
    records: list[dict] = []
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        year_info = find_year(config, station, eval_year)
        while not done and step_count < int(config.get("runtime", {}).get("max_steps", 260)):
            latest = latest_observation_dict(env, obs, info)
            dap = int(round(float(latest.get("dap", step_count) or step_count)))
            raw_action, _ = model.predict(obs, deterministic=True)
            raw_action = np.clip(np.asarray(raw_action, dtype=np.float32).flatten(), -1.0, 1.0)
            obs, reward, terminated, truncated, info = env.step(raw_action)
            done = bool(terminated or truncated)
            latest = latest_observation_dict(env, obs, info)
            design_result = design_result_from_env(env)
            safety_result = safety_result_from_env(env)
            if design_result is None:
                raw_real = safety_result.raw_real_action if safety_result is not None else {"amir": np.nan, "anfer": np.nan}
                filtered_real = dict(raw_real)
                design_flags = {
                    "action_design_rule_triggered": "",
                    "is_decision_day": True,
                    "is_in_fertilization_window": True,
                    "is_in_irrigation_window": True,
                }
            else:
                raw_real = design_result.raw_real_action
                filtered_real = design_result.design_filtered_action
                design_flags = {
                    "action_design_rule_triggered": design_result.action_design_rule_triggered,
                    "is_decision_day": design_result.is_decision_day,
                    "is_in_fertilization_window": design_result.is_in_fertilization_window,
                    "is_in_irrigation_window": design_result.is_in_irrigation_window,
                }
            if safety_result is None:
                safe_real = {"amir": np.nan, "anfer": np.nan}
                safety_flags = {}
            else:
                safe_real = safety_result.safe_real_action
                safety_flags = {
                    "action_clipped_amir": safety_result.action_clipped_amir,
                    "action_clipped_anfer": safety_result.action_clipped_anfer,
                    "season_irrigation_so_far": safety_result.season_irrigation_so_far,
                    "season_n_so_far": safety_result.season_n_so_far,
                    "safety_rule_triggered": safety_result.safety_rule_triggered,
                }
            norm_action = normalized_action_dict(env, raw_action)
            date = pd.Timestamp(year_info["planting_date"]) + pd.Timedelta(days=max(dap - 1, 0))
            totir_raw = float(latest.get("totir", np.nan)) if latest.get("totir", np.nan) is not None else np.nan
            tofer_raw = float(latest.get("tofer", np.nan)) if latest.get("tofer", np.nan) is not None else np.nan
            row = {
                "station": station,
                "policy_name": policy_name,
                "action_design": design["name"],
                "action_design_type": design["type"],
                "train_year": train_year,
                "train_year_label": train_info["label"],
                "eval_year": eval_year,
                "eval_year_label": eval_info["label"],
                "seed": seed,
                "date": date.strftime("%Y-%m-%d"),
                "doy": int(date.dayofyear),
                "dap": float(latest.get("dap", dap) or dap),
                "topwt": float(latest.get("topwt", np.nan)) if latest.get("topwt", np.nan) is not None else np.nan,
                "grnwt": float(latest.get("grnwt", np.nan)) if latest.get("grnwt", np.nan) is not None else np.nan,
                "xlai": float(latest.get("xlai", np.nan)) if latest.get("xlai", np.nan) is not None else np.nan,
                "totir_raw": totir_raw,
                "totir": env.total_irrigation if np.isnan(totir_raw) else totir_raw,
                "tofer_raw": tofer_raw,
                "tofer": env.total_n if np.isnan(tofer_raw) else tofer_raw,
                "swfac": float(latest.get("swfac", np.nan)) if latest.get("swfac", np.nan) is not None else np.nan,
                "nstres": float(latest.get("nstres", np.nan)) if latest.get("nstres", np.nan) is not None else np.nan,
                "reward": float(reward),
                "real_action_amir": float(safe_real.get("amir", np.nan)),
                "real_action_anfer": float(safe_real.get("anfer", np.nan)),
                "normalized_action_amir": norm_action.get("amir", np.nan),
                "normalized_action_anfer": norm_action.get("anfer", np.nan),
                "raw_real_action_amir": float(raw_real.get("amir", np.nan)),
                "raw_real_action_anfer": float(raw_real.get("anfer", np.nan)),
                "design_filtered_action_amir": float(filtered_real.get("amir", np.nan)),
                "design_filtered_action_anfer": float(filtered_real.get("anfer", np.nan)),
                "safe_real_action_amir": float(safe_real.get("amir", np.nan)),
                "safe_real_action_anfer": float(safe_real.get("anfer", np.nan)),
                "done": done,
            }
            row.update(design_flags)
            row.update(safety_flags)
            row.update(env.last_profit_info)
            row.update(row_metadata(config, design, station))
            records.append(row)
            step_count += 1
    finally:
        try:
            env.close()
        except Exception:
            pass

    daily = pd.DataFrame(records)
    daily_dir = OUTPUT_ROOT / "daily_outputs" / station
    daily_dir.mkdir(parents=True, exist_ok=True)
    daily_csv = daily_dir / f"{station}_train{train_year}_{design['name']}_eval{eval_year}_seed{seed}_daily.csv"
    daily.to_csv(daily_csv, index=False, encoding="utf-8-sig")
    fig_dir = OUTPUT_ROOT / "figures" / station / design["name"] / f"eval_{eval_year}"
    plot_action_design_episode(daily, fig_dir)

    episode_completed = bool(records and records[-1]["done"])
    result = {
        "station": station,
        "action_design": design["name"],
        "action_design_type": design["type"],
        "reward_version": config["reward"]["version"],
        "policy_name": policy_name,
        "train_year": train_year,
        "eval_year": eval_year,
        "seed": seed,
        "season_irrigation_cap": float(config["cap"]["season_irrigation_cap"]),
        "season_n_cap": float(config["cap"]["season_n_cap"]),
        "run_status": "ok" if episode_completed else "failed",
        "episode_completed": episode_completed,
        "episode_length": len(daily),
        "final_grnwt": float(daily["grnwt"].iloc[-1]) if len(daily) else np.nan,
        "final_topwt": float(daily["topwt"].iloc[-1]) if len(daily) else np.nan,
        "total_irrigation": float(daily["real_action_amir"].sum()) if len(daily) else 0.0,
        "total_n_fertilizer": float(daily["real_action_anfer"].sum()) if len(daily) else 0.0,
        "mean_swfac": float(pd.to_numeric(daily["swfac"], errors="coerce").mean()) if len(daily) else np.nan,
        "mean_nstres": float(pd.to_numeric(daily["nstres"], errors="coerce").mean()) if len(daily) else np.nan,
        "mean_reward": float(daily["reward"].mean()) if len(daily) else np.nan,
        "sum_reward": float(daily["reward"].sum()) if len(daily) else 0.0,
        "num_decision_days": int(pd.to_numeric(daily.get("is_decision_day", pd.Series(dtype=float)), errors="coerce").sum()) if len(daily) else 0,
        "num_irrigation_allowed_days": int(pd.to_numeric(daily.get("is_in_irrigation_window", pd.Series(dtype=float)), errors="coerce").sum()) if len(daily) else 0,
        "num_fertilization_allowed_days": int(pd.to_numeric(daily.get("is_in_fertilization_window", pd.Series(dtype=float)), errors="coerce").sum()) if len(daily) else 0,
        "num_design_filtered_days": int(daily.get("action_design_rule_triggered", pd.Series(dtype=str)).fillna("").astype(str).str.len().gt(0).sum()) if len(daily) else 0,
        "num_safety_trigger_days": int(daily.get("safety_rule_triggered", pd.Series(dtype=str)).fillna("").astype(str).str.len().gt(0).sum()) if len(daily) else 0,
        "daily_csv_path": str(daily_csv.relative_to(PROJECT_ROOT)),
        "figure_dir": str(fig_dir.relative_to(PROJECT_ROOT)),
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "notes": "action_design_profit_eval",
    }
    result["irrigation_saturation_ratio"] = result["total_irrigation"] / result["season_irrigation_cap"]
    result["n_saturation_ratio"] = result["total_n_fertilizer"] / result["season_n_cap"]
    return result


def append_summary(rows: list[dict]) -> pd.DataFrame:
    out = OUTPUT_ROOT / "evaluation" / "action_design_evaluation_summary.csv"
    new = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_csv(out)
        df = pd.concat([old, new], ignore_index=True)
        df = df.drop_duplicates(["station", "action_design", "train_year", "eval_year", "seed"], keep="last")
    else:
        df = new
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return df


def compare_designs(df: pd.DataFrame) -> pd.DataFrame:
    hla = df[df["station"] == "HLA"].copy()
    hla["quality_gate_pass"] = (hla["run_status"] == "ok") & hla["episode_completed"].astype(str).str.lower().isin(["true", "1"])
    grouped = (
        hla.groupby(["action_design", "action_design_type"], dropna=False)
        .agg(
            eval_count=("eval_year", "count"),
            ok_count=("quality_gate_pass", "sum"),
            mean_yield=("final_grnwt", "mean"),
            std_yield=("final_grnwt", "std"),
            mean_reward=("mean_reward", "mean"),
            mean_irrigation=("total_irrigation", "mean"),
            mean_n=("total_n_fertilizer", "mean"),
            mean_irrigation_saturation_ratio=("irrigation_saturation_ratio", "mean"),
            mean_n_saturation_ratio=("n_saturation_ratio", "mean"),
            mean_swfac=("mean_swfac", "mean"),
            mean_nstres=("mean_nstres", "mean"),
            mean_design_filtered_days=("num_design_filtered_days", "mean"),
            mean_safety_trigger_days=("num_safety_trigger_days", "mean"),
            mean_decision_days=("num_decision_days", "mean"),
        )
        .reset_index()
    )
    base = grouped[grouped["action_design"] == "baseline_daily_action_current"]
    base_yield = float(base.iloc[0]["mean_yield"]) if len(base) else float(grouped["mean_yield"].max())
    grouped["yield_loss_vs_baseline"] = (base_yield - grouped["mean_yield"]) / base_yield
    grouped["input_reduction_vs_baseline"] = 1.0 - (grouped["mean_irrigation"] + grouped["mean_n"]) / (300.0 + 450.0)
    grouped["strict_pass"] = (
        (grouped["mean_irrigation_saturation_ratio"] < 0.95)
        & (grouped["mean_n_saturation_ratio"] < 0.95)
        & (grouped["yield_loss_vs_baseline"] <= 0.15)
        & (grouped["ok_count"] == grouped["eval_count"])
    )
    grouped["relaxed_pass"] = (
        (
            (grouped["mean_irrigation_saturation_ratio"] < 0.95)
            | (grouped["mean_n_saturation_ratio"] < 0.95)
        )
        & (grouped["yield_loss_vs_baseline"] <= 0.15)
        & (grouped["ok_count"] == grouped["eval_count"])
    )
    grouped["action_too_restrictive"] = (
        (grouped["mean_irrigation"] < 1.0)
        & (grouped["mean_n"] < 1.0)
        & (grouped["yield_loss_vs_baseline"] > 0.15)
    )
    out = OUTPUT_ROOT / "evaluation" / "action_design_candidate_comparison.csv"
    grouped.to_csv(out, index=False, encoding="utf-8-sig")
    return grouped


def select_design(comparison: pd.DataFrame) -> str | None:
    passed = comparison[comparison["strict_pass"]].copy()
    if passed.empty:
        passed = comparison[comparison["relaxed_pass"]].copy()
    if passed.empty:
        return None
    passed = passed.sort_values(["yield_loss_vs_baseline", "input_reduction_vs_baseline"], ascending=[True, False])
    return str(passed.iloc[0]["action_design"])


def plot_summary(comparison: pd.DataFrame) -> None:
    fig_root = OUTPUT_ROOT / "figures" / "summary"
    fig_root.mkdir(parents=True, exist_ok=True)
    labels = comparison["action_design"]

    plt.figure(figsize=(11, 6))
    plt.scatter(comparison["mean_irrigation"] + comparison["mean_n"], comparison["mean_yield"])
    for _, row in comparison.iterrows():
        plt.annotate(row["action_design"].replace("_", "\n"), (row["mean_irrigation"] + row["mean_n"], row["mean_yield"]), fontsize=7)
    plt.xlabel("Mean water + nitrogen input")
    plt.ylabel("Mean grain yield")
    plt.tight_layout()
    plt.savefig(fig_root / "action_design_yield_vs_input.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(labels, comparison["mean_irrigation_saturation_ratio"], marker="o", label="irrigation ratio")
    plt.plot(labels, comparison["mean_n_saturation_ratio"], marker="o", label="nitrogen ratio")
    plt.axhline(0.95, color="black", linestyle="--", linewidth=1)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Saturation ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_root / "action_design_saturation_ratio.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.scatter(comparison["input_reduction_vs_baseline"], comparison["yield_loss_vs_baseline"])
    for _, row in comparison.iterrows():
        plt.annotate(row["action_design"].replace("_", "\n"), (row["input_reduction_vs_baseline"], row["yield_loss_vs_baseline"]), fontsize=7)
    plt.axhline(0.15, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Input reduction vs baseline")
    plt.ylabel("Yield loss vs baseline")
    plt.tight_layout()
    plt.savefig(fig_root / "action_design_yield_loss_vs_input_reduction.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.bar(labels, comparison["mean_design_filtered_days"], label="design filtered days")
    plt.bar(labels, comparison["mean_safety_trigger_days"], bottom=comparison["mean_design_filtered_days"], label="safety trigger days", alpha=0.8)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean days")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_root / "action_design_filtered_days.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(labels, comparison["mean_swfac"], marker="o", label="swfac")
    plt.plot(labels, comparison["mean_nstres"], marker="o", label="nstres")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean stress indicators")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_root / "action_design_swfac_nstres.png", dpi=180)
    plt.close()


def run_design(config: dict, design: dict, station: str, train_year: int, eval_years: list[int]) -> list[dict]:
    print(f"[action-design] train {station} {train_year} {design['name']}", flush=True)
    model, model_path, policy_name = train_action_design_policy(config, design, station, train_year)
    rows = []
    for eval_year in eval_years:
        print(f"[action-design] evaluate {station} {design['name']} eval {eval_year}", flush=True)
        rows.append(evaluate_action_design_model(model, config, design, station, train_year, int(eval_year), model_path, policy_name))
    return rows


def run_hla_pilot(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    all_rows: list[dict] = []
    for design in config["action_designs"]:
        try:
            rows = run_design(
                config,
                design,
                config["training"]["hla_pilot_station"],
                int(config["training"]["hla_train_year"]),
                [int(y) for y in config["training"]["hla_eval_years"]],
            )
            all_rows.extend(rows)
            summary = append_summary(all_rows)
            comparison = compare_designs(summary)
            plot_summary(comparison)
        except Exception as exc:
            error_row = {
                "station": config["training"]["hla_pilot_station"],
                "action_design": design["name"],
                "action_design_type": design["type"],
                "reward_version": config["reward"]["version"],
                "train_year": int(config["training"]["hla_train_year"]),
                "eval_year": np.nan,
                "seed": int(config["seed"]),
                "run_status": "failed",
                "episode_completed": False,
                "error_message": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-2000:],
            }
            all_rows.append(error_row)
            append_summary(all_rows)
            raise
    summary = append_summary(all_rows)
    comparison = compare_designs(summary)
    plot_summary(comparison)
    return summary, comparison, select_design(comparison)


def run_extension(config: dict, selected: str | None) -> pd.DataFrame:
    out = OUTPUT_ROOT / "evaluation" / "cross_site_action_design_test.csv"
    if not selected:
        empty = pd.DataFrame([{"selected_action_design": "", "run_status": "skipped", "notes": "No HLA action design passed strict or relaxed criteria."}])
        empty.to_csv(out, index=False, encoding="utf-8-sig")
        write_blocker_report()
        return empty
    design = design_by_name(config, selected)
    all_rows: list[dict] = []
    for station in ["SYA", "LCA"]:
        train_year = int(config["training"]["best_train_years"][station])
        eval_years = [int(item["year"]) for item in config["observed_years"][station]]
        all_rows.extend(run_design(config, design, station, train_year, eval_years))
    cross = pd.DataFrame(all_rows)
    cross.to_csv(out, index=False, encoding="utf-8-sig")
    return cross


def write_blocker_report() -> None:
    text = """# Action design blocker report

Generated at: 2026-06-06

Design A/B wrappers did not identify a strict or relaxed passing HLA pilot candidate. The next action-design step should not be another coefficient scan. Recommended directions:

1. Explicit seasonal budget action: PPO first decides total water/N budget, then allocates within fixed windows.
2. Scheduled discrete events: actions are event amounts at basal, jointing, pre-tasseling, silking/grain-filling windows.
3. Rule-based expert policy plus imitation learning: pretrain management timing from agronomic rules before RL fine tuning.
4. Hybrid PPO over management windows: reduce episode action count to a few stage decisions.
"""
    (OUTPUT_ROOT / "evaluation" / "action_design_blocker_report.md").write_text(text, encoding="utf-8")


def write_report(comparison: pd.DataFrame, cross: pd.DataFrame, selected: str | None) -> None:
    report_table = comparison.copy()
    for col in report_table.select_dtypes(include=["float", "int"]).columns:
        report_table[col] = report_table[col].round(4)
    cross_table = cross.copy()
    if len(cross_table):
        for col in cross_table.select_dtypes(include=["float", "int"]).columns:
            cross_table[col] = cross_table[col].round(4)
    unsaturated = report_table[
        (report_table["mean_irrigation_saturation_ratio"] < 0.95)
        & (report_table["mean_n_saturation_ratio"] < 0.95)
    ]["action_design"].tolist()
    yield_ok = report_table[report_table["yield_loss_vs_baseline"] <= 0.15]["action_design"].tolist()
    md = f"""# Action design debug report

Generated at: 2026-06-06

## Goal

This stage revises action design after reward/cost/terminal-profit tuning all failed to avoid 300/450 cap saturation.

## Implemented wrappers

- `ScheduledActionDesignWrapper`: shared implementation for decision interval and window-gated action filtering.
- `DecisionIntervalActionWrapper`: low-frequency decision wrapper.
- `PhenologyWindowActionWrapper`: phenology window wrapper.

## Wrapper order

`GymDssatWrapper -> ActionDesignWrapper -> SafeActionWrapper -> EpisodeProfitRewardWrapper`

This means PPO raw actions are filtered by the action design first, action safety remains the final physical guard, and terminal profit reward is computed after the environment step.

## HLA comparison

{df_to_markdown(report_table)}

## Pass summary

- Designs with both irrigation and N unsaturated: {', '.join(unsaturated) if unsaturated else 'None'}
- Designs with yield loss <= 15%: {', '.join(yield_ok) if yield_ok else 'None'}
- Recommended action design: `{selected or 'None'}`

## SYA/LCA extension

{df_to_markdown(cross_table) if len(cross_table) else 'Extension was skipped because HLA did not identify a strict or relaxed passing action design.'}

## Interpretation

If A/B wrappers are still saturated, the next step should be explicit seasonal budget action or scheduled discrete event actions. If A/B wrappers reduce input but collapse yield, the windows or stage budgets need agronomic calibration before multi-seed.

## Next step

Proceed to multi-seed only if the selected HLA action design is unsaturated and has yield loss <= 15%. Proceed to rainfall-scaling only after action behavior is interpretable in observed-year HLA.
"""
    DOC_MD.parent.mkdir(parents=True, exist_ok=True)
    DOC_MD.write_text(md, encoding="utf-8")
    shutil.copy2(DOC_MD, OUTPUT_ROOT / "reports" / DOC_MD.name)
    write_ppt(comparison, selected)


def write_ppt(comparison: pd.DataFrame, selected: str | None) -> None:
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt
    except ModuleNotFoundError:
        (OUTPUT_ROOT / "reports" / "ppt_generation_skipped_in_container.txt").write_text(
            "python-pptx is not installed in the DSSAT container. Generate the PPT from Windows Python after training.",
            encoding="utf-8",
        )
        return
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    def add_title(slide, title: str) -> None:
        box = slide.shapes.add_textbox(Inches(0.5), Inches(0.25), Inches(12.3), Inches(0.5))
        p = box.text_frame.paragraphs[0]
        p.text = title
        p.font.name = "Microsoft YaHei"
        p.font.size = Pt(24)
        p.font.bold = True

    def add_body(slide, text: str) -> None:
        box = slide.shapes.add_textbox(Inches(0.7), Inches(1.0), Inches(12.0), Inches(5.8))
        tf = box.text_frame
        tf.word_wrap = True
        for i, line in enumerate(text.split("\n")):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = line
            p.font.name = "Microsoft YaHei"
            p.font.size = Pt(15)

    def picture_slide(title: str, file_name: str) -> None:
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        add_title(slide, title)
        path = OUTPUT_ROOT / "figures" / "summary" / file_name
        if path.exists():
            slide.shapes.add_picture(str(path), Inches(0.8), Inches(1.0), width=Inches(11.8))

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "006_06 Action Design Debug")
    add_body(slide, "目标：从每日连续动作转向低频/窗口化管理动作。\n约束：action-safe PPO；HLA 2011 pilot；不改原始 reward，不覆盖 006_03/04/05。")

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "Why Change Action Design")
    add_body(slide, "每日连续动作让 PPO 每天都能请求水氮，action safety 最终会把持续正动作截断到季节 cap。\n这会让策略学成“每天要，直到 cap 用满”。")

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "Implemented Designs")
    add_body(slide, "Baseline：每日连续动作。\nA7/A10/A15：每 7/10/15 天决策一次。\nB：只允许物候窗口内水氮动作。\nA10+B：10 天决策与物候窗口叠加。")

    picture_slide("Yield vs Input", "action_design_yield_vs_input.png")
    picture_slide("Saturation Ratio", "action_design_saturation_ratio.png")
    picture_slide("Yield Loss vs Input Reduction", "action_design_yield_loss_vs_input_reduction.png")
    picture_slide("Filtered and Safety Days", "action_design_filtered_days.png")
    picture_slide("SWFAC and NSTRES", "action_design_swfac_nstres.png")

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "Conclusion")
    add_body(slide, f"推荐 action design：{selected or 'None'}。\n如果 A/B 仍失败，下一步应转向 explicit seasonal budget action 或 scheduled discrete event actions。")

    DOC_PPT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(DOC_PPT)
    shutil.copy2(DOC_PPT, OUTPUT_ROOT / "reports" / DOC_PPT.name)


def main() -> None:
    ensure_dirs()
    config = load_yaml(CONFIG_PATH)
    write_design_files(config)
    summary, comparison, selected = run_hla_pilot(config)
    cross = run_extension(config, selected)
    write_report(comparison, cross, selected)
    print("[action-design] done")
    print((OUTPUT_ROOT / "evaluation" / "action_design_candidate_comparison.csv").relative_to(PROJECT_ROOT))
    print(f"[action-design] selected={selected}")


if __name__ == "__main__":
    main()
