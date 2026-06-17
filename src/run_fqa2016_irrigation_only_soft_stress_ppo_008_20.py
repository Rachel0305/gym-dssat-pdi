from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_all_year_direct_action_safe_ppo as direct
import run_hla2004_stress_aware_stage_ppo_008_07 as stage_runner
import run_management_scenario_comparison as msc
from ppo_evaluate import latest_observation_dict, scalar
from ppo_safe_rendering import PROJECT_ROOT
from soft_stress_gate_wrapper_008_14 import SoftStressForecastGateDapStageActionWrapper
from stress_aware_stage_action_wrapper import gate_blocked_irrigation_amount


CONFIG_PATH = PROJECT_ROOT / "experiments" / "ppo_observed_years" / "config_008_20_fqa2016_irrigation_only_soft_stress_ppo_validation.yaml"


def root(config: dict[str, Any]) -> Path:
    return PROJECT_ROOT / config["paths"]["output_root"]


def doc_md(config: dict[str, Any]) -> Path:
    return PROJECT_ROOT / config["paths"]["doc_md"]


def ensure_dirs(config: dict[str, Any], station: str) -> None:
    for sub in [
        f"configs",
        f"models/{station}",
        f"tensorboard/{station}",
        f"daily_outputs/{station}",
        "evaluation",
        "figures",
        "reports",
        "rendered_inputs",
        "logs",
    ]:
        (root(config) / sub).mkdir(parents=True, exist_ok=True)
    doc_md(config).parent.mkdir(parents=True, exist_ok=True)


def station_year(config: dict[str, Any]) -> tuple[str, int]:
    item = config["station_years"][0]
    return str(item["station"]), int(item["year"])


def select_station_year(config: dict[str, Any], station: str, year: int) -> pd.DataFrame:
    pool = pd.read_csv(PROJECT_ROOT / config["paths"]["scenario_pool_csv"])
    row = pool[
        (pool["station_code"].astype(str).eq(station))
        & (pd.to_numeric(pool["year"], errors="coerce").eq(year))
    ]
    if row.empty:
        raise KeyError(f"{station} {year} is missing from all-year scenario pool")
    return row.copy()


def build_env_config(config: dict[str, Any], station: str, year: int) -> dict[str, Any]:
    work = dict(config)
    work["paths"] = dict(config["paths"])
    work["runtime"] = dict(config["runtime"])
    work["runtime"]["mode"] = config["runtime"].get("mode", "all")
    work["runtime"]["max_steps"] = int(config["runtime"].get("max_steps", 260))
    work["seed"] = int(config.get("seed", 0))
    work["action_safety"] = dict(config.get("action_safety", {"enabled": False}))
    env_config = direct.build_env_config(work, select_station_year(config, station, year))
    if config.get("forecast_stress_gate", {}).get("enabled"):
        year_info = direct.find_year(env_config, station, year)
        config["forecast_stress_gate"]["planting_date"] = year_info["planting_date"]
    return env_config


def make_env(config: dict[str, Any], env_config: dict[str, Any], station: str, year: int, seed: int, tag: str, evaluation: bool):
    base = direct.make_base_env(env_config, station, year, seed, tag, evaluation=evaluation)
    return SoftStressForecastGateDapStageActionWrapper(base, config)


def model_path(config: dict[str, Any], station: str, year: int, seed: int) -> Path:
    return root(config) / "models" / station / f"ppo_soft_stress_{station}_{year}_seed{seed}.zip"


def ppo_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    allowed = ["learning_rate", "gamma", "n_steps", "batch_size", "n_epochs", "ent_coef", "clip_range"]
    return {k: config["ppo"][k] for k in allowed if k in config.get("ppo", {})}


def train_one(config: dict[str, Any], env_config: dict[str, Any], station: str, year: int, seed: int) -> dict[str, Any]:
    from stable_baselines3 import PPO

    out = model_path(config, station, year, seed)
    env = None
    try:
        env = make_env(config, env_config, station, year, seed, f"{station}_{year}_00820_train_seed{seed}", evaluation=False)
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
            tensorboard_log=str(root(config) / "tensorboard" / station),
            **ppo_kwargs(config),
        )
        model.learn(total_timesteps=int(config["total_timesteps"]), progress_bar=False)
        model.save(str(out.with_suffix("")))
        return {
            "station": station,
            "year": year,
            "seed": seed,
            "total_timesteps": int(config["total_timesteps"]),
            "run_status": "ok",
            "model_path": str(out.relative_to(PROJECT_ROOT)),
            "notes": "",
        }
    except Exception:
        return {
            "station": station,
            "year": year,
            "seed": seed,
            "total_timesteps": int(config["total_timesteps"]),
            "run_status": "failed",
            "model_path": "",
            "notes": traceback.format_exc()[-5000:],
        }
    finally:
        if env is not None:
            env.close()


def add_weather(daily: pd.DataFrame, station: str, year: int) -> pd.DataFrame:
    if daily.empty:
        return daily
    weather = msc.read_weather(station, year)
    if weather.empty:
        for col in ["rain", "srad", "tmax", "tmin"]:
            daily[col] = np.nan
        return daily
    out = daily.copy()
    out["date_dt"] = pd.to_datetime(out["date"])
    out = out.merge(weather, left_on="date_dt", right_on="date", how="left", suffixes=("", "_weather"))
    return out.drop(columns=["date_dt", "date_weather"], errors="ignore")


def evaluate_one(config: dict[str, Any], env_config: dict[str, Any], station: str, year: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    from stable_baselines3 import PPO

    model = PPO.load(str(model_path(config, station, year, seed)))
    env = make_env(config, env_config, station, year, seed, f"{station}_{year}_00820_eval_seed{seed}", evaluation=True)
    daily_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        done = False
        stage_step = 0
        year_info = direct.find_year(env_config, station, year)
        planting = pd.Timestamp(year_info["planting_date"])
        while not done and stage_step < int(config["runtime"]["max_stage_steps"]):
            before = latest_observation_dict(env, obs, info)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            after = latest_observation_dict(env, obs, info)
            stage_info = dict(env.last_action_info)
            stage_step += 1
            stage_rows.append(
                {
                    "station": station,
                    "year": year,
                    "seed": seed,
                    "stage_step": stage_step,
                    "start_dap_observed": int(round(scalar(before.get("dap"), stage_step))),
                    "end_dap_observed": int(round(scalar(after.get("dap"), stage_step))),
                    "reward": float(reward),
                    "done": done,
                    **stage_info,
                }
            )
            for item in env.last_stage_records:
                dap = int(round(float(item["dap_before_step"])))
                date = planting + pd.Timedelta(days=max(dap - 1, 0))
                daily_rows.append(
                    {
                        "station": station,
                        "year": year,
                        "seed": seed,
                        "date": date.strftime("%Y-%m-%d"),
                        "doy": int(date.dayofyear),
                        "dap": dap,
                        "stage_step": stage_step,
                        **item,
                        "reward_stage": float(reward),
                        "done": done,
                    }
                )
    finally:
        env.close()

    daily = add_weather(pd.DataFrame(daily_rows), station, year)
    stages = pd.DataFrame(stage_rows)
    daily_path = root(config) / "daily_outputs" / station / f"{station}_{year}_seed{seed}_soft_stress_stage_daily.csv"
    stage_path = root(config) / "daily_outputs" / station / f"{station}_{year}_seed{seed}_soft_stress_stage_steps.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    stages.to_csv(stage_path, index=False, encoding="utf-8-sig")
    summary = summarize_run(config, station, year, seed, daily, stages, daily_path, stage_path)
    return daily, stages, summary


def summarize_run(
    config: dict[str, Any],
    station: str,
    year: int,
    seed: int,
    daily: pd.DataFrame,
    stages: pd.DataFrame,
    daily_path: Path,
    stage_path: Path,
) -> dict[str, Any]:
    stage_i = pd.to_numeric(stages.get("stage_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    stage_n = pd.to_numeric(stages.get("stage_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    grnwt = pd.to_numeric(daily.get("grnwt", pd.Series(dtype=float)), errors="coerce").dropna()
    topwt = pd.to_numeric(daily.get("topwt", pd.Series(dtype=float)), errors="coerce").dropna()
    swfac = pd.to_numeric(daily.get("swfac", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    nstres = pd.to_numeric(daily.get("nstres", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    rain = pd.to_numeric(daily.get("rain", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    total_i = float(stage_i.sum())
    total_n = float(stage_n.sum())
    final_grnwt = float(grnwt.iloc[-1]) if len(grnwt) else np.nan
    first_i = stages.loc[stage_i > 0, "decision_dap"].iloc[0] if len(stages) and (stage_i > 0).any() else np.nan
    return {
        "station": station,
        "year": year,
        "seed": seed,
        "run_status": "ok" if len(stages) and bool(stages["done"].iloc[-1]) else "failed",
        "stage_steps": int(len(stages)),
        "daily_steps": int(len(daily)),
        "total_irrigation": total_i,
        "total_n": total_n,
        "final_grnwt": final_grnwt,
        "final_topwt": float(topwt.iloc[-1]) if len(topwt) else np.nan,
        "profit_low_water_cost": float(config["economics"]["grain_value_coef"]) * final_grnwt
        - float(config["economics"]["water_cost_low"]) * total_i
        - float(config["economics"]["nitrogen_cost"]) * total_n,
        "max_swfac": float(swfac.max()) if len(swfac) else np.nan,
        "swfac_days_gt_0p05": int((swfac > 0.05).sum()) if len(swfac) else 0,
        "max_nstres": float(nstres.max()) if len(nstres) else np.nan,
        "nstres_days_gt_0p05": int((nstres > 0.05).sum()) if len(nstres) else 0,
        "rain_total_in_episode": float(rain.sum()) if len(rain) else np.nan,
        "first_irrigation_dap": float(first_i) if pd.notna(first_i) else np.nan,
        "irrigation_cap_saturated": total_i >= float(config["success_criteria"]["cap_saturation_threshold_irrigation"]),
        "n_cap_saturated": total_n >= float(config["success_criteria"]["cap_saturation_threshold_n"]),
        "early_irrigation_before_allowed_window": bool(pd.notna(first_i) and float(first_i) < float(config["success_criteria"]["target_first_irrigation_dap_min"])),
        "gate_blocked_irrigation": gate_blocked_irrigation_amount(stages),
        "daily_csv_path": str(daily_path.relative_to(PROJECT_ROOT)),
        "stage_csv_path": str(stage_path.relative_to(PROJECT_ROOT)),
        "model_path": str(model_path(config, station, year, seed).relative_to(PROJECT_ROOT)),
    }


def add_reference_context(config: dict[str, Any], summary: pd.DataFrame, station: str, year: int) -> pd.DataFrame:
    out = summary.copy()
    rule_path = PROJECT_ROOT / config["paths"]["forecast_rule_summary_csv"]
    if rule_path.exists() and not out.empty:
        rule = pd.read_csv(rule_path)
        hit = rule[(rule["station"].astype(str).eq(station)) & (pd.to_numeric(rule["year"], errors="coerce").eq(year))]
        if not hit.empty:
            r = hit.iloc[0]
            out["rule_replay_total_irrigation"] = float(r.get("total_irrigation", np.nan))
            out["rule_replay_total_n"] = float(r.get("total_n", np.nan))
            out["rule_replay_final_grnwt"] = float(r.get("final_grnwt", np.nan))
            out["rule_replay_swfac_days_gt_0p05"] = float(r.get("swfac_days_gt_0p05", np.nan))
            out["grnwt_fraction_vs_rule_replay"] = pd.to_numeric(out["final_grnwt"], errors="coerce") / float(r.get("final_grnwt", np.nan))
    pool_path = PROJECT_ROOT / config["paths"]["scenario_pool_csv"]
    if pool_path.exists() and not out.empty:
        pool = pd.read_csv(pool_path)
        hit = pool[(pool["station_code"].astype(str).eq(station)) & (pd.to_numeric(pool["year"], errors="coerce").eq(year))]
        if not hit.empty:
            p = hit.iloc[0]
            for col in [
                "scenario_type",
                "has_water_stress",
                "irrigation_responsive",
                "yield_gain_from_irrigation_at_same_N",
                "swfac_stress_days_gt_0p05",
                "max_swfac",
            ]:
                out[f"pool_{col}"] = p.get(col, np.nan)
    if "grnwt_fraction_vs_rule_replay" in out.columns:
        out["debug_promising"] = (
            (~out["irrigation_cap_saturated"].astype(bool))
            & (~out["early_irrigation_before_allowed_window"].astype(bool))
            & (pd.to_numeric(out["grnwt_fraction_vs_rule_replay"], errors="coerce") >= float(config["success_criteria"]["min_grnwt_fraction_vs_rule_replay"]))
            & (pd.to_numeric(out["total_irrigation"], errors="coerce") > 1.0)
        )
    return out


def plot_outputs(config: dict[str, Any], station: str, year: int, daily: pd.DataFrame, stages: pd.DataFrame) -> None:
    fig_dir = root(config) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if not stages.empty:
        labels = stages["stage_id"].astype(str)
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=180)
        ax.bar(x - 0.18, pd.to_numeric(stages["stage_action_amir"], errors="coerce"), 0.36, label="Irrigation mm", color="#0072B2")
        ax.bar(x + 0.18, pd.to_numeric(stages["stage_action_anfer"], errors="coerce"), 0.36, label="N kg/ha", color="#009E73")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Amount")
        ax.set_title(f"{station} {year} soft-stress PPO stage actions")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{station}_{year}_soft_stress_stage_actions.png")
        plt.close(fig)
    if not daily.empty:
        fig, axes = plt.subplots(5, 1, figsize=(10, 9.5), dpi=180, sharex=True)
        axes[0].bar(daily["dap"], pd.to_numeric(daily.get("rain", 0), errors="coerce").fillna(0), width=1.0, color="#4F81BD", alpha=0.35)
        axes[0].set_ylabel("Rain mm")
        axes[1].plot(daily["dap"], pd.to_numeric(daily["swfac"], errors="coerce"), label="SWFAC", color="#0072B2")
        axes[1].plot(daily["dap"], pd.to_numeric(daily["nstres"], errors="coerce"), label="NSTRES", color="#D55E00")
        axes[1].legend(frameon=False)
        axes[1].set_ylabel("Stress")
        axes[2].vlines(stages["decision_dap"], 0, pd.to_numeric(stages["stage_action_amir"], errors="coerce"), color="#0072B2", label="Irrigation")
        axes[2].vlines(stages["decision_dap"], 0, pd.to_numeric(stages["stage_action_anfer"], errors="coerce"), color="#009E73", linestyles="dashed", label="N")
        axes[2].legend(frameon=False)
        axes[2].set_ylabel("Actions")
        axes[3].plot(daily["dap"], pd.to_numeric(daily["topwt"], errors="coerce"), label="TOPWT", color="#009E73")
        axes[3].plot(daily["dap"], pd.to_numeric(daily["grnwt"], errors="coerce"), label="GRNWT", color="#CC79A7")
        axes[3].legend(frameon=False)
        axes[3].set_ylabel("Growth")
        axes[4].plot(daily["dap"], pd.to_numeric(daily["reward_stage"], errors="coerce"), color="#E69F00", label="Stage reward")
        axes[4].axhline(0, color="black", linestyle="--", linewidth=0.8)
        axes[4].legend(frameon=False)
        axes[4].set_ylabel("Reward")
        axes[4].set_xlabel("DAP")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{station}_{year}_soft_stress_process.png")
        plt.close(fig)


def df_md(df: pd.DataFrame, max_rows: int = 25) -> str:
    return direct.df_to_markdown(df, max_rows=max_rows) if not df.empty else ""


def write_report(config: dict[str, Any], train_summary: pd.DataFrame, eval_summary: pd.DataFrame, stages: pd.DataFrame) -> None:
    station, year = station_year(config)
    stage_keep = [
        "stage_id",
        "decision_dap",
        "raw_stage_action_irrigation",
        "stage_action_amir",
        "stage_action_anfer",
        "irrigation_before_gate",
        "irrigation_after_gate",
        "gate_future_rain",
        "gate_forecast_trigger",
        "gate_swfac_at_decision",
        "soft_swfac_penalty",
        "reward",
    ]
    stage_table = stages[[c for c in stage_keep if c in stages.columns]].copy() if not stages.empty else stages
    lines = [
        f"# {config.get('report_title', '008_20 FQA 2016 Soft-Stress PPO Validation')}",
        "",
        "## Purpose",
        "",
        "This lightweight run tests whether the HLA 2004 soft-stress stage PPO framework can produce an interpretable irrigation policy in a second water-stress year.",
        "Nitrogen is fixed at N150 by stage prior. PPO controls irrigation only. No hard minimum irrigation gate is used.",
        "",
        "## Training Summary",
        "",
        df_md(train_summary),
        "",
        "## Evaluation Summary",
        "",
        df_md(eval_summary),
        "",
        "## Stage Decisions",
        "",
        df_md(stage_table, max_rows=12),
        "",
        "## Interpretation",
        "",
    ]
    if not eval_summary.empty:
        r = eval_summary.iloc[0]
        lines.extend(
            [
                f"- Total irrigation: {float(r.get('total_irrigation', np.nan)):.2f} mm; total N: {float(r.get('total_n', np.nan)):.2f} kg/ha.",
                f"- Final GRNWT: {float(r.get('final_grnwt', np.nan)):.2f}.",
                f"- GRNWT / 008_11 rule replay: {float(r.get('grnwt_fraction_vs_rule_replay', np.nan)):.3f}.",
                f"- SWFAC stress days: {int(r.get('swfac_days_gt_0p05', 0))}; max SWFAC: {float(r.get('max_swfac', np.nan)):.3f}.",
                f"- First irrigation DAP: {r.get('first_irrigation_dap', np.nan)}.",
                f"- Debug promising: {bool(r.get('debug_promising', False))}.",
            ]
        )
        if float(r.get("total_irrigation", 0.0)) <= 1.0:
            lines.append("- PPO collapsed to zero irrigation in this second-year validation.")
        elif bool(r.get("irrigation_cap_saturated", False)):
            lines.append("- PPO saturated the irrigation cap; this should not be treated as a successful interpretable policy.")
        else:
            lines.append("- PPO produced a nonzero, non-saturated irrigation policy.")
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Training summary: `{(root(config) / 'evaluation' / f'{station}_{year}_soft_stress_ppo_training.csv').relative_to(PROJECT_ROOT)}`",
            f"- Evaluation summary: `{(root(config) / 'evaluation' / f'{station}_{year}_soft_stress_ppo_summary.csv').relative_to(PROJECT_ROOT)}`",
            f"- Daily outputs: `{(root(config) / 'daily_outputs' / station).relative_to(PROJECT_ROOT)}`",
            f"- Figures: `{(root(config) / 'figures').relative_to(PROJECT_ROOT)}`",
        ]
    )
    doc_md(config).write_text("\n".join(lines), encoding="utf-8")


def run(config_path: Path, report_only: bool = False) -> None:
    config = direct.load_yaml(config_path)
    station, year = station_year(config)
    ensure_dirs(config, station)
    direct.write_yaml(config, root(config) / "configs" / config_path.name)
    env_config = build_env_config(config, station, year)
    direct.write_yaml(env_config, root(config) / "configs" / "rendered_env_config.yaml")
    seed = int(config.get("seed", 0))
    train_path = root(config) / "evaluation" / f"{station}_{year}_soft_stress_ppo_training.csv"
    summary_path = root(config) / "evaluation" / f"{station}_{year}_soft_stress_ppo_summary.csv"
    stage_path = root(config) / "daily_outputs" / station / f"{station}_{year}_seed{seed}_soft_stress_stage_steps.csv"
    if report_only:
        train_summary = pd.read_csv(train_path)
        eval_summary = pd.read_csv(summary_path)
        stages = pd.read_csv(stage_path)
    else:
        train_summary = pd.DataFrame([train_one(config, env_config, station, year, seed)])
        train_summary.to_csv(train_path, index=False, encoding="utf-8-sig")
        eval_rows: list[dict[str, Any]] = []
        daily = pd.DataFrame()
        stages = pd.DataFrame()
        if train_summary["run_status"].iloc[0] == "ok":
            daily, stages, summary = evaluate_one(config, env_config, station, year, seed)
            eval_rows.append(summary)
            plot_outputs(config, station, year, daily, stages)
        eval_summary = add_reference_context(config, pd.DataFrame(eval_rows), station, year)
        eval_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    write_report(config, train_summary, eval_summary, stages)
    print(train_path)
    print(summary_path)
    print(doc_md(config))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    run(config_path, report_only=args.report_only)


if __name__ == "__main__":
    main()
