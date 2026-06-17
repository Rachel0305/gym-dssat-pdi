from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_fqa2016_irrigation_only_soft_stress_ppo_008_20 as base_runner
import run_all_year_direct_action_safe_ppo as direct
from joint_soft_stress_gate_wrapper_009 import JointSoftStressForecastGateDapStageActionWrapper
from ppo_safe_rendering import PROJECT_ROOT
from stress_aware_stage_action_wrapper import gate_blocked_irrigation_amount


CONFIG_PATH = PROJECT_ROOT / "experiments" / "ppo_observed_years" / "config_009_02_hla2004_stage_level_water_nitrogen_joint_ppo_smoke.yaml"
HLA_00815_SUMMARY = PROJECT_ROOT / "Leave_One_experiments" / "hla2004_irrigation_only_soft_stress_ppo_008_15_10k" / "evaluation" / "HLA_2004_stress_aware_stage_ppo_summary.csv"


def root(config: dict[str, Any]) -> Path:
    return PROJECT_ROOT / config["paths"]["output_root"]


def doc_md(config: dict[str, Any]) -> Path:
    return PROJECT_ROOT / config["paths"]["doc_md"]


def make_env(config: dict[str, Any], env_config: dict[str, Any], station: str, year: int, seed: int, tag: str, evaluation: bool):
    base = direct.make_base_env(env_config, station, year, seed, tag, evaluation=evaluation)
    return JointSoftStressForecastGateDapStageActionWrapper(base, config)


def model_path(config: dict[str, Any], station: str, year: int, seed: int) -> Path:
    return root(config) / "models" / station / f"ppo_joint_water_n_{station}_{year}_seed{seed}.zip"


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
    base_n = pd.to_numeric(stages.get("stage_nitrogen_base", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    extra_n = pd.to_numeric(stages.get("ppo_extra_n_applied", stages.get("stage_extra_anfer", pd.Series(dtype=float))), errors="coerce").fillna(0.0)
    sw_penalty = pd.to_numeric(stages.get("soft_swfac_penalty", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    n_penalty = pd.to_numeric(stages.get("soft_nstres_penalty", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    grnwt = pd.to_numeric(daily.get("grnwt", pd.Series(dtype=float)), errors="coerce").dropna()
    topwt = pd.to_numeric(daily.get("topwt", pd.Series(dtype=float)), errors="coerce").dropna()
    swfac = pd.to_numeric(daily.get("swfac", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    nstres = pd.to_numeric(daily.get("nstres", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    rain = pd.to_numeric(daily.get("rain", pd.Series(dtype=float)), errors="coerce").fillna(0.0)

    total_i = float(stage_i.sum())
    total_n = float(stage_n.sum())
    total_base_n = float(base_n.sum())
    total_extra_n = float(extra_n.sum())
    final_grnwt = float(grnwt.iloc[-1]) if len(grnwt) else np.nan
    first_i = stages.loc[stage_i > 0, "decision_dap"].iloc[0] if len(stages) and (stage_i > 0).any() else np.nan
    first_extra_n = stages.loc[extra_n > 1e-9, "decision_dap"].iloc[0] if len(stages) and (extra_n > 1e-9).any() else np.nan
    baseline_grnwt = np.nan
    if HLA_00815_SUMMARY.exists():
        baseline = pd.read_csv(HLA_00815_SUMMARY)
        if not baseline.empty:
            baseline_grnwt = float(pd.to_numeric(baseline["final_grnwt"], errors="coerce").iloc[0])

    nstres_days = int((nstres > 0.05).sum()) if len(nstres) else 0
    total_n_min = float(config["success_criteria"].get("min_total_n", 110.0))
    nstres_ok = nstres_days < int(config["success_criteria"].get("max_acceptable_nstres_days", 30))
    irrigation_saturated = total_i >= float(config["success_criteria"]["cap_saturation_threshold_irrigation"])
    n_saturated = total_n >= float(config["success_criteria"]["cap_saturation_threshold_n"])
    early_i = bool(pd.notna(first_i) and float(first_i) < float(config["success_criteria"]["target_first_irrigation_dap_min"]))
    yield_ok = bool(np.isfinite(final_grnwt) and (not np.isfinite(baseline_grnwt) or final_grnwt >= 0.70 * baseline_grnwt))
    debug_joint_promising = bool(
        total_i > 1.0
        and not irrigation_saturated
        and not n_saturated
        and not early_i
        and total_n >= total_n_min
        and (total_extra_n > 1e-9 or nstres_ok)
        and yield_ok
    )

    return {
        "station": station,
        "year": year,
        "seed": seed,
        "run_status": "ok" if len(stages) and bool(stages["done"].iloc[-1]) else "failed",
        "stage_steps": int(len(stages)),
        "daily_steps": int(len(daily)),
        "total_irrigation": total_i,
        "total_ppo_extra_irrigation": total_i,
        "total_n": total_n,
        "total_base_n": total_base_n,
        "total_ppo_extra_n": total_extra_n,
        "final_grnwt": final_grnwt,
        "final_topwt": float(topwt.iloc[-1]) if len(topwt) else np.nan,
        "grnwt_fraction_vs_00815_irrigation_only": final_grnwt / baseline_grnwt if np.isfinite(final_grnwt) and np.isfinite(baseline_grnwt) else np.nan,
        "profit_low_water_cost": float(config["economics"]["grain_value_coef"]) * final_grnwt
        - float(config["economics"]["water_cost_low"]) * total_i
        - float(config["economics"]["nitrogen_cost"]) * total_n,
        "max_swfac": float(swfac.max()) if len(swfac) else np.nan,
        "swfac_days_gt_0p05": int((swfac > 0.05).sum()) if len(swfac) else 0,
        "max_nstres": float(nstres.max()) if len(nstres) else np.nan,
        "nstres_days_gt_0p05": nstres_days,
        "soft_swfac_penalty_total": float(sw_penalty.sum()),
        "soft_nstres_penalty_total": float(n_penalty.sum()),
        "rain_total_in_episode": float(rain.sum()) if len(rain) else np.nan,
        "first_irrigation_dap": float(first_i) if pd.notna(first_i) else np.nan,
        "first_ppo_n_dap": float(first_extra_n) if pd.notna(first_extra_n) else np.nan,
        "irrigation_cap_saturated": irrigation_saturated,
        "n_cap_saturated": n_saturated,
        "early_irrigation_before_allowed_window": early_i,
        "gate_blocked_irrigation": gate_blocked_irrigation_amount(stages),
        "nstres_warning": bool(nstres_days >= int(config["success_criteria"].get("max_acceptable_nstres_days", 30))),
        "debug_joint_promising": debug_joint_promising,
        "daily_csv_path": str(daily_path.relative_to(PROJECT_ROOT)),
        "stage_csv_path": str(stage_path.relative_to(PROJECT_ROOT)),
        "model_path": str(model_path(config, station, year, seed).relative_to(PROJECT_ROOT)),
    }


def write_report(config: dict[str, Any], train_summary: pd.DataFrame, eval_summary: pd.DataFrame, stages: pd.DataFrame) -> None:
    station, year = base_runner.station_year(config)
    stage_keep = [
        "stage_id",
        "decision_dap",
        "raw_stage_action_irrigation",
        "raw_stage_action_nitrogen",
        "stage_action_amir",
        "stage_action_anfer",
        "stage_nitrogen_base",
        "ppo_extra_n_applied",
        "irrigation_before_gate",
        "irrigation_after_gate",
        "gate_future_rain",
        "gate_forecast_trigger",
        "gate_swfac_at_decision",
        "soft_swfac_penalty",
        "soft_nstres_penalty",
        "reward",
    ]
    stage_table = stages[[c for c in stage_keep if c in stages.columns]].copy() if not stages.empty else stages
    lines = [
        f"# {config.get('report_title', '009_02 HLA 2004 Stage-Level Water-Nitrogen Joint PPO Smoke Test')}",
        "",
        "## Purpose",
        "",
        "This smoke test reintroduces PPO-controlled supplemental nitrogen while preserving the 008 stage-level soft-stress irrigation framework.",
        "S1/S2 use diagnostic agronomic base nitrogen. S3-S5 allow PPO supplemental N. PPO also controls irrigation under the no-hard-minimum forecast/stress gate.",
        "",
        "## Training Summary",
        "",
        direct.df_to_markdown(train_summary),
        "",
        "## Evaluation Summary",
        "",
        direct.df_to_markdown(eval_summary),
        "",
        "## Stage Decisions",
        "",
        direct.df_to_markdown(stage_table, max_rows=12),
        "",
        "## Interpretation",
        "",
    ]
    if not eval_summary.empty:
        r = eval_summary.iloc[0]
        lines.extend(
            [
                f"- Total irrigation: {float(r.get('total_irrigation', np.nan)):.2f} mm.",
                f"- Total N: {float(r.get('total_n', np.nan)):.2f} kg/ha = base {float(r.get('total_base_n', np.nan)):.2f} + PPO extra {float(r.get('total_ppo_extra_n', np.nan)):.2f}.",
                f"- Final GRNWT: {float(r.get('final_grnwt', np.nan)):.2f}.",
                f"- GRNWT / 008_15 irrigation-only PPO: {float(r.get('grnwt_fraction_vs_00815_irrigation_only', np.nan)):.3f}.",
                f"- SWFAC stress days: {int(r.get('swfac_days_gt_0p05', 0))}; NSTRES stress days: {int(r.get('nstres_days_gt_0p05', 0))}.",
                f"- Soft SWFAC penalty total: {float(r.get('soft_swfac_penalty_total', np.nan)):.2f}; soft NSTRES penalty total: {float(r.get('soft_nstres_penalty_total', np.nan)):.2f}.",
                f"- First irrigation DAP: {r.get('first_irrigation_dap', np.nan)}; first PPO N DAP: {r.get('first_ppo_n_dap', np.nan)}.",
                f"- Debug joint promising: {bool(r.get('debug_joint_promising', False))}.",
            ]
        )
        if bool(r.get("debug_joint_promising", False)):
            lines.append("- The first water-nitrogen joint smoke test is promising under the predefined criteria.")
        else:
            lines.append("- The first water-nitrogen joint smoke test did not fully meet the predefined criteria; inspect action space before tuning reward.")
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Training summary: `{(root(config) / 'evaluation' / f'{station}_{year}_joint_ppo_training.csv').relative_to(PROJECT_ROOT)}`",
            f"- Evaluation summary: `{(root(config) / 'evaluation' / f'{station}_{year}_joint_ppo_summary.csv').relative_to(PROJECT_ROOT)}`",
            f"- Daily outputs: `{(root(config) / 'daily_outputs' / station).relative_to(PROJECT_ROOT)}`",
            f"- Figures: `{(root(config) / 'figures').relative_to(PROJECT_ROOT)}`",
        ]
    )
    doc_md(config).write_text("\n".join(lines), encoding="utf-8")


def run(config_path: Path, report_only: bool = False) -> None:
    config = direct.load_yaml(config_path)
    station, year = base_runner.station_year(config)
    base_runner.ensure_dirs(config, station)
    direct.write_yaml(config, root(config) / "configs" / config_path.name)
    env_config = base_runner.build_env_config(config, station, year)
    direct.write_yaml(env_config, root(config) / "configs" / "rendered_env_config.yaml")
    seed = int(config.get("seed", 0))
    train_path = root(config) / "evaluation" / f"{station}_{year}_joint_ppo_training.csv"
    summary_path = root(config) / "evaluation" / f"{station}_{year}_joint_ppo_summary.csv"
    stage_path = root(config) / "daily_outputs" / station / f"{station}_{year}_seed{seed}_soft_stress_stage_steps.csv"
    if report_only:
        train_summary = pd.read_csv(train_path)
        eval_summary = pd.read_csv(summary_path)
        stages = pd.read_csv(stage_path)
    else:
        base_runner.make_env = make_env
        base_runner.model_path = model_path
        base_runner.summarize_run = summarize_run
        train_summary = pd.DataFrame([base_runner.train_one(config, env_config, station, year, seed)])
        train_summary.to_csv(train_path, index=False, encoding="utf-8-sig")
        eval_rows: list[dict[str, Any]] = []
        daily = pd.DataFrame()
        stages = pd.DataFrame()
        if train_summary["run_status"].iloc[0] == "ok":
            daily, stages, summary = base_runner.evaluate_one(config, env_config, station, year, seed)
            eval_rows.append(summary)
            base_runner.plot_outputs(config, station, year, daily, stages)
        eval_summary = pd.DataFrame(eval_rows)
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

