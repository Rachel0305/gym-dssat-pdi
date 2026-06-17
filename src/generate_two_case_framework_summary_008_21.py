from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_all_year_direct_action_safe_ppo as direct
import run_management_scenario_comparison as msc
from ppo_safe_rendering import PROJECT_ROOT


OUTPUT_ROOT = PROJECT_ROOT / "Leave_One_experiments" / "two_case_framework_summary_008_21"
FIG_DIR = OUTPUT_ROOT / "figures"
EVAL_DIR = OUTPUT_ROOT / "evaluation"
DOC_MD = PROJECT_ROOT / "docs" / "2026-06-16_008_21_hla2004_fqa2016_two_case_framework_summary.md"

SCENARIO_POOL = PROJECT_ROOT / "Leave_One_experiments" / "all_year_weather_calibration_validation" / "scenario_pool" / "all_year_weather_scenario_pool.csv"
BASE_ENV_CONFIG = PROJECT_ROOT / "experiments" / "ppo_observed_years" / "config_all_year_direct_action_safe_ppo.yaml"

HLA_DAILY_VALUES = PROJECT_ROOT / "Leave_One_experiments" / "hla2004_four_scenario_process_plots_008_19" / "evaluation" / "008_19_hla2004_four_scenario_daily_values_for_plots.csv"
HLA_SUMMARY = PROJECT_ROOT / "Leave_One_experiments" / "hla2004_four_scenario_process_plots_008_19" / "evaluation" / "008_19_hla2004_four_scenario_summary.csv"
FQA_PPO_DAILY = PROJECT_ROOT / "Leave_One_experiments" / "fqa2016_irrigation_only_soft_stress_ppo_008_20" / "daily_outputs" / "FQA" / "FQA_2016_seed0_soft_stress_stage_daily.csv"
FQA_PPO_SUMMARY = PROJECT_ROOT / "Leave_One_experiments" / "fqa2016_irrigation_only_soft_stress_ppo_008_20" / "evaluation" / "FQA_2016_soft_stress_ppo_summary.csv"

SCENARIO_COLORS = {
    "null_zero": "#000000",
    "expert_reference_recorded": "#009E73",
    "dssat_auto_attempt": "#E69F00",
    "ppo_soft_stress_seed0": "#0072B2",
}

FQA_SCENARIOS = {
    "null_zero": {
        "label": "Null zero",
        "runner": "null_zero",
        "note": "No irrigation and no nitrogen.",
    },
    "expert_reference_recorded": {
        "label": "Recorded expert",
        "runner": "expert_reference",
        "note": "Single-year FQA observed management record; not a year-specific optimum.",
    },
    "dssat_auto_attempt": {
        "label": "DSSAT auto attempt",
        "runner": "dssat_automatic",
        "note": "Automatic-management attempt through gym-DSSAT; retained as diagnostic.",
    },
    "ppo_soft_stress_seed0": {
        "label": "PPO soft-stress seed0",
        "runner": "reuse_00820",
        "note": "008_20 FQA 2016 soft-stress stage PPO; fixed N150, PPO controls irrigation.",
    },
}


def ensure_dirs() -> None:
    for sub in ["daily_outputs/FQA", "event_outputs", "evaluation", "figures", "configs", "logs", "rendered_inputs"]:
        (OUTPUT_ROOT / sub).mkdir(parents=True, exist_ok=True)
    DOC_MD.parent.mkdir(parents=True, exist_ok=True)


def num(series: pd.Series | Any, default: float = 0.0) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce").fillna(default)
    return pd.Series(dtype=float)


def safe_to_csv(df: pd.DataFrame, path: Path) -> Path:
    candidates = [
        path,
        path.with_name(path.stem + "_updated.csv"),
        path.with_name(path.stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"),
    ]
    for candidate in candidates:
        try:
            df.to_csv(candidate, index=False, encoding="utf-8-sig")
            return candidate
        except PermissionError:
            continue
    raise PermissionError(path)


def build_env_config(station: str, year: int) -> dict[str, Any]:
    cfg = direct.load_yaml(BASE_ENV_CONFIG)
    cfg["paths"]["output_root"] = str(OUTPUT_ROOT.relative_to(PROJECT_ROOT))
    pool = pd.read_csv(SCENARIO_POOL)
    row = pool[(pool["station_code"].astype(str).eq(station)) & (pd.to_numeric(pool["year"], errors="coerce").eq(year))]
    if row.empty:
        raise KeyError(f"{station} {year} missing from scenario pool")
    env_config = direct.build_env_config(cfg, row)
    direct.write_yaml(env_config, OUTPUT_ROOT / "configs" / f"rendered_env_config_{station}_{year}.yaml")
    return env_config


def select_reward_series(df: pd.DataFrame) -> pd.Series:
    for col in ["diagnostic_reward", "reward_stage", "reward_after_soft_stress", "reward", "env_reward"]:
        if col in df.columns:
            return num(df[col], default=0.0)
    return pd.Series([0.0] * len(df), index=df.index)


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["station", "year", "scenario_key", "scenario_label", "dap"]
    agg = {
        "rain": "first",
        "swfac": "last",
        "nstres": "last",
        "topwt": "last",
        "grnwt": "last",
        "real_action_amir": "sum",
        "real_action_anfer": "sum",
        "plot_reward": "sum",
    }
    if "date" in df.columns:
        agg["date"] = "first"
    if "doy" in df.columns:
        agg["doy"] = "first"
    out = df.groupby(group_cols, as_index=False).agg(agg)
    return out.sort_values(["scenario_key", "dap"]).reset_index(drop=True)


def normalize_daily(df: pd.DataFrame, station: str, year: int, scenario_key: str, label: str, source: str, note: str) -> pd.DataFrame:
    out = df.copy()
    out["station"] = station
    out["year"] = int(year)
    out["scenario_key"] = scenario_key
    out["scenario_label"] = label
    if "dap" not in out.columns:
        out["dap"] = np.arange(1, len(out) + 1)
    for col in ["rain", "swfac", "nstres", "topwt", "grnwt", "real_action_amir", "real_action_anfer"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = num(out[col], default=0.0)
    if scenario_key == "dssat_auto_attempt":
        out["real_action_amir"] = out["real_action_amir"].fillna(0.0)
        out["real_action_anfer"] = out["real_action_anfer"].fillna(0.0)
    out["plot_reward"] = select_reward_series(out)
    out["source_daily_csv"] = source
    out["note"] = note
    return aggregate_daily(out)


def run_fqa_scenarios() -> dict[str, pd.DataFrame]:
    station = "FQA"
    year = 2016
    env_config = build_env_config(station, year)
    original_output_root = msc.OUTPUT_ROOT
    data: dict[str, pd.DataFrame] = {}
    try:
        msc.OUTPUT_ROOT = OUTPUT_ROOT
        msc.ensure_dirs([station])
        for key, spec in FQA_SCENARIOS.items():
            if spec["runner"] == "reuse_00820":
                raw = pd.read_csv(FQA_PPO_DAILY)
                daily = normalize_daily(
                    raw,
                    station,
                    year,
                    key,
                    spec["label"],
                    str(FQA_PPO_DAILY.relative_to(PROJECT_ROOT)),
                    spec["note"],
                )
            else:
                summary, raw, events = msc.run_scenario(env_config, station, year, spec["runner"], seed=0, max_steps=260)
                if raw.empty:
                    raise RuntimeError(f"{station} {year} {key} failed: {summary.get('notes', '')}")
                daily = normalize_daily(
                    raw,
                    station,
                    year,
                    key,
                    spec["label"],
                    str((OUTPUT_ROOT / "daily_outputs" / station / f"{year}_{spec['runner']}_daily.csv").relative_to(PROJECT_ROOT)),
                    spec["note"],
                )
            data[key] = daily
            out = OUTPUT_ROOT / "daily_outputs" / station / f"{station}_{year}_{key}_daily_values_for_plots.csv"
            daily.to_csv(out, index=False, encoding="utf-8-sig")
    finally:
        msc.OUTPUT_ROOT = original_output_root
    return data


def summarize_daily(data: dict[str, pd.DataFrame], station: str, year: int) -> pd.DataFrame:
    rows = []
    for key, df in data.items():
        action_i = num(df["real_action_amir"])
        action_n = num(df["real_action_anfer"])
        final_grnwt = float(num(df["grnwt"], np.nan).dropna().iloc[-1]) if len(num(df["grnwt"], np.nan).dropna()) else np.nan
        rows.append(
            {
                "station": station,
                "year": year,
                "scenario_key": key,
                "scenario_label": FQA_SCENARIOS.get(key, {}).get("label", key),
                "total_irrigation": float(action_i.sum()),
                "total_n": float(action_n.sum()),
                "final_grnwt": final_grnwt,
                "final_topwt": float(num(df["topwt"], np.nan).dropna().iloc[-1]) if len(num(df["topwt"], np.nan).dropna()) else np.nan,
                "max_swfac": float(num(df["swfac"]).max()),
                "swfac_days_gt_0p05": int((num(df["swfac"]) > 0.05).sum()),
                "max_nstres": float(num(df["nstres"]).max()),
                "nstres_days_gt_0p05": int((num(df["nstres"]) > 0.05).sum()),
                "total_reward_for_plot": float(num(df["plot_reward"]).sum()),
                "mean_daily_reward_for_plot": float(num(df["plot_reward"]).mean()),
                "note": FQA_SCENARIOS.get(key, {}).get("note", ""),
            }
        )
    return pd.DataFrame(rows)


def plot_four_scenario_combined(data: dict[str, pd.DataFrame], station: str, year: int) -> Path:
    fig, axes = plt.subplots(7, 1, figsize=(11, 15), dpi=180, sharex=True)
    weather = next(iter(data.values()))
    axes[0].bar(weather["dap"], weather["rain"], color="#4F81BD", alpha=0.55, width=1.0, label="Observed rainfall")
    for key, df in data.items():
        label = FQA_SCENARIOS[key]["label"]
        color = SCENARIO_COLORS.get(key, None)
        axes[1].plot(df["dap"], df["swfac"], label=label, color=color, linewidth=2.2)
        axes[2].plot(df["dap"], df["nstres"], label=label, color=color, linewidth=2.2)
        irrigation = df[df["real_action_amir"].abs() > 1e-9].copy()
        nitrogen = df[df["real_action_anfer"].abs() > 1e-9].copy()
        if not irrigation.empty:
            axes[3].vlines(irrigation["dap"], 0, irrigation["real_action_amir"], color=color, linewidth=2.4, label=label)
            axes[3].scatter(irrigation["dap"], irrigation["real_action_amir"], color=color, s=24)
        if not nitrogen.empty:
            axes[4].vlines(nitrogen["dap"], 0, nitrogen["real_action_anfer"], color=color, linewidth=2.4, label=label)
            axes[4].scatter(nitrogen["dap"], nitrogen["real_action_anfer"], color=color, s=24)
        axes[5].plot(df["dap"], df["grnwt"], label=label, color=color, linewidth=2.2)
        axes[6].plot(df["dap"], df["plot_reward"], label=label, color=color, linewidth=2.2)
    axes[0].set_ylabel("Rain")
    axes[1].set_ylabel("SWFAC")
    axes[2].set_ylabel("NSTRES")
    axes[3].set_ylabel("Irrigation")
    axes[4].set_ylabel("N fertilizer")
    axes[5].set_ylabel("GRNWT")
    axes[6].set_ylabel("Reward")
    axes[6].set_xlabel("DAP")
    axes[0].set_title(f"{station} {year} four-scenario process comparison")
    axes[6].axhline(0, color="black", linestyle="--", linewidth=0.8)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    for idx in [0, 1, 3, 4, 5, 6]:
        axes[idx].legend(frameon=False, ncol=2)
    path = FIG_DIR / f"{station}_{year}_four_scenario_combined_process_comparison.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def load_hla_case() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(HLA_DAILY_VALUES)
    summary = pd.read_csv(HLA_SUMMARY)
    summary = summary.rename(columns={"total_n": "total_n"})
    summary["station"] = "HLA"
    summary["year"] = 2004
    return daily, summary


def build_two_case_summary(hla_summary: pd.DataFrame, fqa_summary: pd.DataFrame) -> pd.DataFrame:
    hla = hla_summary.copy()
    fqa = fqa_summary.copy()
    hla["case_id"] = "HLA 2004"
    fqa["case_id"] = "FQA 2016"
    combined = pd.concat([hla, fqa], ignore_index=True, sort=False)
    ppo_keys = ["ppo_soft_stress_seed0_00815", "ppo_soft_stress_seed0"]
    ppo = combined[combined["scenario_key"].isin(ppo_keys)].copy()
    ppo["case_role"] = np.where(ppo["case_id"].eq("HLA 2004"), "primary_demo", "second_validation")
    ppo["nonzero_irrigation"] = pd.to_numeric(ppo["total_irrigation"], errors="coerce") > 0
    ppo["non_saturated_irrigation"] = pd.to_numeric(ppo["total_irrigation"], errors="coerce") < 150
    ppo["framework_signal"] = np.where(
        ppo["nonzero_irrigation"] & ppo["non_saturated_irrigation"],
        "nonzero_non_saturated_interpretable",
        "needs_diagnosis",
    )
    return ppo[
        [
            "case_id",
            "case_role",
            "station",
            "year",
            "scenario_label",
            "total_irrigation",
            "total_n",
            "final_grnwt",
            "max_swfac",
            "swfac_days_gt_0p05",
            "max_nstres",
            "nstres_days_gt_0p05",
            "total_reward_for_plot",
            "framework_signal",
            "note",
        ]
    ]


def plot_two_case_summary(summary: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), dpi=180)
    x = np.arange(len(summary))
    labels = summary["case_id"].tolist()
    axes[0].bar(x, pd.to_numeric(summary["final_grnwt"], errors="coerce"), color=["#0072B2", "#009E73"])
    axes[0].set_title("Final GRNWT")
    axes[0].set_ylabel("kg ha-1")
    axes[1].bar(x, pd.to_numeric(summary["total_irrigation"], errors="coerce"), color=["#0072B2", "#009E73"])
    axes[1].set_title("PPO irrigation")
    axes[1].set_ylabel("mm")
    axes[2].bar(x, pd.to_numeric(summary["swfac_days_gt_0p05"], errors="coerce"), color=["#0072B2", "#009E73"])
    axes[2].set_title("SWFAC stress days")
    axes[2].set_ylabel("days")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Two-case PPO framework validation")
    fig.tight_layout()
    path = FIG_DIR / "HLA2004_FQA2016_two_case_ppo_summary.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def df_md(df: pd.DataFrame, max_rows: int = 30) -> str:
    out = df.head(max_rows).copy()
    for col in out.select_dtypes(include=["float", "int"]).columns:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(4)
    header = "| " + " | ".join(out.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(out.columns)) + " |"
    rows = ["| " + " | ".join("" if pd.isna(x) else str(x) for x in row) + " |" for row in out.values.tolist()]
    return "\n".join([header, sep, *rows])


def main() -> None:
    ensure_dirs()
    fqa_data = run_fqa_scenarios()
    fqa_daily = pd.concat(fqa_data.values(), ignore_index=True)
    fqa_daily_path = safe_to_csv(fqa_daily, EVAL_DIR / "008_21_fqa2016_four_scenario_daily_values_for_plots.csv")
    fqa_summary = summarize_daily(fqa_data, "FQA", 2016)
    fqa_summary_path = safe_to_csv(fqa_summary, EVAL_DIR / "008_21_fqa2016_four_scenario_summary.csv")
    fqa_fig = plot_four_scenario_combined(fqa_data, "FQA", 2016)

    hla_daily, hla_summary = load_hla_case()
    hla_daily_copy_path = safe_to_csv(hla_daily, EVAL_DIR / "008_21_hla2004_four_scenario_daily_values_for_plots_copy.csv")
    hla_summary_copy_path = safe_to_csv(hla_summary, EVAL_DIR / "008_21_hla2004_four_scenario_summary_copy.csv")

    two_case = build_two_case_summary(hla_summary, fqa_summary)
    two_case_path = safe_to_csv(two_case, EVAL_DIR / "008_21_hla2004_fqa2016_two_case_ppo_summary.csv")
    two_case_fig = plot_two_case_summary(two_case)

    lines = [
        "# 008_21 HLA 2004 + FQA 2016 Two-Case Framework Summary",
        "",
        "## Purpose",
        "",
        "This report summarizes the current two-case evidence that the weather/stress-assisted stage PPO framework is not limited to one year.",
        "HLA 2004 is the primary dry-year demonstration case. FQA 2016 is the second water-stress validation case.",
        "",
        "## Two-Case PPO Summary",
        "",
        df_md(two_case),
        "",
        "## FQA 2016 Four-Scenario Summary",
        "",
        df_md(fqa_summary),
        "",
        "## Interpretation",
        "",
        "- HLA 2004 and FQA 2016 both show nonzero and non-saturated PPO irrigation.",
        "- FQA 2016 extends the framework evidence beyond the primary HLA 2004 demonstration case.",
        "- This still does not prove universal optimality or full water-nitrogen simultaneous optimization.",
        "- The current claim should remain: the framework can generate interpretable irrigation decisions in representative water-stress years under fixed adequate nitrogen supply.",
        "",
        "## Figures",
        "",
        f"- FQA 2016 four-scenario seven-panel process plot: `{fqa_fig.relative_to(PROJECT_ROOT)}`",
        f"- HLA 2004 + FQA 2016 two-case PPO summary plot: `{two_case_fig.relative_to(PROJECT_ROOT)}`",
        "",
        "## Files",
        "",
        f"- FQA daily values for plots: `{fqa_daily_path.relative_to(PROJECT_ROOT)}`",
        f"- FQA four-scenario summary: `{fqa_summary_path.relative_to(PROJECT_ROOT)}`",
        f"- HLA daily values copy: `{hla_daily_copy_path.relative_to(PROJECT_ROOT)}`",
        f"- HLA summary copy: `{hla_summary_copy_path.relative_to(PROJECT_ROOT)}`",
        f"- Two-case PPO summary: `{two_case_path.relative_to(PROJECT_ROOT)}`",
    ]
    DOC_MD.write_text("\n".join(lines), encoding="utf-8")
    print(fqa_daily_path.relative_to(PROJECT_ROOT))
    print(fqa_summary_path.relative_to(PROJECT_ROOT))
    print(two_case_path.relative_to(PROJECT_ROOT))
    print(fqa_fig.relative_to(PROJECT_ROOT))
    print(two_case_fig.relative_to(PROJECT_ROOT))
    print(DOC_MD.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
