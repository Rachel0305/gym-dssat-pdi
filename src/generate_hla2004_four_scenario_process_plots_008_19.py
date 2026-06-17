from __future__ import annotations

from pathlib import Path
from typing import Any
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ppo_safe_rendering import PROJECT_ROOT


OUTPUT_ROOT = PROJECT_ROOT / "Leave_One_experiments" / "hla2004_four_scenario_process_plots_008_19"
FIG_DIR = OUTPUT_ROOT / "figures"
EVAL_DIR = OUTPUT_ROOT / "evaluation"
DOC_MD = PROJECT_ROOT / "docs" / "2026-06-14_008_19_hla2004_four_scenario_process_plots_for_supervisor_report.md"


SCENARIOS = {
    "null_zero": {
        "label": "Null zero",
        "daily": PROJECT_ROOT / "Leave_One_experiments" / "representative_management_comparison_008_06" / "daily_outputs" / "HLA" / "2004_null_zero_daily.csv",
        "note": "No irrigation and no nitrogen; strict lower-bound baseline.",
    },
    "expert_reference_recorded": {
        "label": "Recorded expert",
        "daily": PROJECT_ROOT / "Leave_One_experiments" / "representative_management_comparison_008_06" / "daily_outputs" / "HLA" / "2004_expert_reference_recorded_daily.csv",
        "note": "Single-year observed site management record, not a year-specific optimum.",
    },
    "dssat_auto_attempt": {
        "label": "DSSAT auto attempt",
        "daily": PROJECT_ROOT / "Leave_One_experiments" / "representative_management_comparison_008_06" / "daily_outputs" / "HLA" / "2004_dssat_auto_attempt_daily.csv",
        "note": "Automatic-management attempt through gym-DSSAT; action totals were not fully exposed and output behaved like null.",
    },
    "ppo_soft_stress_seed0_00815": {
        "label": "PPO soft-stress seed0",
        "daily": PROJECT_ROOT / "Leave_One_experiments" / "hla2004_irrigation_only_soft_stress_ppo_008_15_10k" / "daily_outputs" / "HLA" / "HLA_2004_seed0_stress_aware_stage_daily.csv",
        "stage": PROJECT_ROOT / "Leave_One_experiments" / "hla2004_irrigation_only_soft_stress_ppo_008_15_10k" / "daily_outputs" / "HLA" / "HLA_2004_seed0_stress_aware_stage_steps.csv",
        "note": "Current HLA PPO diagnostic policy: no hard minimum gate, fixed N150, PPO controls irrigation.",
    },
}

SCENARIO_COLORS = {
    "null_zero": "#000000",
    "expert_reference_recorded": "#009E73",
    "dssat_auto_attempt": "#E69F00",
    "ppo_soft_stress_seed0_00815": "#0072B2",
}


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    DOC_MD.parent.mkdir(parents=True, exist_ok=True)


def num(series: pd.Series | Any, default: float = 0.0) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce").fillna(default)
    return pd.Series(dtype=float)


def load_daily(key: str, spec: dict[str, Any]) -> pd.DataFrame:
    df = pd.read_csv(spec["daily"])
    out = df.copy()
    out["scenario_key"] = key
    out["scenario_label"] = spec["label"]
    if "dap" not in out.columns:
        out["dap"] = np.arange(1, len(out) + 1)
    out["dap"] = num(out["dap"])
    for col in ["rain", "swfac", "nstres", "topwt", "grnwt", "real_action_amir", "real_action_anfer"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = num(out[col], default=0.0)
    out["plot_reward"] = select_reward_series(out)
    return aggregate_daily(out)


def select_reward_series(df: pd.DataFrame) -> pd.Series:
    for col in ["diagnostic_reward", "reward_stage", "reward_after_soft_stress", "reward", "env_reward"]:
        if col in df.columns:
            return num(df[col], default=0.0)
    return pd.Series([0.0] * len(df), index=df.index)


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated wrapper rows to one row per DAP for plotting.

    Stage wrappers can record both the stage action and internal zero-action
    rows within the same DAP. For process plots, rainfall/stress/growth should
    be shown once per day, while management actions should be summed per day.
    """

    group_cols = ["scenario_key", "scenario_label", "dap"]
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
    return out.sort_values("dap").reset_index(drop=True)


def load_all() -> dict[str, pd.DataFrame]:
    return {key: load_daily(key, spec) for key, spec in SCENARIOS.items()}


def save_plot(fig: plt.Figure, filename: str) -> Path:
    path = FIG_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_weather(key: str, df: pd.DataFrame, label: str) -> Path:
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(df["dap"], df["rain"], color="#4F81BD", alpha=0.55, width=1.0)
    ax.set_title(f"HLA 2004 {label}: rainfall")
    ax.set_xlabel("DAP")
    ax.set_ylabel("Rainfall (mm d-1)")
    ax.grid(axis="y", alpha=0.25)
    return save_plot(fig, f"{key}_01_weather_rainfall.png")


def plot_stress(key: str, df: pd.DataFrame, label: str) -> Path:
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(df["dap"], df["swfac"], color="#4F81BD", label="SWFAC")
    ax.plot(df["dap"], df["nstres"], color="#C0504D", label="NSTRES")
    ax.axhline(0.05, color="black", linestyle="--", linewidth=0.8)
    ax.set_title(f"HLA 2004 {label}: water and nitrogen stress")
    ax.set_xlabel("DAP")
    ax.set_ylabel("Stress index")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    return save_plot(fig, f"{key}_02_stress_swfac_nstres.png")


def plot_actions(key: str, df: pd.DataFrame, label: str) -> Path:
    fig, ax = plt.subplots(figsize=(9, 3.8))
    action = df[(df["real_action_amir"].abs() > 1e-9) | (df["real_action_anfer"].abs() > 1e-9)].copy()
    if action.empty:
        ax.text(0.5, 0.5, "No exposed irrigation/fertilization action", transform=ax.transAxes, ha="center", va="center")
    else:
        ax.vlines(action["dap"], 0, action["real_action_amir"], color="#4F81BD", linewidth=2.0, label="Irrigation (mm)")
        ax.vlines(action["dap"], 0, action["real_action_anfer"], color="#9BBB59", linewidth=2.0, linestyles="dashed", label="N fertilizer (kg ha-1)")
        ax.legend(frameon=False)
    ax.set_title(f"HLA 2004 {label}: management actions")
    ax.set_xlabel("DAP")
    ax.set_ylabel("Amount")
    ax.grid(axis="y", alpha=0.25)
    return save_plot(fig, f"{key}_03_management_actions.png")


def plot_growth(key: str, df: pd.DataFrame, label: str) -> Path:
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(df["dap"], df["topwt"], color="#9BBB59", label="TOPWT")
    ax.plot(df["dap"], df["grnwt"], color="#8064A2", label="GRNWT")
    ax.set_title(f"HLA 2004 {label}: crop growth")
    ax.set_xlabel("DAP")
    ax.set_ylabel("Biomass / grain weight (kg ha-1)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    return save_plot(fig, f"{key}_04_growth_topwt_grnwt.png")


def plot_reward(key: str, df: pd.DataFrame, label: str) -> Path:
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(df["dap"], df["plot_reward"], color="#F79646", label="Daily reward")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title(f"HLA 2004 {label}: daily reward")
    ax.set_xlabel("DAP")
    ax.set_ylabel("Reward")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    return save_plot(fig, f"{key}_05_daily_reward.png")


def plot_combined(data: dict[str, pd.DataFrame]) -> Path:
    fig, axes = plt.subplots(7, 1, figsize=(11, 15), sharex=True)
    weather = next(iter(data.values()))
    axes[0].bar(weather["dap"], weather["rain"], color="#4F81BD", alpha=0.55, width=1.0, label="Observed rainfall")
    for key, df in data.items():
        label = SCENARIOS[key]["label"]
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
    axes[0].set_title("HLA 2004 four-scenario process comparison")
    axes[6].axhline(0, color="black", linestyle="--", linewidth=0.8)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False, ncol=2)
    axes[3].legend(frameon=False, ncol=2)
    axes[4].legend(frameon=False, ncol=2)
    axes[5].legend(frameon=False, ncol=2)
    axes[6].legend(frameon=False, ncol=2)
    return save_plot(fig, "HLA_2004_four_scenario_combined_process_comparison.png")


def summarize(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for key, df in data.items():
        action_i = df["real_action_amir"].where(df["real_action_amir"].notna(), 0.0)
        action_n = df["real_action_anfer"].where(df["real_action_anfer"].notna(), 0.0)
        rows.append(
            {
                "scenario_key": key,
                "scenario_label": SCENARIOS[key]["label"],
                "total_irrigation": float(action_i.sum()),
                "total_n": float(action_n.sum()),
                "final_grnwt": float(df["grnwt"].dropna().iloc[-1]) if len(df["grnwt"].dropna()) else np.nan,
                "final_topwt": float(df["topwt"].dropna().iloc[-1]) if len(df["topwt"].dropna()) else np.nan,
                "max_swfac": float(df["swfac"].max()),
                "swfac_days_gt_0p05": int((df["swfac"] > 0.05).sum()),
                "max_nstres": float(df["nstres"].max()),
                "nstres_days_gt_0p05": int((df["nstres"] > 0.05).sum()),
                "total_reward_for_plot": float(df["plot_reward"].sum()),
                "mean_daily_reward_for_plot": float(df["plot_reward"].mean()),
                "note": SCENARIOS[key]["note"],
            }
        )
    out = pd.DataFrame(rows)
    ref = out.loc[out["scenario_key"].eq("ppo_soft_stress_seed0_00815"), "final_grnwt"]
    fixed = 7062.4658203125
    out["grnwt_fraction_vs_fixed_I120"] = out["final_grnwt"] / fixed
    return out


def build_daily_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine the one-row-per-day plotting data for all scenarios."""

    daily_frames = []
    keep_cols = [
        "scenario_key",
        "scenario_label",
        "dap",
        "date",
        "doy",
        "rain",
        "swfac",
        "nstres",
        "real_action_amir",
        "real_action_anfer",
        "topwt",
        "grnwt",
        "plot_reward",
    ]
    for key, df in data.items():
        out = df.copy()
        for col in keep_cols:
            if col not in out.columns:
                out[col] = np.nan
        out["source_daily_csv"] = str(SCENARIOS[key]["daily"].relative_to(PROJECT_ROOT))
        out["note"] = SCENARIOS[key]["note"]
        daily_frames.append(out[[*keep_cols, "source_daily_csv", "note"]])
    combined = pd.concat(daily_frames, ignore_index=True)
    return combined.sort_values(["scenario_key", "dap"]).reset_index(drop=True)


def df_md(df: pd.DataFrame) -> str:
    out = df.copy()
    for col in out.select_dtypes(include=["float", "int"]).columns:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(4)
    header = "| " + " | ".join(out.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(out.columns)) + " |"
    rows = ["| " + " | ".join("" if pd.isna(x) else str(x) for x in row) + " |" for row in out.values.tolist()]
    return "\n".join([header, sep, *rows])


def safe_to_csv(df: pd.DataFrame, path: Path) -> Path:
    candidates = [
        path,
        path.with_name(path.stem + "_updated.csv"),
        path.with_name(path.stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"),
    ]
    last_error: PermissionError | None = None
    for candidate in candidates:
        try:
            df.to_csv(candidate, index=False, encoding="utf-8-sig")
            return candidate
        except PermissionError as exc:
            last_error = exc
    raise last_error if last_error else PermissionError(path)


def main() -> None:
    ensure_dirs()
    data = load_all()
    figures: list[Path] = []
    for key, df in data.items():
        label = SCENARIOS[key]["label"]
        figures.extend(
            [
                plot_weather(key, df, label),
                plot_stress(key, df, label),
                plot_actions(key, df, label),
                plot_growth(key, df, label),
                plot_reward(key, df, label),
            ]
        )
    figures.append(plot_combined(data))
    daily_table = build_daily_table(data)
    daily_path = EVAL_DIR / "008_19_hla2004_four_scenario_daily_values_for_plots.csv"
    daily_path = safe_to_csv(daily_table, daily_path)
    summary = summarize(data)
    summary_path = EVAL_DIR / "008_19_hla2004_four_scenario_summary.csv"
    summary_path = safe_to_csv(summary, summary_path)
    lines = [
        "# 008_19 HLA 2004 Four-Scenario Process Plots For Supervisor",
        "",
        "## Purpose",
        "",
        "This report reorganizes existing HLA 2004 outputs into the four management scenarios requested by the supervisor.",
        "No PPO training and no DSSAT rerun were performed.",
        "",
        "## Scenarios",
        "",
        "- `null_zero`: no irrigation and no nitrogen.",
        "- `expert_reference_recorded`: single-year observed HLA management reference.",
        "- `dssat_auto_attempt`: gym-DSSAT automatic-management attempt; retained as diagnostic because it behaved like null.",
        "- `ppo_soft_stress_seed0_00815`: current HLA PPO result with no hard-minimum gate and fixed N150.",
        "",
        "## Summary",
        "",
        df_md(summary),
        "",
        "## Interpretation",
        "",
        "- HLA 2004 is the current main water-limited demonstration case.",
        "- PPO should be represented by 008_15 soft-stress PPO, not by the older FQA transfer replay.",
        "- The PPO case used 80 mm irrigation and achieved about 96% of the fixed I120_N150 reference yield.",
        "- `dssat_auto_attempt` should not be described as a fully verified native DSSAT automatic-management baseline.",
        "- PPO contribution is visible in this scenario set, but seed stability remains a documented limitation from 008_16/008_17.",
        "- Reward curves are included for process comparison. They use the available saved daily reward column for each scenario, prioritizing diagnostic reward when available.",
        "",
        "## Figures",
        "",
        *[f"- `{p.relative_to(PROJECT_ROOT)}`" for p in figures],
        "",
        "## Files",
        "",
        f"- Summary CSV: `{summary_path.relative_to(PROJECT_ROOT)}`",
        f"- Daily values CSV for all process plots: `{daily_path.relative_to(PROJECT_ROOT)}`",
    ]
    DOC_MD.write_text("\n".join(lines), encoding="utf-8")
    print(daily_path.relative_to(PROJECT_ROOT))
    print(summary_path.relative_to(PROJECT_ROOT))
    print(DOC_MD.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
