"""Build diagnostic four-scenario visuals replacing rule/old auto with 010_13 DSSAT auto.

This is a plotting-only diagnostic. It does not run DSSAT and does not train PPO.

Important caveat:
    The first, second, and fourth scenarios come from the older 008_19
    comparison. The third scenario comes from 010_13 candidate-IC DSSAT native
    automatic management. Therefore the output is useful for visual design and
    discussion, not a strict same-input four-scenario result.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OLD_DAILY = (
    PROJECT_ROOT
    / "Leave_One_experiments"
    / "hla2004_four_scenario_process_plots_008_19"
    / "evaluation"
    / "008_19_hla2004_four_scenario_daily_values_for_plots.csv"
)
AUTO_DIR = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "HLA_2004"
    / "candidate_ic055_n025_dssat_auto_management_010_13"
)
AUTO_DAILY = AUTO_DIR / "hla2004_candidate_ic_dssat_auto_daily_values.csv"
AUTO_EVENTS = AUTO_DIR / "hla2004_candidate_ic_dssat_auto_management_events.csv"
OUT_DIR = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "HLA_2004"
    / "candidate_ic055_n025_dssat_auto_management_010_13"
    / "diagnostic_four_scenario_visuals"
)


TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

SCENARIOS = [
    "Null zero",
    "Recorded expert",
    "DSSAT auto irrigation + auto-N attempt",
    "PPO soft-stress seed0",
]

COLORS = {
    "Null zero": "#464C55",
    "Recorded expert": "#CC6F47",
    "DSSAT auto irrigation + auto-N attempt": "#5477C4",
    "PPO soft-stress seed0": "#386411",
}


def use_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        },
    )


def load_old_three() -> pd.DataFrame:
    old = pd.read_csv(OLD_DAILY)
    keep = {
        "null_zero": "Null zero",
        "expert_reference_recorded": "Recorded expert",
        "ppo_soft_stress_seed0_00815": "PPO soft-stress seed0",
    }
    old = old[old["scenario_key"].isin(keep)].copy()
    old["scenario_label"] = old["scenario_key"].map(keep)
    for col in ["dap", "rain", "swfac", "nstres", "real_action_amir", "real_action_anfer", "topwt", "grnwt", "plot_reward"]:
        if col not in old.columns:
            old[col] = 0.0
        old[col] = pd.to_numeric(old[col], errors="coerce").fillna(0.0)
    old["source_note"] = "008_19 old comparison inputs"
    return old[
        [
            "scenario_label",
            "dap",
            "rain",
            "swfac",
            "nstres",
            "real_action_amir",
            "real_action_anfer",
            "topwt",
            "grnwt",
            "plot_reward",
            "source_note",
        ]
    ]


def load_new_auto() -> pd.DataFrame:
    daily = pd.read_csv(AUTO_DAILY)
    events = pd.read_csv(AUTO_EVENTS) if AUTO_EVENTS.exists() else pd.DataFrame()
    out = pd.DataFrame(
        {
            "scenario_label": "DSSAT auto irrigation + auto-N attempt",
            "dap": pd.to_numeric(daily["dap"], errors="coerce"),
            "rain": pd.to_numeric(daily.get("rain", 0.0), errors="coerce").fillna(0.0),
            "swfac": pd.to_numeric(daily.get("wspd", 0.0), errors="coerce").fillna(0.0),
            "nstres": pd.to_numeric(daily.get("nstd", 0.0), errors="coerce").fillna(0.0),
            "real_action_amir": 0.0,
            "real_action_anfer": 0.0,
            "topwt": pd.to_numeric(daily.get("cwad", 0.0), errors="coerce").fillna(0.0),
            "grnwt": pd.to_numeric(daily.get("gwad", 0.0), errors="coerce").fillna(0.0),
            "plot_reward": 0.0,
            "source_note": "010_13 candidate IC DSSAT native automatic management",
        }
    )
    if not events.empty:
        for _, event in events.iterrows():
            dap = pd.to_numeric(event.get("dap"), errors="coerce")
            if pd.isna(dap):
                continue
            amount = float(event.get("amount", 0.0))
            operation = str(event.get("operation", ""))
            idx = out["dap"].eq(float(dap))
            if operation.lower().find("irrigation") >= 0:
                out.loc[idx, "real_action_amir"] += amount
            elif "fertil" in operation.lower() or "nitrogen" in operation.lower():
                out.loc[idx, "real_action_anfer"] += amount
    return out


def combine_data() -> pd.DataFrame:
    combined = pd.concat([load_old_three(), load_new_auto()], ignore_index=True)
    combined["scenario_label"] = pd.Categorical(combined["scenario_label"], categories=SCENARIOS, ordered=True)
    combined = combined.sort_values(["scenario_label", "dap"]).reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_DIR / "hla2004_diagnostic_four_scenario_daily_values_with_01013_auto.csv", index=False, encoding="utf-8-sig")
    return combined


def plot_combined_facets(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True)
    rain = df[df["scenario_label"].eq("DSSAT auto irrigation + auto-N attempt")].sort_values("dap")
    if rain.empty:
        rain = df[df["scenario_label"].eq("Null zero")].sort_values("dap")
    axes[0].bar(rain["dap"], rain["rain"], color="#C5CAD3", edgecolor="#7A828F", linewidth=0.3, width=1.0, label="Rainfall")
    axes[0].set_ylabel("Rain\n(mm)")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].set_title(
        "HLA 2004 diagnostic four-scenario process plot",
        loc="left",
        fontsize=13,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    axes[0].text(
        0,
        1.18,
        "Diagnostic mixed-input figure: old 008_19 null/expert/PPO plus 010_13 candidate-IC DSSAT auto scenario. Use for visual design, not final same-input conclusion.",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color=TOKENS["muted"],
    )

    for scenario in SCENARIOS:
        sub = df[df["scenario_label"].eq(scenario)].sort_values("dap")
        color = COLORS[scenario]
        axes[1].plot(sub["dap"], sub["swfac"], color=color, linewidth=1.8, label=scenario)
        axes[2].plot(sub["dap"], sub["nstres"], color=color, linewidth=1.8, label=scenario)
        axes[3].vlines(
            sub.loc[sub["real_action_amir"].abs() > 1e-9, "dap"],
            0,
            sub.loc[sub["real_action_amir"].abs() > 1e-9, "real_action_amir"],
            color=color,
            linewidth=2.2,
            alpha=0.9,
        )
        axes[3].scatter(
            sub.loc[sub["real_action_anfer"].abs() > 1e-9, "dap"],
            sub.loc[sub["real_action_anfer"].abs() > 1e-9, "real_action_anfer"],
            color=color,
            s=28,
            marker="^",
            edgecolor="white",
            linewidth=0.4,
        )
        axes[4].plot(sub["dap"], sub["grnwt"], color=color, linewidth=1.8, label=scenario)
        axes[4].plot(sub["dap"], sub["topwt"], color=color, linewidth=1.2, linestyle="--", alpha=0.75)

    axes[1].set_ylabel("Water\nstress")
    axes[1].set_title("Water stress index by scenario", loc="left", fontsize=10)
    axes[2].set_ylabel("Nitrogen\nstress")
    axes[2].set_title("Nitrogen stress index by scenario", loc="left", fontsize=10)
    axes[3].set_ylabel("Mgmt\namount")
    axes[3].set_title("Management events: irrigation as vertical lines; fertilization as triangle markers", loc="left", fontsize=10)
    axes[4].set_ylabel("kg/ha")
    axes[4].set_title("Crop outcome: solid = grain weight, dashed = aboveground biomass", loc="left", fontsize=10)
    axes[4].set_xlabel("DAP")

    for ax in axes:
        ax.grid(True, axis="y", linestyle="--", alpha=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1].legend(frameon=False, loc="upper left", ncol=2)
    axes[4].legend(frameon=False, loc="upper left", ncol=2)
    fig.tight_layout()
    path = OUT_DIR / "hla2004_diagnostic_four_scenario_process_with_01013_auto.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_final_bars(df: pd.DataFrame) -> Path:
    rows = []
    for scenario, sub in df.groupby("scenario_label", observed=True):
        sub = sub.sort_values("dap")
        rows.append(
            {
                "scenario": str(scenario),
                "grain_yield": float(sub["grnwt"].iloc[-1]),
                "biomass": float(sub["topwt"].iloc[-1]),
                "total_irrigation": float(sub["real_action_amir"].sum()),
                "total_n": float(sub["real_action_anfer"].sum()),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "hla2004_diagnostic_four_scenario_summary_with_01013_auto.csv", index=False, encoding="utf-8-sig")

    long = summary.melt(id_vars=["scenario"], value_vars=["grain_yield", "biomass"], var_name="metric", value_name="kg_ha")
    fig, ax = plt.subplots(figsize=(11.5, 5.9))
    x = np.arange(len(summary))
    width = 0.36
    grain = summary["grain_yield"].to_numpy()
    bio = summary["biomass"].to_numpy()
    ax.bar(x - width / 2, grain, width=width, color="#A3BEFA", edgecolor="#2E4780", label="Grain yield / GWAD")
    ax.bar(x + width / 2, bio, width=width, color="#E2E5EA", edgecolor="#464C55", label="Biomass / CWAD-TOPWT")
    for i, row in summary.iterrows():
        ax.text(i - width / 2, row["grain_yield"] + 120, f"{row['grain_yield']:.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, row["biomass"] + 120, f"{row['biomass']:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["scenario"], rotation=18, ha="right")
    ax.set_ylabel("kg/ha")
    ax.set_title("Final grain yield and biomass by scenario", loc="left", fontsize=13, fontweight="semibold", pad=28)
    ax.text(
        0,
        1.015,
        "Diagnostic mixed-input summary; use only after checking the source-note caveat.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color=TOKENS["muted"],
    )
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, axis="y", linestyle="--", alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = OUT_DIR / "hla2004_diagnostic_four_scenario_final_yield_biomass_with_01013_auto.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def main() -> None:
    use_theme()
    df = combine_data()
    p1 = plot_combined_facets(df)
    p2 = plot_final_bars(df)
    print(f"Wrote {p1}")
    print(f"Wrote {p2}")
    print(f"Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
