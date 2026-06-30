from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2015_fixed_action_counterfactual_012_06"
OUT_DIR = IN_DIR / "figures"
FOUR_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla_2010_2015_four_scenario_with_ppo"


COLORS = {
    "fixed_I0_N0": "#464C55",
    "fixed_I120_N0": "#5477C4",
    "fixed_I120_N50": "#386411",
    "fixed_I120_N100": "#CC6F47",
    "fixed_I120_N150": "#804126",
}
LABELS = {
    "fixed_I0_N0": "I0/N0",
    "fixed_I120_N0": "I120/N0",
    "fixed_I120_N50": "I120/N50",
    "fixed_I120_N100": "I120/N100",
    "fixed_I120_N150": "I120/N150",
}


def style_axis(ax) -> None:
    ax.set_facecolor("#FFFFFF")
    ax.grid(True, axis="both", color="#E6E8F0", linewidth=0.8, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D7DBE7")
    ax.spines["bottom"].set_color("#D7DBE7")
    ax.tick_params(colors="#464C55", labelsize=10)


def load_rain() -> pd.DataFrame:
    base = pd.read_csv(FOUR_DIR / "hla_2010_2015_four_scenario_daily.csv", keep_default_na=False)
    base["scenario"] = base["scenario"].replace({"": "null_zero", "null": "null_zero"}).fillna("null_zero")
    base = base[base["requested_year"].astype(int).eq(2015) & base["scenario"].eq("null_zero")]
    return base[["dap", "rain"]].drop_duplicates("dap").sort_values("dap")


def process_plot(daily: pd.DataFrame, rain: pd.DataFrame, out_path: Path) -> None:
    scenarios = list(LABELS)
    max_dap = int(np.nanmax(daily["dap"]))
    x_max = max_dap + 5
    x_ticks = np.arange(0, x_max + 1, 25)
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(15.8, 13.2),
        sharex=True,
        gridspec_kw={"height_ratios": [0.85, 1.0, 1.0, 1.0, 1.15], "hspace": 0.26},
    )
    fig.patch.set_facecolor("#FCFCFD")
    fig.subplots_adjust(top=0.91)
    fig.text(
        0.08,
        0.965,
        "HLA 2015 fixed action counterfactual",
        ha="left",
        va="top",
        fontsize=16,
        color="#1F2430",
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.94,
        "No training. Fixed irrigation/nitrogen schedules test whether DQN had available gain under the same windows.",
        ha="left",
        va="top",
        fontsize=9.5,
        color="#6F768A",
    )

    axes[0].bar(rain["dap"], rain["rain"], width=1.0, color="#C5CAD3", edgecolor="#7A828F", linewidth=0.45)
    axes[0].set_ylabel("Rain\n(mm)")
    style_axis(axes[0])

    for ax, col, ylabel, title in [
        (axes[1], "swfac", "Water\nstress", "Water stress index"),
        (axes[2], "nstres", "Nitrogen\nstress", "Nitrogen stress index"),
    ]:
        for scenario in scenarios:
            sub = daily[daily["scenario"].eq(scenario)].sort_values("dap")
            ax.plot(sub["dap"], sub[col], color=COLORS[scenario], linewidth=2.0, label=LABELS[scenario])
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.05, max(1.05, float(daily[col].max()) * 1.15))
        ax.set_title(title, loc="left", fontsize=10)
        style_axis(ax)

    for scenario in scenarios:
        if scenario == "fixed_I0_N0":
            continue
        sub = daily[daily["scenario"].eq(scenario)].sort_values("dap")
        color = COLORS[scenario]
        irrig = sub[sub["safe_amir"].fillna(0) > 1e-6]
        fert = sub[sub["safe_anfer"].fillna(0) > 1e-6]
        if not irrig.empty:
            axes[3].vlines(irrig["dap"], 0, irrig["safe_amir"], colors=color, linewidth=2.2, alpha=0.95)
        if not fert.empty:
            axes[3].scatter(fert["dap"], fert["safe_anfer"], marker="^", s=52, color=color, edgecolor="#FFFFFF", linewidth=0.6)
    axes[3].set_ylabel("Mgmt\namount")
    axes[3].set_title("Management events: irrigation as vertical lines; fertilization as triangle markers", loc="left", fontsize=10)
    axes[3].set_ylim(-8, 65)
    style_axis(axes[3])

    for scenario in scenarios:
        sub = daily[daily["scenario"].eq(scenario)].sort_values("dap")
        axes[4].plot(sub["dap"], sub["grnwt"], color=COLORS[scenario], linewidth=2.0)
        axes[4].plot(sub["dap"], sub["topwt"], color=COLORS[scenario], linewidth=1.5, linestyle="--", alpha=0.65)
    axes[4].set_ylabel("kg/ha")
    axes[4].set_xlabel("DAP")
    axes[4].set_title("Crop outcome: solid = grain weight, dashed = aboveground biomass", loc="left", fontsize=10)
    style_axis(axes[4])

    for ax in axes:
        ax.set_xlim(-2, x_max)
        ax.set_xticks(x_ticks)
    handles = [Line2D([0], [0], color=COLORS[s], lw=2.0, label=LABELS[s]) for s in scenarios]
    axes[1].legend(handles=handles, frameon=False, loc="upper left", ncol=3, fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=280, bbox_inches="tight")
    plt.close(fig)


def summary_plot(summary: pd.DataFrame, out_path: Path) -> None:
    summary = summary.copy()
    summary["label"] = summary["scenario"].map(LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    fig.patch.set_facecolor("#FCFCFD")
    fig.suptitle("HLA 2015 fixed action counterfactual summary", x=0.06, ha="left", fontsize=15, fontweight="bold")

    colors = [COLORS[s] for s in summary["scenario"]]
    axes[0].bar(summary["label"], summary["harvest_yield_kg_ha"], color=colors, edgecolor="#333333", linewidth=0.5)
    axes[0].set_ylabel("Harvest yield (kg/ha)")
    axes[0].set_title("Yield")
    axes[0].tick_params(axis="x", rotation=35)
    style_axis(axes[0])

    axes[1].bar(summary["label"], summary["economic_reward_total"], color=colors, edgecolor="#333333", linewidth=0.5)
    axes[1].set_ylabel("Economic reward")
    axes[1].set_title("Reward = ΔGRNWT - 1.0*I - 5.0*N")
    axes[1].tick_params(axis="x", rotation=35)
    style_axis(axes[1])

    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=280, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(IN_DIR / "hla2015_fixed_action_counterfactual_daily.csv")
    summary = pd.read_csv(IN_DIR / "hla2015_fixed_action_counterfactual_summary.csv")
    rain = load_rain()
    process_plot(daily, rain, OUT_DIR / "hla2015_fixed_action_counterfactual_process.png")
    summary_plot(summary, OUT_DIR / "hla2015_fixed_action_counterfactual_summary.png")
    pd.DataFrame(
        {
            "file_type": ["process_figure", "summary_figure", "daily_csv", "summary_csv"],
            "path": [
                str(OUT_DIR / "hla2015_fixed_action_counterfactual_process.png"),
                str(OUT_DIR / "hla2015_fixed_action_counterfactual_summary.png"),
                str(IN_DIR / "hla2015_fixed_action_counterfactual_daily.csv"),
                str(IN_DIR / "hla2015_fixed_action_counterfactual_summary.csv"),
            ],
        }
    ).to_csv(OUT_DIR / "figure_manifest.csv", index=False, encoding="utf-8-sig")
    print("saved", OUT_DIR)


if __name__ == "__main__":
    main()
