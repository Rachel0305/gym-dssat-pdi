from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "yc2014_formal_four_scenario_015_06" / "seed0_seed1_best"
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "yc2014_formal_four_scenario_016_10" / "seed0_best"

DAILY_CSV = BASE_DIR / "015_06_yc2014_formal_four_scenario_daily.csv"
SUMMARY_CSV = BASE_DIR / "015_06_yc2014_formal_four_scenario_summary.csv"
EVENTS_CSV = BASE_DIR / "015_06_yc2014_formal_four_scenario_management_events.csv"

FIG_PATH = OUT_DIR / "figures" / "yc2014_four_scenario_process_nature_style.png"
SVG_PATH = FIG_PATH.with_suffix(".svg")
PDF_PATH = FIG_PATH.with_suffix(".pdf")

SCENARIO_ORDER = ["null", "recorded", "dssat_auto", "dqn_best"]
SCENARIO_LABELS = {
    "null": "Null",
    "recorded": "Recorded expert",
    "dssat_auto": "DSSAT auto",
    "dqn_best": "DQN best checkpoint",
}
SCENARIO_COLORS = {
    "null": "#3F3F3F",
    "recorded": "#C73E3A",
    "dssat_auto": "#B8860B",
    "dqn_best": "#2E8B57",
}
SCENARIO_LINESTYLES = {
    "null": "-",
    "recorded": "--",
    "dssat_auto": "-",
    "dqn_best": "-",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 9,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(DAILY_CSV)
    summary = pd.read_csv(SUMMARY_CSV)
    events = pd.read_csv(EVENTS_CSV)

    daily["scenario"] = daily["scenario"].fillna("null")
    summary["scenario"] = summary["scenario"].fillna("null")
    events["scenario"] = events["scenario"].fillna("null")

    # Only plot the representative DQN line used in the formal four-scenario figure.
    dqn_daily = daily[daily["scenario"].eq("dqn_best")].copy()
    if "seed" in dqn_daily.columns:
        dqn_daily = dqn_daily[dqn_daily["seed"].fillna(-1).eq(0)]
    if "checkpoint_step" in dqn_daily.columns:
        dqn_daily = dqn_daily[dqn_daily["checkpoint_step"].fillna(-1).eq(5000)]

    base_daily = daily[daily["scenario"].isin(["null", "recorded", "dssat_auto"])].copy()
    daily_plot = pd.concat([base_daily, dqn_daily], ignore_index=True)

    dqn_events = events[events["scenario"].eq("dqn_best")].copy()
    if "seed" in dqn_events.columns:
        dqn_events = dqn_events[dqn_events["seed"].fillna(-1).eq(0)]
    if "checkpoint_step" in dqn_events.columns:
        dqn_events = dqn_events[dqn_events["checkpoint_step"].fillna(-1).eq(5000)]

    base_events = events[events["scenario"].isin(["null", "recorded", "dssat_auto"])].copy()
    event_plot = pd.concat([base_events, dqn_events], ignore_index=True)

    dqn_summary = summary[summary["scenario"].eq("dqn_best")].copy()
    if "seed" in dqn_summary.columns:
        dqn_summary = dqn_summary[dqn_summary["seed"].fillna(-1).eq(0)]
    if "checkpoint_step" in dqn_summary.columns:
        dqn_summary = dqn_summary[dqn_summary["checkpoint_step"].fillna(-1).eq(5000)]
    base_summary = summary[summary["scenario"].isin(["null", "recorded", "dssat_auto"])].copy()
    summary_plot = pd.concat([base_summary, dqn_summary], ignore_index=True)

    return daily_plot, summary_plot, event_plot


def make_style_handles():
    return [
        Line2D(
            [0],
            [0],
            color=SCENARIO_COLORS[s],
            lw=2.2,
            ls=SCENARIO_LINESTYLES[s],
            label=SCENARIO_LABELS[s],
        )
        for s in SCENARIO_ORDER
    ]


def plot() -> None:
    daily, summary, events = load_inputs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        6,
        1,
        figsize=(14.8, 12.8),
        sharex=True,
        gridspec_kw={"height_ratios": [0.7, 1.0, 1.0, 1.0, 1.15, 1.0], "hspace": 0.22},
    )

    # a) rainfall
    rain = daily[["dap", "rain"]].drop_duplicates("dap").sort_values("dap")
    axes[0].bar(
        rain["dap"],
        rain["rain"],
        color="#C9CED8",
        edgecolor="#AEB6C2",
        linewidth=0.4,
        width=0.9,
        label="Rainfall",
    )
    axes[0].set_ylabel("Rain\n(mm)")
    axes[0].set_title("YC2014 formal four-scenario process plot", loc="left", fontsize=15, fontweight="bold", pad=8)
    axes[0].legend(handles=[Patch(facecolor="#C9CED8", edgecolor="#AEB6C2", label="Rainfall")], loc="upper left")

    # b/c/e/f lines
    line_handles = make_style_handles()
    for scenario in SCENARIO_ORDER:
        sub = daily[daily["scenario"].eq(scenario)].sort_values("dap")
        if sub.empty:
            continue
        color = SCENARIO_COLORS[scenario]
        ls = SCENARIO_LINESTYLES[scenario]
        z = 5 if scenario == "dqn_best" else 4 if scenario == "recorded" else 3

        axes[1].plot(sub["dap"], sub["swfac"], color=color, ls=ls, lw=2.2, zorder=z)
        axes[2].plot(sub["dap"], sub["nstres"], color=color, ls=ls, lw=2.2, zorder=z)
        axes[4].plot(sub["dap"], sub["grnwt"], color=color, ls=ls, lw=2.2, zorder=z)
        axes[4].plot(sub["dap"], sub["topwt"], color=color, ls=":", lw=1.8, alpha=0.95, zorder=z)
        axes[5].plot(sub["dap"], sub["reward_proxy"], color=color, ls=ls, lw=2.2, zorder=z)

    axes[1].set_ylabel("Water\nstress")
    axes[1].set_title("Water stress index by scenario", loc="left", fontsize=10)
    axes[1].legend(handles=line_handles, loc="upper left", ncol=2, fontsize=9)

    axes[2].set_ylabel("Nitrogen\nstress")
    axes[2].set_title("Nitrogen stress index by scenario", loc="left", fontsize=10)

    # d) management
    for scenario in SCENARIO_ORDER:
        sube = events[events["scenario"].eq(scenario)].copy()
        if sube.empty:
            continue
        color = SCENARIO_COLORS[scenario]
        ls = SCENARIO_LINESTYLES[scenario]

        irr = sube[sube["operation"].eq("Irrigation")]
        fert = sube[sube["operation"].eq("Fertilizer")]
        if not irr.empty:
            axes[3].vlines(
                irr["dap"],
                0,
                irr["amount"],
                colors=color,
                linestyles=ls,
                linewidth=2.4,
                alpha=0.95,
            )
        if not fert.empty:
            axes[3].scatter(
                fert["dap"],
                fert["amount"],
                marker="^",
                s=65,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                zorder=5,
            )

    axes[3].set_ylabel("Mgmt\namount")
    axes[3].set_title("Management events: irrigation as vertical lines; fertilization as triangle markers", loc="left", fontsize=10)
    axes[3].legend(handles=line_handles, loc="upper left", ncol=2, fontsize=9)

    # e/f labels
    axes[4].set_ylabel("kg/ha")
    axes[4].set_title("Crop outcome: solid = grain yield, dotted = aboveground biomass", loc="left", fontsize=10)

    axes[5].set_ylabel("Cum.\nreward")
    axes[5].set_title("Cumulative reward proxy: grain increment - 1×irrigation - 5×fertilizer", loc="left", fontsize=10)
    axes[5].legend(handles=line_handles, loc="upper left", ncol=2, fontsize=9)
    axes[5].set_xlabel("DAP")

    # cosmetics
    max_dap = float(daily["dap"].max()) if not daily.empty else 110
    for ax in axes:
        ax.grid(True, color="#E6E8F0", linewidth=0.8, alpha=0.9)
        ax.set_xlim(0, max_dap + 2)
    axes[1].set_ylim(bottom=-0.02)
    axes[2].set_ylim(bottom=-0.02)
    axes[3].set_ylim(bottom=-10)

    fig.subplots_adjust(top=0.97, bottom=0.06, left=0.07, right=0.985, hspace=0.22)
    fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
    fig.savefig(SVG_PATH, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plot()
    print(FIG_PATH)


if __name__ == "__main__":
    main()
