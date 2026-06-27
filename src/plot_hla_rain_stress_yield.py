from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "HLA_2004"
    / "run_CNHL0407_DSSAT480_IC0_null_2004_2023"
    / "analysis_windows480_vs_gym_pdi_ic0_2004_2023"
)


def load_data() -> pd.DataFrame:
    daily = pd.read_csv(ANALYSIS_DIR / "daily_windows480_vs_gym_pdi_by_DAP.csv")
    weather = pd.read_csv(ANALYSIS_DIR / "windows480_Weather_daily_parsed.csv")
    out = daily.merge(
        weather.rename(columns={"doy": "windows_doy"})[["year", "windows_doy", "rain"]],
        on=["year", "windows_doy"],
        how="left",
    )
    out = out.rename(
        columns={
            "windows_topwt": "cwad",
            "windows_grnwt": "gwad",
            "windows_wspd": "wspd",
            "windows_nstd": "nstd",
        }
    )
    out = out.sort_values(["year", "dap"]).reset_index(drop=True)
    return out


def plot_rain_stress(data: pd.DataFrame) -> Path:
    years = sorted(data["year"].dropna().astype(int).unique().tolist())
    ncols = 4
    nrows = math.ceil(len(years) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15.5, max(3.0 * nrows, 5)), sharex=False)
    axes = list(axes.ravel())
    colors = {"rain": "#C5CAD3", "wspd": "#D62728", "nstd": "#2CA02C"}
    for ax, year in zip(axes, years):
        sub = data[data["year"].eq(year)].sort_values("dap")
        ax2 = ax.twinx()
        ax2.bar(
            sub["dap"],
            sub["rain"],
            width=1.0,
            color=colors["rain"],
            alpha=0.45,
            edgecolor="#7A828F",
            linewidth=0.25,
        )
        ax.plot(sub["dap"], sub["wspd"], color=colors["wspd"], linewidth=2.0, label="WSPD")
        ax.plot(sub["dap"], sub["nstd"], color=colors["nstd"], linewidth=2.0, linestyle=(0, (4, 2)), label="NSTD")
        ax.set_title(str(year), loc="left", fontsize=10)
        ax.set_ylim(-0.03, 1.03)
        rain_max = pd.to_numeric(sub["rain"], errors="coerce").max()
        ax2.set_ylim(0, max(20, rain_max * 1.15 if pd.notna(rain_max) else 20))
        ax.grid(True, axis="y", color="#E6E8F0", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax.tick_params(axis="both", labelsize=8)
        ax2.tick_params(axis="y", labelsize=8)
        if ax in axes[-ncols:]:
            ax.set_xlabel("DAP", fontsize=9)
    for ax in axes[len(years) :]:
        ax.axis("off")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig.legend(
        handles=[
            Patch(facecolor=colors["rain"], alpha=0.45, edgecolor="#7A828F", label="Rainfall (mm/day, right axis)"),
            Line2D([0], [0], color=colors["wspd"], lw=2.0, label="WSPD"),
            Line2D([0], [0], color=colors["nstd"], lw=2.0, linestyle=(0, (4, 2)), label="NSTD"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Hailun IC=0 null: rainfall and DSSAT stress indices", y=1.018, fontsize=14)
    fig.text(
        0.01,
        0.985,
        "Left axis: WSPD/NSTD stress indices. Right axis: rainfall. Lower WSPD/NSTD means stronger stress in DSSAT output convention.",
        fontsize=9,
        color="#6F768A",
    )
    out = ANALYSIS_DIR / "hla_rainfall_wspd_nstd_all_years.png"
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_growth_yield(data: pd.DataFrame) -> Path:
    years = sorted(data["year"].dropna().astype(int).unique().tolist())
    ncols = 4
    nrows = math.ceil(len(years) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15.5, max(3.0 * nrows, 5)), sharex=False)
    axes = list(axes.ravel())
    colors = {"cwad": "#2E4780", "gwad": "#804126"}
    for ax, year in zip(axes, years):
        sub = data[data["year"].eq(year)].sort_values("dap")
        ax.plot(sub["dap"], sub["cwad"], color=colors["cwad"], linewidth=2.1, label="CWAD")
        ax.plot(sub["dap"], sub["gwad"], color=colors["gwad"], linewidth=2.1, linestyle=(0, (4, 2)), label="GWAD")
        final_gwad = pd.to_numeric(sub["gwad"], errors="coerce").dropna()
        title = f"{year}"
        if not final_gwad.empty:
            title += f" | final GWAD={final_gwad.iloc[-1]:.0f}"
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(True, axis="y", color="#E6E8F0", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=8)
        if ax in axes[-ncols:]:
            ax.set_xlabel("DAP", fontsize=9)
    for ax in axes[len(years) :]:
        ax.axis("off")
    from matplotlib.lines import Line2D

    fig.legend(
        handles=[
            Line2D([0], [0], color=colors["cwad"], lw=2.1, label="CWAD biomass (kg/ha)"),
            Line2D([0], [0], color=colors["gwad"], lw=2.1, linestyle=(0, (4, 2)), label="GWAD grain yield (kg/ha)"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("Hailun IC=0 null: biomass and grain trajectories", y=1.018, fontsize=14)
    out = ANALYSIS_DIR / "hla_cwad_gwad_all_years.png"
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    data = load_data()
    data.to_csv(ANALYSIS_DIR / "hla_daily_rain_stress_growth_for_plots.csv", index=False, encoding="utf-8-sig")
    print(plot_rain_stress(data))
    print(plot_growth_yield(data))


if __name__ == "__main__":
    main()
