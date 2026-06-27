from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


CONFIGS = {
    "YC": {
        "label": "Yucheng",
        "run_dir": PROJECT_ROOT / "DSSAT_auto_validation" / "run_CNYC0802_DSSAT480_IC0_null_2000_2023",
        "analysis_dir": PROJECT_ROOT
        / "DSSAT_auto_validation"
        / "run_CNYC0802_DSSAT480_IC0_null_2000_2023"
        / "analysis_windows480_vs_pdi_yearly_ic0",
        "comparison_csv": "daily_windows480_vs_pdi_yearly_by_DAP.csv",
        "years": list(range(2008, 2024)),
    },
    "FQZ": {
        "label": "Fengqiu",
        "run_dir": PROJECT_ROOT
        / "DSSAT_auto_validation"
        / "HLA_2004"
        / "run_CNFQ0802_DSSAT480_standalone"
        / "run_CNFQ0802_DSSAT480_standalone",
        "analysis_dir": PROJECT_ROOT
        / "DSSAT_auto_validation"
        / "HLA_2004"
        / "run_CNFQ0802_DSSAT480_standalone"
        / "run_CNFQ0802_DSSAT480_standalone"
        / "analysis_windows480_vs_pdi_yearly_ic0",
        "comparison_csv": "daily_windows480_vs_pdi_yearly_by_DAP.csv",
        "years": list(range(2007, 2024)),
    },
}


def parse_weather_out(path: Path) -> pd.DataFrame:
    header: list[str] | None = None
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="latin1", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("@"):
                header = stripped.replace("@", "", 1).split()
                continue
            if header and re.match(r"^\d{4}\s+\d+", stripped):
                parts = stripped.split()
                if len(parts) >= len(header):
                    rows.append(dict(zip(header, parts[: len(header)])))
    df = pd.DataFrame(rows)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    keep = [c for c in ["YEAR", "DOY", "DAS", "PRED"] if c in df.columns]
    df = df[keep].rename(columns={"YEAR": "year", "DOY": "doy", "DAS": "das", "PRED": "rain"})
    return df


def load_station_data(config: dict[str, Any]) -> pd.DataFrame:
    daily = pd.read_csv(config["analysis_dir"] / config["comparison_csv"])
    # The comparison CSV uses uppercase DSSAT variable names with suffixes.
    rename = {
        "CWAD_windows": "cwad",
        "GWAD_windows": "gwad",
        "WSPD_windows": "wspd",
        "NSTD_windows": "nstd",
    }
    daily = daily.rename(columns=rename)
    weather = parse_weather_out(config["run_dir"] / "Weather.OUT")
    out = daily.merge(weather[["year", "doy", "rain"]], on=["year", "doy"], how="left")
    out = out[out["year"].isin(config["years"])].copy()
    out = out.sort_values(["year", "dap"]).reset_index(drop=True)
    return out


def plot_rain_stress(station: str, config: dict[str, Any], data: pd.DataFrame) -> Path:
    years = [y for y in config["years"] if y in set(data["year"].dropna().astype(int))]
    ncols = 4
    nrows = math.ceil(len(years) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15.5, max(3.0 * nrows, 5)), sharex=False)
    axes = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]
    colors = {
        "rain": "#C5CAD3",
        "wspd": "#D62728",  # red
        "nstd": "#2CA02C",  # green
    }
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
            label="Rainfall",
            zorder=1,
        )
        ax.plot(sub["dap"], sub["wspd"], color=colors["wspd"], linewidth=2.0, label="WSPD", zorder=4)
        ax.plot(sub["dap"], sub["nstd"], color=colors["nstd"], linewidth=2.0, linestyle=(0, (4, 2)), label="NSTD", zorder=5)
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

    legend_items = [
        Patch(facecolor=colors["rain"], alpha=0.45, edgecolor="#7A828F", label="Rainfall (mm/day, right axis)"),
        Line2D([0], [0], color=colors["wspd"], lw=2.0, label="WSPD"),
        Line2D([0], [0], color=colors["nstd"], lw=2.0, linestyle=(0, (4, 2)), label="NSTD"),
    ]
    fig.legend(handles=legend_items, loc="upper center", bbox_to_anchor=(0.5, 0.992), ncol=3, frameon=False)
    fig.suptitle(f"{config['label']} IC=0 null: rainfall and DSSAT stress indices", y=1.018, fontsize=14)
    fig.text(
        0.01,
        0.985,
        "Left axis: WSPD/NSTD stress indices. Right axis: rainfall. Lower WSPD/NSTD means stronger stress in DSSAT output convention.",
        fontsize=9,
        color="#6F768A",
    )
    out_path = config["analysis_dir"] / f"{station.lower()}_rainfall_wspd_nstd_all_years.png"
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_growth_yield(station: str, config: dict[str, Any], data: pd.DataFrame) -> Path:
    years = [y for y in config["years"] if y in set(data["year"].dropna().astype(int))]
    ncols = 4
    nrows = math.ceil(len(years) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15.5, max(3.0 * nrows, 5)), sharex=False)
    axes = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]
    colors = {
        "cwad": "#2E4780",
        "gwad": "#804126",
    }
    for ax, year in zip(axes, years):
        sub = data[data["year"].eq(year)].sort_values("dap")
        ax.plot(sub["dap"], sub["cwad"], color=colors["cwad"], linewidth=2.1, label="CWAD")
        ax.plot(sub["dap"], sub["gwad"], color=colors["gwad"], linewidth=2.1, linestyle=(0, (4, 2)), label="GWAD")
        final_gwad = pd.to_numeric(sub["gwad"], errors="coerce").dropna()
        label = f"{year}"
        if not final_gwad.empty:
            label += f" | final GWAD={final_gwad.iloc[-1]:.0f}"
        ax.set_title(label, loc="left", fontsize=10)
        ax.grid(True, axis="y", color="#E6E8F0", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=8)
        if ax in axes[-ncols:]:
            ax.set_xlabel("DAP", fontsize=9)
    for ax in axes[len(years) :]:
        ax.axis("off")
    from matplotlib.lines import Line2D

    legend_items = [
        Line2D([0], [0], color=colors["cwad"], lw=2.1, label="CWAD biomass (kg/ha)"),
        Line2D([0], [0], color=colors["gwad"], lw=2.1, linestyle=(0, (4, 2)), label="GWAD grain yield (kg/ha)"),
    ]
    fig.legend(handles=legend_items, loc="upper center", bbox_to_anchor=(0.5, 0.992), ncol=2, frameon=False)
    fig.suptitle(f"{config['label']} IC=0 null: biomass and grain trajectories", y=1.018, fontsize=14)
    out_path = config["analysis_dir"] / f"{station.lower()}_cwad_gwad_all_years.png"
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    outputs = []
    for station, config in CONFIGS.items():
        data = load_station_data(config)
        data.to_csv(config["analysis_dir"] / f"{station.lower()}_daily_rain_stress_growth_for_plots.csv", index=False, encoding="utf-8-sig")
        outputs.append(str(plot_rain_stress(station, config, data)))
        outputs.append(str(plot_growth_yield(station, config, data)))
    print("\n".join(outputs))


if __name__ == "__main__":
    main()
