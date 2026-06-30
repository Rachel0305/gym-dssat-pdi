"""Alternative split-style plots for HLA 2007/2009 management-space check.

This plotting-only script reads existing CSV/OUT files and generates, for each
year, two figures:
1) rainfall + irrigation + fertilizer + water/nitrogen stress
2) grain yield and biomass

No DSSAT/PDI simulation is run.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla_2007_2009_management_space_check"
FIG_DIR = OUT_DIR / "figures_split_style"
FIG_DIR.mkdir(parents=True, exist_ok=True)

YEAR_TO_PDATE_DOY = {2007: 125, 2009: 122}
YEAR_TO_WTH = {
    2007: OUT_DIR / "runs" / "2007_null" / "input" / "CNHL0701.WTH",
    2009: OUT_DIR / "runs" / "2009_null" / "input" / "CNHL0901.WTH",
}
SCENARIOS = ["null", "recorded", "dssat_auto"]
LABELS = {"null": "Null", "recorded": "Recorded", "dssat_auto": "DSSAT auto"}
COLORS = {"null": "#111111", "recorded": "#2E4780", "dssat_auto": "#CC6F47"}
LINESTYLES = {"null": "-", "recorded": "--", "dssat_auto": "-."}

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
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


def add_header(fig, ax, title: str, subtitle: str) -> None:
    left = ax.get_position().x0
    fig.text(left, 0.985, textwrap.fill(title, 90), ha="left", va="top", fontsize=13, fontweight="semibold", color=TOKENS["ink"])
    fig.text(left, 0.94, textwrap.fill(subtitle, 120), ha="left", va="top", fontsize=9, color=TOKENS["muted"])


def read_weather_rain(path: Path, year: int) -> pd.DataFrame:
    header = None
    rows = []
    for raw in path.read_text(encoding="latin-1", errors="ignore").splitlines():
        stripped = raw.strip()
        if stripped.startswith("@") and "DATE" in stripped and "RAIN" in stripped:
            header = stripped.replace("@", "", 1).split()
            continue
        if not header or not stripped or not stripped[0].isdigit():
            continue
        parts = stripped.split()
        row = dict(zip(header, parts))
        date = int(row["DATE"])
        if date // 1000 != year:
            continue
        doy = date % 1000
        rows.append({"year": year, "doy": doy, "dap": doy - YEAR_TO_PDATE_DOY[year], "rain": float(row.get("RAIN", 0.0))})
    return pd.DataFrame(rows)


def parse_events_from_raw(year: int, scenario: str) -> pd.DataFrame:
    path = OUT_DIR / "runs" / f"{year}_{scenario}" / "pdi_tmp_snapshot" / "MgmtEvent.OUT"
    rows = []
    if not path.exists():
        return pd.DataFrame(columns=["year", "scenario", "dap", "operation", "amount", "unit"])
    pattern = re.compile(
        r"^\s*\d+\s+\w+\s+\d+,\s+\d{4}\s+\d+\s+\d+\s+(-?\d+)\s+\w+\s+(.+?)\s+([-+]?\d+(?:\.\d*)?)\s+(\S+)"
    )
    for raw in path.read_text(encoding="latin-1", errors="ignore").splitlines():
        if "Irrigation" not in raw and "Fertil" not in raw and "Nitrogen" not in raw:
            continue
        m = pattern.match(raw)
        if not m:
            continue
        rows.append(
            {
                "year": year,
                "scenario": scenario,
                "dap": int(m.group(1)),
                "operation": m.group(2).strip(),
                "amount": float(m.group(3)),
                "unit": m.group(4),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["year", "scenario", "dap", "operation", "amount", "unit"])
    return pd.DataFrame(rows).drop_duplicates(["year", "scenario", "dap", "operation", "amount", "unit"], keep="first")


def load_events() -> pd.DataFrame:
    frames = [parse_events_from_raw(year, scenario) for year in YEAR_TO_PDATE_DOY for scenario in SCENARIOS]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["year", "scenario", "dap", "operation", "amount", "unit"])
    return pd.concat(frames, ignore_index=True)


def plot_process(year: int, daily: pd.DataFrame, events: pd.DataFrame) -> Path:
    use_theme()
    rain = read_weather_rain(YEAR_TO_WTH[year], year)
    sub_year = daily[daily["year"].eq(year)].copy()
    max_dap = float(sub_year["dap"].max())

    fig, ax1 = plt.subplots(figsize=(14, 7.2))
    fig.subplots_adjust(top=0.82, right=0.88)
    ax2 = ax1.twinx()

    # Rainfall and management are on the left axis (mm or kg/ha).
    ax1.bar(rain["dap"], rain["rain"], color="#C5CAD3", edgecolor="#464C55", linewidth=0.25, width=1.0, label="Rainfall")
    for scenario in SCENARIOS:
        ev = events[(events["year"].eq(year)) & (events["scenario"].eq(scenario))]
        if ev.empty:
            continue
        color = COLORS[scenario]
        irrig = ev[ev["operation"].str.contains("Irrigation", case=False, na=False)]
        fert = ev[ev["operation"].str.contains("Fertil|Nitrogen", case=False, na=False)]
        if not irrig.empty:
            ax1.vlines(irrig["dap"], 0, irrig["amount"], color=color, linestyle=LINESTYLES[scenario], linewidth=2.5, alpha=0.95, label=f"{LABELS[scenario]} irrigation")
        if not fert.empty:
            ax1.scatter(fert["dap"], fert["amount"], color=color, marker="^", s=70, edgecolor="white", linewidth=0.5, label=f"{LABELS[scenario]} N")

    # Stress indices are on the right axis.
    for scenario in SCENARIOS:
        sub = sub_year[sub_year["scenario"].eq(scenario)].sort_values("dap")
        if sub.empty:
            continue
        color = COLORS[scenario]
        ax2.plot(sub["dap"], sub["wspd"], color=color, linestyle=LINESTYLES[scenario], linewidth=2.2, label=f"{LABELS[scenario]} WSPD")
        ax2.plot(sub["dap"], sub["nstd"], color=color, linestyle=":", linewidth=2.5, label=f"{LABELS[scenario]} NSTD")

    ax1.set_xlim(0, max_dap + 2)
    ax1.set_xlabel("DAP")
    ax1.set_ylabel("Rainfall / management amount")
    ax2.set_ylabel("Stress index")
    ax2.set_ylim(-0.03, 1.05)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.45)
    ax2.grid(False)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    # Keep legend compact; process chart is intentionally dense.
    ax1.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.02), ncol=4, fontsize=8)
    add_header(
        fig,
        ax1,
        f"HLA {year}: rainfall, management events, and stress indices",
        "Bars show rainfall; vertical lines show irrigation; triangles show fertilizer; solid/dashed/dash-dot lines show WSPD and dotted lines show NSTD.",
    )
    path = FIG_DIR / f"hla_{year}_split_process_rain_mgmt_stress.png"
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_yield(year: int, daily: pd.DataFrame) -> Path:
    use_theme()
    sub_year = daily[daily["year"].eq(year)].copy()
    max_dap = float(sub_year["dap"].max())
    fig, ax = plt.subplots(figsize=(14, 6.8))
    fig.subplots_adjust(top=0.82)
    for scenario in SCENARIOS:
        sub = sub_year[sub_year["scenario"].eq(scenario)].sort_values("dap")
        if sub.empty:
            continue
        color = COLORS[scenario]
        ls = LINESTYLES[scenario]
        label = LABELS[scenario]
        ax.plot(sub["dap"], sub["gwad"], color=color, linestyle=ls, linewidth=2.4, label=f"{label} GWAD")
        ax.plot(sub["dap"], sub["cwad"], color=color, linestyle=":", linewidth=2.5, alpha=0.95, label=f"{label} CWAD")
        final = sub.iloc[-1]
        ax.text(final["dap"] + 1, final["gwad"], f"{final['gwad']:.0f}", color=color, fontsize=8, va="center")

    ax.set_xlim(0, max_dap + 10)
    ax.set_xlabel("DAP")
    ax.set_ylabel("kg/ha")
    ax.grid(True, axis="y", linestyle="--", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.02), ncol=3, fontsize=9)
    add_header(
        fig,
        ax,
        f"HLA {year}: grain yield and biomass trajectories",
        "GWAD is grain yield; CWAD is aboveground biomass. Dotted lines are biomass; non-dotted lines are grain yield.",
    )
    path = FIG_DIR / f"hla_{year}_split_yield_biomass.png"
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    daily = pd.read_csv(OUT_DIR / "hla_2007_2009_management_space_daily.csv", keep_default_na=False)
    events = load_events()
    events.to_csv(OUT_DIR / "hla_2007_2009_management_space_events_deduplicated.csv", index=False, encoding="utf-8-sig")
    paths = []
    for year in YEAR_TO_PDATE_DOY:
        paths.append(plot_process(year, daily, events))
        paths.append(plot_yield(year, daily))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
