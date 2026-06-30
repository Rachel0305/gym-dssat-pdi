"""Compare HLA null simulations under IC=0 and IC=1 for 2004-2023.

This script is intentionally offline-only: it reads existing DSSAT/PDI outputs,
creates comparison CSV files and static figures, and does not run DSSAT or PPO.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IC0_DIR = ROOT / "DSSAT_auto_validation" / "HLA_2004" / "run_CNHL0407_DSSAT480_IC0_null_2004_2023"
IC1_DIR = ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla_ic1_yearly_diagnostics_2004_2023" / "null"
OUT_DIR = ROOT / "DSSAT_auto_validation" / "HLA_2004" / "analysis_hla_ic0_vs_ic1_null_2004_2023"


TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}


def parse_dssat_table(path: Path) -> pd.DataFrame:
    """Parse DSSAT whitespace table sections with @ headers."""
    rows: list[list[str]] = []
    columns: list[str] | None = None
    with path.open("r", encoding="latin-1", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("@"):
                columns = stripped.split()
                if columns and columns[0] == "@":
                    columns = columns[1:]
                elif columns and columns[0].startswith("@"):
                    columns[0] = columns[0].lstrip("@")
                continue
            if columns is None:
                continue
            if stripped.startswith("*") or stripped.startswith("!") or stripped.startswith("@"):
                continue
            parts = stripped.split()
            if not parts:
                continue
            if not parts[0].lstrip("-").isdigit():
                continue
            if len(parts) < len(columns):
                parts = parts + [""] * (len(columns) - len(parts))
            elif len(parts) > len(columns):
                parts = parts[: len(columns)]
            rows.append(parts)
    df = pd.DataFrame(rows, columns=columns or [])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def standardize_daily(df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = df.copy()
    if "WSPD" in out.columns:
        out["wspd"] = pd.to_numeric(out["WSPD"], errors="coerce")
    elif "swfac" in out.columns:
        out["wspd"] = pd.to_numeric(out["swfac"], errors="coerce")
    else:
        out["wspd"] = pd.NA

    if "NSTD" in out.columns:
        out["nstd"] = pd.to_numeric(out["NSTD"], errors="coerce")
    elif "nstres" in out.columns:
        out["nstd"] = pd.to_numeric(out["nstres"], errors="coerce")
    else:
        out["nstd"] = pd.NA

    if "GWAD" in out.columns:
        out["gwad"] = pd.to_numeric(out["GWAD"], errors="coerce")
    elif "grnwt" in out.columns:
        out["gwad"] = pd.to_numeric(out["grnwt"], errors="coerce")
    else:
        out["gwad"] = pd.NA

    if "CWAD" in out.columns:
        out["cwad"] = pd.to_numeric(out["CWAD"], errors="coerce")
    elif "topwt" in out.columns:
        out["cwad"] = pd.to_numeric(out["topwt"], errors="coerce")
    else:
        out["cwad"] = pd.NA

    if "DAP" in out.columns:
        out["dap"] = pd.to_numeric(out["DAP"], errors="coerce")
    elif "dap" in out.columns:
        out["dap"] = pd.to_numeric(out["dap"], errors="coerce")
    else:
        out["dap"] = pd.NA

    if "YEAR" in out.columns:
        out["year"] = pd.to_numeric(out["YEAR"], errors="coerce")
    elif "year" in out.columns:
        out["year"] = pd.to_numeric(out["year"], errors="coerce")
    else:
        out["year"] = pd.NA

    keep = ["source", "year", "dap", "wspd", "nstd", "gwad", "cwad"]
    out["source"] = source
    return out[keep].dropna(subset=["dap"])


def load_ic0_daily() -> pd.DataFrame:
    plant = standardize_daily(parse_dssat_table(IC0_DIR / "PlantGro.OUT"), "IC0_null")
    return plant


def load_ic1_daily(years: Iterable[int]) -> pd.DataFrame:
    frames = []
    for year in years:
        path = IC1_DIR / str(year) / "pdi_tmp_snapshot" / "PlantGro.OUT"
        if path.exists():
            df = standardize_daily(parse_dssat_table(path), "IC1_null")
            # Some PDI snapshots duplicate summary/output blocks; daily rows should be unique by year/DAP.
            df = df.drop_duplicates(subset=["source", "year", "dap"], keep="last")
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_summary(path: Path, source: str) -> pd.DataFrame:
    df = parse_dssat_table(path)
    if df.empty:
        return df
    for col in ["HYEAR", "HWAM", "CWAM", "MDAT", "PRCP", "ETCP", "EPCP", "ESCP"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop_duplicates(subset=["HYEAR"], keep="last")
    out = pd.DataFrame(
        {
            "source": source,
            "year": df.get("HYEAR"),
            "grain_yield_hwam": df.get("HWAM"),
            "biomass_cwam": df.get("CWAM"),
            "maturity_yrdoy": df.get("MDAT"),
            "season_prcp": df.get("PRCP"),
            "season_et": df.get("ETCP"),
            "season_ep": df.get("EPCP"),
            "season_es": df.get("ESCP"),
        }
    )
    return out.dropna(subset=["year"])


def load_ic1_summary(years: Iterable[int]) -> pd.DataFrame:
    frames = []
    for year in years:
        path = IC1_DIR / str(year) / "pdi_tmp_snapshot" / "Summary.OUT"
        if path.exists():
            frames.append(load_summary(path, "IC1_null"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def daily_final_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """Build a reliable final-state summary from daily PlantGro rows."""
    rows = []
    for (source, year), sub in daily.dropna(subset=["year"]).groupby(["source", "year"]):
        sub = sub.sort_values("dap")
        final = sub.iloc[-1]
        rows.append(
            {
                "source": source,
                "year": int(year),
                "grain_yield_hwam": final["gwad"],
                "biomass_cwam": final["cwad"],
                "final_dap": final["dap"],
                "max_wspd": sub["wspd"].max(),
                "mean_wspd": sub["wspd"].mean(),
                "max_nstd": sub["nstd"].max(),
                "mean_nstd": sub["nstd"].mean(),
            }
        )
    return pd.DataFrame(rows)


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(TOKENS["panel"])
    ax.grid(True, axis="y", color=TOKENS["grid"], linewidth=0.8, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])
    ax.tick_params(colors=TOKENS["muted"])
    ax.xaxis.label.set_color(TOKENS["ink"])
    ax.yaxis.label.set_color(TOKENS["ink"])


def plot_yield(summary_wide: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=TOKENS["surface"])
    ax.plot(
        summary_wide["year"],
        summary_wide["grain_yield_hwam_IC0"],
        color="#2E4780",
        marker="o",
        linewidth=1.8,
        label="IC=0 null",
    )
    ax.plot(
        summary_wide["year"],
        summary_wide["grain_yield_hwam_IC1"],
        color="#CC6F47",
        marker="s",
        linewidth=1.8,
        linestyle="--",
        label="IC=1 null",
    )
    for year in [2004, 2012]:
        ax.axvline(year, color="#7A828F", linestyle=":", linewidth=1)
    ax.set_title("HLA null yield comparison: IC=0 vs IC=1", color=TOKENS["ink"], loc="left")
    ax.set_xlabel("Year")
    ax.set_ylabel("Grain yield HWAM (kg/ha)")
    ax.legend(frameon=False, loc="upper left")
    style_axes(ax)
    fig.tight_layout()
    path = OUT_DIR / "hla_ic0_vs_ic1_null_yield_2004_2023.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_focus_year(daily: pd.DataFrame, year: int) -> Path:
    sub = daily[daily["year"] == year].copy()
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, facecolor=TOKENS["surface"])
    configs = [
        ("wspd", "Water stress WSPD", "Stress index"),
        ("nstd", "Nitrogen stress NSTD", "Stress index"),
        ("gwad", "Grain weight GWAD", "kg/ha"),
    ]
    styles = {
        "IC0_null": {"color": "#2E4780", "linestyle": "-", "marker": None, "label": "IC=0 null"},
        "IC1_null": {"color": "#CC6F47", "linestyle": "--", "marker": None, "label": "IC=1 null"},
    }
    for ax, (metric, title, ylabel) in zip(axes, configs):
        for source, sdf in sub.groupby("source"):
            style = styles.get(source, {})
            ax.plot(
                sdf["dap"],
                sdf[metric],
                color=style.get("color"),
                linestyle=style.get("linestyle", "-"),
                linewidth=1.8,
                label=style.get("label", source),
            )
        ax.set_ylabel(ylabel)
        ax.set_title(title, color=TOKENS["ink"], loc="left", fontsize=11)
        style_axes(ax)
    axes[-1].set_xlabel("DAP")
    axes[0].legend(frameon=False, loc="upper left", ncol=2)
    fig.suptitle(f"HLA {year} null daily comparison: IC=0 vs IC=1", x=0.01, ha="left", color=TOKENS["ink"])
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = OUT_DIR / f"hla_{year}_ic0_vs_ic1_null_daily_stress_yield.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    years = list(range(2004, 2024))
    ic0_daily = load_ic0_daily()
    ic1_daily = load_ic1_daily(years)
    daily = pd.concat([ic0_daily, ic1_daily], ignore_index=True)
    daily.to_csv(OUT_DIR / "hla_ic0_vs_ic1_null_daily_values_2004_2023.csv", index=False)

    # Summary.OUT can contain blank fixed-width fields that are unsafe to parse
    # by whitespace for some IC=1 snapshots. For this comparison, use the final
    # daily PlantGro state as the canonical final yield/biomass summary.
    summary = daily_final_summary(daily)
    summary.to_csv(OUT_DIR / "hla_ic0_vs_ic1_null_summary_long_2004_2023.csv", index=False)

    wide = summary.pivot_table(
        index="year",
        columns="source",
        values=["grain_yield_hwam", "biomass_cwam", "final_dap", "max_wspd", "mean_wspd", "max_nstd", "mean_nstd"],
        aggfunc="last",
    )
    wide.columns = [f"{metric}_{source.replace('_null', '')}" for metric, source in wide.columns]
    wide = wide.reset_index()
    if {"grain_yield_hwam_IC0", "grain_yield_hwam_IC1"}.issubset(wide.columns):
        wide["yield_diff_ic1_minus_ic0"] = wide["grain_yield_hwam_IC1"] - wide["grain_yield_hwam_IC0"]
        wide["yield_ratio_ic1_to_ic0"] = wide["grain_yield_hwam_IC1"] / wide["grain_yield_hwam_IC0"].replace(0, pd.NA)
    if {"biomass_cwam_IC0", "biomass_cwam_IC1"}.issubset(wide.columns):
        wide["biomass_diff_ic1_minus_ic0"] = wide["biomass_cwam_IC1"] - wide["biomass_cwam_IC0"]
    wide.to_csv(OUT_DIR / "hla_ic0_vs_ic1_null_summary_wide_2004_2023.csv", index=False)

    paths = [plot_yield(wide)]
    for year in [2004, 2012]:
        paths.append(plot_focus_year(daily, year))

    print("Wrote:")
    for path in [
        OUT_DIR / "hla_ic0_vs_ic1_null_daily_values_2004_2023.csv",
        OUT_DIR / "hla_ic0_vs_ic1_null_summary_long_2004_2023.csv",
        OUT_DIR / "hla_ic0_vs_ic1_null_summary_wide_2004_2023.csv",
        *paths,
    ]:
        print(path)

    print("\nKey rows:")
    print(wide[wide["year"].isin([2004, 2012])].to_string(index=False))


if __name__ == "__main__":
    main()
