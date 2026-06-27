from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

WINDOWS_RUN_DIR = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "HLA_2004"
    / "run_CNHL0407_DSSAT480_IC0_null_2004_2023"
)

OLD_ANALYSIS_DIR = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "HLA_2004"
    / "run_CNHL0405_IC0_null"
    / "analysis_windows_vs_gym_ic0_null_2004_2023"
)

OUT_DIR = WINDOWS_RUN_DIR / "analysis_windows480_vs_gym_pdi_ic0_2004_2023"
FIG_DIR = OUT_DIR / "high_contrast_figures"


def parse_table_out(path: Path) -> pd.DataFrame:
    """Parse DSSAT OUT files with repeated @ headers and *RUN sections."""
    rows: list[dict[str, object]] = []
    header: list[str] | None = None
    current_run: int | None = None
    current_treatment: str | None = None
    with path.open("r", encoding="latin1", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("*RUN"):
                parts = stripped.split()
                current_run = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                current_treatment = stripped
                header = None
                continue
            if stripped.startswith("@"):
                header = stripped.replace("@", "", 1).split()
                continue
            if not header or not stripped or stripped.startswith(("*", "!", "@")):
                continue
            parts = stripped.split()
            if len(parts) < len(header):
                continue
            rec = dict(zip(header, parts[: len(header)]))
            if current_run is not None:
                rec["RUN"] = current_run
            if current_treatment is not None:
                rec["RUN_HEADER"] = current_treatment
            rows.append(rec)
    df = pd.DataFrame(rows)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def parse_summary(path: Path) -> pd.DataFrame:
    header: list[str] | None = None
    rows: list[list[str]] = []
    with path.open("r", encoding="latin1", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("@"):
                header = stripped.replace("@", "", 1).split()
                continue
            if header and stripped and not stripped.startswith(("*", "!", "@")):
                parts = stripped.split()
                if len(parts) >= len(header):
                    rows.append(parts[: len(header)])
    if not header:
        raise ValueError(f"Cannot find Summary.OUT header in {path}")
    df = pd.DataFrame(rows, columns=header)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def make_windows_daily() -> pd.DataFrame:
    pg = parse_table_out(WINDOWS_RUN_DIR / "PlantGro.OUT")
    keep_cols = ["YEAR", "DOY", "DAS", "DAP", "LAID", "GWAD", "CWAD", "WSPD", "WSGD", "NSTD", "DTTD"]
    missing = [col for col in keep_cols if col not in pg.columns]
    if missing:
        raise ValueError(f"PlantGro.OUT missing expected columns: {missing}")
    out = pg[keep_cols].copy()
    out = out.rename(
        columns={
            "YEAR": "year",
            "DOY": "windows_doy",
            "DAS": "windows_das",
            "DAP": "dap",
            "LAID": "windows_xlai",
            "GWAD": "windows_grnwt",
            "CWAD": "windows_topwt",
            "WSPD": "windows_wspd",
            "WSGD": "windows_wsgd",
            "NSTD": "windows_nstd",
            "DTTD": "windows_dttd",
        }
    )
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["dap"] = pd.to_numeric(out["dap"], errors="coerce")
    return out


def make_rain_daily() -> pd.DataFrame:
    weather = parse_table_out(WINDOWS_RUN_DIR / "Weather.OUT")
    rain = weather[["YEAR", "DOY", "DAS", "PRED"]].copy()
    rain = rain.rename(columns={"YEAR": "year", "DOY": "doy", "DAS": "das", "PRED": "rain"})
    rain["year"] = pd.to_numeric(rain["year"], errors="coerce").astype("Int64")
    return rain


def load_gym_daily() -> pd.DataFrame:
    old = pd.read_csv(OLD_ANALYSIS_DIR / "daily_by_DAP_windows_vs_gym_IC0_null_2004_2023.csv")
    cols = [
        "year",
        "dap",
        "station",
        "gym_date",
        "gym_doy",
        "gym_topwt",
        "gym_grnwt",
        "gym_xlai",
        "gym_totir",
        "gym_tofer",
        "gym_swfac",
        "gym_nstres",
        "gym_done",
    ]
    missing = [col for col in cols if col not in old.columns]
    if missing:
        raise ValueError(f"Existing gym daily CSV missing expected columns: {missing}")
    return old[cols].copy()


def final_by_year(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["year"])
    sub = df.sort_values(["year", "dap"]).groupby("year", as_index=False).tail(1)
    if prefix == "windows":
        return sub[
            ["year", "dap", "windows_doy", "windows_topwt", "windows_grnwt", "windows_wspd", "windows_nstd"]
        ].rename(
            columns={
                "dap": "windows_final_dap",
                "windows_doy": "windows_final_doy",
                "windows_topwt": "windows_final_CWAD",
                "windows_grnwt": "windows_final_GWAD",
                "windows_wspd": "windows_final_WSPD",
                "windows_nstd": "windows_final_NSTD",
            }
        )
    return sub[["year", "dap", "gym_doy", "gym_topwt", "gym_grnwt", "gym_swfac", "gym_nstres"]].rename(
        columns={
            "dap": "gym_final_dap",
            "gym_doy": "gym_final_doy",
            "gym_topwt": "gym_final_TOPWT",
            "gym_grnwt": "gym_final_GRNWT",
            "gym_swfac": "gym_final_SWFAC",
            "gym_nstres": "gym_final_NSTRES",
        }
    )


def plot_year(df: pd.DataFrame, rain: pd.DataFrame, year: int, out_path: Path) -> None:
    sub = df[df["year"].eq(year)].sort_values("dap")
    if sub.empty:
        return
    r = rain[rain["year"].eq(year)].copy()
    if not r.empty:
        # DAS starts at 1 in Weather.OUT; PlantGro DAP starts at 0.
        r["dap"] = pd.to_numeric(r["das"], errors="coerce") - 1
    else:
        r = pd.DataFrame({"dap": [], "rain": []})

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes = axes.ravel()

    axes[0].bar(r["dap"], r["rain"], color="#56B4E9", width=1.0, label="Rainfall")
    axes[0].set_title("Rainfall / PRED")
    axes[0].set_ylabel("mm/day")
    axes[0].grid(True, axis="y", color="#D0D0D0", linewidth=0.8, alpha=0.9)
    axes[1].axis("off")

    specs = [
        ("Biomass: CWAD vs TOPWT", "windows_topwt", "gym_topwt", "kg/ha"),
        ("Grain: GWAD vs GRNWT", "windows_grnwt", "gym_grnwt", "kg/ha"),
        ("Water stress: WSPD vs SWFAC", "windows_wspd", "gym_swfac", "index"),
        ("Nitrogen stress: NSTD vs NSTRES", "windows_nstd", "gym_nstres", "index"),
    ]
    for ax, (title, win_col, gym_col, ylabel) in zip(axes[2:], specs):
        ax.plot(
            sub["dap"],
            sub[win_col],
            color="#D55E00",
            linewidth=3.0,
            linestyle="-",
            label="Windows DSSAT 4.8.0",
        )
        ax.plot(
            sub["dap"],
            sub[gym_col],
            color="#0072B2",
            linewidth=2.2,
            linestyle=(0, (3, 2)),
            marker="o",
            markersize=2.2,
            markevery=max(1, len(sub) // 18),
            label="gym/PDI DSSAT 4.8.0",
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#D0D0D0", linewidth=0.8, alpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[-2:]:
        ax.set_xlabel("DAP")
    handles, labels = axes[2].get_legend_handles_labels()
    fig.suptitle(f"HLA {year} IC=0 null: Windows DSSAT 4.8.0 vs gym/PDI DSSAT 4.8.0", fontsize=14, y=0.985)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=2, frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_difference_year(df: pd.DataFrame, year: int, out_path: Path) -> None:
    sub = df[df["year"].eq(year)].sort_values("dap")
    if sub.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes = axes.ravel()
    specs = [
        ("TOPWT/CWAD", "diff_topwt_win_minus_gym", "kg/ha"),
        ("GRNWT/GWAD", "diff_grnwt_win_minus_gym", "kg/ha"),
        ("Water stress", "diff_water_stress_like_win_minus_gym", "index"),
        ("Nitrogen stress", "diff_nitrogen_stress_like_win_minus_gym", "index"),
    ]
    for ax, (title, col, ylabel) in zip(axes, specs):
        ax.axhline(0, color="#000000", linewidth=1.1)
        ax.plot(
            sub["dap"],
            sub[col],
            color="#CC79A7",
            linewidth=2.5,
            marker="s",
            markersize=2.2,
            markevery=max(1, len(sub) // 18),
        )
        ax.set_title(f"{title} difference")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#D0D0D0", linewidth=0.8, alpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[-2:]:
        ax.set_xlabel("DAP")
    fig.suptitle(f"HLA {year}: Windows 4.8.0 minus gym/PDI 4.8.0", fontsize=14, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_annual_summary(summary: pd.DataFrame, out_path: Path) -> None:
    sub = summary.sort_values("year").copy()
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

    axes[0].plot(
        sub["year"],
        sub["windows_final_GWAD"],
        color="#D55E00",
        linewidth=2.8,
        marker="o",
        label="Windows DSSAT 4.8.0 GWAD",
    )
    axes[0].plot(
        sub["year"],
        sub["gym_final_GRNWT"],
        color="#0072B2",
        linewidth=2.4,
        linestyle=(0, (3, 2)),
        marker="s",
        label="gym/PDI DSSAT 4.8.0 GRNWT",
    )
    axes[0].set_ylabel("Grain yield kg/ha")
    axes[0].set_title("Final grain yield")

    axes[1].plot(
        sub["year"],
        sub["windows_final_CWAD"],
        color="#D55E00",
        linewidth=2.8,
        marker="o",
        label="Windows DSSAT 4.8.0 CWAD",
    )
    axes[1].plot(
        sub["year"],
        sub["gym_final_TOPWT"],
        color="#0072B2",
        linewidth=2.4,
        linestyle=(0, (3, 2)),
        marker="s",
        label="gym/PDI DSSAT 4.8.0 TOPWT",
    )
    axes[1].set_ylabel("Biomass kg/ha")
    axes[1].set_title("Final biomass")

    axes[2].axhline(0, color="#000000", linewidth=1.1)
    axes[2].bar(
        sub["year"] - 0.18,
        sub["diff_final_GWAD_windows_minus_gym"],
        width=0.36,
        color="#CC79A7",
        label="GWAD/GRNWT diff",
    )
    axes[2].bar(
        sub["year"] + 0.18,
        sub["diff_final_CWAD_windows_minus_gym"],
        width=0.36,
        color="#999999",
        label="CWAD/TOPWT diff",
    )
    axes[2].set_ylabel("Windows - gym/PDI")
    axes[2].set_xlabel("Year")
    axes[2].set_title("Final value differences")

    for ax in axes:
        ax.grid(True, color="#D0D0D0", linewidth=0.8, alpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="best", frameon=False)
    axes[2].set_xticks(sub["year"].astype(int).tolist())
    axes[2].tick_params(axis="x", rotation=45)
    fig.suptitle("HLA 2004-2023 IC=0 null: annual final outputs", fontsize=14, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    windows_summary = parse_summary(WINDOWS_RUN_DIR / "Summary.OUT")
    windows_daily = make_windows_daily()
    rain_daily = make_rain_daily()
    gym_daily = load_gym_daily()

    windows_summary.to_csv(OUT_DIR / "windows480_summary_parsed.csv", index=False, encoding="utf-8-sig")
    windows_daily.to_csv(OUT_DIR / "windows480_PlantGro_daily_parsed.csv", index=False, encoding="utf-8-sig")
    rain_daily.to_csv(OUT_DIR / "windows480_Weather_daily_parsed.csv", index=False, encoding="utf-8-sig")
    gym_daily.to_csv(OUT_DIR / "gym_pdi_daily_reused_from_prior_ic0.csv", index=False, encoding="utf-8-sig")

    usable_years = sorted(
        y
        for y in windows_summary.get("WYEAR", pd.Series(dtype=float)).dropna().astype(int).unique().tolist()
        if 2004 <= y <= 2023
    )
    win = windows_daily[windows_daily["year"].isin(usable_years)].copy()
    gym = gym_daily[gym_daily["year"].isin(usable_years)].copy()

    merged = win.merge(gym, on=["year", "dap"], how="outer", indicator=True).sort_values(["year", "dap"])
    merged["diff_topwt_win_minus_gym"] = merged["windows_topwt"] - merged["gym_topwt"]
    merged["diff_grnwt_win_minus_gym"] = merged["windows_grnwt"] - merged["gym_grnwt"]
    merged["diff_xlai_win_minus_gym"] = merged["windows_xlai"] - merged["gym_xlai"]
    merged["diff_water_stress_like_win_minus_gym"] = merged["windows_wspd"] - merged["gym_swfac"]
    merged["diff_nitrogen_stress_like_win_minus_gym"] = merged["windows_nstd"] - merged["gym_nstres"]
    for col in [
        "diff_topwt_win_minus_gym",
        "diff_grnwt_win_minus_gym",
        "diff_xlai_win_minus_gym",
        "diff_water_stress_like_win_minus_gym",
        "diff_nitrogen_stress_like_win_minus_gym",
    ]:
        merged[f"abs{col.replace('diff', 'diff')}"] = merged[col].abs()
    merged.to_csv(OUT_DIR / "daily_windows480_vs_gym_pdi_by_DAP.csv", index=False, encoding="utf-8-sig")

    summary = final_by_year(win, "windows").merge(final_by_year(gym, "gym"), on="year", how="outer", indicator=True)
    summary["diff_final_GWAD_windows_minus_gym"] = summary["windows_final_GWAD"] - summary["gym_final_GRNWT"]
    summary["diff_final_CWAD_windows_minus_gym"] = summary["windows_final_CWAD"] - summary["gym_final_TOPWT"]
    summary["diff_final_NSTD_windows_minus_gym"] = summary["windows_final_NSTD"] - summary["gym_final_NSTRES"]
    summary["diff_final_WSPD_windows_minus_gym"] = summary["windows_final_WSPD"] - summary["gym_final_SWFAC"]
    summary.to_csv(OUT_DIR / "annual_summary_windows480_vs_gym_pdi.csv", index=False, encoding="utf-8-sig")
    plot_annual_summary(summary, OUT_DIR / "annual_final_outputs_windows480_vs_gym_pdi.png")

    availability = pd.DataFrame(
        {
            "year": usable_years,
            "windows480_daily_present": [int((win["year"] == y).any()) for y in usable_years],
            "gym_pdi_daily_present": [int((gym["year"] == y).any()) for y in usable_years],
            "merged_rows": [int((merged["year"] == y).sum()) for y in usable_years],
        }
    )
    availability.to_csv(OUT_DIR / "year_availability_windows480_vs_gym_pdi.csv", index=False, encoding="utf-8-sig")

    manifest_rows: list[dict[str, object]] = []
    for year in usable_years:
        main_path = FIG_DIR / f"HLA_{year}_win480_vs_pdi480_rain.png"
        diff_path = FIG_DIR / f"HLA_{year}_win480_minus_pdi480_diff.png"
        plot_year(merged, rain_daily, year, main_path)
        plot_difference_year(merged, year, diff_path)
        manifest_rows.append({"year": year, "main_figure": str(main_path), "difference_figure": str(diff_path)})
    pd.DataFrame(manifest_rows).to_csv(FIG_DIR / "figure_manifest.csv", index=False, encoding="utf-8-sig")

    readme = [
        "# HLA 2004-2023 Windows DSSAT 4.8.0 vs gym/PDI DSSAT 4.8.0 IC=0 null",
        "",
        "This analysis replaces the previous Windows DSSAT 4.8.5 side with the newly provided Windows DSSAT 4.8.0 standalone output in `run_CNHL0407_DSSAT480_IC0_null_2004_2023`.",
        "",
        "The gym/PDI daily series is reused from the prior IC=0 null HLA 2004-2023 diagnostic CSV because it is already a complete PDI/gym DSSAT 4.8.0 post-state series.",
        "",
        f"Years plotted: {', '.join(map(str, usable_years))}",
        "",
        "Main figures include rainfall from the Windows DSSAT 4.8.0 `Weather.OUT` file and daily trajectories for biomass, grain, water-stress index, and nitrogen-stress index.",
    ]
    (OUT_DIR / "README_hla_windows480_vs_gym_pdi.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"Wrote HLA 2004-2023 Windows 4.8.0 vs gym/PDI 4.8.0 analysis to {OUT_DIR}")
    print(f"Generated {len(manifest_rows)} years x 2 figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
