from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "HLA_2004"
    / "run_CNHL0405_IC0_null"
    / "analysis_windows_vs_gym_ic0_null_2004_2023"
)
OUT_DIR = ANALYSIS_DIR / "high_contrast_all_years"


SERIES = [
    ("Biomass / CWAD vs TOPWT", "windows_topwt", "gym_topwt", "kg/ha"),
    ("Grain / GWAD vs GRNWT", "windows_grnwt", "gym_grnwt", "kg/ha"),
    ("Water stress / WSPD vs SWFAC", "windows_wspd", "gym_swfac", "index"),
    ("Nitrogen stress / NSTD vs NSTRES", "windows_nstd", "gym_nstres", "index"),
]


def plot_one_year(df: pd.DataFrame, year: int, x_col: str, x_label: str, out_path: Path) -> None:
    year_df = df[df["year"] == year].copy()
    if year_df.empty:
        return
    year_df = year_df.sort_values(x_col)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes = axes.ravel()

    for ax, (title, win_col, gym_col, y_label) in zip(axes, SERIES):
        ax.plot(
            year_df[x_col],
            year_df[win_col],
            color="#D55E00",
            linewidth=2.4,
            linestyle="-",
            label="Windows DSSAT 4.8.5",
        )
        ax.plot(
            year_df[x_col],
            year_df[gym_col],
            color="#0072B2",
            linewidth=2.4,
            linestyle=(0, (5, 2)),
            label="gym/PDI DSSAT 4.8.0",
        )
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(y_label)
        ax.grid(True, color="#D0D0D0", linewidth=0.8, alpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[-2:]:
        ax.set_xlabel(x_label)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(f"HLA {year} IC=0 null: Windows DSSAT vs gym/PDI", fontsize=14, y=0.985)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dap_path = ANALYSIS_DIR / "daily_by_DAP_windows_vs_gym_IC0_null_2004_2023.csv"
    doy_path = ANALYSIS_DIR / "daily_by_DOY_windows_vs_gym_IC0_null_2004_2023.csv"
    dap = pd.read_csv(dap_path)
    doy = pd.read_csv(doy_path)

    years = sorted(set(dap["year"].dropna().astype(int)).intersection(doy["year"].dropna().astype(int)))
    manifest_rows: list[dict[str, str | int]] = []

    for year in years:
        dap_out = OUT_DIR / f"high_contrast_DAP_windows_vs_gym_{year}.png"
        doy_out = OUT_DIR / f"high_contrast_DOY_windows_vs_gym_{year}.png"
        plot_one_year(dap, year, "dap", "DAP", dap_out)
        plot_one_year(doy, year, "doy", "DOY", doy_out)
        manifest_rows.append(
            {
                "year": year,
                "dap_figure": str(dap_out),
                "doy_figure": str(doy_out),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUT_DIR / "high_contrast_all_years_manifest.csv", index=False, encoding="utf-8-sig")
    print(f"Generated {len(manifest_rows)} years x 2 figures in {OUT_DIR}")


if __name__ == "__main__":
    main()
