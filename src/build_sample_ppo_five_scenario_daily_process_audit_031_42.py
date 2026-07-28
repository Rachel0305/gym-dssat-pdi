from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SELECTED_PATH = ROOT / "benchmark_results/031_38_five_station_ppo_water_n_saving_summary/tables/031_38_selected_station_year_ppo_deltas_vs_official.csv"
BASELINE_PATH = ROOT / "benchmark_results/031_35_missing_four_baseline_completion_for_03134/evaluation/031_35_full_generated_baseline_daily.csv"
AUTO_PATH = ROOT / "benchmark_results/031_36_missing_dssat_auto_completion_for_03134/evaluation/031_36_full_generated_dssat_auto_daily.csv"

OUT_DIR = ROOT / "benchmark_results/031_42_sample_ppo_five_scenario_daily_process_audit"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
DOCS_DIR = ROOT / "docs"

SAMPLES = [
    ("HLA", 2008, "HLA candidate with complete five-scenario daily sources"),
    ("HLA", 2023, "HLA later-year candidate with complete five-scenario daily sources"),
    ("LCA", 2012, "LC candidate with moderate water and N"),
    ("LCA", 2023, "LC candidate with lower N"),
    ("YCA", 2015, "YC candidate with relatively low irrigation and complete sources"),
    ("YCA", 2023, "YC later-year candidate with complete sources"),
]

SCENARIO_ORDER = ["null", "recorded_farmer", "dssat_auto", "official_extension_expert", "ppo_candidate"]
LABELS = {
    "null": "Null",
    "recorded_farmer": "Recorded farmer",
    "dssat_auto": "DSSAT auto",
    "official_extension_expert": "Official expert",
    "ppo_candidate": "MaskablePPO candidate",
}
COLORS = {
    "null": "#444444",
    "recorded_farmer": "#c94c4c",
    "dssat_auto": "#d99a00",
    "official_extension_expert": "#7b61b8",
    "ppo_candidate": "#2e8b57",
}
LINESTYLES = {
    "null": "-",
    "recorded_farmer": "--",
    "dssat_auto": "-.",
    "official_extension_expert": ":",
    "ppo_candidate": "-",
}


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)


def numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def normalize_baseline(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["scenario"] = out["scenario"].astype(str).str.strip()
    out["scenario"] = out["scenario"].replace(
        {
            "": "null",
            "nan": "null",
            "NaN": "null",
            "<NA>": "null",
            "recorded_farmer_template_02705": "recorded_farmer",
            "official_extension_expert": "official_extension_expert",
        }
    )
    out = out.rename(
        columns={
            "rain": "rainfall_mm",
            "tmax": "tmax_c",
            "tmin": "tmin_c",
            "grnwt": "grain_yield_kg_ha",
            "topwt": "biomass_kg_ha",
            "swfac": "water_stress_index_wspd",
            "nstres": "nitrogen_stress_index_nstd",
            "reward": "reward_step",
        }
    )
    return finalize_daily(out, source_file=str(BASELINE_PATH.relative_to(ROOT)))


def normalize_auto(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["scenario"] = "dssat_auto"
    out = out.rename(
        columns={
            "rain": "rainfall_mm",
            "tmax": "tmax_c",
            "tmin": "tmin_c",
            "grnwt": "grain_yield_kg_ha",
            "topwt": "biomass_kg_ha",
            "swfac": "water_stress_index_wspd",
            "nstres": "nitrogen_stress_index_nstd",
            "external_irrigation_action_mm": "irrigation_executed_mm",
            "external_nitrogen_action_kg_ha": "nitrogen_executed_kg_ha",
            "reward": "reward_step",
        }
    )
    return finalize_daily(out, source_file=str(AUTO_PATH.relative_to(ROOT)))


def normalize_ppo(path: Path, selected_row: pd.Series) -> pd.DataFrame:
    out = pd.read_csv(path)
    out["scenario"] = "ppo_candidate"
    out = out.rename(
        columns={
            "rain": "rainfall_mm",
            "tmax": "tmax_c",
            "tmin": "tmin_c",
            "grnwt": "grain_yield_kg_ha",
            "topwt": "biomass_kg_ha",
            "swfac": "water_stress_index_wspd",
            "nstres": "nitrogen_stress_index_nstd",
            "safe_action_amir": "irrigation_executed_mm",
            "safe_action_anfer": "nitrogen_executed_kg_ha",
            "reward": "reward_step",
        }
    )
    out["selected_seed"] = int(selected_row["seed"])
    out["selected_checkpoint_step"] = int(selected_row["checkpoint_step"])
    return finalize_daily(out, source_file=str(path.relative_to(ROOT)))


def finalize_daily(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    keep = [
        "station_code",
        "site",
        "year",
        "scenario",
        "date",
        "doy",
        "dap",
        "rainfall_mm",
        "tmax_c",
        "tmin_c",
        "grain_yield_kg_ha",
        "biomass_kg_ha",
        "water_stress_index_wspd",
        "nitrogen_stress_index_nstd",
        "irrigation_executed_mm",
        "nitrogen_executed_kg_ha",
        "reward_step",
        "selected_seed",
        "selected_checkpoint_step",
    ]
    for col in keep:
        if col not in df.columns:
            df[col] = np.nan
    df = df[keep].copy()
    numeric(
        df,
        [
            "year",
            "doy",
            "dap",
            "rainfall_mm",
            "tmax_c",
            "tmin_c",
            "grain_yield_kg_ha",
            "biomass_kg_ha",
            "water_stress_index_wspd",
            "nitrogen_stress_index_nstd",
            "irrigation_executed_mm",
            "nitrogen_executed_kg_ha",
            "reward_step",
        ],
    )
    df["irrigation_executed_mm"] = df["irrigation_executed_mm"].fillna(0.0)
    df["nitrogen_executed_kg_ha"] = df["nitrogen_executed_kg_ha"].fillna(0.0)
    df["reward_step"] = df["reward_step"].fillna(0.0)
    df = df.sort_values(["station_code", "year", "scenario", "dap"])
    df["cumulative_available_reward"] = df.groupby(["station_code", "year", "scenario"], dropna=False)["reward_step"].cumsum()
    df["cumulative_irrigation_mm"] = df.groupby(["station_code", "year", "scenario"], dropna=False)["irrigation_executed_mm"].cumsum()
    df["cumulative_n_kg_ha"] = df.groupby(["station_code", "year", "scenario"], dropna=False)["nitrogen_executed_kg_ha"].cumsum()
    df["source_file"] = source_file
    return df


def summarize_daily(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (station, year, scenario), g in df.groupby(["station_code", "year", "scenario"], dropna=False):
        gs = g.sort_values("dap")
        final = gs.iloc[-1]
        irr = float(gs["irrigation_executed_mm"].sum())
        nit = float(gs["nitrogen_executed_kg_ha"].sum())
        grain = float(final["grain_yield_kg_ha"])
        rows.append(
            {
                "station_code": station,
                "year": int(year),
                "scenario": scenario,
                "final_grain_kg_ha": grain,
                "final_biomass_kg_ha": float(final["biomass_kg_ha"]),
                "total_irrigation_mm": irr,
                "total_nitrogen_kg_ha": nit,
                "irrigation_event_count": int((gs["irrigation_executed_mm"] > 0).sum()),
                "nitrogen_event_count": int((gs["nitrogen_executed_kg_ha"] > 0).sum()),
                "six_mm_irrigation_count": int(np.isclose(gs["irrigation_executed_mm"], 6.0, atol=1e-9).sum()),
                "first_irrigation_dap": int(gs.loc[gs["irrigation_executed_mm"] > 0, "dap"].iloc[0]) if (gs["irrigation_executed_mm"] > 0).any() else np.nan,
                "first_n_dap": int(gs.loc[gs["nitrogen_executed_kg_ha"] > 0, "dap"].iloc[0]) if (gs["nitrogen_executed_kg_ha"] > 0).any() else np.nan,
                "max_water_stress_index_wspd": float(gs["water_stress_index_wspd"].max()),
                "max_nitrogen_stress_index_nstd": float(gs["nitrogen_stress_index_nstd"].max()),
                "stress_days_wspd_gt_0p05": int((gs["water_stress_index_wspd"] > 0.05).sum()),
                "stress_days_nstd_gt_0p05": int((gs["nitrogen_stress_index_nstd"] > 0.05).sum()),
                "final_cumulative_available_reward": float(final["cumulative_available_reward"]),
                "source_file": final["source_file"],
            }
        )
    return pd.DataFrame(rows).sort_values(["station_code", "year", "scenario"])


def scenario_subset(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    return df[df["scenario"] == scenario].sort_values("dap")


def plot_sample(df: pd.DataFrame, station: str, year: int, selected: pd.Series) -> tuple[Path, Path]:
    sample = df[(df["station_code"] == station) & (df["year"] == year)].copy()
    fig, axes = plt.subplots(4, 2, figsize=(14, 11), constrained_layout=True)
    ax = axes.ravel()

    weather = scenario_subset(sample, "null")
    if weather.empty:
        weather = sample.sort_values("dap").drop_duplicates("dap")
    ax0 = ax[0]
    ax0.bar(weather["dap"], weather["rainfall_mm"], color="#8bb8d8", alpha=0.8, label="Rain")
    ax0.set_ylabel("Rain (mm)")
    ax0b = ax0.twinx()
    ax0b.plot(weather["dap"], weather["tmax_c"], color="#d94841", label="Tmax")
    ax0b.plot(weather["dap"], weather["tmin_c"], color="#777777", linestyle="--", label="Tmin")
    ax0b.set_ylabel("Temperature (°C)")
    lines0, labels0 = ax0.get_legend_handles_labels()
    lines0b, labels0b = ax0b.get_legend_handles_labels()
    ax0.legend(lines0 + lines0b, labels0 + labels0b, fontsize=8, loc="upper right")
    ax0.set_title("Weather")

    ax1 = ax[1]
    for sc in SCENARIO_ORDER:
        g = scenario_subset(sample, sc)
        if g.empty:
            continue
        ax1.plot(g["dap"], g["cumulative_irrigation_mm"], color=COLORS[sc], linestyle=LINESTYLES[sc], label=LABELS[sc])
    ax1.set_title("Cumulative irrigation (SWTD unavailable)")
    ax1.set_ylabel("mm")
    ax1.legend(fontsize=8, loc="best")

    ax2 = ax[2]
    for sc in SCENARIO_ORDER:
        g = scenario_subset(sample, sc)
        if g.empty:
            continue
        ax2.plot(g["dap"], g["water_stress_index_wspd"], color=COLORS[sc], linestyle=LINESTYLES[sc], label=LABELS[sc])
    ax2.set_title("Water stress index")
    ax2.set_ylabel("WSPD (0=no stress)")

    ax3 = ax[3]
    for sc in SCENARIO_ORDER:
        g = scenario_subset(sample, sc)
        if g.empty:
            continue
        ax3.plot(g["dap"], g["nitrogen_stress_index_nstd"], color=COLORS[sc], linestyle=LINESTYLES[sc], label=LABELS[sc])
    ax3.set_title("Nitrogen stress index")
    ax3.set_ylabel("NSTD (0=no stress)")

    ax4 = ax[4]
    for sc in SCENARIO_ORDER:
        g = scenario_subset(sample, sc)
        ev = g[g["irrigation_executed_mm"] > 0]
        if ev.empty:
            continue
        ax4.vlines(ev["dap"], 0, ev["irrigation_executed_mm"], colors=COLORS[sc], linestyles=LINESTYLES[sc], alpha=0.85, label=LABELS[sc])
        ax4.scatter(ev["dap"], ev["irrigation_executed_mm"], color=COLORS[sc], s=22)
    ax4.set_title("Irrigation events")
    ax4.set_ylabel("mm/event")
    ax4.legend(fontsize=8, loc="upper right", ncol=2)

    ax5 = ax[5]
    for sc in SCENARIO_ORDER:
        g = scenario_subset(sample, sc)
        ev = g[g["nitrogen_executed_kg_ha"] > 0]
        if ev.empty:
            continue
        ax5.vlines(ev["dap"], 0, ev["nitrogen_executed_kg_ha"], colors=COLORS[sc], linestyles=LINESTYLES[sc], alpha=0.85, label=LABELS[sc])
        ax5.scatter(ev["dap"], ev["nitrogen_executed_kg_ha"], color=COLORS[sc], s=22)
    ax5.set_title("Nitrogen application events")
    ax5.set_ylabel("kg/ha/event")
    ax5.legend(fontsize=8, loc="upper right", ncol=2)

    ax6 = ax[6]
    ax6.text(0.01, 0.97, "Thin companion lines are biomass; thick lines are grain.", transform=ax6.transAxes, va="top", fontsize=8)
    for sc in SCENARIO_ORDER:
        g = scenario_subset(sample, sc)
        if g.empty:
            continue
        ax6.plot(g["dap"], g["biomass_kg_ha"], color=COLORS[sc], linestyle=LINESTYLES[sc], alpha=0.35)
        ax6.plot(g["dap"], g["grain_yield_kg_ha"], color=COLORS[sc], linestyle=LINESTYLES[sc], linewidth=2.0, label=LABELS[sc])
    ax6.set_title("Grain and biomass trajectories")
    ax6.set_ylabel("kg/ha")
    ax6.set_xlabel("DAP")

    ax7 = ax[7]
    for sc in SCENARIO_ORDER:
        g = scenario_subset(sample, sc)
        if g.empty:
            continue
        ax7.plot(g["dap"], g["cumulative_available_reward"], color=COLORS[sc], linestyle=LINESTYLES[sc], label=LABELS[sc])
    ax7.set_title("Cumulative available reward/proxy")
    ax7.set_ylabel("source reward units")
    ax7.set_xlabel("DAP")
    ax7.legend(fontsize=8, loc="best")

    for a in ax:
        a.grid(alpha=0.25)

    seed = int(selected["seed"])
    ckpt = int(selected["checkpoint_step"])
    fig.suptitle(f"{station}{year} MaskablePPO five-scenario daily process audit (seed{seed}, ckpt{ckpt})", fontsize=14, fontweight="bold")

    stem = f"031_42_{station.lower()}{year}_maskableppo_five_scenario_daily_process"
    png = FIG_DIR / f"{stem}.png"
    svg = FIG_DIR / f"{stem}.svg"
    fig.savefig(png, dpi=220)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def make_record(manifest: pd.DataFrame, summary: pd.DataFrame, coverage: pd.DataFrame) -> None:
    record = DOCS_DIR / "031_42_sample_ppo_five_scenario_daily_process_audit_record.md"
    ok_count = int((coverage["status"] == "ok").sum())
    total_count = len(coverage)
    lines = [
        "# 031_42 Sample free-timing PPO five-scenario daily process audit record",
        "",
        "## Scope",
        "",
        "This is a plotting and evidence-audit task only. No new training, no DSSAT rerun, no checkpoint reselection, and no reward or hyperparameter change were performed.",
        "",
        "## Important limitation",
        "",
        "The available 031_34/031_35/031_36 daily CSV sources do not include a complete soil-water storage column. The previous soil-water panel was therefore replaced by cumulative irrigation. SWTD was not inferred or fabricated.",
        "",
        "## Fixed sample list",
        "",
        manifest.to_markdown(index=False),
        "",
        "## Coverage",
        "",
        f"- Successful plotted samples: {ok_count}/{total_count}",
        "",
        coverage.to_markdown(index=False),
        "",
        "## Summary of sampled PPO candidates",
        "",
        summary[summary["scenario"] == "ppo_candidate"][
            [
                "station_code",
                "year",
                "final_grain_kg_ha",
                "total_irrigation_mm",
                "total_nitrogen_kg_ha",
                "irrigation_event_count",
                "six_mm_irrigation_count",
                "nitrogen_event_count",
                "max_water_stress_index_wspd",
                "max_nitrogen_stress_index_nstd",
                "stress_days_wspd_gt_0p05",
                "stress_days_nstd_gt_0p05",
                "first_irrigation_dap",
                "first_n_dap",
            ]
        ].to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- These figures are intended for visual inspection of whether the frozen PPO candidate actions look agronomically plausible against weather and stress trajectories.",
        "- They are not causal proof that each action was necessary.",
        "- Questionable events should be followed by same-prefix counterfactual DSSAT audits, e.g. delete or downgrade one action and compare final yield/resource metrics.",
    ]
    record.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    selected = pd.read_csv(SELECTED_PATH)
    base = pd.read_csv(BASELINE_PATH, keep_default_na=False)
    auto = pd.read_csv(AUTO_PATH, keep_default_na=False)
    selected["year"] = pd.to_numeric(selected["year"], errors="coerce").astype("Int64")
    base["year"] = pd.to_numeric(base["year"], errors="coerce").astype("Int64")
    auto["year"] = pd.to_numeric(auto["year"], errors="coerce").astype("Int64")

    manifest_rows = []
    coverage_rows = []
    daily_frames = []

    for station, year, reason in SAMPLES:
        row = selected[(selected["station_code"] == station) & (selected["year"] == year)]
        row_found = len(row) == 1
        ppo_path = None
        ppo_exists = False
        if row_found:
            row0 = row.iloc[0]
            ppo_path = ROOT / str(row0["daily_csv_path"])
            ppo_exists = ppo_path.exists()
        base_rows = base[(base["station_code"] == station) & (base["year"] == year)]
        auto_rows = auto[(auto["station_code"] == station) & (auto["year"] == year)]
        has_base = not base_rows.empty
        has_auto = not auto_rows.empty

        status = "ok" if row_found and ppo_exists and has_base and has_auto else "missing_source"
        coverage_rows.append(
            {
                "station_code": station,
                "year": year,
                "status": status,
                "selected_row_found": row_found,
                "ppo_daily_exists": ppo_exists,
                "baseline_03135_rows": int(len(base_rows)),
                "dssat_auto_03136_rows": int(len(auto_rows)),
                "ppo_daily_path": str(ppo_path.relative_to(ROOT)) if ppo_path and ppo_exists else "",
            }
        )
        manifest_rows.append({"station_code": station, "year": year, "reason": reason})
        if status != "ok":
            continue
        daily_frames.append(normalize_baseline(base_rows))
        daily_frames.append(normalize_auto(auto_rows))
        daily_frames.append(normalize_ppo(ppo_path, row.iloc[0]))

    manifest = pd.DataFrame(manifest_rows)
    coverage = pd.DataFrame(coverage_rows)
    if not daily_frames:
        raise RuntimeError("No samples could be assembled from available sources.")
    daily = pd.concat(daily_frames, ignore_index=True)
    daily = daily.sort_values(["station_code", "year", "scenario", "dap"])
    summary = summarize_daily(daily)

    figure_rows = []
    for station, year, _reason in SAMPLES:
        if not ((coverage["station_code"] == station) & (coverage["year"] == year) & (coverage["status"] == "ok")).any():
            continue
        row0 = selected[(selected["station_code"] == station) & (selected["year"] == year)].iloc[0]
        png, svg = plot_sample(daily, station, year, row0)
        figure_rows.append({"station_code": station, "year": year, "png": str(png.relative_to(ROOT)), "svg": str(svg.relative_to(ROOT))})

    manifest.to_csv(TABLE_DIR / "031_42_sample_manifest.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(TABLE_DIR / "031_42_source_coverage_checks.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(TABLE_DIR / "031_42_sample_five_scenario_daily.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLE_DIR / "031_42_sample_five_scenario_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(figure_rows).to_csv(TABLE_DIR / "031_42_generated_figures.csv", index=False, encoding="utf-8-sig")
    make_record(manifest, summary, coverage)

    print(f"Wrote {len(figure_rows)} figures to {FIG_DIR}")
    print(f"Record: {DOCS_DIR / '031_42_sample_ppo_five_scenario_daily_process_audit_record.md'}")


if __name__ == "__main__":
    main()
