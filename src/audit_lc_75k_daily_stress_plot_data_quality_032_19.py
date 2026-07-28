from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import build_relaxed_success_five_scenario_daily_evidence_027_05 as plot27


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "032_19_lc_75k_daily_stress_plot_data_quality_audit"
TAB = OUT / "tables"
DOC = ROOT / "docs" / "032_19_lc_75k_daily_stress_plot_data_quality_audit_record.md"
PROMPT = ROOT / "prompts" / "032_19_lc_75k_daily_stress_plot_data_quality_audit.md"
DAILY_DIR = ROOT / "benchmark_results" / "032_17_lc_75k_ppo_all_year_five_scenario_daily_package" / "tables"
BASE_SNAPSHOT_03135 = ROOT / "benchmark_results" / "031_35_missing_four_baseline_completion_for_03134" / "snapshots" / "LCA"
AUTO_SNAPSHOT_03136 = ROOT / "benchmark_results" / "031_36_missing_dssat_auto_completion_for_03134" / "snapshots" / "LCA"
AUTO_SNAPSHOT_03218 = ROOT / "benchmark_results" / "032_18_lc_missing_dssat_auto_daily_completion" / "snapshots" / "LCA"
YEARS = list(range(2005, 2021))
SCENARIOS = ["null", "recorded_farmer", "dssat_auto", "official_extension_expert", "rl_candidate"]


def ensure_dirs() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "No rows."
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(6)
    work = work.astype(object).where(pd.notna(work), "")
    header = "| " + " | ".join(map(str, work.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(map(str, values)) + " |" for values in work.to_numpy().tolist()]
    return "\n".join([header, sep, *rows])


def load_year(year: int) -> pd.DataFrame:
    path = DAILY_DIR / f"032_17_lc{year}_75k_ppo_five_scenario_daily.csv"
    df = pd.read_csv(path, keep_default_na=False)
    for col in [
        "year",
        "requested_year",
        "doy",
        "dap",
        "rainfall_mm",
        "irrigation_executed_mm",
        "nitrogen_executed_kg_ha",
        "water_stress_index_wspd",
        "nitrogen_stress_index_nstd",
        "grain_yield_kg_ha",
        "biomass_kg_ha",
    ]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def baseline_snapshot(year: int, scenario: str) -> Path | None:
    if scenario == "recorded_farmer":
        candidates = [BASE_SNAPSHOT_03135 / str(year) / "recorded_farmer_template_02705"]
    elif scenario == "official_extension_expert":
        candidates = [BASE_SNAPSHOT_03135 / str(year) / "official_extension_expert"]
    elif scenario == "null":
        candidates = [BASE_SNAPSHOT_03135 / str(year) / "null"]
    elif scenario == "dssat_auto":
        candidates = [AUTO_SNAPSHOT_03136 / str(year) / "dssat_auto", AUTO_SNAPSHOT_03218 / str(year) / "dssat_auto"]
    else:
        return None
    for c in candidates:
        if (c / "PlantGro.OUT").exists():
            return c
    return None


def parse_plant_stress(snapshot: Path) -> pd.DataFrame:
    plant = plot27.parse_table(snapshot / "PlantGro.OUT")
    if plant.empty:
        return pd.DataFrame()
    cols = {c: c.lower() for c in ["YEAR", "DOY", "DAS", "DAP", "WSPD", "NSTD", "GWAD", "CWAD"] if c in plant.columns}
    out = plant[list(cols)].rename(columns=cols)
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def main() -> None:
    ensure_dirs()
    all_daily = pd.concat([load_year(y) for y in YEARS], ignore_index=True, sort=False)

    row_counts = all_daily.groupby(["requested_year", "scenario"]).size().reset_index(name="rows")
    pivot_counts = row_counts.pivot(index="requested_year", columns="scenario", values="rows").reset_index()
    pivot_counts.to_csv(TAB / "032_19_row_counts_by_year_scenario.csv", index=False, encoding="utf-8-sig")

    dup = (
        all_daily.groupby(["requested_year", "scenario", "date", "dap"])
        .size()
        .reset_index(name="n_rows")
    )
    dup = dup[dup["n_rows"] > 1].sort_values(["requested_year", "scenario", "dap", "date"])
    dup.to_csv(TAB / "032_19_duplicate_date_dap_rows.csv", index=False, encoding="utf-8-sig")

    weather = (
        all_daily.groupby(["requested_year", "scenario"])
        .agg(
            rain_total_mm=("rainfall_mm", "sum"),
            dap_min=("dap", "min"),
            dap_max=("dap", "max"),
            max_wspd=("water_stress_index_wspd", "max"),
            max_nstd=("nitrogen_stress_index_nstd", "max"),
            irrigation_total_mm=("irrigation_executed_mm", "sum"),
            nitrogen_total_kg_ha=("nitrogen_executed_kg_ha", "sum"),
        )
        .reset_index()
    )
    rain_spread = weather.groupby("requested_year")["rain_total_mm"].agg(["min", "max"]).reset_index()
    rain_spread["rain_total_spread_mm"] = rain_spread["max"] - rain_spread["min"]
    weather.to_csv(TAB / "032_19_weather_and_stress_summary_by_year_scenario.csv", index=False, encoding="utf-8-sig")
    rain_spread.to_csv(TAB / "032_19_rain_total_spread_by_year.csv", index=False, encoding="utf-8-sig")

    plant_compare_rows: list[dict[str, Any]] = []
    for year in YEARS:
        daily = all_daily[all_daily["requested_year"].eq(year)]
        for scenario in [s for s in SCENARIOS if s != "rl_candidate"]:
            snap = baseline_snapshot(year, scenario)
            if snap is None:
                plant_compare_rows.append({"year": year, "scenario": scenario, "snapshot_available": False})
                continue
            plant = parse_plant_stress(snap)
            if plant.empty:
                plant_compare_rows.append({"year": year, "scenario": scenario, "snapshot_available": True, "plantgro_empty": True})
                continue
            d = daily[daily["scenario"].eq(scenario)].copy()
            merged = d.merge(plant[["doy", "dap", "wspd", "nstd"]].rename(columns={"dap": "plantgro_dap"}), on=["doy"], how="left")
            daily_w = pd.to_numeric(merged["water_stress_index_wspd"], errors="coerce")
            daily_n = pd.to_numeric(merged["nitrogen_stress_index_nstd"], errors="coerce")
            plant_w = pd.to_numeric(merged["wspd"], errors="coerce")
            plant_n = pd.to_numeric(merged["nstd"], errors="coerce")
            plant_compare_rows.append(
                {
                    "year": year,
                    "scenario": scenario,
                    "snapshot_available": True,
                    "rows_daily": int(len(d)),
                    "rows_merged": int(len(merged)),
                    "daily_max_water_col": float(daily_w.max()),
                    "plantgro_max_WSPD": float(plant_w.max()),
                    "max_abs_diff_water_col_vs_WSPD": float((daily_w - plant_w).abs().max()),
                    "daily_max_n_col": float(daily_n.max()),
                    "plantgro_max_NSTD": float(plant_n.max()),
                    "max_abs_diff_n_col_vs_NSTD": float((daily_n - plant_n).abs().max()),
                    "snapshot_path": str(snap.relative_to(ROOT)).replace("\\", "/"),
                }
            )
    plant_compare = pd.DataFrame(plant_compare_rows)
    plant_compare.to_csv(TAB / "032_19_baseline_daily_stress_vs_plantgro_wspd_nstd.csv", index=False, encoding="utf-8-sig")

    focus_rows = []
    for year in [2008, 2013, 2018]:
        daily = all_daily[all_daily["requested_year"].eq(year)].copy()
        ppo = daily[daily["scenario"].eq("rl_candidate")].copy()
        max_dap = int(ppo.loc[pd.to_numeric(ppo["water_stress_index_wspd"], errors="coerce").idxmax(), "dap"])
        window = daily[daily["dap"].between(max_dap - 4, max_dap + 4)].copy()
        window["focus_year"] = year
        window["ppo_max_wspd_dap"] = max_dap
        focus_rows.append(window)
    focus = pd.concat(focus_rows, ignore_index=True, sort=False)
    focus_cols = [
        "focus_year",
        "ppo_max_wspd_dap",
        "scenario",
        "date",
        "doy",
        "dap",
        "rainfall_mm",
        "irrigation_executed_mm",
        "nitrogen_executed_kg_ha",
        "water_stress_index_wspd",
        "nitrogen_stress_index_nstd",
        "grain_yield_kg_ha",
    ]
    focus[focus_cols].to_csv(TAB / "032_19_focus_windows_lc2008_2013_2018.csv", index=False, encoding="utf-8-sig")

    key_findings = pd.DataFrame(
        [
            {
                "finding": "rl_candidate_duplicate_date_dap_rows",
                "evidence": f"{len(dup)} duplicate date-DAP rows; affected years={sorted(dup['requested_year'].unique().astype(int).tolist()) if not dup.empty else []}",
                "risk": "PPO weather totals and early daily line plots can be distorted if plotted without merging duplicate DAP/date rows.",
            },
            {
                "finding": "baseline_daily_water_col_matches_plantgro_WSPD",
                "evidence": f"max baseline water diff={plant_compare['max_abs_diff_water_col_vs_WSPD'].dropna().max() if not plant_compare.empty else np.nan}",
                "risk": "If large, 032_17 water panel is using an inconsistent water-stress source for baselines.",
            },
            {
                "finding": "baseline_daily_n_col_matches_plantgro_NSTD",
                "evidence": f"max baseline nitrogen diff={plant_compare['max_abs_diff_n_col_vs_NSTD'].dropna().max() if not plant_compare.empty else np.nan}",
                "risk": "If large, 032_17 nitrogen panel is using an inconsistent nitrogen-stress source for baselines.",
            },
            {
                "finding": "weather_total_should_be_scenario_invariant",
                "evidence": f"max rain total spread across scenarios={rain_spread['rain_total_spread_mm'].max()} mm",
                "risk": "Weather is exogenous; nonzero spread usually indicates duplicated rows or date alignment differences, not real weather differences.",
            },
        ]
    )
    key_findings.to_csv(TAB / "032_19_key_findings.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 032_19 LC 75k daily stress plot data-quality audit record",
        "",
        "## Status",
        "",
        "- Completed.",
        "- Training runs: 0.",
        "- DSSAT reruns: 0.",
        "- 032_17 figures were not modified in this audit.",
        "",
        "## Key findings",
        "",
        md_table(key_findings, 20),
        "",
        "## Duplicate date-DAP rows",
        "",
        md_table(dup.head(40), 40),
        "",
        "## Rain total spread by year",
        "",
        md_table(rain_spread, 40),
        "",
        "## Baseline stress columns vs PlantGro WSPD/NSTD",
        "",
        md_table(plant_compare[["year", "scenario", "snapshot_available", "daily_max_water_col", "plantgro_max_WSPD", "max_abs_diff_water_col_vs_WSPD", "daily_max_n_col", "plantgro_max_NSTD", "max_abs_diff_n_col_vs_NSTD"]], 80),
        "",
        "## Interpretation",
        "",
        "- `WSPD/NSTD` panels should be interpreted from PlantGro-derived columns when available.",
        "- `rl_candidate` rows in 032_17 require date-DAP de-duplication before weather totals or daily stress panels are advisor-facing.",
        "- LC years with low rainfall but null WSPD=0 are not automatically erroneous: DSSAT PlantGro may report no water stress while nitrogen stress is severe.",
        "- The next fix should rebuild daily figures from consistent sources, preferably PlantGro/MgmtEvent snapshots for all scenarios, or at least de-duplicate PPO rows and relabel environment-derived `swfac/nstres` cautiously.",
    ]
    text = "\n".join(lines) + "\n"
    DOC.write_text(text, encoding="utf-8")
    (OUT / DOC.name).write_text(text, encoding="utf-8")
    print(key_findings.to_string(index=False))
    print(f"record={DOC.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
