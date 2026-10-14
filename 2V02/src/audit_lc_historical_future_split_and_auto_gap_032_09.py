from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
PROMPT = ROOT / "prompts" / "032_09_lc_historical_future_split_and_auto_gap_audit.md"
BASE_DAILY = ROOT / "benchmark_results" / "031_35_missing_four_baseline_completion_for_03134" / "evaluation" / "031_35_full_generated_baseline_daily.csv"
AUTO_DAILY = ROOT / "benchmark_results" / "031_36_missing_dssat_auto_completion_for_03134" / "evaluation" / "031_36_full_generated_dssat_auto_daily.csv"
LEGACY_LC_AUTO_DAILY = ROOT / "DSSAT_auto_validation" / "lc_fixed_input_year_screening_017_11" / "017_11_lc_fixed_input_daily.csv"
OUT = ROOT / "benchmark_results" / "032_09_lc_historical_future_split_and_auto_gap_audit"
DOC = ROOT / "docs" / "032_09_lc_historical_future_split_and_auto_gap_audit_record.md"
STATION = "LCA"


def ensure_dirs() -> None:
    for rel in ["configs", "tables"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def md_table(df: pd.DataFrame, max_rows: int = 120) -> str:
    if df.empty:
        return "No rows."
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
    work = work.astype(object).where(pd.notna(work), "")
    header = "| " + " | ".join(map(str, work.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in work.to_numpy().tolist()]
    return "\n".join([header, sep, *rows])


def split_label(year: int) -> str:
    if 2000 <= year <= 2010:
        return "train_2000_2010"
    if 2011 <= year <= 2020:
        return "test_2011_2020"
    if 2021 <= year <= 2023:
        return "extension_2021_2023"
    return "outside_scope"


def scenario_pool_years(cfg: dict[str, Any]) -> pd.DataFrame:
    path = ROOT / cfg["paths"]["scenario_pool_csv"]
    pool = pd.read_csv(path)
    pool["year"] = pd.to_numeric(pool["year"], errors="coerce")
    sub = pool[pool["station_code"].eq(STATION) & pool["year"].ge(2000)].copy()
    site_col = "site" if "site" in sub.columns else "station_name"
    keep_cols = ["station_code", site_col, "year"]
    optional = [
        "scenario_type",
        "growing_season_rain",
        "annual_rain",
        "has_water_stress",
        "has_nitrogen_stress",
        "irrigation_responsive",
        "nitrogen_responsive",
        "recommended_for_ppo_train",
        "recommended_for_ppo_eval",
    ]
    keep_cols.extend([c for c in optional if c in sub.columns])
    rows = sub[keep_cols].drop_duplicates()
    if site_col != "site":
        rows = rows.rename(columns={site_col: "site"})
    rows["year"] = rows["year"].astype(int)
    rows["scenario_pool_available"] = True
    rows["scenario_pool_source"] = str(path.relative_to(ROOT)).replace("\\", "/")
    return rows.sort_values("year").reset_index(drop=True)


def weather_years(cfg: dict[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    inv_path = ROOT / cfg["paths"]["weather_inventory_csv"]
    if inv_path.exists():
        inv = pd.read_csv(inv_path)
        year_col = "year" if "year" in inv.columns else None
        station_col = "station_code" if "station_code" in inv.columns else None
        if year_col and station_col:
            sub = inv[inv[station_col].eq(STATION)].copy()
            sub[year_col] = pd.to_numeric(sub[year_col], errors="coerce")
            for year in sorted(sub[sub[year_col].ge(2000)][year_col].dropna().astype(int).unique()):
                records.append({"year": year, "weather_inventory_available": True, "weather_inventory_source": str(inv_path.relative_to(ROOT)).replace("\\", "/")})
    clean_path = ROOT / cfg["paths"]["weather_clean_csv"]
    if clean_path.exists():
        clean = pd.read_csv(clean_path)
        if {"station_code", "date"}.issubset(clean.columns):
            sub = clean[clean["station_code"].eq(STATION)].copy()
            years = pd.to_datetime(sub["date"], errors="coerce").dt.year
            for year in sorted(years[years.ge(2000)].dropna().astype(int).unique()):
                records.append({"year": year, "weather_clean_available": True, "weather_clean_source": str(clean_path.relative_to(ROOT)).replace("\\", "/")})
        elif {"station_code", "year"}.issubset(clean.columns):
            sub = clean[clean["station_code"].eq(STATION)].copy()
            sub["year"] = pd.to_numeric(sub["year"], errors="coerce")
            for year in sorted(sub[sub["year"].ge(2000)]["year"].dropna().astype(int).unique()):
                records.append({"year": year, "weather_clean_available": True, "weather_clean_source": str(clean_path.relative_to(ROOT)).replace("\\", "/")})
    if not records:
        return pd.DataFrame(columns=["year", "weather_inventory_available", "weather_clean_available", "weather_inventory_source", "weather_clean_source"])
    raw = pd.DataFrame(records)
    for col, default in [
        ("weather_inventory_available", False),
        ("weather_clean_available", False),
        ("weather_inventory_source", ""),
        ("weather_clean_source", ""),
    ]:
        if col not in raw:
            raw[col] = default
    out = raw.groupby("year", as_index=False).agg(
        weather_inventory_available=("weather_inventory_available", "max"),
        weather_clean_available=("weather_clean_available", "max"),
        weather_inventory_source=("weather_inventory_source", lambda s: ";".join(sorted(set(str(x) for x in s.dropna())))),
        weather_clean_source=("weather_clean_source", lambda s: ";".join(sorted(set(str(x) for x in s.dropna())))),
    )
    return out


def baseline_coverage() -> pd.DataFrame:
    parts = []
    if BASE_DAILY.exists():
        base = pd.read_csv(BASE_DAILY, keep_default_na=False)
        base = base[base["station_code"].eq(STATION)].copy()
        base["year"] = pd.to_numeric(base["year"], errors="coerce").astype(int)
        base["scenario"] = base["scenario"].replace({"recorded_farmer_template_02705": "recorded_farmer"})
        cov = base.groupby(["year", "scenario"], as_index=False).size()
        parts.append(cov.assign(source=str(BASE_DAILY.relative_to(ROOT)).replace("\\", "/")))
    if AUTO_DAILY.exists():
        auto = pd.read_csv(AUTO_DAILY, keep_default_na=False)
        auto = auto[auto["station_code"].eq(STATION)].copy()
        auto["year"] = pd.to_numeric(auto["year"], errors="coerce").astype(int)
        cov = auto.groupby(["year", "scenario"], as_index=False).size()
        parts.append(cov.assign(source=str(AUTO_DAILY.relative_to(ROOT)).replace("\\", "/")))
    if LEGACY_LC_AUTO_DAILY.exists():
        legacy = pd.read_csv(LEGACY_LC_AUTO_DAILY, keep_default_na=False)
        if {"site", "requested_year", "scenario"}.issubset(legacy.columns):
            legacy = legacy[legacy["site"].eq("LC") & legacy["scenario"].eq("dssat_auto")].copy()
            legacy["year"] = pd.to_numeric(legacy["requested_year"], errors="coerce").astype(int)
            cov = legacy.groupby(["year", "scenario"], as_index=False).size()
            parts.append(cov.assign(source=str(LEGACY_LC_AUTO_DAILY.relative_to(ROOT)).replace("\\", "/")))
    all_cov = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["year", "scenario", "size", "source"])
    rows = []
    for year, g in all_cov.groupby("year"):
        scenarios = set(g["scenario"].astype(str))
        rows.append(
            {
                "year": int(year),
                "has_null": "null" in scenarios,
                "has_official_extension_expert": "official_extension_expert" in scenarios,
                "has_recorded_farmer": "recorded_farmer" in scenarios,
                "has_dssat_auto": "dssat_auto" in scenarios,
                "baseline_scenarios_present": ",".join(sorted(scenarios)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    cfg = load_yaml(CONFIG)
    pool = scenario_pool_years(cfg)
    weather = weather_years(cfg)
    baselines = baseline_coverage()
    years = pool.merge(weather, on="year", how="left").merge(baselines, on="year", how="left")
    for col in ["weather_inventory_available", "weather_clean_available", "has_null", "has_official_extension_expert", "has_recorded_farmer", "has_dssat_auto"]:
        if col not in years:
            years[col] = False
        years[col] = years[col].fillna(False).astype(bool)
    years["split"] = years["year"].apply(split_label)
    years["has_core_four_baseline_daily"] = years["has_null"] & years["has_official_extension_expert"] & years["has_recorded_farmer"] & years["has_dssat_auto"]
    years["rl_env_candidate"] = years["scenario_pool_available"] & (years["weather_inventory_available"] | years["weather_clean_available"])
    years["five_scenario_plot_ready"] = years["rl_env_candidate"] & years["has_core_four_baseline_daily"]
    years["needs_dssat_auto_completion"] = years["rl_env_candidate"] & years["has_null"] & years["has_official_extension_expert"] & years["has_recorded_farmer"] & (~years["has_dssat_auto"])
    years["recommended_use"] = years.apply(
        lambda r: "train_candidate" if r["split"] == "train_2000_2010" and r["rl_env_candidate"] else (
            "future_test_ready" if r["split"] == "test_2011_2020" and r["five_scenario_plot_ready"] else (
                "future_test_needs_auto" if r["split"] == "test_2011_2020" and r["needs_dssat_auto_completion"] else (
                    "extension_test_ready" if r["split"] == "extension_2021_2023" and r["five_scenario_plot_ready"] else (
                        "extension_test_needs_auto" if r["split"] == "extension_2021_2023" and r["needs_dssat_auto_completion"] else "not_ready_or_outside_scope"
                    )
                )
            )
        ),
        axis=1,
    )

    years.to_csv(OUT / "tables" / "032_09_lc_year_split_and_baseline_coverage.csv", index=False, encoding="utf-8-sig")
    gaps = years[years["needs_dssat_auto_completion"]].copy()
    gaps.to_csv(OUT / "tables" / "032_09_lc_missing_dssat_auto_years.csv", index=False, encoding="utf-8-sig")
    split_counts = years.groupby(["split", "recommended_use"], as_index=False).size()
    split_counts.to_csv(OUT / "tables" / "032_09_lc_split_counts.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 032_09 LC historical/future split and DSSAT-auto gap audit record",
        "",
        "## Scope",
        "",
        "- No training.",
        "- No DSSAT run.",
        "- Station: LCA/LC.",
        "- Candidate protocol: train on 2000-2010, test on 2011-2020, extension test on 2021-2023.",
        "",
        "## Split and baseline coverage",
        "",
        md_table(years[[
            "year", "split", "scenario_pool_available", "weather_inventory_available", "weather_clean_available",
            "has_null", "has_official_extension_expert", "has_recorded_farmer", "has_dssat_auto",
            "five_scenario_plot_ready", "needs_dssat_auto_completion", "recommended_use"
        ]], 80),
        "",
        "## Missing DSSAT-auto years",
        "",
        md_table(gaps[["year", "split", "recommended_use", "baseline_scenarios_present"]], 80) if not gaps.empty else "No missing DSSAT-auto gaps among otherwise baseline-ready years.",
        "",
        "## Split counts",
        "",
        md_table(split_counts, 80),
        "",
        "## Recommendation",
        "",
        "- Use all available 2000-2010 LC years as the historical training/development pool if the RL environment can instantiate them.",
        "- Use 2011-2020 as future-year tests after DSSAT-auto gaps are completed.",
        "- Treat 2021-2023 as external/extension tests.",
        "- Complete missing DSSAT-auto years in a separate task before final five-scenario plotting.",
    ]
    text = "\n".join(lines) + "\n"
    DOC.write_text(text, encoding="utf-8")
    (OUT / DOC.name).write_text(text, encoding="utf-8")
    result = {
        "task": "032_09_lc_historical_future_split_and_auto_gap_audit",
        "training_run": False,
        "dssat_run": False,
        "coverage_csv": str((OUT / "tables" / "032_09_lc_year_split_and_baseline_coverage.csv").relative_to(ROOT)).replace("\\", "/"),
        "missing_auto_csv": str((OUT / "tables" / "032_09_lc_missing_dssat_auto_years.csv").relative_to(ROOT)).replace("\\", "/"),
        "record_md": str(DOC.relative_to(ROOT)).replace("\\", "/"),
    }
    (OUT / "032_09_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\nCoverage:")
    print(years[["year", "split", "has_dssat_auto", "five_scenario_plot_ready", "needs_dssat_auto_completion", "recommended_use"]].to_string(index=False))
    print("\nMissing DSSAT-auto:")
    print(gaps[["year", "split", "recommended_use"]].to_string(index=False) if not gaps.empty else "none")


if __name__ == "__main__":
    main()
