"""Versioned HL/FQ five-scenario figures with native auto I and corrected FQ2018.

Reads frozen baseline snapshots and already trained PPO checkpoints. The old
055_03 results are preserved as an original-input comparison.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from hl_fq_8seed_cross_site.plot_hl_fq_five_scenario_05503_style import (
    OUT_ROOT as ORIGINAL, SEEDS_BY_SITE, YEARS, num, plot_daily_year,
    plot_management_bars, plot_management_year, plot_metric_bars, plot_reward,
    read_csv, scenario_key, to_number, write_event_and_daily_tables, write_table_files,
)
import hl_fq_8seed_cross_site.plot_hl_fq_five_scenario_05503_style as viz

OUT = ROOT / "results/hl_fq_8seed_cross_site/055_03_native_auto_corrected2018"
BASELINE_ROOT = {
    "HL": ROOT / "benchmark_results/054_02_hla_lowIC_four_baselines_static_level1",
    "FQ": ROOT / "benchmark_results/051_02_fqa_originIC_four_baselines_static_level1",
}
CORRECTED_BASELINE = ROOT / "benchmark_results/088_00_fq2018_input_quality_correction_run_corrected2018"
RUNNER = {
    "HL": ROOT / "src/054_hla_lowIC_site_transfer/run_054_03_hla_lowIC_five_scenario_figures.py",
    "FQ": ROOT / "src/051_fqa_originIC_site_transfer/run_051_03_fqa_originIC_five_scenario_figures.py",
}
SCENARIOS = ["null", "recorded_farmer_template", "dssat_auto", "official_extension_expert"]
LABELS = {
    "null": "Null", "recorded_farmer_template": "Recorded template",
    "dssat_auto": "DSSAT auto irrigation (N=0)", "official_extension_expert": "Official expert",
}


def load_runner(site: str):
    spec = importlib.util.spec_from_file_location(f"snapshot_reader_{site.lower()}", RUNNER[site])
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot_daily(runner, snapshot: Path, site: str, year: int, scenario: str, seed: int) -> pd.DataFrame:
    raw = runner.daily_from_snapshot(snapshot, year, scenario)
    # MgmtEvent.OUT may contain multiple internal DSSAT runs. The shared
    # parser deduplicates equal rows but can retain earlier-run amounts that
    # differ by 0.1 mm. Only the last @RUN block matches the final Summary.OUT.
    event_text = (snapshot / "MgmtEvent.OUT").read_text(encoding="latin-1", errors="ignore")
    blocks = re.split(r"(?m)^\s*@RUN Date", event_text)
    final_block = blocks[-1] if len(blocks) > 1 else event_text
    pattern = re.compile(
        r"^\s*\d+\s+[A-Z]{3}\s+\d+,\s+\d{4}\s+(\d+)\s+\d+\s+(-?\d+)\s+MZ\s+.*?\b(Irrigation|Fertilizer)\s+([0-9.]+)",
        re.IGNORECASE,
    )
    irrigation = pd.Series(0.0, index=raw.index)
    nitrogen = pd.Series(0.0, index=raw.index)
    for line in final_block.splitlines():
        hit = pattern.search(line)
        if not hit:
            continue
        doy, operation, amount = int(hit.group(1)), hit.group(3).lower(), float(hit.group(4))
        target = raw.doy.eq(doy)
        if operation.startswith("irrig"):
            irrigation.loc[target] += amount
        else:
            nitrogen.loc[target] += amount
    reward = -1.1 * irrigation - 1.58 * nitrogen
    reward.iloc[-1] += 0.158 * num(raw.grain_yield_kg_ha.iloc[-1], 0.0)
    return pd.DataFrame({
        "site": site, "station_code": "HLA" if site == "HL" else "FQA",
        "year": year, "scenario": scenario,
        "scenario_label": f"PPO seed {seed}" if scenario == scenario_key(seed) else LABELS[scenario],
        "date": raw.date.dt.strftime("%Y-%m-%d"), "doy": raw.doy, "dap": raw.dap,
        "rainfall_mm": raw.rainfall_mm, "tmax_c": raw.tmax_c, "tmin_c": raw.tmin_c,
        "grain_yield_kg_ha": raw.grain_yield_kg_ha, "biomass_kg_ha": raw.biomass_kg_ha,
        "water_stress": raw.water_stress_index_wspd, "water_stress_source": "WSPD",
        "nitrogen_stress": raw.nitrogen_stress_index_nstd, "nitrogen_stress_source": "NSTD",
        "soil_water_mm": raw.soil_water_mm,
        "irrigation_executed_mm": irrigation, "nitrogen_executed_kg_ha": nitrogen,
        "common_cumulative_reward": reward.cumsum(),
        "source_kind": "DSSAT snapshot, corrected 2018 WTH" if site == "FQ" and year == 2018 else "Frozen native DSSAT auto snapshot",
    })


def baseline_metric(row: pd.Series, site: str, seed: int, scenario: str) -> dict:
    y = float(row.grain_yield_kg_ha)
    i = float(row.actual_irrigation_mm)
    n = float(row.actual_nitrogen_kg_ha)
    pfp = num(row.PFP_N_kg_kg)
    wp = float(row.WP_ET_kg_m3)
    return {
        "site": site, "station_code": "HLA" if site == "HL" else "FQA",
        "seed": seed, "year": int(row.year), "checkpoint_step": 100000,
        "scenario": scenario, "scenario_label": LABELS[scenario],
        "grain_yield_kg_ha": y, "total_irrigation_mm": i, "total_n_kg_ha": n,
        "etcp_mm": float(row.etcp_mm), "WP_ET_kg_m3": wp, "WUE": wp,
        "PFP_N_kg_grain_per_kg_N": pfp, "NUE": math.nan,
        "common_reward": 0.158 * y - 1.1 * i - 1.58 * n,
        "source_status": "088 isolated corrected FQ2018 native baseline" if site == "FQ" and int(row.year) == 2018 else "054_02/051_02 frozen native auto baseline",
    }


def main() -> None:
    viz.BASELINE_SCENARIOS[2] = "dssat_auto"
    viz.BASELINE_LABELS["dssat_auto"] = LABELS["dssat_auto"]
    if OUT.exists() and any(OUT.iterdir()):
        # Corrected PPO replays are built first and are the only allowed prior output.
        existing = {p.name for p in OUT.iterdir()}
        if existing != {"replays"}:
            raise FileExistsError(f"Refusing to overwrite result directory: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    runners = {site: load_runner(site) for site in ("HL", "FQ")}
    native = {}
    for site in ("HL", "FQ"):
        filename = "054_02_baseline_summary.csv" if site == "HL" else "051_02_baseline_summary.csv"
        native[site] = read_csv(BASELINE_ROOT[site] / "evaluation" / filename)
        native[site]["year"] = to_number(native[site].year).astype(int)
    corrected = read_csv(CORRECTED_BASELINE / "evaluation/051_02_baseline_summary.csv")
    corrected["year"] = to_number(corrected.year).astype(int)
    cached_daily = {}
    for site in ("HL", "FQ"):
        for year in YEARS:
            snapshot = BASELINE_ROOT[site] / "snapshots" / ("HLA" if site == "HL" else "FQA") / str(year) / "dssat_auto"
            if site == "FQ" and year == 2018:
                snapshot = CORRECTED_BASELINE / "snapshots/FQA/2018/dssat_auto"
            cached_daily[(site, year, "dssat_auto")] = snapshot_daily(runners[site], snapshot, site, year, "dssat_auto", 0)
    for scenario in SCENARIOS:
        if scenario == "dssat_auto":
            continue
        snapshot = CORRECTED_BASELINE / "snapshots/FQA/2018" / scenario
        cached_daily[("FQ", 2018, scenario)] = snapshot_daily(runners["FQ"], snapshot, "FQ", 2018, scenario, 0)

    gallery = ["# HL/FQ 15-seed five-scenario figures: native auto irrigation and corrected FQ2018", "",
               "HL/FQ DSSAT auto uses native automatic irrigation with zero external N. FQ2018 uses only the isolated Tmin-corrected observed WTH; original-input figures remain in ../055_03_five_scenario/.", "",
               "The PPO models and training histories are unchanged. Corrected FQ2018 is a new deterministic checkpoint replay, not the old zero-yield endpoint.", ""]
    manifest = []
    for site, seeds in SEEDS_BY_SITE.items():
        for seed in seeds:
            prefix = f"{site.lower()}_seed{seed}"
            previous = ORIGINAL / site / f"best_seed_seed{seed}" / "tables"
            folder = OUT / site / f"seed{seed}"
            figures = folder / "figures"
            tables = folder / "tables"
            if folder.exists():
                raise FileExistsError(folder)
            figures.mkdir(parents=True)
            tables.mkdir(parents=True)
            metrics = read_csv(previous / f"{prefix}_yearly_five_scenario_metrics.csv")
            metrics["year"] = to_number(metrics.year).astype(int)
            metrics.loc[metrics.scenario.eq("dssat_auto_external_n"), "scenario"] = "dssat_auto"
            daily = read_csv(previous / f"{prefix}_five_scenario_daily.csv")
            daily["year"] = to_number(daily.year).astype(int)
            daily = daily.loc[~daily.scenario.eq("dssat_auto_external_n")].copy()
            if site == "FQ":
                daily = daily.loc[~daily.year.eq(2018)].copy()
            appended = []
            for year in YEARS:
                source = corrected if site == "FQ" and year == 2018 else native[site]
                scenarios = SCENARIOS if site == "FQ" and year == 2018 else ["dssat_auto"]
                for scenario in scenarios:
                    rows = source.loc[source.year.eq(year) & source.scenario.eq(scenario)]
                    if len(rows) != 1:
                        raise ValueError(f"Missing/duplicate frozen baseline: {site} {year} {scenario}")
                    data = baseline_metric(rows.iloc[0], site, seed, scenario)
                    old = metrics.index[metrics.year.eq(year) & metrics.scenario.eq(scenario)]
                    if len(old) != 1:
                        raise ValueError(f"Missing target baseline row: {site} seed{seed} {year} {scenario}")
                    for key, value in data.items():
                        metrics.at[old[0], key] = value
                    if scenario == "dssat_auto":
                        appended.append(cached_daily[(site, year, scenario)])
                    elif site == "FQ" and year == 2018:
                        appended.append(cached_daily[(site, year, scenario)])
            if site == "FQ":
                replay_root = OUT / "replays/FQ" / f"seed{seed}"
                corrected_ppo = json.loads((replay_root / "corrected2018_ppo_metrics.json").read_text(encoding="utf-8"))
                idx = metrics.index[metrics.year.eq(2018) & metrics.scenario.eq(scenario_key(seed))]
                if len(idx) != 1:
                    raise ValueError(f"Missing PPO FQ2018 row: seed{seed}")
                idx = idx[0]
                y = float(corrected_ppo["corrected_yield_kg_ha"])
                i = float(corrected_ppo["corrected_irrigation_mm"])
                n = float(corrected_ppo["corrected_n_kg_ha"])
                for key, val in {
                    "grain_yield_kg_ha": y, "total_irrigation_mm": i, "total_n_kg_ha": n,
                    "etcp_mm": corrected_ppo["corrected_etcp_mm"], "WP_ET_kg_m3": corrected_ppo["corrected_WP_ET_kg_m3"],
                    "WUE": corrected_ppo["corrected_WP_ET_kg_m3"],
                    "PFP_N_kg_grain_per_kg_N": corrected_ppo["corrected_PFP_N_kg_kg"],
                    "common_reward": 0.158 * y - 1.1 * i - 1.58 * n,
                    "source_status": "Corrected FQ2018 observed-WTH deterministic 100K replay; old zero-yield endpoint intentionally replaced",
                    "summary_daily_path": (replay_root / "corrected2018_ppo_daily.csv").relative_to(ROOT).as_posix(),
                    "daily_output_path": (replay_root / "corrected2018_ppo_daily.csv").relative_to(ROOT).as_posix(),
                }.items():
                    metrics.at[idx, key] = val
                ppo_daily = snapshot_daily(runners["FQ"], replay_root / "snapshot", "FQ", 2018, scenario_key(seed), seed)
                appended.append(ppo_daily)
            daily = pd.concat([daily, *appended], ignore_index=True, sort=False)
            daily["year"] = to_number(daily.year).astype(int)
            metrics = metrics.sort_values(["year", "scenario"]).reset_index(drop=True)
            keys = set(zip(metrics.year, metrics.scenario))
            expected = {(year, scenario) for year in YEARS for scenario in [*SCENARIOS, scenario_key(seed)]}
            if len(metrics) != 50 or keys != expected or set(daily.year) != set(YEARS):
                raise ValueError(f"Incomplete corrected five-scenario coverage: {site} seed{seed}")
            if not to_number(metrics.loc[metrics.scenario.eq("dssat_auto"), "total_n_kg_ha"]).eq(0).all():
                raise ValueError(f"Native auto contains N: {site} seed{seed}")
            if site == "FQ" and to_number(metrics.loc[metrics.year.eq(2018), "grain_yield_kg_ha"]).le(0).any():
                raise ValueError(f"Corrected FQ2018 still has zero-yield scenario: seed{seed}")
            audit = json.loads((previous / f"{prefix}_action_sequence_audit.json").read_text(encoding="utf-8"))
            if site == "FQ":
                original_audit = audit
                (tables / f"{prefix}_original_weather_action_sequence_audit.json").write_text(
                    json.dumps(original_audit, indent=2), encoding="utf-8")
                corrected_events = daily.loc[
                    daily.year.eq(2018) & daily.scenario.eq(scenario_key(seed)) &
                    (to_number(daily.irrigation_executed_mm).gt(0) | to_number(daily.nitrogen_executed_kg_ha).gt(0)),
                    ["date", "dap", "irrigation_executed_mm", "nitrogen_executed_kg_ha"],
                ].to_dict("records")
                audit = {
                    "site": site, "seed": seed,
                    "original_weather_action_audit_file": f"{prefix}_original_weather_action_sequence_audit.json",
                    "uncorrected_2014_2017_2019_2023_sequences": {
                        year: seq for year, seq in original_audit["yearly_action_sequences"].items() if year != "2018"
                    },
                    "corrected_2018_action_events_from_snapshot": corrected_events,
                    "identical_across_all_validation_years": None,
                    "unique_yearly_action_sequence_count": None,
                    "corrected_2018_timing_note": "Corrected-year events come from DSSAT MgmtEvent.OUT; other years come from PPO validation logs. DAP origin differs, so exact cross-year sequence status is not asserted here.",
                }
            write_table_files(site, seed, metrics, tables, audit)
            ppo_actions = daily.loc[daily.scenario.eq(scenario_key(seed)), [
                "site", "station_code", "year", "date", "doy", "dap", "irrigation_executed_mm", "nitrogen_executed_kg_ha",
            ]].rename(columns={"irrigation_executed_mm": "irrigation_mm", "nitrogen_executed_kg_ha": "nitrogen_kg_ha"})
            write_event_and_daily_tables(site, seed, daily, ppo_actions, tables)
            plot_metric_bars(site, seed, metrics, figures)
            plot_management_bars(site, seed, metrics, figures)
            plot_reward(site, seed, metrics, figures)
            for year in YEARS:
                plot_daily_year(site, seed, year, daily, figures, True)
                plot_management_year(site, seed, year, metrics, figures)
            (folder / "README.md").write_text(
                f"# {site} seed {seed}: native auto irrigation five-scenario figures\n\n"
                + ("FQ2018 uses the isolated corrected observed WTH and a new PPO checkpoint replay. Other years retain the original historical observed WTH.\n\n" if site == "FQ" else "All years retain the historical observed WTH.\n\n")
                + f"- Figures: 23 PNG in `figures/` (10 daily, 10 year-specific water/N bars, 3 across-year).\n"
                + f"- Metrics: `tables/{prefix}_yearly_five_scenario_metrics.csv`.\n"
                + f"- Full daily water/N events: `tables/{prefix}_five_scenario_daily.csv`.\n"
                + ("- Corrected 2018 MgmtEvent timing uses DSSAT DAP convention; see action audit JSON.\n" if site == "FQ" else "")
                , encoding="utf-8"
            )
            gallery += [f"## {site} seed {seed}", "",
                        f"[指标图]({site}/seed{seed}/figures/{prefix}_five_scenario_metrics.png) · [十年水氮图]({site}/seed{seed}/figures/{prefix}_five_scenario_management.png) · [逐年指标表]({site}/seed{seed}/tables/{prefix}_yearly_five_scenario_metrics.csv)", ""]
            for year in YEARS:
                gallery.append(f"- {year}: [五情景逐日图]({site}/seed{seed}/figures/{prefix}_{year}_five_scenario_daily.png) · [灌溉/施氮柱状图]({site}/seed{seed}/figures/{prefix}_{year}_five_scenario_management_bars.png)")
            gallery.append("")
            manifest.append({"site": site, "seed": seed, "scenarios": [*SCENARIOS, scenario_key(seed)], "years": YEARS,
                             "figures": len(list(figures.glob("*.png"))), "native_auto_n_zero": True,
                             "fq2018_corrected": site == "FQ"})
            print(f"DONE {site} seed{seed}: 23 figures, native auto N=0", flush=True)
    (OUT / "GALLERY.md").write_text("\n".join(gallery), encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Complete: {len(manifest)} seeds / {sum(x['figures'] for x in manifest)} PNG", flush=True)


if __name__ == "__main__":
    main()
