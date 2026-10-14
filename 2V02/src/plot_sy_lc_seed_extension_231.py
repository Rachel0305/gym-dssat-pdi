"""Create frozen-style five-scenario validation packages for SY/LC seeds 1-7.

Only PPO is replayed. The four baseline trajectories and exact metrics are
read from the corresponding frozen seed0 report for the same station/year.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/053_lca_lowIC_site_transfer"))

import build_sya_configured_five_scenario_daily_plots_046_06 as sy_plot
import run_053_03_lca_lowIC_five_scenario_figures as lc_plot

BASE = ROOT / "results/sy_lc_8seed_extension_231"
REPORT = BASE / "five_scenario_figures_v2"
FROZEN = {
    "SY": ROOT / "benchmark_results/046_06_sya_originIC_046_10_sya_originIC_expanded_action_maskableppo_five_scenario_daily_ckpt100000",
    "LC": ROOT / "benchmark_results/053_03_lca_lowIC_053_00_lca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000",
}
PREFIX = {"SY": "046_06_sya", "LC": "053_03_lca"}
STATION = {"SY": "SYA", "LC": "LCA"}
TASK = {"SY": "sya_originIC_expanded_action_maskableppo", "LC": "lca_lowIC_expanded_action_maskableppo"}
SCENARIOS = ["null", "recorded_farmer_template", "dssat_auto_external_n", "official_extension_expert", "rl_candidate"]
YEARS = list(range(2014, 2024))
METRICS = ["grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg"]
RESOURCES = ["actual_irrigation_mm", "actual_nitrogen_kg_ha", "unified_reward"]


def source_tables(site: str) -> tuple[Path, Path]:
    folder = FROZEN[site] / "tables"
    return folder / f"{PREFIX[site]}_five_scenario_daily.csv", folder / f"{PREFIX[site]}_five_scenario_season_summary.csv"


def normalize_summary(site: str, summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary.copy()
    if site == "SY":
        frame = frame.rename(columns={
            "final_grain_kg_ha": "grain_yield_kg_ha",
            "final_biomass_kg_ha": "biomass_kg_ha",
            "irrigation_mm": "actual_irrigation_mm",
            "nitrogen_kg_ha": "actual_nitrogen_kg_ha",
        })
        frame.insert(0, "site", "SY")
        frame.insert(0, "station_code", "SYA")
    return frame


def ppo_summary(site: str, year: int, daily: pd.DataFrame, snapshot: Path, plot_module) -> dict:
    last = daily.iloc[-1]
    yield_kg = float(pd.to_numeric(pd.Series([last.grain_yield_kg_ha]), errors="coerce").iloc[0])
    metric = plot_module.baseline.metrics_from_snapshot(snapshot, yield_kg)
    n = float(metric["actual_nitrogen_kg_ha"])
    pfp = metric["PFP_N_kg_kg"]
    return {
        "station_code": STATION[site], "site": site, "year": year,
        "scenario": "rl_candidate", "grain_yield_kg_ha": yield_kg,
        "biomass_kg_ha": float(last.biomass_kg_ha),
        "actual_irrigation_mm": float(metric["actual_irrigation_mm"]),
        "actual_nitrogen_kg_ha": n, "etcp_mm": float(metric["etcp_mm"]),
        "WP_ET_kg_m3": float(metric["WP_ET_kg_m3"]),
        "PFP_N_kg_kg": float(pfp) if pd.notna(pfp) and n > 0 else np.nan,
        "max_wspd": float(pd.to_numeric(daily.water_stress_index_wspd, errors="coerce").max()),
        "max_nstd": float(pd.to_numeric(daily.nitrogen_stress_index_nstd, errors="coerce").max()),
        "unified_reward": float(last.unified_cumulative_reward),
        "snapshot_path": snapshot.relative_to(ROOT).as_posix(),
    }


def run(site: str, seed: int) -> None:
    plot = sy_plot if site == "SY" else lc_plot
    config_path = BASE / "configs" / f"{site.lower()}_seed{seed}.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert cfg["seed"] == seed and cfg["station_code"] == STATION[site]
    ppo_root = ROOT / "benchmark_results" / f"231_{site.lower()}_seed{seed}_{TASK[site]}_seed{seed}_extension"
    formal = json.loads((ppo_root / f"231_{site.lower()}_seed{seed}_formal_result.json").read_text(encoding="utf-8"))
    gate = formal["action_audit" if site == "SY" else "action_gate"]
    if gate["observed_validation_rows"] != 10 or not gate["all_daily_files_exist"] or not gate["all_actions_on_declared_grid"]:
        raise RuntimeError(f"Technical validation incomplete: {site} seed{seed}")
    output = REPORT / site / f"seed_{seed:02d}"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite a replay package: {output}")
    figures, tables = output / "figures", output / "tables"
    figures.mkdir(parents=True)
    tables.mkdir(parents=True)

    daily_path, summary_path = source_tables(site)
    frozen_daily = pd.read_csv(daily_path, keep_default_na=False)
    frozen_summary = normalize_summary(site, pd.read_csv(summary_path, keep_default_na=False))
    frozen_daily = frozen_daily[~frozen_daily.scenario.eq("rl_candidate")].copy()
    frozen_summary = frozen_summary[~frozen_summary.scenario.eq("rl_candidate")].copy()
    expected = {(year, scenario) for year in YEARS for scenario in SCENARIOS[:-1]}
    if set(zip(frozen_summary.year, frozen_summary.scenario)) != expected:
        raise RuntimeError(f"Frozen baseline year/scenario coverage mismatch: {site}")

    day_frames = []
    summary_rows = []
    audits = []
    for year in YEARS:
        snapshot = plot.replay_ppo_snapshot(cfg, 100000, year, output, ppo_root)
        ppo_day = plot.daily_from_snapshot(snapshot, year, "rl_candidate")
        baseline_day = frozen_daily[frozen_daily.year.eq(year)].copy()
        if set(baseline_day.scenario) != set(SCENARIOS[:-1]):
            raise RuntimeError(f"Missing frozen daily baseline: {site} {year}")
        merged = plot.add_unified_reward(pd.concat([baseline_day, ppo_day], ignore_index=True, sort=False))
        plot.plot_year(merged, year, figures)
        day_frames.append(merged)
        summary_rows.extend(frozen_summary[frozen_summary.year.eq(year)].to_dict(orient="records"))
        summary_rows.append(ppo_summary(site, year, merged[merged.scenario.eq("rl_candidate")], snapshot, plot))
        audits.append({"year": year, "snapshot": snapshot.relative_to(ROOT).as_posix(), "ppo_daily_rows": len(ppo_day)})
        print(f"{site} seed{seed} {year} replayed", flush=True)

    full_daily = pd.concat(day_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values(["year", "scenario"]).reset_index(drop=True)
    if len(summary) != 50 or set(zip(summary.year, summary.scenario)) != {(y, s) for y in YEARS for s in SCENARIOS}:
        raise RuntimeError(f"Five-scenario summary incomplete: {site} seed{seed}")
    for column in METRICS + RESOURCES:
        summary[column] = pd.to_numeric(summary[column], errors="coerce")
    event_rows = full_daily[full_daily.irrigation_executed_mm.gt(0) | full_daily.nitrogen_executed_kg_ha.gt(0)].copy()
    full_daily.to_csv(tables / f"{PREFIX[site]}_five_scenario_daily.csv", index=False, encoding="utf-8-sig")
    event_rows.to_csv(tables / f"{PREFIX[site]}_five_scenario_management_events.csv", index=False, encoding="utf-8-sig")
    if site == "SY":
        sy_summary = summary.drop(columns=["station_code", "site", "snapshot_path"], errors="ignore").rename(columns={
            "grain_yield_kg_ha": "final_grain_kg_ha", "biomass_kg_ha": "final_biomass_kg_ha",
            "actual_irrigation_mm": "irrigation_mm", "actual_nitrogen_kg_ha": "nitrogen_kg_ha",
        })
        sy_summary.to_csv(tables / f"{PREFIX[site]}_five_scenario_season_summary.csv", index=False, encoding="utf-8-sig")
    else:
        summary.to_csv(tables / f"{PREFIX[site]}_five_scenario_season_summary.csv", index=False, encoding="utf-8-sig")
    lc_plot.draw_grouped(summary, YEARS, METRICS, figures / f"{PREFIX[site]}_five_scenario_metrics.png")
    lc_plot.draw_grouped(summary, YEARS, RESOURCES, figures / f"{PREFIX[site]}_five_scenario_management.png")
    gap_rows = []
    for year, subset in summary.groupby("year"):
        ppo = subset[subset.scenario.eq("rl_candidate")].iloc[0]
        other = subset[~subset.scenario.eq("rl_candidate")]
        for metric in METRICS + RESOURCES:
            maximum = pd.to_numeric(other[metric], errors="coerce").max()
            value = float(ppo[metric]) if pd.notna(ppo[metric]) else np.nan
            gap_rows.append({"year": year, "metric": metric, "ppo_value": value, "four_baseline_max": maximum, "ppo_minus_four_max": value - maximum if pd.notna(maximum) else np.nan})
    pd.DataFrame(gap_rows).to_csv(tables / f"{PREFIX[site]}_metric_gaps_vs_four_baseline_max.csv", index=False, encoding="utf-8-sig")
    (output / "audit.json").write_text(json.dumps({
        "site": site, "seed": seed, "checkpoint": 100000,
        "source_frozen_baselines": FROZEN[site].relative_to(ROOT).as_posix(),
        "ppo_run": ppo_root.relative_to(ROOT).as_posix(),
        "behavior_gate": gate, "replay_years": audits,
        "figure_count": len(list(figures.glob("*.png"))), "summary_rows": len(summary),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        f"# {site} seed{seed}: 100K five-scenario validation\n\n"
        f"Baseline trajectories and exact metrics are from `{FROZEN[site].relative_to(ROOT).as_posix()}`. "
        f"PPO uses the completed seed{seed} model and 2014–2023 replay. Each replay endpoint "
        "is checked against its saved training evaluation. Poor action diversity is retained as a result.\n\n"
        "Figures: 10 frozen-style daily process panels, one metrics panel, one management panel. "
        "Tables: daily, management events, season summary, and metric gaps.\n",
        encoding="utf-8",
    )
    print(output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=["SY", "LC"], required=True)
    parser.add_argument("--seed", type=int, choices=range(1, 8), required=True)
    args = parser.parse_args()
    run(args.site, args.seed)
