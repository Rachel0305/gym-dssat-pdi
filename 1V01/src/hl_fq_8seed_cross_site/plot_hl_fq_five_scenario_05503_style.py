"""Create YC-055_03-style five-scenario figures for completed HL/FQ PPO seeds.

This is a local, read-only analysis of existing validation summaries, daily PPO
traces, and frozen four-baseline tables. It does not import DSSAT, train PPO,
run a replay, or modify experiment configurations.
"""

from __future__ import annotations

import json
import math
import sys
import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from hl_fq_8seed_cross_site.analyze_hl_fq_8seed import (  # noqa: E402
    YEARS,
    config_and_output,
    daily_output_path,
    extract_year,
    selected_summary,
)


OUT_ROOT = ROOT / "results/hl_fq_8seed_cross_site/055_03_five_scenario"
BASELINE_SCENARIOS = [
    "null",
    "recorded_farmer_template",
    "dssat_auto_external_n",
    "official_extension_expert",
]
COLORS = ["#555555", "#C44E52", "#D8A305", "#7E63B6", "#2A9D55"]
LINE_STYLES = ["-", "--", "-.", ":", "-"]
BASELINE_FILES = {
    "HL": {
        "summary": ROOT / "benchmark_results/054_03_hla_lowIC_054_00_hla_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/tables/054_03_hla_five_scenario_season_summary.csv",
        "daily": ROOT / "benchmark_results/054_03_hla_lowIC_054_00_hla_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/tables/054_03_hla_five_scenario_daily.csv",
    },
    "FQ": {
        "summary": ROOT / "benchmark_results/051_03_fqa_originIC_051_00_fqa_originIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/tables/051_03_fqa_five_scenario_season_summary.csv",
        "daily": ROOT / "benchmark_results/051_03_fqa_originIC_051_00_fqa_originIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/tables/051_03_fqa_five_scenario_daily.csv",
    },
}
SEEDS_BY_SITE = {"HL": list(range(8)), "FQ": list(range(7))}
BASELINE_LABELS = {
    "null": "Null",
    "recorded_farmer_template": "Recorded template",
    "dssat_auto_external_n": "DSSAT auto + external N",
    "official_extension_expert": "Official expert",
}


def num(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    # keep_default_na=False is necessary: pandas otherwise treats scenario="null"
    # as a missing value and silently drops the Null baseline.
    return pd.read_csv(path, keep_default_na=False)


def csv_number(value: Any) -> float:
    return num(value)


def to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def scenario_key(seed: int) -> str:
    return f"ppo_seed_{seed}"


def scenario_label(seed: int) -> str:
    return f"PPO seed {seed}"


def scenario_order(seed: int) -> list[str]:
    return [*BASELINE_SCENARIOS, scenario_key(seed)]


def labels(seed: int) -> list[str]:
    return [*(BASELINE_LABELS[s] for s in BASELINE_SCENARIOS), scenario_label(seed)]


def baseline_paths(site: str) -> tuple[Path, Path]:
    sources = BASELINE_FILES[site]
    return sources["summary"], sources["daily"]


def load_baseline(site: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path, daily_path = baseline_paths(site)
    summary = read_csv(summary_path)
    daily = read_csv(daily_path)
    summary["year"] = to_number(summary["year"]).astype(int)
    summary = summary.loc[summary.scenario.isin(BASELINE_SCENARIOS)].copy()
    expected = {(year, scenario) for year in YEARS for scenario in BASELINE_SCENARIOS}
    actual = set(zip(summary.year.astype(int), summary.scenario.astype(str)))
    if actual != expected or len(summary) != len(expected):
        raise ValueError(f"Incomplete/duplicate {site} four-baseline year coverage")
    daily["year"] = to_number(daily["year"]).astype(int)
    daily = daily.loc[daily.scenario.isin(BASELINE_SCENARIOS)].copy()
    if set(daily.year.unique()) != set(YEARS):
        raise ValueError(f"Incomplete {site} baseline daily year coverage")
    return summary, daily


def load_ppo_seed(site: str, seed: int, base_summary: pd.DataFrame,
                  base_daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    spec = config_and_output(site, seed)
    _config_path, output_root = spec
    station_code = "HLA" if site == "HL" else "FQA"
    summary_by_year = selected_summary(output_root, station_code, seed)
    metrics: list[dict[str, Any]] = []
    daily_records: list[dict[str, Any]] = []
    ppo_full_daily: list[dict[str, Any]] = []
    yearly_sequences: dict[str, list[list[float | int]]] = {}
    weather_audit: dict[str, Any] = {}
    ppo_scenario = scenario_key(seed)

    for year in YEARS:
        summary_row = summary_by_year[year]
        source_daily_path = daily_output_path(output_root, station_code, year, seed)
        ppo_metric, ppo_events, extra = extract_year(
            site, seed, year, summary_row, source_daily_path, output_root
        )
        ppo_metric.update({
            "scenario": ppo_scenario,
            "scenario_label": scenario_label(seed),
            "WP_ET_kg_m3": None,
            "WUE": None,
            "NUE": None,
            "etcp_mm": None,
            "plant_n_uptake_kg_ha": None,
            "source_status": "PPO validation summary/daily trace; no exact ETCP replay or plant N uptake output",
        })
        metrics.append(ppo_metric)
        yearly_sequences[str(year)] = extra["sequence"]["event_sequence"]

        ppo_raw = read_csv(source_daily_path)
        if ppo_raw.empty:
            raise ValueError(f"Empty PPO daily trace: {source_daily_path}")
        ppo_raw["doy"] = to_number(ppo_raw["doy"])
        ppo_raw["dap"] = to_number(ppo_raw["dap"])
        for name in ("rain", "tmax", "tmin", "swfac", "nstres", "topwt", "grnwt",
                     "safe_action_amir", "safe_action_anfer", "discrete_action_index",
                     "requested_discrete_action_index"):
            if name in ppo_raw.columns:
                ppo_raw[name] = to_number(ppo_raw[name])
        action_i = to_number(ppo_raw["safe_action_amir"]).fillna(0.0)
        action_n = to_number(ppo_raw["safe_action_anfer"]).fillna(0.0)
        reward_increment = -1.1 * action_i - 1.58 * action_n
        reward_increment.iloc[-1] += 0.158 * num(ppo_metric["grain_yield_kg_ha"])
        ppo_raw["_common_reward_cumulative"] = reward_increment.cumsum()

        baseline_null = base_daily.loc[
            (base_daily.year == year) & (base_daily.scenario == "null"),
            ["date", "rainfall_mm", "tmax_c", "tmin_c"],
        ].copy()
        weather = baseline_null.merge(
            ppo_raw[["date", "rain", "tmax", "tmin"]], on="date", how="inner"
        )
        weather_diffs = {
            "rain_mm_max_abs_diff": float((to_number(weather.rainfall_mm) - to_number(weather.rain)).abs().max()) if not weather.empty else None,
            "tmax_c_max_abs_diff": float((to_number(weather.tmax_c) - to_number(weather.tmax)).abs().max()) if not weather.empty else None,
            "tmin_c_max_abs_diff": float((to_number(weather.tmin_c) - to_number(weather.tmin)).abs().max()) if not weather.empty else None,
        }
        matched = len(weather)
        # The frozen baseline daily table stores temperatures to one decimal;
        # allow half of that rounding unit while requiring exact rainfall.
        weather_match = (
            matched == len(ppo_raw)
            and weather_diffs["rain_mm_max_abs_diff"] is not None
            and weather_diffs["rain_mm_max_abs_diff"] <= 1e-6
            and weather_diffs["tmax_c_max_abs_diff"] is not None
            and weather_diffs["tmax_c_max_abs_diff"] <= 0.051
            and weather_diffs["tmin_c_max_abs_diff"] is not None
            and weather_diffs["tmin_c_max_abs_diff"] <= 0.051
        )
        weather_audit[str(year)] = {
            "ppo_daily_rows": len(ppo_raw),
            "date_matched_weather_rows": matched,
            "weather_values_match_by_calendar_date": weather_match,
            **weather_diffs,
        }

        # Keep all daily PPO action rows, including no-ops, to preserve a full
        # day-by-day trace alongside the event-only action sequence.
        for idx, raw in ppo_raw.iterrows():
            ppo_full_daily.append({
                "site": site,
                "station_code": station_code,
                "seed": seed,
                "year": year,
                "date": raw.get("date", ""),
                "doy": num(raw.get("doy")),
                "dap": num(raw.get("dap")),
                "discrete_action_index": num(raw.get("discrete_action_index")),
                "requested_discrete_action_index": num(raw.get("requested_discrete_action_index")),
                "mask_forced_noop": raw.get("mask_forced_noop", ""),
                "irrigation_mm": num(raw.get("safe_action_amir"), 0.0),
                "nitrogen_kg_ha": num(raw.get("safe_action_anfer"), 0.0),
                "scenario": ppo_scenario,
                "source_daily_path": source_daily_path.relative_to(ROOT).as_posix(),
            })
            daily_records.append({
                "site": site,
                "station_code": station_code,
                "year": year,
                "scenario": ppo_scenario,
                "scenario_label": scenario_label(seed),
                "date": raw.get("date", ""),
                "doy": num(raw.get("doy")),
                "dap": num(raw.get("dap")),
                "rainfall_mm": num(raw.get("rain")),
                "tmax_c": num(raw.get("tmax")),
                "tmin_c": num(raw.get("tmin")),
                "grain_yield_kg_ha": num(raw.get("grnwt")),
                "biomass_kg_ha": num(raw.get("topwt")),
                "water_stress": num(raw.get("swfac")),
                "water_stress_source": "SWFAC",
                "nitrogen_stress": num(raw.get("nstres")),
                "nitrogen_stress_source": "NSTRES",
                "soil_water_mm": math.nan,
                "irrigation_executed_mm": num(raw.get("safe_action_amir"), 0.0),
                "nitrogen_executed_kg_ha": num(raw.get("safe_action_anfer"), 0.0),
                "common_cumulative_reward": num(raw.get("_common_reward_cumulative")),
                "source_kind": "PPO daily validation log",
            })

    # Add frozen baseline metrics once for this seed's five-scenario table.
    for _, row in base_summary.iterrows():
        year = int(row.year)
        metrics.append({
            "site": site,
            "station_code": station_code,
            "seed": seed,
            "year": year,
            "checkpoint_step": 100000,
            "scenario": str(row.scenario),
            "scenario_label": BASELINE_LABELS[str(row.scenario)],
            "grain_yield_kg_ha": csv_number(row.grain_yield_kg_ha),
            "total_irrigation_mm": csv_number(row.actual_irrigation_mm),
            "total_n_kg_ha": csv_number(row.actual_nitrogen_kg_ha),
            "etcp_mm": csv_number(row.etcp_mm),
            "WP_ET_kg_m3": csv_number(row.WP_ET_kg_m3),
            "PFP_N_kg_grain_per_kg_N": csv_number(row.PFP_N_kg_kg),
            "NUE": math.nan,
            "WUE": math.nan,
            "plant_n_uptake_kg_ha": math.nan,
            "common_reward": csv_number(row.unified_reward),
            "source_status": "Frozen four-baseline seasonal summary",
        })

    # Add the four baseline daily process traces to the five-scenario daily file.
    baseline_daily = base_daily.copy()
    baseline_daily["scenario_label"] = baseline_daily.scenario.map(BASELINE_LABELS)
    for _, raw in baseline_daily.iterrows():
        daily_records.append({
            "site": site,
            "station_code": station_code,
            "year": int(raw.year),
            "scenario": str(raw.scenario),
            "scenario_label": raw.scenario_label,
            "date": raw.get("date", ""),
            "doy": num(raw.get("doy")),
            "dap": num(raw.get("dap")),
            "rainfall_mm": num(raw.get("rainfall_mm")),
            "tmax_c": num(raw.get("tmax_c")),
            "tmin_c": num(raw.get("tmin_c")),
            "grain_yield_kg_ha": num(raw.get("grain_yield_kg_ha")),
            "biomass_kg_ha": num(raw.get("biomass_kg_ha")),
            "water_stress": num(raw.get("water_stress_index_wspd")),
            "water_stress_source": "WSPD",
            "nitrogen_stress": num(raw.get("nitrogen_stress_index_nstd")),
            "nitrogen_stress_source": "NSTD",
            "soil_water_mm": num(raw.get("soil_water_mm")),
            "irrigation_executed_mm": num(raw.get("irrigation_executed_mm"), 0.0),
            "nitrogen_executed_kg_ha": num(raw.get("nitrogen_executed_kg_ha"), 0.0),
            "common_cumulative_reward": num(raw.get("unified_cumulative_reward")),
            "source_kind": "Frozen four-baseline daily trace",
        })

    daily_frame = pd.DataFrame(daily_records)
    daily_frame["scenario"] = daily_frame.scenario.astype(str)
    metric_frame = pd.DataFrame(metrics)
    metric_frame["scenario"] = metric_frame.scenario.astype(str)
    metric_frame = metric_frame.sort_values(["year", "scenario"], key=lambda col: col.map(
        {name: idx for idx, name in enumerate(scenario_order(seed))} if col.name == "scenario" else {}
    ) if col.name == "scenario" else col).reset_index(drop=True)

    counts = Counter(json.dumps(yearly_sequences[str(year)], separators=(",", ":")) for year in YEARS)
    signature_count = len(counts)
    sequence_audit = {
        "site": site,
        "seed": seed,
        "years": YEARS,
        "sequence_definition": "ordered positive irrigation/N event sequence [DAP, irrigation_mm, nitrogen_kg_ha] from validated PPO daily trace",
        "identical_across_all_validation_years": signature_count == 1,
        "unique_yearly_action_sequence_count": signature_count,
        "sequence_frequency": {str(n): count for n, count in counts.items()},
        "yearly_action_sequences": yearly_sequences,
        "weather_input_match_by_calendar_date": weather_audit,
        "weather_match_all_years": all(x["weather_values_match_by_calendar_date"] for x in weather_audit.values()),
    }
    return metric_frame, daily_frame, pd.DataFrame(ppo_full_daily), sequence_audit


def write_table_files(site: str, seed: int, metrics: pd.DataFrame,
                      tables_dir: Path, audit: dict[str, Any]) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    key = scenario_key(seed)
    order = scenario_order(seed)
    metrics = metrics.copy()
    metrics.to_csv(tables_dir / f"{site.lower()}_seed{seed}_yearly_five_scenario_metrics.csv",
                   index=False, encoding="utf-8-sig")

    by_key = {
        (int(row["year"]), str(row["scenario"])): row
        for row in metrics.to_dict(orient="records")
    }
    yield_table: list[dict[str, Any]] = []
    productivity_table: list[dict[str, Any]] = []
    def fmt(value: Any, decimals: int = 1) -> str:
        v = num(value)
        if not math.isfinite(v):
            return "—"
        if abs(v - round(v)) < 1e-8:
            return str(int(round(v)))
        return f"{v:.{decimals}f}"

    pfp_key = "PFP_N_kg_grain_per_kg_N"
    for year in YEARS:
        row1: dict[str, Any] = {"year": year}
        row2: dict[str, Any] = {"year": year}
        for scenario in order:
            item = by_key[(year, scenario)]
            col = scenario_label(seed) if scenario == key else scenario
            row1[col] = " / ".join([
                fmt(item["grain_yield_kg_ha"], 1),
                fmt(item["total_irrigation_mm"], 1),
                fmt(item["total_n_kg_ha"], 1),
            ])
            row2[col] = " / ".join([
                fmt(item["WP_ET_kg_m3"], 2),
                fmt(item[pfp_key], 1),
            ])
        yield_table.append(row1)
        productivity_table.append(row2)

    pd.DataFrame(yield_table).to_csv(
        tables_dir / f"{site.lower()}_seed{seed}_yearly_yield_irrigation_n_table.csv",
        index=False, encoding="utf-8-sig",
    )
    pd.DataFrame(productivity_table).to_csv(
        tables_dir / f"{site.lower()}_seed{seed}_yearly_wp_et_pfp_n_table.csv",
        index=False, encoding="utf-8-sig",
    )
    (tables_dir / f"{site.lower()}_seed{seed}_action_sequence_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_event_and_daily_tables(site: str, seed: int, daily: pd.DataFrame,
                                 ppo_actions: pd.DataFrame, tables_dir: Path) -> None:
    daily = daily.copy()
    daily.to_csv(tables_dir / f"{site.lower()}_seed{seed}_five_scenario_daily.csv",
                 index=False, encoding="utf-8-sig")
    ppo_actions.to_csv(tables_dir / f"{site.lower()}_seed{seed}_ppo_daily_action_trace.csv",
                       index=False, encoding="utf-8-sig")
    events = daily.loc[
        to_number(daily.irrigation_executed_mm).gt(0)
        | to_number(daily.nitrogen_executed_kg_ha).gt(0),
        ["site", "station_code", "year", "scenario", "scenario_label", "date", "doy", "dap",
         "irrigation_executed_mm", "nitrogen_executed_kg_ha"],
    ].copy()
    events.to_csv(tables_dir / f"{site.lower()}_seed{seed}_five_scenario_management_events.csv",
                  index=False, encoding="utf-8-sig")


def plot_metric_bars(site: str, seed: int, metrics: pd.DataFrame, figures_dir: Path) -> None:
    order = scenario_order(seed)
    scenario_labels = labels(seed)
    x = np.arange(len(YEARS))
    width = 0.16
    fig, axes = plt.subplots(3, 1, figsize=(15.5, 11.5), sharex=True)
    fields = [
        ("grain_yield_kg_ha", "grain_yield_kg_ha", False),
        ("WP_ET_kg_m3", "WP_ET_kg_m3", True),
        ("PFP_N_kg_grain_per_kg_N", "PFP_N_kg_kg", False),
    ]
    for ax, (field, ylabel, ppo_wp_panel) in zip(axes, fields):
        for idx, (scenario, label, color) in enumerate(zip(order, scenario_labels, COLORS)):
            rows = metrics.loc[metrics.scenario.eq(scenario)].set_index("year").reindex(YEARS)
            values = to_number(rows[field]).to_numpy(dtype=float)
            valid = np.isfinite(values)
            if valid.any():
                ax.bar(x[valid] + (idx - 2) * width, values[valid], width,
                       label=label, color=color)
        if ppo_wp_panel and not to_number(metrics.loc[metrics.scenario.eq(scenario_key(seed)), field]).notna().all():
            ax.text(0.015, 0.93, "PPO WP_ET unavailable: no exact Summary.OUT + ETCP replay",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.24)
    fq2018 = metrics.loc[metrics.year.eq(2018)] if site == "FQ" else pd.DataFrame()
    if site == "FQ" and len(fq2018) == 5 and to_number(fq2018.grain_yield_kg_ha).eq(0).all():
        axes[0].text(4, axes[0].get_ylim()[1] * 0.08, "2018: observed WTH input anomaly; all yields = 0",
                     ha="center", va="bottom", rotation=90, fontsize=8, color="#9A3333")
    handles = [Patch(color=color, label=label) for color, label in zip(COLORS, scenario_labels)]
    for ax in axes:
        ax.legend(handles=handles, ncol=3, fontsize=8, frameon=True, loc="best")
    axes[-1].set_xticks(x, YEARS)
    axes[-1].set_xlabel("Validation year")
    fig.suptitle(f"{site} | historical-weather PPO seed {seed} vs four baselines",
                 x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(figures_dir / f"{site.lower()}_seed{seed}_five_scenario_metrics.png", dpi=140)
    plt.close(fig)


def plot_management_bars(site: str, seed: int, metrics: pd.DataFrame, figures_dir: Path) -> None:
    order = scenario_order(seed)
    scenario_labels = labels(seed)
    x = np.arange(len(YEARS))
    width = 0.16
    fig, axes = plt.subplots(2, 1, figsize=(15.5, 8.5), sharex=True)
    for ax, field, ylabel in (
        (axes[0], "total_irrigation_mm", "Total irrigation (mm)"),
        (axes[1], "total_n_kg_ha", "Total nitrogen (kg N/ha)"),
    ):
        for idx, (scenario, label, color) in enumerate(zip(order, scenario_labels, COLORS)):
            rows = metrics.loc[metrics.scenario.eq(scenario)].set_index("year").reindex(YEARS)
            values = to_number(rows[field]).to_numpy(dtype=float)
            valid = np.isfinite(values)
            ax.bar(x[valid] + (idx - 2) * width, values[valid], width,
                   color=color, label=label)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.24)
        ax.legend(ncol=3, fontsize=8, frameon=False)
    axes[-1].set_xticks(x, YEARS)
    axes[-1].set_xlabel("Validation year")
    fig.suptitle(f"{site} | five-scenario water and nitrogen management | seed {seed}",
                 x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(figures_dir / f"{site.lower()}_seed{seed}_five_scenario_management.png", dpi=140)
    plt.close(fig)


def plot_management_year(site: str, seed: int, year: int, metrics: pd.DataFrame, figures_dir: Path) -> None:
    """One year, five scenarios, with irrigation and N totals shown explicitly."""
    rows = metrics.loc[metrics.year.eq(year)].set_index("scenario")
    order = scenario_order(seed)
    if set(rows.index) != set(order):
        raise ValueError(f"Missing annual management rows for {site} seed{seed} {year}")
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5))
    for ax, field, ylabel in (
        (axes[0], "total_irrigation_mm", "Irrigation (mm)"),
        (axes[1], "total_n_kg_ha", "Nitrogen (kg N/ha)"),
    ):
        values = [num(rows.loc[key, field], 0.0) for key in order]
        bars = ax.bar(range(5), values, color=COLORS, width=0.68)
        top = max(values) if values else 0
        ax.set_ylim(0, max(1, top * 1.18))
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(1, top * 0.015),
                    f"{value:g}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(5), ["Null", "Recorded", "Auto + N", "Expert", f"PPO s{seed}"], rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(f"{site} {year} | five-scenario water and nitrogen totals | historical-weather PPO seed {seed}",
                 x=0.02, ha="left", fontweight="bold")
    if site == "FQ" and year == 2018:
        zero_yield = to_number(rows.grain_yield_kg_ha).eq(0).all()
        note = ("Original 2018 WTH has documented Tmin sign anomalies; all five yields are zero."
                if zero_yield else "2018 uses the isolated Tmin-corrected observed WTH; original-input results are archived separately.")
        fig.text(0.02, 0.01, note, color="#9A3333" if zero_yield else "#333333", fontsize=8)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(figures_dir / f"{site.lower()}_seed{seed}_{year}_five_scenario_management_bars.png", dpi=140)
    plt.close(fig)


def plot_reward(site: str, seed: int, metrics: pd.DataFrame, figures_dir: Path) -> None:
    order = scenario_order(seed)
    scenario_labels = labels(seed)
    x = np.arange(len(YEARS))
    fig, ax = plt.subplots(figsize=(14, 5.3))
    for scenario, label, color, style in zip(order, scenario_labels, COLORS, LINE_STYLES):
        rows = metrics.loc[metrics.scenario.eq(scenario)].set_index("year").reindex(YEARS)
        values = to_number(rows.common_reward).to_numpy(dtype=float)
        ax.plot(x, values, color=color, ls=style, lw=2 if scenario == scenario_key(seed) else 1.4,
                marker="o", ms=4, label=label)
    ax.set_xticks(x, YEARS)
    ax.set_xlabel("Validation year")
    ax.set_ylabel("Common reward (055_03 formula)")
    ax.set_title("0.158 × yield − 1.1 × irrigation − 1.58 × N", loc="left", fontsize=9)
    ax.grid(axis="y", alpha=0.24)
    ax.legend(ncol=3, fontsize=8, frameon=False)
    fig.suptitle(f"{site} | common reward comparison | seed {seed}", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(figures_dir / f"{site.lower()}_seed{seed}_five_scenario_reward.png", dpi=140)
    plt.close(fig)


def plot_daily_year(site: str, seed: int, year: int, daily: pd.DataFrame,
                    figures_dir: Path, weather_match: bool) -> None:
    base = daily.loc[(daily.year == year) & daily.scenario.isin(BASELINE_SCENARIOS)].copy()
    ppo = daily.loc[(daily.year == year) & daily.scenario.eq(scenario_key(seed))].copy()
    if ppo.empty or base.empty:
        raise ValueError(f"Missing daily process rows for {site} seed {seed}, year {year}")
    fig, axes = plt.subplots(4, 2, figsize=(15.5, 12.5), sharex=False)
    base_null = base.loc[base.scenario.eq("null")].sort_values("doy")
    # Weather is shown against calendar DOY so the same observed .WTH day is
    # aligned even when a simulator reports a different DAP origin.
    axes[0, 0].bar(to_number(base_null.doy), to_number(base_null.rainfall_mm),
                   color="#80A9C7", alpha=0.85, width=1.0, label="Rain")
    temp = axes[0, 0].twinx()
    temp.plot(to_number(base_null.doy), to_number(base_null.tmax_c), color="#C23B32", lw=1.25, label="Tmax")
    temp.plot(to_number(base_null.doy), to_number(base_null.tmin_c), color="#666666", lw=1.1, ls="--", label="Tmin")
    axes[0, 0].set_title("Weather (calendar DOY; shared observed .WTH)")
    axes[0, 0].set_ylabel("Rain (mm)")
    axes[0, 0].set_xlabel("Day of year")
    temp.set_ylabel("Temperature (°C)")
    weather_note = "Baseline/PPO weather aligns within baseline rounding" if weather_match else "Weather trace mismatch; see audit JSON"
    axes[0, 0].text(0.02, 0.96, weather_note, transform=axes[0, 0].transAxes,
                    ha="left", va="top", fontsize=8, color="#444444")

    for scenario, label, color, style in zip(BASELINE_SCENARIOS, labels(seed), COLORS[:4], LINE_STYLES[:4]):
        sub = base.loc[base.scenario.eq(scenario)].sort_values("dap")
        axes[0, 1].plot(sub.dap, sub.soil_water_mm, color=color, ls=style, lw=1.35, label=label)
        axes[1, 0].plot(sub.dap, sub.water_stress, color=color, ls=style, lw=1.35, label=f"{label} WSPD")
        axes[1, 1].plot(sub.dap, sub.nitrogen_stress, color=color, ls=style, lw=1.35, label=f"{label} NSTD")
        axes[3, 0].plot(sub.dap, sub.grain_yield_kg_ha, color=color, ls=style, lw=1.4, label=f"{label} grain")
        axes[3, 0].plot(sub.dap, sub.biomass_kg_ha, color=color, ls=style, lw=0.8, alpha=0.38)
        axes[3, 1].plot(sub.dap, sub.common_cumulative_reward, color=color, ls=style, lw=1.35, label=label)
        for ax, field in ((axes[2, 0], "irrigation_executed_mm"),
                          (axes[2, 1], "nitrogen_executed_kg_ha")):
            event = sub.loc[to_number(sub[field]).gt(0)]
            if not event.empty:
                ax.stem(to_number(event.dap), to_number(event[field]), linefmt=color,
                        markerfmt="o", basefmt=" ", label=label)

    color = COLORS[-1]
    ppo = ppo.sort_values("dap")
    if to_number(ppo.soil_water_mm).notna().any():
        axes[0, 1].plot(ppo.dap, to_number(ppo.soil_water_mm), color=color, lw=1.6,
                        label=scenario_label(seed))
    else:
        axes[0, 1].text(0.5, 0.5, "PPO daily soil-water state\nnot recorded in 8-seed validation log",
                        transform=axes[0, 1].transAxes, ha="center", va="center", fontsize=9, color="#555555")
    ppo_water_source = str(ppo.water_stress_source.iloc[0])
    ppo_n_source = str(ppo.nitrogen_stress_source.iloc[0])
    axes[1, 0].plot(ppo.dap, ppo.water_stress, color=color, lw=1.5, label=f"{scenario_label(seed)} {ppo_water_source}")
    axes[1, 1].plot(ppo.dap, ppo.nitrogen_stress, color=color, lw=1.5, label=f"{scenario_label(seed)} {ppo_n_source}")
    axes[3, 0].plot(ppo.dap, ppo.grain_yield_kg_ha, color=color, lw=1.7, label=f"{scenario_label(seed)} grain")
    axes[3, 0].plot(ppo.dap, ppo.biomass_kg_ha, color=color, lw=0.9, alpha=0.42,
                    label=f"{scenario_label(seed)} biomass")
    for ax, field in ((axes[2, 0], "irrigation_executed_mm"),
                      (axes[2, 1], "nitrogen_executed_kg_ha")):
        event = ppo.loc[to_number(ppo[field]).gt(0)]
        if not event.empty:
            ax.stem(to_number(event.dap), to_number(event[field]), linefmt=color,
                    markerfmt="o", basefmt=" ", label=scenario_label(seed))
    axes[3, 1].plot(ppo.dap, ppo.common_cumulative_reward, color=color, lw=1.55,
                    label=scenario_label(seed))

    panel_titles = (
        (axes[0, 1], "Soil water (five scenarios)" if to_number(ppo.soil_water_mm).notna().any() else "Soil water (baselines; PPO state unavailable)"),
        (axes[1, 0], f"Water status (baseline WSPD; PPO {ppo_water_source})"),
        (axes[1, 1], f"Nitrogen stress (baseline NSTD; PPO {ppo_n_source})"),
        (axes[2, 0], "Irrigation events"),
        (axes[2, 1], "Nitrogen application events"),
        (axes[3, 0], "Grain and biomass trajectories"),
        (axes[3, 1], "Common cumulative reward"),
    )
    for ax, title in panel_titles:
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[1, 0].set_ylabel("WSPD" if ppo_water_source == "WSPD" else "WSPD / SWFAC")
    axes[1, 1].set_ylabel("NSTD" if ppo_n_source == "NSTD" else "NSTD / NSTRES")
    axes[2, 0].set_ylabel("mm/event")
    axes[2, 1].set_ylabel("kg N/ha/event")
    axes[3, 0].set_ylabel("kg/ha")
    axes[3, 1].set_ylabel("common units")
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0], axes[2, 1], axes[3, 1]):
        ax.legend(fontsize=6.8, ncol=2, frameon=True)
    axes[3, 0].legend(fontsize=6.5, ncol=2, frameon=True)
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0], axes[2, 1], axes[3, 0], axes[3, 1]):
        ax.set_xlabel("DAP")
    title = f"{('HLA' if site == 'HL' else 'FQA')}{year} five-scenario daily process | historical-weather PPO seed {seed}"
    if site == "FQ" and year == 2018:
        title += (" | original WTH Tmin anomaly; zero yield across five scenarios"
                  if to_number(ppo.grain_yield_kg_ha).iloc[-1] == 0 else " | isolated Tmin-corrected observed WTH")
    fig.suptitle(title,
                 x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(figures_dir / f"{site.lower()}_seed{seed}_{year}_five_scenario_daily.png", dpi=115)
    plt.close(fig)


def write_seed_readme(site: str, seed: int, metric_frame: pd.DataFrame,
                      sequence_audit: dict[str, Any], seed_dir: Path) -> None:
    ppo_rows = metric_frame.loc[metric_frame.scenario.eq(scenario_key(seed))]
    mean_yield = to_number(ppo_rows.grain_yield_kg_ha).mean()
    mean_i = to_number(ppo_rows.total_irrigation_mm).mean()
    mean_n = to_number(ppo_rows.total_n_kg_ha).mean()
    fig_names = sorted(p.name for p in (seed_dir / "figures").glob("*.png"))
    text = [
        f"# {site} historical-weather PPO seed {seed}: five-scenario results",
        "",
        "Compared with the frozen Null, recorded farmer template, DSSAT auto + external N, and official extension expert baselines.",
        "",
        f"- Validation years: 2014–2023 (10 years).",
        f"- PPO mean yield / irrigation / N: {mean_yield:.1f} kg/ha / {mean_i:.1f} mm / {mean_n:.1f} kg N/ha.",
        f"- Exact same positive-action sequence across all ten years: {'yes' if sequence_audit['identical_across_all_validation_years'] else 'no'} ({sequence_audit['unique_yearly_action_sequence_count']} unique annual sequences).",
        "- WP_ET / WUE: unavailable for PPO because the selected PPO validation artifacts do not contain an exact Summary.OUT + ETCP replay.",
        "- NUE: unavailable because the selected PPO validation artifacts do not contain plant N uptake. PFP_N is reported separately and must not be relabeled as NUE.",
        "- Daily process figures show baseline WSPD/NSTD alongside PPO SWFAC/NSTRES; these are the source-specific stress indicators. PPO daily soil water was not recorded.",
        "- Weather was cross-checked against the matching site/year Null baseline by calendar date (rainfall exact; temperature tolerance 0.051°C for one-decimal baseline rounding); see the action audit JSON.",
        "",
        "## Tables",
        "",
        f"- `tables/{site.lower()}_seed{seed}_yearly_yield_irrigation_n_table.csv`: each cell is yield / irrigation / N.",
        f"- `tables/{site.lower()}_seed{seed}_yearly_wp_et_pfp_n_table.csv`: each cell is WP_ET / PFP_N; missing exact metrics are `—`.",
        f"- `tables/{site.lower()}_seed{seed}_yearly_five_scenario_metrics.csv`: tidy numeric five-scenario metrics.",
        f"- `tables/{site.lower()}_seed{seed}_five_scenario_management_events.csv`: all positive baseline and PPO water/N events.",
        f"- `tables/{site.lower()}_seed{seed}_ppo_daily_action_trace.csv`: full daily PPO action trace, including no-ops.",
        "",
        "## Figures",
        "",
        *[f"- `figures/{name}`" for name in fig_names],
        "",
    ]
    (seed_dir / "README.md").write_text("\n".join(text), encoding="utf-8")


def run() -> None:
    if OUT_ROOT.exists() and any(OUT_ROOT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {OUT_ROOT}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    index_lines = [
        "# HL/FQ 8-seed historical-weather PPO: YC 055_03-style five-scenario output",
        "",
        "This is a read-only post-processing of completed 100K checkpoints and existing four-baseline results. No training, DSSAT replay, WGEN weather generation, or configuration edit was performed.",
        "",
        "Each completed seed has `figures/` and `tables/` under `HL|FQ/best_seed_seedN/`, matching the YC reference folder structure.",
        "",
        "The yearly metric tables contain yield / irrigation / N and WP_ET / PFP_N. PPO WP_ET and NUE remain blank/marked unavailable because exact ETCP and plant N uptake results are absent from the selected validation artifacts. PFP_N is not NUE.",
        "",
        "## Completed seed outputs",
        "",
    ]
    counts: dict[str, int] = {}
    for site in ("HL", "FQ"):
        base_summary, base_daily = load_baseline(site)
        for seed in SEEDS_BY_SITE[site]:
            seed_dir = OUT_ROOT / site / f"best_seed_seed{seed}"
            figures_dir = seed_dir / "figures"
            tables_dir = seed_dir / "tables"
            figures_dir.mkdir(parents=True, exist_ok=True)
            metrics, daily, ppo_actions, sequence_audit = load_ppo_seed(
                site, seed, base_summary, base_daily
            )
            # Verify exactly ten annual rows per scenario before export.
            expected_keys = {(year, scenario) for year in YEARS for scenario in scenario_order(seed)}
            actual_keys = set(zip(metrics.year.astype(int), metrics.scenario.astype(str)))
            if len(metrics) != 50 or actual_keys != expected_keys:
                raise ValueError(f"Five-scenario metric table is incomplete: {site} seed {seed}")
            write_table_files(site, seed, metrics, tables_dir, sequence_audit)
            write_event_and_daily_tables(site, seed, daily, ppo_actions, tables_dir)
            plot_metric_bars(site, seed, metrics, figures_dir)
            plot_management_bars(site, seed, metrics, figures_dir)
            plot_reward(site, seed, metrics, figures_dir)
            for year in YEARS:
                weather_match = bool(sequence_audit["weather_input_match_by_calendar_date"][str(year)]["weather_values_match_by_calendar_date"])
                plot_daily_year(site, seed, year, daily, figures_dir, weather_match)
            write_seed_readme(site, seed, metrics, sequence_audit, seed_dir)
            counts[f"{site}_seed{seed}"] = len(list(figures_dir.glob("*.png")))
            index_lines.append(
                f"- [{site} seed {seed}]({site}/best_seed_seed{seed}/README.md) — {counts[f'{site}_seed{seed}']} PNG figures"
            )
    index_lines += [
        "",
        "## Coverage",
        "",
        "- HL: PPO seeds 0–7 (8 seeds).",
        "- FQ: PPO seeds 0–6 (7 completed seeds). Seed 7 remains excluded because it did not pass the existing formal smoke gate.",
        "- The per-seed action audit reports whether its ten annual positive-action sequences are exactly identical.",
        "",
    ]
    (OUT_ROOT / "README.md").write_text("\n".join(index_lines), encoding="utf-8")
    print(json.dumps({"output_root": OUT_ROOT.relative_to(ROOT).as_posix(), "seed_figure_counts": counts,
                      "total_png": sum(counts.values()), "sites": {k: len(v) for k, v in SEEDS_BY_SITE.items()}},
                     ensure_ascii=False, indent=2))


def refresh_metric_figures() -> None:
    """Replot only the aggregate metric panels from already-exported CSVs."""
    for site, seeds in SEEDS_BY_SITE.items():
        for seed in seeds:
            seed_dir = OUT_ROOT / site / f"best_seed_seed{seed}"
            table_path = seed_dir / "tables" / f"{site.lower()}_seed{seed}_yearly_five_scenario_metrics.csv"
            metrics = read_csv(table_path)
            metrics["year"] = to_number(metrics["year"]).astype(int)
            figures_dir = seed_dir / "figures"
            if not (figures_dir / f"{site.lower()}_seed{seed}_five_scenario_metrics.png").is_file():
                raise FileNotFoundError(f"Missing existing metric figure under {figures_dir}")
            plot_metric_bars(site, seed, metrics, figures_dir)
    print("Refreshed the 15 aggregate five-scenario metric figures from saved CSVs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replot-metrics", action="store_true",
                        help="Regenerate only existing aggregate metric figures from saved metrics CSVs")
    arguments = parser.parse_args()
    if arguments.replot_metrics:
        refresh_metric_figures()
    else:
        run()
