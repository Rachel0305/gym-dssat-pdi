"""Build current five-site PPT figures with Python and frozen later outputs.

This package deliberately does not call the old figure entry points.  It reuses
their visual grammar (five-scenario colours, grouped yearly bars and 4x2 daily
process panels), while assembling the tables from the later frozen baseline,
no-forecast and E2/weather replay outputs.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from build_sya_configured_five_scenario_daily_plots_046_06 import add_unified_reward, daily_from_snapshot
from run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 import num, parse_summary_out


TASK = "190_multisite_ppt_figures_current"
OUT = ROOT / "benchmark_results" / TASK
YEARS = list(range(2014, 2024))
SCENARIOS = ["null", "recorded_farmer_template", "dssat_auto", "official_extension_expert", "rl_candidate"]
LABELS = {
    "null": "Null",
    "recorded_farmer_template": "Recorded template",
    "dssat_auto": "DSSAT auto irrigation",
    "official_extension_expert": "Official expert",
    "rl_candidate": "PPO",
}
COLORS = {
    "null": "#444444",
    "recorded_farmer_template": "#C44E52",
    "dssat_auto": "#D8A305",
    "official_extension_expert": "#7861B2",
    "rl_candidate": "#238B45",
}
STYLES = {
    "null": "-",
    "recorded_farmer_template": "--",
    "dssat_auto": "-.",
    "official_extension_expert": ":",
    "rl_candidate": "-",
}
METRICS = [
    ("grain_yield_kg_ha", "Grain yield (kg/ha)"),
    ("WP_ET_kg_m3", "WP_ET (kg/m³)"),
    ("PFP_N_kg_kg", "PFP-N (kg/kg)"),
]
RESOURCES = [
    ("actual_irrigation_mm", "Total irrigation (mm)"),
    ("actual_nitrogen_kg_ha", "Total nitrogen (kg/ha)"),
]


SITES: dict[str, dict[str, Any]] = {
    "SY": {
        "station": "SYA", "site_label": "SY",
        "baseline_csv": "benchmark_results/046_03_sya_originIC_four_baselines/evaluation/046_03_baseline_summary.csv",
        "baseline_snapshots": "benchmark_results/046_03_sya_originIC_four_baselines/snapshots/SYA",
        "nof_eval": "benchmark_results/046_10_sya_originIC_expanded_action_maskableppo/evaluation/046_10_checkpoint_validation_summary.csv",
        "nof_ckpt": 100000,
        "nof_snapshots": "benchmark_results/046_06_sya_originIC_046_10_sya_originIC_expanded_action_maskableppo_five_scenario_daily_ckpt100000/snapshots/SYA",
        "weather_replay_csv": "benchmark_results/151E2N_wp_et_5k_replay/151_wp_et_replay_by_policy_year.csv",
        "weather_policy": "e2_s0",
        "weather_snapshots": "benchmark_results/151E2N_wp_et_5k_replay/replays/e2_s0/snapshots/SYA",
        "weather_label": "E2 forecast PPO (5K, seed0)",
    },
    "LC": {
        "station": "LCA", "site_label": "LC",
        "baseline_csv": "benchmark_results/053_02_lca_lowIC_four_baselines_static_level1/evaluation/053_02_baseline_summary.csv",
        "baseline_snapshots": "benchmark_results/053_02_lca_lowIC_four_baselines_static_level1/snapshots/LCA",
        "nof_eval": "benchmark_results/053_00_lca_lowIC_expanded_action_maskableppo/evaluation/053_00_checkpoint_validation_summary.csv",
        "nof_ckpt": 100000,
        "nof_snapshots": "benchmark_results/053_03_lca_lowIC_053_00_lca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/snapshots/LCA",
        "weather_replay_csv": "benchmark_results/157LC_lca_lowIC_dual_branch_weather_reader_wp_et_5k_replay/157LC_wp_et_replay_by_policy_year.csv",
        "weather_policy": "e2_s0",
        "weather_snapshots": "benchmark_results/157LC_lca_lowIC_dual_branch_weather_reader_wp_et_5k_replay/replays/e2_s0/snapshots/LCA",
        "weather_label": "E2 forecast PPO (5K, seed0)",
    },
    "HL": {
        "station": "HLA", "site_label": "HL",
        "baseline_csv": "benchmark_results/054_02_hla_lowIC_four_baselines_static_level1/evaluation/054_02_baseline_summary.csv",
        "baseline_snapshots": "benchmark_results/054_02_hla_lowIC_four_baselines_static_level1/snapshots/HLA",
        "nof_eval": "benchmark_results/054_00_hla_lowIC_expanded_action_maskableppo/evaluation/054_00_checkpoint_validation_summary.csv",
        "nof_ckpt": 100000,
        "nof_snapshots": "benchmark_results/054_03_hla_lowIC_054_00_hla_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/snapshots/HLA",
        "weather_replay_csv": "benchmark_results/190_weather_wp_et_missing_replay/178_wp_et_replay_by_policy_year.csv",
        "weather_policy": "hla_forecast_5k",
        "weather_snapshots": "benchmark_results/190_weather_wp_et_missing_replay/replays/hla_forecast_5k/snapshots/HLA",
        "weather_label": "E2 forecast PPO (5K, seed1)",
    },
    "YC": {
        "station": "YCA", "site_label": "YC",
        "baseline_csv": "benchmark_results/055_02_yca_lowIC_four_baselines_static_level1/evaluation/055_02_baseline_summary.csv",
        "baseline_snapshots": "benchmark_results/055_02_yca_lowIC_four_baselines_static_level1/snapshots/YCA",
        "nof_eval": "benchmark_results/055_00_yca_lowIC_expanded_action_maskableppo/evaluation/055_00_checkpoint_validation_summary.csv",
        "nof_ckpt": 100000,
        "nof_snapshots": "benchmark_results/055_03_yca_lowIC_055_00_yca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/snapshots/YCA",
        "weather_replay_csv": "benchmark_results/190_weather_wp_et_missing_replay/178_wp_et_replay_by_policy_year.csv",
        "weather_policy": "yca_forecast_5k",
        "weather_snapshots": "benchmark_results/190_weather_wp_et_missing_replay/replays/yca_forecast_5k/snapshots/YCA",
        "weather_label": "E2 forecast PPO (5K, seed1; collapse retained)",
    },
    "FQ": {
        "station": "FQA", "site_label": "FQ",
        "baseline_csv": "benchmark_results/051_02_fqa_originIC_four_baselines_static_level1/evaluation/051_02_baseline_summary.csv",
        "baseline_snapshots": "benchmark_results/051_02_fqa_originIC_four_baselines_static_level1/snapshots/FQA",
        "nof_eval": "benchmark_results/051_00_fqa_originIC_expanded_action_maskableppo/evaluation/051_00_checkpoint_validation_summary.csv",
        "nof_ckpt": 100000,
        "nof_snapshots": "benchmark_results/051_03_fqa_originIC_051_00_fqa_originIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/snapshots/FQA",
        "weather_replay_csv": "benchmark_results/178_multisite_wp_et_replay/178_wp_et_replay_by_policy_year.csv",
        "weather_policy": "fqa_forecast_5k",
        "weather_snapshots": "benchmark_results/178_multisite_wp_et_replay/replays/fqa_forecast_5k/snapshots/FQA",
        "weather_label": "E2 forecast PPO (5K, seed1)",
        # FQ2018 has an isolated corrected-weather paper record.  Use it only
        # for the no-forecast 100K package; the E2 forecast replay remains on
        # the original input until its own corrected replay is completed.
        "corrected_2018": {
            "baseline_csv": "benchmark_results/088_00_fq2018_input_quality_correction_run_corrected2018/evaluation/051_02_baseline_summary.csv",
            "baseline_snapshots": "benchmark_results/088_00_fq2018_input_quality_correction_run_corrected2018/snapshots/FQA/2018",
            "ppo_snapshot": "benchmark_results/088_00_fq2018_input_quality_correction/ppo_replay_corrected2018_rerun1/snapshots/FQA/2018/ckpt100000",
            "replacement_csv": "benchmark_results/089_00_fq2018_corrected_ten_year_paper_summary/tables/089_00_fqa_native_auto_corrected_2018_replacement.csv",
        },
    },
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def scalar(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def summary_metrics(snapshot: Path, final_yield: float, total_i: float, total_n: float) -> dict[str, float]:
    rows = parse_summary_out(snapshot / "Summary.OUT")
    candidates: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        hwam = num(row, "HWAM")
        if hwam is None:
            continue
        score = abs(float(hwam) - final_yield)
        score += abs(float(num(row, "IRCM") or 0.0) - total_i)
        score += abs(float(num(row, "NICM") or 0.0) - total_n)
        candidates.append((score, row))
    if not candidates:
        raise RuntimeError(f"No matching Summary.OUT row: {snapshot}")
    _, row = min(candidates, key=lambda item: item[0])
    etcp = scalar(num(row, "ETCP"))
    ypem = scalar(num(row, "YPEM"))
    ypnam = scalar(num(row, "YPNAM"))
    wp = ypem * 0.1 if np.isfinite(ypem) and ypem >= 0 else final_yield / (etcp * 10.0) if etcp > 0 else np.nan
    pfp = ypnam if total_n > 0 and np.isfinite(ypnam) and ypnam >= 0 else np.nan
    return {"etcp_mm": etcp, "WP_ET_kg_m3": wp, "PFP_N_kg_kg": pfp}


def corrected_2018_metrics(cfg: dict[str, Any], scenario: str) -> dict[str, float]:
    """Read the later isolated FQ2018 replacement record, not old 051 values."""
    correction = cfg.get("corrected_2018")
    if not correction:
        return {}
    path = ROOT / correction["replacement_csv"]
    df = pd.read_csv(path, keep_default_na=False)
    source_scenario = "ppo" if scenario == "rl_candidate" else scenario
    rows = df[df["year"].astype(str).eq("2018") & df["scenario"].eq(source_scenario)]
    if rows.empty:
        raise RuntimeError(f"Missing corrected FQ2018 record for {scenario}: {path}")
    out: dict[str, float] = {}
    field_map = {
        "grain_yield_kg_ha": "grain_yield_kg_ha",
        "WP_ET_kg_m3": "WP_ET_kg_m3",
        "PFP_N_kg_kg": "PFP_N_kg_kg",
        "actual_irrigation_mm": "actual_irrigation_mm",
        "actual_nitrogen_kg_ha": "actual_nitrogen_kg_ha",
        "etcp_mm": "etcp_mm",
    }
    for field, metric in field_map.items():
        row = rows[rows["metric"].eq(metric)]
        out[field] = scalar(row["corrected_value"].iloc[0]) if not row.empty else float("nan")
    return out


def action_sequence_from_snapshot(snapshot: Path, year: int, scenario: str) -> str:
    daily = daily_from_snapshot(snapshot, year, scenario)
    daily["irrigation_executed_mm"] = pd.to_numeric(daily["irrigation_executed_mm"], errors="coerce").fillna(0.0)
    daily["nitrogen_executed_kg_ha"] = pd.to_numeric(daily["nitrogen_executed_kg_ha"], errors="coerce").fillna(0.0)
    events = daily[daily["irrigation_executed_mm"].gt(0) | daily["nitrogen_executed_kg_ha"].gt(0)].sort_values("dap")
    return "; ".join(
        f"DAP{int(row.dap) + 1} I{row.irrigation_executed_mm:g}/N{row.nitrogen_executed_kg_ha:g}"
        for row in events.itertuples()
    )


def load_baseline(cfg: dict[str, Any], mode: str) -> pd.DataFrame:
    path = ROOT / cfg["baseline_csv"]
    df = pd.read_csv(path, keep_default_na=False)
    df["scenario"] = df["scenario"].replace({"": "null"}).fillna("null")
    df = df[df["scenario"].isin(SCENARIOS[:-1])].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    for col in ["grain_yield_kg_ha", "actual_irrigation_mm", "actual_nitrogen_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["source"] = cfg["baseline_csv"]
    correction = cfg.get("corrected_2018") if mode == "no_forecast" else None
    if correction:
        corrected = pd.read_csv(ROOT / correction["baseline_csv"], keep_default_na=False)
        corrected["year"] = pd.to_numeric(corrected["year"], errors="coerce").astype(int)
        corrected = corrected[corrected["year"].eq(2018) & corrected["scenario"].isin(SCENARIOS[:-1])].copy()
        for col in ["grain_yield_kg_ha", "actual_irrigation_mm", "actual_nitrogen_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg"]:
            corrected[col] = pd.to_numeric(corrected[col], errors="coerce")
        corrected["source"] = correction["baseline_csv"]
        df = pd.concat([df[~(df["year"].eq(2018) & df["scenario"].isin(SCENARIOS[:-1]))], corrected], ignore_index=True)
    return df[["year", "scenario", "grain_yield_kg_ha", "actual_irrigation_mm", "actual_nitrogen_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg", "source"]]


def load_noforecast(cfg: dict[str, Any]) -> pd.DataFrame:
    path = ROOT / cfg["nof_eval"]
    df = pd.read_csv(path, keep_default_na=False)
    df["checkpoint_step"] = pd.to_numeric(df["checkpoint_step"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df = df[df["checkpoint_step"].eq(cfg["nof_ckpt"]) & df["year"].isin(YEARS)].copy()
    if len(df) != len(YEARS):
        raise RuntimeError(f"No-forecast rows incomplete for {cfg['site_label']}: {len(df)}")
    rows: list[dict[str, Any]] = []
    snap_root = ROOT / cfg["nof_snapshots"]
    for _, row in df.sort_values("year").iterrows():
        year = int(row["year"])
        snapshot = snap_root / str(year) / "rl_candidate"
        final_y = scalar(row["final_grnwt"])
        total_i = scalar(row["total_irrigation"])
        total_n = scalar(row["total_n"])
        source = cfg["nof_eval"]
        if year == 2018 and cfg.get("corrected_2018"):
            corrected = cfg["corrected_2018"]
            snapshot = ROOT / corrected["ppo_snapshot"]
            replacement = corrected_2018_metrics(cfg, "rl_candidate")
            final_y = replacement["grain_yield_kg_ha"]
            total_i = replacement["actual_irrigation_mm"]
            total_n = replacement["actual_nitrogen_kg_ha"]
            metric_values = {
                "etcp_mm": replacement["etcp_mm"],
                "WP_ET_kg_m3": replacement["WP_ET_kg_m3"],
                "PFP_N_kg_kg": replacement["PFP_N_kg_kg"],
            }
            action_sequence = action_sequence_from_snapshot(snapshot, year, "rl_candidate")
            source = corrected["replacement_csv"]
        else:
            metric_values = summary_metrics(snapshot, final_y, total_i, total_n)
            action_sequence = str(row.get("action_sequence", ""))
        rows.append({
            "year": year, "scenario": "rl_candidate",
            "grain_yield_kg_ha": final_y, "actual_irrigation_mm": total_i,
            "actual_nitrogen_kg_ha": total_n,
            **metric_values,
            "action_sequence": action_sequence,
            "source": source,
        })
    return pd.DataFrame(rows)


def load_weather(cfg: dict[str, Any]) -> pd.DataFrame:
    path = ROOT / cfg["weather_replay_csv"]
    df = pd.read_csv(path, keep_default_na=False)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df = df[df["policy"].astype(str).eq(cfg["weather_policy"]) & df["year"].isin(YEARS)].copy()
    if len(df) != len(YEARS):
        raise RuntimeError(f"Weather rows incomplete for {cfg['site_label']}: {len(df)} from {path}")
    return pd.DataFrame({
        "year": df["year"].astype(int), "scenario": "rl_candidate",
        "grain_yield_kg_ha": pd.to_numeric(df["final_grnwt"], errors="coerce"),
        "actual_irrigation_mm": pd.to_numeric(df["total_irrigation"], errors="coerce"),
        "actual_nitrogen_kg_ha": pd.to_numeric(df["total_n"], errors="coerce"),
        "WP_ET_kg_m3": pd.to_numeric(df["WP_ET_kg_m3"], errors="coerce"),
        "PFP_N_kg_kg": pd.to_numeric(df["PFP_N_kg_kg"], errors="coerce"),
        "action_sequence": df.get("action_sequence", "").astype(str),
        "source": cfg["weather_replay_csv"],
    })


def assemble(cfg: dict[str, Any], mode: str) -> pd.DataFrame:
    baseline = load_baseline(cfg, mode)
    ppo = load_noforecast(cfg) if mode == "no_forecast" else load_weather(cfg)
    result = pd.concat([baseline, ppo], ignore_index=True, sort=False)
    result["site"] = cfg["site_label"]
    result["station"] = cfg["station"]
    result["mode"] = mode
    expected = len(YEARS) * len(SCENARIOS)
    if len(result) != expected or result.duplicated(["year", "scenario"]).any():
        raise RuntimeError(f"Five-scenario row mismatch for {cfg['site_label']} {mode}: {len(result)}")
    return result.sort_values(["year", "scenario"]).reset_index(drop=True)


def snapshot_map(cfg: dict[str, Any], mode: str, year: int) -> dict[str, Path]:
    base = ROOT / cfg["baseline_snapshots"]
    base_year = base / str(year)
    if mode == "no_forecast":
        ppo = ROOT / cfg["nof_snapshots"] / str(year) / "rl_candidate"
        if year == 2018 and cfg.get("corrected_2018"):
            base_year = ROOT / cfg["corrected_2018"]["baseline_snapshots"]
            ppo = ROOT / cfg["corrected_2018"]["ppo_snapshot"]
    else:
        ppo = ROOT / cfg["weather_snapshots"] / str(year) / "ckpt5000"
    return {
        "null": base_year / "null",
        "recorded_farmer_template": base_year / "recorded_farmer_template",
        "dssat_auto": base_year / "dssat_auto",
        "official_extension_expert": base_year / "official_extension_expert",
        "rl_candidate": ppo,
    }


def plot_daily(daily: pd.DataFrame, site: str, year: int, mode: str, path: Path, ppo_label: str) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(16, 13), sharex=True)
    weather = daily[daily["scenario"].eq("null")].sort_values("dap")
    axes[0, 0].bar(weather["dap"], weather["rainfall_mm"], color="#80A9C7", alpha=0.85, label="Rain")
    temp = axes[0, 0].twinx()
    temp.plot(weather["dap"], weather["tmax_c"], color="#C23B32", lw=1.25, label="Tmax")
    temp.plot(weather["dap"], weather["tmin_c"], color="#666666", lw=1.1, ls="--", label="Tmin")
    axes[0, 0].set_title("Weather"); axes[0, 0].set_ylabel("Rain (mm)"); temp.set_ylabel("Temperature (°C)")
    for scenario in SCENARIOS:
        sub = daily[daily["scenario"].eq(scenario)].sort_values("dap")
        color, style = COLORS[scenario], STYLES[scenario]
        label = ppo_label if scenario == "rl_candidate" else LABELS[scenario]
        axes[0, 1].plot(sub["dap"], sub["soil_water_mm"], color=color, ls=style, lw=1.35, label=label)
        axes[1, 0].plot(sub["dap"], sub["water_stress_index_wspd"], color=color, ls=style, lw=1.35, label=label)
        axes[1, 1].plot(sub["dap"], sub["nitrogen_stress_index_nstd"], color=color, ls=style, lw=1.35, label=label)
        positive_i = sub[sub["irrigation_executed_mm"].gt(0)]
        positive_n = sub[sub["nitrogen_executed_kg_ha"].gt(0)]
        if not positive_i.empty:
            axes[2, 0].vlines(positive_i["dap"], 0, positive_i["irrigation_executed_mm"], color=color, lw=1.35, label=label)
            axes[2, 0].scatter(positive_i["dap"], positive_i["irrigation_executed_mm"], color=color, s=18)
        if not positive_n.empty:
            axes[2, 1].vlines(positive_n["dap"], 0, positive_n["nitrogen_executed_kg_ha"], color=color, lw=1.35, label=label)
            axes[2, 1].scatter(positive_n["dap"], positive_n["nitrogen_executed_kg_ha"], color=color, s=18)
        axes[3, 0].plot(sub["dap"], sub["grain_yield_kg_ha"], color=color, ls=style, lw=1.45, label=f"{label} grain")
        axes[3, 0].plot(sub["dap"], sub["biomass_kg_ha"], color=color, ls=style, lw=0.85, alpha=0.4)
        axes[3, 1].plot(sub["dap"], sub["unified_cumulative_reward"], color=color, ls=style, lw=1.35, label=label)
    for axis, title, ylabel in [
        (axes[0, 1], "Soil water", "SWTD (mm)"), (axes[1, 0], "Water stress index", "WSPD"),
        (axes[1, 1], "Nitrogen stress index", "NSTD"), (axes[2, 0], "Irrigation events", "mm/event"),
        (axes[2, 1], "Nitrogen application events", "kg/ha/event"), (axes[3, 0], "Grain and biomass trajectories", "kg/ha"),
        (axes[3, 1], "Unified cumulative reward", "common units"),
    ]:
        axis.set_title(title); axis.set_ylabel(ylabel); axis.grid(alpha=0.2)
    for axis in [axes[0, 1], axes[2, 0], axes[2, 1], axes[3, 1]]:
        axis.legend(fontsize=7, ncol=2)
    axes[3, 0].set_xlabel("DAP"); axes[3, 1].set_xlabel("DAP")
    mode_label = "no-forecast" if mode == "no_forecast" else "weather forecast"
    fig.suptitle(f"{site}{year} {mode_label} five-scenario daily process", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=220)
    plt.close(fig)


def draw_grouped(summary: pd.DataFrame, fields: list[tuple[str, str]], path: Path, title: str, ppo_label: str) -> None:
    x = np.arange(len(YEARS)); width = 0.15
    fig, axes = plt.subplots(len(fields), 1, figsize=(16, 4.2 * len(fields)), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, (field, ylabel) in zip(axes, fields):
        for idx, scenario in enumerate(SCENARIOS):
            subset = summary[summary["scenario"].eq(scenario)].set_index("year")
            values = [scalar(subset[field].get(year, np.nan)) for year in YEARS]
            label = ppo_label if scenario == "rl_candidate" else LABELS[scenario]
            axis.bar(x + (idx - 2) * width, values, width, label=label, color=COLORS[scenario])
        axis.set_ylabel(ylabel); axis.grid(axis="y", alpha=0.25); axis.legend(ncol=3, fontsize=8)
    axes[-1].set_xticks(x); axes[-1].set_xticklabels(YEARS); axes[-1].set_xlabel("Validation year")
    fig.suptitle(title, x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(path, dpi=220); fig.savefig(path.with_suffix(".svg")); plt.close(fig)


def draw_aggregate(summary: pd.DataFrame, fields: list[tuple[str, str]], path: Path, title: str, reducer: str, ppo_label: str) -> None:
    aggregate = summary.groupby("scenario", as_index=False)[[field for field, _ in fields]].agg(reducer)
    x = np.arange(len(SCENARIOS)); fig, axes = plt.subplots(len(fields), 1, figsize=(12, 4.0 * len(fields)))
    axes = np.atleast_1d(axes)
    for axis, (field, ylabel) in zip(axes, fields):
        values = [scalar(aggregate.loc[aggregate["scenario"].eq(scenario), field].iloc[0]) for scenario in SCENARIOS]
        axis.bar(x, values, color=[COLORS[s] for s in SCENARIOS], width=0.62)
        axis.set_ylabel(ylabel); axis.set_xticks(x); axis.set_xticklabels([ppo_label if s == "rl_candidate" else LABELS[s] for s in SCENARIOS], rotation=18, ha="right"); axis.grid(axis="y", alpha=0.25)
        for pos, value in zip(x, values):
            if np.isfinite(value):
                axis.text(pos, value, f"{value:.2f}" if "WP" in ylabel or "PFP" in ylabel else f"{value:.0f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle(title, x=0.02, ha="left", fontweight="bold"); fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(path, dpi=220); fig.savefig(path.with_suffix(".svg")); plt.close(fig)


def run_site(cfg: dict[str, Any], mode: str) -> dict[str, Any]:
    summary = assemble(cfg, mode)
    site = cfg["site_label"]
    ppo_label = "PPO" if mode == "no_forecast" else cfg["weather_label"]
    out = OUT / ("no_forecast" if mode == "no_forecast" else "weather_forecast") / site
    (out / "figures").mkdir(parents=True, exist_ok=True); (out / "tables").mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "tables" / f"{site}_{mode}_five_scenario_summary.csv", index=False, encoding="utf-8-sig")
    daily_frames: list[pd.DataFrame] = []
    for year in YEARS:
        mapping = snapshot_map(cfg, mode, year)
        missing = [rel(path) for path in mapping.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{site} {mode} {year} missing snapshot(s): {missing}")
        daily = pd.concat([daily_from_snapshot(path, year, scenario) for scenario, path in mapping.items()], ignore_index=True)
        # The reused daily renderer carries SYA/SY as its historical defaults;
        # normalize the metadata here so exported tables remain site-correct.
        daily["station_code"] = cfg["station"]
        daily["site"] = site
        daily = add_unified_reward(daily)
        daily_frames.append(daily)
        daily.to_csv(out / "tables" / f"{site}_{year}_{mode}_five_scenario_daily.csv", index=False, encoding="utf-8-sig")
        plot_daily(daily, site, year, mode, out / "figures" / f"{site}{year}_{mode}_five_scenario_daily.png", ppo_label)
    pd.concat(daily_frames, ignore_index=True).to_csv(out / "tables" / f"{site}_{mode}_all_years_daily.csv", index=False, encoding="utf-8-sig")
    draw_grouped(summary, METRICS, out / "figures" / f"{site}_{mode}_yearly_metrics.png", f"{site} {mode}: yearly yield and water/nitrogen efficiency", ppo_label)
    draw_grouped(summary, RESOURCES, out / "figures" / f"{site}_{mode}_yearly_management.png", f"{site} {mode}: yearly total irrigation and nitrogen", ppo_label)
    draw_aggregate(summary, RESOURCES, out / "figures" / f"{site}_{mode}_total_irrigation_nitrogen_bar.png", f"{site} {mode}: total resource use, 2014–2023", "sum", ppo_label)
    draw_aggregate(summary, METRICS, out / "figures" / f"{site}_{mode}_mean_yield_efficiency_bar.png", f"{site} {mode}: mean yield and efficiency, 2014–2023", "mean", ppo_label)
    return {"site": site, "mode": mode, "rows": int(len(summary)), "daily_figures": len(YEARS), "output": rel(out), "summary": rel(out / "tables" / f"{site}_{mode}_five_scenario_summary.csv")}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifests: list[dict[str, Any]] = []
    for site, cfg in SITES.items():
        for mode in ["no_forecast", "weather_forecast"]:
            manifests.append(run_site(cfg, mode))
    manifest = {
        "task": TASK,
        "python_backend": "matplotlib",
        "years": YEARS,
        "scenarios": SCENARIOS,
        "sites": list(SITES),
        "results": manifests,
        "data_policy": "later frozen outputs only; old figures are style references, not numeric inputs",
        "weather_scope": "SY/LC seed0 E2 5K; HL/YC/FQ seed1 E2 5K; YC 5K collapse retained",
        "fq2018_correction": "FQ no-forecast uses isolated 088/089 corrected-weather replacement; FQ E2 weather remains original 178 replay pending corrected-E2 replay",
    }
    (OUT / "190_multisite_ppt_figures_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
