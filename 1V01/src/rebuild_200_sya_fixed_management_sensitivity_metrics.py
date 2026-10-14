"""Rebuild task 200 metrics from completed snapshots.

Task 200 replayed all 42 cases successfully, but its first summary parser used
whitespace splitting on fixed-width Summary.OUT.  This script reuses those
snapshots and daily files, applies the project's fixed-width parser, and does
not rerun DSSAT.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import run_200_sya_fixed_management_sensitivity as task200
from calculate_five_site_wue_nue_from_summary_019_10 import num as summary_num, parse_summary_out


INPUT = ROOT / "benchmark_results" / "200_sya_fixed_management_sensitivity_retry1"
OUT = ROOT / "benchmark_results" / "200_sya_fixed_management_sensitivity_retry2"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def value(row: dict[str, Any], key: str) -> float:
    raw = summary_num(row, key)
    return task200.valid_num(raw)


def summarize(variant: str, management_id: str, daily: pd.DataFrame, runtime: pd.DataFrame, snapshot: Path, out: Path) -> dict[str, Any]:
    growth = daily[daily["DAP"].notna()].copy()
    grain_series = pd.to_numeric(growth.get("GWAD", pd.Series(dtype=float)), errors="coerce").dropna()
    daily_grain = float(grain_series.iloc[-1]) if not grain_series.empty else np.nan
    requested_i = float(runtime["irrigation_requested_mm"].sum())
    requested_n = float(runtime["nitrogen_requested_kg_ha"].sum())
    rows = parse_summary_out(snapshot / "Summary.OUT")
    summary_row = task200.select_summary_row(rows, daily_grain, requested_i, requested_n)
    grain = value(summary_row, "HWAM")
    if not np.isfinite(grain):
        grain = daily_grain
    irrigation = value(summary_row, "IRCM")
    nitrogen = value(summary_row, "NICM")
    etcp = value(summary_row, "ETCP")
    precip = value(summary_row, "PRCP")
    n_uptake = value(summary_row, "NUCM")
    n_leaching = value(summary_row, "NLCM")
    runoff = value(summary_row, "ROCM")
    wp_et = grain / etcp / 10.0 if np.isfinite(grain) and np.isfinite(etcp) and etcp > 0 else np.nan
    gross_wp = grain / (precip + irrigation) / 10.0 if np.isfinite(grain) and np.isfinite(precip) and np.isfinite(irrigation) and precip + irrigation > 0 else np.nan
    irrigation_wp = grain / irrigation / 10.0 if np.isfinite(grain) and np.isfinite(irrigation) and irrigation > 0 else np.nan
    pfp_n = grain / nitrogen if np.isfinite(grain) and np.isfinite(nitrogen) and nitrogen > 0 else np.nan
    uptake_fraction = n_uptake / nitrogen if np.isfinite(n_uptake) and np.isfinite(nitrogen) and nitrogen > 0 else np.nan
    leaching_fraction = n_leaching / nitrogen if np.isfinite(n_leaching) and np.isfinite(nitrogen) and nitrogen > 0 else np.nan
    daily_copy = out / "daily_outputs" / f"{variant}_{management_id}_daily.csv"
    runtime_copy = out / "runtime_state_daily" / f"{variant}_{management_id}_runtime.csv"
    snapshot_copy = out / "snapshots" / variant / management_id
    daily_copy.parent.mkdir(parents=True, exist_ok=True)
    runtime_copy.parent.mkdir(parents=True, exist_ok=True)
    snapshot_copy.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(daily_copy, index=False, encoding="utf-8-sig")
    runtime.to_csv(runtime_copy, index=False, encoding="utf-8-sig")
    shutil.copytree(snapshot, snapshot_copy)
    return {
        "weather_variant": variant,
        "management_id": management_id,
        "episode_completed": True,
        "runtime_rows": len(runtime),
        "requested_irrigation_total_mm": requested_i,
        "requested_nitrogen_total_kg_ha": requested_n,
        "summary_irrigation_total_mm": irrigation,
        "summary_nitrogen_total_kg_ha": nitrogen,
        "rainfall_mm": precip,
        "grain_yield_kg_ha": grain,
        "ETCP_mm": etcp,
        "WP_ET_kg_m3": wp_et,
        "gross_water_productivity_kg_m3": gross_wp,
        "irrigation_productivity_kg_m3": irrigation_wp,
        "PFP_N_kg_kg": pfp_n,
        "N_uptake_kg_ha": n_uptake,
        "N_uptake_fraction_of_applied": uptake_fraction,
        "N_leaching_kg_ha": n_leaching,
        "N_leaching_fraction_of_applied": leaching_fraction,
        "runoff_cumulative_mm_summary_ROCM": runoff,
        "final_soil_water_SWTD_mm": float(pd.to_numeric(growth["SWTD"], errors="coerce").dropna().iloc[-1]) if growth["SWTD"].notna().any() else np.nan,
        "max_WSPD": float(pd.to_numeric(growth["WSPD"], errors="coerce").max()),
        "max_NSTD": float(pd.to_numeric(growth["NSTD"], errors="coerce").max()),
        "ROFC_final_mm": float(pd.to_numeric(daily["ROFC"], errors="coerce").dropna().iloc[-1]),
        "NLCC_final_kg_ha": float(pd.to_numeric(daily["NLCC"], errors="coerce").dropna().iloc[-1]),
        "daily_csv": rel(daily_copy),
        "snapshot_path": rel(snapshot_copy),
        "runtime_state_csv": rel(runtime_copy),
    }


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)
    if OUT.exists():
        raise FileExistsError(f"Refuse to overwrite {OUT}")
    OUT.mkdir(parents=True, exist_ok=False)
    summaries: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    for snapshot in sorted(INPUT.glob("snapshots/*/*")):
        if not (snapshot / "Summary.OUT").exists():
            continue
        variant = snapshot.parent.name
        management_id = snapshot.name
        daily_path = INPUT / "daily_outputs" / f"{variant}_{management_id}_daily.csv"
        runtime_path = INPUT / "runtime_state_daily" / f"{variant}_{management_id}_runtime.csv"
        daily = pd.read_csv(daily_path)
        runtime = pd.read_csv(runtime_path)
        daily_frames.append(daily)
        summaries.append(summarize(variant, management_id, daily, runtime, snapshot, OUT))

    expected = 6 * len(task200.CASES)
    if len(summaries) != expected:
        raise RuntimeError(f"Expected {expected} completed snapshots, found {len(summaries)}")
    summary_df = pd.DataFrame(summaries).sort_values(["weather_variant", "management_id"])
    combined = pd.concat(daily_frames, ignore_index=True)
    evaluation = OUT / "evaluation"
    evaluation.mkdir(parents=True, exist_ok=True)
    daily_path = evaluation / "six_weather_fixed_management_daily.csv"
    summary_path = evaluation / "six_weather_fixed_management_summary.csv"
    effect_path = evaluation / "treatment_effect_vs_I0_N200.csv"
    combined.to_csv(daily_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    baseline = summary_df[summary_df["management_id"].eq("I0_N200")].set_index("weather_variant")
    effect = summary_df.copy()
    for metric in ["grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg", "N_leaching_kg_ha", "ETCP_mm"]:
        effect[f"delta_{metric}_vs_I0_N200"] = effect.apply(
            lambda row: row[metric] - baseline.loc[row["weather_variant"], metric]
            if row["weather_variant"] in baseline.index and np.isfinite(row[metric]) and np.isfinite(baseline.loc[row["weather_variant"], metric])
            else np.nan,
            axis=1,
        )
    effect.to_csv(effect_path, index=False, encoding="utf-8-sig")
    report = {
        "status": "completed_rebuild_from_existing_snapshots",
        "training_executed": False,
        "input_task200": rel(INPUT),
        "episode_count": len(summary_df),
        "summary_parser": "calculate_five_site_wue_nue_from_summary_019_10.parse_summary_out fixed-width parser",
        "outputs": {"daily": rel(daily_path), "summary": rel(summary_path), "treatment_effect": rel(effect_path), "snapshots": rel(OUT / "snapshots")},
    }
    (OUT / "200_rebuild_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
