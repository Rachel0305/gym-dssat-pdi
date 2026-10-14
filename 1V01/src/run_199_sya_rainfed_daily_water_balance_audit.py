"""199: audit the physical response of the six SYA weather variants.

This is a read-only replay of the fixed ``rainfed_N`` management used in
task 164.  It does not load or train a PPO model.  The output keeps both the
runtime state exposed by PDI and the DSSAT output files so that rainfall,
soil-water storage, stress, runoff/drainage, and inorganic nitrogen can be
checked on the same daily axis.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
from ppo_action_safety import normalize_action
from ppo_evaluate import scalar
import run_164_sya_weather_v2_fixed_management_gate as fixed_gate


TASK_ID = "199"
TASK_NAME = "sya_rainfed_daily_water_balance_audit"
DEFAULT_CONFIG = ROOT / "configs" / "163_sya_weather_scenario_bank_v2.json"
MANIFEST_ROOT = ROOT / "benchmark_results" / "166_sya_weather_scenario_bank_v2_fixed_format"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def parse_dssat_table(path: Path) -> pd.DataFrame:
    """Parse the whitespace tables used by DSSAT OUT files."""
    if not path.exists():
        return pd.DataFrame()
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    for raw in path.read_text(encoding="latin1", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("@"):
            parts = line.split()
            if parts and parts[0] == "@":
                parts = parts[1:]
            elif parts:
                parts[0] = parts[0].lstrip("@")
            header = parts
            # PDI can append several complete DSSAT output blocks to one
            # temporary file.  The last block is the completed replay.
            rows = []
            continue
        if header and re.match(r"^\d", line):
            values = line.split()
            n = min(len(header), len(values))
            rows.append(dict(zip(header[:n], values[:n])))
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    for column in frame.columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().sum() == frame[column].notna().sum():
            frame[column] = converted
    return frame


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def copy_state(raw_state: dict[str, Any]) -> dict[str, Any]:
    """Flatten the useful runtime state without relying on the 25-D PPO schema."""
    row: dict[str, Any] = {}
    for key, value in raw_state.items():
        array = np.asarray(value) if isinstance(value, (list, tuple, np.ndarray)) else None
        if array is not None and array.ndim > 0 and array.size > 1:
            flat = array.reshape(-1)
            for index, item in enumerate(flat, start=1):
                row[f"state_{key}_layer_{index}"] = float(item)
        else:
            row[f"state_{key}"] = scalar(value, np.nan)
    return row


def merge_output_tables(snapshot: Path, variant: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    weather = parse_dssat_table(snapshot / "Weather.OUT")
    soilwat = parse_dssat_table(snapshot / "SoilWat.OUT")
    plantgro = parse_dssat_table(snapshot / "PlantGro.OUT")
    soilni = parse_dssat_table(snapshot / "SoilNi.OUT")
    required = {"Weather.OUT": weather, "SoilWat.OUT": soilwat, "PlantGro.OUT": plantgro, "SoilNi.OUT": soilni}
    missing = [name for name, frame in required.items() if frame.empty]
    if missing:
        raise RuntimeError(f"{variant}: missing or empty DSSAT output tables: {missing}")

    keys = [column for column in ("YEAR", "DOY", "DAS") if column in weather.columns and column in soilwat.columns]
    if not {"DOY", "DAS"}.issubset(keys):
        raise RuntimeError(f"{variant}: DSSAT daily join keys DOY/DAS are unavailable")

    weather_cols = [column for column in ["YEAR", "DOY", "DAS", "PRED", "TMXD", "TMND", "SRAD"] if column in weather.columns]
    soilwat_cols = [
        column
        for column in [
            "DOY", "DAS", "SWTD", "SWXD", "ROFC", "DRNC", "PREC", "IR#C", "IRRC", "TDFD", "TDFC", "ROFD",
            *[f"SW{index}D" for index in range(1, 10)],
        ]
        if column in soilwat.columns
    ]
    plant_cols = [column for column in ["DOY", "DAS", "DAP", "GWAD", "CWAD", "WSPD", "NSTD"] if column in plantgro.columns]
    n_cols = [
        column
        for column in ["DOY", "DAS", "NAPC", "NIAD", "NITD", "NHTD", "NUCM", "NLCC", "TDNC", *[f"NT{index}D" for index in range(1, 10)]]
        if column in soilni.columns
    ]

    daily = weather[weather_cols].copy()
    daily = daily.merge(soilwat[soilwat_cols], on=["DOY", "DAS"], how="left", suffixes=("", "_soilwat"))
    daily = daily.merge(plantgro[plant_cols], on=["DOY", "DAS"], how="left", suffixes=("", "_plantgro"))
    daily = daily.merge(soilni[n_cols], on=["DOY", "DAS"], how="left", suffixes=("", "_soilni"))
    daily.insert(0, "variant", variant)
    daily = daily.sort_values([column for column in ["DAS", "DOY"] if column in daily.columns]).reset_index(drop=True)

    if "YEAR" in daily.columns and "DOY" in daily.columns:
        daily["date"] = pd.to_datetime(
            daily["YEAR"].astype(int).astype(str) + daily["DOY"].astype(int).astype(str).str.zfill(3),
            format="%Y%j",
            errors="coerce",
        )

    plant_window = numeric_series(daily, "DAP").notna()

    for column in ["ROFC", "DRNC", "TDFC"]:
        if column in daily.columns:
            daily[f"{column}_daily_delta"] = numeric_series(daily, column).diff().clip(lower=0.0).fillna(0.0)
    if "SWTD" in daily.columns:
        daily["soil_water_profile_daily_delta_mm"] = numeric_series(daily, "SWTD").diff().fillna(0.0)

    # Keep the DSSAT names, but add explicit aliases for the fields most useful
    # in the diagnostic table.  The raw columns remain authoritative.
    alias_map = {
        "PRED": "rainfall_mm",
        "SWTD": "soil_water_profile_mm",
        "SWXD": "soil_water_extractable_mm",
        "ROFD": "runoff_daily_mm_as_reported",
        "ROFC": "runoff_cumulative_mm_as_reported",
        "DRNC": "drainage_cumulative_mm_as_reported",
        "TDFD": "tile_drainage_daily_mm_as_reported",
        "TDFC": "tile_drainage_cumulative_mm_as_reported",
        "PREC": "soilwat_precipitation_mm",
        "IRRC": "soilwat_irrigation_cumulative_mm",
        "WSPD": "water_stress_index_wspd",
        "NSTD": "nitrogen_stress_index_nstd",
        "NITD": "soil_inorganic_nitrate_total_as_reported",
        "NHTD": "soil_inorganic_ammonium_total_as_reported",
        "NUCM": "nitrogen_uptake_cumulative_kg_ha_as_reported",
        "NLCC": "nitrogen_leaching_cumulative_kg_ha_as_reported",
    }
    for source, target in alias_map.items():
        if source in daily.columns:
            daily[target] = daily[source]
    if "SWXD" in daily.columns:
        daily["soil_water_SWXD_raw"] = daily["SWXD"]
    for index in range(1, 10):
        if f"SW{index}D" in daily.columns:
            daily[f"soil_water_layer_{index}_vwc"] = daily[f"SW{index}D"]
        if f"NT{index}D" in daily.columns:
            daily[f"soil_total_inorganic_n_layer_{index}_ppm"] = daily[f"NT{index}D"]

    summary = {
        "variant": variant,
        "snapshot": rel(snapshot),
        "dssat_daily_rows": int(len(daily)),
        "rainfall_total_mm": float(numeric_series(daily.loc[plant_window], "PRED").sum()),
        "rainfall_total_mm_all_dssat_rows": float(numeric_series(daily, "PRED").sum()),
        "rainfall_total_mm_dap_ge1": float(numeric_series(daily.loc[numeric_series(daily, "DAP") >= 1], "PRED").sum()),
        "final_soil_water_profile_mm": float(numeric_series(daily, "SWTD").dropna().iloc[-1]) if numeric_series(daily, "SWTD").notna().any() else np.nan,
        "soil_water_profile_daily_delta_min_mm": float(numeric_series(daily, "soil_water_profile_daily_delta_mm").min()),
        "soil_water_profile_daily_delta_max_mm": float(numeric_series(daily, "soil_water_profile_daily_delta_mm").max()),
        "soil_water_SWXD_min_mm": float(numeric_series(daily, "SWXD").min()),
        "soil_water_SWXD_max_mm": float(numeric_series(daily, "SWXD").max()),
        "max_water_stress_wspd": float(numeric_series(daily, "WSPD").max()),
        "max_nitrogen_stress_nstd": float(numeric_series(daily, "NSTD").max()),
        "raw_WSPD_nonzero_rows": int((numeric_series(daily, "WSPD").abs() > 1e-12).sum()),
        "raw_TDFD_nonzero_rows": int((numeric_series(daily, "TDFD").abs() > 1e-12).sum()),
        "raw_TDFC_nonzero_rows": int((numeric_series(daily, "TDFC").abs() > 1e-12).sum()),
        "raw_DRNC_nonzero_rows": int((numeric_series(daily, "DRNC").abs() > 1e-12).sum()),
        "final_nitrogen_leaching_cumulative": float(numeric_series(daily, "NLCC").dropna().iloc[-1]) if numeric_series(daily, "NLCC").notna().any() else np.nan,
        "max_runoff_daily_as_reported": float(numeric_series(daily, "ROFD").max()),
        "max_drainage_daily_as_reported": float(numeric_series(daily, "TDFD").max()),
    }
    return daily, summary


def run_variant(
    env_config: dict[str, Any],
    scenario_root: Path,
    variant: str,
    management: dict[int, float],
    output_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = scenario_root
    env = None
    runtime_rows: list[dict[str, Any]] = []
    start = time.time()
    snapshot: Path | None = None
    try:
        env = direct_ppo.make_base_env(env_config, "SYA", 2005, 0, f"{variant}_rainfed_N", evaluation=True)
        obs, info = env.reset()
        done = False
        step_count = 0
        while not done and step_count < int(env_config["runtime"]["max_steps"]):
            raw_before = dict(getattr(env.unwrapped, "_state", {}))
            dap_raw = scalar(raw_before.get("dap"), step_count + 1)
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            action_real = {"amir": 0.0, "anfer": float(management.get(dap, 0.0))}
            action = normalize_action(env.formator.action_names, env.formator.action_space_dict, action_real)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            raw_after = dict(getattr(env.unwrapped, "_state", {}))
            row = {
                "variant": variant,
                "runtime_step": step_count,
                "decision_dap": dap,
                "irrigation_requested_mm": 0.0,
                "nitrogen_requested_kg_ha": action_real["anfer"],
                "reward": float(reward),
                "done": done,
                **{f"before_{key}": value for key, value in copy_state(raw_before).items()},
                **{f"after_{key}": value for key, value in copy_state(raw_after).items()},
            }
            runtime_rows.append(row)
            step_count += 1
        if not runtime_rows or not runtime_rows[-1]["done"]:
            raise RuntimeError(f"{variant}: episode did not terminate within max_steps")
        tmp = getattr(env.unwrapped, "_tmp_folder", None)
        if not tmp:
            raise RuntimeError(f"{variant}: DSSAT temporary folder is unavailable")
        tmp_path = Path(tmp)
        if not (tmp_path / "Summary.OUT").exists():
            raise FileNotFoundError(f"{variant}: missing Summary.OUT in {tmp_path}")
        snapshot = output_root / "snapshots" / variant / "rainfed_N"
        if snapshot.exists():
            raise FileExistsError(f"Refuse to overwrite snapshot: {snapshot}")
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(tmp_path, snapshot)
    finally:
        if env is not None:
            env.close()
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root

    runtime = pd.DataFrame(runtime_rows)
    runtime_path = output_root / "runtime_state_daily" / f"{variant}_rainfed_N_runtime.csv"
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime.to_csv(runtime_path, index=False, encoding="utf-8-sig")
    daily, summary = merge_output_tables(snapshot, variant)
    daily["runtime_state_csv"] = rel(runtime_path)
    daily["snapshot_path"] = rel(snapshot)
    daily_path = output_root / "daily_outputs" / f"{variant}_rainfed_N_dssat_daily.csv"
    daily_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    summary.update(
        {
            "episode_completed": True,
            "runtime_rows": int(len(runtime)),
            "total_irrigation_requested_mm": float(runtime["irrigation_requested_mm"].sum()),
            "total_nitrogen_requested_kg_ha": float(runtime["nitrogen_requested_kg_ha"].sum()),
            "runtime_state_csv": rel(runtime_path),
            "daily_csv": rel(daily_path),
            "elapsed_seconds": time.time() - start,
        }
    )
    return daily, summary


def run(config_path: Path, output_root: Path) -> dict[str, Any]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    manifest_candidates = sorted(MANIFEST_ROOT.glob("*_manifest.csv"))
    if len(manifest_candidates) != 1:
        raise FileNotFoundError(f"Expected one scenario manifest in {MANIFEST_ROOT}, found {manifest_candidates}")
    manifest = pd.read_csv(manifest_candidates[0], keep_default_na=False)
    if output_root.exists():
        raise FileExistsError(f"Refuse to overwrite existing output: {rel(output_root)}")
    output_root.mkdir(parents=True, exist_ok=False)

    env_config = fixed_gate.make_env_config(output_root)
    management = {1: 50.0, 30: 50.0, 60: 50.0}
    all_daily: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for item in manifest.itertuples(index=False):
        variant = str(item.variant)
        scenario_root = ROOT / str(item.scenario_input_root)
        try:
            daily, summary = run_variant(env_config, scenario_root, variant, management, output_root)
            all_daily.append(daily)
            summaries.append(summary)
        except Exception as exc:
            failures.append({"variant": variant, "error": repr(exc)})

    if len(all_daily) != len(manifest):
        raise RuntimeError(f"Some variants failed: {failures}")
    combined = pd.concat(all_daily, ignore_index=True)
    combined_path = output_root / "evaluation" / "rainfed_N_six_weather_daily.csv"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
    summary_df = pd.DataFrame(summaries)
    summary_path = output_root / "evaluation" / "rainfed_N_six_weather_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    raw_audit_columns = [
        "variant",
        "soil_water_SWXD_min_mm",
        "soil_water_SWXD_max_mm",
        "soil_water_profile_daily_delta_min_mm",
        "soil_water_profile_daily_delta_max_mm",
        "max_water_stress_wspd",
        "raw_WSPD_nonzero_rows",
        "raw_TDFD_nonzero_rows",
        "raw_TDFC_nonzero_rows",
        "raw_DRNC_nonzero_rows",
        "total_irrigation_requested_mm",
        "total_nitrogen_requested_kg_ha",
    ]
    raw_audit_path = output_root / "evaluation" / "raw_field_audit.csv"
    summary_df[[column for column in raw_audit_columns if column in summary_df.columns]].to_csv(raw_audit_path, index=False, encoding="utf-8-sig")

    comparison_rows: list[dict[str, Any]] = []
    for dap, group in combined.groupby("DAP", dropna=False):
        row: dict[str, Any] = {"DAP": dap}
        for column in ["PRED", "SWTD", "WSPD", "NSTD", "ROFD", "TDFD", "NLCC", "NT1D", "NT2D", "NT3D", "NT4D", "NT5D", "NT6D", "NT7D", "NT8D", "NT9D"]:
            values = numeric_series(group, column).dropna()
            row[f"{column}_min"] = float(values.min()) if not values.empty else np.nan
            row[f"{column}_max"] = float(values.max()) if not values.empty else np.nan
            row[f"{column}_range"] = float(values.max() - values.min()) if not values.empty else np.nan
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows)
    comparison_path = output_root / "evaluation" / "same_dap_cross_weather_range.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    report = {
        "status": "completed",
        "task": f"{TASK_ID}_{TASK_NAME}",
        "training_executed": False,
        "management": {"irrigation": "none (0 mm every day)", "nitrogen": management, "label": "rainfed_N"},
        "scenario_count": int(len(manifest)),
        "episode_count_completed": int(len(summaries)),
        "failures": failures,
        "manifest": rel(manifest_candidates[0]),
        "outputs": {
            "combined_daily": rel(combined_path),
            "scenario_summary": rel(summary_path),
            "raw_field_audit": rel(raw_audit_path),
            "same_dap_range": rel(comparison_path),
            "raw_snapshots": rel(output_root / "snapshots"),
            "runtime_state": rel(output_root / "runtime_state_daily"),
        },
        "field_notes": {
            "soil_water": "SoilWat.OUT SWTD is profile soil water storage in mm; SW1D-SW9D are layer volumetric water contents as reported by DSSAT. SWXD is kept as extractable-water field; soil_water_profile_daily_delta_mm is the signed day-to-day change computed from SWTD.",
            "stress": "PlantGro.OUT WSPD and NSTD; 0 means no stress in the existing project convention.",
            "runoff_drainage": "The table retains DSSAT SoilWat.OUT raw columns ROFD/ROFC, DRNC, TDFD/TDFC. DRNC is cumulative drainage; TDFD/TDFC are tile-drain water loss daily/cumulative fields. They are reported with raw nonzero counts and are not conflated with transpiration.",
            "nitrogen": "SoilNi.OUT NT1D-NT9D are layer total inorganic-N concentration fields; NITD/NHTD/NUCM/NLCC are retained with explicit aliases.",
            "pdi_leaching": "Runtime tleachd/cleach are retained separately as PDI-exposed nitrogen-leaching state fields, not water percolation.",
        },
    }
    (output_root / "199_audit_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = [
        f"# {TASK_ID} SYA six-weather rainfed daily water-balance audit",
        "",
        "- Training executed: no; this is six fixed-management DSSAT replays.",
        "- Management: no irrigation; N requests at DAP 1/30/60, 50 kg/ha each.",
        f"- Completed episodes: `{len(summaries)}/{len(manifest)}`.",
        "- `SoilWat.OUT` and `SoilNi.OUT` snapshots are copied for independent inspection.",
        "- `SWTD` is profile soil-water storage; `SWXD` is extractable water, while `soil_water_profile_daily_delta_mm` is computed from `SWTD.diff()`.",
        "- `WSPD/NSTD` are read directly from the final-block `PlantGro.OUT`; `raw_field_audit.csv` records their raw nonzero counts.",
        "- `ROFD/ROFC`, `DRNC`, `TDFD/TDFC` are kept as DSSAT raw columns; PDI `tleachd/cleach` are nitrogen-leaching fields and are not called water drainage.",
        "- `rainfed_N` means zero irrigation but nonzero fixed nitrogen; the actual requested totals are included in the scenario summary.",
        "",
        f"- Combined daily table: `{rel(combined_path)}`.",
        f"- Scenario summary: `{rel(summary_path)}`.",
        f"- Raw field audit: `{rel(raw_audit_path)}`.",
        f"- Same-DAP cross-weather ranges: `{rel(comparison_path)}`.",
        f"- Raw snapshots: `{rel(output_root / 'snapshots')}`.",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    print(json.dumps(run(config.resolve(), output_root), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
