"""200: fixed-management sensitivity for SYA's six weather scenarios.

This is a low-cost DSSAT replay matrix, not a PPO experiment.  It separates
the marginal response to nitrogen from the marginal response to irrigation
before changing the policy, reward, or observation design.
"""

from __future__ import annotations

import argparse
import json
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
import run_199_sya_rainfed_daily_water_balance_audit as audit199
from calculate_five_site_wue_nue_from_summary_019_10 import num as summary_num, parse_summary_out


TASK_ID = "200"
TASK_NAME = "sya_fixed_management_sensitivity"
DEFAULT_CONFIG = ROOT / "configs" / "163_sya_weather_scenario_bank_v2.json"
MANIFEST_ROOT = ROOT / "benchmark_results" / "166_sya_weather_scenario_bank_v2_fixed_format"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"


CASES: list[dict[str, Any]] = [
    {"management_id": "I0_N0", "irrigation": {}, "nitrogen": {}},
    {"management_id": "I0_N100", "irrigation": {}, "nitrogen": {1: 25.0, 30: 25.0, 60: 25.0}},
    {"management_id": "I0_N200", "irrigation": {}, "nitrogen": {1: 50.0, 30: 50.0, 60: 50.0}},
    {"management_id": "I0_N300", "irrigation": {}, "nitrogen": {1: 75.0, 30: 75.0, 60: 75.0}},
    {"management_id": "I60_N200", "irrigation": {35: 30.0, 65: 30.0}, "nitrogen": {1: 50.0, 30: 50.0, 60: 50.0}},
    {"management_id": "I120_N200", "irrigation": {35: 50.0, 65: 50.0, 95: 20.0}, "nitrogen": {1: 50.0, 30: 50.0, 60: 50.0}},
    {"management_id": "I240_N200", "irrigation": {35: 50.0, 50: 50.0, 65: 50.0, 80: 50.0, 95: 40.0}, "nitrogen": {1: 50.0, 30: 50.0, 60: 50.0}},
]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def valid_num(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) and number >= 0 else np.nan


def summary_value(summary_row: dict[str, Any], column: str) -> float:
    value = summary_num(summary_row, column)
    return valid_num(value)


def select_summary_row(rows: list[dict[str, Any]], expected_yield: float, expected_i: float, expected_n: float) -> dict[str, Any]:
    candidates: list[tuple[float, int, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        hwam = summary_num(row, "HWAM")
        ircm = summary_num(row, "IRCM")
        nicm = summary_num(row, "NICM")
        if hwam is None or ircm is None or nicm is None:
            continue
        score = abs(float(hwam) - expected_yield) + abs(float(ircm) - expected_i) + abs(float(nicm) - expected_n)
        candidates.append((score, -index, row))
    if not candidates:
        raise ValueError("No usable Summary.OUT row")
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def run_case(
    env_config: dict[str, Any],
    scenario_root: Path,
    variant: str,
    case: dict[str, Any],
    output_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    management_id = str(case["management_id"])
    old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = scenario_root
    env = None
    runtime_rows: list[dict[str, Any]] = []
    start = time.time()
    try:
        env = direct_ppo.make_base_env(env_config, "SYA", 2005, 0, f"{variant}_{management_id}", evaluation=True)
        obs, info = env.reset()
        done = False
        step_count = 0
        while not done and step_count < int(env_config["runtime"]["max_steps"]):
            raw_before = dict(getattr(env.unwrapped, "_state", {}))
            dap_raw = scalar(raw_before.get("dap"), step_count + 1)
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            action_real = {
                "amir": float(case["irrigation"].get(dap, 0.0)),
                "anfer": float(case["nitrogen"].get(dap, 0.0)),
            }
            action = normalize_action(env.formator.action_names, env.formator.action_space_dict, action_real)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            raw_after = dict(getattr(env.unwrapped, "_state", {}))
            runtime_rows.append(
                {
                    "weather_variant": variant,
                    "management_id": management_id,
                    "runtime_step": step_count,
                    "decision_dap": dap,
                    "irrigation_requested_mm": action_real["amir"],
                    "nitrogen_requested_kg_ha": action_real["anfer"],
                    "reward": float(reward),
                    "done": done,
                    **{f"before_{key}": value for key, value in audit199.copy_state(raw_before).items()},
                    **{f"after_{key}": value for key, value in audit199.copy_state(raw_after).items()},
                }
            )
            step_count += 1
        if not runtime_rows or not runtime_rows[-1]["done"]:
            raise RuntimeError(f"{variant}/{management_id}: episode did not terminate")
        tmp = getattr(env.unwrapped, "_tmp_folder", None)
        if not tmp:
            raise RuntimeError(f"{variant}/{management_id}: DSSAT temporary folder is unavailable")
        snapshot = output_root / "snapshots" / variant / management_id
        if snapshot.exists():
            raise FileExistsError(f"Refuse to overwrite snapshot: {snapshot}")
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        from shutil import copytree

        copytree(Path(tmp), snapshot)
    finally:
        if env is not None:
            env.close()
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root

    runtime = pd.DataFrame(runtime_rows)
    runtime_path = output_root / "runtime_state_daily" / f"{variant}_{management_id}_runtime.csv"
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime.to_csv(runtime_path, index=False, encoding="utf-8-sig")

    daily, _ = audit199.merge_output_tables(snapshot, variant)
    daily["weather_variant"] = variant
    daily["management_id"] = management_id
    daily["irrigation_requested_mm"] = pd.to_numeric(daily["DAP"], errors="coerce").map(case["irrigation"]).fillna(0.0)
    daily["nitrogen_requested_kg_ha"] = pd.to_numeric(daily["DAP"], errors="coerce").map(case["nitrogen"]).fillna(0.0)
    daily["runtime_state_csv"] = rel(runtime_path)
    daily["snapshot_path"] = rel(snapshot)
    daily_path = output_root / "daily_outputs" / f"{variant}_{management_id}_daily.csv"
    daily_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")

    summary_rows = parse_summary_out(snapshot / "Summary.OUT")
    grain_series = pd.to_numeric(daily.get("GWAD", pd.Series(dtype=float)), errors="coerce").dropna()
    daily_grain = float(grain_series.iloc[-1]) if not grain_series.empty else np.nan
    expected_i = float(runtime["irrigation_requested_mm"].sum())
    expected_n = float(runtime["nitrogen_requested_kg_ha"].sum())
    summary_row = select_summary_row(summary_rows, daily_grain, expected_i, expected_n)
    grain = summary_value(summary_row, "HWAM")
    if not np.isfinite(grain):
        grain = daily_grain
    irrigation = summary_value(summary_row, "IRCM")
    nitrogen = summary_value(summary_row, "NICM")
    etcp = summary_value(summary_row, "ETCP")
    precip = summary_value(summary_row, "PRCP")
    n_uptake = summary_value(summary_row, "NUCM")
    n_leaching = summary_value(summary_row, "NLCM")
    runoff = summary_value(summary_row, "ROCM")
    wp_et = grain / etcp / 10.0 if np.isfinite(grain) and np.isfinite(etcp) and etcp > 0 else np.nan
    pfp_n = grain / nitrogen if np.isfinite(grain) and np.isfinite(nitrogen) and nitrogen > 0 else np.nan
    wp_gross = grain / (precip + irrigation) / 10.0 if np.isfinite(grain) and np.isfinite(precip) and np.isfinite(irrigation) and precip + irrigation > 0 else np.nan
    irrigation_wp = grain / irrigation / 10.0 if np.isfinite(grain) and np.isfinite(irrigation) and irrigation > 0 else np.nan
    n_uptake_fraction = n_uptake / nitrogen if np.isfinite(n_uptake) and np.isfinite(nitrogen) and nitrogen > 0 else np.nan
    n_leaching_fraction = n_leaching / nitrogen if np.isfinite(n_leaching) and np.isfinite(nitrogen) and nitrogen > 0 else np.nan
    plant_window = daily[daily["DAP"].notna()]
    summary = {
        "weather_variant": variant,
        "management_id": management_id,
        "episode_completed": True,
        "runtime_rows": len(runtime),
        "requested_irrigation_total_mm": float(runtime["irrigation_requested_mm"].sum()),
        "requested_nitrogen_total_kg_ha": float(runtime["nitrogen_requested_kg_ha"].sum()),
        "summary_irrigation_total_mm": irrigation,
        "summary_nitrogen_total_kg_ha": nitrogen,
        "rainfall_mm": precip,
        "grain_yield_kg_ha": grain,
        "ETCP_mm": etcp,
        "WP_ET_kg_m3": wp_et,
        "gross_water_productivity_kg_m3": wp_gross,
        "irrigation_productivity_kg_m3": irrigation_wp,
        "PFP_N_kg_kg": pfp_n,
        "N_uptake_kg_ha": n_uptake,
        "N_uptake_fraction_of_applied": n_uptake_fraction,
        "N_leaching_kg_ha": n_leaching,
        "N_leaching_fraction_of_applied": n_leaching_fraction,
        "runoff_cumulative_mm_summary_ROCM": runoff,
        "final_soil_water_SWTD_mm": float(pd.to_numeric(plant_window["SWTD"], errors="coerce").dropna().iloc[-1]) if not plant_window.empty and plant_window["SWTD"].notna().any() else np.nan,
        "max_WSPD": float(pd.to_numeric(plant_window["WSPD"], errors="coerce").max()) if not plant_window.empty else np.nan,
        "max_NSTD": float(pd.to_numeric(plant_window["NSTD"], errors="coerce").max()) if not plant_window.empty else np.nan,
        "ROFC_final_mm": float(pd.to_numeric(daily["ROFC"], errors="coerce").dropna().iloc[-1]) if daily["ROFC"].notna().any() else np.nan,
        "NLCC_final_kg_ha": float(pd.to_numeric(daily["NLCC"], errors="coerce").dropna().iloc[-1]) if daily["NLCC"].notna().any() else np.nan,
        "daily_csv": rel(daily_path),
        "snapshot_path": rel(snapshot),
        "runtime_state_csv": rel(runtime_path),
        "elapsed_seconds": time.time() - start,
    }
    return daily, summary


def run(config_path: Path, output_root: Path) -> dict[str, Any]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    manifests = sorted(MANIFEST_ROOT.glob("*_manifest.csv"))
    if len(manifests) != 1:
        raise FileNotFoundError(f"Expected one manifest in {MANIFEST_ROOT}, found {manifests}")
    manifest = pd.read_csv(manifests[0], keep_default_na=False)
    if output_root.exists():
        raise FileExistsError(f"Refuse to overwrite existing output: {rel(output_root)}")
    output_root.mkdir(parents=True, exist_ok=False)
    env_config = fixed_gate.make_env_config(output_root)

    all_daily: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for item in manifest.itertuples(index=False):
        variant = str(item.variant)
        scenario_root = ROOT / str(item.scenario_input_root)
        for case in CASES:
            try:
                daily, summary = run_case(env_config, scenario_root, variant, case, output_root)
                all_daily.append(daily)
                summaries.append(summary)
            except Exception as exc:
                failures.append({"weather_variant": variant, "management_id": case["management_id"], "error": repr(exc)})

    expected = len(manifest) * len(CASES)
    if len(summaries) != expected:
        raise RuntimeError(f"Fixed-management sensitivity incomplete: {len(summaries)}/{expected}; {failures}")

    summary_df = pd.DataFrame(summaries)
    combined = pd.concat(all_daily, ignore_index=True)
    combined_path = output_root / "evaluation" / "six_weather_fixed_management_daily.csv"
    summary_path = output_root / "evaluation" / "six_weather_fixed_management_summary.csv"
    effect_path = output_root / "evaluation" / "treatment_effect_vs_I0_N200.csv"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    baseline = summary_df[summary_df["management_id"].eq("I0_N200")].set_index("weather_variant")
    effect = summary_df.copy()
    for metric in ["grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg", "N_leaching_kg_ha", "ETCP_mm"]:
        effect[f"delta_{metric}_vs_I0_N200"] = effect.apply(
            lambda row: float(row[metric] - baseline.loc[row["weather_variant"], metric])
            if row["weather_variant"] in baseline.index and np.isfinite(row[metric]) and np.isfinite(baseline.loc[row["weather_variant"], metric])
            else np.nan,
            axis=1,
        )
    effect.to_csv(effect_path, index=False, encoding="utf-8-sig")

    report = {
        "status": "completed",
        "task": f"{TASK_ID}_{TASK_NAME}",
        "training_executed": False,
        "weather_scenarios": sorted(manifest["variant"].astype(str).tolist()),
        "management_cases": CASES,
        "case_count": len(CASES),
        "scenario_count": len(manifest),
        "episode_count": len(summaries),
        "failures": failures,
        "metrics": {
            "WP_ET": "grain_yield_kg_ha / ETCP_mm / 10",
            "gross_water_productivity": "grain_yield_kg_ha / (rainfall_mm + irrigation_mm) / 10",
            "PFP_N": "grain_yield_kg_ha / applied nitrogen kg/ha",
            "N_uptake_fraction": "NUCM / NICM",
            "N_leaching_fraction": "NLCM / NICM",
        },
        "outputs": {
            "daily": rel(combined_path),
            "summary": rel(summary_path),
            "treatment_effect": rel(effect_path),
            "snapshots": rel(output_root / "snapshots"),
        },
    }
    (output_root / "200_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = [
        f"# {TASK_ID} SYA fixed-management sensitivity",
        "",
        "- Training executed: no; this is a fixed DSSAT replay matrix.",
        f"- Episodes: `{len(summaries)}` = `{len(manifest)}` weather variants x `{len(CASES)}` management cases.",
        "- Nitrogen sensitivity: I0_N0/I0_N100/I0_N200/I0_N300.",
        "- Irrigation sensitivity at fixed N200: I60_N200/I120_N200/I240_N200.",
        "- WP_ET uses Summary.OUT ETCP; PFP_N uses Summary.OUT NICM; N uptake and leaching use NUCM/NLCM.",
        "",
        f"- Daily table: `{rel(combined_path)}`.",
        f"- Summary: `{rel(summary_path)}`.",
        f"- Treatment effect vs I0_N200: `{rel(effect_path)}`.",
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
