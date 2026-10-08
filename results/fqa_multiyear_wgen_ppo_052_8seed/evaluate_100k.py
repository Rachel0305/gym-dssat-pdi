"""Evaluate one completed FQA WGEN PPO seed on fixed 2014-2023 weather.

Evaluation is sequential and saves DSSAT snapshots, daily traces, exact
Summary.OUT/ETCP metrics, and YC-055_03-style per-seed figures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
VALIDATION = BASE / "validation"
FIGURE_ROOT = BASE / "055_03_five_scenario" / "FQ"
YEARS = list(range(2014, 2024))
STATION = "FQA"
SITE = "FQ"
EVAL_SEED = 0
BASELINE_SUMMARY = ROOT / "benchmark_results/051_03_fqa_originIC_051_00_fqa_originIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/tables/051_03_fqa_five_scenario_season_summary.csv"
BASELINE_DAILY = ROOT / "benchmark_results/051_03_fqa_originIC_051_00_fqa_originIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/tables/051_03_fqa_five_scenario_daily.csv"
BASELINE_SCENARIOS = ["null", "recorded_farmer_template", "dssat_auto_external_n", "official_extension_expert"]
BASELINE_LABELS = {
    "null": "Null",
    "recorded_farmer_template": "Recorded template",
    "dssat_auto_external_n": "DSSAT auto + external N",
    "official_extension_expert": "Official expert",
}
INPUT_ROOT = ROOT / "DSSAT_auto_validation/multisite_new_cultivar_inputs_013"
PROMPT = ROOT / "prompt_02/052_fqa_wgen_ppo_100k_8seed_validation_figures.md"
ENGINE_DOC = ROOT / "docs/052_fqa_wgen_ppo_100k_8seed_record.md"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/051_fqa_originIC_site_transfer"))
import ppo_safe_rendering  # noqa: E402
import run_all_year_direct_action_safe_ppo as direct_ppo  # noqa: E402
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base03222  # noqa: E402
import run_multisite_input_ic1_four_baseline_rebuild_034_00 as baseline  # noqa: E402
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine  # noqa: E402
from run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 import snapshot_from_env  # noqa: E402
from run_051_03_fqa_originIC_five_scenario_figures import daily_from_snapshot  # noqa: E402
from hl_fq_8seed_cross_site import plot_hl_fq_five_scenario_05503_style as plots  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def model_path(seed: int) -> Path:
    if seed == 0:
        folder = ROOT / "results/fqa_multiyear_wgen_ppo_051/attempt_01/models"
        gate_path = ROOT / "results/fqa_multiyear_wgen_ppo_051/attempt_01/audit_gate.json"
    else:
        folder = BASE / f"seed_{seed:02d}/attempt_01/models"
        gate_path = BASE / f"seed_{seed:02d}/attempt_01/audit_gate.json"
    if not gate_path.is_file():
        raise FileNotFoundError(f"Seed {seed} has no formal audit gate: {gate_path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS_100K_ARCHIVE_ONLY":
        raise RuntimeError(f"Seed {seed} formal gate is not PASS: {gate.get('status')}")
    checkpoint = folder / "checkpoint_100000.zip"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Exact 100,000-step checkpoint missing for seed {seed}: {checkpoint}")
    return checkpoint


def configure_eval():
    engine.TASK_ID = "051_00"
    engine.TASK_NAME = "fqa_originIC_expanded_action_maskableppo"
    engine.BASE_OUT = VALIDATION
    engine.BASE_DOC = ENGINE_DOC
    engine.PROMPT = PROMPT
    engine.LOWIC_INPUT_ROOT = INPUT_ROOT
    engine.STATION = STATION
    engine.SITES = [STATION]
    engine.BINARY_IRRIGATION_LEVELS = [0.0, 15.0, 30.0, 45.0]
    engine.BINARY_NITROGEN_LEVELS = [0.0, 40.0, 80.0, 120.0]
    base03222.SITE_NAMES[STATION] = SITE
    engine.patch_base_module(VALIDATION, ENGINE_DOC, 100_000, [25_000, 50_000, 75_000, 100_000])
    config = base03222.load_config()
    selection = base03222.build_selection(base03222.load_split())
    env_config = direct_ppo.build_env_config(config, selection)
    env_config["paths"]["output_root"] = VALIDATION.relative_to(ROOT).as_posix()
    return config, env_config


def run_year(seed: int, year: int, model, config, env_config, target: Path, model_sha: str) -> Path:
    metadata_path = target / "evaluation_metadata.json"
    if target.exists():
        if not metadata_path.is_file():
            raise FileExistsError(f"Snapshot folder exists without evaluation metadata; preserving it: {target}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata != {"seed": seed, "year": year, "evaluation_seed": EVAL_SEED, "model_sha256": model_sha}:
            raise RuntimeError(f"Existing snapshot provenance mismatch: {target}")
        if not (target / "Summary.OUT").is_file():
            raise RuntimeError(f"Existing snapshot is incomplete: {target}")
        return target

    env = None
    try:
        env = base03222.base.make_env(
            config, env_config, STATION, int(year), EVAL_SEED,
            f"FQA_{year}_052_seed_{seed}_historical_validation", evaluation=True,
        )
        obs, _info = env.reset()
        from sb3_contrib.common.maskable.utils import get_action_masks

        done = False
        steps = 0
        while not done and steps < 260:
            action, _ = model.predict(obs, action_masks=get_action_masks(env), deterministic=True)
            obs, _reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)
            steps += 1
        if not done:
            raise RuntimeError(f"Seed {seed} FQA{year} validation did not finish in 260 steps")
        source = snapshot_from_env(env)
        if not (source / "Summary.OUT").is_file():
            raise RuntimeError(f"Seed {seed} FQA{year} snapshot has no Summary.OUT")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
        metadata_path.write_text(json.dumps({"seed": seed, "year": year, "evaluation_seed": EVAL_SEED, "model_sha256": model_sha}, indent=2) + "\n", encoding="utf-8")
        return target
    finally:
        if env is not None:
            env.close()


def reward_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    step_reward = -1.1 * pd.to_numeric(result.irrigation_executed_mm, errors="coerce").fillna(0.0)
    step_reward -= 1.58 * pd.to_numeric(result.nitrogen_executed_kg_ha, errors="coerce").fillna(0.0)
    if len(result):
        step_reward.iloc[-1] += 0.158 * float(pd.to_numeric(result.grain_yield_kg_ha, errors="coerce").iloc[-1])
    result["common_reward_step"] = step_reward
    result["common_cumulative_reward"] = step_reward.cumsum()
    return result


def build_seed_frames(seed: int, model, model_file: Path, config, env_config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    baseline_summary = pd.read_csv(BASELINE_SUMMARY, keep_default_na=False)
    baseline_daily = pd.read_csv(BASELINE_DAILY, keep_default_na=False)
    baseline_summary["year"] = pd.to_numeric(baseline_summary.year).astype(int)
    baseline_daily["year"] = pd.to_numeric(baseline_daily.year).astype(int)
    baseline_daily["date"] = pd.to_datetime(baseline_daily["date"])
    baseline_summary = baseline_summary.loc[
        baseline_summary.year.isin(YEARS) & baseline_summary.scenario.isin(BASELINE_SCENARIOS)
    ].copy()
    baseline_daily = baseline_daily.loc[
        baseline_daily.year.isin(YEARS) & baseline_daily.scenario.isin(BASELINE_SCENARIOS)
    ].copy()
    if len(baseline_summary) != len(YEARS) * len(BASELINE_SCENARIOS):
        raise RuntimeError("Frozen FQ baseline summary is incomplete or duplicated")
    if set(baseline_daily.year.unique()) != set(YEARS):
        raise RuntimeError("Frozen FQ baseline daily weather does not cover 2014-2023")

    scenario = plots.scenario_key(seed)
    scenario_label = plots.scenario_label(seed)
    metrics: list[dict[str, Any]] = []
    ppo_daily: list[pd.DataFrame] = []
    ppo_actions: list[dict[str, Any]] = []
    weather_audit: dict[str, Any] = {}
    yearly_sequences: dict[str, list[list[Any]]] = {}
    model_hash = sha256(model_file)

    for year in YEARS:
        target = VALIDATION / f"seed_{seed:02d}/snapshots/FQA/{year}/{scenario}"
        snapshot = run_year(seed, year, model, config, env_config, target, model_hash)
        raw = daily_from_snapshot(snapshot, year, scenario)
        final_yield = float(pd.to_numeric(raw.grain_yield_kg_ha, errors="coerce").iloc[-1])
        measured = baseline.metrics_from_snapshot(snapshot, final_yield)
        ir = pd.to_numeric(raw.irrigation_executed_mm, errors="coerce").fillna(0.0)
        nr = pd.to_numeric(raw.nitrogen_executed_kg_ha, errors="coerce").fillna(0.0)
        raw["site"] = SITE
        raw["scenario_label"] = scenario_label
        raw["water_stress"] = pd.to_numeric(raw.water_stress_index_wspd, errors="coerce")
        raw["water_stress_source"] = "WSPD"
        raw["nitrogen_stress"] = pd.to_numeric(raw.nitrogen_stress_index_nstd, errors="coerce")
        raw["nitrogen_stress_source"] = "NSTD"
        raw["common_reward_step"] = -1.1 * ir - 1.58 * nr
        raw.loc[raw.index[-1], "common_reward_step"] += 0.158 * final_yield
        raw["common_cumulative_reward"] = raw.common_reward_step.cumsum()
        ppo_daily.append(raw)

        sequence = []
        for row in raw.loc[ir.gt(0) | nr.gt(0)].itertuples():
            sequence.append([int(row.dap), round(float(row.irrigation_executed_mm), 3), round(float(row.nitrogen_executed_kg_ha), 3)])
        yearly_sequences[str(year)] = sequence
        weather = baseline_daily.loc[
            (baseline_daily.year.eq(year)) & baseline_daily.scenario.eq("null"),
            ["date", "rainfall_mm", "tmax_c", "tmin_c"],
        ].merge(raw[["date", "rainfall_mm", "tmax_c", "tmin_c"]], on="date", suffixes=("_base", "_ppo"))
        rain_diff = (pd.to_numeric(weather.rainfall_mm_base) - pd.to_numeric(weather.rainfall_mm_ppo)).abs()
        tmax_diff = (pd.to_numeric(weather.tmax_c_base) - pd.to_numeric(weather.tmax_c_ppo)).abs()
        tmin_diff = (pd.to_numeric(weather.tmin_c_base) - pd.to_numeric(weather.tmin_c_ppo)).abs()
        weather_audit[str(year)] = {
            "ppo_daily_rows": len(raw), "date_matched_rows": len(weather),
            "rain_mm_max_abs_diff": float(rain_diff.max()) if len(weather) else None,
            "tmax_c_max_abs_diff": float(tmax_diff.max()) if len(weather) else None,
            "tmin_c_max_abs_diff": float(tmin_diff.max()) if len(weather) else None,
            "matches_frozen_baseline_weather": bool(
                len(weather) == len(raw) and len(weather) > 0
                and rain_diff.max() <= 1e-6 and tmax_diff.max() <= 0.051 and tmin_diff.max() <= 0.051
            ),
        }
        metrics.append({
            "site": SITE, "station_code": STATION, "seed": seed, "year": year,
            "checkpoint_step": 100000, "scenario": scenario, "scenario_label": scenario_label,
            "grain_yield_kg_ha": final_yield, "total_irrigation_mm": measured["actual_irrigation_mm"],
            "total_n_kg_ha": measured["actual_nitrogen_kg_ha"], "etcp_mm": measured["etcp_mm"],
            "WP_ET_kg_m3": measured["WP_ET_kg_m3"], "PFP_N_kg_grain_per_kg_N": measured["PFP_N_kg_kg"],
            "NUE": math.nan, "WUE": measured["WP_ET_kg_m3"], "plant_n_uptake_kg_ha": math.nan,
            "max_wspd": float(pd.to_numeric(raw.water_stress_index_wspd, errors="coerce").max()),
            "max_nstd": float(pd.to_numeric(raw.nitrogen_stress_index_nstd, errors="coerce").max()),
            "common_reward": float(raw.common_cumulative_reward.iloc[-1]),
            "etcp_source": "exact Summary.OUT ETCP", "snapshot_path": snapshot.relative_to(ROOT).as_posix(),
            "model_sha256": model_hash,
        })
        for row in raw.itertuples():
            ppo_actions.append({
                "site": SITE, "station_code": STATION, "seed": seed, "year": year,
                "date": row.date, "doy": row.doy, "dap": row.dap,
                "discrete_action_index": None, "requested_discrete_action_index": None,
                "mask_forced_noop": "not exported by DSSAT snapshot", "irrigation_mm": row.irrigation_executed_mm,
                "nitrogen_kg_ha": row.nitrogen_executed_kg_ha, "scenario": scenario,
                "source_daily_path": snapshot.relative_to(ROOT).as_posix(),
            })
        print(f"DONE seed{seed} FQA{year}: yield={final_yield:.1f}, I={measured['actual_irrigation_mm']:.1f}, N={measured['actual_nitrogen_kg_ha']:.1f}, WP_ET={measured['WP_ET_kg_m3']:.3f}", flush=True)

    if not all(item["matches_frozen_baseline_weather"] for item in weather_audit.values()):
        raise RuntimeError(f"Validation weather differs from frozen baseline: {weather_audit}")
    baseline_metric_rows: list[dict[str, Any]] = []
    for row in baseline_summary.to_dict(orient="records"):
        baseline_metric_rows.append({
            "site": SITE, "station_code": STATION, "seed": seed, "year": int(row["year"]),
            "checkpoint_step": 100000, "scenario": str(row["scenario"]),
            "scenario_label": BASELINE_LABELS[str(row["scenario"])],
            "grain_yield_kg_ha": float(row["grain_yield_kg_ha"]),
            "total_irrigation_mm": float(row["actual_irrigation_mm"]),
            "total_n_kg_ha": float(row["actual_nitrogen_kg_ha"]), "etcp_mm": float(row["etcp_mm"]),
            "WP_ET_kg_m3": float(row["WP_ET_kg_m3"]),
            "PFP_N_kg_grain_per_kg_N": float(row["PFP_N_kg_kg"]) if str(row["PFP_N_kg_kg"]) not in ("", "nan") else math.nan,
            "NUE": math.nan, "WUE": float(row["WP_ET_kg_m3"]), "plant_n_uptake_kg_ha": math.nan,
            "common_reward": float(row["unified_reward"]), "source_status": "frozen 051_03 four-baseline Summary.OUT/ETCP",
        })

    base_daily = baseline_daily.copy()
    base_daily["scenario_label"] = base_daily.scenario.map(BASELINE_LABELS)
    base_daily["water_stress"] = pd.to_numeric(base_daily.water_stress_index_wspd, errors="coerce")
    base_daily["water_stress_source"] = "WSPD"
    base_daily["nitrogen_stress"] = pd.to_numeric(base_daily.nitrogen_stress_index_nstd, errors="coerce")
    base_daily["nitrogen_stress_source"] = "NSTD"
    base_daily["common_cumulative_reward"] = pd.to_numeric(base_daily.unified_cumulative_reward, errors="coerce")
    base_daily["site"] = SITE
    base_daily["seed"] = seed
    ppo_daily_frame = pd.concat(ppo_daily, ignore_index=True)
    daily_frame = pd.concat([base_daily, ppo_daily_frame], ignore_index=True, sort=False)
    metric_frame = pd.DataFrame([*baseline_metric_rows, *metrics]).sort_values(["year", "scenario"]).reset_index(drop=True)
    unique = len({json.dumps(seq, separators=(",", ":")) for seq in yearly_sequences.values()})
    action_audit = {
        "site": SITE, "seed": seed, "years": YEARS,
        "sequence_definition": "ordered positive irrigation/N event sequence [DAP, irrigation_mm, nitrogen_kg_ha] reconstructed from exact MgmtEvent.OUT",
        "identical_across_all_validation_years": unique == 1,
        "unique_yearly_action_sequence_count": unique,
        "yearly_action_sequences": yearly_sequences,
        "weather_input_match_by_calendar_date": weather_audit,
        "weather_match_all_years": True,
        "exact_etcp_replay": True,
        "evaluation_seed_fixed_across_ppo_seeds": EVAL_SEED,
    }
    return metric_frame, daily_frame, pd.DataFrame(ppo_actions), action_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=range(8), required=True)
    args = parser.parse_args()
    if not BASELINE_SUMMARY.is_file() or not BASELINE_DAILY.is_file():
        raise FileNotFoundError("Frozen FQ five-scenario baseline tables are missing")
    VALIDATION.mkdir(parents=True, exist_ok=True)
    config, env_config = configure_eval()
    checkpoint = model_path(args.seed)
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(str(checkpoint), device="cpu")
    metrics, daily, actions, audit = build_seed_frames(args.seed, model, checkpoint, config, env_config)
    seed_dir = FIGURE_ROOT / f"best_seed_seed{args.seed}"
    figures_dir, tables_dir = seed_dir / "figures", seed_dir / "tables"
    if seed_dir.exists() and any(seed_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing figure archive: {seed_dir}")
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots.write_table_files("FQ", args.seed, metrics, tables_dir, audit)
    plots.write_event_and_daily_tables("FQ", args.seed, daily, actions, tables_dir)
    plots.plot_metric_bars("FQ", args.seed, metrics, figures_dir)
    plots.plot_management_bars("FQ", args.seed, metrics, figures_dir)
    plots.plot_reward("FQ", args.seed, metrics, figures_dir)
    for year in YEARS:
        plots.plot_daily_year("FQ", args.seed, year, daily, figures_dir, True)
        plots.plot_management_year("FQ", args.seed, year, metrics, figures_dir)
    plots.write_seed_readme("FQ", args.seed, metrics, audit, seed_dir)
    seed_readme = seed_dir / "README.md"
    text = seed_readme.read_text(encoding="utf-8")
    text = text.replace(
        "- WP_ET / WUE: unavailable for PPO because the selected PPO validation artifacts do not contain an exact Summary.OUT + ETCP replay.",
        "- PPO WP_ET / WUE uses endpoint-matched Summary.OUT ETCP from deterministic 2014–2023 DSSAT replay; plant N uptake is unavailable, so NUE is not inferred.",
    ).replace(
        "- Daily process figures show baseline WSPD/NSTD alongside PPO SWFAC/NSTRES; these are the source-specific stress indicators. PPO daily soil water was not recorded.",
        "- Daily process figures show WSPD/NSTD from PlantGro.OUT for PPO and the frozen baselines; soil water comes from SoilWat.OUT.",
    ).replace(
        "- Weather was cross-checked against the matching site/year Null baseline by calendar date (rainfall exact; temperature tolerance 0.051°C for one-decimal baseline rounding); see the action audit JSON.",
        "- Validation weather was matched to the same-year frozen Null baseline by calendar date; rainfall tolerance is 1e-6 mm and temperature tolerance is 0.051°C.",
    )
    seed_readme.write_text(text, encoding="utf-8")
    print(json.dumps({
        "seed": args.seed, "model": checkpoint.relative_to(ROOT).as_posix(),
        "model_sha256": sha256(checkpoint), "validation_years": YEARS,
        "weather_match_all_years": audit["weather_match_all_years"],
        "exact_etcp_rows": len(metrics.loc[metrics.scenario.eq(plots.scenario_key(args.seed))]),
        "figure_count": len(list(figures_dir.glob("*.png"))),
        "output": seed_dir.relative_to(ROOT).as_posix(),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
