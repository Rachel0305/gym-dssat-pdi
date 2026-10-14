"""096_00: FQ ncost=2.5 25K checkpoint and native-auto five-scenario replay.

This is an isolated continuation of the 095 2K smoke.  It never modifies the
old FQ runners or inputs.  The original 2018 weather is used for the training
endpoint oracle; an isolated corrected-weather 2018 replay is reported
separately and is used for the paper five-scenario table.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine
import run_multisite_input_ic1_four_baseline_rebuild_034_00 as baseline
from build_relaxed_success_five_scenario_daily_evidence_027_05 import parse_management_events, parse_table
from run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 import snapshot_from_env


_FIGURES_SPEC = importlib.util.spec_from_file_location(
    "fqa_figures_051_03",
    ROOT / "src/051_fqa_originIC_site_transfer/run_051_03_fqa_originIC_five_scenario_figures.py",
)
if _FIGURES_SPEC is None or _FIGURES_SPEC.loader is None:
    raise ImportError("cannot load 051_03 FQ figure/replay module")
figures = importlib.util.module_from_spec(_FIGURES_SPEC)
_FIGURES_SPEC.loader.exec_module(figures)


CONFIG = ROOT / "configs/096_00_fqa_reward_ncost25_ckpt25k_five_scenario.json"
PROMPT = ROOT / "prompts/096_00_fqa_reward_ncost25_ckpt25k_five_scenario.md"
BASE_YAML = ROOT / "experiments/ppo_observed_years/config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
STATION, SITE = "FQA", "FQ"
LEVELS_I = [0.0, 15.0, 30.0, 45.0]
LEVELS_N = [0.0, 40.0, 80.0, 120.0]
PPO_N_COST = 2.5
BASELINE_N_COST = 1.58


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def patch_training_config() -> tuple[Path, dict[str, Any]]:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    effective = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))
    effective["ppo"].update(cfg["fixed_ppo"])
    effective["reward"].update(cfg["reward_override"])
    effective_path = ROOT / "configs/096_00_effective_reward_ncost25_ckpt25k_ppo_config.yaml"
    effective_path.write_text(yaml.safe_dump(effective, sort_keys=False), encoding="utf-8")
    return effective_path, cfg


def action_signature(daily: pd.DataFrame) -> str:
    irrigation = pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0.0)
    nitrogen = pd.to_numeric(daily["safe_action_anfer"], errors="coerce").fillna(0.0)
    return "; ".join(sorted({f"I{float(i):g}/N{float(n):g}" for i, n in zip(irrigation, nitrogen) if i > 1e-9 or n > 1e-9}))


def audit_25k(summary: pd.DataFrame, out: Path, checkpoint: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in summary.itertuples(index=False):
        daily_path = ROOT / str(rec.daily_csv_path)
        daily = pd.read_csv(daily_path, keep_default_na=False)
        irrigation = pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0.0)
        nitrogen = pd.to_numeric(daily["safe_action_anfer"], errors="coerce").fillna(0.0)
        dap = pd.to_numeric(daily["dap"], errors="coerce").fillna(0.0)
        previous_i = pd.to_numeric(daily["previous_cumulative_irrigation"], errors="coerce")
        previous_n = pd.to_numeric(daily["previous_cumulative_n"], errors="coerce")
        cumulative_i = pd.to_numeric(daily["season_cumulative_irrigation"], errors="coerce")
        cumulative_n = pd.to_numeric(daily["season_cumulative_n"], errors="coerce")
        positive = (irrigation > 1e-9) | (nitrogen > 1e-9)
        raw_i = pd.to_numeric(daily["raw_action_amir"], errors="coerce")
        raw_n = pd.to_numeric(daily["raw_action_anfer"], errors="coerce")
        transmitted = (~positive) | (
            np.isclose(cumulative_i - previous_i, irrigation, atol=1e-6)
            & np.isclose(cumulative_n - previous_n, nitrogen, atol=1e-6)
        )
        off_grid = (
            ~np.isclose(irrigation.to_numpy()[:, None], np.asarray(LEVELS_I)[None, :], atol=1e-6).any(axis=1)
        ).sum() + (
            ~np.isclose(nitrogen.to_numpy()[:, None], np.asarray(LEVELS_N)[None, :], atol=1e-6).any(axis=1)
        ).sum()
        labels = [f"I{float(i):g}/N{float(n):g}" for i, n in zip(irrigation[positive], nitrogen[positive])]
        rows.append(
            {
                "checkpoint_step": int(rec.checkpoint_step),
                "year": int(rec.year),
                "daily_path": str(rec.daily_csv_path),
                "final_grnwt": float(rec.final_grnwt),
                "total_irrigation": float(rec.total_irrigation),
                "total_n": float(rec.total_n),
                "action_signature": action_signature(daily),
                "positive_action_pair_count": int(len(set(labels))),
                "positive_action_rows_after_dap1": int((positive & (dap > 1)).sum()),
                "raw_to_safe_mismatch_rows": int((~np.isclose(raw_i, irrigation, atol=1e-6) | ~np.isclose(raw_n, nitrogen, atol=1e-6)).sum()),
                "safe_to_dssat_mismatch_rows": int((positive & ~transmitted).sum()),
                "off_grid_rows": int(off_grid),
            }
        )
    audit = pd.DataFrame(rows).sort_values(["checkpoint_step", "year"]).reset_index(drop=True)
    final = audit[audit["checkpoint_step"].eq(checkpoint)].copy()
    non_2018 = final[final["year"].ne(2018)]
    signature_counts = final["action_signature"].value_counts()
    all_pairs = set("; ".join(final["action_signature"]).split("; ")) - {""}
    gate = {
        "validation_rows": int(len(final)),
        "expected_validation_rows": 10,
        "raw_to_safe_mismatch_rows": int(final["raw_to_safe_mismatch_rows"].sum()),
        "safe_to_dssat_mismatch_rows": int(final["safe_to_dssat_mismatch_rows"].sum()),
        "off_grid_rows": int(final["off_grid_rows"].sum()),
        "nonzero_action_pair_count": int(len(all_pairs)),
        "positive_action_rows_after_dap1": int(final["positive_action_rows_after_dap1"].sum()),
        "cross_year_distinct_signatures": int(final["action_signature"].nunique()),
        "minority_signature_year_count": int(signature_counts.iloc[-1]) if len(signature_counts) > 1 else 0,
        "non_2018_cross_year_distinct_signatures": int(non_2018["action_signature"].nunique()),
    }
    gate["mechanism_pass"] = bool(
        gate["validation_rows"] == 10
        and gate["raw_to_safe_mismatch_rows"] == 0
        and gate["safe_to_dssat_mismatch_rows"] == 0
        and gate["off_grid_rows"] == 0
        and gate["nonzero_action_pair_count"] >= 3
        and gate["positive_action_rows_after_dap1"] > 0
        and gate["cross_year_distinct_signatures"] > 1
        and gate["minority_signature_year_count"] >= 2
        and gate["non_2018_cross_year_distinct_signatures"] > 1
    )
    audit.to_csv(out / "audits/096_00_action_audit.csv", index=False, encoding="utf-8-sig")
    return audit, gate


def observed_kwargs(model_path: Path) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(str(model_path), device="cpu")
    return {
        "learning_rate": model.learning_rate,
        "gamma": model.gamma,
        "gae_lambda": model.gae_lambda,
        "n_steps": model.n_steps,
        "batch_size": model.batch_size,
        "n_epochs": model.n_epochs,
        "ent_coef": model.ent_coef,
        "clip_range": model.clip_range,
        "policy_kwargs": model.policy_kwargs,
    }


def state_snapshot() -> dict[str, Any]:
    return {
        "engine": {key: getattr(engine, key) for key in ["TASK_ID", "TASK_NAME", "BASE_OUT", "BASE_DOC", "PROMPT", "LOWIC_INPUT_ROOT", "STATION", "SITES", "BINARY_IRRIGATION_LEVELS", "BINARY_NITROGEN_LEVELS"]},
        "batch": {key: getattr(batch, key) for key in ["CONFIG", "OUT", "DOC", "PROMPT", "SITES", "TOTAL_TIMESTEPS", "CHECKPOINT_STEPS"]},
        "site_names": dict(batch.SITE_NAMES),
        "input_root": ppo_safe_rendering.MULTISITE_INPUT_ROOT,
        "load_config": batch.load_config,
    }


def restore_state(old: dict[str, Any]) -> None:
    for key, value in old["engine"].items():
        setattr(engine, key, value)
    for key, value in old["batch"].items():
        setattr(batch, key, value)
    batch.SITE_NAMES.clear()
    batch.SITE_NAMES.update(old["site_names"])
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = old["input_root"]
    batch.load_config = old["load_config"]


def replay_candidate(
    cfg: dict[str, Any],
    out: Path,
    ppo_root: Path,
    effective_yaml: Path,
    checkpoint: int,
    year: int,
    input_root: Path,
    variant: str,
    strict_endpoint: bool,
) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks

    inventory = pd.read_csv(out / "evaluation/096_00_training_checkpoint_inventory.csv", keep_default_na=False)
    validation = pd.read_csv(out / "evaluation/096_00_checkpoint_validation_summary.csv", keep_default_na=False)
    model_row = inventory[pd.to_numeric(inventory["checkpoint_step"], errors="coerce").eq(checkpoint)]
    expected = validation[
        pd.to_numeric(validation["checkpoint_step"], errors="coerce").eq(checkpoint)
        & pd.to_numeric(validation["year"], errors="coerce").eq(year)
    ]
    if len(model_row) != 1 or len(expected) != 1:
        raise RuntimeError(f"missing model/validation row for checkpoint={checkpoint}, year={year}")
    model_path = ROOT / str(model_row.iloc[0]["model_path"])
    target = out / "snapshots" / STATION / str(year) / variant
    if target.exists():
        raise FileExistsError(target)

    old = state_snapshot()
    env = None
    started = time.time()
    try:
        engine.TASK_ID = cfg["task_id"]
        engine.TASK_NAME = cfg["task_name"]
        engine.BASE_OUT = ppo_root
        engine.BASE_DOC = ROOT / "docs/096_00_fqa_reward_ncost25_ckpt25k_five_scenario_record.md"
        engine.PROMPT = PROMPT
        engine.LOWIC_INPUT_ROOT = input_root
        engine.STATION = STATION
        engine.SITES = [STATION]
        engine.BINARY_IRRIGATION_LEVELS = list(LEVELS_I)
        engine.BINARY_NITROGEN_LEVELS = list(LEVELS_N)
        batch.SITE_NAMES[STATION] = SITE
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = input_root
        engine.patch_base_module(ppo_root, engine.BASE_DOC, int(cfg["training"]["total_timesteps"]), [checkpoint])
        # Preserve the effective reward config for provenance; replay dynamics do
        # not use the reward value, but actions and safety must use the same grid.
        batch.CONFIG = effective_yaml
        config = batch.load_config()
        selection = batch.build_selection(batch.load_split())
        env_config = direct_ppo.build_env_config(config, selection)
        env_config["paths"]["output_root"] = rel(ppo_root)
        model = MaskablePPO.load(str(model_path), device="cpu")
        env = batch.base.make_env(config, env_config, STATION, int(year), int(cfg["seed"]), f"FQA_{year}_096_{variant}", evaluation=True)
        obs, _info = env.reset()
        done, steps = False, 0
        action_records: list[dict[str, Any]] = []
        while not done and steps < 260:
            action, _ = model.predict(obs, action_masks=get_action_masks(env), deterministic=True)
            obs, reward, terminated, truncated, _info = env.step(action)
            action_info = dict(getattr(env, "last_action_info", {}))
            action_records.append(
                {
                    "dap": int(round(float(action_info.get("dap", steps + 1)))),
                    "safe_action_amir": float(action_info.get("safe_action_amir", 0.0)),
                    "safe_action_anfer": float(action_info.get("safe_action_anfer", 0.0)),
                    "reward": float(reward),
                }
            )
            done = bool(terminated or truncated)
            steps += 1
        if not done:
            raise RuntimeError(f"PPO replay did not finish within 260 steps for FQA{year}")
        snapshot_tmp = snapshot_from_env(env)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(snapshot_tmp, target)
        daily = figures.daily_from_snapshot(target, year, "ppo")
        replay_yield = float(pd.to_numeric(daily["grain_yield_kg_ha"], errors="coerce").iloc[-1])
        saved_yield = float(pd.to_numeric(expected.iloc[0]["final_grnwt"], errors="coerce"))
        endpoint_delta = replay_yield - saved_yield
        if strict_endpoint and not np.isclose(replay_yield, saved_yield, rtol=0.0, atol=0.5):
            raise RuntimeError(
                f"PPO replay endpoint mismatch for FQA{year}: saved={saved_yield:.4f}, replay={replay_yield:.4f}"
            )
        action_path = out / "actions" / f"FQA_{year}_{variant}_ckpt{checkpoint}_daily.csv"
        action_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(action_records).to_csv(action_path, index=False, encoding="utf-8-sig")
        return {
            "year": int(year),
            "checkpoint_step": int(checkpoint),
            "variant": variant,
            "input_root": rel(input_root),
            "input_wth_sha256": sha256(input_root / "FQ" / "CNFQ1801.WTH"),
            "model_path": rel(model_path),
            "saved_training_yield": saved_yield,
            "replay_yield": replay_yield,
            "endpoint_delta": endpoint_delta,
            "strict_endpoint_checked": strict_endpoint,
            "action_signature": action_signature(pd.DataFrame(action_records)),
            "total_irrigation_mm": float(pd.to_numeric(daily["irrigation_executed_mm"], errors="coerce").sum()),
            "total_n_kg_ha": float(pd.to_numeric(daily["nitrogen_executed_kg_ha"], errors="coerce").sum()),
            "snapshot_path": rel(target),
            "action_path": rel(action_path),
            "elapsed_seconds": time.time() - started,
        }
    finally:
        if env is not None:
            env.close()
        restore_state(old)


def existing_replay_row(
    cfg: dict[str, Any],
    out: Path,
    checkpoint: int,
    year: int,
    input_root: Path,
    variant: str,
    strict_endpoint: bool,
) -> dict[str, Any]:
    """Finalize already-rendered replay snapshots without retraining or DSSAT."""
    inventory = pd.read_csv(out / "evaluation/096_00_training_checkpoint_inventory.csv", keep_default_na=False)
    validation = pd.read_csv(out / "evaluation/096_00_checkpoint_validation_summary.csv", keep_default_na=False)
    model_row = inventory[pd.to_numeric(inventory["checkpoint_step"], errors="coerce").eq(checkpoint)]
    expected = validation[
        pd.to_numeric(validation["checkpoint_step"], errors="coerce").eq(checkpoint)
        & pd.to_numeric(validation["year"], errors="coerce").eq(year)
    ]
    if len(model_row) != 1 or len(expected) != 1:
        raise RuntimeError(f"missing existing model/validation row for checkpoint={checkpoint}, year={year}")
    target = out / "snapshots" / STATION / str(year) / variant
    action_path = out / "actions" / f"FQA_{year}_{variant}_ckpt{checkpoint}_daily.csv"
    if not (target / "Summary.OUT").exists() or not action_path.exists():
        raise FileNotFoundError(f"incomplete existing replay for FQA{year}/{variant}")
    daily = figures.daily_from_snapshot(target, year, "ppo")
    actions = pd.read_csv(action_path, keep_default_na=False)
    replay_yield = float(pd.to_numeric(daily["grain_yield_kg_ha"], errors="coerce").iloc[-1])
    saved_yield = float(pd.to_numeric(expected.iloc[0]["final_grnwt"], errors="coerce"))
    endpoint_delta = replay_yield - saved_yield
    if strict_endpoint and not np.isclose(replay_yield, saved_yield, rtol=0.0, atol=0.5):
        raise RuntimeError(f"existing replay endpoint mismatch for FQA{year}: saved={saved_yield:.4f}, replay={replay_yield:.4f}")
    return {
        "year": int(year),
        "checkpoint_step": int(checkpoint),
        "variant": variant,
        "input_root": rel(input_root),
        "input_wth_sha256": sha256(input_root / "FQ" / "CNFQ1801.WTH"),
        "model_path": rel(ROOT / str(model_row.iloc[0]["model_path"])),
        "saved_training_yield": saved_yield,
        "replay_yield": replay_yield,
        "endpoint_delta": endpoint_delta,
        "strict_endpoint_checked": strict_endpoint,
        "action_signature": action_signature(actions),
        "total_irrigation_mm": float(pd.to_numeric(daily["irrigation_executed_mm"], errors="coerce").sum()),
        "total_n_kg_ha": float(pd.to_numeric(daily["nitrogen_executed_kg_ha"], errors="coerce").sum()),
        "snapshot_path": rel(target),
        "action_path": rel(action_path),
        "elapsed_seconds": 0.0,
    }


def add_reward_columns(daily: pd.DataFrame) -> pd.DataFrame:
    result = daily.copy()
    result["nitrogen_cost_used"] = np.where(result["scenario"].eq("ppo"), PPO_N_COST, BASELINE_N_COST)
    result["unified_reward_step"] = -1.1 * result["irrigation_executed_mm"] - result["nitrogen_cost_used"] * result["nitrogen_executed_kg_ha"]
    for (_, _), idx in result.groupby(["year", "scenario"]).groups.items():
        positions = list(idx)
        final_yield = float(pd.to_numeric(result.loc[positions, "grain_yield_kg_ha"], errors="coerce").iloc[-1])
        result.loc[positions[-1], "unified_reward_step"] += 0.158 * final_yield
    result["unified_cumulative_reward"] = result.groupby(["year", "scenario"])["unified_reward_step"].cumsum()
    return result


def build_five_scenario_tables(cfg: dict[str, Any], out: Path, replay_rows: list[dict[str, Any]]) -> dict[str, Any]:
    years = list(map(int, cfg["scope"]["validation_years"]))
    replay_map = {(int(row["year"]), row["variant"]): ROOT / row["snapshot_path"] for row in replay_rows}
    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for year in years:
        baseline_root = ROOT / cfg["input_roots"]["baseline_corrected_2018"] if year == 2018 else ROOT / cfg["input_roots"]["baseline_original"]
        scenario_paths = {
            "null": baseline_root / "snapshots" / STATION / str(year) / "null",
            "recorded_farmer_template": baseline_root / "snapshots" / STATION / str(year) / "recorded_farmer_template",
            "dssat_auto": baseline_root / "snapshots" / STATION / str(year) / "dssat_auto",
            "official_extension_expert": baseline_root / "snapshots" / STATION / str(year) / "official_extension_expert",
            "ppo": replay_map[(year, "corrected2018" if year == 2018 else "original")],
        }
        missing = [rel(path) for path in scenario_paths.values() if not (path / "Summary.OUT").exists()]
        if missing:
            raise FileNotFoundError(f"missing five-scenario snapshots for {year}: {missing}")
        year_frames: list[pd.DataFrame] = []
        for scenario, snapshot in scenario_paths.items():
            year_frames.append(figures.daily_from_snapshot(snapshot, year, scenario))
        merged = add_reward_columns(pd.concat(year_frames, ignore_index=True))
        frames.append(merged)
        for scenario, sub in merged.groupby("scenario"):
            last = sub.iloc[-1]
            metrics = baseline.metrics_from_snapshot(scenario_paths[scenario], float(last["grain_yield_kg_ha"]))
            n_total = float(metrics["actual_nitrogen_kg_ha"])
            summaries.append(
                {
                    "station_code": STATION,
                    "site": SITE,
                    "year": int(year),
                    "scenario": scenario,
                    "grain_yield_kg_ha": float(last["grain_yield_kg_ha"]),
                    "biomass_kg_ha": float(last["biomass_kg_ha"]),
                    "actual_irrigation_mm": float(metrics["actual_irrigation_mm"]),
                    "actual_nitrogen_kg_ha": n_total,
                    "etcp_mm": float(metrics["etcp_mm"]),
                    "WP_ET_kg_m3": float(metrics["WP_ET_kg_m3"]),
                    "PFP_N_kg_kg": float(metrics["PFP_N_kg_kg"]) if pd.notna(metrics["PFP_N_kg_kg"]) and n_total > 0 else np.nan,
                    "max_wspd": float(pd.to_numeric(sub["water_stress_index_wspd"], errors="coerce").max()),
                    "max_nstd": float(pd.to_numeric(sub["nitrogen_stress_index_nstd"], errors="coerce").max()),
                    "snapshot_path": rel(scenario_paths[scenario]),
                }
            )
    all_daily = pd.concat(frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    action_events = all_daily[(all_daily["irrigation_executed_mm"].gt(0)) | (all_daily["nitrogen_executed_kg_ha"].gt(0))].copy()
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    all_daily.to_csv(tables / "096_00_fqa_five_scenario_daily.csv", index=False, encoding="utf-8-sig")
    action_events.to_csv(tables / "096_00_fqa_five_scenario_management_events.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(tables / "096_00_fqa_five_scenario_season_summary.csv", index=False, encoding="utf-8-sig")

    gaps: list[dict[str, Any]] = []
    for year, subset in summary.groupby("year"):
        candidate = subset[subset["scenario"].eq("ppo")].iloc[0]
        four = subset[~subset["scenario"].eq("ppo")]
        for metric in ["grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg"]:
            maximum = pd.to_numeric(four[metric], errors="coerce").max(skipna=True)
            value = float(candidate[metric]) if pd.notna(candidate[metric]) else np.nan
            gaps.append({"year": int(year), "metric": metric, "ppo_value": value, "four_baseline_max": maximum, "ppo_minus_four_max": value - maximum if pd.notna(maximum) else np.nan})
    gaps_df = pd.DataFrame(gaps)
    gaps_df.to_csv(tables / "096_00_fqa_metric_gaps_vs_four_native_baselines.csv", index=False, encoding="utf-8-sig")
    wins = gaps_df.groupby("metric")["ppo_minus_four_max"].apply(lambda values: int((pd.to_numeric(values, errors="coerce") > 0).sum())).to_dict()
    means = summary.groupby("scenario")[["grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg", "actual_irrigation_mm", "actual_nitrogen_kg_ha"]].mean(numeric_only=True).reset_index()
    means.to_csv(tables / "096_00_fqa_five_scenario_site_means.csv", index=False, encoding="utf-8-sig")
    return {"rows": int(len(summary)), "years": years, "scenarios": sorted(summary["scenario"].unique().tolist()), "ppo_win_counts": wins, "site_means": means.to_dict(orient="records"), "daily_table": rel(tables / "096_00_fqa_five_scenario_daily.csv"), "season_summary": rel(tables / "096_00_fqa_five_scenario_season_summary.csv"), "management_events": rel(tables / "096_00_fqa_five_scenario_management_events.csv"), "gaps": rel(tables / "096_00_fqa_metric_gaps_vs_four_native_baselines.csv")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true", help="finish files from a completed training/replay without rerunning DSSAT")
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    out = ROOT / cfg["output_root"]
    original_input = ROOT / cfg["input_roots"]["original"]
    corrected_input = ROOT / cfg["input_roots"]["corrected_2018"]
    baseline_original = ROOT / cfg["input_roots"]["baseline_original"]
    baseline_corrected = ROOT / cfg["input_roots"]["baseline_corrected_2018"]
    if args.dry_run:
        print(json.dumps({"task_id": cfg["task_id"], "station": STATION, "training_steps": cfg["training"]["total_timesteps"], "checkpoint_steps": cfg["training"]["checkpoint_steps"], "reward_override": cfg["reward_override"], "paper_scenarios": cfg["scope"]["paper_scenarios"], "external_n_allowed": False, "weather_forecast_allowed": False, "original_input_exists": original_input.exists(), "corrected_2018_exists": (corrected_input / "FQ/CNFQ1801.WTH").exists(), "native_baseline_exists": (baseline_original / "snapshots/FQA/2014/dssat_auto/Summary.OUT").exists(), "prompt_exists": PROMPT.exists()}, ensure_ascii=False, indent=2))
        return
    required = [original_input, corrected_input / "FQ/CNFQ1801.WTH", baseline_original, baseline_corrected]
    missing = [rel(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    if out.exists() and not args.finalize_existing:
        raise FileExistsError(f"refusing to overwrite {out}")
    if not args.finalize_existing:
        (out / "configs").mkdir(parents=True, exist_ok=False)
        (out / "audits").mkdir(parents=True, exist_ok=True)
        shutil.copy2(CONFIG, out / "configs" / CONFIG.name)
        shutil.copy2(PROMPT, out / "configs" / PROMPT.name)
    effective_yaml, cfg = patch_training_config()
    started = time.time()
    if not args.finalize_existing:
        old = state_snapshot()
        try:
            engine.TASK_ID = cfg["task_id"]
            engine.TASK_NAME = cfg["task_name"]
            engine.BASE_OUT = out
            engine.BASE_DOC = ROOT / "docs/096_00_fqa_reward_ncost25_ckpt25k_five_scenario_record.md"
            engine.PROMPT = PROMPT
            engine.LOWIC_INPUT_ROOT = original_input
            engine.STATION = STATION
            engine.SITES = [STATION]
            engine.BINARY_IRRIGATION_LEVELS = list(LEVELS_I)
            engine.BINARY_NITROGEN_LEVELS = list(LEVELS_N)
            batch.CONFIG = effective_yaml
            batch.OUT = out
            batch.DOC = engine.BASE_DOC
            batch.SITE_NAMES[STATION] = SITE
            ppo_safe_rendering.MULTISITE_INPUT_ROOT = original_input
            engine.run_training(int(cfg["training"]["total_timesteps"]), list(map(int, cfg["training"]["checkpoint_steps"])), suffix="")
        finally:
            restore_state(old)

    source_summary = out / "evaluation/042_10_checkpoint_validation_summary.csv"
    source_inventory = out / "evaluation/042_10_training_checkpoint_inventory.csv"
    if not source_summary.exists() or not source_inventory.exists():
        raise FileNotFoundError("training did not produce checkpoint summary and inventory")
    summary = pd.read_csv(source_summary, keep_default_na=False)
    inventory = pd.read_csv(source_inventory, keep_default_na=False)
    summary.to_csv(out / "evaluation/096_00_checkpoint_validation_summary.csv", index=False, encoding="utf-8-sig")
    inventory.to_csv(out / "evaluation/096_00_training_checkpoint_inventory.csv", index=False, encoding="utf-8-sig")
    _, gate = audit_25k(summary, out, 25000)
    model_path = ROOT / str(inventory[pd.to_numeric(inventory["checkpoint_step"], errors="coerce").eq(25000)].iloc[0]["model_path"])
    observed = observed_kwargs(model_path)

    (out / "tables").mkdir(parents=True, exist_ok=True)
    replay_rows: list[dict[str, Any]] = []
    years = list(map(int, cfg["scope"]["validation_years"]))
    if args.finalize_existing:
        # The prior process completed all DSSAT replays; only reconstruct their
        # manifests and tables here.  This branch never calls make_env/DSSAT.
        for year in years:
            replay_rows.append(existing_replay_row(cfg, out, 25000, year, original_input, "original", True))
        replay_rows.append(existing_replay_row(cfg, out, 25000, 2018, corrected_input, "corrected2018", False))
    else:
        # Raw input is the endpoint oracle for all years, including raw 2018.
        for year in years:
            replay_rows.append(replay_candidate(cfg, out, out, effective_yaml, 25000, year, original_input, "original", True))
        # Corrected 2018 is a separate provenance replay and is never compared to
        # the raw-weather zero endpoint.
        replay_rows.append(replay_candidate(cfg, out, out, effective_yaml, 25000, 2018, corrected_input, "corrected2018", False))
    pd.DataFrame(replay_rows).to_csv(out / "tables/096_00_ppo_replay_provenance.csv", index=False, encoding="utf-8-sig")
    five = build_five_scenario_tables(cfg, out, replay_rows)
    raw_2018 = next(row for row in replay_rows if row["variant"] == "original" and row["year"] == 2018)
    corrected_2018 = next(row for row in replay_rows if row["variant"] == "corrected2018")
    result = {
        "task_id": cfg["task_id"],
        "status": "completed_25k_five_scenario_replay",
        "training_run": True,
        "external_n_included": False,
        "weather_forecast_included": False,
        "checkpoint": 25000,
        "effective_yaml": rel(effective_yaml),
        "effective_kwargs_from_loaded_model": observed,
        "reward_override": cfg["reward_override"],
        "action_gate_25k": gate,
        "raw_2018_endpoint_oracle": raw_2018,
        "corrected_2018_provenance_replay": corrected_2018,
        "five_scenario": five,
        "output_root": rel(out),
        "elapsed_seconds": time.time() - started,
        "next_step_allowed": False,
        "next_step_note": "25K only; no automatic 50K/75K/100K. Corrected 2018 remains an isolated input provenance replacement.",
    }
    (out / "096_00_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "096_00_manifest.json").write_text(json.dumps({"config": cfg, "effective_kwargs": observed, "result": result}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = [
        "# 096_00 FQ ncost=2.5 25K checkpoint and native-auto five-scenario replay",
        "",
        "- 唯一 reward 干预：`nitrogen_cost 1.58 -> 2.5`。",
        "- 论文五情景为 `null`、`recorded_farmer_template`、`dssat_auto`（DSSAT 原生自动灌溉）、`official_extension_expert` 和 PPO；不含 external-N。",
        "- 原始 2018 replay 用于训练 endpoint oracle；修正 2018 replay 单独记录，不与原始 0 产量作等值比较。",
        "",
        "## 25K action gate",
        "",
    ]
    lines.extend(f"- `{key}`: `{value}`" for key, value in gate.items())
    lines.extend(["", "## Five-scenario output", "", f"- `{five}`", "", "- 不自动启动 50K/75K/100K。", ""])
    (ROOT / "docs/096_00_fqa_reward_ncost25_ckpt25k_five_scenario_record.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
