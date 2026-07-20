#!/usr/bin/env python3
"""Frozen HLA2010 MaskablePPO transfer to four screened HLA years."""

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
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_extension_expert_baseline_018_03 as extension
import run_hla_five_scenario_completion_020_11 as hla
import run_sy_all_authoritative_years_frozen_stage_ppo_026_06 as metrics_source
from run_hla2010_stage_ppo_readiness_scaler_smoke_027_01 import HLAStageEnv027


OUT = ROOT / "benchmark_results" / "028_08_hla_frozen_maskableppo_screened_year_transfer"
SMOKE_OUT = ROOT / "benchmark_results" / "028_08_smoke_hla2007_seed0"
DOC = ROOT / "docs" / "2026-07-18_028_08_hla_frozen_maskableppo_screened_year_transfer.md"
YEARS = (2007, 2015, 2016, 2022)
SCALER = ROOT / "benchmark_results" / "027_01_attempt2" / "027_01_hla2010_observation_scaler.csv"
MODELS = {
    0: ROOT / "benchmark_results" / "027_02_attempt2" / "checkpoint_000180.zip",
    1: ROOT / "benchmark_results" / "027_03" / "seed1" / "checkpoint_000060.zip",
    2: ROOT / "benchmark_results" / "027_03" / "seed2" / "checkpoint_000060.zip",
}
BASELINE_SUMMARY = ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla_five_scenario_nstep_020_11" / "020_11_hla_five_scenario_summary.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def baseline_rows() -> pd.DataFrame:
    source = pd.read_csv(BASELINE_SUMMARY)
    source["scenario"] = source["scenario"].fillna("null")
    source = source[source["year"].isin(YEARS) & source["scenario"].isin(("null", "recorded_farmer", "dssat_auto", "extension_expert"))]
    rows: list[dict[str, Any]] = []
    mapping = {"recorded_farmer": "recorded", "extension_expert": "official_extension_expert"}
    for _, row in source.iterrows():
        source_file = ROOT / str(row["source_file"])
        snapshot = source_file.parent / "pdi_tmp_snapshot_eval"
        metrics = metrics_source.metrics_from_snapshot(
            snapshot,
            float(row["final_grain_kg_ha"]),
            float(row["irrigation_executed_total_mm"]),
            float(row["nitrogen_executed_total_kg_ha"]),
        )
        rows.append({
            "site": "HLA", "year": int(row["year"]),
            "scenario": mapping.get(str(row["scenario"]), str(row["scenario"])),
            "final_gwad": float(row["final_grain_kg_ha"]),
            "final_cwad": float(row["final_biomass_kg_ha"]),
            "irrigation_total": float(row["irrigation_executed_total_mm"]),
            "fertilizer_total": float(row["nitrogen_executed_total_kg_ha"]),
            "source_snapshot": str(snapshot.relative_to(ROOT)).replace("\\", "/"),
            **metrics,
        })
    frame = pd.DataFrame(rows)
    if len(frame) != len(YEARS) * 4 or not frame.groupby("year")["scenario"].nunique().eq(4).all():
        raise ValueError("HLA four-baseline matrix is incomplete")
    return frame


def prepare(year: int, seed: int) -> tuple[Path, dict[str, Any]]:
    run_dir = OUT / "runs" / str(year) / f"seed{seed}"
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {run_dir}")
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True)
    for item in sorted(hla.YEAR_INPUTS[year].iterdir()):
        if item.is_file():
            shutil.copyfile(item, input_dir / item.name)
    filex = hla.validate_input(input_dir, year)
    env_args = {
        "log_saving_path": str(run_dir / "pdi_gym.log"), "mode": "all", "seed": 0,
        "random_weather": False, "evaluation": True,
        "fileX_template_path": str(filex), "experiment_number": 1,
        "auxiliary_file_paths": [str(path) for path in sorted(input_dir.iterdir()) if path.is_file() and path != filex],
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (run_dir / "env_args.json").write_text(json.dumps(env_args, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir, env_args


def evaluate(year: int, seed: int, model_path: Path, scaler: pd.DataFrame, baselines: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir, env_args = prepare(year, seed)
    local = baselines[baselines["year"] == year]
    null_yield = float(local.loc[local["scenario"] == "null", "final_gwad"].iloc[0])
    gate_yield = float(local.loc[local["scenario"] == "official_extension_expert", "final_gwad"].iloc[0])
    before = sha256(model_path)
    raw = extension.make_raw_env(env_args)
    env = HLAStageEnv027(raw, scaler, null_yield, gate_yield)
    action_rows: list[dict[str, Any]] = []
    total_reward = 0.0
    try:
        model = MaskablePPO.load(model_path, device="cpu")
        obs, info = env.reset()
        done = False
        while not done:
            mask = get_action_masks(env)
            stage_index = int(env.stage_index)
            dap = int(info.get("dap", env._dap()))
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += float(reward)
            action_rows.append({
                "site": "HLA", "year": year, "seed": seed, "stage_index": stage_index,
                "dap": dap, "action_index": int(action), "mask_valid": bool(mask[int(action)]),
                **env.stage_rows[-1],
            })
            done = bool(terminated or truncated)
        if env.last_result is None:
            raise RuntimeError("Season ended without final result")
        tmp = Path(getattr(env.raw_env.unwrapped, "_tmp_folder"))
        snapshot = run_dir / "pdi_tmp_snapshot_eval"
        shutil.copytree(tmp, snapshot, dirs_exist_ok=False)
        metrics = metrics_source.metrics_from_snapshot(snapshot, env.last_result["final_yield"], env.last_result["irrigation_total"], env.last_result["nitrogen_total"])
        result = {
            "site": "HLA", "year": year, "seed": seed,
            "source_model": str(model_path.relative_to(ROOT)).replace("\\", "/"),
            "model_sha256": before,
            "action_sequence": ",".join(str(row["action_index"]) for row in action_rows),
            "final_gwad": float(env.last_result["final_yield"]),
            "final_cwad": float(env.last_result["final_biomass"]),
            "irrigation_total": float(env.last_result["irrigation_total"]),
            "fertilizer_total": float(env.last_result["nitrogen_total"]),
            "episode_reward": total_reward,
            "invalid_action_attempts": int(env.invalid_attempts),
            "run_dir": str(run_dir.relative_to(ROOT)).replace("\\", "/"),
            **metrics,
        }
    finally:
        env.close()
    if sha256(model_path) != before:
        raise RuntimeError(f"Frozen model changed: {model_path}")
    if len(action_rows) != 6:
        raise RuntimeError(f"Expected 6 stage actions, got {len(action_rows)}")
    return result, action_rows


def add_advisor_rule(models: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in models.iterrows():
        refs = baselines[baselines["year"] == int(row["year"])]
        maxima = {
            "yield": float(refs["final_gwad"].max()),
            "wp": float(refs["WP_ET_kg_m3"].max()),
            "pfp": float(refs["PFP_N_kg_kg"].dropna().max()),
        }
        out = row.to_dict()
        pfp = float(row["PFP_N_kg_kg"]) if pd.notna(row["PFP_N_kg_kg"]) else math.nan
        out.update({
            "four_baseline_max_yield": maxima["yield"],
            "four_baseline_max_WP_ET": maxima["wp"],
            "four_baseline_max_PFP_N": maxima["pfp"],
            "yield_strict_win": float(row["final_gwad"]) > maxima["yield"],
            "wp_et_strict_win": float(row["WP_ET_kg_m3"]) > maxima["wp"],
            "pfp_n_strict_win": math.isfinite(pfp) and pfp > maxima["pfp"],
            "yield_gap_pct": 100.0 * (float(row["final_gwad"]) / maxima["yield"] - 1.0),
            "wp_et_gap_pct": 100.0 * (float(row["WP_ET_kg_m3"]) / maxima["wp"] - 1.0),
            "pfp_n_gap_pct": 100.0 * (pfp / maxima["pfp"] - 1.0) if math.isfinite(pfp) else math.nan,
        })
        out["advisor_any_metric_win"] = bool(out["yield_strict_win"] or out["wp_et_strict_win"] or out["pfp_n_strict_win"])
        rows.append(out)
    return pd.DataFrame(rows)


def reuse_smoke_result() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload_path = SMOKE_OUT / "028_08_smoke_result.json"
    actions_path = SMOKE_OUT / "028_08_smoke_actions.csv"
    snapshot = SMOKE_OUT / "runs" / "2007" / "seed0" / "pdi_tmp_snapshot_eval"
    if not payload_path.is_file() or not actions_path.is_file() or not (snapshot / "Summary.OUT").is_file():
        raise FileNotFoundError("028_08 smoke evidence is incomplete and cannot be reused")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    actions = pd.read_csv(actions_path).to_dict("records")
    metrics = metrics_source.metrics_from_snapshot(
        snapshot, float(payload["yield"]), float(payload["irrigation"]), float(payload["nitrogen"])
    )
    plantgro = extension.parse_dssat_table(snapshot / "PlantGro.OUT")
    final_cwad = float(plantgro["CWAD"].dropna().iloc[-1])
    result = {
        "site": "HLA", "year": 2007, "seed": 0,
        "source_model": str(MODELS[0].relative_to(ROOT)).replace("\\", "/"),
        "model_sha256": payload["model_hash"], "action_sequence": payload["action_sequence"],
        "final_gwad": float(payload["yield"]), "final_cwad": final_cwad,
        "irrigation_total": float(payload["irrigation"]), "fertilizer_total": float(payload["nitrogen"]),
        "episode_reward": math.nan, "invalid_action_attempts": 0,
        "run_dir": str((SMOKE_OUT / "runs" / "2007" / "seed0").relative_to(ROOT)).replace("\\", "/"),
        "reused_from_smoke": True, **metrics,
    }
    return result, actions


def run_smoke() -> None:
    global OUT
    OUT = SMOKE_OUT
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    scaler = pd.read_csv(SCALER)
    baselines = baseline_rows()
    result, actions = evaluate(2007, 0, MODELS[0], scaler, baselines)
    payload = {
        "status": "passed" if len(actions) == 6 and result["invalid_action_attempts"] == 0 else "failed",
        "year": 2007, "seed": 0, "training_steps": 0,
        "model_hash": result["model_sha256"], "action_sequence": result["action_sequence"],
        "yield": result["final_gwad"], "irrigation": result["irrigation_total"], "nitrogen": result["fertilizer_total"],
    }
    (OUT / "028_08_smoke_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(actions).to_csv(OUT / "028_08_smoke_actions.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    scaler = pd.read_csv(SCALER)
    baselines = baseline_rows()
    baselines.to_csv(OUT / "028_08_reused_four_baselines.csv", index=False, encoding="utf-8-sig")
    model_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    for year in YEARS:
        for seed, model_path in MODELS.items():
            if year == 2007 and seed == 0:
                result, actions = reuse_smoke_result()
            else:
                result, actions = evaluate(year, seed, model_path, scaler, baselines)
            model_rows.append(result)
            action_rows.extend(actions)
            print(f"OK HLA{year} seed{seed}: Y={result['final_gwad']:.3f} I={result['irrigation_total']:.1f} N={result['fertilizer_total']:.1f}")
    models = add_advisor_rule(pd.DataFrame(model_rows), baselines)
    actions = pd.DataFrame(action_rows)
    models.to_csv(OUT / "028_08_hla_frozen_transfer_summary.csv", index=False, encoding="utf-8-sig")
    actions.to_csv(OUT / "028_08_hla_frozen_transfer_stage_actions.csv", index=False, encoding="utf-8-sig")
    matrix = models.groupby("year").agg(
        seed_count=("seed", "count"), winner_count=("advisor_any_metric_win", "sum"),
        yield_win_count=("yield_strict_win", "sum"), wp_win_count=("wp_et_strict_win", "sum"),
        pfp_win_count=("pfp_n_strict_win", "sum"), invalid_actions=("invalid_action_attempts", "sum"),
    ).reset_index()
    matrix["at_least_two_of_three"] = matrix["winner_count"] >= 2
    matrix.to_csv(OUT / "028_08_hla_year_seed_advisor_matrix.csv", index=False, encoding="utf-8-sig")
    checks = {
        "twelve_seasons": len(models) == 12,
        "all_six_stage_actions": len(actions) == 72,
        "zero_invalid_actions": int(models["invalid_action_attempts"].sum()) == 0,
        "all_model_hashes_unchanged": all(sha256(path) == models.loc[models["seed"] == seed, "model_sha256"].iloc[0] for seed, path in MODELS.items()),
        "zero_training_steps": True,
    }
    result = {"status": "completed" if all(checks.values()) else "failed", "checks": checks, "year_matrix": matrix.to_dict("records"), "training_steps": 0}
    (OUT / "028_08_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 028_08 HLA2010 阶段型 MaskablePPO 固定权重跨年迁移",
        "", f"状态：`{result['status']}`", "",
        "本任务复用三个 HLA2010 selected checkpoint，在 HLA2007/2015/2016/2022 上零训练确定性评估；四基线全部复用，未重跑。",
        "", "## 导师规则结果", "",
        "|year|winner seeds|yield wins|WP_ET wins|PFP_N wins|2/3通过|",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in matrix.iterrows():
        lines.append(f"|{int(row.year)}|{int(row.winner_count)}/3|{int(row.yield_win_count)}|{int(row.wp_win_count)}|{int(row.pfp_win_count)}|{bool(row.at_least_two_of_three)}|")
    lines += ["", "其余两个未领先指标只在 CSV 中报告差值，不设置未经导师确认的接近阈值。", "", "训练步数：0。"]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    run_smoke() if args.smoke_only else main()
