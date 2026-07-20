from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

import run_hla_five_scenario_completion_020_11 as hla
import run_sy_all_authoritative_years_frozen_stage_ppo_026_06 as common


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "026_09"
SMOKE = ROOT / "benchmark_results" / "026_09_smoke_hla2010_seed0"
YEARS = (2007, 2010, 2015, 2016, 2022)
BASELINE_SUMMARY = ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla_five_scenario_nstep_020_11" / "020_11_hla_five_scenario_summary.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_run(year: int, seed: int, root: Path) -> tuple[Path, dict[str, Any]]:
    source = hla.YEAR_INPUTS[year]
    run_dir = root / str(year) / f"seed{seed}" / "frozen_sy_stage_ppo"
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {run_dir}")
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True)
    for item in sorted(source.iterdir()):
        if item.is_file():
            shutil.copyfile(item, input_dir / item.name)
    filex = hla.validate_input(input_dir, year)
    env_args = {
        "log_saving_path": str(run_dir / "pdi_gym.log"),
        "mode": "all",
        "seed": 0,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(filex),
        "experiment_number": 1,
        "auxiliary_file_paths": [str(path) for path in sorted(input_dir.iterdir()) if path.is_file() and path != filex],
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (run_dir / "env_args.json").write_text(json.dumps(env_args, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir, env_args


def baseline_rows() -> pd.DataFrame:
    source = pd.read_csv(BASELINE_SUMMARY)
    source["scenario"] = source["scenario"].fillna("null")
    wanted = source[source["scenario"].isin(["null", "recorded_farmer", "dssat_auto", "extension_expert"])].copy()
    rows: list[dict[str, Any]] = []
    mapping = {"recorded_farmer": "recorded", "extension_expert": "official_extension_expert"}
    for _, row in wanted.iterrows():
        source_file = ROOT / str(row["source_file"])
        snapshot = source_file.parent / "pdi_tmp_snapshot_eval"
        metrics = common.metrics_from_snapshot(
            snapshot,
            float(row["final_grain_kg_ha"]),
            float(row["irrigation_executed_total_mm"]),
            float(row["nitrogen_executed_total_kg_ha"]),
        )
        rows.append(
            {
                "site": "HLA",
                "year": int(row["year"]),
                "scenario": mapping.get(str(row["scenario"]), str(row["scenario"])),
                "final_gwad": float(row["final_grain_kg_ha"]),
                "final_cwad": float(row["final_biomass_kg_ha"]),
                "irrigation_total": float(row["irrigation_executed_total_mm"]),
                "fertilizer_total": float(row["nitrogen_executed_total_kg_ha"]),
                "source_file": str(row["source_file"]),
                **metrics,
            }
        )
    frame = pd.DataFrame(rows)
    counts = frame.groupby("year")["scenario"].nunique()
    if not counts.eq(4).all() or set(frame["year"]) != set(YEARS):
        raise ValueError("HLA four-baseline reuse set is incomplete")
    return frame


def evaluate(year: int, seed: int, model_path: Path, targets: dict[str, float], root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir, env_args = prepare_run(year, seed, root)
    env = common.TransferStageEnv(common.sy.make_raw_env(env_args), common.SCALER)
    before = sha256(model_path)
    actions: list[dict[str, Any]] = []
    total_reward = 0.0
    try:
        model = MaskablePPO.load(model_path, device="cpu")
        obs, info = env.reset()
        done = False
        while not done:
            mask = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            stage_index = env.stage_index
            dap = int(info.get("dap", env._dap()))
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += float(reward)
            actions.append({"site": "HLA", "year": year, "seed": seed, "stage_index": stage_index, "dap": dap, "action_index": int(action), "mask_valid": bool(mask[int(action)]), **env.stage_rows[-1]})
            done = bool(terminated or truncated)
        if env.last_result is None:
            raise RuntimeError("HLA frozen PPO season ended without result")
        tmp = Path(getattr(env.raw_env.unwrapped, "_tmp_folder"))
        metrics = common.metrics_from_snapshot(tmp, env.last_result["final_yield"], env.last_result["irrigation_total"], env.last_result["nitrogen_total"])
        result = {
            "site": "HLA", "year": year, "seed": seed,
            "source_model": str(model_path.relative_to(ROOT)), "model_sha256": before,
            "action_sequence": ",".join(str(row["action_index"]) for row in actions),
            "final_gwad": env.last_result["final_yield"], "final_cwad": env.last_result["final_biomass"],
            "irrigation_total": env.last_result["irrigation_total"], "fertilizer_total": env.last_result["nitrogen_total"],
            "frozen_environment_return": total_reward, "invalid_action_attempts": env.invalid_attempts,
            "run_dir": str(run_dir.relative_to(ROOT)), **metrics,
        }
        result["yield_pass"] = result["final_gwad"] >= targets["yield_min"]
        result["wp_et_pass"] = result["WP_ET_kg_m3"] >= targets["wp_et_min"]
        result["pfp_n_pass"] = result["PFP_N_kg_kg"] >= targets["pfp_n_min"]
        result["local_primary_pass"] = bool(result["yield_pass"] and result["wp_et_pass"] and result["pfp_n_pass"])
    finally:
        env.close()
    if sha256(model_path) != before:
        raise RuntimeError("Frozen model hash changed")
    return result, actions


def run_smoke() -> None:
    if SMOKE.exists():
        raise FileExistsError(f"Refusing to overwrite {SMOKE}")
    baselines = baseline_rows()
    targets = common.local_targets(baselines[baselines["year"].eq(2010)])
    result, actions = evaluate(2010, 0, common.MODELS[0], targets, SMOKE)
    SMOKE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(actions).to_csv(SMOKE / "026_09_smoke_actions.csv", index=False)
    payload = {"status": "passed", "result": result, "targets": targets, "training_steps": 0}
    (SMOKE / "026_09_smoke_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    baselines = baseline_rows()
    baselines.to_csv(OUT / "026_09_hla_reused_four_baselines_with_wp_pfp.csv", index=False)
    model_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    for year in YEARS:
        targets = common.local_targets(baselines[baselines["year"].eq(year)])
        for seed, model_path in common.MODELS.items():
            result, actions = evaluate(year, seed, model_path, targets, OUT / "ppo_runs")
            model_rows.append(result)
            action_rows.extend(actions)
    models = pd.DataFrame(model_rows)
    actions = pd.DataFrame(action_rows)
    recorded = baselines.loc[baselines["scenario"].eq("recorded"), ["year", "final_gwad"]].rename(columns={"final_gwad": "recorded_yield"})
    models = models.merge(recorded, on="year", how="left")
    models["yield_ge_recorded"] = models["final_gwad"] >= models["recorded_yield"]
    models.to_csv(OUT / "026_09_hla_frozen_sy_ppo_summary.csv", index=False)
    actions.to_csv(OUT / "026_09_hla_frozen_sy_ppo_stage_actions.csv", index=False)
    matrix = models.pivot(index="year", columns="seed", values="local_primary_pass").reset_index()
    matrix.columns = ["year", "seed0_local_primary", "seed1_local_primary", "seed2_local_primary"]
    matrix["pass_count"] = matrix.filter(like="_local_primary").sum(axis=1).astype(int)
    matrix["year_transfer_pass"] = matrix["pass_count"] >= 2
    matrix.to_csv(OUT / "026_09_hla_year_seed_pass_matrix.csv", index=False)
    checks = {
        "all_15_models": len(models) == 15,
        "zero_training_steps": True,
        "zero_invalid_actions": int(models["invalid_action_attempts"].sum()) == 0,
        "model_hashes_unchanged": all(sha256(path) == models.loc[models["seed"].eq(seed), "model_sha256"].iloc[0] for seed, path in common.MODELS.items()),
    }
    engineering = all(checks.values())
    all_years = bool(matrix["year_transfer_pass"].all())
    payload = {
        "status": "completed",
        "branch": "A_HLA_all_years_cross_site_transfer" if engineering and all_years else ("B_HLA_partial_year_transfer" if engineering else "C_input_or_execution_blocked"),
        "checks": checks,
        "all_years_at_least_two_of_three": all_years,
        "training_steps": 0,
        "scientific_success_claimed": False,
    }
    (OUT / "026_09_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if args.smoke_only:
        run_smoke()
    else:
        main()
