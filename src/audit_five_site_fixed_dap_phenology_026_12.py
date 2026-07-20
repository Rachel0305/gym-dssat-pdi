from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_hla_five_scenario_completion_020_11 as hla
import run_lc_fixed_input_year_screening_017_11 as lc
import run_sy_all_authoritative_years_frozen_stage_ppo_026_06 as sy_transfer
import run_yc_fq_frozen_nstep_cross_site_020_12 as cross
from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "026_12"
STAGES = (1, 30, 50, 65, 85, 110)
YEARS = {"SY": 2014, "HLA": 2010, "YC": 2014, "FQ": 2016, "LC": 2010}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_env(site: str):
    case = OUT / "runtime" / site
    if site == "SY":
        source_run = ROOT / "benchmark_results" / "026_07_attempt2" / "2014" / "ppo_runs" / "2014" / "seed0" / "transfer_frozen_ppo_seed0"
        input_dir = case / "input"
        input_dir.mkdir(parents=True)
        for source in sorted((source_run / "input").iterdir()):
            if source.is_file():
                shutil.copyfile(source, input_dir / source.name)
        filex = input_dir / "CNSY1201.MZX"
        treatment = 2
    elif site == "HLA":
        input_dir = case / "input"
        input_dir.mkdir(parents=True)
        for source in sorted(hla.YEAR_INPUTS[2010].iterdir()):
            if source.is_file():
                shutil.copyfile(source, input_dir / source.name)
        filex = hla.validate_input(input_dir, 2010)
        treatment = 1
    elif site in {"YC", "FQ"}:
        spec = cross.SITE_SPECS[site]
        filex = cross.copy_site_inputs(spec, case / "input", cross.prepare_site_text(spec, "dqn"), f"{site}_026_12.MZX")
        treatment = spec.treatment
    elif site == "LC":
        lc.OUT_DIR = OUT / "runtime" / "LC_source"
        prepared = lc.prepare_run(2010, "null")
        input_dir = case / "input"
        input_dir.mkdir(parents=True)
        for source in sorted((prepared / "input").iterdir()):
            if source.is_file():
                shutil.copyfile(source, input_dir / source.name)
        filex = next(input_dir.glob("*.MZX"))
        treatment = 3
    else:
        raise ValueError(site)
    env_args = {
        "log_saving_path": str(case / "pdi_gym.log"), "mode": "all", "seed": 0,
        "random_weather": False, "evaluation": True,
        "fileX_template_path": str(filex), "experiment_number": treatment,
        "auxiliary_file_paths": [str(path) for path in sorted(filex.parent.iterdir()) if path.is_file() and path != filex],
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (case / "env_args.json").write_text(json.dumps(env_args, ensure_ascii=False, indent=2), encoding="utf-8")
    return sy_transfer.sy.make_raw_env(env_args), filex


def noop(env: Any) -> np.ndarray:
    return normalize_action(env.formator.action_names, env.formator.action_space_dict, {"amir": 0.0, "anfer": 0.0})


def run_site(site: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env, runtime_filex = make_env(site)
    source_hash_before = sha256(runtime_filex)
    rows: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        done = False
        while True:
            state = latest_observation_dict(env, obs, info)
            rows.append({
                "site": site, "year": YEARS[site], "dap": int(round(float(scalar(state.get("dap"), 0)))),
                "istage": float(scalar(state.get("istage"))), "vstage": float(scalar(state.get("vstage"))),
                "topwt": float(scalar(state.get("topwt"))), "grnwt": float(scalar(state.get("grnwt"))),
            })
            if done:
                break
            obs, _, terminated, truncated, info = env.step(noop(env))
            done = bool(terminated or truncated)
    finally:
        env.close()
    return rows, {
        "site": site, "year": YEARS[site], "runtime_filex": str(runtime_filex.relative_to(ROOT)),
        "runtime_filex_hash_unchanged": sha256(runtime_filex) == source_hash_before,
        "days_recorded": len(rows), "max_dap": max(row["dap"] for row in rows),
    }


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    daily_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for site in ("SY", "HLA", "YC", "FQ", "LC"):
        try:
            rows, manifest = run_site(site)
            daily_rows.extend(rows)
            manifests.append(manifest)
        except Exception as exc:
            errors.append({"site": site, "error_type": type(exc).__name__, "error": str(exc)})
    daily = pd.DataFrame(daily_rows)
    manifest_frame = pd.DataFrame(manifests)
    if not daily.empty:
        snapshots = daily[daily["dap"].isin(STAGES)].drop_duplicates(["site", "dap"], keep="last")
        spread = snapshots.groupby("dap").agg(
            site_count=("site", "nunique"), vstage_min=("vstage", "min"), vstage_max=("vstage", "max"),
            istage_count=("istage", "nunique"),
        ).reset_index()
        spread["vstage_spread"] = spread["vstage_max"] - spread["vstage_min"]
    else:
        snapshots = pd.DataFrame()
        spread = pd.DataFrame()
    daily.to_csv(OUT / "026_12_daily_phenology.csv", index=False, encoding="utf-8-sig")
    snapshots.to_csv(OUT / "026_12_fixed_dap_snapshots.csv", index=False, encoding="utf-8-sig")
    spread.to_csv(OUT / "026_12_cross_site_vstage_spread.csv", index=False, encoding="utf-8-sig")
    manifest_frame.to_csv(OUT / "026_12_runtime_manifests.csv", index=False, encoding="utf-8-sig")
    engineering_pass = bool(
        not errors and len(manifest_frame) == 5 and manifest_frame["runtime_filex_hash_unchanged"].all()
        and set(daily["site"]) == {"SY", "HLA", "YC", "FQ", "LC"}
    )
    fixed_dap_compatible = bool(
        engineering_pass and len(spread) == len(STAGES)
        and spread["site_count"].eq(5).all() and spread["vstage_spread"].le(2.0).all()
    )
    branch = (
        "A_fixed_dap_phenology_approximately_compatible" if fixed_dap_compatible
        else "B_phenology_triggered_stage_adapter_required" if engineering_pass
        else "C_input_or_dssat_execution_failed"
    )
    payload = {
        "status": "completed", "branch": branch, "training_steps": 0,
        "policy_actions_executed": 0, "zero_action_seasons": len(manifest_frame),
        "engineering_pass": engineering_pass, "fixed_dap_compatible": fixed_dap_compatible,
        "fixed_daps": list(STAGES), "vstage_spread_limit": 2.0, "errors": errors,
        "next_step": (
            "Keep fixed DAPs in joint PPO design" if branch.startswith("A_")
            else "Preregister DSSAT-phenology-triggered common decision stages before joint PPO training"
            if branch.startswith("B_") else "Repair input/runtime failure"
        ),
    }
    (OUT / "026_12_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
