#!/usr/bin/env python3
"""Persist only the seven representative DQN snapshots absent from 029_02/029_03."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mask_aware_dqn_029 import MaskAwareDQN
import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as multi
import run_hla2010_stage_maskable_ppo_seed0_027_02 as hla_run
from smoke_sy2014_stage_maskable_ppo_train_026_01 import AuditedStageEnv
import run_sy_all_authoritative_years_frozen_stage_ppo_026_06 as sy_base
import run_sy_icdat_aligned_all_years_frozen_stage_ppo_026_07 as sy_aligned

# Keep the DSSAT runtime path deliberately short; the PDI bridge is sensitive
# to long bind-mounted paths on Windows.
OUT = ROOT / "benchmark_results" / "029_04_dqn_daily_replay"
ANCHOR = ROOT / "benchmark_results" / "029_02_five_site_stage_mask_aware_dqn"
REP = ROOT / "benchmark_results" / "029_04_maskableppo_vs_maskaware_dqn_evidence" / "029_04_dqn_visualization_representatives.csv"
REPLAY_CASES = {("SY", 2012), ("SY", 2014), ("SY", 2015), ("HLA", 2010), ("YC", 2014), ("FQ", 2016), ("LC", 2010)}
SY_SCALER = ROOT / "benchmark_results/021_24/021_24_observation_scaler.csv"
HLA_READY = ROOT / "benchmark_results/027_01_attempt2"
MULTI_READY = ROOT / "benchmark_results/027_07_site_specific_stage_maskable_ppo_attempt2"


def selected(site: str, year: int) -> tuple[int, int, Path]:
    frame = pd.read_csv(REP)
    row = frame[(frame.site == site) & (frame.year.astype(int) == year)].iloc[0]
    seed, checkpoint = int(row.seed), int(row.checkpoint)
    raw_source = row.get("source_model")
    if pd.isna(raw_source) or str(raw_source).strip().lower() == "nan":
        payload = json.loads((ANCHOR / site / f"seed{seed}" / "result.json").read_text(encoding="utf-8"))
        raw_source = payload["selected"]["model_path"]
    return seed, checkpoint, ROOT / str(raw_source)


def run_env(env, model: MaskAwareDQN, checkpoint: int, site: str, year: int, seed: int):
    print(f"TRACE {site}{year} seed{seed}: before reset", flush=True)
    obs, _ = env.reset(); done = False; actions = []
    print(f"TRACE {site}{year} seed{seed}: after reset", flush=True)
    while not done:
        mask = np.asarray(env.action_masks(), dtype=bool)
        stage = int(env.stage_index)
        action = model.select_action(obs, mask, checkpoint, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        print(f"TRACE {site}{year} seed{seed}: stage{stage} action{action} complete", flush=True)
        actions.append({"site": site, "year": year, "seed": seed, "stage_index": stage, **env.stage_rows[-1]})
        done = bool(terminated or truncated)
    if getattr(env, "last_result", None) is None:
        raise RuntimeError("Representative replay ended without final result")
    return actions


def run_anchor(site: str, year: int, seed: int, checkpoint: int, model_path: Path, target: Path) -> None:
    if site == "SY":
        eval_env = AuditedStageEnv(target / "runtime_eval", SY_SCALER, seed=1000 + seed, phase=f"evidence_seed{seed}")
        expected_stages = 6
    elif site == "HLA":
        scaler = pd.read_csv(HLA_READY / "027_01_hla2010_observation_scaler.csv")
        reward = json.loads((HLA_READY / "027_01_hla2010_reward_config.json").read_text(encoding="utf-8"))
        eval_env = hla_run.make_env(target / "runtime_eval", scaler, reward, seed=1000 + seed, phase=f"evidence_seed{seed}")
        expected_stages = 6
    else:
        spec = multi.SPECS[site]
        _, scaler, _ = multi.load_readiness(spec, MULTI_READY / site)
        reward = json.loads((MULTI_READY / site / "readiness" / "readiness_result.json").read_text(encoding="utf-8"))["reward_config"]
        eval_env = multi.make_stage_env(spec, target / "runtime_eval", scaler, reward, 1000 + seed, f"evidence_seed{seed}")
        expected_stages = len(spec.executable_stage_daps)
    model, stored_step = MaskAwareDQN.load(model_path)
    if stored_step != checkpoint:
        raise RuntimeError(f"Checkpoint mismatch: {stored_step} != {checkpoint}")
    try:
        actions = run_env(eval_env, model, checkpoint, site, year, seed)
        snapshot_source = multi.snapshot_from_env(eval_env.raw_env)
        shutil.copytree(snapshot_source, target / "snapshot")
    finally:
        eval_env.close()
    if len(actions) != expected_stages:
        raise RuntimeError(f"{site}{year}: {len(actions)} stages, expected {expected_stages}")
    pd.DataFrame(actions).to_csv(target / "stage_actions.csv", index=False, encoding="utf-8-sig")


def run_sy_transfer(year: int, seed: int, checkpoint: int, model_path: Path, target: Path) -> None:
    print(f"TRACE SY{year}: install patch", flush=True)
    sy_aligned.install_runtime_patch(target / "runtime")
    print(f"TRACE SY{year}: prepare input", flush=True)
    run_dir, env_args = sy_aligned.prepare_linked_run_aligned(year, f"dqn_evidence_seed{seed}", seed, target / "runtime")
    print(f"TRACE SY{year}: construct env", flush=True)
    env = sy_base.TransferStageEnv(sy_base.sy.make_raw_env(env_args), sy_base.SCALER)
    print(f"TRACE SY{year}: load model", flush=True)
    model, stored_step = MaskAwareDQN.load(model_path)
    if stored_step != checkpoint:
        raise RuntimeError(f"Checkpoint mismatch: {stored_step} != {checkpoint}")
    try:
        actions = run_env(env, model, checkpoint, "SY", year, seed)
        shutil.copytree(Path(getattr(env.raw_env.unwrapped, "_tmp_folder")), target / "snapshot")
    finally:
        env.close()
    if len(actions) != 6:
        raise RuntimeError(f"SY{year}: expected six stages")
    pd.DataFrame(actions).to_csv(target / "stage_actions.csv", index=False, encoding="utf-8-sig")


def run_case(site: str, year: int) -> None:
    if (site, year) not in REPLAY_CASES:
        raise ValueError((site, year))
    target = OUT / site / str(year)
    if target.exists():
        if (target / "result.json").is_file():
            raise FileExistsError(target)
        # Preserve an incomplete timeout directory; never delete or overwrite it.
        attempt = 2
        while (OUT / site / f"{year}_attempt{attempt}").exists():
            attempt += 1
        target = OUT / site / f"{year}_attempt{attempt}"
    target.mkdir(parents=True)
    seed, checkpoint, model_path = selected(site, year)
    if site == "SY":
        run_sy_transfer(year, seed, checkpoint, model_path, target)
    else:
        run_anchor(site, year, seed, checkpoint, model_path, target)
    payload = {"status": "completed", "site": site, "year": year, "seed": seed, "checkpoint": checkpoint, "source_model": str(model_path.relative_to(ROOT)).replace("\\", "/"), "training_steps": 0, "purpose": "daily evidence persistence only"}
    (target / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--site", required=True); parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args(); run_case(args.site, args.year)


if __name__ == "__main__":
    main()
