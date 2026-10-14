from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import stable_baselines3
import sb3_contrib
from sb3_contrib import MaskablePPO


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import run_hla2010_stage_maskable_ppo_seed0_027_02 as base
from run_hla2010_stage_ppo_readiness_scaler_smoke_027_01 import OBS_DIM, source_hashes


OUT = ROOT / "benchmark_results" / "027_03"
SEED0_ROOT = ROOT / "benchmark_results" / "027_02_attempt2"
SEED0_RESULT = SEED0_ROOT / "027_02_result.json"
EXPECTED_SEED0_HASH = "6715fffb4bbe251cf0cdad13121331d9e6f875a3e6629c26423500e502e042da"
CHECKPOINTS = (0, 60, 120, 180, 240)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_model(model: MaskablePPO, seed_dir: Path, checkpoint: int) -> tuple[Path, str]:
    stem = seed_dir / f"checkpoint_{checkpoint:06d}"
    model.save(stem)
    path = stem.with_suffix(".zip")
    if not path.exists():
        raise FileNotFoundError(path)
    return path, sha256(path)


def run_seed(seed: int, scaler: pd.DataFrame, reward: dict[str, Any]) -> dict[str, Any]:
    seed_dir = OUT / f"seed{seed}"
    seed_dir.mkdir(parents=True)
    train_env = base.make_env(seed_dir / "runtime_train", scaler, reward, seed=seed, phase=f"train_seed{seed}")
    eval_env = base.make_env(seed_dir / "runtime_eval", scaler, reward, seed=1000 + seed, phase=f"eval_seed{seed}")
    checkpoint_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    learn_calls = 0
    try:
        model = MaskablePPO(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=60,
            batch_size=30,
            n_epochs=5,
            gamma=1.0,
            gae_lambda=1.0,
            ent_coef=0.0,
            policy_kwargs={"net_arch": [32, 32]},
            seed=seed,
            device="cpu",
            verbose=0,
        )
        row, stages = base.evaluate(model, eval_env, 0)
        path, digest = save_model(model, seed_dir, 0)
        row.update({"seed": seed, "model_path": str(path.relative_to(ROOT)), "model_sha256": digest})
        checkpoint_rows.append(row)
        stage_rows.extend({"seed": seed, **stage} for stage in stages)
        hashes["0"] = digest

        for block_index, checkpoint in enumerate(CHECKPOINTS[1:]):
            model.learn(total_timesteps=60, reset_num_timesteps=(block_index == 0), progress_bar=False)
            learn_calls += 1
            if int(model.num_timesteps) != checkpoint:
                raise RuntimeError(f"seed{seed}: expected checkpoint {checkpoint}, got {model.num_timesteps}")
            row, stages = base.evaluate(model, eval_env, checkpoint)
            path, digest = save_model(model, seed_dir, checkpoint)
            row.update({"seed": seed, "model_path": str(path.relative_to(ROOT)), "model_sha256": digest})
            checkpoint_rows.append(row)
            stage_rows.extend({"seed": seed, **stage} for stage in stages)
            hashes[str(checkpoint)] = digest

        selected = base.select_by_reward(checkpoint_rows)
        metric_keys = (
            "final_yield", "final_biomass", "irrigation_total", "nitrogen_total",
            "episode_total_reward", "WP_ET_kg_m3",
        )
        checks = {
            "versions_2p8p0": stable_baselines3.__version__ == "2.8.0" and sb3_contrib.__version__ == "2.8.0",
            "exact_checkpoints": [row["checkpoint"] for row in checkpoint_rows] == list(CHECKPOINTS),
            "exact_240_training_steps": int(model.num_timesteps) == 240,
            "exact_40_training_seasons": len(train_env.completed_episodes) == 40,
            "exact_four_learn_calls": learn_calls == 4,
            "five_evaluation_seasons": len(eval_env.completed_episodes) == 5,
            "all_evaluations_six_stages": len(stage_rows) == 30,
            "zero_masked_action_attempts": train_env.invalid_attempts + eval_env.invalid_attempts == 0,
            "finite_required_metrics": all(
                math.isfinite(float(row[key])) for row in checkpoint_rows for key in metric_keys
            ),
            "finite_model_parameters": base.model_finite(model),
            "reward_components_close": all(
                abs(float(row["episode_total_reward"]) - float(row["resource_reward_total"]) - float(row["terminal_reward_total"])) <= 1e-9
                for row in checkpoint_rows
            ),
            "five_unique_model_hashes": len(set(hashes.values())) == 5,
        }
        pd.DataFrame(checkpoint_rows).to_csv(
            seed_dir / f"027_03_hla2010_seed{seed}_checkpoint_summary.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(stage_rows).to_csv(
            seed_dir / f"027_03_hla2010_seed{seed}_checkpoint_stage_actions.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(train_env.completed_episodes).to_csv(
            seed_dir / f"027_03_hla2010_seed{seed}_training_episode_summary.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(train_env.all_stage_rows).to_csv(
            seed_dir / f"027_03_hla2010_seed{seed}_training_stage_actions.csv", index=False, encoding="utf-8-sig"
        )
        payload = {
            "seed": seed,
            "checks": checks,
            "engineering_pass": all(checks.values()),
            "selected_checkpoint": selected,
            "checkpoint_results": checkpoint_rows,
            "training_seasons": len(train_env.completed_episodes),
            "evaluation_seasons": len(eval_env.completed_episodes),
            "learn_calls": learn_calls,
        }
        (seed_dir / f"027_03_seed{seed}_result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
        )
        return payload
    finally:
        train_env.close()
        eval_env.close()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    seed0_payload = json.loads(SEED0_RESULT.read_text(encoding="utf-8"))
    seed0_selected = dict(seed0_payload["selected_checkpoint"])
    seed0_path = ROOT / seed0_selected["model_path"]
    prechecks = {
        "seed0_branch_authorized": seed0_payload["branch"] == "A_seed0_primary_signal",
        "seed0_selected_checkpoint_180": int(seed0_selected["checkpoint"]) == 180,
        "seed0_selected_hash_declared": seed0_selected["model_sha256"] == EXPECTED_SEED0_HASH,
        "seed0_selected_model_exists": seed0_path.exists(),
        "seed0_selected_model_hash_matches": seed0_path.exists() and sha256(seed0_path) == EXPECTED_SEED0_HASH,
    }
    if not all(prechecks.values()):
        raise RuntimeError(f"027_03 prechecks failed: {prechecks}")
    hashes_before = source_hashes()
    scaler = pd.read_csv(base.SCALER_PATH)
    reward = json.loads(base.REWARD_PATH.read_text(encoding="utf-8"))
    seed_results = [run_seed(seed, scaler, reward) for seed in (1, 2)]

    seed0_summary = {
        "seed": 0,
        "engineering_pass": True,
        "selected_checkpoint": int(seed0_selected["checkpoint"]),
        "selected_model_sha256": seed0_selected["model_sha256"],
        "selected_yield": float(seed0_selected["final_yield"]),
        "selected_irrigation": float(seed0_selected["irrigation_total"]),
        "selected_nitrogen": float(seed0_selected["nitrogen_total"]),
        "selected_WP_ET": float(seed0_selected["WP_ET_kg_m3"]),
        "selected_PFP_N": float(seed0_selected["PFP_N_kg_kg"]),
        "selected_reward": float(seed0_selected["episode_total_reward"]),
        "selected_primary_pass": bool(seed0_selected["primary_pass"]),
        "selected_recorded_pass": bool(seed0_selected["recorded_all_comparable_pass"]),
    }
    summaries = [seed0_summary]
    for payload in seed_results:
        selected = payload["selected_checkpoint"]
        summaries.append({
            "seed": int(payload["seed"]),
            "engineering_pass": bool(payload["engineering_pass"]),
            "selected_checkpoint": int(selected["checkpoint"]),
            "selected_model_sha256": selected["model_sha256"],
            "selected_yield": float(selected["final_yield"]),
            "selected_irrigation": float(selected["irrigation_total"]),
            "selected_nitrogen": float(selected["nitrogen_total"]),
            "selected_WP_ET": float(selected["WP_ET_kg_m3"]),
            "selected_PFP_N": float(selected["PFP_N_kg_kg"]) if pd.notna(selected["PFP_N_kg_kg"]) else math.nan,
            "selected_reward": float(selected["episode_total_reward"]),
            "selected_primary_pass": bool(selected["primary_pass"]),
            "selected_recorded_pass": bool(selected["recorded_all_comparable_pass"]),
        })
    summary = pd.DataFrame(summaries).sort_values("seed")
    engineering_pass = bool(summary["engineering_pass"].all() and source_hashes() == hashes_before)
    primary_count = int(summary["selected_primary_pass"].sum())
    branch = (
        "A_three_seed_primary_replicated"
        if engineering_pass and primary_count >= 2
        else ("B_three_seed_primary_not_replicated" if engineering_pass else "C_execution_failed")
    )
    summary.to_csv(OUT / "027_03_hla2010_three_seed_selected_summary.csv", index=False, encoding="utf-8-sig")
    payload = {
        "status": "completed",
        "branch": branch,
        "prechecks": prechecks,
        "source_hashes_unchanged": source_hashes() == hashes_before,
        "selection_rule": "per seed: max deterministic episode_total_reward among checkpoints 60/120/180/240; exact ties choose earlier",
        "primary_pass_count": primary_count,
        "recorded_pass_count": int(summary["selected_recorded_pass"].sum()),
        "three_seed_selected": summary.to_dict("records"),
        "seed1_result": seed_results[0],
        "seed2_result": seed_results[1],
        "scientific_success_claimed": branch == "A_three_seed_primary_replicated",
        "next_step_allowed": branch == "A_three_seed_primary_replicated",
        "next_step": "Freeze three selected checkpoints and preregister HLA within-site cross-year transfer" if branch == "A_three_seed_primary_replicated" else "Stop HLA cross-year transfer under preregistered rule",
    }
    (OUT / "027_03_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps({
        "status": payload["status"], "branch": branch,
        "primary_pass_count": primary_count,
        "recorded_pass_count": payload["recorded_pass_count"],
        "three_seed_selected": payload["three_seed_selected"],
        "next_step_allowed": payload["next_step_allowed"],
    }, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
