from __future__ import annotations

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
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_sy2014_stage_maskable_ppo_seed0_primary_026_03 import select_by_reward
from run_sy2014_stage_maskable_ppo_seed1_curve_026_02 import CHECKPOINTS, evaluate, finite_model
from smoke_sy2014_stage_maskable_ppo_train_026_01 import AuditedStageEnv


OUT = ROOT / "benchmark_results" / "026_04"
SCALER = ROOT / "benchmark_results" / "021_24" / "021_24_observation_scaler.csv"
THRESHOLDS_PATH = ROOT / "benchmark_results" / "022_01" / "022_01_thresholds_and_provenance.json"
SEED0_RESULT = ROOT / "benchmark_results" / "026_03" / "026_03_result.json"
SEED1_RESULT = ROOT / "benchmark_results" / "026_02" / "026_02_result.json"


def existing_seed_summary(seed: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = select_by_reward(rows)
    later_count = sum(bool(row["primary_pass"]) for row in rows if int(row["checkpoint"]) in (120, 180, 240))
    return {
        "seed": seed,
        "selected_checkpoint": int(selected["checkpoint"]),
        "selected_reward": float(selected["episode_total_reward"]),
        "selected_yield": float(selected["final_yield"]),
        "selected_irrigation": float(selected["irrigation_total"]),
        "selected_nitrogen": float(selected["nitrogen_total"]),
        "selected_wp_et": float(selected["WP_ET_kg_m3"]),
        "selected_pfp_n": float(selected["PFP_N_kg_kg"]),
        "selected_primary": bool(selected["primary_pass"]),
        "selected_strict": bool(selected["strict_pass"]),
        "later_primary_count": int(later_count),
    }


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    thresholds = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    seed0_payload = json.loads(SEED0_RESULT.read_text(encoding="utf-8"))
    seed1_payload = json.loads(SEED1_RESULT.read_text(encoding="utf-8"))
    seed0_rows = seed0_payload["seed0_checkpoint_results"]
    seed1_rows = seed1_payload["checkpoint_results"]

    train_env = AuditedStageEnv(OUT / "runtime_train", SCALER, seed=2, phase="train_seed2")
    eval_env = AuditedStageEnv(OUT / "runtime_eval", SCALER, seed=1002, phase="eval_seed2")
    checkpoint_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
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
            seed=2,
            device="cpu",
            verbose=0,
        )
        row, stages = evaluate(model, eval_env, 0, thresholds)
        checkpoint_rows.append(row)
        stage_rows.extend(stages)
        model.save(OUT / "checkpoint_000000")
        for block_index, checkpoint in enumerate(CHECKPOINTS[1:]):
            model.learn(total_timesteps=60, reset_num_timesteps=(block_index == 0), progress_bar=False)
            if int(model.num_timesteps) != checkpoint:
                raise RuntimeError(f"Expected checkpoint {checkpoint}, got {model.num_timesteps}")
            row, stages = evaluate(model, eval_env, checkpoint, thresholds)
            checkpoint_rows.append(row)
            stage_rows.extend(stages)
            model.save(OUT / f"checkpoint_{checkpoint:06d}")

        finite_metrics = all(
            math.isfinite(float(row[key]))
            for row in checkpoint_rows
            for key in ("final_yield", "final_biomass", "irrigation_total", "nitrogen_total", "episode_total_reward", "WP_ET_kg_m3", "PFP_N_kg_kg")
        )
        seed2_selected = select_by_reward(checkpoint_rows)
        seed2_later_primary = sum(bool(row["primary_pass"]) for row in checkpoint_rows if int(row["checkpoint"]) in (120, 180, 240))
        checks = {
            "versions_2p8p0": stable_baselines3.__version__ == "2.8.0" and sb3_contrib.__version__ == "2.8.0",
            "exact_checkpoints": [row["checkpoint"] for row in checkpoint_rows] == list(CHECKPOINTS),
            "exact_240_training_steps": int(model.num_timesteps) == 240,
            "exact_40_training_seasons": len(train_env.completed_episodes) == 40,
            "zero_masked_action_attempts": train_env.invalid_attempts + eval_env.invalid_attempts == 0,
            "all_evaluations_six_stages": all(len([stage for stage in stage_rows if stage["checkpoint"] == checkpoint]) == 6 for checkpoint in CHECKPOINTS),
            "finite_metrics": finite_metrics,
            "finite_model_parameters": finite_model(model),
        }
        summaries = [
            existing_seed_summary(0, seed0_rows),
            existing_seed_summary(1, seed1_rows),
            existing_seed_summary(2, checkpoint_rows),
        ]
        scientific_checks = {
            "seed2_selected_primary": bool(seed2_selected["primary_pass"]),
            "seed2_later_primary_at_least_2_of_3": seed2_later_primary >= 2,
            "all_three_selected_primary": all(row["selected_primary"] for row in summaries),
            "all_three_later_primary_at_least_2_of_3": all(int(row["later_primary_count"]) >= 2 for row in summaries),
        }
        engineering_pass = all(checks.values())
        three_seed_signal = engineering_pass and all(scientific_checks.values())
        branch = "A_three_seed_primary_signal" if three_seed_signal else ("B_two_of_three_primary_only" if engineering_pass else "C_execution_failed")
        pd.DataFrame(checkpoint_rows).to_csv(OUT / "026_04_seed2_checkpoint_summary.csv", index=False)
        pd.DataFrame(stage_rows).to_csv(OUT / "026_04_seed2_checkpoint_stage_actions.csv", index=False)
        pd.DataFrame(train_env.completed_episodes).to_csv(OUT / "026_04_seed2_training_episode_summary.csv", index=False)
        pd.DataFrame(summaries).to_csv(OUT / "026_04_three_seed_comparison.csv", index=False)
        payload = {
            "status": "completed",
            "branch": branch,
            "checks": checks,
            "scientific_checks": scientific_checks,
            "selection_rule": "max deterministic episode_total_reward among checkpoints 60/120/180/240; ties choose earlier",
            "three_seed_selected_summary": summaries,
            "seed2_checkpoint_results": checkpoint_rows,
            "scientific_success_claimed": False,
            "next_step_allowed": bool(three_seed_signal),
        }
        (OUT / "026_04_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
