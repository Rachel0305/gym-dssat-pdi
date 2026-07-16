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

from run_sy2014_stage_maskable_ppo_seed1_curve_026_02 import CHECKPOINTS, evaluate, finite_model
from smoke_sy2014_stage_maskable_ppo_train_026_01 import AuditedStageEnv


OUT = ROOT / "benchmark_results" / "026_03"
SCALER = ROOT / "benchmark_results" / "021_24" / "021_24_observation_scaler.csv"
THRESHOLDS_PATH = ROOT / "benchmark_results" / "022_01" / "022_01_thresholds_and_provenance.json"
SEED1_RESULT = ROOT / "benchmark_results" / "026_02" / "026_02_result.json"


def select_by_reward(rows: list[dict[str, Any]]) -> dict[str, Any]:
    trained = [row for row in rows if int(row["checkpoint"]) > 0]
    if not trained:
        raise ValueError("No trained checkpoints available")
    return max(trained, key=lambda row: (float(row["episode_total_reward"]), -int(row["checkpoint"])))


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    thresholds = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    seed1_payload = json.loads(SEED1_RESULT.read_text(encoding="utf-8"))
    seed1_rows = seed1_payload["checkpoint_results"]
    seed1_selected = select_by_reward(seed1_rows)
    seed1_later_primary = sum(bool(row["primary_pass"]) for row in seed1_rows if int(row["checkpoint"]) in (120, 180, 240))

    train_env = AuditedStageEnv(OUT / "runtime_train", SCALER, seed=0, phase="train_seed0")
    eval_env = AuditedStageEnv(OUT / "runtime_eval", SCALER, seed=1000, phase="eval_seed0")
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
            seed=0,
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
        seed0_selected = select_by_reward(checkpoint_rows)
        seed0_later_primary = sum(bool(row["primary_pass"]) for row in checkpoint_rows if int(row["checkpoint"]) in (120, 180, 240))
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
        engineering_pass = all(checks.values())
        scientific_checks = {
            "seed0_selected_primary": bool(seed0_selected["primary_pass"]),
            "seed0_later_primary_at_least_2_of_3": seed0_later_primary >= 2,
            "seed1_selected_primary": bool(seed1_selected["primary_pass"]),
            "seed1_later_primary_at_least_2_of_3": seed1_later_primary >= 2,
        }
        cross_seed_signal = engineering_pass and all(scientific_checks.values())
        branch = "A_cross_seed_primary_signal" if cross_seed_signal else ("B_seed0_not_replicated" if engineering_pass else "C_execution_failed")
        pd.DataFrame(checkpoint_rows).to_csv(OUT / "026_03_seed0_checkpoint_summary.csv", index=False)
        pd.DataFrame(stage_rows).to_csv(OUT / "026_03_seed0_checkpoint_stage_actions.csv", index=False)
        pd.DataFrame(train_env.completed_episodes).to_csv(OUT / "026_03_seed0_training_episode_summary.csv", index=False)
        comparison = pd.DataFrame([
            {"seed": 0, "selected_checkpoint": seed0_selected["checkpoint"], "selected_reward": seed0_selected["episode_total_reward"], "selected_primary": seed0_selected["primary_pass"], "selected_strict": seed0_selected["strict_pass"], "later_primary_count": seed0_later_primary},
            {"seed": 1, "selected_checkpoint": seed1_selected["checkpoint"], "selected_reward": seed1_selected["episode_total_reward"], "selected_primary": seed1_selected["primary_pass"], "selected_strict": seed1_selected["strict_pass"], "later_primary_count": seed1_later_primary},
        ])
        comparison.to_csv(OUT / "026_03_cross_seed_comparison.csv", index=False)
        payload = {
            "status": "completed",
            "branch": branch,
            "checks": checks,
            "scientific_checks": scientific_checks,
            "selection_rule": "max deterministic episode_total_reward among checkpoints 60/120/180/240; ties choose earlier",
            "seed0_selected": seed0_selected,
            "seed1_selected_from_026_02": seed1_selected,
            "seed0_later_primary_count": seed0_later_primary,
            "seed1_later_primary_count": seed1_later_primary,
            "seed0_checkpoint_results": checkpoint_rows,
            "scientific_success_claimed": False,
            "next_step_allowed": bool(cross_seed_signal),
        }
        (OUT / "026_03_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
