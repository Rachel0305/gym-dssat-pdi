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
from sb3_contrib.common.maskable.utils import get_action_masks


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_sy2014_stage_mc_dqn_seed1_short_022_02 import summary_metrics
from smoke_sy2014_stage_maskable_ppo_train_026_01 import AuditedStageEnv


OUT = ROOT / "benchmark_results" / "026_02"
SCALER = ROOT / "benchmark_results" / "021_24" / "021_24_observation_scaler.csv"
THRESHOLDS_PATH = ROOT / "benchmark_results" / "022_01" / "022_01_thresholds_and_provenance.json"
CHECKPOINTS = (0, 60, 120, 180, 240)


def evaluate(model: MaskablePPO, env: AuditedStageEnv, checkpoint: int, thresholds: dict[str, float]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    before = len(env.all_stage_rows)
    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        mask = get_action_masks(env)
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        total_reward += float(reward)
        done = bool(terminated or truncated)
    if env.last_result is None:
        raise RuntimeError("Evaluation ended without terminal result")
    metrics = summary_metrics(env.raw_env, env.last_result, thresholds)
    stages = []
    for row, executed in zip(env.all_stage_rows[before:], env.last_result["stage_rows"]):
        stages.append({"checkpoint": checkpoint, **row, **{f"executed_{key}": value for key, value in executed.items() if key not in row}})
    result = {
        "checkpoint": checkpoint,
        "action_sequence": ",".join(str(row["action_index"]) for row in env.all_stage_rows[before:]),
        "final_yield": env.last_result["final_yield"],
        "final_biomass": env.last_result["final_biomass"],
        "irrigation_total": env.last_result["irrigation_total"],
        "nitrogen_total": env.last_result["nitrogen_total"],
        "episode_total_reward": total_reward,
        **metrics,
    }
    return result, stages


def finite_model(model: MaskablePPO) -> bool:
    return all(bool(np.isfinite(parameter.detach().cpu().numpy()).all()) for parameter in model.policy.parameters())


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    thresholds = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    train_env = AuditedStageEnv(OUT / "runtime_train", SCALER, seed=1, phase="train")
    eval_env = AuditedStageEnv(OUT / "runtime_eval", SCALER, seed=1001, phase="eval")
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
            seed=1,
            device="cpu",
            verbose=0,
        )
        row, stages = evaluate(model, eval_env, 0, thresholds)
        checkpoint_rows.append(row)
        stage_rows.extend(stages)
        model.save(OUT / "checkpoint_000000")
        for block_index, checkpoint in enumerate(CHECKPOINTS[1:]):
            model.learn(
                total_timesteps=60,
                reset_num_timesteps=(block_index == 0),
                progress_bar=False,
            )
            if int(model.num_timesteps) != checkpoint:
                raise RuntimeError(f"Expected checkpoint {checkpoint}, got {model.num_timesteps}")
            row, stages = evaluate(model, eval_env, checkpoint, thresholds)
            checkpoint_rows.append(row)
            stage_rows.extend(stages)
            model.save(OUT / f"checkpoint_{checkpoint:06d}")

        finite_checkpoints = all(
            math.isfinite(float(row[key]))
            for row in checkpoint_rows
            for key in (
                "final_yield",
                "final_biomass",
                "irrigation_total",
                "nitrogen_total",
                "episode_total_reward",
                "WP_ET_kg_m3",
                "PFP_N_kg_kg",
            )
        )
        later = [row for row in checkpoint_rows if row["checkpoint"] in (120, 180, 240)]
        final_row = checkpoint_rows[-1]
        checks = {
            "versions_2p8p0": stable_baselines3.__version__ == "2.8.0" and sb3_contrib.__version__ == "2.8.0",
            "exact_checkpoints": [row["checkpoint"] for row in checkpoint_rows] == list(CHECKPOINTS),
            "exact_240_training_steps": int(model.num_timesteps) == 240,
            "exact_40_training_seasons": len(train_env.completed_episodes) == 40,
            "zero_masked_action_attempts": train_env.invalid_attempts + eval_env.invalid_attempts == 0,
            "all_evaluations_six_stages": all(len([stage for stage in stage_rows if stage["checkpoint"] == checkpoint]) == 6 for checkpoint in CHECKPOINTS),
            "finite_checkpoint_metrics": finite_checkpoints,
            "finite_model_parameters": finite_model(model),
        }
        engineering_pass = all(checks.values())
        stable_signal = bool(
            engineering_pass
            and final_row["strict_pass"]
            and sum(bool(row["strict_pass"]) for row in later) >= 2
        )
        branch = "A_candidate_ready_for_seed_replication" if stable_signal else ("B_engineering_valid_but_not_stable" if engineering_pass else "C_execution_failed")
        pd.DataFrame(checkpoint_rows).to_csv(OUT / "026_02_checkpoint_summary.csv", index=False)
        pd.DataFrame(stage_rows).to_csv(OUT / "026_02_checkpoint_stage_actions.csv", index=False)
        pd.DataFrame(train_env.completed_episodes).to_csv(OUT / "026_02_training_episode_summary.csv", index=False)
        payload = {
            "status": "completed",
            "branch": branch,
            "checks": checks,
            "config": {
                "seed": 1,
                "total_timesteps": 240,
                "n_steps": 60,
                "batch_size": 30,
                "n_epochs": 5,
                "gamma": 1.0,
                "gae_lambda": 1.0,
                "learning_rate": 3e-4,
                "net_arch": [32, 32],
                "checkpoints": list(CHECKPOINTS),
            },
            "training_seasons": len(train_env.completed_episodes),
            "evaluation_seasons": len(checkpoint_rows),
            "later_strict_pass_count": sum(bool(row["strict_pass"]) for row in later),
            "final_strict_pass": bool(final_row["strict_pass"]),
            "checkpoint_results": checkpoint_rows,
            "scientific_success_claimed": False,
            "next_step_allowed": stable_signal,
        }
        (OUT / "026_02_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
