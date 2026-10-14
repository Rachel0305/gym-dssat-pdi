from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from mask_aware_dqn_029 import MaskAwareDQN
import run_hla2010_stage_maskable_ppo_seed0_027_02 as hla_run
import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as multi_run
import run_sy2014_stage_maskable_ppo_seed1_curve_026_02 as sy_eval
from smoke_sy2014_stage_maskable_ppo_train_026_01 import AuditedStageEnv


OUT = ROOT / "benchmark_results" / "029_02_five_site_stage_mask_aware_dqn"
CHECKPOINTS = (0, 60, 120, 180, 240)
SY_SCALER = ROOT / "benchmark_results" / "021_24" / "021_24_observation_scaler.csv"
SY_THRESHOLDS = ROOT / "benchmark_results" / "022_01" / "022_01_thresholds_and_provenance.json"
HLA_READINESS = ROOT / "benchmark_results" / "027_01_attempt2"
MULTI_READINESS = ROOT / "benchmark_results" / "027_07_site_specific_stage_maskable_ppo_attempt2"


class EvaluationAdapter:
    def __init__(self, model: MaskAwareDQN, checkpoint: int) -> None:
        self.model = model
        self.checkpoint = int(checkpoint)

    def predict(self, observation, action_masks, deterministic=True):
        action = self.model.select_action(
            np.asarray(observation, dtype=np.float32),
            np.asarray(action_masks, dtype=bool),
            self.checkpoint,
            deterministic=bool(deterministic),
        )
        return action, None


def site_components(site: str, seed: int, seed_dir: Path):
    if site == "SY":
        train_env = AuditedStageEnv(seed_dir / "runtime_train", SY_SCALER, seed=seed, phase=f"train_seed{seed}")
        eval_env = AuditedStageEnv(seed_dir / "runtime_eval", SY_SCALER, seed=1000 + seed, phase=f"eval_seed{seed}")
        thresholds = json.loads(SY_THRESHOLDS.read_text(encoding="utf-8"))

        def evaluate(model: MaskAwareDQN, env, checkpoint: int):
            return sy_eval.evaluate(EvaluationAdapter(model, checkpoint), env, checkpoint, thresholds)

        return train_env, eval_env, evaluate, 25, 6

    if site == "HLA":
        scaler = pd.read_csv(HLA_READINESS / "027_01_hla2010_observation_scaler.csv")
        reward = json.loads((HLA_READINESS / "027_01_hla2010_reward_config.json").read_text(encoding="utf-8"))
        train_env = hla_run.make_env(seed_dir / "runtime_train", scaler, reward, seed=seed, phase=f"train_seed{seed}")
        eval_env = hla_run.make_env(seed_dir / "runtime_eval", scaler, reward, seed=1000 + seed, phase=f"eval_seed{seed}")

        def evaluate(model: MaskAwareDQN, env, checkpoint: int):
            return hla_run.evaluate(EvaluationAdapter(model, checkpoint), env, checkpoint)

        return train_env, eval_env, evaluate, 24, 6

    spec = multi_run.SPECS[site]
    readiness, scaler, baselines = multi_run.load_readiness(spec, MULTI_READINESS / site)
    reward = readiness["reward_config"]
    train_env = multi_run.make_stage_env(spec, seed_dir / "runtime_train", scaler, reward, seed, f"train_seed{seed}")
    eval_env = multi_run.make_stage_env(spec, seed_dir / "runtime_eval", scaler, reward, 1000 + seed, f"eval_seed{seed}")

    def evaluate(model: MaskAwareDQN, env, checkpoint: int):
        return multi_run.evaluate(EvaluationAdapter(model, checkpoint), env, checkpoint, baselines)

    return train_env, eval_env, evaluate, spec.observation_dimension, len(spec.executable_stage_daps)


def select_by_reward(rows: list[dict[str, Any]]) -> dict[str, Any]:
    trained = [row for row in rows if int(row["checkpoint"]) > 0]
    return max(trained, key=lambda row: (float(row["episode_total_reward"]), -int(row["checkpoint"])))


def run(site: str, seed: int) -> None:
    seed_dir = OUT / site / f"seed{seed}"
    if seed_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {seed_dir}")
    seed_dir.mkdir(parents=True)
    train_env, eval_env, evaluate, observation_dim, stages_per_episode = site_components(site, seed, seed_dir)
    model = MaskAwareDQN(observation_dim, seed=seed)
    checkpoint_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    try:
        row, stages = evaluate(model, eval_env, 0)
        model_path = seed_dir / "checkpoint_000000.pt"
        model.save(model_path, 0)
        row.update({"site": site, "seed": seed, "model_path": str(model_path.relative_to(ROOT)), "model_sha256": model.module_hash(model.online)})
        checkpoint_rows.append(row)
        stage_rows.extend({"site": site, "seed": seed, **stage} for stage in stages)
        obs, _ = train_env.reset()
        for global_step in range(1, 241):
            mask = train_env.action_masks().copy()
            action = model.select_action(obs, mask, global_step - 1, deterministic=False)
            next_obs, reward, terminated, truncated, _ = train_env.step(action)
            done = bool(terminated or truncated)
            next_mask = np.zeros(9, dtype=bool) if done else train_env.action_masks().copy()
            model.add_transition(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                done=done,
                mask=mask,
                next_mask=next_mask,
            )
            transition_rows.append(
                {
                    "site": site,
                    "seed": seed,
                    "global_step": global_step,
                    "episode_index": (global_step - 1) // stages_per_episode,
                    "stage_index": (global_step - 1) % stages_per_episode,
                    "action": action,
                    "valid_actions": ",".join(map(str, np.flatnonzero(mask))),
                    "reward": float(reward),
                    "done": done,
                    "epsilon": model.epsilon(global_step - 1),
                }
            )
            if global_step >= 60:
                update_rows.append({"site": site, "seed": seed, "global_step": global_step, **model.train_step()})
            if global_step % 60 == 0:
                model.sync_target()
            if done and global_step < 240:
                obs, _ = train_env.reset()
            else:
                obs = next_obs
            if global_step in CHECKPOINTS[1:]:
                row, stages = evaluate(model, eval_env, global_step)
                model_path = seed_dir / f"checkpoint_{global_step:06d}.pt"
                model.save(model_path, global_step)
                row.update({"site": site, "seed": seed, "model_path": str(model_path.relative_to(ROOT)), "model_sha256": model.module_hash(model.online)})
                checkpoint_rows.append(row)
                stage_rows.extend({"site": site, "seed": seed, **stage} for stage in stages)

        selected = select_by_reward(checkpoint_rows)
        invalid_attempts = int(getattr(train_env, "invalid_attempts", 0)) + int(getattr(eval_env, "invalid_attempts", 0))
        mask_rows = list(getattr(train_env, "all_stage_rows", [])) + list(getattr(eval_env, "all_stage_rows", []))
        checks = {
            "exact_checkpoints": [int(row["checkpoint"]) for row in checkpoint_rows] == list(CHECKPOINTS),
            "exact_240_environment_steps": len(transition_rows) == 240,
            "expected_training_seasons": len(train_env.completed_episodes) == 240 // stages_per_episode,
            "exact_181_optimizer_updates": model.optimizer_updates == 181,
            "exact_four_target_updates": model.target_updates == 4,
            "zero_invalid_attempts": invalid_attempts == 0,
            "all_logged_masks_valid": all(bool(row.get("mask_valid", True)) for row in mask_rows),
            "finite_updates": all(math.isfinite(float(value)) for row in update_rows for key, value in row.items() if key not in {"site", "seed", "global_step"}),
            "five_unique_online_hashes": len({row["model_sha256"] for row in checkpoint_rows}) == 5,
        }
        pd.DataFrame(checkpoint_rows).to_csv(seed_dir / "checkpoint_summary.csv", index=False)
        pd.DataFrame(stage_rows).to_csv(seed_dir / "checkpoint_stage_actions.csv", index=False)
        pd.DataFrame(transition_rows).to_csv(seed_dir / "training_transitions.csv", index=False)
        pd.DataFrame(update_rows).to_csv(seed_dir / "training_updates.csv", index=False)
        pd.DataFrame(train_env.completed_episodes).to_csv(seed_dir / "training_episodes.csv", index=False)
        payload = {
            "status": "completed" if all(checks.values()) else "failed",
            "site": site,
            "seed": seed,
            "checks": checks,
            "selected": selected,
            "checkpoint_results": checkpoint_rows,
            "optimizer_updates": model.optimizer_updates,
            "target_updates": model.target_updates,
        }
        (seed_dir / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
        print(json.dumps({"site": site, "seed": seed, "status": payload["status"], "selected": selected}, ensure_ascii=False, indent=2, allow_nan=True))
        if not all(checks.values()):
            raise RuntimeError(f"{site}/seed{seed} engineering checks failed: {checks}")
    finally:
        train_env.close()
        eval_env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=["SY", "HLA", "YC", "FQ", "LC"], required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    args = parser.parse_args()
    run(args.site, args.seed)


if __name__ == "__main__":
    main()
