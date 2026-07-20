from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from mask_aware_dqn_029 import MaskAwareDQN
from run_sy2014_stage_mc_dqn_seed1_short_022_02 import summary_metrics
from smoke_sy2014_stage_maskable_ppo_train_026_01 import AuditedStageEnv


OUT = ROOT / "benchmark_results" / "029_01_sy2014_mask_aware_dqn_smoke"
SCALER = ROOT / "benchmark_results" / "021_24" / "021_24_observation_scaler.csv"
THRESHOLDS = ROOT / "benchmark_results" / "022_01" / "022_01_thresholds_and_provenance.json"
TOTAL_STEPS = 60


def evaluate(model: MaskAwareDQN, env: AuditedStageEnv, checkpoint: int) -> dict[str, object]:
    obs, _ = env.reset()
    action_rows: list[dict[str, object]] = []
    total_reward = 0.0
    done = False
    while not done:
        mask = env.action_masks().copy()
        stage_index = int(env.stage_index)
        dap = int(env._dap())
        action = model.select_action(obs, mask, checkpoint, deterministic=True)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        action_rows.append(
            {
                "checkpoint": checkpoint,
                "stage_index": stage_index,
                "dap": dap,
                "action": action,
                "valid_actions": ",".join(map(str, np.flatnonzero(mask))),
            }
        )
        obs = next_obs
        total_reward += float(reward)
        done = bool(terminated or truncated)
    if env.last_result is None:
        raise RuntimeError("Evaluation did not produce a terminal result")
    thresholds = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    metrics = summary_metrics(env.raw_env, env.last_result, thresholds)
    return {
        "checkpoint": checkpoint,
        "action_sequence": ",".join(str(row["action"]) for row in action_rows),
        "episode_total_reward": total_reward,
        "final_yield": env.last_result["final_yield"],
        "final_biomass": env.last_result["final_biomass"],
        "irrigation_total": env.last_result["irrigation_total"],
        "nitrogen_total": env.last_result["nitrogen_total"],
        **metrics,
        "actions": action_rows,
    }


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    train_env = AuditedStageEnv(OUT / "runtime_train", SCALER, seed=0, phase="train")
    eval_env = AuditedStageEnv(OUT / "runtime_eval", SCALER, seed=1000, phase="eval")
    model = MaskAwareDQN(25, seed=0)
    transition_rows: list[dict[str, object]] = []
    update_rows: list[dict[str, object]] = []
    try:
        evaluation_before = evaluate(model, eval_env, 0)
        obs, _ = train_env.reset()
        for global_step in range(1, TOTAL_STEPS + 1):
            mask = train_env.action_masks().copy()
            action = model.select_action(obs, mask, global_step - 1, deterministic=False)
            next_obs, reward, terminated, truncated, _ = train_env.step(action)
            done = bool(terminated or truncated)
            next_mask = np.zeros(model.action_dim, dtype=bool) if done else train_env.action_masks().copy()
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
                    "global_step": global_step,
                    "action": action,
                    "action_was_valid": bool(mask[action]),
                    "mask": "".join("1" if value else "0" for value in mask),
                    "next_mask": "".join("1" if value else "0" for value in next_mask),
                    "reward": float(reward),
                    "done": done,
                    "epsilon": model.epsilon(global_step - 1),
                }
            )
            if global_step >= 60:
                update_rows.append({"global_step": global_step, **model.train_step()})
            if global_step % 60 == 0:
                model.sync_target()
            if done and global_step < TOTAL_STEPS:
                obs, _ = train_env.reset()
            else:
                obs = next_obs

        checkpoint = OUT / "checkpoint_000060.pt"
        model.save(checkpoint, global_step=TOTAL_STEPS)
        evaluation_after = evaluate(model, eval_env, 60)
        restored, restored_step = MaskAwareDQN.load(checkpoint)
        restored_evaluation = evaluate(restored, eval_env, 60)
        required_numeric = [
            value
            for row in (evaluation_before, evaluation_after)
            for key, value in row.items()
            if key in {"episode_total_reward", "final_yield", "final_biomass", "irrigation_total", "nitrogen_total", "WP_ET_kg_m3"}
        ]
        checks = {
            "exact_60_environment_steps": len(transition_rows) == 60,
            "exact_10_training_seasons": len(train_env.completed_episodes) == 10,
            "all_collected_actions_valid": all(bool(row["action_was_valid"]) for row in transition_rows),
            "environment_zero_invalid_attempts": train_env.invalid_attempts + eval_env.invalid_attempts == 0,
            "replay_size_60": model.replay.size == 60,
            "one_optimizer_update_at_learning_start": model.optimizer_updates == 1,
            "one_target_sync": model.target_updates == 1,
            "finite_update_metrics": all(math.isfinite(float(value)) for row in update_rows for key, value in row.items() if key != "global_step"),
            "finite_evaluation_metrics": all(math.isfinite(float(value)) for value in required_numeric),
            "checkpoint_restored_step_60": restored_step == 60,
            "checkpoint_roundtrip_same_online_hash": restored.module_hash(restored.online) == model.module_hash(model.online),
            "checkpoint_roundtrip_same_action_sequence": restored_evaluation["action_sequence"] == evaluation_after["action_sequence"],
            "reward_roundtrip_close": abs(float(restored_evaluation["episode_total_reward"]) - float(evaluation_after["episode_total_reward"])) <= 1e-12,
        }
        pd.DataFrame(transition_rows).to_csv(OUT / "029_01_transitions.csv", index=False)
        pd.DataFrame(update_rows).to_csv(OUT / "029_01_updates.csv", index=False)
        pd.DataFrame(train_env.completed_episodes).to_csv(OUT / "029_01_training_episodes.csv", index=False)
        pd.DataFrame([{key: value for key, value in row.items() if key != "actions"} for row in (evaluation_before, evaluation_after)]).to_csv(
            OUT / "029_01_evaluations.csv", index=False
        )
        result = {
            "status": "completed" if all(checks.values()) else "failed",
            "engineering_pass": all(checks.values()),
            "checks": checks,
            "evaluation_before": evaluation_before,
            "evaluation_after": evaluation_after,
            "online_hash": model.module_hash(model.online),
            "target_hash": model.module_hash(model.target),
        }
        (OUT / "029_01_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        if not all(checks.values()):
            raise RuntimeError("029_01 engineering smoke failed")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
