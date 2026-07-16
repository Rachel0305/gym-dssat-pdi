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
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import BaseCallback


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from stage_decision_env_ppo_026 import StageDecisionEnv026


OUT = ROOT / "benchmark_results" / "026_01"
SCALER = ROOT / "benchmark_results" / "021_24" / "021_24_observation_scaler.csv"
MODEL_PATH = OUT / "026_01_model"
TOTAL_TIMESTEPS = 24


class AuditedStageEnv(StageDecisionEnv026):
    def __init__(self, *args: Any, phase: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.phase = phase
        self.valid_steps = 0
        self.invalid_attempts = 0
        self.completed_episodes: list[dict[str, Any]] = []
        self.all_stage_rows: list[dict[str, Any]] = []

    def step(self, action_index: int):
        action_index = int(action_index)
        stage_index = self.stage_index
        dap = int(round(float(self._dap())))
        mask = self.action_masks().copy()
        if action_index < 0 or action_index >= len(mask) or not bool(mask[action_index]):
            self.invalid_attempts += 1
        output = super().step(action_index)
        self.valid_steps += 1
        self.all_stage_rows.append(
            {
                "phase": self.phase,
                "episode_index": len(self.completed_episodes),
                "stage_index": stage_index,
                "dap": dap,
                "action_index": action_index,
                "mask_valid": bool(mask[action_index]),
                "valid_actions": ",".join(map(str, np.flatnonzero(mask).tolist())),
            }
        )
        _, reward, terminated, truncated, _ = output
        if terminated or truncated:
            if self.last_result is None:
                raise RuntimeError("Terminal episode is missing last_result")
            self.completed_episodes.append(
                {
                    "phase": self.phase,
                    "episode_index": len(self.completed_episodes),
                    "final_yield": self.last_result["final_yield"],
                    "final_biomass": self.last_result["final_biomass"],
                    "irrigation_total": self.last_result["irrigation_total"],
                    "nitrogen_total": self.last_result["nitrogen_total"],
                    "terminal_step_reward": float(reward),
                    "stage_count": len(self.last_result["stage_rows"]),
                }
            )
        return output


class RolloutCounter(BaseCallback):
    def __init__(self) -> None:
        super().__init__(verbose=0)
        self.rollouts = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.rollouts += 1


def deterministic_episode(model: MaskablePPO, env: AuditedStageEnv) -> tuple[dict[str, Any], np.ndarray, np.ndarray, int]:
    obs, _ = env.reset()
    initial_obs = obs.copy()
    initial_mask = get_action_masks(env).copy()
    initial_action, _ = model.predict(obs, action_masks=initial_mask, deterministic=True)
    done = False
    total_reward = 0.0
    while not done:
        mask = get_action_masks(env)
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        total_reward += float(reward)
        done = bool(terminated or truncated)
    result = dict(env.completed_episodes[-1])
    result["episode_total_reward"] = total_reward
    return result, initial_obs, initial_mask, int(initial_action)


def state_dict_finite(model: MaskablePPO) -> bool:
    return all(bool(np.isfinite(param.detach().cpu().numpy()).all()) for param in model.policy.parameters())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)

    train_env = AuditedStageEnv(OUT / "runtime_train", SCALER, seed=1, phase="train")
    pre_env = AuditedStageEnv(OUT / "runtime_eval_pre", SCALER, seed=101, phase="eval_pre")
    post_env = AuditedStageEnv(OUT / "runtime_eval_post", SCALER, seed=102, phase="eval_post")
    callback = RolloutCounter()
    try:
        model = MaskablePPO(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=12,
            batch_size=12,
            n_epochs=2,
            gamma=1.0,
            gae_lambda=1.0,
            ent_coef=0.0,
            policy_kwargs={"net_arch": [32, 32]},
            seed=1,
            device="cpu",
            verbose=0,
        )
        pre_result, _, _, _ = deterministic_episode(model, pre_env)
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, progress_bar=False)
        post_result, initial_obs, initial_mask, action_before_save = deterministic_episode(model, post_env)
        model.save(MODEL_PATH)
        model_zip = MODEL_PATH.with_suffix(".zip")
        reloaded = MaskablePPO.load(model_zip, device="cpu")
        action_after_load, _ = reloaded.predict(initial_obs, action_masks=initial_mask, deterministic=True)

        episode_rows = [pre_result, *train_env.completed_episodes, post_result]
        stage_rows = [*pre_env.all_stage_rows, *train_env.all_stage_rows, *post_env.all_stage_rows]
        finite_episode_values = all(
            math.isfinite(float(row[key]))
            for row in episode_rows
            for key in ("final_yield", "final_biomass", "irrigation_total", "nitrogen_total", "terminal_step_reward")
        )
        checks = {
            "sb3_version_2p8p0": stable_baselines3.__version__ == "2.8.0",
            "sb3_contrib_version_2p8p0": sb3_contrib.__version__ == "2.8.0",
            "num_timesteps_24": int(model.num_timesteps) == TOTAL_TIMESTEPS,
            "four_training_seasons": len(train_env.completed_episodes) == 4,
            "at_least_two_rollouts": callback.rollouts >= 2,
            "zero_masked_action_attempts": sum(env.invalid_attempts for env in (train_env, pre_env, post_env)) == 0,
            "all_episodes_six_stages": all(int(row["stage_count"]) == 6 for row in episode_rows),
            "all_logged_actions_mask_valid": all(bool(row["mask_valid"]) for row in stage_rows),
            "finite_episode_values": finite_episode_values,
            "finite_model_parameters": state_dict_finite(model),
            "model_saved": model_zip.exists(),
            "reload_action_identical": int(action_after_load) == action_before_save,
        }
        branch = "A_smoke_passed" if all(checks.values()) else "C_smoke_failed"
        pd.DataFrame(episode_rows).to_csv(OUT / "026_01_episode_summary.csv", index=False)
        pd.DataFrame(stage_rows).to_csv(OUT / "026_01_stage_actions.csv", index=False)
        payload = {
            "status": "completed",
            "branch": branch,
            "checks": checks,
            "config": {
                "seed": 1,
                "total_timesteps": TOTAL_TIMESTEPS,
                "n_steps": 12,
                "batch_size": 12,
                "n_epochs": 2,
                "gamma": 1.0,
                "gae_lambda": 1.0,
                "learning_rate": 3e-4,
                "net_arch": [32, 32],
            },
            "rollouts": callback.rollouts,
            "training_seasons": len(train_env.completed_episodes),
            "total_dssat_seasons": len(episode_rows),
            "invalid_action_attempts": sum(env.invalid_attempts for env in (train_env, pre_env, post_env)),
            "pre_training_evaluation": pre_result,
            "post_training_evaluation": post_result,
            "model_sha256": sha256(model_zip),
            "scientific_success_claimed": False,
            "next_step_allowed": all(checks.values()),
        }
        (OUT / "026_01_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        train_env.close()
        pre_env.close()
        post_env.close()


if __name__ == "__main__":
    main()
