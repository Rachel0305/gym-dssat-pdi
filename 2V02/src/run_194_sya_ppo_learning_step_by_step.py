"""Task 194: a very explicit first PPO learning lesson.

This is intentionally not a general framework.  It is a teaching script for
one fixed SYA 2005 original-weather episode.  The code is written in the same
order as the algorithm:

1. create the environment;
2. create a new MaskablePPO model;
3. collect one rollout of 144 transitions;
4. print/save the variables stored for every transition;
5. compute returns and advantages;
6. stop before the parameter update unless RUN_UPDATE is changed to True.

The purpose of the first run is observation, not diagnosis.  The script uses
the current reward/action/observation chain but does not modify existing PPO
scripts or checkpoints.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

# These are the already audited environment-construction modules.  We do not
# hide the PPO learning loop inside them; they are used only to build the same
# kind of environment as the current SYA PPO experiment.
import run_184_sya_ppo_critic_td_action_value_audit as task184
import run_189_sya_gap1_early_irrigation_reward_critic_audit as task189


TASK_ID = "194"
OUTPUT_DIR = ROOT / "benchmark_results" / "194_sya_ppo_learning_step_by_step"
RUN_UPDATE = False

# These are the current PPO settings used by Task191.  They are written here
# explicitly so a beginner can see what is being passed to MaskablePPO.
LEARNING_RATE = 0.0003
GAMMA = 1.0
GAE_LAMBDA = 1.0
N_STEPS = 144
BATCH_SIZE = 144
N_EPOCHS = 5
ENT_COEF = 0.02
CLIP_RANGE = 0.2
NET_ARCH = [64, 64]


def scalar(value: Any, default: float = np.nan) -> float:
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
        if array.size == 0:
            return default
        number = float(array[0])
        return number if np.isfinite(number) else default
    except Exception:
        return default


def action_label(action_index: int) -> str:
    irrigation = [0, 15, 30, 45][int(action_index) // 4]
    nitrogen = [0, 40, 80, 120][int(action_index) % 4]
    return f"I{irrigation}/N{nitrogen}"


def probabilities_for_one_state(model, observation: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
    """Read the policy's current probability for each action.

    This is the exact quantity used for action selection.  We read it directly
    from the policy before and, optionally, after model.train().
    """
    import torch
    from stable_baselines3.common.utils import obs_as_tensor

    observation_batch = np.asarray(observation, dtype=np.float32).reshape(1, -1)
    mask_batch = np.asarray(action_mask, dtype=bool).reshape(1, -1)
    observation_tensor = obs_as_tensor(observation_batch, model.device)
    with torch.no_grad():
        distribution = model.policy.get_distribution(observation_tensor, action_masks=mask_batch)
        probabilities = distribution.distribution.probs.detach().cpu().numpy()[0]
    return np.asarray(probabilities, dtype=float)


class RecordOneRollout:
    """Small callback needed by SB3 to expose each collected transition."""

    def __init__(self, model_placeholder=None):
        from stable_baselines3.common.callbacks import BaseCallback

        outer = self

        class Callback(BaseCallback):
            def __init__(self) -> None:
                super().__init__(verbose=0)

            def _on_step(self) -> bool:
                # collect_rollouts() places these local variables here.  They
                # are the exact inputs/outputs of one environment step.
                pre_observation = np.asarray(self.locals["self"]._last_obs).reshape(-1).copy()
                action = int(np.asarray(self.locals["actions"]).reshape(-1)[0])
                reward = float(np.asarray(self.locals["rewards"]).reshape(-1)[0])
                done = bool(np.asarray(self.locals["dones"]).reshape(-1)[0])
                value_before = float(np.asarray(self.locals["values"].detach().cpu()).reshape(-1)[0])
                log_probability_before = float(np.asarray(self.locals["log_probs"].detach().cpu()).reshape(-1)[0])
                action_mask = np.asarray(self.locals["action_masks"]).reshape(-1).copy()
                probability_before = probabilities_for_one_state(self.model, pre_observation, action_mask)

                outer.rows.append(
                    {
                        "rollout_step": len(outer.rows),
                        "action_index": action,
                        "action": action_label(action),
                        "reward": reward,
                        "done": done,
                        "value_before": value_before,
                        "log_probability_selected_before": log_probability_before,
                        "probability_selected_before": float(np.exp(log_probability_before)),
                        "observation_vector": json.dumps(np.round(pre_observation, 6).tolist()),
                        "action_mask": "".join("1" if flag else "0" for flag in action_mask.tolist()),
                        "all_action_probabilities_before": json.dumps(np.round(probability_before, 8).tolist()),
                    }
                )

                # This print is deliberately plain.  It lets us see the data
                # before interpreting it.
                print(
                    f"step={len(outer.rows)-1:03d} "
                    f"action={action_label(action):>9s} "
                    f"reward={reward: .6f} "
                    f"V={value_before: .6f} "
                    f"old_log_prob={log_probability_before: .6f} "
                    f"legal={int(action_mask.sum())}",
                    flush=True,
                )
                return True

        self.rows: list[dict[str, Any]] = []
        self.callback = Callback()


# ---------------------------------------------------------------------------
# Step 1. Load the same physical scenario manifest and environment chain.
# ---------------------------------------------------------------------------
base_config_path = ROOT / "configs" / "191_sya_balanced_irrigation_gap1_100k_learning_curve.json"
base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
task178 = task184.task178
task168 = task178.task177.task174.task173.task172.task168
gate = task168.verify_gate(base_config)
manifest = task168.read_manifest(base_config)
split = task168.load_split(manifest)
if not bool(gate.get("physical_response_passed", True)):
    raise RuntimeError("The existing physical response gate did not pass.")

# Task189 supplies the already audited gap=1 runtime wrapper.  It is restored
# at the end; no source file or original result is overwritten.
runtime_patch_owner, original_patch, original_restore = task189.patch_gap1_runtime()
saved_runtime = runtime_patch_owner.patch_runtime_178(manifest)

try:
    engine = task168.engine
    base03222 = task168.base03222
    runtime_config = engine.load_config()
    env_config = base03222.direct_ppo.build_env_config(runtime_config, split)

    # One fixed environment is used first so there is only one episode and one
    # weather situation to understand.  We will add the six-scenario sampler
    # only after this basic process is understood.
    env = base03222.base.make_env(
        runtime_config,
        env_config,
        "SYA",
        2005,
        0,
        "SYA_2005_original_194_learning_lesson",
        evaluation=False,
    )

    # -----------------------------------------------------------------------
    # Step 2. Create a new PPO.  No old checkpoint is loaded here.
    # -----------------------------------------------------------------------
    from sb3_contrib import MaskablePPO

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=0,
        seed=0,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        ent_coef=ENT_COEF,
        clip_range=CLIP_RANGE,
        policy_kwargs={"net_arch": NET_ARCH},
    )

    print("\n[模型参数]")
    print(f"learning_rate={model.learning_rate}")
    print(f"gamma={model.gamma}")
    print(f"gae_lambda={model.gae_lambda}")
    print(f"n_steps={model.n_steps}")
    print(f"batch_size={model.batch_size}")
    print(f"n_epochs={model.n_epochs}")
    print(f"ent_coef={model.ent_coef}")
    print(f"clip_range={model.clip_range}")

    # -----------------------------------------------------------------------
    # Step 3. Collect one rollout, but do not call model.learn().
    # -----------------------------------------------------------------------
    # PPO first collects n_steps transitions.  This is the same collection
    # stage that model.learn() normally performs internally.
    recorder = RecordOneRollout()
    total_timesteps, callback = model._setup_learn(
        total_timesteps=N_STEPS,
        callback=recorder.callback,
        reset_num_timesteps=True,
        tb_log_name="194_first_rollout",
        progress_bar=False,
    )
    callback.on_training_start(locals(), globals())
    collected = model.collect_rollouts(
        model.get_env(),
        callback,
        model.rollout_buffer,
        n_rollout_steps=N_STEPS,
        use_masking=True,
    )
    callback.on_training_end()

    if not collected:
        raise RuntimeError("collect_rollouts returned False before the first rollout ended.")

    # -----------------------------------------------------------------------
    # Step 4. PPO now computes the future return and advantage for each row.
    # -----------------------------------------------------------------------
    # This does not change the policy yet.  It only fills the training labels
    # that model.train() will use.
    buffer = model.rollout_buffer
    for index, row in enumerate(recorder.rows):
        row["return_used_for_training"] = float(buffer.returns[index, 0])
        row["advantage_used_for_training"] = float(buffer.advantages[index, 0])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    (OUTPUT_DIR / "rollout_transitions.csv").write_text(
        pd.DataFrame(recorder.rows).to_csv(index=False), encoding="utf-8-sig"
    )
    (OUTPUT_DIR / "ppo_parameters.json").write_text(
        json.dumps(
            {
                "learning_rate": LEARNING_RATE,
                "gamma": GAMMA,
                "gae_lambda": GAE_LAMBDA,
                "n_steps": N_STEPS,
                "batch_size": BATCH_SIZE,
                "n_epochs": N_EPOCHS,
                "ent_coef": ENT_COEF,
                "clip_range": CLIP_RANGE,
                "net_arch": NET_ARCH,
                "run_update": RUN_UPDATE,
                "training_scenario": "SYA 2005 original fixed episode",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n[第一次收集结束]")
    print(f"collected={len(recorder.rows)} transitions")
    print("现在已经有了 PPO 第一次更新所需的原始数据，但还没有更新神经网络参数。")
    print("已保存：benchmark_results/194_sya_ppo_learning_step_by_step/rollout_transitions.csv")

    # This flag is deliberately visible near the top of the file.  The first
    # lesson stops here.  Only when we explicitly set RUN_UPDATE=True do we
    # call the real SB3 PPO train() method.
    if RUN_UPDATE:
        print("\n[开始第一次 PPO 参数更新]")
        model.train()
        print("[第一次 PPO 参数更新结束]")
        print("训练日志变量：")
        for key, value in sorted(model.logger.name_to_value.items()):
            if str(key).startswith("train/"):
                print(f"{key} = {value}")
finally:
    runtime_patch_owner.restore_runtime_178(saved_runtime)
    task178.patch_runtime_178 = original_patch
    task178.restore_runtime_178 = original_restore
    try:
        env.close()
    except Exception:
        pass
