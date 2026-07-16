from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from frozen_nstep_dqn_config_020_11 import ACTION_TABLE_9, apply_environment_constants, dqn_kwargs
from minimal_dqfd_style import DemonstrationBatch, MinimalDQfDStyle, large_margin_loss
from ppo_evaluate import latest_observation_dict, scalar
import run_yc2014_linked_dqn_5k_multiseed_013_07 as shared
import run_yc_fq_frozen_nstep_cross_site_020_12 as legacy
from run_fq_yc_new_cultivar_forward_screening_013_01 import parse_dssat_table
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback


OUT = ROOT / "benchmark_results" / "021_20"
ENV_SOURCE = ROOT / "benchmark_results/021_14/021_14_sy2014_reward_scale_seed1_25k_retry__sy_2014_seed1/dqn_env_args.json"
NULL_YIELD = 5408.0
REFERENCE_YIELD = 11205.0
SCHEDULE = {22: 1, 29: 4, 42: 7, 56: 4, 79: 1}
GAMMA = 0.99
N_STEP = 5
PLANNED_TIMESTEPS = 50_000
STOP_TIMESTEPS = 5_000
CHECKPOINT_INTERVAL = 1_000
DEMO_PRETRAIN_UPDATES = 100


def environment_args(label: str) -> dict[str, Any]:
    args = json.loads(ENV_SOURCE.read_text(encoding="utf-8"))
    run = OUT / label
    run.mkdir(parents=True, exist_ok=True)
    args["log_saving_path"] = str(run / "pdi_gym.log")
    return args


def make_env(args: dict[str, Any]):
    apply_environment_constants(shared)
    return legacy.make_train_env(args, NULL_YIELD)


def snapshot(env: Any, destination: Path) -> None:
    tmp = getattr(env.unwrapped, "_tmp_folder", None)
    if not tmp or not Path(tmp).exists():
        raise RuntimeError("PDI temporary output unavailable")
    shutil.copytree(Path(tmp), destination)


def final_yield(snapshot_dir: Path) -> tuple[float, float]:
    frame = parse_dssat_table(snapshot_dir / "PlantGro.OUT")
    gwad = pd.to_numeric(frame["GWAD"], errors="coerce").dropna()
    cwad = pd.to_numeric(frame["CWAD"], errors="coerce").dropna()
    return float(gwad.iloc[-1]), float(cwad.iloc[-1])


def build_n_step(
    rewards: np.ndarray,
    next_observations: np.ndarray,
    dones: np.ndarray,
    n_step: int,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(rewards)
    returns = np.zeros(count, dtype=np.float32)
    next_states = np.zeros_like(next_observations, dtype=np.float32)
    n_dones = np.zeros(count, dtype=np.float32)
    discounts = np.zeros(count, dtype=np.float32)
    horizons = np.zeros(count, dtype=np.int64)
    for start in range(count):
        value = 0.0
        last = start
        steps = 0
        for offset in range(n_step):
            index = start + offset
            if index >= count:
                break
            value += (gamma ** offset) * float(rewards[index])
            last = index
            steps += 1
            if bool(dones[index]):
                break
        returns[start] = value
        next_states[start] = next_observations[last]
        n_dones[start] = float(dones[last])
        discounts[start] = float(gamma ** steps)
        horizons[start] = steps
    return returns, next_states, n_dones, discounts, horizons


def collect_demonstration() -> tuple[DemonstrationBatch, pd.DataFrame, dict[str, Any]]:
    args = environment_args("demonstration_replay")
    env = make_env(args)
    observations: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    next_observations: list[np.ndarray] = []
    dones: list[float] = []
    rows: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        for step in range(420):
            latest = latest_observation_dict(env, obs, info)
            dap = int(round(scalar(latest.get("dap", step)) or 0))
            action = int(SCHEDULE.get(dap, 0))
            before = np.asarray(obs, dtype=np.float32).copy()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            safe = dict(getattr(env, "last_safe_real_action", {}) or {})
            observations.append(before)
            actions.append(action)
            rewards.append(float(reward))
            next_observations.append(np.asarray(next_obs, dtype=np.float32).copy())
            dones.append(float(done))
            rows.append({
                "step": step,
                "dap_action": dap,
                "target_action": action,
                "target_amir": ACTION_TABLE_9[action]["amir"],
                "target_anfer": ACTION_TABLE_9[action]["anfer"],
                "safe_amir": float(safe.get("amir", 0.0)),
                "safe_anfer": float(safe.get("anfer", 0.0)),
                "reward": float(reward),
                "done": done,
            })
            obs = next_obs
            if done:
                break
    finally:
        snap = OUT / "demonstration_replay" / "pdi_tmp_snapshot_eval"
        snapshot(env, snap)
        env.close()

    obs_array = np.asarray(observations, dtype=np.float32)
    action_array = np.asarray(actions, dtype=np.int64)
    reward_array = np.asarray(rewards, dtype=np.float32)
    next_array = np.asarray(next_observations, dtype=np.float32)
    done_array = np.asarray(dones, dtype=np.float32)
    n_returns, n_next, n_dones, n_discounts, horizons = build_n_step(
        reward_array, next_array, done_array, N_STEP, GAMMA
    )
    batch = DemonstrationBatch(
        observations=obs_array,
        actions=action_array,
        rewards=reward_array,
        next_observations=next_array,
        dones=done_array,
        n_step_returns=n_returns,
        n_step_next_observations=n_next,
        n_step_dones=n_dones,
        n_step_discounts=n_discounts,
    )
    batch.validate()
    frame = pd.DataFrame(rows)
    frame["n_step_return"] = n_returns
    frame["n_step_done"] = n_dones
    frame["n_step_discount"] = n_discounts
    frame["n_step_horizon"] = horizons
    gwad, cwad = final_yield(snap)
    nonzero = frame[frame.target_action.ne(0)]
    audit = {
        "yield_kg_ha": gwad,
        "biomass_kg_ha": cwad,
        "irrigation_mm": float(frame.safe_amir.sum()),
        "nitrogen_kg_ha": float(frame.safe_anfer.sum()),
        "transition_count": int(len(frame)),
        "terminal_count": int(frame.done.sum()),
        "last_transition_terminal": bool(frame.done.iloc[-1]),
        "nonzero_event_count": int(len(nonzero)),
        "nonzero_all_uncropped": bool(
            np.allclose(nonzero.target_amir, nonzero.safe_amir)
            and np.allclose(nonzero.target_anfer, nonzero.safe_anfer)
        ),
        "all_arrays_finite": bool(
            np.isfinite(obs_array).all()
            and np.isfinite(next_array).all()
            and np.isfinite(reward_array).all()
            and np.isfinite(n_returns).all()
        ),
        "yield_reproduction_error": abs(gwad - REFERENCE_YIELD),
    }
    audit["passed"] = bool(
        audit["yield_reproduction_error"] <= 2
        and audit["irrigation_mm"] == 75
        and audit["nitrogen_kg_ha"] == 200
        and audit["nonzero_event_count"] == 5
        and audit["nonzero_all_uncropped"]
        and audit["terminal_count"] == 1
        and audit["last_transition_terminal"]
        and audit["all_arrays_finite"]
    )
    frame.to_csv(OUT / "021_20_demonstration_transitions.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        OUT / "021_20_demonstration_transitions.npz",
        observations=obs_array,
        actions=action_array,
        rewards=reward_array,
        next_observations=next_array,
        dones=done_array,
        n_step_returns=n_returns,
        n_step_next_observations=n_next,
        n_step_dones=n_dones,
        n_step_discounts=n_discounts,
        n_step_horizons=horizons,
    )
    (OUT / "021_20_demonstration_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not audit["passed"]:
        raise RuntimeError(f"Demonstration audit failed: {audit}")
    return batch, frame, audit


def load_demonstration() -> DemonstrationBatch:
    data = np.load(OUT / "021_20_demonstration_transitions.npz")
    return DemonstrationBatch(
        observations=data["observations"], actions=data["actions"], rewards=data["rewards"],
        next_observations=data["next_observations"], dones=data["dones"],
        n_step_returns=data["n_step_returns"], n_step_next_observations=data["n_step_next_observations"],
        n_step_dones=data["n_step_dones"], n_step_discounts=data["n_step_discounts"],
    )


def run_unit_tests(batch: DemonstrationBatch) -> dict[str, Any]:
    import gymnasium as gym

    batch.validate()
    q_zero = torch.tensor([[1.0, 0.1, 0.0]], dtype=torch.float32)
    q_positive = torch.tensor([[1.0, 0.5, 0.0]], dtype=torch.float32)
    expert = torch.tensor([0])
    zero_loss = float(large_margin_loss(q_zero, expert, 0.8))
    positive_loss = float(large_margin_loss(q_positive, expert, 0.8))

    toy_q = torch.nn.Parameter(torch.zeros((1, 9), dtype=torch.float32))
    optimizer = torch.optim.Adam([toy_q], lr=0.1)

    def toy_loss() -> torch.Tensor:
        td = F.smooth_l1_loss(toy_q[:, 4], torch.ones(1))
        margin = large_margin_loss(toy_q, torch.tensor([4]), 0.8)
        return td + margin

    before_loss = float(toy_loss().detach())
    before_margin = float((toy_q[0, 4] - torch.max(torch.cat([toy_q[0, :4], toy_q[0, 5:]]))).detach())
    optimizer.zero_grad()
    toy_loss().backward()
    optimizer.step()
    after_loss = float(toy_loss().detach())
    after_margin = float((toy_q[0, 4] - torch.max(torch.cat([toy_q[0, :4], toy_q[0, 5:]]))).detach())

    class DummyEnv(gym.Env):
        def __init__(self) -> None:
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=batch.observations.shape[1:], dtype=np.float32
            )
            self.action_space = gym.spaces.Discrete(9)

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            super().reset(seed=seed)
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}

        def step(self, action: int):
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, False, {}

    dummy = DummyEnv()
    demo_model = MinimalDQfDStyle(
        "MlpPolicy", dummy, verbose=0, seed=0, learning_starts=1,
        buffer_size=32, batch_size=2, train_freq=1, gradient_steps=1,
    )
    demo_model.set_demonstrations(batch)
    permanent_demo_count = demo_model.demonstration_count
    ordinary_replay_size = int(demo_model.replay_buffer.size())

    result = {
        "transition_count": int(len(batch.actions)),
        "last_done": bool(batch.dones[-1]),
        "all_n_step_discounts_positive": bool(np.all(batch.n_step_discounts > 0)),
        "large_margin_zero_case": zero_loss,
        "large_margin_positive_case": positive_loss,
        "toy_loss_before": before_loss,
        "toy_loss_after": after_loss,
        "toy_expert_margin_before": before_margin,
        "toy_expert_margin_after": after_margin,
        "permanent_demo_count": permanent_demo_count,
        "ordinary_replay_size_before_online_training": ordinary_replay_size,
    }
    result["passed"] = bool(
        result["transition_count"] == 160
        and result["last_done"]
        and result["all_n_step_discounts_positive"]
        and abs(zero_loss) <= 1e-7
        and positive_loss > 0
        and after_loss < before_loss
        and after_margin > before_margin
        and permanent_demo_count == 160
        and ordinary_replay_size == 0
    )
    (OUT / "021_20_unit_tests.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not result["passed"]:
        raise RuntimeError(f"Unit tests failed: {result}")
    return result


class SaveAndStopCallback(BaseCallback):
    def __init__(self, destination: Path, stop_at: int, interval: int) -> None:
        super().__init__(verbose=0)
        self.destination = destination
        self.stop_at = int(stop_at)
        self.interval = int(interval)
        self.records: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        step = int(self.model.num_timesteps)
        if step % self.interval == 0:
            path = self.destination / f"checkpoint_{step}"
            path.mkdir(parents=True, exist_ok=False)
            self.model.save(str(path / "model"))
            finite = bool(all(torch.isfinite(p).all().item() for p in self.model.policy.parameters()))
            row = {
                "checkpoint": step,
                "exploration_rate": float(self.model.exploration_rate),
                "n_updates": int(self.model._n_updates),
                "parameters_finite": finite,
            }
            self.records.append(row)
            (path / "training_state.json").write_text(
                json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        return step < self.stop_at


def train_arm(label: str, batch: DemonstrationBatch, treatment: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    args = environment_args(f"training/{label}")
    env = make_env(args)
    checkpoint_root = OUT / "training" / label / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    kwargs = dqn_kwargs(seed=0)
    callback = SaveAndStopCallback(checkpoint_root, STOP_TIMESTEPS, CHECKPOINT_INTERVAL)
    pretrain_records: list[dict[str, float]] = []
    try:
        if treatment:
            model = MinimalDQfDStyle(
                "MlpPolicy", env, verbose=0,
                demo_batch_size=32, demo_margin=0.8,
                lambda_n_step=1.0, lambda_margin=1.0, lambda_l2=1e-5,
                **kwargs,
            )
            model.set_demonstrations(batch)
            initial_demo_count = model.demonstration_count
            pretrain_records = model.pretrain_demonstrations(DEMO_PRETRAIN_UPDATES)
            pd.DataFrame(pretrain_records).to_csv(
                OUT / "training" / label / "pretrain_losses.csv", index=False, encoding="utf-8-sig"
            )
        else:
            model = DQN("MlpPolicy", env, verbose=0, **kwargs)
            initial_demo_count = 0
        model.learn(
            total_timesteps=PLANNED_TIMESTEPS,
            reset_num_timesteps=True,
            progress_bar=False,
            callback=callback,
        )
        final_demo_count = int(getattr(model, "demonstration_count", 0))
        final_demo_updates = int(getattr(model, "demo_update_count", 0))
    finally:
        env.close()
    records = pd.DataFrame(callback.records)
    records.to_csv(OUT / "training" / label / "training_checkpoints.csv", index=False, encoding="utf-8-sig")
    audit = {
        "arm": label,
        "treatment": treatment,
        "planned_schedule_timesteps": PLANNED_TIMESTEPS,
        "actual_stop_timesteps": int(records.checkpoint.max()),
        "checkpoint_count": int(len(records)),
        "exploration_rate_at_5k": float(records.set_index("checkpoint").loc[5000, "exploration_rate"]),
        "all_parameters_finite": bool(records.parameters_finite.all()),
        "demonstration_count_before_training": initial_demo_count,
        "demonstration_count_after_training": final_demo_count,
        "demonstration_update_count": final_demo_updates,
        "pretrain_updates": len(pretrain_records),
    }
    audit["passed"] = bool(
        audit["actual_stop_timesteps"] == STOP_TIMESTEPS
        and audit["checkpoint_count"] == 5
        and 0.70 <= audit["exploration_rate_at_5k"] <= 0.76
        and audit["all_parameters_finite"]
        and (not treatment or (initial_demo_count == final_demo_count == 160))
    )
    (OUT / "training" / label / "training_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not audit["passed"]:
        raise RuntimeError(f"Training audit failed for {label}: {audit}")
    return records, audit


def evaluate_model(label: str, checkpoint: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    model_path = OUT / "training" / label / "checkpoints" / f"checkpoint_{checkpoint}" / "model.zip"
    model = DQN.load(str(model_path), device="cpu")
    args = environment_args(f"evaluations/{label}_checkpoint_{checkpoint}")
    env = make_env(args)
    rows: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        for step in range(420):
            latest = latest_observation_dict(env, obs, info)
            dap = int(round(scalar(latest.get("dap", step)) or 0))
            action, _ = model.predict(obs, deterministic=True)
            action = int(np.asarray(action).item())
            with torch.no_grad():
                q_values = model.q_net(torch.as_tensor(obs, dtype=torch.float32, device=model.device).reshape(1, -1))
            obs, reward, terminated, truncated, info = env.step(action)
            safe = dict(getattr(env, "last_safe_real_action", {}) or {})
            rows.append({
                "step": step, "dap_action": dap, "action": action,
                "safe_amir": float(safe.get("amir", 0.0)),
                "safe_anfer": float(safe.get("anfer", 0.0)),
                "reward": float(reward), "q_min": float(q_values.min()), "q_max": float(q_values.max()),
                "terminated": bool(terminated), "truncated": bool(truncated),
            })
            if terminated or truncated:
                break
    finally:
        destination = OUT / "evaluations" / f"{label}_checkpoint_{checkpoint}" / "pdi_tmp_snapshot_eval"
        snapshot(env, destination)
        env.close()
    frame = pd.DataFrame(rows)
    gwad, cwad = final_yield(destination)
    action_counts = frame.action.value_counts().sort_index().to_dict()
    result = {
        "arm": label, "checkpoint": checkpoint,
        "yield_kg_ha": gwad, "biomass_kg_ha": cwad,
        "irrigation_mm": float(frame.safe_amir.sum()),
        "nitrogen_kg_ha": float(frame.safe_anfer.sum()),
        "reward_total": float(frame.reward.sum()),
        "late_n_after_dap90_kg_ha": float(frame.loc[frame.dap_action > 90, "safe_anfer"].sum()),
        "q_values_finite": bool(np.isfinite(frame[["q_min", "q_max"]].to_numpy()).all()),
        "action_counts": {str(int(key)): int(value) for key, value in action_counts.items()},
    }
    eval_dir = OUT / "evaluations" / f"{label}_checkpoint_{checkpoint}"
    frame.to_csv(eval_dir / "eval_daily.csv", index=False, encoding="utf-8-sig")
    (eval_dir / "eval_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return frame, result


def run_prepare() -> dict[str, Any]:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing {OUT}")
    OUT.mkdir(parents=True)
    batch, _frame, audit = collect_demonstration()
    unit = run_unit_tests(batch)
    summary = {"phase": "prepare", "demonstration": audit, "unit_tests": unit, "passed": True}
    (OUT / "021_20_prepare_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def run_ab() -> dict[str, Any]:
    prepare = json.loads((OUT / "021_20_prepare_summary.json").read_text(encoding="utf-8"))
    if not prepare.get("passed"):
        raise RuntimeError("Prepare/unit-test phase did not pass")
    if (OUT / "training").exists() or (OUT / "evaluations").exists():
        raise FileExistsError("Refusing to overwrite an existing 021_20 A/B run")
    batch = load_demonstration()
    train_audits = {}
    for label, treatment in (("control", False), ("minimal_dqfd_style", True)):
        _records, audit = train_arm(label, batch, treatment)
        train_audits[label] = audit
    rows: list[dict[str, Any]] = []
    for label in ("control", "minimal_dqfd_style"):
        for checkpoint in range(CHECKPOINT_INTERVAL, STOP_TIMESTEPS + 1, CHECKPOINT_INTERVAL):
            _daily, result = evaluate_model(label, checkpoint)
            flat = dict(result)
            flat["action_counts"] = json.dumps(flat["action_counts"], sort_keys=True)
            rows.append(flat)
    trajectory = pd.DataFrame(rows)
    trajectory.to_csv(OUT / "021_20_ab_checkpoint_trajectory.csv", index=False, encoding="utf-8-sig")

    treatment = trajectory[trajectory.arm.eq("minimal_dqfd_style")]
    control = trajectory[trajectory.arm.eq("control")].set_index("checkpoint")
    candidates: list[dict[str, Any]] = []
    for row in treatment.to_dict("records"):
        matched = control.loc[int(row["checkpoint"])]
        strict_resource_yield = bool(
            row["yield_kg_ha"] >= 11077
            and row["irrigation_mm"] <= 90
            and row["nitrogen_kg_ha"] <= 250
            and row["late_n_after_dap90_kg_ha"] == 0
        )
        resource_better = bool(
            row["irrigation_mm"] < matched["irrigation_mm"]
            or row["nitrogen_kg_ha"] < matched["nitrogen_kg_ha"]
        )
        yield_better = bool(row["yield_kg_ha"] > matched["yield_kg_ha"])
        no_material_yield_harm = bool(row["yield_kg_ha"] >= matched["yield_kg_ha"] - 100)
        no_material_resource_harm = bool(
            row["irrigation_mm"] <= matched["irrigation_mm"] + 15
            and row["nitrogen_kg_ha"] <= matched["nitrogen_kg_ha"] + 50
        )
        beats_control = bool(
            (resource_better and no_material_yield_harm)
            or (yield_better and no_material_resource_harm)
        )
        candidates.append({
            "checkpoint": int(row["checkpoint"]),
            "strict_resource_yield": strict_resource_yield,
            "beats_control_without_material_tradeoff": beats_control,
            "passed": strict_resource_yield and beats_control and bool(row["q_values_finite"]),
        })
    candidate_frame = pd.DataFrame(candidates)
    candidate_frame.to_csv(OUT / "021_20_preregistered_candidate_checks.csv", index=False, encoding="utf-8-sig")
    passed = bool(candidate_frame.passed.any())
    summary = {
        "status": "completed",
        "method_label": "minimal DQfD-style; not a full DQfD reproduction",
        "prepare": prepare,
        "training_audits": train_audits,
        "preregistered_candidate_passed": passed,
        "passing_checkpoints": candidate_frame.loc[candidate_frame.passed, "checkpoint"].astype(int).tolist(),
        "stop_rule": "If false, do not tune margin/lambda or expand seed/training length.",
    }
    (OUT / "021_20_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "ab"), required=True)
    args = parser.parse_args()
    result = run_prepare() if args.phase == "prepare" else run_ab()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
