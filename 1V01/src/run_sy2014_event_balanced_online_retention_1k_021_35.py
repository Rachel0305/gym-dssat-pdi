from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections import deque
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_sy2014_dqfd_real_network_loss_diagnostic_021_22 as base
import run_sy2014_event_balanced_frozen_forward_021_34 as prior
import run_sy2014_fixed_observation_standardization_diagnostic_021_24 as normbase
import run_sy2014_minimal_dqfd_style_5k_ab_021_20 as oldab
from literature_aligned_dqfd import PrioritizedDemonstrationReplay, ReplaySample, arrays_sha256
from ppo_evaluate import latest_observation_dict, scalar
from run_sy2014_demo_return_to_go_ab_021_27 import full_return_to_go
from stable_baselines3 import DQN


OUT = ROOT / "benchmark_results" / "021_35"
DOC = ROOT / "docs" / "2026-07-15_021_35_sy2014_event_balanced_online_retention_1k_smoke.md"
PRIOR_VALIDATION = ROOT / "benchmark_results" / "021_34" / "021_34_validation.json"
PRIOR_SUMMARY = ROOT / "benchmark_results" / "021_34" / "021_34_frozen_policy_summary.json"
STEPS = 1000
CHECKPOINTS = [250, 500, 750, 1000]
BATCH_SIZE = 32
BETA = 0.6
EVAL_LIMIT = 420
MAX_GRAD_NORM = 10.0


def q_hash(model: DQN) -> str:
    digest = hashlib.sha256()
    for tensor in model.q_net.state_dict().values():
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_env(relative: str):
    args = json.loads(base.ENV_SOURCE.read_text(encoding="utf-8"))
    destination = OUT / relative
    destination.mkdir(parents=True, exist_ok=True)
    args["log_saving_path"] = str(destination / "pdi_gym.log")
    return base.make_env(args)


def mixed_sample(
    replay: PrioritizedDemonstrationReplay,
    demo_actions: np.ndarray,
    rng: np.random.Generator,
    update: int,
) -> ReplaySample:
    noop = np.where(demo_actions == 0)[0]
    nonzero = np.where(demo_actions != 0)[0]
    probabilities = replay.sampling_probabilities()
    demo_prob = probabilities[: replay.demo_count]
    noop_cond = demo_prob[noop] / demo_prob[noop].sum()
    chosen_noop = rng.choice(noop, size=8, replace=True, p=noop_cond)

    counts = np.ones(len(nonzero), dtype=np.int64)
    for offset in range(3):
        counts[(update - 1 + offset) % len(nonzero)] += 1
    chosen_nonzero = np.repeat(nonzero, counts)

    if replay.agent_size <= 0:
        raise RuntimeError("Agent replay is empty at online update")
    agent_global = np.arange(replay.demo_count, replay.total_size, dtype=np.int64)
    agent_cond = probabilities[replay.demo_count :] / probabilities[replay.demo_count :].sum()
    chosen_agent = rng.choice(agent_global, size=16, replace=True, p=agent_cond)
    chosen = np.concatenate([chosen_noop, chosen_nonzero, chosen_agent]).astype(np.int64)
    rng.shuffle(chosen)

    noop_lookup = {int(i): float(p) for i, p in zip(noop, noop_cond)}
    agent_lookup = {int(i): float(p) for i, p in zip(agent_global, agent_cond)}
    mixture = np.empty(BATCH_SIZE, dtype=np.float64)
    for position, index in enumerate(chosen):
        if index < replay.demo_count and demo_actions[index] == 0:
            mixture[position] = 0.25 * noop_lookup[int(index)]
        elif index < replay.demo_count:
            mixture[position] = 0.25 * 0.2
        else:
            mixture[position] = 0.50 * agent_lookup[int(index)]
    weights = (replay.total_size * mixture) ** (-BETA)
    weights /= weights.max()
    data = replay._gather(chosen)
    return ReplaySample(
        global_indices=chosen,
        probabilities=mixture,
        importance_weights=weights.astype(np.float32),
        is_demonstration=chosen < replay.demo_count,
        data=data,
    )


def online_update(
    model: DQN,
    replay: PrioritizedDemonstrationReplay,
    demo_actions: np.ndarray,
    rng: np.random.Generator,
    *,
    update: int,
    env_step: int,
    epsilon: float,
) -> dict[str, Any]:
    sample = mixed_sample(replay, demo_actions, rng, update)
    device = model.device
    observations = base.tensor(sample.data["observations"], dtype=torch.float32, device=device)
    actions = base.tensor(sample.data["actions"], dtype=torch.long, device=device).reshape(-1)
    weights = base.tensor(sample.importance_weights, dtype=torch.float32, device=device)
    demo_mask = base.tensor(sample.is_demonstration, dtype=torch.bool, device=device)
    agent_mask = ~demo_mask
    target_1, target_n = base.compute_targets(model, sample.data)
    q_values = model.q_net(observations)
    chosen_q = q_values.gather(1, actions[:, None]).squeeze(1)

    td1_each = F.smooth_l1_loss(chosen_q, target_1, reduction="none")
    td1 = (td1_each[agent_mask] * weights[agent_mask]).mean()
    tdn_each = F.smooth_l1_loss(chosen_q, target_n, reduction="none")
    tdn = (tdn_each * weights).mean()
    margins = torch.full_like(q_values, 0.8)
    margins.scatter_(1, actions[:, None], 0.0)
    margin_each = torch.max(q_values + margins, dim=1).values - chosen_q
    margin = (margin_each[demo_mask] * weights[demo_mask]).mean()
    parameters = list(model.q_net.parameters())
    l2 = 1e-5 * sum(parameter.square().sum() for parameter in parameters)
    total = td1 + tdn + margin + l2

    model.policy.optimizer.zero_grad()
    total.backward()
    gradient = float(torch.nn.utils.clip_grad_norm_(parameters, MAX_GRAD_NORM))
    model.policy.optimizer.step()
    with torch.no_grad():
        errors = torch.abs(target_1 - chosen_q).detach().cpu().numpy()
    priority_updates: dict[int, float] = {}
    for index, error in zip(sample.global_indices, errors):
        priority_updates[int(index)] = max(priority_updates.get(int(index), 0.0), float(error))
    replay.update_priorities(priority_updates.keys(), priority_updates.values())
    values = {
        "td1_agent_only": float(td1.detach().cpu()),
        "tdn_all": float(tdn.detach().cpu()),
        "margin_demo_only": float(margin.detach().cpu()),
        "l2": float(l2.detach().cpu()),
        "total": float(total.detach().cpu()),
    }
    return {
        "update": update, "env_step": env_step, "epsilon": epsilon,
        "sample_demo_count": int(demo_mask.sum().item()),
        "sample_agent_count": int(agent_mask.sum().item()),
        "sample_demo_noop_count": int(((actions == 0) & demo_mask).sum().item()),
        "sample_demo_nonzero_count": int(((actions != 0) & demo_mask).sum().item()),
        "replay_agent_count": replay.agent_size,
        **values,
        "gradient_before_clip": gradient,
        "gradient_would_clip": gradient > MAX_GRAD_NORM,
        "q_abs_max": float(q_values.detach().abs().max().cpu()),
        "all_finite": bool(
            all(np.isfinite(value) for value in values.values())
            and np.isfinite(gradient)
            and torch.isfinite(q_values).all().item()
            and all(torch.isfinite(parameter).all().item() for parameter in parameters)
        ),
    }


def train_online(
    *, action_seed: int = 0, sample_seed: int = 21035,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    model, _old_replay, validation, reproduction_curve, _offline_updates = prior.build_model()
    prior_check = json.loads(PRIOR_VALIDATION.read_text(encoding="utf-8"))
    expected_hash = prior_check["parameter_hash_after_forward"]
    initial_hash = q_hash(model)
    if initial_hash != expected_hash:
        raise RuntimeError(f"021_34 start hash mismatch: {initial_hash} != {expected_hash}")

    raw, standardized, scaler = normbase.prepare_demonstrations()
    if not scaler["all_pre_run_checks_pass"]:
        raise RuntimeError("Standardization validation failed")
    demonstrations = full_return_to_go(standardized)
    demo_actions = np.asarray(demonstrations["actions"], dtype=np.int64).reshape(-1)
    replay = PrioritizedDemonstrationReplay(
        demonstrations, agent_capacity=20_000, alpha=0.4,
        epsilon_demo=1.0, epsilon_agent=0.001, seed=21022,
    )
    demo_hash_before = arrays_sha256(replay.demonstrations)
    model.q_net_target.load_state_dict(model.q_net.state_dict())
    hash_after_target_sync = q_hash(model)
    if hash_after_target_sync != initial_hash:
        raise RuntimeError("Target sync unexpectedly changed online Q")

    env = make_env("training/environment")
    action_rng = np.random.default_rng(action_seed)
    sample_rng = np.random.default_rng(sample_seed)
    pending: deque[dict[str, Any]] = deque()
    interactions: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    checkpoint_dir = OUT / "training" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(checkpoint_dir / "checkpoint_0"))
    try:
        raw_obs, info = env.reset()
        for step in range(1, STEPS + 1):
            latest = latest_observation_dict(env, raw_obs, info)
            dap = int(round(scalar(latest.get("dap", step)) or 0))
            obs = normbase.normalize(raw_obs, scaler["mean"], scaler["scale"])
            epsilon = base.epsilon_at(step)
            if action_rng.random() < epsilon:
                action = int(action_rng.integers(0, 9)); source = "epsilon_random"
            else:
                with torch.no_grad():
                    q = model.q_net(torch.as_tensor(obs, dtype=torch.float32, device=model.device).reshape(1, -1))
                action = int(q.argmax(dim=1).item()); source = "greedy"
            next_raw, reward, terminated, truncated, info = env.step(action)
            next_obs = normbase.normalize(next_raw, scaler["mean"], scaler["scale"])
            done = bool(terminated or truncated)
            safe = dict(getattr(env, "last_safe_real_action", {}) or {})
            pending.append({
                "observation": obs.copy(), "action": action, "reward": float(reward),
                "next_observation": next_obs.copy(), "done": done,
            })
            added = 0
            while len(pending) >= base.N_STEP or (done and pending):
                transition = base.matured_transition(pending)
                replay.add_agent(transition, td_error=base.initial_td_error(model, transition))
                pending.popleft(); added += 1
                if not done and len(pending) < base.N_STEP:
                    break
            if step >= base.LEARNING_STARTS:
                updates.append(online_update(
                    model, replay, demo_actions, sample_rng,
                    update=step - base.LEARNING_STARTS + 1,
                    env_step=step, epsilon=epsilon,
                ))
            interactions.append({
                "env_step": step, "episode_dap": dap, "epsilon": epsilon,
                "action": action, "action_source": source,
                "safe_amir": float(safe.get("amir", 0.0)),
                "safe_anfer": float(safe.get("anfer", 0.0)),
                "reward": float(reward), "done": done,
                "agent_transitions_added": added, "replay_agent_count": replay.agent_size,
            })
            if step in CHECKPOINTS:
                model.save(str(checkpoint_dir / f"checkpoint_{step}"))
            if done and step < STEPS:
                raw_obs, info = env.reset(); pending.clear()
            else:
                raw_obs = next_raw
    finally:
        tmp = getattr(env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            shutil.copytree(Path(tmp), OUT / "training" / "pdi_tmp_snapshot_after_training")
        env.close()

    interactions_frame = pd.DataFrame(interactions)
    updates_frame = pd.DataFrame(updates)
    demo_hash_after = arrays_sha256(replay.demonstrations)
    audit = {
        "initial_q_hash": initial_hash,
        "expected_021_34_q_hash": expected_hash,
        "initial_hash_matches": initial_hash == expected_hash,
        "target_sync_changed_online_q": hash_after_target_sync != initial_hash,
        "demonstration_hash_before": demo_hash_before,
        "demonstration_hash_after": demo_hash_after,
        "demonstrations_unchanged": demo_hash_before == demo_hash_after,
        "replay_agent_count_final": replay.agent_size,
        "all_batches_16_demo_16_agent": bool(
            (updates_frame.sample_demo_count == 16).all()
            and (updates_frame.sample_agent_count == 16).all()
        ),
        "all_demo_halves_8_noop_8_nonzero": bool(
            (updates_frame.sample_demo_noop_count == 8).all()
            and (updates_frame.sample_demo_nonzero_count == 8).all()
        ),
        "all_updates_finite": bool(updates_frame.all_finite.all()),
        "gradient_clip_fraction": float(updates_frame.gradient_would_clip.mean()),
        "reproduction_500_nonzero_recall": float(reproduction_curve.iloc[-1].nonzero_action_recall),
        "online_action_seed": int(action_seed),
        "online_sample_seed": int(sample_seed),
    }
    return interactions_frame, updates_frame, audit, scaler


def evaluate_checkpoint(checkpoint: int, mean: np.ndarray, scale: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
    model = DQN.load(str(OUT / "training" / "checkpoints" / f"checkpoint_{checkpoint}.zip"), device="cpu")
    env = make_env(f"evaluations/checkpoint_{checkpoint}")
    rows: list[dict[str, Any]] = []
    destination = OUT / "evaluations" / f"checkpoint_{checkpoint}" / "pdi_tmp_snapshot_eval"
    try:
        raw_obs, info = env.reset()
        for step in range(EVAL_LIMIT):
            latest = latest_observation_dict(env, raw_obs, info)
            dap = int(round(scalar(latest.get("dap", step)) or 0))
            obs = normbase.normalize(raw_obs, mean, scale)
            with torch.no_grad():
                q = model.q_net(torch.as_tensor(obs, dtype=torch.float32, device=model.device).reshape(1, -1))
            action = int(q.argmax(dim=1).item())
            raw_obs, reward, terminated, truncated, info = env.step(action)
            safe = dict(getattr(env, "last_safe_real_action", {}) or {})
            rows.append({
                "checkpoint": checkpoint, "step": step, "dap_action": dap,
                "action": action, "safe_amir": float(safe.get("amir", 0.0)),
                "safe_anfer": float(safe.get("anfer", 0.0)), "reward": float(reward),
                "q_min": float(q.min()), "q_max": float(q.max()),
                "terminated": bool(terminated), "truncated": bool(truncated),
            })
            if terminated or truncated:
                break
    finally:
        oldab.snapshot(env, destination)
        env.close()
    frame = pd.DataFrame(rows)
    gwad, cwad = oldab.final_yield(destination)
    result = {
        "checkpoint": checkpoint, "yield_kg_ha": gwad, "biomass_kg_ha": cwad,
        "irrigation_mm": float(frame.safe_amir.sum()),
        "nitrogen_kg_ha": float(frame.safe_anfer.sum()),
        "late_n_after_dap90_kg_ha": float(frame.loc[frame.dap_action > 90, "safe_anfer"].sum()),
        "nonzero_operation_count": int((frame.safe_amir.gt(0) | frame.safe_anfer.gt(0)).sum()),
        "reward_total": float(frame.reward.sum()),
        "q_values_finite": bool(np.isfinite(frame[["q_min", "q_max"]].to_numpy()).all()),
        "terminated_or_truncated": bool(frame.iloc[-1].terminated or frame.iloc[-1].truncated),
    }
    result["expert_efficiency_gate"] = bool(
        gwad >= 11077 and result["irrigation_mm"] <= 120
        and result["nitrogen_kg_ha"] <= 300
        and result["late_n_after_dap90_kg_ha"] == 0
    )
    frame.to_csv(OUT / "evaluations" / f"checkpoint_{checkpoint}" / "daily.csv", index=False, encoding="utf-8-sig")
    return frame, result


def plot_results(trajectory: pd.DataFrame, updates: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    axes[0, 0].plot(trajectory.checkpoint, trajectory.yield_kg_ha, marker="o", color="#0072B2")
    axes[0, 0].axhline(11077, color="#D55E00", linestyle="--", label="official expert")
    axes[0, 0].axhline(9613, color="#333333", linestyle=":", label="recorded")
    axes[0, 0].set_ylabel("Yield (kg/ha)"); axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(trajectory.checkpoint, trajectory.irrigation_mm, marker="o", label="Irrigation", color="#0072B2")
    axes[0, 1].plot(trajectory.checkpoint, trajectory.nitrogen_kg_ha, marker="o", label="Nitrogen", color="#D55E00")
    axes[0, 1].set_ylabel("Seasonal input"); axes[0, 1].legend(frameon=False)
    axes[1, 0].plot(trajectory.checkpoint, trajectory.late_n_after_dap90_kg_ha, marker="o", color="#CC79A7")
    axes[1, 0].set_ylabel("N after DAP90 (kg/ha)")
    axes[1, 1].plot(updates.env_step, updates.total, color="#009E73", alpha=0.8, label="Total loss")
    axes[1, 1].plot(updates.env_step, updates.gradient_before_clip, color="#E69F00", alpha=0.7, label="Gradient")
    axes[1, 1].set_yscale("symlog", linthresh=1e-3); axes[1, 1].legend(frameon=False)
    for ax in axes.flat:
        ax.grid(alpha=0.2); ax.set_xlabel("Online environment steps")
    fig.suptitle("SY2014 online retention smoke from 021_34 frozen network")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "021_35_online_retention_1k.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "021_35_online_retention_1k.svg", bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    if DOC.exists():
        raise FileExistsError(f"Refusing to overwrite {DOC}")
    OUT.mkdir(parents=True)
    interactions, updates, audit, scaler = train_online()
    interactions.to_csv(OUT / "021_35_training_interactions.csv", index=False, encoding="utf-8-sig")
    updates.to_csv(OUT / "021_35_online_update_log.csv", index=False, encoding="utf-8-sig")

    prior_summary = json.loads(PRIOR_SUMMARY.read_text(encoding="utf-8"))
    initial = {
        "checkpoint": 0,
        "yield_kg_ha": prior_summary["yield_kg_ha"],
        "biomass_kg_ha": prior_summary["biomass_kg_ha"],
        "irrigation_mm": prior_summary["irrigation_mm"],
        "nitrogen_kg_ha": prior_summary["nitrogen_kg_ha"],
        "late_n_after_dap90_kg_ha": prior_summary["late_n_after_dap90_kg_ha"],
        "nonzero_operation_count": prior_summary["nonzero_operation_count"],
        "reward_total": prior_summary["reward_total"],
        "q_values_finite": prior_summary["q_values_finite"],
        "terminated_or_truncated": prior_summary["terminated_or_truncated"],
        "expert_efficiency_gate": True,
        "source": "reused_021_34",
    }
    rows = [initial]
    for checkpoint in CHECKPOINTS:
        _daily, result = evaluate_checkpoint(checkpoint, scaler["mean"], scaler["scale"])
        result["source"] = "021_35_deterministic_evaluation"
        rows.append(result)
    trajectory = pd.DataFrame(rows)
    trajectory.to_csv(OUT / "021_35_checkpoint_trajectory.csv", index=False, encoding="utf-8-sig")

    online = trajectory[trajectory.checkpoint > 0]
    pass_count = int(online.expert_efficiency_gate.sum())
    final = online.loc[online.checkpoint == 1000].iloc[0]
    if pass_count >= 3 and bool(final.expert_efficiency_gate) and float(online.yield_kg_ha.min()) >= 9613:
        branch = "A"
    elif pass_count >= 1 or float(final.yield_kg_ha) >= 9613:
        branch = "B"
    else:
        branch = "C"
    validation = {
        **audit,
        "all_evaluations_finite": bool(trajectory.q_values_finite.all()),
        "all_evaluations_terminated": bool(trajectory.terminated_or_truncated.all()),
        "online_checkpoint_pass_count": pass_count,
        "final_checkpoint_passed": bool(final.expert_efficiency_gate),
        "all_required_checks_pass": bool(
            audit["initial_hash_matches"] and not audit["target_sync_changed_online_q"]
            and audit["demonstrations_unchanged"] and audit["replay_agent_count_final"] > 0
            and audit["all_batches_16_demo_16_agent"]
            and audit["all_demo_halves_8_noop_8_nonzero"] and audit["all_updates_finite"]
            and trajectory.q_values_finite.all() and trajectory.terminated_or_truncated.all()
        ),
    }
    (OUT / "021_35_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    summary = {
        "status": "completed" if validation["all_required_checks_pass"] else "failed_validation",
        "branch": branch,
        "pass_count_out_of_4": pass_count,
        "final_checkpoint": final.to_dict(),
        "online_5k_started": False,
        "interpretation": {
            "A": "1K在线更新总体保持了021_34优质策略，可另立5K与多seed任务。",
            "B": "在线更新只在部分检查点保持优质策略，尚不能放大训练。",
            "C": "在线更新未保持优质起点，当前配置停止，不进入5K。",
        }[branch],
    }
    (OUT / "021_35_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    plot_results(trajectory, updates)

    display = trajectory[[
        "checkpoint", "yield_kg_ha", "irrigation_mm", "nitrogen_kg_ha",
        "late_n_after_dap90_kg_ha", "expert_efficiency_gate",
    ]].round(3)
    doc = f"""# 021_35 SY2014 event-balanced 网络在线保持性 1K smoke 记录

## 目的与边界

从021_34逐位复现的优质冻结网络出发，只增加在线交互和梯度更新。仅seed0、1K；未改reward、IC、动作、预算和DSSAT输入，未启动5K。

## 实现验证

- 起点Q哈希与021_34一致：{audit['initial_hash_matches']}。
- target同步没有改变online Q：{not audit['target_sync_changed_online_q']}。
- demonstration训练前后哈希一致：{audit['demonstrations_unchanged']}。
- 每批16 demo/16 agent：{audit['all_batches_16_demo_16_agent']}；demo内8 no-op/8 nonzero：{audit['all_demo_halves_8_noop_8_nonzero']}。
- 最终agent replay数量：{audit['replay_agent_count_final']}；全部更新有限：{audit['all_updates_finite']}。
- 梯度裁剪比例：{audit['gradient_clip_fraction']:.3f}。

## 确定性检查点结果

{markdown_table(display)}

## 预注册判定

- 在线4个检查点通过数：{pass_count}/4。
- 分支：**{branch}**。
- {summary['interpretation']}

## 限制

这是1K seed0 smoke，不代表跨seed稳定性。demonstration保持full return-to-go，agent transition使用冻结5-step回报；这是预注册的在线衔接设计，不应误写成完整原版DQfD复现。

## 输出

- `benchmark_results/021_35/021_35_training_interactions.csv`
- `benchmark_results/021_35/021_35_online_update_log.csv`
- `benchmark_results/021_35/021_35_checkpoint_trajectory.csv`
- `benchmark_results/021_35/021_35_validation.json`
- `benchmark_results/021_35/021_35_summary.json`
- `benchmark_results/021_35/021_35_online_retention_1k.png/.svg`
"""
    DOC.write_text(doc, encoding="utf-8")
    print(json.dumps({"status": summary["status"], "branch": branch, "trajectory": trajectory.to_dict("records"), "validation": validation}, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
