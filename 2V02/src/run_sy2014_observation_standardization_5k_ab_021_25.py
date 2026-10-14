from __future__ import annotations

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


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_sy2014_dqfd_real_network_loss_diagnostic_021_22 as base
import run_sy2014_fixed_observation_standardization_diagnostic_021_24 as normbase
import run_sy2014_minimal_dqfd_style_5k_ab_021_20 as oldab
from frozen_nstep_dqn_config_020_11 import dqn_kwargs
from literature_aligned_dqfd import PrioritizedDemonstrationReplay, dqfd_loss_components
from ppo_evaluate import latest_observation_dict, scalar
from stable_baselines3 import DQN


OUT = ROOT / "benchmark_results" / "021_25"
DOC = ROOT / "docs" / "2026-07-15_021_25_sy2014_observation_standardization_5k_ab.md"
STOP_STEPS = 5_000
CHECKPOINT_INTERVAL = 1_000
EVAL_LIMIT = 420


def env_args(relative: str) -> dict[str, Any]:
    args = json.loads(base.ENV_SOURCE.read_text(encoding="utf-8"))
    destination = OUT / relative
    destination.mkdir(parents=True, exist_ok=True)
    args["log_saving_path"] = str(destination / "pdi_gym.log")
    return args


def transformed_demo(standardized: bool) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, Any]]:
    raw, transformed, validation = normbase.prepare_demonstrations()
    demo = transformed if standardized else {name: np.asarray(value).copy() for name, value in raw.items()}
    return demo, validation["mean"], validation["scale"], validation


def transform_obs(value: np.ndarray, standardized: bool, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return normbase.normalize(value, mean, scale) if standardized else np.asarray(value, dtype=np.float32).copy()


def fast_update(
    model: DQN,
    replay: PrioritizedDemonstrationReplay,
    *,
    phase: str,
    update: int,
    env_step: int,
    epsilon: float,
) -> dict[str, Any]:
    sample = replay.sample(base.BATCH_SIZE, beta=0.6)
    device = model.device
    observations = base.tensor(sample.data["observations"], dtype=torch.float32, device=device)
    actions = base.tensor(sample.data["actions"], dtype=torch.long, device=device).reshape(-1)
    weights = base.tensor(sample.importance_weights, dtype=torch.float32, device=device)
    demo_mask = base.tensor(sample.is_demonstration, dtype=torch.bool, device=device)
    target_1, target_n = base.compute_targets(model, sample.data)
    q_values = model.q_net(observations)
    parameters = list(model.q_net.parameters())
    losses = dqfd_loss_components(
        q_values=q_values,
        actions=actions,
        target_1_step=target_1,
        target_n_step=target_n,
        importance_weights=weights,
        demonstration_mask=demo_mask,
        l2_parameters=parameters,
        margin=0.8,
        lambda_n_step=1.0,
        lambda_margin=1.0,
        lambda_l2=1e-5,
    )
    model.policy.optimizer.zero_grad()
    losses["total"].backward()
    total_gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, base.MAX_GRAD_NORM))
    model.policy.optimizer.step()
    with torch.no_grad():
        chosen_q = q_values.gather(1, actions[:, None]).squeeze(1)
        td_errors = torch.abs(target_1 - chosen_q).detach().cpu().numpy()
    priority_updates: dict[int, float] = {}
    for index, error in zip(sample.global_indices, td_errors):
        priority_updates[int(index)] = max(priority_updates.get(int(index), 0.0), float(error))
    replay.update_priorities(priority_updates.keys(), priority_updates.values())
    values = {name: float(value.detach().cpu()) for name, value in losses.items()}
    all_finite = bool(
        all(np.isfinite(value) for value in values.values())
        and np.isfinite(total_gradient_norm)
        and torch.isfinite(q_values).all().item()
        and all(torch.isfinite(parameter).all().item() for parameter in parameters)
    )
    return {
        "phase": phase, "update": int(update), "env_step": int(env_step), "epsilon": float(epsilon),
        "sample_demo_count": int(sample.is_demonstration.sum()),
        "sample_agent_count": int((~sample.is_demonstration).sum()),
        "replay_demo_count": replay.demo_count, "replay_agent_count": replay.agent_size,
        **values,
        "grad_total_before_clip": total_gradient_norm,
        "gradient_would_clip": total_gradient_norm > base.MAX_GRAD_NORM,
        "q_abs_mean": float(q_values.detach().abs().mean().cpu()),
        "q_abs_max": float(q_values.detach().abs().max().cpu()),
        "all_finite": all_finite,
    }


def run_arm(label: str, standardized: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    np.random.seed(0)
    torch.manual_seed(0)
    action_rng = np.random.default_rng(0)
    demo, mean, scale, validation = transformed_demo(standardized)
    replay = PrioritizedDemonstrationReplay(
        demo, agent_capacity=20_000, alpha=0.4,
        epsilon_demo=1.0, epsilon_agent=0.001, seed=21022,
    )
    env = base.make_env(env_args(f"training/{label}/environment"))
    model = DQN("MlpPolicy", env, verbose=0, **dqn_kwargs(seed=0))
    parameters = list(model.q_net.parameters())
    initial_parameters = base.parameter_vector(parameters)
    diagnostic_rows: list[dict[str, Any]] = []
    for update in range(1, base.PRETRAIN_UPDATES + 1):
        diagnostic_rows.append(fast_update(
            model, replay, phase="pretrain", update=update, env_step=0, epsilon=1.0
        ))
    model.q_net_target.load_state_dict(model.q_net.state_dict())
    online_start_parameters = base.parameter_vector(parameters)

    interactions: list[dict[str, Any]] = []
    pending: deque[dict[str, Any]] = deque()
    checkpoints = OUT / "training" / label / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=False)
    try:
        raw_obs, info = env.reset()
        for step in range(1, STOP_STEPS + 1):
            latest = latest_observation_dict(env, raw_obs, info)
            dap = int(round(scalar(latest.get("dap", step)) or 0))
            obs = transform_obs(raw_obs, standardized, mean, scale)
            epsilon = base.epsilon_at(step)
            if action_rng.random() < epsilon:
                action = int(action_rng.integers(0, 9)); action_source = "epsilon_random"
            else:
                with torch.no_grad():
                    q_values = model.q_net(torch.as_tensor(obs, dtype=torch.float32, device=model.device).reshape(1, -1))
                action = int(q_values.argmax(dim=1).item()); action_source = "greedy"
            next_raw_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = transform_obs(next_raw_obs, standardized, mean, scale)
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
                diagnostic_rows.append(fast_update(
                    model, replay, phase="online", update=step - base.LEARNING_STARTS + 1,
                    env_step=step, epsilon=epsilon,
                ))
            interactions.append({
                "arm": label, "env_step": step, "episode_dap": dap, "epsilon": epsilon,
                "action": action, "action_source": action_source,
                "safe_amir": float(safe.get("amir", 0.0)),
                "safe_anfer": float(safe.get("anfer", 0.0)),
                "reward": float(reward), "done": done,
                "agent_transitions_added": added, "replay_agent_count": replay.agent_size,
            })
            if step % CHECKPOINT_INTERVAL == 0:
                model.save(str(checkpoints / f"checkpoint_{step}"))
            if done and step < STOP_STEPS:
                raw_obs, info = env.reset()
                pending.clear()
            else:
                raw_obs = next_raw_obs
    finally:
        tmp = getattr(env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            shutil.copytree(Path(tmp), OUT / "training" / label / "pdi_tmp_snapshot_after_training")
        env.close()

    final_parameters = base.parameter_vector(parameters)
    cumulative_relative_l2 = float(torch.linalg.vector_norm(final_parameters - online_start_parameters)) / max(
        float(torch.linalg.vector_norm(online_start_parameters)), 1e-12
    )
    diagnostics = pd.DataFrame(diagnostic_rows)
    interaction_frame = pd.DataFrame(interactions)
    diagnostics.to_csv(OUT / "training" / label / "training_diagnostics.csv", index=False, encoding="utf-8-sig")
    interaction_frame.to_csv(OUT / "training" / label / "training_interactions.csv", index=False, encoding="utf-8-sig")
    online = diagnostics[diagnostics.phase == "online"]
    audit = {
        "arm": label, "standardized": standardized,
        "steps": STOP_STEPS, "pretrain_updates": base.PRETRAIN_UPDATES,
        "checkpoint_count": len(list(checkpoints.glob("checkpoint_*.zip"))),
        "all_finite": bool(diagnostics.all_finite.all()),
        "median_online_gradient": float(online.grad_total_before_clip.median()),
        "max_online_gradient": float(online.grad_total_before_clip.max()),
        "online_gradient_clip_fraction": float(online.gradient_would_clip.mean()),
        "online_cumulative_parameter_relative_l2": cumulative_relative_l2,
        "reward_was_scaled": False,
        "standardization_validation_passed": bool(validation["all_pre_run_checks_pass"]),
    }
    (OUT / "training" / label / "training_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return diagnostics, interaction_frame, audit


def evaluate(label: str, checkpoint: int, standardized: bool, mean: np.ndarray, scale: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
    model = DQN.load(str(OUT / "training" / label / "checkpoints" / f"checkpoint_{checkpoint}.zip"), device="cpu")
    env = base.make_env(env_args(f"evaluations/{label}_checkpoint_{checkpoint}"))
    rows: list[dict[str, Any]] = []
    destination = OUT / "evaluations" / f"{label}_checkpoint_{checkpoint}" / "pdi_tmp_snapshot_eval"
    try:
        raw_obs, info = env.reset()
        for step in range(EVAL_LIMIT):
            latest = latest_observation_dict(env, raw_obs, info)
            dap = int(round(scalar(latest.get("dap", step)) or 0))
            obs = transform_obs(raw_obs, standardized, mean, scale)
            with torch.no_grad():
                q = model.q_net(torch.as_tensor(obs, dtype=torch.float32, device=model.device).reshape(1, -1))
            action = int(q.argmax(dim=1).item())
            raw_obs, reward, terminated, truncated, info = env.step(action)
            safe = dict(getattr(env, "last_safe_real_action", {}) or {})
            rows.append({
                "step": step, "dap_action": dap, "action": action,
                "safe_amir": float(safe.get("amir", 0.0)),
                "safe_anfer": float(safe.get("anfer", 0.0)),
                "reward": float(reward), "q_min": float(q.min()), "q_max": float(q.max()),
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
        "arm": label, "checkpoint": checkpoint,
        "yield_kg_ha": gwad, "biomass_kg_ha": cwad,
        "irrigation_mm": float(frame.safe_amir.sum()),
        "nitrogen_kg_ha": float(frame.safe_anfer.sum()),
        "late_n_after_dap90_kg_ha": float(frame.loc[frame.dap_action > 90, "safe_anfer"].sum()),
        "reward_total": float(frame.reward.sum()),
        "q_abs_max": float(frame[["q_min", "q_max"]].abs().max().max()),
        "q_values_finite": bool(np.isfinite(frame[["q_min", "q_max"]].to_numpy()).all()),
        "nonzero_action_count": int((frame.safe_amir.gt(0) | frame.safe_anfer.gt(0)).sum()),
    }
    eval_dir = OUT / "evaluations" / f"{label}_checkpoint_{checkpoint}"
    frame.to_csv(eval_dir / "eval_daily.csv", index=False, encoding="utf-8-sig")
    (eval_dir / "eval_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return frame, result


def plot_summary(trajectory: pd.DataFrame, audits: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    colors = {"control_raw_observation": "#777777", "treatment_standardized_observation": "#009E73"}
    for arm, frame in trajectory.groupby("arm"):
        axes[0, 0].plot(frame.checkpoint, frame.yield_kg_ha, marker="o", color=colors[arm], label=arm)
        axes[0, 1].plot(frame.checkpoint, frame.irrigation_mm, marker="o", color=colors[arm], label=arm)
        axes[1, 0].plot(frame.checkpoint, frame.nitrogen_kg_ha, marker="o", color=colors[arm], label=arm)
    axes[0, 0].axhline(11077, color="#D55E00", linestyle="--", linewidth=1, label="official expert yield")
    axes[0, 0].set_ylabel("Yield (kg/ha)"); axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].set_ylabel("Irrigation (mm)"); axes[1, 0].set_ylabel("Nitrogen (kg/ha)")
    positions = np.arange(len(audits)); width = 0.35
    axes[1, 1].bar(positions - width/2, audits.median_online_gradient, width, color="#4C78A8", label="median gradient")
    axes[1, 1].bar(positions + width/2, audits.online_gradient_clip_fraction, width, color="#F58518", label="clip fraction")
    axes[1, 1].set_yscale("symlog", linthresh=1e-3)
    axes[1, 1].set_xticks(positions, ["control", "standardized"])
    axes[1, 1].legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.2); ax.set_xlabel("Training checkpoint" if ax is not axes[1, 1] else "Arm")
    fig.suptitle("SY2014 5K observation-standardization A/B")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "021_25_observation_standardization_5k_ab.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "021_25_observation_standardization_5k_ab.svg", bbox_inches="tight")
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
    OUT.mkdir(parents=True)
    raw_demo, mean, scale, validation = transformed_demo(False)
    if not validation["all_pre_run_checks_pass"]:
        raise RuntimeError("021_24 scaler validation did not pass")
    pd.DataFrame({
        "observation_index": np.arange(25), "observation_label": normbase.LABELS,
        "mean": mean, "scale": scale,
    }).to_csv(OUT / "021_25_frozen_observation_scaler.csv", index=False, encoding="utf-8-sig")

    arm_specs = [
        ("control_raw_observation", False),
        ("treatment_standardized_observation", True),
    ]
    audits = []
    for label, standardized in arm_specs:
        _diagnostics, _interactions, audit = run_arm(label, standardized)
        audits.append(audit)
    audit_frame = pd.DataFrame(audits)
    audit_frame.to_csv(OUT / "021_25_training_audits.csv", index=False, encoding="utf-8-sig")

    eval_rows = []
    for label, standardized in arm_specs:
        for checkpoint in range(CHECKPOINT_INTERVAL, STOP_STEPS + 1, CHECKPOINT_INTERVAL):
            _daily, result = evaluate(label, checkpoint, standardized, mean, scale)
            eval_rows.append(result)
    trajectory = pd.DataFrame(eval_rows)
    trajectory.to_csv(OUT / "021_25_checkpoint_trajectory.csv", index=False, encoding="utf-8-sig")

    control = trajectory[trajectory.arm == "control_raw_observation"].set_index("checkpoint")
    treatment = trajectory[trajectory.arm == "treatment_standardized_observation"].copy()
    checks = []
    for row in treatment.itertuples():
        matched = control.loc[row.checkpoint]
        agronomic = bool(
            row.yield_kg_ha >= 11077 and row.irrigation_mm <= 90
            and row.nitrogen_kg_ha <= 250 and row.late_n_after_dap90_kg_ha == 0
        )
        matched_control = bool(row.yield_kg_ha >= float(matched.yield_kg_ha) - 100)
        checks.append({
            "checkpoint": row.checkpoint, "agronomic_gate": agronomic,
            "no_material_yield_loss_vs_control": matched_control,
            "checkpoint_pass": agronomic and matched_control and row.q_values_finite,
        })
    check_frame = pd.DataFrame(checks)
    pass_values = check_frame.checkpoint_pass.tolist()
    consecutive_two = any(pass_values[i] and pass_values[i + 1] for i in range(len(pass_values) - 1))
    passed_checkpoints = check_frame.loc[check_frame.checkpoint_pass, "checkpoint"].tolist()
    if passed_checkpoints:
        first = min(passed_checkpoints)
        no_later_collapse = bool(treatment.loc[treatment.checkpoint >= first, "yield_kg_ha"].min() >= 9000)
    else:
        no_later_collapse = False
    check_frame.to_csv(OUT / "021_25_preregistered_checks.csv", index=False, encoding="utf-8-sig")

    audit_by_arm = audit_frame.set_index("arm")
    control_grad = float(audit_by_arm.loc["control_raw_observation", "median_online_gradient"])
    treatment_grad = float(audit_by_arm.loc["treatment_standardized_observation", "median_online_gradient"])
    treatment_clip = float(audit_by_arm.loc["treatment_standardized_observation", "online_gradient_clip_fraction"])
    numeric_pass = bool(
        treatment_grad <= 0.25 * control_grad and treatment_clip < 0.10
        and audit_by_arm.loc["treatment_standardized_observation", "all_finite"]
    )
    agronomic_pass = bool(consecutive_two and no_later_collapse)
    branch = "A_numeric_and_agronomic_pass" if numeric_pass and agronomic_pass else (
        "B_numeric_pass_agronomic_fail" if numeric_pass else "C_numeric_gate_fail"
    )
    summary = {
        "status": "completed", "no_training_beyond_5k": True,
        "only_scientific_variable": "fixed_observation_standardization",
        "numeric_gate_passed": numeric_pass, "agronomic_gate_passed": agronomic_pass,
        "consecutive_two_passing_checkpoints": consecutive_two,
        "no_later_collapse_after_first_pass": no_later_collapse,
        "passing_checkpoints": passed_checkpoints,
        "pre_registered_interpretation_branch": branch,
        "control_median_online_gradient": control_grad,
        "treatment_median_online_gradient": treatment_grad,
        "treatment_to_control_gradient_ratio": treatment_grad / max(control_grad, 1e-12),
        "treatment_gradient_clip_fraction": treatment_clip,
    }
    (OUT / "021_25_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_summary(trajectory, audit_frame)

    display = trajectory[[
        "arm", "checkpoint", "yield_kg_ha", "biomass_kg_ha", "irrigation_mm",
        "nitrogen_kg_ha", "late_n_after_dap90_kg_ha", "reward_total",
    ]].round(3)
    record = f"""# 021_25 SY2014 固定观测标准化 5K 严格 A/B 记录

## 设计

Control 和 Treatment 顺序运行；唯一科学变量是是否应用 021_24 冻结的固定观测标准化。两组均使用原始 reward、相同 seed、示范、literature-aligned DQfD 组件和 5K/1K checkpoint 协议。没有训练超过 5K。

## Checkpoint 结果

{markdown_table(display)}

## 训练数值

{markdown_table(audit_frame[["arm", "median_online_gradient", "max_online_gradient", "online_gradient_clip_fraction", "online_cumulative_parameter_relative_l2", "all_finite"]].round(6))}

## 预注册判定

- numeric gate: `{numeric_pass}`
- agronomic gate: `{agronomic_pass}`
- consecutive two passing checkpoints: `{consecutive_two}`
- no later collapse: `{no_later_collapse}`
- branch: `{branch}`

本任务不因结果好坏自动延长到 25K，也不现场修改标准化、reward 或 DQfD 参数。

后续可使用独立离线脚本审计 checkpoint 对 5 个稀疏非零示范动作的召回率；该审计不得回写本次预注册门槛或改变 A/B 结果。

## 输出

- `benchmark_results/021_25/training/`
- `benchmark_results/021_25/evaluations/`
- `benchmark_results/021_25/021_25_checkpoint_trajectory.csv`
- `benchmark_results/021_25/021_25_training_audits.csv`
- `benchmark_results/021_25/021_25_preregistered_checks.csv`
- `benchmark_results/021_25/021_25_summary.json`
- `benchmark_results/021_25/021_25_observation_standardization_5k_ab.png/.svg`
"""
    DOC.write_text(record, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
