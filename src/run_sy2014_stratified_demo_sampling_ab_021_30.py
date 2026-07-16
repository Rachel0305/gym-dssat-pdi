from __future__ import annotations

import json
import sys
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
from frozen_nstep_dqn_config_020_11 import dqn_kwargs
from literature_aligned_dqfd import (
    PrioritizedDemonstrationReplay,
    ReplaySample,
    dqfd_loss_components,
)
from run_sy2014_demo_return_to_go_ab_021_27 import full_return_to_go
from run_sy2014_standardized_demo_pretraining_curve_021_26 import (
    OfflineShapeEnv,
    evaluate_demonstrations,
)
from stable_baselines3 import DQN


OUT = ROOT / "benchmark_results" / "021_30"
DOC = ROOT / "docs" / "2026-07-15_021_30_sy2014_stratified_demonstration_sampling_offline_ab.md"
MILESTONES = [0, 10, 25, 50, 100, 250, 500, 1000]
BATCH_SIZE = base.BATCH_SIZE
BETA = 0.6


def stratified_sample(
    replay: PrioritizedDemonstrationReplay,
    actions: np.ndarray,
    rng: np.random.Generator,
) -> ReplaySample:
    noop_indices = np.where(actions == 0)[0]
    nonzero_indices = np.where(actions != 0)[0]
    probabilities = replay.sampling_probabilities()
    noop_conditional = probabilities[noop_indices] / probabilities[noop_indices].sum()
    nonzero_conditional = probabilities[nonzero_indices] / probabilities[nonzero_indices].sum()
    half = BATCH_SIZE // 2
    chosen_noop = rng.choice(noop_indices, size=half, replace=True, p=noop_conditional)
    chosen_nonzero = rng.choice(nonzero_indices, size=half, replace=True, p=nonzero_conditional)
    chosen = np.concatenate([chosen_noop, chosen_nonzero]).astype(np.int64)
    # Shuffle positions while preserving the pre-registered 32/32 composition.
    rng.shuffle(chosen)

    mixture_probabilities = np.empty(BATCH_SIZE, dtype=np.float64)
    conditional_probabilities = np.empty(BATCH_SIZE, dtype=np.float64)
    noop_lookup = {int(index): float(value) for index, value in zip(noop_indices, noop_conditional)}
    nonzero_lookup = {
        int(index): float(value) for index, value in zip(nonzero_indices, nonzero_conditional)
    }
    for position, index in enumerate(chosen):
        conditional = noop_lookup[int(index)] if actions[index] == 0 else nonzero_lookup[int(index)]
        conditional_probabilities[position] = conditional
        mixture_probabilities[position] = 0.5 * conditional
    group_sizes = np.where(actions[chosen] == 0, len(noop_indices), len(nonzero_indices))
    importance = (group_sizes * conditional_probabilities) ** (-BETA)
    importance = importance / importance.max()
    data = {name: np.asarray(values)[chosen] for name, values in replay.demonstrations.items()}
    return ReplaySample(
        global_indices=chosen,
        probabilities=mixture_probabilities,
        importance_weights=importance.astype(np.float32),
        is_demonstration=np.ones(BATCH_SIZE, dtype=bool),
        data=data,
    )


def update_model(
    model: DQN,
    replay: PrioritizedDemonstrationReplay,
    actions_all: np.ndarray,
    rng: np.random.Generator,
    *,
    arm: str,
    update: int,
) -> dict[str, Any]:
    if arm == "control_global_per":
        sample = replay.sample(BATCH_SIZE, beta=BETA)
    else:
        sample = stratified_sample(replay, actions_all, rng)
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
    # Both arms mask demonstration 1-step TD; it remains logged and drives priority updates.
    # Preserve the exact 021_28 computation graph; only the sampler may differ.
    total = (
        0.0 * losses["td_1_weighted"]
        + losses["td_n_weighted"]
        + losses["margin_weighted"]
        + losses["l2_weighted"]
    )
    model.policy.optimizer.zero_grad()
    total.backward()
    total_gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, base.MAX_GRAD_NORM))
    model.policy.optimizer.step()
    with torch.no_grad():
        chosen_q = q_values.gather(1, actions[:, None]).squeeze(1)
        td_errors = torch.abs(target_1 - chosen_q).detach().cpu().numpy()
    priority_updates: dict[int, float] = {}
    for index, error in zip(sample.global_indices, td_errors):
        priority_updates[int(index)] = max(priority_updates.get(int(index), 0.0), float(error))
    replay.update_priorities(priority_updates.keys(), priority_updates.values())
    sample_actions = np.asarray(sample.data["actions"]).reshape(-1)
    values = {name: float(value.detach().cpu()) for name, value in losses.items()}
    return {
        "arm": arm,
        "phase": "offline_pretrain",
        "update": int(update),
        "sample_noop_count": int((sample_actions == 0).sum()),
        "sample_nonzero_count": int((sample_actions != 0).sum()),
        "sample_demo_count": int(sample.is_demonstration.sum()),
        "sample_agent_count": int((~sample.is_demonstration).sum()),
        "importance_weight_min": float(sample.importance_weights.min()),
        "importance_weight_max": float(sample.importance_weights.max()),
        **values,
        "effective_total": float(total.detach().cpu()),
        "grad_total_before_clip": total_gradient_norm,
        "gradient_would_clip": total_gradient_norm > base.MAX_GRAD_NORM,
        "all_finite": bool(
            np.isfinite(total_gradient_norm)
            and torch.isfinite(total).item()
            and torch.isfinite(q_values).all().item()
            and all(torch.isfinite(parameter).all().item() for parameter in parameters)
        ),
    }


def run_arm(
    arm: str,
    demonstrations: dict[str, np.ndarray],
    raw_observations: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    np.random.seed(0)
    torch.manual_seed(0)
    rng = np.random.default_rng(21022)
    actions = np.asarray(demonstrations["actions"], dtype=np.int64).reshape(-1)
    observations = demonstrations["observations"]
    env = OfflineShapeEnv(observations.shape[1])
    model = DQN("MlpPolicy", env, verbose=0, **dqn_kwargs(seed=0))
    replay = PrioritizedDemonstrationReplay(
        demonstrations, agent_capacity=1, alpha=0.4,
        epsilon_demo=1.0, epsilon_agent=0.001, seed=21022,
    )
    curve_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    summary, events = evaluate_demonstrations(
        model, observations, actions, raw_observations, 0, pd.DataFrame()
    )
    summary["arm"] = arm
    for event in events:
        event["arm"] = arm
    curve_rows.append(summary)
    event_rows.extend(events)
    previous = 0
    for milestone in MILESTONES[1:]:
        interval_rows: list[dict[str, Any]] = []
        for update in range(previous + 1, milestone + 1):
            row = update_model(model, replay, actions, rng, arm=arm, update=update)
            interval_rows.append(row)
            update_rows.append(row)
        interval = pd.DataFrame(interval_rows)
        summary, events = evaluate_demonstrations(
            model, observations, actions, raw_observations, milestone, interval
        )
        summary["arm"] = arm
        for event in events:
            event["arm"] = arm
        curve_rows.append(summary)
        event_rows.extend(events)
        previous = milestone
    env.close()
    return pd.DataFrame(curve_rows), pd.DataFrame(event_rows), pd.DataFrame(update_rows)


def consecutive_pass(frame: pd.DataFrame) -> bool:
    values = frame.sort_values("milestone_updates").milestone_gate_passed.tolist()
    return any(values[index] and values[index + 1] for index in range(len(values) - 1))


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def plot_results(curve: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    colors = {"control_global_per": "#666666", "treatment_stratified_16_16": "#009E73"}
    for arm, frame in curve.groupby("arm"):
        frame = frame.sort_values("milestone_updates")
        axes[0, 0].plot(frame.milestone_updates, frame.nonzero_action_recall, marker="o", color=colors[arm], label=arm)
        axes[0, 1].plot(frame.milestone_updates, frame.noop_accuracy, marker="o", color=colors[arm], label=arm)
        axes[1, 0].plot(frame.milestone_updates, frame.positive_expert_margin_fraction_nonzero, marker="o", color=colors[arm], label=arm)
        axes[1, 1].plot(frame.milestone_updates, frame.mean_expert_margin_nonzero, marker="o", color=colors[arm], label=arm)
    axes[0, 0].axhline(0.8, color="#333333", linestyle="--", linewidth=1)
    axes[0, 1].axhline(0.95, color="#333333", linestyle="--", linewidth=1)
    axes[1, 0].axhline(0.8, color="#333333", linestyle="--", linewidth=1)
    axes[1, 1].axhline(0, color="#333333", linestyle="--", linewidth=1)
    axes[0, 0].set_ylabel("Nonzero action recall")
    axes[0, 1].set_ylabel("No-op accuracy")
    axes[1, 0].set_ylabel("Positive expert-margin fraction")
    axes[1, 1].set_ylabel("Mean expert Q margin")
    for ax in axes.flat:
        ax.set_xlabel("Demonstration updates")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("SY2014 stratified demonstration sampling: offline A/B")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "021_30_stratified_demonstration_sampling_ab.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "021_30_stratified_demonstration_sampling_ab.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    raw, standardized, validation = normbase.prepare_demonstrations()
    if not validation["all_pre_run_checks_pass"]:
        raise RuntimeError("Frozen standardization validation failed")
    demonstrations = full_return_to_go(standardized)

    curves: list[pd.DataFrame] = []
    events: list[pd.DataFrame] = []
    updates: list[pd.DataFrame] = []
    for arm in ("control_global_per", "treatment_stratified_16_16"):
        curve, event, update = run_arm(arm, demonstrations, raw["observations"])
        curves.append(curve)
        events.append(event)
        updates.append(update)
    curve = pd.concat(curves, ignore_index=True)
    event_frame = pd.concat(events, ignore_index=True)
    update_frame = pd.concat(updates, ignore_index=True)
    curve.to_csv(OUT / "021_30_learning_curve.csv", index=False, encoding="utf-8-sig")
    event_frame.to_csv(OUT / "021_30_nonzero_event_predictions.csv", index=False, encoding="utf-8-sig")
    update_frame.to_csv(OUT / "021_30_update_diagnostics.csv", index=False, encoding="utf-8-sig")

    reference = pd.read_csv(ROOT / "benchmark_results" / "021_28" / "021_28_learning_curve.csv")
    reference = reference[reference.arm == "treatment_mask_demo_td1"].sort_values("milestone_updates")
    control = curve[curve.arm == "control_global_per"].sort_values("milestone_updates")
    comparison_columns = [
        "noop_accuracy", "nonzero_action_recall",
        "positive_expert_margin_fraction_nonzero", "mean_expert_margin_nonzero",
    ]
    max_errors = {
        column: float(np.max(np.abs(control[column].to_numpy() - reference[column].to_numpy())))
        for column in comparison_columns
    }
    treatment_updates = update_frame[update_frame.arm == "treatment_stratified_16_16"]
    validation_result = {
        "control_021_28_max_abs_errors": max_errors,
        "control_reproduced": bool(max(max_errors.values()) < 1e-6),
        "frozen_batch_size_is_32": bool(BATCH_SIZE == 32),
        "treatment_all_batches_16_noop": bool((treatment_updates.sample_noop_count == 16).all()),
        "treatment_all_batches_16_nonzero": bool((treatment_updates.sample_nonzero_count == 16).all()),
        "all_samples_demonstrations": bool((update_frame.sample_agent_count == 0).all()),
        "all_updates_finite": bool(update_frame.all_finite.all()),
    }
    validation_result["passed"] = bool(all(validation_result.values()))
    (OUT / "021_30_validation.json").write_text(
        json.dumps(validation_result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not validation_result["passed"]:
        raise RuntimeError(f"Validation failed: {validation_result}")

    treatment = curve[curve.arm == "treatment_stratified_16_16"]
    control_consecutive = consecutive_pass(control)
    treatment_consecutive = consecutive_pass(treatment)
    treatment_any = bool(treatment.milestone_gate_passed.any())
    partial = bool(
        treatment.nonzero_action_recall.max() > control.nonzero_action_recall.max()
        or treatment.positive_expert_margin_fraction_nonzero.max()
        > control.positive_expert_margin_fraction_nonzero.max()
    )
    if treatment_consecutive and not control_consecutive:
        branch = "A_stratified_balance_is_important_mechanism"
    elif treatment_any or partial:
        branch = "B_partial_signal_but_joint_gate_not_stable"
    else:
        branch = "C_stratified_balance_not_sufficient"
    result = {
        "status": "completed_offline_no_dssat",
        "validation_passed": validation_result["passed"],
        "control_passing_milestones": control.loc[control.milestone_gate_passed, "milestone_updates"].astype(int).tolist(),
        "treatment_passing_milestones": treatment.loc[treatment.milestone_gate_passed, "milestone_updates"].astype(int).tolist(),
        "control_two_consecutive_pass": control_consecutive,
        "treatment_two_consecutive_pass": treatment_consecutive,
        "treatment_max_nonzero_recall": float(treatment.nonzero_action_recall.max()),
        "treatment_max_noop_accuracy": float(treatment.noop_accuracy.max()),
        "treatment_max_positive_margin_fraction": float(treatment.positive_expert_margin_fraction_nonzero.max()),
        "interpretation_branch": branch,
        "no_online_training_performed": True,
    }
    (OUT / "021_30_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_results(curve)

    display = curve[[
        "arm", "milestone_updates", "noop_accuracy", "nonzero_action_recall",
        "positive_expert_margin_fraction_nonzero", "mean_expert_margin_nonzero",
        "median_total_gradient", "gradient_clip_fraction", "milestone_gate_passed",
    ]].round(6)
    record = f"""# 021_30 SY2014 分层平衡示范采样离线 A/B

## 边界

完全离线。Control 使用全局 PER；Treatment 保持冻结 batch size 32，每个 batch 固定 16 no-op + 16 非零示范，并在组内按 priority 抽样。其他训练目标和参数完全相同，两组均屏蔽示范 1-step TD 梯度。

## 强制验证

```json
{json.dumps(validation_result, indent=2, ensure_ascii=False)}
```

## 学习曲线

{markdown_table(display)}

## 预注册判定

- Control passing milestones: `{result['control_passing_milestones']}`
- Treatment passing milestones: `{result['treatment_passing_milestones']}`
- Treatment two consecutive pass: `{treatment_consecutive}`
- Branch: `{branch}`

## 结论边界

`{branch}`。Treatment 是类别平衡机制实验，不声称是原始全局 PER 目标的无偏估计。本轮没有调用 DSSAT、没有在线训练，也没有现场修改 16/16 比例或其他超参数。
"""
    DOC.write_text(record, encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
