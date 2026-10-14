from __future__ import annotations

import hashlib
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
from literature_aligned_dqfd import PrioritizedDemonstrationReplay, dqfd_loss_components
from run_sy2014_demo_return_to_go_ab_021_27 import full_return_to_go
from run_sy2014_standardized_demo_pretraining_curve_021_26 import (
    OfflineShapeEnv,
    evaluate_demonstrations,
)
from stable_baselines3 import DQN


OUT = ROOT / "benchmark_results" / "021_28"
DOC = ROOT / "docs" / "2026-07-15_021_28_sy2014_demo_td1_conflict_mask_offline_ab.md"
MILESTONES = [0, 10, 25, 50, 100, 250, 500, 1000]


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def masked_update(
    model: DQN,
    replay: PrioritizedDemonstrationReplay,
    *,
    update: int,
    include_demo_td1: bool,
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
    td1_in_total_weight = 1.0 if include_demo_td1 else 0.0
    total = (
        td1_in_total_weight * losses["td_1_weighted"]
        + losses["td_n_weighted"]
        + losses["margin_weighted"]
        + losses["l2_weighted"]
    )
    model.policy.optimizer.zero_grad()
    total.backward()
    total_gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, base.MAX_GRAD_NORM))
    model.policy.optimizer.step()

    # Keep PER priority updates frozen to the original 1-step TD error.
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
        and np.isfinite(float(total.detach().cpu()))
        and np.isfinite(total_gradient_norm)
        and torch.isfinite(q_values).all().item()
        and all(torch.isfinite(parameter).all().item() for parameter in parameters)
    )
    return {
        "phase": "offline_pretrain",
        "update": int(update),
        "sample_demo_count": int(sample.is_demonstration.sum()),
        "sample_agent_count": int((~sample.is_demonstration).sum()),
        "sample_indices_sha256": array_hash(sample.global_indices),
        "target_1_sha256": array_hash(target_1.detach().cpu().numpy()),
        "target_n_sha256": array_hash(target_n.detach().cpu().numpy()),
        **values,
        "original_total_with_td1": values["total"],
        "td1_in_total_weight": td1_in_total_weight,
        "effective_total": float(total.detach().cpu()),
        "grad_total_before_clip": total_gradient_norm,
        "gradient_would_clip": total_gradient_norm > base.MAX_GRAD_NORM,
        "q_abs_mean": float(q_values.detach().abs().mean().cpu()),
        "q_abs_max": float(q_values.detach().abs().max().cpu()),
        "all_finite": all_finite,
    }


def run_arm(
    arm: str,
    demonstrations: dict[str, np.ndarray],
    raw_observations: np.ndarray,
    *,
    include_demo_td1: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    np.random.seed(0)
    torch.manual_seed(0)
    observations = demonstrations["observations"]
    actions = np.asarray(demonstrations["actions"], dtype=np.int64).reshape(-1)
    env = OfflineShapeEnv(observations.shape[1])
    model = DQN("MlpPolicy", env, verbose=0, **dqn_kwargs(seed=0))
    replay = PrioritizedDemonstrationReplay(
        demonstrations,
        agent_capacity=1,
        alpha=0.4,
        epsilon_demo=1.0,
        epsilon_agent=0.001,
        seed=21022,
    )
    curve_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    summary, events = evaluate_demonstrations(
        model, observations, actions, raw_observations, 0, pd.DataFrame()
    )
    summary["arm"] = arm
    summary["include_demo_td1"] = include_demo_td1
    for event in events:
        event["arm"] = arm
    curve_rows.append(summary)
    event_rows.extend(events)

    previous = 0
    for milestone in MILESTONES[1:]:
        interval_rows: list[dict[str, Any]] = []
        for update in range(previous + 1, milestone + 1):
            row = masked_update(
                model,
                replay,
                update=update,
                include_demo_td1=include_demo_td1,
            )
            row["arm"] = arm
            interval_rows.append(row)
            update_rows.append(row)
        interval = pd.DataFrame(interval_rows)
        summary, events = evaluate_demonstrations(
            model, observations, actions, raw_observations, milestone, interval
        )
        summary["arm"] = arm
        summary["include_demo_td1"] = include_demo_td1
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
    colors = {"control_keep_td1": "#666666", "treatment_mask_demo_td1": "#0072B2"}
    for arm, frame in curve.groupby("arm"):
        frame = frame.sort_values("milestone_updates")
        axes[0, 0].plot(frame.milestone_updates, frame.nonzero_action_recall, marker="o", color=colors[arm], label=arm)
        axes[0, 1].plot(frame.milestone_updates, frame.noop_accuracy, marker="o", color=colors[arm], label=arm)
        axes[1, 0].plot(frame.milestone_updates, frame.positive_expert_margin_fraction_nonzero, marker="o", color=colors[arm], label=arm)
        axes[1, 1].plot(frame.milestone_updates, frame.mean_expert_margin_nonzero, marker="o", color=colors[arm], label=arm)
    axes[0, 0].axhline(0.8, color="#333333", linestyle="--", linewidth=1)
    axes[0, 1].axhline(0.95, color="#333333", linestyle="--", linewidth=1)
    axes[1, 0].axhline(0.8, color="#333333", linestyle="--", linewidth=1)
    axes[1, 1].axhline(0.0, color="#333333", linestyle="--", linewidth=1)
    axes[0, 0].set_ylabel("Nonzero action recall")
    axes[0, 1].set_ylabel("No-op accuracy")
    axes[1, 0].set_ylabel("Positive expert-margin fraction")
    axes[1, 1].set_ylabel("Mean expert Q margin")
    for ax in axes.flat:
        ax.set_xlabel("Demonstration updates")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("SY2014 demonstration 1-step TD conflict mask: offline A/B")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "021_28_demo_td1_conflict_mask_ab.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "021_28_demo_td1_conflict_mask_ab.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    raw, standardized, scaler_validation = normbase.prepare_demonstrations()
    if not scaler_validation["all_pre_run_checks_pass"]:
        raise RuntimeError("Frozen standardization validation failed")
    demonstrations = full_return_to_go(standardized)
    if not np.all(np.asarray(demonstrations["n_step_dones"]) == 1):
        raise RuntimeError("Full return-to-go targets are not terminal")

    curves: list[pd.DataFrame] = []
    events: list[pd.DataFrame] = []
    updates: list[pd.DataFrame] = []
    arms = (
        ("control_keep_td1", True),
        ("treatment_mask_demo_td1", False),
    )
    for arm, include_demo_td1 in arms:
        curve, event, update = run_arm(
            arm,
            demonstrations,
            raw["observations"],
            include_demo_td1=include_demo_td1,
        )
        curves.append(curve)
        events.append(event)
        updates.append(update)

    curve = pd.concat(curves, ignore_index=True)
    event_frame = pd.concat(events, ignore_index=True)
    update_frame = pd.concat(updates, ignore_index=True)
    curve.to_csv(OUT / "021_28_learning_curve.csv", index=False, encoding="utf-8-sig")
    event_frame.to_csv(OUT / "021_28_nonzero_event_predictions.csv", index=False, encoding="utf-8-sig")
    update_frame.to_csv(OUT / "021_28_update_diagnostics.csv", index=False, encoding="utf-8-sig")

    first = update_frame[update_frame["update"] == 1].set_index("arm")
    validation = {
        "same_first_sample_indices": bool(first.sample_indices_sha256.nunique() == 1),
        "same_first_target_1": bool(first.target_1_sha256.nunique() == 1),
        "same_first_target_n": bool(first.target_n_sha256.nunique() == 1),
        "control_td1_weight": float(first.loc["control_keep_td1", "td1_in_total_weight"]),
        "treatment_td1_weight": float(first.loc["treatment_mask_demo_td1", "td1_in_total_weight"]),
        "all_samples_demonstrations": bool((update_frame.sample_agent_count == 0).all()),
        "all_updates_finite": bool(update_frame.all_finite.all()),
        "no_dssat_or_online_training": True,
    }
    validation["passed"] = bool(
        validation["same_first_sample_indices"]
        and validation["same_first_target_1"]
        and validation["same_first_target_n"]
        and validation["control_td1_weight"] == 1.0
        and validation["treatment_td1_weight"] == 0.0
        and validation["all_samples_demonstrations"]
        and validation["all_updates_finite"]
    )
    (OUT / "021_28_implementation_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not validation["passed"]:
        raise RuntimeError(f"Implementation validation failed: {validation}")

    control = curve[curve.arm == "control_keep_td1"]
    treatment = curve[curve.arm == "treatment_mask_demo_td1"]
    control_consecutive = consecutive_pass(control)
    treatment_consecutive = consecutive_pass(treatment)
    treatment_any = bool(treatment.milestone_gate_passed.any())
    treatment_better = bool(
        treatment.nonzero_action_recall.max() > control.nonzero_action_recall.max()
        or treatment.positive_expert_margin_fraction_nonzero.max()
        > control.positive_expert_margin_fraction_nonzero.max()
    )
    if treatment_consecutive and not control_consecutive:
        branch = "A_demo_td1_conflict_is_important_mechanism"
    elif treatment_any or treatment_better:
        branch = "B_partial_or_unstable_signal"
    else:
        branch = "C_masking_demo_td1_not_sufficient"
    summary = {
        "status": "completed_offline_no_dssat",
        "implementation_validation_passed": validation["passed"],
        "control_passing_milestones": control.loc[control.milestone_gate_passed, "milestone_updates"].astype(int).tolist(),
        "treatment_passing_milestones": treatment.loc[treatment.milestone_gate_passed, "milestone_updates"].astype(int).tolist(),
        "control_two_consecutive_pass": control_consecutive,
        "treatment_two_consecutive_pass": treatment_consecutive,
        "pre_registered_interpretation_branch": branch,
        "no_online_training_performed": True,
    }
    (OUT / "021_28_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_results(curve)

    display = curve[[
        "arm",
        "milestone_updates",
        "noop_accuracy",
        "nonzero_action_recall",
        "positive_expert_margin_fraction_nonzero",
        "mean_expert_margin_nonzero",
        "median_td_1_weighted",
        "median_td_n_weighted",
        "median_margin_weighted",
        "median_total_gradient",
        "gradient_clip_fraction",
        "milestone_gate_passed",
    ]].round(6)
    record = f"""# 021_28 SY2014 示范 1-step TD 冲突屏蔽离线 A/B

## 目的与边界

021_27 证明五个非零示范动作上，负的即时成本 1-step TD 与正的完整季节 return TD 形成方向相反的饱和梯度。本任务只检验：在示范样本上屏蔽冲突的 1-step TD 梯度，是否能让网络学会这些稀疏非零动作。

本任务完全离线，不调用 DSSAT/PDI，不进行在线环境交互。两组均使用相同的 160 条示范、固定标准化、完整 return-to-go、网络初始化、PER、margin、L2 和 priority 更新规则。唯一差异是示范 1-step TD 是否进入总损失。

## 实现一致性验证

```json
{json.dumps(validation, indent=2, ensure_ascii=False)}
```

## 学习曲线

{markdown_table(display)}

## 预注册判定

- Control passing milestones: `{summary['control_passing_milestones']}`
- Treatment passing milestones: `{summary['treatment_passing_milestones']}`
- Treatment consecutive pass: `{summary['treatment_two_consecutive_pass']}`
- Branch: `{branch}`

## 结论

`{branch}`。本轮只回答离线示范信用分配问题；无论结果如何，均未自动启动在线 5K 训练，也未修改 reward、观测、PER、网络或动作空间。

## 输出

- `benchmark_results/021_28/021_28_learning_curve.csv`
- `benchmark_results/021_28/021_28_nonzero_event_predictions.csv`
- `benchmark_results/021_28/021_28_update_diagnostics.csv`
- `benchmark_results/021_28/021_28_implementation_validation.json`
- `benchmark_results/021_28/021_28_summary.json`
- `benchmark_results/021_28/021_28_demo_td1_conflict_mask_ab.png/.svg`
"""
    DOC.write_text(record, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
