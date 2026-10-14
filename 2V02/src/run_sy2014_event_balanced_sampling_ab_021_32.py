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

import run_sy2014_fixed_observation_standardization_diagnostic_021_24 as normbase
import run_sy2014_stratified_demo_sampling_ab_021_30 as exp
from literature_aligned_dqfd import ReplaySample
from run_sy2014_demo_return_to_go_ab_021_27 import full_return_to_go


OUT = ROOT / "benchmark_results" / "021_32"
DOC = ROOT / "docs" / "2026-07-15_021_32_sy2014_event_balanced_nonzero_sampling_offline_ab.md"
MILESTONES = [0, 10, 25, 50, 100, 250, 500, 1000]
BETA = 0.6


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def plot_results(curve: pd.DataFrame, events: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    colors = {"control_within_nonzero_per": "#666666", "treatment_event_balanced": "#CC79A7"}
    for arm, frame in curve.groupby("arm"):
        frame = frame.sort_values("milestone_updates")
        axes[0, 0].plot(frame.milestone_updates, frame.nonzero_action_recall, marker="o", color=colors[arm], label=arm)
        axes[0, 1].plot(frame.milestone_updates, frame.noop_accuracy, marker="o", color=colors[arm], label=arm)
        axes[1, 0].plot(frame.milestone_updates, frame.positive_expert_margin_fraction_nonzero, marker="o", color=colors[arm], label=arm)
    treatment = events[events.arm == "treatment_event_balanced"]
    for index, frame in treatment.groupby("transition_index"):
        label = f"DAP{int(frame.dap.iloc[0])}/a{int(frame.target_action.iloc[0])}"
        axes[1, 1].plot(frame.milestone_updates, frame.expert_margin, marker="o", label=label)
    axes[0, 0].axhline(0.8, color="#333333", linestyle="--", linewidth=1)
    axes[0, 1].axhline(0.95, color="#333333", linestyle="--", linewidth=1)
    axes[1, 0].axhline(0.8, color="#333333", linestyle="--", linewidth=1)
    axes[1, 1].axhline(0, color="#333333", linestyle="--", linewidth=1)
    axes[0, 0].set_ylabel("Nonzero action recall")
    axes[0, 1].set_ylabel("No-op accuracy")
    axes[1, 0].set_ylabel("Positive expert-margin fraction")
    axes[1, 1].set_ylabel("Treatment expert Q margin")
    for ax in axes.flat:
        ax.set_xlabel("Demonstration updates")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("SY2014 event-balanced nonzero sampling: offline A/B")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "021_32_event_balanced_sampling_ab.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "021_32_event_balanced_sampling_ab.svg", bbox_inches="tight")
    plt.close(fig)


def run_arm(
    arm: str,
    demonstrations: dict[str, np.ndarray],
    raw_observations: np.ndarray,
    *,
    event_balanced: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    np.random.seed(0)
    torch.manual_seed(0)
    observations = demonstrations["observations"]
    actions = np.asarray(demonstrations["actions"], dtype=np.int64).reshape(-1)
    nonzero_indices = np.where(actions != 0)[0]
    noop_indices = np.where(actions == 0)[0]
    env = exp.OfflineShapeEnv(observations.shape[1])
    model = exp.DQN("MlpPolicy", env, verbose=0, **exp.dqn_kwargs(seed=0))
    replay = exp.PrioritizedDemonstrationReplay(
        demonstrations, agent_capacity=1, alpha=0.4,
        epsilon_demo=1.0, epsilon_agent=0.001, seed=21022,
    )
    rng = np.random.default_rng(21022)
    state: dict[str, int] = {"update": 0}
    sampling_rows: list[dict[str, Any]] = []
    original_sampler = exp.stratified_sample

    def event_balanced_sampler(replay_object, action_values, generator):
        probabilities = replay_object.sampling_probabilities()
        noop_conditional = probabilities[noop_indices] / probabilities[noop_indices].sum()
        chosen_noop = generator.choice(noop_indices, size=16, replace=True, p=noop_conditional)
        counts = np.full(len(nonzero_indices), 3, dtype=np.int64)
        counts[(state["update"] - 1) % len(nonzero_indices)] += 1
        chosen_nonzero = np.repeat(nonzero_indices, counts)
        chosen = np.concatenate([chosen_noop, chosen_nonzero]).astype(np.int64)
        generator.shuffle(chosen)

        conditional = np.empty(len(chosen), dtype=np.float64)
        noop_lookup = {int(index): float(value) for index, value in zip(noop_indices, noop_conditional)}
        for position, index in enumerate(chosen):
            conditional[position] = noop_lookup[int(index)] if actions[index] == 0 else 0.2
        group_sizes = np.where(actions[chosen] == 0, len(noop_indices), len(nonzero_indices))
        importance = (group_sizes * conditional) ** (-BETA)
        importance = importance / importance.max()
        mixture_probability = 0.5 * conditional
        data = {name: np.asarray(values)[chosen] for name, values in replay_object.demonstrations.items()}
        for index, count in zip(nonzero_indices, counts):
            sampling_rows.append({
                "update": int(state["update"]),
                "five_update_block": int((state["update"] - 1) // 5),
                "transition_index": int(index),
                "dap": int(round(float(raw_observations[index, 1]))),
                "target_action": int(actions[index]),
                "priority_before_sample": float(replay_object.combined_priorities()[index]),
                "batch_sample_count": int(count),
            })
        return ReplaySample(
            global_indices=chosen,
            probabilities=mixture_probability,
            importance_weights=importance.astype(np.float32),
            is_demonstration=np.ones(len(chosen), dtype=bool),
            data=data,
        )

    if event_balanced:
        exp.stratified_sample = event_balanced_sampler
    else:
        exp.stratified_sample = original_sampler
    curve_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    summary, events = exp.evaluate_demonstrations(
        model, observations, actions, raw_observations, 0, pd.DataFrame()
    )
    summary["arm"] = arm
    for event in events:
        event["arm"] = arm
    curve_rows.append(summary)
    event_rows.extend(events)
    previous = 0
    try:
        for milestone in MILESTONES[1:]:
            interval_rows: list[dict[str, Any]] = []
            for update in range(previous + 1, milestone + 1):
                state["update"] = update
                row = exp.update_model(model, replay, actions, rng, arm=arm, update=update)
                interval_rows.append(row)
                update_rows.append(row)
            summary, events = exp.evaluate_demonstrations(
                model, observations, actions, raw_observations,
                milestone, pd.DataFrame(interval_rows),
            )
            summary["arm"] = arm
            for event in events:
                event["arm"] = arm
            curve_rows.append(summary)
            event_rows.extend(events)
            previous = milestone
    finally:
        exp.stratified_sample = original_sampler
        env.close()
    return (
        pd.DataFrame(curve_rows),
        pd.DataFrame(event_rows),
        pd.DataFrame(update_rows),
        pd.DataFrame(sampling_rows),
    )


def consecutive_pass(frame: pd.DataFrame) -> bool:
    values = frame.sort_values("milestone_updates").milestone_gate_passed.tolist()
    return any(values[index] and values[index + 1] for index in range(len(values) - 1))


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    raw, standardized, validation = normbase.prepare_demonstrations()
    if not validation["all_pre_run_checks_pass"]:
        raise RuntimeError("Frozen standardization validation failed")
    demonstrations = full_return_to_go(standardized)

    outputs = []
    for arm, balanced in (
        ("control_within_nonzero_per", False),
        ("treatment_event_balanced", True),
    ):
        outputs.append(run_arm(arm, demonstrations, raw["observations"], event_balanced=balanced))
    curve = pd.concat([value[0] for value in outputs], ignore_index=True)
    events = pd.concat([value[1] for value in outputs], ignore_index=True)
    updates = pd.concat([value[2] for value in outputs], ignore_index=True)
    event_sampling = outputs[1][3]
    curve.to_csv(OUT / "021_32_learning_curve.csv", index=False, encoding="utf-8-sig")
    events.to_csv(OUT / "021_32_event_predictions.csv", index=False, encoding="utf-8-sig")
    updates.to_csv(OUT / "021_32_update_diagnostics.csv", index=False, encoding="utf-8-sig")
    event_sampling.to_csv(OUT / "021_32_event_sampling.csv", index=False, encoding="utf-8-sig")

    reference = pd.read_csv(ROOT / "benchmark_results" / "021_30" / "021_30_learning_curve.csv")
    reference = reference[reference.arm == "treatment_stratified_16_16"].sort_values("milestone_updates")
    control = curve[curve.arm == "control_within_nonzero_per"].sort_values("milestone_updates")
    comparison_columns = [
        "noop_accuracy", "nonzero_action_recall",
        "positive_expert_margin_fraction_nonzero", "mean_expert_margin_nonzero",
    ]
    max_errors = {
        column: float(np.max(np.abs(control[column].to_numpy() - reference[column].to_numpy())))
        for column in comparison_columns
    }
    treatment_updates = updates[updates.arm == "treatment_event_balanced"]
    block_counts = event_sampling.groupby(["five_update_block", "transition_index"]).batch_sample_count.sum()
    total_counts = event_sampling.groupby("transition_index").batch_sample_count.sum()
    validation_result = {
        "control_021_30_max_abs_errors": max_errors,
        "control_reproduced": bool(max(max_errors.values()) < 1e-6),
        "treatment_all_batches_16_noop": bool((treatment_updates.sample_noop_count == 16).all()),
        "treatment_all_batches_16_nonzero": bool((treatment_updates.sample_nonzero_count == 16).all()),
        "each_event_has_16_samples_per_five_update_block": bool((block_counts == 16).all()),
        "each_event_has_3200_total_samples": bool((total_counts == 3200).all()),
        "all_updates_finite": bool(updates.all_finite.all()),
    }
    validation_result["passed"] = bool(all(validation_result.values()))
    (OUT / "021_32_validation.json").write_text(
        json.dumps(validation_result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not validation_result["passed"]:
        raise RuntimeError(f"Validation failed: {validation_result}")

    treatment = curve[curve.arm == "treatment_event_balanced"]
    control_consecutive = consecutive_pass(control)
    treatment_consecutive = consecutive_pass(treatment)
    treatment_any = bool(treatment.milestone_gate_passed.any())
    partial = bool(
        treatment.nonzero_action_recall.max() > control.nonzero_action_recall.max()
        or treatment.positive_expert_margin_fraction_nonzero.max()
        > control.positive_expert_margin_fraction_nonzero.max()
    )
    if treatment_consecutive and not control_consecutive:
        branch = "A_event_level_per_bias_is_key_remaining_mechanism"
    elif treatment_any or partial:
        branch = "B_event_balance_partial_but_not_stable"
    else:
        branch = "C_event_balance_not_sufficient"
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
    (OUT / "021_32_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_results(curve, events)

    display = curve[[
        "arm", "milestone_updates", "noop_accuracy", "nonzero_action_recall",
        "positive_expert_margin_fraction_nonzero", "mean_expert_margin_nonzero",
        "median_total_gradient", "gradient_clip_fraction", "milestone_gate_passed",
    ]].round(6)
    final_events = events[
        (events.arm == "treatment_event_balanced") & (events.milestone_updates == 1000)
    ][[
        "transition_index", "dap", "target_action", "predicted_action",
        "target_action_recalled", "expert_margin",
    ]].round(6)
    record = f"""# 021_32 SY2014 非零事件等频采样离线 A/B

## 边界

Control 保留 021_30 的非零组内 PER；Treatment 只把五个非零事件改为每五个 batch 完全等频。no-op 采样、priority 更新、训练目标和所有其他参数未变。全程离线，无 DSSAT/PDI 调用。

## 强制验证

```json
{json.dumps(validation_result, indent=2, ensure_ascii=False)}
```

## 学习曲线

{markdown_table(display)}

## Treatment 1000 次更新事件结果

{markdown_table(final_events)}

## 预注册判定

- Control passing milestones: `{result['control_passing_milestones']}`
- Treatment passing milestones: `{result['treatment_passing_milestones']}`
- Treatment two consecutive pass: `{treatment_consecutive}`
- Branch: `{branch}`

## 结论边界

`{branch}`。本轮不根据结果调整事件频率，也未自动启动在线训练。
"""
    DOC.write_text(record, encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
