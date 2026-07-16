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
from run_sy2014_demo_return_to_go_ab_021_27 import full_return_to_go


OUT = ROOT / "benchmark_results" / "021_31"
DOC = ROOT / "docs" / "2026-07-15_021_31_sy2014_action_level_sampling_audit.md"
MILESTONES = [0, 10, 25, 50, 100, 250, 500, 1000]


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def plot_results(
    per_update: pd.DataFrame,
    category_summary: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    colors = {1: "#0072B2", 4: "#D55E00", 7: "#009E73"}
    final_categories = category_summary[category_summary.milestone == 1000]
    axes[0, 0].bar(
        final_categories.target_action.astype(str),
        final_categories.cumulative_sample_share,
        color=[colors[int(value)] for value in final_categories.target_action],
    )
    axes[0, 0].scatter(["1", "4", "7"], [0.4, 0.4, 0.2], color="#222222", marker="x", s=55, label="event-count reference")
    axes[0, 0].set_ylabel("Cumulative sample share")
    axes[0, 0].legend(frameon=False, fontsize=8)

    audit = per_update[per_update["update"].isin(MILESTONES[1:])]
    for action, frame in audit.groupby("target_action"):
        grouped = frame.groupby("update", as_index=False).conditional_nonzero_probability.sum()
        axes[0, 1].plot(grouped["update"], grouped.conditional_nonzero_probability, marker="o", color=colors[int(action)], label=f"action {int(action)}")
    axes[0, 1].set_ylabel("Conditional nonzero probability mass")
    axes[0, 1].legend(frameon=False, fontsize=8)

    for index, frame in predictions.groupby("transition_index"):
        label = f"DAP{int(frame.dap.iloc[0])}/a{int(frame.target_action.iloc[0])}"
        axes[1, 0].plot(frame.milestone_updates, frame.expert_margin, marker="o", label=label)
    axes[1, 0].axhline(0, color="#555555", linestyle="--")
    axes[1, 0].set_ylabel("Expert Q margin")
    axes[1, 0].legend(frameon=False, fontsize=7, ncol=2)

    for action, frame in category_summary.groupby("target_action"):
        axes[1, 1].plot(frame.milestone, frame.cumulative_sample_share, marker="o", color=colors[int(action)], label=f"action {int(action)}")
    axes[1, 1].set_ylabel("Cumulative sample share")
    axes[1, 1].legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("Update or action")
        ax.grid(alpha=0.2)
    fig.suptitle("SY2014 action-level sampling audit within the nonzero stratum")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "021_31_action_level_sampling_audit.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "021_31_action_level_sampling_audit.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    np.random.seed(0)
    torch.manual_seed(0)
    raw, standardized, scaler_validation = normbase.prepare_demonstrations()
    if not scaler_validation["all_pre_run_checks_pass"]:
        raise RuntimeError("Frozen standardization validation failed")
    demonstrations = full_return_to_go(standardized)
    actions = np.asarray(demonstrations["actions"], dtype=np.int64).reshape(-1)
    observations = demonstrations["observations"]
    nonzero_indices = np.where(actions != 0)[0]
    if len(nonzero_indices) != 5:
        raise RuntimeError(f"Expected five nonzero events, got {len(nonzero_indices)}")

    env = exp.OfflineShapeEnv(observations.shape[1])
    model = exp.DQN("MlpPolicy", env, verbose=0, **exp.dqn_kwargs(seed=0))
    replay = exp.PrioritizedDemonstrationReplay(
        demonstrations, agent_capacity=1, alpha=0.4,
        epsilon_demo=1.0, epsilon_agent=0.001, seed=21022,
    )
    rng = np.random.default_rng(21022)
    original_sampler = exp.stratified_sample
    state: dict[str, Any] = {"update": 0}
    cumulative = {int(index): 0 for index in nonzero_indices}
    sampling_rows: list[dict[str, Any]] = []

    def capturing_sampler(replay_object, action_values, generator):
        probabilities = replay_object.sampling_probabilities()
        priorities = replay_object.combined_priorities()
        nonzero_mass = float(probabilities[nonzero_indices].sum())
        sample = original_sampler(replay_object, action_values, generator)
        selected = sample.global_indices.astype(int)
        for index in nonzero_indices:
            count = int((selected == index).sum())
            cumulative[int(index)] += count
            sampling_rows.append({
                "update": int(state["update"]),
                "transition_index": int(index),
                "dap": int(round(float(raw["observations"][index, 1]))),
                "target_action": int(actions[index]),
                "immediate_reward": float(demonstrations["rewards"][index]),
                "priority_before_sample": float(priorities[index]),
                "global_probability": float(probabilities[index]),
                "conditional_nonzero_probability": float(probabilities[index] / nonzero_mass),
                "batch_sample_count": count,
                "cumulative_sample_count": int(cumulative[int(index)]),
            })
        return sample

    exp.stratified_sample = capturing_sampler
    curve_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    initial, events = exp.evaluate_demonstrations(
        model, observations, actions, raw["observations"], 0, pd.DataFrame()
    )
    curve_rows.append(initial)
    event_rows.extend(events)
    previous = 0
    try:
        for milestone in MILESTONES[1:]:
            interval_rows: list[dict[str, Any]] = []
            for update in range(previous + 1, milestone + 1):
                state["update"] = update
                row = exp.update_model(
                    model, replay, actions, rng,
                    arm="treatment_stratified_16_16", update=update,
                )
                interval_rows.append(row)
                update_rows.append(row)
            summary, events = exp.evaluate_demonstrations(
                model, observations, actions, raw["observations"],
                milestone, pd.DataFrame(interval_rows),
            )
            curve_rows.append(summary)
            event_rows.extend(events)
            previous = milestone
    finally:
        exp.stratified_sample = original_sampler
        env.close()

    per_update = pd.DataFrame(sampling_rows)
    curve = pd.DataFrame(curve_rows)
    predictions = pd.DataFrame(event_rows)
    updates = pd.DataFrame(update_rows)
    per_update.to_csv(OUT / "021_31_action_sampling_per_update.csv", index=False, encoding="utf-8-sig")
    curve.to_csv(OUT / "021_31_learning_curve.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUT / "021_31_event_predictions.csv", index=False, encoding="utf-8-sig")
    updates.to_csv(OUT / "021_31_update_diagnostics.csv", index=False, encoding="utf-8-sig")

    category_rows: list[dict[str, Any]] = []
    for milestone in MILESTONES[1:]:
        subset = per_update[per_update["update"] <= milestone]
        total = int(subset.drop_duplicates(["update", "transition_index"]).batch_sample_count.sum())
        for action in sorted(np.unique(actions[nonzero_indices])):
            action_subset = subset[subset.target_action == action]
            count = int(action_subset.batch_sample_count.sum())
            category_rows.append({
                "milestone": milestone,
                "target_action": int(action),
                "event_count": int((actions[nonzero_indices] == action).sum()),
                "event_count_reference_share": float((actions[nonzero_indices] == action).mean()),
                "cumulative_sample_count": count,
                "cumulative_sample_share": float(count / total),
            })
    category_summary = pd.DataFrame(category_rows)
    category_summary.to_csv(OUT / "021_31_action_sampling_summary.csv", index=False, encoding="utf-8-sig")

    reference = pd.read_csv(ROOT / "benchmark_results" / "021_30" / "021_30_learning_curve.csv")
    reference = reference[reference.arm == "treatment_stratified_16_16"].sort_values("milestone_updates")
    aligned = curve.sort_values("milestone_updates")
    comparison_columns = [
        "noop_accuracy", "nonzero_action_recall",
        "positive_expert_margin_fraction_nonzero", "mean_expert_margin_nonzero",
    ]
    max_errors = {
        column: float(np.max(np.abs(aligned[column].to_numpy() - reference[column].to_numpy())))
        for column in comparison_columns
    }
    per_batch_nonzero = per_update.groupby("update").batch_sample_count.sum()
    validation = {
        "021_30_treatment_max_abs_errors": max_errors,
        "learning_curve_reproduced": bool(max(max_errors.values()) < 1e-6),
        "all_batches_have_16_nonzero": bool((per_batch_nonzero == 16).all()),
        "all_batches_have_16_noop": bool((updates.sample_noop_count == 16).all()),
        "all_updates_finite": bool(updates.all_finite.all()),
    }
    validation["passed"] = bool(all(validation.values()))
    (OUT / "021_31_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not validation["passed"]:
        raise RuntimeError(f"Validation failed: {validation}")

    late = per_update[(per_update["update"] >= 251) & (per_update["update"] <= 1000)]
    late_total = int(late.batch_sample_count.sum())
    action1_late = int(late[late.target_action == 1].batch_sample_count.sum())
    action1_late_share = float(action1_late / late_total)
    final_predictions = predictions[predictions.milestone_updates == 1000]
    action1_all_missed = bool(
        (~final_predictions[final_predictions.target_action == 1].target_action_recalled).all()
    )
    other_all_recalled = bool(
        final_predictions[final_predictions.target_action.isin([4, 7])].target_action_recalled.all()
    )
    if action1_late_share < 0.20 and action1_all_missed and other_all_recalled:
        branch = "A_strong_within_nonzero_per_bias_support"
    elif action1_late_share < 0.35 and action1_all_missed:
        branch = "B_partial_within_nonzero_per_bias_support"
    else:
        branch = "C_action1_not_materially_under_sampled"
    result = {
        "status": "completed_offline_no_dssat",
        "validation_passed": validation["passed"],
        "action_reference_shares": {"1": 0.4, "4": 0.4, "7": 0.2},
        "action1_sample_share_updates_251_1000": action1_late_share,
        "action1_all_missed_at_1000": action1_all_missed,
        "action4_and_action7_all_recalled_at_1000": other_all_recalled,
        "interpretation_branch": branch,
        "no_online_training_performed": True,
    }
    (OUT / "021_31_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_results(per_update, category_summary, predictions)

    final_categories = category_summary[category_summary.milestone == 1000].round(6)
    final_events = final_predictions[[
        "transition_index", "dap", "target_action", "predicted_action",
        "target_action_recalled", "expert_margin",
    ]].round(6)
    record = f"""# 021_31 SY2014 非零示范动作级采样审计

## 边界

本任务完全离线，并通过包装 021_30 已验证的 `stratified_sample()` 捕获真实抽样；未修改 sampler、priority、训练目标、网络或 seed。

## 强制验证

```json
{json.dumps(validation, indent=2, ensure_ascii=False)}
```

## 1000 次更新累计动作采样

{markdown_table(final_categories)}

## 1000 次更新动作预测

{markdown_table(final_events)}

## 预注册判定

- action1 事件数参照份额：`0.40`
- updates 251–1000 的 action1 实际采样份额：`{action1_late_share:.6f}`
- 两个 action1 均未召回：`{action1_all_missed}`
- action4/action7 均已召回：`{other_all_recalled}`
- Branch：`{branch}`

## 结论边界

本轮只判断组内 PER 是否对 action1 形成明显低采样。未现场修改采样规则，也未启动在线训练。
"""
    DOC.write_text(record, encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
