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


OUT = ROOT / "benchmark_results" / "021_33"
DOC = ROOT / "docs" / "2026-07-15_021_33_sy2014_false_positive_temporal_neighborhood_audit.md"
MILESTONES = [0, 10, 25, 50, 100, 250, 500, 1000]
STATE_AUDIT_MILESTONES = {250, 500, 1000}
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


def state_predictions(
    model,
    observations: np.ndarray,
    raw_observations: np.ndarray,
    target_actions: np.ndarray,
    milestone: int,
) -> pd.DataFrame:
    tensor = torch.as_tensor(observations, dtype=torch.float32, device=model.device)
    with torch.no_grad():
        q_values = model.q_net(tensor).detach().cpu().numpy()
    predictions = q_values.argmax(axis=1)
    sorted_q = np.sort(q_values, axis=1)
    predicted_margin = sorted_q[:, -1] - sorted_q[:, -2]
    daps = np.rint(raw_observations[:, 1]).astype(int)
    oracle_indices = np.where(target_actions != 0)[0]
    oracle_daps = daps[oracle_indices]
    oracle_actions = target_actions[oracle_indices]
    rows: list[dict[str, Any]] = []
    for index, (dap, target, predicted) in enumerate(zip(daps, target_actions, predictions)):
        distances = np.abs(oracle_daps - dap)
        nearest_position = int(np.argmin(distances))
        nearest_distance = int(distances[nearest_position])
        nearest_dap = int(oracle_daps[nearest_position])
        nearest_action = int(oracle_actions[nearest_position])
        same_positions = np.where(oracle_actions == predicted)[0]
        if len(same_positions):
            same_distances = np.abs(oracle_daps[same_positions] - dap)
            same_local = int(np.argmin(same_distances))
            same_position = int(same_positions[same_local])
            same_distance: float = float(same_distances[same_local])
            same_dap: float = float(oracle_daps[same_position])
        else:
            same_distance = float("nan")
            same_dap = float("nan")
        false_positive = bool(target == 0 and predicted != 0)
        row = {
            "milestone_updates": int(milestone),
            "transition_index": int(index),
            "dap": int(dap),
            "target_action": int(target),
            "predicted_action": int(predicted),
            "false_positive_noop": false_positive,
            "prediction_correct": bool(target == predicted),
            "predicted_q_margin": float(predicted_margin[index]),
            "nearest_oracle_dap": nearest_dap,
            "nearest_oracle_action": nearest_action,
            "nearest_oracle_distance_dap": nearest_distance,
            "nearest_same_action_oracle_dap": same_dap,
            "nearest_same_action_distance_dap": same_distance,
            "q_values_finite": bool(np.isfinite(q_values[index]).all()),
        }
        for window in (1, 3, 7):
            row[f"within_any_oracle_{window}dap"] = bool(nearest_distance <= window)
            row[f"within_same_action_oracle_{window}dap"] = bool(
                np.isfinite(same_distance) and same_distance <= window
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_results(all_states: pd.DataFrame, false_positives: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for axis, milestone in zip(axes[0], (500, 1000)):
        frame = all_states[all_states.milestone_updates == milestone]
        predicted = frame[frame.predicted_action != 0]
        oracle = frame[frame.target_action != 0]
        axis.scatter(predicted.dap, predicted.predicted_action, color="#D55E00", marker="o", label="predicted nonzero")
        axis.scatter(oracle.dap, oracle.target_action, color="#0072B2", marker="x", s=55, label="oracle event")
        axis.set_title(f"{milestone} updates")
        axis.set_ylabel("Action index")
        axis.legend(frameon=False, fontsize=8)
    for milestone, color in ((500, "#009E73"), (1000, "#CC79A7")):
        frame = false_positives[false_positives.milestone_updates == milestone]
        axes[1, 0].hist(
            frame.nearest_oracle_distance_dap,
            bins=np.arange(-0.5, max(8.5, frame.nearest_oracle_distance_dap.max() + 1.5), 1),
            alpha=0.55, color=color, label=f"{milestone}",
        )
    axes[1, 0].set_ylabel("False-positive count")
    axes[1, 0].set_xlabel("Distance to nearest oracle event (DAP)")
    axes[1, 0].legend(frameon=False, fontsize=8)
    plot_summary = summary[summary.milestone_updates.isin([500, 1000])]
    x = np.arange(len(plot_summary)); width = 0.36
    axes[1, 1].bar(x - width / 2, plot_summary.fraction_within_any_3dap, width, label="any event ±3 DAP")
    axes[1, 1].bar(x + width / 2, plot_summary.fraction_within_same_action_3dap, width, label="same action ±3 DAP")
    axes[1, 1].set_xticks(x, plot_summary.milestone_updates.astype(str))
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_ylabel("False-positive fraction")
    axes[1, 1].set_xlabel("Updates")
    axes[1, 1].legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
        if ax in axes[0]:
            ax.set_xlabel("DAP")
    fig.suptitle("SY2014 temporal neighborhood of false-positive operations")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "021_33_false_positive_temporal_neighborhood.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "021_33_false_positive_temporal_neighborhood.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    np.random.seed(0)
    torch.manual_seed(0)
    raw, standardized, validation = normbase.prepare_demonstrations()
    if not validation["all_pre_run_checks_pass"]:
        raise RuntimeError("Frozen standardization validation failed")
    demonstrations = full_return_to_go(standardized)
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
    state = {"update": 0}
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
        data = {name: np.asarray(values)[chosen] for name, values in replay_object.demonstrations.items()}
        return ReplaySample(
            global_indices=chosen,
            probabilities=0.5 * conditional,
            importance_weights=importance.astype(np.float32),
            is_demonstration=np.ones(len(chosen), dtype=bool),
            data=data,
        )

    exp.stratified_sample = event_balanced_sampler
    curve_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    all_state_frames: list[pd.DataFrame] = []
    initial, _ = exp.evaluate_demonstrations(
        model, observations, actions, raw["observations"], 0, pd.DataFrame()
    )
    curve_rows.append(initial)
    previous = 0
    try:
        for milestone in MILESTONES[1:]:
            interval_rows: list[dict[str, Any]] = []
            for update in range(previous + 1, milestone + 1):
                state["update"] = update
                row = exp.update_model(
                    model, replay, actions, rng,
                    arm="treatment_event_balanced", update=update,
                )
                interval_rows.append(row); update_rows.append(row)
            summary, _ = exp.evaluate_demonstrations(
                model, observations, actions, raw["observations"],
                milestone, pd.DataFrame(interval_rows),
            )
            curve_rows.append(summary)
            if milestone in STATE_AUDIT_MILESTONES:
                all_state_frames.append(state_predictions(
                    model, observations, raw["observations"], actions, milestone
                ))
            previous = milestone
    finally:
        exp.stratified_sample = original_sampler
        env.close()

    curve = pd.DataFrame(curve_rows)
    updates = pd.DataFrame(update_rows)
    all_states = pd.concat(all_state_frames, ignore_index=True)
    false_positives = all_states[all_states.false_positive_noop].copy()
    summary_rows: list[dict[str, Any]] = []
    for milestone in sorted(STATE_AUDIT_MILESTONES):
        frame = all_states[all_states.milestone_updates == milestone]
        fp = frame[frame.false_positive_noop]
        row: dict[str, Any] = {
            "milestone_updates": milestone,
            "false_positive_count": int(len(fp)),
            "noop_state_count": int((frame.target_action == 0).sum()),
            "noop_accuracy_from_states": float(
                (frame[frame.target_action == 0].predicted_action == 0).mean()
            ),
        }
        for window in (1, 3, 7):
            row[f"fraction_within_any_{window}dap"] = float(fp[f"within_any_oracle_{window}dap"].mean())
            row[f"fraction_within_same_action_{window}dap"] = float(fp[f"within_same_action_oracle_{window}dap"].mean())
        summary_rows.append(row)
    temporal_summary = pd.DataFrame(summary_rows)
    curve.to_csv(OUT / "021_33_learning_curve.csv", index=False, encoding="utf-8-sig")
    updates.to_csv(OUT / "021_33_update_diagnostics.csv", index=False, encoding="utf-8-sig")
    all_states.to_csv(OUT / "021_33_all_state_predictions.csv", index=False, encoding="utf-8-sig")
    false_positives.to_csv(OUT / "021_33_false_positive_states.csv", index=False, encoding="utf-8-sig")
    temporal_summary.to_csv(OUT / "021_33_temporal_summary.csv", index=False, encoding="utf-8-sig")

    reference = pd.read_csv(ROOT / "benchmark_results" / "021_32" / "021_32_learning_curve.csv")
    reference = reference[reference.arm == "treatment_event_balanced"].sort_values("milestone_updates")
    aligned = curve.sort_values("milestone_updates")
    comparison_columns = [
        "noop_accuracy", "nonzero_action_recall",
        "positive_expert_margin_fraction_nonzero", "mean_expert_margin_nonzero",
    ]
    max_errors = {
        column: float(np.max(np.abs(aligned[column].to_numpy() - reference[column].to_numpy())))
        for column in comparison_columns
    }
    count_checks = {}
    for milestone in (500, 1000):
        curve_row = aligned[aligned.milestone_updates == milestone].iloc[0]
        expected = int(round(155 * (1.0 - float(curve_row.noop_accuracy))))
        observed = int(temporal_summary.loc[
            temporal_summary.milestone_updates == milestone, "false_positive_count"
        ].iloc[0])
        count_checks[str(milestone)] = {"expected": expected, "observed": observed, "match": expected == observed}
    validation_result = {
        "021_32_treatment_max_abs_errors": max_errors,
        "learning_curve_reproduced": bool(max(max_errors.values()) < 1e-6),
        "false_positive_count_checks": count_checks,
        "false_positive_counts_match": bool(all(value["match"] for value in count_checks.values())),
        "all_q_values_finite": bool(all_states.q_values_finite.all()),
        "all_updates_finite": bool(updates.all_finite.all()),
    }
    validation_result["passed"] = bool(
        validation_result["learning_curve_reproduced"]
        and validation_result["false_positive_counts_match"]
        and validation_result["all_q_values_finite"]
        and validation_result["all_updates_finite"]
    )
    (OUT / "021_33_validation.json").write_text(
        json.dumps(validation_result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not validation_result["passed"]:
        raise RuntimeError(f"Validation failed: {validation_result}")

    primary = temporal_summary[temporal_summary.milestone_updates == 500].iloc[0]
    any3 = float(primary.fraction_within_any_3dap)
    same3 = float(primary.fraction_within_same_action_3dap)
    if any3 >= 0.8 and same3 >= 0.8:
        branch = "A_false_positives_temporally_local"
    elif any3 >= 0.5 or same3 >= 0.5:
        branch = "B_mixed_temporal_and_dispersed_false_positives"
    else:
        branch = "C_false_positives_dispersed_across_season"
    result = {
        "status": "completed_offline_no_dssat",
        "validation_passed": validation_result["passed"],
        "primary_milestone": 500,
        "false_positive_count_500": int(primary.false_positive_count),
        "fraction_within_any_oracle_3dap_500": any3,
        "fraction_within_same_action_oracle_3dap_500": same3,
        "interpretation_branch": branch,
        "agronomic_equivalence_not_claimed": True,
        "no_online_training_performed": True,
    }
    (OUT / "021_33_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_results(all_states, false_positives, temporal_summary)

    summary_display = temporal_summary.round(6)
    fp500 = false_positives[false_positives.milestone_updates == 500][[
        "transition_index", "dap", "predicted_action",
        "nearest_oracle_dap", "nearest_oracle_action", "nearest_oracle_distance_dap",
        "nearest_same_action_oracle_dap", "nearest_same_action_distance_dap",
        "predicted_q_margin",
    ]].round(6)
    record = f"""# 021_33 SY2014 false-positive 时序邻域离线审计

## 边界

本任务严格复现 021_32 event-balanced Treatment，只在 250/500/1000 次更新读取全部状态预测；未修改训练，也未调用 DSSAT/PDI。“时序接近”不等于“农艺等价”。

## 强制验证

```json
{json.dumps(validation_result, indent=2, ensure_ascii=False)}
```

## 时序邻域汇总

{markdown_table(summary_display)}

## 500 次更新 false positives

{markdown_table(fp500)}

## 预注册判定

- 500 次 false-positive 数：`{int(primary.false_positive_count)}`
- 任一 oracle 事件 ±3 DAP 比例：`{any3:.6f}`
- 同动作 oracle 事件 ±3 DAP 比例：`{same3:.6f}`
- Branch：`{branch}`

## 结论边界

`{branch}`。该结论只区分 false positive 的时序分布，不证明相邻日期操作具有相同农艺效果，也未自动启动在线训练。
"""
    DOC.write_text(record, encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
