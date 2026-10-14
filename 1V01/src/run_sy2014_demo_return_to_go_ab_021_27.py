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
from frozen_nstep_dqn_config_020_11 import dqn_kwargs
from literature_aligned_dqfd import PrioritizedDemonstrationReplay
from run_sy2014_observation_standardization_5k_ab_021_25 import fast_update
from run_sy2014_standardized_demo_pretraining_curve_021_26 import OfflineShapeEnv, evaluate_demonstrations
from stable_baselines3 import DQN


OUT = ROOT / "benchmark_results" / "021_27"
DOC = ROOT / "docs" / "2026-07-15_021_27_sy2014_demo_return_to_go_credit_assignment_ab.md"
MILESTONES = [0, 10, 25, 50, 100, 250, 500, 1000]
GAMMA = 0.99


def full_return_to_go(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    treatment = {name: np.asarray(value).copy() for name, value in data.items()}
    rewards = np.asarray(data["rewards"], dtype=np.float32).reshape(-1)
    dones = np.asarray(data["dones"], dtype=np.float32).reshape(-1)
    next_observations = np.asarray(data["next_observations"], dtype=np.float32)
    count = len(rewards)
    returns = np.zeros(count, dtype=np.float32)
    n_next = np.zeros_like(next_observations)
    n_dones = np.zeros(count, dtype=np.float32)
    discounts = np.zeros(count, dtype=np.float32)
    horizons = np.zeros(count, dtype=np.int64)
    for start in range(count):
        value = 0.0
        last = start
        horizon = 0
        for index in range(start, count):
            value += (GAMMA ** (index - start)) * float(rewards[index])
            last = index
            horizon += 1
            if bool(dones[index]):
                break
        if not bool(dones[last]):
            raise RuntimeError(f"Transition {start} does not reach terminal state")
        returns[start] = value
        n_next[start] = next_observations[last]
        n_dones[start] = 1.0
        discounts[start] = float(GAMMA ** horizon)
        horizons[start] = horizon
    treatment["n_step_returns"] = returns
    treatment["n_step_next_observations"] = n_next
    treatment["n_step_dones"] = n_dones
    treatment["n_step_discounts"] = discounts
    treatment["n_step_horizons"] = horizons
    return treatment


def target_audit(
    raw: dict[str, np.ndarray], control: dict[str, np.ndarray], treatment: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    actions = np.asarray(raw["actions"]).reshape(-1)
    nonzero_indices = np.where(actions != 0)[0]
    rows = []
    for index in nonzero_indices:
        rows.append({
            "transition_index": int(index),
            "dap": int(round(float(raw["observations"][index, 1]))),
            "action": int(actions[index]),
            "immediate_reward": float(raw["rewards"][index]),
            "control_5step_return": float(control["n_step_returns"][index]),
            "treatment_full_return_to_go": float(treatment["n_step_returns"][index]),
            "control_horizon": int(control["n_step_horizons"][index]),
            "treatment_horizon": int(treatment["n_step_horizons"][index]),
            "treatment_terminal": bool(treatment["n_step_dones"][index]),
        })
    frame = pd.DataFrame(rows)
    unchanged = ["observations", "actions", "rewards", "next_observations", "dones"]
    validation = {
        "unchanged_fields_max_abs_error": {
            name: float(np.max(np.abs(np.asarray(control[name]) - np.asarray(treatment[name]))))
            for name in unchanged
        },
        "all_treatment_targets_terminal": bool(np.all(treatment["n_step_dones"] == 1)),
        "all_nonzero_full_returns_positive": bool((frame.treatment_full_return_to_go > 0).all()),
        "all_arrays_finite": bool(all(np.isfinite(value).all() for value in treatment.values())),
    }
    validation["passed"] = bool(
        max(validation["unchanged_fields_max_abs_error"].values()) == 0.0
        and validation["all_treatment_targets_terminal"]
        and validation["all_nonzero_full_returns_positive"]
        and validation["all_arrays_finite"]
    )
    return frame, validation


def run_arm(
    arm: str,
    demonstrations: dict[str, np.ndarray],
    raw_observations: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    np.random.seed(0)
    torch.manual_seed(0)
    observations = demonstrations["observations"]
    actions = np.asarray(demonstrations["actions"], dtype=np.int64).reshape(-1)
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
    curve_rows.append(summary); event_rows.extend(events)
    previous = 0
    for milestone in MILESTONES[1:]:
        interval_rows = []
        for update in range(previous + 1, milestone + 1):
            row = fast_update(
                model, replay, phase="offline_pretrain", update=update,
                env_step=0, epsilon=1.0,
            )
            row["arm"] = arm
            interval_rows.append(row); update_rows.append(row)
        interval = pd.DataFrame(interval_rows)
        summary, events = evaluate_demonstrations(
            model, observations, actions, raw_observations, milestone, interval
        )
        summary["arm"] = arm
        for event in events:
            event["arm"] = arm
        curve_rows.append(summary); event_rows.extend(events)
        previous = milestone
    env.close()
    return pd.DataFrame(curve_rows), pd.DataFrame(event_rows), pd.DataFrame(update_rows)


def consecutive_pass(frame: pd.DataFrame) -> bool:
    values = frame.sort_values("milestone_updates").milestone_gate_passed.tolist()
    return any(values[index] and values[index + 1] for index in range(len(values) - 1))


def plot_results(curve: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    colors = {"control_5step": "#777777", "treatment_full_return_to_go": "#009E73"}
    for arm, frame in curve.groupby("arm"):
        frame = frame.sort_values("milestone_updates")
        axes[0, 0].plot(frame.milestone_updates, frame.nonzero_action_recall, marker="o", color=colors[arm], label=arm)
        axes[0, 1].plot(frame.milestone_updates, frame.noop_accuracy, marker="o", color=colors[arm], label=arm)
        axes[1, 0].plot(frame.milestone_updates, frame.mean_expert_margin_nonzero, marker="o", color=colors[arm], label=arm)
        axes[1, 1].plot(frame.milestone_updates, frame.median_total_gradient, marker="o", color=colors[arm], label=arm)
    axes[0, 0].axhline(0.8, color="#333333", linestyle="--", linewidth=1); axes[0, 0].set_ylabel("Nonzero recall")
    axes[0, 1].axhline(0.95, color="#333333", linestyle="--", linewidth=1); axes[0, 1].set_ylabel("No-op accuracy")
    axes[1, 0].axhline(0, color="#333333", linestyle="--", linewidth=1); axes[1, 0].set_ylabel("Mean expert Q margin")
    axes[1, 1].set_yscale("symlog", linthresh=1e-3); axes[1, 1].set_ylabel("Interval median gradient")
    for ax in axes.flat:
        ax.set_xlabel("Demonstration updates"); ax.grid(alpha=0.2); ax.legend(frameon=False, fontsize=8)
    fig.suptitle("SY2014 demonstration credit assignment: 5-step vs full return-to-go")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "021_27_return_to_go_credit_assignment_ab.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "021_27_return_to_go_credit_assignment_ab.svg", bbox_inches="tight")
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
    raw, control, scaler_validation = normbase.prepare_demonstrations()
    if not scaler_validation["all_pre_run_checks_pass"]:
        raise RuntimeError("Frozen standardization validation failed")
    treatment = full_return_to_go(control)
    target_frame, validation = target_audit(raw, control, treatment)
    target_frame.to_csv(OUT / "021_27_target_audit.csv", index=False, encoding="utf-8-sig")
    (OUT / "021_27_target_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not validation["passed"]:
        raise RuntimeError(f"Return-to-go validation failed: {validation}")

    curves = []; events = []; updates = []
    for arm, data in (("control_5step", control), ("treatment_full_return_to_go", treatment)):
        curve, event, update = run_arm(arm, data, raw["observations"])
        curves.append(curve); events.append(event); updates.append(update)
    curve = pd.concat(curves, ignore_index=True)
    event_frame = pd.concat(events, ignore_index=True)
    update_frame = pd.concat(updates, ignore_index=True)
    curve.to_csv(OUT / "021_27_learning_curve.csv", index=False, encoding="utf-8-sig")
    event_frame.to_csv(OUT / "021_27_nonzero_event_predictions.csv", index=False, encoding="utf-8-sig")
    update_frame.to_csv(OUT / "021_27_update_diagnostics.csv", index=False, encoding="utf-8-sig")

    control_curve = curve[curve.arm == "control_5step"]
    treatment_curve = curve[curve.arm == "treatment_full_return_to_go"]
    control_consecutive = consecutive_pass(control_curve)
    treatment_consecutive = consecutive_pass(treatment_curve)
    treatment_any = bool(treatment_curve.milestone_gate_passed.any())
    if treatment_consecutive and not control_consecutive:
        branch = "A_full_return_credit_assignment_participant"
    elif treatment_any or float(treatment_curve.nonzero_action_recall.max()) > float(control_curve.nonzero_action_recall.max()):
        branch = "B_partial_but_not_stable_signal"
    else:
        branch = "C_full_return_to_go_not_sufficient"
    summary = {
        "status": "completed_offline_no_dssat",
        "target_validation_passed": validation["passed"],
        "control_passing_milestones": control_curve.loc[control_curve.milestone_gate_passed, "milestone_updates"].astype(int).tolist(),
        "treatment_passing_milestones": treatment_curve.loc[treatment_curve.milestone_gate_passed, "milestone_updates"].astype(int).tolist(),
        "control_two_consecutive_pass": control_consecutive,
        "treatment_two_consecutive_pass": treatment_consecutive,
        "pre_registered_interpretation_branch": branch,
        "no_online_training_performed": True,
    }
    (OUT / "021_27_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_results(curve)

    display = curve[[
        "arm", "milestone_updates", "noop_accuracy", "nonzero_action_recall",
        "positive_expert_margin_fraction_nonzero", "mean_expert_margin_nonzero",
        "median_total_gradient", "gradient_clip_fraction", "milestone_gate_passed",
    ]].round(6)
    record = f"""# 021_27 SY2014 示范 return-to-go 信用分配离线 A/B

## 边界

本任务完全离线。Control与Treatment唯一差异是示范长期目标：5-step return对比终止前完整折扣return-to-go。即时reward、1-step target、标准化、网络、PER和loss权重均未改变。

## 非零动作目标审计

{markdown_table(target_frame.round(6))}

## 学习曲线

{markdown_table(display)}

## 预注册判定

- control passing milestones: `{summary['control_passing_milestones']}`
- treatment passing milestones: `{summary['treatment_passing_milestones']}`
- treatment consecutive pass: `{treatment_consecutive}`
- branch: `{branch}`

该结果只证明或否定离线示范信用目标的作用，不代表在线DQN产量或稳定性；不自动进入在线训练。

## 输出

- `benchmark_results/021_27/021_27_target_audit.csv`
- `benchmark_results/021_27/021_27_target_validation.json`
- `benchmark_results/021_27/021_27_learning_curve.csv`
- `benchmark_results/021_27/021_27_nonzero_event_predictions.csv`
- `benchmark_results/021_27/021_27_update_diagnostics.csv`
- `benchmark_results/021_27/021_27_summary.json`
- `benchmark_results/021_27/021_27_return_to_go_credit_assignment_ab.png/.svg`
"""
    DOC.write_text(record, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
