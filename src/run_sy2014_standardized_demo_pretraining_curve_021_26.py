from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
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
from stable_baselines3 import DQN


OUT = ROOT / "benchmark_results" / "021_26"
DOC = ROOT / "docs" / "2026-07-15_021_26_sy2014_standardized_demo_pretraining_sufficiency_curve.md"
MILESTONES = [0, 10, 25, 50, 100, 250, 500, 1000, 2000, 5000]


class OfflineShapeEnv(gym.Env):
    metadata: dict[str, Any] = {}

    def __init__(self, dimensions: int) -> None:
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(dimensions,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(9)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action: int):
        raise RuntimeError("021_26 is offline-only; environment stepping is forbidden")


def evaluate_demonstrations(
    model: DQN,
    observations: np.ndarray,
    actions: np.ndarray,
    raw_observations: np.ndarray,
    milestone: int,
    recent_updates: pd.DataFrame,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=model.device)
    target_actions = torch.as_tensor(actions, dtype=torch.long, device=model.device)
    with torch.no_grad():
        q = model.q_net(obs_tensor)
    predictions = q.argmax(dim=1)
    expert_q = q.gather(1, target_actions[:, None]).squeeze(1)
    other = q.clone()
    other[torch.arange(len(actions), device=model.device), target_actions] = -torch.inf
    margins = expert_q - other.max(dim=1).values
    prediction_array = predictions.cpu().numpy()
    margin_array = margins.cpu().numpy()
    nonzero = actions != 0
    parameters_finite = all(torch.isfinite(parameter).all().item() for parameter in model.q_net.parameters())
    q_finite = bool(torch.isfinite(q).all().item())
    nonzero_recall = float(np.mean(prediction_array[nonzero] == actions[nonzero]))
    noop_accuracy = float(np.mean(prediction_array[~nonzero] == 0))
    positive_margin_fraction = float(np.mean(margin_array[nonzero] > 0))
    passed = bool(
        nonzero_recall >= 0.8 and noop_accuracy >= 0.95
        and positive_margin_fraction >= 0.8 and parameters_finite and q_finite
    )
    if recent_updates.empty:
        update_stats = {
            "interval_update_count": 0,
            "median_td_1_weighted": np.nan,
            "median_td_n_weighted": np.nan,
            "median_margin_weighted": np.nan,
            "median_total_gradient": np.nan,
            "gradient_clip_fraction": np.nan,
        }
    else:
        update_stats = {
            "interval_update_count": int(len(recent_updates)),
            "median_td_1_weighted": float(recent_updates.td_1_weighted.median()),
            "median_td_n_weighted": float(recent_updates.td_n_weighted.median()),
            "median_margin_weighted": float(recent_updates.margin_weighted.median()),
            "median_total_gradient": float(recent_updates.grad_total_before_clip.median()),
            "gradient_clip_fraction": float(recent_updates.gradient_would_clip.mean()),
        }
    summary = {
        "milestone_updates": milestone,
        "overall_accuracy": float(np.mean(prediction_array == actions)),
        "noop_accuracy": noop_accuracy,
        "nonzero_action_recall": nonzero_recall,
        "predicted_nonzero_fraction": float(np.mean(prediction_array != 0)),
        "positive_expert_margin_fraction_nonzero": positive_margin_fraction,
        "mean_expert_margin_nonzero": float(np.mean(margin_array[nonzero])),
        "minimum_expert_margin_nonzero": float(np.min(margin_array[nonzero])),
        "q_abs_mean": float(q.abs().mean().cpu()),
        "q_abs_max": float(q.abs().max().cpu()),
        "q_values_finite": q_finite,
        "parameters_finite": parameters_finite,
        "milestone_gate_passed": passed,
        **update_stats,
    }
    event_rows = []
    for index in np.where(nonzero)[0]:
        event_rows.append({
            "milestone_updates": milestone,
            "transition_index": int(index),
            "dap": int(round(float(raw_observations[index, 1]))),
            "target_action": int(actions[index]),
            "predicted_action": int(prediction_array[index]),
            "target_action_recalled": bool(prediction_array[index] == actions[index]),
            "expert_q": float(expert_q[index].cpu()),
            "best_other_q": float(other[index].max().cpu()),
            "expert_margin": float(margin_array[index]),
        })
    return summary, event_rows


def plot_curve(curve: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot(curve.milestone_updates, curve.nonzero_action_recall, marker="o", label="nonzero recall", color="#D55E00")
    axes[0, 0].plot(curve.milestone_updates, curve.noop_accuracy, marker="o", label="no-op accuracy", color="#0072B2")
    axes[0, 0].axhline(0.8, color="#D55E00", linestyle="--", linewidth=1)
    axes[0, 0].axhline(0.95, color="#0072B2", linestyle="--", linewidth=1)
    axes[0, 0].set_ylim(-0.03, 1.03); axes[0, 0].set_ylabel("Fraction"); axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(curve.milestone_updates, curve.mean_expert_margin_nonzero, marker="o", color="#009E73", label="mean margin")
    axes[0, 1].plot(curve.milestone_updates, curve.minimum_expert_margin_nonzero, marker="o", color="#CC79A7", label="min margin")
    axes[0, 1].axhline(0, color="#333333", linestyle="--", linewidth=1)
    axes[0, 1].set_ylabel("Expert Q margin"); axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(curve.milestone_updates, curve.median_td_1_weighted, marker="o", label="TD1", color="#4C78A8")
    axes[1, 0].plot(curve.milestone_updates, curve.median_td_n_weighted, marker="o", label="n-step", color="#F58518")
    axes[1, 0].plot(curve.milestone_updates, curve.median_margin_weighted, marker="o", label="margin", color="#54A24B")
    axes[1, 0].set_yscale("symlog", linthresh=1e-3); axes[1, 0].set_ylabel("Interval median loss"); axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(curve.milestone_updates, curve.median_total_gradient, marker="o", color="#E45756", label="gradient")
    axes[1, 1].plot(curve.milestone_updates, curve.gradient_clip_fraction, marker="o", color="#72B7B2", label="clip fraction")
    axes[1, 1].set_yscale("symlog", linthresh=1e-3); axes[1, 1].set_ylabel("Interval metric"); axes[1, 1].legend(frameon=False)
    for ax in axes.flat:
        ax.set_xlabel("Demonstration pretraining updates"); ax.grid(alpha=0.2)
    fig.suptitle("SY2014 standardized-observation demonstration pretraining curve")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "021_26_pretraining_sufficiency_curve.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "021_26_pretraining_sufficiency_curve.svg", bbox_inches="tight")
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
    np.random.seed(0)
    torch.manual_seed(0)
    raw, standardized, validation = normbase.prepare_demonstrations()
    if not validation["all_pre_run_checks_pass"]:
        raise RuntimeError("Frozen standardization validation failed")
    observations = standardized["observations"]
    actions = np.asarray(standardized["actions"], dtype=np.int64).reshape(-1)
    env = OfflineShapeEnv(observations.shape[1])
    model = DQN("MlpPolicy", env, verbose=0, **dqn_kwargs(seed=0))
    replay = PrioritizedDemonstrationReplay(
        standardized, agent_capacity=1, alpha=0.4,
        epsilon_demo=1.0, epsilon_agent=0.001, seed=21022,
    )

    curve_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    previous = 0
    summary, events = evaluate_demonstrations(
        model, observations, actions, raw["observations"], 0, pd.DataFrame()
    )
    curve_rows.append(summary); event_rows.extend(events)
    for milestone in MILESTONES[1:]:
        interval_rows = []
        for update in range(previous + 1, milestone + 1):
            row = fast_update(
                model, replay, phase="offline_pretrain", update=update,
                env_step=0, epsilon=1.0,
            )
            interval_rows.append(row); update_rows.append(row)
        interval = pd.DataFrame(interval_rows)
        summary, events = evaluate_demonstrations(
            model, observations, actions, raw["observations"], milestone, interval
        )
        curve_rows.append(summary); event_rows.extend(events)
        previous = milestone
    env.close()

    curve = pd.DataFrame(curve_rows)
    events = pd.DataFrame(event_rows)
    updates = pd.DataFrame(update_rows)
    curve.to_csv(OUT / "021_26_pretraining_learning_curve.csv", index=False, encoding="utf-8-sig")
    events.to_csv(OUT / "021_26_nonzero_event_predictions.csv", index=False, encoding="utf-8-sig")
    updates.to_csv(OUT / "021_26_update_diagnostics.csv", index=False, encoding="utf-8-sig")

    passes = curve.milestone_gate_passed.tolist()
    consecutive = any(passes[index] and passes[index + 1] for index in range(len(passes) - 1))
    isolated = bool(any(passes) and not consecutive)
    branch = "A_pretraining_insufficiency_participant" if consecutive else (
        "B_isolated_unstable_pass" if isolated else "C_more_current_pretraining_not_sufficient"
    )
    summary_json = {
        "status": "completed_offline_no_dssat",
        "milestones": MILESTONES,
        "demo_transition_count": int(len(actions)),
        "demo_nonzero_action_count": int((actions != 0).sum()),
        "passing_milestones": curve.loc[curve.milestone_gate_passed, "milestone_updates"].astype(int).tolist(),
        "two_consecutive_milestones_passed": consecutive,
        "pre_registered_interpretation_branch": branch,
        "no_online_training_performed": True,
        "no_parameter_changed_after_results": True,
    }
    (OUT / "021_26_summary.json").write_text(
        json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_curve(curve)

    display = curve[[
        "milestone_updates", "overall_accuracy", "noop_accuracy", "nonzero_action_recall",
        "positive_expert_margin_fraction_nonzero", "mean_expert_margin_nonzero",
        "median_total_gradient", "gradient_clip_fraction", "milestone_gate_passed",
    ]].round(6)
    record = f"""# 021_26 SY2014 标准化观测示范预训练充分度学习曲线

## 边界

本任务完全离线，不调用 DSSAT/PDI、不进行在线训练。所有里程碑、DQfD/PER参数、target冻结方式和判据均在运行前固定；结果不能直接用于挑选一个最好预训练步数。

## 学习曲线

{markdown_table(display)}

## 预注册判定

- passing milestones: `{summary_json['passing_milestones']}`
- two consecutive milestones passed: `{consecutive}`
- branch: `{branch}`

## 解释边界

该结果只判断“当前固定损失下，单纯延长示范预训练是否足以学习5个稀疏动作”，不代表在线策略、产量或跨seed稳定性。

## 输出

- `benchmark_results/021_26/021_26_pretraining_learning_curve.csv`
- `benchmark_results/021_26/021_26_nonzero_event_predictions.csv`
- `benchmark_results/021_26/021_26_update_diagnostics.csv`
- `benchmark_results/021_26/021_26_summary.json`
- `benchmark_results/021_26/021_26_pretraining_sufficiency_curve.png/.svg`
"""
    DOC.write_text(record, encoding="utf-8")
    print(json.dumps(summary_json, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
