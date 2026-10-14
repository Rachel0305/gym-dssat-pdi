from __future__ import annotations

import json
import sys
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
import run_sy2014_fixed_observation_standardization_diagnostic_021_24 as normbase
from frozen_nstep_dqn_config_020_11 import dqn_kwargs
from literature_aligned_dqfd import PrioritizedDemonstrationReplay, dqfd_loss_components
from run_sy2014_demo_return_to_go_ab_021_27 import full_return_to_go
from run_sy2014_standardized_demo_pretraining_curve_021_26 import (
    OfflineShapeEnv,
    evaluate_demonstrations,
)
from stable_baselines3 import DQN


OUT = ROOT / "benchmark_results" / "021_29"
DOC = ROOT / "docs" / "2026-07-15_021_29_sy2014_demo_sampling_gradient_balance_audit.md"
MILESTONES = [0, 1, 10, 25, 50, 100, 250, 500, 1000]
AUDIT_UPDATES = set(MILESTONES[1:])


def gradient_vector(loss: torch.Tensor, parameters: list[torch.Tensor]) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True
    )
    values = []
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            values.append(torch.zeros_like(parameter).reshape(-1))
        else:
            values.append(gradient.reshape(-1))
    return torch.cat(values)


def safe_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = left.norm() * right.norm()
    if float(denominator.detach().cpu()) == 0.0:
        return float("nan")
    return float(torch.dot(left, right).div(denominator).detach().cpu())


def audited_update(
    model: DQN,
    replay: PrioritizedDemonstrationReplay,
    demo_actions: np.ndarray,
    demo_dones: np.ndarray,
    *,
    update: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    probabilities = replay.sampling_probabilities()
    priorities = replay.combined_priorities()
    demo_nonzero = demo_actions != 0
    demo_terminal = demo_dones.astype(bool)

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

    chosen_q = q_values.gather(1, actions[:, None]).squeeze(1)
    td_n_each = F.smooth_l1_loss(chosen_q, target_n, reduction="none") * weights
    margins = torch.full_like(q_values, 0.8)
    margins.scatter_(1, actions[:, None], 0.0)
    margin_each = (torch.max(q_values + margins, dim=1).values - chosen_q) * weights
    sample_nonzero = actions != 0
    sample_noop = ~sample_nonzero
    batch_size = float(len(actions))

    def contribution(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        td = (td_n_each * mask.float()).sum() / batch_size
        margin = (margin_each * mask.float()).sum() / batch_size
        return td, margin, td + margin

    noop_td, noop_margin, noop_objective = contribution(sample_noop)
    nonzero_td, nonzero_margin, nonzero_objective = contribution(sample_nonzero)
    total = noop_objective + nonzero_objective + losses["l2_weighted"]
    reconstruction_error = float(
        torch.abs(total - (losses["td_n_weighted"] + losses["margin_weighted"] + losses["l2_weighted"]))
        .detach().cpu()
    )
    reconstruction_relative_error = reconstruction_error / max(
        abs(float(total.detach().cpu())), 1.0
    )

    audit: dict[str, Any] | None = None
    if update in AUDIT_UPDATES:
        noop_gradient = gradient_vector(noop_objective, parameters)
        nonzero_gradient = gradient_vector(nonzero_objective, parameters)
        sample_indices = sample.global_indices.astype(int)
        sample_demo_actions = demo_actions[sample_indices]
        sample_demo_dones = demo_dones[sample_indices].astype(bool)

        def priority_stats(mask: np.ndarray, prefix: str) -> dict[str, Any]:
            values = priorities[mask]
            return {
                f"{prefix}_population_count": int(mask.sum()),
                f"{prefix}_priority_median": float(np.median(values)),
                f"{prefix}_priority_max": float(np.max(values)),
                f"{prefix}_probability_mass": float(probabilities[mask].sum()),
            }

        audit = {
            "update": int(update),
            "batch_size": int(len(actions)),
            "sample_noop_count": int((sample_demo_actions == 0).sum()),
            "sample_nonzero_count": int((sample_demo_actions != 0).sum()),
            "sample_terminal_count": int(sample_demo_dones.sum()),
            "sample_nonzero_fraction": float((sample_demo_actions != 0).mean()),
            "sample_terminal_fraction": float(sample_demo_dones.mean()),
            **priority_stats(~demo_nonzero, "noop"),
            **priority_stats(demo_nonzero, "nonzero"),
            **priority_stats(demo_terminal, "terminal"),
            "noop_full_return_td_contribution": float(noop_td.detach().cpu()),
            "noop_margin_contribution": float(noop_margin.detach().cpu()),
            "noop_objective_contribution": float(noop_objective.detach().cpu()),
            "nonzero_full_return_td_contribution": float(nonzero_td.detach().cpu()),
            "nonzero_margin_contribution": float(nonzero_margin.detach().cpu()),
            "nonzero_objective_contribution": float(nonzero_objective.detach().cpu()),
            "noop_gradient_norm": float(noop_gradient.norm().detach().cpu()),
            "nonzero_gradient_norm": float(nonzero_gradient.norm().detach().cpu()),
            "noop_to_nonzero_gradient_norm_ratio": (
                float(noop_gradient.norm().div(nonzero_gradient.norm()).detach().cpu())
                if float(nonzero_gradient.norm().detach().cpu()) > 0 else float("inf")
            ),
            "noop_nonzero_gradient_cosine": safe_cosine(noop_gradient, nonzero_gradient),
            "data_loss_reconstruction_error": reconstruction_error,
            "data_loss_reconstruction_relative_error": reconstruction_relative_error,
        }

    model.policy.optimizer.zero_grad()
    total.backward()
    total_gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, base.MAX_GRAD_NORM))
    model.policy.optimizer.step()
    with torch.no_grad():
        td_errors = torch.abs(target_1 - chosen_q).detach().cpu().numpy()
    priority_updates: dict[int, float] = {}
    for index, error in zip(sample.global_indices, td_errors):
        priority_updates[int(index)] = max(priority_updates.get(int(index), 0.0), float(error))
    replay.update_priorities(priority_updates.keys(), priority_updates.values())

    row = {
        "phase": "offline_pretrain",
        "update": int(update),
        "td_1_weighted": float(losses["td_1_weighted"].detach().cpu()),
        "td_n_weighted": float(losses["td_n_weighted"].detach().cpu()),
        "margin_weighted": float(losses["margin_weighted"].detach().cpu()),
        "l2_weighted": float(losses["l2_weighted"].detach().cpu()),
        "effective_total": float(total.detach().cpu()),
        "grad_total_before_clip": total_gradient_norm,
        "gradient_would_clip": total_gradient_norm > base.MAX_GRAD_NORM,
        "all_finite": bool(
            np.isfinite(total_gradient_norm)
            and torch.isfinite(total).item()
            and all(torch.isfinite(parameter).all().item() for parameter in parameters)
        ),
    }
    return row, audit


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def plot_results(audit: pd.DataFrame, curve: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot(audit["update"], audit.sample_nonzero_fraction, marker="o", label="sampled nonzero fraction")
    axes[0, 0].plot(audit["update"], audit.nonzero_probability_mass, marker="s", label="PER nonzero mass")
    axes[0, 0].axhline(5 / 160, color="#555555", linestyle="--", label="population fraction")
    axes[0, 0].set_ylabel("Fraction")
    axes[0, 1].plot(audit["update"], audit.noop_gradient_norm, marker="o", label="no-op gradient")
    axes[0, 1].plot(audit["update"], audit.nonzero_gradient_norm, marker="s", label="nonzero gradient")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("Gradient norm")
    axes[1, 0].plot(
        audit["update"], audit.noop_nonzero_gradient_cosine,
        marker="o", color="#D55E00", label="gradient cosine",
    )
    axes[1, 0].axhline(0, color="#555555", linestyle="--")
    axes[1, 0].set_ylabel("No-op vs nonzero gradient cosine")
    axes[1, 1].plot(curve.milestone_updates, curve.noop_accuracy, marker="o", label="no-op accuracy")
    axes[1, 1].plot(curve.milestone_updates, curve.nonzero_action_recall, marker="s", label="nonzero recall")
    axes[1, 1].set_ylabel("Accuracy / recall")
    for ax in axes.flat:
        ax.set_xlabel("Demonstration update")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("SY2014 demonstration sampling and gradient balance audit")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "021_29_sampling_gradient_balance_audit.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "021_29_sampling_gradient_balance_audit.svg", bbox_inches="tight")
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
    actions = np.asarray(demonstrations["actions"], dtype=np.int64).reshape(-1)
    dones = np.asarray(demonstrations["dones"], dtype=np.float32).reshape(-1)
    env = OfflineShapeEnv(demonstrations["observations"].shape[1])
    model = DQN("MlpPolicy", env, verbose=0, **dqn_kwargs(seed=0))
    replay = PrioritizedDemonstrationReplay(
        demonstrations, agent_capacity=1, alpha=0.4,
        epsilon_demo=1.0, epsilon_agent=0.001, seed=21022,
    )

    curve_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    summary, _ = evaluate_demonstrations(
        model, demonstrations["observations"], actions, raw["observations"], 0, pd.DataFrame()
    )
    curve_rows.append(summary)
    previous = 0
    for milestone in MILESTONES[1:]:
        interval_rows: list[dict[str, Any]] = []
        for update in range(previous + 1, milestone + 1):
            row, audit = audited_update(model, replay, actions, dones, update=update)
            update_rows.append(row)
            interval_rows.append(row)
            if audit is not None:
                audit_rows.append(audit)
        summary, _ = evaluate_demonstrations(
            model,
            demonstrations["observations"],
            actions,
            raw["observations"],
            milestone,
            pd.DataFrame(interval_rows),
        )
        curve_rows.append(summary)
        previous = milestone
    env.close()

    curve = pd.DataFrame(curve_rows)
    updates = pd.DataFrame(update_rows)
    audit = pd.DataFrame(audit_rows)
    curve.to_csv(OUT / "021_29_learning_curve.csv", index=False, encoding="utf-8-sig")
    updates.to_csv(OUT / "021_29_update_diagnostics.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(OUT / "021_29_sampling_gradient_audit.csv", index=False, encoding="utf-8-sig")

    reference = pd.read_csv(ROOT / "benchmark_results" / "021_28" / "021_28_learning_curve.csv")
    reference = reference[reference.arm == "treatment_mask_demo_td1"].sort_values("milestone_updates")
    aligned = curve.sort_values("milestone_updates")
    comparison = aligned.merge(
        reference,
        on="milestone_updates",
        how="inner",
        suffixes=("_current", "_reference"),
        validate="one_to_one",
    )
    comparison_columns = [
        "noop_accuracy", "nonzero_action_recall",
        "positive_expert_margin_fraction_nonzero", "mean_expert_margin_nonzero",
    ]
    max_errors = {
        column: float(np.max(np.abs(
            comparison[f"{column}_current"].to_numpy()
            - comparison[f"{column}_reference"].to_numpy()
        )))
        for column in comparison_columns
    }
    replication = {
        "reference": "021_28 treatment_mask_demo_td1",
        "reference_milestones_all_matched": bool(
            comparison.milestone_updates.tolist() == reference.milestone_updates.tolist()
        ),
        "additional_audit_milestones": sorted(
            set(aligned.milestone_updates.tolist()) - set(reference.milestone_updates.tolist())
        ),
        "max_abs_errors": max_errors,
        "all_finite": bool(
            updates.all_finite.all()
            and np.isfinite(audit.data_loss_reconstruction_error).all()
            and np.isfinite(audit.data_loss_reconstruction_relative_error).all()
        ),
        "max_data_loss_reconstruction_error": float(audit.data_loss_reconstruction_error.max()),
        "max_data_loss_reconstruction_relative_error": float(
            audit.data_loss_reconstruction_relative_error.max()
        ),
        "passed": bool(
            comparison.milestone_updates.tolist() == reference.milestone_updates.tolist()
            and max(max_errors.values()) < 1e-6
            and audit.data_loss_reconstruction_relative_error.max() < 1e-6
        ),
    }
    (OUT / "021_29_replication_validation.json").write_text(
        json.dumps(replication, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not replication["passed"]:
        raise RuntimeError(f"021_28 replication failed: {replication}")

    median_nonzero_mass = float(audit.nonzero_probability_mass.median())
    median_sample_nonzero = float(audit.sample_nonzero_fraction.median())
    median_ratio = float(audit.noop_to_nonzero_gradient_norm_ratio.replace([np.inf], np.nan).median())
    median_cosine = float(audit.noop_nonzero_gradient_cosine.median())
    if median_nonzero_mass < 0.05 and median_ratio > 1.0:
        branch = "A_nonzero_sampling_and_noop_gradient_dominance"
    elif median_nonzero_mass >= 0.05 and median_cosine < -0.25:
        branch = "B_nonzero_sampled_but_gradients_conflict"
    else:
        branch = "C_no_single_clear_sampling_or_gradient_explanation"
    result = {
        "status": "completed_offline_no_dssat",
        "replication_validation_passed": replication["passed"],
        "nonzero_population_fraction": float((actions != 0).mean()),
        "median_nonzero_probability_mass": median_nonzero_mass,
        "median_sample_nonzero_fraction": median_sample_nonzero,
        "median_noop_to_nonzero_gradient_norm_ratio": median_ratio,
        "median_noop_nonzero_gradient_cosine": median_cosine,
        "interpretation_branch": branch,
        "no_online_training_performed": True,
    }
    (OUT / "021_29_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_results(audit, curve)

    audit_display = audit[[
        "update", "sample_nonzero_count", "sample_nonzero_fraction",
        "nonzero_probability_mass", "terminal_probability_mass",
        "noop_gradient_norm", "nonzero_gradient_norm",
        "noop_to_nonzero_gradient_norm_ratio", "noop_nonzero_gradient_cosine",
    ]].round(6)
    curve_display = curve[[
        "milestone_updates", "noop_accuracy", "nonzero_action_recall",
        "positive_expert_margin_fraction_nonzero", "mean_expert_margin_nonzero",
    ]].round(6)
    record = f"""# 021_29 SY2014 示范采样与梯度平衡离线审计

## 边界

本任务完全离线复现 021_28 Treatment，不调用 DSSAT/PDI、不在线训练、不修改任何训练参数。分组损失均按原 batch 总大小归一化，L2 未归入 no-op 或非零样本贡献。

## 复现验证

```json
{json.dumps(replication, indent=2, ensure_ascii=False)}
```

## PER 采样与分组梯度

{markdown_table(audit_display)}

## 学习结果复现

{markdown_table(curve_display)}

## 描述性结果

- 原始非零动作占比：`{result['nonzero_population_fraction']:.4f}`
- 非零动作 PER 概率质量中位数：`{median_nonzero_mass:.4f}`
- batch 非零动作占比中位数：`{median_sample_nonzero:.4f}`
- no-op / 非零梯度范数比中位数：`{median_ratio:.4f}`
- 两组梯度 cosine 中位数：`{median_cosine:.4f}`
- 预注册解释分支：`{branch}`

## 结论边界

本任务只描述 021_28 Treatment 中 PER 采样和两类示范梯度的实际结构，不现场修改采样、损失权重或网络，也不自动启动在线训练。
"""
    DOC.write_text(record, encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
