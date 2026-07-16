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
import run_sy2014_event_balanced_online_retention_1k_021_35 as ab


OUT = ROOT / "benchmark_results" / "021_39"
DOC = ROOT / "docs" / "2026-07-15_021_39_sy2014_exact_replay_component_gradient_audit.md"
REFERENCE = ROOT / "benchmark_results" / "021_35" / "021_35_checkpoint_trajectory.csv"
COMPONENT_GRADS = ["grad_td1_agent", "grad_tdn_all", "grad_margin_demo"]
COSINES = ["cos_td1_tdn", "cos_td1_margin", "cos_tdn_margin"]


def flat_gradient(loss: torch.Tensor, parameters: list[torch.Tensor]) -> torch.Tensor:
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([
        (torch.zeros_like(parameter) if grad is None else grad).reshape(-1)
        for parameter, grad in zip(parameters, grads)
    ]).detach()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denominator) == 0:
        return float("nan")
    return float(torch.dot(a, b) / denominator)


def diagnostic_update(model, replay, demo_actions, rng, *, update, env_step, epsilon) -> dict[str, Any]:
    sample = ab.mixed_sample(replay, demo_actions, rng, update)
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

    grad_td1 = flat_gradient(td1, parameters)
    grad_tdn = flat_gradient(tdn, parameters)
    grad_margin = flat_gradient(margin, parameters)
    diagnostics = {
        "grad_td1_agent": float(torch.linalg.vector_norm(grad_td1)),
        "grad_tdn_all": float(torch.linalg.vector_norm(grad_tdn)),
        "grad_margin_demo": float(torch.linalg.vector_norm(grad_margin)),
        "cos_td1_tdn": cosine(grad_td1, grad_tdn),
        "cos_td1_margin": cosine(grad_td1, grad_margin),
        "cos_tdn_margin": cosine(grad_tdn, grad_margin),
    }

    model.policy.optimizer.zero_grad()
    total.backward()
    gradient = float(torch.nn.utils.clip_grad_norm_(parameters, ab.MAX_GRAD_NORM))
    model.policy.optimizer.step()
    with torch.no_grad():
        errors = torch.abs(target_1 - chosen_q).detach().cpu().numpy()
    priority_updates: dict[int, float] = {}
    for index, error in zip(sample.global_indices, errors):
        priority_updates[int(index)] = max(priority_updates.get(int(index), 0.0), float(error))
    replay.update_priorities(priority_updates.keys(), priority_updates.values())
    values = {
        "td1_agent_only": float(td1.detach()), "tdn_all": float(tdn.detach()),
        "margin_demo_only": float(margin.detach()), "l2": float(l2.detach()),
        "total": float(total.detach()),
    }
    return {
        "update": update, "env_step": env_step, "epsilon": epsilon,
        "sample_demo_count": int(demo_mask.sum()), "sample_agent_count": int(agent_mask.sum()),
        "sample_demo_noop_count": int(((actions == 0) & demo_mask).sum()),
        "sample_demo_nonzero_count": int(((actions != 0) & demo_mask).sum()),
        "replay_agent_count": replay.agent_size, **values, **diagnostics,
        "gradient_before_clip": gradient, "gradient_would_clip": gradient > ab.MAX_GRAD_NORM,
        "q_abs_max": float(q_values.detach().abs().max()),
        "all_finite": bool(
            all(np.isfinite(value) for value in [*values.values(), *diagnostics.values(), gradient])
            and torch.isfinite(q_values).all()
            and all(torch.isfinite(parameter).all() for parameter in parameters)
        ),
    }


def ratio(a: float, b: float) -> float:
    return float(a / max(abs(b), 1e-12))


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
    ab.OUT = OUT
    ab.online_update = diagnostic_update
    interactions, updates, audit, scaler = ab.train_online()
    interactions.to_csv(OUT / "021_39_training_interactions.csv", index=False, encoding="utf-8-sig")
    updates.to_csv(OUT / "021_39_component_gradient_update_log.csv", index=False, encoding="utf-8-sig")

    reference = pd.read_csv(REFERENCE)
    eval_rows = [reference.loc[reference.checkpoint == 0].iloc[0].to_dict()]
    for checkpoint in ab.CHECKPOINTS:
        _daily, result = ab.evaluate_checkpoint(checkpoint, scaler["mean"], scaler["scale"])
        eval_rows.append(result)
    trajectory = pd.DataFrame(eval_rows).sort_values("checkpoint")
    trajectory.to_csv(OUT / "021_39_checkpoint_trajectory.csv", index=False, encoding="utf-8-sig")
    compare_columns = ["yield_kg_ha", "biomass_kg_ha", "irrigation_mm", "nitrogen_kg_ha", "late_n_after_dap90_kg_ha"]
    ref = reference.sort_values("checkpoint").reset_index(drop=True)
    cur = trajectory.sort_values("checkpoint").reset_index(drop=True)
    reproduction_errors = {
        column: float(np.max(np.abs(pd.to_numeric(ref[column]) - pd.to_numeric(cur[column]))))
        for column in compare_columns
    }

    collapse = updates[updates.env_step.between(50, 500)]
    recovery = updates[updates.env_step.between(501, 1000)]
    comparison_rows = []
    for metric in COMPONENT_GRADS + COSINES:
        c = float(collapse[metric].median()); r = float(recovery[metric].median())
        comparison_rows.append({
            "metric": metric, "collapse_median": c, "recovery_median": r,
            "collapse_to_recovery_ratio": ratio(c, r),
            "absolute_stage_difference": abs(c - r),
        })
    comparison = pd.DataFrame(comparison_rows)
    grad_candidate = comparison[
        comparison.metric.isin(COMPONENT_GRADS)
        & ((comparison.collapse_to_recovery_ratio >= 2) | (comparison.collapse_to_recovery_ratio <= 0.5))
    ]
    cosine_candidate = comparison[
        comparison.metric.isin(COSINES) & (comparison.absolute_stage_difference >= 0.2)
    ]
    if len(grad_candidate) or len(cosine_candidate):
        branch = "A"
    elif bool(comparison.loc[comparison.metric.isin(COMPONENT_GRADS), "collapse_to_recovery_ratio"].between(0.5, 2).all()) and bool(
        (comparison.loc[comparison.metric.isin(COSINES), "absolute_stage_difference"] < 0.1).all()
    ):
        branch = "B"
    else:
        branch = "C"
    comparison.to_csv(OUT / "021_39_collapse_recovery_gradient_comparison.csv", index=False, encoding="utf-8-sig")
    validation = {
        "checkpoint_reproduction_max_abs_errors": reproduction_errors,
        "checkpoint_trajectory_exactly_reproduced": max(reproduction_errors.values()) == 0,
        "update_rows": len(updates), "expected_update_rows": 951,
        "all_updates_finite": bool(updates.all_finite.all()),
        "all_required_checks_pass": bool(
            max(reproduction_errors.values()) == 0 and len(updates) == 951 and updates.all_finite.all()
        ),
    }
    (OUT / "021_39_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    summary = {
        "status": "completed" if validation["all_required_checks_pass"] else "failed_validation",
        "branch": branch,
        "gradient_candidate_metrics": grad_candidate.metric.tolist(),
        "cosine_candidate_metrics": cosine_candidate.metric.tolist(),
        "interpretation": {
            "A": "至少一个分项梯度范数或方向在坍缩/恢复期出现预注册的时段差异。",
            "B": "分项梯度在坍缩/恢复期没有足够时间区分力。",
            "C": "分项梯度存在混合变化，但未达到单一强候选条件。",
        }[branch],
        "causal_claim_allowed": False,
    }
    (OUT / "021_39_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for metric in COMPONENT_GRADS:
        axes[0, 0].plot(updates.env_step, updates[metric].rolling(40, min_periods=1).median(), label=metric)
    axes[0, 0].set_yscale("log"); axes[0, 0].set_ylabel("Component gradient norm"); axes[0, 0].legend(frameon=False, fontsize=7)
    for metric in COSINES:
        axes[0, 1].plot(updates.env_step, updates[metric].rolling(40, min_periods=1).median(), label=metric)
    axes[0, 1].axhline(0, color="#333333", linestyle="--"); axes[0, 1].set_ylabel("Gradient cosine"); axes[0, 1].legend(frameon=False, fontsize=7)
    axes[1, 0].plot(trajectory.checkpoint, trajectory.yield_kg_ha, marker="o", color="#0072B2")
    axes[1, 0].set_ylabel("Yield (kg/ha)")
    axes[1, 1].plot(trajectory.checkpoint, trajectory.irrigation_mm, marker="o", label="I")
    axes[1, 1].plot(trajectory.checkpoint, trajectory.nitrogen_kg_ha, marker="o", label="N")
    axes[1, 1].set_ylabel("Seasonal input"); axes[1, 1].legend(frameon=False)
    for ax in axes.flat:
        for x in (250, 500, 750, 1000):
            ax.axvline(x, color="#AAAAAA", linestyle=":", linewidth=0.8)
        ax.grid(alpha=0.2); ax.set_xlabel("Online environment steps")
    fig.suptitle("SY2014 exact replay: per-component gradient audit")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "021_39_component_gradient_audit.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "021_39_component_gradient_audit.svg", bbox_inches="tight")
    plt.close(fig)

    display = comparison.round(5)
    doc = f"""# 021_39 SY2014 精确重放与分项梯度审计记录

## 强制复现

021_35 checkpoint轨迹最大绝对误差：{max(reproduction_errors.values()):.6g}；逐项复现通过：{validation['checkpoint_trajectory_exactly_reproduced']}。只增加autograd诊断，没有改变optimizer更新。

## 坍缩与恢复期分项梯度

{markdown_table(display)}

预注册分支：**{branch}**。{summary['interpretation']}

## 边界

这是同一训练轨迹上的时段对照，仍然是观察性机制证据，不证明某项梯度是根因。本任务不改任何loss权重，也未超过1K。

## 输出

- `benchmark_results/021_39/021_39_component_gradient_update_log.csv`
- `benchmark_results/021_39/021_39_checkpoint_trajectory.csv`
- `benchmark_results/021_39/021_39_collapse_recovery_gradient_comparison.csv`
- `benchmark_results/021_39/021_39_component_gradient_audit.png/.svg`
"""
    DOC.write_text(doc, encoding="utf-8")
    print(json.dumps({"summary": summary, "validation": validation, "comparison": display.to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()
