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


OUT = ROOT / "benchmark_results" / "021_41"
DOC = ROOT / "docs" / "2026-07-15_021_41_sy2014_online_demo_nstep_mask_1k_ab.md"
CONTROL_SOURCE = ROOT / "benchmark_results" / "021_35" / "021_35_checkpoint_trajectory.csv"


def treatment_update(model, replay, demo_actions, rng, *, update, env_step, epsilon) -> dict[str, Any]:
    sample = ab.mixed_sample(replay, demo_actions, rng, update)
    device = model.device
    observations = base.tensor(sample.data["observations"], dtype=torch.float32, device=device)
    actions = base.tensor(sample.data["actions"], dtype=torch.long, device=device).reshape(-1)
    weights = base.tensor(sample.importance_weights, dtype=torch.float32, device=device)
    demo_mask = base.tensor(sample.is_demonstration, dtype=torch.bool, device=device)
    agent_mask = ~demo_mask
    target_1, target_n = base.compute_targets(model, sample.data)
    q = model.q_net(observations)
    chosen = q.gather(1, actions[:, None]).squeeze(1)
    td1_each = F.smooth_l1_loss(chosen, target_1, reduction="none")
    td1 = (td1_each[agent_mask] * weights[agent_mask]).mean()
    tdn_each = F.smooth_l1_loss(chosen, target_n, reduction="none")
    # Preserve each agent sample's original 1/32 contribution; delete only demo TDn terms.
    tdn_agent_only = (tdn_each[agent_mask] * weights[agent_mask]).sum() / ab.BATCH_SIZE
    margins = torch.full_like(q, 0.8); margins.scatter_(1, actions[:, None], 0.0)
    margin_each = torch.max(q + margins, dim=1).values - chosen
    margin = (margin_each[demo_mask] * weights[demo_mask]).mean()
    parameters = list(model.q_net.parameters())
    l2 = 1e-5 * sum(parameter.square().sum() for parameter in parameters)
    total = td1 + tdn_agent_only + margin + l2
    model.policy.optimizer.zero_grad(); total.backward()
    gradient = float(torch.nn.utils.clip_grad_norm_(parameters, ab.MAX_GRAD_NORM)); model.policy.optimizer.step()
    with torch.no_grad():
        errors = torch.abs(target_1 - chosen).detach().cpu().numpy()
    priorities: dict[int, float] = {}
    for index, error in zip(sample.global_indices, errors):
        priorities[int(index)] = max(priorities.get(int(index), 0.0), float(error))
    replay.update_priorities(priorities.keys(), priorities.values())
    values = {
        "td1_agent_only": float(td1.detach()),
        "tdn_agent_only_preserved_scale": float(tdn_agent_only.detach()),
        "margin_demo_only": float(margin.detach()), "l2": float(l2.detach()),
        "total": float(total.detach()),
    }
    return {
        "update": update, "env_step": env_step, "epsilon": epsilon,
        "sample_demo_count": int(demo_mask.sum()), "sample_agent_count": int(agent_mask.sum()),
        "sample_demo_noop_count": int(((actions == 0) & demo_mask).sum()),
        "sample_demo_nonzero_count": int(((actions != 0) & demo_mask).sum()),
        "replay_agent_count": replay.agent_size, **values,
        "demo_nstep_weight": 0.0,
        "gradient_before_clip": gradient, "gradient_would_clip": gradient > ab.MAX_GRAD_NORM,
        "q_abs_max": float(q.detach().abs().max()),
        "all_finite": bool(all(np.isfinite(value) for value in [*values.values(), gradient])),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    if OUT.exists(): raise FileExistsError(f"Refusing to overwrite {OUT}")
    if DOC.exists(): raise FileExistsError(f"Refusing to overwrite {DOC}")
    OUT.mkdir(parents=True)
    control = pd.read_csv(CONTROL_SOURCE).copy(); control["arm"] = "control_021_35_all_nstep"
    ab.OUT = OUT; ab.online_update = treatment_update
    interactions, updates, audit, scaler = ab.train_online()
    interactions.to_csv(OUT / "021_41_training_interactions.csv", index=False, encoding="utf-8-sig")
    updates.to_csv(OUT / "021_41_treatment_update_log.csv", index=False, encoding="utf-8-sig")
    initial = control.loc[control.checkpoint == 0].iloc[0].to_dict()
    treatment_rows = [initial]
    for checkpoint in ab.CHECKPOINTS:
        _daily, result = ab.evaluate_checkpoint(checkpoint, scaler["mean"], scaler["scale"])
        treatment_rows.append(result)
    treatment = pd.DataFrame(treatment_rows); treatment["arm"] = "treatment_mask_demo_nstep"
    trajectory = pd.concat([control, treatment], ignore_index=True, sort=False)
    trajectory.to_csv(OUT / "021_41_checkpoint_trajectory_ab.csv", index=False, encoding="utf-8-sig")
    treatment_online = treatment[treatment.checkpoint > 0]
    control_online = control[control.checkpoint > 0]
    treatment_pass = int(treatment_online.expert_efficiency_gate.astype(bool).sum())
    control_pass = int(control_online.expert_efficiency_gate.astype(bool).sum())
    final = treatment_online.loc[treatment_online.checkpoint == 1000].iloc[0]
    if treatment_pass >= 3 and bool(final.expert_efficiency_gate) and float(treatment_online.yield_kg_ha.min()) >= 9613 and treatment_pass > control_pass:
        branch = "A"
    elif treatment_pass > control_pass or bool(final.expert_efficiency_gate):
        branch = "B"
    else:
        branch = "C"
    validation = {
        **audit,
        "all_treatment_demo_nstep_weights_zero": bool((updates.demo_nstep_weight == 0).all()),
        "control_pass_count": control_pass, "treatment_pass_count": treatment_pass,
        "all_treatment_evaluations_finite": bool(treatment.q_values_finite.astype(bool).all()),
        "all_treatment_evaluations_terminated": bool(treatment.terminated_or_truncated.astype(bool).all()),
        "all_required_checks_pass": bool(
            audit["initial_hash_matches"] and audit["demonstrations_unchanged"]
            and audit["all_batches_16_demo_16_agent"] and audit["all_demo_halves_8_noop_8_nonzero"]
            and audit["all_updates_finite"] and (updates.demo_nstep_weight == 0).all()
            and treatment.q_values_finite.astype(bool).all() and treatment.terminated_or_truncated.astype(bool).all()
        ),
    }
    (OUT / "021_41_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    summary = {
        "status": "completed" if validation["all_required_checks_pass"] else "failed_validation",
        "branch": branch, "control_pass_count": control_pass, "treatment_pass_count": treatment_pass,
        "final_treatment": final.to_dict(), "online_5k_started": False,
        "interpretation": {
            "A": "屏蔽在线demo n-step显著提高优质策略保持性，可另立5K/多seed任务。",
            "B": "Treatment有所改善或终点恢复，但过程仍不稳定，不能直接放大。",
            "C": "屏蔽在线demo n-step未提高保持性，该单变量不支持继续。",
        }[branch],
    }
    (OUT / "021_41_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors = {"control_021_35_all_nstep":"#777777", "treatment_mask_demo_nstep":"#0072B2"}
    for arm, frame in trajectory.groupby("arm"):
        frame = frame.sort_values("checkpoint")
        axes[0].plot(frame.checkpoint, frame.yield_kg_ha, marker="o", color=colors[arm], label=arm)
        axes[1].plot(frame.checkpoint, frame.irrigation_mm, marker="o", color=colors[arm], label=arm)
        axes[2].plot(frame.checkpoint, frame.nitrogen_kg_ha, marker="o", color=colors[arm], label=arm)
    axes[0].axhline(11077, color="#D55E00", linestyle="--"); axes[0].set_ylabel("Yield (kg/ha)")
    axes[1].set_ylabel("Irrigation (mm)"); axes[2].set_ylabel("Nitrogen (kg/ha)")
    for ax in axes:
        ax.grid(alpha=0.2); ax.set_xlabel("Online environment steps"); ax.legend(frameon=False, fontsize=7)
    fig.suptitle("SY2014 online demo n-step mask 1K A/B")
    fig.tight_layout(rect=(0,0,1,0.92))
    fig.savefig(OUT / "021_41_demo_nstep_mask_1k_ab.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "021_41_demo_nstep_mask_1k_ab.svg", bbox_inches="tight")
    plt.close(fig)

    display = trajectory[["arm","checkpoint","yield_kg_ha","irrigation_mm","nitrogen_kg_ha","late_n_after_dap90_kg_ha","expert_efficiency_gate"]].round(3)
    doc = f"""# 021_41 SY2014 在线demo n-step屏蔽1K A/B记录

## 单变量

Control复用021_35；Treatment仅删除在线混合batch中demo样本的n-step梯度，agent n-step的原始1/32缩放、agent TD1、demo margin及其他设置均不变。离线500次学习不变。

## 结果

{markdown_table(display)}

Control通过数={control_pass}/4，Treatment通过数={treatment_pass}/4。预注册分支：**{branch}**。{summary['interpretation']}

## 边界

仅seed0 1K；没有启动5K或多seed，也没有根据结果调其他权重。
"""
    DOC.write_text(doc, encoding="utf-8")
    print(json.dumps({"summary": summary, "validation": validation, "trajectory": display.to_dict("records")}, indent=2, allow_nan=True))


if __name__ == "__main__": main()
