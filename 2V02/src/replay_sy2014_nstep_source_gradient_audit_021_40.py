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
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_sy2014_dqfd_real_network_loss_diagnostic_021_22 as base
import run_sy2014_event_balanced_online_retention_1k_021_35 as ab


OUT = ROOT / "benchmark_results" / "021_40"
DOC = ROOT / "docs" / "2026-07-15_021_40_sy2014_agent_demo_nstep_gradient_split_audit.md"
REFERENCE_CHECKPOINTS = ROOT / "benchmark_results" / "021_39" / "training" / "checkpoints"


def flat_gradient(loss: torch.Tensor, parameters: list[torch.Tensor]) -> torch.Tensor:
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([
        (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
        for parameter, gradient in zip(parameters, gradients)
    ]).detach()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    return float(torch.dot(a, b) / denominator) if float(denominator) else float("nan")


def model_hash(path: Path) -> str:
    model = ab.DQN.load(str(path), device="cpu")
    digest = hashlib.sha256()
    for tensor in model.q_net.state_dict().values():
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def split_update(model, replay, demo_actions, rng, *, update, env_step, epsilon) -> dict[str, Any]:
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
    tdn = (tdn_each * weights).mean()
    tdn_demo = (tdn_each[demo_mask] * weights[demo_mask]).mean()
    tdn_agent = (tdn_each[agent_mask] * weights[agent_mask]).mean()
    margins = torch.full_like(q, 0.8); margins.scatter_(1, actions[:, None], 0.0)
    margin_each = torch.max(q + margins, dim=1).values - chosen
    margin = (margin_each[demo_mask] * weights[demo_mask]).mean()
    parameters = list(model.q_net.parameters())
    l2 = 1e-5 * sum(parameter.square().sum() for parameter in parameters)
    total = td1 + tdn + margin + l2

    gradients = {
        "tdn_demo": flat_gradient(tdn_demo, parameters),
        "tdn_agent": flat_gradient(tdn_agent, parameters),
        "margin": flat_gradient(margin, parameters),
    }
    diagnostics = {
        "grad_tdn_demo": float(torch.linalg.vector_norm(gradients["tdn_demo"])),
        "grad_tdn_agent": float(torch.linalg.vector_norm(gradients["tdn_agent"])),
        "grad_margin_demo": float(torch.linalg.vector_norm(gradients["margin"])),
        "cos_tdn_demo_margin": cosine(gradients["tdn_demo"], gradients["margin"]),
        "cos_tdn_agent_margin": cosine(gradients["tdn_agent"], gradients["margin"]),
        "cos_tdn_demo_agent": cosine(gradients["tdn_demo"], gradients["tdn_agent"]),
    }

    model.policy.optimizer.zero_grad(); total.backward()
    gradient = float(torch.nn.utils.clip_grad_norm_(parameters, ab.MAX_GRAD_NORM)); model.policy.optimizer.step()
    with torch.no_grad():
        errors = torch.abs(target_1 - chosen).detach().cpu().numpy()
    priorities: dict[int, float] = {}
    for index, error in zip(sample.global_indices, errors):
        priorities[int(index)] = max(priorities.get(int(index), 0.0), float(error))
    replay.update_priorities(priorities.keys(), priorities.values())
    values = {
        "td1_agent_only": float(td1.detach()), "tdn_all": float(tdn.detach()),
        "margin_demo_only": float(margin.detach()), "l2": float(l2.detach()), "total": float(total.detach()),
    }
    return {
        "update": update, "env_step": env_step, "epsilon": epsilon,
        "sample_demo_count": int(demo_mask.sum()), "sample_agent_count": int(agent_mask.sum()),
        "sample_demo_noop_count": int(((actions == 0) & demo_mask).sum()),
        "sample_demo_nonzero_count": int(((actions != 0) & demo_mask).sum()),
        "replay_agent_count": replay.agent_size, **values, **diagnostics,
        "gradient_before_clip": gradient, "gradient_would_clip": gradient > ab.MAX_GRAD_NORM,
        "q_abs_max": float(q.detach().abs().max()),
        "all_finite": bool(all(np.isfinite(x) for x in [*values.values(), *diagnostics.values(), gradient])),
    }


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
    ab.OUT = OUT; ab.online_update = split_update
    interactions, updates, audit, _scaler = ab.train_online()
    interactions.to_csv(OUT / "021_40_training_interactions.csv", index=False, encoding="utf-8-sig")
    updates.to_csv(OUT / "021_40_nstep_source_gradient_log.csv", index=False, encoding="utf-8-sig")

    hashes = []
    for checkpoint in [0, 250, 500, 750, 1000]:
        current = model_hash(OUT / "training" / "checkpoints" / f"checkpoint_{checkpoint}.zip")
        reference = model_hash(REFERENCE_CHECKPOINTS / f"checkpoint_{checkpoint}.zip")
        hashes.append({"checkpoint": checkpoint, "current_hash": current, "reference_021_39_hash": reference, "equal": current == reference})
    hash_frame = pd.DataFrame(hashes)
    hash_frame.to_csv(OUT / "021_40_checkpoint_hash_validation.csv", index=False, encoding="utf-8-sig")

    collapse = updates[updates.env_step.between(50, 500)]
    recovery = updates[updates.env_step.between(501, 1000)]
    rows = []
    for metric in ["grad_tdn_demo", "grad_tdn_agent", "grad_margin_demo", "cos_tdn_demo_margin", "cos_tdn_agent_margin", "cos_tdn_demo_agent"]:
        c = float(collapse[metric].median()); r = float(recovery[metric].median())
        rows.append({"metric": metric, "collapse_median": c, "recovery_median": r, "absolute_stage_difference": abs(c-r)})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "021_40_collapse_recovery_source_comparison.csv", index=False, encoding="utf-8-sig")
    demo_diff = float(comparison.loc[comparison.metric.eq("cos_tdn_demo_margin"), "absolute_stage_difference"].iloc[0])
    agent_diff = float(comparison.loc[comparison.metric.eq("cos_tdn_agent_margin"), "absolute_stage_difference"].iloc[0])
    if agent_diff >= 0.2 and agent_diff > demo_diff + 0.05:
        branch = "agent_nstep_priority_candidate"
    elif demo_diff >= 0.2 and demo_diff > agent_diff + 0.05:
        branch = "demo_nstep_priority_candidate"
    elif demo_diff >= 0.2 and agent_diff >= 0.2 and abs(demo_diff-agent_diff) < 0.05:
        branch = "joint_candidate"
    else:
        branch = "no_clear_source"
    validation = {
        "checkpoint_hashes_all_match_021_39": bool(hash_frame.equal.all()),
        "update_rows": len(updates), "all_updates_finite": bool(updates.all_finite.all()),
        "no_dssat_evaluation_rerun": True,
        "all_required_checks_pass": bool(hash_frame.equal.all() and len(updates) == 951 and updates.all_finite.all()),
    }
    (OUT / "021_40_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    summary = {
        "status": "completed" if validation["all_required_checks_pass"] else "failed_validation",
        "branch": branch, "demo_cosine_stage_difference": demo_diff,
        "agent_cosine_stage_difference": agent_diff,
        "causal_claim_allowed": False,
    }
    (OUT / "021_40_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for metric in ["grad_tdn_demo", "grad_tdn_agent", "grad_margin_demo"]:
        axes[0].plot(updates.env_step, updates[metric].rolling(40, min_periods=1).median(), label=metric)
    axes[0].set_yscale("log"); axes[0].set_ylabel("Gradient norm"); axes[0].legend(frameon=False, fontsize=7)
    for metric in ["cos_tdn_demo_margin", "cos_tdn_agent_margin", "cos_tdn_demo_agent"]:
        axes[1].plot(updates.env_step, updates[metric].rolling(40, min_periods=1).median(), label=metric)
    axes[1].axhline(0, color="#333333", linestyle="--"); axes[1].set_ylabel("Gradient cosine"); axes[1].legend(frameon=False, fontsize=7)
    for ax in axes:
        for x in [250, 500, 750, 1000]: ax.axvline(x, color="#AAAAAA", linestyle=":", linewidth=0.8)
        ax.grid(alpha=0.2); ax.set_xlabel("Online environment steps")
    fig.suptitle("SY2014 agent/demo n-step gradient source audit")
    fig.tight_layout(rect=(0,0,1,0.92))
    fig.savefig(OUT / "021_40_nstep_source_gradient_audit.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "021_40_nstep_source_gradient_audit.svg", bbox_inches="tight")
    plt.close(fig)

    doc = f"""# 021_40 SY2014 agent/demo n-step梯度来源拆分记录

## 复现验证

五个checkpoint online-Q哈希全部与021_39相同：{validation['checkpoint_hashes_all_match_021_39']}。因此新增autograd日志没有改变训练轨迹。本轮没有重复DSSAT评估。

## 坍缩与恢复期

{markdown_table(comparison.round(5))}

预注册来源判定：**{branch}**。demo cosine阶段差={demo_diff:.4f}，agent cosine阶段差={agent_diff:.4f}。

## 边界

本结果只定位候选来源，不证明因果，也未现场修改任何loss。
"""
    DOC.write_text(doc, encoding="utf-8")
    print(json.dumps({"summary": summary, "validation": validation, "comparison": comparison.round(5).to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()
