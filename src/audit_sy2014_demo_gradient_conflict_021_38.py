from __future__ import annotations

import json
import sys
from pathlib import Path

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
from run_sy2014_demo_return_to_go_ab_021_27 import full_return_to_go
from stable_baselines3 import DQN


OUT = ROOT / "benchmark_results" / "021_38"
DOC = ROOT / "docs" / "2026-07-15_021_38_sy2014_demo_gradient_conflict_checkpoint_audit.md"
CHECKPOINT_DIR = ROOT / "benchmark_results" / "021_35" / "training" / "checkpoints"
CHECKPOINTS = [0, 250, 500, 750, 1000]


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
    raw, standardized, validation = normbase.prepare_demonstrations()
    if not validation["all_pre_run_checks_pass"]:
        raise RuntimeError("Frozen standardization validation failed")
    demo = full_return_to_go(standardized)
    actions_np = np.asarray(demo["actions"], dtype=np.int64).reshape(-1)
    noop_np = actions_np == 0
    weights_np = np.where(noop_np, 0.5 / noop_np.sum(), 0.5 / (~noop_np).sum()).astype(np.float32)
    rows = []
    target_hashes = []
    for checkpoint in CHECKPOINTS:
        model = DQN.load(str(CHECKPOINT_DIR / f"checkpoint_{checkpoint}.zip"), device="cpu")
        device = model.device
        observations = torch.as_tensor(demo["observations"], dtype=torch.float32, device=device)
        actions = torch.as_tensor(actions_np, dtype=torch.long, device=device)
        weights = torch.as_tensor(weights_np, dtype=torch.float32, device=device)
        noop = torch.as_tensor(noop_np, dtype=torch.bool, device=device)
        target_n = base.compute_targets(model, demo)[1]
        q = model.q_net(observations)
        chosen = q.gather(1, actions[:, None]).squeeze(1)
        n_each = F.smooth_l1_loss(chosen, target_n, reduction="none")
        n_loss = torch.sum(weights * n_each)
        n_noop = n_each[noop].mean()
        n_nonzero = n_each[~noop].mean()
        margins = torch.full_like(q, 0.8)
        margins.scatter_(1, actions[:, None], 0.0)
        margin_each = torch.max(q + margins, dim=1).values - chosen
        margin_loss = torch.sum(weights * margin_each)
        parameters = list(model.q_net.parameters())
        grad_n = flat_gradient(n_loss, parameters)
        grad_margin = flat_gradient(margin_loss, parameters)
        grad_noop = flat_gradient(n_noop, parameters)
        grad_nonzero = flat_gradient(n_nonzero, parameters)
        norm_n = float(torch.linalg.vector_norm(grad_n))
        norm_margin = float(torch.linalg.vector_norm(grad_margin))
        ratio = norm_n / max(norm_margin, 1e-12)
        cos_nm = cosine(grad_n, grad_margin)
        conflict = bool(ratio >= 10 and cos_nm <= -0.3)
        target_digest = __import__("hashlib").sha256()
        for tensor in model.q_net_target.state_dict().values():
            target_digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        target_hashes.append(target_digest.hexdigest())
        rows.append({
            "checkpoint": checkpoint,
            "demo_nstep_loss_group_balanced": float(n_loss.detach()),
            "demo_margin_loss_group_balanced": float(margin_loss.detach()),
            "grad_norm_demo_nstep": norm_n,
            "grad_norm_demo_margin": norm_margin,
            "nstep_to_margin_grad_norm_ratio": ratio,
            "cosine_demo_nstep_vs_margin": cos_nm,
            "cosine_nstep_noop_vs_nonzero": cosine(grad_noop, grad_nonzero),
            "conflict_checkpoint": conflict,
            "all_finite": bool(
                torch.isfinite(q).all() and torch.isfinite(target_n).all()
                and torch.isfinite(grad_n).all() and torch.isfinite(grad_margin).all()
            ),
        })
    frame = pd.DataFrame(rows)
    collapse_conflict = bool(frame.loc[frame.checkpoint.isin([250, 500]), "conflict_checkpoint"].all())
    endpoint_nonconflict = bool((~frame.loc[frame.checkpoint.isin([0, 1000]), "conflict_checkpoint"]).any())
    if collapse_conflict and endpoint_nonconflict:
        branch = "A"
    elif bool(frame.conflict_checkpoint.all()) or bool((~frame.conflict_checkpoint).all()):
        branch = "B"
    else:
        branch = "C"
    frame.to_csv(OUT / "021_38_demo_gradient_conflict_by_checkpoint.csv", index=False, encoding="utf-8-sig")
    validation_result = {
        "checkpoint_count": len(frame),
        "all_finite": bool(frame.all_finite.all()),
        "target_hash_unique_count": len(set(target_hashes)),
        "target_network_constant_across_1k": len(set(target_hashes)) == 1,
        "no_optimizer_step_or_dssat_call": True,
        "all_required_checks_pass": bool(len(frame) == 5 and frame.all_finite.all()),
    }
    (OUT / "021_38_validation.json").write_text(json.dumps(validation_result, indent=2), encoding="utf-8")
    summary = {
        "status": "completed" if validation_result["all_required_checks_pass"] else "failed_validation",
        "branch": branch,
        "conflict_checkpoints": frame.loc[frame.conflict_checkpoint, "checkpoint"].astype(int).tolist(),
        "interpretation": {
            "A": "坍缩checkpoint的demo n-step与margin梯度冲突增强，构成优先候选。",
            "B": "demo n-step/margin梯度冲突指标在时间上没有区分力，不能解释坍缩与恢复。",
            "C": "冲突指标呈混合变化，可能参与但不是单一时间解释。",
        }[branch],
        "agent_gradient_not_available": True,
        "causal_claim_allowed": False,
    }
    (OUT / "021_38_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(frame.checkpoint, frame.grad_norm_demo_nstep, marker="o", label="demo n-step")
    axes[0].plot(frame.checkpoint, frame.grad_norm_demo_margin, marker="o", label="demo margin")
    axes[0].set_yscale("log"); axes[0].set_ylabel("Gradient norm"); axes[0].legend(frameon=False)
    axes[1].plot(frame.checkpoint, frame.nstep_to_margin_grad_norm_ratio, marker="o", color="#D55E00")
    axes[1].axhline(10, color="#333333", linestyle="--"); axes[1].set_ylabel("n-step / margin gradient norm")
    axes[2].plot(frame.checkpoint, frame.cosine_demo_nstep_vs_margin, marker="o", color="#009E73")
    axes[2].axhline(-0.3, color="#333333", linestyle="--"); axes[2].axhline(0, color="#999999", linestyle=":")
    axes[2].set_ylabel("Gradient cosine")
    for ax in axes:
        ax.grid(alpha=0.2); ax.set_xlabel("Online environment steps")
    fig.suptitle("SY2014 demonstration n-step vs margin gradient audit")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "021_38_demo_gradient_conflict.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "021_38_demo_gradient_conflict.svg", bbox_inches="tight")
    plt.close(fig)

    display = frame.drop(columns=["all_finite"]).round(5)
    doc = f"""# 021_38 SY2014 demonstration梯度冲突离线审计记录

## 边界

只在五个已有checkpoint和同一组160条oracle demonstration上计算梯度；无optimizer.step、无训练、无DSSAT。采用no-op/nonzero各0.5的group-balanced期望权重。agent梯度未保存，不能在本轮恢复。

## 结果

{markdown_table(display)}

预注册分支：**{branch}**。{summary['interpretation']}

target network在五个checkpoint中的唯一哈希数：{validation_result['target_hash_unique_count']}，符合1K内target冻结设置。

## 限制

这是固定demonstration上的局部梯度诊断，不等同于真实混合batch的总更新方向，也不构成因果证明。本轮不调整margin、n-step或采样权重。

## 输出

- `benchmark_results/021_38/021_38_demo_gradient_conflict_by_checkpoint.csv`
- `benchmark_results/021_38/021_38_validation.json`
- `benchmark_results/021_38/021_38_summary.json`
- `benchmark_results/021_38/021_38_demo_gradient_conflict.png/.svg`
"""
    DOC.write_text(doc, encoding="utf-8")
    print(json.dumps({"summary": summary, "validation": validation_result, "rows": display.to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()
