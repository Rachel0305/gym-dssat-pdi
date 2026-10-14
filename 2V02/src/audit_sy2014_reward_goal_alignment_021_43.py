from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "021_43"
DOC = ROOT / "docs" / "2026-07-15_021_43_sy2014_reward_goal_alignment_offline_audit.md"
NULL_YIELD = 5408.0
WATER_COST = 1.0
NITROGEN_COST = 5.0


def reward(yield_value: float, irrigation: float, nitrogen: float, n_cost: float = NITROGEN_COST) -> float:
    return max(0.0, yield_value - NULL_YIELD) - WATER_COST * irrigation - n_cost * nitrogen


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
    seed0 = pd.read_csv(ROOT / "benchmark_results/021_41/021_41_checkpoint_trajectory_ab.csv")
    seed0 = seed0[(seed0.arm == "treatment_mask_demo_nstep") & (seed0.checkpoint > 0)].copy()
    seed0["online_seed"] = 0
    seed12 = pd.read_csv(ROOT / "benchmark_results/021_42/021_42_checkpoint_trajectories.csv")
    all_rows = pd.concat([seed0, seed12[seed12.checkpoint > 0]], ignore_index=True, sort=False)
    all_rows["recalculated_reward"] = [
        reward(y, i, n) for y, i, n in zip(all_rows.yield_kg_ha, all_rows.irrigation_mm, all_rows.nitrogen_kg_ha)
    ]
    all_rows["expert_gate_recomputed"] = (
        (all_rows.yield_kg_ha >= 11077) & (all_rows.irrigation_mm <= 120)
        & (all_rows.nitrogen_kg_ha <= 300) & (all_rows.late_n_after_dap90_kg_ha == 0)
    )
    all_rows.to_csv(OUT / "021_43_checkpoint_reward_alignment.csv", index=False, encoding="utf-8-sig")

    summaries = []
    for seed, frame in all_rows.groupby("online_seed"):
        frame = frame.sort_values("checkpoint")
        best_reward = frame.sort_values(["recalculated_reward", "checkpoint"], ascending=[False, True]).iloc[0]
        passing = frame[frame.expert_gate_recomputed]
        best_passing = passing.sort_values(["recalculated_reward", "checkpoint"], ascending=[False, True]).iloc[0] if len(passing) else None
        mismatch = bool(not best_reward.expert_gate_recomputed and best_passing is not None and best_reward.recalculated_reward > best_passing.recalculated_reward)
        break_even = np.nan
        if mismatch and best_passing.nitrogen_kg_ha != best_reward.nitrogen_kg_ha:
            # Solve Yp-Ip-c*Np = Yf-If-c*Nf.
            break_even = (
                (best_passing.yield_kg_ha - best_reward.yield_kg_ha)
                - (best_passing.irrigation_mm - best_reward.irrigation_mm)
            ) / (best_passing.nitrogen_kg_ha - best_reward.nitrogen_kg_ha)
        summaries.append({
            "online_seed": int(seed),
            "reward_best_checkpoint": int(best_reward.checkpoint),
            "reward_best_yield": float(best_reward.yield_kg_ha),
            "reward_best_irrigation": float(best_reward.irrigation_mm),
            "reward_best_nitrogen": float(best_reward.nitrogen_kg_ha),
            "reward_best_score": float(best_reward.recalculated_reward),
            "reward_best_passes_expert_gate": bool(best_reward.expert_gate_recomputed),
            "best_passing_checkpoint": int(best_passing.checkpoint) if best_passing is not None else np.nan,
            "best_passing_score": float(best_passing.recalculated_reward) if best_passing is not None else np.nan,
            "reward_advantage_of_failing_choice": float(best_reward.recalculated_reward - best_passing.recalculated_reward) if mismatch else 0.0,
            "nitrogen_cost_break_even": float(break_even),
            "objective_mismatch": mismatch,
        })
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(OUT / "021_43_seed_objective_alignment_summary.csv", index=False, encoding="utf-8-sig")
    affected = set(summary_frame.loc[summary_frame.objective_mismatch, "online_seed"].astype(int))
    if {1, 2}.issubset(affected): branch = "A"
    elif 1 in affected or 2 in affected: branch = "B"
    else: branch = "C"
    validation = {
        "seed_count": int(summary_frame.online_seed.nunique()),
        "checkpoint_rows": len(all_rows),
        "formula_constants": {"null_yield": NULL_YIELD, "water_cost": WATER_COST, "nitrogen_cost": NITROGEN_COST},
        "expert_gate_matches_saved": bool(
            (all_rows.expert_gate_recomputed.astype(bool) == all_rows.expert_efficiency_gate.astype(bool)).all()
        ),
        "no_training_or_dssat_calls": True,
        "all_required_checks_pass": bool(
            summary_frame.online_seed.nunique() == 3 and len(all_rows) == 12
            and (all_rows.expert_gate_recomputed.astype(bool) == all_rows.expert_efficiency_gate.astype(bool)).all()
        ),
    }
    (OUT / "021_43_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    summary = {
        "status": "completed" if validation["all_required_checks_pass"] else "failed_validation",
        "branch": branch, "affected_online_seeds": sorted(affected),
        "interpretation": {
            "A": "seed1/2均出现当前reward偏好低于expert产量的低氮策略，存在系统性目标错位。",
            "B": "一个额外seed出现reward与严格expert gate错位。",
            "C": "未观察到reward最高checkpoint与严格expert gate错位。",
        }[branch],
        "reward_modified": False,
    }
    (OUT / "021_43_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    for seed, frame in all_rows.groupby("online_seed"):
        axes[0].plot(frame.checkpoint, frame.recalculated_reward, marker="o", label=f"seed{int(seed)}")
        axes[1].plot(frame.checkpoint, frame.yield_kg_ha, marker="o", label=f"seed{int(seed)}")
    axes[1].axhline(11077, color="#333333", linestyle="--", label="expert yield gate")
    axes[0].set_ylabel("Frozen reward"); axes[1].set_ylabel("Yield (kg/ha)")
    for ax in axes: ax.grid(alpha=0.2); ax.set_xlabel("Online steps"); ax.legend(frameon=False)
    fig.suptitle("SY2014 reward versus strict expert-gate alignment")
    fig.tight_layout(rect=(0,0,1,0.92))
    fig.savefig(OUT / "021_43_reward_goal_alignment.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "021_43_reward_goal_alignment.svg", bbox_inches="tight")
    plt.close(fig)

    doc = f"""# 021_43 SY2014 reward与导师目标一致性审计记录

## 当前公式

`R = max(0, Y - 5408) - 1*I - 5*N`。本轮只重算已有checkpoint，不训练、不改reward。

## 结果

{markdown_table(summary_frame.round(4))}

预注册分支：**{branch}**。{summary['interpretation']}

## 含义与边界

如果低氮策略虽然未达到expert产量，却获得更高reward，那么DQN选择它不一定是训练失败，而可能是正确优化了当前标量目标。盈亏平衡系数只用于解释当前权衡，不授权现场把氮成本改成该数值；是否采用“产量硬门槛+门槛内资源效率”需要单独预注册。
"""
    DOC.write_text(doc, encoding="utf-8")
    print(json.dumps({"summary": summary, "validation": validation, "seeds": summary_frame.round(5).to_dict("records")}, indent=2))


if __name__ == "__main__": main()
