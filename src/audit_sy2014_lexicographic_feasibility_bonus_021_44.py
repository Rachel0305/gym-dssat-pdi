from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "021_44"
DOC = ROOT / "docs" / "2026-07-15_021_44_sy2014_lexicographic_feasibility_bonus_offline_audit.md"
SOURCE = ROOT / "benchmark_results" / "021_43" / "021_43_checkpoint_reward_alignment.csv"
WATER_COST = 1.0
NITROGEN_COST = 5.0
IRRIGATION_BUDGET = 120.0
NITROGEN_BUDGET = 300.0
BONUS = WATER_COST * IRRIGATION_BUDGET + NITROGEN_COST * NITROGEN_BUDGET


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows(): lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    if OUT.exists(): raise FileExistsError(f"Refusing to overwrite {OUT}")
    if DOC.exists(): raise FileExistsError(f"Refusing to overwrite {DOC}")
    OUT.mkdir(parents=True)
    frame = pd.read_csv(SOURCE)
    frame["feasibility_bonus"] = frame.expert_gate_recomputed.astype(bool).astype(float) * BONUS
    frame["candidate_lexicographic_reward"] = frame.recalculated_reward + frame.feasibility_bonus
    frame.to_csv(OUT / "021_44_checkpoint_counterfactual_scores.csv", index=False, encoding="utf-8-sig")
    rows = []
    for seed, group in frame.groupby("online_seed"):
        old = group.sort_values(["recalculated_reward", "checkpoint"], ascending=[False, True]).iloc[0]
        new = group.sort_values(["candidate_lexicographic_reward", "checkpoint"], ascending=[False, True]).iloc[0]
        rows.append({
            "online_seed": int(seed),
            "old_best_checkpoint": int(old.checkpoint), "old_best_passes_gate": bool(old.expert_gate_recomputed),
            "old_best_reward": float(old.recalculated_reward),
            "candidate_best_checkpoint": int(new.checkpoint), "candidate_best_passes_gate": bool(new.expert_gate_recomputed),
            "candidate_best_reward": float(new.candidate_lexicographic_reward),
            "candidate_best_yield": float(new.yield_kg_ha),
            "candidate_best_irrigation": float(new.irrigation_mm),
            "candidate_best_nitrogen": float(new.nitrogen_kg_ha),
        })
    summary_frame = pd.DataFrame(rows)
    summary_frame.to_csv(OUT / "021_44_seed_ranking_summary.csv", index=False, encoding="utf-8-sig")
    pass_count = int(summary_frame.candidate_best_passes_gate.sum())
    branch = "A" if pass_count == 3 else ("B" if pass_count == 2 else "C")
    validation = {
        "bonus_derivation": f"{WATER_COST}*{IRRIGATION_BUDGET}+{NITROGEN_COST}*{NITROGEN_BUDGET}",
        "bonus_value": BONUS,
        "checkpoint_rows": len(frame), "seed_count": int(frame.online_seed.nunique()),
        "no_training_or_dssat_calls": True,
        "existing_reward_implementation_modified": False,
        "all_required_checks_pass": bool(len(frame) == 12 and frame.online_seed.nunique() == 3 and BONUS == 1620),
    }
    (OUT / "021_44_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    summary = {
        "status": "completed" if validation["all_required_checks_pass"] else "failed_validation",
        "branch": branch, "candidate_best_gate_pass_count": pass_count,
        "candidate_formula": "base_reward + 1620 * I[yield >= local official expert yield]",
        "interpretation": {
            "A": "候选可行性优先公式在三个在线seed上均把最高分排序转向expert-gate可行策略。",
            "B": "候选公式仅在两个seed上对齐严格目标。",
            "C": "候选公式未能稳定对齐严格目标。",
        }[branch],
        "training_authorized": False,
    }
    (OUT / "021_44_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    for seed, group in frame.groupby("online_seed"):
        axes[0].plot(group.checkpoint, group.recalculated_reward, marker="o", label=f"seed{int(seed)}")
        axes[1].plot(group.checkpoint, group.candidate_lexicographic_reward, marker="o", label=f"seed{int(seed)}")
    axes[0].set_title("Current reward"); axes[1].set_title("Candidate feasibility-priority reward")
    for ax in axes: ax.grid(alpha=0.2); ax.set_xlabel("Online steps"); ax.set_ylabel("Score"); ax.legend(frameon=False)
    fig.suptitle("SY2014 counterfactual reward ranking")
    fig.tight_layout(rect=(0,0,1,0.92))
    fig.savefig(OUT / "021_44_lexicographic_reward_ranking.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "021_44_lexicographic_reward_ranking.svg", bbox_inches="tight")
    plt.close(fig)

    doc = f"""# 021_44 SY2014 产量可行性优先reward离线审计记录

## 候选公式

`R_candidate = R_current + 1620 * I[Y >= local official expert yield]`，其中1620严格由冻结预算与成本`1*120+5*300`推导，不按本次结果拟合。

## 结果

{markdown_table(summary_frame.round(3))}

预注册分支：**{branch}**。{summary['interpretation']}

## 边界

这只是反事实排序检查，现有reward代码没有修改，也未授权训练。下一步若执行，必须单独实现terminal bonus、先做单元/smoke验证，再进行1K单变量A/B。
"""
    DOC.write_text(doc, encoding="utf-8")
    print(json.dumps({"summary": summary, "validation": validation, "seeds": summary_frame.round(4).to_dict("records")}, indent=2))


if __name__ == "__main__": main()
