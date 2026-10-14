from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_sy2014_fixed_observation_standardization_diagnostic_021_24 as normbase
from run_sy2014_standardized_demo_pretraining_curve_021_26 import evaluate_demonstrations
from stable_baselines3 import DQN


OUT = ROOT / "benchmark_results" / "021_36"
DOC = ROOT / "docs" / "2026-07-15_021_36_sy2014_online_checkpoint_q_ranking_offline_audit.md"
CHECKPOINT_DIR = ROOT / "benchmark_results" / "021_35" / "training" / "checkpoints"
TRAJECTORY = ROOT / "benchmark_results" / "021_35" / "021_35_checkpoint_trajectory.csv"
CHECKPOINTS = [0, 250, 500, 750, 1000]


def spearman(x: pd.Series, y: pd.Series) -> float:
    xr = x.rank(method="average").to_numpy(dtype=float)
    yr = y.rank(method="average").to_numpy(dtype=float)
    if np.std(xr) == 0 or np.std(yr) == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


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
    actions = np.asarray(standardized["actions"], dtype=np.int64).reshape(-1)
    summaries: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINTS:
        path = CHECKPOINT_DIR / f"checkpoint_{checkpoint}.zip"
        if not path.exists():
            raise FileNotFoundError(path)
        model = DQN.load(str(path), device="cpu")
        summary, event_rows = evaluate_demonstrations(
            model, standardized["observations"], actions,
            raw["observations"], checkpoint, pd.DataFrame(),
        )
        summary["checkpoint"] = checkpoint
        for row in event_rows:
            row["checkpoint"] = checkpoint
        summaries.append(summary); events.extend(event_rows)
    qcurve = pd.DataFrame(summaries).sort_values("checkpoint")
    event_frame = pd.DataFrame(events).sort_values(["checkpoint", "dap"])
    trajectory = pd.read_csv(TRAJECTORY)
    combined = trajectory.merge(qcurve, on="checkpoint", validate="one_to_one")
    metrics = [
        "nonzero_action_recall", "positive_expert_margin_fraction_nonzero",
        "mean_expert_margin_nonzero", "noop_accuracy", "q_abs_max",
    ]
    correlations = pd.DataFrame([
        {"metric": metric, "spearman_with_yield": spearman(combined["yield_kg_ha"], combined[metric])}
        for metric in metrics
    ])
    recall_rho = float(correlations.loc[correlations.metric.eq("nonzero_action_recall"), "spearman_with_yield"].iloc[0])
    margin_fraction_rho = float(correlations.loc[
        correlations.metric.eq("positive_expert_margin_fraction_nonzero"), "spearman_with_yield"
    ].iloc[0])
    if recall_rho >= 0.8 and margin_fraction_rho >= 0.8:
        branch = "A"
    elif abs(recall_rho) < 0.3 and abs(margin_fraction_rho) < 0.3:
        branch = "B"
    else:
        branch = "C"

    qcurve.to_csv(OUT / "021_36_fixed_demo_q_curve.csv", index=False, encoding="utf-8-sig")
    event_frame.to_csv(OUT / "021_36_nonzero_event_predictions.csv", index=False, encoding="utf-8-sig")
    combined.to_csv(OUT / "021_36_yield_q_metric_alignment.csv", index=False, encoding="utf-8-sig")
    correlations.to_csv(OUT / "021_36_spearman_descriptive.csv", index=False, encoding="utf-8-sig")
    checks = {
        "checkpoint_count": len(qcurve),
        "expected_checkpoint_count": len(CHECKPOINTS),
        "event_rows": len(event_frame),
        "expected_event_rows": len(CHECKPOINTS) * int((actions != 0).sum()),
        "all_q_and_parameters_finite": bool(qcurve.q_values_finite.all() and qcurve.parameters_finite.all()),
        "no_training_or_dssat_calls": True,
        "all_required_checks_pass": bool(
            len(qcurve) == len(CHECKPOINTS)
            and len(event_frame) == len(CHECKPOINTS) * int((actions != 0).sum())
            and qcurve.q_values_finite.all() and qcurve.parameters_finite.all()
        ),
    }
    (OUT / "021_36_validation.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    summary = {
        "status": "completed" if checks["all_required_checks_pass"] else "failed_validation",
        "branch": branch,
        "spearman_yield_vs_nonzero_recall": recall_rho,
        "spearman_yield_vs_positive_margin_fraction": margin_fraction_rho,
        "interpretation": {
            "A": "固定示范状态的非零动作Q排序与闭环产量坍缩/恢复高度同步。",
            "B": "固定示范状态Q指标不能解释闭环产量变化，需查真实轨迹分布或闭环放大。",
            "C": "固定示范状态Q排序与产量存在部分对应，但不是单一解释。",
        }[branch],
        "causal_claim_allowed": False,
    }
    (OUT / "021_36_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    axes[0, 0].plot(combined.checkpoint, combined.yield_kg_ha, marker="o", color="#0072B2")
    axes[0, 0].set_ylabel("Yield (kg/ha)")
    axes[0, 1].plot(combined.checkpoint, combined.nonzero_action_recall, marker="o", color="#D55E00", label="Nonzero recall")
    axes[0, 1].plot(combined.checkpoint, combined.noop_accuracy, marker="o", color="#009E73", label="No-op accuracy")
    axes[0, 1].set_ylabel("Fraction"); axes[0, 1].legend(frameon=False)
    axes[1, 0].plot(combined.checkpoint, combined.positive_expert_margin_fraction_nonzero, marker="o", color="#CC79A7")
    axes[1, 0].set_ylabel("Positive expert-margin fraction")
    for (dap, action), frame in event_frame.groupby(["dap", "target_action"]):
        axes[1, 1].plot(frame.checkpoint, frame.expert_margin, marker="o", label=f"DAP{dap}/a{action}")
    axes[1, 1].axhline(0, color="#333333", linestyle="--", linewidth=1)
    axes[1, 1].set_ylabel("Expert Q margin"); axes[1, 1].legend(frameon=False, fontsize=7)
    for ax in axes.flat:
        ax.grid(alpha=0.2); ax.set_xlabel("Online environment steps")
    fig.suptitle("SY2014 checkpoint Q-ranking audit on fixed oracle states")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "021_36_checkpoint_q_ranking_audit.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "021_36_checkpoint_q_ranking_audit.svg", bbox_inches="tight")
    plt.close(fig)

    display = combined[[
        "checkpoint", "yield_kg_ha", "nonzero_action_recall", "noop_accuracy",
        "positive_expert_margin_fraction_nonzero", "mean_expert_margin_nonzero",
    ]].round(4)
    doc = f"""# 021_36 SY2014 在线checkpoint Q排序离线审计记录

## 边界

只读取021_35五个checkpoint，在同一组160个标准化oracle示范状态上计算Q排序。未训练、未调用DSSAT、未修改模型。

## 结果

{markdown_table(display)}

描述性Spearman：yield vs nonzero recall = {recall_rho:.4f}；yield vs positive-margin fraction = {margin_fraction_rho:.4f}。

预注册分支：**{branch}**。{summary['interpretation']}

## 限制

只有5个checkpoint，且固定示范状态不是完整闭环状态分布；相关性仅用于定位下一步，不构成因果证明。

## 输出

- `benchmark_results/021_36/021_36_fixed_demo_q_curve.csv`
- `benchmark_results/021_36/021_36_nonzero_event_predictions.csv`
- `benchmark_results/021_36/021_36_yield_q_metric_alignment.csv`
- `benchmark_results/021_36/021_36_spearman_descriptive.csv`
- `benchmark_results/021_36/021_36_checkpoint_q_ranking_audit.png/.svg`
"""
    DOC.write_text(doc, encoding="utf-8")
    print(json.dumps({"summary": summary, "validation": checks, "alignment": display.to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()
