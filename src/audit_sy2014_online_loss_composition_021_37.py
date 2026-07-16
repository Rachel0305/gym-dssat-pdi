from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "021_37"
DOC = ROOT / "docs" / "2026-07-15_021_37_sy2014_online_loss_composition_offline_audit.md"
UPDATE_SOURCE = ROOT / "benchmark_results" / "021_35" / "021_35_online_update_log.csv"
Q_SOURCE = ROOT / "benchmark_results" / "021_36" / "021_36_yield_q_metric_alignment.csv"
COMPONENTS = ["td1_agent_only", "tdn_all", "margin_demo_only", "l2"]
MAJOR = ["td1_agent_only", "tdn_all", "margin_demo_only"]


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
    updates = pd.read_csv(UPDATE_SOURCE)
    qcurve = pd.read_csv(Q_SOURCE)
    expected = set(COMPONENTS + ["env_step", "total", "gradient_before_clip"])
    missing = expected - set(updates.columns)
    if missing:
        raise KeyError(f"Missing update-log columns: {sorted(missing)}")
    updates["component_sum"] = updates[COMPONENTS].sum(axis=1)
    for component in COMPONENTS:
        updates[f"share_{component}"] = updates[component] / updates.component_sum.clip(lower=1e-12)

    interval_specs = [
        ("collapse_entry_50_250", 50, 250),
        ("collapse_hold_251_500", 251, 500),
        ("recovery_501_750", 501, 750),
        ("recovery_751_1000", 751, 1000),
    ]
    rows = []
    for label, low, high in interval_specs:
        frame = updates[updates.env_step.between(low, high)]
        row = {
            "interval": label, "step_low": low, "step_high": high, "updates": len(frame),
            "median_total": float(frame.total.median()),
            "median_total_gradient": float(frame.gradient_before_clip.median()),
            "p95_total_gradient": float(frame.gradient_before_clip.quantile(0.95)),
            "gradient_clip_fraction": float(frame.gradient_would_clip.mean()),
        }
        for component in COMPONENTS:
            row[f"median_{component}"] = float(frame[component].median())
            row[f"median_share_{component}"] = float(frame[f"share_{component}"].median())
        rows.append(row)
    intervals = pd.DataFrame(rows)

    collapse = updates[updates.env_step.between(50, 500)]
    recovery = updates[updates.env_step.between(501, 1000)]
    comparisons = []
    for component in COMPONENTS:
        collapse_value = float(collapse[component].median())
        recovery_value = float(recovery[component].median())
        collapse_share = float(collapse[f"share_{component}"].median())
        recovery_share = float(recovery[f"share_{component}"].median())
        comparisons.append({
            "component": component,
            "collapse_median": collapse_value,
            "recovery_median": recovery_value,
            "collapse_to_recovery_ratio": ratio(collapse_value, recovery_value),
            "collapse_median_share": collapse_share,
            "recovery_median_share": recovery_share,
            "absolute_share_difference": abs(collapse_share - recovery_share),
        })
    comparison = pd.DataFrame(comparisons)
    gradient_ratio = ratio(
        float(collapse.gradient_before_clip.median()),
        float(recovery.gradient_before_clip.median()),
    )
    major = comparison[comparison.component.isin(MAJOR)]
    candidate = major[
        (major.absolute_share_difference >= 0.10)
        | ((major.collapse_to_recovery_ratio >= 2.0) & (major.absolute_share_difference >= 0.05))
        | ((major.collapse_to_recovery_ratio <= 0.5) & (major.absolute_share_difference >= 0.05))
    ]
    if len(candidate):
        branch = "A"
    elif bool((major.absolute_share_difference < 0.05).all()) and 0.5 <= gradient_ratio <= 2.0:
        branch = "B"
    else:
        branch = "C"

    updates.to_csv(OUT / "021_37_update_log_with_component_shares.csv", index=False, encoding="utf-8-sig")
    intervals.to_csv(OUT / "021_37_interval_summary.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT / "021_37_collapse_recovery_comparison.csv", index=False, encoding="utf-8-sig")
    validation = {
        "update_rows": len(updates),
        "expected_update_rows": 951,
        "interval_rows_sum": int(intervals.updates.sum()),
        "all_values_finite": bool(np.isfinite(updates[COMPONENTS + ["total", "gradient_before_clip"]]).all().all()),
        "component_sum_matches_total_max_abs_error": float(np.max(np.abs(updates.component_sum - updates.total))),
        "no_training_or_dssat_calls": True,
        "all_required_checks_pass": False,
    }
    validation["all_required_checks_pass"] = bool(
        validation["update_rows"] == 951
        and validation["interval_rows_sum"] == 951
        and validation["all_values_finite"]
        and validation["component_sum_matches_total_max_abs_error"] < 1e-3
    )
    (OUT / "021_37_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    summary = {
        "status": "completed" if validation["all_required_checks_pass"] else "failed_validation",
        "branch": branch,
        "gradient_collapse_to_recovery_ratio": gradient_ratio,
        "candidate_components": candidate.component.tolist(),
        "interpretation": {
            "A": "至少一个损失项在坍缩期与恢复期出现预注册的显著构成变化，可作为下一轮优先候选。",
            "B": "聚合loss构成和总梯度在坍缩/恢复期没有明显时段异常，不能据此解释Q排序翻转。",
            "C": "loss构成存在混合变化，但没有单一项满足强候选判据。",
        }[branch],
        "causal_claim_allowed": False,
        "per_component_gradient_available": False,
    }
    (OUT / "021_37_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rolling = updates.set_index("env_step")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for component, color in zip(MAJOR, ["#0072B2", "#D55E00", "#009E73"]):
        axes[0, 0].plot(rolling.index, rolling[component].rolling(40, min_periods=1).median(), color=color, label=component)
        axes[0, 1].plot(rolling.index, rolling[f"share_{component}"].rolling(40, min_periods=1).median(), color=color, label=component)
    axes[0, 0].set_yscale("symlog", linthresh=1e-3); axes[0, 0].set_ylabel("Rolling median loss")
    axes[0, 1].set_ylabel("Rolling median component share")
    axes[0, 0].legend(frameon=False, fontsize=8); axes[0, 1].legend(frameon=False, fontsize=8)
    axes[1, 0].plot(rolling.index, rolling.gradient_before_clip.rolling(40, min_periods=1).median(), color="#E69F00")
    axes[1, 0].set_ylabel("Rolling median total gradient")
    axes[1, 1].plot(qcurve.checkpoint, qcurve.nonzero_action_recall, marker="o", color="#D55E00", label="nonzero recall")
    axes[1, 1].plot(qcurve.checkpoint, qcurve.positive_expert_margin_fraction_nonzero, marker="o", color="#CC79A7", label="positive margin fraction")
    axes[1, 1].set_ylabel("Fixed-state Q metric"); axes[1, 1].legend(frameon=False)
    for ax in axes.flat:
        for x in (250, 500, 750, 1000):
            ax.axvline(x, color="#AAAAAA", linestyle=":", linewidth=0.8)
        ax.grid(alpha=0.2); ax.set_xlabel("Online environment steps")
    fig.suptitle("SY2014 online loss composition: collapse vs recovery")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "021_37_online_loss_composition_audit.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "021_37_online_loss_composition_audit.svg", bbox_inches="tight")
    plt.close(fig)

    display = comparison.round(5)
    doc = f"""# 021_37 SY2014 在线损失构成离线审计记录

## 边界

只读取021_35的951条更新日志和021_36的checkpoint Q指标；未训练、未调用DSSAT。由于未保存逐样本观测和分项梯度，本结果只能定位聚合loss构成，不能证明因果。

## 坍缩期与恢复期

{markdown_table(display)}

- 总梯度中位数collapse/recovery比：{gradient_ratio:.4f}。
- 预注册强候选项：{candidate.component.tolist()}。
- 分支：**{branch}**。{summary['interpretation']}

## 结论边界

即使某项loss数值占比很高，也不等同于其参数梯度或决策方向占主导；Huber loss、样本状态和梯度方向均会影响实际更新。本轮不调整任何权重。

## 输出

- `benchmark_results/021_37/021_37_update_log_with_component_shares.csv`
- `benchmark_results/021_37/021_37_interval_summary.csv`
- `benchmark_results/021_37/021_37_collapse_recovery_comparison.csv`
- `benchmark_results/021_37/021_37_online_loss_composition_audit.png/.svg`
"""
    DOC.write_text(doc, encoding="utf-8")
    print(json.dumps({"summary": summary, "validation": validation, "comparison": display.to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()
