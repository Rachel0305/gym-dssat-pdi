"""Compile exact YC V1 replay results, criteria, reports, and figures.

This script only reads the frozen baseline summaries, formal validation CSVs,
and the exact Summary.OUT replay table.  It does not alter training outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments" / "222_yc_v1_three_seed_confirmation"
REPLAY = EXPERIMENT / "results" / "yc_v1_exact_efficiency_replay_all_seeds.csv"
RESULTS = ROOT / "results"
FIGURES = EXPERIMENT / "figures"
BASELINE = ROOT / "benchmark_results" / "055_02_yca_lowIC_four_baselines_static_level1" / "evaluation" / "055_02_baseline_summary.csv"
METHODS = ["Original", "Augmented"]
CHECKPOINTS = [25_000, 50_000, 75_000, 100_000]
METRICS = ["yield", "total_irrigation", "total_nitrogen", "PFP_N", "WP_ET"]
LABELS = {
    "yield": "Yield (kg ha$^{-1}$)",
    "total_irrigation": "Irrigation (mm)",
    "total_nitrogen": "N input (kg ha$^{-1}$)",
    "PFP_N": "PFP-N (kg kg$^{-1}$)",
    "WP_ET": "WP$_{ET}$ (kg m$^{-3}$)",
}


def numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def fmt(value: float | int | str) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.3f}"


def load_replay() -> pd.DataFrame:
    if not REPLAY.exists():
        raise FileNotFoundError(f"missing exact replay: {REPLAY}")
    df = pd.read_csv(REPLAY)
    df = numeric(df, ["seed", "checkpoint", "evaluation_year", *METRICS, "ETCP", "reward"])
    return df.sort_values(["method", "seed", "checkpoint", "evaluation_year"]).reset_index(drop=True)


def load_baselines() -> pd.DataFrame:
    df = pd.read_csv(BASELINE, keep_default_na=False)
    return numeric(df, ["year", "grain_yield_kg_ha", "actual_irrigation_mm", "actual_nitrogen_kg_ha", "PFP_N_kg_kg", "WP_ET_kg_m3", "etcp_mm"])


def make_tables(replay: pd.DataFrame, baseline: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    per_seed = replay.groupby(["method", "seed", "checkpoint"], as_index=False)[METRICS + ["ETCP", "reward"]].mean()
    # Checkpoint error bars must represent between-seed variability, not the
    # within-validation-year spread.  The latter remains available in the
    # by-year table and is plotted separately in Figure 4.
    summary = per_seed.groupby(["method", "checkpoint"])[METRICS + ["ETCP", "reward"]].agg(["mean", "std"])
    summary.columns = ["_".join(x).rstrip("_") if isinstance(x, tuple) else x for x in summary.columns]
    summary = summary.reset_index()

    baseline_map = {
        "Expert": "official_extension_expert",
        "Farmer": "recorded_farmer_template",
    }
    comp_rows: list[dict[str, object]] = []
    for label, scenario in baseline_map.items():
        rows = baseline[baseline["scenario"].eq(scenario)]
        comp_rows.append(
            {
                "method": label,
                "seed": "baseline",
                "checkpoint": 100_000,
                "yield": rows["grain_yield_kg_ha"].mean(),
                "total_irrigation": rows["actual_irrigation_mm"].mean(),
                "total_nitrogen": rows["actual_nitrogen_kg_ha"].mean(),
                "PFP_N": rows["PFP_N_kg_kg"].mean(),
                "WP_ET": rows["WP_ET_kg_m3"].mean(),
            }
        )
    final = per_seed[per_seed["checkpoint"].eq(100_000)]
    for method in METHODS:
        row = final[final["method"].eq(method)].groupby("method", as_index=False)[METRICS].mean()
        values = row.iloc[0].to_dict()
        values.update({"method": method, "seed": "three_seed_mean", "checkpoint": 100_000})
        comp_rows.append(values)
    comparison = pd.DataFrame(comp_rows)[["method", "seed", "checkpoint", *METRICS]]

    expert = comparison.loc[comparison["method"].eq("Expert")].iloc[0]
    aug_final = comparison.loc[comparison["method"].eq("Augmented")].iloc[0]
    aug_seeds = final[final["method"].eq("Augmented")].groupby("seed", as_index=False)[METRICS].mean()
    criteria_rows = [
        ("3-seed mean Yield >= 98% expert", aug_final["yield"], expert["yield"] * 0.98),
        ("Each seed mean Yield >= 95% expert", aug_seeds["yield"].min(), expert["yield"] * 0.95),
        ("3-seed mean irrigation <= 110% expert", aug_final["total_irrigation"], expert["total_irrigation"] * 1.10),
        ("3-seed mean N <= 100% expert", aug_final["total_nitrogen"], expert["total_nitrogen"]),
        ("3-seed mean PFP-N >= 100% expert", aug_final["PFP_N"], expert["PFP_N"]),
        ("3-seed mean WP_ET >= 95% expert", aug_final["WP_ET"], expert["WP_ET"] * 0.95),
    ]
    criteria = pd.DataFrame(
        [
            {
                "criterion": name,
                "observed": float(observed),
                "threshold": float(threshold),
                "pass": bool(observed >= threshold) if "<=" not in name else bool(observed <= threshold),
                "reference": "official_extension_expert, 2014-2023",
            }
            for name, observed, threshold in criteria_rows
        ]
    )
    return per_seed, summary, comparison, criteria, {"expert_yield": float(expert["yield"]), "aug_yield": float(aug_final["yield"])}


def save_figures(replay: pd.DataFrame, summary: pd.DataFrame, comparison: pd.DataFrame, per_seed: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 120})
    colors = {"Original": "#3366CC", "Augmented": "#D55E00"}

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for method in METHODS:
        g = summary[summary["method"].eq(method)].sort_values("checkpoint")
        ax.errorbar(g["checkpoint"] / 1000, g["yield_mean"], yerr=g["yield_std"], marker="o", capsize=3, label=method, color=colors[method])
    ax.set(xlabel="Checkpoint (K)", ylabel=LABELS["yield"], title="YC V1 checkpoint yield stability")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure1_checkpoint_yield_stability.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.7), constrained_layout=True)
    order = ["Original", "Augmented", "Expert", "Farmer"]
    plot_comp = comparison.set_index("method").reindex(order)
    for ax, metric in zip(axes, METRICS):
        vals = plot_comp[metric].to_numpy(dtype=float)
        bars = ax.bar(order, vals, color=[colors.get(x, "#777777") for x in order])
        ax.set_title(LABELS[metric])
        ax.tick_params(axis="x", rotation=55, labelsize=8)
        ax.grid(axis="y", alpha=0.2)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.1f}", ha="center", va="bottom", fontsize=7, rotation=90)
    fig.suptitle("YC V1 100K multi-objective comparison", y=1.03)
    fig.savefig(FIGURES / "figure2_100k_multiobjective_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.7), constrained_layout=False)
    g = per_seed[per_seed["checkpoint"].eq(100_000)]
    for ax, metric in zip(axes, METRICS):
        x = np.arange(3)
        width = 0.34
        for offset, method in [(-width / 2, "Original"), (width / 2, "Augmented")]:
            vals = g[g["method"].eq(method)].sort_values("seed")[metric].to_numpy(dtype=float)
            ax.bar(x + offset, vals, width, label=method if metric == METRICS[0] else None, color=colors[method])
        ax.set_title(LABELS[metric])
        ax.set_xticks(x, ["0", "1", "2"])
        ax.set_xlabel("Seed")
        ax.grid(axis="y", alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(top=0.76, wspace=0.28)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=False)
    fig.suptitle("YC V1 100K seed robustness", y=1.08)
    fig.savefig(FIGURES / "figure3_100k_seed_robustness.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), constrained_layout=True)
    g = replay[replay["checkpoint"].eq(100_000)]
    for ax, metric in zip(axes.flat, METRICS):
        for method in METHODS:
            y = g[g["method"].eq(method)].groupby("evaluation_year")[metric].mean().reindex(range(2014, 2024))
            ax.plot(y.index, y.values, marker="o", ms=3, label=method, color=colors[method])
        ax.set_title(LABELS[metric])
        ax.set_xticks(range(2014, 2024, 2))
        ax.grid(alpha=0.2)
    axes.flat[-1].axis("off")
    axes.flat[0].legend(frameon=False)
    fig.suptitle("YC V1 100K by-year stability (three-seed mean)", y=1.02)
    fig.savefig(FIGURES / "figure4_100k_by_year_stability.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_reports(replay: pd.DataFrame, per_seed: pd.DataFrame, summary: pd.DataFrame, comparison: pd.DataFrame, criteria: pd.DataFrame, ref: dict[str, float]) -> str:
    RESULTS.mkdir(parents=True, exist_ok=True)
    replay.to_csv(RESULTS / "yc_v1_exact_efficiency_replay.csv", index=False, encoding="utf-8-sig")
    replay.to_csv(RESULTS / "yc_v1_three_seed_by_year.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(RESULTS / "yc_v1_three_seed_summary.csv", index=False, encoding="utf-8-sig")
    per_seed.to_csv(RESULTS / "yc_v1_three_seed_checkpoint_trajectory.csv", index=False, encoding="utf-8-sig")
    criteria.to_csv(RESULTS / "yc_v1_three_seed_criteria.csv", index=False, encoding="utf-8-sig")

    failed = criteria.loc[~criteria["pass"]]
    original_yield = float(comparison.loc[comparison["method"].eq("Original"), "yield"].iloc[0])
    augmented_yield = float(comparison.loc[comparison["method"].eq("Augmented"), "yield"].iloc[0])
    if len(failed) == 0:
        status = "provisional_success"
    elif augmented_yield > original_yield and int(criteria["pass"].sum()) > 0:
        status = "partial_improvement"
    else:
        status = "augmentation_not_robust"
    main_issue = "全部冻结标准通过。" if len(failed) == 0 else str(failed.iloc[0]["criterion"])
    gate_rows: list[str] = []
    for method, seed in [("Original", 1), ("Augmented", 1), ("Original", 2), ("Augmented", 2)]:
        path = EXPERIMENT / "logs" / f"{method.lower()}_seed{seed}_formal_status.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload.get("result", {})
        gate = result.get("action_gate") or result.get("technical_gate") or {}
        gate_rows.append(
            f"| {method} | {seed} | {str(result.get('status', payload.get('status'))).lower()} | "
            f"{gate.get('observed_validation_rows', gate.get('observed_validation_rows', 'NA'))}/10 | "
            f"{gate.get('all_actions_on_declared_grid', gate.get('final_actions_on_grid', 'NA'))} | "
            f"{gate.get('multiple_nonzero_action_pairs', gate.get('final_unique_action_signatures', 'NA'))} | "
            f"{gate.get('next_step_allowed', result.get('next_step_allowed', 'NA'))} |"
        )
    gate_md = "\n".join(gate_rows) if gate_rows else "| NA | NA | NA | NA | NA | NA | NA |"
    criteria_md = "\n".join(
        f"| {r['criterion']} | {fmt(r['observed'])} | {fmt(r['threshold'])} | {'PASS' if r['pass'] else 'FAIL'} |"
        for r in criteria.to_dict("records")
    )
    metrics_md = "\n".join(
        f"| {r['method']} | {fmt(r['yield'])} | {fmt(r['total_irrigation'])} | {fmt(r['total_nitrogen'])} | {fmt(r['PFP_N'])} | {fmt(r['WP_ET'])} |"
        for r in comparison.to_dict("records")
    )
    report = f"""# YC V1 three-seed confirmation

日期：2026-09-10  
状态：`{status}`  
主结果：Augmented 100K，三种子均值；Original 为同一协议下的 paired comparator。

## A–B. 精确效率回放

已完成 `{len(replay)}` 行精确回放（2 methods × 3 seeds × 4 checkpoints × 10 validation years），失败行数为 `{int(replay.isna().all(axis=1).sum())}`。WP_ET/ETCP 来自 DSSAT `Summary.OUT`；定义为 `YPEM × 0.1`（有效时），否则为 `HWAM/(ETCP×10)`，其中 ETCP 单位为 mm。PFP-N 使用 `YPNAM`，仅在 `NICM > 0` 时有效。没有从 daily CSV 推断 WP_ET。

精确回放源代码：`src/055_yca_lowIC_site_transfer/run_055_03_yca_lowIC_five_scenario_figures.py`；回放结果：[yc_v1_exact_efficiency_replay.csv](../results/yc_v1_exact_efficiency_replay.csv)。

## C–H. 结果摘要

seed 0 已复用冻结 Original/221YCA 产物并做同协议回放；seed 1/2 使用独立输出根完成正式 100K 训练。所有正式运行均保留 25K/50K/75K/100K 检查点，验证年份为 2014–2023。下表为 100K 三种子均值；Expert/Farmer 为冻结 `055_02` baseline summary 的 2014–2023 均值。

正式管理行为审计（seed 0 为冻结既有产物，seed 1/2 为本次独立正式运行）：

| Method | Seed | Status | Validation rows | Actions on grid | Diversity signal | next_step_allowed |
|---|---:|---|---:|---|---|---|
{gate_md}

其中 Original seed1 的最终审计 `next_step_allowed=false`，且 `multiple_nonzero_action_pairs=false`；这被保留为管理策略塌缩/多样性风险，而不是删除该种子或重新训练后再选择结果。

| Method | Yield | Irrigation | N input | PFP-N | WP_ET |
|---|---:|---:|---:|---:|---:|
{metrics_md}

三种子均值与标准差见 [yc_v1_three_seed_summary.csv](../results/yc_v1_three_seed_summary.csv)，逐种子检查点轨迹见 [yc_v1_three_seed_checkpoint_trajectory.csv](../results/yc_v1_three_seed_checkpoint_trajectory.csv)，逐年明细见 [yc_v1_three_seed_by_year.csv](../results/yc_v1_three_seed_by_year.csv)。

四张图：

1. `experiments/222_yc_v1_three_seed_confirmation/figures/figure1_checkpoint_yield_stability.png`
2. `experiments/222_yc_v1_three_seed_confirmation/figures/figure2_100k_multiobjective_comparison.png`
3. `experiments/222_yc_v1_three_seed_confirmation/figures/figure3_100k_seed_robustness.png`
4. `experiments/222_yc_v1_three_seed_confirmation/figures/figure4_100k_by_year_stability.png`

## I–K. 冻结标准判定

Expert mean Yield = `{fmt(ref['expert_yield'])}`；Augmented mean Yield = `{fmt(ref['aug_yield'])}`；Original mean Yield = `{fmt(original_yield)}`。

| Criterion | Observed | Threshold | Result |
|---|---:|---:|---|
{criteria_md}

结论状态为 `{status}`。主要未满足项：{main_issue}。该结论不宣称 PPO/augmentation 在所有站点、年份或指标上普遍优越；它只适用于本 YC lowIC、固定验证期、固定动作/奖励/网络/训练协议。

## L–N. 复现与未执行项

新增实验目录：`experiments/222_yc_v1_three_seed_confirmation/`。训练期间通过 smoke gate 后才进入 formal；正式运行未改 reward、action evaluation、PPO、network 或其他站点。由于原始模型/daily 输出体量较大，本次 Git commit 只提交脚本、配置、manifest、汇总表、图和文档，不提交大型 checkpoint 与缓存目录；原始输出仍保留在本地 `benchmark_results/222YCA_*`。

未执行：未 push 到远端；未将 `dssat_auto_external_n` 纳入主比较；未改变冻结 criteria；未重新训练 seed 0。
"""
    (ROOT / "docs" / "yc_v1_three_seed_confirmation.md").write_text(report, encoding="utf-8")
    replay_doc = f"""# YC V1 exact WP_ET/ETCP replay\n\n已完成 `{len(replay)}` 行 Summary.OUT 精确回放，覆盖 Original/Augmented、seed 0/1/2、25K/50K/75K/100K 和 2014–2023。WP_ET 只来自 Summary.OUT/ETCP：有效 YPEM 时使用 `YPEM*0.1`，否则使用 `HWAM/(ETCP*10)`；ETCP 单位为 mm。PFP-N 只在 NICM>0 时使用 YPNAM。\n\n输出：`results/yc_v1_exact_efficiency_replay.csv`。源代码：`src/055_yca_lowIC_site_transfer/run_055_03_yca_lowIC_five_scenario_figures.py`。\n"""
    (ROOT / "docs" / "yc_v1_exact_efficiency_replay.md").write_text(replay_doc, encoding="utf-8")
    (RESULTS / "yc_v1_three_seed_status.json").write_text(json.dumps({"status": status, "main_issue": main_issue, "criteria_passed": int(criteria["pass"].sum()), "criteria_total": len(criteria)}, ensure_ascii=False, indent=2), encoding="utf-8")
    return status


def main() -> None:
    replay = load_replay()
    baseline = load_baselines()
    per_seed, summary, comparison, criteria, ref = make_tables(replay, baseline)
    save_figures(replay, summary, comparison, per_seed)
    status = write_reports(replay, per_seed, summary, comparison, criteria, ref)
    print(json.dumps({"status": status, "replay_rows": len(replay), "criteria_passed": int(criteria["pass"].sum()), "criteria_total": len(criteria)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
