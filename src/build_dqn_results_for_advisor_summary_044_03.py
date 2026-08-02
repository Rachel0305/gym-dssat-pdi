from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "044_03_dqn_results_for_advisor_summary"
TABLES = OUT / "tables"
FIGS = OUT / "figures"
DOC = ROOT / "docs" / "044_03_dqn_results_for_advisor_summary_record.md"


SOURCES = {
    "普通DQN_040_01": {
        "kind": "validation",
        "summary": ROOT
        / "benchmark_results/040_01_sya_lowIC_free_timing_dqn/evaluation/040_01_validation_summary_by_checkpoint.csv",
        "detail": ROOT
        / "benchmark_results/040_01_sya_lowIC_free_timing_dqn/evaluation/040_01_checkpoint_validation_summary.csv",
        "note": "SB3 DQN + 外部安全修正；不是严格 MaskableDQN。",
    },
    "严格MaskableDQN_040_02": {
        "kind": "validation",
        "summary": ROOT
        / "benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/evaluation/040_02_validation_summary_by_checkpoint.csv",
        "detail": ROOT
        / "benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/evaluation/040_02_checkpoint_validation_summary.csv",
        "note": "自定义严格 MaskableDQN；训练和评估阶段屏蔽非法动作。",
    },
    "DemoDQN_DQfD_044_00": {
        "kind": "validation",
        "summary": ROOT
        / "benchmark_results/044_00_sya_lowIC_binary_forecast_demo_dqn_smoke_smoke2k/evaluation/044_00_validation_summary_by_checkpoint.csv",
        "detail": ROOT
        / "benchmark_results/044_00_sya_lowIC_binary_forecast_demo_dqn_smoke_smoke2k/evaluation/044_00_checkpoint_validation_summary.csv",
        "note": "Demo replay + margin loss 的 2K smoke；检查示范经验是否能避免 no-op。",
    },
}

Q_AUDITS = {
    "DemoDQN在线2K_044_01": ROOT
    / "benchmark_results/044_01_sya_lowIC_demo_dqn_q_ranking_audit/tables/044_01_q_ranking_by_checkpoint.csv",
    "DemoOnly预训练_044_02": ROOT
    / "benchmark_results/044_02_sya_lowIC_demo_only_pretrain_dqn_audit/tables/044_02_q_ranking_by_checkpoint.csv",
}


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    for font_path in [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=str(font_path)).get_name()]
            break
    plt.rcParams["axes.unicode_minus"] = False


def read_validation_summaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    checkpoint_rows = []
    best_year_rows = []

    for method, spec in SOURCES.items():
        summary = pd.read_csv(spec["summary"])
        detail = pd.read_csv(spec["detail"])

        if "mean_yield" in summary.columns:
            summary = summary.rename(
                columns={
                    "mean_yield": "mean_final_grnwt",
                    "mean_irrigation": "mean_total_irrigation",
                    "mean_nitrogen": "mean_total_n",
                    "max_swfac": "max_swfac_over_years",
                    "max_nstres": "max_nstres_over_years",
                }
            )
            summary["mean_PFP_N"] = pd.NA
            summary["mean_swfac_stress_days_gt_0p05"] = pd.NA
            summary["mean_nstres_days_gt_0p05"] = pd.NA

        summary["method"] = method
        summary["source_note"] = spec["note"]
        checkpoint_rows.append(summary)

        best_ckpt = (
            summary.sort_values("mean_final_grnwt", ascending=False)
            .iloc[0]["checkpoint_step"]
        )
        detail_best = detail.loc[detail["checkpoint_step"] == best_ckpt].copy()
        detail_best["method"] = method
        detail_best["selected_checkpoint_by"] = "mean_final_grnwt_max"
        detail_best["source_note"] = spec["note"]
        if "grain_yield_kg_ha" in detail_best.columns:
            detail_best = detail_best.rename(
                columns={
                    "grain_yield_kg_ha": "final_grnwt",
                    "total_nitrogen": "total_n",
                }
            )
            detail_best["PFP_N"] = detail_best.apply(
                lambda r: r["final_grnwt"] / r["total_n"]
                if r.get("total_n", 0) not in (0, 0.0)
                else pd.NA,
                axis=1,
            )
            detail_best["swfac_stress_days_gt_0p05"] = pd.NA
            detail_best["nstres_days_gt_0p05"] = pd.NA
        best_year_rows.append(detail_best)

    checkpoints = pd.concat(checkpoint_rows, ignore_index=True)
    yearly_best = pd.concat(best_year_rows, ignore_index=True, sort=False)
    return checkpoints, yearly_best


def read_q_audits() -> pd.DataFrame:
    rows = []
    for method, path in Q_AUDITS.items():
        df = pd.read_csv(path)
        df["method"] = method
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def save_tables(checkpoints: pd.DataFrame, yearly_best: pd.DataFrame, q: pd.DataFrame) -> dict[str, str]:
    checkpoint_cols = [
        "method",
        "checkpoint_step",
        "validation_years",
        "mean_final_grnwt",
        "mean_total_irrigation",
        "mean_total_n",
        "mean_PFP_N",
        "mean_swfac_stress_days_gt_0p05",
        "mean_nstres_days_gt_0p05",
        "source_note",
    ]
    checkpoint_cols = [c for c in checkpoint_cols if c in checkpoints.columns]
    checkpoints_out = TABLES / "044_03_dqn_checkpoint_mean_metrics.csv"
    checkpoints[checkpoint_cols].to_csv(checkpoints_out, index=False, encoding="utf-8-sig")

    yearly_cols = [
        "method",
        "year",
        "checkpoint_step",
        "final_grnwt",
        "total_irrigation",
        "total_n",
        "PFP_N",
        "swfac_stress_days_gt_0p05",
        "nstres_days_gt_0p05",
        "action_sequence",
        "daily_csv_path",
        "source_note",
    ]
    yearly_cols = [c for c in yearly_cols if c in yearly_best.columns]
    yearly_out = TABLES / "044_03_dqn_best_checkpoint_yearly_validation.csv"
    yearly_best[yearly_cols].to_csv(yearly_out, index=False, encoding="utf-8-sig")

    q_out = TABLES / "044_03_demo_dqn_q_ranking_summary.csv"
    q.to_csv(q_out, index=False, encoding="utf-8-sig")

    return {
        "checkpoint_metrics": str(checkpoints_out.relative_to(ROOT)),
        "yearly_best": str(yearly_out.relative_to(ROOT)),
        "q_ranking": str(q_out.relative_to(ROOT)),
    }


def plot_checkpoint_metrics(checkpoints: pd.DataFrame) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    metrics = [
        ("mean_final_grnwt", "平均籽粒产量 (kg/ha)"),
        ("mean_total_irrigation", "平均总灌溉 (mm)"),
        ("mean_total_n", "平均总施氮 (kg/ha)"),
        ("mean_swfac_stress_days_gt_0p05", "平均水分胁迫天数 (>0.05)"),
    ]
    for ax, (col, title) in zip(axes.ravel(), metrics):
        for method, sub in checkpoints.groupby("method", sort=False):
            if col not in sub.columns or sub[col].isna().all():
                continue
            ax.plot(sub["checkpoint_step"], sub[col], marker="o", label=method)
        ax.set_title(title)
        ax.set_xlabel("checkpoint step")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("DQN / MaskableDQN / Demo-DQN 验证年 checkpoint 指标对比", fontsize=14)
    out = FIGS / "044_03_dqn_checkpoint_metric_overview.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return str(out.relative_to(ROOT))


def plot_yearly_best(yearly_best: pd.DataFrame) -> str:
    methods = list(yearly_best["method"].drop_duplicates())
    years = sorted(yearly_best["year"].dropna().unique())
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    width = 0.25
    offsets = {m: (i - (len(methods) - 1) / 2) * width for i, m in enumerate(methods)}
    for method in methods:
        sub = yearly_best[yearly_best["method"] == method]
        values = [sub.loc[sub["year"] == y, "final_grnwt"].iloc[0] if (sub["year"] == y).any() else 0 for y in years]
        ax.bar([y + offsets[method] for y in years], values, width=width, label=method)
    ax.set_title("各 DQN 方法最佳 checkpoint 在 2014–2023 验证年的籽粒产量")
    ax.set_xlabel("验证年份")
    ax.set_ylabel("籽粒产量 (kg/ha)")
    ax.set_xticks(years)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    out = FIGS / "044_03_dqn_best_checkpoint_yearly_yield.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return str(out.relative_to(ROOT))


def plot_q_ranking(q: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for method, sub in q.groupby("method", sort=False):
        axes[0].plot(sub["checkpoint_step"], sub["nonzero_teacher_argmax_rate"], marker="o", label=method)
        axes[1].plot(sub["checkpoint_step"], sub["mean_q_teacher_minus_noop"], marker="o", label=method)
    axes[0].set_title("非零 teacher 动作成为 argmax 的比例")
    axes[0].set_ylabel("rate")
    axes[1].set_title("Q(teacher) - Q(no-op) 平均值")
    axes[1].axhline(0, color="black", linewidth=0.8)
    for ax in axes:
        ax.set_xlabel("checkpoint / pretrain epoch")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Demo-DQN / DQfD 示范动作 Q 排序诊断", fontsize=14)
    out = FIGS / "044_03_demo_dqn_q_ranking_diagnostics.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return str(out.relative_to(ROOT))


def write_record(checkpoints: pd.DataFrame, q: pd.DataFrame, outputs: dict[str, str], figures: list[str]) -> None:
    best = (
        checkpoints.sort_values(["method", "mean_final_grnwt"], ascending=[True, False])
        .groupby("method", as_index=False)
        .head(1)
    )
    lines = [
        "# 044_03 DQN 结果整理：导师汇报用",
        "",
        "## 任务性质",
        "",
        "本任务只整理已有 DQN 结果，不重新训练，不重新运行 DSSAT。",
        "",
        "## 纳入结果",
        "",
        "- 040_01：普通 DQN + 外部动作安全修正，不是严格 MaskableDQN。",
        "- 040_02：严格 MaskableDQN，训练和评估时屏蔽非法动作。",
        "- 044_00–044_02：Demo-DQN / DQfD 风格示范经验尝试与 Q 排序诊断。",
        "",
        "## 核心结论",
        "",
        "1. 普通 DQN 的最好 checkpoint 是按平均产量选出的，但验证年平均产量仍偏低，且动作序列高度固定，早期大量施氮/灌溉特征明显。",
        "2. 严格 MaskableDQN 在 25K 后迅速退化：50K、75K、100K 的平均产量继续下降，并出现少施氮或 no-op 倾向。",
        "3. Demo-DQN / DQfD 路线已确认示范经验进入训练诊断链，但非零 teacher 动作没有被稳定学成 Q 值最高动作；2K smoke 的验证策略为 no-op，产量极低。",
        "4. 这些结果支持把 DQN 作为当前阶段的对照/阴性证据保留，而不是继续无限修 DQN；主线仍应优先回到 PPO 的天气响应性和策略合理性改进。",
        "",
        "## 各方法最佳 checkpoint（按验证年平均产量）",
        "",
        best[
            [
                "method",
                "checkpoint_step",
                "validation_years",
                "mean_final_grnwt",
                "mean_total_irrigation",
                "mean_total_n",
                "mean_PFP_N",
                "mean_swfac_stress_days_gt_0p05",
                "mean_nstres_days_gt_0p05",
            ]
        ].to_markdown(index=False),
        "",
        "## Demo-DQN / DQfD Q 排序摘要",
        "",
        q[
            [
                "method",
                "checkpoint_step",
                "sample_count",
                "nonzero_count",
                "nonzero_teacher_argmax_rate",
                "mean_q_teacher_minus_noop",
                "margin_satisfied_rate",
            ]
        ].to_markdown(index=False),
        "",
        "## 输出文件",
        "",
    ]
    for label, path in outputs.items():
        lines.append(f"- {label}: `{path}`")
    for fig in figures:
        lines.append(f"- figure: `{fig}`")
    lines.extend(
        [
            "",
            "## 汇报口径建议",
            "",
            "可以说：我们按导师要求补了 DQN/DQfD 路线，并且在同一 lowIC、自由时序、动作约束框架下做了普通 DQN、严格 MaskableDQN 和 Demo-DQN/DQfD 对照。结果显示，当前 DQN 系列没有比 PPO 更可靠：普通 DQN 有早期固定动作倾向，严格 MaskableDQN 随训练加深退化，Demo-DQN/DQfD 虽接入示范经验但仍无法把关键非零动作学成高 Q 值。因此当前阶段把 DQN 作为对照保留，主线继续优化 PPO 的天气响应性更合理。",
            "",
        ]
    )
    DOC.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    checkpoints, yearly_best = read_validation_summaries()
    q = read_q_audits()
    outputs = save_tables(checkpoints, yearly_best, q)
    figures = [
        plot_checkpoint_metrics(checkpoints),
        plot_yearly_best(yearly_best),
        plot_q_ranking(q),
    ]
    write_record(checkpoints, q, outputs, figures)
    result = {
        "task": "044_03_dqn_results_for_advisor_summary",
        "tables": outputs,
        "figures": figures,
        "record_md": str(DOC.relative_to(ROOT)),
    }
    (OUT / "044_03_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
