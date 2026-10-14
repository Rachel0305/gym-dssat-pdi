"""Aggregate all eight FQA WGEN PPO seeds without best-seed selection."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
FIGURE_ROOT = BASE / "055_03_five_scenario/FQ"
OUT = FIGURE_ROOT / "cohort_8seed_summary"
SEEDS = list(range(8))
YEARS = list(range(2014, 2024))
METRICS = {
    "grain_yield_kg_ha": "Yield (kg/ha)",
    "WP_ET_kg_m3": "WP_ET (kg/m³)",
    "PFP_N_kg_grain_per_kg_N": "PFP_N (kg grain/kg N)",
    "total_irrigation_mm": "Irrigation (mm)",
    "total_n_kg_ha": "Nitrogen (kg/ha)",
}
BASELINES = ["null", "recorded_farmer_template", "dssat_auto_external_n", "official_extension_expert"]
LABELS = {
    "null": "Null",
    "recorded_farmer_template": "Recorded",
    "dssat_auto_external_n": "Auto + N",
    "official_extension_expert": "Expert",
    "ppo": "PPO cohort",
}
COLORS = ["#555555", "#C44E52", "#D8A305", "#7E63B6", "#2A9D55"]


def main() -> None:
    frames = []
    for seed in SEEDS:
        path = FIGURE_ROOT / f"best_seed_seed{seed}/tables/fq_seed{seed}_yearly_five_scenario_metrics.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing seed {seed} five-scenario table: {path}")
        frame = pd.read_csv(path, keep_default_na=False)
        if len(frame) != len(YEARS) * 5:
            raise ValueError(f"Seed {seed} expected 50 year/scenario rows, found {len(frame)}")
        frame["seed"] = seed
        frame["year"] = pd.to_numeric(frame.year).astype(int)
        for metric in METRICS:
            frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
        frames.append(frame)
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows.to_csv(OUT / "all_seed_year_scenario_metrics.csv", index=False, encoding="utf-8-sig")

    baseline = all_rows.loc[all_rows.scenario.isin(BASELINES)].copy()
    for scenario in BASELINES:
        rows = baseline.loc[baseline.scenario.eq(scenario), ["year", *METRICS.keys()]].drop_duplicates()
        if len(rows) != len(YEARS):
            raise ValueError(f"Baseline {scenario} must have one identical row/year across all seeds")
    baseline = baseline.drop_duplicates(subset=["scenario", "year"]).copy()

    ppo_rows = all_rows.loc[all_rows.scenario.eq(all_rows.seed.map(lambda x: f"ppo_seed_{int(x)}"))].copy()
    if len(ppo_rows) != len(SEEDS) * len(YEARS):
        raise ValueError(f"Expected 80 PPO seed-year rows, found {len(ppo_rows)}")
    seed_means = ppo_rows.groupby("seed", as_index=False)[list(METRICS)].mean()
    seed_means.to_csv(OUT / "ppo_seed_10year_means.csv", index=False, encoding="utf-8-sig")

    annual_rows = []
    for year in YEARS:
        for metric, label in METRICS.items():
            ppo = ppo_rows.loc[ppo_rows.year.eq(year), metric].astype(float)
            row = {
                "year": year, "metric": metric, "metric_label": label,
                "ppo_mean_across_8_seeds": float(ppo.mean()), "ppo_sd_across_8_seeds": float(ppo.std(ddof=1)),
                "ppo_min": float(ppo.min()), "ppo_max": float(ppo.max()),
            }
            for scenario in BASELINES:
                vals = baseline.loc[(baseline.year.eq(year)) & baseline.scenario.eq(scenario), metric].astype(float).unique()
                if len(vals) != 1:
                    raise ValueError(f"Expected one value for {scenario} {year} {metric}")
                row[f"{scenario}_baseline"] = float(vals[0])
                row[f"ppo_minus_{scenario}"] = float(ppo.mean() - vals[0])
            annual_rows.append(row)
    annual = pd.DataFrame(annual_rows)
    annual.to_csv(OUT / "cohort_by_year_mean_sd_and_baseline_deltas.csv", index=False, encoding="utf-8-sig")

    paired_rows = []
    for metric, label in METRICS.items():
        ppo_seed_mean = ppo_rows.groupby("seed")[metric].mean()
        for scenario in BASELINES:
            base_mean = baseline.loc[baseline.scenario.eq(scenario)].groupby("year")[metric].first().mean()
            diffs = ppo_seed_mean - float(base_mean)
            paired_rows.append({
                "metric": metric, "metric_label": label, "comparison": scenario,
                "baseline_mean_10y": float(base_mean), "ppo_mean_across_seed_means": float(ppo_seed_mean.mean()),
                "ppo_between_seed_sd_of_10y_means": float(ppo_seed_mean.std(ddof=1)),
                "mean_paired_delta_across_8_seeds": float(diffs.mean()),
                "sd_paired_delta_across_8_seeds": float(diffs.std(ddof=1)),
                "seeds_above_baseline": int((diffs > 0).sum()),
                "seed_count": len(diffs),
                "interpretation": "descriptive multi-seed comparison; no significance claim",
            })
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(OUT / "cohort_paired_10year_comparison.csv", index=False, encoding="utf-8-sig")

    summary = []
    for metric, label in METRICS.items():
        s = ppo_seed_mean = ppo_rows.groupby("seed")[metric].mean()
        baseline_means = {
            scenario: float(baseline.loc[baseline.scenario.eq(scenario)].groupby("year")[metric].first().mean())
            for scenario in BASELINES
        }
        summary.append({
            "metric": metric, "metric_label": label,
            "ppo_8seed_mean_of_10y_means": float(s.mean()),
            "ppo_8seed_sd": float(s.std(ddof=1)), "ppo_8seed_min": float(s.min()), "ppo_8seed_max": float(s.max()),
            **{f"{scenario}_10y_mean": value for scenario, value in baseline_means.items()},
            **{f"seeds_above_{scenario}": int((s > value).sum()) for scenario, value in baseline_means.items()},
        })
    overall = pd.DataFrame(summary)
    overall.to_csv(OUT / "cohort_overall_mean_sd.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    axes = axes.ravel()
    for ax, (metric, label) in zip(axes, METRICS.items()):
        ppo_means = ppo_rows.groupby("seed")[metric].mean()
        baseline_values = [
            float(baseline.loc[baseline.scenario.eq(scenario)].groupby("year")[metric].first().mean())
            for scenario in BASELINES
        ]
        x = np.arange(5)
        vals = [*baseline_values, float(ppo_means.mean())]
        err = [0, 0, 0, 0, float(ppo_means.std(ddof=1))]
        ax.bar(x, vals, color=COLORS, yerr=err, capsize=4)
        ax.scatter(np.repeat(4, len(ppo_means)), ppo_means.to_numpy(), color="#111111", s=22, zorder=3, label="PPO seed means")
        ax.set_xticks(x, [LABELS[s] for s in BASELINES] + [LABELS["ppo"]], rotation=22, ha="right")
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=7, loc="best")
    axes[-1].axis("off")
    fig.suptitle("FQ WGEN PPO 100K: eight-seed mean ± between-seed SD", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "cohort_8seed_mean_sd_vs_baselines.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for ax, metric, label in zip(axes.flat, ["grain_yield_kg_ha", "total_irrigation_mm", "total_n_kg_ha", "PFP_N_kg_grain_per_kg_N"], ["Yield (kg/ha)", "Irrigation (mm)", "Nitrogen (kg/ha)", "PFP_N (kg grain/kg N)"]):
        for seed in SEEDS:
            values = ppo_rows.loc[ppo_rows.seed.eq(seed)].set_index("year").reindex(YEARS)[metric]
            ax.plot(YEARS, values, marker="o", ms=3, lw=1, alpha=0.68, label=f"seed {seed}")
        for scenario, color in zip(BASELINES, COLORS[:4]):
            values = baseline.loc[baseline.scenario.eq(scenario)].set_index("year").reindex(YEARS)[metric]
            ax.plot(YEARS, values, color=color, lw=1.5, ls="--", alpha=0.85, label=LABELS[scenario])
        ax.set_title(label)
        ax.set_xticks(YEARS[::2])
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("FQ WGEN PPO 100K: all eight seeds by validation year", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "cohort_8seed_yearly_traces.png", dpi=160)
    plt.close(fig)

    lines = [
        "# FQ WGEN PPO 100K — 八个随机种子汇总", "",
        "范围：PPO 训练种子 0–7；各模型在 2014–2023 固定历史天气上确定性评估；四个对照采用冻结的 051_03 结果。",
        "每个种子的 PPO 10 年平均值为一个统计单位；表中 ±SD 表示 8 个训练种子之间的标准差。这里是描述性比较，不作显著性推断。",
        "WP_ET 使用逐季 DSSAT Summary.OUT 的 ETCP 精确计算；PFP_N 沿用 Summary.OUT 口径；没有作物吸氮量时不报告 NUE。",
        "", "## PPO 8 种子均值与四个冻结基线", "",
        "| 指标 | PPO mean ± SD | PPO min–max | Null | Recorded | Auto + N | Expert | PPO 高于基线的种子数（Null / Recorded / Auto+N / Expert） |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall.to_dict(orient="records"):
        fmt = lambda x: "—" if pd.isna(x) else f"{float(x):.2f}"
        win = " / ".join(str(row[f"seeds_above_{s}"]) + "/8" for s in BASELINES)
        lines.append(
            f"| {row['metric_label']} | {fmt(row['ppo_8seed_mean_of_10y_means'])} ± {fmt(row['ppo_8seed_sd'])} | "
            f"{fmt(row['ppo_8seed_min'])}–{fmt(row['ppo_8seed_max'])} | {fmt(row['null_10y_mean'])} | "
            f"{fmt(row['recorded_farmer_template_10y_mean'])} | {fmt(row['dssat_auto_external_n_10y_mean'])} | "
            f"{fmt(row['official_extension_expert_10y_mean'])} | {win} |"
        )
    lines += [
        "", "## 解释边界", "",
        "- 所有种子使用相同的 WGEN 训练 episode 调度，因此这里主要反映 PPO 初始化和优化随机性，不是 WGEN 天气池不确定性的独立重复抽样。",
        "- 同一套 2014–2023 验证天气与四个冻结基线配对；seed 0–7 全部保留，不按结果选择‘最佳种子’。",
        "- FQ 2018 原始 Tmin 异常按照冻结基线输入保留；相关图表和逐季快照可追溯。",
        "", "## 文件", "",
        "- `cohort_overall_mean_sd.csv`: 8 个种子的 10 年平均值及 4 个基线均值。",
        "- `cohort_paired_10year_comparison.csv`: 每个 PPO seed 对四基线的 10 年配对差值。",
        "- `cohort_by_year_mean_sd_and_baseline_deltas.csv`: 年度 PPO 均值、标准差及基线差。",
        "- `ppo_seed_10year_means.csv`: 每个 seed 的 10 年均值。",
        "- `all_seed_year_scenario_metrics.csv`: 全部 8×10×5 逐年情景指标。",
        "- `cohort_8seed_mean_sd_vs_baselines.png` 和 `cohort_8seed_yearly_traces.png`: 整组图。",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "PASS_8_SEED_AGGREGATION", "rows": len(all_rows), "ppo_seed_year_rows": len(ppo_rows), "output": OUT.relative_to(ROOT).as_posix()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main()
