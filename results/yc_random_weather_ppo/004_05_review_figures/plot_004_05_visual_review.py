"""Render the 004_05 visual review using the frozen 222 YC figure style."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "results" / "yc_random_weather_ppo" / "004_05"
OUT = ROOT / "results" / "yc_random_weather_ppo" / "004_05_review_figures"
REPORT = ROOT / "docs" / "yc_random_weather_004_05_visual_review.md"
STYLE_SCRIPT = ROOT / "experiments" / "222_yc_v1_three_seed_confirmation" / "scripts" / "compile_confirmation.py"
STYLE_FIGURES = ROOT / "experiments" / "222_yc_v1_three_seed_confirmation" / "figures"
EPISODES = SOURCE / "all_evaluation_episode_level_0_7.csv"
PAIRED = SOURCE / "paired_seed_performance_comparison.csv"
ARCHETYPES = SOURCE / "policy_archetype_assignment.csv"

METHODS = ("Historical", "Random-weather")
REGIMES = {"HISTORICAL_WEATHER": "Historical", "RANDOM_WEATHER_WGEN": "Random-weather"}
COLORS = {"Historical": "#3366CC", "Random-weather": "#D55E00"}
YEARS = list(range(2014, 2024))
SEEDS = list(range(8))
METRICS = ("yield", "total_irrigation", "total_fertilizer", "reward")
MANAGEMENT = ("total_irrigation", "total_fertilizer", "irrigation_event_count", "fertilizer_event_count")
LABELS = {
    "yield": "Yield (kg ha$^{-1}$)",
    "total_irrigation": "Irrigation (mm)",
    "total_fertilizer": "N input (kg ha$^{-1}$)",
    "reward": "Reward",
    "irrigation_event_count": "Irrigation events",
    "fertilizer_event_count": "N application events",
}
STEMS = {
    "yield": "yield",
    "total_irrigation": "irrigation",
    "total_fertilizer": "nitrogen",
    "reward": "reward",
}
TITLES = {"yield": "yield", "total_irrigation": "irrigation", "total_fertilizer": "N input", "reward": "reward"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    evaluation_qc = json.loads((SOURCE / "evaluation_qc.json").read_text(encoding="utf-8"))
    training_qc = json.loads((SOURCE / "new_training_qc.json").read_text(encoding="utf-8"))
    probe_qc = json.loads((SOURCE / "policy_probe" / "probe_manifest.json").read_text(encoding="utf-8"))
    decision = json.loads((SOURCE / "multi_seed_decision.json").read_text(encoding="utf-8"))
    require(evaluation_qc.get("passed") is True and evaluation_qc.get("expected_all_evaluation_episodes") == 480, "004_05 evaluation QC is not final/passed")
    require(training_qc.get("passed") is True and probe_qc.get("passed") is True, "004_05 training/probe QC is not passed")

    columns = [
        "model_id", "training_regime", "ppo_seed", "evaluation_weather_type",
        "evaluation_weather_seed", "evaluation_weather_year", "reward", "yield",
        "total_irrigation", "total_fertilizer", "irrigation_event_count", "fertilizer_event_count",
    ]
    episodes = pd.read_csv(EPISODES, usecols=columns, low_memory=False)
    require(len(episodes) == 480, f"Expected 480 004_05 episodes, got {len(episodes)}")
    episodes["method"] = episodes["training_regime"].map(REGIMES)
    require(episodes["method"].notna().all(), "Unknown training regime")
    for column in ["ppo_seed", "evaluation_weather_seed", "evaluation_weather_year", *METRICS, "irrigation_event_count", "fertilizer_event_count"]:
        episodes[column] = pd.to_numeric(episodes[column], errors="coerce")
    require(episodes[["ppo_seed", *METRICS, "irrigation_event_count", "fertilizer_event_count"]].notna().all().all(), "Missing numeric episode metric")
    episodes["ppo_seed"] = episodes["ppo_seed"].astype(int)
    require(set(episodes["ppo_seed"]) == set(SEEDS), "Unexpected PPO seed set")
    require((episodes["model_id"] == episodes["method"].map({"Historical": "H", "Random-weather": "W"}) + episodes["ppo_seed"].astype(str)).all(), "Model ID and regime/seed disagree")

    observed = episodes[episodes["evaluation_weather_type"].eq("observed_weather")].copy()
    heldout = episodes[episodes["evaluation_weather_type"].eq("heldout_wgen")].copy()
    require(len(observed) == 160 and len(heldout) == 320, "Observed/held-out episode count differs from 004_05 contract")
    require(observed["evaluation_weather_year"].notna().all(), "Observed year missing")
    observed["year"] = observed["evaluation_weather_year"].astype(int)
    require(set(observed["year"]) == set(YEARS), "Observed year set differs from 2014-2023")
    require(observed.groupby(["method", "ppo_seed", "year"]).size().eq(1).all() and len(observed.groupby(["method", "ppo_seed", "year"])) == 160, "Observed seed-year pair is missing or duplicated")
    require(heldout["evaluation_weather_seed"].notna().all(), "Held-out weather seed missing")
    heldout["weather_seed"] = heldout["evaluation_weather_seed"].astype(int)
    require(set(heldout["weather_seed"]) == set(range(1081, 1101)), "Held-out WGEN seed set differs from 1081-1100")
    require(heldout.groupby(["method", "ppo_seed", "weather_seed"]).size().eq(1).all() and len(heldout.groupby(["method", "ppo_seed", "weather_seed"])) == 320, "Held-out seed pair is missing or duplicated")

    paired = pd.read_csv(PAIRED)
    require(len(paired) == 8 and set(paired["ppo_seed"].astype(int)) == set(SEEDS), "Paired summary has unexpected seed set")
    for seed in SEEDS:
        source_row = paired.loc[paired["ppo_seed"].eq(seed)].iloc[0]
        for domain, frame in (("observed_weather", observed), ("heldout_wgen", heldout)):
            for method, suffix in (("Historical", "historical"), ("Random-weather", "random")):
                part = frame[frame["ppo_seed"].eq(seed) & frame["method"].eq(method)]
                for metric, field in (("reward", "reward"), ("yield", "yield"), ("total_irrigation", "irrigation"), ("total_fertilizer", "fertilizer")):
                    expected = float(source_row[f"{domain}_{field}_{suffix}"])
                    actual = float(part[metric].mean())
                    require(np.isclose(actual, expected, rtol=0, atol=1e-9), f"Episode/paired summary mismatch: {domain}/{method}/seed{seed}/{metric}")

    assignments = pd.read_csv(ARCHETYPES, usecols=["model_id", "training_regime", "ppo_seed", "archetype_label"])
    require(len(assignments) == 16 and set(assignments["model_id"]) == set(episodes["model_id"]), "Archetype labels do not cover all 16 models")
    require(assignments["archetype_label"].notna().all(), "Archetype label missing")
    assignments["method"] = assignments["training_regime"].map(REGIMES)
    return observed, heldout, paired, {"assignments": assignments, "decision": decision, "evaluation_qc": evaluation_qc}


def paired_differences(observed: pd.DataFrame) -> pd.DataFrame:
    index = ["ppo_seed", "year"]
    historical = observed[observed["method"].eq("Historical")].set_index(index).sort_index()
    random = observed[observed["method"].eq("Random-weather")].set_index(index).sort_index()
    require(historical.index.equals(random.index), "Observed seed-year pairing differs between training regimes")
    differences = random[[*METRICS, "irrigation_event_count", "fertilizer_event_count"]] - historical[[*METRICS, "irrigation_event_count", "fertilizer_event_count"]]
    return differences.reset_index()


def save_tables(observed: pd.DataFrame, differences: pd.DataFrame, assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = [*METRICS, "irrigation_event_count", "fertilizer_event_count"]
    group_rows = []
    difference_rows = []
    for year in YEARS:
        year_delta = differences[differences["year"].eq(year)]
        row: dict[str, int | float] = {"year": year}
        for metric in metrics:
            values = year_delta[metric].to_numpy(dtype=float)
            row[f"{metric}_mean_delta"] = float(values.mean())
            row[f"{metric}_seed_sd_delta"] = float(values.std(ddof=1))
            row[f"{metric}_positive_pairs"] = int((values > 0).sum())
            row[f"{metric}_negative_pairs"] = int((values < 0).sum())
        difference_rows.append(row)
        for method in METHODS:
            part = observed[observed["year"].eq(year) & observed["method"].eq(method)]
            for metric in metrics:
                group_rows.append({"year": year, "method": method, "metric": metric, "seed_count": len(part), "mean": float(part[metric].mean()), "seed_sd": float(part[metric].std(ddof=1))})
    by_year = pd.DataFrame(group_rows)
    by_year_delta = pd.DataFrame(difference_rows)
    by_year.to_csv(OUT / "observed_by_year_group_summary.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    by_year_delta.to_csv(OUT / "observed_by_year_paired_difference.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    differences.to_csv(OUT / "observed_seed_year_paired_difference.csv", index=False, encoding="utf-8-sig", lineterminator="\n")

    seed_management = observed.groupby(["model_id", "method", "ppo_seed"], as_index=False)[[*METRICS, "irrigation_event_count", "fertilizer_event_count"]].mean()
    seed_management = seed_management.merge(assignments[["model_id", "archetype_label"]], on="model_id", how="left", validate="one_to_one")
    seed_management.to_csv(OUT / "observed_seed_management_summary.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    return by_year, by_year_delta, seed_management


def year_axis(ax: plt.Axes, observed: pd.DataFrame, metric: str, legend: bool = False) -> None:
    for method in METHODS:
        frame = observed[observed["method"].eq(method)]
        for seed in SEEDS:
            per_seed = frame[frame["ppo_seed"].eq(seed)].sort_values("year")
            ax.plot(per_seed["year"], per_seed[metric], color=COLORS[method], alpha=0.18, lw=0.7)
        mean = frame.groupby("year")[metric].mean().reindex(YEARS)
        ax.plot(YEARS, mean, marker="o", ms=3, lw=2, label=method, color=COLORS[method])
    ax.set_title(LABELS[metric])
    ax.set_xticks(range(2014, 2024, 2))
    ax.grid(alpha=0.2)
    if legend:
        ax.legend(frameon=False)


def difference_axis(ax: plt.Axes, differences: pd.DataFrame, metric: str, legend: bool = False) -> None:
    mean = differences.groupby("year")[metric].mean().reindex(YEARS)
    ax.axhline(0, color="#333333", lw=1.3)
    ax.plot(YEARS, mean, marker="o", ms=3, lw=2, label="Random-weather - Historical", color=COLORS["Random-weather"])
    ax.set_title(LABELS[metric])
    ax.set_xticks(range(2014, 2024, 2))
    ax.grid(alpha=0.2)
    if legend:
        ax.legend(frameon=False)


def save_figure(fig: plt.Figure, name: str, tight: bool = False) -> None:
    if tight:
        fig.savefig(OUT / name, dpi=180, bbox_inches="tight")
    else:
        fig.tight_layout()
        fig.savefig(OUT / name, dpi=180)
    plt.close(fig)


def draw_a(observed: pd.DataFrame) -> list[str]:
    names = []
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        year_axis(ax, observed, metric, legend=True)
        ax.set(xlabel="Year", ylabel=LABELS[metric])
        name = f"figureA_by_year_{STEMS[metric]}.png"
        save_figure(fig, name)
        names.append(name)
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.6), constrained_layout=True)
    for ax, metric in zip(axes.flat, METRICS):
        year_axis(ax, observed, metric, legend=metric == "yield")
    fig.suptitle("YC 100K by-year stability (eight-seed mean)", y=1.02)
    name = "figureA_by_year_stability.png"
    save_figure(fig, name, tight=True)
    return [*names, name]


def draw_b(differences: pd.DataFrame) -> list[str]:
    names = []
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        difference_axis(ax, differences, metric, legend=True)
        ax.set(xlabel="Year", ylabel=f"Difference in {LABELS[metric]}")
        name = f"figureB_by_year_{STEMS[metric]}_difference.png"
        save_figure(fig, name)
        names.append(name)
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.6), constrained_layout=True)
    for ax, metric in zip(axes.flat, METRICS):
        difference_axis(ax, differences, metric, legend=metric == "yield")
    fig.suptitle("YC 100K paired by-year difference (Random-weather - Historical)", y=1.02)
    name = "figureB_by_year_paired_difference.png"
    save_figure(fig, name, tight=True)
    return [*names, name]


def draw_c(frame: pd.DataFrame, name: str, title: str) -> str:
    per_seed = frame.groupby(["method", "ppo_seed"], as_index=False)[list(METRICS)].mean()
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.7), constrained_layout=False)
    for ax, metric in zip(axes, METRICS):
        x = np.arange(8)
        width = 0.34
        for offset, method in ((-width / 2, "Historical"), (width / 2, "Random-weather")):
            values = per_seed[per_seed["method"].eq(method)].sort_values("ppo_seed")[metric].to_numpy(dtype=float)
            ax.bar(x + offset, values, width, label=method if metric == METRICS[0] else None, color=COLORS[method])
        ax.set_title(LABELS[metric])
        ax.set_xticks(x, [str(seed) for seed in SEEDS])
        ax.set_xlabel("Seed")
        ax.grid(axis="y", alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(top=0.76, wspace=0.28)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=False)
    fig.suptitle(title, y=1.08)
    save_figure(fig, name, tight=True)
    return name


def draw_d(observed: pd.DataFrame, assignments: pd.DataFrame) -> str:
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), constrained_layout=True)
    for ax, metric in zip(axes.flat, MANAGEMENT):
        year_axis(ax, observed, metric, legend=metric == MANAGEMENT[0])
    ax = axes.flat[4]
    counts = assignments.groupby(["method", "archetype_label"]).size().unstack(fill_value=0)
    labels = sorted(counts.columns.tolist())
    x = np.arange(len(labels))
    width = 0.34
    for offset, method in ((-width / 2, "Historical"), (width / 2, "Random-weather")):
        ax.bar(x + offset, counts.loc[method, labels].to_numpy(), width, color=COLORS[method])
    ax.set_title("Policy archetype (seed count)")
    ax.set_xticks(x, labels, rotation=55, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.2)
    axes.flat[5].axis("off")
    fig.suptitle("YC 100K management strategy profile (2014-2023)", y=1.02)
    name = "figureD_management_strategy_profile.png"
    save_figure(fig, name, tight=True)
    return name


def md_link(path: Path, label: str | None = None) -> str:
    relative = path.relative_to(ROOT).as_posix()
    return f"[{label or path.name}](../{relative})"


def years_where(frame: pd.DataFrame, column: str, positive: bool) -> str:
    selected = frame.loc[frame[column].gt(0) if positive else frame[column].lt(0), "year"].astype(int).tolist()
    return ", ".join(map(str, selected)) if selected else "无"


def make_report(
    observed: pd.DataFrame,
    heldout: pd.DataFrame,
    paired: pd.DataFrame,
    assignments: pd.DataFrame,
    differences: pd.DataFrame,
    by_year_delta: pd.DataFrame,
    seed_management: pd.DataFrame,
    decision: dict,
    figure_names: list[str],
) -> None:
    h = observed[observed["method"].eq("Historical")]
    w = observed[observed["method"].eq("Random-weather")]
    held_h = heldout[heldout["method"].eq("Historical")]
    held_w = heldout[heldout["method"].eq("Random-weather")]
    report_years = by_year_delta[[
        "year", "reward_mean_delta", "reward_positive_pairs", "yield_mean_delta",
        "yield_positive_pairs", "total_irrigation_mean_delta", "total_fertilizer_mean_delta",
    ]].copy()
    report_years.columns = ["Year", "Reward Δ", "Reward + pairs", "Yield Δ", "Yield + pairs", "Irrigation Δ (mm)", "N Δ (kg/ha)"]
    report_years["Reward Δ"] = report_years["Reward Δ"].map(lambda value: f"{value:+.3f}")
    for column in ("Yield Δ", "Irrigation Δ (mm)", "N Δ (kg/ha)"):
        report_years[column] = report_years[column].map(lambda value: f"{value:+.1f}")
    report_years["Reward + pairs"] = report_years["Reward + pairs"].map(lambda value: f"{value}/8")
    report_years["Yield + pairs"] = report_years["Yield + pairs"].map(lambda value: f"{value}/8")

    seed_rows = []
    for seed in SEEDS:
        hs = seed_management[seed_management["model_id"].eq(f"H{seed}")].iloc[0]
        ws = seed_management[seed_management["model_id"].eq(f"W{seed}")].iloc[0]
        seed_rows.append({
            "Seed": seed,
            "Historical archetype": hs["archetype_label"],
            "Random-weather archetype": ws["archetype_label"],
            "Irrigation Δ (mm)": f"{ws['total_irrigation'] - hs['total_irrigation']:+.1f}",
            "N Δ (kg/ha)": f"{ws['total_fertilizer'] - hs['total_fertilizer']:+.1f}",
            "I events Δ": f"{ws['irrigation_event_count'] - hs['irrigation_event_count']:+.1f}",
            "N events Δ": f"{ws['fertilizer_event_count'] - hs['fertilizer_event_count']:+.1f}",
        })
    seed_table = pd.DataFrame(seed_rows)
    observed_positive_seeds = paired.loc[paired["observed_weather_reward_difference"].gt(0), "ppo_seed"].astype(int).tolist()
    heldout_positive_seeds = paired.loc[paired["heldout_wgen_reward_difference"].gt(0), "ppo_seed"].astype(int).tolist()
    both_positive_seeds = sorted(set(observed_positive_seeds) & set(heldout_positive_seeds))

    figure_table = pd.DataFrame([
        {"图": name, "内容": (
            "逐年指标与 seed 线" if name.startswith("figureA") else
            "逐年配对差值均值与零线" if name.startswith("figureB") else
            "8 个配对 seed 的四指标柱图" if name.startswith("figureC") else
            "逐年管理次数、资源量与 archetype 频数"
        ), "数据": (
            "004_05 all_evaluation_episode_level_0_7.csv；observed_weather" if name.startswith(("figureA", "figureB", "figureD")) or name == "figureC_seed_robustness.png" else
            "004_05 all_evaluation_episode_level_0_7.csv；heldout_wgen"
        ) + ("；policy_archetype_assignment.csv" if name.startswith("figureD") else "")}
        for name in figure_names
    ])
    figure_table["图"] = figure_table["图"].map(lambda name: md_link(OUT / name, name))

    report = f"""# YC 004_05 weather augmentation 视觉审查

## 样式来源

| 作用 | 旧图或脚本 |
|---|---|
| 逐年稳定性 | {md_link(STYLE_FIGURES / 'figure4_100k_by_year_stability.png')} |
| seed 稳健性 | {md_link(STYLE_FIGURES / 'figure3_100k_seed_robustness.png')} |
| 管理资源对比 | {md_link(STYLE_FIGURES / 'figure2_100k_multiobjective_comparison.png')} |
| 单栏线图 | {md_link(STYLE_FIGURES / 'figure1_checkpoint_yield_stability.png')} |
| 原绘图代码 | {md_link(STYLE_SCRIPT)} |
| 原文档引用 | {md_link(ROOT / 'docs' / 'yc_v1_three_seed_confirmation.md')} |

```yaml
style_source_figure_by_year: experiments/222_yc_v1_three_seed_confirmation/figures/figure4_100k_by_year_stability.png
style_source_figure_seed: experiments/222_yc_v1_three_seed_confirmation/figures/figure3_100k_seed_robustness.png
style_source_figure_management: experiments/222_yc_v1_three_seed_confirmation/figures/figure2_100k_multiobjective_comparison.png
style_source_script: experiments/222_yc_v1_three_seed_confirmation/scripts/compile_confirmation.py
style_source_palette: "Historical #3366CC; Random-weather #D55E00; reference #777777"
```

直接复用了旧脚本的 `font.size=9`、`axes.titlesize=10`、白底、180 dpi、蓝橙配色、圆点、网格透明度 0.2、无边框 legend、`6.8 × 4.2` 单栏线图、`15 × 3.7` seed 并列柱图及 `12 × 6.6` 多面板图。旧套图只有 PNG，故按旧交付格式输出 PNG。`figureA` 和 `figureD` 继承旧 by-year 图；`figureC` 继承旧 seed robustness；`figureB` 沿用旧线图并增加清晰的零线。

## 数据与必要调整

全部绘图只读 004_05 的正式 episode 表、paired-seed 表和 archetype 表：{md_link(EPISODES)}、{md_link(PAIRED)}、{md_link(ARCHETYPES)}。训练、评估和 probe QC 均为通过；逐年对照严格筛选 `observed_weather`，共 2 组 × 8 seed × 10 年 = 160 行，每一配对只有一行。held-out WGEN 为 320 行，天气种子 1081–1100。episode 均值与 paired-seed 正式表逐项核对一致。

旧 `figure4` 为五指标 `2 × 3` 布局；本轮核心指标为 Yield、irrigation、N 和 reward，因此 `figureA`/`figureB` 用 `2 × 2` 汇总布局，另输出四张单指标图。为展示八个 seed，`figureA`/`figureD` 在组均值粗线下增加同色浅细单 seed 线；`figureB` 复用旧逐年图的均值线，配对明细另存 CSV，以便零线附近的小差值可见。旧 `figure3` 的五栏改为四栏。`figureD` 保留旧 `2 × 3` 结构：四个逐年管理面板、一个 archetype 频数面板，末格留白。图中 Historical 与 Random-weather 分别对应旧图的 Original 蓝与 Augmented 橙；旧图的 Expert/Farmer 不属于这轮配对实验，故未加入。N 为零的 episode 有 {int(observed['total_fertilizer'].eq(0).sum())} 行，本轮未计算 PFP_N；没有可直接复用的 ETCP 精确回放，因此未绘制 WP_ET/NUE。

## 图与数据源

{figure_table.to_markdown(index=False)}

汇总数据：{md_link(OUT / 'observed_by_year_group_summary.csv')}、{md_link(OUT / 'observed_by_year_paired_difference.csv')}、{md_link(OUT / 'observed_seed_management_summary.csv')}；80 条原始配对差值为 {md_link(OUT / 'observed_seed_year_paired_difference.csv')}。差值均为同一 seed、同一年 `Random-weather - Historical`，`figureB` 画八个差值的均值；各年正差值 seed 数和标准差可从汇总表核查。`figureC_seed_robustness.png` 为 observed 2014–2023，`figureC_seed_robustness_heldout_wgen.png` 为 held-out WGEN 1081–1100。

## 逐年直接观察

正的 reward/yield 差值表示 random-weather 的该指标更高；负的 irrigation/N 差值仅表示投入更少，不单独构成综合胜出。

{report_years.to_markdown(index=False)}

按八 seed 均值，random-weather 的 reward 更高年份：**{years_where(by_year_delta, 'reward_mean_delta', True)}**；更低年份：**{years_where(by_year_delta, 'reward_mean_delta', False)}**。Yield 更高年份：**{years_where(by_year_delta, 'yield_mean_delta', True)}**；更低年份：**{years_where(by_year_delta, 'yield_mean_delta', False)}**。每年 `Reward + pairs` 和 `Yield + pairs` 显示八个同 seed 配对中有多少个差值为正；年份均值为正不表示八个 seed 一致获益。

## Seed 与管理行为

Observed 2014–2023 的八 seed 平均：Historical reward `{h['reward'].mean():.4f}`、yield `{h['yield'].mean():.1f}` kg/ha、灌溉 `{h['total_irrigation'].mean():.1f}` mm、施氮 `{h['total_fertilizer'].mean():.1f}` kg/ha；Random-weather 分别为 `{w['reward'].mean():.4f}`、`{w['yield'].mean():.1f}`、`{w['total_irrigation'].mean():.1f}`、`{w['total_fertilizer'].mean():.1f}`。管理次数均值为 Historical 灌溉 `{h['irrigation_event_count'].mean():.2f}`、施氮 `{h['fertilizer_event_count'].mean():.2f}`，Random-weather 灌溉 `{w['irrigation_event_count'].mean():.2f}`、施氮 `{w['fertilizer_event_count'].mean():.2f}`。总体表现为较少灌溉及灌溉次数、较多施氮、较低 yield/reward；各 seed 的资源方向并不相同。

{seed_table.to_markdown(index=False)}

Archetype 按 004_05 固定 probe 分类，不依 reward 重新贴标签。组频数和配对转换参见原 {md_link(SOURCE / 'archetype_frequency_by_regime.csv')} 与 {md_link(SOURCE / 'paired_seed_archetype_transition.csv')}。图中 `VERY_LOW_INPUT`、`OTHER_NEW` 等标签就是该正式分类；八个配对中 {decision['paired_seed_switch_count']}/8 更换标签，但主转换方向仅占 {decision['dominant_transition_share_among_switches']:.1%}，所以不把它解释为统一策略迁移。

## 是否胜出

Observed mean reward 的局部正差值 seed：**{', '.join(map(str, observed_positive_seeds)) if observed_positive_seeds else '无'}**；held-out WGEN mean reward 的局部正差值 seed：**{', '.join(map(str, heldout_positive_seeds)) if heldout_positive_seeds else '无'}**；两个域均为正的 seed：**{', '.join(map(str, both_positive_seeds)) if both_positive_seeds else '无'}**。Held-out WGEN 的八 seed 均值 reward 为 Historical `{held_h['reward'].mean():.4f}`、Random-weather `{held_w['reward'].mean():.4f}`，yield 为 `{held_h['yield'].mean():.1f}` 与 `{held_w['yield'].mean():.1f}` kg/ha。004_05 预设的跨域资源/产量/reward 联合成功条件为 **{decision['success_count']}/8**；图中没有这一轮整体稳定胜出的视觉证据。Archetype 频率描述为 `{decision['archetype_distribution_finding']}`，与性能成功分开解读。

Observed 2014–2023 为独立比较期，不宣称 pristine final test set。以上图是已完成正式评估的可视化，没有新训练、DSSAT 运行或策略变更。
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8", newline="\n")


def main() -> None:
    require(STYLE_SCRIPT.is_file() and all((STYLE_FIGURES / name).is_file() for name in (
        "figure1_checkpoint_yield_stability.png", "figure2_100k_multiobjective_comparison.png",
        "figure3_100k_seed_robustness.png", "figure4_100k_by_year_stability.png",
    )), "Frozen 222 YC style source is missing")
    observed, heldout, paired, metadata = load_data()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 120})
    differences = paired_differences(observed)
    _, by_year_delta, seed_management = save_tables(observed, differences, metadata["assignments"])
    figures = [
        *draw_a(observed),
        *draw_b(differences),
        draw_c(observed, "figureC_seed_robustness.png", "YC 100K seed robustness (observed 2014-2023)"),
        draw_c(heldout, "figureC_seed_robustness_heldout_wgen.png", "YC 100K seed robustness (held-out WGEN)"),
        draw_d(observed, metadata["assignments"]),
    ]
    make_report(observed, heldout, paired, metadata["assignments"], differences, by_year_delta, seed_management, metadata["decision"], figures)
    manifest = {
        "source_episode_rows": 480,
        "observed_rows": len(observed),
        "heldout_rows": len(heldout),
        "observed_seed_year_pairs": 80,
        "paired_summary_reconciled": True,
        "figure_count": len(figures),
        "figures": [{"path": (OUT / name).relative_to(ROOT).as_posix(), "sha256": sha256(OUT / name)} for name in figures],
        "input_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in (EPISODES, PAIRED, ARCHETYPES, SOURCE / "evaluation_qc.json", STYLE_SCRIPT)},
        "report": REPORT.relative_to(ROOT).as_posix(),
        "reward_component_used_for_plots": "episode total reward only; no unrecorded step components",
        "new_training_or_dssat_evaluation": False,
    }
    (OUT / "visual_review_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"figures": len(figures), "observed_rows": len(observed), "heldout_rows": len(heldout), "report": str(REPORT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
