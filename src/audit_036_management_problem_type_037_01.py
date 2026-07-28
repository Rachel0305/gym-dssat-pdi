from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK = "037_01_036_management_problem_type_audit"
OUT = ROOT / "benchmark_results" / TASK
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
DOC = ROOT / "docs" / f"{TASK}_record.md"

YEAR_AUDIT = (
    ROOT
    / "benchmark_results"
    / "036_05_selected_ppo_management_rationality_audit"
    / "tables"
    / "036_05_year_level_management_audit.csv"
)
EVENT_AUDIT = (
    ROOT
    / "benchmark_results"
    / "036_05_selected_ppo_management_rationality_audit"
    / "tables"
    / "036_05_event_level_audit.csv"
)
METRICS = (
    ROOT
    / "benchmark_results"
    / "036_04_select_checkpoint_and_plot_03601_03603_summary"
    / "tables"
    / "036_04_selected_year_level_comparison.csv"
)

MAX_REASONABLE_IRR_EVENTS = 4
MAX_REASONABLE_N_EVENTS = 4
EARLY_SHARE_THRESHOLD = 0.80


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    years = pd.read_csv(YEAR_AUDIT)
    events = pd.read_csv(EVENT_AUDIT)
    metrics = pd.read_csv(METRICS)
    required_year_cols = {
        "station_code",
        "year",
        "checkpoint_step",
        "final_grnwt",
        "total_irrigation",
        "total_n",
        "irrigation_event_count",
        "n_event_count",
        "early_irrigation_share",
        "early_n_share",
        "irrigation_interval_violations_lt7d",
        "n_interval_violations_lt7d",
        "mostly_preventive_warning",
        "max_swfac",
        "max_nstres",
    }
    missing = sorted(required_year_cols - set(years.columns))
    if missing:
        raise RuntimeError(f"036_05 year audit missing columns: {missing}")
    required_metric_cols = {
        "station_code",
        "year",
        "checkpoint_step",
        "yield_gap_vs_four_max",
        "wp_et_gap_vs_four_max",
        "pfp_n_gap_vs_four_max",
        "any_metric_win_four_max",
    }
    missing = sorted(required_metric_cols - set(metrics.columns))
    if missing:
        raise RuntimeError(f"036_04 metrics missing columns: {missing}")
    return years, events, metrics


def classify_years(years: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["station_code", "year", "checkpoint_step"]
    m = metrics[
        keys
        + [
            "yield_gap_vs_four_max",
            "wp_et_gap_vs_four_max",
            "pfp_n_gap_vs_four_max",
            "any_metric_win_four_max",
        ]
    ].copy()
    df = years.merge(m, on=keys, how="left", validate="one_to_one", suffixes=("_mgmt", ""))
    if "any_metric_win_four_max" not in df.columns and "any_metric_win_four_max_mgmt" in df.columns:
        df["any_metric_win_four_max"] = df["any_metric_win_four_max_mgmt"]
    df["mostly_preventive_warning"] = as_bool(df["mostly_preventive_warning"])
    df["any_metric_win_four_max"] = as_bool(df["any_metric_win_four_max"])

    df["too_many_irrigation_events"] = df["irrigation_event_count"] > MAX_REASONABLE_IRR_EVENTS
    df["too_many_n_events"] = df["n_event_count"] > MAX_REASONABLE_N_EVENTS
    df["interval_violation"] = (
        df["irrigation_interval_violations_lt7d"].fillna(0).gt(0)
        | df["n_interval_violations_lt7d"].fillna(0).gt(0)
    )
    df["frequent_operation_problem"] = (
        df["too_many_irrigation_events"] | df["too_many_n_events"] | df["interval_violation"]
    )

    df["early_irrigation_concentration"] = (
        df["total_irrigation"].gt(0) & df["early_irrigation_share"].fillna(0).ge(EARLY_SHARE_THRESHOLD)
    )
    df["early_n_concentration"] = df["total_n"].gt(0) & df["early_n_share"].fillna(0).ge(EARLY_SHARE_THRESHOLD)
    df["early_dump_problem"] = df["early_irrigation_concentration"] | df["early_n_concentration"]

    df["yield_win_four_max"] = df["yield_gap_vs_four_max"].gt(0)
    df["wp_et_win_four_max"] = df["wp_et_gap_vs_four_max"].gt(0)
    df["pfp_n_win_four_max"] = df["pfp_n_gap_vs_four_max"].gt(0)
    df["win_metric_count"] = df[["yield_win_four_max", "wp_et_win_four_max", "pfp_n_win_four_max"]].sum(axis=1)

    conditions = [
        df["early_dump_problem"] & df["frequent_operation_problem"],
        df["early_dump_problem"] & ~df["frequent_operation_problem"],
        ~df["early_dump_problem"] & df["frequent_operation_problem"],
        ~df["early_dump_problem"] & ~df["frequent_operation_problem"] & df["mostly_preventive_warning"],
        ~df["early_dump_problem"] & ~df["frequent_operation_problem"] & ~df["mostly_preventive_warning"],
    ]
    choices = [
        "早期集中投入+频繁/间隔问题",
        "主要是早期集中投入",
        "主要是频繁/间隔问题",
        "无硬问题但偏预防性",
        "未见明显过程问题",
    ]
    df["problem_type"] = np.select(conditions, choices, default="待核查")
    return df.sort_values(["station_code", "year"]).reset_index(drop=True)


def summarize_station(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby("station_code", as_index=False).agg(
        years=("year", "nunique"),
        metric_success_years=("any_metric_win_four_max", "sum"),
        mean_win_metric_count=("win_metric_count", "mean"),
        early_dump_years=("early_dump_problem", "sum"),
        frequent_problem_years=("frequent_operation_problem", "sum"),
        preventive_years=("mostly_preventive_warning", "sum"),
        too_many_irrigation_years=("too_many_irrigation_events", "sum"),
        too_many_n_years=("too_many_n_events", "sum"),
        interval_violation_years=("interval_violation", "sum"),
        mean_irrigation_events=("irrigation_event_count", "mean"),
        mean_n_events=("n_event_count", "mean"),
        mean_early_irrigation_share=("early_irrigation_share", "mean"),
        mean_early_n_share=("early_n_share", "mean"),
        mean_yield_gap_vs_four_max=("yield_gap_vs_four_max", "mean"),
        mean_wp_et_gap_vs_four_max=("wp_et_gap_vs_four_max", "mean"),
        mean_pfp_n_gap_vs_four_max=("pfp_n_gap_vs_four_max", "mean"),
    )
    out["early_dump_rate"] = out["early_dump_years"] / out["years"]
    out["frequent_problem_rate"] = out["frequent_problem_years"] / out["years"]
    return out


def savefig(fig: plt.Figure, stem: str) -> list[str]:
    files = []
    for suffix in ["png", "svg"]:
        p = FIG_DIR / f"{stem}.{suffix}"
        fig.savefig(p, dpi=220, bbox_inches="tight")
        files.append(str(p.relative_to(ROOT)))
    plt.close(fig)
    return files


def plot_problem_composition(station: pd.DataFrame) -> list[str]:
    s = station.sort_values("station_code")
    x = np.arange(len(s))
    width = 0.22
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - width, s["early_dump_years"], width, label="early dump")
    ax.bar(x, s["frequent_problem_years"], width, label="frequent/interval")
    ax.bar(x + width, s["preventive_years"], width, label="preventive")
    ax.set_xticks(x)
    ax.set_xticklabels(s["station_code"])
    ax.set_ylim(0, max(10, int(s["years"].max())) + 1)
    ax.set_ylabel("years")
    ax.set_title("036 PPO management problem type by station")
    ax.legend(ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    return savefig(fig, "037_01_station_problem_type_counts")


def plot_event_counts(df: pd.DataFrame) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, col, title in [
        (axes[0], "irrigation_event_count", "Irrigation event count"),
        (axes[1], "n_event_count", "Nitrogen event count"),
    ]:
        data = [g[col].dropna().to_numpy() for _, g in df.groupby("station_code")]
        labels = [k for k, _ in df.groupby("station_code")]
        ax.boxplot(data, labels=labels, showmeans=True)
        ax.axhline(4, color="tab:red", ls="--", lw=1, label="event count = 4")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("events per season")
    axes[1].legend(frameon=False)
    fig.suptitle("036 PPO event counts across selected validation years")
    return savefig(fig, "037_01_event_count_distribution")


def plot_early_share(df: pd.DataFrame) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, col, title in [
        (axes[0], "early_irrigation_share", "DAP1-10 irrigation share"),
        (axes[1], "early_n_share", "DAP1-10 nitrogen share"),
    ]:
        data = [g[col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy() for _, g in df.groupby("station_code")]
        labels = [k for k, _ in df.groupby("station_code")]
        ax.boxplot(data, labels=labels, showmeans=True)
        ax.axhline(EARLY_SHARE_THRESHOLD, color="tab:red", ls="--", lw=1, label="0.80 threshold")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("share of seasonal amount")
    axes[1].legend(frameon=False)
    fig.suptitle("036 PPO early concentration diagnostic")
    return savefig(fig, "037_01_early_share_distribution")


def write_record(df: pd.DataFrame, station: pd.DataFrame, figures: list[str]) -> None:
    overall = pd.DataFrame(
        [
            {
                "year_count": len(df),
                "metric_success_years": int(df["any_metric_win_four_max"].sum()),
                "early_dump_years": int(df["early_dump_problem"].sum()),
                "frequent_problem_years": int(df["frequent_operation_problem"].sum()),
                "preventive_years": int(df["mostly_preventive_warning"].sum()),
                "too_many_irrigation_years": int(df["too_many_irrigation_events"].sum()),
                "too_many_n_years": int(df["too_many_n_events"].sum()),
                "interval_violation_years": int(df["interval_violation"].sum()),
            }
        ]
    )
    overall.to_csv(TABLE_DIR / "037_01_overall_problem_type_summary.csv", index=False, encoding="utf-8-sig")

    lines: list[str] = []
    lines.append("# 037_01：036 主线 PPO 措施问题类型审计记录\n")
    lines.append("\n## 任务边界\n")
    lines.append("- 不训练、不重跑 DSSAT、不修改 reward 或动作约束。\n")
    lines.append("- 本任务只判断 036 措施问题主要属于频繁操作，还是早期集中投入。\n")
    lines.append("\n## 预注册判据\n")
    lines.append("- 频繁/碎片化：灌溉事件数 > 4，或施氮事件数 > 4，或同类操作间隔 < 7 天。\n")
    lines.append("- 早期集中投入：DAP1-10 灌溉或施氮占季节总量 >= 80%。\n")
    lines.append("- 预防性操作不直接判错，只作为解释性标签。\n")
    lines.append("\n## 总体结果\n")
    lines.append(overall.to_markdown(index=False))
    lines.append("\n\n## 站点级结果\n")
    cols = [
        "station_code",
        "years",
        "metric_success_years",
        "early_dump_years",
        "frequent_problem_years",
        "preventive_years",
        "too_many_irrigation_years",
        "too_many_n_years",
        "interval_violation_years",
        "mean_irrigation_events",
        "mean_n_events",
        "mean_early_irrigation_share",
        "mean_early_n_share",
    ]
    lines.append(station[cols].round(3).to_markdown(index=False))
    lines.append("\n\n## 年份级问题类型样例\n")
    sample_cols = [
        "station_code",
        "year",
        "checkpoint_step",
        "final_grnwt",
        "total_irrigation",
        "total_n",
        "irrigation_event_count",
        "n_event_count",
        "early_irrigation_share",
        "early_n_share",
        "frequent_operation_problem",
        "early_dump_problem",
        "mostly_preventive_warning",
        "problem_type",
        "win_metric_count",
    ]
    lines.append(df[sample_cols].round(3).to_markdown(index=False))
    lines.append("\n\n## 判定\n")
    early = int(df["early_dump_problem"].sum())
    freq = int(df["frequent_operation_problem"].sum())
    if early > freq:
        lines.append(
            f"- 早期集中投入年份 {early}/50，多于频繁/间隔问题年份 {freq}/50。下一步不应优先加最大事件数，而应优先处理 early dump。\n"
        )
    elif freq > early:
        lines.append(
            f"- 频繁/间隔问题年份 {freq}/50，多于早期集中投入年份 {early}/50。下一步可优先考虑最大事件数或更强间隔约束。\n"
        )
    else:
        lines.append(f"- 早期集中投入和频繁/间隔问题均为 {early}/50，需要结合站点分布决定下一步。\n")
    lines.append(
        "- 预防性操作为解释性现象，不应单独作为失败标准；但如果导师追问，需要通过五情景过程图或反事实消融说明其必要性。\n"
    )
    lines.append("\n## 输出图件\n")
    for f in figures:
        lines.append(f"- `{f}`\n")
    lines.append("\n## 输出表格\n")
    for f in [
        "benchmark_results/037_01_036_management_problem_type_audit/tables/037_01_year_level_problem_type.csv",
        "benchmark_results/037_01_036_management_problem_type_audit/tables/037_01_station_level_problem_type.csv",
        "benchmark_results/037_01_036_management_problem_type_audit/tables/037_01_overall_problem_type_summary.csv",
    ]:
        lines.append(f"- `{f}`\n")
    DOC.write_text("".join(lines), encoding="utf-8-sig")


def main() -> None:
    ensure_dirs()
    years, events, metrics = load_data()
    classified = classify_years(years, metrics)
    station = summarize_station(classified)
    classified.to_csv(TABLE_DIR / "037_01_year_level_problem_type.csv", index=False, encoding="utf-8-sig")
    station.to_csv(TABLE_DIR / "037_01_station_level_problem_type.csv", index=False, encoding="utf-8-sig")

    figures: list[str] = []
    figures.extend(plot_problem_composition(station))
    figures.extend(plot_event_counts(classified))
    figures.extend(plot_early_share(classified))
    write_record(classified, station, figures)
    print(f"Wrote {TABLE_DIR}")
    print(f"Wrote {FIG_DIR}")
    print(f"Wrote {DOC}")


if __name__ == "__main__":
    main()
