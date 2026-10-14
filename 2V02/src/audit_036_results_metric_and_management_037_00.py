from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK = "037_00_036_results_metric_and_management_audit"
OUT = ROOT / "benchmark_results" / TASK
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
DOC = ROOT / "docs" / f"{TASK}_record.md"

SELECTED_YEAR = (
    ROOT
    / "benchmark_results"
    / "036_04_select_checkpoint_and_plot_03601_03603_summary"
    / "tables"
    / "036_04_selected_year_level_comparison.csv"
)
SELECTED_CKPT = (
    ROOT
    / "benchmark_results"
    / "036_04_select_checkpoint_and_plot_03601_03603_summary"
    / "tables"
    / "036_04_selected_checkpoints.csv"
)
MGMT_AUDIT = (
    ROOT
    / "benchmark_results"
    / "036_05_selected_ppo_management_rationality_audit"
    / "tables"
    / "036_05_year_level_management_audit.csv"
)


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def require_columns(df: pd.DataFrame, cols: set[str], name: str) -> None:
    missing = sorted(cols - set(df.columns))
    if missing:
        raise RuntimeError(f"{name} missing required columns: {missing}")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = pd.read_csv(SELECTED_YEAR)
    ckpts = pd.read_csv(SELECTED_CKPT)
    mgmt = pd.read_csv(MGMT_AUDIT)

    require_columns(
        selected,
        {
            "station_code",
            "year",
            "checkpoint_step",
            "final_grnwt",
            "total_irrigation",
            "total_n",
            "WP_ET_kg_m3",
            "PFP_N_kg_kg",
            "yield_gap_vs_four_max",
            "wp_et_gap_vs_four_max",
            "pfp_n_gap_vs_four_max",
            "any_metric_win_four_max",
            "water_saving_vs_expert_mm",
            "n_saving_vs_expert_kg_ha",
            "yield_gap_vs_expert",
            "wp_et_gap_vs_expert",
            "pfp_n_gap_vs_expert",
            "action_sequence",
            "daily_csv_path",
        },
        "036_04 selected year table",
    )
    require_columns(
        mgmt,
        {
            "station_code",
            "year",
            "checkpoint_step",
            "irrigation_event_count",
            "n_event_count",
            "first_irrigation_dap",
            "first_n_dap",
            "early_dump_warning",
            "interval_warning",
            "late_n_warning",
            "zero_yield_warning",
            "mostly_preventive_warning",
            "management_risk_flag_count",
            "max_swfac",
            "max_nstres",
            "swfac_days_gt_0p05",
            "nstres_days_gt_0p05",
        },
        "036_05 management audit table",
    )
    return selected, ckpts, mgmt


def as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def build_year_table(selected: pd.DataFrame, mgmt: pd.DataFrame) -> pd.DataFrame:
    merge_keys = ["station_code", "year", "checkpoint_step"]
    cols_selected = [
        "station_code",
        "year",
        "checkpoint_step",
        "final_grnwt",
        "total_irrigation",
        "total_n",
        "WP_ET_kg_m3",
        "PFP_N_kg_kg",
        "yield_gap_vs_four_max",
        "wp_et_gap_vs_four_max",
        "pfp_n_gap_vs_four_max",
        "any_metric_win_four_max",
        "water_saving_vs_expert_mm",
        "n_saving_vs_expert_kg_ha",
        "yield_gap_vs_expert",
        "wp_et_gap_vs_expert",
        "pfp_n_gap_vs_expert",
        "action_sequence",
        "daily_csv_path",
    ]
    cols_mgmt = [
        "station_code",
        "year",
        "checkpoint_step",
        "irrigation_event_count",
        "n_event_count",
        "first_irrigation_dap",
        "first_n_dap",
        "early_irrigation_share",
        "early_n_share",
        "early_dump_warning",
        "interval_warning",
        "late_n_warning",
        "zero_yield_warning",
        "mostly_preventive_warning",
        "management_risk_flag_count",
        "management_risk_flags",
        "max_swfac",
        "max_nstres",
        "swfac_days_gt_0p05",
        "nstres_days_gt_0p05",
    ]
    df = selected[cols_selected].merge(mgmt[cols_mgmt], on=merge_keys, how="left", validate="one_to_one")

    for c in [
        "any_metric_win_four_max",
        "early_dump_warning",
        "interval_warning",
        "late_n_warning",
        "zero_yield_warning",
        "mostly_preventive_warning",
    ]:
        df[c] = as_bool(df[c])

    df["yield_win_four_max"] = df["yield_gap_vs_four_max"] > 0
    df["wp_et_win_four_max"] = df["wp_et_gap_vs_four_max"] > 0
    df["pfp_n_win_four_max"] = df["pfp_n_gap_vs_four_max"] > 0
    df["win_metric_count"] = df[["yield_win_four_max", "wp_et_win_four_max", "pfp_n_win_four_max"]].sum(axis=1)
    df["hard_risk_count"] = df[["early_dump_warning", "interval_warning", "late_n_warning", "zero_yield_warning"]].sum(axis=1)
    df["process_clean_hard"] = df["hard_risk_count"].eq(0)
    df["指标结论"] = np.select(
        [
            df["win_metric_count"].ge(2),
            df["win_metric_count"].eq(1),
            df["win_metric_count"].eq(0),
        ],
        ["至少2项超过四情景最高值", "1项超过四情景最高值", "无指标超过四情景最高值"],
        default="待核查",
    )
    df["措施结论"] = np.select(
        [
            df["zero_yield_warning"],
            df["hard_risk_count"].gt(0),
            df["mostly_preventive_warning"],
            df["process_clean_hard"],
        ],
        ["硬风险：零产量", "存在硬风险", "无硬风险但偏预防性", "无硬风险"],
        default="待核查",
    )
    return df.sort_values(["station_code", "year"]).reset_index(drop=True)


def build_station_table(years: pd.DataFrame) -> pd.DataFrame:
    agg = years.groupby("station_code", as_index=False).agg(
        years=("year", "nunique"),
        selected_checkpoint=("checkpoint_step", "first"),
        any_metric_win_years=("any_metric_win_four_max", "sum"),
        yield_win_years=("yield_win_four_max", "sum"),
        wp_et_win_years=("wp_et_win_four_max", "sum"),
        pfp_n_win_years=("pfp_n_win_four_max", "sum"),
        two_or_more_metric_win_years=("win_metric_count", lambda x: int((x >= 2).sum())),
        mean_yield_gap_vs_four_max=("yield_gap_vs_four_max", "mean"),
        mean_wp_et_gap_vs_four_max=("wp_et_gap_vs_four_max", "mean"),
        mean_pfp_n_gap_vs_four_max=("pfp_n_gap_vs_four_max", "mean"),
        mean_yield_gap_vs_expert=("yield_gap_vs_expert", "mean"),
        mean_water_saving_vs_expert_mm=("water_saving_vs_expert_mm", "mean"),
        std_water_saving_vs_expert_mm=("water_saving_vs_expert_mm", "std"),
        mean_n_saving_vs_expert_kg_ha=("n_saving_vs_expert_kg_ha", "mean"),
        std_n_saving_vs_expert_kg_ha=("n_saving_vs_expert_kg_ha", "std"),
        mean_total_irrigation=("total_irrigation", "mean"),
        mean_total_n=("total_n", "mean"),
        mean_irrigation_event_count=("irrigation_event_count", "mean"),
        mean_n_event_count=("n_event_count", "mean"),
        hard_risk_years=("hard_risk_count", lambda x: int((x > 0).sum())),
        clean_hard_years=("process_clean_hard", "sum"),
        mostly_preventive_years=("mostly_preventive_warning", "sum"),
        max_swfac=("max_swfac", "max"),
        max_nstres=("max_nstres", "max"),
    )
    agg["any_metric_win_rate"] = agg["any_metric_win_years"] / agg["years"]
    agg["hard_risk_rate"] = agg["hard_risk_years"] / agg["years"]
    return agg


def savefig(fig: plt.Figure, stem: str) -> list[str]:
    out = []
    for suffix in ["png", "svg"]:
        p = FIG_DIR / f"{stem}.{suffix}"
        fig.savefig(p, dpi=220, bbox_inches="tight")
        out.append(str(p.relative_to(ROOT)))
    plt.close(fig)
    return out


def plot_metric_counts(station: pd.DataFrame) -> list[str]:
    s = station.sort_values("station_code")
    x = np.arange(len(s))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - 1.5 * width, s["any_metric_win_years"], width, label="any metric")
    ax.bar(x - 0.5 * width, s["yield_win_years"], width, label="yield")
    ax.bar(x + 0.5 * width, s["wp_et_win_years"], width, label="WP_ET")
    ax.bar(x + 1.5 * width, s["pfp_n_win_years"], width, label="PFP_N")
    ax.set_xticks(x)
    ax.set_xticklabels(s["station_code"])
    ax.set_ylim(0, max(10, int(s["years"].max())) + 1)
    ax.set_ylabel("years out of station validation years")
    ax.set_title("036 selected PPO: metric wins vs four-scenario maximum")
    ax.legend(ncol=4, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    return savefig(fig, "037_00_station_metric_win_counts")


def plot_management_flags(station: pd.DataFrame) -> list[str]:
    s = station.sort_values("station_code")
    x = np.arange(len(s))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - width, s["hard_risk_years"], width, label="hard risk years")
    ax.bar(x, s["mostly_preventive_years"], width, label="preventive years")
    ax.bar(x + width, s["clean_hard_years"], width, label="hard-risk clean years")
    ax.set_xticks(x)
    ax.set_xticklabels(s["station_code"])
    ax.set_ylim(0, max(10, int(s["years"].max())) + 1)
    ax.set_ylabel("years")
    ax.set_title("036 selected PPO: management-process audit")
    ax.legend(ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    return savefig(fig, "037_00_station_management_audit_counts")


def plot_gap_savings(station: pd.DataFrame) -> list[str]:
    s = station.sort_values("station_code")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    axes[0].bar(s["station_code"], s["mean_yield_gap_vs_four_max"])
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_title("Mean yield gap vs four-scenario max")
    axes[0].set_ylabel("kg/ha")

    axes[1].bar(s["station_code"], s["mean_water_saving_vs_expert_mm"], yerr=s["std_water_saving_vs_expert_mm"].fillna(0))
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title("Water saving vs expert")
    axes[1].set_ylabel("mm")

    axes[2].bar(s["station_code"], s["mean_n_saving_vs_expert_kg_ha"], yerr=s["std_n_saving_vs_expert_kg_ha"].fillna(0))
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_title("Nitrogen saving vs expert")
    axes[2].set_ylabel("kg/ha")

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("036 selected PPO: average performance and resource savings")
    return savefig(fig, "037_00_station_mean_gap_and_savings")


def write_record(years: pd.DataFrame, station: pd.DataFrame, ckpts: pd.DataFrame, figures: list[str]) -> None:
    overall = {
        "station_count": int(station["station_code"].nunique()),
        "year_count": int(len(years)),
        "any_metric_win_years": int(years["any_metric_win_four_max"].sum()),
        "yield_win_years": int(years["yield_win_four_max"].sum()),
        "wp_et_win_years": int(years["wp_et_win_four_max"].sum()),
        "pfp_n_win_years": int(years["pfp_n_win_four_max"].sum()),
        "two_or_more_metric_win_years": int((years["win_metric_count"] >= 2).sum()),
        "hard_risk_years": int((years["hard_risk_count"] > 0).sum()),
        "hard_risk_clean_years": int(years["process_clean_hard"].sum()),
        "mostly_preventive_years": int(years["mostly_preventive_warning"].sum()),
    }
    overall_df = pd.DataFrame([overall])
    overall_df.to_csv(TABLE_DIR / "037_00_overall_summary.csv", index=False, encoding="utf-8-sig")

    top_problem = years.sort_values(["hard_risk_count", "win_metric_count"], ascending=[False, True]).head(12)
    top_success = years.sort_values(["win_metric_count", "yield_gap_vs_four_max"], ascending=[False, False]).head(12)

    lines: list[str] = []
    lines.append("# 037_00：036 主线结果指标表现与措施合理性总审计记录\n")
    lines.append("## 任务边界\n")
    lines.append("- 本任务不训练模型、不重跑 DSSAT、不修改 reward 或动作约束。\n")
    lines.append("- 输入为 036_04 指标对照表和 036_05 措施合理性审计表。\n")
    lines.append("- 本任务目的：把 036 主线结果整理成“指标表现 + 措施合理性”的可汇报总览。\n")
    lines.append("\n## 036 主线配置回顾\n")
    lines.append("- 数据源：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013`。\n")
    lines.append("- 输入与执行修复：`IC=1`，动态管理为 `IRRIG=L, FERTI=L`。\n")
    lines.append("- 算法：自由时序 `MaskablePPO`。\n")
    lines.append("- 灌溉档位：`{0, 15, 30, 45}` mm；施氮档位：`{0, 40, 80, 120}` kg/ha。\n")
    lines.append("- 约束：灌溉/施氮最小间隔 7 天、单季水氮上限、DAP90 后禁氮。\n")
    lines.append("- checkpoint：25K/50K/75K/100K；每站点按 036_04 预定义规则选一个 checkpoint。\n")
    lines.append("\n## 总体摘要\n")
    lines.append(overall_df.to_markdown(index=False))
    lines.append("\n\n## 每站点指标与措施总览\n")
    display_cols = [
        "station_code",
        "selected_checkpoint",
        "years",
        "any_metric_win_years",
        "yield_win_years",
        "wp_et_win_years",
        "pfp_n_win_years",
        "mean_yield_gap_vs_four_max",
        "mean_water_saving_vs_expert_mm",
        "mean_n_saving_vs_expert_kg_ha",
        "hard_risk_years",
        "mostly_preventive_years",
    ]
    lines.append(station[display_cols].round(3).to_markdown(index=False))
    lines.append("\n\n## 代表性成功年份（按胜出指标数和产量 gap 排序，最多 12 行）\n")
    success_cols = [
        "station_code",
        "year",
        "checkpoint_step",
        "final_grnwt",
        "total_irrigation",
        "total_n",
        "WP_ET_kg_m3",
        "PFP_N_kg_kg",
        "yield_gap_vs_four_max",
        "wp_et_gap_vs_four_max",
        "pfp_n_gap_vs_four_max",
        "win_metric_count",
        "措施结论",
    ]
    lines.append(top_success[success_cols].round(3).to_markdown(index=False))
    lines.append("\n\n## 需要重点解释或复核的年份（按硬风险数量排序，最多 12 行）\n")
    problem_cols = [
        "station_code",
        "year",
        "checkpoint_step",
        "final_grnwt",
        "total_irrigation",
        "total_n",
        "irrigation_event_count",
        "n_event_count",
        "first_irrigation_dap",
        "first_n_dap",
        "hard_risk_count",
        "mostly_preventive_warning",
        "management_risk_flags",
    ]
    lines.append(top_problem[problem_cols].round(3).to_markdown(index=False))
    lines.append("\n\n## 解释边界\n")
    lines.append("- `any_metric_win_four_max` 表示至少一个指标超过四情景最高值，不表示三个指标全部同时超过。\n")
    lines.append("- `mostly_preventive_warning` 不直接等于错误；它表示操作前 3 天未出现明显胁迫，需要用过程图或反事实试验解释。\n")
    lines.append("- 硬风险包括早期集中投入、间隔违规、DAP90 后施氮、零产量；这些是下一轮改进或汇报时优先说明的问题。\n")
    lines.append("- 与 expert 的节水节氮用于说明资源投入变化；与四情景最高值的 gap 用于说明是否真正超过所有对照。\n")
    lines.append("\n## 输出图件\n")
    for f in figures:
        lines.append(f"- `{f}`\n")
    lines.append("\n## 输出表格\n")
    for f in [
        "benchmark_results/037_00_036_results_metric_and_management_audit/tables/037_00_year_level_metric_management_overview.csv",
        "benchmark_results/037_00_036_results_metric_and_management_audit/tables/037_00_station_level_metric_management_overview.csv",
        "benchmark_results/037_00_036_results_metric_and_management_audit/tables/037_00_overall_summary.csv",
    ]:
        lines.append(f"- `{f}`\n")
    # Use utf-8-sig so Windows editors can open the Chinese record without mojibake.
    DOC.write_text("".join(lines), encoding="utf-8-sig")


def main() -> None:
    ensure_dirs()
    selected, ckpts, mgmt = load_inputs()
    years = build_year_table(selected, mgmt)
    station = build_station_table(years)

    years.to_csv(TABLE_DIR / "037_00_year_level_metric_management_overview.csv", index=False, encoding="utf-8-sig")
    station.to_csv(TABLE_DIR / "037_00_station_level_metric_management_overview.csv", index=False, encoding="utf-8-sig")

    figures: list[str] = []
    figures.extend(plot_metric_counts(station))
    figures.extend(plot_management_flags(station))
    figures.extend(plot_gap_savings(station))

    write_record(years, station, ckpts, figures)
    print(f"Wrote {TABLE_DIR}")
    print(f"Wrote {FIG_DIR}")
    print(f"Wrote {DOC}")


if __name__ == "__main__":
    main()
