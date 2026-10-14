from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK = "036_05_selected_ppo_management_rationality_audit"
OUT = ROOT / "benchmark_results" / TASK
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
DOC = ROOT / "docs" / f"{TASK}_record.md"

SELECTED = ROOT / "benchmark_results" / "036_04_select_checkpoint_and_plot_03601_03603_summary" / "tables" / "036_04_selected_year_level_comparison.csv"

MIN_INTERVAL_DAYS = 7
EARLY_DAP_CUTOFF = 10
EARLY_SHARE_WARN = 0.80
STRESS_THRESHOLD = 0.05
NO_STRESS_THRESHOLD = 0.001
LATE_N_DAP = 90


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def load_selected() -> pd.DataFrame:
    df = pd.read_csv(SELECTED)
    required = {"station_code", "year", "checkpoint_step", "daily_csv_path", "final_grnwt", "total_irrigation", "total_n"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Selected table missing required columns: {missing}")
    return df


def event_pre_stress(daily: pd.DataFrame, dap: float, stress_col: str) -> tuple[float, float]:
    before = daily.loc[(daily["dap"] >= dap - 3) & (daily["dap"] <= dap), stress_col]
    if before.empty:
        return np.nan, np.nan
    return float(before.max()), float(before.mean())


def audit_one(row: pd.Series) -> tuple[list[dict], dict]:
    path = ROOT / str(row["daily_csv_path"])
    daily = pd.read_csv(path)
    daily["dap"] = pd.to_numeric(daily["dap"], errors="coerce")
    for col in ["safe_action_amir", "safe_action_anfer", "swfac", "nstres", "rain", "reward_stress_aware"]:
        if col in daily.columns:
            daily[col] = pd.to_numeric(daily[col], errors="coerce").fillna(0.0)

    events: list[dict] = []
    for _, drow in daily.iterrows():
        dap = float(drow["dap"])
        i_amt = float(drow.get("safe_action_amir", 0.0))
        n_amt = float(drow.get("safe_action_anfer", 0.0))
        if i_amt > 0:
            mx, mean = event_pre_stress(daily, dap, "swfac")
            events.append(
                {
                    "station_code": row["station_code"],
                    "year": int(row["year"]),
                    "checkpoint_step": int(row["checkpoint_step"]),
                    "resource": "irrigation",
                    "dap": dap,
                    "amount": i_amt,
                    "pre3d_max_stress": mx,
                    "pre3d_mean_stress": mean,
                    "stress_responsive": bool(mx >= STRESS_THRESHOLD),
                    "preventive_no_stress": bool(mx <= NO_STRESS_THRESHOLD),
                }
            )
        if n_amt > 0:
            mx, mean = event_pre_stress(daily, dap, "nstres")
            events.append(
                {
                    "station_code": row["station_code"],
                    "year": int(row["year"]),
                    "checkpoint_step": int(row["checkpoint_step"]),
                    "resource": "nitrogen",
                    "dap": dap,
                    "amount": n_amt,
                    "pre3d_max_stress": mx,
                    "pre3d_mean_stress": mean,
                    "stress_responsive": bool(mx >= STRESS_THRESHOLD),
                    "preventive_no_stress": bool(mx <= NO_STRESS_THRESHOLD),
                }
            )

    ev = pd.DataFrame(events)
    total_i = float(row["total_irrigation"])
    total_n = float(row["total_n"])
    early_i = float(daily.loc[daily["dap"].le(EARLY_DAP_CUTOFF), "safe_action_amir"].sum())
    early_n = float(daily.loc[daily["dap"].le(EARLY_DAP_CUTOFF), "safe_action_anfer"].sum())

    def min_gap(resource: str) -> float:
        if ev.empty:
            return np.nan
        daps = sorted(ev.loc[ev["resource"].eq(resource), "dap"].tolist())
        if len(daps) < 2:
            return np.nan
        return float(np.min(np.diff(daps)))

    def count_interval_viol(resource: str) -> int:
        if ev.empty:
            return 0
        daps = sorted(ev.loc[ev["resource"].eq(resource), "dap"].tolist())
        if len(daps) < 2:
            return 0
        return int((np.diff(daps) < MIN_INTERVAL_DAYS).sum())

    irr_events = ev.loc[ev["resource"].eq("irrigation")] if not ev.empty else pd.DataFrame()
    n_events = ev.loc[ev["resource"].eq("nitrogen")] if not ev.empty else pd.DataFrame()
    n_late = n_events.loc[n_events["dap"].gt(LATE_N_DAP)] if not n_events.empty else pd.DataFrame()

    year_audit = {
        "station_code": row["station_code"],
        "year": int(row["year"]),
        "checkpoint_step": int(row["checkpoint_step"]),
        "final_grnwt": float(row["final_grnwt"]),
        "total_irrigation": total_i,
        "total_n": total_n,
        "WP_ET_kg_m3": float(row.get("WP_ET_kg_m3", np.nan)),
        "PFP_N_kg_kg": float(row.get("PFP_N_kg_kg", np.nan)),
        "any_metric_win_four_max": bool(row.get("any_metric_win_four_max", False)),
        "yield_gap_vs_four_max": float(row.get("yield_gap_vs_four_max", np.nan)),
        "wp_et_gap_vs_four_max": float(row.get("wp_et_gap_vs_four_max", np.nan)),
        "pfp_n_gap_vs_four_max": float(row.get("pfp_n_gap_vs_four_max", np.nan)),
        "irrigation_event_count": int(len(irr_events)),
        "n_event_count": int(len(n_events)),
        "first_irrigation_dap": float(irr_events["dap"].min()) if len(irr_events) else np.nan,
        "first_n_dap": float(n_events["dap"].min()) if len(n_events) else np.nan,
        "early_irrigation_dap1_10": early_i,
        "early_n_dap1_10": early_n,
        "early_irrigation_share": early_i / total_i if total_i > 0 else np.nan,
        "early_n_share": early_n / total_n if total_n > 0 else np.nan,
        "min_irrigation_gap_days": min_gap("irrigation"),
        "min_n_gap_days": min_gap("nitrogen"),
        "irrigation_interval_violations_lt7d": count_interval_viol("irrigation"),
        "n_interval_violations_lt7d": count_interval_viol("nitrogen"),
        "late_n_after_dap90_count": int(len(n_late)),
        "late_n_after_dap90_total": float(n_late["amount"].sum()) if len(n_late) else 0.0,
        "max_swfac": float(pd.to_numeric(daily["swfac"], errors="coerce").max()),
        "max_nstres": float(pd.to_numeric(daily["nstres"], errors="coerce").max()),
        "swfac_days_gt_0p05": int((pd.to_numeric(daily["swfac"], errors="coerce") > STRESS_THRESHOLD).sum()),
        "nstres_days_gt_0p05": int((pd.to_numeric(daily["nstres"], errors="coerce") > STRESS_THRESHOLD).sum()),
        "preventive_irrigation_events_no_pre_stress": int(irr_events["preventive_no_stress"].sum()) if len(irr_events) else 0,
        "preventive_n_events_no_pre_stress": int(n_events["preventive_no_stress"].sum()) if len(n_events) else 0,
        "stress_responsive_irrigation_events": int(irr_events["stress_responsive"].sum()) if len(irr_events) else 0,
        "stress_responsive_n_events": int(n_events["stress_responsive"].sum()) if len(n_events) else 0,
    }

    year_audit["early_dump_warning"] = bool(
        (total_i > 0 and year_audit["early_irrigation_share"] >= EARLY_SHARE_WARN)
        or (total_n > 0 and year_audit["early_n_share"] >= EARLY_SHARE_WARN)
    )
    year_audit["interval_warning"] = bool(
        year_audit["irrigation_interval_violations_lt7d"] > 0 or year_audit["n_interval_violations_lt7d"] > 0
    )
    year_audit["late_n_warning"] = bool(year_audit["late_n_after_dap90_count"] > 0)
    year_audit["zero_yield_warning"] = bool(year_audit["final_grnwt"] <= 0)
    year_audit["mostly_preventive_warning"] = bool(
        (
            year_audit["irrigation_event_count"] > 0
            and year_audit["preventive_irrigation_events_no_pre_stress"] == year_audit["irrigation_event_count"]
        )
        or (
            year_audit["n_event_count"] > 0
            and year_audit["preventive_n_events_no_pre_stress"] == year_audit["n_event_count"]
        )
    )
    warnings = [
        k
        for k in [
            "early_dump_warning",
            "interval_warning",
            "late_n_warning",
            "zero_yield_warning",
            "mostly_preventive_warning",
        ]
        if year_audit[k]
    ]
    year_audit["management_risk_flags"] = ";".join(warnings)
    year_audit["management_risk_flag_count"] = len(warnings)
    return events, year_audit


def build_audits(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    event_rows: list[dict] = []
    year_rows: list[dict] = []
    for _, row in selected.iterrows():
        events, year_audit = audit_one(row)
        event_rows.extend(events)
        year_rows.append(year_audit)
    events = pd.DataFrame(event_rows)
    years = pd.DataFrame(year_rows)
    station = (
        years.groupby("station_code", as_index=False)
        .agg(
            years=("year", "nunique"),
            any_metric_win_years=("any_metric_win_four_max", "sum"),
            mean_total_irrigation=("total_irrigation", "mean"),
            mean_total_n=("total_n", "mean"),
            mean_irrigation_event_count=("irrigation_event_count", "mean"),
            mean_n_event_count=("n_event_count", "mean"),
            mean_first_irrigation_dap=("first_irrigation_dap", "mean"),
            mean_first_n_dap=("first_n_dap", "mean"),
            early_dump_warning_years=("early_dump_warning", "sum"),
            interval_warning_years=("interval_warning", "sum"),
            late_n_warning_years=("late_n_warning", "sum"),
            zero_yield_warning_years=("zero_yield_warning", "sum"),
            mostly_preventive_warning_years=("mostly_preventive_warning", "sum"),
            mean_management_risk_flag_count=("management_risk_flag_count", "mean"),
            max_swfac=("max_swfac", "max"),
            max_nstres=("max_nstres", "max"),
        )
    )
    return events, years, station


def savefig(fig: plt.Figure, stem: str) -> list[str]:
    paths = []
    for suffix in ["png", "svg"]:
        path = FIG_DIR / f"{stem}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return paths


def plot_event_timeline(events: pd.DataFrame) -> list[str]:
    if events.empty:
        return []
    data = events.copy()
    data["site_year"] = data["station_code"].astype(str) + data["year"].astype(str)
    order = (
        data[["station_code", "year", "site_year"]]
        .drop_duplicates()
        .sort_values(["station_code", "year"])
        ["site_year"]
        .tolist()
    )
    ymap = {label: i for i, label in enumerate(order)}
    fig, ax = plt.subplots(figsize=(13, max(8, 0.22 * len(order))), constrained_layout=True)
    colors = {"irrigation": "#4C78A8", "nitrogen": "#F58518"}
    markers = {"irrigation": "o", "nitrogen": "^"}
    for resource, sub in data.groupby("resource"):
        ax.scatter(
            sub["dap"],
            sub["site_year"].map(ymap),
            s=np.clip(sub["amount"] * 2.5, 25, 260),
            color=colors[resource],
            marker=markers[resource],
            edgecolor="#333333",
            linewidth=0.4,
            alpha=0.82,
            label=resource,
        )
    ax.axvspan(0, EARLY_DAP_CUTOFF, color="#DDDDDD", alpha=0.35, label="DAP1-10 early window")
    ax.axvline(LATE_N_DAP, color="#333333", linestyle="--", linewidth=0.8, label="DAP90")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=7)
    ax.set_xlabel("DAP")
    ax.set_title("036_05 PPO selected checkpoint event timeline by station-year")
    ax.grid(axis="x", alpha=0.22)
    ax.legend(loc="upper right", fontsize=8)
    return savefig(fig, "036_05_ppo_event_timeline_by_station_year")


def plot_risk_counts(station: pd.DataFrame) -> list[str]:
    cols = [
        "early_dump_warning_years",
        "mostly_preventive_warning_years",
        "interval_warning_years",
        "late_n_warning_years",
        "zero_yield_warning_years",
    ]
    labels = ["Early concentrated", "Preventive/no pre-stress", "<7d interval", "N after DAP90", "Zero yield"]
    data = station.sort_values("station_code")
    x = np.arange(len(data))
    bottom = np.zeros(len(data))
    fig, ax = plt.subplots(figsize=(11, 5.6), constrained_layout=True)
    palette = ["#B279A2", "#72B7B2", "#F58518", "#E45756", "#4C78A8"]
    for col, label, color in zip(cols, labels, palette):
        vals = data[col].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, label=label, color=color, edgecolor="#333333", linewidth=0.35)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(data["station_code"])
    ax.set_ylabel("Number of validation years flagged")
    ax.set_title("036_05 management rationality warning counts by station")
    ax.set_ylim(0, max(10, bottom.max() * 1.15))
    ax.legend(ncol=2, fontsize=8)
    ax.grid(axis="y", alpha=0.22)
    return savefig(fig, "036_05_management_risk_counts_by_station")


def write_record(events: pd.DataFrame, years: pd.DataFrame, station: pd.DataFrame, figures: list[str]) -> None:
    def md_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_None._"
        return df.to_markdown(index=False)

    high_risk = years.loc[years["management_risk_flag_count"].gt(0)].copy()
    representative = years.sort_values(["management_risk_flag_count", "station_code", "year"], ascending=[False, True, True]).head(20)
    lines = [
        "# 036_05 PPO措施合理性审计记录",
        "",
        "## 任务边界",
        "",
        "- 不训练。",
        "- 不修改reward、模型、动作空间或DSSAT输入。",
        "- 只审计036_04选中的每站点代表checkpoint对应的50个验证年。",
        "",
        "## 审计规则",
        "",
        f"- 早期集中投入：DAP1-{EARLY_DAP_CUTOFF} 使用水或氮占季节总量 >= {EARLY_SHARE_WARN:.0%}。",
        f"- 胁迫响应：操作前3天最大 `swfac/nstres` >= {STRESS_THRESHOLD} 记为响应性操作。",
        f"- 预防性/无胁迫操作：操作前3天最大 `swfac/nstres` <= {NO_STRESS_THRESHOLD}。",
        f"- 最小间隔检查：同类水/氮操作间隔 < {MIN_INTERVAL_DAYS} 天记为违反。",
        f"- 后期施氮：DAP > {LATE_N_DAP} 仍施氮记为风险。",
        "",
        "## 站点级审计汇总",
        "",
        md_table(station.round(4)),
        "",
        "## 风险年份样例（最多20行）",
        "",
        md_table(
            representative[
                [
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
                    "early_irrigation_share",
                    "early_n_share",
                    "max_swfac",
                    "max_nstres",
                    "any_metric_win_four_max",
                    "management_risk_flags",
                ]
            ].round(4)
        ),
        "",
        "## 初步结论",
        "",
    ]
    total_years = len(years)
    early_years = int(years["early_dump_warning"].sum())
    preventive_years = int(years["mostly_preventive_warning"].sum())
    zero_years = int(years["zero_yield_warning"].sum())
    interval_years = int(years["interval_warning"].sum())
    late_n_years = int(years["late_n_warning"].sum())
    lines.extend(
        [
            f"- 在 {total_years} 个验证年中，早期集中投入风险年份为 {early_years} 个。",
            f"- 操作多为无前置胁迫/预防性投入的年份为 {preventive_years} 个。",
            f"- 同类操作间隔小于7天的年份为 {interval_years} 个。",
            f"- DAP90后施氮年份为 {late_n_years} 个。",
            f"- 零产量年份为 {zero_years} 个。",
            "- 因此，036_04 的“指标候选成功”不能直接等价为“措施过程合理”。需要将过程合理性作为单独结论汇报。",
            "",
            "## 输出图件",
            "",
        ]
    )
    lines.extend([f"- `{p}`" for p in figures])
    lines.extend(
        [
            "",
            "## 输出表格",
            "",
            f"- `{(TABLE_DIR / '036_05_event_level_audit.csv').relative_to(ROOT)}`",
            f"- `{(TABLE_DIR / '036_05_year_level_management_audit.csv').relative_to(ROOT)}`",
            f"- `{(TABLE_DIR / '036_05_station_level_management_audit.csv').relative_to(ROOT)}`",
        ]
    )
    DOC.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    selected = load_selected()
    events, years, station = build_audits(selected)
    events.to_csv(TABLE_DIR / "036_05_event_level_audit.csv", index=False, encoding="utf-8-sig")
    years.to_csv(TABLE_DIR / "036_05_year_level_management_audit.csv", index=False, encoding="utf-8-sig")
    station.to_csv(TABLE_DIR / "036_05_station_level_management_audit.csv", index=False, encoding="utf-8-sig")
    figures: list[str] = []
    figures.extend(plot_event_timeline(events))
    figures.extend(plot_risk_counts(station))
    write_record(events, years, station, figures)
    print(
        json.dumps(
            {
                "task": TASK,
                "event_rows": int(len(events)),
                "year_rows": int(len(years)),
                "station_rows": int(len(station)),
                "early_dump_warning_years": int(years["early_dump_warning"].sum()),
                "mostly_preventive_warning_years": int(years["mostly_preventive_warning"].sum()),
                "interval_warning_years": int(years["interval_warning"].sum()),
                "late_n_warning_years": int(years["late_n_warning"].sum()),
                "zero_yield_warning_years": int(years["zero_yield_warning"].sum()),
                "figures": figures,
                "record_md": str(DOC.relative_to(ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
