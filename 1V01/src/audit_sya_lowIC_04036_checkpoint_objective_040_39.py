from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK = "040_39_sya_lowIC_04036_checkpoint_objective_audit"
OUT = ROOT / "benchmark_results" / TASK
FIG_DIR = OUT / "figures"
TAB_DIR = OUT / "tables"
DOC = ROOT / "docs" / f"{TASK}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK}.md"

PPO_SUMMARY = (
    ROOT
    / "benchmark_results"
    / "040_36_sya_lowIC_ppo_late_irrigation_reserve_mask"
    / "evaluation"
    / "040_36_checkpoint_validation_summary.csv"
)
BASELINE_SUMMARY = (
    ROOT
    / "benchmark_results"
    / "040_21_sya_lowIC_four_baseline_rebuild"
    / "evaluation"
    / "040_21_baseline_summary.csv"
)
CKPT100_GAPS = (
    ROOT
    / "benchmark_results"
    / "040_38_sya_lowIC_04036_ckpt100k_validation_five_scenario_metric_bars"
    / "tables"
    / "040_38_sya_lowIC_04036_ckpt100k_metric_gaps_vs_four_max.csv"
)

STATION = "SYA"
SITE = "SY"
YEARS = list(range(2014, 2024))
FOUR_SCENARIOS = [
    "null",
    "recorded_farmer_template",
    "dssat_auto",
    "official_extension_expert",
]


def ensure_dirs() -> None:
    for path in [OUT, FIG_DIR, TAB_DIR, OUT / "configs"]:
        path.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    if PROMPT.exists():
        shutil.copy2(PROMPT, OUT / "configs" / PROMPT.name)


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "无记录。"
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number", "bool"]).columns:
        if work[col].dtype == bool:
            continue
        work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
    work = work.astype(object).where(pd.notna(work), "")
    lines = [
        "| " + " | ".join(map(str, work.columns)) + " |",
        "| " + " | ".join(["---"] * len(work.columns)) + " |",
    ]
    for row in work.to_numpy().tolist():
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def load_ppo() -> pd.DataFrame:
    df = pd.read_csv(PPO_SUMMARY, keep_default_na=False)
    df = df[
        df["station_code"].astype(str).eq(STATION)
        & pd.to_numeric(df["year"], errors="coerce").isin(YEARS)
        & df["run_status"].astype(str).eq("ok")
    ].copy()
    numeric_cols = [
        "year",
        "checkpoint_step",
        "final_grnwt",
        "total_irrigation",
        "total_n",
        "PFP_N",
        "profit_simple",
        "max_swfac",
        "swfac_days_gt_0p05",
        "max_nstres",
        "nstres_days_gt_0p05",
        "irrigation_event_count",
        "n_event_count",
        "irrigation_dap1_30",
        "irrigation_dap31_60",
        "irrigation_dap61_90",
        "irrigation_dap91_plus",
        "late_reserve_violation",
        "has_forbidden_15mm_irrigation",
        "has_forbidden_40kg_n",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_baseline_four_max() -> pd.DataFrame:
    b = pd.read_csv(BASELINE_SUMMARY, keep_default_na=False)
    b = b[
        b["station_code"].astype(str).eq(STATION)
        & pd.to_numeric(b["year"], errors="coerce").isin(YEARS)
        & b["scenario"].astype(str).isin(FOUR_SCENARIOS)
    ].copy()
    for col in [
        "year",
        "grain_yield_kg_ha",
        "actual_irrigation_mm",
        "actual_nitrogen_kg_ha",
        "WP_ET_kg_m3",
        "PFP_N_kg_kg",
        "max_water_stress",
        "max_nitrogen_stress",
    ]:
        if col in b.columns:
            b[col] = pd.to_numeric(b[col], errors="coerce")

    records: list[dict[str, object]] = []
    for year, g in b.groupby("year"):
        yrow = g.loc[g["grain_yield_kg_ha"].idxmax()]
        wp_g = g.dropna(subset=["WP_ET_kg_m3"])
        wp_row = wp_g.loc[wp_g["WP_ET_kg_m3"].idxmax()] if len(wp_g) else None
        pfp_g = g.dropna(subset=["PFP_N_kg_kg"])
        pfp_row = pfp_g.loc[pfp_g["PFP_N_kg_kg"].idxmax()] if len(pfp_g) else None
        expert = g[g["scenario"].eq("official_extension_expert")]
        expert_row = expert.iloc[0] if len(expert) else None
        records.append(
            {
                "year": int(year),
                "four_max_yield": float(yrow["grain_yield_kg_ha"]),
                "four_max_yield_scenario": yrow["scenario"],
                "four_max_wp_et": float(wp_row["WP_ET_kg_m3"]) if wp_row is not None else np.nan,
                "four_max_wp_et_scenario": wp_row["scenario"] if wp_row is not None else "",
                "four_max_pfp_n": float(pfp_row["PFP_N_kg_kg"]) if pfp_row is not None else np.nan,
                "four_max_pfp_n_scenario": pfp_row["scenario"] if pfp_row is not None else "",
                "expert_yield": float(expert_row["grain_yield_kg_ha"]) if expert_row is not None else np.nan,
                "expert_irrigation": float(expert_row["actual_irrigation_mm"]) if expert_row is not None else np.nan,
                "expert_n": float(expert_row["actual_nitrogen_kg_ha"]) if expert_row is not None else np.nan,
                "expert_pfp_n": float(expert_row["PFP_N_kg_kg"]) if expert_row is not None else np.nan,
            }
        )
    return pd.DataFrame(records)


def add_ckpt100_wp_et_if_available(year_rows: pd.DataFrame) -> pd.DataFrame:
    year_rows = year_rows.copy()
    year_rows["ppo_wp_et_exact"] = np.nan
    year_rows["gap_wp_et_vs_four_max_exact"] = np.nan
    year_rows["win_wp_et_four_exact"] = False
    if not CKPT100_GAPS.exists():
        return year_rows
    g = pd.read_csv(CKPT100_GAPS, keep_default_na=False)
    for col in ["year", "ppo_wp_et", "gap_wp_et_vs_four_max", "win_wp_et_vs_four_max"]:
        if col in g.columns:
            if col != "win_wp_et_four":
                g[col] = pd.to_numeric(g[col], errors="coerce")
    g = g[["year", "ppo_wp_et", "gap_wp_et_vs_four_max", "win_wp_et_vs_four_max"]].copy()
    g = g.rename(
        columns={
            "ppo_wp_et": "ppo_wp_et_exact",
            "gap_wp_et_vs_four_max": "gap_wp_et_vs_four_max_exact",
            "win_wp_et_vs_four_max": "win_wp_et_four_exact",
        }
    )
    mask = year_rows["checkpoint_step"].eq(100000)
    merged_100 = year_rows[mask].drop(
        columns=["ppo_wp_et_exact", "gap_wp_et_vs_four_max_exact", "win_wp_et_four_exact"]
    ).merge(g, on="year", how="left")
    return pd.concat([year_rows[~mask], merged_100], ignore_index=True).sort_values(
        ["checkpoint_step", "year"]
    )


def build_year_rows() -> pd.DataFrame:
    ppo = load_ppo()
    base = load_baseline_four_max()
    rows = ppo.merge(base, on="year", how="left")
    rows["gap_yield_vs_four_max"] = rows["final_grnwt"] - rows["four_max_yield"]
    rows["gap_yield_vs_expert"] = rows["final_grnwt"] - rows["expert_yield"]
    rows["win_yield_four"] = rows["gap_yield_vs_four_max"] >= -1e-9
    rows["gap_pfp_n_vs_four_max"] = rows["PFP_N"] - rows["four_max_pfp_n"]
    rows["win_pfp_n_four"] = rows["gap_pfp_n_vs_four_max"] >= -1e-9
    rows["save_i_vs_expert"] = rows["expert_irrigation"] - rows["total_irrigation"]
    rows["save_n_vs_expert"] = rows["expert_n"] - rows["total_n"]
    rows["no_small_actions"] = (
        pd.to_numeric(rows["has_forbidden_15mm_irrigation"], errors="coerce").fillna(0).eq(0)
        & pd.to_numeric(rows["has_forbidden_40kg_n"], errors="coerce").fillna(0).eq(0)
    )
    rows["no_late_reserve_violation"] = (
        pd.to_numeric(rows["late_reserve_violation"], errors="coerce").fillna(0).eq(0)
    )
    rows = add_ckpt100_wp_et_if_available(rows)
    cols = [
        "checkpoint_step",
        "year",
        "final_grnwt",
        "four_max_yield",
        "four_max_yield_scenario",
        "gap_yield_vs_four_max",
        "gap_yield_vs_expert",
        "win_yield_four",
        "total_irrigation",
        "total_n",
        "PFP_N",
        "four_max_pfp_n",
        "gap_pfp_n_vs_four_max",
        "win_pfp_n_four",
        "ppo_wp_et_exact",
        "four_max_wp_et",
        "gap_wp_et_vs_four_max_exact",
        "win_wp_et_four_exact",
        "save_i_vs_expert",
        "save_n_vs_expert",
        "max_swfac",
        "swfac_days_gt_0p05",
        "max_nstres",
        "nstres_days_gt_0p05",
        "irrigation_event_count",
        "n_event_count",
        "irrigation_dap1_30",
        "irrigation_dap31_60",
        "irrigation_dap61_90",
        "irrigation_dap91_plus",
        "no_small_actions",
        "no_late_reserve_violation",
        "action_sequence",
    ]
    return rows[[c for c in cols if c in rows.columns]].sort_values(["checkpoint_step", "year"])


def summarize(year_rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    ckpt100 = year_rows[year_rows["checkpoint_step"].eq(100000)]
    ref_yield = float(ckpt100["final_grnwt"].mean()) if len(ckpt100) else np.nan
    ref_i = float(ckpt100["total_irrigation"].mean()) if len(ckpt100) else np.nan
    ref_n = float(ckpt100["total_n"].mean()) if len(ckpt100) else np.nan
    for ckpt, g in year_rows.groupby("checkpoint_step"):
        g = g.sort_values("year")
        records.append(
            {
                "checkpoint_step": int(ckpt),
                "years": int(len(g)),
                "mean_yield": float(g["final_grnwt"].mean()),
                "mean_gap_yield_vs_four_max": float(g["gap_yield_vs_four_max"].mean()),
                "min_gap_yield_vs_four_max": float(g["gap_yield_vs_four_max"].min()),
                "yield_win_years": int(g["win_yield_four"].sum()),
                "mean_irrigation": float(g["total_irrigation"].mean()),
                "mean_n": float(g["total_n"].mean()),
                "mean_pfp_n": float(g["PFP_N"].mean()),
                "pfp_n_win_years": int(g["win_pfp_n_four"].sum()),
                "mean_save_i_vs_expert": float(g["save_i_vs_expert"].mean()),
                "mean_save_n_vs_expert": float(g["save_n_vs_expert"].mean()),
                "mean_max_swfac": float(g["max_swfac"].mean()),
                "mean_swfac_days_gt_0p05": float(g["swfac_days_gt_0p05"].mean()),
                "mean_max_nstres": float(g["max_nstres"].mean()),
                "mean_nstres_days_gt_0p05": float(g["nstres_days_gt_0p05"].mean()),
                "mean_i_events": float(g["irrigation_event_count"].mean()),
                "mean_n_events": float(g["n_event_count"].mean()),
                "mean_i_dap1_30": float(g["irrigation_dap1_30"].mean()),
                "mean_i_dap31_60": float(g["irrigation_dap31_60"].mean()),
                "mean_i_dap61_90": float(g["irrigation_dap61_90"].mean()),
                "mean_i_dap91_plus": float(g["irrigation_dap91_plus"].mean()),
                "small_action_violation_years": int((~g["no_small_actions"]).sum()),
                "late_reserve_violation_years": int((~g["no_late_reserve_violation"]).sum()),
                "mean_wp_et_exact_if_available": float(g["ppo_wp_et_exact"].mean())
                if g["ppo_wp_et_exact"].notna().any()
                else np.nan,
                "wp_et_win_years_exact_if_available": int(g["win_wp_et_four_exact"].sum())
                if g["ppo_wp_et_exact"].notna().any()
                else np.nan,
                "delta_mean_yield_vs_100k": float(g["final_grnwt"].mean() - ref_yield),
                "delta_mean_i_vs_100k": float(g["total_irrigation"].mean() - ref_i),
                "delta_mean_n_vs_100k": float(g["total_n"].mean() - ref_n),
            }
        )
    out = pd.DataFrame(records).sort_values("checkpoint_step")

    # A small transparent score: prioritize yield gap, then PFP_N wins and resource savings,
    # penalize stress/violations. This is only a ranking aid, not a success criterion.
    out["audit_score_higher_is_better"] = (
        out["mean_gap_yield_vs_four_max"] / 1000.0
        + 0.10 * out["yield_win_years"]
        + 0.05 * out["pfp_n_win_years"]
        + out["mean_save_i_vs_expert"] / 200.0
        + out["mean_save_n_vs_expert"] / 300.0
        - out["mean_swfac_days_gt_0p05"] / 20.0
        - 2.0 * out["small_action_violation_years"]
        - 2.0 * out["late_reserve_violation_years"]
    )
    return out


def branch(summary: pd.DataFrame) -> tuple[str, str]:
    row100 = summary[summary["checkpoint_step"].eq(100000)]
    if row100.empty:
        return "B_need_reward_v3", "缺少 100K 参照，不能认为已有 checkpoint 足够。"
    r100 = row100.iloc[0]
    candidates = summary[
        summary["small_action_violation_years"].eq(0)
        & summary["late_reserve_violation_years"].eq(0)
        & (summary["mean_yield"] >= float(r100["mean_yield"]) - 1e-9)
        & (summary["mean_n"] <= float(r100["mean_n"]) + 1e-9)
        & (
            (summary["mean_irrigation"] <= float(r100["mean_irrigation"]) + 1e-9)
            | (
                (summary["mean_yield"] - float(r100["mean_yield"]))
                >= 10.0 * (summary["mean_irrigation"] - float(r100["mean_irrigation"]))
            )
        )
    ].copy()
    candidates = candidates[candidates["checkpoint_step"].ne(100000)]
    if not candidates.empty:
        best = candidates.sort_values("audit_score_higher_is_better", ascending=False).iloc[0]
        return (
            "A_existing_checkpoint_candidate",
            f"已有 checkpoint {int(best['checkpoint_step'])} 在预注册硬条件下优于/不劣于 100K，可另开任务补 WP_ET replay。",
        )
    return (
        "B_need_reward_v3",
        "040_36 已有 checkpoint 没有同时改善产量与资源效率；下一步应设计保产优先的 PPO reward v3，而不是仅换 checkpoint。",
    )


def plot_tradeoff(summary: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = summary["checkpoint_step"].astype(str)
    axes[0, 0].bar(x, summary["mean_yield"], color="#4C78A8")
    axes[0, 0].set_title("Mean yield")
    axes[0, 0].set_ylabel("kg/ha")
    axes[0, 1].bar(x, summary["mean_gap_yield_vs_four_max"], color="#F58518")
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].set_title("Mean yield gap vs four-scenario max")
    axes[0, 1].set_ylabel("kg/ha")
    width = 0.35
    idx = np.arange(len(summary))
    axes[1, 0].bar(idx - width / 2, summary["mean_irrigation"], width, label="Irrigation", color="#54A24B")
    axes[1, 0].bar(idx + width / 2, summary["mean_n"], width, label="Nitrogen", color="#B279A2")
    axes[1, 0].set_xticks(idx)
    axes[1, 0].set_xticklabels(x)
    axes[1, 0].set_title("Mean resource inputs")
    axes[1, 0].set_ylabel("mm or kg/ha")
    axes[1, 0].legend()
    axes[1, 1].bar(idx - width / 2, summary["yield_win_years"], width, label="Yield wins", color="#E45756")
    axes[1, 1].bar(idx + width / 2, summary["pfp_n_win_years"], width, label="PFP_N wins", color="#72B7B2")
    axes[1, 1].set_xticks(idx)
    axes[1, 1].set_xticklabels(x)
    axes[1, 1].set_ylim(0, 10)
    axes[1, 1].set_title("Years winning vs four-scenario max")
    axes[1, 1].legend()
    for ax in axes.ravel():
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("040_39 SYA lowIC 040_36 checkpoint tradeoff audit")
    fig.tight_layout()
    out = FIG_DIR / "040_39_checkpoint_tradeoff_summary.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_yearly_gap(year_rows: pd.DataFrame) -> Path:
    pivot = year_rows.pivot(index="year", columns="checkpoint_step", values="gap_yield_vs_four_max")
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(kind="bar", ax=ax, width=0.82)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Yearly yield gap vs four-scenario max by checkpoint")
    ax.set_ylabel("PPO yield - four-scenario max yield (kg/ha)")
    ax.set_xlabel("Validation year")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = FIG_DIR / "040_39_yearly_yield_gap_by_checkpoint.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def write_record(year_rows: pd.DataFrame, summary: pd.DataFrame, branch_name: str, reason: str, figs: list[Path]) -> None:
    with DOC.open("w", encoding="utf-8") as f:
        f.write(f"# {TASK} 实验记录\n\n")
        f.write("## 任务目的\n\n")
        f.write(
            "本任务只复查 040_36 已有 checkpoint，不训练、不改 reward、不重跑 DSSAT。"
            "目标是判断当前指标不足是否能通过 checkpoint 选择解决，还是需要进入 PPO reward v3。\n\n"
        )
        f.write("## 输入文件\n\n")
        f.write(f"- PPO checkpoint 验证摘要：`{PPO_SUMMARY.relative_to(ROOT).as_posix()}`\n")
        f.write(f"- 四情景基线：`{BASELINE_SUMMARY.relative_to(ROOT).as_posix()}`\n")
        if CKPT100_GAPS.exists():
            f.write(f"- 100K 精确 WP_ET 表：`{CKPT100_GAPS.relative_to(ROOT).as_posix()}`\n")
        f.write("\n## 判定结果\n\n")
        f.write(f"- 分支：`{branch_name}`\n")
        f.write(f"- 理由：{reason}\n\n")
        f.write("## checkpoint 汇总\n\n")
        display_cols = [
            "checkpoint_step",
            "mean_yield",
            "mean_gap_yield_vs_four_max",
            "min_gap_yield_vs_four_max",
            "yield_win_years",
            "mean_irrigation",
            "mean_n",
            "mean_pfp_n",
            "pfp_n_win_years",
            "mean_save_i_vs_expert",
            "mean_save_n_vs_expert",
            "mean_swfac_days_gt_0p05",
            "mean_i_dap91_plus",
            "small_action_violation_years",
            "late_reserve_violation_years",
            "audit_score_higher_is_better",
        ]
        f.write(md_table(summary[[c for c in display_cols if c in summary.columns]]) + "\n\n")
        f.write("## 逐年产量缺口和资源投入\n\n")
        year_cols = [
            "checkpoint_step",
            "year",
            "final_grnwt",
            "four_max_yield",
            "gap_yield_vs_four_max",
            "gap_yield_vs_expert",
            "total_irrigation",
            "total_n",
            "PFP_N",
            "gap_pfp_n_vs_four_max",
            "max_swfac",
            "swfac_days_gt_0p05",
            "irrigation_dap91_plus",
        ]
        f.write(md_table(year_rows[[c for c in year_cols if c in year_rows.columns]], max_rows=60) + "\n\n")
        f.write("## 图件\n\n")
        for fig in figs:
            f.write(f"- `{fig.relative_to(ROOT).as_posix()}`\n")
        f.write("\n## 边界说明\n\n")
        f.write(
            "- 25K/50K/75K checkpoint 未重跑 DSSAT 补 ETCP，因此本任务不报告这些 checkpoint 的精确 WP_ET。\n"
            "- 100K checkpoint 的 WP_ET 来自 040_38 的固定动作 replay。\n"
            "- 本任务只是 checkpoint 目标复查，不构成新训练结果。\n"
        )


def main() -> None:
    ensure_dirs()
    year_rows = build_year_rows()
    summary = summarize(year_rows)
    branch_name, reason = branch(summary)

    year_path = TAB_DIR / "040_39_checkpoint_year_metric_gaps.csv"
    summary_path = TAB_DIR / "040_39_checkpoint_summary.csv"
    year_rows.to_csv(year_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    figs = [plot_tradeoff(summary), plot_yearly_gap(year_rows)]
    write_record(year_rows, summary, branch_name, reason, figs)
    print(
        json.dumps(
            {
                "task": TASK,
                "branch": branch_name,
                "reason": reason,
                "record_md": DOC.relative_to(ROOT).as_posix(),
                "year_rows": year_path.relative_to(ROOT).as_posix(),
                "summary": summary_path.relative_to(ROOT).as_posix(),
                "figures": [p.relative_to(ROOT).as_posix() for p in figs],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
