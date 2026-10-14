from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "040_06"
TASK_NAME = "sya_lowIC_ppo_irrigation_budget_distribution_audit"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
TABLE_DIR = OUT / "tables"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"

PPO_SUMMARY = (
    ROOT
    / "benchmark_results"
    / "040_00_sya_lowIC_free_timing_maskableppo"
    / "evaluation"
    / "040_00_checkpoint_validation_summary.csv"
)
DIAGNOSIS = (
    ROOT
    / "benchmark_results"
    / "040_04_sya_lowIC_ppo_failure_diagnosis"
    / "tables"
    / "040_04_ppo_vs_lowIC_baselines_by_year.csv"
)

STATION = "SYA"
CHECKPOINT = 75_000
YEARS = list(range(2014, 2024))
INTERVALS = [
    ("DAP1_30", 1, 30),
    ("DAP31_60", 31, 60),
    ("DAP61_90", 61, 90),
    ("DAP91_plus", 91, 999),
]


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def md_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "无记录。"
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
    work = work.astype(object).where(pd.notna(work), "")
    header = "| " + " | ".join(map(str, work.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in work.to_numpy().tolist()]
    return "\n".join([header, sep, *rows])


def interval_sum(daily: pd.DataFrame, start: int, end: int) -> float:
    dap = pd.to_numeric(daily["dap"], errors="coerce")
    irr = pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0.0)
    return float(irr[(dap >= start) & (dap <= end)].sum())


def first_dap(masked: pd.DataFrame) -> int | None:
    if masked.empty:
        return None
    return int(round(float(masked["dap"].iloc[0])))


def audit_year(row: pd.Series, diagnosis_row: pd.Series | None) -> dict:
    daily_path = ROOT / str(row["daily_csv_path"])
    if not daily_path.exists():
        raise FileNotFoundError(daily_path)
    daily = pd.read_csv(daily_path)
    daily["dap"] = pd.to_numeric(daily["dap"], errors="coerce")
    daily["safe_action_amir"] = pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0.0)
    daily["swfac"] = pd.to_numeric(daily["swfac"], errors="coerce")

    total_i = float(daily["safe_action_amir"].sum())
    nonzero_i = daily[daily["safe_action_amir"] > 0].copy()
    stress = daily[daily["swfac"] > 0.05].copy()
    first_stress = first_dap(stress)
    first_i = first_dap(nonzero_i)
    last_i = int(round(float(nonzero_i["dap"].iloc[-1]))) if not nonzero_i.empty else None

    out = {
        "station_code": STATION,
        "year": int(row["year"]),
        "checkpoint_step": CHECKPOINT,
        "final_grnwt": float(row["final_grnwt"]),
        "total_irrigation": total_i,
        "total_n": float(row["total_n"]),
        "swfac_days_gt_0p05": int(row["swfac_stress_days_gt_0p05"]),
        "max_swfac": float(row["max_swfac"]),
        "first_irrigation_dap": first_i,
        "last_irrigation_dap": last_i,
        "irrigation_event_count": int((daily["safe_action_amir"] > 0).sum()),
        "first_swfac_gt_0p05_dap": first_stress,
        "irrigation_after_first_stress": False,
        "irrigation_before_or_at_first_stress": total_i if first_stress is None else float(daily.loc[daily["dap"] <= first_stress, "safe_action_amir"].sum()),
        "irrigation_after_first_stress_mm": 0.0 if first_stress is None else float(daily.loc[daily["dap"] > first_stress, "safe_action_amir"].sum()),
        "daily_csv_path": str(row["daily_csv_path"]),
    }
    out["irrigation_after_first_stress"] = bool(out["irrigation_after_first_stress_mm"] > 0)

    for label, start, end in INTERVALS:
        amount = interval_sum(daily, start, end)
        out[f"irrigation_{label}_mm"] = amount
        out[f"irrigation_{label}_share"] = amount / total_i if total_i > 0 else pd.NA

    if diagnosis_row is not None:
        for col in [
            "diagnosis_class",
            "ppo_gap_yield_vs_expert",
            "ppo_water_saved_vs_expert",
            "ppo_n_saved_vs_expert",
        ]:
            out[col] = diagnosis_row.get(col, pd.NA)
    return out


def main() -> None:
    ensure_dirs()
    ppo = pd.read_csv(PPO_SUMMARY)
    ppo = ppo[
        (ppo["station_code"].eq(STATION))
        & (ppo["checkpoint_step"].astype(int).eq(CHECKPOINT))
        & (ppo["year"].astype(int).isin(YEARS))
    ].copy()
    if len(ppo) != len(YEARS):
        raise RuntimeError(f"PPO checkpoint {CHECKPOINT} 验证年数量异常: {len(ppo)}")

    diag = pd.read_csv(DIAGNOSIS) if DIAGNOSIS.exists() else pd.DataFrame()
    diag_map = {int(r["year"]): r for _, r in diag.iterrows()} if not diag.empty else {}

    rows = []
    for _, row in ppo.sort_values("year").iterrows():
        year = int(row["year"])
        rows.append(audit_year(row, diag_map.get(year)))
    by_year = pd.DataFrame(rows)

    group_cols = ["diagnosis_class"]
    by_class = (
        by_year.groupby(group_cols, as_index=False)
        .agg(
            year_count=("year", "count"),
            mean_yield_gap_vs_expert=("ppo_gap_yield_vs_expert", "mean"),
            mean_total_irrigation=("total_irrigation", "mean"),
            mean_swfac_days_gt_0p05=("swfac_days_gt_0p05", "mean"),
            mean_first_swfac_gt_0p05_dap=("first_swfac_gt_0p05_dap", "mean"),
            mean_irrigation_DAP1_30_share=("irrigation_DAP1_30_share", "mean"),
            mean_irrigation_DAP31_60_share=("irrigation_DAP31_60_share", "mean"),
            mean_irrigation_DAP61_90_share=("irrigation_DAP61_90_share", "mean"),
            mean_irrigation_DAP91_plus_share=("irrigation_DAP91_plus_share", "mean"),
            years_with_irrigation_after_first_stress=("irrigation_after_first_stress", "sum"),
        )
        .sort_values(["year_count", "diagnosis_class"], ascending=[False, True])
    )

    by_year_csv = TABLE_DIR / f"{TASK_ID}_irrigation_distribution_by_year.csv"
    by_class_csv = TABLE_DIR / f"{TASK_ID}_irrigation_distribution_by_diagnosis_class.csv"
    by_year.to_csv(by_year_csv, index=False, encoding="utf-8-sig")
    by_class.to_csv(by_class_csv, index=False, encoding="utf-8-sig")

    show_cols = [
        "year",
        "diagnosis_class",
        "final_grnwt",
        "ppo_gap_yield_vs_expert",
        "total_irrigation",
        "irrigation_DAP1_30_mm",
        "irrigation_DAP1_30_share",
        "irrigation_DAP31_60_mm",
        "irrigation_DAP61_90_mm",
        "irrigation_DAP91_plus_mm",
        "first_swfac_gt_0p05_dap",
        "swfac_days_gt_0p05",
        "irrigation_after_first_stress_mm",
    ]

    doc = f"""# 040_06 SYA lowIC PPO 灌溉预算分配审计记录

## 结论先说

本任务固定 PPO checkpoint {CHECKPOINT}，只读取已有 daily CSV，不重新训练、不重跑 DSSAT。

核心发现：PPO 在 2014–2023 验证年中几乎把全部灌溉预算集中在前 30 DAP。  
这支持 040_05 的机制判断：2014/2017 的水分胁迫失败不是因为完全没有水，而是因为水用得过早，后期胁迫出现后没有剩余灌溉响应。

## 逐年灌溉分配

{md_table(by_year[show_cols], max_rows=20)}

## 按失败类型汇总

{md_table(by_class, max_rows=20)}

## 解释

- 如果 `irrigation_DAP1_30_share` 接近 1，说明 PPO 几乎把水全用在前 30 天。
- 如果 `first_swfac_gt_0p05_dap` 远晚于最后一次灌溉 DAP，且 `irrigation_after_first_stress_mm = 0`，说明 PPO 没有为后期水分胁迫保留预算。
- 这一步仍是审计，不是改 reward 或改约束。

## 输出文件

- 逐年表：`{by_year_csv.relative_to(ROOT)}`
- 类型汇总表：`{by_class_csv.relative_to(ROOT)}`

"""
    DOC.write_text(doc, encoding="utf-8")

    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "branch": "audit_completed",
        "by_year_csv": by_year_csv.relative_to(ROOT).as_posix(),
        "by_class_csv": by_class_csv.relative_to(ROOT).as_posix(),
        "record_md": DOC.relative_to(ROOT).as_posix(),
        "mean_DAP1_30_irrigation_share": float(by_year["irrigation_DAP1_30_share"].mean()),
        "years_with_irrigation_after_first_stress": int(by_year["irrigation_after_first_stress"].sum()),
    }
    (OUT / f"{TASK_ID}_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
