from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "040_09"
TASK_NAME = "sya_lowIC_ppo_checkpoint_water_guardrail_audit"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"

PPO_DETAIL = (
    ROOT
    / "benchmark_results"
    / "040_00_sya_lowIC_free_timing_maskableppo"
    / "evaluation"
    / "040_00_checkpoint_validation_summary.csv"
)

STATION = "SYA"
YEARS = list(range(2014, 2024))
CHECKPOINTS = [25_000, 50_000, 75_000, 100_000]
WATER_FAILURE_STRESS_DAYS = 20


def ensure_dirs() -> None:
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
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


def load_rows() -> pd.DataFrame:
    df = pd.read_csv(PPO_DETAIL)
    rows = df[
        df["station_code"].eq(STATION)
        & df["year"].astype(int).isin(YEARS)
        & df["checkpoint_step"].astype(int).isin(CHECKPOINTS)
    ].copy()
    rows["year"] = rows["year"].astype(int)
    rows["checkpoint_step"] = rows["checkpoint_step"].astype(int)
    expected = len(YEARS) * len(CHECKPOINTS)
    if len(rows) != expected:
        raise RuntimeError(f"040_00 checkpoint validation rows incomplete: expected={expected}, got={len(rows)}")
    return rows.sort_values(["checkpoint_step", "year"]).reset_index(drop=True)


def irrigation_buckets(daily_csv_path: str) -> dict[str, Any]:
    daily = pd.read_csv(ROOT / daily_csv_path)
    dap = pd.to_numeric(daily["dap"], errors="coerce")
    irrigation = pd.to_numeric(daily.get("safe_action_amir", 0.0), errors="coerce").fillna(0.0)
    total = float(irrigation.sum())
    buckets = {
        "irrigation_DAP1_30_mm": float(irrigation[(dap >= 1) & (dap <= 30)].sum()),
        "irrigation_DAP31_60_mm": float(irrigation[(dap >= 31) & (dap <= 60)].sum()),
        "irrigation_DAP61_90_mm": float(irrigation[(dap >= 61) & (dap <= 90)].sum()),
        "irrigation_DAP91_plus_mm": float(irrigation[dap >= 91].sum()),
        "has_irrigation_after_DAP30": bool(float(irrigation[dap > 30].sum()) > 1e-9),
        "last_irrigation_dap_recalc": np.nan,
    }
    pos = daily[irrigation > 1e-9]
    if not pos.empty:
        buckets["last_irrigation_dap_recalc"] = int(round(float(pos["dap"].max())))
    for key in ["irrigation_DAP1_30_mm", "irrigation_DAP31_60_mm", "irrigation_DAP61_90_mm", "irrigation_DAP91_plus_mm"]:
        buckets[key.replace("_mm", "_share")] = buckets[key] / total if total > 0 else np.nan
    return buckets


def build_detail(rows: pd.DataFrame) -> pd.DataFrame:
    details = []
    for _, row in rows.iterrows():
        out = row.to_dict()
        out.update(irrigation_buckets(str(row["daily_csv_path"])))
        out["water_failure_year"] = int(float(row["swfac_days_gt_0p05"]) >= WATER_FAILURE_STRESS_DAYS)
        details.append(out)
    detail = pd.DataFrame(details)
    keep = [
        "station_code",
        "year",
        "checkpoint_step",
        "final_grnwt",
        "total_irrigation",
        "total_n",
        "profit_simple",
        "PFP_N",
        "swfac_days_gt_0p05",
        "max_swfac",
        "nstres_days_gt_0p05",
        "irrigation_event_count",
        "first_irrigation_dap",
        "last_irrigation_dap_recalc",
        "water_failure_year",
        "irrigation_DAP1_30_mm",
        "irrigation_DAP1_30_share",
        "irrigation_DAP31_60_mm",
        "irrigation_DAP31_60_share",
        "irrigation_DAP61_90_mm",
        "irrigation_DAP61_90_share",
        "irrigation_DAP91_plus_mm",
        "irrigation_DAP91_plus_share",
        "has_irrigation_after_DAP30",
        "action_sequence",
        "daily_csv_path",
    ]
    return detail[keep]


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    summary = (
        detail.groupby("checkpoint_step", as_index=False)
        .agg(
            mean_yield=("final_grnwt", "mean"),
            min_yield=("final_grnwt", "min"),
            mean_profit_simple=("profit_simple", "mean"),
            mean_total_irrigation=("total_irrigation", "mean"),
            mean_total_n=("total_n", "mean"),
            mean_swfac_days=("swfac_days_gt_0p05", "mean"),
            max_swfac_days=("swfac_days_gt_0p05", "max"),
            water_failure_year_count=("water_failure_year", "sum"),
            mean_max_swfac=("max_swfac", "mean"),
            mean_nstres_days=("nstres_days_gt_0p05", "mean"),
            mean_DAP1_30_irrigation_share=("irrigation_DAP1_30_share", "mean"),
            mean_DAP31_60_irrigation_share=("irrigation_DAP31_60_share", "mean"),
            mean_DAP61_90_irrigation_share=("irrigation_DAP61_90_share", "mean"),
            years_with_irrigation_after_DAP30=("has_irrigation_after_DAP30", "sum"),
            mean_irrigation_events=("irrigation_event_count", "mean"),
        )
        .sort_values("checkpoint_step")
        .reset_index(drop=True)
    )
    best_yield = float(summary["mean_yield"].max())
    summary["is_original_mean_yield_best"] = summary["mean_yield"].eq(best_yield)
    summary["yield_gap_vs_best_mean"] = summary["mean_yield"] - best_yield
    summary["within_2pct_of_best_mean_yield"] = summary["mean_yield"] >= best_yield * 0.98
    return summary


def pareto(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, a in summary.iterrows():
        for _, b in summary.iterrows():
            if int(a["checkpoint_step"]) == int(b["checkpoint_step"]):
                continue
            ge_yield = float(a["mean_yield"]) >= float(b["mean_yield"]) - 1e-9
            le_mean_stress = float(a["mean_swfac_days"]) <= float(b["mean_swfac_days"]) + 1e-9
            le_max_stress = float(a["max_swfac_days"]) <= float(b["max_swfac_days"]) + 1e-9
            strict = (
                float(a["mean_yield"]) > float(b["mean_yield"]) + 1e-9
                or float(a["mean_swfac_days"]) < float(b["mean_swfac_days"]) - 1e-9
                or float(a["max_swfac_days"]) < float(b["max_swfac_days"]) - 1e-9
            )
            rows.append(
                {
                    "candidate_checkpoint": int(a["checkpoint_step"]),
                    "compared_checkpoint": int(b["checkpoint_step"]),
                    "candidate_dominates_compared": bool(ge_yield and le_mean_stress and le_max_stress and strict),
                    "delta_mean_yield": float(a["mean_yield"]) - float(b["mean_yield"]),
                    "delta_mean_swfac_days": float(a["mean_swfac_days"]) - float(b["mean_swfac_days"]),
                    "delta_max_swfac_days": float(a["max_swfac_days"]) - float(b["max_swfac_days"]),
                }
            )
    dom = pd.DataFrame(rows)
    dominated = dom[dom["candidate_dominates_compared"]].groupby("compared_checkpoint")["candidate_checkpoint"].apply(list)
    summary_cp = summary["checkpoint_step"].astype(int).tolist()
    nondominated = [cp for cp in summary_cp if cp not in dominated.index.tolist()]
    dom["nondominated_checkpoints"] = json.dumps(nondominated)
    return dom


def write_record(summary: pd.DataFrame, detail: pd.DataFrame, dom: pd.DataFrame) -> str:
    best_cp = int(summary.loc[summary["mean_yield"].idxmax(), "checkpoint_step"])
    dominators = dom[(dom["compared_checkpoint"].eq(best_cp)) & (dom["candidate_dominates_compared"])]
    close = summary[(~summary["checkpoint_step"].eq(best_cp)) & (summary["within_2pct_of_best_mean_yield"])]
    if not dominators.empty:
        branch = "A_existing_checkpoint_dominates_yield_best"
        conclusion = f"存在 checkpoint 支配原始平均产量最优 checkpoint {best_cp}，可考虑改 checkpoint 选择。"
    elif not close.empty:
        branch = "B_close_yield_alternative_exists_but_not_dominating"
        conclusion = f"没有 checkpoint 支配原始平均产量最优 checkpoint {best_cp}，但存在平均产量接近的候选，需要人工查看胁迫收益是否值得。"
    else:
        branch = "C_no_existing_guardrail_checkpoint"
        conclusion = f"现有 040_00 训练轨迹中没有明显可替代 {best_cp} 的水分 guardrail checkpoint。"
    lines = [
        "# 040_09 SYA lowIC PPO checkpoint 水分 guardrail 离线审计记录",
        "",
        "## 结论先说",
        "",
        f"- 分支：`{branch}`。",
        f"- {conclusion}",
        "- 本任务没有重新训练，也没有重新跑 DSSAT，只读取 040_00 已有 checkpoint 评估与逐日 CSV。",
        "",
        "## checkpoint 汇总",
        "",
        md_table(summary, max_rows=20),
        "",
        "## Pareto 支配关系",
        "",
        md_table(dom[dom["candidate_dominates_compared"]], max_rows=40),
        "",
        "## 逐年明细预览",
        "",
        md_table(detail, max_rows=40),
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return branch


def main() -> None:
    ensure_dirs()
    if not PROMPT.exists():
        raise FileNotFoundError(PROMPT)
    rows = load_rows()
    detail = build_detail(rows)
    summary = summarize(detail)
    dom = pareto(summary)
    detail.to_csv(OUT / "tables" / f"{TASK_ID}_checkpoint_year_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "tables" / f"{TASK_ID}_checkpoint_guardrail_summary.csv", index=False, encoding="utf-8-sig")
    dom.to_csv(OUT / "tables" / f"{TASK_ID}_pareto_dominance.csv", index=False, encoding="utf-8-sig")
    branch = write_record(summary, detail, dom)
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "branch": branch,
        "summary_csv": (OUT / "tables" / f"{TASK_ID}_checkpoint_guardrail_summary.csv").relative_to(ROOT).as_posix(),
        "detail_csv": (OUT / "tables" / f"{TASK_ID}_checkpoint_year_detail.csv").relative_to(ROOT).as_posix(),
        "pareto_csv": (OUT / "tables" / f"{TASK_ID}_pareto_dominance.csv").relative_to(ROOT).as_posix(),
        "record_md": DOC.relative_to(ROOT).as_posix(),
    }
    (OUT / f"{TASK_ID}_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
