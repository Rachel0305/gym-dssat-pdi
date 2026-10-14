"""040_44: audit whether 040_40 SYA lowIC PPO reacts to validation-year variation.

This script is intentionally read-only with respect to model training.  It reads
the existing 040_40 checkpoint100k daily validation outputs and summarizes
whether irrigation/fertilization timing changes across SYA validation years.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "040_44"
TASK_NAME = "sya_lowIC_ppo_policy_response_audit"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"

PPO_DAILY_DIR = ROOT / "benchmark_results" / "040_40_sya_lowIC_ppo_yield_guardrail_v3" / "daily_outputs" / "SYA"
YEARS = list(range(2014, 2024))
CHECKPOINT = 100_000


def ensure_dirs() -> None:
    for rel in ["configs", "tables"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def read_daily(year: int) -> pd.DataFrame:
    path = PPO_DAILY_DIR / f"SYA_{int(year)}_seed0_ckpt{CHECKPOINT}_daily.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, keep_default_na=False)
    df["source_daily_csv"] = path.relative_to(ROOT).as_posix()
    return df


def event_sequence(df: pd.DataFrame, action_col: str, label: str) -> str:
    dap = pd.to_numeric(df["dap"], errors="coerce")
    amount = pd.to_numeric(df[action_col], errors="coerce").fillna(0.0)
    parts = [
        f"DAP{int(d)}:{label}{float(a):g}"
        for d, a in zip(dap, amount)
        if np.isfinite(d) and float(a) > 1e-9
    ]
    return "; ".join(parts)


def rain_window(df: pd.DataFrame, dap_value: int, left: int, right: int) -> float:
    dap = pd.to_numeric(df["dap"], errors="coerce")
    rain = pd.to_numeric(df["rain"], errors="coerce").fillna(0.0)
    mask = (dap >= dap_value + left) & (dap <= dap_value + right)
    return float(rain[mask].sum())


def value_at_dap(df: pd.DataFrame, col: str, dap_value: int) -> float:
    rows = df[pd.to_numeric(df["dap"], errors="coerce").eq(dap_value)]
    if rows.empty or col not in rows.columns:
        return np.nan
    return float(pd.to_numeric(rows[col], errors="coerce").iloc[0])


def summarize_year(year: int, df: pd.DataFrame) -> dict[str, Any]:
    dap = pd.to_numeric(df["dap"], errors="coerce")
    irr = pd.to_numeric(df["safe_action_amir"], errors="coerce").fillna(0.0)
    n = pd.to_numeric(df["safe_action_anfer"], errors="coerce").fillna(0.0)
    rain = pd.to_numeric(df["rain"], errors="coerce").fillna(0.0)
    swfac = pd.to_numeric(df["swfac"], errors="coerce")
    nstres = pd.to_numeric(df["nstres"], errors="coerce")
    grnwt = pd.to_numeric(df["grnwt"], errors="coerce")
    return {
        "year": int(year),
        "final_grnwt": float(grnwt.iloc[-1]) if len(grnwt) else np.nan,
        "rain_total": float(rain.sum()),
        "total_irrigation": float(irr.sum()),
        "total_n": float(n.sum()),
        "max_swfac": float(swfac.max()) if len(swfac) else np.nan,
        "max_nstres": float(nstres.max()) if len(nstres) else np.nan,
        "swfac_days_gt_0p05": int((swfac > 0.05).sum()),
        "nstres_days_gt_0p05": int((nstres > 0.05).sum()),
        "irrigation_dap1_30": float(irr[dap <= 30].sum()),
        "irrigation_dap31_60": float(irr[(dap > 30) & (dap <= 60)].sum()),
        "irrigation_dap61_90": float(irr[(dap > 60) & (dap <= 90)].sum()),
        "irrigation_dap91_plus": float(irr[dap > 90].sum()),
        "irrigation_sequence": event_sequence(df, "safe_action_amir", "I"),
        "nitrogen_sequence": event_sequence(df, "safe_action_anfer", "N"),
        "source_daily_csv": str(df["source_daily_csv"].iloc[0]),
    }


def irrigation_event_context(year: int, df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    irr = pd.to_numeric(df["safe_action_amir"], errors="coerce").fillna(0.0)
    for idx in np.flatnonzero(irr.to_numpy() > 1e-9):
        dap = int(round(float(df.iloc[idx]["dap"])))
        rows.append(
            {
                "year": int(year),
                "dap": dap,
                "irrigation_mm": float(irr.iloc[idx]),
                "rain_prev_7d": rain_window(df, dap, -7, -1),
                "rain_same_day": rain_window(df, dap, 0, 0),
                "rain_next_7d": rain_window(df, dap, 1, 7),
                "swfac_at_action": value_at_dap(df, "swfac", dap),
                "swfac_prev_day": value_at_dap(df, "swfac", dap - 1),
                "swfac_next_day": value_at_dap(df, "swfac", dap + 1),
            }
        )
    return rows


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
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


def write_record(summary: pd.DataFrame, event_ctx: pd.DataFrame, result: dict[str, Any]) -> None:
    cols = [
        "year",
        "rain_total",
        "final_grnwt",
        "total_irrigation",
        "total_n",
        "max_swfac",
        "swfac_days_gt_0p05",
        "irrigation_dap1_30",
        "irrigation_dap31_60",
        "irrigation_dap61_90",
        "irrigation_dap91_plus",
        "irrigation_sequence",
        "nitrogen_sequence",
    ]
    lines = [
        "# 040_44 SYA lowIC PPO 策略年际响应性审计记录",
        "",
        "## 结论先说",
        "",
        f"- 分支：`{result['branch']}`。",
        f"- 验证年份：{', '.join(map(str, YEARS))}。",
        f"- 灌溉唯一序列数：{result['unique_irrigation_sequence_count']}。",
        f"- 施氮唯一序列数：{result['unique_nitrogen_sequence_count']}。",
        "- 本任务未训练模型、未修改 reward、未修改约束。",
        "",
        "## 年度措施与指标摘要",
        "",
        md_table(summary[[c for c in cols if c in summary.columns]], max_rows=30),
        "",
        "## 灌溉事件前后天气/胁迫上下文",
        "",
        md_table(event_ctx, max_rows=80),
        "",
        "## 解释边界",
        "",
        "- 若灌溉唯一序列数为 1，只能说明当前 040_40 checkpoint100k 在验证年份上采用了固定灌溉日程；不能单独说明 PPO 必然如此。",
        "- 若不同年份天气差异很大但灌溉日程不变，则后续应考虑 SAC、天气预报输入、reward/状态响应性改造等方向。",
        "- 本记录不作为调参依据，只作为后续算法候选实验的诊断依据。",
    ]
    text = "\n".join(lines) + "\n"
    DOC.write_text(text, encoding="utf-8")
    (OUT / DOC.name).write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    if PROMPT.exists():
        (OUT / "configs" / PROMPT.name).write_text(PROMPT.read_text(encoding="utf-8"), encoding="utf-8")
    summaries: list[dict[str, Any]] = []
    contexts: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for year in YEARS:
        try:
            daily = read_daily(year)
            summaries.append(summarize_year(year, daily))
            contexts.extend(irrigation_event_context(year, daily))
        except Exception as exc:
            failures.append({"year": int(year), "error": repr(exc)})
    summary = pd.DataFrame(summaries)
    event_ctx = pd.DataFrame(contexts)
    summary_path = OUT / "tables" / "040_44_sya_lowIC_ppo_policy_response_year_summary.csv"
    event_path = OUT / "tables" / "040_44_sya_lowIC_ppo_irrigation_event_context.csv"
    failure_path = OUT / "tables" / "040_44_failures.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    event_ctx.to_csv(event_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(failures).to_csv(failure_path, index=False, encoding="utf-8-sig")
    unique_i = int(summary["irrigation_sequence"].nunique()) if not summary.empty else 0
    unique_n = int(summary["nitrogen_sequence"].nunique()) if not summary.empty else 0
    branch = "fixed_irrigation_schedule_detected" if unique_i == 1 and not failures else "year_responsive_or_incomplete"
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "branch": branch,
        "years": YEARS,
        "checkpoint": CHECKPOINT,
        "unique_irrigation_sequence_count": unique_i,
        "unique_nitrogen_sequence_count": unique_n,
        "failures": len(failures),
        "summary_csv": summary_path.relative_to(ROOT).as_posix(),
        "event_context_csv": event_path.relative_to(ROOT).as_posix(),
        "record_md": DOC.relative_to(ROOT).as_posix(),
    }
    (OUT / "040_44_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_record(summary, event_ctx, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not summary.empty:
        print(summary[["year", "rain_total", "final_grnwt", "total_irrigation", "total_n", "max_swfac", "irrigation_sequence", "nitrogen_sequence"]].to_string(index=False))


if __name__ == "__main__":
    main()

