from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "DSSAT_auto_validation" / "fq2016_leaching_aware_reward_smoke_019_05"
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "fq2016_leaching_cost_sensitivity_019_06"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-10_019_06_fq2016_leaching_cost_sensitivity_smoke_record.md"


def collect() -> pd.DataFrame:
    rows = []
    if not ROOT.exists():
        return pd.DataFrame()

    for summary_path in ROOT.glob("seed0_500steps_lc*/leaching_aware_checkpoint_summary.csv"):
        run_dir = summary_path.parent
        df = pd.read_csv(summary_path)
        if df.empty:
            continue

        row = df.iloc[-1].to_dict()
        row["run_dir"] = str(run_dir.relative_to(PROJECT_ROOT))

        daily = run_dir / "leaching_aware_eval_daily.csv"
        if daily.exists():
            daily_df = pd.read_csv(daily)
            row["nonzero_irrigation_events"] = int(
                (pd.to_numeric(daily_df.get("irrigation_mm", 0), errors="coerce").fillna(0) > 0).sum()
            )
            row["nonzero_fertilizer_events"] = int(
                (pd.to_numeric(daily_df.get("fertilizer_kg_ha", 0), errors="coerce").fillna(0) > 0).sum()
            )
            row["max_leaching_cost_term"] = float(
                pd.to_numeric(daily_df.get("leaching_cost_term", 0), errors="coerce").fillna(0).max()
            )

        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(["leaching_cost", "run_dir"], kind="stable")
    latest_rows = []
    for _, group in out.groupby("leaching_cost", sort=True):
        latest_rows.append(group.iloc[-1])

    return pd.DataFrame(latest_rows).reset_index(drop=True)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_无结果。_"

    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.3f}")
        else:
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else str(v))

    lines = [
        "| " + " | ".join(out.columns) + " |",
        "| " + " | ".join(["---"] * len(out.columns)) + " |",
    ]
    for row in out.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def write_doc(summary: pd.DataFrame) -> None:
    show_cols = [
        "leaching_cost",
        "final_grain_kg_ha",
        "action_irrigation_total",
        "action_fertilizer_total",
        "final_cleach",
        "soilni_final_NLCC",
        "sum_delta_cleach",
        "sum_leaching_cost_term",
        "total_reward",
        "nonzero_irrigation_events",
        "nonzero_fertilizer_events",
        "run_dir",
    ]
    table = dataframe_to_markdown(summary[[c for c in show_cols if c in summary.columns]])
    lines = [
        "# 019_06 FQ2016 leaching-cost sensitivity smoke 记录",
        "",
        "## 结论先行",
        "",
        "本轮比较 `leaching_cost=0/20/50/100` 的 500-step smoke 结果。注意：500 steps 只能用于检查奖励项方向和链路稳定性，不能作为最终策略优劣结论。",
        "",
        "500-step smoke 下，`leaching_cost=20` 是当前最值得进入 5K 短训练的候选；它在本轮中达到最高 `GWAD=8012 kg/ha`，灌溉总量为 `90 mm`，施氮总量仍为 `300 kg/ha`，且 `final_cleach/NLCC=0`。但这不是最终策略结论，因为 500 steps 训练量太小，且所有系数下施氮总量仍为 300 kg/ha。",
        "",
        "`leaching_cost=50/100` 在本轮明显把灌溉压到 `30 mm`，并导致产量下降到约 `7700 kg/ha`，不建议直接放大做长训练。淋洗惩罚目前更像是在影响灌溉/时机，而不是直接降低总施氮。",
        "",
        "## 结果",
        "",
        table,
        "",
        "## 文件",
        "",
        f"- 汇总表：`{(OUT_DIR / '019_06_fq2016_leaching_cost_sensitivity_summary.csv').relative_to(PROJECT_ROOT)}`",
        f"- 来源目录：`{ROOT.relative_to(PROJECT_ROOT)}`",
        "",
        "## 下一步建议",
        "",
        "1. 先用 `leaching_cost=20` 做 FQ2016 5K seed0 短训练。",
        "2. 保留 `leaching_cost=0` 作为无淋洗惩罚对照，不要只看单一结果。",
        "3. 暂不建议直接上 `leaching_cost=50/100` 的长训练，因为短训练已经显示高系数可能压低灌溉和产量。",
        "4. 如果 5K seed0 能维持高产并降低 `cleach/NLCC`，再做 seed1 稳定性复核。",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = collect()
    summary.to_csv(
        OUT_DIR / "019_06_fq2016_leaching_cost_sensitivity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_doc(summary)
    print(summary.to_string(index=False))
    print(DOC_PATH)


if __name__ == "__main__":
    main()
