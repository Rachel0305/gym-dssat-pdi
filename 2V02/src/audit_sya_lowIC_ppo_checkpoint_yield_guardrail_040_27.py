"""040_27: yield guardrail audit for 040_26 SYA lowIC PPO checkpoints.

No training is performed.  This script checks whether any 040_26 checkpoint
already satisfies a pre-registered yield floor relative to the official expert.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK = "040_27_sya_lowIC_ppo_checkpoint_yield_guardrail"
OUT = ROOT / "benchmark_results" / TASK
DOC = ROOT / "docs" / f"{TASK}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK}.md"

PPO_CSV = ROOT / "benchmark_results" / "040_26_sya_lowIC_ppo_i240_staged_reserve" / "evaluation" / "040_26_checkpoint_validation_summary.csv"
BASELINE_CSV = ROOT / "benchmark_results" / "040_21_sya_lowIC_four_baseline_rebuild" / "evaluation" / "040_21_baseline_summary.csv"

MEAN_YIELD_RATIO_FLOOR = 0.95
YEAR_YIELD_RATIO_FLOOR = 0.90
MIN_YEAR_PASS_COUNT = 8


def ensure_dirs() -> None:
    for rel in ["tables"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "无记录。"
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
    work = work.astype(object).where(pd.notna(work), "")
    lines = [
        "| " + " | ".join(map(str, work.columns)) + " |",
        "| " + " | ".join(["---"] * len(work.columns)) + " |",
    ]
    for row in work.to_numpy().tolist():
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not PPO_CSV.exists():
        raise FileNotFoundError(PPO_CSV)
    if not BASELINE_CSV.exists():
        raise FileNotFoundError(BASELINE_CSV)
    ppo = pd.read_csv(PPO_CSV, keep_default_na=False)
    base = pd.read_csv(BASELINE_CSV, keep_default_na=False)
    base["scenario"] = base["scenario"].replace({"": "null"})
    return ppo, base


def compute_audit() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    ppo, base = load_inputs()
    expert = base[base["scenario"].eq("official_extension_expert")].copy()
    expert = expert[["station_code", "year", "grain_yield_kg_ha", "actual_irrigation_mm", "actual_nitrogen_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg"]].rename(
        columns={
            "grain_yield_kg_ha": "expert_yield",
            "actual_irrigation_mm": "expert_irrigation",
            "actual_nitrogen_kg_ha": "expert_n",
            "WP_ET_kg_m3": "expert_WP_ET",
            "PFP_N_kg_kg": "expert_PFP_N",
        }
    )
    ppo = ppo[ppo["run_status"].astype(str).eq("ok")].copy()
    for col in ["year", "checkpoint_step", "final_grnwt", "total_irrigation", "total_n", "PFP_N", "max_swfac", "swfac_days_gt_0p05"]:
        if col in ppo.columns:
            ppo[col] = pd.to_numeric(ppo[col], errors="coerce")
    merged = ppo.merge(expert, on=["station_code", "year"], how="left")
    merged["yield_ratio_vs_expert"] = merged["final_grnwt"] / merged["expert_yield"]
    merged["year_yield_guardrail_pass"] = merged["yield_ratio_vs_expert"] >= YEAR_YIELD_RATIO_FLOOR
    merged["yield_gap_vs_expert"] = merged["final_grnwt"] - merged["expert_yield"]
    merged["irrigation_saving_vs_expert"] = merged["expert_irrigation"] - merged["total_irrigation"]
    merged["n_saving_vs_expert"] = merged["expert_n"] - merged["total_n"]

    expert_mean_yield = float(expert["expert_yield"].mean())
    rows = []
    for step, g in merged.groupby("checkpoint_step"):
        mean_yield = float(g["final_grnwt"].mean())
        mean_ratio = mean_yield / expert_mean_yield if expert_mean_yield else np.nan
        year_pass_count = int(g["year_yield_guardrail_pass"].sum())
        rows.append(
            {
                "checkpoint_step": int(step),
                "validation_year_count": int(len(g)),
                "mean_yield": mean_yield,
                "expert_mean_yield": expert_mean_yield,
                "mean_yield_ratio_vs_expert": mean_ratio,
                "mean_yield_guardrail_pass": bool(mean_ratio >= MEAN_YIELD_RATIO_FLOOR),
                "year_yield_guardrail_pass_count": year_pass_count,
                "year_yield_guardrail_required_count": MIN_YEAR_PASS_COUNT,
                "year_yield_guardrail_pass": bool(year_pass_count >= MIN_YEAR_PASS_COUNT),
                "overall_guardrail_pass": bool(mean_ratio >= MEAN_YIELD_RATIO_FLOOR and year_pass_count >= MIN_YEAR_PASS_COUNT),
                "mean_irrigation": float(g["total_irrigation"].mean()),
                "mean_n": float(g["total_n"].mean()),
                "mean_PFP_N": float(g["PFP_N"].mean(skipna=True)) if g["PFP_N"].notna().any() else np.nan,
                "mean_irrigation_saving_vs_expert": float(g["irrigation_saving_vs_expert"].mean()),
                "mean_n_saving_vs_expert": float(g["n_saving_vs_expert"].mean()),
                "mean_swfac_days_gt_0p05": float(g["swfac_days_gt_0p05"].mean()) if "swfac_days_gt_0p05" in g.columns else np.nan,
                "max_swfac": float(g["max_swfac"].max()) if "max_swfac" in g.columns else np.nan,
            }
        )
    summary = pd.DataFrame(rows).sort_values(["overall_guardrail_pass", "mean_yield"], ascending=[False, False])
    thresholds = {
        "mean_yield_ratio_floor": MEAN_YIELD_RATIO_FLOOR,
        "year_yield_ratio_floor": YEAR_YIELD_RATIO_FLOOR,
        "min_year_pass_count": MIN_YEAR_PASS_COUNT,
        "expert_mean_yield": expert_mean_yield,
        "mean_yield_floor_kg_ha": expert_mean_yield * MEAN_YIELD_RATIO_FLOOR,
    }
    return merged, summary, thresholds


def write_record(yearly: pd.DataFrame, summary: pd.DataFrame, thresholds: dict[str, float]) -> str:
    passed = summary[summary["overall_guardrail_pass"].astype(bool)].copy()
    branch = "A_checkpoint_passes_yield_guardrail" if not passed.empty else "B_no_checkpoint_passes_yield_guardrail"

    yearly_cols = [
        "year",
        "checkpoint_step",
        "final_grnwt",
        "expert_yield",
        "yield_ratio_vs_expert",
        "year_yield_guardrail_pass",
        "total_irrigation",
        "total_n",
        "PFP_N",
        "swfac_days_gt_0p05",
        "action_sequence",
    ]
    yearly_cols = [c for c in yearly_cols if c in yearly.columns]
    lines = [
        "# 040_27 SYA lowIC PPO checkpoint 产量底线筛选记录",
        "",
        "## 结论",
        "",
        f"- 分支：`{branch}`。",
        f"- 均值产量底线：official expert 平均产量的 {MEAN_YIELD_RATIO_FLOOR:.0%}，即 {thresholds['mean_yield_floor_kg_ha']:.1f} kg/ha。",
        f"- 逐年底线：至少 {MIN_YEAR_PASS_COUNT}/10 个验证年达到同年 official expert 产量的 {YEAR_YIELD_RATIO_FLOOR:.0%}。",
        "",
    ]
    if passed.empty:
        lines.extend(
            [
                "- 040_26 没有任何 checkpoint 通过产量底线。",
                "- 因此，040_26 的失败不能通过简单 checkpoint 选择解决；下一步需要把产量或水分胁迫底线接入训练目标、reward 或更强的训练期 guardrail。",
                "",
            ]
        )
    else:
        best = passed.iloc[0]
        lines.extend(
            [
                f"- 通过 guardrail 的候选 checkpoint：`{int(best['checkpoint_step'])}`。",
                "- 可进入五情景图和进一步 DSSAT 过程合理性检查。",
                "",
            ]
        )

    lines.extend(
        [
            "## checkpoint 汇总",
            "",
            md_table(summary, 80),
            "",
            "## 逐年明细",
            "",
            md_table(yearly[yearly_cols].sort_values(["checkpoint_step", "year"]), 240),
            "",
            "## 输出",
            "",
            f"- yearly：`{(OUT / 'tables' / '040_27_yearly_guardrail_audit.csv').relative_to(ROOT).as_posix()}`",
            f"- summary：`{(OUT / 'tables' / '040_27_checkpoint_guardrail_summary.csv').relative_to(ROOT).as_posix()}`",
        ]
    )
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return branch


def main() -> None:
    ensure_dirs()
    yearly, summary, thresholds = compute_audit()
    yearly.to_csv(OUT / "tables" / "040_27_yearly_guardrail_audit.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "tables" / "040_27_checkpoint_guardrail_summary.csv", index=False, encoding="utf-8-sig")
    branch = write_record(yearly, summary, thresholds)
    result = {
        "task": TASK,
        "branch": branch,
        "prompt": PROMPT.relative_to(ROOT).as_posix(),
        "record_md": DOC.relative_to(ROOT).as_posix(),
        "summary_csv": (OUT / "tables" / "040_27_checkpoint_guardrail_summary.csv").relative_to(ROOT).as_posix(),
        "yearly_csv": (OUT / "tables" / "040_27_yearly_guardrail_audit.csv").relative_to(ROOT).as_posix(),
        "thresholds": thresholds,
        "checkpoint_pass_count": int(summary["overall_guardrail_pass"].sum()),
    }
    (OUT / "040_27_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

