from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base


OUT = ROOT / "benchmark_results" / "033_03_ic_enabled_five_site_half_split_stress_aware_maskableppo_batch"
DOC = ROOT / "docs" / "033_03_ic_enabled_five_site_half_split_stress_aware_maskableppo_batch_record.md"
PROMPT = ROOT / "prompts" / "033_03_ic_enabled_five_site_half_split_stress_aware_maskableppo_batch.md"


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


def add_comparison_flags_without_old_baselines(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Do not compare IC-enabled PPO against historical IC=0 baselines."""
    if eval_df.empty:
        return eval_df
    out = eval_df.copy()
    out["baseline_rows"] = pd.NA
    out["gap_yield_vs_four_max"] = pd.NA
    out["gap_wp_et_vs_four_max"] = pd.NA
    out["gap_pfp_n_vs_four_max"] = pd.NA
    out["any_metric_win_four"] = pd.NA
    out["baseline_comparison_note"] = "not_computed_in_033_03_old_baselines_are_ic0_need_ic_enabled_baseline_rerun"
    return out


def write_record(
    split: pd.DataFrame,
    train_df: pd.DataFrame,
    reset_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    by_station: pd.DataFrame,
) -> None:
    failed = train_df[~train_df["run_status"].astype(str).str.startswith("ok")].copy() if not train_df.empty else pd.DataFrame()
    lines = [
        "# 033_03 启用 IC 后五站点 half-split stress-aware MaskablePPO 重跑记录",
        "",
        "## 结论先说",
        "",
        f"- 状态：`{'completed' if failed.empty else 'partial'}`。",
        f"- 算法：MaskablePPO；seed={base.SEED}；每站点训练 {base.TOTAL_TIMESTEPS} timesteps。",
        f"- checkpoint：{', '.join(map(str, base.CHECKPOINT_STEPS))}。",
        "- 本轮唯一关键改变：通过 `ppo_safe_rendering.safe_render_template` 渲染出的 treatment 行启用 `IC=1`。",
        "- 本轮不调参、不改 reward、不改动作空间、不跨站点迁移。",
        "- 本轮不和旧四情景基线做最终比较，因为旧四情景基线在 033_01 中已确认多数为 `IC=0`。",
        "",
        "## 固定年份切分",
        "",
        md_table(split[["station_code", "site", "year", "split"]], max_rows=140),
        "",
        "## 训练 checkpoint 库存",
        "",
        md_table(train_df, max_rows=100),
        "",
        "## 训练年份采样次数",
        "",
        md_table(reset_df, max_rows=140),
        "",
        "## 验证集按站点和 checkpoint 汇总",
        "",
        md_table(by_station, max_rows=100),
        "",
        "## 逐年验证结果",
        "",
        md_table(eval_df, max_rows=200),
        "",
        "## 解释边界",
        "",
        "- 033_03 是 IC-enabled PPO 主实验重跑，不是最终四情景优劣比较。",
        "- 下一步必须用同样 IC-enabled 渲染链条重跑 null、recorded、official expert、DSSAT auto 四情景基线。",
        "- 历史 032_22/031_35/031_36 数值保留为 IC=0 历史诊断结果，不再作为正式 IC-enabled 汇报结论。",
    ]
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    base.OUT = OUT
    base.DOC = DOC
    base.PROMPT = PROMPT
    base.add_comparison_flags = add_comparison_flags_without_old_baselines
    base.write_record = write_record
    try:
        base.main()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "033_03_error_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    result_path = OUT / "033_03_result.json"
    if not result_path.exists() and (OUT / "032_22_result.json").exists():
        result = json.loads((OUT / "032_22_result.json").read_text(encoding="utf-8"))
        result["task"] = "033_03_ic_enabled_five_site_half_split_stress_aware_maskableppo_batch"
        result["record_md"] = str(DOC.relative_to(ROOT)).replace("\\", "/")
        result["ic_enabled"] = True
        result["baseline_comparison"] = "not_computed_old_baselines_are_ic0"
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
