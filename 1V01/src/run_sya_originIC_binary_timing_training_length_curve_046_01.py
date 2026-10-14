"""046_01: Training-length curve for SYA originIC binary-timing PPO.

Thin wrapper around 042_10. It runs one training trajectory with checkpoints
at 1K, 2K, 5K, 10K, and 25K, then adds an action-diversity summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_sya_lowIC_binary_timing_maskableppo_042_10 as base04210


TASK_ID = "046_01"
TASK_NAME = "sya_lowIC_binary_timing_training_length_curve"
SUFFIX = "length_curve"
CHECKPOINT_STEPS = [1000, 2000, 5000, 10000, 25000]
TOTAL_TIMESTEPS = 25000

OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"


def md_table(df: pd.DataFrame, max_rows: int = 120) -> str:
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


def event_position_variability(eval_df: pd.DataFrame, action_type: str) -> float:
    event_rows = []
    for row in eval_df.itertuples(index=False):
        daily_path = ROOT / str(row.daily_csv_path)
        if not daily_path.exists():
            continue
        daily = pd.read_csv(daily_path)
        dap = pd.to_numeric(daily.get("dap", pd.Series(dtype=float)), errors="coerce")
        if action_type == "irrigation":
            amount = pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        else:
            amount = pd.to_numeric(daily.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        sub = pd.DataFrame({"dap": dap, "amount": amount})
        sub = sub[sub["amount"] > 1e-9].sort_values("dap").copy()
        for order, erow in enumerate(sub.itertuples(index=False), start=1):
            event_rows.append(
                {
                    "checkpoint_step": int(row.checkpoint_step),
                    "year": int(row.year),
                    "event_order": int(order),
                    "dap": float(erow.dap),
                    "amount": float(erow.amount),
                }
            )
    if not event_rows:
        return 0.0
    events = pd.DataFrame(event_rows)
    pos = (
        events.groupby(["checkpoint_step", "event_order"], as_index=False)
        .agg(sd_dap=("dap", lambda s: float(pd.to_numeric(s, errors="coerce").std(ddof=0))))
    )
    return float(pos["sd_dap"].mean()) if not pos.empty else 0.0


def summarize_action_diversity(eval_path: Path) -> pd.DataFrame:
    eval_df = pd.read_csv(eval_path, keep_default_na=False)
    rows: list[dict[str, Any]] = []
    for ckpt, g in eval_df.groupby("checkpoint_step"):
        action_seq = g["action_sequence"].astype(str)
        rows.append(
            {
                "checkpoint_step": int(ckpt),
                "validation_years": int(g["year"].nunique()),
                "unique_action_signatures": int(action_seq.nunique()),
                "sd_total_irrigation": float(pd.to_numeric(g["total_irrigation"], errors="coerce").std(ddof=0)),
                "sd_total_n": float(pd.to_numeric(g["total_n"], errors="coerce").std(ddof=0)),
                "mean_total_irrigation": float(pd.to_numeric(g["total_irrigation"], errors="coerce").mean()),
                "mean_total_n": float(pd.to_numeric(g["total_n"], errors="coerce").mean()),
                "mean_yield": float(pd.to_numeric(g["final_grnwt"], errors="coerce").mean()),
                "mean_pfp_n": float(pd.to_numeric(g["PFP_N"], errors="coerce").mean()),
                "any_metric_win_years": int(pd.to_numeric(g["any_metric_win_four"], errors="coerce").fillna(0).sum()),
                "mean_irrigation_event_dap_sd": event_position_variability(g, "irrigation"),
                "mean_nitrogen_event_dap_sd": event_position_variability(g, "nitrogen"),
                "template_like": bool(action_seq.nunique() <= 1),
            }
        )
    return pd.DataFrame(rows).sort_values("checkpoint_step")


def decide_branch(div: pd.DataFrame) -> tuple[str, bool]:
    if div.empty:
        return "C_no_useful_diversity", False
    early = div[div["checkpoint_step"] <= 2000]
    late = div[div["checkpoint_step"] >= 5000]
    early_diverse = bool((early["unique_action_signatures"] >= 3).any())
    late_template = bool((late["unique_action_signatures"] <= 1).any())
    persistent = bool((div["unique_action_signatures"] >= 3).all())
    if early_diverse and late_template:
        return "A_early_diversity_then_late_collapse", False
    if persistent:
        return "B_diversity_persists", True
    return "C_no_useful_diversity", False


def write_record(result: dict[str, Any], div: pd.DataFrame, by_ckpt: pd.DataFrame, eval_df: pd.DataFrame) -> None:
    lines = [
        f"# {TASK_ID} SYA lowIC binary-timing PPO 训练长度曲线",
        "",
        "## 结论",
        "",
        f"- 分支：`{result['branch']}`",
        f"- next_step_allowed：`{str(result['next_step_allowed']).lower()}`",
        "- 本任务不选择最终 checkpoint，只记录训练长度和动作多样性的关系。",
        "",
        "## 动作多样性曲线",
        "",
        md_table(div, 80),
        "",
        "## endpoint 汇总",
        "",
        md_table(by_ckpt, 80),
        "",
        "## 逐年结果",
        "",
        md_table(eval_df, 220),
        "",
        "## 输出文件",
        "",
    ]
    for k, v in result["outputs"].items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    # Route 042_10 outputs into this task-specific folder by monkey-patching its
    # naming helpers. Keep its actual framework unchanged.
    old_task_id = base04210.TASK_ID
    old_task_name = base04210.TASK_NAME
    old_base_out = base04210.BASE_OUT
    old_base_doc = base04210.BASE_DOC
    old_prompt = base04210.PROMPT
    old_renames = dict(base04210.RENAMES)
    try:
        base04210.TASK_ID = TASK_ID
        base04210.TASK_NAME = TASK_NAME
        base04210.BASE_OUT = OUT
        base04210.BASE_DOC = DOC
        base04210.PROMPT = PROMPT
        base04210.RENAMES = {
            "configs/032_22_half_split_selection.csv": "configs/042_11_half_split_selection.csv",
            "evaluation/032_22_training_checkpoint_inventory_partial.csv": "evaluation/042_11_training_checkpoint_inventory_partial.csv",
            "evaluation/032_22_checkpoint_validation_summary_partial.csv": "evaluation/042_11_checkpoint_validation_summary_partial.csv",
            "evaluation/032_22_training_checkpoint_inventory.csv": "evaluation/042_11_training_checkpoint_inventory.csv",
            "evaluation/032_22_checkpoint_validation_summary.csv": "evaluation/042_11_checkpoint_validation_summary.csv",
            "evaluation/032_22_validation_summary_by_station_checkpoint.csv": "evaluation/042_11_validation_summary_by_station_checkpoint.csv",
            "logs/032_22_training_year_reset_counts.csv": "logs/042_11_training_year_reset_counts.csv",
            "032_22_result.json": "042_11_result.json",
        }
        base04210.run_training(TOTAL_TIMESTEPS, CHECKPOINT_STEPS, suffix="")
    finally:
        base04210.TASK_ID = old_task_id
        base04210.TASK_NAME = old_task_name
        base04210.BASE_OUT = old_base_out
        base04210.BASE_DOC = old_base_doc
        base04210.PROMPT = old_prompt
        base04210.RENAMES = old_renames

    eval_path = OUT / "evaluation" / "042_11_checkpoint_validation_summary.csv"
    by_path = OUT / "evaluation" / "042_11_validation_summary_by_station_checkpoint.csv"
    div_path = OUT / "evaluation" / "042_11_action_diversity_curve.csv"
    result_path = OUT / "042_11_result.json"

    div = summarize_action_diversity(eval_path)
    div.to_csv(div_path, index=False, encoding="utf-8-sig")
    branch, next_allowed = decide_branch(div)
    by_ckpt = pd.read_csv(by_path, keep_default_na=False) if by_path.exists() else pd.DataFrame()
    eval_df = pd.read_csv(eval_path, keep_default_na=False) if eval_path.exists() else pd.DataFrame()
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "branch": branch,
        "next_step_allowed": next_allowed,
        "total_timesteps": TOTAL_TIMESTEPS,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "outputs": {
            "record_md": str(DOC.relative_to(ROOT)),
            "validation_summary": str(eval_path.relative_to(ROOT)),
            "by_checkpoint": str(by_path.relative_to(ROOT)),
            "action_diversity_curve": str(div_path.relative_to(ROOT)),
            "result_json": str(result_path.relative_to(ROOT)),
        },
    }
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_record(result, div, by_ckpt, eval_df)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
