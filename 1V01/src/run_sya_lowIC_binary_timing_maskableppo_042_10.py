"""042_10: SYA lowIC binary-timing MaskablePPO.

This is a minimal fork of the 040_36 mainline. It keeps the free-timing
MaskablePPO framework, lowIC inputs, reward, and safety masks, but collapses the
discrete action grid to:

    irrigation: [0, 45] mm
    nitrogen:   [0, 80] kg/ha

The goal is to test timing learning before re-opening dose learning.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ppo_safe_rendering
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base03222
import run_sya_lowIC_ppo_early_irrigation_reserve_penalty_040_10 as util04010
import run_sya_lowIC_ppo_i240_staged_reserve_040_26 as base04026
import run_sya_lowIC_ppo_late_irrigation_reserve_mask_040_36 as base04036


ORIGINAL_SHUTIL_COPY2 = shutil.copy2
TASK_ID = "042_10"
TASK_NAME = "sya_lowIC_binary_timing_maskableppo"
BASE_OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
BASE_DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"

LOWIC_INPUT_ROOT = base04036.LOWIC_INPUT_ROOT
STATION = base04036.STATION
SITES = [STATION]
DEFAULT_TOTAL_TIMESTEPS = 100_000
DEFAULT_CHECKPOINT_STEPS = [25_000, 50_000, 75_000, 100_000]

BINARY_IRRIGATION_LEVELS = [0.0, 45.0]
BINARY_NITROGEN_LEVELS = [0.0, 80.0]

RENAMES = {
    "configs/032_22_half_split_selection.csv": "configs/042_10_half_split_selection.csv",
    "evaluation/032_22_training_checkpoint_inventory_partial.csv": "evaluation/042_10_training_checkpoint_inventory_partial.csv",
    "evaluation/032_22_checkpoint_validation_summary_partial.csv": "evaluation/042_10_checkpoint_validation_summary_partial.csv",
    "evaluation/032_22_training_checkpoint_inventory.csv": "evaluation/042_10_training_checkpoint_inventory.csv",
    "evaluation/032_22_checkpoint_validation_summary.csv": "evaluation/042_10_checkpoint_validation_summary.csv",
    "evaluation/032_22_validation_summary_by_station_checkpoint.csv": "evaluation/042_10_validation_summary_by_station_checkpoint.csv",
    "logs/032_22_training_year_reset_counts.csv": "logs/042_10_training_year_reset_counts.csv",
    "032_22_result.json": "042_10_result.json",
}


def safe_copy2(src: str | Path, dst: str | Path, *args, **kwargs) -> str:
    try:
        return str(ORIGINAL_SHUTIL_COPY2(src, dst, *args, **kwargs))
    except PermissionError:
        dst_path = Path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst_path)
        return str(dst_path)


def out_for_suffix(suffix: str) -> Path:
    return ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}_{suffix}" if suffix else BASE_OUT


def doc_for_suffix(suffix: str) -> Path:
    return ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_{suffix}_record.md" if suffix else BASE_DOC


def parse_checkpoint_steps(raw: str | None, total_timesteps: int) -> list[int]:
    if not raw:
        return [x for x in DEFAULT_CHECKPOINT_STEPS if x <= total_timesteps]
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def load_config() -> dict[str, Any]:
    cfg = base04036.load_config()
    cfg["discrete_actions"]["irrigation_levels"] = list(BINARY_IRRIGATION_LEVELS)
    cfg["discrete_actions"]["nitrogen_levels"] = list(BINARY_NITROGEN_LEVELS)
    cfg["discrete_actions"]["binary_timing_source"] = (
        "042_10 collapses dose choices to fixed I45/N80 so PPO first learns timing; "
        "reward and safety masks are inherited from 040_36."
    )
    return cfg


def patch_base_module(out: Path, doc: Path, total_timesteps: int, checkpoint_steps: list[int]) -> None:
    base03222.OUT = out
    base03222.DOC = doc
    base03222.PROMPT = PROMPT
    base03222.SITES = list(SITES)
    base03222.TOTAL_TIMESTEPS = int(total_timesteps)
    base03222.CHECKPOINT_STEPS = [int(x) for x in checkpoint_steps]
    base03222.summarize_by_station = util04010.summarize_by_station_safe
    base03222.load_config = load_config
    base03222.base.StressAwareDiscreteWrapper = base04036.LateIrrigationReserveMaskWrapper
    base03222.shutil.copy2 = safe_copy2


def copy_with_task_names(out: Path, suffix: str) -> None:
    for old_rel, new_rel in RENAMES.items():
        old = out / old_rel
        if suffix:
            new_rel = new_rel.replace("042_10_", f"042_10_{suffix}_")
        new = out / new_rel
        if old.exists():
            new.parent.mkdir(parents=True, exist_ok=True)
            safe_copy2(old, new)
    generic_result = out / "032_22_result.json"
    result_path = out / ("042_10_result.json" if not suffix else f"042_10_{suffix}_result.json")
    if generic_result.exists():
        result = json.loads(generic_result.read_text(encoding="utf-8"))
        result.update(
            {
                "task": f"{TASK_ID}_{TASK_NAME}" + (f"_{suffix}" if suffix else ""),
                "input_root": LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
                "station_scope": SITES,
                "binary_timing_action_levels": {
                    "irrigation_levels": BINARY_IRRIGATION_LEVELS,
                    "nitrogen_levels": BINARY_NITROGEN_LEVELS,
                    "combined_action_count": len(BINARY_IRRIGATION_LEVELS) * len(BINARY_NITROGEN_LEVELS),
                    "reward_changed": False,
                    "safety_constraints_changed": False,
                },
                "inherited_from": "040_36_sya_lowIC_ppo_late_irrigation_reserve_mask",
            }
        )
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


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


def add_04210_summaries(out: Path, suffix: str) -> None:
    prefix = "042_10" if not suffix else f"042_10_{suffix}"
    eval_path = out / "evaluation" / f"{prefix}_checkpoint_validation_summary.csv"
    if not eval_path.exists():
        return
    eval_df = pd.read_csv(eval_path, keep_default_na=False)
    rows: list[dict[str, Any]] = []
    for _, row in eval_df.iterrows():
        daily_path = ROOT / str(row.get("daily_csv_path", ""))
        rec: dict[str, Any] = {
            "station_code": row.get("station_code"),
            "year": row.get("year"),
            "checkpoint_step": row.get("checkpoint_step"),
            "binary_irrigation_event_count": np.nan,
            "binary_nitrogen_event_count": np.nan,
            "has_non_binary_irrigation_amount": np.nan,
            "has_non_binary_nitrogen_amount": np.nan,
            "irrigation_dap1_30": np.nan,
            "irrigation_dap31_60": np.nan,
            "irrigation_dap61_90": np.nan,
            "irrigation_dap91_plus": np.nan,
            "nitrogen_dap1_30": np.nan,
            "nitrogen_dap31_60": np.nan,
            "nitrogen_dap61_90": np.nan,
        }
        if daily_path.exists():
            daily = pd.read_csv(daily_path)
            dap = pd.to_numeric(daily.get("dap", pd.Series(dtype=float)), errors="coerce")
            irr = pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
            n = pd.to_numeric(daily.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
            irr_positive = irr[irr > 1e-9]
            n_positive = n[n > 1e-9]
            rec.update(
                {
                    "binary_irrigation_event_count": int((irr > 1e-9).sum()),
                    "binary_nitrogen_event_count": int((n > 1e-9).sum()),
                    "has_non_binary_irrigation_amount": bool((~np.isclose(irr_positive, 45.0)).any()) if len(irr_positive) else False,
                    "has_non_binary_nitrogen_amount": bool((~np.isclose(n_positive, 80.0)).any()) if len(n_positive) else False,
                    "irrigation_dap1_30": float(irr[dap <= 30].sum()),
                    "irrigation_dap31_60": float(irr[(dap > 30) & (dap <= 60)].sum()),
                    "irrigation_dap61_90": float(irr[(dap > 60) & (dap <= 90)].sum()),
                    "irrigation_dap91_plus": float(irr[dap > 90].sum()),
                    "nitrogen_dap1_30": float(n[dap <= 30].sum()),
                    "nitrogen_dap31_60": float(n[(dap > 30) & (dap <= 60)].sum()),
                    "nitrogen_dap61_90": float(n[(dap > 60) & (dap <= 90)].sum()),
                }
            )
        rows.append(rec)
    extra_df = pd.DataFrame(rows)
    drop_cols = [c for c in extra_df.columns if c in eval_df.columns and c not in ["station_code", "year", "checkpoint_step"]]
    if drop_cols:
        eval_df = eval_df.drop(columns=drop_cols)
    merged = eval_df.merge(extra_df, on=["station_code", "year", "checkpoint_step"], how="left")
    merged.to_csv(eval_path, index=False, encoding="utf-8-sig")

    by_path = out / "evaluation" / f"{prefix}_validation_summary_by_station_checkpoint.csv"
    if by_path.exists():
        by_station = util04010.summarize_by_station_safe(merged)
        extra = (
            merged.groupby(["station_code", "checkpoint_step"], as_index=False)
            .agg(
                mean_binary_irrigation_event_count=("binary_irrigation_event_count", "mean"),
                mean_binary_nitrogen_event_count=("binary_nitrogen_event_count", "mean"),
                non_binary_irrigation_amount_count=("has_non_binary_irrigation_amount", "sum"),
                non_binary_nitrogen_amount_count=("has_non_binary_nitrogen_amount", "sum"),
                mean_irrigation_dap1_30=("irrigation_dap1_30", "mean"),
                mean_irrigation_dap31_60=("irrigation_dap31_60", "mean"),
                mean_irrigation_dap61_90=("irrigation_dap61_90", "mean"),
                mean_irrigation_dap91_plus=("irrigation_dap91_plus", "mean"),
                mean_nitrogen_dap1_30=("nitrogen_dap1_30", "mean"),
                mean_nitrogen_dap31_60=("nitrogen_dap31_60", "mean"),
                mean_nitrogen_dap61_90=("nitrogen_dap61_90", "mean"),
            )
        )
        by_station = by_station.merge(extra, on=["station_code", "checkpoint_step"], how="left")
        by_station.to_csv(by_path, index=False, encoding="utf-8-sig")


def write_clean_record(out: Path, doc: Path, suffix: str, total_timesteps: int, checkpoint_steps: list[int]) -> None:
    prefix = "042_10" if not suffix else f"042_10_{suffix}"
    split = read_csv_or_empty(out / "configs" / f"{prefix}_half_split_selection.csv")
    train = read_csv_or_empty(out / "evaluation" / f"{prefix}_training_checkpoint_inventory.csv")
    eval_df = read_csv_or_empty(out / "evaluation" / f"{prefix}_checkpoint_validation_summary.csv")
    by_station = read_csv_or_empty(out / "evaluation" / f"{prefix}_validation_summary_by_station_checkpoint.csv")
    split_show = split[[c for c in ["station_code", "site", "year", "split"] if c in split.columns]] if not split.empty else split

    lines = [
        f"# {prefix} SYA lowIC binary-timing MaskablePPO 记录",
        "",
        "## 结论边界",
        "",
        "- 本任务继承 040_36 的 lowIC 自由时序 PPO、reward 和安全约束。",
        "- 唯一核心改动：动作档位收缩为灌溉 [0,45] mm、施氮 [0,80] kg/ha。",
        "- 目的不是最终优化剂量，而是先检验 PPO 能否学习自由决策时机。",
        f"- 训练步数：`{total_timesteps}`；checkpoint：`{', '.join(map(str, checkpoint_steps))}`。",
        "",
        "## 固定配置",
        "",
        f"- lowIC 输入目录：`{LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix()}`",
        f"- 输入目录存在：`{LOWIC_INPUT_ROOT.exists()}`",
        f"- 灌溉档位：`{BINARY_IRRIGATION_LEVELS}`",
        f"- 施氮档位：`{BINARY_NITROGEN_LEVELS}`",
        f"- 组合动作数：`{len(BINARY_IRRIGATION_LEVELS) * len(BINARY_NITROGEN_LEVELS)}`",
        "- reward：继承 040_36/040_28，不新增 reward 项。",
        "- safety：继承 040_36，包括 7 天间隔、季节上限、DAP90 后禁氮、晚期灌溉保留。",
        "",
        "## 年份划分",
        "",
        md_table(split_show, 120),
        "",
        "## 训练 checkpoint 库存",
        "",
        md_table(train, 80),
        "",
        "## 验证集按 checkpoint 汇总",
        "",
        md_table(by_station, 80),
        "",
        "## 逐年验证结果",
        "",
        md_table(eval_df, 240),
        "",
    ]
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("\n".join(lines), encoding="utf-8")


def dry_run(total_timesteps: int, checkpoint_steps: list[int], suffix: str) -> None:
    out = out_for_suffix(suffix)
    doc = doc_for_suffix(suffix)
    patch_base_module(out, doc, total_timesteps, checkpoint_steps)
    cfg = load_config()
    split = base03222.load_split().copy()
    split = split[split["station_code"].eq(STATION)].sort_values(["year"]).reset_index(drop=True)
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}" + (f"_{suffix}" if suffix else ""),
        "mode": "dry_run",
        "station": STATION,
        "lowIC_input_root": LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
        "lowIC_input_root_exists": LOWIC_INPUT_ROOT.exists(),
        "prompt_exists": PROMPT.exists(),
        "split_years": split[["station_code", "site", "year", "split"]].to_dict(orient="records"),
        "total_timesteps": int(total_timesteps),
        "checkpoint_steps": checkpoint_steps,
        "discrete_actions": cfg["discrete_actions"],
        "action_safety": cfg["action_safety"],
        "reward": cfg["reward"],
        "combined_action_count": len(BINARY_IRRIGATION_LEVELS) * len(BINARY_NITROGEN_LEVELS),
        "next_step_allowed": bool(LOWIC_INPUT_ROOT.exists() and PROMPT.exists() and not split.empty),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


def run_training(total_timesteps: int, checkpoint_steps: list[int], suffix: str) -> None:
    out = out_for_suffix(suffix)
    doc = doc_for_suffix(suffix)
    patch_base_module(out, doc, total_timesteps, checkpoint_steps)
    old_input_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = LOWIC_INPUT_ROOT
    try:
        base03222.main()
        copy_with_task_names(out, suffix)
        add_04210_summaries(out, suffix)
        write_clean_record(out, doc, suffix, total_timesteps, checkpoint_steps)
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root

    prefix = "042_10" if not suffix else f"042_10_{suffix}"
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}" + (f"_{suffix}" if suffix else ""),
        "algorithm": "MaskablePPO",
        "record_md": doc.relative_to(ROOT).as_posix(),
        "train_inventory": (out / "evaluation" / f"{prefix}_training_checkpoint_inventory.csv").relative_to(ROOT).as_posix(),
        "validation_summary": (out / "evaluation" / f"{prefix}_checkpoint_validation_summary.csv").relative_to(ROOT).as_posix(),
        "by_checkpoint": (out / "evaluation" / f"{prefix}_validation_summary_by_station_checkpoint.csv").relative_to(ROOT).as_posix(),
        "total_timesteps": int(total_timesteps),
        "checkpoint_steps": checkpoint_steps,
        "input_root": LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
        "binary_irrigation_levels": BINARY_IRRIGATION_LEVELS,
        "binary_nitrogen_levels": BINARY_NITROGEN_LEVELS,
    }
    result_path = out / (f"{prefix}_wrapper_result.json")
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument("--checkpoint-steps", type=str, default=None)
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()
    checkpoint_steps = parse_checkpoint_steps(args.checkpoint_steps, args.timesteps)
    if args.dry_run:
        dry_run(args.timesteps, checkpoint_steps, args.suffix)
    else:
        run_training(args.timesteps, checkpoint_steps, args.suffix)


if __name__ == "__main__":
    main()
