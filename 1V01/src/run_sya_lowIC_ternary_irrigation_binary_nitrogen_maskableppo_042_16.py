"""042_16: SYA lowIC ternary-irrigation / binary-nitrogen MaskablePPO.

This is a narrow fork of 042_10.  It keeps the same lowIC input root, reward,
MaskablePPO training framework, safety masks, and train/validation split, but
expands only the irrigation dose grid:

    irrigation: [0, 30, 45] mm
    nitrogen:   [0, 80] kg/ha

The intent is to test whether the 042_10 binary 45 mm irrigation dose is too
coarse while avoiding a full return to the earlier dense action grid that
caused frequent small operations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_sya_lowIC_binary_timing_maskableppo_042_10 as base04210


TASK_ID = "042_16"
TASK_NAME = "sya_lowIC_ternary_irrigation_binary_nitrogen_maskableppo"
BASE_OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
BASE_DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"

TERNARY_IRRIGATION_LEVELS = [0.0, 30.0, 45.0]
BINARY_NITROGEN_LEVELS = [0.0, 80.0]


def patch_04210_globals() -> None:
    """Redirect the 042_10 reusable runner to the 042_16 task namespace."""

    base04210.TASK_ID = TASK_ID
    base04210.TASK_NAME = TASK_NAME
    base04210.BASE_OUT = BASE_OUT
    base04210.BASE_DOC = BASE_DOC
    base04210.PROMPT = PROMPT
    base04210.BINARY_IRRIGATION_LEVELS = list(TERNARY_IRRIGATION_LEVELS)
    base04210.BINARY_NITROGEN_LEVELS = list(BINARY_NITROGEN_LEVELS)
    base04210.RENAMES = {
        "configs/032_22_half_split_selection.csv": "configs/042_16_half_split_selection.csv",
        "evaluation/032_22_training_checkpoint_inventory_partial.csv": "evaluation/042_16_training_checkpoint_inventory_partial.csv",
        "evaluation/032_22_checkpoint_validation_summary_partial.csv": "evaluation/042_16_checkpoint_validation_summary_partial.csv",
        "evaluation/032_22_training_checkpoint_inventory.csv": "evaluation/042_16_training_checkpoint_inventory.csv",
        "evaluation/032_22_checkpoint_validation_summary.csv": "evaluation/042_16_checkpoint_validation_summary.csv",
        "evaluation/032_22_validation_summary_by_station_checkpoint.csv": "evaluation/042_16_validation_summary_by_station_checkpoint.csv",
        "logs/032_22_training_year_reset_counts.csv": "logs/042_16_training_year_reset_counts.csv",
        "032_22_result.json": "042_16_result.json",
    }


def load_config() -> dict[str, Any]:
    cfg = base04210.base04036.load_config()
    cfg["discrete_actions"]["irrigation_levels"] = list(TERNARY_IRRIGATION_LEVELS)
    cfg["discrete_actions"]["nitrogen_levels"] = list(BINARY_NITROGEN_LEVELS)
    cfg["discrete_actions"]["ternary_irrigation_binary_nitrogen_source"] = (
        "042_16 expands only irrigation dose from [0,45] to [0,30,45] while "
        "keeping nitrogen [0,80], reward, lowIC inputs, and safety masks inherited from 042_10/040_36."
    )
    return cfg


def patch_base_module(out: Path, doc: Path, total_timesteps: int, checkpoint_steps: list[int]) -> None:
    base04210.patch_base_module(out, doc, total_timesteps, checkpoint_steps)
    base04210.base03222.load_config = load_config


def dry_run(total_timesteps: int, checkpoint_steps: list[int], suffix: str) -> None:
    patch_04210_globals()
    out = base04210.out_for_suffix(suffix)
    doc = base04210.doc_for_suffix(suffix)
    patch_base_module(out, doc, total_timesteps, checkpoint_steps)
    cfg = load_config()
    split = base04210.base03222.load_split().copy()
    split = split[split["station_code"].eq(base04210.STATION)].sort_values(["year"]).reset_index(drop=True)
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}" + (f"_{suffix}" if suffix else ""),
        "mode": "dry_run",
        "station": base04210.STATION,
        "lowIC_input_root": base04210.LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
        "lowIC_input_root_exists": base04210.LOWIC_INPUT_ROOT.exists(),
        "prompt_exists": PROMPT.exists(),
        "split_years": split[["station_code", "site", "year", "split"]].to_dict(orient="records"),
        "total_timesteps": int(total_timesteps),
        "checkpoint_steps": checkpoint_steps,
        "discrete_actions": cfg["discrete_actions"],
        "action_safety": cfg["action_safety"],
        "reward": cfg["reward"],
        "combined_action_count": len(TERNARY_IRRIGATION_LEVELS) * len(BINARY_NITROGEN_LEVELS),
        "only_planned_change_vs_04210": "irrigation_levels [0,45] -> [0,30,45]",
        "next_step_allowed": bool(base04210.LOWIC_INPUT_ROOT.exists() and PROMPT.exists() and not split.empty),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


def run_training(total_timesteps: int, checkpoint_steps: list[int], suffix: str) -> None:
    patch_04210_globals()
    out = base04210.out_for_suffix(suffix)
    doc = base04210.doc_for_suffix(suffix)
    patch_base_module(out, doc, total_timesteps, checkpoint_steps)
    old_input_root = base04210.ppo_safe_rendering.MULTISITE_INPUT_ROOT
    base04210.ppo_safe_rendering.MULTISITE_INPUT_ROOT = base04210.LOWIC_INPUT_ROOT
    try:
        base04210.base03222.main()
        base04210.copy_with_task_names(out, suffix)
        base04210.add_04210_summaries(out, suffix)
        write_clean_record(out, doc, suffix, total_timesteps, checkpoint_steps)
    finally:
        base04210.ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root

    prefix = TASK_ID if not suffix else f"{TASK_ID}_{suffix}"
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}" + (f"_{suffix}" if suffix else ""),
        "algorithm": "MaskablePPO",
        "record_md": doc.relative_to(ROOT).as_posix(),
        "train_inventory": (out / "evaluation" / f"{prefix}_training_checkpoint_inventory.csv").relative_to(ROOT).as_posix(),
        "validation_summary": (out / "evaluation" / f"{prefix}_checkpoint_validation_summary.csv").relative_to(ROOT).as_posix(),
        "by_checkpoint": (out / "evaluation" / f"{prefix}_validation_summary_by_station_checkpoint.csv").relative_to(ROOT).as_posix(),
        "total_timesteps": int(total_timesteps),
        "checkpoint_steps": checkpoint_steps,
        "input_root": base04210.LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
        "irrigation_levels": TERNARY_IRRIGATION_LEVELS,
        "nitrogen_levels": BINARY_NITROGEN_LEVELS,
        "combined_action_count": len(TERNARY_IRRIGATION_LEVELS) * len(BINARY_NITROGEN_LEVELS),
        "only_planned_change_vs_04210": "irrigation_levels [0,45] -> [0,30,45]",
    }
    result_path = out / f"{prefix}_wrapper_result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def md_table(df: pd.DataFrame, max_rows: int = 120) -> str:
    return base04210.md_table(df, max_rows=max_rows)


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    return base04210.read_csv_or_empty(path)


def write_clean_record(out: Path, doc: Path, suffix: str, total_timesteps: int, checkpoint_steps: list[int]) -> None:
    prefix = TASK_ID if not suffix else f"{TASK_ID}_{suffix}"
    split = read_csv_or_empty(out / "configs" / f"{prefix}_half_split_selection.csv")
    train = read_csv_or_empty(out / "evaluation" / f"{prefix}_training_checkpoint_inventory.csv")
    eval_df = read_csv_or_empty(out / "evaluation" / f"{prefix}_checkpoint_validation_summary.csv")
    by_station = read_csv_or_empty(out / "evaluation" / f"{prefix}_validation_summary_by_station_checkpoint.csv")
    split_show = split[[c for c in ["station_code", "site", "year", "split"] if c in split.columns]] if not split.empty else split

    lines = [
        f"# {prefix} SYA lowIC 三档灌溉/二档施氮 MaskablePPO 记录",
        "",
        "## 结论边界",
        "",
        "- 本任务继承 042_10 的 SYA lowIC 自由时序 MaskablePPO 主线。",
        "- 唯一核心改动：灌溉档位从 `[0,45]` 扩展为 `[0,30,45]`；施氮档位保持 `[0,80]`。",
        "- reward、lowIC 输入、训练/验证年份划分、安全约束、后期灌溉 reserve mask 均保持不变。",
        "- 本任务的目的不是重新调 reward，而是检查 042_10 的 45 mm 单一灌溉剂量是否过粗。",
        f"- 训练步数：`{total_timesteps}`；checkpoint：`{', '.join(map(str, checkpoint_steps))}`。",
        "",
        "## 固定配置",
        "",
        f"- lowIC 输入目录：`{base04210.LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix()}`",
        f"- 输入目录存在：`{base04210.LOWIC_INPUT_ROOT.exists()}`",
        f"- 灌溉档位：`{TERNARY_IRRIGATION_LEVELS}`",
        f"- 施氮档位：`{BINARY_NITROGEN_LEVELS}`",
        f"- 组合动作数：`{len(TERNARY_IRRIGATION_LEVELS) * len(BINARY_NITROGEN_LEVELS)}`",
        "- safety：继承 040_36，包括 7 天间隔、单季上限、DAP90 后禁氮、后期灌溉保留 mask。",
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
        "## 后续判断",
        "",
        "- 若 042_16 比 042_10/042_12 指标更好且措施不重新碎片化，说明增加 30 mm 灌溉档位有价值。",
        "- 若 042_16 出现频繁小灌或模板化加重，则说明剂量自由度会重新带来措施合理性风险，不应继续盲目扩展动作档位。",
    ]
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_checkpoint_steps(raw: str | None, timesteps: int) -> list[int]:
    return base04210.parse_checkpoint_steps(raw, timesteps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timesteps", type=int, default=base04210.DEFAULT_TOTAL_TIMESTEPS)
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
