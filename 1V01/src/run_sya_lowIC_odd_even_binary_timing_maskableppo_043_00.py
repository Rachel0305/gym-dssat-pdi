"""043_00: SYA lowIC odd-year train / even-year validation MaskablePPO.

This is a narrow split-design fork of 042_10.  It keeps the lowIC input root,
free-timing MaskablePPO framework, binary timing action grid, reward, and safety
masks unchanged.  The only scientific change is the year split:

    odd years  -> train
    even years -> validation

The goal is to test whether the previous front-half/back-half split caused
unnecessarily strong weather distribution shift and encouraged template-like
policies.
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


TASK_ID = "043_00"
TASK_NAME = "sya_lowIC_odd_even_binary_timing_maskableppo"
BASE_OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
BASE_DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"

ORIGINAL_LOAD_SPLIT = base04210.base03222.load_split


RENAMES = {
    "configs/032_22_half_split_selection.csv": "configs/043_00_odd_even_selection.csv",
    "evaluation/032_22_training_checkpoint_inventory_partial.csv": "evaluation/043_00_training_checkpoint_inventory_partial.csv",
    "evaluation/032_22_checkpoint_validation_summary_partial.csv": "evaluation/043_00_checkpoint_validation_summary_partial.csv",
    "evaluation/032_22_training_checkpoint_inventory.csv": "evaluation/043_00_training_checkpoint_inventory.csv",
    "evaluation/032_22_checkpoint_validation_summary.csv": "evaluation/043_00_checkpoint_validation_summary.csv",
    "evaluation/032_22_validation_summary_by_station_checkpoint.csv": "evaluation/043_00_validation_summary_by_station_checkpoint.csv",
    "logs/032_22_training_year_reset_counts.csv": "logs/043_00_training_year_reset_counts.csv",
    "032_22_result.json": "043_00_result.json",
}


def out_for_suffix(suffix: str) -> Path:
    return ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}_{suffix}" if suffix else BASE_OUT


def doc_for_suffix(suffix: str) -> Path:
    return ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_{suffix}_record.md" if suffix else BASE_DOC


def task_prefix(suffix: str) -> str:
    return TASK_ID if not suffix else f"{TASK_ID}_{suffix}"


def load_odd_even_split() -> pd.DataFrame:
    """Load the original year inventory and rewrite only the split column."""

    split = ORIGINAL_LOAD_SPLIT().copy()
    split = split[split["station_code"].eq(base04210.STATION)].copy()
    split["year"] = pd.to_numeric(split["year"], errors="coerce").astype(int)
    split["split"] = split["year"].apply(lambda y: "train" if int(y) % 2 == 1 else "validation")
    split["selection_reason"] = "043_00_odd_year_train_even_year_validation"
    return split.sort_values(["station_code", "year"]).reset_index(drop=True)


def patch_04210_globals() -> None:
    """Redirect reusable 042_10 runner names to the 043_00 namespace."""

    base04210.TASK_ID = TASK_ID
    base04210.TASK_NAME = TASK_NAME
    base04210.BASE_OUT = BASE_OUT
    base04210.BASE_DOC = BASE_DOC
    base04210.PROMPT = PROMPT
    base04210.RENAMES = dict(RENAMES)


def patch_base_module(out: Path, doc: Path, total_timesteps: int, checkpoint_steps: list[int]) -> None:
    patch_04210_globals()
    base04210.patch_base_module(out, doc, total_timesteps, checkpoint_steps)
    base04210.base03222.load_split = load_odd_even_split


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


def copy_with_task_names(out: Path, suffix: str) -> None:
    base04210.copy_with_task_names(out, suffix)
    generic_result = out / "032_22_result.json"
    result_path = out / (f"{task_prefix(suffix)}_result.json")
    if generic_result.exists():
        result = json.loads(generic_result.read_text(encoding="utf-8"))
        result.update(
            {
                "task": f"{TASK_ID}_{TASK_NAME}" + (f"_{suffix}" if suffix else ""),
                "split_design": "odd_year_train_even_year_validation",
                "only_planned_change_vs_04210": "year split changed from front-half/back-half to odd/even",
                "input_root": base04210.LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
                "station_scope": [base04210.STATION],
                "binary_timing_action_levels": {
                    "irrigation_levels": list(base04210.BINARY_IRRIGATION_LEVELS),
                    "nitrogen_levels": list(base04210.BINARY_NITROGEN_LEVELS),
                    "combined_action_count": len(base04210.BINARY_IRRIGATION_LEVELS)
                    * len(base04210.BINARY_NITROGEN_LEVELS),
                    "reward_changed": False,
                    "safety_constraints_changed": False,
                    "action_grid_changed": False,
                },
            }
        )
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def write_clean_record(out: Path, doc: Path, suffix: str, total_timesteps: int, checkpoint_steps: list[int]) -> None:
    prefix = task_prefix(suffix)
    split = read_csv_or_empty(out / "configs" / f"{prefix}_odd_even_selection.csv")
    train = read_csv_or_empty(out / "evaluation" / f"{prefix}_training_checkpoint_inventory.csv")
    eval_df = read_csv_or_empty(out / "evaluation" / f"{prefix}_checkpoint_validation_summary.csv")
    by_station = read_csv_or_empty(out / "evaluation" / f"{prefix}_validation_summary_by_station_checkpoint.csv")
    reset = read_csv_or_empty(out / "logs" / f"{prefix}_training_year_reset_counts.csv")

    split_show = split[[c for c in ["station_code", "site", "year", "split"] if c in split.columns]] if not split.empty else split

    lines = [
        f"# {prefix} SYA lowIC 奇数年训练 / 偶数年验证 binary-timing MaskablePPO 记录",
        "",
        "## 结论边界",
        "",
        "- 本任务继承 042_10 的 SYA lowIC 自由时序 binary-timing MaskablePPO。",
        "- 唯一计划改动：年份划分从前半段训练/后半段验证，改为奇数年训练、偶数年验证。",
        "- 不改变 reward、不改变 lowIC 输入、不改变动作档位、不改变 7 天间隔、季节上限、DAP90 后禁氮和后期灌溉 reserve mask。",
        "- 本任务用于判断交错年份划分是否能减轻时间分布偏移，并检查 PPO 是否更能根据年份气象/土壤差异改变措施。",
        f"- 训练步数：`{total_timesteps}`；checkpoint：`{', '.join(map(str, checkpoint_steps))}`。",
        "",
        "## 固定配置",
        "",
        f"- 输入目录：`{base04210.LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix()}`",
        f"- 输入目录存在：`{base04210.LOWIC_INPUT_ROOT.exists()}`",
        f"- 灌溉档位：`{list(base04210.BINARY_IRRIGATION_LEVELS)}`",
        f"- 施氮档位：`{list(base04210.BINARY_NITROGEN_LEVELS)}`",
        f"- 组合动作数：`{len(base04210.BINARY_IRRIGATION_LEVELS) * len(base04210.BINARY_NITROGEN_LEVELS)}`",
        "- 解释重点：若指标改善但管理序列仍高度模板化，只能说明 split 改善性能，不能说明 PPO 已经充分学会天气响应。",
        "",
        "## 年份划分",
        "",
        md_table(split_show, 120),
        "",
        "## 训练年份 reset 计数",
        "",
        md_table(reset, 120),
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
        "## 下一步判读",
        "",
        "- 若偶数验证年中 2014/2016/2018/2020/2022 等年份指标和措施响应性同步改善，说明前半段/后半段 split 可能确实造成了过强分布偏移。",
        "- 若仍出现固定灌溉/施氮模板，则下一步应继续做输入敏感性或天气响应 guardrail，而不是只靠重新划分年份。",
        "- 若结果明显退化，则保留 042_10/042_12 冻结线，043_00 仅作为 split 对照阴性结果。",
        "",
    ]
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("\n".join(lines), encoding="utf-8")


def dry_run(total_timesteps: int, checkpoint_steps: list[int], suffix: str) -> None:
    out = out_for_suffix(suffix)
    doc = doc_for_suffix(suffix)
    patch_base_module(out, doc, total_timesteps, checkpoint_steps)
    cfg = base04210.load_config()
    split = load_odd_even_split()
    train_years = split[split["split"].eq("train")]["year"].astype(int).tolist()
    validation_years = split[split["split"].eq("validation")]["year"].astype(int).tolist()
    result: dict[str, Any] = {
        "task": f"{TASK_ID}_{TASK_NAME}" + (f"_{suffix}" if suffix else ""),
        "mode": "dry_run",
        "station": base04210.STATION,
        "lowIC_input_root": base04210.LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
        "lowIC_input_root_exists": base04210.LOWIC_INPUT_ROOT.exists(),
        "prompt_exists": PROMPT.exists(),
        "split_design": "odd_year_train_even_year_validation",
        "train_years": train_years,
        "validation_years": validation_years,
        "train_year_count": len(train_years),
        "validation_year_count": len(validation_years),
        "split_years": split[["station_code", "site", "year", "split"]].to_dict(orient="records"),
        "total_timesteps": int(total_timesteps),
        "checkpoint_steps": checkpoint_steps,
        "discrete_actions": cfg["discrete_actions"],
        "action_safety": cfg["action_safety"],
        "reward": cfg["reward"],
        "combined_action_count": len(base04210.BINARY_IRRIGATION_LEVELS) * len(base04210.BINARY_NITROGEN_LEVELS),
        "only_planned_change_vs_04210": "year split changed from front-half/back-half to odd/even",
        "next_step_allowed": bool(base04210.LOWIC_INPUT_ROOT.exists() and PROMPT.exists() and not split.empty),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


def run_training(total_timesteps: int, checkpoint_steps: list[int], suffix: str) -> None:
    out = out_for_suffix(suffix)
    doc = doc_for_suffix(suffix)
    patch_base_module(out, doc, total_timesteps, checkpoint_steps)
    old_input_root = base04210.ppo_safe_rendering.MULTISITE_INPUT_ROOT
    base04210.ppo_safe_rendering.MULTISITE_INPUT_ROOT = base04210.LOWIC_INPUT_ROOT
    try:
        base04210.base03222.main()
        copy_with_task_names(out, suffix)
        base04210.add_04210_summaries(out, suffix)
        write_clean_record(out, doc, suffix, total_timesteps, checkpoint_steps)
    finally:
        base04210.ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root

    prefix = task_prefix(suffix)
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
        "split_design": "odd_year_train_even_year_validation",
        "binary_irrigation_levels": list(base04210.BINARY_IRRIGATION_LEVELS),
        "binary_nitrogen_levels": list(base04210.BINARY_NITROGEN_LEVELS),
    }
    result_path = out / f"{prefix}_wrapper_result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


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
