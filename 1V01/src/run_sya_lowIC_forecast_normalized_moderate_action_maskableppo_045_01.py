"""045_01: SYA lowIC forecast-normalized MaskablePPO with moderate action expansion.

This is a narrow, auditable wrapper around 043_02.  It keeps the normalized
weather/perfect-forecast observation path from 043_02 and changes only the
discrete action grid:

    irrigation: [0, 30, 45] mm
    nitrogen:   [0, 40, 80] kg/ha

The intent is to test whether the current 042_15 frozen candidate can keep its
metric/action quality while addressing two clear limitations: coarse binary
doses and absent useful forecast benefit.
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

import run_sya_lowIC_binary_timing_forecast_normalized_maskableppo_043_02 as base04302
import run_sya_lowIC_binary_timing_maskableppo_042_10 as base04210


TASK_ID = "045_01"
TASK_NAME = "sya_lowIC_forecast_normalized_moderate_action_maskableppo"
BASE_OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
BASE_DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"

IRRIGATION_LEVELS = [0.0, 30.0, 45.0]
NITROGEN_LEVELS = [0.0, 40.0, 80.0]

RENAMES = {
    "configs/032_22_half_split_selection.csv": "configs/045_01_half_split_selection.csv",
    "evaluation/032_22_training_checkpoint_inventory_partial.csv": "evaluation/045_01_training_checkpoint_inventory_partial.csv",
    "evaluation/032_22_checkpoint_validation_summary_partial.csv": "evaluation/045_01_checkpoint_validation_summary_partial.csv",
    "evaluation/032_22_training_checkpoint_inventory.csv": "evaluation/045_01_training_checkpoint_inventory.csv",
    "evaluation/032_22_checkpoint_validation_summary.csv": "evaluation/045_01_checkpoint_validation_summary.csv",
    "evaluation/032_22_validation_summary_by_station_checkpoint.csv": "evaluation/045_01_validation_summary_by_station_checkpoint.csv",
    "logs/032_22_training_year_reset_counts.csv": "logs/045_01_training_year_reset_counts.csv",
    "032_22_result.json": "045_01_result.json",
}


def patch_globals() -> None:
    """Redirect 043_02 reusable code to the 045_01 namespace/action grid."""

    base04302.TASK_ID = TASK_ID
    base04302.TASK_NAME = TASK_NAME
    base04302.BASE_OUT = BASE_OUT
    base04302.BASE_DOC = BASE_DOC
    base04302.PROMPT = PROMPT
    base04302.RENAMES = RENAMES
    base04210.BINARY_IRRIGATION_LEVELS = list(IRRIGATION_LEVELS)
    base04210.BINARY_NITROGEN_LEVELS = list(NITROGEN_LEVELS)


def out_for_suffix(suffix: str) -> Path:
    return ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}_{suffix}" if suffix else BASE_OUT


def doc_for_suffix(suffix: str) -> Path:
    return ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_{suffix}_record.md" if suffix else BASE_DOC


def load_config() -> dict[str, Any]:
    cfg = base04302.load_config()
    cfg["discrete_actions"]["irrigation_levels"] = list(IRRIGATION_LEVELS)
    cfg["discrete_actions"]["nitrogen_levels"] = list(NITROGEN_LEVELS)
    cfg["discrete_actions"]["045_01_moderate_action_expansion"] = {
        "enabled": True,
        "irrigation_levels": list(IRRIGATION_LEVELS),
        "nitrogen_levels": list(NITROGEN_LEVELS),
        "combined_action_count": len(IRRIGATION_LEVELS) * len(NITROGEN_LEVELS),
        "rationale": (
            "Moderately expand the frozen 042_15 binary action grid without returning "
            "to the earlier dense small-dose grid that caused fragmented operations."
        ),
    }
    return cfg


def patch_base_module(out: Path, doc: Path, total_timesteps: int, checkpoint_steps: list[int]) -> None:
    patch_globals()
    base04302.patch_base_module(out, doc, total_timesteps, checkpoint_steps)
    base04302.base03222.load_config = load_config


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


def write_clean_record(out: Path, doc: Path, suffix: str, total_timesteps: int, checkpoint_steps: list[int], smoke_csv: Path) -> None:
    prefix = TASK_ID if not suffix else f"{TASK_ID}_{suffix}"
    split = read_csv_or_empty(out / "configs" / f"{prefix}_half_split_selection.csv")
    train = read_csv_or_empty(out / "evaluation" / f"{prefix}_training_checkpoint_inventory.csv")
    eval_df = read_csv_or_empty(out / "evaluation" / f"{prefix}_checkpoint_validation_summary.csv")
    by_checkpoint = read_csv_or_empty(out / "evaluation" / f"{prefix}_validation_summary_by_station_checkpoint.csv")
    smoke = read_csv_or_empty(smoke_csv)
    split_show = split[[c for c in ["station_code", "site", "year", "split"] if c in split.columns]] if not split.empty else split

    lines = [
        f"# {prefix} SYA lowIC 天气预报归一化 + 温和扩展动作组 MaskablePPO 记录",
        "",
        "## 结论边界",
        "",
        "- 本任务复用 043_02 的天气/完美预报/归一化 observation。",
        "- 本任务只把动作组温和扩展为灌溉 `[0,30,45]`、施氮 `[0,40,80]`。",
        "- 不改 lowIC 输入、不改训练/验证年份、不使用 teacher warm-start、不做 reward 系数扫描。",
        "- 本任务目标不是必须超过 042_15，而是检查能否在补足动作粒度和天气输入两个缺点后，维持 042_15 的可用水平。",
        f"- 训练步数：`{total_timesteps}`；checkpoint：`{', '.join(map(str, checkpoint_steps))}`。",
        "",
        "## 固定配置",
        "",
        f"- lowIC 输入目录：`{base04302.LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix()}`",
        f"- 灌溉动作：`{IRRIGATION_LEVELS}`",
        f"- 施氮动作：`{NITROGEN_LEVELS}`",
        f"- 组合动作数：`{len(IRRIGATION_LEVELS) * len(NITROGEN_LEVELS)}`",
        f"- observation 维度：`{len(base04302.ENHANCED_OBS_NAMES)}`",
        f"- 天气/预报变量：`{base04302.WEATHER_FEATURE_NAMES}`",
        "",
        "## Observation smoke 审计",
        "",
        md_table(smoke, 30),
        "",
        "## 年份划分",
        "",
        md_table(split_show, 80),
        "",
        "## 训练 checkpoint 库存",
        "",
        md_table(train, 80),
        "",
        "## 验证集按 checkpoint 汇总",
        "",
        md_table(by_checkpoint, 80),
        "",
        "## 逐年验证结果",
        "",
        md_table(eval_df, 240),
        "",
        "## 后续判读",
        "",
        "- 若指标接近 042_15，且措施没有重新碎片化，则 045_01 可作为更完整但仍简洁的候选配置。",
        "- 若明显不如 042_15，则冻结 042_15，不再把动作组/天气输入强行合并到主线。",
    ]
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_with_task_names(out: Path, suffix: str) -> None:
    for old_rel, new_rel in RENAMES.items():
        old = out / old_rel
        if suffix:
            new_rel = new_rel.replace("045_01_", f"045_01_{suffix}_")
        new = out / new_rel
        if old.exists():
            new.parent.mkdir(parents=True, exist_ok=True)
            base04302.safe_copy2(old, new)
    generic_result = out / "032_22_result.json"
    result_path = out / ("045_01_result.json" if not suffix else f"045_01_{suffix}_result.json")
    if generic_result.exists():
        result = json.loads(generic_result.read_text(encoding="utf-8"))
        result.update(
            {
                "task": f"{TASK_ID}_{TASK_NAME}" + (f"_{suffix}" if suffix else ""),
                "input_root": base04302.LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
                "station_scope": base04302.SITES,
                "action_levels": {
                    "irrigation_levels": IRRIGATION_LEVELS,
                    "nitrogen_levels": NITROGEN_LEVELS,
                    "combined_action_count": len(IRRIGATION_LEVELS) * len(NITROGEN_LEVELS),
                },
                "enhanced_observation_dim": len(base04302.ENHANCED_OBS_NAMES),
                "weather_forecast_features": base04302.WEATHER_FEATURE_NAMES,
                "teacher_warmstart_enabled": False,
            }
        )
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def dry_run(total_timesteps: int, checkpoint_steps: list[int], suffix: str) -> None:
    out = out_for_suffix(suffix)
    doc = doc_for_suffix(suffix)
    patch_base_module(out, doc, total_timesteps, checkpoint_steps)
    cfg = load_config()
    split = base04302.base03222.load_split().copy()
    split = split[split["station_code"].astype(str).eq(base04302.STATION)].sort_values(["year"]).reset_index(drop=True)
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}" + (f"_{suffix}" if suffix else ""),
        "mode": "dry_run",
        "station": base04302.STATION,
        "lowIC_input_root": base04302.LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
        "lowIC_input_root_exists": base04302.LOWIC_INPUT_ROOT.exists(),
        "prompt_exists": PROMPT.exists(),
        "split_years": split[["station_code", "site", "year", "split"]].to_dict(orient="records"),
        "total_timesteps": int(total_timesteps),
        "checkpoint_steps": checkpoint_steps,
        "discrete_actions": cfg["discrete_actions"],
        "action_safety": cfg["action_safety"],
        "reward": cfg["reward"],
        "enhanced_observation_dim": len(base04302.ENHANCED_OBS_NAMES),
        "weather_forecast_features": base04302.WEATHER_FEATURE_NAMES,
        "teacher_warmstart_enabled": False,
        "next_step_allowed": bool(base04302.LOWIC_INPUT_ROOT.exists() and PROMPT.exists() and not split.empty),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


def run_training(total_timesteps: int, checkpoint_steps: list[int], suffix: str) -> None:
    out = out_for_suffix(suffix)
    doc = doc_for_suffix(suffix)
    patch_base_module(out, doc, total_timesteps, checkpoint_steps)
    out.mkdir(parents=True, exist_ok=True)
    smoke_pass, smoke_csv = base04302.observation_smoke_audit(out)
    if not smoke_pass:
        raise RuntimeError(f"045_01 observation smoke failed: {smoke_csv.relative_to(ROOT).as_posix()}")

    old_input_root = base04302.ppo_safe_rendering.MULTISITE_INPUT_ROOT
    base04302.ppo_safe_rendering.MULTISITE_INPUT_ROOT = base04302.LOWIC_INPUT_ROOT
    try:
        base04302.base03222.main()
        copy_with_task_names(out, suffix)
        write_clean_record(out, doc, suffix, total_timesteps, checkpoint_steps, smoke_csv)
    finally:
        base04302.ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root
        base04302.base03222.base.make_env = base04302.ORIGINAL_BASE_MAKE_ENV

    prefix = TASK_ID if not suffix else f"{TASK_ID}_{suffix}"
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}" + (f"_{suffix}" if suffix else ""),
        "algorithm": "MaskablePPO",
        "record_md": doc.relative_to(ROOT).as_posix(),
        "observation_smoke_csv": smoke_csv.relative_to(ROOT).as_posix(),
        "train_inventory": (out / "evaluation" / f"{prefix}_training_checkpoint_inventory.csv").relative_to(ROOT).as_posix(),
        "validation_summary": (out / "evaluation" / f"{prefix}_checkpoint_validation_summary.csv").relative_to(ROOT).as_posix(),
        "by_checkpoint": (out / "evaluation" / f"{prefix}_validation_summary_by_station_checkpoint.csv").relative_to(ROOT).as_posix(),
        "total_timesteps": int(total_timesteps),
        "checkpoint_steps": checkpoint_steps,
        "input_root": base04302.LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
        "irrigation_levels": IRRIGATION_LEVELS,
        "nitrogen_levels": NITROGEN_LEVELS,
        "enhanced_observation_dim": len(base04302.ENHANCED_OBS_NAMES),
        "teacher_warmstart_enabled": False,
    }
    result_path = out / f"{prefix}_wrapper_result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def parse_checkpoint_steps(raw: str | None, total_timesteps: int) -> list[int]:
    return base04302.parse_checkpoint_steps(raw, total_timesteps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timesteps", type=int, default=base04302.DEFAULT_TOTAL_TIMESTEPS)
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
