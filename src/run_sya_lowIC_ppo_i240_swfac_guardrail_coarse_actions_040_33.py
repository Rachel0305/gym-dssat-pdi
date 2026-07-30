"""040_33: SYA lowIC PPO I240 + SWFAC guardrail + coarse action levels.

This task returns to the 040_28 reward and masks, but removes the smallest
non-zero action levels to prevent fragmented 15 mm irrigation and 40 kg/ha N
applications.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import ppo_safe_rendering
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base03222
import run_sya_lowIC_ppo_early_irrigation_reserve_penalty_040_10 as util04010
import run_sya_lowIC_ppo_i240_staged_reserve_040_26 as base04026
import run_sya_lowIC_ppo_i240_swfac_guardrail_reward_040_28 as base04028


ORIGINAL_SHUTIL_COPY2 = shutil.copy2
TASK_ID = "040_33"
TASK_NAME = "sya_lowIC_ppo_i240_swfac_guardrail_coarse_actions"
BASE_OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
BASE_DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"

LOWIC_INPUT_ROOT = base04028.LOWIC_INPUT_ROOT
STATION = base04028.STATION
SITES = [STATION]
DEFAULT_TOTAL_TIMESTEPS = 100_000
DEFAULT_CHECKPOINT_STEPS = [25_000, 50_000, 75_000, 100_000]

IRRIGATION_LEVELS = [0.0, 30.0, 45.0]
NITROGEN_LEVELS = [0.0, 80.0, 120.0]

RENAMES = {
    "configs/032_22_half_split_selection.csv": "configs/040_33_half_split_selection.csv",
    "evaluation/032_22_training_checkpoint_inventory_partial.csv": "evaluation/040_33_training_checkpoint_inventory_partial.csv",
    "evaluation/032_22_checkpoint_validation_summary_partial.csv": "evaluation/040_33_checkpoint_validation_summary_partial.csv",
    "evaluation/032_22_training_checkpoint_inventory.csv": "evaluation/040_33_training_checkpoint_inventory.csv",
    "evaluation/032_22_checkpoint_validation_summary.csv": "evaluation/040_33_checkpoint_validation_summary.csv",
    "evaluation/032_22_validation_summary_by_station_checkpoint.csv": "evaluation/040_33_validation_summary_by_station_checkpoint.csv",
    "logs/032_22_training_year_reset_counts.csv": "logs/040_33_training_year_reset_counts.csv",
    "032_22_result.json": "040_33_result.json",
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
    cfg = base04028.load_config()
    cfg["discrete_actions"]["irrigation_levels"] = list(IRRIGATION_LEVELS)
    cfg["discrete_actions"]["nitrogen_levels"] = list(NITROGEN_LEVELS)
    cfg["discrete_actions"]["coarse_action_source"] = "040_33 removes 15 mm and 40 kg/ha fragmented action levels; reward unchanged from 040_28"
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
    base03222.base.StressAwareDiscreteWrapper = base04028.I240SwfacGuardrailWrapper
    base03222.shutil.copy2 = safe_copy2


def copy_with_task_names(out: Path, suffix: str) -> None:
    for old_rel, new_rel in RENAMES.items():
        old = out / old_rel
        if suffix:
            new_rel = new_rel.replace("040_33_", f"040_33_{suffix}_")
        new = out / new_rel
        if old.exists():
            new.parent.mkdir(parents=True, exist_ok=True)
            safe_copy2(old, new)
    generic_result = out / "032_22_result.json"
    result_path = out / ("040_33_result.json" if not suffix else f"040_33_{suffix}_result.json")
    if generic_result.exists():
        result = json.loads(generic_result.read_text(encoding="utf-8"))
        result.update(
            {
                "task": f"{TASK_ID}_{TASK_NAME}" + (f"_{suffix}" if suffix else ""),
                "input_root": LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
                "station_scope": SITES,
                "coarse_action_levels": {
                    "irrigation_levels": IRRIGATION_LEVELS,
                    "nitrogen_levels": NITROGEN_LEVELS,
                    "reward_changed": False,
                },
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


def add_04033_summaries(out: Path, suffix: str) -> None:
    prefix = "040_33" if not suffix else f"040_33_{suffix}"
    eval_path = out / "evaluation" / f"{prefix}_checkpoint_validation_summary.csv"
    if not eval_path.exists():
        return
    eval_df = pd.read_csv(eval_path, keep_default_na=False)
    rows: list[dict[str, Any]] = []
    for _, row in eval_df.iterrows():
        path = ROOT / str(row.get("daily_csv_path", ""))
        rec: dict[str, Any] = {
            "station_code": row.get("station_code"),
            "year": row.get("year"),
            "checkpoint_step": row.get("checkpoint_step"),
            "coarse_irrigation_event_count": np.nan,
            "coarse_nitrogen_event_count": np.nan,
            "has_forbidden_15mm_irrigation": np.nan,
            "has_forbidden_40kg_n": np.nan,
            "irrigation_dap1_30": np.nan,
            "irrigation_dap31_60": np.nan,
            "irrigation_dap61_90": np.nan,
            "irrigation_dap91_plus": np.nan,
        }
        if path.exists():
            daily = pd.read_csv(path)
            dap = pd.to_numeric(daily.get("dap", pd.Series(dtype=float)), errors="coerce")
            irr = pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
            n = pd.to_numeric(daily.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
            rec.update(
                {
                    "coarse_irrigation_event_count": int((irr > 1e-9).sum()),
                    "coarse_nitrogen_event_count": int((n > 1e-9).sum()),
                    "has_forbidden_15mm_irrigation": bool(np.isclose(irr, 15.0).any()),
                    "has_forbidden_40kg_n": bool(np.isclose(n, 40.0).any()),
                    "irrigation_dap1_30": float(irr[dap <= base04026.EARLY_DAP_END].sum()),
                    "irrigation_dap31_60": float(irr[(dap > base04026.EARLY_DAP_END) & (dap <= base04026.MID_DAP_END)].sum()),
                    "irrigation_dap61_90": float(irr[(dap > base04026.MID_DAP_END) & (dap <= 90)].sum()),
                    "irrigation_dap91_plus": float(irr[dap > 90].sum()),
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
                mean_coarse_irrigation_event_count=("coarse_irrigation_event_count", "mean"),
                mean_coarse_nitrogen_event_count=("coarse_nitrogen_event_count", "mean"),
                forbidden_15mm_count=("has_forbidden_15mm_irrigation", "sum"),
                forbidden_40kg_count=("has_forbidden_40kg_n", "sum"),
                mean_irrigation_dap1_30=("irrigation_dap1_30", "mean"),
                mean_irrigation_dap31_60=("irrigation_dap31_60", "mean"),
                mean_irrigation_dap61_90=("irrigation_dap61_90", "mean"),
                mean_irrigation_dap91_plus=("irrigation_dap91_plus", "mean"),
            )
        )
        by_station = by_station.merge(extra, on=["station_code", "checkpoint_step"], how="left")
        by_station.to_csv(by_path, index=False, encoding="utf-8-sig")


def write_clean_record(out: Path, doc: Path, suffix: str, total_timesteps: int, checkpoint_steps: list[int]) -> None:
    prefix = "040_33" if not suffix else f"040_33_{suffix}"
    split = read_csv_or_empty(out / "configs" / f"{prefix}_half_split_selection.csv")
    train = read_csv_or_empty(out / "evaluation" / f"{prefix}_training_checkpoint_inventory.csv")
    eval_df = read_csv_or_empty(out / "evaluation" / f"{prefix}_checkpoint_validation_summary.csv")
    by_station = read_csv_or_empty(out / "evaluation" / f"{prefix}_validation_summary_by_station_checkpoint.csv")
    lines = [
        f"# {prefix} SYA lowIC PPO I240 + 水分胁迫惩罚 + 粗化动作档位记录",
        "",
        "## 结论边界",
        "",
        "- 本任务回到 040_28 reward，只删除 15mm 灌溉档位和 40kg/ha 施氮档位。",
        "- 不使用 040_31 事件成本；不使用 040_32 最大事件数；不改输入、不改年份、不改 PPO 超参数、不改水氮总上限。",
        f"- 训练步数：`{total_timesteps}`；checkpoint：`{', '.join(map(str, checkpoint_steps))}`。",
        "",
        "## 输入与动作自检",
        "",
        f"- lowIC 输入目录：`{LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix()}`",
        f"- lowIC 输入目录存在：`{LOWIC_INPUT_ROOT.exists()}`",
        f"- 灌溉档位：`{IRRIGATION_LEVELS}`",
        f"- 施氮档位：`{NITROGEN_LEVELS}`",
        "",
        "## 年份划分",
        "",
        md_table(split[["station_code", "site", "year", "split"]] if {"station_code", "site", "year", "split"}.issubset(split.columns) else split, 120),
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
    ]
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dry_run(total_timesteps: int, checkpoint_steps: list[int], suffix: str) -> None:
    patch_base_module(out_for_suffix(suffix), doc_for_suffix(suffix), total_timesteps, checkpoint_steps)
    config = load_config()
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
        "action_safety": config["action_safety"],
        "discrete_actions": config["discrete_actions"],
        "reward": config["reward"],
        "next_step_allowed": LOWIC_INPUT_ROOT.exists() and PROMPT.exists() and not split.empty,
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
        add_04033_summaries(out, suffix)
        write_clean_record(out, doc, suffix, total_timesteps, checkpoint_steps)
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root

    prefix = "040_33" if not suffix else f"040_33_{suffix}"
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
    }
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
