"""040_31: SYA lowIC PPO I240 + SWFAC guardrail + operation event cost.

This task keeps 040_28 unchanged except for one reward term: each actual
irrigation/fertilization event receives a fixed management cost.  The purpose is
to test whether the frequent-small-irrigation / frequent-small-fertilization
pattern is caused by the absence of an event cost in the reward.
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
TASK_ID = "040_31"
TASK_NAME = "sya_lowIC_ppo_i240_swfac_guardrail_event_cost"
BASE_OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
BASE_DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"

LOWIC_INPUT_ROOT = base04028.LOWIC_INPUT_ROOT
STATION = base04028.STATION
SITES = [STATION]
DEFAULT_TOTAL_TIMESTEPS = 100_000
DEFAULT_CHECKPOINT_STEPS = [25_000, 50_000, 75_000, 100_000]

WATER_COST = 1.1
NITROGEN_COST = 1.58
MIN_IRRIGATION_LEVEL = 15.0
MIN_NITROGEN_LEVEL = 40.0
IRRIGATION_EVENT_COST_UNSCALED = MIN_IRRIGATION_LEVEL * WATER_COST
NITROGEN_EVENT_COST_UNSCALED = MIN_NITROGEN_LEVEL * NITROGEN_COST

RENAMES = {
    "configs/032_22_half_split_selection.csv": "configs/040_31_half_split_selection.csv",
    "evaluation/032_22_training_checkpoint_inventory_partial.csv": "evaluation/040_31_training_checkpoint_inventory_partial.csv",
    "evaluation/032_22_checkpoint_validation_summary_partial.csv": "evaluation/040_31_checkpoint_validation_summary_partial.csv",
    "evaluation/032_22_training_checkpoint_inventory.csv": "evaluation/040_31_training_checkpoint_inventory.csv",
    "evaluation/032_22_checkpoint_validation_summary.csv": "evaluation/040_31_checkpoint_validation_summary.csv",
    "evaluation/032_22_validation_summary_by_station_checkpoint.csv": "evaluation/040_31_validation_summary_by_station_checkpoint.csv",
    "logs/032_22_training_year_reset_counts.csv": "logs/040_31_training_year_reset_counts.csv",
    "032_22_result.json": "040_31_result.json",
}


def safe_copy2(src: str | Path, dst: str | Path, *args, **kwargs) -> str:
    try:
        return str(ORIGINAL_SHUTIL_COPY2(src, dst, *args, **kwargs))
    except PermissionError:
        dst_path = Path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst_path)
        return str(dst_path)


class I240SwfacEventCostWrapper(base04028.I240SwfacGuardrailWrapper):
    """040_28 wrapper plus fixed per-event management costs."""

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        actual_i = base04026.base032.scalar(self.last_action_info.get("safe_action_amir", 0.0), 0.0)
        actual_n = base04026.base032.scalar(self.last_action_info.get("safe_action_anfer", 0.0), 0.0)
        irrigation_event = bool(np.isfinite(actual_i) and float(actual_i) > 1e-9)
        nitrogen_event = bool(np.isfinite(actual_n) and float(actual_n) > 1e-9)
        event_cost_unscaled = 0.0
        if irrigation_event:
            event_cost_unscaled += IRRIGATION_EVENT_COST_UNSCALED
        if nitrogen_event:
            event_cost_unscaled += NITROGEN_EVENT_COST_UNSCALED
        reward_scale = float(self.config.get("reward", {}).get("reward_scale", 1.0))
        event_cost_scaled = event_cost_unscaled * reward_scale
        reward_after_event_cost = float(reward) - event_cost_scaled
        self.last_action_info.update(
            {
                "operation_event_cost_enabled": True,
                "irrigation_event_for_cost": irrigation_event,
                "nitrogen_event_for_cost": nitrogen_event,
                "irrigation_event_cost_unscaled": IRRIGATION_EVENT_COST_UNSCALED if irrigation_event else 0.0,
                "nitrogen_event_cost_unscaled": NITROGEN_EVENT_COST_UNSCALED if nitrogen_event else 0.0,
                "operation_event_cost_unscaled": event_cost_unscaled,
                "operation_event_cost_scaled": event_cost_scaled,
                "reward_before_operation_event_cost": float(reward),
                "reward_after_operation_event_cost": reward_after_event_cost,
            }
        )
        return obs, reward_after_event_cost, terminated, truncated, info


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
    cfg["reward"]["operation_event_cost"] = {
        "enabled": True,
        "irrigation_event_cost_unscaled": IRRIGATION_EVENT_COST_UNSCALED,
        "nitrogen_event_cost_unscaled": NITROGEN_EVENT_COST_UNSCALED,
        "irrigation_event_cost_source": f"{MIN_IRRIGATION_LEVEL} mm * water_cost {WATER_COST}",
        "nitrogen_event_cost_source": f"{MIN_NITROGEN_LEVEL} kg/ha * nitrogen_cost {NITROGEN_COST}",
        "formula": "reward_04031 = reward_04028 - event_cost_unscaled * reward_scale",
    }
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
    base03222.base.StressAwareDiscreteWrapper = I240SwfacEventCostWrapper
    base03222.shutil.copy2 = safe_copy2


def copy_with_task_names(out: Path, suffix: str) -> None:
    for old_rel, new_rel in RENAMES.items():
        old = out / old_rel
        if suffix:
            new_rel = new_rel.replace("040_31_", f"040_31_{suffix}_")
        new = out / new_rel
        if old.exists():
            new.parent.mkdir(parents=True, exist_ok=True)
            safe_copy2(old, new)
    generic_result = out / "032_22_result.json"
    result_path = out / ("040_31_result.json" if not suffix else f"040_31_{suffix}_result.json")
    if generic_result.exists():
        result = json.loads(generic_result.read_text(encoding="utf-8"))
        result.update(
            {
                "task": f"{TASK_ID}_{TASK_NAME}" + (f"_{suffix}" if suffix else ""),
                "input_root": LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
                "station_scope": SITES,
                "operation_event_cost": {
                    "irrigation_event_cost_unscaled": IRRIGATION_EVENT_COST_UNSCALED,
                    "nitrogen_event_cost_unscaled": NITROGEN_EVENT_COST_UNSCALED,
                    "reward_changed": True,
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


def add_04031_summaries(out: Path, suffix: str) -> None:
    prefix = "040_31" if not suffix else f"040_31_{suffix}"
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
            "event_cost_irrigation_events": np.nan,
            "event_cost_nitrogen_events": np.nan,
            "operation_event_cost_sum_unscaled": np.nan,
            "operation_event_cost_sum_scaled": np.nan,
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
            cost_u = pd.to_numeric(daily.get("operation_event_cost_unscaled", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
            cost_s = pd.to_numeric(daily.get("operation_event_cost_scaled", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
            rec.update(
                {
                    "event_cost_irrigation_events": int((irr > 1e-9).sum()),
                    "event_cost_nitrogen_events": int((n > 1e-9).sum()),
                    "operation_event_cost_sum_unscaled": float(cost_u.sum()),
                    "operation_event_cost_sum_scaled": float(cost_s.sum()),
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
                mean_event_cost_irrigation_events=("event_cost_irrigation_events", "mean"),
                mean_event_cost_nitrogen_events=("event_cost_nitrogen_events", "mean"),
                mean_operation_event_cost_unscaled=("operation_event_cost_sum_unscaled", "mean"),
                mean_irrigation_dap1_30=("irrigation_dap1_30", "mean"),
                mean_irrigation_dap31_60=("irrigation_dap31_60", "mean"),
                mean_irrigation_dap61_90=("irrigation_dap61_90", "mean"),
                mean_irrigation_dap91_plus=("irrigation_dap91_plus", "mean"),
            )
        )
        by_station = by_station.merge(extra, on=["station_code", "checkpoint_step"], how="left")
        by_station.to_csv(by_path, index=False, encoding="utf-8-sig")


def write_clean_record(out: Path, doc: Path, suffix: str, total_timesteps: int, checkpoint_steps: list[int]) -> None:
    prefix = "040_31" if not suffix else f"040_31_{suffix}"
    split = read_csv_or_empty(out / "configs" / f"{prefix}_half_split_selection.csv")
    train = read_csv_or_empty(out / "evaluation" / f"{prefix}_training_checkpoint_inventory.csv")
    eval_df = read_csv_or_empty(out / "evaluation" / f"{prefix}_checkpoint_validation_summary.csv")
    by_station = read_csv_or_empty(out / "evaluation" / f"{prefix}_validation_summary_by_station_checkpoint.csv")
    lines = [
        f"# {prefix} SYA lowIC PPO I240 + 水分胁迫惩罚 + 操作事件成本记录",
        "",
        "## 结论边界",
        "",
        "- 本任务只在 040_28 基础上新增每次灌溉/施肥固定操作成本。",
        "- 不改输入、不改年份、不改 PPO 超参数、不改水氮上限、不改动作档位。",
        f"- 训练步数：`{total_timesteps}`；checkpoint：`{', '.join(map(str, checkpoint_steps))}`。",
        "",
        "## 输入自检",
        "",
        f"- lowIC 输入目录：`{LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix()}`",
        f"- lowIC 输入目录存在：`{LOWIC_INPUT_ROOT.exists()}`",
        f"- 站点范围：`{', '.join(SITES)}`",
        "",
        "## 新增 reward 项",
        "",
        "```text",
        f"irrigation_event_cost_unscaled = {IRRIGATION_EVENT_COST_UNSCALED} if actual_irrigation_mm > 0 else 0",
        f"nitrogen_event_cost_unscaled   = {NITROGEN_EVENT_COST_UNSCALED} if actual_nitrogen_kg_ha > 0 else 0",
        "reward_04031 = reward_04028 - event_cost_unscaled * reward_scale",
        "```",
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
        add_04031_summaries(out, suffix)
        write_clean_record(out, doc, suffix, total_timesteps, checkpoint_steps)
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root

    prefix = "040_31" if not suffix else f"040_31_{suffix}"
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
