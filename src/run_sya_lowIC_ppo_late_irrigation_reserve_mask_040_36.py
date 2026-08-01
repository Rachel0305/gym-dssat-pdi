"""040_36: SYA lowIC MaskablePPO with a broad late-irrigation reserve mask.

This is a narrow follow-up to 040_33.  It keeps the 040_33 coarse action
levels and the 040_28 reward, and adds only one broad irrigation mask:

    if DAP <= 90: cumulative_irrigation_after_action <= 195 mm

Given the season cap of 240 mm, this reserves at least one 45 mm irrigation
opportunity for DAP91+.  It does not force any exact DAP such as DAP96.
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
import run_sya_lowIC_ppo_i240_swfac_guardrail_coarse_actions_040_33 as base04033
import run_sya_lowIC_ppo_i240_swfac_guardrail_reward_040_28 as base04028


ORIGINAL_SHUTIL_COPY2 = shutil.copy2
TASK_ID = "040_36"
TASK_NAME = "sya_lowIC_ppo_late_irrigation_reserve_mask"
BASE_OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
BASE_DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"

LOWIC_INPUT_ROOT = base04028.LOWIC_INPUT_ROOT
STATION = base04028.STATION
SITES = [STATION]
DEFAULT_TOTAL_TIMESTEPS = 100_000
DEFAULT_CHECKPOINT_STEPS = [25_000, 50_000, 75_000, 100_000]

IRRIGATION_LEVELS = list(base04033.IRRIGATION_LEVELS)
NITROGEN_LEVELS = list(base04033.NITROGEN_LEVELS)
LATE_RESERVE_DAP_END = 90
PRE_LATE_CUM_IRRIGATION_CAP = 195.0
LATE_RESERVED_IRRIGATION_MM = base04026.SEASON_IRRIGATION_CAP - PRE_LATE_CUM_IRRIGATION_CAP

RENAMES = {
    "configs/032_22_half_split_selection.csv": "configs/040_36_half_split_selection.csv",
    "evaluation/032_22_training_checkpoint_inventory_partial.csv": "evaluation/040_36_training_checkpoint_inventory_partial.csv",
    "evaluation/032_22_checkpoint_validation_summary_partial.csv": "evaluation/040_36_checkpoint_validation_summary_partial.csv",
    "evaluation/032_22_training_checkpoint_inventory.csv": "evaluation/040_36_training_checkpoint_inventory.csv",
    "evaluation/032_22_checkpoint_validation_summary.csv": "evaluation/040_36_checkpoint_validation_summary.csv",
    "evaluation/032_22_validation_summary_by_station_checkpoint.csv": "evaluation/040_36_validation_summary_by_station_checkpoint.csv",
    "logs/032_22_training_year_reset_counts.csv": "logs/040_36_training_year_reset_counts.csv",
    "032_22_result.json": "040_36_result.json",
}


def safe_copy2(src: str | Path, dst: str | Path, *args, **kwargs) -> str:
    try:
        return str(ORIGINAL_SHUTIL_COPY2(src, dst, *args, **kwargs))
    except PermissionError:
        dst_path = Path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst_path)
        return str(dst_path)


class LateIrrigationReserveMaskWrapper(base04028.I240SwfacGuardrailWrapper):
    """040_28/040_33 wrapper plus a DAP90 pre-late irrigation cap."""

    def _action_is_legal_without_clipping(self, raw: dict[str, float], dap: int) -> bool:
        if not super()._action_is_legal_without_clipping(raw, dap):
            return False
        irrigation = float(raw.get("amir", 0.0) or 0.0)
        if irrigation <= 1e-9:
            return True
        after_i = float(self.safety_state.cumulative_irrigation) + irrigation
        if dap <= LATE_RESERVE_DAP_END and after_i - PRE_LATE_CUM_IRRIGATION_CAP > 1e-9:
            return False
        return True

    def step(self, action):
        prev_dap = self._dap()
        prev_i = float(self.safety_state.cumulative_irrigation)
        obs, reward, terminated, truncated, info = super().step(action)
        current_i = float(self.safety_state.cumulative_irrigation)
        self.last_action_info.update(
            {
                "late_irrigation_reserve_mask_enabled": True,
                "late_reserve_dap_end": LATE_RESERVE_DAP_END,
                "pre_late_cum_irrigation_cap_mm": PRE_LATE_CUM_IRRIGATION_CAP,
                "late_reserved_irrigation_mm": LATE_RESERVED_IRRIGATION_MM,
                "late_reserve_active_before_action": bool(prev_dap <= LATE_RESERVE_DAP_END),
                "late_reserve_irrigation_before_action_mm": prev_i,
                "late_reserve_irrigation_after_action_mm": current_i,
                "late_reserve_remaining_before_action_mm": max(0.0, PRE_LATE_CUM_IRRIGATION_CAP - prev_i)
                if prev_dap <= LATE_RESERVE_DAP_END
                else np.nan,
            }
        )
        return obs, reward, terminated, truncated, info


def out_for_suffix(suffix: str) -> Path:
    return ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}_{suffix}" if suffix else BASE_OUT


def doc_for_suffix(suffix: str) -> Path:
    return ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_{suffix}_record.md" if suffix else BASE_DOC


def parse_checkpoint_steps(raw: str | None, total_timesteps: int) -> list[int]:
    if not raw:
        return [x for x in DEFAULT_CHECKPOINT_STEPS if x <= total_timesteps]
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def load_config() -> dict[str, Any]:
    cfg = base04033.load_config()
    cfg["action_safety"]["late_irrigation_reserve_mask"] = {
        "enabled": True,
        "dap_end": LATE_RESERVE_DAP_END,
        "pre_late_cum_irrigation_cap_mm": PRE_LATE_CUM_IRRIGATION_CAP,
        "reserved_for_after_dap90_mm": LATE_RESERVED_IRRIGATION_MM,
        "reason": "reserve at least one 45 mm irrigation opportunity after DAP90; do not force an exact irrigation day",
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
    base03222.base.StressAwareDiscreteWrapper = LateIrrigationReserveMaskWrapper
    base03222.shutil.copy2 = safe_copy2


def copy_with_task_names(out: Path, suffix: str) -> None:
    for old_rel, new_rel in RENAMES.items():
        old = out / old_rel
        if suffix:
            new_rel = new_rel.replace("040_36_", f"040_36_{suffix}_")
        new = out / new_rel
        if old.exists():
            new.parent.mkdir(parents=True, exist_ok=True)
            safe_copy2(old, new)
    generic_result = out / "032_22_result.json"
    result_path = out / ("040_36_result.json" if not suffix else f"040_36_{suffix}_result.json")
    if generic_result.exists():
        result = json.loads(generic_result.read_text(encoding="utf-8"))
        result.update(
            {
                "task": f"{TASK_ID}_{TASK_NAME}" + (f"_{suffix}" if suffix else ""),
                "input_root": LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
                "station_scope": SITES,
                "late_irrigation_reserve_mask": {
                    "dap_end": LATE_RESERVE_DAP_END,
                    "pre_late_cum_irrigation_cap_mm": PRE_LATE_CUM_IRRIGATION_CAP,
                    "reserved_for_after_dap90_mm": LATE_RESERVED_IRRIGATION_MM,
                    "reward_changed": False,
                },
                "coarse_action_levels": {
                    "irrigation_levels": IRRIGATION_LEVELS,
                    "nitrogen_levels": NITROGEN_LEVELS,
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


def add_04036_summaries(out: Path, suffix: str) -> None:
    prefix = "040_36" if not suffix else f"040_36_{suffix}"
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
            "irrigation_dap1_90": np.nan,
            "late_reserve_violation": np.nan,
            "late_reserve_used_after_dap90": np.nan,
        }
        if path.exists():
            daily = pd.read_csv(path)
            dap = pd.to_numeric(daily.get("dap", pd.Series(dtype=float)), errors="coerce")
            irr = pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
            n = pd.to_numeric(daily.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
            i1_30 = float(irr[dap <= base04026.EARLY_DAP_END].sum())
            i31_60 = float(irr[(dap > base04026.EARLY_DAP_END) & (dap <= base04026.MID_DAP_END)].sum())
            i61_90 = float(irr[(dap > base04026.MID_DAP_END) & (dap <= LATE_RESERVE_DAP_END)].sum())
            i91 = float(irr[dap > LATE_RESERVE_DAP_END].sum())
            i1_90 = i1_30 + i31_60 + i61_90
            rec.update(
                {
                    "coarse_irrigation_event_count": int((irr > 1e-9).sum()),
                    "coarse_nitrogen_event_count": int((n > 1e-9).sum()),
                    "has_forbidden_15mm_irrigation": bool(np.isclose(irr, 15.0).any()),
                    "has_forbidden_40kg_n": bool(np.isclose(n, 40.0).any()),
                    "irrigation_dap1_30": i1_30,
                    "irrigation_dap31_60": i31_60,
                    "irrigation_dap61_90": i61_90,
                    "irrigation_dap91_plus": i91,
                    "irrigation_dap1_90": i1_90,
                    "late_reserve_violation": bool(i1_90 - PRE_LATE_CUM_IRRIGATION_CAP > 1e-9),
                    "late_reserve_used_after_dap90": bool(i91 > 1e-9),
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
                mean_irrigation_dap1_90=("irrigation_dap1_90", "mean"),
                late_reserve_violation_count=("late_reserve_violation", "sum"),
                late_reserve_used_after_dap90_count=("late_reserve_used_after_dap90", "sum"),
            )
        )
        by_station = by_station.merge(extra, on=["station_code", "checkpoint_step"], how="left")
        by_station.to_csv(by_path, index=False, encoding="utf-8-sig")


def write_clean_record(out: Path, doc: Path, suffix: str, total_timesteps: int, checkpoint_steps: list[int]) -> None:
    prefix = "040_36" if not suffix else f"040_36_{suffix}"
    split = read_csv_or_empty(out / "configs" / f"{prefix}_half_split_selection.csv")
    train = read_csv_or_empty(out / "evaluation" / f"{prefix}_training_checkpoint_inventory.csv")
    eval_df = read_csv_or_empty(out / "evaluation" / f"{prefix}_checkpoint_validation_summary.csv")
    by_station = read_csv_or_empty(out / "evaluation" / f"{prefix}_validation_summary_by_station_checkpoint.csv")
    cols = ["station_code", "site", "year", "split"]
    split_show = split[cols] if set(cols).issubset(split.columns) else split
    lines = [
        f"# {prefix} SYA lowIC PPO 晚期灌溉保留 mask 记录",
        "",
        "## 结论边界",
        "",
        "- 本任务只在 040_33 基础上增加 DAP90 前留水约束；不改 reward、不改年份、不改 PPO 超参数。",
        "- 约束不是指定 DAP96，而是 DAP≤90 累计灌溉不得超过 195mm，给 DAP91+ 至少保留 45mm。",
        "- 动作档位仍为灌溉 [0,30,45] mm、施氮 [0,80,120] kg/ha，避免 15mm/40kg 小碎片动作。",
        f"- 训练步数：`{total_timesteps}`；checkpoint：`{', '.join(map(str, checkpoint_steps))}`。",
        "",
        "## 固定配置",
        "",
        f"- lowIC 输入目录：`{LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix()}`",
        f"- 输入目录存在：`{LOWIC_INPUT_ROOT.exists()}`",
        f"- DAP1–30 上限：{base04026.EARLY_CUM_IRRIGATION_CAP} mm",
        f"- DAP1–60 上限：{base04026.MID_CUM_IRRIGATION_CAP} mm",
        f"- DAP1–90 上限：{PRE_LATE_CUM_IRRIGATION_CAP} mm",
        f"- DAP91+ 保留量：{LATE_RESERVED_IRRIGATION_MM} mm",
        f"- 总灌溉上限：{base04026.SEASON_IRRIGATION_CAP} mm",
        f"- 灌溉档位：`{IRRIGATION_LEVELS}`",
        f"- 施氮档位：`{NITROGEN_LEVELS}`",
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
        "late_irrigation_reserve_mask": config["action_safety"]["late_irrigation_reserve_mask"],
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
        add_04036_summaries(out, suffix)
        write_clean_record(out, doc, suffix, total_timesteps, checkpoint_steps)
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root

    prefix = "040_36" if not suffix else f"040_36_{suffix}"
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
