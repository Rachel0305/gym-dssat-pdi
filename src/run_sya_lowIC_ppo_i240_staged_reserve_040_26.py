"""040_26: SYA lowIC MaskablePPO with I240 staged irrigation reserve.

This task is a narrow follow-up to 040_24 and 040_25.

Only the irrigation budget and staged water-reserve thresholds are changed:

    season_irrigation_soft_limit: 160 -> 240 mm
    DAP <= 30: cumulative irrigation after action <= 75 mm
    DAP <= 60: cumulative irrigation after action <= 150 mm

Reward, lowIC input root, year split, nitrogen cap, action levels and PPO
hyperparameters are intentionally inherited from the current free-timing
stress-aware PPO pipeline.
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
import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as base032
import run_sya_lowIC_ppo_early_irrigation_reserve_penalty_040_10 as util04010
import run_all_year_direct_action_safe_ppo as direct_ppo


ORIGINAL_SHUTIL_COPY2 = shutil.copy2
TASK_ID = "040_26"
TASK_NAME = "sya_lowIC_ppo_i240_staged_reserve"
BASE_OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
BASE_DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"

LOWIC_INPUT_ROOT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual"
STATION = "SYA"
SITES = [STATION]
DEFAULT_TOTAL_TIMESTEPS = 100_000
DEFAULT_CHECKPOINT_STEPS = [25_000, 50_000, 75_000, 100_000]

EARLY_DAP_END = 30
MID_DAP_END = 60
EARLY_CUM_IRRIGATION_CAP = 75.0
MID_CUM_IRRIGATION_CAP = 150.0
SEASON_IRRIGATION_CAP = 240.0

RENAMES = {
    "configs/032_22_half_split_selection.csv": "configs/040_26_half_split_selection.csv",
    "evaluation/032_22_training_checkpoint_inventory_partial.csv": "evaluation/040_26_training_checkpoint_inventory_partial.csv",
    "evaluation/032_22_checkpoint_validation_summary_partial.csv": "evaluation/040_26_checkpoint_validation_summary_partial.csv",
    "evaluation/032_22_training_checkpoint_inventory.csv": "evaluation/040_26_training_checkpoint_inventory.csv",
    "evaluation/032_22_checkpoint_validation_summary.csv": "evaluation/040_26_checkpoint_validation_summary.csv",
    "evaluation/032_22_validation_summary_by_station_checkpoint.csv": "evaluation/040_26_validation_summary_by_station_checkpoint.csv",
    "logs/032_22_training_year_reset_counts.csv": "logs/040_26_training_year_reset_counts.csv",
    "032_22_result.json": "040_26_result.json",
}


def safe_copy2(src: str | Path, dst: str | Path, *args, **kwargs) -> str:
    """Copy file contents; fall back when Windows-mounted Docker blocks utime."""
    try:
        return str(ORIGINAL_SHUTIL_COPY2(src, dst, *args, **kwargs))
    except PermissionError:
        dst_path = Path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst_path)
        return str(dst_path)


class I240StagedIrrigationReserveMaskWrapper(base032.StressAwareDiscreteWrapper):
    """Hard action mask that keeps water available for mid/late season."""

    def _action_is_legal_without_clipping(self, raw: dict[str, float], dap: int) -> bool:
        if not super()._action_is_legal_without_clipping(raw, dap):
            return False
        irrigation = float(raw.get("amir", 0.0) or 0.0)
        if irrigation <= 1e-9:
            return True
        after_i = float(self.safety_state.cumulative_irrigation) + irrigation
        if dap <= EARLY_DAP_END and after_i - EARLY_CUM_IRRIGATION_CAP > 1e-9:
            return False
        if dap <= MID_DAP_END and after_i - MID_CUM_IRRIGATION_CAP > 1e-9:
            return False
        return True

    def step(self, action):
        prev_dap = self._dap()
        prev_i = float(self.safety_state.cumulative_irrigation)
        obs, reward, terminated, truncated, info = super().step(action)
        current_i = float(self.safety_state.cumulative_irrigation)
        self.last_action_info.update(
            {
                "i240_staged_irrigation_reserve_enabled": True,
                "early_dap_end": EARLY_DAP_END,
                "mid_dap_end": MID_DAP_END,
                "early_cum_irrigation_cap_mm": EARLY_CUM_IRRIGATION_CAP,
                "mid_cum_irrigation_cap_mm": MID_CUM_IRRIGATION_CAP,
                "season_irrigation_cap_mm": SEASON_IRRIGATION_CAP,
                "irrigation_before_action_mm": prev_i,
                "irrigation_after_action_mm": current_i,
                "active_stage_for_irrigation_mask": "early" if prev_dap <= EARLY_DAP_END else ("mid" if prev_dap <= MID_DAP_END else "late"),
                "early_remaining_before_action_mm": max(0.0, EARLY_CUM_IRRIGATION_CAP - prev_i) if prev_dap <= EARLY_DAP_END else np.nan,
                "mid_remaining_before_action_mm": max(0.0, MID_CUM_IRRIGATION_CAP - prev_i) if prev_dap <= MID_DAP_END else np.nan,
                "late_reserved_reference_before_action_mm": max(0.0, SEASON_IRRIGATION_CAP - MID_CUM_IRRIGATION_CAP)
                if prev_dap <= MID_DAP_END
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


def load_i240_config() -> dict[str, Any]:
    cfg = direct_ppo.load_yaml(base03222.CONFIG)
    cfg = json.loads(json.dumps(cfg))
    cfg["seed"] = base03222.SEED
    cfg["total_timesteps"] = base03222.TOTAL_TIMESTEPS
    cfg["paths"]["output_root"] = str(base03222.OUT.relative_to(ROOT)).replace("\\", "/")
    cfg["runtime"]["smoke_station"] = SITES[0]
    cfg["action_safety"]["season_irrigation_soft_limit"] = float(SEASON_IRRIGATION_CAP)
    return cfg


def patch_base_module(out: Path, doc: Path, total_timesteps: int, checkpoint_steps: list[int]) -> None:
    base03222.OUT = out
    base03222.DOC = doc
    base03222.PROMPT = PROMPT
    base03222.SITES = list(SITES)
    base03222.TOTAL_TIMESTEPS = int(total_timesteps)
    base03222.CHECKPOINT_STEPS = [int(x) for x in checkpoint_steps]
    base03222.summarize_by_station = util04010.summarize_by_station_safe
    base03222.load_config = load_i240_config
    base03222.base.StressAwareDiscreteWrapper = I240StagedIrrigationReserveMaskWrapper
    base03222.shutil.copy2 = safe_copy2


def copy_with_task_names(out: Path, suffix: str) -> None:
    for old_rel, new_rel in RENAMES.items():
        old = out / old_rel
        if suffix:
            new_rel = new_rel.replace("040_26_", f"040_26_{suffix}_")
        new = out / new_rel
        if old.exists():
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, new)
    generic_result = out / "032_22_result.json"
    result_path = out / ("040_26_result.json" if not suffix else f"040_26_{suffix}_result.json")
    if generic_result.exists():
        result = json.loads(generic_result.read_text(encoding="utf-8"))
        result.update(
            {
                "task": f"{TASK_ID}_{TASK_NAME}" + (f"_{suffix}" if suffix else ""),
                "input_root": LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
                "station_scope": SITES,
                "i240_staged_irrigation_reserve_mask": {
                    "early_dap_end": EARLY_DAP_END,
                    "early_cum_irrigation_cap_mm": EARLY_CUM_IRRIGATION_CAP,
                    "mid_dap_end": MID_DAP_END,
                    "mid_cum_irrigation_cap_mm": MID_CUM_IRRIGATION_CAP,
                    "season_irrigation_cap_mm": SEASON_IRRIGATION_CAP,
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


def add_04026_summaries(out: Path, suffix: str) -> None:
    prefix = "040_26" if not suffix else f"040_26_{suffix}"
    eval_path = out / "evaluation" / f"{prefix}_checkpoint_validation_summary.csv"
    if not eval_path.exists():
        return
    eval_df = pd.read_csv(eval_path, keep_default_na=False)
    drop_existing = [
        c
        for c in eval_df.columns
        if c.startswith("irrigation_dap")
        or c.startswith("i240_staged_irrigation")
        or c.endswith("_cap_violation")
    ]
    if drop_existing:
        eval_df = eval_df.drop(columns=drop_existing)
    rows: list[dict[str, Any]] = []
    for _, row in eval_df.iterrows():
        path = ROOT / str(row.get("daily_csv_path", ""))
        rec: dict[str, Any] = {
            "station_code": row.get("station_code"),
            "year": row.get("year"),
            "checkpoint_step": row.get("checkpoint_step"),
            "irrigation_dap1_30": np.nan,
            "irrigation_dap31_60": np.nan,
            "irrigation_dap61_90": np.nan,
            "irrigation_dap91_plus": np.nan,
            "irrigation_dap1_60": np.nan,
            "early_cap_violation": np.nan,
            "mid_cap_violation": np.nan,
        }
        if path.exists():
            daily = pd.read_csv(path)
            dap = pd.to_numeric(daily.get("dap", pd.Series(dtype=float)), errors="coerce")
            irr = pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
            i1_30 = float(irr[dap <= EARLY_DAP_END].sum())
            i31_60 = float(irr[(dap > EARLY_DAP_END) & (dap <= MID_DAP_END)].sum())
            i61_90 = float(irr[(dap > MID_DAP_END) & (dap <= 90)].sum())
            i91 = float(irr[dap > 90].sum())
            rec.update(
                {
                    "irrigation_dap1_30": i1_30,
                    "irrigation_dap31_60": i31_60,
                    "irrigation_dap61_90": i61_90,
                    "irrigation_dap91_plus": i91,
                    "irrigation_dap1_60": i1_30 + i31_60,
                    "early_cap_violation": bool(i1_30 - EARLY_CUM_IRRIGATION_CAP > 1e-9),
                    "mid_cap_violation": bool((i1_30 + i31_60) - MID_CUM_IRRIGATION_CAP > 1e-9),
                }
            )
        rows.append(rec)
    merged = eval_df.merge(pd.DataFrame(rows), on=["station_code", "year", "checkpoint_step"], how="left")
    merged.to_csv(eval_path, index=False, encoding="utf-8-sig")
    by_path = out / "evaluation" / f"{prefix}_validation_summary_by_station_checkpoint.csv"
    if by_path.exists():
        by_station = util04010.summarize_by_station_safe(merged)
        extra = (
            merged.groupby(["station_code", "checkpoint_step"], as_index=False)
            .agg(
                mean_irrigation_dap1_30=("irrigation_dap1_30", "mean"),
                mean_irrigation_dap31_60=("irrigation_dap31_60", "mean"),
                mean_irrigation_dap61_90=("irrigation_dap61_90", "mean"),
                mean_irrigation_dap91_plus=("irrigation_dap91_plus", "mean"),
                early_cap_violation_count=("early_cap_violation", "sum"),
                mid_cap_violation_count=("mid_cap_violation", "sum"),
            )
        )
        by_station = by_station.merge(extra, on=["station_code", "checkpoint_step"], how="left")
        by_station.to_csv(by_path, index=False, encoding="utf-8-sig")


def write_clean_record(out: Path, doc: Path, suffix: str, total_timesteps: int, checkpoint_steps: list[int]) -> None:
    prefix = "040_26" if not suffix else f"040_26_{suffix}"
    split = read_csv_or_empty(out / "configs" / f"{prefix}_half_split_selection.csv")
    train = read_csv_or_empty(out / "evaluation" / f"{prefix}_training_checkpoint_inventory.csv")
    eval_df = read_csv_or_empty(out / "evaluation" / f"{prefix}_checkpoint_validation_summary.csv")
    by_station = read_csv_or_empty(out / "evaluation" / f"{prefix}_validation_summary_by_station_checkpoint.csv")

    lines = [
        f"# {prefix} SYA lowIC PPO I240 分阶段储备约束记录",
        "",
        "## 结论边界",
        "",
        "- 本任务是 040_25 反事实审计后的窄幅训练改动。",
        "- 只把灌溉上限从 160 mm 放宽到 240 mm，并同步放宽分阶段储备阈值；不改 reward、不改输入、不改年份划分、不改施氮上限。",
        f"- 训练步数：`{total_timesteps}`；checkpoint：`{', '.join(map(str, checkpoint_steps))}`。",
        "",
        "## 本次约束",
        "",
        "```text",
        f"season_irrigation_soft_limit = {SEASON_IRRIGATION_CAP} mm",
        f"if DAP <= {EARLY_DAP_END}: cumulative_irrigation_after_action <= {EARLY_CUM_IRRIGATION_CAP} mm",
        f"if DAP <= {MID_DAP_END}: cumulative_irrigation_after_action <= {MID_CUM_IRRIGATION_CAP} mm",
        "season_n_soft_limit = 250 kg/ha",
        "reward unchanged",
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
        "",
        "## 解释规则",
        "",
        "- 若产量和水分胁迫明显改善，说明 040_24 的主要限制是 I160 水分预算不足。",
        "- 若灌溉升至约 240 mm 但产量仍低，才考虑 reward、checkpoint guardrail 或更强天气/土壤水响应信号。",
        "- 本任务不根据结果现场继续调 75/150/240 mm 阈值。",
    ]
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dry_run(total_timesteps: int, checkpoint_steps: list[int], suffix: str) -> None:
    patch_base_module(out_for_suffix(suffix), doc_for_suffix(suffix), total_timesteps, checkpoint_steps)
    config = load_i240_config()
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
        "i240_staged_irrigation_reserve_mask": {
            "early_dap_end": EARLY_DAP_END,
            "early_cum_irrigation_cap_mm": EARLY_CUM_IRRIGATION_CAP,
            "mid_dap_end": MID_DAP_END,
            "mid_cum_irrigation_cap_mm": MID_CUM_IRRIGATION_CAP,
            "season_irrigation_cap_mm": SEASON_IRRIGATION_CAP,
            "reward_changed": False,
        },
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
        add_04026_summaries(out, suffix)
        write_clean_record(out, doc, suffix, total_timesteps, checkpoint_steps)
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root

    prefix = "040_26" if not suffix else f"040_26_{suffix}"
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
