from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import ppo_safe_rendering  # noqa: F401  # keep safe rendering patch active
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base03222
import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as base032
import run_sya_lowIC_ppo_early_irrigation_reserve_penalty_040_10 as util04010


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "040_13"
TASK_NAME = "sya_lowIC_ppo_early_irrigation_hard_reserve_mask"
BASE_OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
BASE_DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"

LOWIC_INPUT_ROOT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual"
STATION = "SYA"
SITES = [STATION]
DEFAULT_TOTAL_TIMESTEPS = 100_000
DEFAULT_CHECKPOINT_STEPS = [25_000, 50_000, 75_000, 100_000]

EARLY_RESERVE_DAP_END = 30
EARLY_IRRIGATION_HARD_CAP = 90.0

MAINLINE_SUMMARY = (
    ROOT
    / "benchmark_results"
    / "040_00_sya_lowIC_free_timing_maskableppo"
    / "evaluation"
    / "040_00_checkpoint_validation_summary.csv"
)
MAINLINE_CHECKPOINT = 75_000

RENAMES = {
    "configs/032_22_half_split_selection.csv": "configs/040_13_half_split_selection.csv",
    "evaluation/032_22_training_checkpoint_inventory_partial.csv": "evaluation/040_13_training_checkpoint_inventory_partial.csv",
    "evaluation/032_22_checkpoint_validation_summary_partial.csv": "evaluation/040_13_checkpoint_validation_summary_partial.csv",
    "evaluation/032_22_training_checkpoint_inventory.csv": "evaluation/040_13_training_checkpoint_inventory.csv",
    "evaluation/032_22_checkpoint_validation_summary.csv": "evaluation/040_13_checkpoint_validation_summary.csv",
    "evaluation/032_22_validation_summary_by_station_checkpoint.csv": "evaluation/040_13_validation_summary_by_station_checkpoint.csv",
    "logs/032_22_training_year_reset_counts.csv": "logs/040_13_training_year_reset_counts.csv",
    "032_22_result.json": "040_13_result.json",
}


class EarlyIrrigationHardReserveMaskWrapper(base032.StressAwareDiscreteWrapper):
    """040_13: hard action mask keeping at least 70 mm water for DAP31+."""

    def _action_is_legal_without_clipping(self, raw: dict[str, float], dap: int) -> bool:
        if not super()._action_is_legal_without_clipping(raw, dap):
            return False
        i = float(raw.get("amir", 0.0) or 0.0)
        if i > 0 and dap <= EARLY_RESERVE_DAP_END:
            after_i = float(self.safety_state.cumulative_irrigation) + i
            if after_i - EARLY_IRRIGATION_HARD_CAP > 1e-9:
                return False
        return True

    def step(self, action):
        prev_dap = self._dap()
        prev_i = float(self.safety_state.cumulative_irrigation)
        obs, reward, terminated, truncated, info = super().step(action)
        current_i = float(self.safety_state.cumulative_irrigation)
        self.last_action_info.update(
            {
                "early_irrigation_hard_reserve_enabled": True,
                "early_reserve_dap_end": EARLY_RESERVE_DAP_END,
                "early_irrigation_hard_cap_mm": EARLY_IRRIGATION_HARD_CAP,
                "early_irrigation_before_action_mm": prev_i if prev_dap <= EARLY_RESERVE_DAP_END else np.nan,
                "early_irrigation_after_action_mm": current_i if prev_dap <= EARLY_RESERVE_DAP_END else np.nan,
                "early_hard_cap_remaining_before_action_mm": max(0.0, EARLY_IRRIGATION_HARD_CAP - prev_i)
                if prev_dap <= EARLY_RESERVE_DAP_END
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


def patch_base_module(out: Path, doc: Path, total_timesteps: int, checkpoint_steps: list[int]) -> None:
    base03222.OUT = out
    base03222.DOC = doc
    base03222.PROMPT = PROMPT
    base03222.SITES = list(SITES)
    base03222.TOTAL_TIMESTEPS = int(total_timesteps)
    base03222.CHECKPOINT_STEPS = [int(x) for x in checkpoint_steps]
    base03222.summarize_by_station = util04010.summarize_by_station_safe
    base03222.base.StressAwareDiscreteWrapper = EarlyIrrigationHardReserveMaskWrapper


def copy_with_task_names(out: Path, suffix: str) -> None:
    for old_rel, new_rel in RENAMES.items():
        old = out / old_rel
        if suffix:
            new_rel = new_rel.replace("040_13_", f"040_13_{suffix}_")
        new = out / new_rel
        if old.exists():
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, new)
    generic_result = out / "032_22_result.json"
    result_path = out / ("040_13_result.json" if not suffix else f"040_13_{suffix}_result.json")
    if generic_result.exists():
        result = json.loads(generic_result.read_text(encoding="utf-8"))
        result.update(
            {
                "task": f"{TASK_ID}_{TASK_NAME}" + (f"_{suffix}" if suffix else ""),
                "input_root": LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
                "station_scope": SITES,
                "early_irrigation_hard_reserve_mask": {
                    "dap_end": EARLY_RESERVE_DAP_END,
                    "hard_cap_mm": EARLY_IRRIGATION_HARD_CAP,
                    "reward_changed": False,
                },
            }
        )
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def add_04013_summaries(out: Path, suffix: str) -> None:
    prefix = "040_13" if not suffix else f"040_13_{suffix}"
    eval_path = out / "evaluation" / f"{prefix}_checkpoint_validation_summary.csv"
    if not eval_path.exists():
        return
    eval_df = pd.read_csv(eval_path, keep_default_na=False)
    drop_existing = [
        c
        for c in eval_df.columns
        if c.startswith("early_dap1_30_irrigation")
        or c.startswith("early_dap1_30_cap_violation")
        or c.startswith("irrigation_after_dap30")
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
            "early_dap1_30_irrigation": np.nan,
            "irrigation_after_dap30": np.nan,
            "early_dap1_30_cap_violation": np.nan,
        }
        if path.exists():
            daily = pd.read_csv(path)
            dap = pd.to_numeric(daily.get("dap", pd.Series(dtype=float)), errors="coerce")
            irr = pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
            early_i = float(irr[dap <= EARLY_RESERVE_DAP_END].sum())
            late_i = float(irr[dap > EARLY_RESERVE_DAP_END].sum())
            rec.update(
                {
                    "early_dap1_30_irrigation": early_i,
                    "irrigation_after_dap30": late_i,
                    "early_dap1_30_cap_violation": bool(early_i - EARLY_IRRIGATION_HARD_CAP > 1e-9),
                }
            )
        rows.append(rec)
    merged = eval_df.merge(pd.DataFrame(rows), on=["station_code", "year", "checkpoint_step"], how="left")
    merged.to_csv(eval_path, index=False, encoding="utf-8-sig")
    by_path = out / "evaluation" / f"{prefix}_validation_summary_by_station_checkpoint.csv"
    if by_path.exists():
        by_station = util04010.summarize_by_station_safe(merged)
        # Add hard-cap-specific aggregates.
        extra = (
            merged.groupby(["station_code", "checkpoint_step"], as_index=False)
            .agg(
                mean_early_dap1_30_irrigation=("early_dap1_30_irrigation", "mean"),
                mean_irrigation_after_dap30=("irrigation_after_dap30", "mean"),
                early_cap_violation_count=("early_dap1_30_cap_violation", "sum"),
            )
        )
        by_station = by_station.merge(extra, on=["station_code", "checkpoint_step"], how="left")
        by_station.to_csv(by_path, index=False, encoding="utf-8-sig")


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


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


def compare_to_mainline(eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df.empty or not MAINLINE_SUMMARY.exists():
        return pd.DataFrame()
    main = pd.read_csv(MAINLINE_SUMMARY, keep_default_na=False)
    main = main[pd.to_numeric(main["checkpoint_step"], errors="coerce").eq(MAINLINE_CHECKPOINT)].copy()
    if main.empty:
        return pd.DataFrame()
    base = main.set_index("year")
    rows = []
    for _, row in eval_df.iterrows():
        year = int(row["year"])
        if year not in base.index:
            continue
        b = base.loc[year]
        rows.append(
            {
                "checkpoint_step": int(row["checkpoint_step"]),
                "year": year,
                "final_grnwt": float(row["final_grnwt"]),
                "delta_yield_vs_040_00_ckpt75000": float(row["final_grnwt"]) - float(b["final_grnwt"]),
                "total_irrigation": float(row["total_irrigation"]),
                "delta_irrigation_vs_040_00_ckpt75000": float(row["total_irrigation"]) - float(b["total_irrigation"]),
                "swfac_days_gt_0p05": int(row["swfac_stress_days_gt_0p05"]),
                "delta_swfac_days_vs_040_00_ckpt75000": int(row["swfac_stress_days_gt_0p05"]) - int(b["swfac_stress_days_gt_0p05"]),
                "early_dap1_30_irrigation": float(row.get("early_dap1_30_irrigation", np.nan)),
                "irrigation_after_dap30": float(row.get("irrigation_after_dap30", np.nan)),
                "action_sequence": row.get("action_sequence", ""),
            }
        )
    return pd.DataFrame(rows)


def write_clean_record(out: Path, doc: Path, suffix: str, total_timesteps: int, checkpoint_steps: list[int]) -> None:
    prefix = "040_13" if not suffix else f"040_13_{suffix}"
    split = read_csv_or_empty(out / "configs" / f"{prefix}_half_split_selection.csv")
    train = read_csv_or_empty(out / "evaluation" / f"{prefix}_training_checkpoint_inventory.csv")
    eval_df = read_csv_or_empty(out / "evaluation" / f"{prefix}_checkpoint_validation_summary.csv")
    by_station = read_csv_or_empty(out / "evaluation" / f"{prefix}_validation_summary_by_station_checkpoint.csv")
    comp = compare_to_mainline(eval_df)
    comp_path = out / "evaluation" / f"{prefix}_vs_040_00_ckpt75000_year_level.csv"
    if not comp.empty:
        comp.to_csv(comp_path, index=False, encoding="utf-8-sig")

    branch = "not_evaluated"
    if not by_station.empty:
        base_mean = np.nan
        base_min = np.nan
        base_swfac = np.nan
        if MAINLINE_SUMMARY.exists():
            main = pd.read_csv(MAINLINE_SUMMARY, keep_default_na=False)
            main = main[pd.to_numeric(main["checkpoint_step"], errors="coerce").eq(MAINLINE_CHECKPOINT)].copy()
            if not main.empty:
                base_mean = pd.to_numeric(main["final_grnwt"], errors="coerce").mean()
                base_min = pd.to_numeric(main["final_grnwt"], errors="coerce").min()
                base_swfac = pd.to_numeric(main["swfac_stress_days_gt_0p05"], errors="coerce").mean()
        candidates = []
        for _, row in by_station.iterrows():
            ck = int(row["checkpoint_step"])
            sub = eval_df[pd.to_numeric(eval_df["checkpoint_step"], errors="coerce").eq(ck)]
            mean_y = pd.to_numeric(sub["final_grnwt"], errors="coerce").mean()
            min_y = pd.to_numeric(sub["final_grnwt"], errors="coerce").min()
            mean_sw = pd.to_numeric(sub["swfac_stress_days_gt_0p05"], errors="coerce").mean()
            violations = pd.to_numeric(sub.get("early_dap1_30_cap_violation", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
            if (
                np.isfinite(base_mean)
                and mean_y >= base_mean
                and min_y >= base_min
                and mean_sw <= base_swfac
                and violations == 0
            ):
                candidates.append(ck)
        branch = "A_hard_reserve_candidate_passes" if candidates else "B_no_checkpoint_dominates_mainline"

    lines = [
        f"# {prefix} SYA lowIC PPO 早期灌溉硬保留 mask 记录",
        "",
        "## 结论先说",
        "",
        f"- 分支判定：`{branch}`。",
        f"- 本轮训练步数：`{total_timesteps}`；checkpoint：`{', '.join(map(str, checkpoint_steps))}`。",
        "- 本轮不是改 reward，而是在 action mask / safety layer 中加入早期灌溉硬约束。",
        "",
        "## 唯一新增约束",
        "",
        "```text",
        "if DAP <= 30:",
        "    cumulative_irrigation_after_action <= 90 mm",
        "```",
        "",
        "这意味着 PPO 仍然每天观察、每天自由决策，但不能在前30天把全季灌溉预算提前用光。",
        "",
        "## 保持不变",
        "",
        "- lowIC 输入目录不变。",
        "- SYA 训练年份 2005–2013，验证年份 2014–2023。",
        "- MaskablePPO 超参数、动作档位、单季水氮上限、7天最小间隔、DAP90后禁氮均沿用 040_00。",
        "- reward 沿用 040_00 stress-aware reward，不叠加 040_10/040_11 的额外惩罚。",
        "",
        "## 年份划分",
        "",
        md_table(split[["station_code", "site", "year", "split"]] if {"station_code", "site", "year", "split"}.issubset(split.columns) else split, max_rows=120),
        "",
        "## 训练 checkpoint 库存",
        "",
        md_table(train, max_rows=80),
        "",
        "## 验证集按 checkpoint 汇总",
        "",
        md_table(by_station, max_rows=80),
        "",
        "## 相对 040_00 checkpoint 75000 的逐年差异",
        "",
        md_table(comp, max_rows=120),
        "",
        "## 解释边界",
        "",
        "- 本轮测试的是硬保留约束是否比软 reward 惩罚更稳。",
        "- 如果失败，不自动调 DAP30 或 90mm 阈值；需要另开任务重新预注册。",
    ]
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dry_run(total_timesteps: int, checkpoint_steps: list[int], suffix: str) -> None:
    patch_base_module(out_for_suffix(suffix), doc_for_suffix(suffix), total_timesteps, checkpoint_steps)
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
        "early_irrigation_hard_reserve_mask": {
            "dap_end": EARLY_RESERVE_DAP_END,
            "hard_cap_mm": EARLY_IRRIGATION_HARD_CAP,
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
        add_04013_summaries(out, suffix)
        write_clean_record(out, doc, suffix, total_timesteps, checkpoint_steps)
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root

    prefix = "040_13" if not suffix else f"040_13_{suffix}"
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}" + (f"_{suffix}" if suffix else ""),
        "algorithm": "MaskablePPO",
        "record_md": doc.relative_to(ROOT).as_posix(),
        "train_inventory": (out / "evaluation" / f"{prefix}_training_checkpoint_inventory.csv").relative_to(ROOT).as_posix(),
        "validation_summary": (out / "evaluation" / f"{prefix}_checkpoint_validation_summary.csv").relative_to(ROOT).as_posix(),
        "by_checkpoint": (out / "evaluation" / f"{prefix}_validation_summary_by_station_checkpoint.csv").relative_to(ROOT).as_posix(),
        "comparison_vs_040_00": (out / "evaluation" / f"{prefix}_vs_040_00_ckpt75000_year_level.csv").relative_to(ROOT).as_posix(),
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
