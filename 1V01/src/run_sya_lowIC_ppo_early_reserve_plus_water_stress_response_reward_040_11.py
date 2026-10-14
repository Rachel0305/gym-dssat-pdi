from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import ppo_safe_rendering
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base03222
import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as base032
import run_sya_lowIC_ppo_early_irrigation_reserve_penalty_040_10 as base04010


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "040_11"
TASK_NAME = "sya_lowIC_ppo_early_reserve_plus_water_stress_response_reward"
BASE_OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
BASE_DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"

LOWIC_INPUT_ROOT = base04010.LOWIC_INPUT_ROOT
STATION = "SYA"
SITES = [STATION]
DEFAULT_TOTAL_TIMESTEPS = 100_000
DEFAULT_CHECKPOINT_STEPS = [25_000, 50_000, 75_000, 100_000]

WATER_STRESS_RESPONSE_START_DAP = 31
WATER_STRESS_THRESHOLD = 0.05
WATER_STRESS_NO_RESPONSE_COST = 45.0


RENAMES = {
    "configs/032_22_half_split_selection.csv": "configs/040_11_half_split_selection.csv",
    "evaluation/032_22_training_checkpoint_inventory_partial.csv": "evaluation/040_11_training_checkpoint_inventory_partial.csv",
    "evaluation/032_22_checkpoint_validation_summary_partial.csv": "evaluation/040_11_checkpoint_validation_summary_partial.csv",
    "evaluation/032_22_training_checkpoint_inventory.csv": "evaluation/040_11_training_checkpoint_inventory.csv",
    "evaluation/032_22_checkpoint_validation_summary.csv": "evaluation/040_11_checkpoint_validation_summary.csv",
    "evaluation/032_22_validation_summary_by_station_checkpoint.csv": "evaluation/040_11_validation_summary_by_station_checkpoint.csv",
    "logs/032_22_training_year_reset_counts.csv": "logs/040_11_training_year_reset_counts.csv",
    "032_22_result.json": "040_11_result.json",
}


class EarlyReserveWaterStressResponseWrapper(base04010.EarlyIrrigationReservePenaltyWrapper):
    """040_11 wrapper: 040_10 plus DAP30+ water-stress no-response penalty."""

    def step(self, action):
        prev = dict(self.last_obs_dict)
        prev_dap_raw = base032.scalar(prev.get("dap", 1))
        prev_dap = int(round(prev_dap_raw)) if np.isfinite(prev_dap_raw) and prev_dap_raw > 0 else 1
        swfac_before = base032.scalar(prev.get("swfac"), 0.0)

        obs, reward, terminated, truncated, info = super().step(action)

        info_dict = dict(self.last_action_info)
        safe_i = float(info_dict.get("safe_action_amir", 0.0) or 0.0)
        reward_scale = float(info_dict.get("reward_scale", self.config.get("reward", {}).get("reward_scale", 1.0)))
        no_response_penalty_unscaled = 0.0
        if prev_dap >= WATER_STRESS_RESPONSE_START_DAP and swfac_before > WATER_STRESS_THRESHOLD and safe_i <= 1e-9:
            no_response_penalty_unscaled = WATER_STRESS_NO_RESPONSE_COST * float(swfac_before)
        no_response_penalty_scaled = no_response_penalty_unscaled * reward_scale
        new_reward = float(reward) - no_response_penalty_scaled
        self.last_action_info.update(
            {
                "water_stress_response_start_dap": WATER_STRESS_RESPONSE_START_DAP,
                "water_stress_threshold": WATER_STRESS_THRESHOLD,
                "swfac_before_action_for_response": float(swfac_before),
                "water_stress_no_response_cost": WATER_STRESS_NO_RESPONSE_COST,
                "water_stress_no_response_penalty_unscaled": float(no_response_penalty_unscaled),
                "water_stress_no_response_penalty_scaled": float(no_response_penalty_scaled),
                "reward_before_water_stress_response_penalty": float(reward),
                "reward_stress_aware": float(new_reward),
                "reward_unscaled_04011": float(info_dict.get("reward_unscaled_04010", info_dict.get("reward_unscaled", 0.0)))
                - float(no_response_penalty_unscaled),
            }
        )
        return obs, new_reward, terminated, truncated, info


def out_for_suffix(suffix: str) -> Path:
    return ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}_{suffix}" if suffix else BASE_OUT


def doc_for_suffix(suffix: str) -> Path:
    return ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_{suffix}_record.md" if suffix else BASE_DOC


def patch_base_module(out: Path, doc: Path, total_timesteps: int, checkpoint_steps: list[int]) -> None:
    base03222.OUT = out
    base03222.DOC = doc
    base03222.PROMPT = PROMPT
    base03222.SITES = list(SITES)
    base03222.TOTAL_TIMESTEPS = int(total_timesteps)
    base03222.CHECKPOINT_STEPS = [int(x) for x in checkpoint_steps]
    base03222.summarize_by_station = base04010.summarize_by_station_safe
    base03222.base.StressAwareDiscreteWrapper = EarlyReserveWaterStressResponseWrapper


def parse_checkpoint_steps(raw: str | None, total_timesteps: int) -> list[int]:
    if not raw:
        return [x for x in DEFAULT_CHECKPOINT_STEPS if x <= total_timesteps]
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def copy_with_task_names(out: Path, suffix: str) -> None:
    for old_rel, new_rel in RENAMES.items():
        old = out / old_rel
        if suffix:
            new_rel = new_rel.replace("040_11_", f"040_11_{suffix}_")
        new = out / new_rel
        if old.exists():
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, new)
    generic_result = out / "032_22_result.json"
    result_path = out / ("040_11_result.json" if not suffix else f"040_11_{suffix}_result.json")
    if generic_result.exists():
        result = json.loads(generic_result.read_text(encoding="utf-8"))
        result.update(
            {
                "task": f"{TASK_ID}_{TASK_NAME}" + (f"_{suffix}" if suffix else ""),
                "input_root": LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
                "station_scope": SITES,
                "early_reserve_penalty": {
                    "dap_end": base04010.EARLY_RESERVE_DAP_END,
                    "free_allowance_mm": base04010.EARLY_IRRIGATION_FREE_ALLOWANCE,
                    "early_overuse_cost_source": base04010.EARLY_OVERUSE_COST_SOURCE,
                },
                "water_stress_response_penalty": {
                    "start_dap": WATER_STRESS_RESPONSE_START_DAP,
                    "threshold": WATER_STRESS_THRESHOLD,
                    "no_response_cost": WATER_STRESS_NO_RESPONSE_COST,
                },
            }
        )
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def add_04011_penalty_summaries(out: Path, suffix: str) -> None:
    prefix = "040_11" if not suffix else f"040_11_{suffix}"
    eval_path = out / "evaluation" / f"{prefix}_checkpoint_validation_summary.csv"
    if not eval_path.exists():
        return
    eval_df = pd.read_csv(eval_path, keep_default_na=False)
    drop_existing = [
        c
        for c in eval_df.columns
        if c.startswith("early_reserve_penalty_sum_unscaled")
        or c.startswith("early_overuse_total_mm")
        or c.startswith("water_stress_no_response_penalty_sum_unscaled")
        or c.startswith("water_stress_no_response_days")
    ]
    if drop_existing:
        eval_df = eval_df.drop(columns=drop_existing)
    rows: list[dict[str, Any]] = []
    for _, row in eval_df.iterrows():
        path = ROOT / str(row.get("daily_csv_path", ""))
        if path.exists():
            daily = pd.read_csv(path)
            early_penalty = pd.to_numeric(daily.get("early_reserve_penalty_unscaled", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
            early_overuse = pd.to_numeric(daily.get("early_overuse_increment", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
            water_penalty = pd.to_numeric(daily.get("water_stress_no_response_penalty_unscaled", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
            rows.append(
                {
                    "station_code": row.get("station_code"),
                    "year": row.get("year"),
                    "checkpoint_step": row.get("checkpoint_step"),
                    "early_reserve_penalty_sum_unscaled": float(early_penalty.sum()),
                    "early_overuse_total_mm": float(early_overuse.sum()),
                    "water_stress_no_response_penalty_sum_unscaled": float(water_penalty.sum()),
                    "water_stress_no_response_days": int((water_penalty > 0).sum()),
                }
            )
        else:
            rows.append(
                {
                    "station_code": row.get("station_code"),
                    "year": row.get("year"),
                    "checkpoint_step": row.get("checkpoint_step"),
                    "early_reserve_penalty_sum_unscaled": np.nan,
                    "early_overuse_total_mm": np.nan,
                    "water_stress_no_response_penalty_sum_unscaled": np.nan,
                    "water_stress_no_response_days": np.nan,
                }
            )
    merged = eval_df.merge(pd.DataFrame(rows), on=["station_code", "year", "checkpoint_step"], how="left")
    merged.to_csv(eval_path, index=False, encoding="utf-8-sig")


def write_record(out: Path, doc: Path, suffix: str, total_timesteps: int, checkpoint_steps: list[int]) -> None:
    prefix = "040_11" if not suffix else f"040_11_{suffix}"
    split = base04010.read_csv_or_empty(out / "configs" / f"{prefix}_half_split_selection.csv")
    train = base04010.read_csv_or_empty(out / "evaluation" / f"{prefix}_training_checkpoint_inventory.csv")
    eval_df = base04010.read_csv_or_empty(out / "evaluation" / f"{prefix}_checkpoint_validation_summary.csv")
    by_station = base04010.read_csv_or_empty(out / "evaluation" / f"{prefix}_validation_summary_by_station_checkpoint.csv")
    lines = [
        f"# {prefix} SYA lowIC PPO 早期保留 + 水分胁迫响应奖励记录",
        "",
        "## 结论先说",
        "",
        "- 本任务复用 040_10 的早期灌溉预算保留惩罚。",
        "- 新增 DAP30 后水分胁迫未响应惩罚：如果 swfac_before_action > 0.05 且本步没有灌溉，则扣分。",
        f"- 训练步数：`{total_timesteps}`；checkpoint：`{', '.join(map(str, checkpoint_steps))}`。",
        "",
        "## 新增项",
        "",
        "```text",
        "if DAP >= 31 and swfac_before_action > 0.05 and irrigation == 0:",
        "    no_response_penalty = 45.0 * swfac_before_action",
        "reward_04011 = reward_04010 - no_response_penalty * reward_scale",
        "```",
        "",
        "## 年份划分",
        "",
        base04010.md_table(split, max_rows=120),
        "",
        "## 训练 checkpoint 库存",
        "",
        base04010.md_table(train, max_rows=120),
        "",
        "## 验证集按 checkpoint 汇总",
        "",
        base04010.md_table(by_station, max_rows=120),
        "",
        "## 逐年验证结果",
        "",
        base04010.md_table(eval_df, max_rows=200),
    ]
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dry_run(out: Path, doc: Path, total_timesteps: int, checkpoint_steps: list[int], suffix: str) -> None:
    patch_base_module(out, doc, total_timesteps, checkpoint_steps)
    out.mkdir(parents=True, exist_ok=True)
    (out / "configs").mkdir(parents=True, exist_ok=True)
    split = base04010.load_split_for_sya()
    config = base03222.load_config()
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}" + (f"_{suffix}" if suffix else ""),
        "mode": "dry_run",
        "station": STATION,
        "lowIC_input_root": LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
        "lowIC_input_root_exists": LOWIC_INPUT_ROOT.exists(),
        "prompt_exists": PROMPT.exists(),
        "split_years": split[["station_code", "site", "year", "split"]].to_dict(orient="records"),
        "seed": base03222.SEED,
        "total_timesteps": total_timesteps,
        "checkpoint_steps": checkpoint_steps,
        "config_reward": config.get("reward", {}),
        "early_reserve_penalty": {
            "dap_end": base04010.EARLY_RESERVE_DAP_END,
            "free_allowance_mm": base04010.EARLY_IRRIGATION_FREE_ALLOWANCE,
        },
        "water_stress_response_penalty": {
            "start_dap": WATER_STRESS_RESPONSE_START_DAP,
            "threshold": WATER_STRESS_THRESHOLD,
            "no_response_cost": WATER_STRESS_NO_RESPONSE_COST,
        },
        "next_step_allowed": bool(LOWIC_INPUT_ROOT.exists() and PROMPT.exists() and not split.empty),
    }
    (out / f"{TASK_ID}_dry_run_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("# 040_11 dry-run 记录\n\n```json\n" + json.dumps(result, indent=2, ensure_ascii=False) + "\n```\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def run_training(out: Path, doc: Path, total_timesteps: int, checkpoint_steps: list[int], suffix: str) -> None:
    if not LOWIC_INPUT_ROOT.exists():
        raise FileNotFoundError(LOWIC_INPUT_ROOT)
    if not PROMPT.exists():
        raise FileNotFoundError(PROMPT)
    patch_base_module(out, doc, total_timesteps, checkpoint_steps)
    old_input_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = LOWIC_INPUT_ROOT
    try:
        base03222.main()
        copy_with_task_names(out, suffix)
        add_04011_penalty_summaries(out, suffix)
        write_record(out, doc, suffix, total_timesteps, checkpoint_steps)
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root


def main() -> None:
    parser = argparse.ArgumentParser(description="040_11 SYA lowIC PPO early reserve plus water stress response reward")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument("--checkpoint-steps", default=None)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    checkpoint_steps = parse_checkpoint_steps(args.checkpoint_steps, int(args.timesteps))
    out = out_for_suffix(str(args.suffix))
    doc = doc_for_suffix(str(args.suffix))
    if args.dry_run:
        dry_run(out, doc, int(args.timesteps), checkpoint_steps, str(args.suffix))
    else:
        run_training(out, doc, int(args.timesteps), checkpoint_steps, str(args.suffix))


def parse_checkpoint_steps(raw: str | None, total_timesteps: int) -> list[int]:
    if not raw:
        return [x for x in DEFAULT_CHECKPOINT_STEPS if x <= total_timesteps]
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


if __name__ == "__main__":
    main()
