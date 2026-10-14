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


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "040_10"
TASK_NAME = "sya_lowIC_ppo_early_irrigation_reserve_penalty"
BASE_OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
BASE_DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"

LOWIC_INPUT_ROOT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual"
LOWIC_CLASSIFICATION = (
    ROOT
    / "benchmark_results"
    / "039_02_original_vs_lowIC_three_baseline_audit"
    / "tables"
    / "039_02_lowIC_usability_classification.csv"
)

STATION = "SYA"
SITES = [STATION]
DEFAULT_TOTAL_TIMESTEPS = 100_000
DEFAULT_CHECKPOINT_STEPS = [25_000, 50_000, 75_000, 100_000]

EARLY_RESERVE_DAP_END = 30
EARLY_IRRIGATION_FREE_ALLOWANCE = 90.0
EARLY_OVERUSE_COST_SOURCE = "water_cost"


RENAMES = {
    "configs/032_22_half_split_selection.csv": "configs/040_10_half_split_selection.csv",
    "evaluation/032_22_training_checkpoint_inventory_partial.csv": "evaluation/040_10_training_checkpoint_inventory_partial.csv",
    "evaluation/032_22_checkpoint_validation_summary_partial.csv": "evaluation/040_10_checkpoint_validation_summary_partial.csv",
    "evaluation/032_22_training_checkpoint_inventory.csv": "evaluation/040_10_training_checkpoint_inventory.csv",
    "evaluation/032_22_checkpoint_validation_summary.csv": "evaluation/040_10_checkpoint_validation_summary.csv",
    "evaluation/032_22_validation_summary_by_station_checkpoint.csv": "evaluation/040_10_validation_summary_by_station_checkpoint.csv",
    "logs/032_22_training_year_reset_counts.csv": "logs/040_10_training_year_reset_counts.csv",
    "032_22_result.json": "040_10_result.json",
}


def out_for_suffix(suffix: str) -> Path:
    return ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}_{suffix}" if suffix else BASE_OUT


def doc_for_suffix(suffix: str) -> Path:
    return ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_{suffix}_record.md" if suffix else BASE_DOC


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


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


class EarlyIrrigationReservePenaltyWrapper(base032.StressAwareDiscreteWrapper):
    """040_10 wrapper: add soft penalty for DAP1-30 irrigation above 90 mm."""

    def step(self, action):
        reward_obs = super().step(action)
        obs, reward, terminated, truncated, info = reward_obs
        dap = self._dap()
        info_dict = dict(self.last_action_info)
        previous_i = float(info_dict.get("previous_cumulative_irrigation", 0.0) or 0.0)
        current_i = float(info_dict.get("season_cumulative_irrigation", previous_i) or previous_i)
        reward_scale = float(info_dict.get("reward_scale", self.config.get("reward", {}).get("reward_scale", 1.0)))
        water_cost = float(self.config.get("reward", {}).get("water_cost", 1.1))
        early_overuse_cost = water_cost
        increment = 0.0
        if dap <= EARLY_RESERVE_DAP_END:
            before_excess = max(0.0, previous_i - EARLY_IRRIGATION_FREE_ALLOWANCE)
            after_excess = max(0.0, current_i - EARLY_IRRIGATION_FREE_ALLOWANCE)
            increment = max(0.0, after_excess - before_excess)
        penalty_unscaled = early_overuse_cost * increment
        penalty_scaled = penalty_unscaled * reward_scale
        new_reward = float(reward) - penalty_scaled
        self.last_action_info.update(
            {
                "early_reserve_dap_end": EARLY_RESERVE_DAP_END,
                "early_irrigation_free_allowance": EARLY_IRRIGATION_FREE_ALLOWANCE,
                "early_overuse_increment": float(increment),
                "early_overuse_cost": float(early_overuse_cost),
                "early_reserve_penalty_unscaled": float(penalty_unscaled),
                "early_reserve_penalty_scaled": float(penalty_scaled),
                "reward_before_early_reserve_penalty": float(reward),
                "reward_stress_aware": float(new_reward),
                "reward_unscaled_04010": float(info_dict.get("reward_unscaled", 0.0)) - float(penalty_unscaled),
            }
        )
        return obs, new_reward, terminated, truncated, info


def summarize_by_station_safe(eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df.empty:
        return pd.DataFrame()
    ok = eval_df[eval_df["run_status"].astype(str).str.startswith("ok")].copy()
    if ok.empty:
        return pd.DataFrame()
    if "site" not in ok.columns and "station_code" in ok.columns:
        ok["site"] = ok["station_code"].map(base03222.SITE_NAMES)
    for col in [
        "final_grnwt",
        "total_irrigation",
        "total_n",
        "PFP_N",
        "max_nstres",
        "max_swfac",
        "gap_yield_vs_four_max",
        "gap_pfp_n_vs_four_max",
        "early_reserve_penalty_sum_unscaled",
    ]:
        if col not in ok.columns:
            ok[col] = pd.NA
        ok[col] = pd.to_numeric(ok[col], errors="coerce")
    if "any_metric_win_four" not in ok.columns:
        ok["any_metric_win_four"] = False
    ok["any_metric_win_four_bool"] = ok["any_metric_win_four"].map(
        lambda x: str(x).strip().lower() in {"true", "1", "yes"}
    )
    return (
        ok.groupby(["station_code", "site", "checkpoint_step"], as_index=False)
        .agg(
            validation_years=("year", "nunique"),
            mean_final_grnwt=("final_grnwt", "mean"),
            mean_total_irrigation=("total_irrigation", "mean"),
            mean_total_n=("total_n", "mean"),
            mean_PFP_N=("PFP_N", "mean"),
            max_nstres=("max_nstres", "max"),
            max_swfac=("max_swfac", "max"),
            mean_early_reserve_penalty_unscaled=("early_reserve_penalty_sum_unscaled", "mean"),
            any_metric_win_four_count=("any_metric_win_four_bool", "sum"),
            mean_gap_yield_vs_four_max=("gap_yield_vs_four_max", "mean"),
            mean_gap_pfp_n_vs_four_max=("gap_pfp_n_vs_four_max", "mean"),
        )
        .sort_values(["station_code", "checkpoint_step"])
        .reset_index(drop=True)
    )


def patch_base_module(out: Path, doc: Path, total_timesteps: int, checkpoint_steps: list[int]) -> None:
    base03222.OUT = out
    base03222.DOC = doc
    base03222.PROMPT = PROMPT
    base03222.SITES = list(SITES)
    base03222.TOTAL_TIMESTEPS = int(total_timesteps)
    base03222.CHECKPOINT_STEPS = [int(x) for x in checkpoint_steps]
    base03222.summarize_by_station = summarize_by_station_safe
    base03222.base.StressAwareDiscreteWrapper = EarlyIrrigationReservePenaltyWrapper


def load_split_for_sya() -> pd.DataFrame:
    split = base03222.load_split().copy()
    split["year"] = pd.to_numeric(split["year"], errors="coerce").astype(int)
    return split.sort_values(["station_code", "year"]).reset_index(drop=True)


def load_lowic_classification_for_sya() -> pd.DataFrame:
    if not LOWIC_CLASSIFICATION.exists():
        return pd.DataFrame()
    df = pd.read_csv(LOWIC_CLASSIFICATION, keep_default_na=False)
    if "station_code" not in df.columns:
        return pd.DataFrame()
    df = df[df["station_code"].eq(STATION)].copy()
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    return df.sort_values(["station_code", "year"]).reset_index(drop=True)


def add_04010_penalty_summaries(out: Path, suffix: str) -> None:
    prefix = "040_10" if not suffix else f"040_10_{suffix}"
    eval_path = out / "evaluation" / f"{prefix}_checkpoint_validation_summary.csv"
    if not eval_path.exists():
        return
    eval_df = pd.read_csv(eval_path, keep_default_na=False)
    drop_existing = [
        c
        for c in eval_df.columns
        if c == "early_reserve_penalty_sum_unscaled"
        or c == "early_overuse_total_mm"
        or c.startswith("early_reserve_penalty_sum_unscaled_")
        or c.startswith("early_overuse_total_mm_")
    ]
    if drop_existing:
        eval_df = eval_df.drop(columns=drop_existing)
    penalty_sums: list[dict[str, Any]] = []
    for _, row in eval_df.iterrows():
        path = ROOT / str(row.get("daily_csv_path", ""))
        if not path.exists():
            penalty_sums.append({"station_code": row.get("station_code"), "year": row.get("year"), "checkpoint_step": row.get("checkpoint_step"), "early_reserve_penalty_sum_unscaled": np.nan, "early_overuse_total_mm": np.nan})
            continue
        daily = pd.read_csv(path)
        penalty = pd.to_numeric(daily.get("early_reserve_penalty_unscaled", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        overuse = pd.to_numeric(daily.get("early_overuse_increment", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        penalty_sums.append(
            {
                "station_code": row.get("station_code"),
                "year": row.get("year"),
                "checkpoint_step": row.get("checkpoint_step"),
                "early_reserve_penalty_sum_unscaled": float(penalty.sum()),
                "early_overuse_total_mm": float(overuse.sum()),
            }
        )
    p = pd.DataFrame(penalty_sums)
    merged = eval_df.merge(p, on=["station_code", "year", "checkpoint_step"], how="left")
    merged.to_csv(eval_path, index=False, encoding="utf-8-sig")
    by_path = out / "evaluation" / f"{prefix}_validation_summary_by_station_checkpoint.csv"
    if by_path.exists():
        by_station = summarize_by_station_safe(merged)
        by_station.to_csv(by_path, index=False, encoding="utf-8-sig")


def copy_with_task_names(out: Path, suffix: str) -> None:
    for old_rel, new_rel in RENAMES.items():
        old = out / old_rel
        if suffix:
            new_rel = new_rel.replace("040_10_", f"040_10_{suffix}_")
        new = out / new_rel
        if old.exists():
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, new)
    result_path = out / ("040_10_result.json" if not suffix else f"040_10_{suffix}_result.json")
    generic_result = out / "032_22_result.json"
    if generic_result.exists():
        result = json.loads(generic_result.read_text(encoding="utf-8"))
        result.update(
            {
                "task": f"{TASK_ID}_{TASK_NAME}" + (f"_{suffix}" if suffix else ""),
                "input_root": LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
                "station_scope": SITES,
                "early_reserve_penalty": {
                    "dap_end": EARLY_RESERVE_DAP_END,
                    "free_allowance_mm": EARLY_IRRIGATION_FREE_ALLOWANCE,
                    "early_overuse_cost_source": EARLY_OVERUSE_COST_SOURCE,
                },
            }
        )
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def write_clean_record(out: Path, doc: Path, suffix: str, total_timesteps: int, checkpoint_steps: list[int]) -> None:
    prefix = "040_10" if not suffix else f"040_10_{suffix}"
    split = read_csv_or_empty(out / "configs" / f"{prefix}_half_split_selection.csv")
    train = read_csv_or_empty(out / "evaluation" / f"{prefix}_training_checkpoint_inventory.csv")
    reset = read_csv_or_empty(out / "logs" / f"{prefix}_training_year_reset_counts.csv")
    eval_df = read_csv_or_empty(out / "evaluation" / f"{prefix}_checkpoint_validation_summary.csv")
    by_station = read_csv_or_empty(out / "evaluation" / f"{prefix}_validation_summary_by_station_checkpoint.csv")
    cls = load_lowic_classification_for_sya()
    lines = [
        f"# {prefix} SYA lowIC PPO 早期灌溉预算保留惩罚训练记录",
        "",
        "## 结论先说",
        "",
        "- 本任务复用 040_00 自由时序 MaskablePPO 主线，只新增 DAP1-30 早期灌溉超过 90 mm 的软惩罚。",
        "- 不改变动作空间、总季节上限、7天最小间隔、DAP90后禁氮、PPO网络结构或训练/验证年份划分。",
        f"- 训练步数：`{total_timesteps}`；checkpoint：`{', '.join(map(str, checkpoint_steps))}`。",
        "",
        "## 新增 reward 项",
        "",
        "```text",
        "early_overuse_increment = max(0, cumulative_I_after - 90) - max(0, cumulative_I_before - 90), only if DAP <= 30",
        "early_reserve_penalty = water_cost * early_overuse_increment",
        "reward_04010 = reward_04000 - early_reserve_penalty * reward_scale",
        "```",
        "",
        "## SYA lowIC 可用性分类",
        "",
        md_table(
            cls[["station_code", "year", "lowIC_usability_class", "lowIC_usability_reason"]]
            if {"station_code", "year", "lowIC_usability_class", "lowIC_usability_reason"}.issubset(cls.columns)
            else cls,
            max_rows=120,
        ),
        "",
        "## 年份划分",
        "",
        md_table(split[[c for c in ["station_code", "site", "year", "split", "selected_for_train", "selected_for_eval"] if c in split.columns]], max_rows=120),
        "",
        "## 训练 checkpoint 库存",
        "",
        md_table(train, max_rows=120),
        "",
        "## 训练年份采样次数",
        "",
        md_table(reset, max_rows=120),
        "",
        "## 验证集按 checkpoint 汇总",
        "",
        md_table(by_station, max_rows=120),
        "",
        "## 逐年验证结果",
        "",
        md_table(eval_df, max_rows=200),
    ]
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dry_run(out: Path, doc: Path, total_timesteps: int, checkpoint_steps: list[int], suffix: str) -> None:
    patch_base_module(out, doc, total_timesteps, checkpoint_steps)
    out.mkdir(parents=True, exist_ok=True)
    (out / "configs").mkdir(parents=True, exist_ok=True)
    split = load_split_for_sya()
    config = base03222.load_config()
    summary = {
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
        "config_action_safety": config.get("action_safety", {}),
        "config_discrete_actions": config.get("discrete_actions", {}),
        "config_reward": config.get("reward", {}),
        "early_reserve_penalty": {
            "dap_end": EARLY_RESERVE_DAP_END,
            "free_allowance_mm": EARLY_IRRIGATION_FREE_ALLOWANCE,
            "early_overuse_cost_source": EARLY_OVERUSE_COST_SOURCE,
        },
        "next_step_allowed": bool(LOWIC_INPUT_ROOT.exists() and PROMPT.exists() and not split.empty),
    }
    split.to_csv(out / "configs" / f"{TASK_ID}_half_split_selection_dry_run.csv", index=False, encoding="utf-8-sig")
    (out / f"{TASK_ID}_dry_run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("# 040_10 dry-run 记录\n\n```json\n" + json.dumps(summary, indent=2, ensure_ascii=False) + "\n```\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


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
        add_04010_penalty_summaries(out, suffix)
        write_clean_record(out, doc, suffix, total_timesteps, checkpoint_steps)
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root


def parse_checkpoint_steps(raw: str | None, total_timesteps: int) -> list[int]:
    if not raw:
        return [x for x in DEFAULT_CHECKPOINT_STEPS if x <= total_timesteps]
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="040_10 SYA lowIC PPO with early irrigation reserve penalty")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument("--checkpoint-steps", default=None)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    checkpoint_steps = parse_checkpoint_steps(args.checkpoint_steps, int(args.timesteps))
    out = out_for_suffix(args.suffix)
    doc = doc_for_suffix(args.suffix)
    if args.dry_run:
        dry_run(out, doc, int(args.timesteps), checkpoint_steps, str(args.suffix))
    else:
        run_training(out, doc, int(args.timesteps), checkpoint_steps, str(args.suffix))


if __name__ == "__main__":
    main()
