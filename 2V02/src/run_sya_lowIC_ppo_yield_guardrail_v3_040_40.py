"""040_40: SYA lowIC free-timing MaskablePPO with terminal yield guardrail.

This is a narrow follow-up to 040_36/040_39.

It keeps the 040_36 action constraints and reward terms, and adds only one
terminal penalty when final yield falls below 98% of the same-year official
extension expert yield.
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
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base03222
import run_sya_lowIC_ppo_early_irrigation_reserve_penalty_040_10 as util04010
import run_sya_lowIC_ppo_late_irrigation_reserve_mask_040_36 as base04036


TASK_ID = "040_40"
TASK_NAME = "sya_lowIC_ppo_yield_guardrail_v3"
BASE_OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
BASE_DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"

LOWIC_INPUT_ROOT = base04036.LOWIC_INPUT_ROOT
STATION = base04036.STATION
SITES = [STATION]
DEFAULT_TOTAL_TIMESTEPS = 100_000
DEFAULT_CHECKPOINT_STEPS = [25_000, 50_000, 75_000, 100_000]

YIELD_TARGET_FRACTION = 0.98
BASELINE_SUMMARY = (
    ROOT
    / "benchmark_results"
    / "040_21_sya_lowIC_four_baseline_rebuild"
    / "evaluation"
    / "040_21_baseline_summary.csv"
)
BASELINE_TARGET_CSV = (
    ROOT
    / "benchmark_results"
    / "040_41_sya_lowIC_official_expert_yield_targets"
    / "evaluation"
    / "040_41_official_expert_yield_targets.csv"
)

ORIGINAL_COPY2 = shutil.copy2
ORIGINAL_BASE_MAKE_ENV = base03222.base.make_env

RENAMES = {
    "configs/032_22_half_split_selection.csv": "configs/040_40_half_split_selection.csv",
    "evaluation/032_22_training_checkpoint_inventory_partial.csv": "evaluation/040_40_training_checkpoint_inventory_partial.csv",
    "evaluation/032_22_checkpoint_validation_summary_partial.csv": "evaluation/040_40_checkpoint_validation_summary_partial.csv",
    "evaluation/032_22_training_checkpoint_inventory.csv": "evaluation/040_40_training_checkpoint_inventory.csv",
    "evaluation/032_22_checkpoint_validation_summary.csv": "evaluation/040_40_checkpoint_validation_summary.csv",
    "evaluation/032_22_validation_summary_by_station_checkpoint.csv": "evaluation/040_40_validation_summary_by_station_checkpoint.csv",
    "logs/032_22_training_year_reset_counts.csv": "logs/040_40_training_year_reset_counts.csv",
    "032_22_result.json": "040_40_result.json",
}


def safe_copy2(src: str | Path, dst: str | Path, *args, **kwargs) -> str:
    try:
        return str(ORIGINAL_COPY2(src, dst, *args, **kwargs))
    except PermissionError:
        dst_path = Path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst_path)
        return str(dst_path)


def expert_yield_targets() -> dict[tuple[str, int], float]:
    source = BASELINE_TARGET_CSV if BASELINE_TARGET_CSV.exists() else BASELINE_SUMMARY
    if not source.exists():
        raise FileNotFoundError(source)
    df = pd.read_csv(source, keep_default_na=False)
    df = df[
        df["station_code"].astype(str).eq(STATION)
        & df["scenario"].astype(str).eq("official_extension_expert")
    ].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df["grain_yield_kg_ha"] = pd.to_numeric(df["grain_yield_kg_ha"], errors="coerce")
    return {
        (str(row.station_code), int(row.year)): float(row.grain_yield_kg_ha) * YIELD_TARGET_FRACTION
        for row in df.itertuples(index=False)
        if np.isfinite(float(row.grain_yield_kg_ha))
    }


YIELD_TARGETS: dict[tuple[str, int], float] | None = None


def get_target_yield(station: str, year: int) -> float:
    global YIELD_TARGETS
    if YIELD_TARGETS is None:
        YIELD_TARGETS = expert_yield_targets()
    key = (str(station), int(year))
    if key not in YIELD_TARGETS:
        raise KeyError(f"Missing yield guardrail target for {station}{year}")
    return float(YIELD_TARGETS[key])


class YieldGuardrailLateReserveWrapper(base04036.LateIrrigationReserveMaskWrapper):
    """040_36 wrapper plus terminal yield-deficit penalty."""

    def __init__(self, env, config: dict, station: str, year: int):
        super().__init__(env, config)
        self.guardrail_station = str(station)
        self.guardrail_year = int(year)
        self.guardrail_target_yield = get_target_yield(station, year)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        done = bool(terminated or truncated)
        reward_scale = float(self.config.get("reward", {}).get("reward_scale", 1.0))
        yield_coef = float(self.config.get("reward", {}).get("yield_coef", 0.158))
        final_yield = base04036.base04028.base04026.base032.scalar(self.last_obs_dict.get("grnwt", np.nan), np.nan)
        deficit = 0.0
        penalty_unscaled = 0.0
        penalty_scaled = 0.0
        if done and np.isfinite(final_yield):
            deficit = max(float(self.guardrail_target_yield) - float(final_yield), 0.0)
            penalty_unscaled = yield_coef * deficit
            penalty_scaled = penalty_unscaled * reward_scale
            reward = float(reward) - penalty_scaled
        self.last_action_info.update(
            {
                "yield_guardrail_enabled": True,
                "yield_guardrail_target_fraction": YIELD_TARGET_FRACTION,
                "yield_guardrail_target_yield": float(self.guardrail_target_yield),
                "yield_guardrail_final_yield": float(final_yield) if np.isfinite(final_yield) else np.nan,
                "yield_guardrail_deficit": float(deficit),
                "yield_guardrail_penalty_unscaled": float(penalty_unscaled),
                "yield_guardrail_penalty_scaled": float(penalty_scaled),
                "reward_after_yield_guardrail": float(reward),
            }
        )
        return obs, float(reward), terminated, truncated, info


def make_env_with_yield_guardrail(
    config: dict,
    env_config: dict,
    station: str,
    year: int,
    seed: int,
    run_tag: str,
    evaluation: bool = False,
):
    base_env = direct_ppo.make_base_env(env_config, station, year, seed, run_tag, evaluation=evaluation)
    return YieldGuardrailLateReserveWrapper(base_env, config, station, int(year))


def out_for_suffix(suffix: str) -> Path:
    return ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}_{suffix}" if suffix else BASE_OUT


def doc_for_suffix(suffix: str) -> Path:
    return ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_{suffix}_record.md" if suffix else BASE_DOC


def parse_checkpoint_steps(raw: str | None, total_timesteps: int) -> list[int]:
    if not raw:
        return [x for x in DEFAULT_CHECKPOINT_STEPS if x <= total_timesteps]
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def load_config() -> dict[str, Any]:
    cfg = base04036.load_config()
    cfg["reward"]["terminal_yield_guardrail"] = {
        "enabled": True,
        "target": "0.98 * same-year official_extension_expert grain yield from 040_41 target table when available, otherwise 040_21 lowIC baseline",
        "target_fraction": YIELD_TARGET_FRACTION,
        "penalty_formula": "yield_coef * max(0, target_yield - final_grnwt) * reward_scale",
        "yield_coef": float(cfg["reward"]["yield_coef"]),
        "reward_scale": float(cfg["reward"].get("reward_scale", 1.0)),
        "reason": "040_39 found checkpoint selection alone cannot solve insufficient yield; add a conservative yield-deficit guardrail without changing action constraints.",
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
    base03222.base.make_env = make_env_with_yield_guardrail
    base03222.shutil.copy2 = safe_copy2


def restore_base_module() -> None:
    base03222.base.make_env = ORIGINAL_BASE_MAKE_ENV


def copy_with_task_names(out: Path, suffix: str) -> None:
    for old_rel, new_rel in RENAMES.items():
        old = out / old_rel
        if suffix:
            new_rel = new_rel.replace("040_40_", f"040_40_{suffix}_")
        new = out / new_rel
        if old.exists():
            new.parent.mkdir(parents=True, exist_ok=True)
            safe_copy2(old, new)
    generic_result = out / "032_22_result.json"
    result_path = out / ("040_40_result.json" if not suffix else f"040_40_{suffix}_result.json")
    if generic_result.exists():
        result = json.loads(generic_result.read_text(encoding="utf-8"))
        result.update(
            {
                "task": f"{TASK_ID}_{TASK_NAME}" + (f"_{suffix}" if suffix else ""),
                "input_root": LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
                "station_scope": SITES,
                "terminal_yield_guardrail": {
                    "enabled": True,
                    "target_fraction": YIELD_TARGET_FRACTION,
        "target_source": (BASELINE_TARGET_CSV if BASELINE_TARGET_CSV.exists() else BASELINE_SUMMARY).relative_to(ROOT).as_posix(),
                },
                "late_irrigation_reserve_mask": {
                    "dap_end": base04036.LATE_RESERVE_DAP_END,
                    "pre_late_cum_irrigation_cap_mm": base04036.PRE_LATE_CUM_IRRIGATION_CAP,
                    "reserved_for_after_dap90_mm": base04036.LATE_RESERVED_IRRIGATION_MM,
                },
                "coarse_action_levels": {
                    "irrigation_levels": base04036.IRRIGATION_LEVELS,
                    "nitrogen_levels": base04036.NITROGEN_LEVELS,
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


def add_04040_summaries(out: Path, suffix: str) -> None:
    prefix = "040_40" if not suffix else f"040_40_{suffix}"
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
            "yield_guardrail_target_yield": np.nan,
            "yield_guardrail_deficit": np.nan,
            "yield_guardrail_penalty_unscaled": np.nan,
            "yield_guardrail_penalty_scaled": np.nan,
        }
        if path.exists():
            daily = pd.read_csv(path)
            dap = pd.to_numeric(daily.get("dap", pd.Series(dtype=float)), errors="coerce")
            irr = pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
            n = pd.to_numeric(daily.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
            i1_30 = float(irr[dap <= base04036.base04026.EARLY_DAP_END].sum())
            i31_60 = float(
                irr[
                    (dap > base04036.base04026.EARLY_DAP_END)
                    & (dap <= base04036.base04026.MID_DAP_END)
                ].sum()
            )
            i61_90 = float(
                irr[
                    (dap > base04036.base04026.MID_DAP_END)
                    & (dap <= base04036.LATE_RESERVE_DAP_END)
                ].sum()
            )
            i91 = float(irr[dap > base04036.LATE_RESERVE_DAP_END].sum())
            i1_90 = i1_30 + i31_60 + i61_90
            terminal = daily.tail(1)
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
                    "late_reserve_violation": bool(i1_90 - base04036.PRE_LATE_CUM_IRRIGATION_CAP > 1e-9),
                    "late_reserve_used_after_dap90": bool(i91 > 1e-9),
                }
            )
            if len(terminal):
                for col in [
                    "yield_guardrail_target_yield",
                    "yield_guardrail_deficit",
                    "yield_guardrail_penalty_unscaled",
                    "yield_guardrail_penalty_scaled",
                ]:
                    rec[col] = pd.to_numeric(terminal.get(col, pd.Series([np.nan])).iloc[0], errors="coerce")
        rows.append(rec)
    extra = pd.DataFrame(rows)
    drop_cols = [c for c in extra.columns if c in eval_df.columns and c not in ["station_code", "year", "checkpoint_step"]]
    if drop_cols:
        eval_df = eval_df.drop(columns=drop_cols)
    merged = eval_df.merge(extra, on=["station_code", "year", "checkpoint_step"], how="left")
    merged.to_csv(eval_path, index=False, encoding="utf-8-sig")

    by_path = out / "evaluation" / f"{prefix}_validation_summary_by_station_checkpoint.csv"
    if by_path.exists():
        by_station = util04010.summarize_by_station_safe(merged)
        extra_by = (
            merged.groupby(["station_code", "checkpoint_step"], as_index=False)
            .agg(
                mean_yield_guardrail_deficit=("yield_guardrail_deficit", "mean"),
                max_yield_guardrail_deficit=("yield_guardrail_deficit", "max"),
                mean_yield_guardrail_penalty_scaled=("yield_guardrail_penalty_scaled", "mean"),
            )
        )
        by_station = by_station.merge(extra_by, on=["station_code", "checkpoint_step"], how="left")
        action_by = (
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
        by_station = by_station.merge(action_by, on=["station_code", "checkpoint_step"], how="left")
        by_station.to_csv(by_path, index=False, encoding="utf-8-sig")


def write_clean_record(out: Path, doc: Path, suffix: str, total_timesteps: int, checkpoint_steps: list[int]) -> None:
    prefix = "040_40" if not suffix else f"040_40_{suffix}"
    split = read_csv_or_empty(out / "configs" / f"{prefix}_half_split_selection.csv")
    train = read_csv_or_empty(out / "evaluation" / f"{prefix}_training_checkpoint_inventory.csv")
    eval_df = read_csv_or_empty(out / "evaluation" / f"{prefix}_checkpoint_validation_summary.csv")
    by_station = read_csv_or_empty(out / "evaluation" / f"{prefix}_validation_summary_by_station_checkpoint.csv")
    cols = ["station_code", "site", "year", "split"]
    split_show = split[cols] if set(cols).issubset(split.columns) else split
    lines = [
        f"# {prefix} SYA lowIC PPO 保产优先 reward v3 记录",
        "",
        "## 结论边界",
        "",
        "- 本任务只在 040_36 基础上新增 terminal yield guardrail，不改动作空间、不改年份划分、不改 lowIC 输入。",
        "- 目的不是让 PPO 无条件追产量，而是防止为了节水节氮造成明显减产。",
        f"- 低产线：同年 official_extension_expert 产量的 {YIELD_TARGET_FRACTION:.2%}。",
        f"- 训练步数：`{total_timesteps}`；checkpoint：`{', '.join(map(str, checkpoint_steps))}`。",
        "",
        "## 固定配置",
        "",
        f"- lowIC 输入目录：`{LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix()}`",
        f"- 输入目录存在：`{LOWIC_INPUT_ROOT.exists()}`",
        f"- 产量目标来源：`{(BASELINE_TARGET_CSV if BASELINE_TARGET_CSV.exists() else BASELINE_SUMMARY).relative_to(ROOT).as_posix()}`",
        f"- 灌溉档位：`{base04036.IRRIGATION_LEVELS}`",
        f"- 施氮档位：`{base04036.NITROGEN_LEVELS}`",
        f"- DAP90 前累计灌溉上限：{base04036.PRE_LATE_CUM_IRRIGATION_CAP} mm",
        f"- DAP91+ 保留水量：{base04036.LATE_RESERVED_IRRIGATION_MM} mm",
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
    out = out_for_suffix(suffix)
    doc = doc_for_suffix(suffix)
    patch_base_module(out, doc, total_timesteps, checkpoint_steps)
    config = load_config()
    split = base03222.load_split().copy()
    split = split[split["station_code"].eq(STATION)].sort_values(["year"]).reset_index(drop=True)
    targets = expert_yield_targets()
    target_preview = [
        {"station_code": s, "year": y, "target_yield_98pct_expert": round(v, 3)}
        for (s, y), v in sorted(targets.items())
        if s == STATION
    ]
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
        "yield_guardrail_target_preview": target_preview,
        "next_step_allowed": LOWIC_INPUT_ROOT.exists() and PROMPT.exists() and not split.empty and len(target_preview) >= len(split),
    }
    restore_base_module()
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
        add_04040_summaries(out, suffix)
        write_clean_record(out, doc, suffix, total_timesteps, checkpoint_steps)
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root
        restore_base_module()

    prefix = "040_40" if not suffix else f"040_40_{suffix}"
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
        "yield_guardrail_target_fraction": YIELD_TARGET_FRACTION,
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
