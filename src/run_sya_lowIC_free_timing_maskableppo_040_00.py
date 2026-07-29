from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

import ppo_safe_rendering
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base03222


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "040_00"
TASK_NAME = "sya_lowIC_free_timing_maskableppo"

LOWIC_INPUT_ROOT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual"
LOWIC_CLASSIFICATION = (
    ROOT
    / "benchmark_results"
    / "039_02_original_vs_lowIC_three_baseline_audit"
    / "tables"
    / "039_02_lowIC_usability_classification.csv"
)

OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"

STATION = "SYA"
SITES = [STATION]


RENAMES = {
    "configs/032_22_half_split_selection.csv": "configs/040_00_half_split_selection.csv",
    "evaluation/032_22_training_checkpoint_inventory_partial.csv": "evaluation/040_00_training_checkpoint_inventory_partial.csv",
    "evaluation/032_22_checkpoint_validation_summary_partial.csv": "evaluation/040_00_checkpoint_validation_summary_partial.csv",
    "evaluation/032_22_training_checkpoint_inventory.csv": "evaluation/040_00_training_checkpoint_inventory.csv",
    "evaluation/032_22_checkpoint_validation_summary.csv": "evaluation/040_00_checkpoint_validation_summary.csv",
    "evaluation/032_22_validation_summary_by_station_checkpoint.csv": "evaluation/040_00_validation_summary_by_station_checkpoint.csv",
    "logs/032_22_training_year_reset_counts.csv": "logs/040_00_training_year_reset_counts.csv",
    "032_22_result.json": "040_00_result.json",
}


def ensure_dirs() -> None:
    for rel in ["configs", "evaluation", "logs", "models", "daily_outputs"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


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


def patch_base_module() -> None:
    """Route the existing 032_22 training engine to the 040_00 SYA lowIC task."""

    base03222.OUT = OUT
    base03222.DOC = DOC
    base03222.PROMPT = PROMPT
    base03222.SITES = list(SITES)
    base03222.summarize_by_station = summarize_by_station_safe


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


def summarize_by_station_safe(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize validation rows without assuming four-scenario baseline gaps exist.

    The original 032_22 summarizer assumes columns such as
    ``gap_yield_vs_four_max`` exist after baseline comparison.  In 040_00 the
    lowIC run can be executed before lowIC four-scenario baselines are rebuilt,
    so those comparison columns may be absent.  They are useful, but not needed
    for the training/evaluation bookkeeping itself.  Missing comparison fields
    are therefore filled with NaN/False instead of crashing at the final record
    stage.
    """

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
    ]:
        if col not in ok.columns:
            ok[col] = pd.NA
        ok[col] = pd.to_numeric(ok[col], errors="coerce")
    if "any_metric_win_four" not in ok.columns:
        ok["any_metric_win_four"] = False

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
            any_metric_win_four_count=("any_metric_win_four", lambda s: int(pd.Series(s).fillna(False).sum())),
            mean_gap_yield_vs_four_max=("gap_yield_vs_four_max", "mean"),
            mean_gap_pfp_n_vs_four_max=("gap_pfp_n_vs_four_max", "mean"),
        )
        .sort_values(["station_code", "checkpoint_step"])
        .reset_index(drop=True)
    )


def build_dry_run_summary() -> dict[str, Any]:
    patch_base_module()
    split = load_split_for_sya()
    cls = load_lowic_classification_for_sya()
    config = base03222.load_config()
    return {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "mode": "dry_run",
        "station": STATION,
        "lowIC_input_root": LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
        "lowIC_input_root_exists": LOWIC_INPUT_ROOT.exists(),
        "prompt_exists": PROMPT.exists(),
        "split_years": split[["station_code", "site", "year", "split"]].to_dict(orient="records"),
        "lowIC_class_counts": cls["lowIC_usability_class"].value_counts().to_dict() if "lowIC_usability_class" in cls.columns else {},
        "seed": base03222.SEED,
        "total_timesteps": base03222.TOTAL_TIMESTEPS,
        "checkpoint_steps": list(base03222.CHECKPOINT_STEPS),
        "config_action_safety": config.get("action_safety", {}),
        "config_discrete_actions": config.get("discrete_actions", {}),
        "config_reward": config.get("reward", {}),
        "next_step_allowed": bool(LOWIC_INPUT_ROOT.exists() and not split.empty),
    }


def write_dry_run_files(summary: dict[str, Any]) -> None:
    ensure_dirs()
    split = pd.DataFrame(summary["split_years"])
    split.to_csv(OUT / "configs" / "040_00_half_split_selection_dry_run.csv", index=False, encoding="utf-8-sig")
    (OUT / "040_00_dry_run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# 040_00 dry-run 记录",
        "",
        "## 结论先说",
        "",
        f"- next_step_allowed: `{summary['next_step_allowed']}`",
        f"- 输入根目录：`{summary['lowIC_input_root']}`",
        f"- 站点：`{summary['station']}`",
        f"- 训练步数：`{summary['total_timesteps']}`",
        f"- checkpoint：`{', '.join(map(str, summary['checkpoint_steps']))}`",
        "",
        "## SYA 年份划分",
        "",
        md_table(split, max_rows=120),
        "",
        "## lowIC 分类计数",
        "",
        "```json",
        json.dumps(summary["lowIC_class_counts"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## 奖励与约束快照",
        "",
        "```json",
        json.dumps(
            {
                "reward": summary["config_reward"],
                "action_safety": summary["config_action_safety"],
                "discrete_actions": summary["config_discrete_actions"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        "```",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_with_task_names() -> None:
    for old_rel, new_rel in RENAMES.items():
        old = OUT / old_rel
        new = OUT / new_rel
        if old.exists():
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, new)

    result_path = OUT / "040_00_result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result.update(
            {
                "task": f"{TASK_ID}_{TASK_NAME}",
                "input_root": LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
                "station_scope": SITES,
                "record_md": DOC.relative_to(ROOT).as_posix(),
                "train_inventory": (OUT / "evaluation" / "040_00_training_checkpoint_inventory.csv").relative_to(ROOT).as_posix(),
                "validation_summary": (OUT / "evaluation" / "040_00_checkpoint_validation_summary.csv").relative_to(ROOT).as_posix(),
                "by_station": (OUT / "evaluation" / "040_00_validation_summary_by_station_checkpoint.csv").relative_to(ROOT).as_posix(),
            }
        )
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def write_clean_record() -> None:
    split_path = OUT / "configs" / "040_00_half_split_selection.csv"
    train_path = OUT / "evaluation" / "040_00_training_checkpoint_inventory.csv"
    reset_path = OUT / "logs" / "040_00_training_year_reset_counts.csv"
    eval_path = OUT / "evaluation" / "040_00_checkpoint_validation_summary.csv"
    by_station_path = OUT / "evaluation" / "040_00_validation_summary_by_station_checkpoint.csv"

    split = read_csv_or_empty(split_path)
    train = read_csv_or_empty(train_path)
    reset = read_csv_or_empty(reset_path)
    eval_df = read_csv_or_empty(eval_path)
    by_station = read_csv_or_empty(by_station_path)
    cls = load_lowic_classification_for_sya()

    lines = [
        "# 040_00 SYA lowIC 自由时序 MaskablePPO 训练记录",
        "",
        "## 结论先说",
        "",
        "- 本任务是 032_22 主线训练框架的 SYA lowIC 包装运行。",
        "- 科学变量只有一个：输入根目录切换到 lowIC 手工修订数据。",
        "- 算法、奖励、动作空间、约束、seed、训练步数、checkpoint 均沿用原配置。",
        "",
        "## 固定配置",
        "",
        f"- 输入根目录：`{LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix()}`",
        f"- 站点：`{', '.join(SITES)}`",
        f"- seed：`{base03222.SEED}`",
        f"- 训练步数：`{base03222.TOTAL_TIMESTEPS}`",
        f"- checkpoint：`{', '.join(map(str, base03222.CHECKPOINT_STEPS))}`",
        "- 注意：底层复用 032_22 训练引擎，部分运行标签可能保留 `032_22` 字样；040_00 命名表格已另存。",
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
        "",
        "## 解释边界",
        "",
        "- 040_00 不是新算法，也不是调参实验。",
        "- 040_00 只能说明 lowIC 输入条件下，原自由时序 MaskablePPO 主线在 SYA 上的表现。",
        "- 如果结果改善或变差，优先解释为初始土壤水氮条件改变后的训练/验证响应，而不是算法结构改变。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_training() -> None:
    if not LOWIC_INPUT_ROOT.exists():
        raise FileNotFoundError(f"lowIC input root does not exist: {LOWIC_INPUT_ROOT}")
    if not PROMPT.exists():
        raise FileNotFoundError(f"prompt does not exist: {PROMPT}")

    patch_base_module()
    old_input_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = LOWIC_INPUT_ROOT
    try:
        base03222.main()
        copy_with_task_names()
        write_clean_record()
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root


def main() -> None:
    parser = argparse.ArgumentParser(description="040_00 SYA lowIC free-timing MaskablePPO wrapper")
    parser.add_argument("--dry-run", action="store_true", help="Only verify routing/config/year split; do not train.")
    args = parser.parse_args()

    if args.dry_run:
        summary = build_dry_run_summary()
        write_dry_run_files(summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    run_training()


if __name__ == "__main__":
    main()
