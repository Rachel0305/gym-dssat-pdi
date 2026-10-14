from __future__ import annotations

import importlib
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
base = importlib.import_module("run_five_site_half_split_stress_aware_maskableppo_batch_032_22")

TASK_ID = "036_01"
OUT = ROOT / "benchmark_results" / "036_01_original_free_timing_maskableppo_ic1_linked_five_site_half_split_rerun"
DOC = ROOT / "docs" / "036_01_original_free_timing_maskableppo_ic1_linked_five_site_half_split_rerun_record.md"
PROMPT = ROOT / "prompts" / "036_01_original_free_timing_maskableppo_ic1_linked_five_site_half_split_rerun.md"


base.OUT = OUT
base.DOC = DOC
base.PROMPT = PROMPT


def copy_with_036_names() -> None:
    eval_dir = OUT / "evaluation"
    log_dir = OUT / "logs"
    pairs = [
        (eval_dir / "032_22_training_checkpoint_inventory.csv", eval_dir / "036_01_training_checkpoint_inventory.csv"),
        (eval_dir / "032_22_checkpoint_validation_summary.csv", eval_dir / "036_01_checkpoint_validation_summary.csv"),
        (eval_dir / "032_22_validation_summary_by_station_checkpoint.csv", eval_dir / "036_01_validation_summary_by_station_checkpoint.csv"),
        (eval_dir / "032_22_training_checkpoint_inventory_partial.csv", eval_dir / "036_01_training_checkpoint_inventory_partial.csv"),
        (eval_dir / "032_22_checkpoint_validation_summary_partial.csv", eval_dir / "036_01_checkpoint_validation_summary_partial.csv"),
        (log_dir / "032_22_training_year_reset_counts.csv", log_dir / "036_01_training_year_reset_counts.csv"),
        (OUT / "032_22_result.json", OUT / "036_01_result.json"),
    ]
    for src, dst in pairs:
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)

    result_path = OUT / "036_01_result.json"
    if result_path.exists():
        data = json.loads(result_path.read_text(encoding="utf-8"))
        data["task"] = "036_01_original_free_timing_maskableppo_ic1_linked_five_site_half_split_rerun"
        data["record_md"] = DOC.relative_to(ROOT).as_posix()
        data["train_inventory"] = (eval_dir / "036_01_training_checkpoint_inventory.csv").relative_to(ROOT).as_posix()
        data["validation_summary"] = (eval_dir / "036_01_checkpoint_validation_summary.csv").relative_to(ROOT).as_posix()
        data["by_station"] = (eval_dir / "036_01_validation_summary_by_station_checkpoint.csv").relative_to(ROOT).as_posix()
        result_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_036_record_appendix() -> None:
    if not DOC.exists():
        return
    text = DOC.read_text(encoding="utf-8")
    appendix = [
        "",
        "## 036_01 主线说明",
        "",
        "- 本次是 032_22 原自由时序 stress-aware MaskablePPO 在 IC=1 + linked action 修复后的正式重跑。",
        "- 原训练框架、reward、动作约束、PPO 超参数、100K 训练步数、25K/50K/75K/100K checkpoint 均保持不变。",
        "- 本次不采用 035_04/035_06 的 reward 改动；035_04/035_06 只作为旁支探索记录保留。",
        "- 036_00 已在同一主线下确认 IC=1、multisite_013 输入源和 linked 动作接口。",
        "",
    ]
    if "## 036_01 主线说明" not in text:
        DOC.write_text(text + "\n".join(appendix), encoding="utf-8")


def print_quick_summary() -> None:
    val_path = OUT / "evaluation" / "036_01_checkpoint_validation_summary.csv"
    by_path = OUT / "evaluation" / "036_01_validation_summary_by_station_checkpoint.csv"
    if not val_path.exists():
        return
    val = pd.read_csv(val_path)
    by = pd.read_csv(by_path) if by_path.exists() else pd.DataFrame()
    print(
        json.dumps(
            {
                "task": TASK_ID,
                "validation_rows": int(len(val)),
                "stations": sorted(val["station_code"].dropna().astype(str).unique().tolist()) if "station_code" in val else [],
                "record_md": DOC.relative_to(ROOT).as_posix(),
                "validation_summary": val_path.relative_to(ROOT).as_posix(),
                "by_station": by_path.relative_to(ROOT).as_posix() if by_path.exists() else "",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not by.empty:
        print(by.to_string(index=False))


if __name__ == "__main__":
    base.main()
    copy_with_036_names()
    write_036_record_appendix()
    print_quick_summary()
