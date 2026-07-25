from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base
from ppo_safe_rendering import source_weather_path


OUT = ROOT / "benchmark_results" / "033_04_multisite_input_enabled_five_site_half_split_maskableppo_rerun"
DOC = ROOT / "docs" / "033_04_multisite_input_enabled_five_site_half_split_maskableppo_rerun_record.md"
PROMPT = ROOT / "prompts" / "033_04_multisite_input_enabled_five_site_half_split_maskableppo_rerun.md"
SKIPPED_WTH = OUT / "configs" / "033_04_skipped_missing_multisite_wth.csv"
FILTERED_SPLIT = OUT / "configs" / "033_04_available_weather_half_split_years.csv"

_ORIGINAL_LOAD_SPLIT = base.load_split


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


def load_split_available_weather() -> pd.DataFrame:
    split = _ORIGINAL_LOAD_SPLIT()
    rows: list[dict] = []
    skipped: list[dict] = []
    for _, row in split.iterrows():
        station = str(row["station_code"])
        year = int(row["year"])
        try:
            source_weather_path(station, year)
            rows.append(row.to_dict())
        except FileNotFoundError as exc:
            out = row.to_dict()
            out["skip_reason"] = "missing_multisite_wth"
            out["missing_path"] = str(exc)
            skipped.append(out)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "configs").mkdir(parents=True, exist_ok=True)
    skipped_df = pd.DataFrame(skipped)
    skipped_df.to_csv(SKIPPED_WTH, index=False, encoding="utf-8-sig")
    filtered = pd.DataFrame(rows)
    filtered = filtered.sort_values(["station_code", "year"]).reset_index(drop=True)
    filtered.to_csv(FILTERED_SPLIT, index=False, encoding="utf-8-sig")
    return filtered


def add_comparison_flags_without_old_baselines(eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df.empty:
        return eval_df
    out = eval_df.copy()
    out["baseline_rows"] = pd.NA
    out["gap_yield_vs_four_max"] = pd.NA
    out["gap_wp_et_vs_four_max"] = pd.NA
    out["gap_pfp_n_vs_four_max"] = pd.NA
    out["any_metric_win_four"] = pd.NA
    out["baseline_comparison_note"] = "033_04不与旧四情景基线比较；旧基线来自旧输入链，需要后续用multisite输入重跑"
    return out


def write_record(
    split: pd.DataFrame,
    train_df: pd.DataFrame,
    reset_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    by_station: pd.DataFrame,
) -> None:
    if SKIPPED_WTH.exists() and SKIPPED_WTH.stat().st_size > 4:
        skipped_df = pd.read_csv(SKIPPED_WTH, keep_default_na=False)
    else:
        skipped_df = pd.DataFrame()
    failed = train_df[~train_df["run_status"].astype(str).str.startswith("ok")].copy() if not train_df.empty else pd.DataFrame()
    split_summary = (
        split.groupby(["station_code", "site", "split"], as_index=False)
        .agg(n_years=("year", "nunique"), years=("year", lambda s: ",".join(map(str, sorted(pd.to_numeric(s).astype(int).tolist())))))
        .sort_values(["station_code", "split"])
    ) if not split.empty else pd.DataFrame()
    skipped_summary = (
        skipped_df.groupby(["station_code", "site", "split"], as_index=False)
        .agg(n_skipped=("year", "nunique"), skipped_years=("year", lambda s: ",".join(map(str, sorted(pd.to_numeric(s).astype(int).tolist())))))
        .sort_values(["station_code", "split"])
    ) if not skipped_df.empty else pd.DataFrame()

    lines = [
        "# 033_04 使用 multisite_new_cultivar_inputs_013 的五站点 half-split MaskablePPO 重跑记录",
        "",
        "## 结论先说",
        "",
        f"- 状态：`{'completed' if failed.empty else 'partial'}`。",
        f"- 算法：MaskablePPO；seed={base.SEED}；每站点训练 {base.TOTAL_TIMESTEPS} timesteps。",
        f"- checkpoint：{', '.join(map(str, base.CHECKPOINT_STEPS))}。",
        "- 输入源：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013`。",
        "- 渲染链条：目标年 WSTA 与目标年 `.WTH` 文件一致；treatment 1 启用 `IC=1, MI=1, MF=1`。",
        "- 本轮不调参、不改 reward、不改动作空间、不做跨站点迁移。",
        "- 本轮不和旧四情景基线做最终比较，因为旧四情景基线使用了旧输入链；需要后续用同一 multisite 输入源重跑基线。",
        "",
        "## 可用天气年份过滤",
        "",
        "- 033_05 已确认：有 multisite WTH 的 81 个站点-年份全部通过 IC/WSTA 渲染审计。",
        "- 缺少 multisite WTH 的年份不会静默训练，也不会回退到旧 `my_data`。",
        "- HLA 当前输入包只有 2007、2009、2010、2011 四个可用天气年份，且均落在原 half-split 的训练段；因此 033_04 对 HLA 可训练，但没有验证年份可评估。",
        "",
        "### 实际纳入年份汇总",
        "",
        md_table(split_summary, max_rows=80),
        "",
        "### 因缺 WTH 跳过的年份",
        "",
        md_table(skipped_summary, max_rows=80),
        "",
        "## 固定年份切分（过滤后）",
        "",
        md_table(split[["station_code", "site", "year", "split"]], max_rows=140),
        "",
        "## 训练 checkpoint 库存",
        "",
        md_table(train_df, max_rows=100),
        "",
        "## 训练年份采样次数",
        "",
        md_table(reset_df, max_rows=140),
        "",
        "## 验证集按站点和 checkpoint 汇总",
        "",
        md_table(by_station, max_rows=100),
        "",
        "## 逐年验证结果",
        "",
        md_table(eval_df, max_rows=240),
        "",
        "## 解释边界",
        "",
        "- 033_04 是在正确 multisite 输入源、IC=1 链条下的 PPO 主实验重跑，不是最终五情景优劣比较。",
        "- 历史 032_22/031_35/031_36 数值保留为旧输入链历史结果，不再作为正式汇报结论。",
        "- 若后续需要导师汇报图，必须基于同一输入源重跑 null、recorded、official expert、DSSAT auto 后再汇总。",
    ]
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    base.OUT = OUT
    base.DOC = DOC
    base.PROMPT = PROMPT
    base.load_split = load_split_available_weather
    base.add_comparison_flags = add_comparison_flags_without_old_baselines
    base.write_record = write_record
    try:
        base.main()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "033_04_error_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    result_path = OUT / "033_04_result.json"
    legacy_result = OUT / "032_22_result.json"
    if legacy_result.exists():
        result = json.loads(legacy_result.read_text(encoding="utf-8"))
        result["task"] = "033_04_multisite_input_enabled_five_site_half_split_maskableppo_rerun"
        result["record_md"] = str(DOC.relative_to(ROOT)).replace("\\", "/")
        result["input_source"] = "DSSAT_auto_validation/multisite_new_cultivar_inputs_013"
        result["filtered_split"] = str(FILTERED_SPLIT.relative_to(ROOT)).replace("\\", "/")
        result["skipped_missing_multisite_wth"] = str(SKIPPED_WTH.relative_to(ROOT)).replace("\\", "/")
        result["baseline_comparison"] = "not_computed_old_baselines_use_old_input_chain"
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
