from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

import build_sya_lowIC_04028_ckpt75k_sample_five_scenario_daily_plots_040_30 as daily_base


ROOT = Path(__file__).resolve().parents[1]
TASK = "045_00_sya_lowIC_04215_frozen_candidate_all_validation_daily_plots"
OUT = ROOT / "benchmark_results" / TASK
FIG_DIR = OUT / "figures"
TAB_DIR = OUT / "tables"
SNAP_DIR = OUT / "snapshots"
DOC = ROOT / "docs" / f"{TASK}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK}.md"

YEARS = list(range(2014, 2024))
CHECKPOINT = 25_000
PPO_EVAL = (
    ROOT
    / "benchmark_results"
    / "042_11_sya_lowIC_binary_timing_training_length_curve"
    / "evaluation"
    / "042_11_checkpoint_validation_summary.csv"
)
SOURCE_METRIC_SUMMARY = (
    ROOT
    / "benchmark_results"
    / "042_12_sya_lowIC_binary_timing_25k_candidate_audit"
    / "tables"
    / "042_12_sya_lowIC_binary_timing_ckpt25k_five_scenario_metric_summary.csv"
)
SOURCE_METRIC_GAPS = (
    ROOT
    / "benchmark_results"
    / "042_12_sya_lowIC_binary_timing_25k_candidate_audit"
    / "tables"
    / "042_12_sya_lowIC_binary_timing_ckpt25k_metric_gaps_vs_four_max.csv"
)


def configure_reused_daily_plotter() -> None:
    """Reuse the existing five-scenario daily plotting implementation.

    The reused module already contains the vetted eight-panel plotting style and
    the lowIC four-baseline loading path.  Here we only redirect output paths,
    checkpoint metadata, labels, and year list to the frozen 042_15 candidate.
    """

    daily_base.TASK = TASK
    daily_base.OUT = OUT
    daily_base.FIG_DIR = FIG_DIR
    daily_base.TAB_DIR = TAB_DIR
    daily_base.SNAP_DIR = SNAP_DIR
    daily_base.DOC = DOC
    daily_base.PROMPT = PROMPT
    daily_base.CHECKPOINT = CHECKPOINT
    daily_base.YEARS = YEARS
    daily_base.PLOT_TAG = "042_15 frozen binary-timing PPO ckpt25k"
    daily_base.FILE_TAG = "045_00_lowIC_04215_binary_timing_ckpt25k"
    daily_base.TABLE_TAG = "045_00_sya_2014_2023_lowIC_04215_binary_timing_ckpt25k"
    daily_base.PPO04028_EVAL = PPO_EVAL
    daily_base.LABELS = dict(daily_base.LABELS)
    daily_base.LABELS["rl_candidate"] = "042_15 frozen PPO ckpt25k"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, keep_default_na=False)
    except EmptyDataError:
        return pd.DataFrame()


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
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


def write_clean_record(result: dict) -> None:
    summary_path = TAB_DIR / "045_00_sya_2014_2023_lowIC_04215_binary_timing_ckpt25k_five_scenario_summary.csv"
    actions_path = TAB_DIR / "045_00_sya_2014_2023_lowIC_04215_binary_timing_ckpt25k_ppo_management_events.csv"
    failures_path = TAB_DIR / "040_30_failures.csv"
    daily_path = TAB_DIR / "045_00_sya_2014_2023_lowIC_04215_binary_timing_ckpt25k_five_scenario_daily.csv"
    metric_summary_path = TAB_DIR / "045_00_sya_lowIC_04215_binary_timing_ckpt25k_five_scenario_metric_summary.csv"
    metric_gaps_path = TAB_DIR / "045_00_sya_lowIC_04215_binary_timing_ckpt25k_metric_gaps_vs_four_max.csv"

    summary = read_csv(summary_path)
    actions = read_csv(actions_path)
    failures = read_csv(failures_path)
    metric_gaps = read_csv(metric_gaps_path)

    ppo_summary = summary[summary.get("scenario", pd.Series(dtype=str)).astype(str).eq("rl_candidate")].copy()

    lines = [
        "# 045_00 SYA lowIC 042_15 冻结候选全年验证五情景日过程图记录",
        "",
        "## 结论先说",
        "",
        "- 本任务没有训练、没有改 reward、没有改 DSSAT 输入、没有重新选择 checkpoint。",
        "- 本任务只把当前阶段性冻结候选 `042_11 binary-timing MaskablePPO ckpt25k` 在 2014--2023 全部验证年份上画成五情景日过程图。",
        "- 图形样式复用既有八面板五情景图：天气、土壤水、WSPD、NSTD、灌溉、施肥、grain/biomass、统一累计 reward。",
        "- 累计 reward 使用绘图统一口径，只用于情景间可视化对照，不代表重新训练信号。",
        "",
        "## 运行对象",
        "",
        f"- 输入根目录：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`",
        f"- PPO 评估表：`{PPO_EVAL.relative_to(ROOT).as_posix()}`",
        f"- checkpoint：`{CHECKPOINT}`",
        f"- 验证年份：`{', '.join(map(str, YEARS))}`",
        "",
        "## PPO 终值摘要",
        "",
        md_table(
            ppo_summary[
                [
                    c
                    for c in [
                        "year",
                        "final_grain_kg_ha",
                        "final_biomass_kg_ha",
                        "rain_total_mm",
                        "irrigation_event_total_mm",
                        "nitrogen_event_total_kg_ha",
                        "max_water_stress_wspd",
                        "max_nitrogen_stress_nstd",
                        "WP_ET_kg_m3",
                        "PFP_N_kg_kg",
                    ]
                    if c in ppo_summary.columns
                ]
            ],
            max_rows=30,
        ),
        "",
        "## PPO 管理事件",
        "",
        md_table(actions, max_rows=160),
        "",
        "## PPO 相对四基线最高值的指标差值",
        "",
        md_table(metric_gaps, max_rows=30),
        "",
        "## 失败记录",
        "",
        md_table(failures, max_rows=40),
        "",
        "## 输出文件",
        "",
        f"- 全年日值表：`{daily_path.relative_to(ROOT).as_posix()}`",
        f"- 五情景总结表：`{summary_path.relative_to(ROOT).as_posix()}`",
        f"- PPO 管理事件表：`{actions_path.relative_to(ROOT).as_posix()}`",
        f"- 五情景指标总览表：`{metric_summary_path.relative_to(ROOT).as_posix()}`",
        f"- PPO vs 四基线最高值差值表：`{metric_gaps_path.relative_to(ROOT).as_posix()}`",
        f"- 失败记录：`{failures_path.relative_to(ROOT).as_posix()}`",
        "",
        "## 图件",
        "",
        *[f"- `{p}`" for p in result.get("figures_png", [])],
        "",
        "## 汇报边界",
        "",
        "这批图可以用于汇报：SY 站点已经得到一个阶段性可用、指标表现良好、措施基本可解释的自由时序 PPO 候选策略。",
        "但仍需说明：它不是最终完美策略，仍有模板成分；后续任务应转向其他站点复用流程，或专门增强天气响应性。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if not PPO_EVAL.exists():
        raise FileNotFoundError(PPO_EVAL)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "configs").mkdir(parents=True, exist_ok=True)
    if PROMPT.exists():
        try:
            shutil.copy2(PROMPT, OUT / "configs" / PROMPT.name)
        except OSError:
            shutil.copyfile(PROMPT, OUT / "configs" / PROMPT.name)

    configure_reused_daily_plotter()
    daily_base.main()

    metric_summary_path = TAB_DIR / "045_00_sya_lowIC_04215_binary_timing_ckpt25k_five_scenario_metric_summary.csv"
    metric_gaps_path = TAB_DIR / "045_00_sya_lowIC_04215_binary_timing_ckpt25k_metric_gaps_vs_four_max.csv"
    if SOURCE_METRIC_SUMMARY.exists():
        shutil.copy2(SOURCE_METRIC_SUMMARY, metric_summary_path)
    if SOURCE_METRIC_GAPS.exists():
        shutil.copy2(SOURCE_METRIC_GAPS, metric_gaps_path)

    figures = sorted(FIG_DIR.glob("*.png"))
    result = {
        "task": TASK,
        "source_candidate": "042_15 frozen binary-timing PPO; 042_11 checkpoint 25k",
        "checkpoint": CHECKPOINT,
        "years": YEARS,
        "figures_png": [p.relative_to(ROOT).as_posix() for p in figures],
        "daily_table": (TAB_DIR / "045_00_sya_2014_2023_lowIC_04215_binary_timing_ckpt25k_five_scenario_daily.csv").relative_to(ROOT).as_posix(),
        "summary_table": (TAB_DIR / "045_00_sya_2014_2023_lowIC_04215_binary_timing_ckpt25k_five_scenario_summary.csv").relative_to(ROOT).as_posix(),
        "actions_table": (TAB_DIR / "045_00_sya_2014_2023_lowIC_04215_binary_timing_ckpt25k_ppo_management_events.csv").relative_to(ROOT).as_posix(),
        "metric_summary": metric_summary_path.relative_to(ROOT).as_posix(),
        "metric_gaps_vs_four_max": metric_gaps_path.relative_to(ROOT).as_posix(),
        "record_md": DOC.relative_to(ROOT).as_posix(),
        "failures_csv": (TAB_DIR / "040_30_failures.csv").relative_to(ROOT).as_posix(),
    }
    (OUT / f"{TASK}_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_clean_record(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
