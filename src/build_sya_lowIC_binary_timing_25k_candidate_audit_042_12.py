from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

import build_sya_lowIC_04028_ckpt75k_sample_five_scenario_daily_plots_040_30 as daily_base
import build_sya_lowIC_04036_ckpt100k_validation_five_scenario_metric_bars_040_38 as metric_base
import run_sya_lowIC_binary_timing_maskableppo_042_10 as ppo04210


ROOT = Path(__file__).resolve().parents[1]
TASK = "042_12_sya_lowIC_binary_timing_25k_candidate_audit"
OUT = ROOT / "benchmark_results" / TASK
FIG_DIR = OUT / "figures"
TAB_DIR = OUT / "tables"
SNAP_DIR = OUT / "snapshots"
DAILY_DIR = OUT / "daily_outputs"
DOC = ROOT / "docs" / f"{TASK}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK}.md"

STATION_CODE = "SYA"
SITE = "SY"
CHECKPOINT = 25_000
YEARS = list(range(2014, 2024))
KEY_YEARS = [2014, 2017, 2021, 2022, 2023]
PPO_EVAL = (
    ROOT
    / "benchmark_results"
    / "042_11_sya_lowIC_binary_timing_training_length_curve"
    / "evaluation"
    / "042_11_checkpoint_validation_summary.csv"
)
DIVERSITY_CURVE = (
    ROOT
    / "benchmark_results"
    / "042_11_sya_lowIC_binary_timing_training_length_curve"
    / "evaluation"
    / "042_11_action_diversity_curve.csv"
)


def ensure_dirs() -> None:
    for path in [OUT, FIG_DIR, TAB_DIR, SNAP_DIR, DAILY_DIR, OUT / "configs", OUT / "logs"]:
        path.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    if PROMPT.exists():
        shutil.copy2(PROMPT, OUT / "configs" / PROMPT.name)


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


def patch_metric_base() -> None:
    metric_base.TASK = TASK
    metric_base.OUT = OUT
    metric_base.FIG_DIR = FIG_DIR
    metric_base.TAB_DIR = TAB_DIR
    metric_base.SNAP_DIR = SNAP_DIR
    metric_base.DAILY_DIR = DAILY_DIR
    metric_base.DOC = DOC
    metric_base.PROMPT = PROMPT
    metric_base.CHECKPOINT = CHECKPOINT
    metric_base.YEARS = YEARS
    metric_base.PPO_EVAL = PPO_EVAL
    metric_base.LOWIC_INPUT_ROOT = ppo04210.LOWIC_INPUT_ROOT
    metric_base.ppo04036 = ppo04210
    metric_base.SCENARIO_LABELS = dict(metric_base.SCENARIO_LABELS)
    metric_base.SCENARIO_LABELS["rl_candidate"] = "Binary-timing PPO 042_11 ckpt25k"


def patch_daily_base() -> None:
    daily_base.TASK = TASK
    daily_base.OUT = OUT
    daily_base.FIG_DIR = FIG_DIR
    daily_base.TAB_DIR = TAB_DIR
    daily_base.SNAP_DIR = SNAP_DIR
    daily_base.DOC = DOC
    daily_base.PROMPT = PROMPT
    daily_base.CHECKPOINT = CHECKPOINT
    daily_base.YEARS = KEY_YEARS
    daily_base.PPO04028_EVAL = PPO_EVAL
    daily_base.PLOT_TAG = "Binary-timing PPO 042_11 ckpt25k"
    daily_base.FILE_TAG = "042_12_lowIC_binary_timing_ckpt25k"
    daily_base.TABLE_TAG = "042_12_sya_2014_2017_2021_2022_2023_lowIC_binary_timing_ckpt25k"
    daily_base.LABELS = dict(daily_base.LABELS)
    daily_base.LABELS["rl_candidate"] = "Binary-timing PPO 042_11 ckpt25k"


def rename_metric_outputs() -> dict[str, str]:
    """The reused 040_38 plotting function writes historical filenames.

    Keep the numerical content, but rename them to the 042_12 namespace so the
    result folder is understandable later.
    """

    replacements = {
        "040_38_sya_lowIC_04036_ckpt100k": "042_12_sya_lowIC_binary_timing_ckpt25k",
        "040_38_sya_lowIC_04036": "042_12_sya_lowIC_binary_timing",
        "040_38": "042_12",
    }
    renamed: dict[str, str] = {}
    for folder in [TAB_DIR, FIG_DIR]:
        for path in list(folder.glob("*")):
            new_name = path.name
            for old, new in replacements.items():
                new_name = new_name.replace(old, new)
            if new_name != path.name:
                target = path.with_name(new_name)
                if target.exists():
                    target.unlink()
                path.replace(target)
                renamed[path.name] = target.name
    return renamed


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, keep_default_na=False)


def load_checkpoint_rows() -> pd.DataFrame:
    eval_df = pd.read_csv(PPO_EVAL, keep_default_na=False)
    return eval_df[
        eval_df["station_code"].astype(str).eq(STATION_CODE)
        & pd.to_numeric(eval_df["checkpoint_step"], errors="coerce").eq(CHECKPOINT)
    ].copy()


def summarize_action_events(eval_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in eval_rows.iterrows():
        daily_path = ROOT / str(row["daily_csv_path"])
        daily = pd.read_csv(daily_path, keep_default_na=False)
        i = pd.to_numeric(daily.get("safe_action_amir", daily.get("irrigation_mm_action")), errors="coerce").fillna(0.0)
        n = pd.to_numeric(daily.get("safe_action_anfer", daily.get("nitrogen_kg_ha_action")), errors="coerce").fillna(0.0)
        dap = pd.to_numeric(daily["dap"], errors="coerce")
        i_events = daily.loc[i.gt(0), ["dap"]].copy()
        n_events = daily.loc[n.gt(0), ["dap"]].copy()
        rows.append(
            {
                "year": int(row["year"]),
                "action_sequence": row.get("action_sequence", ""),
                "total_irrigation_mm": float(i.sum()),
                "total_nitrogen_kg_ha": float(n.sum()),
                "irrigation_event_count": int(i.gt(0).sum()),
                "nitrogen_event_count": int(n.gt(0).sum()),
                "irrigation_event_daps": ";".join(map(str, dap[i.gt(0)].astype(int).tolist())),
                "nitrogen_event_daps": ";".join(map(str, dap[n.gt(0)].astype(int).tolist())),
                "first_irrigation_dap": int(dap[i.gt(0)].min()) if i.gt(0).any() else "",
                "first_nitrogen_dap": int(dap[n.gt(0)].min()) if n.gt(0).any() else "",
            }
        )
    return pd.DataFrame(rows)


def write_clean_record(result: dict) -> None:
    gaps_path = TAB_DIR / "042_12_sya_lowIC_binary_timing_ckpt25k_metric_gaps_vs_four_max.csv"
    summary_path = TAB_DIR / "042_12_sya_lowIC_binary_timing_ckpt25k_five_scenario_metric_summary.csv"
    actions_path = TAB_DIR / "042_12_binary_timing_ckpt25k_action_events.csv"
    diversity_path = TAB_DIR / "042_12_binary_timing_ckpt25k_diversity_row.csv"

    gaps = read_csv_if_exists(gaps_path)
    summary = read_csv_if_exists(summary_path)
    actions = read_csv_if_exists(actions_path)
    diversity = read_csv_if_exists(diversity_path)

    lines = [
        "# 042_12 SYA lowIC binary-timing PPO 25K 候选策略审计记录",
        "",
        "## 结论先说",
        "",
        "- 本任务没有训练、没有调参、没有改 checkpoint，只审计 042_11 的 25K checkpoint。",
        "- 25K 的含义是：训练过程中第 25,000 个环境交互步保存的冻结模型，不是 25 年、25 个 episode，也不是最终收敛点。",
        "- 指标柱状图使用 042_10 binary-timing PPO 配置重放 DSSAT，避免误用 040_36/040_28 的旧动作空间。",
        "- 日过程图使用 042_11 25K checkpoint 已保存的 validation daily CSV，与四基线放在同一图中展示。",
        "",
        "## 胜出计数",
        "",
        f"- 验证年份数：{result.get('years', '')}",
        f"- 至少一项指标超过四基线最高值：{result.get('any_metric_win', '')}",
        f"- 产量超过四基线最高值：{result.get('yield_win', '')}",
        f"- WP_ET 超过四基线最高值：{result.get('wp_et_win', '')}",
        f"- PFP_N 超过四基线最高值：{result.get('pfp_n_win', '')}",
        "",
        "## 25K 动作多样性",
        "",
        md_table(diversity, max_rows=10),
        "",
        "## 25K 每年管理事件",
        "",
        md_table(actions, max_rows=30),
        "",
        "## 指标差值表",
        "",
        md_table(gaps, max_rows=30),
        "",
        "## 五情景终值摘要",
        "",
        md_table(summary, max_rows=80),
        "",
        "## 输出文件",
        "",
        *[f"- `{p}`" for p in result.get("outputs", [])],
        "",
        "## 结论边界",
        "",
        "- 042_12 只判断 25K checkpoint 是否值得作为候选策略继续展示/诊断。",
        "- 如果 25K 指标较好但动作仍模板化，下一步仍应做天气响应性和反事实必要性审计，而不是直接宣称自由时序 PPO 已经完全解决决策合理性问题。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ensure_dirs()
    if not PPO_EVAL.exists():
        raise FileNotFoundError(PPO_EVAL)

    patch_metric_base()
    metric_base.main()
    rename_metric_outputs()

    patch_daily_base()
    daily_base.main()

    eval_rows = load_checkpoint_rows()
    actions = summarize_action_events(eval_rows)
    actions_path = TAB_DIR / "042_12_binary_timing_ckpt25k_action_events.csv"
    actions.to_csv(actions_path, index=False, encoding="utf-8-sig")

    diversity = read_csv_if_exists(DIVERSITY_CURVE)
    diversity_row = diversity[pd.to_numeric(diversity.get("checkpoint_step"), errors="coerce").eq(CHECKPOINT)].copy()
    diversity_path = TAB_DIR / "042_12_binary_timing_ckpt25k_diversity_row.csv"
    diversity_row.to_csv(diversity_path, index=False, encoding="utf-8-sig")

    gaps_path = TAB_DIR / "042_12_sya_lowIC_binary_timing_ckpt25k_metric_gaps_vs_four_max.csv"
    gaps = read_csv_if_exists(gaps_path)
    # 040_38-style gap table is already PPO-only, one row per validation year.
    # Older summary helpers sometimes used a scenario column; support both to
    # avoid silently reporting zero wins when the table is already filtered.
    if "scenario" in gaps.columns:
        rl_gaps = gaps[gaps["scenario"].eq("rl_candidate")].copy()
    else:
        rl_gaps = gaps.copy()
    any_metric = 0
    yield_win = 0
    wp_win = 0
    pfp_win = 0
    if not rl_gaps.empty:
        yield_win = int(pd.to_numeric(rl_gaps["gap_yield_vs_four_max"], errors="coerce").ge(0).sum())
        wp_win = int(pd.to_numeric(rl_gaps["gap_wp_et_vs_four_max"], errors="coerce").ge(0).sum())
        pfp_win = int(pd.to_numeric(rl_gaps["gap_pfp_n_vs_four_max"], errors="coerce").ge(0).sum())
        any_metric = int(
            (
                pd.to_numeric(rl_gaps["gap_yield_vs_four_max"], errors="coerce").ge(0)
                | pd.to_numeric(rl_gaps["gap_wp_et_vs_four_max"], errors="coerce").ge(0)
                | pd.to_numeric(rl_gaps["gap_pfp_n_vs_four_max"], errors="coerce").ge(0)
            ).sum()
        )

    outputs = sorted(
        [
            p.relative_to(ROOT).as_posix()
            for folder in [TAB_DIR, FIG_DIR]
            for p in folder.glob("*")
            if p.is_file()
        ]
    )
    result = {
        "task": TASK,
        "checkpoint": CHECKPOINT,
        "years": len(YEARS),
        "key_years": KEY_YEARS,
        "any_metric_win": any_metric,
        "yield_win": yield_win,
        "wp_et_win": wp_win,
        "pfp_n_win": pfp_win,
        "unique_action_signatures": int(diversity_row["unique_action_signatures"].iloc[0]) if not diversity_row.empty else None,
        "record_md": DOC.relative_to(ROOT).as_posix(),
        "outputs": outputs,
    }
    write_clean_record(result)
    result_path = OUT / "042_12_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
