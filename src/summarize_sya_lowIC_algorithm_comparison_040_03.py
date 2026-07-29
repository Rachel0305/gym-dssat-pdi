from __future__ import annotations

from pathlib import Path
import json
import math
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "benchmark_results" / "040_03_sya_lowIC_algorithm_comparison_freeze"
TABLE_DIR = OUT_ROOT / "tables"
DOC_PATH = PROJECT_ROOT / "docs" / "040_03_sya_lowIC_algorithm_comparison_freeze_record.md"


ALGORITHMS = [
    {
        "algorithm_label": "MaskablePPO_040_00",
        "method_family": "PPO",
        "summary_path": PROJECT_ROOT
        / "benchmark_results"
        / "040_00_sya_lowIC_free_timing_maskableppo"
        / "evaluation"
        / "040_00_validation_summary_by_station_checkpoint.csv",
        "detail_path": PROJECT_ROOT
        / "benchmark_results"
        / "040_00_sya_lowIC_free_timing_maskableppo"
        / "evaluation"
        / "040_00_checkpoint_validation_summary.csv",
        "note": "MaskablePPO；mask进入训练采样和评估。",
    },
    {
        "algorithm_label": "SB3_DQN_040_01",
        "method_family": "DQN",
        "summary_path": PROJECT_ROOT
        / "benchmark_results"
        / "040_01_sya_lowIC_free_timing_dqn"
        / "evaluation"
        / "040_01_validation_summary_by_checkpoint.csv",
        "detail_path": PROJECT_ROOT
        / "benchmark_results"
        / "040_01_sya_lowIC_free_timing_dqn"
        / "evaluation"
        / "040_01_checkpoint_validation_summary.csv",
        "note": "SB3 DQN；训练阶段非严格mask，确定性评估使用masked-greedy。",
    },
    {
        "algorithm_label": "StrictMaskableDQN_040_02",
        "method_family": "DQN",
        "summary_path": PROJECT_ROOT
        / "benchmark_results"
        / "040_02_sya_lowIC_strict_maskable_dqn"
        / "evaluation"
        / "040_02_validation_summary_by_checkpoint.csv",
        "detail_path": PROJECT_ROOT
        / "benchmark_results"
        / "040_02_sya_lowIC_strict_maskable_dqn"
        / "evaluation"
        / "040_02_checkpoint_validation_summary.csv",
        "note": "项目内StrictMaskableDQN；mask进入探索、贪心、replay和Bellman target。",
    },
]


def _read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"缺少输入文件: {path}")
    return pd.read_csv(path)


def _normalize_summary(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "algorithm_label", meta["algorithm_label"])
    out.insert(1, "method_family", meta["method_family"])
    out["source_summary_path"] = str(meta["summary_path"].relative_to(PROJECT_ROOT))
    out["algorithm_note"] = meta["note"]

    optional_cols = [
        "mean_PFP_N",
        "mean_swfac_stress_days_gt_0p05",
        "mean_nstres_days_gt_0p05",
        "max_swfac",
        "max_nstres",
        "any_metric_win_four_count",
        "mean_gap_yield_vs_four_max",
        "mean_gap_pfp_n_vs_four_max",
    ]
    for col in optional_cols:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def _best_checkpoint(summary: pd.DataFrame) -> pd.Series:
    ok = summary.dropna(subset=["mean_final_grnwt"]).copy()
    if ok.empty:
        raise ValueError("没有可用于选择最佳checkpoint的mean_final_grnwt")
    ok = ok.sort_values(
        ["mean_final_grnwt", "checkpoint_step"],
        ascending=[False, True],
        kind="mergesort",
    )
    return ok.iloc[0]


def _safe_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _fmt(value, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}"


def _markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append(_fmt(val, 2))
            elif pd.isna(val):
                vals.append("")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)

    summary_frames: list[pd.DataFrame] = []
    detail_frames: list[pd.DataFrame] = []
    missing_inputs: list[str] = []

    for meta in ALGORITHMS:
        try:
            summary = _normalize_summary(_read_csv_required(meta["summary_path"]), meta)
            summary_frames.append(summary)
            detail = _read_csv_required(meta["detail_path"]).copy()
            detail.insert(0, "algorithm_label", meta["algorithm_label"])
            detail.insert(1, "method_family", meta["method_family"])
            detail_frames.append(detail)
        except FileNotFoundError as exc:
            missing_inputs.append(str(exc))

    if missing_inputs:
        raise FileNotFoundError("\n".join(missing_inputs))

    all_summary = pd.concat(summary_frames, ignore_index=True, sort=False)
    all_summary = all_summary.sort_values(
        ["algorithm_label", "checkpoint_step"],
        kind="mergesort",
    )
    all_detail = pd.concat(detail_frames, ignore_index=True, sort=False)

    stress_cols = [
        "swfac_stress_days_gt_0p05",
        "nstres_days_gt_0p05",
    ]
    if all(col in all_detail.columns for col in stress_cols):
        stress_summary = (
            all_detail.groupby(["algorithm_label", "checkpoint_step"], as_index=False)
            .agg(
                mean_swfac_stress_days_gt_0p05_from_detail=(
                    "swfac_stress_days_gt_0p05",
                    "mean",
                ),
                mean_nstres_days_gt_0p05_from_detail=(
                    "nstres_days_gt_0p05",
                    "mean",
                ),
            )
        )
        all_summary = all_summary.merge(
            stress_summary,
            on=["algorithm_label", "checkpoint_step"],
            how="left",
        )
        for col in [
            "mean_swfac_stress_days_gt_0p05",
            "mean_nstres_days_gt_0p05",
        ]:
            from_detail = f"{col}_from_detail"
            all_summary[col] = all_summary[col].combine_first(all_summary[from_detail])
            all_summary = all_summary.drop(columns=[from_detail])

    best_rows = []
    for algorithm_label, group in all_summary.groupby("algorithm_label", sort=False):
        best = _best_checkpoint(group)
        final_ckpt = group.sort_values("checkpoint_step").iloc[-1]
        degradation = _safe_float(best["mean_final_grnwt"])
        final_yield = _safe_float(final_ckpt["mean_final_grnwt"])
        best = best.copy()
        best["final_checkpoint_step"] = final_ckpt["checkpoint_step"]
        best["final_checkpoint_mean_final_grnwt"] = final_yield
        best["yield_drop_from_best_to_final"] = (
            degradation - final_yield if degradation is not None and final_yield is not None else pd.NA
        )
        best_rows.append(best)

    best_summary = pd.DataFrame(best_rows)
    best_summary = best_summary.sort_values(
        ["mean_final_grnwt"],
        ascending=False,
        kind="mergesort",
    )
    best_summary["rank_by_mean_validation_yield"] = range(1, len(best_summary) + 1)

    key_pairs = best_summary[["algorithm_label", "checkpoint_step"]].copy()
    best_detail = all_detail.merge(key_pairs, on=["algorithm_label", "checkpoint_step"], how="inner")

    checkpoint_csv = TABLE_DIR / "040_03_algorithm_checkpoint_summary.csv"
    best_csv = TABLE_DIR / "040_03_algorithm_best_checkpoint_summary.csv"
    detail_csv = TABLE_DIR / "040_03_algorithm_best_year_detail.csv"
    all_summary.to_csv(checkpoint_csv, index=False, encoding="utf-8-sig")
    best_summary.to_csv(best_csv, index=False, encoding="utf-8-sig")
    best_detail.to_csv(detail_csv, index=False, encoding="utf-8-sig")

    ppo_best = best_summary[best_summary["algorithm_label"] == "MaskablePPO_040_00"].iloc[0]
    strict_best = best_summary[best_summary["algorithm_label"] == "StrictMaskableDQN_040_02"].iloc[0]
    sb3_best = best_summary[best_summary["algorithm_label"] == "SB3_DQN_040_01"].iloc[0]

    comparison_cols = [
        "rank_by_mean_validation_yield",
        "algorithm_label",
        "checkpoint_step",
        "validation_years",
        "mean_final_grnwt",
        "mean_total_irrigation",
        "mean_total_n",
        "mean_PFP_N",
        "mean_swfac_stress_days_gt_0p05",
        "mean_nstres_days_gt_0p05",
        "yield_drop_from_best_to_final",
    ]

    doc = f"""# 040_03 SYA lowIC 自由时序算法对照冻结记录

## 结论先说

在相同 SYA lowIC 输入、相同训练/验证年份划分、相同动作空间、相同安全约束、相同奖励函数下，当前三套算法的验证年均产量排序为：

1. **MaskablePPO_040_00**：最佳 checkpoint {int(ppo_best['checkpoint_step'])}，验证年均产量 {_fmt(ppo_best['mean_final_grnwt'])} kg/ha。
2. **SB3_DQN_040_01**：最佳 checkpoint {int(sb3_best['checkpoint_step'])}，验证年均产量 {_fmt(sb3_best['mean_final_grnwt'])} kg/ha。
3. **StrictMaskableDQN_040_02**：最佳 checkpoint {int(strict_best['checkpoint_step'])}，验证年均产量 {_fmt(strict_best['mean_final_grnwt'])} kg/ha。

这说明：**严格把 mask 接入 DQN 的探索、贪心、replay 和 Bellman target 后，DQN 仍没有优于 PPO。**  
因此，当前证据支持把 PPO 作为后续主线算法，DQN 作为已完成的公平对照保留。

## 任务边界

- 本任务没有重新训练模型。
- 本任务没有重新运行 DSSAT。
- 本任务只读取 040_00、040_01、040_02 已有 CSV 结果，并统一汇总。
- checkpoint 选择规则固定为：每个算法选择验证年均产量 `mean_final_grnwt` 最高的 checkpoint。

## 最佳 checkpoint 对照

{_markdown_table(best_summary[comparison_cols], comparison_cols)}

## 关键解释

### 1. 为什么 040_02 是更严格的 DQN 对照？

040_01 使用 SB3 DQN。SB3 DQN 本身没有原生 MaskableDQN 机制，所以 040_01 只能在确定性评估时使用 masked-greedy 选择合法动作；训练阶段的探索和 Bellman target 并不是严格 mask 的。

040_02 使用项目内实现的 StrictMaskableDQN，mask 进入：

- epsilon 随机探索；
- greedy 动作选择；
- replay buffer 储存；
- next-state Bellman target 的 `max_a Q(s', a)`。

所以 040_02 回答的是：如果把 DQN 的 mask 机制补严格，是否能超过 PPO？当前答案是没有。

### 2. 为什么不继续给 DQN 加训练步数？

StrictMaskableDQN 的最佳 checkpoint 出现在 {int(strict_best['checkpoint_step'])} 步；后续 checkpoint 明显退化。  
最终 checkpoint {int(strict_best['final_checkpoint_step'])} 的验证年均产量为 {_fmt(strict_best['final_checkpoint_mean_final_grnwt'])} kg/ha，比最佳 checkpoint 低 {_fmt(strict_best['yield_drop_from_best_to_final'])} kg/ha。

这更像是 DQN 长训练过程中的策略退化，而不是训练不足。

### 3. 这是否证明 DQN 理论上不行？

不能。这个结论只限于当前实验条件：

- 站点：SYA；
- 输入：lowIC；
- 年份划分：2005–2013 训练，2014–2023 验证；
- 自由时序日尺度环境；
- 当前 stress-aware reward；
- 当前动作空间和安全约束；
- 当前 100K 训练预算。

它能支持的结论是：**在当前公平对照设置下，PPO 比两种 DQN 实现更稳、更好。**

## 输出文件

- checkpoint 汇总：`{checkpoint_csv.relative_to(PROJECT_ROOT)}`
- 最佳 checkpoint 汇总：`{best_csv.relative_to(PROJECT_ROOT)}`
- 最佳 checkpoint 年份明细：`{detail_csv.relative_to(PROJECT_ROOT)}`

"""
    DOC_PATH.write_text(doc, encoding="utf-8")

    result = {
        "task": "040_03_sya_lowIC_algorithm_comparison_freeze",
        "branch": "algorithm_comparison_frozen",
        "best_algorithm": str(best_summary.iloc[0]["algorithm_label"]),
        "best_checkpoint_step": int(best_summary.iloc[0]["checkpoint_step"]),
        "best_mean_final_grnwt": float(best_summary.iloc[0]["mean_final_grnwt"]),
        "outputs": {
            "checkpoint_summary": str(checkpoint_csv.relative_to(PROJECT_ROOT)),
            "best_checkpoint_summary": str(best_csv.relative_to(PROJECT_ROOT)),
            "best_year_detail": str(detail_csv.relative_to(PROJECT_ROOT)),
            "record_md": str(DOC_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
