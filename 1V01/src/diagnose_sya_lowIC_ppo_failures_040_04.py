from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "benchmark_results" / "040_04_sya_lowIC_ppo_failure_diagnosis"
TABLE_DIR = OUT_ROOT / "tables"
DOC_PATH = PROJECT_ROOT / "docs" / "040_04_sya_lowIC_ppo_failure_diagnosis_record.md"

PPO_DETAIL = (
    PROJECT_ROOT
    / "benchmark_results"
    / "040_00_sya_lowIC_free_timing_maskableppo"
    / "evaluation"
    / "040_00_checkpoint_validation_summary.csv"
)
BASELINE_SUMMARY = (
    PROJECT_ROOT
    / "benchmark_results"
    / "039_02_original_vs_lowIC_three_baseline_audit"
    / "tables"
    / "039_02_full_summary.csv"
)

PPO_CHECKPOINT = 75000
VALIDATION_YEARS = list(range(2014, 2024))


def _scenario_label(value) -> str:
    if pd.isna(value) or value == "":
        return "null"
    return str(value)


def _fmt(x, nd=1) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):.{nd}f}"


def _markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                vals.append(_fmt(value, 2))
            elif pd.isna(value):
                vals.append("")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)

    ppo = pd.read_csv(PPO_DETAIL)
    base = pd.read_csv(BASELINE_SUMMARY)

    ppo = ppo[
        (ppo["station_code"] == "SYA")
        & (ppo["checkpoint_step"] == PPO_CHECKPOINT)
        & (ppo["year"].isin(VALIDATION_YEARS))
    ].copy()
    if len(ppo) != len(VALIDATION_YEARS):
        raise ValueError(f"PPO验证年数量异常: {len(ppo)}")

    base = base[
        (base["station_code"] == "SYA")
        & (base["condition"] == "lowIC")
        & (base["year"].isin(VALIDATION_YEARS))
        & (base["run_status"] == "ok")
    ].copy()
    base["scenario_label"] = base["scenario"].map(_scenario_label)

    rows = []
    for _, prow in ppo.sort_values("year").iterrows():
        year = int(prow["year"])
        b = base[base["year"] == year].copy()
        if b.empty:
            raise ValueError(f"缺少SYA {year} lowIC基线")
        by_scenario = {r["scenario_label"]: r for _, r in b.iterrows()}
        expert = by_scenario.get("official_extension_expert")

        baseline_yield_max = b["grain_yield_kg_ha"].max()
        baseline_wp_max = b["WP_ET_kg_m3"].max()
        baseline_pfp_max = b["PFP_N_kg_kg"].dropna().max()

        ppo_yield = float(prow["final_grnwt"])
        ppo_irrig = float(prow["total_irrigation"])
        ppo_n = float(prow["total_n"])
        ppo_pfp = float(prow["PFP_N"]) if not pd.isna(prow["PFP_N"]) else pd.NA

        row = {
            "station_code": "SYA",
            "site": "SY",
            "year": year,
            "ppo_checkpoint_step": PPO_CHECKPOINT,
            "ppo_yield": ppo_yield,
            "ppo_irrigation": ppo_irrig,
            "ppo_nitrogen": ppo_n,
            "ppo_PFP_N": ppo_pfp,
            "ppo_swfac_days_gt_0p05": prow["swfac_stress_days_gt_0p05"],
            "ppo_nstres_days_gt_0p05": prow["nstres_days_gt_0p05"],
            "ppo_max_swfac": prow["max_swfac"],
            "ppo_max_nstres": prow["max_nstres"],
            "baseline_max_yield": baseline_yield_max,
            "baseline_max_WP_ET": baseline_wp_max,
            "baseline_max_PFP_N": baseline_pfp_max,
            "ppo_gap_yield_vs_baseline_max": ppo_yield - baseline_yield_max,
            "ppo_gap_PFP_N_vs_baseline_max": (
                ppo_pfp - baseline_pfp_max if not pd.isna(ppo_pfp) and not pd.isna(baseline_pfp_max) else pd.NA
            ),
            "ppo_yield_beats_all_three_baselines": ppo_yield > baseline_yield_max,
            "ppo_PFP_N_beats_all_defined_baselines": (
                ppo_pfp > baseline_pfp_max if not pd.isna(ppo_pfp) and not pd.isna(baseline_pfp_max) else False
            ),
        }

        if expert is not None:
            row.update(
                {
                    "expert_yield": expert["grain_yield_kg_ha"],
                    "expert_irrigation": expert["actual_irrigation_mm"],
                    "expert_nitrogen": expert["actual_nitrogen_kg_ha"],
                    "expert_WP_ET": expert["WP_ET_kg_m3"],
                    "expert_PFP_N": expert["PFP_N_kg_kg"],
                    "ppo_gap_yield_vs_expert": ppo_yield - expert["grain_yield_kg_ha"],
                    "ppo_water_saved_vs_expert": expert["actual_irrigation_mm"] - ppo_irrig,
                    "ppo_n_saved_vs_expert": expert["actual_nitrogen_kg_ha"] - ppo_n,
                    "ppo_gap_PFP_N_vs_expert": (
                        ppo_pfp - expert["PFP_N_kg_kg"] if not pd.isna(ppo_pfp) else pd.NA
                    ),
                }
            )
        rows.append(row)

    out = pd.DataFrame(rows)

    def classify(row) -> str:
        if row["ppo_yield_beats_all_three_baselines"]:
            return "yield_win"
        if row["ppo_gap_yield_vs_expert"] >= -200 and row["ppo_water_saved_vs_expert"] > 0 and row["ppo_n_saved_vs_expert"] >= 0:
            return "near_yield_resource_saving"
        if row["ppo_swfac_days_gt_0p05"] >= 20:
            return "water_stress_failure"
        if row["ppo_nstres_days_gt_0p05"] >= 20:
            return "nitrogen_stress_failure"
        return "yield_gap_other"

    out["diagnosis_class"] = out.apply(classify, axis=1)

    class_summary = (
        out.groupby("diagnosis_class", as_index=False)
        .agg(
            year_count=("year", "count"),
            mean_yield_gap_vs_expert=("ppo_gap_yield_vs_expert", "mean"),
            mean_water_saved_vs_expert=("ppo_water_saved_vs_expert", "mean"),
            mean_n_saved_vs_expert=("ppo_n_saved_vs_expert", "mean"),
            mean_swfac_days=("ppo_swfac_days_gt_0p05", "mean"),
        )
        .sort_values("year_count", ascending=False)
    )

    out_csv = TABLE_DIR / "040_04_ppo_vs_lowIC_baselines_by_year.csv"
    class_csv = TABLE_DIR / "040_04_ppo_failure_type_summary.csv"
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    class_summary.to_csv(class_csv, index=False, encoding="utf-8-sig")

    show_cols = [
        "year",
        "ppo_yield",
        "expert_yield",
        "ppo_gap_yield_vs_expert",
        "ppo_water_saved_vs_expert",
        "ppo_n_saved_vs_expert",
        "ppo_swfac_days_gt_0p05",
        "ppo_nstres_days_gt_0p05",
        "diagnosis_class",
    ]

    doc = f"""# 040_04 SYA lowIC PPO 失败年份诊断记录

## 结论先说

本任务回到 PPO 主线，固定使用 040_03 选出的 PPO 最佳 checkpoint：**{PPO_CHECKPOINT}**。  
诊断对象是 SYA lowIC 验证年份 2014–2023。

这一步没有重新训练，也没有重新运行 DSSAT；只读取已有 PPO 评估结果和 039_02 lowIC 三基线结果。

## 逐年诊断表

{_markdown_table(out[show_cols], show_cols)}

## 类型汇总

{_markdown_table(class_summary, list(class_summary.columns))}

## 解释

- `water_stress_failure`：PPO 相比 expert 产量明显偏低，同时水分胁迫天数较多，说明下一步 PPO 优化应优先处理水分胁迫压不住的问题。
- `near_yield_resource_saving`：产量接近 expert，同时节水或节氮，属于有希望通过轻量 guardrail 或 checkpoint 选择改善的年份。
- `yield_win`：PPO 产量超过三基线最高值。

## 输出文件

- 逐年诊断表：`{out_csv.relative_to(PROJECT_ROOT)}`
- 类型汇总表：`{class_csv.relative_to(PROJECT_ROOT)}`

"""
    DOC_PATH.write_text(doc, encoding="utf-8")

    result = {
        "task": "040_04_sya_lowIC_ppo_failure_diagnosis",
        "ppo_checkpoint_step": PPO_CHECKPOINT,
        "validation_years": VALIDATION_YEARS,
        "outputs": {
            "by_year": str(out_csv.relative_to(PROJECT_ROOT)),
            "class_summary": str(class_csv.relative_to(PROJECT_ROOT)),
            "record_md": str(DOC_PATH.relative_to(PROJECT_ROOT)),
        },
        "class_counts": out["diagnosis_class"].value_counts().to_dict(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
