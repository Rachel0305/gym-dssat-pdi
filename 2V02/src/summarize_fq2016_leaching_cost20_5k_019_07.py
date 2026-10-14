from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "DSSAT_auto_validation" / "fq2016_leaching_aware_reward_smoke_019_05"
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "fq2016_leaching_cost20_5k_019_07"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-10_019_07_fq2016_leaching_cost20_5k_record.md"


RUN_DIRS = {
    "lc0_5k": ROOT / "seed0_5000steps_lc0",
    "lc20_5k": ROOT / "seed0_5000steps_lc20",
}


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_无结果。_"
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.3f}")
        else:
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else str(v))
    lines = [
        "| " + " | ".join(out.columns) + " |",
        "| " + " | ".join(["---"] * len(out.columns)) + " |",
    ]
    for row in out.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def collect() -> pd.DataFrame:
    rows = []
    for label, run_dir in RUN_DIRS.items():
        summary_path = run_dir / "leaching_aware_checkpoint_summary.csv"
        if not summary_path.exists():
            continue
        df = pd.read_csv(summary_path)
        if df.empty:
            continue
        df = df.copy()
        df.insert(0, "run_label", label)
        df["run_dir"] = str(run_dir.relative_to(PROJECT_ROOT))
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def write_doc(summary: pd.DataFrame) -> None:
    show_cols = [
        "run_label",
        "checkpoint_step",
        "leaching_cost",
        "final_grain_kg_ha",
        "final_biomass_kg_ha",
        "action_irrigation_total",
        "action_fertilizer_total",
        "final_cleach",
        "soilni_final_NLCC",
        "sum_leaching_cost_term",
        "max_water_stress",
        "max_nitrogen_stress",
        "total_reward",
        "run_dir",
    ]
    table = dataframe_to_markdown(summary[[c for c in show_cols if c in summary.columns]])

    best_by_yield = pd.DataFrame()
    if not summary.empty and "final_grain_kg_ha" in summary.columns:
        idx = summary.groupby("run_label")["final_grain_kg_ha"].idxmax()
        best_by_yield = summary.loc[idx].sort_values("run_label")

    best_table = dataframe_to_markdown(best_by_yield[[c for c in show_cols if c in best_by_yield.columns]])

    lines = [
        "# 019_07 FQ2016 leaching_cost=20 5K 短训练记录",
        "",
        "## 结论先行",
        "",
        "本轮跑了 FQ2016、seed0、5000 steps 的两个对照：`leaching_cost=0` 和 `leaching_cost=20`，每 1000 steps 评估一次。",
        "",
        "最重要的结果：`leaching_cost=20` 在 2000-step checkpoint 表现很好（GWAD=8012 kg/ha，I=90 mm，N=300 kg/ha，final_cleach=0），但继续训练到 4000/5000 steps 后退化为不灌溉、只施氮，产量降到 7106 kg/ha。因此它不能简单取 final checkpoint；如果继续这条线，必须使用 best checkpoint 选择和 seed 稳定性复核。",
        "",
        "`leaching_cost=0` 在 4000/5000 steps 达到同样 GWAD=8012 kg/ha，I=120 mm，N=300 kg/ha，final_cleach=0。也就是说，在本轮 5K 下，无淋洗惩罚对照并不差，甚至更稳定；`leaching_cost=20` 的优势主要出现在中间 checkpoint 的节水 30 mm，而不是最终 checkpoint。",
        "",
        "这一轮的直接判断：淋洗惩罚链路有效，但 `leaching_cost=20` 仍不是可以直接放大的正式配置。下一步若继续，应该围绕 `best checkpoint` 和 `0 vs 20` 的多 seed 复核，而不是盲目加训练步数。",
        "",
        "## 各 checkpoint 结果",
        "",
        table,
        "",
        "## 各组按产量选出的 best checkpoint",
        "",
        best_table,
        "",
        "## 文件",
        "",
        f"- 汇总表：`{(OUT_DIR / '019_07_fq2016_leaching_cost20_5k_summary.csv').relative_to(PROJECT_ROOT)}`",
        f"- 来源目录：`{ROOT.relative_to(PROJECT_ROOT)}`",
        "",
        "## 下一步建议",
        "",
        "1. 不建议直接把 `leaching_cost=20` 继续训练到更长步数，因为 5K 已经出现后期退化。",
        "2. 如果导师希望纳入淋洗惩罚，建议使用 `best checkpoint` 逻辑，并先做 seed1 复核。",
        "3. 如果当前主线目标仍是产量和水氮效率优先，FQ2016 这一轮暂时更支持把 `leaching_cost=0` 作为稳定对照，把淋洗惩罚作为敏感性/环境效益扩展。",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = collect()
    summary.to_csv(OUT_DIR / "019_07_fq2016_leaching_cost20_5k_summary.csv", index=False, encoding="utf-8-sig")
    write_doc(summary)
    print(summary.to_string(index=False))
    print(DOC_PATH)


if __name__ == "__main__":
    main()
