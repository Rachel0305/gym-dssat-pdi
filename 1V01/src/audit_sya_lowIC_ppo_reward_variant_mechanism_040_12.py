from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


TASK = "040_12_sya_lowIC_ppo_reward_variant_mechanism_audit"
ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "benchmark_results" / TASK
TABLE_DIR = OUT_ROOT / "tables"
DOC_PATH = ROOT / "docs" / f"{TASK}_record.md"

MAINLINE_PATH = (
    ROOT
    / "benchmark_results"
    / "040_00_sya_lowIC_free_timing_maskableppo"
    / "evaluation"
    / "040_00_checkpoint_validation_summary.csv"
)
V11_PATH = (
    ROOT
    / "benchmark_results"
    / "040_11_sya_lowIC_ppo_early_reserve_plus_water_stress_response_reward"
    / "evaluation"
    / "040_11_checkpoint_validation_summary.csv"
)


def _read_selected(path: Path, label: str, checkpoint: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    rows = df[df["checkpoint_step"].astype(int).eq(checkpoint)].copy()
    if rows.empty:
        raise ValueError(f"No rows for {label} checkpoint {checkpoint}: {path}")
    rows["variant"] = label
    rows["variant_checkpoint"] = checkpoint
    return rows


def _select_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "variant",
        "variant_checkpoint",
        "station_code",
        "year",
        "final_grnwt",
        "total_irrigation",
        "total_n",
        "profit_simple",
        "PFP_N",
        "swfac_stress_days_gt_0p05",
        "nstres_days_gt_0p05",
        "max_swfac",
        "max_nstres",
        "early_dap1_10_irrigation",
        "irrigation_event_count",
        "n_event_count",
        "first_irrigation_dap",
        "first_n_dap",
        "action_sequence",
    ]
    optional = [
        "early_reserve_penalty_sum_unscaled",
        "early_overuse_total_mm",
        "water_stress_no_response_penalty_sum_unscaled",
        "water_stress_no_response_days",
    ]
    existing = [c for c in cols + optional if c in df.columns]
    return df[existing].copy()


def _with_deltas(all_rows: pd.DataFrame) -> pd.DataFrame:
    base = all_rows[all_rows["variant"].eq("040_00_ckpt75000")].copy()
    if base.empty:
        raise ValueError("Missing 040_00 baseline rows")
    base = base.set_index("year")
    metrics = [
        "final_grnwt",
        "total_irrigation",
        "total_n",
        "profit_simple",
        "PFP_N",
        "swfac_stress_days_gt_0p05",
        "nstres_days_gt_0p05",
        "max_swfac",
        "early_dap1_10_irrigation",
        "irrigation_event_count",
    ]
    out = all_rows.copy()
    for metric in metrics:
        if metric not in out.columns or metric not in base.columns:
            continue
        out[f"delta_{metric}_vs_040_00"] = out.apply(
            lambda r: r[metric] - base.loc[int(r["year"]), metric]
            if int(r["year"]) in base.index
            else pd.NA,
            axis=1,
        )
    return out


def _summarize(comp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, g in comp.groupby("variant", sort=False):
        row = {
            "variant": variant,
            "checkpoint": int(g["variant_checkpoint"].iloc[0]),
            "year_count": int(g["year"].nunique()),
            "mean_yield": g["final_grnwt"].mean(),
            "min_yield": g["final_grnwt"].min(),
            "mean_irrigation": g["total_irrigation"].mean(),
            "mean_n": g["total_n"].mean(),
            "mean_pfp_n": g["PFP_N"].mean(),
            "mean_swfac_days_gt_0p05": g["swfac_stress_days_gt_0p05"].mean(),
            "max_swfac_days_gt_0p05": g["swfac_stress_days_gt_0p05"].max(),
            "mean_nstres_days_gt_0p05": g["nstres_days_gt_0p05"].mean(),
            "mean_early_dap1_10_irrigation": g["early_dap1_10_irrigation"].mean(),
            "mean_irrigation_event_count": g["irrigation_event_count"].mean(),
        }
        if variant != "040_00_ckpt75000":
            row["years_yield_improved_vs_040_00"] = int((g["delta_final_grnwt_vs_040_00"] > 0).sum())
            row["years_yield_harmed_vs_040_00"] = int((g["delta_final_grnwt_vs_040_00"] < 0).sum())
            row["mean_yield_delta_vs_040_00"] = g["delta_final_grnwt_vs_040_00"].mean()
            row["mean_swfac_days_delta_vs_040_00"] = g[
                "delta_swfac_stress_days_gt_0p05_vs_040_00"
            ].mean()
            row["years_swfac_days_improved_vs_040_00"] = int(
                (g["delta_swfac_stress_days_gt_0p05_vs_040_00"] < 0).sum()
            )
            row["years_swfac_days_harmed_vs_040_00"] = int(
                (g["delta_swfac_stress_days_gt_0p05_vs_040_00"] > 0).sum()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _write_record(summary: pd.DataFrame, comp: pd.DataFrame, branch: str) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_md = summary.round(3).to_markdown(index=False)
    focus_cols = [
        "variant",
        "year",
        "final_grnwt",
        "delta_final_grnwt_vs_040_00",
        "total_irrigation",
        "swfac_stress_days_gt_0p05",
        "delta_swfac_stress_days_gt_0p05_vs_040_00",
        "early_dap1_10_irrigation",
        "irrigation_event_count",
        "action_sequence",
    ]
    focus = comp[comp["variant"].ne("040_00_ckpt75000")][[c for c in focus_cols if c in comp.columns]].copy()
    focus_md = focus.round(3).to_markdown(index=False)
    text = f"""# 040_12 SYA lowIC PPO reward variant mechanism audit

## 任务性质

本任务只读取 040_00 与 040_11 已有评估 CSV，不训练 PPO，不调用 DSSAT，不修改 reward。

## 输入

- `{MAINLINE_PATH.relative_to(ROOT)}`
- `{V11_PATH.relative_to(ROOT)}`

## 对比对象

- `040_00_ckpt75000`：当前主线最佳 checkpoint。
- `040_11_ckpt25000`：早期蓄水惩罚 + 水分胁迫未响应惩罚的近似候选。
- `040_11_ckpt50000`：资源更省但产量更低的候选。

## 分支判定

`{branch}`

## checkpoint 汇总

{summary_md}

## 040_11 相对 040_00 的逐年差异

{focus_md}

## 结论

040_11 没有形成可以直接替换 040_00 的支配性改进。它在部分年份缓解或改变了早期灌溉模式，但不是稳定收益；因此下一步不应继续现场调 040_11 的系数，而应先决定是否引入更明确的 checkpoint guardrail 或重新设计水分响应机制。
"""
    DOC_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    selected = pd.concat(
        [
            _read_selected(MAINLINE_PATH, "040_00_ckpt75000", 75000),
            _read_selected(V11_PATH, "040_11_ckpt25000", 25000),
            _read_selected(V11_PATH, "040_11_ckpt50000", 50000),
        ],
        ignore_index=True,
    )
    selected = _select_columns(selected)
    comp = _with_deltas(selected)
    summary = _summarize(comp)

    comp_path = TABLE_DIR / "040_12_year_level_variant_comparison.csv"
    summary_path = TABLE_DIR / "040_12_checkpoint_variant_summary.csv"
    comp.to_csv(comp_path, index=False)
    summary.to_csv(summary_path, index=False)

    # Conservative branch logic: a variant must improve mean and min yield while
    # not increasing average water stress days to be considered dominant.
    nonbase = summary[summary["variant"].ne("040_00_ckpt75000")].copy()
    base = summary[summary["variant"].eq("040_00_ckpt75000")].iloc[0]
    dominant = nonbase[
        (nonbase["mean_yield"] >= base["mean_yield"])
        & (nonbase["min_yield"] >= base["min_yield"])
        & (nonbase["mean_swfac_days_gt_0p05"] <= base["mean_swfac_days_gt_0p05"])
    ]
    branch = (
        "A_variant_dominates_040_00"
        if not dominant.empty
        else "B_no_reward_variant_dominates_040_00"
    )
    _write_record(summary, comp, branch)

    result = {
        "task": TASK,
        "branch": branch,
        "inputs": {
            "mainline": str(MAINLINE_PATH.relative_to(ROOT)),
            "v11": str(V11_PATH.relative_to(ROOT)),
        },
        "outputs": {
            "year_level_comparison": str(comp_path.relative_to(ROOT)),
            "checkpoint_summary": str(summary_path.relative_to(ROOT)),
            "record_md": str(DOC_PATH.relative_to(ROOT)),
        },
        "dominant_variant_count": int(len(dominant)),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

