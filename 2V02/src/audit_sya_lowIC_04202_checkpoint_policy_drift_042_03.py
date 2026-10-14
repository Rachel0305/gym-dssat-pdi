from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "042_03"
TASK_NAME = "sya_lowIC_04202_checkpoint_policy_drift_audit"
SRC = ROOT / "benchmark_results" / "042_02_sya_lowIC_normalized_weather_teacher_warmstart_stress_response_rerun100k"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
TABLES = OUT / "tables"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"

SUMMARY = SRC / "evaluation" / "041_03_checkpoint_validation_summary.csv"
BY_CKPT = SRC / "evaluation" / "041_03_validation_summary_by_checkpoint.csv"
TRAIN_INV = SRC / "evaluation" / "041_03_training_checkpoint_inventory.csv"


def require_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size <= 0:
        raise ValueError(f"Empty CSV: {path}")
    return pd.read_csv(path, keep_default_na=False)


def md_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "无记录。"
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number", "bool"]).columns:
        work[col] = pd.to_numeric(work[col], errors="ignore")
        if pd.api.types.is_numeric_dtype(work[col]):
            work[col] = work[col].round(4)
    work = work.astype(object).where(pd.notna(work), "")
    lines = [
        "| " + " | ".join(map(str, work.columns)) + " |",
        "| " + " | ".join(["---"] * len(work.columns)) + " |",
    ]
    for row in work.to_numpy().tolist():
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def parse_daily_actions(path: Path) -> dict[str, object]:
    df = require_csv(path)
    for col in ["dap", "irrigation_mm_action", "nitrogen_kg_ha_action", "swfac", "nstres"]:
        if col not in df.columns:
            raise KeyError(f"{path} missing {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    nonzero_i = df[df["irrigation_mm_action"].fillna(0) > 0]
    nonzero_n = df[df["nitrogen_kg_ha_action"].fillna(0) > 0]
    def fmt_events(sub: pd.DataFrame, amount_col: str) -> str:
        if sub.empty:
            return ""
        return "; ".join(f"DAP{int(round(r.dap))}:{float(getattr(r, amount_col)):.0f}" for r in sub.itertuples(index=False))
    return {
        "irrigation_events": fmt_events(nonzero_i, "irrigation_mm_action"),
        "nitrogen_events": fmt_events(nonzero_n, "nitrogen_kg_ha_action"),
        "irrigation_event_count": int(len(nonzero_i)),
        "nitrogen_event_count": int(len(nonzero_n)),
        "total_irrigation_from_daily": float(nonzero_i["irrigation_mm_action"].sum()),
        "total_n_from_daily": float(nonzero_n["nitrogen_kg_ha_action"].sum()),
        "first_irrigation_dap": int(round(nonzero_i["dap"].iloc[0])) if len(nonzero_i) else np.nan,
        "first_n_dap": int(round(nonzero_n["dap"].iloc[0])) if len(nonzero_n) else np.nan,
        "max_swfac_from_daily": float(df["swfac"].max()),
        "max_nstres_from_daily": float(df["nstres"].max()),
    }


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)

    summary = require_csv(SUMMARY)
    by_ckpt = require_csv(BY_CKPT)
    train_inv = require_csv(TRAIN_INV)

    rows = []
    for r in summary.itertuples(index=False):
        path = ROOT / str(r.daily_csv_path)
        action_info = parse_daily_actions(path)
        row = {
            "year": int(r.year),
            "checkpoint_step": int(r.checkpoint_step),
            "stage": str(getattr(r, "stage", "")),
            "grain_yield_kg_ha": float(r.grain_yield_kg_ha),
            "WP_ET_kg_m3": float(r.WP_ET_kg_m3),
            "PFP_N_kg_kg": float(r.PFP_N_kg_kg) if str(r.PFP_N_kg_kg) not in {"", "nan"} else np.nan,
            "summary_irrigation_total": float(r.summary_irrigation_total),
            "summary_nitrogen_total": float(r.summary_nitrogen_total),
            "gap_yield_vs_four_max": float(r.gap_yield_vs_four_max),
            "gap_wp_et_vs_four_max": float(r.gap_wp_et_vs_four_max),
            "gap_pfp_n_vs_four_max": float(r.gap_pfp_n_vs_four_max) if str(r.gap_pfp_n_vs_four_max) not in {"", "nan"} else np.nan,
            "any_metric_win_vs_four_max": str(r.any_metric_win_vs_four_max).lower() == "true",
            "all3_win_vs_four_max": str(r.all3_win_vs_four_max).lower() == "true",
            **action_info,
            "daily_csv_path": str(r.daily_csv_path),
        }
        rows.append(row)

    actions = pd.DataFrame(rows).sort_values(["checkpoint_step", "year"])
    actions_path = TABLES / "042_03_checkpoint_year_action_metrics.csv"
    actions.to_csv(actions_path, index=False, encoding="utf-8-sig")

    compact = (
        actions.groupby("checkpoint_step", as_index=False)
        .agg(
            mean_yield=("grain_yield_kg_ha", "mean"),
            mean_wp_et=("WP_ET_kg_m3", "mean"),
            mean_pfp_n=("PFP_N_kg_kg", "mean"),
            mean_irrigation=("summary_irrigation_total", "mean"),
            mean_nitrogen=("summary_nitrogen_total", "mean"),
            mean_irrigation_event_count=("irrigation_event_count", "mean"),
            mean_nitrogen_event_count=("nitrogen_event_count", "mean"),
            any_metric_win_years=("any_metric_win_vs_four_max", "sum"),
            all3_win_years=("all3_win_vs_four_max", "sum"),
            max_swfac=("max_swfac_from_daily", "max"),
            max_nstres=("max_nstres_from_daily", "max"),
        )
        .sort_values("checkpoint_step")
    )
    compact_path = TABLES / "042_03_checkpoint_drift_summary.csv"
    compact.to_csv(compact_path, index=False, encoding="utf-8-sig")

    pivot = actions.pivot(index="year", columns="checkpoint_step", values="nitrogen_events").reset_index()
    pivot_path = TABLES / "042_03_nitrogen_events_by_year_checkpoint.csv"
    pivot.to_csv(pivot_path, index=False, encoding="utf-8-sig")

    cmp75 = actions[actions["checkpoint_step"].eq(75000)].set_index("year")
    cmp100 = actions[actions["checkpoint_step"].eq(100000)].set_index("year")
    paired = cmp75[[
        "grain_yield_kg_ha",
        "WP_ET_kg_m3",
        "PFP_N_kg_kg",
        "summary_irrigation_total",
        "summary_nitrogen_total",
        "irrigation_events",
        "nitrogen_events",
        "all3_win_vs_four_max",
    ]].join(
        cmp100[[
            "grain_yield_kg_ha",
            "WP_ET_kg_m3",
            "PFP_N_kg_kg",
            "summary_irrigation_total",
            "summary_nitrogen_total",
            "irrigation_events",
            "nitrogen_events",
            "all3_win_vs_four_max",
        ]],
        how="inner",
        lsuffix="_75k",
        rsuffix="_100k",
    ).reset_index()
    paired["delta_yield_100k_minus_75k"] = paired["grain_yield_kg_ha_100k"] - paired["grain_yield_kg_ha_75k"]
    paired["delta_n_100k_minus_75k"] = paired["summary_nitrogen_total_100k"] - paired["summary_nitrogen_total_75k"]
    paired["nitrogen_lost_by_100k"] = (paired["summary_nitrogen_total_75k"] > 0) & (paired["summary_nitrogen_total_100k"] == 0)
    paired_path = TABLES / "042_03_75k_vs_100k_policy_drift.csv"
    paired.to_csv(paired_path, index=False, encoding="utf-8-sig")

    n_loss_years = int(paired["nitrogen_lost_by_100k"].sum())
    branch = "A_late_finetune_nitrogen_collapse_confirmed" if n_loss_years == len(paired) else "B_mixed_or_unconfirmed_drift"

    lines = [
        "# 042_03 SYA lowIC 042_02 checkpoint 策略漂移审计记录",
        "",
        "## 任务性质",
        "",
        "- 只读取 `042_02_rerun100k` 已有结果；不训练；不运行 DSSAT；不修改原始结果。",
        "- 目的：解释为什么 75K checkpoint 表现较好，而 100K checkpoint 退化。",
        "",
        "## 输入",
        "",
        f"- validation summary: `{SUMMARY.relative_to(ROOT)}`",
        f"- by checkpoint: `{BY_CKPT.relative_to(ROOT)}`",
        f"- training inventory: `{TRAIN_INV.relative_to(ROOT)}`",
        "",
        "## checkpoint 汇总",
        "",
        md_table(compact, max_rows=20),
        "",
        "## 75K vs 100K 配对差异",
        "",
        md_table(
            paired[[
                "year",
                "grain_yield_kg_ha_75k",
                "grain_yield_kg_ha_100k",
                "delta_yield_100k_minus_75k",
                "summary_nitrogen_total_75k",
                "summary_nitrogen_total_100k",
                "delta_n_100k_minus_75k",
                "nitrogen_lost_by_100k",
                "all3_win_vs_four_max_75k",
                "all3_win_vs_four_max_100k",
            ]],
            max_rows=20,
        ),
        "",
        "## 判定",
        "",
        f"- 分支：`{branch}`",
        f"- 100K 相对 75K 丢失施氮的年份数：{n_loss_years}/{len(paired)}。",
        "- 如果本分支为 A，说明 100K 退化不是随机个别年份，而是后期 fine-tune 系统性把施氮策略推到 N=0。",
        "- 本任务不能直接授权事后采用 75K；只能支持下一步预注册 checkpoint 选择或 early stopping 规则。",
        "",
        "## 限制",
        "",
        "- `042_02` 训练器没有写出训练年份 reset log，因此本任务不能审计训练年份采样偏差。",
        "- 如需检查 reset 分布，下一轮训练脚本需显式保存 `RandomYearEnv.switch_log`。",
        "",
        "## 输出",
        "",
        f"- action metrics: `{actions_path.relative_to(ROOT)}`",
        f"- checkpoint summary: `{compact_path.relative_to(ROOT)}`",
        f"- nitrogen events pivot: `{pivot_path.relative_to(ROOT)}`",
        f"- 75K vs 100K paired drift: `{paired_path.relative_to(ROOT)}`",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "branch": branch,
        "source": str(SRC.relative_to(ROOT)),
        "n_paired_years": int(len(paired)),
        "nitrogen_lost_by_100k_years": n_loss_years,
        "outputs": {
            "record_md": str(DOC.relative_to(ROOT)),
            "action_metrics": str(actions_path.relative_to(ROOT)),
            "checkpoint_summary": str(compact_path.relative_to(ROOT)),
            "nitrogen_events_pivot": str(pivot_path.relative_to(ROOT)),
            "paired_drift": str(paired_path.relative_to(ROOT)),
        },
    }
    (OUT / "042_03_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
