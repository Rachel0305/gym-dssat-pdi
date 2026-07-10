from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_sy_local_dqn_train_cross_year_transfer_017_08 as sy


OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "sy2014_seed1_minimal_reproduction_018_08"
FIG_DIR = OUT_DIR / "figures"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-09_018_08_sy2014_seed1_minimal_reproduction_record.md"
SEED0_SUMMARY = PROJECT_ROOT / "DSSAT_auto_validation" / "sy_local_dqn_train_cross_year_transfer_017_08" / "017_08_sy_dqn_checkpoint_summary.csv"


def fmt(v: object) -> str:
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else f"{float(v):.2f}"
    return str(v)


def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    cols = [c for c in cols if c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_record(
    smoke_summary: pd.DataFrame,
    formal_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    smoke_ok: bool,
    formal_ok: bool,
) -> None:
    lines = [
        "# 018_08 SY2014 seed1 最小复现实验记录",
        "",
        "## 目的",
        "",
        "- 不覆盖 017_08 旧结果。",
        "- 在与 seed0 相同框架、奖励、动作空间、预算和输入条件下，补做 SY2014 seed1。",
        "- 严格先做 5K smoke，再决定是否继续 50K。",
        "",
        "## 设置",
        "",
        "- 训练年：SY2014",
        "- seed：1",
        "- 预算：I120 / N300",
        "- 动作表：沿用 017_08 的 9 个离散联合动作",
        "- 奖励：沿用 017_08 的 baseline-relative DQN 奖励",
        "",
        "## 5K smoke 结果",
        "",
        f"- smoke 是否成功：{'是' if smoke_ok else '否'}",
        "",
        markdown_table(
            smoke_summary.sort_values(["checkpoint"]),
            ["year", "scenario", "checkpoint", "final_gwad", "irrigation_total", "fertilizer_total", "total_reward", "yield_diff_vs_null", "yield_diff_vs_dssat_auto"],
        ) if not smoke_summary.empty else "无结果",
        "",
        "## 50K 正式结果",
        "",
        f"- formal 是否执行成功：{'是' if formal_ok else '否'}",
        "",
        markdown_table(
            formal_summary.sort_values(["checkpoint"]),
            ["year", "scenario", "checkpoint", "final_gwad", "irrigation_total", "fertilizer_total", "total_reward", "yield_diff_vs_null", "yield_diff_vs_dssat_auto"],
        ) if not formal_summary.empty else "未执行或无结果",
        "",
        "## seed0 / seed1 对比",
        "",
        markdown_table(
            comparison,
            ["seed", "checkpoint", "final_gwad", "irrigation_total", "fertilizer_total", "total_reward", "yield_diff_vs_null", "yield_diff_vs_dssat_auto"],
        ) if not comparison.empty else "无可比结果",
        "",
        "## 结论",
        "",
    ]
    if not smoke_ok:
        lines += [
            "- 5K smoke 未通过，因此未继续 50K。",
        ]
    elif not formal_ok:
        lines += [
            "- 5K smoke 通过，但 50K 正式训练未成功完成。",
        ]
    else:
        lines += [
            "- 5K smoke 与 50K 正式训练均已完成。",
            "- 是否达到与 seed0 接近的高产/资源配置，需要看上表的 best checkpoint 对比。",
        ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def extract_dqn_only(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    keep = summary["scenario"].astype(str).str.startswith("dqn_ckpt")
    out = summary.loc[keep].copy()
    return out.sort_values(["checkpoint"]).reset_index(drop=True)


def add_relative_for(summary: pd.DataFrame, baseline_summary: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([baseline_summary, summary], ignore_index=True)
    combined = sy.add_relative(combined)
    return extract_dqn_only(combined)


def build_seed_compare(seed1_formal: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if SEED0_SUMMARY.exists():
        seed0 = pd.read_csv(SEED0_SUMMARY)
        seed0 = seed0[(seed0["year"] == 2014)].copy()
        if not seed0.empty:
            best0 = seed0.sort_values(["total_reward", "final_gwad"], ascending=[False, False]).head(1).copy()
            best0["seed"] = 0
            rows.append(best0)
    if not seed1_formal.empty:
        best1 = seed1_formal.sort_values(["total_reward", "final_gwad"], ascending=[False, False]).head(1).copy()
        best1["seed"] = 1
        rows.append(best1)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["checkpoint"] = out["checkpoint"].astype(int)
    cols = ["seed", "checkpoint", "final_gwad", "irrigation_total", "fertilizer_total", "total_reward", "yield_diff_vs_null", "yield_diff_vs_dssat_auto"]
    return out[cols].sort_values("seed").reset_index(drop=True)


def main() -> None:
    sy.OUT_DIR = OUT_DIR
    sy.FIG_DIR = FIG_DIR
    sy.DOC_PATH = DOC_PATH
    sy.configure_globals()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline_daily, baseline_events, baseline_summary = sy.screen_baselines([2014])
    baseline_summary = sy.add_relative(baseline_summary)
    baseline_daily.to_csv(OUT_DIR / "018_08_sy2014_baseline_daily.csv", index=False, encoding="utf-8-sig")
    baseline_events.to_csv(OUT_DIR / "018_08_sy2014_baseline_events.csv", index=False, encoding="utf-8-sig")
    baseline_summary.to_csv(OUT_DIR / "018_08_sy2014_baseline_summary.csv", index=False, encoding="utf-8-sig")

    smoke_ok = False
    formal_ok = False
    smoke_dqn = pd.DataFrame()
    formal_dqn = pd.DataFrame()

    try:
        smoke_daily, smoke_events, smoke_summary = sy.train_checkpoint(2014, 1, 5000, 5000)
        smoke_dqn = add_relative_for(smoke_summary, baseline_summary)
        smoke_daily.to_csv(OUT_DIR / "018_08_smoke_checkpoint_daily.csv", index=False, encoding="utf-8-sig")
        smoke_events.to_csv(OUT_DIR / "018_08_smoke_checkpoint_events.csv", index=False, encoding="utf-8-sig")
        smoke_dqn.to_csv(OUT_DIR / "018_08_smoke_checkpoint_summary.csv", index=False, encoding="utf-8-sig")
        smoke_ok = not smoke_dqn.empty
    except Exception as exc:  # pragma: no cover
        (OUT_DIR / "018_08_smoke_error.txt").write_text(str(exc), encoding="utf-8")

    if smoke_ok:
        try:
            formal_daily, formal_events, formal_summary = sy.train_checkpoint(2014, 1, 50000, 5000)
            formal_dqn = add_relative_for(formal_summary, baseline_summary)
            formal_daily.to_csv(OUT_DIR / "018_08_formal_checkpoint_daily.csv", index=False, encoding="utf-8-sig")
            formal_events.to_csv(OUT_DIR / "018_08_formal_checkpoint_events.csv", index=False, encoding="utf-8-sig")
            formal_dqn.to_csv(OUT_DIR / "018_08_formal_checkpoint_summary.csv", index=False, encoding="utf-8-sig")
            formal_ok = not formal_dqn.empty
        except Exception as exc:  # pragma: no cover
            (OUT_DIR / "018_08_formal_error.txt").write_text(str(exc), encoding="utf-8")

    comparison = build_seed_compare(formal_dqn)
    if not comparison.empty:
        comparison.to_csv(OUT_DIR / "018_08_seed0_vs_seed1_comparison.csv", index=False, encoding="utf-8-sig")
    write_record(smoke_dqn, formal_dqn, comparison, smoke_ok, formal_ok)
    print(f"[018_08] smoke_ok={smoke_ok} formal_ok={formal_ok} out_dir={OUT_DIR}")
    if not comparison.empty:
        print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
