from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK = "042_14_sya_lowIC_binary_timing_checkpoint_guardrail_audit"
OUT = ROOT / "benchmark_results" / TASK
FIG_DIR = OUT / "figures"
TAB_DIR = OUT / "tables"
DOC = ROOT / "docs" / f"{TASK}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK}.md"

STATION_CODE = "SYA"
EVAL_SUMMARY = (
    ROOT
    / "benchmark_results"
    / "042_11_sya_lowIC_binary_timing_training_length_curve"
    / "evaluation"
    / "042_11_checkpoint_validation_summary.csv"
)
BASELINE_SUMMARY = (
    ROOT
    / "benchmark_results"
    / "040_21_sya_lowIC_four_baseline_rebuild"
    / "evaluation"
    / "040_21_baseline_summary.csv"
)

GUARDRAIL = {
    "min_unique_action_signatures": 5,
    "min_mean_yield_ratio_vs_official_expert": 0.90,
    "max_low_yield_years_below_90pct_expert": 2,
    "max_severe_swfac_failure_years": 2,
    "max_fixed_event_slots": 3,
}


def ensure_dirs() -> None:
    for path in [OUT, FIG_DIR, TAB_DIR, OUT / "configs"]:
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


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def event_daps_from_daily(path_value: str) -> tuple[list[int], list[int]]:
    path = ROOT / str(path_value)
    daily = pd.read_csv(path, keep_default_na=False)
    dap = numeric(daily["dap"])
    i = numeric(daily["safe_action_amir"]).fillna(0.0)
    n = numeric(daily["safe_action_anfer"]).fillna(0.0)
    return dap[i.gt(0)].astype(int).tolist(), dap[n.gt(0)].astype(int).tolist()


def slot_stats(events_by_year: dict[int, list[int]], event_type: str, checkpoint: int) -> pd.DataFrame:
    max_len = max((len(v) for v in events_by_year.values()), default=0)
    rows = []
    for idx in range(max_len):
        vals = [v[idx] for v in events_by_year.values() if len(v) > idx]
        rows.append(
            {
                "checkpoint_step": checkpoint,
                "event_type": event_type,
                "slot": idx + 1,
                "year_count": len(vals),
                "dap_min": min(vals) if vals else np.nan,
                "dap_max": max(vals) if vals else np.nan,
                "dap_range": (max(vals) - min(vals)) if vals else np.nan,
                "dap_std": float(np.std(vals)) if vals else np.nan,
                "fixed_across_years": bool(len(set(vals)) == 1) if vals else False,
            }
        )
    return pd.DataFrame(rows)


def load_eval_and_baseline() -> tuple[pd.DataFrame, pd.DataFrame]:
    eval_df = pd.read_csv(EVAL_SUMMARY, keep_default_na=False)
    eval_df = eval_df[eval_df["station_code"].astype(str).eq(STATION_CODE)].copy()
    baseline = pd.read_csv(BASELINE_SUMMARY, keep_default_na=False)
    expert = baseline[
        baseline["station_code"].astype(str).eq(STATION_CODE)
        & baseline["scenario"].astype(str).eq("official_extension_expert")
    ][["year", "grain_yield_kg_ha"]].copy()
    expert["year"] = numeric(expert["year"]).astype(int)
    expert["official_expert_yield"] = numeric(expert["grain_yield_kg_ha"])
    return eval_df, expert[["year", "official_expert_yield"]]


def build_guardrail_tables(eval_df: pd.DataFrame, expert: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = eval_df.merge(expert, on="year", how="left")
    merged["final_grnwt"] = numeric(merged["final_grnwt"])
    merged["official_expert_yield"] = numeric(merged["official_expert_yield"])
    merged["yield_ratio_vs_official_expert"] = merged["final_grnwt"] / merged["official_expert_yield"]
    merged["swfac_days_gt_0p05"] = numeric(merged["swfac_days_gt_0p05"]).fillna(0).astype(int)
    merged["nstres_days_gt_0p05"] = numeric(merged["nstres_days_gt_0p05"]).fillna(0).astype(int)
    merged["total_irrigation"] = numeric(merged["total_irrigation"])
    merged["total_n"] = numeric(merged["total_n"])
    merged["PFP_N"] = numeric(merged["PFP_N"])

    slot_frames = []
    summary_rows = []
    for checkpoint, sub in merged.groupby(numeric(merged["checkpoint_step"]).astype(int)):
        sub = sub.sort_values("year").copy()
        irr_by_year: dict[int, list[int]] = {}
        nit_by_year: dict[int, list[int]] = {}
        for _, row in sub.iterrows():
            year = int(row["year"])
            irr, nit = event_daps_from_daily(str(row["daily_csv_path"]))
            irr_by_year[year] = irr
            nit_by_year[year] = nit
        irr_slots = slot_stats(irr_by_year, "irrigation", int(checkpoint))
        nit_slots = slot_stats(nit_by_year, "nitrogen", int(checkpoint))
        slot = pd.concat([irr_slots, nit_slots], ignore_index=True)
        slot_frames.append(slot)

        fixed_event_slots = int(slot["fixed_across_years"].sum()) if not slot.empty else 0
        max_slot_dap_range = int(pd.to_numeric(slot["dap_range"], errors="coerce").max()) if not slot.empty else 0
        unique_signatures = int(sub["action_sequence"].nunique())
        mean_ratio = float(sub["yield_ratio_vs_official_expert"].mean())
        low_yield_years = int(sub["yield_ratio_vs_official_expert"].lt(0.90).sum())
        severe_swfac_years = int(sub["swfac_days_gt_0p05"].ge(20).sum())
        candidate_pass = (
            unique_signatures >= GUARDRAIL["min_unique_action_signatures"]
            and mean_ratio >= GUARDRAIL["min_mean_yield_ratio_vs_official_expert"]
            and low_yield_years <= GUARDRAIL["max_low_yield_years_below_90pct_expert"]
            and severe_swfac_years <= GUARDRAIL["max_severe_swfac_failure_years"]
            and fixed_event_slots <= GUARDRAIL["max_fixed_event_slots"]
        )
        summary_rows.append(
            {
                "checkpoint_step": int(checkpoint),
                "validation_years": int(sub["year"].nunique()),
                "mean_yield": float(sub["final_grnwt"].mean()),
                "mean_official_expert_yield": float(sub["official_expert_yield"].mean()),
                "mean_yield_ratio_vs_official_expert": mean_ratio,
                "low_yield_years_below_90pct_expert": low_yield_years,
                "mean_total_irrigation": float(sub["total_irrigation"].mean()),
                "mean_total_n": float(sub["total_n"].mean()),
                "mean_pfp_n": float(sub["PFP_N"].mean(skipna=True)),
                "mean_swfac_days_gt_0p05": float(sub["swfac_days_gt_0p05"].mean()),
                "severe_swfac_failure_years": severe_swfac_years,
                "mean_nstres_days_gt_0p05": float(sub["nstres_days_gt_0p05"].mean()),
                "unique_action_signatures": unique_signatures,
                "fixed_event_slots": fixed_event_slots,
                "max_slot_dap_range": max_slot_dap_range,
                "candidate_pass": bool(candidate_pass),
            }
        )
    return pd.DataFrame(summary_rows).sort_values("checkpoint_step"), pd.concat(slot_frames, ignore_index=True)


def plot_guardrail(summary: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = summary["checkpoint_step"].astype(str)
    axes[0, 0].bar(x, summary["mean_yield_ratio_vs_official_expert"], color="#2A7FB8")
    axes[0, 0].axhline(GUARDRAIL["min_mean_yield_ratio_vs_official_expert"], color="#555555", ls="--", lw=1)
    axes[0, 0].set_ylabel("Mean yield / official expert")
    axes[0, 0].set_title("Yield guardrail", loc="left", fontweight="bold")

    axes[0, 1].bar(x, summary["unique_action_signatures"], color="#2A9D55")
    axes[0, 1].axhline(GUARDRAIL["min_unique_action_signatures"], color="#555555", ls="--", lw=1)
    axes[0, 1].set_ylabel("Unique action signatures")
    axes[0, 1].set_title("Action diversity", loc="left", fontweight="bold")

    axes[1, 0].bar(x, summary["fixed_event_slots"], color="#D8A305")
    axes[1, 0].axhline(GUARDRAIL["max_fixed_event_slots"], color="#555555", ls="--", lw=1)
    axes[1, 0].set_ylabel("Fixed event slots")
    axes[1, 0].set_title("Template component guardrail", loc="left", fontweight="bold")

    axes[1, 1].bar(x, summary["severe_swfac_failure_years"], color="#C44E52")
    axes[1, 1].axhline(GUARDRAIL["max_severe_swfac_failure_years"], color="#555555", ls="--", lw=1)
    axes[1, 1].set_ylabel("Years with WSPD days >=20")
    axes[1, 1].set_title("Water-stress guardrail", loc="left", fontweight="bold")

    for ax in axes.flat:
        ax.set_xlabel("Checkpoint step")
        ax.grid(axis="y", color="#E8E8E8", linewidth=0.7)
    fig.suptitle("042_14 binary-timing PPO checkpoint guardrail audit", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = FIG_DIR / "042_14_binary_timing_checkpoint_guardrail_summary.png"
    svg = png.with_suffix(".svg")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return [png, svg]


def write_record(summary: pd.DataFrame, slot: pd.DataFrame, result: dict) -> None:
    passed = summary[summary["candidate_pass"]]
    if passed.empty:
        decision = "没有 checkpoint 通过本轮候选 guardrail 草案。"
    else:
        selected = int(passed.sort_values(["mean_yield_ratio_vs_official_expert", "unique_action_signatures"], ascending=[False, False]).iloc[0]["checkpoint_step"])
        decision = f"通过草案 guardrail 的 checkpoint：{', '.join(map(str, passed['checkpoint_step'].astype(int).tolist()))}；按产量比例优先的候选为 {selected}。"

    lines = [
        "# 042_14 SYA lowIC binary-timing PPO checkpoint 综合 guardrail 审计记录",
        "",
        "## 结论先说",
        "",
        f"- {decision}",
        "- 本任务不训练、不调参，只读取 042_11 已有 checkpoint 结果。",
        "- 这里的 guardrail 是诊断性草案，用于把“指标表现”和“动作合理性”同时放进 checkpoint 审计；不是最终冻结协议。",
        "",
        "## Guardrail 草案",
        "",
        md_table(pd.DataFrame([GUARDRAIL]), max_rows=5),
        "",
        "## Checkpoint 综合表",
        "",
        md_table(summary, max_rows=20),
        "",
        "## 事件槽位变异性",
        "",
        md_table(slot, max_rows=80),
        "",
        "## 输出图",
        "",
        *[f"- `{p}`" for p in result["figures"]],
        "",
        "## 结论边界",
        "",
        "- 如果要把该规则正式用于下一轮训练，必须在训练前写进 prompt，而不是继续事后挑 checkpoint。",
        "- 25K 可以作为当前候选，但它仍有固定 DAP1/DAP2/DAP91 模板成分；后续若追求更强天气响应，需要把响应性指标前置到正式选择协议中。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ensure_dirs()
    eval_df, expert = load_eval_and_baseline()
    summary, slot = build_guardrail_tables(eval_df, expert)
    summary_path = TAB_DIR / "042_14_binary_timing_checkpoint_guardrail_summary.csv"
    slot_path = TAB_DIR / "042_14_binary_timing_event_slot_variability_by_checkpoint.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    slot.to_csv(slot_path, index=False, encoding="utf-8-sig")
    figures = plot_guardrail(summary)
    result = {
        "task": TASK,
        "passed_checkpoints": summary.loc[summary["candidate_pass"], "checkpoint_step"].astype(int).tolist(),
        "summary_csv": summary_path.relative_to(ROOT).as_posix(),
        "slot_csv": slot_path.relative_to(ROOT).as_posix(),
        "figures": [p.relative_to(ROOT).as_posix() for p in figures],
        "record_md": DOC.relative_to(ROOT).as_posix(),
    }
    write_record(summary, slot, result)
    (OUT / "042_14_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

