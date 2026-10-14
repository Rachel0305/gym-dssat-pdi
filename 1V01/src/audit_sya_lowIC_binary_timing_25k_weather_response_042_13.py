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
TASK = "042_13_sya_lowIC_binary_timing_25k_weather_response_audit"
OUT = ROOT / "benchmark_results" / TASK
FIG_DIR = OUT / "figures"
TAB_DIR = OUT / "tables"
DOC = ROOT / "docs" / f"{TASK}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK}.md"

STATION_CODE = "SYA"
CHECKPOINT = 25_000
EVAL_SUMMARY = (
    ROOT
    / "benchmark_results"
    / "042_11_sya_lowIC_binary_timing_training_length_curve"
    / "evaluation"
    / "042_11_checkpoint_validation_summary.csv"
)


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


def load_eval_rows() -> pd.DataFrame:
    df = pd.read_csv(EVAL_SUMMARY, keep_default_na=False)
    sub = df[
        df["station_code"].astype(str).eq(STATION_CODE)
        & pd.to_numeric(df["checkpoint_step"], errors="coerce").eq(CHECKPOINT)
    ].copy()
    if sub.empty:
        raise RuntimeError(f"No {STATION_CODE} checkpoint {CHECKPOINT} rows in {EVAL_SUMMARY}")
    return sub.sort_values("year").reset_index(drop=True)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def daily_from_path(path_value: str) -> pd.DataFrame:
    path = ROOT / str(path_value)
    if not path.exists():
        raise FileNotFoundError(path)
    daily = pd.read_csv(path, keep_default_na=False)
    for col in [
        "year",
        "dap",
        "rain",
        "tmax",
        "tmin",
        "swfac",
        "nstres",
        "safe_action_amir",
        "safe_action_anfer",
        "grnwt",
        "topwt",
    ]:
        if col in daily.columns:
            daily[col] = numeric(daily[col])
    return daily.sort_values("dap").reset_index(drop=True)


def window_stats(daily: pd.DataFrame, dap: float, stress_col: str) -> dict[str, float]:
    dap_series = numeric(daily["dap"])
    pre7 = daily[dap_series.between(dap - 7, dap - 1, inclusive="both")]
    post7 = daily[dap_series.between(dap, dap + 7, inclusive="both")]
    row = daily[dap_series.eq(dap)]
    return {
        "rain_pre7_mm": float(numeric(pre7["rain"]).sum()) if len(pre7) else 0.0,
        "rain_post7_mm": float(numeric(post7["rain"]).sum()) if len(post7) else 0.0,
        f"{stress_col}_at_event": float(numeric(row[stress_col]).iloc[0]) if len(row) else np.nan,
        f"{stress_col}_pre7_max": float(numeric(pre7[stress_col]).max()) if len(pre7) else np.nan,
        f"{stress_col}_post7_max": float(numeric(post7[stress_col]).max()) if len(post7) else np.nan,
        f"{stress_col}_season_max": float(numeric(daily[stress_col]).max()),
    }


def collect_event_context(eval_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    yearly_rows: list[dict] = []
    irrigation_rows: list[dict] = []
    nitrogen_rows: list[dict] = []
    for _, ev in eval_rows.iterrows():
        year = int(ev["year"])
        daily = daily_from_path(str(ev["daily_csv_path"]))
        dap = numeric(daily["dap"])
        i = numeric(daily["safe_action_amir"]).fillna(0.0)
        n = numeric(daily["safe_action_anfer"]).fillna(0.0)
        i_event_daps = dap[i.gt(0)].astype(int).tolist()
        n_event_daps = dap[n.gt(0)].astype(int).tolist()
        yearly_rows.append(
            {
                "year": year,
                "final_grain_kg_ha": float(ev["final_grnwt"]),
                "total_irrigation_mm": float(ev["total_irrigation"]),
                "total_nitrogen_kg_ha": float(ev["total_n"]),
                "irrigation_event_count": int(i.gt(0).sum()),
                "nitrogen_event_count": int(n.gt(0).sum()),
                "irrigation_event_daps": ";".join(map(str, i_event_daps)),
                "nitrogen_event_daps": ";".join(map(str, n_event_daps)),
                "max_swfac": float(ev["max_swfac"]),
                "max_nstres": float(ev["max_nstres"]),
                "swfac_days_gt_0p05": int(ev["swfac_days_gt_0p05"]),
                "nstres_days_gt_0p05": int(ev["nstres_days_gt_0p05"]),
                "action_sequence": str(ev["action_sequence"]),
            }
        )
        for slot, event_dap in enumerate(i_event_daps, start=1):
            stats = window_stats(daily, float(event_dap), "swfac")
            irrigation_rows.append(
                {
                    "year": year,
                    "slot": slot,
                    "dap": event_dap,
                    "amount_mm": float(i[dap.eq(event_dap)].iloc[0]),
                    **stats,
                }
            )
        for slot, event_dap in enumerate(n_event_daps, start=1):
            stats = window_stats(daily, float(event_dap), "nstres")
            nitrogen_rows.append(
                {
                    "year": year,
                    "slot": slot,
                    "dap": event_dap,
                    "amount_kg_ha": float(n[dap.eq(event_dap)].iloc[0]),
                    **stats,
                }
            )
    return pd.DataFrame(yearly_rows), pd.DataFrame(irrigation_rows), pd.DataFrame(nitrogen_rows)


def slot_variability(events: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows = []
    for slot, sub in events.groupby("slot"):
        rows.append(
            {
                "event_type": prefix,
                "slot": int(slot),
                "year_count": int(sub["year"].nunique()),
                "dap_min": int(sub["dap"].min()),
                "dap_max": int(sub["dap"].max()),
                "dap_range": int(sub["dap"].max() - sub["dap"].min()),
                "dap_std": float(pd.to_numeric(sub["dap"], errors="coerce").std(ddof=0)),
                "fixed_across_years": bool(sub["dap"].nunique() == 1),
            }
        )
    return pd.DataFrame(rows)


def plot_event_daps(yearly: pd.DataFrame, irrigation: pd.DataFrame, nitrogen: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, events, title, color in [
        (axes[0], irrigation, "Irrigation event DAPs", "#2A7FB8"),
        (axes[1], nitrogen, "Nitrogen event DAPs", "#2A9D55"),
    ]:
        if not events.empty:
            sc = ax.scatter(events["dap"], events["year"], s=90, c=events["slot"], cmap="viridis", edgecolor="black", linewidth=0.4)
            for _, row in events.iterrows():
                ax.text(row["dap"], row["year"] + 0.08, str(int(row["slot"])), ha="center", va="bottom", fontsize=7)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("DAP")
        ax.grid(color="#E8E8E8", linewidth=0.7)
        ax.set_xlim(0, 125)
    axes[0].set_ylabel("Validation year")
    axes[0].set_yticks(sorted(yearly["year"].astype(int).unique()))
    fig.suptitle("042_13 binary-timing PPO ckpt25k action timing across years", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIG_DIR / "042_13_binary_timing_ckpt25k_event_dap_scatter.png"
    svg = out.with_suffix(".svg")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return [out, svg]


def plot_event_context(irrigation: pd.DataFrame, nitrogen: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=False)
    if not irrigation.empty:
        irr = irrigation.groupby("year", as_index=False).agg(
            mean_event_dap=("dap", "mean"),
            rain_pre7_mm=("rain_pre7_mm", "mean"),
            rain_post7_mm=("rain_post7_mm", "mean"),
            swfac_pre7_max=("swfac_pre7_max", "max"),
            swfac_post7_max=("swfac_post7_max", "max"),
        )
        axes[0, 0].bar(irr["year"] - 0.2, irr["rain_pre7_mm"], width=0.4, label="pre 7d rain", color="#88BDE6")
        axes[0, 0].bar(irr["year"] + 0.2, irr["rain_post7_mm"], width=0.4, label="post 7d rain", color="#2A7FB8")
        axes[0, 1].bar(irr["year"] - 0.2, irr["swfac_pre7_max"], width=0.4, label="pre 7d max WSPD", color="#F4B183")
        axes[0, 1].bar(irr["year"] + 0.2, irr["swfac_post7_max"], width=0.4, label="post 7d max WSPD", color="#C55A11")
    if not nitrogen.empty:
        nit = nitrogen.groupby("year", as_index=False).agg(
            mean_event_dap=("dap", "mean"),
            nstres_pre7_max=("nstres_pre7_max", "max"),
            nstres_post7_max=("nstres_post7_max", "max"),
            nstres_season_max=("nstres_season_max", "max"),
        )
        axes[1, 0].bar(nit["year"] - 0.2, nit["nstres_pre7_max"], width=0.4, label="pre 7d max NSTD", color="#B7D7A8")
        axes[1, 0].bar(nit["year"] + 0.2, nit["nstres_post7_max"], width=0.4, label="post 7d max NSTD", color="#38761D")
        axes[1, 1].scatter(nit["mean_event_dap"], nit["nstres_season_max"], s=60, color="#38761D")
        for _, row in nit.iterrows():
            axes[1, 1].text(row["mean_event_dap"], row["nstres_season_max"], str(int(row["year"])), fontsize=7)
    titles = [
        (axes[0, 0], "Irrigation event rainfall context", "Rain around irrigation events (mm)"),
        (axes[0, 1], "Irrigation event water-stress context", "WSPD max around irrigation events"),
        (axes[1, 0], "Nitrogen event nitrogen-stress context", "NSTD max around N events"),
        (axes[1, 1], "Mean N-event timing vs season N stress", "Season max NSTD"),
    ]
    for ax, title, ylabel in titles:
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(color="#E8E8E8", linewidth=0.7)
        ax.legend(fontsize=8, loc="best") if ax.get_legend_handles_labels()[0] else None
    for ax in axes.flat:
        ax.set_xlabel("Year" if ax is not axes[1, 1] else "Mean N-event DAP")
    fig.suptitle("042_13 event context audit", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIG_DIR / "042_13_binary_timing_ckpt25k_event_context.png"
    svg = out.with_suffix(".svg")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return [out, svg]


def classify_branch(slot_var: pd.DataFrame, yearly: pd.DataFrame) -> tuple[str, str]:
    fixed_slots = int(slot_var["fixed_across_years"].sum()) if not slot_var.empty else 0
    max_range = float(slot_var["dap_range"].max()) if not slot_var.empty else 0.0
    unique_signatures = int(yearly["action_sequence"].nunique())
    if unique_signatures <= 2:
        return "C_template_like", "动作签名很少，整体接近固定模板。"
    if fixed_slots >= 3 and max_range >= 14:
        return "B_partial_response_with_strong_template_components", "动作签名跨年变化，但多个事件槽位固定，属于部分响应而非完全自由天气响应。"
    if unique_signatures >= 7 and fixed_slots <= 2:
        return "A_weather_responsive_candidate", "动作签名变化较多，固定槽位较少，可作为较强天气响应候选。"
    return "B_partial_response", "动作有变化，但仍不足以判定为强天气响应。"


def write_record(result: dict, yearly: pd.DataFrame, irrigation: pd.DataFrame, nitrogen: pd.DataFrame, slot_var: pd.DataFrame) -> None:
    lines = [
        "# 042_13 SYA lowIC binary-timing PPO 25K 天气/胁迫响应性审计记录",
        "",
        "## 结论先说",
        "",
        f"- 分支：`{result['branch']}`。",
        f"- 解释：{result['branch_reason']}",
        f"- 10 个验证年份中，25K checkpoint 的动作签名数为 `{result['unique_action_signatures']}`。",
        "- 这一步不训练、不调参，只读取 042_11/042_12 已有结果。",
        "",
        "## 关键观察",
        "",
        f"- 灌溉事件槽位固定数：`{result['fixed_irrigation_slots']}` / `{result['irrigation_slot_count']}`。",
        f"- 施氮事件槽位固定数：`{result['fixed_nitrogen_slots']}` / `{result['nitrogen_slot_count']}`。",
        f"- 所有事件槽位最大 DAP 跨年范围：`{result['max_slot_dap_range']}` 天。",
        "- DAP1 灌溉、DAP2 施氮、DAP91 灌溉这类固定动作说明策略仍保留明显模板成分。",
        "- 中期灌溉/施氮 DAP 在不同年份之间会漂移，说明它不是完全静态模板。",
        "",
        "## 年度动作摘要",
        "",
        md_table(yearly, max_rows=20),
        "",
        "## 事件槽位变异性",
        "",
        md_table(slot_var, max_rows=20),
        "",
        "## 灌溉事件上下文",
        "",
        md_table(irrigation, max_rows=60),
        "",
        "## 施氮事件上下文",
        "",
        md_table(nitrogen, max_rows=60),
        "",
        "## 输出图表",
        "",
        *[f"- `{p}`" for p in result["figures"]],
        "",
        "## 结论边界",
        "",
        "- 25K 指标表现可以作为候选，但当前动作解释应表述为“部分天气/年份响应，仍含固定模板成分”。",
        "- 如果下一步要强化动作合理性，建议不要再只看产量/WP_ET/PFP_N，而要把动作响应性 guardrail 放入 checkpoint 选择或训练诊断。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ensure_dirs()
    eval_rows = load_eval_rows()
    yearly, irrigation, nitrogen = collect_event_context(eval_rows)
    irr_slots = slot_variability(irrigation, "irrigation")
    nit_slots = slot_variability(nitrogen, "nitrogen")
    slot_var = pd.concat([irr_slots, nit_slots], ignore_index=True)

    yearly_path = TAB_DIR / "042_13_binary_timing_ckpt25k_yearly_action_summary.csv"
    irrigation_path = TAB_DIR / "042_13_binary_timing_ckpt25k_irrigation_event_context.csv"
    nitrogen_path = TAB_DIR / "042_13_binary_timing_ckpt25k_nitrogen_event_context.csv"
    slot_path = TAB_DIR / "042_13_binary_timing_ckpt25k_event_slot_variability.csv"
    yearly.to_csv(yearly_path, index=False, encoding="utf-8-sig")
    irrigation.to_csv(irrigation_path, index=False, encoding="utf-8-sig")
    nitrogen.to_csv(nitrogen_path, index=False, encoding="utf-8-sig")
    slot_var.to_csv(slot_path, index=False, encoding="utf-8-sig")

    figures = []
    figures.extend(plot_event_daps(yearly, irrigation, nitrogen))
    figures.extend(plot_event_context(irrigation, nitrogen))

    branch, reason = classify_branch(slot_var, yearly)
    fixed_irr = int(irr_slots["fixed_across_years"].sum()) if not irr_slots.empty else 0
    fixed_nit = int(nit_slots["fixed_across_years"].sum()) if not nit_slots.empty else 0
    result = {
        "task": TASK,
        "checkpoint": CHECKPOINT,
        "branch": branch,
        "branch_reason": reason,
        "unique_action_signatures": int(yearly["action_sequence"].nunique()),
        "irrigation_slot_count": int(len(irr_slots)),
        "nitrogen_slot_count": int(len(nit_slots)),
        "fixed_irrigation_slots": fixed_irr,
        "fixed_nitrogen_slots": fixed_nit,
        "max_slot_dap_range": int(slot_var["dap_range"].max()) if not slot_var.empty else 0,
        "tables": [
            yearly_path.relative_to(ROOT).as_posix(),
            irrigation_path.relative_to(ROOT).as_posix(),
            nitrogen_path.relative_to(ROOT).as_posix(),
            slot_path.relative_to(ROOT).as_posix(),
        ],
        "figures": [p.relative_to(ROOT).as_posix() for p in figures],
        "record_md": DOC.relative_to(ROOT).as_posix(),
    }
    write_record(result, yearly, irrigation, nitrogen, slot_var)
    (OUT / "042_13_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

