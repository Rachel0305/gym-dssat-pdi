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
TASK = "045_02_sya_lowIC_04215_reporting_bars"
OUT = ROOT / "benchmark_results" / TASK
FIG_DIR = OUT / "figures"
TAB_DIR = OUT / "tables"
CFG_DIR = OUT / "configs"
DOC = ROOT / "docs" / f"{TASK}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK}.md"

SOURCE_DIR = ROOT / "benchmark_results" / "042_12_sya_lowIC_binary_timing_25k_candidate_audit" / "tables"
SUMMARY_SRC = SOURCE_DIR / "042_12_sya_lowIC_binary_timing_ckpt25k_five_scenario_metric_summary.csv"
GAPS_SRC = SOURCE_DIR / "042_12_sya_lowIC_binary_timing_ckpt25k_metric_gaps_vs_four_max.csv"

YEARS = list(range(2014, 2024))
SCENARIO_ORDER = [
    "null",
    "recorded_farmer_template",
    "dssat_auto",
    "official_extension_expert",
    "rl_candidate",
]
SCENARIO_LABELS = {
    "null": "Null",
    "recorded_farmer_template": "Recorded farmer",
    "dssat_auto": "DSSAT auto",
    "official_extension_expert": "Official expert",
    "rl_candidate": "PPO 042_15",
}
COLORS = {
    "null": "#555555",
    "recorded_farmer_template": "#C44E52",
    "dssat_auto": "#D8A305",
    "official_extension_expert": "#7E63B6",
    "rl_candidate": "#2A9D55",
}


def ensure_dirs() -> None:
    for path in [OUT, FIG_DIR, TAB_DIR, CFG_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    if PROMPT.exists():
        shutil.copy2(PROMPT, CFG_DIR / PROMPT.name)
    # Remove the previous 2-panel resource-only artifacts from this same
    # generated-output task so the folder contains only the current report
    # version requested by the user.
    for obsolete in [
        FIG_DIR / "045_02_sya_lowIC_04215_validation_five_scenario_management_inputs.png",
        FIG_DIR / "045_02_sya_lowIC_04215_validation_five_scenario_management_inputs.svg",
        TAB_DIR / "045_02_sya_lowIC_04215_resource_gaps_vs_four_baseline_max.csv",
    ]:
        if obsolete.exists():
            obsolete.unlink()


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


def load_summary() -> pd.DataFrame:
    if not SUMMARY_SRC.exists():
        raise FileNotFoundError(SUMMARY_SRC)
    df = pd.read_csv(SUMMARY_SRC, keep_default_na=False)
    expected_rows = len(YEARS) * len(SCENARIO_ORDER)
    if len(df) != expected_rows:
        raise RuntimeError(f"五情景汇总表行数异常：expected {expected_rows}, got {len(df)}")
    missing = []
    for year in YEARS:
        got = set(df.loc[pd.to_numeric(df["year"], errors="coerce").eq(year), "scenario"].astype(str))
        for scenario in SCENARIO_ORDER:
            if scenario not in got:
                missing.append((year, scenario))
    if missing:
        raise RuntimeError(f"五情景缺项：{missing[:10]}")
    numeric_cols = [
        "grain_yield_kg_ha",
        "WP_ET_kg_m3",
        "PFP_N_kg_kg",
        "actual_irrigation_mm",
        "actual_nitrogen_kg_ha",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    return df[df["year"].isin(YEARS) & df["scenario"].isin(SCENARIO_ORDER)].copy()


def build_yearly_gap_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for year, group in summary.groupby("year"):
        ppo = group[group["scenario"].eq("rl_candidate")]
        bases = group[~group["scenario"].eq("rl_candidate")]
        if ppo.empty or bases.empty:
            continue
        ppo = ppo.iloc[0]
        for metric, col, unit in [
            ("yield", "grain_yield_kg_ha", "kg/ha"),
            ("irrigation", "actual_irrigation_mm", "mm"),
            ("nitrogen", "actual_nitrogen_kg_ha", "kg/ha"),
        ]:
            vals = pd.to_numeric(bases[col], errors="coerce")
            max_val = float(vals.max()) if vals.notna().any() else np.nan
            max_scenario = str(bases.loc[vals.idxmax(), "scenario"]) if vals.notna().any() else ""
            ppo_val = float(ppo[col]) if pd.notna(ppo[col]) else np.nan
            rows.append(
                {
                    "year": int(year),
                    "metric": metric,
                    "unit": unit,
                    "ppo_value": ppo_val,
                    "four_baseline_max": max_val,
                    "four_baseline_max_scenario": max_scenario,
                    "ppo_minus_four_baseline_max": ppo_val - max_val if np.isfinite(ppo_val) and np.isfinite(max_val) else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TAB_DIR / "045_02_sya_lowIC_04215_yearly_gaps_vs_four_baseline_max.csv", index=False, encoding="utf-8-sig")
    return out


def build_site_mean_std_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for scenario in SCENARIO_ORDER:
        sub = summary[summary["scenario"].eq(scenario)].copy()
        for metric, col, unit in [
            ("yield", "grain_yield_kg_ha", "kg/ha"),
            ("WP_ET", "WP_ET_kg_m3", "kg/m3"),
            ("PFP_N", "PFP_N_kg_kg", "kg/kg"),
        ]:
            vals = pd.to_numeric(sub[col], errors="coerce")
            rows.append(
                {
                    "scenario": scenario,
                    "scenario_label": SCENARIO_LABELS[scenario],
                    "metric": metric,
                    "unit": unit,
                    "n_valid_years": int(vals.notna().sum()),
                    "mean": float(vals.mean()) if vals.notna().any() else np.nan,
                    "std": float(vals.std(ddof=1)) if vals.notna().sum() >= 2 else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TAB_DIR / "045_02_sya_lowIC_04215_site_mean_std_metrics.csv", index=False, encoding="utf-8-sig")
    return out


def copy_input_tables(summary: pd.DataFrame) -> None:
    summary.to_csv(TAB_DIR / "045_02_sya_lowIC_04215_five_scenario_metric_summary.csv", index=False, encoding="utf-8-sig")
    if GAPS_SRC.exists():
        shutil.copy2(GAPS_SRC, TAB_DIR / "045_02_sya_lowIC_04215_metric_gaps_vs_four_max.csv")


def add_ppo_delta_label(ax: plt.Axes, xpos: float, ypos: float, delta: float, unit: str) -> None:
    if not np.isfinite(delta):
        return
    sign = "+" if delta > 0 else ""
    text = f"Δ{sign}{delta:.0f}"
    ax.text(
        xpos,
        ypos,
        text,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#1F6F3A",
        rotation=0,
        clip_on=False,
    )


def plot_yearly_yield_resource_bars(summary: pd.DataFrame, yearly_gaps: pd.DataFrame) -> list[Path]:
    metrics = [
        ("grain_yield_kg_ha", "Grain yield (kg/ha)", "Yield", "yield"),
        ("actual_irrigation_mm", "Total irrigation (mm)", "Irrigation"),
        ("actual_nitrogen_kg_ha", "Total nitrogen (kg/ha)", "Nitrogen"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(15, 10.2), sharex=True)
    x = np.arange(len(YEARS))
    width = 0.16
    offsets = np.linspace(-2, 2, len(SCENARIO_ORDER)) * width
    for ax, spec in zip(axes, metrics):
        if len(spec) == 4:
            col, ylabel, title, metric_key = spec
        else:
            col, ylabel, title = spec
            metric_key = "irrigation" if "irrigation" in col else "nitrogen"
        max_y = 0.0
        for scenario, offset in zip(SCENARIO_ORDER, offsets):
            vals = []
            for year in YEARS:
                sub = summary[(summary["year"].eq(year)) & (summary["scenario"].eq(scenario))]
                vals.append(float(sub[col].iloc[0]) if len(sub) and pd.notna(sub[col].iloc[0]) else np.nan)
            max_y = max(max_y, np.nanmax(vals) if len(vals) else 0.0)
            bars = ax.bar(
                x + offset,
                vals,
                width,
                label=SCENARIO_LABELS[scenario],
                color=COLORS[scenario],
                edgecolor="white",
                linewidth=0.4,
            )
            if scenario == "rl_candidate":
                for i, (year, bar) in enumerate(zip(YEARS, bars)):
                    row = yearly_gaps[
                        yearly_gaps["year"].eq(year) & yearly_gaps["metric"].eq(metric_key)
                    ]
                    if len(row):
                        delta = float(row["ppo_minus_four_baseline_max"].iloc[0])
                        add_ppo_delta_label(ax, x[i] + offset, bar.get_height() + max_y * 0.025, delta, "")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=0.22)
        ax.set_ylim(0, max_y * 1.18 if max_y > 0 else 1)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([str(y) for y in YEARS])
    axes[0].legend(ncol=5, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.24))
    fig.suptitle(
        "SYA lowIC 042_15 validation years: five-scenario yield and management input comparison",
        x=0.01,
        y=0.995,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.945,
        "Green labels on PPO bars show PPO minus the maximum value among the four non-RL baselines; positive is better for yield, negative means lower resource input.",
        ha="left",
        va="top",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    png = FIG_DIR / "045_02_sya_lowIC_04215_validation_five_scenario_yield_management_inputs.png"
    svg = png.with_suffix(".svg")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return [png, svg]


def plot_site_mean_std(site_stats: pd.DataFrame) -> list[Path]:
    metrics = [
        ("yield", "Grain yield (kg/ha)", "Yield"),
        ("WP_ET", "WP_ET (kg/m3)", "Water productivity"),
        ("PFP_N", "PFP_N (kg/kg)", "Nitrogen productivity"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    x = np.arange(len(SCENARIO_ORDER))
    for ax, (metric, ylabel, title) in zip(axes, metrics):
        sub = site_stats[site_stats["metric"].eq(metric)].set_index("scenario").loc[SCENARIO_ORDER].reset_index()
        vals = pd.to_numeric(sub["mean"], errors="coerce").to_numpy(dtype=float)
        errs = pd.to_numeric(sub["std"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        bars = ax.bar(
            x,
            vals,
            yerr=errs,
            capsize=4,
            color=[COLORS[s] for s in SCENARIO_ORDER],
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIO_ORDER], rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.22)
        ymax = float(np.nanmax(vals + errs)) if len(vals) else 1.0
        ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1.0)
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.025, f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[s]) for s in SCENARIO_ORDER]
    fig.legend(handles, [SCENARIO_LABELS[s] for s in SCENARIO_ORDER], ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.03), fontsize=8)
    fig.suptitle(
        "SYA lowIC 042_15 validation years: five-scenario mean ± SD across 2014–2023",
        x=0.01,
        y=0.995,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    png = FIG_DIR / "045_02_sya_lowIC_04215_site_mean_sd_yield_wp_pfp.png"
    svg = png.with_suffix(".svg")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return [png, svg]


def write_record(
    summary: pd.DataFrame,
    yearly_gaps: pd.DataFrame,
    site_stats: pd.DataFrame,
    figures: list[Path],
) -> None:
    gaps = pd.read_csv(GAPS_SRC, keep_default_na=False) if GAPS_SRC.exists() else pd.DataFrame()
    for col in [
        "win_yield_vs_four_max",
        "win_wp_et_vs_four_max",
        "win_pfp_n_vs_four_max",
        "any_metric_win_four",
    ]:
        if col in gaps:
            gaps[col] = gaps[col].astype(str).str.lower().isin(["true", "1", "yes"])
    lines = [
        "# 045_02 SYA lowIC 042_15 汇报补图记录",
        "",
        "## 结论先说",
        "",
        "- 本任务只补图，不训练模型、不改 reward、不改 DSSAT 输入、不重跑 DSSAT。",
        "- 使用冻结候选：042_15 / 042_11 binary-timing MaskablePPO 25K checkpoint。",
        "- 已输出两张汇报图：逐年产量与水肥投入五情景柱状图、站点五情景产量/WP_ET/PFP_N 均值±标准差图。",
        "",
        "## 042_15 指标胜出计数",
        "",
        f"- 验证年份：{len(gaps)} 年，2014–2023。",
        f"- 至少一项指标超过四基线最高值：{int(gaps['any_metric_win_four'].sum()) if 'any_metric_win_four' in gaps else 'NA'} / {len(gaps)}。",
        f"- 产量超过四基线最高值：{int(gaps['win_yield_vs_four_max'].sum()) if 'win_yield_vs_four_max' in gaps else 'NA'} / {len(gaps)}。",
        f"- WP_ET 超过四基线最高值：{int(gaps['win_wp_et_vs_four_max'].sum()) if 'win_wp_et_vs_four_max' in gaps else 'NA'} / {len(gaps)}。",
        f"- PFP_N 超过四基线最高值：{int(gaps['win_pfp_n_vs_four_max'].sum()) if 'win_pfp_n_vs_four_max' in gaps else 'NA'} / {len(gaps)}。",
        "",
        "## 逐年产量和水肥投入差值表",
        "",
        md_table(yearly_gaps, 40),
        "",
        "## 五情景均值 ± 标准差表",
        "",
        md_table(site_stats, 30),
        "",
        "## 输出图",
        "",
        *[f"- `{p.relative_to(ROOT).as_posix()}`" for p in figures],
        "",
        "## 数据来源",
        "",
        f"- `{SUMMARY_SRC.relative_to(ROOT).as_posix()}`",
        f"- `{GAPS_SRC.relative_to(ROOT).as_posix()}`",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ensure_dirs()
    summary = load_summary()
    copy_input_tables(summary)
    yearly_gaps = build_yearly_gap_table(summary)
    site_stats = build_site_mean_std_table(summary)
    figures: list[Path] = []
    figures += plot_yearly_yield_resource_bars(summary, yearly_gaps)
    figures += plot_site_mean_std(site_stats)
    write_record(summary, yearly_gaps, site_stats, figures)
    result = {
        "task": TASK,
        "source": SUMMARY_SRC.relative_to(ROOT).as_posix(),
        "years": YEARS,
        "figures": [p.relative_to(ROOT).as_posix() for p in figures],
        "tables": [
            (TAB_DIR / "045_02_sya_lowIC_04215_five_scenario_metric_summary.csv").relative_to(ROOT).as_posix(),
            (TAB_DIR / "045_02_sya_lowIC_04215_metric_gaps_vs_four_max.csv").relative_to(ROOT).as_posix(),
            (TAB_DIR / "045_02_sya_lowIC_04215_yearly_gaps_vs_four_baseline_max.csv").relative_to(ROOT).as_posix(),
            (TAB_DIR / "045_02_sya_lowIC_04215_site_mean_std_metrics.csv").relative_to(ROOT).as_posix(),
        ],
        "record_md": DOC.relative_to(ROOT).as_posix(),
    }
    (OUT / "045_02_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
