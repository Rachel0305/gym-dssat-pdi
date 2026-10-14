from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as siteppo


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "031_37_ppo_water_n_saving_summary"
DOC = ROOT / "docs" / "031_37_ppo_water_n_saving_summary_record.md"
COMP = ROOT / "benchmark_results" / "031_36_missing_dssat_auto_completion_for_03134" / "evaluation" / "031_36_full_ppo_vs_completed_template_aware_four_baselines.csv"
BASE = ROOT / "benchmark_results" / "031_36_missing_dssat_auto_completion_for_03134" / "evaluation" / "031_36_full_completed_template_aware_unified_baseline_summary.csv"


def ensure_dirs() -> None:
    for rel in ["figures", "tables"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "No rows."
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(3)
    work = work.astype(object).where(pd.notna(work), "")
    header = "| " + " | ".join(map(str, work.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(map(str, values)) + " |" for values in work.to_numpy().tolist()]
    return "\n".join([header, sep, *rows])


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    comp = pd.read_csv(COMP, keep_default_na=False)
    base = pd.read_csv(BASE, keep_default_na=False)
    for df, cols in [
        (comp, ["year", "seed", "checkpoint_step", "final_grain_kg_ha", "irrigation_event_total_mm", "nitrogen_event_total_kg_ha", "wp_et_kg_m3", "pfp_n_kg_kg", "profit_simple"]),
        (base, ["year", "grain_yield_kg_ha", "actual_irrigation_mm", "actual_nitrogen_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg"]),
    ]:
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["yield_strict_win", "wp_et_strict_win", "pfp_n_strict_win", "advisor_any_metric_strict_winner"]:
        if col in comp.columns:
            comp[col] = as_bool(comp[col])
    return comp, base


def official_expert_baseline(base: pd.DataFrame) -> pd.DataFrame:
    off = base[base["scenario"].astype(str).eq("official_extension_expert")].copy()
    keep = [
        "station_code",
        "site",
        "year",
        "grain_yield_kg_ha",
        "actual_irrigation_mm",
        "actual_nitrogen_kg_ha",
        "WP_ET_kg_m3",
        "PFP_N_kg_kg",
    ]
    off = off[[c for c in keep if c in off.columns]].drop_duplicates(["station_code", "year"])
    off = fill_official_missing_management_from_summary(out=off)
    return off.rename(
        columns={
            "grain_yield_kg_ha": "official_yield_kg_ha",
            "actual_irrigation_mm": "official_irrigation_mm",
            "actual_nitrogen_kg_ha": "official_nitrogen_kg_ha",
            "WP_ET_kg_m3": "official_wp_et_kg_m3",
            "PFP_N_kg_kg": "official_pfp_n_kg_kg",
        }
    )


def official_summary_path(station_code: str, site: str, year: int) -> Path | None:
    if site == "FQ":
        return ROOT / "benchmark_results" / "028_07_missing_official_expert_six_season_completion" / "runs_completed" / f"FQ{year}" / "official_extension_expert" / "pdi_tmp_snapshot_eval" / "Summary.OUT"
    if site == "YC":
        return ROOT / "benchmark_results" / "028_07_missing_official_expert_six_season_completion" / "runs_completed" / f"YC{year}" / "official_extension_expert" / "pdi_tmp_snapshot_eval" / "Summary.OUT"
    if site == "HLA":
        return ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla_five_scenario_nstep_020_11" / "runs" / str(year) / "extension_expert" / "pdi_tmp_snapshot_eval" / "Summary.OUT"
    if site == "LC":
        return ROOT / "benchmark_results" / "027_07_site_specific_stage_maskable_ppo_attempt2" / "LC" / "readiness" / "baseline_runs" / "official_extension_expert" / "pdi_tmp_snapshot_eval" / "Summary.OUT"
    return None


def summary_management_totals(summary_out: Path, target_yield: float) -> tuple[float, float] | None:
    if not summary_out.exists():
        return None
    rows = siteppo.parse_summary_out(summary_out)
    candidates: list[tuple[float, int, dict[str, Any]]] = []
    for idx, row in enumerate(rows):
        hwam = siteppo.num(row, "HWAM")
        if hwam is None:
            continue
        candidates.append((abs(float(hwam) - float(target_yield)), -idx, row))
    if not candidates:
        return None
    _, _, row = min(candidates, key=lambda item: (item[0], item[1]))
    return float(siteppo.num(row, "IRCM") or 0.0), float(siteppo.num(row, "NICM") or 0.0)


def fill_official_missing_management_from_summary(out: pd.DataFrame) -> pd.DataFrame:
    out = out.copy()
    for idx, row in out.iterrows():
        i = pd.to_numeric(row.get("actual_irrigation_mm"), errors="coerce")
        n = pd.to_numeric(row.get("actual_nitrogen_kg_ha"), errors="coerce")
        if pd.notna(i) and pd.notna(n):
            continue
        path = official_summary_path(str(row.get("station_code")), str(row.get("site")), int(row.get("year")))
        if path is None:
            continue
        totals = summary_management_totals(path, float(pd.to_numeric(row.get("grain_yield_kg_ha"), errors="coerce")))
        if totals is None:
            continue
        out.loc[idx, "actual_irrigation_mm"] = totals[0]
        out.loc[idx, "actual_nitrogen_kg_ha"] = totals[1]
        out.loc[idx, "official_management_fill_source"] = str(path.relative_to(ROOT)).replace("\\", "/")
    return out


def make_all_seed_deltas(comp: pd.DataFrame, off: pd.DataFrame) -> pd.DataFrame:
    df = comp.merge(off, on=["station_code", "site", "year"], how="left", validate="many_to_one")
    df["metric_win_count"] = df[["yield_strict_win", "wp_et_strict_win", "pfp_n_strict_win"]].sum(axis=1)
    df["delta_irrigation_vs_official_mm"] = df["irrigation_event_total_mm"] - df["official_irrigation_mm"]
    df["delta_nitrogen_vs_official_kg_ha"] = df["nitrogen_event_total_kg_ha"] - df["official_nitrogen_kg_ha"]
    df["saved_irrigation_vs_official_mm"] = -df["delta_irrigation_vs_official_mm"]
    df["saved_nitrogen_vs_official_kg_ha"] = -df["delta_nitrogen_vs_official_kg_ha"]
    df["delta_yield_vs_official_kg_ha"] = df["final_grain_kg_ha"] - df["official_yield_kg_ha"]
    df["delta_wp_et_vs_official_kg_m3"] = df["wp_et_kg_m3"] - df["official_wp_et_kg_m3"]
    df["delta_pfp_n_vs_official_kg_kg"] = df["pfp_n_kg_kg"] - df["official_pfp_n_kg_kg"]
    return df


def select_candidate_per_year(all_seed: pd.DataFrame) -> pd.DataFrame:
    work = all_seed.copy()
    work["_advisor_rank"] = work["advisor_any_metric_strict_winner"].astype(int)
    work = work.sort_values(
        [
            "station_code",
            "year",
            "metric_win_count",
            "_advisor_rank",
            "profit_simple",
            "final_grain_kg_ha",
            "irrigation_event_total_mm",
            "nitrogen_event_total_kg_ha",
            "seed",
        ],
        ascending=[True, True, False, False, False, False, True, True, True],
    )
    return work.drop_duplicates(["station_code", "year"], keep="first").drop(columns=["_advisor_rank"])


def station_summary(selected: pd.DataFrame, all_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for station, group in selected.groupby("station_code"):
        all_group = all_seed[all_seed["station_code"].eq(station)]
        rows.append(
            {
                "station_code": station,
                "years": int(group["year"].nunique()),
                "selected_any_metric_win_years": int(group["advisor_any_metric_strict_winner"].sum()),
                "selected_yield_win_years": int(group["yield_strict_win"].sum()),
                "selected_wp_et_win_years": int(group["wp_et_strict_win"].sum()),
                "selected_pfp_n_win_years": int(group["pfp_n_strict_win"].sum()),
                "mean_irrigation_saving_vs_official_mm": float(group["saved_irrigation_vs_official_mm"].mean()),
                "median_irrigation_saving_vs_official_mm": float(group["saved_irrigation_vs_official_mm"].median()),
                "min_irrigation_saving_vs_official_mm": float(group["saved_irrigation_vs_official_mm"].min()),
                "max_irrigation_saving_vs_official_mm": float(group["saved_irrigation_vs_official_mm"].max()),
                "mean_nitrogen_saving_vs_official_kg_ha": float(group["saved_nitrogen_vs_official_kg_ha"].mean()),
                "median_nitrogen_saving_vs_official_kg_ha": float(group["saved_nitrogen_vs_official_kg_ha"].median()),
                "min_nitrogen_saving_vs_official_kg_ha": float(group["saved_nitrogen_vs_official_kg_ha"].min()),
                "max_nitrogen_saving_vs_official_kg_ha": float(group["saved_nitrogen_vs_official_kg_ha"].max()),
                "all_seed_rows": int(len(all_group)),
                "all_seed_any_metric_wins": int(all_group["advisor_any_metric_strict_winner"].sum()),
            }
        )
    return pd.DataFrame(rows)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "axes.titleweight": "bold",
            "figure.dpi": 160,
            "savefig.dpi": 220,
        }
    )


def savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_station(station: str, selected: pd.DataFrame) -> Path:
    data = selected[selected["station_code"].eq(station)].sort_values("year").copy()
    x = np.arange(len(data))
    years = data["year"].astype(int).astype(str).tolist()
    colors_i = np.where(data["saved_irrigation_vs_official_mm"] >= 0, "#2F6BBA", "#C4554D")
    colors_n = np.where(data["saved_nitrogen_vs_official_kg_ha"] >= 0, "#A66A00", "#C4554D")
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 6.8), sharex=True)
    fig.suptitle(f"{station} PPO selected candidates: water and nitrogen saving vs official expert", fontsize=13, y=0.98)
    subtitle = "Negative saving means PPO used more input than official expert; green markers indicate at least one strict four-baseline metric win."
    fig.text(0.5, 0.94, subtitle, ha="center", va="top", fontsize=9, color="#555555")
    axes[0].bar(x, data["saved_irrigation_vs_official_mm"], color=colors_i, edgecolor="#333333", linewidth=0.3)
    axes[0].axhline(0, color="#333333", linewidth=0.8)
    axes[0].set_ylabel("Irrigation saving\n(mm)")
    axes[0].grid(axis="y", color="#E5E5E5", linewidth=0.7)
    axes[1].bar(x, data["saved_nitrogen_vs_official_kg_ha"], color=colors_n, edgecolor="#333333", linewidth=0.3)
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_ylabel("Nitrogen saving\n(kg/ha)")
    axes[1].grid(axis="y", color="#E5E5E5", linewidth=0.7)
    for ax, value_col in [(axes[0], "saved_irrigation_vs_official_mm"), (axes[1], "saved_nitrogen_vs_official_kg_ha")]:
        yvals = data[value_col].to_numpy(dtype=float)
        span = max(abs(np.nanmin(yvals)), abs(np.nanmax(yvals)), 1.0)
        marker_y = np.where(yvals >= 0, yvals + 0.06 * span, yvals - 0.08 * span)
        winners = data["advisor_any_metric_strict_winner"].to_numpy(dtype=bool)
        ax.scatter(x[winners], marker_y[winners], marker="o", s=42, facecolor="#2E8B57", edgecolor="white", linewidth=0.6, zorder=3, label="Any strict metric win")
        ax.scatter(x[~winners], marker_y[~winners], marker="x", s=38, color="#666666", linewidth=1.0, zorder=3, label="No strict metric win")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(years, rotation=45, ha="right", fontsize=8)
    axes[1].set_xlabel("Target year")
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    path = OUT / "figures" / f"031_37_{station.lower()}_ppo_water_n_saving_vs_official"
    savefig(fig, path)
    return path.with_suffix(".png")


def plot_overall(summary: pd.DataFrame) -> Path:
    data = summary.sort_values("station_code").copy()
    y = np.arange(len(data))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    fig.suptitle("PPO water and nitrogen saving vs official expert by station", fontsize=13, y=0.98, fontweight="bold")
    fig.text(0.5, 0.93, "Bars show mean selected-candidate saving; whiskers show min-max across target years.", ha="center", fontsize=9, color="#555555")
    specs = [
        ("mean_irrigation_saving_vs_official_mm", "min_irrigation_saving_vs_official_mm", "max_irrigation_saving_vs_official_mm", "Irrigation saving (mm)", "#2F6BBA"),
        ("mean_nitrogen_saving_vs_official_kg_ha", "min_nitrogen_saving_vs_official_kg_ha", "max_nitrogen_saving_vs_official_kg_ha", "Nitrogen saving (kg/ha)", "#A66A00"),
    ]
    for ax, (mean_col, min_col, max_col, label, color) in zip(axes, specs):
        means = data[mean_col].to_numpy(dtype=float)
        lower = means - data[min_col].to_numpy(dtype=float)
        upper = data[max_col].to_numpy(dtype=float) - means
        bar_colors = np.where(means >= 0, color, "#C4554D")
        ax.barh(y, means, color=bar_colors, edgecolor="#333333", linewidth=0.4)
        ax.errorbar(means, y, xerr=[lower, upper], fmt="none", ecolor="#333333", elinewidth=1.0, capsize=3)
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_xlabel(label)
        ax.grid(axis="x", color="#E5E5E5", linewidth=0.7)
        for yi, val in zip(y, means):
            ax.text(val + (2 if val >= 0 else -2), yi, f"{val:.1f}", va="center", ha="left" if val >= 0 else "right", fontsize=8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(data["station_code"].tolist())
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    path = OUT / "figures" / "031_37_overall_ppo_water_n_saving_vs_official"
    savefig(fig, path)
    return path.with_suffix(".png")


def main() -> None:
    ensure_dirs()
    setup_style()
    comp, base = load_data()
    off = official_expert_baseline(base)
    all_seed = make_all_seed_deltas(comp, off)
    selected = select_candidate_per_year(all_seed)
    summary = station_summary(selected, all_seed)
    all_seed.to_csv(OUT / "tables" / "031_37_all_seed_ppo_deltas_vs_official.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT / "tables" / "031_37_selected_station_year_ppo_deltas_vs_official.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "tables" / "031_37_station_water_n_saving_summary.csv", index=False, encoding="utf-8-sig")
    figure_paths = [plot_station(st, selected) for st in sorted(selected["station_code"].unique())]
    overall_path = plot_overall(summary)
    lines = [
        "# 031_37 PPO water/nitrogen saving summary record",
        "",
        "## Scope",
        "",
        "- No PPO/DQN training.",
        "- No DSSAT rerun.",
        "- Read 031_36 completed four-baseline comparison table and completed template-aware baseline table.",
        "- Main saving baseline is `official_extension_expert`; four-baseline metric wins still use the 031_36 strict winner flags.",
        "",
        "## Candidate selection rule",
        "",
        "One display candidate per station-year was selected by: metric win count, advisor-any-win flag, profit, yield, lower irrigation, lower nitrogen, lower seed.",
        "",
        "## Station summary",
        "",
        md_table(summary, 80),
        "",
        "## Figure outputs",
        "",
        md_table(pd.DataFrame({"figure": [str(p.relative_to(ROOT)).replace('\\\\', '/') for p in [*figure_paths, overall_path]]}), 20),
        "",
        "## Interpretation boundary",
        "",
        "These plots summarize input saving relative to official expert. They do not by themselves prove the causal necessity of individual PPO actions.",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / DOC.name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "task": "031_37",
                "selected_rows": int(len(selected)),
                "all_seed_rows": int(len(all_seed)),
                "station_figures": [str(p.relative_to(ROOT)).replace("\\", "/") for p in figure_paths],
                "overall_figure": str(overall_path.relative_to(ROOT)).replace("\\", "/"),
                "record_md": str(DOC.relative_to(ROOT)).replace("\\", "/"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
