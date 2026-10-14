from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "031_38_five_station_ppo_water_n_saving_summary"
DOC = ROOT / "docs" / "031_38_five_station_ppo_water_n_saving_summary_record.md"

FOUR_SITE_ALL = ROOT / "benchmark_results" / "031_37_ppo_water_n_saving_summary" / "tables" / "031_37_all_seed_ppo_deltas_vs_official.csv"
FOUR_SITE_SELECTED = ROOT / "benchmark_results" / "031_37_ppo_water_n_saving_summary" / "tables" / "031_37_selected_station_year_ppo_deltas_vs_official.csv"
SY_ALL = ROOT / "benchmark_results" / "031_30_sy_ppo_vs_completed_baselines" / "evaluation" / "031_30_candidate_vs_baseline_envelope.csv"


def ensure_dirs() -> None:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
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


def numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def normalize_sy_all() -> pd.DataFrame:
    sy = pd.read_csv(SY_ALL, keep_default_na=False)
    sy = numeric(
        sy,
        [
            "year",
            "seed",
            "checkpoint_step",
            "final_grain_kg_ha",
            "irrigation_event_total_mm",
            "nitrogen_event_total_kg_ha",
            "total_irrigation",
            "total_n",
            "profit_simple",
            "wp_et_kg_m3",
            "pfp_n_kg_kg",
            "official_expert_irrigation",
            "official_expert_n",
            "max_baseline_yield",
            "max_baseline_wp_et",
            "max_baseline_pfp_n",
            "yield_delta_vs_best_baseline",
            "wp_et_delta_vs_best_baseline",
            "pfp_n_delta_vs_best_baseline",
            "water_saving_vs_official_expert",
            "n_saving_vs_official_expert",
        ],
    )
    for source, target in [
        ("yield_winner", "yield_strict_win"),
        ("wp_et_winner", "wp_et_strict_win"),
        ("pfp_n_winner", "pfp_n_strict_win"),
        ("any_metric_winner", "advisor_any_metric_strict_winner"),
    ]:
        sy[target] = as_bool(sy[source]) if source in sy.columns else False

    sy["metric_win_count"] = sy[["yield_strict_win", "wp_et_strict_win", "pfp_n_strict_win"]].sum(axis=1)
    sy["baseline_comparison_status"] = np.where(sy.get("baseline_count", 0).astype(float) > 0, "ok", "missing")
    sy["baseline_row_count"] = sy.get("baseline_count", np.nan)
    sy["baseline_max_yield"] = sy.get("max_baseline_yield", np.nan)
    sy["baseline_max_wp_et"] = sy.get("max_baseline_wp_et", np.nan)
    sy["baseline_max_pfp_n"] = sy.get("max_baseline_pfp_n", np.nan)
    sy["gap_yield"] = sy.get("yield_delta_vs_best_baseline", np.nan)
    sy["gap_wp_et"] = sy.get("wp_et_delta_vs_best_baseline", np.nan)
    sy["gap_pfp_n"] = sy.get("pfp_n_delta_vs_best_baseline", np.nan)
    sy["official_irrigation_mm"] = sy.get("official_expert_irrigation", np.nan)
    sy["official_nitrogen_kg_ha"] = sy.get("official_expert_n", np.nan)
    sy["saved_irrigation_vs_official_mm"] = sy.get("water_saving_vs_official_expert", np.nan)
    sy["saved_nitrogen_vs_official_kg_ha"] = sy.get("n_saving_vs_official_expert", np.nan)
    sy["delta_irrigation_vs_official_mm"] = -sy["saved_irrigation_vs_official_mm"]
    sy["delta_nitrogen_vs_official_kg_ha"] = -sy["saved_nitrogen_vs_official_kg_ha"]
    sy["delta_yield_vs_official_kg_ha"] = np.nan
    sy["delta_wp_et_vs_official_kg_m3"] = np.nan
    sy["delta_pfp_n_vs_official_kg_kg"] = np.nan
    sy["winning_metrics"] = sy.apply(
        lambda r: ",".join(
            [
                name
                for name, col in [
                    ("yield", "yield_strict_win"),
                    ("WP_ET", "wp_et_strict_win"),
                    ("PFP_N", "pfp_n_strict_win"),
                ]
                if bool(r[col])
            ]
        ),
        axis=1,
    )
    return sy


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


def load_merged() -> tuple[pd.DataFrame, pd.DataFrame]:
    four_all = pd.read_csv(FOUR_SITE_ALL, keep_default_na=False)
    four_selected = pd.read_csv(FOUR_SITE_SELECTED, keep_default_na=False)
    sy_all = normalize_sy_all()

    for df in [four_all, four_selected, sy_all]:
        for col in ["yield_strict_win", "wp_et_strict_win", "pfp_n_strict_win", "advisor_any_metric_strict_winner"]:
            if col in df.columns:
                df[col] = as_bool(df[col])
        df["station_code"] = df["station_code"].astype(str)

    sy_selected = select_candidate_per_year(sy_all)
    all_cols = sorted(set(four_all.columns).union(sy_all.columns))
    selected_cols = sorted(set(four_selected.columns).union(sy_selected.columns))
    all_seed = pd.concat([four_all.reindex(columns=all_cols), sy_all.reindex(columns=all_cols)], ignore_index=True)
    selected = pd.concat([four_selected.reindex(columns=selected_cols), sy_selected.reindex(columns=selected_cols)], ignore_index=True)

    for df in [all_seed, selected]:
        df = numeric(df, [])
    return all_seed, selected


def station_summary(selected: pd.DataFrame, all_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for station, group in selected.groupby("station_code", sort=True):
        all_group = all_seed[all_seed["station_code"].eq(station)]
        i_save = pd.to_numeric(group["saved_irrigation_vs_official_mm"], errors="coerce")
        n_save = pd.to_numeric(group["saved_nitrogen_vs_official_kg_ha"], errors="coerce")
        rows.append(
            {
                "station_code": station,
                "years": int(group["year"].nunique()),
                "selected_any_metric_win_years": int(group["advisor_any_metric_strict_winner"].sum()),
                "selected_yield_win_years": int(group["yield_strict_win"].sum()),
                "selected_wp_et_win_years": int(group["wp_et_strict_win"].sum()),
                "selected_pfp_n_win_years": int(group["pfp_n_strict_win"].sum()),
                "mean_irrigation_saving_vs_official_mm": float(i_save.mean()),
                "sd_irrigation_saving_vs_official_mm": float(i_save.std(ddof=1)),
                "min_irrigation_saving_vs_official_mm": float(i_save.min()),
                "max_irrigation_saving_vs_official_mm": float(i_save.max()),
                "mean_nitrogen_saving_vs_official_kg_ha": float(n_save.mean()),
                "sd_nitrogen_saving_vs_official_kg_ha": float(n_save.std(ddof=1)),
                "min_nitrogen_saving_vs_official_kg_ha": float(n_save.min()),
                "max_nitrogen_saving_vs_official_kg_ha": float(n_save.max()),
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
    data["year"] = pd.to_numeric(data["year"], errors="coerce").astype(int)
    for col in ["saved_irrigation_vs_official_mm", "saved_nitrogen_vs_official_kg_ha"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    x = np.arange(len(data))
    years = data["year"].astype(str).tolist()
    colors_i = np.where(data["saved_irrigation_vs_official_mm"] >= 0, "#2F6BBA", "#C4554D")
    colors_n = np.where(data["saved_nitrogen_vs_official_kg_ha"] >= 0, "#A66A00", "#C4554D")
    mean_i = data["saved_irrigation_vs_official_mm"].mean()
    sd_i = data["saved_irrigation_vs_official_mm"].std(ddof=1)
    mean_n = data["saved_nitrogen_vs_official_kg_ha"].mean()
    sd_n = data["saved_nitrogen_vs_official_kg_ha"].std(ddof=1)

    fig, axes = plt.subplots(2, 1, figsize=(13.2, 7.0), sharex=True)
    fig.suptitle(f"{station} PPO selected candidates: water and nitrogen saving vs official expert", fontsize=13, y=0.985)
    subtitle = f"Irrigation saving mean±SD {mean_i:.1f}±{sd_i:.1f} mm; N saving mean±SD {mean_n:.1f}±{sd_n:.1f} kg/ha. Green dots: at least one strict four-baseline metric win."
    fig.text(0.5, 0.945, subtitle, ha="center", va="top", fontsize=9, color="#555555")

    specs = [
        (axes[0], "saved_irrigation_vs_official_mm", colors_i, "Irrigation saving\n(mm)"),
        (axes[1], "saved_nitrogen_vs_official_kg_ha", colors_n, "Nitrogen saving\n(kg/ha)"),
    ]
    for ax, value_col, colors, ylabel in specs:
        vals = data[value_col].to_numpy(dtype=float)
        ax.bar(x, vals, color=colors, edgecolor="#333333", linewidth=0.3)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
        span = max(abs(np.nanmin(vals)), abs(np.nanmax(vals)), 1.0)
        marker_y = np.where(vals >= 0, vals + 0.06 * span, vals - 0.08 * span)
        winners = data["advisor_any_metric_strict_winner"].to_numpy(dtype=bool)
        ax.scatter(x[winners], marker_y[winners], marker="o", s=42, facecolor="#2E8B57", edgecolor="white", linewidth=0.6, zorder=3, label="Any strict metric win")
        ax.scatter(x[~winners], marker_y[~winners], marker="x", s=38, color="#666666", linewidth=1.0, zorder=3, label="No strict metric win")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(years, rotation=45, ha="right", fontsize=8)
    axes[1].set_xlabel("Target year")
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.06, 1, 0.925])
    path = OUT / "figures" / f"031_38_{station.lower()}_ppo_water_n_saving_vs_official"
    savefig(fig, path)
    return path.with_suffix(".png")


def plot_overall(summary: pd.DataFrame) -> Path:
    data = summary.sort_values("station_code").copy()
    y = np.arange(len(data))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), sharey=True)
    fig.suptitle("Five-station PPO water and nitrogen saving vs official expert", fontsize=13, y=0.985, fontweight="bold")
    fig.text(0.5, 0.935, "Bars show mean selected-candidate saving; whiskers show ±1 SD across target years.", ha="center", fontsize=9, color="#555555")
    specs = [
        ("mean_irrigation_saving_vs_official_mm", "sd_irrigation_saving_vs_official_mm", "Irrigation saving (mm)", "#2F6BBA"),
        ("mean_nitrogen_saving_vs_official_kg_ha", "sd_nitrogen_saving_vs_official_kg_ha", "Nitrogen saving (kg/ha)", "#A66A00"),
    ]
    for ax, (mean_col, sd_col, label, color) in zip(axes, specs):
        means = data[mean_col].to_numpy(dtype=float)
        sds = data[sd_col].to_numpy(dtype=float)
        bar_colors = np.where(means >= 0, color, "#C4554D")
        ax.barh(y, means, color=bar_colors, edgecolor="#333333", linewidth=0.4)
        ax.errorbar(means, y, xerr=sds, fmt="none", ecolor="#333333", elinewidth=1.0, capsize=3)
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_xlabel(label)
        ax.grid(axis="x", color="#E5E5E5", linewidth=0.7)
        for yi, val in zip(y, means):
            ax.text(
                val + (2 if val >= 0 else -2),
                yi,
                f"{val:+.1f}",
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=8,
            )
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(data["station_code"].tolist())
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    path = OUT / "figures" / "031_38_overall_ppo_water_n_saving_vs_official"
    savefig(fig, path)
    return path.with_suffix(".png")


def main() -> None:
    ensure_dirs()
    setup_style()
    all_seed, selected = load_merged()
    summary = station_summary(selected, all_seed)

    all_seed.to_csv(OUT / "tables" / "031_38_all_seed_ppo_deltas_vs_official.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT / "tables" / "031_38_selected_station_year_ppo_deltas_vs_official.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "tables" / "031_38_station_water_n_saving_summary.csv", index=False, encoding="utf-8-sig")

    station_paths = [plot_station(station, selected) for station in sorted(selected["station_code"].unique())]
    overall_path = plot_overall(summary)

    lines = [
        "# 031_38 five-station PPO water/nitrogen saving summary record",
        "",
        "## Scope",
        "",
        "- No training.",
        "- No DSSAT rerun.",
        "- Adds SYA results from 031_30 to the 031_37 four-station summary.",
        "- Water/nitrogen saving is relative to official expert.",
        "- Metric-win markers are evaluated against the four-baseline envelope.",
        "",
        "## Source tables",
        "",
        f"- `{FOUR_SITE_ALL.relative_to(ROOT)}`",
        f"- `{FOUR_SITE_SELECTED.relative_to(ROOT)}`",
        f"- `{SY_ALL.relative_to(ROOT)}`",
        "",
        "## Candidate display selection",
        "",
        "One selected candidate per station-year is chosen by metric win count, any-win flag, profit, yield, lower irrigation, lower nitrogen, then lower seed id.",
        "",
        "## Station summary",
        "",
        md_table(summary, 20),
        "",
        "## Outputs",
        "",
        md_table(pd.DataFrame({"figure": [str(p.relative_to(ROOT)).replace('\\\\', '/') for p in [*station_paths, overall_path]]}), 20),
        "",
        "## Interpretation boundary",
        "",
        "The saving baseline is official expert because null and DSSAT auto may use zero water or zero nitrogen. Four-baseline maxima remain the correct benchmark for yield/WP_ET/PFP_N metric wins.",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / DOC.name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "task": "031_38",
                "all_seed_rows": int(len(all_seed)),
                "selected_rows": int(len(selected)),
                "stations": sorted(selected["station_code"].unique().tolist()),
                "station_summary": summary.to_dict(orient="records"),
                "figures": [str(p.relative_to(ROOT)).replace("\\", "/") for p in [*station_paths, overall_path]],
                "record": str(DOC.relative_to(ROOT)).replace("\\", "/"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
