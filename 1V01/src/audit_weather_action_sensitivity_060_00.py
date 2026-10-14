"""060_00: weather/action sensitivity audit for old PPO vs forecast PPO.

This is a read-only audit over existing daily evaluation CSVs. It does not
train, replay DSSAT, or load PPO checkpoints. The purpose is to test whether
the learned policy actions vary with yearly weather/stress conditions or
mostly collapse to a fixed DAP/resource template.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "060_00"
TASK_NAME = "weather_action_sensitivity_audit"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
TAB = OUT / "tables"
FIG = OUT / "figures"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"


RUNS = [
    {
        "run_key": "SY_old_04610_ckpt100000",
        "label": "SY old PPO 046_10 ckpt100000",
        "site": "SY",
        "station": "SYA",
        "daily_dir": ROOT / "benchmark_results" / "046_10_sya_originIC_expanded_action_maskableppo" / "daily_outputs" / "SYA",
        "checkpoint": 100_000,
        "has_forecast_features": False,
        "primary": True,
    },
    {
        "run_key": "SY_forecast_04609_ckpt100000",
        "label": "SY forecast PPO 046_09 ckpt100000",
        "site": "SY",
        "station": "SYA",
        "daily_dir": ROOT
        / "benchmark_results"
        / "046_09_sya_originIC_binary_timing_forecast_raw_observation_ppo"
        / "daily_outputs"
        / "SYA",
        "checkpoint": 100_000,
        "has_forecast_features": True,
        "primary": True,
    },
    {
        "run_key": "LC_old_05300_ckpt100000",
        "label": "LC old PPO 053_00 ckpt100000",
        "site": "LC",
        "station": "LCA",
        "daily_dir": ROOT / "benchmark_results" / "053_00_lca_lowIC_expanded_action_maskableppo" / "daily_outputs" / "LCA",
        "checkpoint": 100_000,
        "has_forecast_features": False,
        "primary": False,
    },
    {
        "run_key": "LC_stress_reward_05900_ckpt50000",
        "label": "LC stress-reward PPO 059_00 ckpt50000",
        "site": "LC",
        "station": "LCA",
        "daily_dir": ROOT / "benchmark_results" / "059_00_lca_lowIC_stress_response_reward_maskableppo" / "daily_outputs" / "LCA",
        "checkpoint": 50_000,
        "has_forecast_features": False,
        "primary": False,
    },
]


COLORS = {
    "SY_old_04610_ckpt100000": "#4C78A8",
    "SY_forecast_04609_ckpt100000": "#F58518",
    "LC_old_05300_ckpt100000": "#54A24B",
    "LC_stress_reward_05900_ckpt50000": "#B279A2",
}


def ensure_dirs() -> None:
    for path in [OUT, TAB, FIG, DOC.parent]:
        path.mkdir(parents=True, exist_ok=True)


def numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_run(run: dict) -> pd.DataFrame:
    daily_dir = Path(run["daily_dir"])
    if not daily_dir.exists():
        raise FileNotFoundError(daily_dir)
    frames: list[pd.DataFrame] = []
    for path in sorted(daily_dir.glob("*_daily.csv")):
        df0 = pd.read_csv(path, keep_default_na=False)
        if "checkpoint_step" not in df0.columns:
            continue
        ck = pd.to_numeric(df0["checkpoint_step"], errors="coerce")
        df0 = df0[ck.eq(int(run["checkpoint"]))].copy()
        if df0.empty:
            continue
        df0["source_daily_csv"] = path.relative_to(ROOT).as_posix()
        df0["run_key"] = run["run_key"]
        df0["run_label"] = run["label"]
        df0["site"] = run["site"]
        df0["station"] = run["station"]
        df0["has_forecast_features"] = bool(run["has_forecast_features"])
        df0["primary_comparison"] = bool(run["primary"])
        frames.append(df0)
    if not frames:
        raise RuntimeError(f"No checkpoint {run['checkpoint']} daily CSVs under {daily_dir}")
    df = pd.concat(frames, ignore_index=True)
    df = numeric(
        df,
        [
            "year",
            "checkpoint_step",
            "date",
            "doy",
            "dap",
            "rain",
            "srad",
            "tmax",
            "tmin",
            "swfac",
            "nstres",
            "grnwt",
            "topwt",
            "xlai",
            "safe_action_amir",
            "safe_action_anfer",
            "raw_action_amir",
            "raw_action_anfer",
            "discrete_action_index",
            "season_cumulative_irrigation",
            "season_cumulative_n",
            "obs_rain_today_mm",
            "obs_rain_past7_mm",
            "obs_rain_future7_mm",
            "obs_tmean_future7_c",
        ],
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["safe_action_amir"] = pd.to_numeric(df["safe_action_amir"], errors="coerce").fillna(0.0)
    df["safe_action_anfer"] = pd.to_numeric(df["safe_action_anfer"], errors="coerce").fillna(0.0)
    df["has_irrigation"] = df["safe_action_amir"].gt(0)
    df["has_nitrogen"] = df["safe_action_anfer"].gt(0)
    df["has_any_action"] = df["has_irrigation"] | df["has_nitrogen"]
    for col in ["rain", "swfac", "nstres", "tmax", "tmin"]:
        if col not in df.columns:
            df[col] = np.nan
    df = df.sort_values(["run_key", "year", "dap"]).reset_index(drop=True)
    df["past7_rain_mm"] = df.groupby(["run_key", "year"])["rain"].transform(lambda s: s.fillna(0).rolling(7, min_periods=1).sum())
    df["future7_rain_mm"] = (
        df.iloc[::-1]
        .groupby(["run_key", "year"])["rain"]
        .transform(lambda s: s.fillna(0).rolling(7, min_periods=1).sum())
        .iloc[::-1]
        .to_numpy()
    )
    df["tmean"] = (df["tmax"] + df["tmin"]) / 2.0
    return df


def action_signature(sub: pd.DataFrame) -> str:
    events = []
    for row in sub.sort_values("dap").itertuples(index=False):
        i = float(getattr(row, "safe_action_amir", 0.0))
        n = float(getattr(row, "safe_action_anfer", 0.0))
        if i > 0 or n > 0:
            events.append(f"DAP{int(round(float(row.dap)))}:I{i:.0f}/N{n:.0f}")
    return "; ".join(events)


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or x.nunique() <= 1 or y.nunique() <= 1:
        return np.nan
    return float(x.corr(y))


def standardized_difference(pos: pd.Series, neg: pd.Series) -> float:
    x1 = pd.to_numeric(pos, errors="coerce").dropna()
    x0 = pd.to_numeric(neg, errors="coerce").dropna()
    if len(x1) < 2 or len(x0) < 2:
        return np.nan
    pooled = math.sqrt(((len(x1) - 1) * x1.var(ddof=1) + (len(x0) - 1) * x0.var(ddof=1)) / max(len(x1) + len(x0) - 2, 1))
    if not np.isfinite(pooled) or pooled == 0:
        return 0.0
    return float((x1.mean() - x0.mean()) / pooled)


def build_yearly(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (run_key, year), sub in daily.groupby(["run_key", "year"], sort=True):
        sub = sub.sort_values("dap")
        i_events = sub[sub["has_irrigation"]]
        n_events = sub[sub["has_nitrogen"]]
        total_i = float(sub["safe_action_amir"].sum())
        total_n = float(sub["safe_action_anfer"].sum())
        rows.append(
            {
                "run_key": run_key,
                "run_label": str(sub["run_label"].iloc[0]),
                "site": str(sub["site"].iloc[0]),
                "station": str(sub["station"].iloc[0]),
                "year": int(year),
                "checkpoint_step": int(pd.to_numeric(sub["checkpoint_step"], errors="coerce").dropna().iloc[0]),
                "has_forecast_features": bool(sub["has_forecast_features"].iloc[0]),
                "primary_comparison": bool(sub["primary_comparison"].iloc[0]),
                "season_rain_mm": float(pd.to_numeric(sub["rain"], errors="coerce").fillna(0).sum()),
                "mean_tmean_c": float(pd.to_numeric(sub["tmean"], errors="coerce").mean()),
                "max_swfac": float(pd.to_numeric(sub["swfac"], errors="coerce").max()),
                "max_nstres": float(pd.to_numeric(sub["nstres"], errors="coerce").max()),
                "stress_days_swfac_gt_0p05": int(pd.to_numeric(sub["swfac"], errors="coerce").gt(0.05).sum()),
                "stress_days_nstres_gt_0p05": int(pd.to_numeric(sub["nstres"], errors="coerce").gt(0.05).sum()),
                "final_grain_kg_ha": float(pd.to_numeric(sub["grnwt"], errors="coerce").iloc[-1]),
                "total_irrigation_mm": total_i,
                "total_nitrogen_kg_ha": total_n,
                "irrigation_event_count": int(len(i_events)),
                "nitrogen_event_count": int(len(n_events)),
                "first_irrigation_dap": float(i_events["dap"].iloc[0]) if len(i_events) else np.nan,
                "first_nitrogen_dap": float(n_events["dap"].iloc[0]) if len(n_events) else np.nan,
                "irrigation_event_daps": ";".join(str(int(round(x))) for x in i_events["dap"].tolist()),
                "nitrogen_event_daps": ";".join(str(int(round(x))) for x in n_events["dap"].tolist()),
                "action_signature": action_signature(sub),
                "source_daily_csv": str(sub["source_daily_csv"].iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values(["run_key", "year"]).reset_index(drop=True)


def build_diversity(yearly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run_key, sub in yearly.groupby("run_key", sort=False):
        mode_share = float(sub["action_signature"].value_counts(normalize=True).iloc[0]) if len(sub) else np.nan
        rows.append(
            {
                "run_key": run_key,
                "run_label": str(sub["run_label"].iloc[0]),
                "site": str(sub["site"].iloc[0]),
                "checkpoint_step": int(sub["checkpoint_step"].iloc[0]),
                "has_forecast_features": bool(sub["has_forecast_features"].iloc[0]),
                "years": int(sub["year"].nunique()),
                "unique_action_signatures": int(sub["action_signature"].nunique()),
                "mode_action_signature_share": mode_share,
                "season_rain_range_mm": float(sub["season_rain_mm"].max() - sub["season_rain_mm"].min()),
                "max_swfac_range": float(sub["max_swfac"].max() - sub["max_swfac"].min()),
                "max_nstres_range": float(sub["max_nstres"].max() - sub["max_nstres"].min()),
                "mean_total_irrigation_mm": float(sub["total_irrigation_mm"].mean()),
                "sd_total_irrigation_mm": float(sub["total_irrigation_mm"].std(ddof=0)),
                "range_total_irrigation_mm": float(sub["total_irrigation_mm"].max() - sub["total_irrigation_mm"].min()),
                "mean_total_nitrogen_kg_ha": float(sub["total_nitrogen_kg_ha"].mean()),
                "sd_total_nitrogen_kg_ha": float(sub["total_nitrogen_kg_ha"].std(ddof=0)),
                "range_total_nitrogen_kg_ha": float(sub["total_nitrogen_kg_ha"].max() - sub["total_nitrogen_kg_ha"].min()),
                "corr_total_i_vs_rain": safe_corr(sub["total_irrigation_mm"], sub["season_rain_mm"]),
                "corr_total_i_vs_max_swfac": safe_corr(sub["total_irrigation_mm"], sub["max_swfac"]),
                "corr_total_n_vs_max_nstres": safe_corr(sub["total_nitrogen_kg_ha"], sub["max_nstres"]),
                "first_i_dap_sd": float(pd.to_numeric(sub["first_irrigation_dap"], errors="coerce").std(ddof=0)),
                "first_n_dap_sd": float(pd.to_numeric(sub["first_nitrogen_dap"], errors="coerce").std(ddof=0)),
            }
        )
    return pd.DataFrame(rows)


def build_event_context(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    features = ["dap", "rain", "past7_rain_mm", "future7_rain_mm", "swfac", "nstres", "tmean"]
    for rk, sub in daily.groupby("run_key", sort=False):
        for action_type, flag_col in [("irrigation", "has_irrigation"), ("nitrogen", "has_nitrogen")]:
            pos = sub[sub[flag_col]]
            neg = sub[~sub[flag_col]]
            for feature in features:
                rows.append(
                    {
                        "run_key": rk,
                        "run_label": str(sub["run_label"].iloc[0]),
                        "action_type": action_type,
                        "feature": feature,
                        "action_days": int(len(pos)),
                        "no_action_days": int(len(neg)),
                        "mean_on_action_days": float(pd.to_numeric(pos[feature], errors="coerce").mean()) if len(pos) else np.nan,
                        "mean_on_no_action_days": float(pd.to_numeric(neg[feature], errors="coerce").mean()) if len(neg) else np.nan,
                        "standardized_difference": standardized_difference(pos[feature], neg[feature]),
                    }
                )
    return pd.DataFrame(rows)


def build_event_rows(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in daily[daily["has_any_action"]].itertuples(index=False):
        if float(r.safe_action_amir) > 0:
            rows.append(
                {
                    "run_key": r.run_key,
                    "run_label": r.run_label,
                    "site": r.site,
                    "year": int(r.year),
                    "action_type": "irrigation",
                    "dap": float(r.dap),
                    "amount": float(r.safe_action_amir),
                    "rain": float(r.rain) if pd.notna(r.rain) else np.nan,
                    "past7_rain_mm": float(r.past7_rain_mm) if pd.notna(r.past7_rain_mm) else np.nan,
                    "future7_rain_mm": float(r.future7_rain_mm) if pd.notna(r.future7_rain_mm) else np.nan,
                    "swfac": float(r.swfac) if pd.notna(r.swfac) else np.nan,
                    "nstres": float(r.nstres) if pd.notna(r.nstres) else np.nan,
                }
            )
        if float(r.safe_action_anfer) > 0:
            rows.append(
                {
                    "run_key": r.run_key,
                    "run_label": r.run_label,
                    "site": r.site,
                    "year": int(r.year),
                    "action_type": "nitrogen",
                    "dap": float(r.dap),
                    "amount": float(r.safe_action_anfer),
                    "rain": float(r.rain) if pd.notna(r.rain) else np.nan,
                    "past7_rain_mm": float(r.past7_rain_mm) if pd.notna(r.past7_rain_mm) else np.nan,
                    "future7_rain_mm": float(r.future7_rain_mm) if pd.notna(r.future7_rain_mm) else np.nan,
                    "swfac": float(r.swfac) if pd.notna(r.swfac) else np.nan,
                    "nstres": float(r.nstres) if pd.notna(r.nstres) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_action_timing(events: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    if events.empty:
        return paths
    runs = events["run_key"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(runs), 2, figsize=(13, max(3.2 * len(runs), 4.0)), sharex=True)
    axes = np.atleast_2d(axes)
    for row_idx, rk in enumerate(runs):
        sub_run = events[events["run_key"].eq(rk)]
        label = str(sub_run["run_label"].iloc[0])
        for col_idx, action_type in enumerate(["irrigation", "nitrogen"]):
            ax = axes[row_idx, col_idx]
            sub = sub_run[sub_run["action_type"].eq(action_type)]
            if not sub.empty:
                sizes = 25 + 1.7 * pd.to_numeric(sub["amount"], errors="coerce").fillna(0)
                ax.scatter(sub["dap"], sub["year"], s=sizes, color=COLORS.get(rk, "#555555"), alpha=0.82, edgecolor="black", linewidth=0.35)
            ax.set_title(f"{label}: {action_type} event DAP", loc="left", fontsize=9)
            ax.set_ylabel("Year")
            ax.grid(color="#E8E8E8", linewidth=0.7)
            ax.set_xlim(0, 125)
            if not sub_run.empty:
                ax.set_yticks(sorted(sub_run["year"].unique()))
    for ax in axes[-1, :]:
        ax.set_xlabel("DAP")
    fig.suptitle("PPO action timing across years", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = FIG / "060_00_action_timing_scatter.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    paths.extend([out, out.with_suffix(".svg")])
    return paths


def plot_weather_vs_actions(yearly: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2))
    for rk, sub in yearly.groupby("run_key", sort=False):
        color = COLORS.get(rk, "#555555")
        label = str(sub["run_label"].iloc[0])
        axes[0, 0].scatter(sub["season_rain_mm"], sub["total_irrigation_mm"], color=color, label=label, s=55, alpha=0.85)
        axes[0, 1].scatter(sub["max_swfac"], sub["total_irrigation_mm"], color=color, label=label, s=55, alpha=0.85)
        axes[1, 0].scatter(sub["max_nstres"], sub["total_nitrogen_kg_ha"], color=color, label=label, s=55, alpha=0.85)
        axes[1, 1].scatter(sub["season_rain_mm"], sub["final_grain_kg_ha"], color=color, label=label, s=55, alpha=0.85)
        for _, rr in sub.iterrows():
            for ax, xcol, ycol in [
                (axes[0, 0], "season_rain_mm", "total_irrigation_mm"),
                (axes[0, 1], "max_swfac", "total_irrigation_mm"),
                (axes[1, 0], "max_nstres", "total_nitrogen_kg_ha"),
                (axes[1, 1], "season_rain_mm", "final_grain_kg_ha"),
            ]:
                ax.text(rr[xcol], rr[ycol], str(int(rr["year"]))[-2:], fontsize=6, ha="center", va="bottom", alpha=0.75)
    panels = [
        (axes[0, 0], "Season rain vs irrigation", "Season rain (mm)", "Total irrigation (mm)"),
        (axes[0, 1], "Water stress vs irrigation", "Max WSF/WSPD", "Total irrigation (mm)"),
        (axes[1, 0], "Nitrogen stress vs N application", "Max NSF/NSTRES", "Total N (kg/ha)"),
        (axes[1, 1], "Season rain vs yield", "Season rain (mm)", "Final grain yield (kg/ha)"),
    ]
    for ax, title, xlabel, ylabel in panels:
        ax.set_title(title, loc="left", fontweight="bold", fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(color="#E8E8E8", linewidth=0.7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle("Weather/stress variation versus PPO resource totals", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.955))
    out = FIG / "060_00_weather_stress_vs_action_totals.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    paths.extend([out, out.with_suffix(".svg")])
    return paths


def md_table(df: pd.DataFrame, max_rows: int = 60) -> str:
    if df.empty:
        return "No rows."
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


def write_doc(yearly: pd.DataFrame, diversity: pd.DataFrame, assoc: pd.DataFrame, figures: list[Path]) -> None:
    primary = diversity[diversity["run_key"].isin(["SY_old_04610_ckpt100000", "SY_forecast_04609_ckpt100000"])].copy()
    compact_cols = [
        "run_key",
        "years",
        "unique_action_signatures",
        "mode_action_signature_share",
        "season_rain_range_mm",
        "max_swfac_range",
        "max_nstres_range",
        "mean_total_irrigation_mm",
        "sd_total_irrigation_mm",
        "mean_total_nitrogen_kg_ha",
        "sd_total_nitrogen_kg_ha",
        "corr_total_i_vs_rain",
        "corr_total_i_vs_max_swfac",
        "corr_total_n_vs_max_nstres",
    ]
    yearly_preview = yearly[
        yearly["run_key"].isin(["SY_old_04610_ckpt100000", "SY_forecast_04609_ckpt100000"])
    ][
        [
            "run_key",
            "year",
            "season_rain_mm",
            "max_swfac",
            "max_nstres",
            "total_irrigation_mm",
            "total_nitrogen_kg_ha",
            "irrigation_event_daps",
            "nitrogen_event_daps",
            "final_grain_kg_ha",
        ]
    ]
    dap_assoc = assoc[
        assoc["run_key"].isin(["SY_old_04610_ckpt100000", "SY_forecast_04609_ckpt100000"])
        & assoc["feature"].isin(["dap", "past7_rain_mm", "future7_rain_mm", "swfac", "nstres"])
    ].copy()
    lines = [
        "# 060_00 weather/action sensitivity audit",
        "",
        "This read-only audit compares old PPO and forecast-enabled PPO daily evaluation outputs. It asks whether action timing and resource totals vary with yearly weather or simulated water/nitrogen stress.",
        "",
        "Primary comparison: `046_10` SY old PPO checkpoint `100000` versus `046_09` SY raw-weather/forecast-observation PPO checkpoint `100000`. LC `053_00` and `059_00` are included as secondary fixed-template references.",
        "",
        "## Main finding",
        "",
        "The forecast-enabled SY PPO has forecast columns in the daily output, but its realized action pattern is still dominated by fixed early-season DAP/resource templates. Adding raw forecast features did not produce clear weather-responsive action adaptation in the existing run.",
        "",
        "## Diversity and weather association summary",
        "",
        md_table(primary[compact_cols]),
        "",
        "## Primary yearly action table",
        "",
        md_table(yearly_preview, max_rows=30),
        "",
        "## Action-day versus no-action-day context",
        "",
        "Large standardized differences for `dap` combined with weak/unstable differences for rain or stress indicate a timing-template policy rather than a weather-triggered policy.",
        "",
        md_table(dap_assoc, max_rows=40),
        "",
        "## Generated figures",
        "",
    ]
    for path in figures:
        lines.append(f"- `{path.relative_to(ROOT).as_posix()}`")
    lines.extend(
        [
            "",
            "## Output tables",
            "",
            f"- `{(TAB / '060_00_daily_combined.csv').relative_to(ROOT).as_posix()}`",
            f"- `{(TAB / '060_00_yearly_action_weather_summary.csv').relative_to(ROOT).as_posix()}`",
            f"- `{(TAB / '060_00_action_diversity_by_run.csv').relative_to(ROOT).as_posix()}`",
            f"- `{(TAB / '060_00_action_day_context_association.csv').relative_to(ROOT).as_posix()}`",
            f"- `{(TAB / '060_00_action_events.csv').relative_to(ROOT).as_posix()}`",
            "",
            "## Interpretation for manuscript planning",
            "",
            "Use these results as a guardrail against over-claiming weather-responsive RL. The defensible statement is that PPO can find high-performing resource templates under selected site conditions, while current forecast-feature augmentation alone did not demonstrate adaptive weather-conditioned action timing.",
        ]
    )
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    frames = [load_run(run) for run in RUNS]
    daily = pd.concat(frames, ignore_index=True)
    yearly = build_yearly(daily)
    diversity = build_diversity(yearly)
    assoc = build_event_context(daily)
    events = build_event_rows(daily)

    daily.to_csv(TAB / "060_00_daily_combined.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(TAB / "060_00_yearly_action_weather_summary.csv", index=False, encoding="utf-8-sig")
    diversity.to_csv(TAB / "060_00_action_diversity_by_run.csv", index=False, encoding="utf-8-sig")
    assoc.to_csv(TAB / "060_00_action_day_context_association.csv", index=False, encoding="utf-8-sig")
    events.to_csv(TAB / "060_00_action_events.csv", index=False, encoding="utf-8-sig")

    figures = []
    figures.extend(plot_action_timing(events))
    figures.extend(plot_weather_vs_actions(yearly))
    write_doc(yearly, diversity, assoc, figures)
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(f"Wrote {DOC.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
