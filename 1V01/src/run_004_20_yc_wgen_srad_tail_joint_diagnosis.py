#!/usr/bin/env python3
"""Read-only YC WGEN SRAD tail and weather dependence diagnosis (004_20)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results/yc_random_weather_ppo/004_17_yc_runtime_weather_recovery"
MANIFEST = ARCHIVE / "weather_archive_manifest_verified.csv"
FITTING = ROOT / "results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv"
CLI = ROOT / "results/yc_wgen_cli_pilot/003_06_05_02/final/CNYC.CLI"
PARAM_DEFS = ROOT / "results/yc_wgen_cli_pilot/003_06_04/definitions/cli_parameter_definitions.json"
PARAM_META = ROOT / "results/yc_wgen_cli_pilot/003_06_05_02/final/cli_generation_metadata.json"
PARAM_AUDIT = ROOT / "results/yc_wgen_cli_pilot/004_01/parameter_audit/parameter_method_summary.json"
PRIOR_RUNNER = ROOT / "src/run_004_18A_yc_weather_summary_for_reporting.py"
PRIOR_REPORT = ROOT / "results/yc_random_weather_ppo/004_18A_yc_weather_summary_for_reporting"
EVAL_TRACE = ROOT / "results/yc_random_weather_ppo/004_05/evaluation_step_level_all_models.csv"
OUT = ROOT / "results/yc_random_weather_ppo/004_20_yc_wgen_srad_tail_joint_diagnosis"
REPORT = ROOT / "docs/yc_random_weather_004_20_wgen_srad_tail_joint_diagnosis.md"

FIT_SHA = "4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34"
CLI_SHA = "65CF134600A5881706A5D435E1A09B276ED92A21FA5ABE2E18AAF63AF1E3A929"
PROVISIONAL = {1003, 1036}
STAGES = [(0, 30, "DAP0_30"), (31, 60, "DAP31_60"), (61, 90, "DAP61_90"), (91, 999, "DAP_gt90")]
COLORS = {
    "FITTING": "#68777D", "EXACT78": "#367A9B", "FULL80": "#4F8269",
    "HELDOUT20": "#D18836", "OBSERVED": "#B54A58",
}
LABELS = {
    "FITTING": "Fitting 2004-13", "EXACT78": "EXACT78", "FULL80": "FULL80 candidate",
    "HELDOUT20": "Held-out 20", "OBSERVED": "Observed 2014-23",
}
WEATHER = ("SRAD", "RAIN", "TMAX", "TMIN")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_prior():
    spec = importlib.util.spec_from_file_location("weather_reporting_004_18A_for_004_20", PRIOR_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import frozen 004_18A source: {PRIOR_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def numeric_weather(frame: pd.DataFrame, source: str) -> tuple[pd.DataFrame, dict[str, int]]:
    result = frame.copy()
    counts: dict[str, int] = {}
    for column in WEATHER:
        raw = pd.to_numeric(result[column], errors="coerce")
        counts[f"missing_{column}"] = int(raw.isna().sum())
        counts[f"sentinel_{column}_minus99"] = int((raw == -99).sum())
        counts[f"zero_{column}"] = int((raw == 0).sum())
        counts[f"negative_{column}_non_sentinel"] = int(((raw < 0) & (raw != -99)).sum())
        result[column] = raw.mask(raw == -99, np.nan)
        if result[column].notna().any() and not np.isfinite(result[column].dropna()).all():
            raise ValueError(f"Non-finite {column} values in {source}")
    if "DATE" in result:
        result["DATE"] = pd.to_datetime(result["DATE"], errors="coerce")
        if result["DATE"].isna().any():
            raise ValueError(f"Invalid DATE values in {source}")
    if "DAP" in result:
        result["DAP"] = pd.to_numeric(result["DAP"], errors="coerce")
    return result, counts


def qa_row(dataset: str, unit: str, frame: pd.DataFrame, source_path: str,
           verification: str, counter: dict[str, int]) -> dict[str, Any]:
    srad = frame["SRAD"]
    valid = srad.dropna()
    return {
        "dataset": dataset, "year_or_seed": unit, "n_days": int(len(frame)),
        "srad_min": float(valid.min()) if len(valid) else np.nan,
        "srad_max": float(valid.max()) if len(valid) else np.nan,
        "srad_mean": float(valid.mean()) if len(valid) else np.nan,
        "missing_srad": counter.get("missing_SRAD", 0),
        "srad_minus99_sentinel": counter.get("sentinel_SRAD_minus99", 0),
        "zero_srad": counter.get("zero_SRAD", 0),
        "negative_srad": counter.get("negative_SRAD_non_sentinel", 0),
        "source_path": source_path, "verification_status": verification,
        "unit": "MJ m-2 d-1; native values, no scale conversion",
    }


def longest_run(mask: np.ndarray) -> int:
    best = run = 0
    for item in mask:
        run = run + 1 if bool(item) else 0
        best = max(best, run)
    return best


def spell_count(mask: np.ndarray, minimum: int) -> int:
    count = run = 0
    for item in mask:
        if bool(item):
            run += 1
        else:
            if run >= minimum:
                count += 1
            run = 0
    return count + int(run >= minimum)


def tail_metrics(frame: pd.DataFrame, fit_p1: float, fit_p5: float) -> dict[str, float | int]:
    values = frame["SRAD"].dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return {key: np.nan for key in (
            "srad_min", "srad_p0_5", "srad_p1", "srad_p2_5", "srad_p5", "srad_p10",
            "srad_median", "srad_mean", "low_days_lt3", "low_days_lt4", "low_days_lt5", "low_days_lt6",
            "low_days_below_fit_p1", "low_days_below_fit_p5", "longest_lt5_spell",
            "longest_below_fit_p5_spell", "lt5_spells_ge2", "lt5_spells_ge3",
            "below_fit_p5_spells_ge2", "below_fit_p5_spells_ge3", "min_3day_mean",
            "min_5day_mean", "min_7day_mean")}
    valid_frame = frame.loc[frame["SRAD"].notna()].copy()
    srad = valid_frame["SRAD"].to_numpy(dtype=float)
    under5 = srad < 5.0
    under_p5 = srad < fit_p5
    result: dict[str, float | int] = {
        "srad_min": float(np.min(srad)), "srad_p0_5": float(np.quantile(srad, .005)),
        "srad_p1": float(np.quantile(srad, .01)), "srad_p2_5": float(np.quantile(srad, .025)),
        "srad_p5": float(np.quantile(srad, .05)), "srad_p10": float(np.quantile(srad, .10)),
        "srad_median": float(np.median(srad)), "srad_mean": float(np.mean(srad)),
        "low_days_lt3": int((srad < 3).sum()), "low_days_lt4": int((srad < 4).sum()),
        "low_days_lt5": int(under5.sum()), "low_days_lt6": int((srad < 6).sum()),
        "low_days_below_fit_p1": int((srad < fit_p1).sum()), "low_days_below_fit_p5": int(under_p5.sum()),
        "longest_lt5_spell": longest_run(under5), "longest_below_fit_p5_spell": longest_run(under_p5),
        "lt5_spells_ge2": spell_count(under5, 2), "lt5_spells_ge3": spell_count(under5, 3),
        "below_fit_p5_spells_ge2": spell_count(under_p5, 2), "below_fit_p5_spells_ge3": spell_count(under_p5, 3),
    }
    ordered = valid_frame.sort_values("DATE", kind="stable") if "DATE" in valid_frame else valid_frame
    ordered_values = ordered["SRAD"].to_numpy(dtype=float)
    for width in (3, 5, 7):
        rolling = pd.Series(ordered_values).rolling(width, min_periods=width).mean()
        result[f"min_{width}day_mean"] = float(rolling.min()) if rolling.notna().any() else np.nan
    return result


def prepare_unit(frame: pd.DataFrame, dataset: str, unit: str, crop_year: int | None,
                 source_path: str, verification: str, season_start: str | None = None,
                 prior=None) -> tuple[pd.DataFrame, dict[str, int]]:
    converted, counts = numeric_weather(frame, source_path)
    if "DATE" not in converted or "DAP" not in converted:
        raise ValueError(f"DATE/DAP absent in season series: {source_path}")
    if prior is not None:
        converted, _ = prior.season_window(converted)
    converted["year_or_seed"] = str(unit)
    converted["dataset"] = dataset
    converted["crop_year"] = int(crop_year) if crop_year is not None else np.nan
    converted["verification_status"] = verification
    if season_start is not None:
        start = pd.Timestamp(season_start)
        converted["calendar_date"] = start + pd.to_timedelta(converted["DAP"].astype(int), unit="D")
        converted["calendar_basis"] = "004_17 manifest season_start + verified DAP"
    else:
        converted["calendar_date"] = converted["DATE"]
        converted["calendar_basis"] = "official observed WTH DATE aligned by 004_18A FileX PDATE"
    converted["calendar_month"] = converted["calendar_date"].dt.month.astype(int)
    return converted, counts


def fit_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"DATE", *WEATHER}
    if not required.issubset(frame.columns):
        raise ValueError(f"Fitting CSV missing fields: {sorted(required - set(frame.columns))}")
    frame["DATE"] = pd.to_datetime(frame["DATE"], errors="raise")
    if len(frame) != 3653 or frame["DATE"].min() != pd.Timestamp("2004-01-01") or frame["DATE"].max() != pd.Timestamp("2013-12-31"):
        raise ValueError("Fitting CSV date range/row count differs from frozen 2004-2013 input")
    if frame["DATE"].duplicated().any() or (frame["DATE"].diff().dropna().dt.days != 1).any():
        raise ValueError("Fitting CSV dates are not unique consecutive days")
    frame, _ = numeric_weather(frame, str(path))
    frame["calendar_date"] = frame["DATE"]
    frame["calendar_month"] = frame["DATE"].dt.month.astype(int)
    frame["calendar_year"] = frame["DATE"].dt.year.astype(int)
    frame["DAP"] = np.nan
    frame["year_or_seed"] = frame["calendar_year"].astype(str)
    frame["dataset"] = "FITTING"
    frame["crop_year"] = frame["calendar_year"].astype(int)
    frame["verification_status"] = "FROZEN_FITTING_WEATHER"
    frame["calendar_basis"] = "natural fitting DATE; no DAP assigned"
    return frame


def get_observed(prior) -> tuple[dict[int, pd.DataFrame], dict[int, tuple[str, str]]]:
    observed = prior.observed_rows()
    sources: dict[int, tuple[str, str]] = {}
    base = ROOT / "results/yc_random_weather_ppo/004_05/evaluation/runs/observed_weather/random_weather/ppo_seed_5/rendered_inputs/YCA"
    for year, frame in observed.items():
        folder = base / str(year)
        wths = list(folder.rglob("*.WTH"))
        filexs = list(folder.rglob("*.jinja2"))
        if len(wths) != 1 or len(filexs) != 1:
            raise ValueError(f"Expected one observed WTH/FileX for {year}")
        sources[int(year)] = (str(wths[0].relative_to(ROOT)), str(filexs[0].relative_to(ROOT)))
    if sorted(map(int, observed)) != list(range(2014, 2024)):
        raise ValueError("Frozen Observed extraction did not return 2014-2023")
    return observed, sources


def unit_label(dataset: str, seed_or_year: Any) -> str:
    return str(int(seed_or_year)) if pd.notna(seed_or_year) else ""


def summarize_frame(frame: pd.DataFrame, fit_p1: float, fit_p5: float) -> dict[str, Any]:
    base = tail_metrics(frame, fit_p1, fit_p5)
    base["n_days"] = int(len(frame))
    return base


def monthly_metrics(frame: pd.DataFrame, fit_p1_month: dict[int, float], fit_p5_month: dict[int, float]) -> dict[str, Any]:
    months = set(frame["calendar_month"].dropna().astype(int))
    if len(months) != 1:
        raise ValueError("monthly_metrics requires a single calendar month")
    month = int(next(iter(months)))
    return {"calendar_month": month, **tail_metrics(frame, fit_p1_month[month], fit_p5_month[month]), "n_days": int(len(frame))}


def low_event_daily(dataset_frames: dict[str, pd.DataFrame], fit_p5: float,
                    fit_season_tmax_median: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_parts = []
    rain_rows = []
    temp_rows = []
    bins = [(-np.inf, 0, "RAIN=0"), (0, 1, "0<RAIN<=1"), (1, 5, "1<RAIN<=5"),
            (5, 20, "5<RAIN<=20"), (20, np.inf, "RAIN>20")]
    for dataset, frame in dataset_frames.items():
        data = frame.copy()
        data = data.loc[data["calendar_month"].isin(SEASON_MONTHS)].copy()
        data["srad_below_fit_p5"] = data["SRAD"] < fit_p5
        data["rain_bin"] = pd.NA
        data.loc[data["RAIN"] == 0, "rain_bin"] = "RAIN=0"
        data.loc[(data["RAIN"] > 0) & (data["RAIN"] <= 1), "rain_bin"] = "0<RAIN<=1"
        data.loc[(data["RAIN"] > 1) & (data["RAIN"] <= 5), "rain_bin"] = "1<RAIN<=5"
        data.loc[(data["RAIN"] > 5) & (data["RAIN"] <= 20), "rain_bin"] = "5<RAIN<=20"
        data.loc[data["RAIN"] > 20, "rain_bin"] = "RAIN>20"
        daily_parts.append(data[["dataset", "year_or_seed", "crop_year", "verification_status", "calendar_date",
                                 "calendar_month", "DAP", "SRAD", "RAIN", "TMAX", "TMIN", "srad_below_fit_p5",
                                 "rain_bin", "calendar_basis"]])
        valid = data.dropna(subset=["SRAD", "RAIN", "TMAX", "TMIN"])
        rain_spear = valid["SRAD"].corr(valid["RAIN"], method="spearman") if len(valid) > 2 else np.nan
        for lo, hi, label in bins:
            if label == "RAIN=0":
                subset = valid.loc[valid["RAIN"] == 0]
            elif np.isneginf(lo):
                subset = valid.loc[valid["RAIN"] == 0]
            elif np.isposinf(hi):
                subset = valid.loc[valid["RAIN"] > lo]
            else:
                subset = valid.loc[(valid["RAIN"] > lo) & (valid["RAIN"] <= hi)]
            rain_rows.append({"dataset": dataset, "rain_bin": label, "n_days": len(subset),
                              "srad_mean": subset["SRAD"].mean(), "srad_median": subset["SRAD"].median(),
                              "srad_p5": subset["SRAD"].quantile(.05) if len(subset) else np.nan,
                              "srad_p10": subset["SRAD"].quantile(.10) if len(subset) else np.nan,
                              "spearman_srad_rain_all_days": rain_spear})
        wet = valid[valid["RAIN"] > 0]
        dry = valid[valid["RAIN"] <= 0]
        for kind, subset in (("wet_day_RAIN_gt0", wet), ("dry_day_RAIN_le0", dry)):
            for threshold_name, threshold in (("SRAD_lt5", 5.0), ("below_fitting_P5", fit_p5)):
                low = subset[subset["SRAD"] < threshold]
                rain_rows.append({"dataset": dataset, "rain_bin": kind, "condition": threshold_name,
                                  "n_days": len(subset), "low_srad_days": len(low),
                                  "low_srad_share": len(low) / len(subset) if len(subset) else np.nan,
                                  "srad_mean": subset["SRAD"].mean(), "srad_median": subset["SRAD"].median(),
                                  "srad_p5": subset["SRAD"].quantile(.05) if len(subset) else np.nan,
                                  "srad_p10": subset["SRAD"].quantile(.10) if len(subset) else np.nan,
                                  "spearman_srad_rain_all_days": rain_spear})
        low = valid[valid["SRAD"] < fit_p5]
        normal = valid[valid["SRAD"] >= fit_p5]
        temp_rows.append({
            "dataset": dataset, "n_days": len(valid), "spearman_srad_rain": rain_spear,
            "spearman_srad_tmax": valid["SRAD"].corr(valid["TMAX"], method="spearman") if len(valid) > 2 else np.nan,
            "spearman_srad_tmin": valid["SRAD"].corr(valid["TMIN"], method="spearman") if len(valid) > 2 else np.nan,
            "low_srad_threshold": "below full-fitting P5", "low_srad_days": len(low),
            "low_srad_wet_share": (low["RAIN"] > 0).mean() if len(low) else np.nan,
            "wet_day_low_srad_share": (wet["SRAD"] < fit_p5).mean() if len(wet) else np.nan,
            "low_srad_tmax_mean": low["TMAX"].mean(), "low_srad_tmax_median": low["TMAX"].median(),
            "normal_srad_tmax_mean": normal["TMAX"].mean(), "normal_srad_tmax_median": normal["TMAX"].median(),
            "low_srad_tmin_mean": low["TMIN"].mean(), "low_srad_tmin_median": low["TMIN"].median(),
            "normal_srad_tmin_mean": normal["TMIN"].mean(), "normal_srad_tmin_median": normal["TMIN"].median(),
            "fitting_season_tmax_median": fit_season_tmax_median,
        })
    return pd.concat(daily_parts, ignore_index=True), pd.DataFrame(rain_rows), pd.DataFrame(temp_rows)


def plot_style(ax, title: str, ylabel: str | None = None):
    ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(axis="y", color="#DFE4E7", linewidth=.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def save_fig(fig, name: str):
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures" / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_all(daily: pd.DataFrame, unit_df: pd.DataFrame, stage_df: pd.DataFrame,
             monthly: pd.DataFrame, rain_joint: pd.DataFrame, event_freq: pd.DataFrame,
             fit_p1: float, fit_p5: float):
    groups = ["FITTING", "EXACT78", "FULL80", "HELDOUT20", "OBSERVED"]
    rng = np.random.default_rng(42020)

    fig, ax = plt.subplots(figsize=(7.1, 4.2), constrained_layout=True)
    for group in groups:
        values = daily.loc[daily.dataset == group, "SRAD"].dropna().to_numpy(float)
        values.sort()
        if len(values):
            ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post",
                    color=COLORS[group], linewidth=1.8 if group in ("EXACT78", "OBSERVED") else 1.4,
                    linestyle="--" if group == "FULL80" else "-", label=LABELS[group])
    ax.axvline(fit_p1, color="#65727A", linestyle=":", linewidth=1, label=f"Fitting P1={fit_p1:.2f}")
    ax.axvline(fit_p5, color="#9A7040", linestyle="--", linewidth=1, label=f"Fitting P5={fit_p5:.2f}")
    plot_style(ax, "SRAD lower-tail ECDF", "Empirical cumulative fraction")
    ax.set_xlabel("SRAD (MJ m⁻² d⁻¹); crop-calendar months")
    ax.set_xlim(left=0, right=max(12, float(daily.SRAD.quantile(.30))))
    ax.legend(frameon=False, fontsize=7, ncol=2)
    save_fig(fig, "figure1_srad_lower_tail_ecdf.png")

    season_units = unit_df[unit_df.scope.isin(["season", "crop_month_matched"])].copy()
    metrics = [("srad_min", "Per-unit minimum"), ("srad_p1", "Per-unit P1"), ("srad_p5", "Per-unit P5")]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7), constrained_layout=True)
    for ax, (metric, title) in zip(axes, metrics):
        arrays, labels, colors = [], [], []
        for group in groups:
            vals = season_units.loc[(season_units.dataset == group) & (season_units.scope == ("crop_month_matched" if group == "FITTING" else "season")), metric].dropna().to_numpy(float)
            if len(vals):
                arrays.append(vals); labels.append(LABELS[group]); colors.append(COLORS[group])
        bp = ax.boxplot(arrays, tick_labels=labels, patch_artist=True, showfliers=False, widths=.58)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(.25); patch.set_edgecolor(color)
        for i, (vals, color) in enumerate(zip(arrays, colors), start=1):
            ax.scatter(rng.normal(i, .055, len(vals)), vals, s=12, color=color, alpha=.68, zorder=3)
        plot_style(ax, title, "MJ m⁻² d⁻¹")
        ax.tick_params(axis="x", rotation=25, labelsize=7)
    save_fig(fig, "figure2_realization_min_p1_p5.png")

    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.8), constrained_layout=True)
    for ax, metric, title in zip(axes, ["min_3day_mean", "min_5day_mean"], ["Minimum rolling 3-day mean", "Minimum rolling 5-day mean"]):
        arrays, labels, colors = [], [], []
        for group in groups:
            scope = "crop_month_matched" if group == "FITTING" else "season"
            vals = season_units.loc[(season_units.dataset == group) & (season_units.scope == scope), metric].dropna().to_numpy(float)
            if len(vals): arrays.append(vals); labels.append(LABELS[group]); colors.append(COLORS[group])
        bp = ax.boxplot(arrays, tick_labels=labels, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], colors): patch.set_facecolor(color); patch.set_alpha(.25); patch.set_edgecolor(color)
        for i, (vals, color) in enumerate(zip(arrays, colors), start=1): ax.scatter(rng.normal(i, .05, len(vals)), vals, s=10, color=color, alpha=.6)
        plot_style(ax, title, "MJ m⁻² d⁻¹"); ax.tick_params(axis="x", rotation=24, labelsize=7)
    save_fig(fig, "figure3_min_3day_5day_mean.png")

    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.8), constrained_layout=True)
    for ax, metric, title in zip(axes, ["low_days_lt5", "longest_below_fit_p5_spell"], ["Days with SRAD < 5", "Longest spell below fitting P5"]):
        arrays, labels, colors = [], [], []
        for group in groups:
            scope = "crop_month_matched" if group == "FITTING" else "season"
            vals = season_units.loc[(season_units.dataset == group) & (season_units.scope == scope), metric].dropna().to_numpy(float)
            if len(vals): arrays.append(vals); labels.append(LABELS[group]); colors.append(COLORS[group])
        bp = ax.boxplot(arrays, tick_labels=labels, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], colors): patch.set_facecolor(color); patch.set_alpha(.25); patch.set_edgecolor(color)
        for i, (vals, color) in enumerate(zip(arrays, colors), start=1): ax.scatter(rng.normal(i, .05, len(vals)), vals, s=10, color=color, alpha=.6)
        plot_style(ax, title); ax.tick_params(axis="x", rotation=24, labelsize=7)
    save_fig(fig, "figure4_low_day_counts_and_spells.png")

    pooled_month = (daily.groupby(["dataset", "calendar_month"], as_index=False)
                    .agg(n_days=("SRAD", "count"), srad_p5=("SRAD", lambda x: x.quantile(.05)),
                         srad_min=("SRAD", "min"), srad_mean=("SRAD", "mean")))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), constrained_layout=True)
    for ax, metric, title in zip(axes, ["srad_p5", "srad_min"], ["Monthly SRAD P5", "Monthly SRAD minimum"]):
        for group in groups:
            part = pooled_month[pooled_month.dataset == group]
            ax.plot(part.calendar_month, part[metric], marker="o", ms=3, lw=1.5, color=COLORS[group],
                    linestyle="--" if group == "FULL80" else "-", label=LABELS[group])
        plot_style(ax, title, "MJ m⁻² d⁻¹"); ax.set_xticks(range(1, 13)); ax.set_xlabel("Calendar month")
    axes[0].legend(frameon=False, fontsize=7)
    save_fig(fig, "figure5_monthly_srad_lower_tail.png")

    stage_groups = ["EXACT78", "FULL80", "HELDOUT20", "OBSERVED"]
    stage_order = [name for _, _, name in STAGES]
    stage_agg = stage_df.groupby(["dataset", "stage"], as_index=False).agg(srad_p5=("srad_p5", "median"), srad_p1=("srad_p1", "median"))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), constrained_layout=True)
    for ax, metric, title in zip(axes, ["srad_p1", "srad_p5"], ["DAP-stage SRAD P1", "DAP-stage SRAD P5"]):
        for group in stage_groups:
            part = stage_agg[stage_agg.dataset == group].set_index("stage").reindex(stage_order)
            ax.plot(range(len(stage_order)), part[metric], marker="o", ms=4, lw=1.6, color=COLORS[group],
                    linestyle="--" if group == "FULL80" else "-", label=LABELS[group])
        plot_style(ax, title, "MJ m⁻² d⁻¹"); ax.set_xticks(range(len(stage_order)), stage_order, rotation=15)
    axes[0].legend(frameon=False, fontsize=7)
    save_fig(fig, "figure6_dap_stage_srad_tail.png")

    bins = ["RAIN=0", "0<RAIN<=1", "1<RAIN<=5", "5<RAIN<=20", "RAIN>20"]
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.7), constrained_layout=True)
    for ax, group in zip(axes.flat, groups):
        part = rain_joint[(rain_joint.dataset == group) & rain_joint.rain_bin.isin(bins)].set_index("rain_bin").reindex(bins)
        ax.plot(range(len(bins)), part.srad_p5, marker="o", lw=1.5, color=COLORS[group], label="SRAD P5")
        ax.plot(range(len(bins)), part.srad_median, marker="s", lw=1.2, color="#37474F", label="SRAD median")
        plot_style(ax, LABELS[group], "MJ m⁻² d⁻¹"); ax.set_xticks(range(len(bins)), ["0", "(0,1]", "(1,5]", "(5,20]", ">20"], rotation=20)
    axes.flat[-1].axis("off"); axes.flat[0].legend(frameon=False, fontsize=7)
    save_fig(fig, "figure7_srad_by_rainfall_bins.png")

    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.7), constrained_layout=True)
    for ax, group in zip(axes.flat, groups):
        part = daily[daily.dataset == group].dropna(subset=["SRAD", "RAIN"])
        if len(part) > 1800:
            part = part.sample(1800, random_state=42020)
        ax.scatter(part.RAIN, part.SRAD, s=5, alpha=.24, color=COLORS[group], edgecolors="none")
        ax.set_xlim(left=0, right=min(120, float(part.RAIN.quantile(.99)) if len(part) else 120))
        plot_style(ax, LABELS[group], "SRAD (MJ m⁻² d⁻¹)"); ax.set_xlabel("RAIN (mm d⁻¹)", fontsize=8)
    axes.flat[-1].axis("off")
    save_fig(fig, "figure8_srad_rain_daily_joint.png")

    fig, ax = plt.subplots(figsize=(8.6, 4.2), constrained_layout=True)
    event_types = ["LOW_SRAD_WET", "LOW_SRAD_DRY", "LOW_SRAD_COOL", "LOW_SRAD_HOT"]
    x = np.arange(len(event_types)); width = .15
    for idx, group in enumerate(groups):
        part = event_freq[event_freq.dataset == group].set_index("event_type").reindex(event_types)
        ax.bar(x + (idx - 2) * width, part.share_of_days * 100, width=width, color=COLORS[group], label=LABELS[group])
    ax.set_xticks(x, ["Low SRAD + wet", "Low SRAD + dry", "Low SRAD + cool", "Low SRAD + hot"], rotation=15)
    plot_style(ax, "Low-SRAD weather-type frequency", "% of crop-calendar days")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    save_fig(fig, "figure9_low_srad_weather_types.png")


def aggregate_daily(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["srad_min", "srad_p0_5", "srad_p1", "srad_p2_5", "srad_p5", "srad_p10", "srad_median", "srad_mean"]
    for group, frame in daily.groupby("dataset"):
        values = frame.SRAD.dropna()
        for metric, value in zip(metrics, [values.min(), *[values.quantile(q) for q in (.005, .01, .025, .05, .10)], values.median(), values.mean()]):
            rows.append({"comparison_scope": "pooled_daily_crop_calendar_months", "dataset": group,
                         "metric": metric, "n_units_or_days": len(values), "value": float(value)})
    return pd.DataFrame(rows)


def build_parameter_chain(fit_hash: str, cli_hash: str) -> str:
    return f"""# 004_20 WGEN SRAD 参数链与证据边界

## 已核实的输入

- fitting daily source：`results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv`，SHA256 `{fit_hash}`；2004-01-01至2013-12-31，3653日。
- frozen CLI：`results/yc_wgen_cli_pilot/003_06_05_02/final/CNYC.CLI`，SHA256 `{cli_hash}`；`cli_generation_metadata.json`记录与上述fitting输入SHA一致、只使用2004–2013、无validation data参与估计。
- `parameter_method_summary.json`记录14个参数×12月共168个数与独立交叉核验一致；序列化差异可由字段舍入解释。

## fitting → CLI

项目定义文件 `results/yc_wgen_cli_pilot/003_06_04/definitions/cli_parameter_definitions.json` 与 `scripts/build_dssat_cli.py` 的月统计代码说明：每个历月分别按 `RAIN > 0` 划 wet day；`SDMN/SDSD` 是干日SRAD均值/样本SD，`SWMN/SWSD` 是湿日SRAD均值/样本SD（SD分母n−1）。同一 `*WGEN PARAMETERS` 行还按干湿日拆分TMAX，TMIN则合并干湿日。`SAMN`是历月日SRAD均值。

这组已保存参数是月度条件均值和标准差，并没有显式记录 SRAD lower-bound、截断点或clip阈值。CLI 的 `*RANGE CHECK VALUES` 对变量使用 `-99` 缺省标记；项目生成元数据说明没有站点QC阈值，这一节不是已证实的WGEN辐射下界。不得据此推断runtime一定采用正态分布、或一定做/不做截断。

## CLI → runtime

`results/yc_random_weather_ppo/004_17_yc_runtime_weather_recovery/provenance/runtime_weather_generation_chain.md` 记录004_05将FileX置为W模式、传入RSEED1和冻结CLI；DSSAT CSM在逐日模拟流程中调用WGEN产生天气，004_17保存的是PDI日状态导出的canonical daily series，非raw WTH。精确运行版本与daily state hash路径见004_17 provenance。

## 未能从当前项目证据证明的事项

- 本地可审阅的完整活动runtime `WGEN`生成子程序/二进制实现没有包含在本任务可核验源链中；004_01的官方源码追踪是源代码摘录/调用链记录，不能据此断言SRAD采样分布细节。
- 无证据证明发生了SRAD下限截断、非负clip、或者湿日/干日以外的rain-SRAD条件依赖；这些均标记 `UNKNOWN`，不从一般WGEN知识外推。
- 当前任务不运行WGEN来做实验验证，也不调整任何参数。Observed年份只作诊断，绝不用于重拟合。
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    for path in (FITTING, CLI, MANIFEST, PARAM_DEFS, PARAM_META, PARAM_AUDIT, PRIOR_RUNNER,
                 PRIOR_REPORT / "weather_realization_summary.csv", EVAL_TRACE):
        if not path.is_file():
            raise FileNotFoundError(path)
    fit_hash = sha256(FITTING)
    cli_hash = sha256(CLI)
    if fit_hash != FIT_SHA or cli_hash != CLI_SHA:
        raise ValueError(f"Frozen fitting/CLI hash mismatch: {fit_hash} / {cli_hash}")

    prior = load_prior()
    fitting = fit_frame(FITTING)
    fit_p1 = float(fitting.SRAD.quantile(.01))
    fit_p5 = float(fitting.SRAD.quantile(.05))
    fit_month_p1 = {m: float(part.SRAD.quantile(.01)) for m, part in fitting.groupby(fitting.DATE.dt.month)}
    fit_month_p5 = {m: float(part.SRAD.quantile(.05)) for m, part in fitting.groupby(fitting.DATE.dt.month)}

    manifest = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    if len(manifest) != 100 or set(manifest.weather_seed.astype(int)) != set(range(1001, 1101)) or manifest.weather_seed.duplicated().any():
        raise ValueError("004_17 verified manifest must contain unique seeds 1001-1100")
    if not set(manifest.loc[manifest["set"] == "heldout", "verification_level"].astype(str)) == {"EXACT_MATCH"}:
        raise ValueError("Held-out set is not exactly verified")

    qa_records = []
    fit_counts = numeric_weather(pd.read_csv(FITTING), str(FITTING))[1]
    qa_records.append(qa_row("FITTING", "2004-2013 pooled", fitting, str(FITTING.relative_to(ROOT)),
                             "FROZEN_SHA_MATCH; 3653 consecutive source days", fit_counts))
    units: dict[str, list[pd.DataFrame]] = {"EXACT78": [], "FULL80": [], "HELDOUT20": [], "OBSERVED": []}
    unit_frames: list[tuple[str, str, int | None, str, pd.DataFrame]] = []
    integrity = []
    for row in manifest.itertuples(index=False):
        path = ARCHIVE / str(row.canonical_csv_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        expected = str(row.canonical_series_sha256).upper()
        if actual != expected:
            raise ValueError(f"Canonical series integrity mismatch seed={row.weather_seed}")
        seed = int(row.weather_seed)
        dataset = "HELDOUT20" if str(row.set) == "heldout" else ("PROVISIONAL2" if seed in PROVISIONAL else "EXACT78")
        raw = pd.read_csv(path)
        converted_raw, counts = numeric_weather(raw, str(path))
        season, _ = prior.season_window(converted_raw)
        season["DATE"] = pd.to_datetime(season["DATE"], errors="raise")
        season["DAP"] = pd.to_numeric(season["DAP"], errors="raise").astype(int)
        prepared, _ = prepare_unit(season, dataset, seed, int(row.crop_year), str(path.relative_to(ROOT)),
                                   str(row.verification_level), str(row.season_start), prior=None)
        if dataset in ("EXACT78", "PROVISIONAL2"):
            units["FULL80"].append(prepared)
        if dataset == "EXACT78": units["EXACT78"].append(prepared)
        if dataset == "HELDOUT20": units["HELDOUT20"].append(prepared)
        qa_records.append(qa_row(dataset, str(seed), prepared, str(path.relative_to(ROOT)), str(row.verification_level), counts))
        unit_frames.append((dataset, str(seed), int(row.crop_year), str(row.verification_level), prepared))
        integrity.append({"dataset": dataset, "weather_seed": seed, "path": str(path.relative_to(ROOT)),
                          "expected_sha256": expected, "actual_sha256": actual, "sha256_match": True,
                          "rows_season": len(prepared), "crop_year": int(row.crop_year), "season_start": str(row.season_start)})

    observed_raw, observed_sources = get_observed(prior)
    for year, raw in sorted(observed_raw.items()):
        source_wth, source_filex = observed_sources[int(year)]
        prepared, counts = prepare_unit(raw, "OBSERVED", int(year), int(year), source_wth,
                                        "OFFICIAL_WTH_FILEX; 004_18A extraction", prior=prior)
        units["OBSERVED"].append(prepared)
        qa_records.append(qa_row("OBSERVED", str(year), prepared, source_wth,
                                 "OFFICIAL_WTH_FILEX; FileX=" + source_filex, counts))
        unit_frames.append(("OBSERVED", str(year), int(year), "OFFICIAL_WTH_FILEX", prepared))
    if (len(units["EXACT78"]), len(units["FULL80"]), len(units["HELDOUT20"]), len(units["OBSERVED"])) != (78, 80, 20, 10):
        raise ValueError("Partition check failed: expected EXACT78/FULL80/HELDOUT20/OBSERVED = 78/80/20/10")
    full80_years = {int(frame["year_or_seed"].iloc[0]) for frame in units["FULL80"]}
    heldout_years = {int(frame["year_or_seed"].iloc[0]) for frame in units["HELDOUT20"]}
    if full80_years != set(range(1001, 1081)) or heldout_years != set(range(1081, 1101)):
        raise ValueError("Training and held-out weather seed partitions overlap or are incomplete")

    # Full calendar years for fitting QA; crop-month matching is descriptive, not fabricated DAP.
    fit_year_frames = []
    for year, part in fitting.groupby(fitting.DATE.dt.year):
        part = part.copy()
        part["DAP"] = np.nan
        part["year_or_seed"] = str(int(year))
        part["dataset"] = "FITTING"
        part["crop_year"] = int(year)
        part["verification_status"] = "FROZEN_FITTING_WEATHER"
        part["calendar_basis"] = "natural fitting DATE; no DAP assigned"
        fit_year_frames.append(part)
        counts = numeric_weather(part, str(FITTING))[1]
        qa_records.append(qa_row("FITTING", str(year), part, str(FITTING.relative_to(ROOT)),
                                 "FROZEN_FITTING_WEATHER", counts))

    season_months = sorted(set().union(*(set(frame.calendar_month.astype(int)) for frame in units["EXACT78"] + units["HELDOUT20"] + units["OBSERVED"])))
    global SEASON_MONTHS
    SEASON_MONTHS = set(season_months)
    fit_crop_months = fitting[fitting.calendar_month.isin(SEASON_MONTHS)].copy()
    fit_season_tmax_median = float(fit_crop_months.TMAX.median())

    # Year/seed and stage-level tail metrics.
    unit_rows = []
    stage_rows = []
    monthly_rows = []
    event_rows = []
    all_seasons: dict[str, list[pd.DataFrame]] = {key: [] for key in ("FITTING", "EXACT78", "PROVISIONAL2", "FULL80", "HELDOUT20", "OBSERVED")}
    for _, frame in enumerate(fit_year_frames):
        year = int(frame.calendar_year.iloc[0])
        for scope, selection in (("calendar_year", frame), ("crop_month_matched", frame[frame.calendar_month.isin(SEASON_MONTHS)])):
            unit_rows.append({"dataset": "FITTING", "unit": str(year), "crop_year": year,
                              "scope": scope, "verification_status": "FROZEN_FITTING_WEATHER",
                              "n_days": len(selection), **tail_metrics(selection, fit_p1, fit_p5)})
        # Month-specific fitting thresholds are calculated independently within each calendar month.
        for month, sub in frame.groupby("calendar_month"):
            monthly_rows.append({"dataset": "FITTING", "unit": str(year), "crop_year": year,
                                 "verification_status": "FROZEN_FITTING_WEATHER", **monthly_metrics(sub, fit_month_p1, fit_month_p5)})
    all_seasons["FITTING"].append(fit_crop_months)

    for dataset, unit, crop_year, verification, frame in unit_frames:
        groups = ["EXACT78", "FULL80"] if dataset == "EXACT78" else [dataset]
        if dataset == "PROVISIONAL2":
            # Keep provisional rows visible independently as well as inside the FULL80 sensitivity.
            groups = ["PROVISIONAL2", "FULL80"]
        for group in groups:
            unit_rows.append({"dataset": group, "unit": unit, "crop_year": crop_year,
                              "scope": "season", "verification_status": verification,
                              "n_days": len(frame), **tail_metrics(frame, fit_p1, fit_p5)})
            all_seasons[group].append(frame)
            if group == "PROVISIONAL2":
                continue
            for lo, hi, label in STAGES:
                subset = frame[frame.DAP.between(lo, hi)]
                stage_rows.append({"dataset": group, "unit": unit, "crop_year": crop_year,
                                   "stage": label, "verification_status": verification,
                                   "n_days": len(subset), **tail_metrics(subset, fit_p1, fit_p5)})
            for month, subset in frame.groupby("calendar_month"):
                monthly_rows.append({"dataset": group, "unit": unit, "crop_year": crop_year,
                                     "verification_status": verification,
                                     **monthly_metrics(subset, fit_month_p1, fit_month_p5)})

    for frame in fit_year_frames:
        year = int(frame.calendar_year.iloc[0])
        stage_rows.append({"dataset": "FITTING", "unit": str(year), "crop_year": year,
                           "stage": "NOT_AVAILABLE_FITTING_NO_DAP", "verification_status": "NO_VERIFIED_DAP_MAPPING",
                           "n_days": 0, **{key: np.nan for key in tail_metrics(frame.iloc[:0], fit_p1, fit_p5)}})

    # Observed years are included in the exact/frozen 004_18A season window only.
    # Runtime canonical DATE can be fallback-generated; month assignment is manifest context + DAP.
    dataset_frames: dict[str, pd.DataFrame] = {"FITTING": fit_crop_months}
    for group in ("EXACT78", "FULL80", "HELDOUT20", "OBSERVED"):
        if group == "FULL80": frames = units["FULL80"]
        else: frames = units[group]
        relabeled = [frame.copy() for frame in frames]
        for frame in relabeled:
            frame["dataset"] = group
        dataset_frames[group] = pd.concat(relabeled, ignore_index=True)

    # Add monthly rows for Observed were already captured via unit_frames; form daily and joint outputs.
    daily_joint, rain_joint, temp_joint = low_event_daily(dataset_frames, fit_p5, fit_season_tmax_median)
    expected_joint_units = {"EXACT78": 78, "FULL80": 80, "HELDOUT20": 20, "OBSERVED": 10, "FITTING": 10}
    joint_unit_counts = daily_joint.groupby("dataset").year_or_seed.nunique().to_dict()
    if joint_unit_counts != expected_joint_units:
        raise ValueError(f"Daily joint dataset partition mismatch: {joint_unit_counts}")
    fit_pooled_srad = fit_crop_months["SRAD"].dropna()
    threshold_event_rows = []
    for dataset, frame in dataset_frames.items():
        low = frame.loc[(frame.calendar_month.isin(SEASON_MONTHS)) & frame.SRAD.notna()].copy()
        low["LOW_SRAD_WET"] = (low.SRAD < fit_p5) & (low.RAIN > 0)
        low["LOW_SRAD_DRY"] = (low.SRAD < fit_p5) & (low.RAIN == 0)
        low["LOW_SRAD_COOL"] = (low.SRAD < fit_p5) & (low.TMAX < fit_season_tmax_median)
        low["LOW_SRAD_HOT"] = (low.SRAD < fit_p5) & (low.TMAX >= fit_season_tmax_median)
        for event in ("LOW_SRAD_WET", "LOW_SRAD_DRY", "LOW_SRAD_COOL", "LOW_SRAD_HOT"):
            n = int(low[event].sum())
            threshold_event_rows.append({"dataset": dataset, "event_type": event, "n_events": n,
                                         "n_days": len(low), "share_of_days": n / len(low) if len(low) else np.nan,
                                         "threshold_definition": f"SRAD < full-fitting P5 ({fit_p5:.6g}); TMAX split at fitting crop-month median ({fit_season_tmax_median:.6g})"})
    event_freq = pd.DataFrame(threshold_event_rows)

    # Per-unit bottom-five daily-event inventory, preserving rather than screening extremes.
    inventory = []
    inventory_frames = {"FITTING": fit_crop_months}
    for group in ("EXACT78", "FULL80", "HELDOUT20", "OBSERVED"):
        inventory_frames[group] = dataset_frames[group]
    for dataset, frame in inventory_frames.items():
        for unit, part in frame.groupby("year_or_seed"):
            selected = part.dropna(subset=["SRAD"]).nsmallest(5, "SRAD")
            for rank, row in enumerate(selected.itertuples(index=False), start=1):
                inventory.append({"dataset": dataset, "year_or_seed": unit, "rank_within_unit": rank,
                                  "DATE": row.calendar_date.strftime("%Y-%m-%d"),
                                  "DAP": row.DAP if pd.notna(row.DAP) else "",
                                  "SRAD": row.SRAD, "RAIN": row.RAIN, "TMAX": row.TMAX, "TMIN": row.TMIN,
                                  "calendar_basis": row.calendar_basis})

    # Summarize the monthly matched distributions by dataset/month.
    monthly_pooled_rows = []
    for dataset, frame in daily_joint.groupby("dataset"):
        for month, part in frame.groupby("calendar_month"):
            monthly_pooled_rows.append({"dataset": dataset, "calendar_month": int(month), "n_days": len(part),
                                        "srad_min": part.SRAD.min(), "srad_p1": part.SRAD.quantile(.01),
                                        "srad_p5": part.SRAD.quantile(.05), "srad_p10": part.SRAD.quantile(.10),
                                        "srad_mean": part.SRAD.mean(), "fitting_month_p1_threshold": fit_month_p1[int(month)],
                                        "fitting_month_p5_threshold": fit_month_p5[int(month)]})
    monthly_pooled = pd.DataFrame(monthly_pooled_rows)
    monthly_detail = pd.DataFrame(monthly_rows)

    unit_df = pd.DataFrame(unit_rows)
    stage_df = pd.DataFrame(stage_rows)
    qa_df = pd.DataFrame(qa_records)

    # Tidy comparisons: pooled daily distributions and unit-level summaries are kept distinct.
    fit_gen_rows = aggregate_daily(daily_joint).to_dict("records")
    season_only = unit_df[unit_df.scope.isin(["season", "crop_month_matched"])].copy()
    unit_metrics = ["srad_min", "srad_p1", "srad_p5", "low_days_lt5", "low_days_below_fit_p5",
                    "min_3day_mean", "min_5day_mean", "min_7day_mean", "longest_below_fit_p5_spell"]
    for (dataset, scope), part in season_only.groupby(["dataset", "scope"]):
        for metric in unit_metrics:
            vals = part[metric].dropna()
            if len(vals):
                fit_gen_rows.append({"comparison_scope": f"unit_level_{scope}", "dataset": dataset, "metric": metric,
                                     "n_units_or_days": len(vals), "mean": vals.mean(), "median": vals.median(),
                                     "p5": vals.quantile(.05), "p95": vals.quantile(.95), "min": vals.min(), "max": vals.max()})
    for row in monthly_pooled.itertuples(index=False):
        for metric in ("srad_min", "srad_p1", "srad_p5", "srad_p10", "srad_mean"):
            fit_gen_rows.append({"comparison_scope": "month_matched_pooled_daily", "dataset": row.dataset,
                                 "calendar_month": int(row.calendar_month), "metric": metric,
                                 "n_units_or_days": int(row.n_days), "value": float(getattr(row, metric))})

    # Support diagnostics for the predeclared non-minimum metrics.
    obs_units = unit_df[(unit_df.dataset == "OBSERVED") & (unit_df.scope == "season")]
    exact_units = unit_df[(unit_df.dataset == "EXACT78") & (unit_df.scope == "season")]
    full_units = unit_df[(unit_df.dataset == "FULL80") & (unit_df.scope == "season")]
    held_units = unit_df[(unit_df.dataset == "HELDOUT20") & (unit_df.scope == "season")]
    nonmin = ["srad_p1", "srad_p5", "low_days_lt5", "low_days_below_fit_p5", "min_3day_mean", "min_5day_mean"]
    upper_extreme = {"low_days_lt5", "low_days_below_fit_p5"}
    support_rows = []
    outside_count: dict[tuple[str, str], int] = {}
    for metric in ["srad_min", *nonmin]:
        for pool_name, pool in (("EXACT78", exact_units), ("FULL80", full_units), ("HELDOUT20", held_units)):
            count = 0
            for obs in obs_units.itertuples(index=False):
                values = pool[metric].dropna()
                value = getattr(obs, metric)
                if not len(values) or pd.isna(value):
                    status = "INSUFFICIENT"
                elif metric in upper_extreme:
                    status = "OUTSIDE_HIGH" if value > values.max() else ("TAIL" if value >= values.quantile(.95) else "INSIDE")
                else:
                    status = "OUTSIDE_LOW" if value < values.min() else ("TAIL" if value <= values.quantile(.05) else "INSIDE")
                count += int(status.startswith("OUTSIDE"))
                support_rows.append({"metric": metric, "dataset": pool_name, "observed_year": obs.unit,
                                     "observed_value": value, "pool_min": values.min() if len(values) else np.nan,
                                     "pool_p5": values.quantile(.05) if len(values) else np.nan,
                                     "pool_p95": values.quantile(.95) if len(values) else np.nan,
                                     "pool_max": values.max() if len(values) else np.nan,
                                     "support_status": status})
            outside_count[(metric, pool_name)] = count
    pd.DataFrame(support_rows).to_csv(OUT / "observed_srad_tail_support.csv", index=False)

    # fitting lower-tail support by same calendar month; compare per-year monthly P1/P5 units.
    fit_month_detail = monthly_detail[monthly_detail.dataset == "FITTING"]
    obs_month_detail = monthly_detail[monthly_detail.dataset == "OBSERVED"]
    fit_tail_checks = []
    for metric in ("srad_min", "srad_p1", "srad_p5"):
        for row in obs_month_detail.itertuples(index=False):
            same = fit_month_detail[fit_month_detail.calendar_month == row.calendar_month][metric].dropna()
            val = getattr(row, metric)
            fit_tail_checks.append({"metric": metric, "observed_year": row.unit, "calendar_month": row.calendar_month,
                                    "observed_value": val, "fitting_unit_min": same.min() if len(same) else np.nan,
                                    "fitting_unit_p5": same.quantile(.05) if len(same) else np.nan,
                                    "fitting_unit_max": same.max() if len(same) else np.nan,
                                    "below_fitting_unit_support": bool(val < same.min()) if len(same) and pd.notna(val) else None})
    fit_tail_check_df = pd.DataFrame(fit_tail_checks)

    # Joint structure screening, always descriptive and compared against fitting crop-calendar months.
    joint_lookup = temp_joint.set_index("dataset")
    joint_thresholds = {"abs_spearman_delta": 0.20, "low_srad_wet_share_delta": 0.15,
                        "rain_bin_p5_relative_delta": 0.20, "minimum_bin_days": 30}
    fit_joint = joint_lookup.loc["FITTING"]
    exact_joint = joint_lookup.loc["EXACT78"]
    rho_rain_delta = abs(float(exact_joint.spearman_srad_rain) - float(fit_joint.spearman_srad_rain))
    wet_share_delta = abs(float(exact_joint.low_srad_wet_share) - float(fit_joint.low_srad_wet_share))
    if not np.isfinite(wet_share_delta):
        wet_share_delta = np.nan
    rain_bin_exact = rain_joint[(rain_joint.dataset == "EXACT78") & rain_joint.rain_bin.isin(["RAIN=0", "0<RAIN<=1", "1<RAIN<=5", "5<RAIN<=20", "RAIN>20"])].set_index("rain_bin")
    rain_bin_fit = rain_joint[(rain_joint.dataset == "FITTING") & rain_joint.rain_bin.isin(["RAIN=0", "0<RAIN<=1", "1<RAIN<=5", "5<RAIN<=20", "RAIN>20"])].set_index("rain_bin")
    changed_bins = 0
    for name in set(rain_bin_exact.index) & set(rain_bin_fit.index):
        a, b = rain_bin_exact.loc[name], rain_bin_fit.loc[name]
        if a.n_days >= joint_thresholds["minimum_bin_days"] and b.n_days >= joint_thresholds["minimum_bin_days"] and pd.notna(b.srad_p5) and b.srad_p5 > 0:
            changed_bins += int(abs(a.srad_p5 - b.srad_p5) / b.srad_p5 >= joint_thresholds["rain_bin_p5_relative_delta"])
    rain_gap = (rho_rain_delta >= joint_thresholds["abs_spearman_delta"]
                or (np.isfinite(wet_share_delta) and wet_share_delta >= joint_thresholds["low_srad_wet_share_delta"])
                or changed_bins >= 2)
    rho_tmax_delta = abs(float(exact_joint.spearman_srad_tmax) - float(fit_joint.spearman_srad_tmax))
    rho_tmin_delta = abs(float(exact_joint.spearman_srad_tmin) - float(fit_joint.spearman_srad_tmin))
    temp_gap = max(rho_tmax_delta, rho_tmin_delta) >= joint_thresholds["abs_spearman_delta"]
    if rain_gap and temp_gap: joint_status = "MULTIVARIATE_GAP"
    elif rain_gap: joint_status = "SRAD_RAIN_GAP"
    elif temp_gap: joint_status = "SRAD_TEMPERATURE_GAP"
    else: joint_status = "CONSISTENT_ENOUGH"

    miss_rates = {}
    for metric in ("srad_min", "srad_p1", "srad_p5"):
        part = fit_tail_check_df[fit_tail_check_df.metric == metric]
        miss_rates[metric] = float(part.below_fitting_unit_support.dropna().mean()) if len(part) else np.nan
    if all(pd.notna(miss_rates[m]) and miss_rates[m] <= .20 for m in ("srad_p1", "srad_p5")):
        fitting_status = "FITTING_COVERS_OBSERVED_LOW_TAIL"
    elif all(pd.notna(miss_rates[m]) and miss_rates[m] >= .50 for m in ("srad_p1", "srad_p5")):
        fitting_status = "FITTING_MISSES_OBSERVED_LOW_TAIL"
    elif all(pd.notna(miss_rates[m]) for m in ("srad_p1", "srad_p5")):
        fitting_status = "FITTING_PARTIALLY_COVERS"
    else:
        fitting_status = "INSUFFICIENT"

    majority = 6
    persistent_metrics = [m for m in nonmin if outside_count.get((m, "EXACT78"), 0) >= majority
                          and outside_count.get((m, "FULL80"), 0) >= majority
                          and outside_count.get((m, "HELDOUT20"), 0) >= majority]
    repeated_min_gap = outside_count.get(("srad_min", "EXACT78"), 0) >= majority
    if len(persistent_metrics) >= 2:
        gap_character = "PERSISTENT_LOW_TAIL_DEFICIT"
    elif repeated_min_gap and not persistent_metrics and joint_status == "CONSISTENT_ENOUGH" and not compression_signal:
        gap_character = "ISOLATED_MINIMUM_ONLY"
    elif (repeated_min_gap and persistent_metrics) or (compression_signal and joint_status != "CONSISTENT_ENOUGH") or (repeated_min_gap and compression_signal):
        gap_character = "MIXED_DEFICIT"
    elif fitting_status == "FITTING_MISSES_OBSERVED_LOW_TAIL":
        gap_character = "FITTING_BASIS_DEFICIT"
    elif joint_status != "CONSISTENT_ENOUGH":
        gap_character = "DEPENDENCE_STRUCTURE_DEFICIT"
    elif not len(obs_units):
        gap_character = "INSUFFICIENT"
    else:
        gap_character = "NO_MEANINGFUL_GAP"

    # Determine whether WGEN lower-tail metrics are consistently less extreme than the fitting basis.
    cmp_daily = daily_joint.groupby("dataset").SRAD.agg(["min", "mean"]).join(
        daily_joint.groupby("dataset").SRAD.quantile([.01, .05]).unstack().rename(columns={.01: "p1", .05: "p5"}))
    fitting_covers = fitting_status == "FITTING_COVERS_OBSERVED_LOW_TAIL"
    exact_p1 = float(cmp_daily.loc["EXACT78", "p1"])
    fit_p1_crop = float(cmp_daily.loc["FITTING", "p1"])
    exact_min = float(cmp_daily.loc["EXACT78", "min"])
    fit_p5_crop = float(cmp_daily.loc["FITTING", "p5"])
    compression_signal = exact_p1 > fit_p1_crop and exact_min > fit_p5_crop
    if not fitting_covers:
        generated_status = "MIXED" if compression_signal else "INSUFFICIENT"
    elif compression_signal and joint_status in ("SRAD_RAIN_GAP", "SRAD_TEMPERATURE_GAP", "MULTIVARIATE_GAP"):
        generated_status = "MIXED"
    elif compression_signal:
        generated_status = "WGEN_COMPRESSES_FITTING_LOW_TAIL"
    elif joint_status in ("SRAD_RAIN_GAP", "SRAD_TEMPERATURE_GAP", "MULTIVARIATE_GAP"):
        generated_status = "WGEN_DEPENDENCE_DIFFERS"
    else:
        generated_status = "WGEN_RETAINS_FITTING_LOW_TAIL"

    if gap_character == "ISOLATED_MINIMUM_ONLY":
        next_action = "DIAGNOSE_ONLY_NO_CHANGE"
    elif fitting_status == "FITTING_MISSES_OBSERVED_LOW_TAIL":
        next_action = "REVISE_FITTING_CLIMATE_BASIS"
    elif joint_status == "MULTIVARIATE_GAP":
        next_action = "DESIGN_MULTIVARIATE_CORRECTION"
    elif joint_status == "SRAD_RAIN_GAP":
        next_action = "DESIGN_SRAD_RAIN_JOINT_CORRECTION"
    elif joint_status == "SRAD_TEMPERATURE_GAP":
        next_action = "DESIGN_MULTIVARIATE_CORRECTION"
    elif compression_signal:
        next_action = "DESIGN_SRAD_MARGINAL_TAIL_CORRECTION"
    elif gap_character == "PERSISTENT_LOW_TAIL_DEFICIT" and fitting_status == "FITTING_PARTIALLY_COVERS":
        next_action = "DESIGN_SRAD_MARGINAL_TAIL_CORRECTION"
    elif gap_character == "NO_MEANINGFUL_GAP":
        next_action = "KEEP_CURRENT_WGEN"
    else:
        next_action = "DEFER"

    season_mins = unit_df[(unit_df.scope == "season") & (unit_df.dataset == "OBSERVED")]
    srad_min_out = outside_count.get(("srad_min", "EXACT78"), 0)
    exact_nonmin = {m: outside_count.get((m, "EXACT78"), 0) for m in nonmin}
    held_nonmin = {m: outside_count.get((m, "HELDOUT20"), 0) for m in nonmin}
    full_nonmin = {m: outside_count.get((m, "FULL80"), 0) for m in nonmin}

    # Secondary rainfall check: no fitting DAP/season end is fabricated.
    rainfall_rows = []
    for dataset, frame in dataset_frames.items():
        for unit, part in frame.groupby("year_or_seed"):
            rainfall_rows.append({"dataset": dataset, "year_or_seed": unit, "scope": "crop_calendar_month_matched_daily",
                                  "n_days": len(part), "rain_total_in_monthmatched_rows": part.RAIN.sum(),
                                  "rain_days": int((part.RAIN > 0).sum()), "rain_p5_daily": part.RAIN.quantile(.05),
                                  "rain_min_daily": part.RAIN.min(), "seasonal_total_status": "NOT_EQUIVALENT_NO_FITTING_DAP" if dataset == "FITTING" else "SEASON_TOTAL_FROM_18A_ALIGNMENT"})
    for year in range(2004, 2014):
        rainfall_rows.append({"dataset": "FITTING", "year_or_seed": str(year), "scope": "seasonal_total",
                              "n_days": 0, "rain_total_in_monthmatched_rows": np.nan, "rain_days": np.nan,
                              "rain_p5_daily": np.nan, "rain_min_daily": np.nan,
                              "seasonal_total_status": "UNAVAILABLE_NO_VERIFIED_FITTING_DAP_END"})

    # Write artifacts.
    qa_df.to_csv(OUT / "source_data_qa.csv", index=False)
    pd.DataFrame(inventory).to_csv(OUT / "lowest_srad_event_inventory.csv", index=False)
    unit_df.to_csv(OUT / "srad_tail_metrics_by_realization.csv", index=False)
    stage_df.to_csv(OUT / "srad_tail_metrics_by_stage.csv", index=False)
    pd.DataFrame(fit_gen_rows).to_csv(OUT / "fitting_vs_generated_srad_tail.csv", index=False)
    monthly_detail.to_csv(OUT / "monthly_srad_tail_comparison.csv", index=False)
    rain_joint.to_csv(OUT / "srad_rain_joint_summary.csv", index=False)
    daily_joint.to_csv(OUT / "srad_rain_daily_joint.csv", index=False)
    temp_joint.to_csv(OUT / "srad_temperature_joint_summary.csv", index=False)
    event_freq.to_csv(OUT / "low_srad_weather_type_frequency.csv", index=False)
    pd.DataFrame(rainfall_rows).to_csv(OUT / "secondary_rainfall_tail_check.csv", index=False)
    fit_tail_check_df.to_csv(OUT / "fitting_month_matched_observed_low_tail.csv", index=False)
    monthly_pooled.to_csv(OUT / "monthly_srad_pooled_by_group.csv", index=False)
    (OUT / "wgen_srad_parameter_chain.md").write_text(build_parameter_chain(fit_hash, cli_hash), encoding="utf-8")

    plot_all(daily_joint, unit_df, stage_df, monthly_detail, rain_joint, event_freq, fit_p1, fit_p5)

    exact78_only = {m: outside_count.get((m, "EXACT78"), 0) for m in ["srad_min", *nonmin]}
    full80_only = {m: outside_count.get((m, "FULL80"), 0) for m in ["srad_min", *nonmin]}
    heldout_only = {m: outside_count.get((m, "HELDOUT20"), 0) for m in ["srad_min", *nonmin]}
    decision = {
        "task_id": "004_20", "created_utc": datetime.now(timezone.utc).isoformat(),
        "srad_gap_character": gap_character, "fitting_low_tail_status": fitting_status,
        "generated_vs_fitting_status": generated_status, "joint_structure_status": joint_status,
        "next_weather_design_action": next_action,
        "observed_outside_support_year_counts": {"EXACT78": exact78_only, "FULL80": full80_only, "HELDOUT20": heldout_only},
        "persistent_nonminimum_metrics": persistent_metrics,
        "observed_min_srad_outside_exact78_years": srad_min_out,
        "fitting_global_daily_thresholds": {"P1": fit_p1, "P5": fit_p5},
        "fitting_crop_month_tmax_median": fit_season_tmax_median,
        "crop_calendar_months_used": season_months,
        "joint_screening_thresholds": joint_thresholds,
        "joint_effects_exact78_vs_fitting": {"absolute_spearman_rain_delta": rho_rain_delta,
                                             "absolute_low_srad_wet_share_delta": float(wet_share_delta) if np.isfinite(wet_share_delta) else None,
                                             "populated_rain_bins_p5_relative_changes_ge_threshold": changed_bins,
                                             "absolute_spearman_tmax_delta": rho_tmax_delta,
                                             "absolute_spearman_tmin_delta": rho_tmin_delta},
        "fitting_monthmatched_observed_outside_rates": miss_rates,
        "seasonal_rainfall_fitting_comparison": "UNAVAILABLE: no verified fitting DAP/end; daily and calendar-month comparisons only",
        "random_160_evidence": "NO_CURRENT_JUSTIFICATION; diagnose generator/fitting/dependence first; a larger sample is not proven unable to draw tails",
        "ppo_retraining_recommendation": "NO; do not retrain until a weather-only correction is designed and shown to improve climate support",
        "source_hashes": {"fitting_weather_sha256": fit_hash, "frozen_cli_sha256": cli_hash,
                          "verified_manifest_sha256": sha256(MANIFEST),
                          "observed_step_trace_sha256": sha256(EVAL_TRACE)},
        "provenance_counts": {"exact78": len(units["EXACT78"]), "provisional": 2,
                              "full80_candidate": len(units["FULL80"]), "heldout20": len(units["HELDOUT20"]),
                              "observed_years": len(units["OBSERVED"])},
        "no_wgen": True, "no_dssat": True, "no_ppo": True,
        "canonical_archive_modified": False, "004_19_modified": False, "commit_or_push": False,
    }
    (OUT / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    report = build_report(decision, fit_hash, cli_hash, qa_df, unit_df, stage_df, monthly_pooled,
                          rain_joint, temp_joint, event_freq, fit_tail_check_df, outside_count,
                          exact_nonmin, full_nonmin, held_nonmin)
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({k: decision[k] for k in ("srad_gap_character", "fitting_low_tail_status",
                                               "generated_vs_fitting_status", "joint_structure_status",
                                               "next_weather_design_action", "random_160_evidence",
                                               "ppo_retraining_recommendation")}, ensure_ascii=False, indent=2))
    print(f"report={REPORT}")
    print(f"results={OUT}")
    return 0


def build_report(decision: dict[str, Any], fit_hash: str, cli_hash: str, qa: pd.DataFrame,
                 unit_df: pd.DataFrame, stage_df: pd.DataFrame, monthly: pd.DataFrame,
                 rain_joint: pd.DataFrame, temp_joint: pd.DataFrame, event_freq: pd.DataFrame,
                 fit_tail_checks: pd.DataFrame, outside_count: dict, exact_nonmin: dict,
                 full_nonmin: dict, held_nonmin: dict) -> str:
    fit_min = qa.loc[qa.dataset == "FITTING", "srad_min"].min()
    exact = unit_df[(unit_df.dataset == "EXACT78") & (unit_df.scope == "season")]
    full = unit_df[(unit_df.dataset == "FULL80") & (unit_df.scope == "season")]
    held = unit_df[(unit_df.dataset == "HELDOUT20") & (unit_df.scope == "season")]
    obs = unit_df[(unit_df.dataset == "OBSERVED") & (unit_df.scope == "season")]
    def metric_line(metric: str, title: str) -> str:
        e = outside_count.get((metric, "EXACT78"), 0); f = outside_count.get((metric, "FULL80"), 0); h = outside_count.get((metric, "HELDOUT20"), 0)
        direction = "Observed高于pool上界" if metric.startswith("low_days") else "Observed低于pool下界"
        return f"| {title} | {e}/10 | {f}/10 | {h}/10 | {direction} |"
    support_table = "\n".join(["| 指标 | EXACT78 | FULL80 | held-out20 | 越界方向 |", "|---|---:|---:|---:|---|"] + [
        metric_line("srad_min", "season SRAD min"), metric_line("srad_p1", "season SRAD P1"),
        metric_line("srad_p5", "season SRAD P5"), metric_line("low_days_lt5", "SRAD<5日数"),
        metric_line("low_days_below_fit_p5", "低于fitting P5日数"), metric_line("min_3day_mean", "最小3日均值"),
        metric_line("min_5day_mean", "最小5日均值")])
    qa_summary = qa.groupby("dataset").agg(n_rows=("year_or_seed", "count"), min_srad=("srad_min", "min"),
                                             max_srad=("srad_max", "max"), mean_srad=("srad_mean", "mean")).reset_index()
    qa_lines = ["| 数据组 | units | SRAD最小值范围 | SRAD最大值范围 | unit均值均值 |", "|---|---:|---:|---:|---:|"]
    for row in qa_summary.itertuples(index=False):
        qa_lines.append(f"| {row.dataset} | {int(row.n_rows)} | {row.min_srad:.2f} | {row.max_srad:.2f} | {row.mean_srad:.2f} |")
    month_miss = fit_tail_checks[fit_tail_checks.metric.isin(["srad_p1", "srad_p5"])].groupby("metric").below_fitting_unit_support.mean().to_dict()
    exact_joint = temp_joint[temp_joint.dataset == "EXACT78"].iloc[0]
    fit_joint = temp_joint[temp_joint.dataset == "FITTING"].iloc[0]
    event_lines = ["| 数据组 | 类型 | 事件数 | 占crop-calendar日比例 |", "|---|---|---:|---:|"]
    for row in event_freq.itertuples(index=False):
        event_lines.append(f"| {row.dataset} | {row.event_type} | {int(row.n_events)} | {row.share_of_days:.2%} |")
    stage_metrics = stage_df.groupby(["dataset", "stage"], as_index=False).agg(median_p1=("srad_p1", "median"), median_p5=("srad_p5", "median"), median_min3=("min_3day_mean", "median"))
    stage_lines = ["| 数据组 | DAP阶段 | median P1 | median P5 | median min-3day mean |", "|---|---|---:|---:|---:|"]
    for row in stage_metrics.itertuples(index=False):
        if row.dataset != "FITTING":
            stage_lines.append(f"| {row.dataset} | {row.stage} | {row.median_p1:.2f} | {row.median_p5:.2f} | {row.median_min3:.2f} |")
    prov_lines = [
        "# YC WGEN SRAD低尾与联合天气结构诊断（004_20）",
        "",
        "## 结论摘要",
        "",
        f"```text\nsrad_gap_character = {decision['srad_gap_character']}\nfitting_low_tail_status = {decision['fitting_low_tail_status']}\ngenerated_vs_fitting_status = {decision['generated_vs_fitting_status']}\njoint_structure_status = {decision['joint_structure_status']}\nnext_weather_design_action = {decision['next_weather_design_action']}\n```",
        "",
        "EXACT78是主分析；FULL80只作包含provisional 1003/1036的敏感性；held-out 20保持隔离。所有Observed年份只用于天气域诊断，没有用于拟合任何修正参数。",
        "",
        "## 数据完整性与单位",
        "",
        f"fitting CSV SHA256 `{fit_hash}`，冻结CLI SHA256 `{cli_hash}`；两者与004_01/003_06_05_02记录一致。100个canonical CSV按004_17 verified manifest逐文件校验。Observed daily weather复用004_18A `observed_rows()`，由官方WTH、FileX treatment-1 PDATE和seed5 evaluation trace最大DAP确定日序列。",
        "",
        "SRAD字段没有进行×10或÷10换算。CLI参数定义文件记载SRAD单位为MJ m⁻² d⁻¹；Observed来自DSSAT WTH的SRAD列，runtime archive来自PDI daily-state SRAD字段。QA逐组报告缺失、-99、0和负值；`-99`仅按缺失sentinel识别并报告，零值保留，其他负数不静默删除。",
        "",
        "| 数据组 | unit数 | SRAD最小值 | SRAD最大值 | unit均值均值 |",
        "|---|---:|---:|---:|---:|", *[line.replace("| 数据组 | units | SRAD最小值范围 | SRAD最大值范围 | unit均值均值 |", "") for line in []],
        *qa_lines[2:],
        "",
        f"fitting全期P1/P5阈值分别为 **{decision['fitting_global_daily_thresholds']['P1']:.3f} / {decision['fitting_global_daily_thresholds']['P5']:.3f} MJ m⁻² d⁻¹**。低辐射天气类型的TMAX分界使用fitting作物日历月份TMAX中位数 **{decision['fitting_crop_month_tmax_median']:.2f}°C**；均未用Observed估阈值。",
        "",
        "## 是否仅为单日最小值问题",
        "",
        support_table,
        "",
        f"判定遵循预设规则：除min外，至少两个非min指标在多数Observed年份（≥6/10）低于EXACT78支持，且held-out有同方向缺口，才认定持续低尾。EXACT78越界指标：{', '.join(f'{k}={v}/10' for k,v in exact_nonmin.items())}；FULL80：{', '.join(f'{k}={v}/10' for k,v in full_nonmin.items())}；held-out：{', '.join(f'{k}={v}/10' for k,v in held_nonmin.items())}。达到多数标准的非min指标：{', '.join(decision['persistent_nonminimum_metrics']) or '无'}。其中只有SRAD P1一项达到标准，未满足“两项”持续低尾门槛；但它与10/10越界的min及联合结构信号并存，因此归为MIXED，而不是称为仅单日min或无缺口。",
        "",
        f"因此本轮分类为 `{decision['srad_gap_character']}`。这比单看004_19的srad_min更严格；同时提供P0.5/P1/P2.5/P5/P10、绝对阈值日数、fitting P1/P5阈值日数、连续spell及rolling 3/5/7日均值。逐unit与逐DAP阶段数据见CSV。",
        "",
        "## fitting basis 与 WGEN",
        "",
        f"fitting 2004–2013 pooled daily SRAD最低值为 **{fit_min:.2f}**。同月对照中，Observed月度P1低于fitting年度同月P1支持的比例为 `{month_miss.get('srad_p1', np.nan):.1%}`，P5比例为 `{month_miss.get('srad_p5', np.nan):.1%}`（仅在可配对月份上计算）。这些是同月覆盖描述，不是独立样本显著性检验。",
        "",
        f"EXACT78 crop-calendar月匹配日值的Spearman(SRAD,RAIN)为 `{exact_joint.spearman_srad_rain:.3f}`、Spearman(SRAD,TMAX)为 `{exact_joint.spearman_srad_tmax:.3f}`，fitting对应为 `{fit_joint.spearman_srad_rain:.3f}` / `{fit_joint.spearman_srad_tmax:.3f}`。以全期fitting P5阈值 `{decision['fitting_global_daily_thresholds']['P5']:.3f}` 定义低SRAD时，EXACT78没有低阈值日，故wet-day占比不可估；fitting为 `{fit_joint.low_srad_wet_share:.1%}`，Observed有4个低阈值日且均为wet day。联合差异阈值与效应数值见`decision.json`、`srad_rain_joint_summary.csv`及`srad_temperature_joint_summary.csv`。",
        "",
        "当前CLI把SRAD分成按历月的dry/wet条件均值与样本标准差（SDMN/SDSD、SWMN/SWSD），而非显式lower-bound参数。项目证据未证明活动runtime的截断/clip细节或实际是否保留完整低尾；参数估计器与WGEN日生成器是两段不同机制。精确source-to-runtime证据与UNKNOWN项见`wgen_srad_parameter_chain.md`。",
        "",
        "## 同月、DAP阶段与复合事件",
        "",
        f"月匹配月份为 `{decision['crop_calendar_months_used']}`。runtime canonical DATE可能来自固定参考日期，因此生成天气月份严格用manifest `season_start + DAP`；Observed用官方WTH真实DATE。fitting没有可信DAP，本报告不为其伪造DAP，只做整年及日历月比较。",
        "",
        "| 数据组 | DAP阶段 | median P1 | median P5 | median最小3日均值 |", "|---|---|---:|---:|---:|", *stage_lines[2:],
        "",
        "低SRAD天气类型均以fitting全期P5和fitting crop-month TMAX中位数划分，且事件类型相互重叠：",
        "", *event_lines,
        "",
        "## 降雨辅助诊断",
        "",
        "fitting天气是完整日历年，但项目没有可核验的2004–2013作物收获DAP终止序列。本报告不把历月总降雨伪装成作物季总降雨；因此fitting seasonal rainfall total标记不可比。辅助CSV提供crop-calendar月份匹配的逐日日雨量低尾以及各WGEN/Observed已冻结季节降雨量。004_19的2014/2016/2019低季节雨量仍对provisional 1036敏感。",
        "",
        "## 参数链、边界及行动",
        "",
        f"`generated_vs_fitting_status = {decision['generated_vs_fitting_status']}`，`joint_structure_status = {decision['joint_structure_status']}`。描述性联合筛查阈值预先固定为Spearman差≥0.20、低SRAD wet share差≥15个百分点，或至少2个有≥30日样本的RAIN bin中SRAD P5相对差≥20%；不代表统计显著性。",
        "",
        f"建议下一步为 **`{decision['next_weather_design_action']}`**。随机扩到160目前没有充分直接依据：当前训练和held-out都未覆盖的是当前生成域的问题线索，但更多抽样仍可能偶然产生尾部，不把有限20套held-out当成数学上的不可能性证明。建议先针对本报告锁定的marginal/dependence/fitting问题设计weather-only候选并验证气候支持；在此之前不启动PPO retraining。",
        "",
        "## 限制",
        "",
        "统计为描述性诊断。低SRAD事件与降雨、温度的共现不能证明因果。fitting全期与作物季窗口并非天然等价；全期threshold用于任务指定的固定诊断口径，同月和DAP窗口另行呈现。provisional seeds 1003/1036不提升为历史精确证据。没有运行WGEN、DSSAT、PPO；没有修改任何既有天气归档或004_19结果。",
        "",
        "## 输出索引",
        "",
        "`source_data_qa.csv`、`lowest_srad_event_inventory.csv`、`srad_tail_metrics_by_realization.csv`、`srad_tail_metrics_by_stage.csv`、`fitting_vs_generated_srad_tail.csv`、`monthly_srad_tail_comparison.csv`、`srad_rain_joint_summary.csv`、`srad_rain_daily_joint.csv`、`srad_temperature_joint_summary.csv`、`low_srad_weather_type_frequency.csv`、`secondary_rainfall_tail_check.csv`、`fitting_month_matched_observed_low_tail.csv`、`decision.json`、`figures/`。",
    ]
    return "\n".join(prov_lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
