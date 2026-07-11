"""Generic publication-ready plots for benchmark outputs.

Figure contract
---------------
Core claim: scenario performance must be interpretable jointly through yield,
resource use, efficiency, and reward rather than a single terminal number.
The figures use a quantitative-grid archetype, English labels for manuscript
reuse, 300 dpi PNG, and editable-text SVG.  Missing fields produce an explicit
"data unavailable" panel instead of a fabricated value or a pipeline failure.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .statistics import EFFICIENCY_COLUMNS, add_efficiency_metrics, normalize_scenario
from .summary_builder import normalize_daily_trajectory, normalize_season_summary

LOGGER = logging.getLogger(__name__)

SCENARIO_COLORS = {
    "null": "#333333",
    "recorded": "#C44E52",
    "dssat_auto": "#CCB974",
    "official_extension_expert": "#4C72B0",
    "dqn": "#3A7D44",
}

SCENARIO_ORDER = {
    "null": 0,
    "recorded": 1,
    "dssat_auto": 2,
    "official_extension_expert": 3,
    "dqn": 4,
}


def _pyplot():
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    return plt


def _save_pair(fig: Any, output_dir: Path, stem: str, dpi: int) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    svg = output_dir / f"{stem}.svg"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    return {"png": png, "svg": svg}


def _scenario_color(value: Any) -> str:
    return SCENARIO_COLORS.get(normalize_scenario(value), "#777777")


def _scenario_sort_key(value: Any) -> tuple[int, str]:
    normalized = normalize_scenario(value)
    return SCENARIO_ORDER.get(normalized, 99), str(value)


def _placeholder(ax: Any, title: str, message: str) -> None:
    ax.set_title(title, loc="left", fontweight="bold")
    ax.text(0.5, 0.5, message, ha="center", va="center", color="#555555", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _write_source(frame: pd.DataFrame, output_dir: Path, stem: str) -> Path:
    path = output_dir / f"{stem}_source.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def plot_baseline_comparison(
    season_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    dpi: int = 300,
) -> dict[str, Any]:
    """Plot mean yield by scenario with cross-seed standard deviations."""

    plt = _pyplot()
    destination = Path(output_dir)
    season = normalize_season_summary(season_df)
    yield_value = pd.to_numeric(season["yield_kg_ha"], errors="coerce")
    source = season.assign(yield_kg_ha=yield_value).dropna(subset=["scenario", "yield_kg_ha"])
    if not source.empty:
        summary = (
            source.groupby("scenario", dropna=False, sort=False)["yield_kg_ha"]
            .agg(mean="mean", std="std", n="count")
            .reset_index()
        )
        summary = summary.sort_values(
            "scenario", key=lambda series: series.map(_scenario_sort_key)
        ).reset_index(drop=True)
    else:
        summary = pd.DataFrame(columns=["scenario", "mean", "std", "n"])

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    if summary.empty:
        _placeholder(ax, "Scenario yield comparison", "Yield or scenario data unavailable")
    else:
        x = np.arange(len(summary))
        error = pd.to_numeric(summary["std"], errors="coerce").fillna(0).to_numpy()
        ax.bar(
            x,
            summary["mean"],
            yerr=error,
            capsize=3,
            color=[_scenario_color(value) for value in summary["scenario"]],
            edgecolor="#222222",
            linewidth=0.5,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(summary["scenario"].astype(str), rotation=20, ha="right")
        ax.set_ylabel("Grain yield (kg ha$^{-1}$)")
        ax.set_title("Scenario yield comparison", loc="left", fontweight="bold")
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    paths = _save_pair(fig, destination, "baseline_comparison", dpi)
    plt.close(fig)
    return {"files": paths, "source": _write_source(summary, destination, "baseline_comparison")}


def plot_reward_curve(
    daily_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    dpi: int = 300,
) -> dict[str, Any]:
    """Plot cumulative reward from step rewards for each available run."""

    plt = _pyplot()
    destination = Path(output_dir)
    daily = normalize_daily_trajectory(daily_df)
    daily["reward"] = pd.to_numeric(daily["reward"], errors="coerce")
    daily["dap"] = pd.to_numeric(daily["dap"], errors="coerce")
    source = daily.dropna(subset=["reward"]).copy()
    identity = [
        name
        for name in ("station_code", "year", "scenario", "seed")
        if name in source.columns
    ]
    if not source.empty:
        if not identity:
            source["_run"] = "run"
            identity = ["_run"]
        source["step"] = source.groupby(identity, dropna=False).cumcount()
        source["cumulative_reward"] = source.groupby(identity, dropna=False)["reward"].cumsum()

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    if source.empty:
        _placeholder(ax, "Cumulative reward", "Step reward data unavailable")
    else:
        for key, subset in source.groupby(identity, dropna=False, sort=False):
            key_tuple = key if isinstance(key, tuple) else (key,)
            label = " | ".join(str(item) for item in key_tuple)
            scenario = subset["scenario"].iloc[0] if "scenario" in subset.columns else "run"
            x = subset["dap"] if subset["dap"].notna().any() else subset["step"]
            ax.plot(x, subset["cumulative_reward"], label=label, color=_scenario_color(scenario), lw=1.4)
        ax.set_xlabel("DAP" if source["dap"].notna().any() else "Decision step")
        ax.set_ylabel("Cumulative reward")
        ax.set_title("Cumulative reward", loc="left", fontweight="bold")
        ax.grid(color="#E5E5E5", linewidth=0.7)
        ax.legend(fontsize=6, loc="best")
    source = source.drop(columns="_run", errors="ignore")
    paths = _save_pair(fig, destination, "reward_curve", dpi)
    plt.close(fig)
    source_columns = [
        name for name in (*identity, "dap", "step", "reward", "cumulative_reward") if name in source.columns
    ]
    return {
        "files": paths,
        "source": _write_source(source.loc[:, source_columns], destination, "reward_curve"),
    }


def plot_resource_comparison(
    season_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    dpi: int = 300,
) -> dict[str, Any]:
    """Plot irrigation and nitrogen inputs side by side."""

    plt = _pyplot()
    destination = Path(output_dir)
    season = normalize_season_summary(season_df)
    source = season[["scenario", "irrigation_mm", "nitrogen_kg_ha"]].copy()
    for column in ("irrigation_mm", "nitrogen_kg_ha"):
        source[column] = pd.to_numeric(source[column], errors="coerce")
    source = source.dropna(subset=["scenario"])
    summary = source.groupby("scenario", sort=False, dropna=False).mean(numeric_only=True).reset_index()
    if not summary.empty:
        summary = summary.sort_values(
            "scenario", key=lambda series: series.map(_scenario_sort_key)
        ).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), constrained_layout=True)
    specs = (
        ("irrigation_mm", "Irrigation (mm)", "Seasonal irrigation"),
        ("nitrogen_kg_ha", "Applied N (kg ha$^{-1}$)", "Seasonal nitrogen"),
    )
    for ax, (column, ylabel, title) in zip(axes, specs):
        available = not summary.empty and summary[column].notna().any()
        if not available:
            _placeholder(ax, title, "Resource data unavailable")
            continue
        x = np.arange(len(summary))
        ax.bar(
            x,
            summary[column],
            color=[_scenario_color(value) for value in summary["scenario"]],
            edgecolor="#222222",
            linewidth=0.5,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(summary["scenario"].astype(str), rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    paths = _save_pair(fig, destination, "resource_comparison", dpi)
    plt.close(fig)
    return {"files": paths, "source": _write_source(summary, destination, "resource_comparison")}


def plot_efficiency_comparison(
    season_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    dpi: int = 300,
) -> dict[str, Any]:
    """Plot four explicitly named water/nitrogen efficiency metrics."""

    plt = _pyplot()
    destination = Path(output_dir)
    season = add_efficiency_metrics(normalize_season_summary(season_df))
    source = season[["scenario", *EFFICIENCY_COLUMNS]].copy()
    for column in EFFICIENCY_COLUMNS:
        source[column] = pd.to_numeric(source[column], errors="coerce")
    source = source.dropna(subset=["scenario"])
    summary = source.groupby("scenario", sort=False, dropna=False).mean(numeric_only=True).reset_index()
    if not summary.empty:
        summary = summary.sort_values(
            "scenario", key=lambda series: series.map(_scenario_sort_key)
        ).reset_index(drop=True)
    labels = {
        "WP_ET_kg_m3": "WP_ET (kg m$^{-3}$)",
        "IWP_gross_kg_m3": "IWP_gross (kg m$^{-3}$)",
        "PFP_N_kg_kg": "PFP_N (kg kg$^{-1}$)",
        "NUtE_kg_kg": "NUtE (kg kg$^{-1}$)",
    }
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.0), constrained_layout=True)
    for ax, metric in zip(axes.flat, EFFICIENCY_COLUMNS):
        if summary.empty or not summary[metric].notna().any():
            _placeholder(ax, labels[metric], "Metric unavailable")
            continue
        x = np.arange(len(summary))
        ax.bar(
            x,
            summary[metric],
            color=[_scenario_color(value) for value in summary["scenario"]],
            edgecolor="#222222",
            linewidth=0.5,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(summary["scenario"].astype(str), rotation=25, ha="right")
        ax.set_ylabel(labels[metric])
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    fig.suptitle("Water and nitrogen efficiency", x=0.01, ha="left", fontweight="bold")
    paths = _save_pair(fig, destination, "efficiency_comparison", dpi)
    plt.close(fig)
    return {"files": paths, "source": _write_source(summary, destination, "efficiency_comparison")}


def plot_yield_resource_pareto(
    season_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    dpi: int = 300,
) -> dict[str, Any]:
    """Plot yield against irrigation and nitrogen without claiming optimality."""

    plt = _pyplot()
    destination = Path(output_dir)
    season = normalize_season_summary(season_df)
    source = season[["scenario", "yield_kg_ha", "irrigation_mm", "nitrogen_kg_ha"]].copy()
    for column in ("yield_kg_ha", "irrigation_mm", "nitrogen_kg_ha"):
        source[column] = pd.to_numeric(source[column], errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), constrained_layout=True)
    specs = (
        ("irrigation_mm", "Irrigation (mm)", "Yield–irrigation space"),
        ("nitrogen_kg_ha", "Applied N (kg ha$^{-1}$)", "Yield–nitrogen space"),
    )
    for ax, (resource, xlabel, title) in zip(axes, specs):
        available = source.dropna(subset=[resource, "yield_kg_ha"])
        if available.empty:
            _placeholder(ax, title, "Comparable data unavailable")
            continue
        for scenario, subset in available.groupby("scenario", dropna=False, sort=False):
            ax.scatter(
                subset[resource],
                subset["yield_kg_ha"],
                s=35,
                color=_scenario_color(scenario),
                edgecolor="#222222",
                linewidth=0.4,
                label=str(scenario),
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Grain yield (kg ha$^{-1}$)")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(color="#E5E5E5", linewidth=0.7)
        ax.legend(fontsize=6)
    paths = _save_pair(fig, destination, "yield_resource_pareto", dpi)
    plt.close(fig)
    return {"files": paths, "source": _write_source(source, destination, "yield_resource_pareto")}


def plot_action_timeline(
    daily_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    dpi: int = 300,
) -> dict[str, Any]:
    """Plot executed irrigation and nitrogen actions against DAP."""

    plt = _pyplot()
    destination = Path(output_dir)
    daily = normalize_daily_trajectory(daily_df)
    for column in ("dap", "action_irrigation", "action_nitrogen"):
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    source = daily.dropna(subset=["dap"]).copy()
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 4.8), sharex=True, constrained_layout=True)
    specs = (
        ("action_irrigation", "Irrigation (mm)", "Irrigation actions"),
        ("action_nitrogen", "Applied N (kg ha$^{-1}$)", "Nitrogen actions"),
    )
    identities = [name for name in ("scenario", "seed") if name in source.columns]
    for ax, (column, ylabel, title) in zip(axes, specs):
        available = source.dropna(subset=[column])
        if available.empty:
            _placeholder(ax, title, "Action data unavailable")
            continue
        groups = available.groupby(identities, dropna=False, sort=False) if identities else [("run", available)]
        for key, subset in groups:
            key_tuple = key if isinstance(key, tuple) else (key,)
            scenario = subset["scenario"].iloc[0] if "scenario" in subset.columns else "run"
            ax.vlines(
                subset["dap"],
                0,
                subset[column],
                color=_scenario_color(scenario),
                alpha=0.8,
                linewidth=1.1,
                label=" | ".join(str(item) for item in key_tuple),
            )
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
        ax.legend(fontsize=6)
    axes[-1].set_xlabel("DAP")
    paths = _save_pair(fig, destination, "action_timeline", dpi)
    plt.close(fig)
    keep = [
        name
        for name in ("station_code", "year", "scenario", "seed", "dap", "action_irrigation", "action_nitrogen")
        if name in source.columns
    ]
    return {"files": paths, "source": _write_source(source[keep], destination, "action_timeline")}


def build_plots(
    season_df: pd.DataFrame | None,
    daily_df: pd.DataFrame | None,
    output_dir: str | Path,
    *,
    action_df: pd.DataFrame | None = None,
    config: Mapping[str, Any] | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate the generic plot package and return all artifact paths."""

    del action_df, manifest  # reserved for station-specific adapters
    reporting = dict((config or {}).get("reporting", {})) if isinstance(config, Mapping) else {}
    dpi = int(reporting.get("dpi", 300))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    season = pd.DataFrame() if season_df is None else season_df
    daily = pd.DataFrame() if daily_df is None else daily_df
    builders = {
        "baseline_comparison": lambda: plot_baseline_comparison(season, destination, dpi=dpi),
        "reward_curve": lambda: plot_reward_curve(daily, destination, dpi=dpi),
        "resource_comparison": lambda: plot_resource_comparison(season, destination, dpi=dpi),
        "efficiency_comparison": lambda: plot_efficiency_comparison(season, destination, dpi=dpi),
        "yield_resource_pareto": lambda: plot_yield_resource_pareto(season, destination, dpi=dpi),
        "action_timeline": lambda: plot_action_timeline(daily, destination, dpi=dpi),
    }
    artifacts: dict[str, Any] = {}
    failures: list[dict[str, str]] = []
    for name, builder in builders.items():
        try:
            artifacts[name] = builder()
        except Exception as exc:  # continue other report-only outputs
            LOGGER.exception("Plot generation failed: %s", name)
            failures.append({"plot": name, "error_type": type(exc).__name__, "error": str(exc)})
    return {"artifacts": artifacts, "failures": failures, "output_dir": destination}

