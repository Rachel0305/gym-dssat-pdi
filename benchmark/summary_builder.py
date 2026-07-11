"""Build normalized CSV and Excel benchmark summaries.

This module is an adapter over standardized data frames.  It does not train or
evaluate an agent and therefore can regenerate tables from reusable legacy
results without touching existing experiment directories.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .statistics import (
    EFFICIENCY_ALIASES,
    add_efficiency_metrics,
    build_baseline_comparison,
    first_existing_column,
    summarize_cross_seed,
)

LOGGER = logging.getLogger(__name__)


SEASON_COLUMNS = (
    "experiment_id",
    "config_hash",
    "station_code",
    "year",
    "seed",
    "scenario",
    "checkpoint",
    "yield_kg_ha",
    "biomass_kg_ha",
    "irrigation_mm",
    "nitrogen_kg_ha",
    "nitrogen_uptake_kg_ha",
    "et_mm",
    "reward_total",
    "number_of_irrigation_events",
    "number_of_nitrogen_events",
    "water_budget_use_ratio",
    "nitrogen_budget_use_ratio",
    "WP_ET_kg_m3",
    "IWP_gross_kg_m3",
    "PFP_N_kg_kg",
    "NUtE_kg_kg",
)

DAILY_COLUMNS = (
    "experiment_id",
    "config_hash",
    "station_code",
    "year",
    "seed",
    "scenario",
    "checkpoint",
    "dap",
    "date",
    "istage",
    "vstage",
    "topwt",
    "grnwt",
    "xlai",
    "swfac",
    "nstres",
    "totir",
    "cumulative_irrigation",
    "cumulative_nitrogen",
    "action_irrigation",
    "action_nitrogen",
    "raw_action",
    "executed_action",
    "reward",
    "terminal_reward",
    "final_yield",
)

ACTION_COLUMNS = (
    "experiment_id",
    "config_hash",
    "station_code",
    "year",
    "seed",
    "scenario",
    "checkpoint",
    "action_count",
    "irrigation_mm",
    "nitrogen_kg_ha",
    "number_of_irrigation_events",
    "number_of_nitrogen_events",
    "nonzero_action_ratio",
)

FAILURE_COLUMNS = (
    "experiment_id",
    "config_hash",
    "station_code",
    "year",
    "seed",
    "stage",
    "status",
    "error_type",
    "error_message",
    "timestamp",
)

SEASON_ALIASES: dict[str, tuple[str, ...]] = {
    **EFFICIENCY_ALIASES,
    "biomass_kg_ha": (
        "biomass_kg_ha",
        "final_biomass_kg_ha",
        "CWAD",
        "CWAM",
        "topwt",
    ),
    "reward_total": (
        "reward_total",
        "unified_reward_total",
        "eval_reward_total",
        "economic_reward_total",
        "economic_reward_total_eval",
    ),
    "number_of_irrigation_events": (
        "number_of_irrigation_events",
        "irrigation_event_count",
    ),
    "number_of_nitrogen_events": (
        "number_of_nitrogen_events",
        "fertilizer_event_count",
        "nitrogen_event_count",
    ),
}


def _ensure_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = pd.NA
    ordered = list(columns) + [name for name in result.columns if name not in columns]
    return result.loc[:, ordered]


def _coalesce_aliases(frame: pd.DataFrame, aliases: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    result = frame.copy()
    for canonical, candidates in aliases.items():
        source = first_existing_column(result, candidates)
        if source is None:
            continue
        values = result[source]
        if canonical not in result.columns:
            result[canonical] = values
        else:
            result[canonical] = result[canonical].where(result[canonical].notna(), values)
    return result


def normalize_season_summary(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Normalize legacy season-level names while preserving source columns."""

    work = pd.DataFrame() if frame is None else frame.copy()
    work = _coalesce_aliases(work, SEASON_ALIASES)
    work = add_efficiency_metrics(work)
    return _ensure_columns(work, SEASON_COLUMNS)


def normalize_daily_trajectory(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Normalize daily/evaluation trajectories without inventing observations."""

    work = pd.DataFrame() if frame is None else frame.copy()
    aliases = {
        "action_irrigation": ("action_irrigation", "amir", "irrigation_action"),
        "action_nitrogen": ("action_nitrogen", "anfer", "nitrogen_action"),
        "reward": ("reward", "reward_step_unified", "economic_reward"),
        "final_yield": ("final_yield", "final_grain_kg_ha", "yield_kg_ha"),
    }
    work = _coalesce_aliases(work, aliases)
    return _ensure_columns(work, DAILY_COLUMNS)


def build_action_summary(
    daily_frame: pd.DataFrame,
    action_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create per-run action totals from an action table or daily trajectory."""

    source = action_frame.copy() if action_frame is not None and not action_frame.empty else daily_frame.copy()
    if source.empty:
        return pd.DataFrame(columns=ACTION_COLUMNS)
    source = _coalesce_aliases(
        source,
        {
            "action_irrigation": ("action_irrigation", "amir", "irrigation_action"),
            "action_nitrogen": ("action_nitrogen", "anfer", "nitrogen_action"),
        },
    )
    for column in ("action_irrigation", "action_nitrogen"):
        if column not in source.columns:
            source[column] = np.nan
        source[column] = pd.to_numeric(source[column], errors="coerce")

    group_candidates = (
        "experiment_id",
        "config_hash",
        "station_code",
        "year",
        "seed",
        "scenario",
        "checkpoint",
    )
    groups = [name for name in group_candidates if name in source.columns]
    if not groups:
        source = source.assign(_all="all")
        groups = ["_all"]

    rows: list[dict[str, Any]] = []
    for key, subset in source.groupby(groups, dropna=False, sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(zip(groups, key_tuple))
        irrigation = pd.to_numeric(subset["action_irrigation"], errors="coerce")
        nitrogen = pd.to_numeric(subset["action_nitrogen"], errors="coerce")
        known_action = irrigation.notna() | nitrogen.notna()
        nonzero = irrigation.fillna(0).ne(0) | nitrogen.fillna(0).ne(0)
        row.update(
            {
                "action_count": int(known_action.sum()),
                "irrigation_mm": irrigation.sum(min_count=1),
                "nitrogen_kg_ha": nitrogen.sum(min_count=1),
                "number_of_irrigation_events": int(irrigation.fillna(0).gt(0).sum()),
                "number_of_nitrogen_events": int(nitrogen.fillna(0).gt(0).sum()),
                "nonzero_action_ratio": (
                    float(nonzero[known_action].mean()) if known_action.any() else np.nan
                ),
            }
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    if "_all" in result.columns:
        result = result.drop(columns="_all")
    return _ensure_columns(result, ACTION_COLUMNS)


def build_daily_trajectory_index(daily_frame: pd.DataFrame) -> pd.DataFrame:
    """Create a compact workbook index instead of duplicating all daily rows."""

    groups = [
        name
        for name in (
            "experiment_id",
            "config_hash",
            "station_code",
            "year",
            "seed",
            "scenario",
            "checkpoint",
        )
        if name in daily_frame.columns
    ]
    if daily_frame.empty:
        return pd.DataFrame(columns=[*groups, "row_count", "dap_min", "dap_max"])
    if not groups:
        daily_frame = daily_frame.assign(_all="all")
        groups = ["_all"]
    rows: list[dict[str, Any]] = []
    for key, subset in daily_frame.groupby(groups, dropna=False, sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        dap = pd.to_numeric(subset.get("dap"), errors="coerce")
        row = dict(zip(groups, key_tuple))
        row.update(
            {
                "row_count": len(subset),
                "dap_min": dap.min() if dap.notna().any() else np.nan,
                "dap_max": dap.max() if dap.notna().any() else np.nan,
                "has_reward": bool(pd.to_numeric(subset.get("reward"), errors="coerce").notna().any()),
            }
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    return result.drop(columns="_all", errors="ignore")


def flatten_mapping(value: Mapping[str, Any] | None, prefix: str = "") -> pd.DataFrame:
    """Flatten a nested mapping into a two-column configuration index."""

    rows: list[dict[str, Any]] = []

    def walk(item: Any, key: str) -> None:
        if isinstance(item, Mapping):
            for child_key, child in item.items():
                walk(child, f"{key}.{child_key}" if key else str(child_key))
        elif isinstance(item, (list, tuple)):
            rows.append({"config_key": key, "config_value": json.dumps(item, ensure_ascii=False)})
        else:
            rows.append({"config_key": key, "config_value": item})

    walk(value or {}, prefix)
    return pd.DataFrame(rows, columns=["config_key", "config_value"])


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _artifact_manifest(
    paths: Mapping[str, Path],
    tables: Mapping[str, pd.DataFrame],
    supplied: pd.DataFrame | None,
) -> pd.DataFrame:
    generated = []
    for name, path in paths.items():
        generated.append(
            {
                "artifact_type": name,
                "path": str(path),
                "status": "generated",
                "row_count": len(tables[name]) if name in tables else pd.NA,
                "source": "benchmark.summary_builder",
                "reused": False,
            }
        )
    result = pd.DataFrame(generated)
    if supplied is not None and not supplied.empty:
        result = pd.concat([supplied.copy(), result], ignore_index=True, sort=False)
    return result


def build_summary_outputs(
    season_df: pd.DataFrame | None,
    daily_df: pd.DataFrame | None,
    output_dir: str | Path,
    *,
    action_df: pd.DataFrame | None = None,
    config: Mapping[str, Any] | None = None,
    manifest: Mapping[str, Any] | None = None,
    config_index_df: pd.DataFrame | None = None,
    failure_df: pd.DataFrame | None = None,
    reused_results_df: pd.DataFrame | None = None,
    result_manifest_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Generate the framework's canonical CSV files and Excel workbook.

    The caller controls ``output_dir``.  Existing files in unrelated legacy
    result directories are never read, modified, or deleted by this function.
    """

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    season = normalize_season_summary(season_df)
    daily = normalize_daily_trajectory(daily_df)
    action = build_action_summary(daily, action_df)
    cross_seed = summarize_cross_seed(season)
    baseline = build_baseline_comparison(season)
    failures = _ensure_columns(
        pd.DataFrame() if failure_df is None else failure_df.copy(), FAILURE_COLUMNS
    )
    config_index = (
        config_index_df.copy()
        if config_index_df is not None
        else flatten_mapping({"config": config or {}, "manifest": manifest or {}})
    )
    reused = pd.DataFrame() if reused_results_df is None else reused_results_df.copy()
    daily_index = build_daily_trajectory_index(daily)

    tables: dict[str, pd.DataFrame] = {
        "season_summary": season,
        "daily_trajectory": daily,
        "action_summary": action,
        "cross_seed_summary": cross_seed,
        "baseline_comparison": baseline,
        "failure_summary": failures,
    }
    paths: dict[str, Path] = {}
    for name, table in tables.items():
        paths[name] = _write_csv(table, destination / f"{name}.csv")

    result_manifest = _artifact_manifest(paths, tables, result_manifest_df)
    result_manifest_path = _write_csv(result_manifest, destination / "result_manifest.csv")
    tables["result_manifest"] = result_manifest
    paths["result_manifest"] = result_manifest_path

    workbook_path = destination / "benchmark_summary.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        season.to_excel(writer, sheet_name="season_summary", index=False)
        daily_index.to_excel(writer, sheet_name="daily_trajectory_index", index=False)
        action.to_excel(writer, sheet_name="action_summary", index=False)
        cross_seed.to_excel(writer, sheet_name="cross_seed_summary", index=False)
        baseline.to_excel(writer, sheet_name="baseline_comparison", index=False)
        config_index.to_excel(writer, sheet_name="config_index", index=False)
        failures.to_excel(writer, sheet_name="failure_summary", index=False)
        reused.to_excel(writer, sheet_name="reused_results", index=False)
    paths["benchmark_summary"] = workbook_path
    LOGGER.info("Benchmark summaries written to %s", destination)
    return {
        "tables": tables,
        "paths": paths,
        "excel_path": workbook_path,
        "daily_trajectory_index": daily_index,
        "config_index": config_index,
        "reused_results": reused,
    }


# Short alias used by runners and external scripts.
build_summaries = build_summary_outputs

