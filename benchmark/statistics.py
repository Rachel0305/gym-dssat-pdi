"""Statistical helpers for the configurable benchmark framework.

The functions in this module are deliberately independent from gym-DSSAT.
They accept normalized :class:`pandas.DataFrame` objects, preserve missing
measurements as ``NA``, and never replace a zero denominator with infinity.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd


EFFICIENCY_ALIASES: dict[str, tuple[str, ...]] = {
    "yield_kg_ha": (
        "yield_kg_ha",
        "final_yield",
        "final_grain_kg_ha",
        "harvest_yield_kg_ha",
        "GWAD",
        "HWAM",
        "grnwt",
    ),
    "et_mm": ("et_mm", "ET_mm", "seasonal_et_mm", "ETCM", "etcm"),
    "irrigation_mm": (
        "irrigation_mm",
        "irrigation_executed_total_mm",
        "action_irrigation_total",
        "cumulative_irrigation",
        "totir",
        "IRCM",
    ),
    "nitrogen_kg_ha": (
        "nitrogen_kg_ha",
        "nitrogen_executed_total_kg_ha",
        "action_nitrogen_total",
        "cumulative_nitrogen",
        "NICM",
    ),
    "nitrogen_uptake_kg_ha": (
        "nitrogen_uptake_kg_ha",
        "n_uptake_kg_ha",
        "NUCM",
        "nucm",
    ),
}

EFFICIENCY_COLUMNS = (
    "WP_ET_kg_m3",
    "IWP_gross_kg_m3",
    "PFP_N_kg_kg",
    "NUtE_kg_kg",
)


def first_existing_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Return the first candidate present in *frame*, otherwise ``None``."""

    return next((name for name in candidates if name in frame.columns), None)


def numeric_series(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    *,
    index: pd.Index | None = None,
) -> pd.Series:
    """Return the first available numeric series or an all-NA float series."""

    target_index = frame.index if index is None else index
    column = first_existing_column(frame, candidates)
    if column is None:
        return pd.Series(np.nan, index=target_index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").reindex(target_index)


def safe_divide(
    numerator: pd.Series | np.ndarray | float | int,
    denominator: pd.Series | np.ndarray | float | int,
    *,
    scale: float = 1.0,
) -> pd.Series | np.ndarray | float:
    """Divide safely, returning ``NA`` where the denominator is zero/missing.

    Negative denominators are not silently discarded because this helper is
    also useful for signed differences.  Input validation should reject
    negative resource totals before reporting.
    """

    scalar = np.isscalar(numerator) and np.isscalar(denominator)
    num = np.asarray(numerator, dtype="float64")
    den = np.asarray(denominator, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.divide(
            num * scale,
            den,
            out=np.full(np.broadcast(num, den).shape, np.nan, dtype="float64"),
            where=np.isfinite(den) & (den != 0),
        )
    if scalar:
        return float(value)
    if isinstance(numerator, pd.Series):
        return pd.Series(value, index=numerator.index, dtype="float64")
    if isinstance(denominator, pd.Series):
        return pd.Series(value, index=denominator.index, dtype="float64")
    return value


def add_efficiency_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Add canonical water- and nitrogen-efficiency metrics.

    Formulas follow the project's frozen definitions:

    ``WP_ET = yield / ET / 10`` (kg m-3),
    ``IWP_gross = yield / irrigation / 10`` (kg m-3),
    ``PFP_N = yield / applied N`` (kg kg-1), and
    ``NUtE = yield / crop N uptake`` (kg kg-1).

    A zero or unavailable denominator yields ``NA``; no infinite value or
    pseudo-efficiency is fabricated.
    """

    result = frame.copy()
    yield_value = numeric_series(result, EFFICIENCY_ALIASES["yield_kg_ha"])
    et_value = numeric_series(result, EFFICIENCY_ALIASES["et_mm"])
    irrigation = numeric_series(result, EFFICIENCY_ALIASES["irrigation_mm"])
    nitrogen = numeric_series(result, EFFICIENCY_ALIASES["nitrogen_kg_ha"])
    uptake = numeric_series(result, EFFICIENCY_ALIASES["nitrogen_uptake_kg_ha"])

    calculated = {
        "WP_ET_kg_m3": safe_divide(yield_value, et_value, scale=0.1),
        "IWP_gross_kg_m3": safe_divide(yield_value, irrigation, scale=0.1),
        "PFP_N_kg_kg": safe_divide(yield_value, nitrogen),
        "NUtE_kg_kg": safe_divide(yield_value, uptake),
    }
    for name, values in calculated.items():
        values = pd.Series(values, index=result.index, dtype="float64")
        if name in result.columns:
            existing = pd.to_numeric(result[name], errors="coerce")
            result[name] = existing.where(existing.notna(), values)
        else:
            result[name] = values
    return result


def summarize_cross_seed(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str] | None = None,
    metric_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return descriptive cross-seed statistics without significance tests."""

    if frame.empty:
        return pd.DataFrame(
            columns=["station_code", "year", "scenario", "seed_count"]
        )

    default_groups = ("experiment_id", "station_code", "year", "scenario")
    groups = [name for name in (group_columns or default_groups) if name in frame.columns]
    if not groups:
        work = frame.assign(_all="all")
        groups = ["_all"]
    else:
        work = frame.copy()

    defaults = (
        "yield_kg_ha",
        "irrigation_mm",
        "nitrogen_kg_ha",
        "reward_total",
        *EFFICIENCY_COLUMNS,
    )
    metrics = [name for name in (metric_columns or defaults) if name in work.columns]
    for name in metrics:
        work[name] = pd.to_numeric(work[name], errors="coerce")

    grouped = work.groupby(groups, dropna=False, sort=False)
    rows: list[dict[str, Any]] = []
    for key, subset in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(zip(groups, key_tuple))
        row["seed_count"] = (
            int(subset["seed"].nunique(dropna=True)) if "seed" in subset.columns else len(subset)
        )
        for metric in metrics:
            values = pd.to_numeric(subset[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = values.mean() if not values.empty else np.nan
            row[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else np.nan
            row[f"{metric}_min"] = values.min() if not values.empty else np.nan
            row[f"{metric}_max"] = values.max() if not values.empty else np.nan
            row[f"{metric}_median"] = values.median() if not values.empty else np.nan
        mean_yield = row.get("yield_kg_ha_mean", np.nan)
        std_yield = row.get("yield_kg_ha_std", np.nan)
        row["yield_cross_seed_cv"] = (
            abs(std_yield / mean_yield)
            if pd.notna(std_yield) and pd.notna(mean_yield) and mean_yield != 0
            else np.nan
        )
        rows.append(row)

    result = pd.DataFrame(rows)
    if "_all" in result.columns:
        result = result.drop(columns="_all")
    return result


def normalize_scenario(value: Any) -> str:
    """Normalize common legacy scenario spellings for comparison matching."""

    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "null_zero": "null",
        "noop": "null",
        "no_op": "null",
        "record": "recorded",
        "recorded_expert": "recorded",
        "farmer": "recorded",
        "farmer_experience": "recorded",
        "auto": "dssat_auto",
        "dssat_automatic": "dssat_auto",
        "extension_expert": "official_extension_expert",
        "official_expert": "official_extension_expert",
        "expert": "official_extension_expert",
        "dqn_policy": "dqn",
    }
    return aliases.get(text, text)


def build_baseline_comparison(
    frame: pd.DataFrame,
    *,
    target_scenarios: Sequence[str] = ("dqn",),
    baseline_scenarios: Sequence[str] = (
        "null",
        "recorded",
        "dssat_auto",
        "official_extension_expert",
    ),
) -> pd.DataFrame:
    """Compare target scenarios with available baselines by station and year.

    Missing baselines produce explicit ``missing_baseline`` rows with ``NA``
    metrics, allowing downstream report generation to continue transparently.
    """

    base_columns = [
        "experiment_id",
        "station_code",
        "year",
        "seed",
        "target_scenario",
        "baseline_scenario",
        "comparison_status",
        "target_yield_kg_ha",
        "baseline_yield_kg_ha",
        "yield_diff_kg_ha",
        "yield_ratio",
        "irrigation_diff_mm",
        "nitrogen_diff_kg_ha",
        "WP_ET_diff_kg_m3",
        "IWP_gross_diff_kg_m3",
        "PFP_N_diff_kg_kg",
        "NUtE_diff_kg_kg",
    ]
    if frame.empty or "scenario" not in frame.columns:
        return pd.DataFrame(columns=base_columns)

    work = add_efficiency_metrics(frame)
    work["_scenario_key"] = work["scenario"].map(normalize_scenario)
    targets = {normalize_scenario(name) for name in target_scenarios}
    baselines = [normalize_scenario(name) for name in baseline_scenarios]
    target_rows = work[work["_scenario_key"].isin(targets)]
    if target_rows.empty:
        return pd.DataFrame(columns=base_columns)

    identity = [name for name in ("station_code", "year") if name in work.columns]
    rows: list[dict[str, Any]] = []
    numeric_pairs = {
        "yield_kg_ha": "yield_diff_kg_ha",
        "irrigation_mm": "irrigation_diff_mm",
        "nitrogen_kg_ha": "nitrogen_diff_kg_ha",
        "WP_ET_kg_m3": "WP_ET_diff_kg_m3",
        "IWP_gross_kg_m3": "IWP_gross_diff_kg_m3",
        "PFP_N_kg_kg": "PFP_N_diff_kg_kg",
        "NUtE_kg_kg": "NUtE_diff_kg_kg",
    }

    for _, target in target_rows.iterrows():
        pool = work
        for column in identity:
            pool = pool[pool[column].eq(target[column])]
        for baseline_name in baselines:
            candidates = pool[pool["_scenario_key"].eq(baseline_name)]
            if "seed" in candidates.columns and "seed" in target.index and pd.notna(target["seed"]):
                same_seed = candidates[candidates["seed"].eq(target["seed"])]
                if not same_seed.empty:
                    candidates = same_seed
            baseline = candidates.iloc[0] if not candidates.empty else None
            row: dict[str, Any] = {
                "experiment_id": target.get("experiment_id", pd.NA),
                "station_code": target.get("station_code", pd.NA),
                "year": target.get("year", pd.NA),
                "seed": target.get("seed", pd.NA),
                "target_scenario": target.get("scenario", "dqn"),
                "baseline_scenario": baseline_name,
                "comparison_status": "matched" if baseline is not None else "missing_baseline",
                "target_yield_kg_ha": target.get("yield_kg_ha", np.nan),
                "baseline_yield_kg_ha": (
                    baseline.get("yield_kg_ha", np.nan) if baseline is not None else np.nan
                ),
            }
            for metric, difference_name in numeric_pairs.items():
                target_value = pd.to_numeric(pd.Series([target.get(metric)]), errors="coerce").iloc[0]
                baseline_value = (
                    pd.to_numeric(pd.Series([baseline.get(metric)]), errors="coerce").iloc[0]
                    if baseline is not None
                    else np.nan
                )
                row[difference_name] = (
                    target_value - baseline_value
                    if pd.notna(target_value) and pd.notna(baseline_value)
                    else np.nan
                )
            row["yield_ratio"] = safe_divide(
                row["target_yield_kg_ha"], row["baseline_yield_kg_ha"]
            )
            rows.append(row)
    return pd.DataFrame(rows, columns=base_columns)

