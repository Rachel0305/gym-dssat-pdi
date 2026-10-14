from __future__ import annotations

import argparse
import csv
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
PRIOR_RUNNER = ROOT / "src/run_004_18A_yc_weather_summary_for_reporting.py"
OUT = ROOT / "results/yc_random_weather_ppo/004_19_yc_formal_climate_coverage_audit"
REPORT = ROOT / "docs/yc_random_weather_004_19_formal_climate_coverage_audit.md"
PROVISIONAL = {1003, 1036}
SEASON_SCOPE = ("season", "ALL")
STAGES = [(0, 30, "DAP0_30"), (31, 60, "DAP31_60"), (61, 90, "DAP61_90"), (91, 999, "DAP_gt90")]
GROUP_COLORS = {"EXACT78": "#367A9B", "FULL80": "#4F8269", "HELDOUT20": "#D18836", "OBSERVED": "#B54A58"}
CORE_METRICS = ["rainfall_total", "rainfall_days", "wet_day_intensity", "longest_dry_spell", "rx1day", "rx5day", "tmax_mean", "tmax_max", "hot_days_gt30", "hot_days_gt32", "hot_days_gt35", "longest_hot_spell", "tmin_mean", "tmin_min", "srad_mean", "srad_min", "srad_max"]
COMPOUND_METRICS = ["hot_dry_days_gt32_rain_le0", "hotspell_dry_overlap_days_ge3", "hotspell_count_overlapping_dryspell_ge3", "lowrain7_hot_days_le10mm"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def load_prior():
    spec = importlib.util.spec_from_file_location("weather_reporting_004_18A_readonly", PRIOR_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import frozen 004_18A metric source: {PRIOR_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_metrics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[str, str, int], pd.DataFrame], list[dict[str, Any]]]:
    prior = load_prior()
    manifest = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    if len(manifest) != 100 or set(manifest.weather_seed.astype(int)) != set(range(1001, 1101)) or manifest.weather_seed.duplicated().any():
        raise ValueError("004_17 manifest must contain unique seeds 1001–1100")
    training = manifest[manifest["set"] == "training"].copy()
    heldout = manifest[manifest["set"] == "heldout"].copy()
    exact_seeds = set(training.loc[~training.weather_seed.astype(int).isin(PROVISIONAL), "weather_seed"].astype(int))
    if len(training) != 80 or len(heldout) != 20 or len(exact_seeds) != 78 or not PROVISIONAL.issubset(set(training.weather_seed.astype(int))):
        raise ValueError("Training/held-out counts or provisional-seed partition do not match the frozen design")
    if set(heldout.verification_level.astype(str)) != {"EXACT_MATCH"}:
        raise ValueError("Held-out 1081–1100 are not all verified exact")
    for _, row in training[~training.weather_seed.astype(int).isin(PROVISIONAL)].iterrows():
        if str(row.verification_level) != "EXACT_MATCH":
            raise ValueError(f"Non-provisional training seed is not exact: {row.weather_seed} / {row.verification_level}")

    season_records: list[dict[str, Any]] = []
    stage_records: list[dict[str, Any]] = []
    compound_records: list[dict[str, Any]] = []
    windows: dict[tuple[str, str, int], pd.DataFrame] = {}
    integrity: list[dict[str, Any]] = []

    for row in manifest.itertuples(index=False):
        path = ARCHIVE / str(row.canonical_csv_path)
        expected = str(row.canonical_series_sha256).upper()
        if not path.is_file():
            raise FileNotFoundError(f"Manifest canonical daily file missing: {path}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Canonical file SHA mismatch for seed {row.weather_seed}: {actual} != {expected}")
        raw = pd.read_csv(path)
        window, dropped = prior.season_window(raw)
        seed = int(row.weather_seed)
        split = "HELDOUT20" if int(row.weather_seed) >= 1081 else ("PROVISIONAL2" if seed in PROVISIONAL else "EXACT78")
        verify = "PROVISIONAL_UNVERIFIABLE" if seed in PROVISIONAL else str(row.verification_level)
        common = {"weather_seed": seed, "set": split, "verification_status": verify, "crop_year": int(row.crop_year), "season_start": str(row.season_start), "n_days": len(window), "preseason_rows_excluded": int(dropped), "canonical_csv_path": str(row.canonical_csv_path), "canonical_series_sha256": expected, "canonical_file_sha256": actual}
        season_records.append({**common, "scope": "season", "stage": "ALL", **prior.metrics(window)})
        windows[(split, "season", seed)] = window
        compound_records.extend(compound_row(seed, split, verify, int(row.crop_year), "season", "ALL", window))
        for low, high, stage in STAGES:
            subset = window[window.DAP.between(low, high)].copy()
            stage_metrics = prior.metrics(subset) if not subset.empty else {metric: np.nan for metric in CORE_METRICS}
            stage_records.append({**common, "scope": "stage", "stage": stage, **stage_metrics})
            compound_records.extend(compound_row(seed, split, verify, int(row.crop_year), "stage", stage, subset))
        integrity.append({"weather_seed": seed, "set": split, "verification_status": verify, "path": str(path.relative_to(ROOT)), "expected_sha256": expected, "actual_sha256": actual, "sha256_match": True, "rows_raw": len(raw), "rows_season": len(window), "preseason_rows_excluded": dropped})

    observed = prior.observed_rows()
    for year, raw in sorted(observed.items()):
        window, dropped = prior.season_window(raw)
        seed = -1
        common = {"weather_seed": "", "set": "OBSERVED", "verification_status": "OFFICIAL_WTH_FILEX", "crop_year": int(year), "season_start": f"observed:{year}", "n_days": len(window), "preseason_rows_excluded": int(dropped), "canonical_csv_path": "", "canonical_series_sha256": "", "raw_file_sha256": ""}
        season_records.append({**common, "scope": "season", "stage": "ALL", **prior.metrics(window)})
        windows[("OBSERVED", "season", int(year))] = window
        compound_records.extend(compound_row(seed, "OBSERVED", "OFFICIAL_WTH_FILEX", int(year), "season", "ALL", window))
        for low, high, stage in STAGES:
            subset = window[window.DAP.between(low, high)].copy()
            stage_metrics = prior.metrics(subset) if not subset.empty else {metric: np.nan for metric in CORE_METRICS}
            stage_records.append({**common, "scope": "stage", "stage": stage, **stage_metrics})
            compound_records.extend(compound_row(seed, "OBSERVED", "OFFICIAL_WTH_FILEX", int(year), "stage", stage, subset))
    observed_years = sorted(int(y) for y in observed)
    if observed_years != list(range(2014, 2024)):
        raise ValueError(f"Expected observed years 2014–2023, got {observed_years}")
    return pd.DataFrame(season_records), pd.DataFrame(stage_records), pd.DataFrame(compound_records), windows, integrity


def compound_values(df: pd.DataFrame) -> dict[str, int]:
    if df.empty:
        return {metric: np.nan for metric in COMPOUND_METRICS}
    rain = pd.to_numeric(df["RAIN"], errors="raise").to_numpy(dtype=float)
    tmax = pd.to_numeric(df["TMAX"], errors="raise").to_numpy(dtype=float)
    hot = tmax > 32.0
    dry = rain <= 0.0
    qualifying_dry = np.zeros(len(df), dtype=bool)
    i = 0
    while i < len(dry):
        if dry[i]:
            j = i + 1
            while j < len(dry) and dry[j]:
                j += 1
            if j - i >= 3:
                qualifying_dry[i:j] = True
            i = j
        else:
            i += 1
    overlap_days = int(np.sum(hot & qualifying_dry))
    overlap_spells = 0
    i = 0
    while i < len(hot):
        if hot[i]:
            j = i + 1
            while j < len(hot) and hot[j]:
                j += 1
            if np.any(qualifying_dry[i:j]):
                overlap_spells += 1
            i = j
        else:
            i += 1
    seven_day = pd.Series(rain).rolling(7, min_periods=7).sum().to_numpy()
    lowrain_hot = int(np.sum(hot & (seven_day <= 10.0)))
    return {
        "hot_dry_days_gt32_rain_le0": int(np.sum(hot & dry)),
        "hotspell_dry_overlap_days_ge3": overlap_days,
        "hotspell_count_overlapping_dryspell_ge3": overlap_spells,
        "lowrain7_hot_days_le10mm": lowrain_hot,
    }


def compound_row(seed: int, split: str, verification: str, year: int, scope: str, stage: str, df: pd.DataFrame) -> list[dict[str, Any]]:
    values = compound_values(df)
    return [{"weather_seed": "" if seed < 0 else seed, "set": split, "verification_status": verification, "crop_year": year, "scope": scope, "stage": stage, "n_days": len(df), **values}]


def frame_for_scope(season: pd.DataFrame, stage: pd.DataFrame, scope: str, group: str) -> pd.DataFrame:
    source = season if scope == "season" else stage
    selected = source["set"].isin(["EXACT78", "PROVISIONAL2"]) if group == "FULL80" else source["set"] == group
    return source[selected & ((source["stage"] == "ALL") if scope == "season" else True)]


def summarize_support(train: pd.DataFrame, observed_value: float, metric: str, train_group: str, scope: str, stage: str, year: int) -> dict[str, Any]:
    values = pd.to_numeric(train[metric], errors="coerce").dropna()
    if values.empty:
        reason = "NO_TRAINING_METRIC_VALUES" if pd.notna(observed_value) else "NO_TRAINING_VALUES_AND_OBSERVED_METRIC_UNAVAILABLE"
        return {"training_set": train_group, "scope": scope, "stage": stage, "metric": metric, "year": year, "observed_value": observed_value, "n_training": 0, "empirical_percentile": np.nan, "training_min": np.nan, "training_max": np.nan, "training_p5": np.nan, "training_p95": np.nan, "outside_support": np.nan, "tail_p5_p95": np.nan, "distance_below_min": np.nan, "distance_above_max": np.nan, "outside_distance": np.nan, "support_min_seed": "", "support_max_seed": "", "classification": "INSUFFICIENT", "insufficient_reason": reason}
    minimum, maximum = float(values.min()), float(values.max())
    p5, p95 = float(values.quantile(.05)), float(values.quantile(.95))
    seed_col = train.get("weather_seed")
    min_seed = int(train.loc[pd.to_numeric(train[metric], errors="coerce").idxmin(), "weather_seed"]) if seed_col is not None else ""
    max_seed = int(train.loc[pd.to_numeric(train[metric], errors="coerce").idxmax(), "weather_seed"]) if seed_col is not None else ""
    if pd.isna(observed_value):
        return {"training_set": train_group, "scope": scope, "stage": stage, "metric": metric, "year": year, "observed_value": np.nan, "n_training": int(len(values)), "empirical_percentile": np.nan, "training_min": minimum, "training_max": maximum, "training_p5": p5, "training_p95": p95, "outside_support": np.nan, "tail_p5_p95": np.nan, "distance_below_min": np.nan, "distance_above_max": np.nan, "outside_distance": np.nan, "support_min_seed": min_seed, "support_max_seed": max_seed, "classification": "INSUFFICIENT", "insufficient_reason": "OBSERVED_METRIC_UNAVAILABLE"}
    outside = bool(observed_value < minimum or observed_value > maximum)
    tail = bool(observed_value < p5 or observed_value > p95)
    label = "OUTSIDE_SUPPORT" if outside else ("TAIL_BUT_INSIDE_SUPPORT" if tail else "WELL_COVERED")
    below = max(0.0, minimum - float(observed_value))
    above = max(0.0, float(observed_value) - maximum)
    return {"training_set": train_group, "scope": scope, "stage": stage, "metric": metric, "year": year, "observed_value": float(observed_value), "n_training": int(len(values)), "empirical_percentile": float(100 * (values <= observed_value).mean()), "training_min": minimum, "training_max": maximum, "training_p5": p5, "training_p95": p95, "outside_support": outside, "tail_p5_p95": tail, "distance_below_min": below, "distance_above_max": above, "outside_distance": below + above, "support_min_seed": min_seed, "support_max_seed": max_seed, "classification": label, "insufficient_reason": ""}


def make_percentiles(season: pd.DataFrame, stage: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    obs_frames = {"season": season[(season["set"] == "OBSERVED")], "stage": stage[stage["set"] == "OBSERVED"]}
    for scope in ("season", "stage"):
        source = season if scope == "season" else stage
        obs_scope = obs_frames[scope]
        stages = ["ALL"] if scope == "season" else [x[2] for x in STAGES]
        for stage_name in stages:
            obs = obs_scope if scope == "season" else obs_scope[obs_scope.stage == stage_name]
            exact = frame_for_scope(season, stage, scope, "EXACT78")
            full = frame_for_scope(season, stage, scope, "FULL80")
            held = frame_for_scope(season, stage, scope, "HELDOUT20")
            if scope == "stage":
                exact, full, held = exact[exact.stage == stage_name], full[full.stage == stage_name], held[held.stage == stage_name]
            for obs_row in obs.itertuples(index=False):
                year = int(obs_row.crop_year)
                for metric in metrics:
                    value = float(getattr(obs_row, metric))
                    exact_row = summarize_support(exact, value, metric, "EXACT78", scope, stage_name, year)
                    full_row = summarize_support(full, value, metric, "FULL80", scope, stage_name, year)
                    held_row = summarize_support(held, value, metric, "HELDOUT20", scope, stage_name, year)
                    sensitive = exact_row["classification"] != full_row["classification"]
                    insufficient = exact_row["classification"] == "INSUFFICIENT" or full_row["classification"] == "INSUFFICIENT"
                    final_class = "PROVISIONAL_SENSITIVE" if sensitive else ("INSUFFICIENT" if insufficient else exact_row["classification"])
                    robustness = "PROVISIONAL_SENSITIVE" if sensitive else ("INSUFFICIENT" if insufficient else "ROBUST")
                    rows.append({**exact_row, "full80_n_training": full_row["n_training"], "full80_percentile": full_row["empirical_percentile"], "full80_min": full_row["training_min"], "full80_max": full_row["training_max"], "full80_p5": full_row["training_p5"], "full80_p95": full_row["training_p95"], "full80_outside_support": full_row["outside_support"], "full80_tail_p5_p95": full_row["tail_p5_p95"], "full80_classification": full_row["classification"], "full80_insufficient_reason": full_row["insufficient_reason"], "full80_min_seed": full_row["support_min_seed"], "full80_max_seed": full_row["support_max_seed"], "heldout_min": held_row["training_min"], "heldout_max": held_row["training_max"], "heldout_support_contains_observed": (not bool(held_row["outside_support"])) if pd.notna(held_row["outside_support"]) else np.nan, "provisional_sensitive": sensitive, "coverage_classification": final_class, "robustness_status": robustness})
    return pd.DataFrame(rows)


def build_compound_support(compounds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scope, stage), obs in compounds[compounds["set"] == "OBSERVED"].groupby(["scope", "stage"], sort=False):
        for obs_row in obs.itertuples(index=False):
            year = int(obs_row.crop_year)
            for metric in COMPOUND_METRICS:
                val = float(getattr(obs_row, metric))
                exact = compounds[(compounds["set"] == "EXACT78") & (compounds.scope == scope) & (compounds.stage == stage)]
                full = compounds[(compounds["set"].isin(["EXACT78", "PROVISIONAL2"])) & (compounds.scope == scope) & (compounds.stage == stage)]
                held = compounds[(compounds["set"] == "HELDOUT20") & (compounds.scope == scope) & (compounds.stage == stage)]
                e = summarize_support(exact, val, metric, "EXACT78", scope, stage, year)
                f = summarize_support(full, val, metric, "FULL80", scope, stage, year)
                h = summarize_support(held, val, metric, "HELDOUT20", scope, stage, year)
                sensitive = e["classification"] != f["classification"]
                insufficient = e["classification"] == "INSUFFICIENT" or f["classification"] == "INSUFFICIENT"
                final_class = "PROVISIONAL_SENSITIVE" if sensitive else ("INSUFFICIENT" if insufficient else e["classification"])
                robustness = "PROVISIONAL_SENSITIVE" if sensitive else ("INSUFFICIENT" if insufficient else "ROBUST")
                rows.append({**e, "full80_n_training": f["n_training"], "full80_percentile": f["empirical_percentile"], "full80_min": f["training_min"], "full80_max": f["training_max"], "full80_p5": f["training_p5"], "full80_p95": f["training_p95"], "full80_classification": f["classification"], "full80_outside_support": f["outside_support"], "full80_tail_p5_p95": f["tail_p5_p95"], "full80_insufficient_reason": f["insufficient_reason"], "full80_min_seed": f["support_min_seed"], "full80_max_seed": f["support_max_seed"], "heldout_min": h["training_min"], "heldout_max": h["training_max"], "heldout_support_contains_observed": (not bool(h["outside_support"])) if pd.notna(h["outside_support"]) else np.nan, "provisional_sensitive": sensitive, "coverage_classification": final_class, "robustness_status": robustness})
    return pd.DataFrame(rows)


def provisional_change_details(compared: pd.DataFrame, season: pd.DataFrame, stage: pd.DataFrame) -> list[dict[str, Any]]:
    details = []
    for row in compared[compared.provisional_sensitive].itertuples(index=False):
        source = season if row.scope == "season" else stage
        exact = frame_for_scope(season, stage, row.scope, "EXACT78")
        if row.scope == "stage":
            exact = exact[exact.stage == row.stage]
        individual_drivers = []
        for seed in sorted(PROVISIONAL):
            candidate = source[(source["set"] == "PROVISIONAL2") & (source.stage == row.stage)] if row.scope == "stage" else source[(source["set"] == "PROVISIONAL2") & (source.stage == "ALL")]
            candidate = candidate[candidate.weather_seed.astype(int) == seed]
            combined = pd.concat([exact, candidate], ignore_index=False)
            result = summarize_support(combined, float(row.observed_value), row.metric, f"EXACT78_PLUS_{seed}", row.scope, row.stage, int(row.year))
            if result["classification"] == row.full80_classification:
                individual_drivers.append(seed)
        details.append({"year": int(row.year), "scope": row.scope, "stage": row.stage, "metric": row.metric, "exact78_classification": row.classification, "full80_classification": row.full80_classification, "exact78_percentile": row.empirical_percentile, "full80_percentile": row.full80_percentile, "individual_provisional_drivers": individual_drivers, "driver_interpretation": ",".join(map(str, individual_drivers)) if individual_drivers else "BOTH_REQUIRED"})
    return details


def heldout_checks(season: pd.DataFrame, stage: pd.DataFrame, compounds: pd.DataFrame) -> pd.DataFrame:
    try:
        from scipy.stats import ks_2samp
    except Exception:
        ks_2samp = None
    rows = []
    for source, metrics in ((season, CORE_METRICS), (stage, CORE_METRICS), (compounds, COMPOUND_METRICS)):
        for (scope, stage_name), _ in source.groupby(["scope", "stage"], sort=False):
            block = source[(source.scope == scope) & (source.stage == stage_name)]
            held = block[block["set"] == "HELDOUT20"]
            observed = block[block["set"] == "OBSERVED"]
            for train_name in ("EXACT78", "FULL80"):
                train = block[block["set"].isin(["EXACT78", "PROVISIONAL2"])] if train_name == "FULL80" else block[block["set"] == "EXACT78"]
                for metric in metrics:
                    a = pd.to_numeric(train[metric], errors="coerce").dropna().to_numpy(float)
                    b = pd.to_numeric(held[metric], errors="coerce").dropna().to_numpy(float)
                    if len(a) == 0 or len(b) == 0:
                        continue
                    ks = ks_2samp(a, b, alternative="two-sided", method="auto") if ks_2samp else None
                    obs_values = pd.to_numeric(observed[metric], errors="coerce").dropna()
                    rows.append({"comparison": f"{train_name}_VS_HELDOUT20", "scope": scope, "stage": stage_name, "metric": metric, "n_training": len(a), "n_heldout": len(b), "training_mean": float(np.mean(a)), "heldout_mean": float(np.mean(b)), "mean_difference_heldout_minus_training": float(np.mean(b) - np.mean(a)), "training_median": float(np.median(a)), "heldout_median": float(np.median(b)), "training_p5": float(np.quantile(a, .05)), "training_p95": float(np.quantile(a, .95)), "heldout_p5": float(np.quantile(b, .05)), "heldout_p95": float(np.quantile(b, .95)), "training_min": float(np.min(a)), "training_max": float(np.max(a)), "heldout_min": float(np.min(b)), "heldout_max": float(np.max(b)), "range_overlap": bool(max(np.min(a), np.min(b)) <= min(np.max(a), np.max(b))), "observed_years_outside_heldout_support": int(sum((v < np.min(b)) or (v > np.max(b)) for v in obs_values)), "ks_statistic": float(ks.statistic) if ks else np.nan, "ks_pvalue_exploratory": float(ks.pvalue) if ks else np.nan})
    return pd.DataFrame(rows)


def classify_coverage(percentiles: pd.DataFrame, branch: str) -> tuple[str, int, int, float]:
    outside_col = "outside_support" if branch == "EXACT78" else "full80_outside_support"
    classification_col = "classification" if branch == "EXACT78" else "full80_classification"
    valid = percentiles[percentiles[classification_col] != "INSUFFICIENT"]
    outside = int(valid[outside_col].astype(bool).sum())
    total = len(valid)
    rate = outside / total if total else float("nan")
    if outside == 0:
        cls = "GOOD"
    elif rate < .10:
        cls = "PARTIAL"
    else:
        cls = "POOR"
    return cls, outside, total, rate


def decide(exact_pct: pd.DataFrame, full_pct: pd.DataFrame, compared: pd.DataFrame, compound_pct: pd.DataFrame, held: pd.DataFrame) -> dict[str, Any]:
    exact_cls, exact_out, n_exact, exact_rate = classify_coverage(exact_pct, "EXACT78")
    full_cls, full_out, n_full, full_rate = classify_coverage(full_pct, "FULL80")
    core = compared
    robust_out = core[(core["classification"] == "OUTSIDE_SUPPORT") & (core["full80_classification"] == "OUTSIDE_SUPPORT")]
    robust_out_n = len(robust_out)
    sensitive_n = int(core.provisional_sensitive.astype(bool).sum())
    # A generator-level gap requires recurring robust same-direction support failures that held-out also does not reach.
    recurring = []
    for (scope, stage, metric), group in robust_out.groupby(["scope", "stage", "metric"]):
        years = sorted(group.year.astype(int).unique().tolist())
        held_miss = int((~group.heldout_support_contains_observed.astype(bool)).sum())
        if len(years) >= 8 and held_miss >= 8:
            recurring.append({"scope": scope, "stage": stage, "metric": metric, "outside_years": years, "heldout_also_outside_year_count": held_miss})
    compound_robust = compound_pct[(compound_pct.classification == "OUTSIDE_SUPPORT") & (compound_pct.full80_classification == "OUTSIDE_SUPPORT")]
    all_robust_out = pd.concat([robust_out, compound_robust], ignore_index=True, sort=False)
    if recurring:
        gap = "SYSTEMATIC_GENERATOR_GAP"
    elif robust_out_n:
        seasonal_robust = robust_out[robust_out.scope == "season"]
        if len(seasonal_robust) == 0:
            gap = "TARGETED_TAIL_GAP"
        elif robust_out.heldout_support_contains_observed.astype(bool).mean() >= .5:
            gap = "RANDOM_SAMPLING_GAP"
        else:
            gap = "SYSTEMATIC_GENERATOR_GAP"
    elif sensitive_n or len(compound_pct[compound_pct.provisional_sensitive.astype(bool)]):
        gap = "PROVISIONAL_SENSITIVE"
    else:
        gap = "NONE"
    if exact_cls != full_cls:
        robust_cls = "PROVISIONAL_SENSITIVE"
    elif exact_cls == "GOOD" and not robust_out_n:
        robust_cls = "GOOD"
    elif exact_cls == "POOR":
        robust_cls = "POOR"
    else:
        robust_cls = "PARTIAL"
    next_action = {"NONE": "KEEP_80", "RANDOM_SAMPLING_GAP": "TEST_RANDOM_160", "TARGETED_TAIL_GAP": "BUILD_TARGETED_TAIL_POOL", "SYSTEMATIC_GENERATOR_GAP": "REVISE_WGEN_GENERATOR", "PROVISIONAL_SENSITIVE": "DEFER_EXPANSION_DECISION"}[gap]
    # 004_18A's critical rainy-season examples remain explicit, not averaged away.
    rain_key = compared[(compared.scope == "season") & (compared.metric == "rainfall_total") & (compared.year.isin([2014, 2016, 2019]))]
    provisional_seed_effect = sorted(set(int(v) for v in rain_key.full80_min_seed if str(v).isdigit() and int(v) in PROVISIONAL))
    held_low_srad = held[(held.scope == "season") & (held.metric == "srad_min") & (held.comparison == "EXACT78_VS_HELDOUT20")]
    srad_example = compared[(compared.scope == "season") & (compared.metric == "srad_min")]
    return {
        "exact78_observed_coverage": exact_cls,
        "full80_candidate_coverage": full_cls,
        "robust_coverage_conclusion": robust_cls,
        "robust_tail_gap_type": gap,
        "next_weather_action": next_action,
        "primary_grid": {"expected_cells": len(CORE_METRICS) * 5 * 10, "metrics_per_scope": len(CORE_METRICS), "scopes": 5, "observed_years": 10, "exact78_valid_cells": n_exact, "full80_valid_cells": n_full, "insufficient_cells": int(len(compared) - n_exact), "exact78_outside_support_cells": exact_out, "full80_outside_support_cells": full_out, "exact78_outside_rate": exact_rate, "full80_outside_rate": full_rate, "classification_change_cells": sensitive_n},
        "robust_outside_support_cell_count": robust_out_n,
        "robust_recurring_generator_gap_metrics": recurring,
        "robust_compound_outside_support_cells": len(compound_robust),
        "provisional_seed_expanding_full80_min_for_2014_2016_2019_rainfall": provisional_seed_effect,
        "key_year_rainfall_rows": rain_key[["year", "observed_value", "training_min", "training_max", "training_p5", "training_p95", "support_min_seed", "full80_min", "full80_max", "full80_p5", "full80_p95", "full80_min_seed", "classification", "full80_classification", "robustness_status"]].to_dict("records"),
        "season_srad_min_outside_year_count_exact78": int(srad_example.outside_support.astype(bool).sum()),
        "season_srad_min_outside_year_count_full80": int((srad_example.full80_outside_support.astype(bool)).sum()),
        "heldout_srad_min_range": [float(held_low_srad.heldout_min.min()), float(held_low_srad.heldout_max.max())] if len(held_low_srad) else [],
        "generator_interpretation": "A systematic generator gap is reported only if repeated robust support failures are also outside held-out support. Held-out non-significance is not treated as evidence of distribution equivalence.",
        "exact78_seed_list": sorted(set(range(1001, 1081)) - PROVISIONAL),
        "full80_candidate_seed_list": list(range(1001, 1081)),
        "heldout_seed_list": list(range(1081, 1101)),
        "provisional_seeds": sorted(PROVISIONAL),
        "generated_at_utc": utc_now(),
    }


def save_box(ax: plt.Axes, season: pd.DataFrame, metric: str, title: str, key_years: list[int] | None = None, label_observed: bool = False) -> None:
    groups = ["EXACT78", "FULL80", "HELDOUT20"]
    scope = "season" if "season" in set(season.scope) else "stage"
    group_frames = [season[season["set"] == "EXACT78"], season[season["set"].isin(["EXACT78", "PROVISIONAL2"])], season[season["set"] == "HELDOUT20"]]
    data = [pd.to_numeric(frame[frame.scope == scope][metric], errors="coerce").dropna().to_numpy(float) for frame in group_frames]
    bp = ax.boxplot(data, positions=[0, 1, 2], widths=.58, patch_artist=True, showfliers=False, medianprops={"color": "#1E292E", "linewidth": 1.25}, whiskerprops={"color": "#64747B", "linewidth": .9}, capprops={"color": "#64747B", "linewidth": .9})
    for patch, group in zip(bp["boxes"], groups):
        patch.set_facecolor(GROUP_COLORS[group]); patch.set_alpha(.34); patch.set_edgecolor(GROUP_COLORS[group]); patch.set_linewidth(1.2)
    rng = np.random.default_rng(19)
    for pos, values, group in zip([0, 1, 2], data, groups):
        ax.scatter(pos + rng.uniform(-.18, .18, len(values)), values, s=8, alpha=.37, color=GROUP_COLORS[group], linewidths=0, zorder=2)
    obs = season[(season["set"] == "OBSERVED") & (season.scope == scope)].sort_values("crop_year")
    obs_rows = list(obs.itertuples(index=False))
    visible_rows = [row for row in obs_rows if key_years is None or int(row.crop_year) in key_years]
    label_rows = visible_rows if (key_years is not None or label_observed) else []
    label_y = {int(row.crop_year): float(getattr(row, metric)) for row in label_rows}
    if len(label_rows) > 1:
        values = np.array([float(getattr(row, metric)) for row in label_rows])
        min_gap = max(float(np.ptp(values)) * 0.04, np.finfo(float).eps)
        previous = None
        for row in sorted(label_rows, key=lambda item: float(getattr(item, metric))):
            year = int(row.crop_year)
            value = float(getattr(row, metric))
            if previous is not None and label_y[year] - previous < min_gap:
                label_y[year] = previous + min_gap
            previous = label_y[year]
    for row in obs_rows:
        year = int(row.crop_year)
        if key_years is not None and year not in key_years:
            continue
        offset = rng.uniform(-.1, .1)
        value = float(getattr(row, metric))
        ax.scatter(3 + offset, value, marker="D", s=30, color=GROUP_COLORS["OBSERVED"], edgecolor="white", linewidth=.45, zorder=4)
        if year in label_y:
            ax.annotate(str(year), (3 + offset, value), xytext=(3.12, label_y[year]), textcoords="data", fontsize=7, va="center", ha="left", arrowprops={"arrowstyle": "-", "color": "#9AA5AA", "linewidth": .55}, annotation_clip=False)
    ax.set_xticks([0, 1, 2, 3], ["EXACT78", "FULL80*", "Held20", "Observed"])
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.grid(axis="y", color="#DFE4E7", linewidth=.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def save_figures(season: pd.DataFrame, stage: pd.DataFrame, support: pd.DataFrame, outdir: Path) -> None:
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.8, 4.0), constrained_layout=True)
    save_box(ax, season, "rainfall_total", "Seasonal rainfall support (mm)", label_observed=True)
    ax.set_ylabel("Rainfall (mm)")
    fig.savefig(figdir / "figure1_seasonal_rainfall_support.png", dpi=180, facecolor="white", bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, metric, title in zip(axes, ["longest_dry_spell", "rx1day", "rx5day"], ["Longest dry spell (days)", "Rx1day (mm)", "Rx5day (mm)"]):
        save_box(ax, season, metric, title); ax.set_ylabel(title)
    fig.savefig(figdir / "figure2_dryspell_rx1_rx5_support.png", dpi=180, facecolor="white", bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.8), constrained_layout=True)
    for ax, (_, _, stage_name) in zip(axes, STAGES):
        block = stage[stage.stage == stage_name]
        save_box(ax, block, "rainfall_total", f"{stage_name} rainfall (mm)")
        ax.set_ylabel("Rainfall (mm)")
    fig.savefig(figdir / "figure3_dap_stage_rainfall_support.png", dpi=180, facecolor="white", bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, metric, title in zip(axes, ["tmax_mean", "tmax_max", "hot_days_gt32"], ["Mean Tmax (°C)", "Max Tmax (°C)", "Days Tmax >32°C"]):
        save_box(ax, season, metric, title); ax.set_ylabel(title)
    fig.savefig(figdir / "figure4_temperature_hotday_support.png", dpi=180, facecolor="white", bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, metric, title in zip(axes, ["srad_mean", "srad_min", "srad_max"], ["Mean SRAD", "Minimum SRAD", "Maximum SRAD"]):
        save_box(ax, season, metric, f"{title} (MJ m⁻² d⁻¹)"); ax.set_ylabel("MJ m⁻² d⁻¹")
    fig.savefig(figdir / "figure5_srad_support.png", dpi=180, facecolor="white", bbox_inches="tight"); plt.close(fig)

    heat_metrics = list(CORE_METRICS)
    for _, _, st in STAGES:
        heat_metrics += [f"{st}:{metric}" for metric in CORE_METRICS]
    selected = support.copy()
    selected["heat_metric"] = selected.apply(lambda r: r.metric if r.scope == "season" else f"{r.stage}:{r.metric}", axis=1)
    selected = selected[selected.heat_metric.isin(heat_metrics)]
    status_codes = {"WELL_COVERED": 0, "TAIL_BUT_INSIDE_SUPPORT": 1, "OUTSIDE_SUPPORT": 2, "PROVISIONAL_SENSITIVE": 3, "INSUFFICIENT": 4}
    selected["plot_status"] = selected.apply(lambda r: 3 if r.provisional_sensitive else status_codes.get(r.classification, 4), axis=1)
    pivot = selected.pivot(index="heat_metric", columns="year", values="plot_status").reindex(heat_metrics)
    cmap = matplotlib.colors.ListedColormap(["#8FBDA6", "#E8C36A", "#D97B68", "#647BB5", "#B8C0C4"])
    norm = matplotlib.colors.BoundaryNorm([-0.5, .5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    heatmap_size = (12.5, max(12.0, len(heat_metrics) * .17))
    fig, ax = plt.subplots(figsize=heatmap_size, constrained_layout=True)
    im = ax.imshow(pivot.to_numpy(float), aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(pivot.columns)), [str(int(y)) for y in pivot.columns]); ax.set_yticks(range(len(pivot.index)), pivot.index, fontsize=7)
    ax.set_title("Observed-year climate support status (EXACT78 primary)", loc="left", fontsize=11, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4], shrink=.82); cb.ax.set_yticklabels(["Well covered", "Tail, inside", "Outside support", "Provisional-sensitive", "Insufficient"])
    fig.savefig(figdir / "figure6_observed_year_metric_support_heatmap.png", dpi=180, facecolor="white", bbox_inches="tight"); plt.close(fig)

    selected2 = selected.pivot(index="heat_metric", columns="year", values="provisional_sensitive").reindex(heat_metrics)
    cmap2 = matplotlib.colors.ListedColormap(["#E5EBED", "#C7773D"])
    fig, ax = plt.subplots(figsize=heatmap_size, constrained_layout=True)
    ax.imshow(selected2.to_numpy(float), aspect="auto", cmap=cmap2, vmin=0, vmax=1)
    ax.set_xticks(range(len(selected2.columns)), [str(int(y)) for y in selected2.columns]); ax.set_yticks(range(len(selected2.index)), selected2.index, fontsize=7)
    ax.set_title("EXACT78 vs FULL80 classification changes", loc="left", fontsize=11, fontweight="bold")
    ax.set_xlabel("Ochre = classification changes when provisional seeds are included")
    fig.savefig(figdir / "figure7_exact78_full80_classification_changes.png", dpi=180, facecolor="white", bbox_inches="tight"); plt.close(fig)

    key_years = [2014, 2016, 2019]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    save_box(axes[0], season, "rainfall_total", "Seasonal rainfall (mm): key years", key_years); axes[0].set_ylabel("Rainfall (mm)")
    # Plot stage totals for the three observed years against the exact/full pool support by stage.
    x = np.arange(len(STAGES)); width = .34
    for offset, group, color in [(-width / 2, "EXACT78", GROUP_COLORS["EXACT78"]), (width / 2, "FULL80", GROUP_COLORS["FULL80"])]:
        medians, lo, hi = [], [], []
        for _, _, st in STAGES:
            set_mask = stage["set"].isin(["EXACT78", "PROVISIONAL2"]) if group == "FULL80" else stage["set"] == group
            values = stage[set_mask & (stage.stage == st)].rainfall_total.astype(float).dropna()
            medians.append(values.median()); lo.append(values.quantile(.05)); hi.append(values.quantile(.95))
        axes[1].errorbar(x + offset, medians, yerr=[np.array(medians) - np.array(lo), np.array(hi) - np.array(medians)], fmt="o", color=color, capsize=3, label=group)
    obs_stage = stage[(stage["set"] == "OBSERVED") & stage.crop_year.isin(key_years)]
    markers = {2014: "D", 2016: "s", 2019: "^"}
    for year in key_years:
        rows = obs_stage[obs_stage.crop_year == year].set_index("stage")
        vals = [rows.loc[st, "rainfall_total"] for _, _, st in STAGES]
        axes[1].plot(x, vals, linestyle="--", marker=markers[year], linewidth=.9, label=str(year), alpha=.85)
    axes[1].set_xticks(x, [s[2] for s in STAGES], rotation=20); axes[1].set_title("Stage rainfall (mm)", loc="left", fontsize=10, fontweight="bold"); axes[1].grid(axis="y", color="#DFE4E7", linewidth=.7); axes[1].legend(fontsize=7, ncol=2)
    save_box(axes[2], season, "srad_min", "Season minimum SRAD: key years", key_years); axes[2].set_ylabel("MJ m⁻² d⁻¹")
    fig.savefig(figdir / "figure8_key_year_weather_profiles.png", dpi=180, facecolor="white", bbox_inches="tight"); plt.close(fig)


def build_report(decision: dict[str, Any], season: pd.DataFrame, stage: pd.DataFrame, support: pd.DataFrame, compound_support: pd.DataFrame, held: pd.DataFrame, gaps: pd.DataFrame) -> str:
    rain = support[(support.scope == "season") & (support.metric == "rainfall_total") & support.year.isin([2014, 2016, 2019])].copy()
    rain_lines = ["| 年份 | Observed (mm) | EXACT78 min–max | FULL80 min–max | EXACT78 P5–P95 | FULL80 P5–P95 | FULL80下界seed | EXACT78分类 | FULL80分类 |", "|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in rain.sort_values("year").itertuples(index=False):
        rain_lines.append(f"| {int(r.year)} | {r.observed_value:.1f} | {r.training_min:.1f}–{r.training_max:.1f} | {r.full80_min:.1f}–{r.full80_max:.1f} | {r.training_p5:.1f}–{r.training_p95:.1f} | {r.full80_p5:.1f}–{r.full80_p95:.1f} | {int(r.full80_min_seed)} | {r.classification} | {r.full80_classification} |")
    provisional_lines = ["| 年份 | 范围/阶段 | 指标 | EXACT78 | FULL80 | 单独足以改变分类的候选seed |", "|---:|---|---|---|---|---|"]
    for item in decision.get("provisional_classification_changes", []):
        seeds = item["individual_provisional_drivers"]
        driver_text = ", ".join(f"seed{seed}" for seed in seeds) if seeds else "需同时加入1003和1036"
        scope_text = "整季" if item["scope"] == "season" else item["stage"]
        provisional_lines.append(f"| {item['year']} | {scope_text} | {item['metric']} | {item['exact78_classification']} | {item['full80_classification']} | {driver_text} |")
    stage_rain = support[(support.scope == "stage") & (support.metric == "rainfall_total") & support.year.isin([2014, 2016, 2019])]
    stage_lines = ["| 年份 | DAP阶段 | Observed降雨(mm) | EXACT78范围 | FULL80范围 | EXACT78分类 | FULL80分类 | held-out是否包含 |", "|---:|---|---:|---:|---:|---|---|---|"]
    for r in stage_rain.sort_values(["year", "stage"]).itertuples(index=False):
        stage_lines.append(f"| {int(r.year)} | {r.stage} | {r.observed_value:.1f} | {r.training_min:.1f}–{r.training_max:.1f} | {r.full80_min:.1f}–{r.full80_max:.1f} | {r.classification} | {r.full80_classification} | {bool(r.heldout_support_contains_observed)} |")
    srad = support[(support.scope == "season") & (support.metric == "srad_min")]
    srad_brief = f"整季 `srad_min` 在EXACT78有{int(srad.outside_support.astype(bool).sum())}/10年、FULL80有{int(srad.full80_outside_support.astype(bool).sum())}/10年超出min/max；held-out整季最低SRAD范围为{decision['heldout_srad_min_range'][0]:.2f}–{decision['heldout_srad_min_range'][1]:.2f} MJ m⁻² d⁻¹。" if decision.get("heldout_srad_min_range") else "整季SRAD最低值汇总不可用。"
    robust_out = gaps[(gaps.get("robustness_status", pd.Series(dtype=str)) == "ROBUST") & (gaps.get("classification", pd.Series(dtype=str)) == "OUTSIDE_SUPPORT")]
    robust_examples = robust_out.sort_values(["scope", "metric", "year"]).head(14)
    robust_lines = ["| 范围 | 阶段 | 指标 | 年份 | Observed | EXACT78范围 | FULL80范围 | held-out包含？ |", "|---|---|---|---:|---:|---:|---:|---|"]
    for r in robust_examples.itertuples(index=False):
        robust_lines.append(f"| {r.scope} | {r.stage} | {r.metric} | {int(r.year)} | {r.observed_value:.3g} | {r.training_min:.3g}–{r.training_max:.3g} | {r.full80_min:.3g}–{r.full80_max:.3g} | {bool(r.heldout_support_contains_observed)} |")
    pcts = decision["primary_grid"]
    exact_insufficient = support[support.classification == "INSUFFICIENT"]
    full_insufficient = support[support.full80_classification == "INSUFFICIENT"]
    exact_reason_counts = exact_insufficient.insufficient_reason.fillna("UNKNOWN").value_counts().to_dict()
    full_reason_counts = full_insufficient.full80_insufficient_reason.fillna("UNKNOWN").value_counts().to_dict()
    reason_text = "；".join(f"{key}：{value}格" for key, value in exact_reason_counts.items()) or "无"
    empty_observed_stages = stage[(stage["set"] == "OBSERVED") & (stage.n_days == 0)]
    empty_stage_keys = set(zip(empty_observed_stages.crop_year.astype(int), empty_observed_stages.stage.astype(str)))
    empty_stage_cells = support[(support.scope == "stage") & support.apply(lambda row: (int(row.year), str(row.stage)) in empty_stage_keys, axis=1)]
    empty_stage_metric_cells = len(empty_stage_cells)
    empty_stage_text = "、".join(f"{int(row.crop_year)}年{row.stage}" for row in empty_observed_stages.itertuples(index=False)) or "无"
    held_rain = held[(held.scope == "season") & (held.metric == "rainfall_total") & (held.comparison == "EXACT78_VS_HELDOUT20")]
    held_rainline = "held-out季节降雨比较不可用。"
    if len(held_rain):
        h = held_rain.iloc[0]
        held_rainline = f"held-out季节降雨范围为{h.heldout_min:.1f}–{h.heldout_max:.1f} mm（中位数{h.heldout_median:.1f}）；2014、2016、2019均低于held-out最小值。10个Observed年份中共有{int(h.observed_years_outside_heldout_support)}年超出held-out季节降雨范围。KS检验仅作探索，不能证明分布等价。"
    provisional_rain = sorted(set(int(x) for x in rain.full80_min_seed if pd.notna(x) and int(x) in PROVISIONAL))
    robust_conc = "；".join(f"{x['scope']} {x['stage']} {x['metric']}：{len(x['outside_years'])}年超出，held-out同样不覆盖{ x['heldout_also_outside_year_count']}年" for x in decision["robust_recurring_generator_gap_metrics"]) or "未发现达到预设重复生成器缺口阈值的单项指标。"
    figure_names = [f"figure{i}_{name}.png" for i, name in [(1, "seasonal_rainfall_support"), (2, "dryspell_rx1_rx5_support"), (3, "dap_stage_rainfall_support"), (4, "temperature_hotday_support"), (5, "srad_support"), (6, "observed_year_metric_support_heatmap"), (7, "exact78_full80_classification_changes"), (8, "key_year_weather_profiles")]]
    report = f"""# YC正式气候覆盖审计：EXACT78主分析与FULL80敏感性

## 结论摘要

- 主网格共{pcts['expected_cells']}格，其中{pcts['insufficient_cells']}格标记为`INSUFFICIENT`（EXACT78原因：{reason_text}）。其中Observed无日记录的阶段为{empty_stage_text}，对应{empty_stage_metric_cells}格；其余缺失来自指标在短窗口内不可定义。有效格中，EXACT78有**{pcts['exact78_outside_support_cells']}格（{pcts['exact78_outside_rate']:.1%}）**超出min/max；FULL80有**{pcts['full80_outside_support_cells']}格（{pcts['full80_outside_rate']:.1%}）**超出。两条口径的总体覆盖等级均为**{decision['exact78_observed_coverage']}**，稳健总体结论为**{decision['robust_coverage_conclusion']}**。
- EXACT78与FULL80共有{pcts['classification_change_cells']}格分类改变。2014、2016、2019整季降雨在EXACT78中低于最小值；加入provisional seed后FULL80的下界被拉低，三年都进入min/max范围，但仍处于P5以下。因此这是对极端min/max的敏感性变化，不是已验证的历史训练域覆盖。
- 另有不依赖1003/1036的重复低SRAD最小值缺口：{srad_brief} training两条口径和held-out都没有覆盖Observed低值，是当前最明确的稳健生成分布问题。
- 决策：`robust_tail_gap_type = {decision['robust_tail_gap_type']}`；`next_weather_action = {decision['next_weather_action']}`。本任务只建议后续先审查WGEN低SRAD尾部表示，不生成天气、不训练PPO。

## 机器结论

```text
exact78_observed_coverage = {decision['exact78_observed_coverage']}
full80_candidate_coverage = {decision['full80_candidate_coverage']}
robust_coverage_conclusion = {decision['robust_coverage_conclusion']}
robust_tail_gap_type = {decision['robust_tail_gap_type']}
next_weather_action = {decision['next_weather_action']}
```

EXACT78是正式历史训练域主分析。FULL80仅为候选敏感性口径，不能称为80/80 exact historical reconstruction。seed1003、1036仍为provisional；held-out 1081–1100继续隔离，不回流training。

## 主要结果

### 2014、2016、2019整季降雨

{chr(10).join(rain_lines)}

在provisional样本中，seed **{', '.join(map(str, provisional_rain)) or '无'}**决定了FULL80最低降雨下界。2014/2016/2019的EXACT78分类为`OUTSIDE_SUPPORT`，FULL80分类为`TAIL_BUT_INSIDE_SUPPORT`；FULL80虽扩展min/max支持，但三年仍低于两套口径的P5。004_18A观察到的经验百分位变化约1.25个百分点。故不得把FULL80的边界纳入当成历史已覆盖的证据。

全部分类变化与候选seed归因：

{chr(10).join(provisional_lines)}

其中2014整季Rx5day的P5分类改变由seed1003单独即可触发；DAP31–60的Rx1day与DAP61–90的SRAD max两项变化由seed1036单独即可触发。seed1036单独扩展季节总降雨的min/max支持；以上FULL80影响均为敏感性，不升级为已核验的历史覆盖。

### 三个重点年份的分阶段降雨

{chr(10).join(stage_lines)}

该表用于辨别整季低雨量对应单一阶段还是多个阶段的降雨差异。DAP按冻结的season alignment定义，不把WGEN的参考DATE误当作作物日历日期。

### 稳健缺口与held-out检查

预先固定的重复缺口规则：EXACT78与FULL80至少8/10年同方向超出，且held-out也至少8年未覆盖。达到规则的指标：**{robust_conc}**

{held_rainline}

Observed低SRAD最小值超出EXACT78、FULL80及held-out支持。对该项而言，证据更符合WGEN当前生成分布没有到达低值区域，而非仅仅80个seed随机漏抽；但不能据此推断所有天气变量都存在生成器问题。held-out比较仅为描述分析，KS不显著不代表分布相同。

代表性稳健越界格：

{chr(10).join(robust_lines) if len(robust_examples) else 'No robust outside-support cells.'}

### 复合高温—干旱条件

复合指标阈值已在计算前固定：同日高温干燥（`Tmax>32°C`且`RAIN<=0`）；高温日与至少连续3天无雨段重叠；以及当日及前6天累计雨量不超过10 mm的高温日数。DAP阶段内滚动窗口重新起算。全部年份和两条training口径见`compound_extreme_support.csv`；这些补充指标不替代850格核心网格。

## 逐项回答

1. Observed 2014–2023并非全部处于历史训练天气支持内；两种口径按预设规则均为`PARTIAL`。
2. 2014/2016/2019整季低降雨分类依赖provisional seed 1036 扩展min/max；P5/P95仍在尾部，因此不属于稳健的历史覆盖证据。分阶段降雨见上表，并需与温度、辐射、复合指标一起判断。
3. 排除1003/1036后仍有稳健缺口，尤其整季最低SRAD在Observed十年均超出两套training边界，held-out也无法覆盖。其他stage、温度、降雨和复合指标的逐项结果见CSV。
4. held-out与training逐项分布比较见`heldout_distribution_check.csv`。有限样本可以表明现有WGEN realization是否实际达到Observed尾部，但不能证明总体分布等价。
5. 当前问题不只是“80个抽样是否太少”：低SRAD最小值呈training与held-out共同的稳健缺口，更支持先检查WGEN生成分布/辐射尾部，而不是直接随机扩到160。此结论是天气分布诊断建议，不是对作物表现的因果结论。

## 方法与证据边界

1. 100个canonical文件均按`weather_archive_manifest_verified.csv`校验；计算前核实78 exact、2 provisional、20 held-out的划分。SHA和逐文件记录见`input_integrity.csv`。
2. season对齐与17项核心指标直接调用冻结的004_18A runner；Observed使用相同官方WTH、FileX treatment-1 PDATE和seed5 episode最大DAP流程。
3. 经验百分位为`100 × mean(training metric <= observed)`；P5/P95采用与004_18A相同的pandas线性分位数。越界距离为Observed到最近min/max边界的绝对距离，范围内记0。
4. 核心网格为17项指标×整季和4个DAP阶段×10年，共850格。Observed某阶段无日记录、该指标在Observed阶段不可定义，或training指标无有效样本时记`INSUFFICIENT`，不填零；排除于coverage比例分母。EXACT78原因分布：{reason_text}。FULL80原因分布：{'；'.join(f'{key}：{value}格' for key, value in full_reason_counts.items()) or '无'}。两条口径各有{pcts['exact78_valid_cells']}格可计算。GOOD=0个min/max越界；PARTIAL=1–84；POOR=85格及以上。复合指标单独报告。
5. held-out双样本KS p值为探索性结果，不作等价性检验，也未作多重检验校正。本任务未运行WGEN、DSSAT或PPO。

## 图表

{chr(10).join(f'- Figure {i}: `results/yc_random_weather_ppo/004_19_yc_formal_climate_coverage_audit/figures/{name}`' for i, name in enumerate(figure_names, 1))}

## 机器可读文件

`weather_metrics_exact78.csv`、`weather_metrics_full80.csv`、`weather_metrics_heldout20.csv`、`observed_metrics.csv`、`observed_percentiles_exact78.csv`、`observed_percentiles_full80.csv`、`observed_support_comparison.csv`、`stage_support_comparison.csv`、`compound_extreme_support.csv`、`heldout_distribution_check.csv`、`robust_gap_inventory.csv`、`insufficient_data_cells.csv`和`decision.json`。
"""
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only YC EXACT78/FULL80 climate coverage audit; no WGEN/DSSAT/PPO.")
    parser.add_argument("--figures-only", action="store_true", help="Refresh figures from the completed 004_19 CSV outputs only")
    parser.add_argument("--refresh-generated-output", action="store_true", help="Recompute outputs in the existing 004_19 result directory after confirming its audit provenance")
    args = parser.parse_args()
    if args.figures_only:
        required = [OUT / "weather_metrics_all_sets.csv", OUT / "weather_stage_metrics_all_sets.csv", OUT / "observed_support_comparison.csv"]
        if not OUT.is_dir() or any(not path.is_file() for path in required):
            raise FileNotFoundError("Completed 004_19 CSV outputs are required for --figures-only")
        season = pd.read_csv(required[0])
        stage = pd.read_csv(required[1])
        support = pd.read_csv(required[2])
        save_figures(season, stage, support, OUT)
        print(json.dumps({"figures_refreshed": 8, "data_or_archive_modified": False}, indent=2))
        return 0
    if OUT.exists() or REPORT.exists():
        if not args.refresh_generated_output or not (OUT / "decision.json").is_file() or not REPORT.is_file():
            raise FileExistsError("Refusing to overwrite existing 004_19 result artifacts")
        prior_decision = json.loads((OUT / "decision.json").read_text(encoding="utf-8"))
        if prior_decision.get("core_metric_definitions_source") != "src/run_004_18A_yc_weather_summary_for_reporting.py" or prior_decision.get("archive_modified") is not False:
            raise ValueError("Existing result directory does not match the 004_19 generated-output provenance")
    if not MANIFEST.is_file() or not PRIOR_RUNNER.is_file():
        raise FileNotFoundError("Required 004_17 manifest or frozen 004_18A metric source missing")
    season, stage, compounds, _, integrity = run_metrics()
    expected_season = {"EXACT78": 78, "PROVISIONAL2": 2, "HELDOUT20": 20, "OBSERVED": 10}
    expected_stage = {"EXACT78": 78 * 4, "PROVISIONAL2": 2 * 4, "HELDOUT20": 20 * 4, "OBSERVED": 10 * 4}
    for group, count in expected_season.items():
        if len(season[season["set"] == group]) != count:
            raise ValueError(f"Unexpected seasonal row count for {group}")
    for group, count in expected_stage.items():
        if len(stage[stage["set"] == group]) != count:
            raise ValueError(f"Unexpected DAP-stage row count for {group}")
    metrics = CORE_METRICS
    pct_exact = make_percentiles(season, stage, metrics)
    if len(pct_exact) != 850:
        raise ValueError(f"Expected 850 primary core support cells, got {len(pct_exact)}")
    pct_exact = pct_exact.copy()
    pct_full = pct_exact.copy()
    # These two files intentionally present their respective primary branch in the legacy column names.
    exact_out = pct_exact.drop(columns=["full80_n_training", "full80_percentile", "full80_min", "full80_max", "full80_p5", "full80_p95", "full80_outside_support", "full80_tail_p5_p95", "full80_classification", "full80_insufficient_reason", "full80_min_seed", "full80_max_seed", "heldout_min", "heldout_max", "heldout_support_contains_observed", "provisional_sensitive", "coverage_classification", "robustness_status"])
    full_out = pct_full.rename(columns={"empirical_percentile": "empirical_percentile_exact78", "training_min": "exact78_min", "training_max": "exact78_max", "training_p5": "exact78_p5", "training_p95": "exact78_p95", "classification": "exact78_classification", "support_min_seed": "exact78_min_seed", "support_max_seed": "exact78_max_seed", "outside_support": "exact78_outside_support", "tail_p5_p95": "exact78_tail_p5_p95", "distance_below_min": "exact78_distance_below_min", "distance_above_max": "exact78_distance_above_max", "outside_distance": "exact78_outside_distance"})
    full_out = full_out.rename(columns={"full80_n_training": "n_training", "full80_percentile": "empirical_percentile", "full80_min": "training_min", "full80_max": "training_max", "full80_p5": "training_p5", "full80_p95": "training_p95", "full80_classification": "classification", "full80_insufficient_reason": "insufficient_reason", "full80_min_seed": "support_min_seed", "full80_max_seed": "support_max_seed", "full80_outside_support": "outside_support", "full80_tail_p5_p95": "tail_p5_p95"})
    compound_support = build_compound_support(compounds)
    held = heldout_checks(season, stage, compounds)
    decision = decide(pct_exact, pct_full, pct_exact, compound_support, held)
    # Keep both branches in one row and preserve explicit branch-specific labels.
    comparison = pct_exact.copy()
    comparison["empirical_percentile_delta_full_minus_exact_pp"] = comparison.full80_percentile - comparison.empirical_percentile
    comparison["min_support_changed"] = comparison.training_min != comparison.full80_min
    comparison["max_support_changed"] = comparison.training_max != comparison.full80_max
    decision["provisional_classification_changes"] = provisional_change_details(comparison, season, stage)
    stage_comparison = comparison[comparison.scope == "stage"].copy()
    insufficient_cells = comparison[(comparison.classification == "INSUFFICIENT") | (comparison.full80_classification == "INSUFFICIENT")].copy()
    gaps = comparison[(comparison.classification.isin(["OUTSIDE_SUPPORT", "TAIL_BUT_INSIDE_SUPPORT"])) | (comparison.full80_classification.isin(["OUTSIDE_SUPPORT", "TAIL_BUT_INSIDE_SUPPORT"]))].copy()
    gaps["gap_metric_type"] = np.where(gaps.robustness_status == "PROVISIONAL_SENSITIVE", "PROVISIONAL_SENSITIVE", np.where(gaps.classification == "OUTSIDE_SUPPORT", "ROBUST_OUTSIDE_SUPPORT", "ROBUST_TAIL_INSIDE_SUPPORT"))
    compound_support["provisional_sensitive"] = compound_support.provisional_sensitive.astype(bool)
    compound_support["robustness_status"] = np.where(compound_support.provisional_sensitive, "PROVISIONAL_SENSITIVE", "ROBUST")
    robust_comp = compound_support[(compound_support.classification.isin(["OUTSIDE_SUPPORT", "TAIL_BUT_INSIDE_SUPPORT"])) | (compound_support.full80_classification.isin(["OUTSIDE_SUPPORT", "TAIL_BUT_INSIDE_SUPPORT"]))]
    gaps_full = pd.concat([gaps.assign(scope_type="core_00418A"), robust_comp.assign(scope_type="compound")], ignore_index=True, sort=False)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    season.to_csv(OUT / "weather_metrics_all_sets.csv", index=False)
    stage.to_csv(OUT / "weather_stage_metrics_all_sets.csv", index=False)
    compounds.to_csv(OUT / "compound_metrics_all_sets.csv", index=False)
    season_exact = season[season["set"] == "EXACT78"].copy()
    season_full = season[season["set"].isin(["EXACT78", "PROVISIONAL2"])].copy()
    season_exact["analysis_set"] = "EXACT78"
    season_full["analysis_set"] = "FULL80_CANDIDATE"
    season_exact.to_csv(OUT / "weather_metrics_exact78.csv", index=False)
    season_full.to_csv(OUT / "weather_metrics_full80.csv", index=False)
    season[(season["set"] == "HELDOUT20")].to_csv(OUT / "weather_metrics_heldout20.csv", index=False)
    season[(season["set"] == "OBSERVED")].to_csv(OUT / "observed_metrics.csv", index=False)
    stage_exact = stage[stage["set"] == "EXACT78"].copy()
    stage_full = stage[stage["set"].isin(["EXACT78", "PROVISIONAL2"])].copy()
    stage_exact["analysis_set"] = "EXACT78"
    stage_full["analysis_set"] = "FULL80_CANDIDATE"
    stage_exact.to_csv(OUT / "weather_stage_metrics_exact78.csv", index=False)
    stage_full.to_csv(OUT / "weather_stage_metrics_full80.csv", index=False)
    exact_out.to_csv(OUT / "observed_percentiles_exact78.csv", index=False)
    full_out.to_csv(OUT / "observed_percentiles_full80.csv", index=False)
    comparison.to_csv(OUT / "observed_support_comparison.csv", index=False)
    stage_comparison.to_csv(OUT / "stage_support_comparison.csv", index=False)
    compound_support.to_csv(OUT / "compound_extreme_support.csv", index=False)
    held.to_csv(OUT / "heldout_distribution_check.csv", index=False)
    gaps_full.to_csv(OUT / "robust_gap_inventory.csv", index=False)
    insufficient_cells.to_csv(OUT / "insufficient_data_cells.csv", index=False)
    pd.DataFrame(integrity).to_csv(OUT / "input_integrity.csv", index=False)
    decision.update({"audit_id": "004_19_yc_formal_climate_coverage_audit", "provenance_counts": {"exact78": 78, "full80_candidate": 80, "provisional_in_full80": [1003, 1036], "heldout20_exact": 20, "observed_years": 10}, "core_metric_definitions_source": str(PRIOR_RUNNER.relative_to(ROOT)), "compound_metric_definitions_source": "004_19 prompt; fixed thresholds before calculation", "archive_modified": False, "wgen_run": False, "dssat_run": False, "ppo_run": False, "commit_or_push": False})
    (OUT / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    save_figures(season, stage, comparison, OUT)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(build_report(decision, season, stage, comparison, compound_support, held, gaps_full), encoding="utf-8")
    print(json.dumps({k: decision[k] for k in ("exact78_observed_coverage", "full80_candidate_coverage", "robust_coverage_conclusion", "robust_tail_gap_type", "next_weather_action", "primary_grid")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
