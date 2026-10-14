#!/usr/bin/env python3
"""Fitting-only diagnostics for the YC external-weather method freeze.

This runner intentionally reads exactly one input file: the frozen 2004-2013
YC fitting-weather CSV. It never opens observed-period data or runtime files.
It does not generate candidate weather files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv"
OUT = ROOT / "results/yc_external_weather_augmentation_004_24A/extended_block_diagnostics_v3"
VARIABLES = ("RAIN", "SRAD", "TMAX", "TMIN")
# The first diagnostic pass showed persistent temperature autocorrelation and
# dry spells beyond 14 days, so evaluate longer blocks without replacing it.
BLOCK_LENGTHS = (3, 5, 7, 10, 14, 21, 28, 35, 42, 56)
LAGS = (1, 2, 3, 5, 7, 10, 14)
WET_THRESHOLD = 0.1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def quantile(values: list[float], p: float) -> float:
    xs = sorted(values)
    if not xs:
        return float("nan")
    pos = (len(xs) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    mx, my = statistics.mean(x), statistics.mean(y)
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    den = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    return sum(a * b for a, b in zip(dx, dy)) / den if den else 0.0


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(average_ranks(x), average_ranks(y))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def spell_lengths(flags: list[bool]) -> list[int]:
    out: list[int] = []
    run = 0
    for flag in flags:
        if flag:
            run += 1
        elif run:
            out.append(run)
            run = 0
    if run:
        out.append(run)
    return out


def acf(values: list[float], lag: int) -> float:
    if len(values) <= lag + 1:
        return float("nan")
    return pearson(values[:-lag], values[lag:])


def phase_ordinal(d: date) -> float:
    """Map month/day to a no-leap annual phase; Feb 29 lies between Feb 28/Mar 1."""
    if d.month == 2 and d.day == 29:
        return 58.5
    ref = date(2001, d.month, d.day)
    return float((ref - date(2001, 1, 1)).days)


def circular_distance(a: float, b: float, period: float = 365.0) -> float:
    delta = abs(a - b) % period
    return min(delta, period - delta)


def fmt(x: float) -> float | str:
    return round(x, 8) if math.isfinite(x) else "NA"


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUT),
                        help="New output directory inside the project; existing directories are never overwritten.")
    args = parser.parse_args()
    requested_out = Path(args.output_dir)
    if not requested_out.is_absolute():
        requested_out = ROOT / requested_out
    OUT = requested_out.resolve()
    try:
        OUT.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError("Output directory must remain inside the project root") from exc
    if not INPUT.is_file():
        raise FileNotFoundError(f"Required fitting-only input not found: {INPUT}")
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing 004_24A output: {OUT}")
    OUT.mkdir(parents=True)

    # Parse only the frozen fitting file and only the fields needed for analysis.
    years: dict[int, list[dict]] = defaultdict(list)
    with INPUT.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"DATE", "YEAR", "DOY", *VARIABLES}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Fitting input missing columns: {sorted(missing)}")
        for raw in reader:
            d = date.fromisoformat(raw["DATE"])
            row = {"date": d, "year": int(raw["YEAR"]), "doy": int(raw["DOY"])}
            for name in VARIABLES:
                row[name] = float(raw[name])
            if row["year"] != d.year or row["doy"] != d.timetuple().tm_yday:
                raise ValueError(f"DATE/YEAR/DOY mismatch: {raw['DATE']}")
            years[row["year"]].append(row)

    found_years = sorted(years)
    if found_years != list(range(2004, 2014)):
        raise ValueError(f"Expected fitting years 2004-2013; found {found_years}")
    for year, rows in years.items():
        rows.sort(key=lambda r: r["date"])
        expected = (date(year + 1, 1, 1) - date(year, 1, 1)).days
        if len(rows) != expected:
            raise ValueError(f"Year {year} has {len(rows)} rows; expected {expected}")
        for a, b in zip(rows, rows[1:]):
            if b["date"] - a["date"] != timedelta(days=1):
                raise ValueError(f"Non-contiguous fitting dates in {year}: {a['date']} {b['date']}")
    all_rows = [r for year in found_years for r in years[year]]
    if any(not math.isfinite(r[v]) for r in all_rows for v in VARIABLES):
        raise ValueError("Non-finite weather value in fitting input")
    if any(r["RAIN"] < 0 or r["SRAD"] < 0 or r["TMAX"] < r["TMIN"] for r in all_rows):
        raise ValueError("Fitting input violates RAIN/SRAD/TMAX/TMIN physical checks")

    fitting_sha = sha256_file(INPUT)
    qa = [{
        "source_path": str(INPUT), "source_sha256": fitting_sha,
        "years": f"{found_years[0]}-{found_years[-1]}", "n_years": len(found_years),
        "n_days": len(all_rows), "expected_days": sum(366 if y % 4 == 0 and (y % 100 != 0 or y % 400 == 0) else 365 for y in found_years),
        "date_continuous": True, "duplicate_dates": 0, "missing_values": 0,
        "wet_day_threshold_mm": WET_THRESHOLD, "physical_checks_pass": True,
        "accessed_other_weather_files": False,
    }]
    write_csv(OUT / "fitting_source_qa.csv", qa)

    summary_rows = []
    groups = [("FITTING_POOL", all_rows)] + [(str(y), years[y]) for y in found_years]
    probs = (0, .01, .05, .10, .50, .90, .95, .99, 1)
    for label, rows in groups:
        for var in VARIABLES:
            vals = [r[var] for r in rows]
            m, sd = mean_sd(vals)
            rec = {"group": label, "n_days": len(rows), "variable": var, "mean": fmt(m), "sd": fmt(sd)}
            rec.update({f"p{int(p*100):02d}": fmt(quantile(vals, p)) for p in probs})
            summary_rows.append(rec)
    write_csv(OUT / "fitting_summary.csv", summary_rows)

    # ACF uses only within-year day pairs so no artificial year-boundary lag is introduced.
    acf_rows = []
    series_map: dict[str, dict[int, list[float]]] = {v: {} for v in VARIABLES}
    series_map["WET"] = {}
    for y in found_years:
        for var in VARIABLES:
            series_map[var][y] = [r[var] for r in years[y]]
        series_map["WET"][y] = [1.0 if r["RAIN"] >= WET_THRESHOLD else 0.0 for r in years[y]]
    for metric, ys in series_map.items():
        for lag in LAGS:
            xs, zs = [], []
            year_acfs = []
            for y in found_years:
                arr = ys[y]
                xs.extend(arr[:-lag])
                zs.extend(arr[lag:])
                year_acfs.append(acf(arr, lag))
            acf_rows.append({
                "metric": metric, "lag_days": lag, "n_pairs": len(xs),
                "pooled_within_year_acf": fmt(pearson(xs, zs)),
                "mean_year_acf": fmt(statistics.mean(year_acfs)),
                "sd_year_acf": fmt(statistics.stdev(year_acfs)),
            })
    write_csv(OUT / "autocorrelation.csv", acf_rows)
    acf_lookup = {(r["metric"], r["lag_days"]): float(r["pooled_within_year_acf"]) for r in acf_rows}

    thresholds = {
        "low_srad_p05": quantile([r["SRAD"] for r in all_rows], .05),
        "low_srad_p10": quantile([r["SRAD"] for r in all_rows], .10),
        "hot_tmax_p90": quantile([r["TMAX"] for r in all_rows], .90),
        "cool_tmin_p10": quantile([r["TMIN"] for r in all_rows], .10),
    }
    spell_rows = []
    spell_lookup: dict[str, list[int]] = defaultdict(list)
    for y in found_years:
        rows = years[y]
        flags_by_name = {
            "dry_day": [r["RAIN"] < WET_THRESHOLD for r in rows],
            "wet_day": [r["RAIN"] >= WET_THRESHOLD for r in rows],
            "low_srad_p05": [r["SRAD"] < thresholds["low_srad_p05"] for r in rows],
            "hot_tmax_p90": [r["TMAX"] > thresholds["hot_tmax_p90"] for r in rows],
            "cool_tmin_p10": [r["TMIN"] < thresholds["cool_tmin_p10"] for r in rows],
        }
        for name, flags in flags_by_name.items():
            lens = spell_lengths(flags)
            spell_lookup[name].extend(lens)
            spell_rows.append({
                "year": y, "spell_type": name, "threshold": fmt(WET_THRESHOLD if name in ("dry_day", "wet_day") else thresholds[name]),
                "n_spells": len(lens), "mean_days": fmt(statistics.mean(lens) if lens else 0),
                "median_days": fmt(quantile(lens, .5) if lens else 0),
                "p90_days": fmt(quantile(lens, .9) if lens else 0), "max_days": max(lens, default=0),
            })
    for name, lens in spell_lookup.items():
        spell_rows.append({
            "year": "POOL", "spell_type": name,
            "threshold": fmt(WET_THRESHOLD if name in ("dry_day", "wet_day") else thresholds[name]),
            "n_spells": len(lens), "mean_days": fmt(statistics.mean(lens) if lens else 0),
            "median_days": fmt(quantile(lens, .5) if lens else 0),
            "p90_days": fmt(quantile(lens, .9) if lens else 0), "max_days": max(lens, default=0),
        })
    write_csv(OUT / "spell_statistics.csv", spell_rows)

    # Correlations and conditional structure; pooled daily correlations are descriptive.
    dep_rows = []
    pairs = [("SRAD", "RAIN"), ("SRAD", "TMAX"), ("SRAD", "TMIN"),
             ("TMAX", "TMIN"), ("RAIN", "TMAX"), ("RAIN", "TMIN")]
    for xname, yname in pairs:
        x = [r[xname] for r in all_rows]
        y = [r[yname] for r in all_rows]
        dep_rows.append({"analysis": "correlation", "x": xname, "y": yname, "group": "ALL", "n": len(x),
                         "pearson": fmt(pearson(x, y)), "spearman": fmt(spearman(x, y)),
                         "x_mean": fmt(statistics.mean(x)), "y_mean": fmt(statistics.mean(y))})
    low_thr = thresholds["low_srad_p05"]
    for label, predicate in (("WET", lambda r: r["RAIN"] >= WET_THRESHOLD),
                             ("DRY", lambda r: r["RAIN"] < WET_THRESHOLD),
                             ("LOW_SRAD_P05", lambda r: r["SRAD"] < low_thr),
                             ("NORMAL_OR_HIGH_SRAD", lambda r: r["SRAD"] >= low_thr)):
        subset = [r for r in all_rows if predicate(r)]
        for var in VARIABLES:
            vals = [r[var] for r in subset]
            dep_rows.append({"analysis": "conditional", "x": "group", "y": var, "group": label, "n": len(vals),
                             "pearson": "NA", "spearman": "NA", "x_mean": "NA",
                             "y_mean": fmt(statistics.mean(vals) if vals else float("nan")),
                             "y_p05": fmt(quantile(vals, .05) if vals else float("nan")),
                             "y_median": fmt(quantile(vals, .5) if vals else float("nan"))})
    write_csv(OUT / "dependence_summary.csv", dep_rows)

    # Calendar diagnostics are month-based; no DAP is inferred.
    month_rows = []
    for month in range(1, 13):
        rows = [r for r in all_rows if r["date"].month == month]
        month_rows.append({
            "month": month, "n_days": len(rows), "n_years": len({r["year"] for r in rows}),
            "rain_total_mean_per_day": fmt(statistics.mean(r["RAIN"] for r in rows)),
            "wet_day_fraction": fmt(sum(r["RAIN"] >= WET_THRESHOLD for r in rows) / len(rows)),
            "wet_day_intensity_mean": fmt(statistics.mean([r["RAIN"] for r in rows if r["RAIN"] >= WET_THRESHOLD]) if any(r["RAIN"] >= WET_THRESHOLD for r in rows) else float("nan")),
            "SRAD_mean": fmt(statistics.mean(r["SRAD"] for r in rows)),
            "SRAD_P05": fmt(quantile([r["SRAD"] for r in rows], .05)),
            "TMAX_mean": fmt(statistics.mean(r["TMAX"] for r in rows)),
            "TMIN_mean": fmt(statistics.mean(r["TMIN"] for r in rows)),
        })
    write_csv(OUT / "seasonal_calendar_diagnostics.csv", month_rows)

    # Adjacent-day deltas define an immutable, fitting-derived seam reference.
    delta_values: dict[str, list[float]] = defaultdict(list)
    rain_transitions = defaultdict(int)
    for y in found_years:
        rows = years[y]
        for a, b in zip(rows, rows[1:]):
            for var in VARIABLES:
                delta_values[var].append(abs(b[var] - a[var]))
            a_wet, b_wet = a["RAIN"] >= WET_THRESHOLD, b["RAIN"] >= WET_THRESHOLD
            rain_transitions[f"{int(a_wet)}->{int(b_wet)}"] += 1
    seam_rows = []
    for var, vals in delta_values.items():
        seam_rows.append({"variable": var, "transition": "", "n_adjacent_pairs": len(vals), "fitting_fraction": "",
                          "p50_abs_delta": fmt(quantile(vals, .5)),
                          "p95_abs_delta": fmt(quantile(vals, .95)), "p99_abs_delta": fmt(quantile(vals, .99)),
                          "max_abs_delta": fmt(max(vals)), "frozen_violation_threshold": fmt(quantile(vals, .99))})
    for transition, count in sorted(rain_transitions.items()):
        seam_rows.append({"variable": "RAIN_WET_STATE", "transition": transition, "n_adjacent_pairs": count,
                          "fitting_fraction": fmt(count / sum(rain_transitions.values())), "p50_abs_delta": "NA",
                          "p95_abs_delta": "NA", "p99_abs_delta": "NA", "max_abs_delta": "NA",
                          "frozen_violation_threshold": "DESCRIPTIVE_REFERENCE_ONLY"})
    write_csv(OUT / "adjacent_day_delta_seam_reference.csv", seam_rows)

    # Equal-year metrics support candidate-level outlier fences and pool-level
    # cluster-bootstrap reference intervals without treating daily rows as iid.
    annual_rows: list[dict] = []
    for y in found_years:
        rows = years[y]
        rain = [r["RAIN"] for r in rows]
        srad = [r["SRAD"] for r in rows]
        tmax = [r["TMAX"] for r in rows]
        tmin = [r["TMIN"] for r in rows]
        wet = [v >= WET_THRESHOLD for v in rain]
        dry = [not v for v in wet]
        lowrad = [v < thresholds["low_srad_p05"] for v in srad]
        hot = [v > thresholds["hot_tmax_p90"] for v in tmax]
        cool = [v < thresholds["cool_tmin_p10"] for v in tmin]
        wet_amounts = [v for v, flag in zip(rain, wet) if flag]
        rolling_rain = [sum(rain[i:i + 5]) for i in range(len(rain) - 4)]
        dtr = [a - b for a, b in zip(tmax, tmin)]
        annual = {
            "year": y,
            "n_days": len(rows),
            "seasonal_rain_total": sum(rain),
            "wet_day_count": sum(wet),
            "wet_day_fraction": sum(wet) / len(wet),
            "wet_day_intensity_mean": statistics.mean(wet_amounts) if wet_amounts else 0.0,
            "max_dry_spell_days": max(spell_lengths(dry), default=0),
            "p90_dry_spell_days": quantile(spell_lengths(dry), .9),
            "max_wet_spell_days": max(spell_lengths(wet), default=0),
            "rx1day": max(rain),
            "rx5day": max(rolling_rain),
            "srad_min": min(srad),
            "srad_p01": quantile(srad, .01),
            "srad_p05": quantile(srad, .05),
            "srad_p10": quantile(srad, .10),
            "srad_mean": statistics.mean(srad),
            "srad_lt_fit_p05_fraction": sum(lowrad) / len(lowrad),
            "srad_lt_fit_p10_fraction": sum(v < thresholds["low_srad_p10"] for v in srad) / len(srad),
            "max_low_srad_spell_days": max(spell_lengths(lowrad), default=0),
            "tmax_mean": statistics.mean(tmax),
            "tmax_max": max(tmax),
            "hot_day_fraction_fit_p90": sum(hot) / len(hot),
            "max_hot_spell_days": max(spell_lengths(hot), default=0),
            "tmin_mean": statistics.mean(tmin),
            "tmin_min": min(tmin),
            "cool_day_fraction_fit_p10": sum(cool) / len(cool),
            "max_cool_spell_days": max(spell_lengths(cool), default=0),
            "diurnal_range_mean": statistics.mean(dtr),
            "srad_tmax_spearman": spearman(srad, tmax),
            "srad_rain_spearman": spearman(srad, rain),
            "tmax_tmin_spearman": spearman(tmax, tmin),
            "wetday_srad_mean": statistics.mean([r["SRAD"] for r, flag in zip(rows, wet) if flag]),
            "dryday_srad_mean": statistics.mean([r["SRAD"] for r, flag in zip(rows, wet) if not flag]),
        }
        for metric in ("RAIN", "SRAD", "TMAX", "TMIN", "WET"):
            annual[f"acf_{metric.lower()}_lag1"] = acf(series_map[metric][y], 1)
            annual[f"acf_{metric.lower()}_lag3"] = acf(series_map[metric][y], 3)
            annual[f"acf_{metric.lower()}_lag7"] = acf(series_map[metric][y], 7)
        annual_rows.append(annual)
    write_csv(OUT / "annual_weather_metrics.csv", annual_rows)

    qa_seed = 20260927
    bootstrap_reps = 5000
    metric_names = [k for k in annual_rows[0] if k not in ("year", "n_days")]
    rng = random.Random(qa_seed)
    boot_values = {m: [] for m in metric_names}
    for _ in range(bootstrap_reps):
        chosen = [annual_rows[rng.randrange(len(annual_rows))] for _ in range(20)]
        for metric in metric_names:
            boot_values[metric].append(statistics.mean(float(x[metric]) for x in chosen))
    ref_rows = []
    for metric in metric_names:
        vals = [float(r[metric]) for r in annual_rows]
        sims = boot_values[metric]
        ref_rows.append({
            "metric": metric, "reference_n_years": len(annual_rows), "candidate_pool_size": 20,
            "fitting_year_mean": fmt(statistics.mean(vals)),
            "fitting_year_sd": fmt(statistics.stdev(vals)),
            "bootstrap_mean_p025": fmt(quantile(sims, .025)),
            "bootstrap_mean_p975": fmt(quantile(sims, .975)),
            "bootstrap_seed": qa_seed, "bootstrap_replicates": bootstrap_reps,
            "resampling_unit": "whole fitting calendar year; equal probability with replacement",
        })
    write_csv(OUT / "pool_qa_bootstrap_reference.csv", ref_rows)

    fence_rows = []
    for metric in metric_names:
        vals = [float(r[metric]) for r in annual_rows]
        q1, q3 = quantile(vals, .25), quantile(vals, .75)
        iqr = q3 - q1
        fence_rows.append({"metric": metric, "fitting_q1": fmt(q1), "fitting_q3": fmt(q3),
                           "fitting_iqr": fmt(iqr), "candidate_fail_below": fmt(q1 - 3 * iqr),
                           "candidate_fail_above": fmt(q3 + 3 * iqr),
                           "rule": "outside Tukey 3xIQR fence across fitting-year annual metrics"})
    write_csv(OUT / "candidate_level_annual_metric_fences.csv", fence_rows)

    # Compare candidate block lengths using the exact expected within-block ACF retention
    # factor (1 - lag/L)+ and fitting spell support. No candidate weather is materialized.
    block_rows = []
    key_metrics = ("RAIN", "SRAD", "TMAX", "TMIN", "WET")
    for length in BLOCK_LENGTHS:
        errs = []
        for metric in key_metrics:
            for lag in (1, 2, 3, 5, 7, 10, 14):
                base = acf_lookup[(metric, lag)]
                retained = base * max(0.0, 1.0 - lag / length)
                errs.append(abs(base - retained))
        spell_long = {}
        for name, lens in spell_lookup.items():
            spell_long[name] = sum(n > length for n in lens) / len(lens) if lens else 0.0
        block_rows.append({
            "block_length_days": length,
            "expected_mean_abs_acf_loss_lags_1_14": fmt(statistics.mean(errs)),
            "expected_mean_abs_acf_loss_lags_1_7": fmt(statistics.mean(
                abs(acf_lookup[(m, lag)] * min(1.0, lag / length))
                for m in key_metrics for lag in (1, 2, 3, 5, 7))),
            "spell_fraction_longer_than_block_dry": fmt(spell_long["dry_day"]),
            "spell_fraction_longer_than_block_wet": fmt(spell_long["wet_day"]),
            "spell_fraction_longer_than_block_low_srad": fmt(spell_long["low_srad_p05"]),
            "spell_fraction_longer_than_block_hot": fmt(spell_long["hot_tmax_p90"]),
            "spell_fraction_longer_than_block_cool": fmt(spell_long["cool_tmin_p10"]),
        })
    write_csv(OUT / "block_length_comparison.csv", block_rows)

    # Donor availability for actual target blocks in 365- and 366-day years,
    # including the shorter final tail block. Source blocks never cross years.
    donor_blocks: dict[int, list[float]] = {}
    for length in BLOCK_LENGTHS:
        mids = []
        for y in found_years:
            rows = years[y]
            for start in range(0, len(rows) - length + 1):
                end = start + length - 1
                a, b = rows[start]["date"], rows[end]["date"]
                pa, pb = phase_ordinal(a), phase_ordinal(b)
                # Circular midpoint handles the Dec/Jan boundary; the block itself does not cross years.
                forward = (pb - pa) % 365.0
                mids.append((pa + forward / 2.0) % 365.0)
        donor_blocks[length] = mids
    donor_rows = []
    donor_detail_rows = []
    for length in BLOCK_LENGTHS:
        counts = []
        target_contexts = []
        for target_year in (2001, 2004):
            n_target = (date(target_year + 1, 1, 1) - date(target_year, 1, 1)).days
            cursor = 0
            while cursor < n_target:
                block_n = min(length, n_target - cursor)
                start_date = date(target_year, 1, 1) + timedelta(days=cursor)
                end_date = start_date + timedelta(days=block_n - 1)
                p0, p1 = phase_ordinal(start_date), phase_ordinal(end_date)
                target_mid = (p0 + ((p1 - p0) % 365.0) / 2.0) % 365.0
                eligible = 0
                for source_year in found_years:
                    src = years[source_year]
                    for start in range(0, len(src) - block_n + 1):
                        a = src[start]["date"]
                        b = src[start + block_n - 1]["date"]
                        pmid = (phase_ordinal(a) + ((phase_ordinal(b) - phase_ordinal(a)) % 365.0) / 2.0) % 365.0
                        if circular_distance(pmid, target_mid) <= 7.0:
                            eligible += 1
                counts.append(eligible)
                target_contexts.append({"target_year": target_year,
                                        "target_year_type": "leap" if n_target == 366 else "nonleap",
                                        "block_index": len(target_contexts) + 1, "target_block_days": block_n,
                                        "target_mid_phase": round(target_mid, 3), "eligible_donor_blocks": eligible,
                                        "block_length_days": length})
                cursor += block_n
        donor_rows.append({
            "block_length_days": length, "phase_window_days": 7,
            "source_blocks_total_full_length": len(donor_blocks[length]),
            "target_blocks_checked": len(counts), "eligible_donors_min": min(counts),
            "eligible_donors_p05": fmt(quantile(counts, .05)), "eligible_donors_median": fmt(quantile(counts, .5)),
            "eligible_donors_p95": fmt(quantile(counts, .95)), "eligible_donors_max": max(counts),
            "includes_short_tail_blocks": True, "cross_calendar_year_blocks_allowed": False,
        })
        donor_detail_rows.extend(target_contexts)
    write_csv(OUT / "donor_availability_diagnostics.csv", donor_rows)
    write_csv(OUT / "donor_availability_by_target_block.csv", donor_detail_rows)

    method = {
        "analysis": "fitting_only_method_freeze_diagnostics",
        "input_path": str(INPUT), "input_sha256": fitting_sha,
        "unit_metadata_in_input_csv": False,
        "units_status": "WTH/fitting convention labels only; no unit conversion performed",
        "python_version": sys.version.split()[0],
        "input_years": found_years, "input_days": len(all_rows),
        "observed_data_opened": False, "other_weather_files_opened": False,
        "candidate_weather_materialized": False, "wgen_dssat_ppo_executed": False,
        "fitting_thresholds": {k: fmt(v) for k, v in thresholds.items()},
        "wet_day_definition": f"RAIN >= {WET_THRESHOLD} mm",
        "block_lengths_evaluated_days": list(BLOCK_LENGTHS),
        "phase_matching_donor_availability_rule": "circular 365-day phase distance between target and donor block midpoints <= 7 days; source blocks must be continuous and remain within one source calendar year; final target tail uses a donor block of exactly the remaining length",
        "seam_reference": "fitting pooled within-year adjacent-day absolute-delta P99, reported by variable",
    }
    (OUT / "analysis_manifest.json").write_text(json.dumps(method, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "fitting_source": str(INPUT), "fitting_years": f"{found_years[0]}-{found_years[-1]}",
        "n_days": len(all_rows), "sha256": fitting_sha, "observed_data_accessed": False,
        "diagnostics_dir": str(OUT), "candidate_weather_materialized": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
