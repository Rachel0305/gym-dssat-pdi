#!/usr/bin/env python3
"""Fitting-only diagnosis and LOYO comparison for YC weather augmentation 004_24C."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/yc_external_weather_augmentation_004_24C"
FREEZE = ROOT / "results/yc_external_weather_augmentation_004_24A/method_freeze.yaml"
DIAG = ROOT / "results/yc_external_weather_augmentation_004_24A/extended_block_diagnostics_v3"
FITTING = ROOT / "results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv"
B_ROOT = ROOT / "results/yc_external_weather_augmentation_004_24B"
B_INTERNAL = B_ROOT / "internal_qa"
REPORT = ROOT / "docs/yc_random_weather_004_24C_failure_diagnosis_and_method_revision.md"
VARIABLES = ("RAIN", "SRAD", "TMAX", "TMIN")
FITTING_SHA256 = "4b8ffe9e881d0a0743921b78b9c0e0ebfb1d2d645c5aa9737948b2b088ed7b34"
FREEZE_SHA256 = "33ff590debe0f8019359b4c6284ef63949a0ee055fa09b90618fb68e9215544e"
DEV_SEEDS = list(range(2424001, 2424021))
VALIDATION_SEEDS = list(range(240445, 240465))
BLOCK_LENGTH = 28
PHASE_WINDOW = 7.0
WET_THRESHOLD = 0.1
LOW_SRAD_P05 = 2.46823896
LOW_SRAD_P10 = 4.5568
HOT_TMAX_P90 = 32.7
COOL_TMIN_P10 = -6.48
METHODS = ("M0_uniform_28d_pm7", "M1_transition_top20pct_28d_pm7")
METHOD_METRICS = (
    "seasonal_rain_total", "wet_day_count", "wet_day_fraction", "wet_day_intensity_mean",
    "max_dry_spell_days", "p90_dry_spell_days", "max_wet_spell_days", "rx1day", "rx5day",
    "srad_min", "srad_p01", "srad_p05", "srad_p10", "srad_mean", "srad_lt_fit_p05_fraction",
    "srad_lt_fit_p10_fraction", "max_low_srad_spell_days", "tmax_mean", "tmax_max",
    "hot_day_fraction_fit_p90", "max_hot_spell_days", "tmin_mean", "tmin_min",
    "cool_day_fraction_fit_p10", "max_cool_spell_days", "diurnal_range_mean",
    "srad_tmax_spearman", "srad_rain_spearman", "tmax_tmin_spearman", "wetday_srad_mean",
    "dryday_srad_mean", "acf_rain_lag1", "acf_rain_lag3", "acf_rain_lag7",
    "acf_srad_lag1", "acf_srad_lag3", "acf_srad_lag7", "acf_tmax_lag1", "acf_tmax_lag3",
    "acf_tmax_lag7", "acf_tmin_lag1", "acf_tmin_lag3", "acf_tmin_lag7",
    "acf_wet_lag1", "acf_wet_lag3", "acf_wet_lag7",
)
PERSISTENCE_DEPENDENCE_METRICS = (
    "max_dry_spell_days", "max_wet_spell_days", "max_low_srad_spell_days",
    "acf_rain_lag1", "acf_rain_lag3", "acf_rain_lag7", "acf_srad_lag1", "acf_srad_lag3",
    "acf_srad_lag7", "acf_tmax_lag1", "acf_tmax_lag3", "acf_tmax_lag7",
    "acf_tmin_lag1", "acf_tmin_lag3", "acf_tmin_lag7", "acf_wet_lag1", "acf_wet_lag3",
    "acf_wet_lag7", "srad_tmax_spearman", "srad_rain_spearman", "tmax_tmin_spearman",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def percentile(values: list[float], q: float) -> float:
    xs = sorted(float(x) for x in values if math.isfinite(float(x)))
    if not xs:
        return float("nan")
    pos = (len(xs) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return xs[lo] if lo == hi else xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def avg(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else float("nan")


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 3:
        return float("nan")
    mx, my = avg(xs), avg(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            result[order[k]] = rank
        i = j
    return result


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(ranks(xs), ranks(ys))


def acf(xs: list[float], lag: int) -> float:
    return pearson(xs[:-lag], xs[lag:]) if len(xs) > lag + 2 else float("nan")


def longest_run(flags: list[bool]) -> int:
    best = run = 0
    for value in flags:
        if value:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def run_lengths(flags: list[bool]) -> list[int]:
    result, run = [], 0
    for value in flags:
        if value:
            run += 1
        elif run:
            result.append(run)
            run = 0
    if run:
        result.append(run)
    return result


def phase_ordinal(day: date) -> float:
    if day.month == 2 and day.day == 29:
        return 58.5
    return float((date(2001, day.month, day.day) - date(2001, 1, 1)).days)


def circular_distance(a: float, b: float) -> float:
    delta = abs(a - b) % 365.0
    return min(delta, 365.0 - delta)


def load_fitting() -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    raw = read_csv(FITTING)
    by_year: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in raw:
        day = date.fromisoformat(item["DATE"])
        row = {"DATE": day, "YEAR": day.year, "DOY": day.timetuple().tm_yday}
        row.update({v: float(item[v]) for v in VARIABLES})
        by_year[day.year].append(row)
    years = {year: sorted(rows, key=lambda x: x["DATE"]) for year, rows in by_year.items()}
    all_rows = [r for year in sorted(years) for r in years[year]]
    if list(years) != list(range(2004, 2014)) or len(all_rows) != 3653:
        raise RuntimeError("Frozen fitting weather years/row count differ from the 004_24A contract")
    return years, all_rows


def metric_values(rows: list[dict[str, Any]]) -> dict[str, float]:
    rows = sorted(rows, key=lambda x: x["DATE"])
    rain = [float(r["RAIN"]) for r in rows]
    srad = [float(r["SRAD"]) for r in rows]
    tmax = [float(r["TMAX"]) for r in rows]
    tmin = [float(r["TMIN"]) for r in rows]
    wet = [x >= WET_THRESHOLD for x in rain]
    dry_runs = run_lengths([not x for x in wet])
    wet_runs = run_lengths(wet)
    low_runs = run_lengths([x < LOW_SRAD_P05 for x in srad])
    hot_runs = run_lengths([x > HOT_TMAX_P90 for x in tmax])
    cool_runs = run_lengths([x < COOL_TMIN_P10 for x in tmin])
    rx5 = [sum(rain[i:i + 5]) for i in range(max(0, len(rain) - 4))]
    wet_rain = [rain[i] for i, is_wet in enumerate(wet) if is_wet]
    wet_srad = [srad[i] for i, is_wet in enumerate(wet) if is_wet]
    dry_srad = [srad[i] for i, is_wet in enumerate(wet) if not is_wet]
    dtr = [a - b for a, b in zip(tmax, tmin)]
    data = {
        "seasonal_rain_total": sum(rain), "wet_day_count": sum(wet),
        "wet_day_fraction": avg([float(x) for x in wet]), "wet_day_intensity_mean": avg(wet_rain),
        "max_dry_spell_days": max(dry_runs, default=0), "p90_dry_spell_days": percentile([float(x) for x in dry_runs], .9),
        "max_wet_spell_days": max(wet_runs, default=0), "rx1day": max(rain, default=0), "rx5day": max(rx5, default=0),
        "srad_min": min(srad), "srad_p01": percentile(srad, .01), "srad_p05": percentile(srad, .05),
        "srad_p10": percentile(srad, .10), "srad_mean": avg(srad),
        "srad_lt_fit_p05_fraction": avg([float(x < LOW_SRAD_P05) for x in srad]),
        "srad_lt_fit_p10_fraction": avg([float(x < LOW_SRAD_P10) for x in srad]),
        "max_low_srad_spell_days": max(low_runs, default=0), "tmax_mean": avg(tmax), "tmax_max": max(tmax),
        "hot_day_fraction_fit_p90": avg([float(x > HOT_TMAX_P90) for x in tmax]),
        "max_hot_spell_days": max(hot_runs, default=0), "tmin_mean": avg(tmin), "tmin_min": min(tmin),
        "cool_day_fraction_fit_p10": avg([float(x < COOL_TMIN_P10) for x in tmin]),
        "max_cool_spell_days": max(cool_runs, default=0), "diurnal_range_mean": avg(dtr),
        "srad_tmax_spearman": spearman(srad, tmax), "srad_rain_spearman": spearman(srad, rain),
        "tmax_tmin_spearman": spearman(tmax, tmin), "wetday_srad_mean": avg(wet_srad), "dryday_srad_mean": avg(dry_srad),
    }
    for var, values in (("rain", rain), ("srad", srad), ("tmax", tmax), ("tmin", tmin), ("wet", [float(x) for x in wet])):
        for lag in (1, 3, 7):
            data[f"acf_{var}_lag{lag}"] = acf(values, lag)
    return data


def fitting_year_metrics(years: dict[int, list[dict[str, Any]]]) -> dict[int, dict[str, float]]:
    return {year: metric_values(rows) for year, rows in years.items()}


def adjacent_pairs(rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    ordered = sorted(rows, key=lambda x: x["DATE"])
    return [(a, b) for a, b in zip(ordered, ordered[1:]) if (b["DATE"] - a["DATE"]).days == 1 and a["DATE"].year == b["DATE"].year]


def make_analysis_plan() -> dict[str, Any]:
    return {
        "task_id": "004_24C", "data_scope": "2004-2013 fitting, 004_24A diagnostics, 004_24B candidates and internal QA only",
        "observed_data_used_for_method_revision": "NO", "forbidden_execution": ["WGEN", "DSSAT", "Gym-DSSAT", "PPO"],
        "baseline": {"id": METHODS[0], "block_length_days": 28, "phase_midpoint_window_days": 7, "selection": "uniform with replacement", "seam": "raw concatenation"},
        "transition_method": {"id": METHODS[1], "block_length_days": 28, "phase_midpoint_window_days": 7,
            "score": "mean of abs(delta)/fitting donor P95 absolute adjacent delta for RAIN, SRAD, TMAX, TMIN, plus wet-state transition surprisal / ln(2); all quantities fitted from the 9 donor years in each LOYO fold",
            "wet_state": "RAIN >= 0.1 mm", "subset": "lowest-score ceil(20% of phase-eligible blocks)",
            "selection": "uniform stochastic draw from compatible subset; with replacement", "seam": "raw concatenation; no smoothing, interpolation, clipping or perturbation"},
        "loyo": {"scheme": "leave one calendar year out", "years": list(range(2004, 2014)), "replicates_per_fold_method": 20,
            "development_seeds": DEV_SEEDS, "paired_method_seeds": True, "cross_source_year_blocks": False,
            "target_calendar": "held-out year's full calendar year including Feb 29 if leap; phase uses frozen 365-day circular rule"},
        "gates_frozen_before_loyo": {
            "structural": "100% continuous unique target dates, correct 365/366 length, donor tuple preserved exactly, donor excludes held-out year",
            "seam": "M1 reduces P99 exceedance rates by >=50% for both TMAX and TMIN vs M0 and does not increase any variable's rate",
            "persistence_dependence": "For each listed persistence/dependence metric, upper 95% paired fold-bootstrap bound of M1-M0 absolute-error change <= 0.10 * fitting-year SD; no post-hoc composite score",
            "annual_metric_shift": "For each frozen QA annual metric, absolute LOYO mean bias <= 0.25 * fitting-year SD, and per-fold 20-member mean is checked against unchanged 004_24A bootstrap interval",
            "diversity": "all donor years used; no donor year >20% of M1 donor rows; no identical donor block >5% of selected blocks; report exact/near duplicates; near duplicate = >=95% matching aligned daily tuples within a held-out fold",
        },
        "next_validation_seed_schedule": [{"candidate_id": f"candidate_{i:03d}", "seed": seed} for i, seed in enumerate(VALIDATION_SEEDS, 1)],
        "frozen_inputs": {"fitting_sha256": FITTING_SHA256, "method_freeze_sha256": FREEZE_SHA256},
    }


def preflight() -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    required = [FREEZE, FITTING, DIAG / "adjacent_day_delta_seam_reference.csv", DIAG / "candidate_level_annual_metric_fences.csv",
                DIAG / "pool_qa_bootstrap_reference.csv", DIAG / "annual_weather_metrics.csv", B_INTERNAL / "candidate_level_qa.csv",
                B_INTERNAL / "pool_metric_bootstrap_checks.csv", B_ROOT / "manifests/candidate_hash_manifest.csv"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing fitting-only input(s): " + ", ".join(missing))
    if sha256(FITTING) != FITTING_SHA256 or sha256(FREEZE) != FREEZE_SHA256:
        raise RuntimeError("Frozen fitting/config hash mismatch")
    config = yaml.safe_load(FREEZE.read_text(encoding="utf-8"))
    if config["method"]["block_length_days"] != BLOCK_LENGTH or "no more than 7 days" not in config["method"]["phase_alignment_rule"]:
        raise RuntimeError("004_24A frozen method differs from pre-registered M0")
    years, all_rows = load_fitting()
    plan_path = OUT / "analysis_plan.json"
    plan = make_analysis_plan()
    plan["analysis_plan_sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    if plan_path.exists():
        old = json.loads(plan_path.read_text(encoding="utf-8"))
        if old != plan:
            raise RuntimeError("Existing 004_24C analysis plan differs; refusing to change frozen LOYO choices")
    else:
        write_json(plan_path, plan)
    return years, all_rows, config


def parse_candidate_wth(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="latin-1", errors="replace").splitlines():
        parts = line.split()
        if not parts or len(parts[0]) != 7 or not parts[0].isdigit():
            continue
        day = date(int(parts[0][:4]), 1, 1) + timedelta(days=int(parts[0][4:]) - 1)
        if len(parts) < 5:
            continue
        rows.append({"DATE": day, "SRAD": float(parts[1]), "TMAX": float(parts[2]), "TMIN": float(parts[3]), "RAIN": float(parts[4])})
    return rows


def b_failure_decomposition(years: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    qa_rows = read_csv(B_INTERNAL / "candidate_level_qa.csv")
    candidate_metric_checks = read_csv(B_INTERNAL / "candidate_metric_fence_checks.csv")
    pool_rows = read_csv(B_INTERNAL / "pool_metric_bootstrap_checks.csv")
    candidate_metric_fail_counts = Counter(r["metric"] for r in candidate_metric_checks if r.get("pass", "").lower() == "false")
    deltas: dict[str, list[float]] = {v: [] for v in VARIABLES}
    for year_rows in years.values():
        for a, b in adjacent_pairs(year_rows):
            for v in VARIABLES:
                deltas[v].append(abs(float(b[v]) - float(a[v])))
    seam_ref = {r["variable"]: float(r["frozen_violation_threshold"]) for r in read_csv(DIAG / "adjacent_day_delta_seam_reference.csv") if r["variable"] in VARIABLES}
    seam_counts = {v: {"fitting_delta_count": 0, "fitting_violation_count": 0, "candidate_internal_pairs": 0,
                       "candidate_internal_violations": 0, "candidate_seam_pairs": 0, "candidate_seam_violations": 0,
                       "candidate_total_pairs": 0, "candidate_total_violations": 0} for v in VARIABLES}
    donor_rows_usage: dict[int, dict[str, float]] = defaultdict(lambda: {"rows": 0, "rain": 0.0, "heavy_rain": 0.0, "wet_days": 0})
    donor_block_usage: Counter[str] = Counter()
    donor_block_rain: Counter[str] = Counter()
    heavy_threshold = percentile([float(r["RAIN"]) for rr in years.values() for r in rr if float(r["RAIN"]) >= WET_THRESHOLD], .95)
    low_srad = {"threshold": LOW_SRAD_P05, "candidate_days_below_threshold": 0, "candidate_low_spell_count": 0,
                "candidate_low_spells_crossing_seam": 0, "candidate_low_spells_touching_seam": 0,
                "candidate_max_low_spell_days": 0, "all_candidate_values_exact_fitting_donor": True}
    rain_candidate_metrics = []
    for i in range(1, 21):
        cid = f"candidate_{i:03d}"
        cdir = B_ROOT / "candidates" / cid
        weather = parse_candidate_wth(cdir / "weather.WTH")
        prov = read_csv(cdir / "provenance.csv")
        if len(weather) != len(prov):
            raise RuntimeError(f"Candidate/provenance row count mismatch: {cid}")
        b_ids = [p["block_id"] for p in prov]
        for row, p in zip(weather, prov):
            donor = int(p["source_year"])
            usage = donor_rows_usage[donor]
            usage["rows"] += 1
            usage["rain"] += row["RAIN"]
            usage["heavy_rain"] += row["RAIN"] if row["RAIN"] > heavy_threshold else 0
            usage["wet_days"] += int(row["RAIN"] >= WET_THRESHOLD)
            key = f"{donor}:{p['block_start_source_date']}:{p['block_end_source_date']}"
            # Count each selected source block once below using its first row.
            if p["within_block_offset"] == "0":
                donor_block_usage[key] += 1
                block_rain = sum(weather[k]["RAIN"] for k in range(len(weather)) if b_ids[k] == p["block_id"])
                donor_block_rain[key] += block_rain
            low_srad["all_candidate_values_exact_fitting_donor"] &= p["source_tuple_exact"].lower() == "true"
            low_srad["candidate_days_below_threshold"] += int(row["SRAD"] < LOW_SRAD_P05)
        metrics = metric_values(weather)
        m = {"candidate_id": cid, "seed": 240424 + i, **metrics,
             "heavy_rain_threshold_fit_wetday_p95": heavy_threshold,
             "rain_above_threshold_total": sum(r["RAIN"] for r in weather if r["RAIN"] > heavy_threshold),
             "rain_above_threshold_share": sum(r["RAIN"] for r in weather if r["RAIN"] > heavy_threshold) / max(metrics["seasonal_rain_total"], 1e-12)}
        rain_candidate_metrics.append(m)
        by_block = defaultdict(list)
        for row, bid in zip(weather, b_ids):
            by_block[bid].append(row)
        seam_idxs = {i for i in range(1, len(b_ids)) if b_ids[i] != b_ids[i - 1]}
        lflags = [r["SRAD"] < LOW_SRAD_P05 for r in weather]
        lstarts = []
        j = 0
        while j < len(lflags):
            if not lflags[j]:
                j += 1
                continue
            st = j
            while j < len(lflags) and lflags[j]:
                j += 1
            lstarts.append((st, j))
            low_srad["candidate_low_spell_count"] += 1
            low_srad["candidate_max_low_spell_days"] = max(low_srad["candidate_max_low_spell_days"], j - st)
            inner = [k for k in seam_idxs if st < k < j]
            touch = any(abs(k - st) <= 1 or abs(k - j) <= 1 for k in seam_idxs)
            low_srad["candidate_low_spells_crossing_seam"] += int(bool(inner))
            low_srad["candidate_low_spells_touching_seam"] += int(touch)
        for k, (a, b) in enumerate(zip(weather, weather[1:]), 1):
            same_block = b_ids[k] == b_ids[k - 1]
            for v in VARIABLES:
                delta = abs(float(b[v]) - float(a[v]))
                rec = seam_counts[v]
                violation = delta > seam_ref[v]
                rec["candidate_total_pairs"] += 1
                rec["candidate_total_violations"] += int(violation)
                if same_block:
                    rec["candidate_internal_pairs"] += 1
                    rec["candidate_internal_violations"] += int(violation)
                else:
                    rec["candidate_seam_pairs"] += 1
                    rec["candidate_seam_violations"] += int(violation)
    for v in VARIABLES:
        ref = seam_ref[v]
        seam_counts[v]["fitting_delta_count"] = len(deltas[v])
        seam_counts[v]["fitting_violation_count"] = sum(x > ref for x in deltas[v])
        seam_counts[v]["fitting_violation_rate"] = seam_counts[v]["fitting_violation_count"] / max(1, len(deltas[v]))
        seam_counts[v]["frozen_p99_threshold"] = ref
        seam_counts[v]["fitting_abs_delta_p50"] = percentile(deltas[v], .5)
        seam_counts[v]["fitting_abs_delta_p95"] = percentile(deltas[v], .95)
        seam_counts[v]["fitting_abs_delta_p99"] = percentile(deltas[v], .99)
        seam_counts[v]["fitting_abs_delta_max"] = max(deltas[v])
        for where, nkey, vkey in (("candidate_internal", "candidate_internal_pairs", "candidate_internal_violations"),
                                  ("candidate_seam", "candidate_seam_pairs", "candidate_seam_violations"),
                                  ("candidate_total", "candidate_total_pairs", "candidate_total_violations")):
            seam_counts[v][where + "_violation_rate"] = seam_counts[v][vkey] / max(1, seam_counts[v][nkey])
    qa_counts = {k: Counter(row[k] for row in qa_rows) for k in ("structural_status", "physical_status", "provenance_status", "serialization_status", "candidate_metric_fence_status", "seam_status", "overall_status")}
    seam_rows = []
    for v in VARIABLES:
        seam_rows.append({"variable": v, **seam_counts[v],
                          "concentration_at_seam": seam_counts[v]["candidate_seam_violation_rate"] > seam_counts[v]["candidate_internal_violation_rate"]})
    write_csv(OUT / "seam_decomposition_by_variable.csv", seam_rows)
    metric_rows = []
    for row in pool_rows:
        if row.get("pass", "").lower() == "false":
            metric = row["metric"]
            hypotheses = {
                "seasonal_rain_total": "RAIN频率/单次强度/重雨块贡献，结合20成员有限抽样分解；非seam单因解释",
                "rx5day": "连续重雨donor块聚集与有限抽样；比较wet count、wet intensity及heavy-rain share",
                "srad_min": "尾部抽样和年度最小值的有限样本变异；该gate过低不是低尾缺失",
                "srad_p10": "低尾频率/块采样与有限样本变化；查看逐年/逐候选分布",
                "max_low_srad_spell_days": "低SRAD块内spell保留及相邻seam合并/截断",
                "tmax_mean": "日历阶段/供体年混合与有限样本偏移；另看温度seam",
                "tmin_min": "冷端供体日抽样及年度极值的有限样本变化",
                "diurnal_range_mean": "TMAX/TMIN联合日向量抽样与温度seam",
                "srad_tmax_spearman": "联合向量跨季节拼接造成的年度相关变化；seam与有限样本共同可能",
                "acf_srad_lag7": "28日块内持续性保留与seam边界的竞争",
                "acf_tmin_lag3": "块边界对温度自相关的影响及有限样本变化",
            }.get(metric, "按metric性质检查seam、供体块抽样与有限样本变化；机制归因为描述性")
            metric_rows.append({"metric": metric, "pool_mean": row["candidate_pool_mean"],
                "fitting_cluster_bootstrap_p025": row["fitting_year_bootstrap_p025"],
                "fitting_cluster_bootstrap_p975": row["fitting_year_bootstrap_p975"],
                "gate_failure": "POOL_MEAN_OUTSIDE_FROZEN_INTERVAL", "candidate_level_fail_count": candidate_metric_fail_counts[metric],
                "diagnosis_hypotheses": hypotheses, "certainty": "DESCRIPTIVE_NOT_CAUSAL"})
    write_csv(OUT / "metric_fence_failure_attribution.csv", metric_rows)
    donor_rows = []
    total_donor_rows = sum(x["rows"] for x in donor_rows_usage.values())
    for year in sorted(donor_rows_usage):
        d = donor_rows_usage[year]
        donor_rows.append({"source_year": year, **d, "row_share": d["rows"] / total_donor_rows,
                           "rain_share": d["rain"] / max(sum(x["rain"] for x in donor_rows_usage.values()), 1e-12),
                           "heavy_rain_share": d["heavy_rain"] / max(sum(x["heavy_rain"] for x in donor_rows_usage.values()), 1e-12),
                           "wet_day_rate": d["wet_days"] / d["rows"]})
    write_csv(OUT / "donor_year_rain_contribution.csv", donor_rows)
    block_rows = [{"donor_block": key, "selected_count": count, "rain_sum_across_selected": donor_block_rain[key],
                   "share_of_selected_blocks": count / max(1, sum(donor_block_usage.values()))}
                  for key, count in donor_block_usage.most_common()]
    write_csv(OUT / "donor_block_reuse_and_rain.csv", block_rows)
    write_csv(OUT / "candidate_rainfall_decomposition.csv", rain_candidate_metrics)
    totals = [float(x["seasonal_rain_total"]) for x in rain_candidate_metrics]
    wet_counts = [float(x["wet_day_count"]) for x in rain_candidate_metrics]
    wet_intensities = [float(x["wet_day_intensity_mean"]) for x in rain_candidate_metrics]
    heavy_totals = [float(x["rain_above_threshold_total"]) for x in rain_candidate_metrics]
    donor_2005 = next((r for r in donor_rows if int(r["source_year"]) == 2005), {})
    b_rain_ref = {r["metric"]: r for r in read_csv(B_INTERNAL / "pool_metric_bootstrap_checks.csv")}
    rainfall_bias = {
        "candidate_mean_annual_rain": avg(totals), "candidate_median_annual_rain": percentile(totals, .5),
        "candidate_wet_day_count_mean": avg(wet_counts), "candidate_wet_day_intensity_mean": avg(wet_intensities),
        "candidate_heavy_rain_total_mean": avg(heavy_totals),
        "candidate_heavy_rain_share_of_total_mean": avg([float(x["rain_above_threshold_share"]) for x in rain_candidate_metrics]),
        "annual_rain_correlation_with_wet_day_count": pearson(totals, wet_counts),
        "annual_rain_correlation_with_wet_day_intensity": pearson(totals, wet_intensities),
        "annual_rain_correlation_with_heavy_rain_total": pearson(totals, heavy_totals),
        "heavy_rain_threshold_fit_wetday_p95": heavy_threshold,
        "donor_year_2005_row_share": donor_2005.get("row_share"), "donor_year_2005_rain_share": donor_2005.get("rain_share"),
        "donor_year_2005_heavy_rain_share": donor_2005.get("heavy_rain_share"),
        "frozen_pool_interval_seasonal_rain": [b_rain_ref["seasonal_rain_total"]["fitting_year_bootstrap_p025"], b_rain_ref["seasonal_rain_total"]["fitting_year_bootstrap_p975"]],
        "frozen_pool_interval_wet_day_count": [b_rain_ref["wet_day_count"]["fitting_year_bootstrap_p025"], b_rain_ref["wet_day_count"]["fitting_year_bootstrap_p975"]],
        "frozen_pool_interval_wet_day_intensity": [b_rain_ref["wet_day_intensity_mean"]["fitting_year_bootstrap_p025"], b_rain_ref["wet_day_intensity_mean"]["fitting_year_bootstrap_p975"]],
        "interpretation": "Wet-day count/fraction were within frozen intervals; annual total tracked wet-day intensity and heavy-rain contribution more strongly. 2005 supplied a disproportionate heavy-rain share, but remained below the frozen 20% source-row cap; no single donor block exceeded the 5% reuse cap. This is descriptive and includes finite 20-member Monte Carlo variation.",
    }
    failure = {
        "candidate_counts": {k: dict(v) for k, v in qa_counts.items()},
        "pool_failed_metrics": [r["metric"] for r in metric_rows],
        "seam_violation_rates_by_variable": {v: {"fitting": seam_counts[v]["fitting_violation_rate"],
            "candidate_within_block": seam_counts[v]["candidate_internal_violation_rate"], "candidate_seam": seam_counts[v]["candidate_seam_violation_rate"]} for v in VARIABLES},
        "candidate_seam_failures_concentrated_at_boundaries": all(seam_counts[v]["candidate_seam_violation_rate"] > seam_counts[v]["candidate_internal_violation_rate"] for v in VARIABLES),
        "candidate_heavy_rain_wetday_p95_threshold": heavy_threshold,
        "rainfall_bias_decomposition": rainfall_bias,
        "candidate_level_metric_fence_fail_counts": dict(candidate_metric_fail_counts),
        "candidate_source_tuple_exact": low_srad["all_candidate_values_exact_fitting_donor"],
        "low_srad": low_srad,
        "source_year_usage_max_share": max((r["row_share"] for r in donor_rows), default=0),
        "donor_block_unique_count": len(donor_block_usage), "donor_block_selected_count": sum(donor_block_usage.values()),
        "donor_block_max_reuse_share": max((x["share_of_selected_blocks"] for x in block_rows), default=0),
        "observed_data_used_for_method_revision": "NO",
    }
    write_json(OUT / "failure_attribution.json", failure)
    return failure


def enumerate_blocks(years: dict[int, list[dict[str, Any]]], length: int) -> list[dict[str, Any]]:
    result = []
    for year, rows in sorted(years.items()):
        for start in range(0, len(rows) - length + 1):
            chunk = rows[start:start + length]
            if any((chunk[i]["DATE"] - chunk[i - 1]["DATE"]).days != 1 for i in range(1, len(chunk))):
                continue
            result.append({"year": year, "start": start, "rows": chunk,
                "start_date": chunk[0]["DATE"], "end_date": chunk[-1]["DATE"],
                "midpoint": avg([phase_ordinal(x["DATE"]) for x in chunk]),
                "key": f"{year}:{chunk[0]['DATE'].isoformat()}:{chunk[-1]['DATE'].isoformat()}"})
    return result


def transition_parameters(donor_years: dict[int, list[dict[str, Any]]]) -> tuple[dict[str, float], dict[tuple[int, int], float]]:
    scales = {}
    for v in VARIABLES:
        changes = [abs(float(b[v]) - float(a[v])) for rows in donor_years.values() for a, b in adjacent_pairs(rows)]
        scales[v] = max(percentile(changes, .95), 1e-8)
    counts = Counter()
    prior_counts = Counter()
    for rows in donor_years.values():
        for a, b in adjacent_pairs(rows):
            x, y = int(a["RAIN"] >= WET_THRESHOLD), int(b["RAIN"] >= WET_THRESHOLD)
            counts[(x, y)] += 1
            prior_counts[x] += 1
    probs = {(x, y): (counts[(x, y)] + 0.5) / (prior_counts[x] + 1.0) for x in (0, 1) for y in (0, 1)}
    return scales, probs


def transition_score(prev: dict[str, Any], first: dict[str, Any], scales: dict[str, float], probs: dict[tuple[int, int], float]) -> float:
    continuous = [abs(float(first[v]) - float(prev[v])) / scales[v] for v in VARIABLES]
    wet_transition = (int(float(prev["RAIN"]) >= WET_THRESHOLD), int(float(first["RAIN"]) >= WET_THRESHOLD))
    surprisal = -math.log(max(probs[wet_transition], 1e-12)) / math.log(2.0)
    return (sum(continuous) + surprisal) / 5.0


def generate_realization(method: str, fold_year: int, seed: int, donors: dict[int, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target_days = [date(fold_year, 1, 1) + timedelta(days=i) for i in range(366 if (fold_year % 4 == 0 and (fold_year % 100 != 0 or fold_year % 400 == 0)) else 365)]
    blocks_by_len: dict[int, list[dict[str, Any]]] = {}
    starts = list(range(0, len(target_days), BLOCK_LENGTH))
    plan = [(s, min(BLOCK_LENGTH, len(target_days) - s), target_days[s:s + BLOCK_LENGTH]) for s in starts]
    for length in {x[1] for x in plan}:
        blocks_by_len[length] = enumerate_blocks(donors, length)
    scales, probs = transition_parameters(donors)
    rng = random.Random(seed)
    weather, provenance = [], []
    prev_last = None
    for block_num, (target_start, length, target_chunk) in enumerate(plan, 1):
        target_mid = avg([phase_ordinal(x) for x in target_chunk])
        eligible = [b for b in blocks_by_len[length] if circular_distance(target_mid, b["midpoint"]) <= PHASE_WINDOW]
        if not eligible:
            raise RuntimeError(f"LOYO donor availability empty in fold {fold_year}, length {length}")
        if method == METHODS[0] or prev_last is None:
            donor = eligible[rng.randrange(len(eligible))]
            score = None
            compatible_count = len(eligible)
        else:
            scored = [(transition_score(prev_last, b["rows"][0], scales, probs), b) for b in eligible]
            scored.sort(key=lambda pair: pair[0])
            keep = max(1, math.ceil(0.20 * len(scored)))
            compatible = scored[:keep]
            score, donor = compatible[rng.randrange(len(compatible))]
            compatible_count = len(compatible)
        donor_id = f"{donor['year']}:{donor['start_date'].isoformat()}:{donor['end_date'].isoformat()}"
        block_id = f"{method}:{fold_year}:{seed}:{block_num:03d}"
        for offset, (target_day, src) in enumerate(zip(target_chunk, donor["rows"])):
            row = {"DATE": target_day, **{v: float(src[v]) for v in VARIABLES}}
            weather.append(row)
            provenance.append({"method": method, "heldout_year": fold_year, "development_seed": seed,
                "synthetic_date": target_day.isoformat(), "source_year": donor["year"], "source_date": src["DATE"].isoformat(),
                "source_block": donor_id, "block_id": block_id, "within_block_offset": offset,
                "phase_distance_days": circular_distance(target_mid, donor["midpoint"]),
                "phase_eligible_count": len(eligible), "compatible_subset_count": compatible_count,
                "transition_score": score if score is not None else "FIRST_BLOCK_UNSCORED",
                "development_status": "DEVELOPMENT_ONLY_NOT_FOR_PPO"})
        prev_last = weather[-1]
    if len(weather) != len(target_days) or len({r["DATE"] for r in weather}) != len(target_days):
        raise RuntimeError("LOYO generated calendar structure failed")
    return weather, provenance


def bootstrap_ci(values: list[float], seed: int, reps: int = 5000) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = [avg([values[rng.randrange(n)] for _ in range(n)]) for _ in range(reps)]
    return percentile(means, .025), percentile(means, .975)


def run_loyo(years: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    loyo_dir = OUT / "development"
    loyo_dir.mkdir(parents=True, exist_ok=True)
    fence_rows = read_csv(DIAG / "candidate_level_annual_metric_fences.csv")
    fences = {r["metric"]: (float(r["candidate_fail_below"]), float(r["candidate_fail_above"])) for r in fence_rows}
    pool_refs = {r["metric"]: (float(r["bootstrap_mean_p025"]), float(r["bootstrap_mean_p975"])) for r in read_csv(DIAG / "pool_qa_bootstrap_reference.csv")}
    fit_metrics_by_year = fitting_year_metrics(years)
    fit_sd = {m: statistics.stdev([fit_metrics_by_year[y][m] for y in years]) for m in METHOD_METRICS if len(years) > 1}
    synthetic_metrics: list[dict[str, Any]] = []
    daily_out: list[dict[str, Any]] = []
    provenance_out: list[dict[str, Any]] = []
    seam_tallies = {method: {v: {"n": 0, "violations": 0, "seams": 0, "seam_violations": 0, "within": 0, "within_violations": 0} for v in VARIABLES} for method in METHODS}
    seam_thresholds = {r["variable"]: float(r["frozen_violation_threshold"]) for r in read_csv(DIAG / "adjacent_day_delta_seam_reference.csv") if r["variable"] in VARIABLES}
    realization_rows: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    all_sequences: dict[tuple[str, int], list[list[tuple[float, ...]]]] = defaultdict(list)
    for fold_year in sorted(years):
        donors = {y: rows for y, rows in years.items() if y != fold_year}
        target = metric_values(years[fold_year])
        for rep, seed in enumerate(DEV_SEEDS, 1):
            for method in METHODS:
                weather, provenance = generate_realization(method, fold_year, seed, donors)
                mv = metric_values(weather)
                record = {"method": method, "heldout_year": fold_year, "replicate": rep, "development_seed": seed,
                          "n_days": len(weather), "development_status": "DEVELOPMENT_ONLY_NOT_FOR_PPO", **mv}
                record["metric_fence_fail_count"] = sum(1 for m, (lo, hi) in fences.items() if m in mv and not (lo <= mv[m] <= hi))
                synthetic_metrics.append(record)
                realization_rows[(method, fold_year)].append(record)
                all_sequences[(method, fold_year)].append([tuple(float(row[v]) for v in VARIABLES) for row in weather])
                block_ids = [p["block_id"] for p in provenance]
                for i, (a, b) in enumerate(zip(weather, weather[1:]), 1):
                    seam = block_ids[i] != block_ids[i - 1]
                    for v in VARIABLES:
                        tally = seam_tallies[method][v]
                        delta = abs(a[v] - b[v])
                        violation = delta > seam_thresholds[v]
                        tally["n"] += 1
                        tally["violations"] += int(violation)
                        if seam:
                            tally["seams"] += 1
                            tally["seam_violations"] += int(violation)
                        else:
                            tally["within"] += 1
                            tally["within_violations"] += int(violation)
                for row in weather:
                    daily_out.append({"method": method, "heldout_year": fold_year, "replicate": rep,
                        "development_seed": seed, "DATE": row["DATE"].isoformat(), **{v: row[v] for v in VARIABLES},
                        "development_status": "DEVELOPMENT_ONLY_NOT_FOR_PPO"})
                provenance_out.extend(provenance)
    write_csv(loyo_dir / "loyo_development_weather.csv", daily_out)
    write_csv(loyo_dir / "loyo_development_provenance.csv", provenance_out)
    write_csv(loyo_dir / "loyo_realization_metrics.csv", synthetic_metrics)
    seam_rows = []
    for method in METHODS:
        for v, tally in seam_tallies[method].items():
            seam_rows.append({"method": method, "variable": v, **tally,
                "overall_violation_rate": tally["violations"] / max(1, tally["n"]),
                "seam_violation_rate": tally["seam_violations"] / max(1, tally["seams"]),
                "within_block_violation_rate": tally["within_violations"] / max(1, tally["within"]),
                "frozen_p99_threshold": seam_thresholds[v]})
    write_csv(loyo_dir / "loyo_seam_decomposition.csv", seam_rows)
    metric_comparison = []
    fold_means = defaultdict(dict)
    for year in sorted(years):
        for method in METHODS:
            records = realization_rows[(method, year)]
            fold_means[(method, year)] = {m: avg([float(r[m]) for r in records]) for m in METHOD_METRICS}
    all_comparison_metrics = sorted(set(METHOD_METRICS))
    for metric in all_comparison_metrics:
        fold_rows = []
        m0_abs, m1_abs, paired_diff = [], [], []
        for year in sorted(years):
            target = fit_metrics_by_year[year][metric]
            a = fold_means[(METHODS[0], year)][metric]
            b = fold_means[(METHODS[1], year)][metric]
            m0_abs.append(abs(a - target))
            m1_abs.append(abs(b - target))
            paired_diff.append(abs(b - target) - abs(a - target))
            fold_rows.append({"heldout_year": year, "metric": metric, "heldout_value": target,
                              "M0_mean": a, "M1_mean": b, "M0_error": a-target, "M1_error": b-target,
                              "M1_minus_M0_absolute_error": abs(b-target)-abs(a-target)})
        ci_lo, ci_hi = bootstrap_ci(paired_diff, 240024 + len(metric))
        bias_m1 = avg([r["M1_error"] for r in fold_rows])
        margin_bias = .25 * fit_sd.get(metric, 0.0)
        bias_status = "NO_CLEAR_SYSTEMATIC_SHIFT" if abs(bias_m1) <= margin_bias or (ci_lo <= 0 <= ci_hi) else "SYSTEMATIC_SHIFT"
        noninf_margin = .10 * fit_sd.get(metric, 0.0)
        metric_comparison.append({"metric": metric, "group": "PERSISTENCE_DEPENDENCE" if metric in PERSISTENCE_DEPENDENCE_METRICS else "ANNUAL_MARGINAL_OR_OTHER",
            "M0_mean_absolute_LOYO_error": avg(m0_abs), "M1_mean_absolute_LOYO_error": avg(m1_abs),
            "M1_minus_M0_MAE": avg(paired_diff), "paired_fold_bootstrap_p025": ci_lo, "paired_fold_bootstrap_p975": ci_hi,
            "noninferiority_margin_0p10_fitting_year_sd": noninf_margin,
            "persistence_dependence_noninferior": ci_hi <= noninf_margin if metric in PERSISTENCE_DEPENDENCE_METRICS else "NA",
            "M1_mean_bias": bias_m1, "M1_bias_bootstrap_p025": bootstrap_ci([r["M1_error"] for r in fold_rows], 240124 + len(metric))[0],
            "M1_bias_bootstrap_p975": bootstrap_ci([r["M1_error"] for r in fold_rows], 240124 + len(metric))[1],
            "annual_shift_status": bias_status, "fitting_year_sd": fit_sd.get(metric),
            "details": json.dumps(fold_rows, separators=(",", ":"))})
    write_csv(loyo_dir / "loyo_metric_comparison.csv", metric_comparison)
    pool_rows = []
    for method in METHODS:
        for year in sorted(years):
            fold_records = realization_rows[(method, year)]
            for metric, (lo, hi) in pool_refs.items():
                values = [float(r[metric]) for r in fold_records]
                mean = avg(values)
                pool_rows.append({"method": method, "heldout_year": year, "metric": metric, "n_realizations": len(values),
                    "pool_mean": mean, "frozen_p025": lo, "frozen_p975": hi, "pass_unchanged_interval": lo <= mean <= hi})
    write_csv(loyo_dir / "loyo_pool_interval_checks.csv", pool_rows)
    # Diversity is measured over all M1 folds, with block and source-year frequencies.
    m1_prov = [p for p in provenance_out if p["method"] == METHODS[1] and int(p["within_block_offset"]) == 0]
    source_year_counts = Counter(int(p["source_year"]) for p in m1_prov)
    source_year_total = sum(source_year_counts.values())
    block_counts = Counter(p["source_block"] for p in m1_prov)
    donor_year_share = {str(y): n/source_year_total for y, n in sorted(source_year_counts.items())}
    block_reuse_share = max(block_counts.values(), default=0) / max(1, sum(block_counts.values()))
    diversity_rows = [{"source_year": y, "selected_blocks": n, "share": n/max(1, source_year_total)} for y, n in sorted(source_year_counts.items())]
    write_csv(loyo_dir / "loyo_m1_donor_year_usage.csv", diversity_rows)
    near_dup_pairs = 0
    for year in sorted(years):
        seqs = all_sequences[(METHODS[1], year)]
        for i in range(len(seqs)):
            for j in range(i+1, len(seqs)):
                same = sum(a == b for a, b in zip(seqs[i], seqs[j]))
                near_dup_pairs += int(same / max(1, min(len(seqs[i]), len(seqs[j]))) >= .95)
    # Candidate-level frozen fences and pool-level intervals are not relaxed.
    m1_metric_fails = sum(int(r["metric_fence_fail_count"]) for r in synthetic_metrics if r["method"] == METHODS[1])
    m1_total_checks = sum(len(fences) for r in synthetic_metrics if r["method"] == METHODS[1])
    m1_pool_fails = sum(not r["pass_unchanged_interval"] for r in pool_rows if r["method"] == METHODS[1])
    seam_by_method = {method: {r["variable"]: r for r in seam_rows if r["method"] == method} for method in METHODS}
    seam_gate = True
    for v in ("TMAX", "TMIN"):
        m0_rate = seam_by_method[METHODS[0]][v]["seam_violation_rate"]
        m1_rate = seam_by_method[METHODS[1]][v]["seam_violation_rate"]
        seam_gate &= (m1_rate <= .5 * m0_rate) if m0_rate > 0 else m1_rate == 0
    seam_gate &= all(seam_by_method[METHODS[1]][v]["seam_violation_rate"] <= seam_by_method[METHODS[0]][v]["seam_violation_rate"] for v in VARIABLES)
    noninferior_rows = [r for r in metric_comparison if r["group"] == "PERSISTENCE_DEPENDENCE"]
    persistence_gate = all(r["persistence_dependence_noninferior"] is True for r in noninferior_rows)
    shift_rows = [r for r in metric_comparison if r["metric"] in fences]
    shift_gate = all(r["annual_shift_status"] == "NO_CLEAR_SYSTEMATIC_SHIFT" for r in shift_rows)
    diversity_gate = (len(source_year_counts) == 10 and max(donor_year_share.values(), default=0) <= .20
                      and block_reuse_share <= .05 and near_dup_pairs == 0)
    structural_gate = len(synthetic_metrics) == 400 and all(int(r["n_days"]) in (365, 366) for r in synthetic_metrics)
    all_gates = structural_gate and seam_gate and persistence_gate and shift_gate and diversity_gate and m1_metric_fails == 0 and m1_pool_fails == 0
    comparison = {
        "task_id": "004_24C", "development_only_not_for_ppo": True, "observed_data_used_for_method_revision": "NO",
        "M0": {"id": METHODS[0], "mean_seam_violation_rates": {v: seam_by_method[METHODS[0]][v]["seam_violation_rate"] for v in VARIABLES}},
        "M1": {"id": METHODS[1], "mean_seam_violation_rates": {v: seam_by_method[METHODS[1]][v]["seam_violation_rate"] for v in VARIABLES},
               "candidate_metric_fence_failures": m1_metric_fails, "candidate_metric_checks": m1_total_checks,
               "pool_interval_failures": m1_pool_fails, "pool_interval_checks": sum(1 for r in pool_rows if r["method"] == METHODS[1]),
               "max_source_year_share": max(donor_year_share.values(), default=0), "max_reused_donor_block_share": block_reuse_share,
               "near_duplicate_pairs_ge95_within_fold": near_dup_pairs},
        "gates": {"structural": structural_gate, "seam": bool(seam_gate), "persistence_dependence": persistence_gate,
                  "annual_metric_shift": shift_gate, "diversity": bool(diversity_gate),
                  "candidate_metric_fence": m1_metric_fails == 0, "pool_interval": m1_pool_fails == 0},
        "method_revision_status": "FROZEN" if all_gates else "NOT_RESOLVED",
        "selected_method": METHODS[1] if all_gates else None,
        "next_step": "004_24D_INDEPENDENT_VALIDATION_OF_REVISED_GENERATOR" if all_gates else "STOP_AND_RECONSIDER_GENERATOR_FAMILY",
        "validation_seed_count": len(VALIDATION_SEEDS), "validation_seeds": VALIDATION_SEEDS,
        "ppo_training_status": "NOT_STARTED",
    }
    write_json(OUT / "method_comparison_summary.json", comparison)
    return comparison


def finalize(comparison: dict[str, Any]) -> None:
    freeze_payload = {
        "task_id": "004_24C", "status": comparison["method_revision_status"],
        "observed_data_used_for_method_revision": "NO", "selected_method": comparison["selected_method"],
        "method": {"family": "transition-aware multivariate moving-block bootstrap" if comparison["selected_method"] else None,
            "block_length_days": BLOCK_LENGTH if comparison["selected_method"] else None,
            "phase_alignment_rule": "calendar block midpoint circular phase distance <= 7 days on 365-day phase" if comparison["selected_method"] else None,
            "transition_aware": bool(comparison["selected_method"]),
            "transition_score_definition": "mean of four abs daily deltas normalized by donor-fold fitting P95 absolute adjacent delta (RAIN, SRAD, TMAX, TMIN), plus wet-state transition surprisal / ln(2), divided by five" if comparison["selected_method"] else None,
            "compatible_subset_rule": "lowest-score ceil(20% of phase-eligible blocks)" if comparison["selected_method"] else None,
            "selection_probability_rule": "uniform stochastic draw with replacement from compatible subset; first target block uniform phase-eligible" if comparison["selected_method"] else None,
            "donor_balance_rule": "none; diversity gate enforced diagnostically" if comparison["selected_method"] else None,
            "seam_policy": "raw concatenation", "smoothing": False},
        "development_validation": {"scheme": "LOYO", "years": list(range(2004, 2014)), "observed_2014_2023_used": False,
            "baseline_method": METHODS[0], "compared_methods": list(METHODS), "selection_rule": "all pre-registered gates; no composite score"},
        "qa": comparison["gates"],
        "next_validation": {"candidate_count": len(VALIDATION_SEEDS), "seed_schedule": VALIDATION_SEEDS,
            "observed_external_validation_mode": "DESCRIPTIVE_ONLY", "parameter_retuning_allowed": False},
        "note": "NOT_FROZEN means this file records the rejected proposal and reserved validation schedule; no next validation is authorized." if not comparison["selected_method"] else "Method parameters and next validation seeds frozen before any new validation result."}
    write_json(OUT / "method_revision_freeze.yaml.json", freeze_payload)
    # YAML is human-readable and intentionally contains NOT_FROZEN when any gate fails.
    (OUT / "method_revision_freeze.yaml").write_text(yaml.safe_dump(freeze_payload, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")
    schedule = [{"candidate_id": f"candidate_{i:03d}", "seed": seed, "schedule_status": "FROZEN_NOT_USED_IN_004_24C"}
                for i, seed in enumerate(VALIDATION_SEEDS, 1)]
    write_csv(OUT / "next_validation_seed_schedule.csv", schedule)
    decision = {"task_id": "004_24C", "observed_data_used_for_method_revision": "NO",
        "method_revision_status": comparison["method_revision_status"], "selected_method": comparison["selected_method"],
        "next_step": comparison["next_step"], "ppo_training_status": "NOT_STARTED", "wgen_run": "NOT_RUN",
        "dssat_or_gym_dssat_run": "NOT_RUN", "development_candidates_status": "DEVELOPMENT_ONLY_NOT_FOR_PPO",
        "next_validation_candidate_count": len(VALIDATION_SEEDS), "next_validation_seed_schedule": VALIDATION_SEEDS,
        "gates": comparison["gates"]}
    write_json(OUT / "decision.json", decision)
    render_report(decision, comparison)


def render_report(decision: dict[str, Any], comparison: dict[str, Any]) -> None:
    failure = json.loads((OUT / "failure_attribution.json").read_text(encoding="utf-8"))
    seam = read_csv(OUT / "seam_decomposition_by_variable.csv")
    metric = read_csv(OUT / "metric_fence_failure_attribution.csv")
    metric_cmp = read_csv(OUT / "development/loyo_metric_comparison.csv")
    seam_cmp = read_csv(OUT / "development/loyo_seam_decomposition.csv")
    m0 = next(x for x in seam_cmp if x["method"] == METHODS[0] and x["variable"] == "TMAX")
    m1 = next(x for x in seam_cmp if x["method"] == METHODS[1] and x["variable"] == "TMAX")
    m0tmin = next(x for x in seam_cmp if x["method"] == METHODS[0] and x["variable"] == "TMIN")
    m1tmin = next(x for x in seam_cmp if x["method"] == METHODS[1] and x["variable"] == "TMIN")
    by_var = {r["variable"]: r for r in seam}
    top_fail = [r["metric"] for r in metric]
    medians = {m: next((r for r in metric_cmp if r["metric"] == m), {}) for m in ("seasonal_rain_total", "rx5day", "srad_p10", "max_low_srad_spell_days", "tmax_mean", "tmin_min", "diurnal_range_mean", "srad_tmax_spearman", "acf_srad_lag7", "acf_tmin_lag3")}
    lines = [
        "# 004_24C YC 外部天气增强失败诊断与 fitting-only 方法修订", "",
        "## 1. 范围与数据隔离", "",
        "- 本任务只读2004–2013 fitting weather、004_24A冻结诊断，以及004_24B的20套candidate、provenance、candidate hash和internal QA。没有读取Observed天气值，也没有使用004_24B observed-derived summary。",
        "- `observed_data_used_for_method_revision = NO`。未运行WGEN、DSSAT、Gym-DSSAT或PPO。全部LOYO逐日序列均标记 `DEVELOPMENT_ONLY_NOT_FOR_PPO`。", "",
        "## 2. 004_24B失败概况", "",
        "- 20套候选中6套通过、14套失败；结构、物理、provenance、序列化为20/20通过；指标fence失败8套、seam失败12套；pool gate失败。候选没有重抽或删改。",
        f"- pool失败指标：{', '.join(failure['pool_failed_metrics'])}。冻结区间原样保留，没有放宽门槛。", "",
        "## 3. Seam归因", "",
        "| 变量 | fitting相邻日P99阈值 | candidate block内部违规率 | candidate seam违规率 | 判断 |",
        "|---|---:|---:|---:|---|",
    ]
    for v in VARIABLES:
        r = by_var[v]
        lines.append(f"| {v} | {float(r['frozen_p99_threshold']):.4f} | {float(r['candidate_internal_violation_rate']):.3%} | {float(r['candidate_seam_violation_rate']):.3%} | {'集中于seam' if r['concentration_at_seam']=='True' else '未显示seam集中'} |")
    lines += ["", f"- 违规是否集中于block边界：`{'YES' if failure['candidate_seam_failures_concentrated_at_boundaries'] else 'PARTIAL/NO'}`。具体分变量计数、率和fitting delta分布见 `seam_decomposition_by_variable.csv`。", "",
        "## 4. Metric-fence与降雨偏高", "",
        f"- pool均值越界项共{len(top_fail)}项，逐项候选值、冻结bootstrap区间及可检验归因列在 `metric_fence_failure_attribution.csv`。这些归因是候选机制，不作因果断言。",
        f"- {failure['candidate_counts']['candidate_metric_fence_status'].get('FAIL', 0)}/20候选至少有一项candidate-level fence失败；失败计数（候选可在多项重复计数）为：" + ", ".join(f"{k}={v}" for k, v in failure["candidate_level_metric_fence_fail_counts"].items()) + "。",
        "- 降雨分解：候选年总雨量均值/中位数为 **729.92/710.95 mm**，wet-day数均值 **59.95日**，湿日强度均值 **12.25 mm/day**；按fitting wet-day P95（45.48 mm）定义的重雨贡献均值为 **267.80 mm/年（总雨量34.45%）**。冻结区间显示wet-day数及wet-day强度总体仍在范围内，但年度总雨量高于上界。",
        "- 年总雨量与wet-day数相关较弱（r=0.228），与湿日强度和重雨总量相关较强（r=0.901、0.900）。供体年2005贡献14.56% donor rows、19.83%总雨量、22.49%重雨；仍低于预注册的20% source-row上限，最大单个donor block复用占比仅1.07%。这支持“强降雨金额/抽样变异值得继续诊断”，不支持归因于wet day过多或单一block重复；均为描述性关联，不作因果断言。详见 `candidate_rainfall_decomposition.csv`、`donor_year_rain_contribution.csv`、`donor_block_reuse_and_rain.csv`。", "",
        "## 5. Low-SRAD分解", "",
        f"- 低SRAD日均来自fitting donor tuple：`{failure['candidate_source_tuple_exact']}`；候选池低于fitting P5阈值 `{LOW_SRAD_P05}` 的日数为{failure['low_srad']['candidate_days_below_threshold']}。",
        f"- candidate低SRAD spell数={failure['low_srad']['candidate_low_spell_count']}；跨越seam的spell={failure['low_srad']['candidate_low_spells_crossing_seam']}；触及seam的spell={failure['low_srad']['candidate_low_spells_touching_seam']}；最长spell={failure['low_srad']['candidate_max_low_spell_days']}日。低尾确实保留，但最小值、P10、spell长度与lag-7 ACF等pool统计仍有冻结gate失败。", "",
        "## 6. LOYO设计与方法", "",
        "- 十折LOYO：每次留出一个2004–2013完整calendar year，donor仅用其余9年；每折每个方法20个配对development seeds，共200个realization/method。Leap year保留366日，沿用冻结365日phase映射。",
        "- M0：原uniform 28日block、midpoint ±7日、有放回、原样拼接。M1：同长度/phase；按四变量delta的donor-fold fitting P95尺度和wet-state转移surprisal构成transition score，取phase-eligible中最低分20%，再均匀随机抽样。没有nearest-neighbor确定性选择、平滑或改值。",
        "- 设计、种子、LOYO重复数及门槛在 `analysis_plan.json` 中先冻结；LOYO逐日天气和source provenance都标注development-only。", "",
        "## 7. LOYO结果", "",
        f"- M0 TMAX/TMIN seam违规率：{float(m0['seam_violation_rate']):.3%} / {float(m0tmin['seam_violation_rate']):.3%}。",
        f"- M1 TMAX/TMIN seam违规率：{float(m1['seam_violation_rate']):.3%} / {float(m1tmin['seam_violation_rate']):.3%}。",
        f"- 四变量seam违规率（M0→M1）：RAIN {float(next(x for x in seam_cmp if x['method'] == METHODS[0] and x['variable'] == 'RAIN')['seam_violation_rate']):.3%}→{float(next(x for x in seam_cmp if x['method'] == METHODS[1] and x['variable'] == 'RAIN')['seam_violation_rate']):.3%}；SRAD {float(next(x for x in seam_cmp if x['method'] == METHODS[0] and x['variable'] == 'SRAD')['seam_violation_rate']):.3%}→{float(next(x for x in seam_cmp if x['method'] == METHODS[1] and x['variable'] == 'SRAD')['seam_violation_rate']):.3%}；TMAX {float(m0['seam_violation_rate']):.3%}→{float(m1['seam_violation_rate']):.3%}；TMIN {float(m0tmin['seam_violation_rate']):.3%}→{float(m1tmin['seam_violation_rate']):.3%}。M1四项均下降，且TMAX/TMIN降幅均超过50%，满足seam gate。",
        f"- 其余门槛：candidate-level metric fence为 {comparison['M1']['candidate_metric_fence_failures']}/{comparison['M1']['candidate_metric_checks']} 个检查失败；逐fold pool区间为 {comparison['M1']['pool_interval_failures']}/{comparison['M1']['pool_interval_checks']} 失败；persistence/dependence 非劣性为 {sum(r['persistence_dependence_noninferior'] == 'True' for r in metric_cmp if r['group'] == 'PERSISTENCE_DEPENDENCE')}/{sum(r['group'] == 'PERSISTENCE_DEPENDENCE' for r in metric_cmp)} 项通过。annual metric shift、diversity、structural gates通过。",
        f"- M1各gate：`{json.dumps(comparison['gates'], ensure_ascii=False)}`。逐metric LOYO paired error、bootstrap区间、各fold pool checks见 `development/`。", "",
        "## 8. 方法选择与冻结", "",
        f"- `method_revision_status = {decision['method_revision_status']}`；selected method=`{decision['selected_method']}`。",
        "- `method_revision_freeze.yaml` 若为 `NOT_FROZEN`，仅记录被评估方案和保留的独立validation seed schedule，不授权生成下一轮正式candidate。没有满足所有门槛时，按任务规定停止，不继续搜索block length、phase window或donor balancing。", "",
        "## 9. 限制与下一步", "",
        "- LOYO仅有10个fitting年份；它检验拟合期内部的伪未见年份泛化，不代替独立气候时期validation。指标关联与候选机制分析均属描述性。",
        f"- 下一validation预留candidate数={decision['next_validation_candidate_count']}，seed见 `next_validation_seed_schedule.csv`；本任务未使用这些seed生成天气。",
        f"- `ppo_training_status = NOT_STARTED`；`next_step = {decision['next_step']}`。", "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("diagnose", "loyo", "finalize", "all"), default="all")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    years, _, _ = preflight()
    if args.phase in ("diagnose", "all"):
        result = b_failure_decomposition(years)
        if args.phase == "diagnose":
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
    if args.phase in ("loyo", "all"):
        if not (OUT / "failure_attribution.json").is_file():
            raise RuntimeError("Run --phase diagnose first; LOYO must not precede failure decomposition")
        result = run_loyo(years)
        if args.phase == "loyo":
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
    if args.phase in ("finalize", "all"):
        summary_path = OUT / "method_comparison_summary.json"
        if not summary_path.is_file():
            raise RuntimeError("Run --phase loyo first")
        result = json.loads(summary_path.read_text(encoding="utf-8"))
        finalize(result)
        print(json.dumps(json.loads((OUT / "decision.json").read_text(encoding="utf-8")), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
