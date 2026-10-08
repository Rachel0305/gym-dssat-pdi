"""Predeclared FQA 80/20 WGEN climate screen against the frozen 2005-2013 fitting weather."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FIT = ROOT / "results/fqa_weather_resume_041/fitting_weather.csv"
PLAN = ROOT / "results/fqa_wgen_multiyear_heldout_gate_045/full_schedule_plan.json"
FIELDS = ("RAIN", "SRAD", "TMAX", "TMIN")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parsed(rows):
    return [{"date": date.fromisoformat(r["DATE"]), **{k: float(r[k]) for k in FIELDS}} for r in rows]


def continuous(rows):
    return bool(rows) and len({r["date"] for r in rows}) == len(rows) and all(b["date"] - a["date"] == timedelta(days=1) for a, b in zip(rows, rows[1:]))


def physical(rows):
    return continuous(rows) and all(all(math.isfinite(r[k]) for k in FIELDS) and r["RAIN"] >= 0 and r["SRAD"] >= 0 and r["TMAX"] >= r["TMIN"] for r in rows)


def mean(xs):
    return statistics.fmean(xs) if xs else None


def sd(xs):
    return statistics.stdev(xs) if len(xs) > 1 else None


def pearson(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    a, b = [x - mx for x in xs], [y - my for y in ys]
    denom = math.sqrt(sum(x*x for x in a) * sum(y*y for y in b))
    return sum(x*y for x, y in zip(a, b)) / denom if denom else None


def longest_dry(rain):
    longest = current = 0
    for value in rain:
        current = current + 1 if value <= 0 else 0
        longest = max(longest, current)
    return longest


def metric(rows):
    rain = [r["RAIN"] for r in rows]
    wet = [x for x in rain if x > 0]
    tmax = [r["TMAX"] for r in rows]
    tmin = [r["TMIN"] for r in rows]
    srad = [r["SRAD"] for r in rows]
    wet_binary = [1.0 if x > 0 else 0.0 for x in rain]
    return {"days": len(rows), "rain_total_mm": sum(rain), "wet_days": len(wet), "wet_intensity_mm": mean(wet), "max_daily_rain_mm": max(rain), "longest_dry_spell_days": longest_dry(rain), "tmax_mean_c": mean(tmax), "tmin_mean_c": mean(tmin), "srad_mean_mj_m2_day": mean(srad), "corr_rain_srad": pearson(rain, srad), "corr_rain_tmax": pearson(rain, tmax), "corr_rain_tmin": pearson(rain, tmin), "corr_tmax_tmin": pearson(tmax, tmin), "lag1_wet": pearson(wet_binary[:-1], wet_binary[1:]), "lag1_tmax": pearson(tmax[:-1], tmax[1:])}


def window(rows, start_md, end_md):
    year = rows[0]["date"].year
    start = date(year, *start_md)
    end = date(year, *end_md)
    selected = [r for r in rows if start <= r["date"] <= end]
    expected = (end - start).days + 1
    if len(selected) != expected or selected[0]["date"] != start or selected[-1]["date"] != end or not physical(selected):
        raise RuntimeError(f"incomplete/invalid fixed window {year} {start_md}..{end_md}")
    return selected


def bias_sd(syn, fit):
    stdev = sd(fit)
    return abs(mean(syn) - mean(fit)) / stdev if stdev and stdev > 0 else None


def quantile(values, p):
    xs = sorted(values)
    at = (len(xs) - 1) * p
    lo = math.floor(at)
    hi = math.ceil(at)
    return xs[lo] + (xs[hi] - xs[lo]) * (at - lo)


def main():
    frozen = json.loads((ROOT / "results/fqa_weather_resume_041/final_gate.json").read_text(encoding="utf-8"))
    if sha(FIT) != frozen["output_sha256"]["fitting_weather.csv"].upper():
        raise RuntimeError("fitting source hash changed")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    manifests = {s: read_csv(OUT / s / "episode_manifest.csv") for s in ("train", "heldout")}
    results = {s: json.loads((OUT / s / "result.json").read_text(encoding="utf-8")) for s in manifests}
    rows = manifests["train"] + manifests["heldout"]
    if len(rows) != 100 or len(manifests["train"]) != 80 or len(manifests["heldout"]) != 20:
        raise RuntimeError("80/20 archive count incomplete")
    if any(results[s]["status"] != "GENERATED_ARCHIVED" for s in results):
        raise RuntimeError("pool generation not complete")
    if [(int(r["year"]), int(r["weather_seed"])) for r in manifests["train"]] != [(x["historical_year"], x["weather_seed"]) for x in plan["train"]] or [(int(r["year"]), int(r["weather_seed"])) for r in manifests["heldout"]] != [(x["historical_year"], x["weather_seed"]) for x in plan["heldout"]]:
        raise RuntimeError("actual manifest differs from frozen schedule")
    archive = []
    for row in rows:
        path = ROOT / row["weather_path"]
        if not path.is_file() or sha(path) != row["weather_sha256"]:
            raise RuntimeError(f"weather archive hash mismatch: {path}")
        daily = parsed(read_csv(path))
        if len(daily) != int(row["days"]) or not physical(daily):
            raise RuntimeError(f"archive row/physical check failed: {path}")
        proof = json.loads((ROOT / row["runtime_evidence_path"]).read_text(encoding="utf-8"))
        if not (proof["wther"] == "W" and proof["wsta_confirmed"] and proof["yaml_bootstrap_confirmed"] and proof["runtime_rseed1"] == int(row["weather_seed"]) and proof["runtime_cli_sha256"] == proof["source_cli_sha256"]):
            raise RuntimeError(f"runtime proof mismatch: {row['split']} {row['episode_index']}")
        archive.append((row, daily))
    if len({r["weather_sha256"] for r in rows}) != 100 or {int(r["weather_seed"]) for r in rows} != set(range(1001, 1101)):
        raise RuntimeError("seed or realization diversity failed")
    start_md = (6, 11)
    earliest_end = min((daily[-1]["date"].month, daily[-1]["date"].day) for _, daily in archive)
    end_md = min((9, 19), earliest_end)
    common_days = (date(2007, *end_md) - date(2007, *start_md)).days + 1
    if common_days < 90:
        raise RuntimeError(f"common weather window too short: {common_days} days")
    fit_by_year = defaultdict(list)
    for r in parsed(read_csv(FIT)):
        fit_by_year[r["date"].year].append(r)
    if set(fit_by_year) != set(range(2005, 2014)):
        raise RuntimeError("fitting years drift")
    fit_units = [(year, window(fit_by_year[year], start_md, end_md)) for year in range(2005, 2014)]
    syn_units = [(row, window(daily, start_md, end_md)) for row, daily in archive]
    fit_metrics = [{"year": year, **metric(daily)} for year, daily in fit_units]
    syn_metrics = [{"split": row["split"], "episode_index": int(row["episode_index"]), "year": int(row["year"]), "weather_seed": int(row["weather_seed"]), "weather_sha256": row["weather_sha256"], **metric(daily)} for row, daily in syn_units]
    write_csv(OUT / "fitting_year_fixed_window_metrics.csv", fit_metrics)
    write_csv(OUT / "synthetic_fixed_window_metrics.csv", syn_metrics)
    monthly = []
    for group, units in (("fitting", fit_units), ("train", [(r, d) for r, d in syn_units if r["split"] == "train"]), ("heldout", [(r, d) for r, d in syn_units if r["split"] == "heldout"]), ("all", syn_units)):
        for month in (6, 7, 8, 9):
            totals = [sum(day["RAIN"] for day in daily if day["date"].month == month) for _, daily in units]
            monthly.append({"group": group, "month": month, "mean_rain_mm": mean(totals), "sd_rain_mm": sd(totals), "n": len(totals)})
    write_csv(OUT / "monthly_rain_profile.csv", monthly)
    monthly_fit = [r["mean_rain_mm"] for r in monthly if r["group"] == "fitting"]
    monthly_all = [r["mean_rain_mm"] for r in monthly if r["group"] == "all"]
    fit_pooled = [r for _, daily in fit_units for r in daily]
    syn_pooled = [r for _, daily in syn_units for r in daily]
    tests = {}
    diagnostics = {}
    for key in ("rain_total_mm", "wet_days", "tmax_mean_c", "tmin_mean_c", "srad_mean_mj_m2_day"):
        fit_values = [r[key] for r in fit_metrics]
        syn_values = [r[key] for r in syn_metrics]
        z = bias_sd(syn_values, fit_values)
        diagnostics[key] = {"fitting_mean": mean(fit_values), "fitting_interannual_sd": sd(fit_values), "train_mean": mean([r[key] for r in syn_metrics if r["split"] == "train"]), "heldout_mean": mean([r[key] for r in syn_metrics if r["split"] == "heldout"]), "synthetic_mean": mean(syn_values), "absolute_bias_in_fitting_sd": z, "synthetic_p05": quantile(syn_values, 0.05), "synthetic_p95": quantile(syn_values, 0.95)}
        tests[key + "_bias_le_2sd"] = z is not None and z <= 2.0
    profile_corr = pearson(monthly_fit, monthly_all)
    diagnostics["monthly_rain_profile_pearson"] = profile_corr
    tests["monthly_rain_profile_ge_0p5"] = profile_corr is not None and profile_corr >= 0.5
    for field in ("TMAX", "TMIN", "SRAD"):
        ratio = sd([r[field] for r in syn_pooled]) / sd([r[field] for r in fit_pooled])
        diagnostics[field.lower() + "_daily_sd_ratio"] = ratio
        tests[field.lower() + "_daily_sd_ratio_0p35_2"] = 0.35 <= ratio <= 2.0
    dep_keys = ("corr_rain_srad", "corr_rain_tmax", "corr_rain_tmin", "corr_tmax_tmin", "lag1_wet", "lag1_tmax")
    dep_diffs = {}
    for key in dep_keys:
        fit_vals = [r[key] for r in fit_metrics if r[key] is not None]
        syn_vals = [r[key] for r in syn_metrics if r[key] is not None]
        dep_diffs[key] = abs(mean(fit_vals) - mean(syn_vals)) if fit_vals and syn_vals else None
    max_dep = max(v for v in dep_diffs.values() if v is not None)
    diagnostics["dependence_group_mean_absolute_differences"] = dep_diffs
    diagnostics["dependence_max_difference"] = max_dep
    tests["dependence_max_le_0p60"] = max_dep <= 0.60 and all(v is not None for v in dep_diffs.values())
    diagnostics["extremes_descriptive"] = {"fit_max_daily_rain_mm": max(r["max_daily_rain_mm"] for r in fit_metrics), "synthetic_max_daily_rain_mm": max(r["max_daily_rain_mm"] for r in syn_metrics), "fit_mean_longest_dry_spell_days": mean([r["longest_dry_spell_days"] for r in fit_metrics]), "synthetic_mean_longest_dry_spell_days": mean([r["longest_dry_spell_days"] for r in syn_metrics]), "fit_mean_wet_intensity_mm": mean([r["wet_intensity_mm"] for r in fit_metrics if r["wet_intensity_mm"] is not None]), "synthetic_mean_wet_intensity_mm": mean([r["wet_intensity_mm"] for r in syn_metrics if r["wet_intensity_mm"] is not None]), "fit_tmax_p95_c": quantile([r["TMAX"] for r in fit_pooled], 0.95), "synthetic_tmax_p95_c": quantile([r["TMAX"] for r in syn_pooled], 0.95), "fit_tmin_p05_c": quantile([r["TMIN"] for r in fit_pooled], 0.05), "synthetic_tmin_p05_c": quantile([r["TMIN"] for r in syn_pooled], 0.05)}
    resource = read_csv(OUT / "train/resource_usage.csv") + read_csv(OUT / "heldout/resource_usage.csv")
    peak_rss = max(float(r["process_tree_rss_mb"]) for r in resource)
    structural = {"generation_100_complete": len(rows) == 100 and all(results[s]["status"] == "GENERATED_ARCHIVED" for s in results), "train_heldout_seed_separation": {int(r["weather_seed"]) for r in manifests["train"]}.isdisjoint({int(r["weather_seed"]) for r in manifests["heldout"]}), "all_weather_files_and_runtime_proofs_verified": True, "all_100_realizations_distinct": len({r["weather_sha256"] for r in rows}) == 100, "fixed_window_at_least_90_days": common_days >= 90, "rss_below_1536_mb": peak_rss < 1536}
    climate_pass = all(tests.values())
    overall = "PASS_INPUT_QC_ONLY" if all(structural.values()) and climate_pass else "FAIL_CLIMATE_SCREEN" if all(structural.values()) else "FAIL_STRUCTURAL"
    summary = {"status": overall, "fitting_source": "041 D222 rain values unchanged; D222 blanks -> 0 mm modeling assumption; candidate overlays for missing T/SRAD", "fitting_years": list(range(2005, 2014)), "generation_episode_counts": {s: len(manifests[s]) for s in manifests}, "actual_weather_days": sum(int(r["days"]) for r in rows), "unique_realization_hashes": 100, "common_window": {"start_month_day": "06-11", "end_month_day": f"{end_md[0]:02d}-{end_md[1]:02d}", "days": common_days}, "structural_checks": structural, "climate_tests": tests, "diagnostics": diagnostics, "peak_process_tree_rss_mb": round(peak_rss, 2), "decision_rules": {"rain_total_wet_days_mean_bias": "<=2 fitting interannual SD", "monthly_rain_profile_pearson": ">=0.50", "tmax_tmin_srad_mean_bias": "<=2 fitting interannual SD", "tmax_tmin_srad_daily_sd_ratio": "0.35 to 2.0", "dependence_group_mean_max_abs_difference": "<=0.60", "interpretation": "heuristic descriptive screen, not hypothesis test; failed screen does not authorize source or seed changes"}, "limits": ["checkpoint used only for deterministic actions, no PPO learning or policy conclusion", "fixed-window generation does not certify full-year weather", "native FIELD coordinate warning remains unresolved"]}
    write_json(OUT / "final_gate.json", summary)
    print(json.dumps({"status": overall, "episodes": 100, "window": summary["common_window"], "failed_climate_tests": [k for k, v in tests.items() if not v], "peak_rss_mb": summary["peak_process_tree_rss_mb"]}, ensure_ascii=False))
    return 0 if overall == "PASS_INPUT_QC_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
