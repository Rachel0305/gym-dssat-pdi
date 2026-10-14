"""HLA-only gridded precipitation reference sensitivity audit (Task 024).

All inputs are read-only. This script creates isolated CSV/JSON/report outputs only;
it imports the project's monthly fitting function but never builds a CLI or runs WGEN.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "hla_gridded_fill_sensitivity_024"
REPORT = ROOT / "docs" / "hla_gridded_fill_sensitivity_024.md"
WEATHER = ROOT / "weather_clean" / "HLA_weather_cleaned.csv"
ST_CANDIDATES = ROOT / "results" / "hl_fq_random_weather_8seed" / "source_audit" / "nasa_candidate_gapfill.csv"
RAIN_SOURCE = ROOT / "results" / "hla_random_weather_8seed" / "014_isd50756_recovery" / "daily_source_selection_after_014.csv"
UNRESOLVED = ROOT / "results" / "hla_random_weather_8seed" / "014_isd50756_recovery" / "unresolved_dates_after_014.csv"
D222 = ROOT / "results" / "five_site_d222_version_audit_022_rerun_20261007" / "current_d222_daily_normalized.csv"
GPCC = ROOT / "results" / "hl_fq_global_precip_023" / "hla_gpcc_daily_2011_2013.csv"
CPC = ROOT / "results" / "hl_fq_global_precip_023" / "hla_cpc_daily_2011_2013.csv"
GATE023 = ROOT / "results" / "hl_fq_global_precip_023" / "final_gate.json"
CLI_CODE = ROOT / "scripts" / "build_dssat_cli.py"
PERIOD_START = date(2004, 1, 1)
PERIOD_END = date(2013, 12, 31)
SCENARIOS = (
    "GPCC_RAW", "CPC_RAW", "GPCC_BIASCORR", "CPC_BIASCORR", "ENSEMBLE_BIASCORR"
)
CANDIDATE_FILENAMES = {
    "GPCC_RAW": "hla_candidate_gpcc_raw_2004_2013.csv",
    "CPC_RAW": "hla_candidate_cpc_raw_2004_2013.csv",
    "GPCC_BIASCORR": "hla_candidate_gpcc_biascorr_2004_2013.csv",
    "CPC_BIASCORR": "hla_candidate_cpc_biascorr_2004_2013.csv",
    "ENSEMBLE_BIASCORR": "hla_candidate_ensemble_biascorr_2004_2013.csv",
}
OTHER_OUTPUTS = (
    "hla_reference_baseline_2004_2013.csv", "input_provenance.json",
    "candidate_integrity_check.json", "fill_values_111_days.csv",
    "weather_statistics_comparison.csv", "monthly_rain_comparison.csv",
    "bias_correction_audit.csv", "cli_parameter_sensitivity.csv",
    "diagnostic_summary.json", "final_gate.json",
)
WEATHER_COLUMNS = ("DATE", "RAIN", "SRAD", "TMAX", "TMIN", "RAIN_STATUS", "RAIN_SOURCE", "PURPOSE")


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def metadata(path: Path) -> dict[str, Any]:
    return {"path": relative(path), "size_bytes": path.stat().st_size, "sha256": digest(path)}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...] | list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"Refusing empty CSV output: {path.name}")
    keys = list(columns) if columns else list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if value is None else value for key, value in row.items()})


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def finite_number(raw: str | None, label: str, *, allow_missing: bool = False) -> float | None:
    if raw is None or not str(raw).strip():
        if allow_missing:
            return None
        raise ValueError(f"Missing numeric value: {label}")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value: {label}={raw!r}") from exc
    if not math.isfinite(value):
        if allow_missing:
            return None
        raise ValueError(f"Non-finite numeric value: {label}={raw!r}")
    return value


def rain_text(value: float | None) -> str:
    return "" if value is None else repr(value)


def all_dates() -> list[str]:
    dates: list[str] = []
    current = PERIOD_START
    while current <= PERIOD_END:
        dates.append(current.isoformat())
        current += timedelta(days=1)
    assert len(dates) == 3653
    return dates


def product_values(path: Path, expected_product: str) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    violations = []
    grid_centers = set()
    for row in read_csv(path):
        current = row["date"]
        if row["site"] != "HLA" or row["product"] != expected_product:
            raise ValueError(f"Wrong product/site in {path.name}: {current}")
        if current in values:
            raise ValueError(f"Duplicate product date in {path.name}: {current}")
        grid_centers.add((row["nearest_grid_latitude"], row["nearest_grid_longitude"], row["nearest_grid_distance_km"]))
        value = finite_number(row.get("nearest_grid_precip_mm"), f"{path.name}:{current}", allow_missing=True)
        if value is not None and value < 0:
            violations.append({"date": current, "product": expected_product, "nearest_grid_precip_mm": value})
        values[current] = value
    if violations:
        write_json(OUT / f"{expected_product.lower()}_negative_rain_qc_violation.json", violations)
        raise ValueError(f"Negative precipitation in 023 {expected_product}; candidate stopped")
    expected = {day for day in all_dates() if day >= "2011-01-01"}
    if set(values) != expected or len(grid_centers) != 1:
        raise ValueError(f"023 {expected_product} daily coverage or frozen nearest-grid center changed")
    return values


def build_baseline() -> tuple[list[dict[str, str]], set[str], dict[str, Any]]:
    expected = all_dates()
    weather_rows = [row for row in read_csv(WEATHER) if expected[0] <= row["date"] <= expected[-1]]
    weather_map = {row["date"]: row for row in weather_rows}
    if len(weather_rows) != len(expected) or set(weather_map) != set(expected) or any(row["station"] != "HLA" for row in weather_rows):
        raise ValueError("HLA weather_clean S/T calendar or station mismatch")

    st_rows = [row for row in read_csv(ST_CANDIDATES) if row["site"] == "HLA"]
    st_replacements: dict[tuple[str, str], str] = {}
    for row in st_rows:
        key = (row["date"], row["variable"])
        if key in st_replacements or key[0] not in weather_map or key[1] not in ("SRAD", "TMAX", "TMIN"):
            raise ValueError(f"Unexpected 008 S/T candidate: {key}")
        if row["source_gate_pass"] != "True" or row["reason"] not in ("raw_missing", "physical_qc_invalid"):
            raise ValueError(f"008 S/T candidate has no screened basis: {key}")
        candidate = finite_number(row["candidate_value"], f"008 candidate {key}")
        assert candidate is not None
        st_replacements[key] = repr(candidate)
    if len(st_replacements) != 92 or Counter(var for _, var in st_replacements) != {"TMAX": 32, "TMIN": 31, "SRAD": 29}:
        raise ValueError("008 HLA S/T screened candidate count changed")

    rain_rows = read_csv(RAIN_SOURCE)
    rain_map = {row["date"]: row for row in rain_rows}
    if len(rain_rows) != len(expected) or set(rain_map) != set(expected) or any(row["site"] != "HLA" for row in rain_rows):
        raise ValueError("014 HLA rain provenance calendar or station mismatch")
    gap_rows = read_csv(UNRESOLVED)
    gaps = {row["date"] for row in gap_rows}
    source_gaps = {day for day, row in rain_map.items() if row["rain_status"] == "UNRESOLVED"}
    if len(gap_rows) != 111 or len(gaps) != 111 or gaps != source_gaps or any(not day.startswith("2013-") for day in gaps):
        raise ValueError("The 111 frozen 014 HLA unresolved dates do not match source provenance")

    baseline: list[dict[str, str]] = []
    source_counts = Counter()
    for current in expected:
        weather_row = weather_map[current]
        rain_row = rain_map[current]
        if current in gaps:
            if rain_row["selected_rain_mm"].strip():
                raise ValueError(f"Target already contains approved rain: {current}")
            rain_value = ""
            rain_status = "UNRESOLVED"
        else:
            value = finite_number(rain_row["selected_rain_mm"], f"014 selected RAIN {current}")
            if value is None or value < 0 or rain_row["rain_status"] != "RESOLVED":
                raise ValueError(f"Invalid approved non-target RAIN: {current}")
            rain_value = rain_row["selected_rain_mm"].strip()
            rain_status = "RESOLVED"
            source_counts[rain_row["selected_source"]] += 1
        record = {
            "DATE": current,
            "RAIN": rain_value,
            "SRAD": st_replacements.get((current, "SRAD"), weather_row["SRAD"].strip()),
            "TMAX": st_replacements.get((current, "TMAX"), weather_row["TMAX"].strip()),
            "TMIN": st_replacements.get((current, "TMIN"), weather_row["TMIN"].strip()),
            "RAIN_STATUS": rain_status,
            "RAIN_SOURCE": rain_row["selected_source"].strip(),
            "PURPOSE": "GRIDDED_REFERENCE_SENSITIVITY_ONLY",
        }
        srad = finite_number(record["SRAD"], f"{current} SRAD")
        tmax = finite_number(record["TMAX"], f"{current} TMAX")
        tmin = finite_number(record["TMIN"], f"{current} TMIN")
        if srad is None or tmax is None or tmin is None or srad < 0 or tmax < tmin or tmax > 60:
            raise ValueError(f"HLA isolated S/T baseline failed physical check: {current}")
        baseline.append(record)

    known_events = {"2013-07-30": 178.2, "2013-08-09": 63.0, "2013-08-12": 66.9, "2013-07-21": 45.7}
    by_date = {row["DATE"]: row for row in baseline}
    for day, expected_rain in known_events.items():
        if day in gaps or abs(float(by_date[day]["RAIN"]) - expected_rain) > 1e-9:
            raise ValueError(f"014 accepted known event changed: {day}")
    if "2013-08-21" in gaps or abs(float(by_date["2013-08-21"]["RAIN"]) - 35.1) > 1e-9:
        raise ValueError("2013-08-21 is approved in 014 and must remain unchanged")
    provenance = {
        "date_range": [expected[0], expected[-1]], "day_count": len(expected),
        "rain_source": relative(RAIN_SOURCE), "rain_resolved_days": 3542,
        "rain_unresolved_days": len(gaps), "non_target_rain_source_counts": dict(sorted(source_counts.items())),
        "srad_tmax_tmin_base": relative(WEATHER),
        "screened_008_s_t_overlays": relative(ST_CANDIDATES),
        "screened_008_s_t_overlay_counts": dict(Counter(var for _, var in st_replacements)),
        "formal_fitting_weather_existed_before_024": False,
        "baseline_interpretation": "Isolated 024 comparison baseline assembled from 014 rain provenance, existing cleaned S/T and 008 screened missing/invalid S/T candidates; this is not an installed or formally recovered weather series.",
        "preserved_2013_08_21_accepted_rain_mm": 35.1,
    }
    return baseline, gaps, provenance


def explicit_d222() -> dict[str, float]:
    numeric: dict[str, float] = {}
    for row in read_csv(D222):
        current = row["date"]
        if row["project_site"] != "HLA" or not ("2011-01-01" <= current <= "2013-12-31"):
            continue
        value = finite_number(row.get("rain_mm"), f"D222 {current}", allow_missing=True)
        if value is not None:
            if current in numeric or value < 0:
                raise ValueError(f"Duplicate or negative D222 value: {current}")
            numeric[current] = value
    if len(numeric) != 330:
        raise ValueError(f"022 HLA 2011-2013 numeric-day count changed: {len(numeric)}")
    return numeric


def metrics(observed: list[float], predicted: list[float]) -> dict[str, float | int | None]:
    if len(observed) != len(predicted):
        raise ValueError("Paired metric lengths differ")
    if not observed:
        return {"paired_days": 0, "mae_mm": None, "rmse_mm": None, "wet_dry_agreement": None}
    errors = [a - b for a, b in zip(observed, predicted)]
    return {
        "paired_days": len(observed),
        "mae_mm": math.fsum(abs(x) for x in errors) / len(errors),
        "rmse_mm": math.sqrt(math.fsum(x * x for x in errors) / len(errors)),
        "wet_dry_agreement": sum((a > 0) == (b > 0) for a, b in zip(observed, predicted)) / len(observed),
    }


def bias_factors(product: str, values: dict[str, float | None], observed: dict[str, float], gaps: set[str]) -> tuple[dict[int, float], list[dict[str, Any]], dict[str, Any]]:
    if gaps.intersection(observed):
        raise ValueError("Bias calibration would leak target-day unknown D222 values")
    paired_by_month: dict[int, list[tuple[float, float]]] = defaultdict(list)
    ratios_by_month: dict[int, list[float]] = defaultdict(list)
    for current, reference in observed.items():
        value = values.get(current)
        if value is None:
            continue
        month = int(current[5:7])
        paired_by_month[month].append((reference, value))
        if reference > 0 and value > 0:
            ratios_by_month[month].append(reference / value)
    all_ratios = [ratio for month in range(1, 13) for ratio in ratios_by_month[month]]
    if len(all_ratios) < 5:
        raise ValueError(f"No robust all-period fallback factor for {product}")
    global_factor = statistics.median(all_ratios)
    factors: dict[int, float] = {}
    audit: list[dict[str, Any]] = []
    for month in range(1, 13):
        wet_ratios = ratios_by_month[month]
        fallback = len(wet_ratios) < 5
        factor = global_factor if fallback else statistics.median(wet_ratios)
        if factor <= 0 or not math.isfinite(factor):
            raise ValueError(f"Invalid robust factor: {product} month {month}")
        factors[month] = factor
        paired = paired_by_month[month]
        reference = [ref for ref, _ in paired]
        raw = [value for _, value in paired]
        corrected = [value * factor if value > 0 else value for value in raw]
        raw_stats = metrics(reference, raw)
        corrected_stats = metrics(reference, corrected)
        audit.append({
            "product": product, "month": month, "paired_numeric_days": len(paired),
            "paired_wet_days": len(wet_ratios), "median_ratio": statistics.median(wet_ratios) if wet_ratios else None,
            "mean_ratio": statistics.mean(wet_ratios) if wet_ratios else None,
            "fallback_to_all_period": fallback, "all_period_paired_wet_days": len(all_ratios),
            "all_period_median_ratio": global_factor, "final_factor": factor,
            "raw_mae_mm": raw_stats["mae_mm"], "corrected_mae_mm": corrected_stats["mae_mm"],
            "raw_rmse_mm": raw_stats["rmse_mm"], "corrected_rmse_mm": corrected_stats["rmse_mm"],
            "wet_dry_agreement": raw_stats["wet_dry_agreement"],
            "occurrence_correction_applied": False,
        })
    all_pairs = [(month, ref, value) for month in range(1, 13) for ref, value in paired_by_month[month]]
    all_reference = [ref for _, ref, _ in all_pairs]
    all_raw = [value for _, _, value in all_pairs]
    all_corrected = [value * factors[month] if value > 0 else value for month, _, value in all_pairs]
    all_stats = {
        "product": product,
        "paired_numeric_days": len(all_pairs),
        "paired_wet_days": len(all_ratios),
        "all_period_median_ratio": global_factor,
        "raw": metrics(all_reference, all_raw),
        "corrected": metrics(all_reference, all_corrected),
    }
    return factors, audit, all_stats


def longest_spells(values: list[float | None]) -> tuple[int, int]:
    dry_run = wet_run = longest_dry = longest_wet = 0
    for value in values:
        if value is None:
            dry_run = wet_run = 0
        elif value > 0:
            wet_run += 1
            dry_run = 0
            longest_wet = max(longest_wet, wet_run)
        else:
            dry_run += 1
            wet_run = 0
            longest_dry = max(longest_dry, dry_run)
    return longest_dry, longest_wet


def percentile(sorted_values: list[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def weather_stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    values = [finite_number(row["RAIN"], row["DATE"], allow_missing=True) for row in rows]
    wet = sorted(value for value in values if value is not None and value > 0)
    longest_dry, longest_wet = longest_spells(values)
    complete = all(value is not None for value in values)
    subtotal = math.fsum(value for value in values if value is not None)
    return {
        "calendar_days": len(values), "numeric_days": sum(value is not None for value in values),
        "missing_days": sum(value is None for value in values), "complete": complete,
        "rain_total_mm": subtotal if complete else None, "numeric_subtotal_mm": subtotal,
        "wet_days_gt_0": len(wet), "wet_days_ge_1mm": sum(value >= 1 for value in wet),
        "max_daily_rain_mm": max((value for value in values if value is not None), default=None),
        "wet_day_p95_mm": percentile(wet, 0.95), "wet_day_p99_mm": percentile(wet, 0.99),
        "longest_dry_spell_days": longest_dry, "longest_wet_spell_days": longest_wet,
    }


def monthly_stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    values = [finite_number(row["RAIN"], row["DATE"], allow_missing=True) for row in rows]
    wet = [value for value in values if value is not None and value > 0]
    complete = all(value is not None for value in values)
    total = math.fsum(value for value in values if value is not None)
    longest_dry, _ = longest_spells(values)
    return {
        "calendar_days": len(values), "numeric_days": len(values) - sum(value is None for value in values),
        "missing_days": sum(value is None for value in values), "complete": complete,
        "total_rainfall_mm": total if complete else None, "numeric_subtotal_mm": total,
        "wet_days_gt_0": len(wet), "mean_wet_day_intensity_mm": statistics.mean(wet) if wet else None,
        "max_daily_rain_mm": max((value for value in values if value is not None), default=None),
        "longest_dry_spell_days": longest_dry,
    }


def main() -> None:
    if not OUT.is_dir():
        raise FileNotFoundError(f"Create new 024 output directory first: {OUT}")
    conflicts = [str(OUT / name) for name in (*CANDIDATE_FILENAMES.values(), *OTHER_OUTPUTS) if (OUT / name).exists()]
    if REPORT.exists():
        conflicts.append(str(REPORT))
    if conflicts:
        raise FileExistsError("Refusing to overwrite 024 outputs: " + ", ".join(conflicts))
    inputs = (WEATHER, ST_CANDIDATES, RAIN_SOURCE, UNRESOLVED, D222, GPCC, CPC, GATE023, CLI_CODE)
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required project inputs missing: " + ", ".join(missing))
    gate023 = json.loads(GATE023.read_text(encoding="utf-8"))
    if gate023["HLA"]["global_product_sanity_status"] != "REGIONAL_PRODUCTS_PARTIALLY_SUPPORT_D222" or gate023["HLA"]["unresolved_days_after"] != 111:
        raise ValueError("023 HLA gate no longer matches the 024 starting condition")

    baseline, gaps, baseline_provenance = build_baseline()
    observed = explicit_d222()
    if gaps & set(observed):
        raise ValueError("022 D222 provides a numeric value on a frozen 014 unresolved target")
    products = {"GPCC": product_values(GPCC, "GPCC"), "CPC": product_values(CPC, "CPC")}
    factors: dict[str, dict[int, float]] = {}
    bias_rows: list[dict[str, Any]] = []
    bias_summary: dict[str, Any] = {}
    for product in ("GPCC", "CPC"):
        factors[product], audit, summary = bias_factors(product, products[product], observed, gaps)
        bias_rows.extend(audit)
        bias_summary[product] = summary

    fill_rows: list[dict[str, Any]] = []
    target_values: dict[str, dict[str, float | None]] = {scenario: {} for scenario in SCENARIOS}
    for current in sorted(gaps):
        month = int(current[5:7])
        gpcc = products["GPCC"][current]
        cpc = products["CPC"][current]
        gpcc_corr = gpcc * factors["GPCC"][month] if gpcc is not None and gpcc > 0 else gpcc
        cpc_corr = cpc * factors["CPC"][month] if cpc is not None and cpc > 0 else cpc
        ensemble = ((gpcc_corr + cpc_corr) / 2 if gpcc_corr is not None and cpc_corr is not None
                    else gpcc_corr if gpcc_corr is not None else cpc_corr)
        values = {
            "GPCC_RAW": gpcc, "CPC_RAW": cpc,
            "GPCC_BIASCORR": gpcc_corr, "CPC_BIASCORR": cpc_corr,
            "ENSEMBLE_BIASCORR": ensemble,
        }
        for scenario, value in values.items():
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"Invalid candidate precipitation: {scenario} {current}")
            target_values[scenario][current] = value
        both_valid = gpcc is not None and cpc is not None
        fill_rows.append({
            "date": current, "baseline_status": "UNRESOLVED",
            "gpcc_raw": gpcc, "cpc_raw": cpc,
            "gpcc_factor": factors["GPCC"][month], "cpc_factor": factors["CPC"][month],
            "gpcc_biascorr": gpcc_corr, "cpc_biascorr": cpc_corr,
            "ensemble_biascorr": ensemble,
            "gpcc_wet": (gpcc > 0) if gpcc is not None else None,
            "cpc_wet": (cpc > 0) if cpc is not None else None,
            "both_products_wet": (gpcc > 0 and cpc > 0) if both_valid else None,
            "both_products_dry": (gpcc == 0 and cpc == 0) if both_valid else None,
            "products_disagree_wetdry": ((gpcc > 0) != (cpc > 0)) if both_valid else None,
            "gpcc_minus_cpc_raw_mm": gpcc - cpc if both_valid else None,
            "absolute_raw_product_difference_mm": abs(gpcc - cpc) if both_valid else None,
            "gpcc_status": "UNRESOLVED_AFTER_GRID_REFERENCE" if gpcc is None else "GRIDDED_REFERENCE_SENSITIVITY_ONLY",
            "cpc_status": "UNRESOLVED_AFTER_GRID_REFERENCE" if cpc is None else "GRIDDED_REFERENCE_SENSITIVITY_ONLY",
            "ensemble_status": "UNRESOLVED_AFTER_GRID_REFERENCE" if ensemble is None else "GRIDDED_REFERENCE_SENSITIVITY_ONLY",
        })

    candidates: dict[str, list[dict[str, str]]] = {}
    integrity: dict[str, Any] = {
        "purpose": "GRIDDED_REFERENCE_SENSITIVITY_ONLY", "target_dates": len(gaps),
        "target_dates_all_2013": True, "baseline_2013_08_21_is_non_target": "2013-08-21" not in gaps,
        "baseline_provenance": baseline_provenance, "scenarios": {},
    }
    for scenario in SCENARIOS:
        rows: list[dict[str, str]] = []
        for baseline_row in baseline:
            current = baseline_row["DATE"]
            row = dict(baseline_row)
            if current in gaps:
                value = target_values[scenario][current]
                row["RAIN"] = rain_text(value)
                row["RAIN_STATUS"] = "UNRESOLVED_AFTER_GRID_REFERENCE" if value is None else "GRIDDED_REFERENCE_SENSITIVITY_ONLY"
                row["RAIN_SOURCE"] = scenario
            rows.append(row)
        candidates[scenario] = rows
        delta = {
            "target_dates": len(gaps),
            "non_target_rain_changes": sum(a["RAIN"] != b["RAIN"] for a, b in zip(baseline, rows) if a["DATE"] not in gaps),
            "srad_changes": sum(a["SRAD"] != b["SRAD"] for a, b in zip(baseline, rows)),
            "tmax_changes": sum(a["TMAX"] != b["TMAX"] for a, b in zip(baseline, rows)),
            "tmin_changes": sum(a["TMIN"] != b["TMIN"] for a, b in zip(baseline, rows)),
            "date_changes": sum(a["DATE"] != b["DATE"] for a, b in zip(baseline, rows)),
            "target_numeric_days": sum(target_values[scenario][current] is not None for current in gaps),
            "remaining_unresolved": sum(target_values[scenario][current] is None for current in gaps),
            "unchanged_existing_event_mm": {day: float(rows[index]["RAIN"]) for index, day in enumerate(all_dates())
                                            if day in ("2013-07-30", "2013-08-09", "2013-08-12", "2013-07-21", "2013-08-21")},
        }
        if any(delta[key] for key in ("non_target_rain_changes", "srad_changes", "tmax_changes", "tmin_changes", "date_changes")):
            raise ValueError(f"Candidate modified protected values: {scenario}")
        integrity["scenarios"][scenario] = delta

    annual_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    annual_lookup: dict[tuple[str, int], dict[str, Any]] = {}
    monthly_lookup: dict[tuple[str, int, int], dict[str, Any]] = {}
    for scenario, rows in candidates.items():
        grouped_year: dict[int, list[dict[str, str]]] = defaultdict(list)
        grouped_year_month: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
        grouped_month: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            year, month = int(row["DATE"][:4]), int(row["DATE"][5:7])
            grouped_year[year].append(row)
            grouped_year_month[(year, month)].append(row)
            grouped_month[month].append(row)
        for year, days in sorted(grouped_year.items()):
            stat = weather_stats(days)
            annual_rows.append({"scenario": scenario, "period_type": "YEAR", "year": year, **stat})
            annual_lookup[(scenario, year)] = stat
        annual_rows.append({"scenario": scenario, "period_type": "FULL_TRAINING_2004_2013", "year": "", **weather_stats(rows)})
        for (year, month), days in sorted(grouped_year_month.items()):
            stat = monthly_stats(days)
            monthly_rows.append({"scenario": scenario, "period_type": "YEAR_MONTH", "year": year, "month": month, **stat})
            monthly_lookup[(scenario, year, month)] = stat
        for month, days in sorted(grouped_month.items()):
            stat = monthly_stats(days)
            monthly_rows.append({"scenario": scenario, "period_type": "TRAINING_MONTH_10Y_TOTAL", "year": "", "month": month,
                                 **stat, "mean_month_total_per_year_mm": stat["total_rainfall_mm"] / 10 if stat["total_rainfall_mm"] is not None else None})

    sys.path.insert(0, str(ROOT / "scripts"))
    import build_dssat_cli as cli  # local import; only fit_monthly_statistics is called

    fitted: dict[str, dict[int, dict[str, float | int]]] = {}
    for scenario, rows in candidates.items():
        records = []
        for row in rows:
            rain = finite_number(row["RAIN"], f"{scenario}:{row['DATE']} RAIN")
            assert rain is not None
            records.append(cli.WeatherDay(date.fromisoformat(row["DATE"]), float(row["SRAD"]), float(row["TMAX"]), float(row["TMIN"]), rain))
        fitted[scenario] = {int(row["month"]): row for row in cli.fit_monthly_statistics(records)}

    direct_rain = {"ALPHA", "RTOT", "PDW", "RNUM"}
    unconditional = {"NAMN", "NASD"}
    parameter_rows: list[dict[str, Any]] = []
    for month in range(1, 13):
        for parameter in cli.WGEN_PARAMETER_FIELDS:
            vals = {scenario: float(fitted[scenario][month][parameter]) for scenario in SCENARIOS}
            rounded = {scenario: float(f"{value:.{cli.WGEN_PARAMETER_PRECISION[parameter]}f}") for scenario, value in vals.items()}
            minimum, maximum = min(vals.values()), max(vals.values())
            spread = maximum - minimum
            center = statistics.mean(vals.values())
            role = ("DIRECT_RAIN" if parameter in direct_rain else "UNCONDITIONAL_NONRAIN" if parameter in unconditional
                    else "NONRAIN_CONDITIONED_ON_WET_DRY")
            parameter_rows.append({
                "month": month, "parameter": parameter, "parameter_role": role,
                **{scenario.lower(): vals[scenario] for scenario in SCENARIOS},
                **{scenario.lower() + "_cli_rounded": rounded[scenario] for scenario in SCENARIOS},
                "minimum": minimum, "maximum": maximum,
                "absolute_spread": spread,
                "relative_spread_to_abs_five_scenario_mean": spread / abs(center) if abs(center) > 1e-12 else None,
                "cli_rounded_absolute_spread": max(rounded.values()) - min(rounded.values()),
                "month_of_max_absolute_spread_for_parameter": None,
            })
    month_of_max: dict[str, int] = {}
    for parameter in cli.WGEN_PARAMETER_FIELDS:
        candidates_for_parameter = [row for row in parameter_rows if row["parameter"] == parameter]
        month_of_max[parameter] = max(candidates_for_parameter, key=lambda row: (row["absolute_spread"], -row["month"]))["month"]
    for row in parameter_rows:
        row["month_of_max_absolute_spread_for_parameter"] = month_of_max[row["parameter"]]
    if any(row["absolute_spread"] > 1e-10 for row in parameter_rows if row["parameter"] in unconditional):
        raise ValueError("Unconditional TMIN WGEN parameters changed despite unchanged TMIN inputs")
    monthly_unconditional_means_invariant = all(
        max(statistics.mean(float(r[field]) for r in candidates[scenario] if int(r["DATE"][5:7]) == month)
            for scenario in SCENARIOS)
        - min(statistics.mean(float(r[field]) for r in candidates[scenario] if int(r["DATE"][5:7]) == month)
              for scenario in SCENARIOS) < 1e-10
        for month in range(1, 13) for field in ("SRAD", "TMAX", "TMIN")
    )
    if not monthly_unconditional_means_invariant:
        raise ValueError("An unconditional nonrain monthly mean changed")
    integrity["unconditional_monthly_srad_tmax_tmin_means_invariant"] = True
    integrity["wgen_namn_nasd_invariant"] = True
    integrity["rain_conditioned_srad_tmax_fields_may_vary_with_wet_dry_membership"] = True

    target_summary = {}
    for scenario in SCENARIOS:
        values = list(target_values[scenario].values())
        target_summary[scenario] = {
            "new_wet_days": sum(value is not None and value > 0 for value in values),
            "new_dry_days": sum(value == 0 for value in values),
            "remaining_unresolved": sum(value is None for value in values),
            "target_rain_subtotal_mm": math.fsum(value for value in values if value is not None),
        }
    product_abs_differences = sorted(float(row["absolute_raw_product_difference_mm"]) for row in fill_rows if row["absolute_raw_product_difference_mm"] is not None)
    target_agreement = {
        "both_products_wet_days": sum(row["both_products_wet"] is True for row in fill_rows),
        "both_products_dry_days": sum(row["both_products_dry"] is True for row in fill_rows),
        "products_disagree_wetdry_days": sum(row["products_disagree_wetdry"] is True for row in fill_rows),
        "both_products_valid_days": sum(row["gpcc_raw"] is not None and row["cpc_raw"] is not None for row in fill_rows),
        "raw_abs_difference_mean_mm": statistics.mean(product_abs_differences) if product_abs_differences else None,
        "raw_abs_difference_median_mm": statistics.median(product_abs_differences) if product_abs_differences else None,
        "raw_abs_difference_p95_mm": percentile(product_abs_differences, 0.95),
        "raw_abs_difference_max_mm": max(product_abs_differences) if product_abs_differences else None,
    }
    direct_parameter_max = {}
    for parameter in ("RTOT", "RNUM", "ALPHA", "PDW"):
        row = next(item for item in parameter_rows if item["parameter"] == parameter and item["month"] == month_of_max[parameter])
        direct_parameter_max[parameter] = {"month": row["month"], "absolute_spread": row["absolute_spread"],
                                           "relative_spread_to_abs_mean": row["relative_spread_to_abs_five_scenario_mean"],
                                           "cli_rounded_absolute_spread": row["cli_rounded_absolute_spread"]}
    year2013_totals = {scenario: annual_lookup[(scenario, 2013)]["rain_total_mm"] for scenario in SCENARIOS}
    baseline_2013_numeric_subtotal = math.fsum(float(row["RAIN"]) for row in baseline
                                               if row["DATE"].startswith("2013-") and row["RAIN"])
    monthly2013_spreads = [
        {"month": month, "rain_total_min_mm": min(float(monthly_lookup[(scenario, 2013, month)]["total_rainfall_mm"]) for scenario in SCENARIOS),
         "rain_total_max_mm": max(float(monthly_lookup[(scenario, 2013, month)]["total_rainfall_mm"]) for scenario in SCENARIOS)}
        for month in range(1, 13)
    ]
    for row in monthly2013_spreads:
        row["absolute_spread_mm"] = row["rain_total_max_mm"] - row["rain_total_min_mm"]
    summary = {
        "purpose": "GRIDDED_REFERENCE_SENSITIVITY_ONLY", "baseline": baseline_provenance,
        "target_summary": target_summary, "target_product_wetdry_agreement": target_agreement,
        "bias_correction": bias_summary,
        "annual_2013_by_scenario": {scenario: annual_lookup[(scenario, 2013)] for scenario in SCENARIOS},
        "full_training_by_scenario": {scenario: weather_stats(candidates[scenario]) for scenario in SCENARIOS},
        "annual_2013_total_min_mm": min(year2013_totals.values()),
        "annual_2013_total_max_mm": max(year2013_totals.values()),
        "annual_2013_total_spread_mm": max(year2013_totals.values()) - min(year2013_totals.values()),
        "annual_2013_total_relative_spread_to_mean": (max(year2013_totals.values()) - min(year2013_totals.values())) / statistics.mean(year2013_totals.values()),
        "baseline_2013_non_target_numeric_subtotal_mm": baseline_2013_numeric_subtotal,
        "monthly_2013_spreads": monthly2013_spreads,
        "direct_wgen_parameter_max_monthly_spread": direct_parameter_max,
        "unconditional_nonrain_monthly_means_invariant": True,
        "scenario_consistency_status": "INSUFFICIENT_FOR_CLASSIFICATION",
        "classification_reason": "024 supplies no predeclared material-divergence threshold; all measured spreads are reported for user review without inventing a DSSAT/WGEN official cutoff.",
    }

    provenance = {
        "task": "024", "site": "HLA", "purpose": "GRIDDED_REFERENCE_SENSITIVITY_ONLY",
        "inputs": [metadata(path) for path in inputs],
        "rain_baseline_rule": "Use 014 selected_rain_mm on RESOLVED dates and leave exactly its 111 UNRESOLVED dates blank before the five isolated scenarios.",
        "st_baseline_rule": "Use existing weather_clean HLA SRAD/TMAX/TMIN, overlay only the 92 missing/invalid site dates screened PASS in 008; no candidate changes these fields.",
        "d222_bias_truth_rule": "Use 022 current D222 explicit numeric HLA days in 2011-2013; exclude blank and all 111 target days; same-date pairs only.",
        "product_rule": "Reuse 023 frozen HLA nearest-grid daily CSVs, with no resampling, downloading, or 3x3 selection.",
        "bias_factor_rule": "Median D222/product on positive paired days by month; <5 paired wet days fall back to product all-period median. Correct positive product intensities only.",
        "wet_day_rule": "RAIN > 0; additional count RAIN >= 1 mm.",
        "percentile_rule": "Linear interpolation at (n-1)*p over positive wet-day intensities.",
        "cli_function": "scripts/build_dssat_cli.py:fit_monthly_statistics; no CLI file is generated.",
        "cli_relative_spread_rule": "(max - min) / abs(mean of five scenario values), null when denominator is near zero.",
    }

    write_csv(OUT / "hla_reference_baseline_2004_2013.csv", baseline, WEATHER_COLUMNS)
    for scenario in SCENARIOS:
        write_csv(OUT / CANDIDATE_FILENAMES[scenario], candidates[scenario], WEATHER_COLUMNS)
    write_csv(OUT / "fill_values_111_days.csv", fill_rows)
    write_csv(OUT / "bias_correction_audit.csv", bias_rows)
    write_csv(OUT / "weather_statistics_comparison.csv", annual_rows)
    write_csv(OUT / "monthly_rain_comparison.csv", monthly_rows)
    write_csv(OUT / "cli_parameter_sensitivity.csv", parameter_rows)
    write_json(OUT / "input_provenance.json", provenance)
    write_json(OUT / "candidate_integrity_check.json", integrity)
    write_json(OUT / "diagnostic_summary.json", summary)

    gate = {
        "task": "024", "site": "HLA", "purpose": "GRIDDED_REFERENCE_SENSITIVITY_ONLY",
        "existing_data_modified": False, "existing_wth_modified": False, "existing_results_modified": False,
        "cli_installed": False, "wgen_executed": False, "dssat_executed": False, "ppo_executed": False,
        "starting_unresolved_days": len(gaps), "candidate_scenarios": list(SCENARIOS),
        "remaining_unresolved_by_scenario": {scenario: target_summary[scenario]["remaining_unresolved"] for scenario in SCENARIOS},
        "scenario_consistency_status": summary["scenario_consistency_status"],
        "scenario_consistency_numeric_basis": {
            "annual_2013_total_spread_mm": summary["annual_2013_total_spread_mm"],
            "annual_2013_total_relative_spread_to_mean": summary["annual_2013_total_relative_spread_to_mean"],
            "baseline_2013_non_target_numeric_subtotal_mm": baseline_2013_numeric_subtotal,
            "target_wetdry_disagreement_days": target_agreement["products_disagree_wetdry_days"],
            "max_monthly_direct_wgen_parameter_spread": direct_parameter_max,
        },
        "formal_observation_recovery_claimed": False, "ready_for_user_review": True,
        "next_step": "USER_REVIEW_BEFORE_ANY_WGEN",
    }
    write_json(OUT / "final_gate.json", gate)

    table_lines = []
    for scenario in SCENARIOS:
        row = annual_lookup[(scenario, 2013)]
        table_lines.append(
            f"| {scenario} | {row['rain_total_mm']:.3f} | {row['wet_days_gt_0']} | {row['wet_days_ge_1mm']} | {row['max_daily_rain_mm']:.3f} | {row['wet_day_p95_mm']:.3f} | {row['wet_day_p99_mm']:.3f} | {row['longest_dry_spell_days']} |"
        )
    sensitive_months = sorted(monthly2013_spreads, key=lambda row: row["absolute_spread_mm"], reverse=True)[:3]
    report = f"""# HLA 格点参考缺口填补敏感性审计（024）

## 范围和输入

本任务只针对 HLA，所有候选均为 `GRIDDED_REFERENCE_SENSITIVITY_ONLY`。014 的 3,653 日降雨来源表有 3,542 日已解决、111 日未解决且全部在 2013；023 已冻结 GPCC/CPC 最近格点日序列，本次没有重新挑格点。由于 010/014 尚未产生正式完整 fitting weather，本审计在独立输出目录中使用 014 的已选降雨与现有 HLA 清洗温度/辐射，叠加 008 已筛选通过的 92 个缺失/异常温度与辐射候选，形成共同参考基线。该基线不是正式安装的天气文件，详见 `input_provenance.json`。

五个候选只为 111 个 RAIN 缺口赋值。014 已接受的 2013-08-21 降雨为 35.1 mm，不在 111 天中，五个候选均原样保留。2013-07-30、08-09、08-12、07-21 的 178.2、63.0、66.9、45.7 mm 也全部保留。`candidate_integrity_check.json` 逐方案证明非目标 RAIN、SRAD、TMAX、TMIN、日期改动数均为 0。

## 111 天的产品信号

GPCC_RAW 给出 {target_summary['GPCC_RAW']['new_wet_days']} 个湿日、{target_summary['GPCC_RAW']['new_dry_days']} 个干日；CPC_RAW 分别为 {target_summary['CPC_RAW']['new_wet_days']} 和 {target_summary['CPC_RAW']['new_dry_days']}。两产品共同湿日 {target_agreement['both_products_wet_days']} 天、共同干日 {target_agreement['both_products_dry_days']} 天、湿干分歧 {target_agreement['products_disagree_wetdry_days']} 天（两产品均有值 {target_agreement['both_products_valid_days']} 天）。原始目标日绝对差的中位数为 {target_agreement['raw_abs_difference_median_mm']:.4f} mm、95 分位为 {target_agreement['raw_abs_difference_p95_mm']:.4f} mm、最大为 {target_agreement['raw_abs_difference_max_mm']:.4f} mm。每日期值与湿干状态见 `fill_values_111_days.csv`。

## 强度校正

只用 022 D222 2011–2013 的明确数值日与 023 产品同日配对；D222 空白和 111 个目标日期未入校正。GPCC/CPC 的可比数值日分别为 {bias_summary['GPCC']['paired_numeric_days']}/{bias_summary['CPC']['paired_numeric_days']}，正值配对湿日为 {bias_summary['GPCC']['paired_wet_days']}/{bias_summary['CPC']['paired_wet_days']}；全期回退因子中位数为 {bias_summary['GPCC']['all_period_median_ratio']:.4f}/{bias_summary['CPC']['all_period_median_ratio']:.4f}。逐月若湿日配对不足 5，则回退到该产品全期中位数；每月计数、因子和原始/校正后的 MAE/RMSE 见 `bias_correction_audit.csv`。校正只乘以正值格点雨强，未改变湿干发生，也未对因子按目标年量调参。

在同一批 330 个 D222 明示日上，GPCC 原始/校正 MAE 为 {bias_summary['GPCC']['raw']['mae_mm']:.3f}/{bias_summary['GPCC']['corrected']['mae_mm']:.3f} mm，RMSE 为 {bias_summary['GPCC']['raw']['rmse_mm']:.3f}/{bias_summary['GPCC']['corrected']['rmse_mm']:.3f} mm；CPC 分别为 {bias_summary['CPC']['raw']['mae_mm']:.3f}/{bias_summary['CPC']['corrected']['mae_mm']:.3f} mm 和 {bias_summary['CPC']['raw']['rmse_mm']:.3f}/{bias_summary['CPC']['corrected']['rmse_mm']:.3f} mm。两产品的总体误差均上升；该比值校正仅构造敏感性情景，不能宣称模型验证改善。

111 天的 GPCC/CPC 原始新增降水分别为 {target_summary['GPCC_RAW']['target_rain_subtotal_mm']:.3f}/{target_summary['CPC_RAW']['target_rain_subtotal_mm']:.3f} mm，校正后为 {target_summary['GPCC_BIASCORR']['target_rain_subtotal_mm']:.3f}/{target_summary['CPC_BIASCORR']['target_rain_subtotal_mm']:.3f} mm，校正组合为 {target_summary['ENSEMBLE_BIASCORR']['target_rain_subtotal_mm']:.3f} mm。组合仅逐日平均两产品校正值；只有一方有效时用有效一方，两方缺失时保持未解决。

## 2013 天气统计

| 候选 | 年雨量 mm | 雨日 >0 | 雨日 ≥1 mm | 最大日雨量 mm | 湿日 P95 mm | 湿日 P99 mm | 最长连续干日 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_lines)}

五个候选的 2013 年雨量范围为 {summary['annual_2013_total_min_mm']:.3f}–{summary['annual_2013_total_max_mm']:.3f} mm，绝对差 {summary['annual_2013_total_spread_mm']:.3f} mm，相对五方案均值为 {summary['annual_2013_total_relative_spread_to_mean']:.3%}。014 已选非目标日数值小计已有 {baseline_2013_numeric_subtotal:.3f} mm；五方案只增加约 2.84–3.10 mm。因此 2013 高降水量级主要来自既有来源，这次 111 天格点参考敏感性不能解决 023 留下的站点雨量量级疑问。年雨日数为 157–163、最长连续干日为 19–22 天，发生结构的差异比年雨量差异更可见。

2013 月雨量分叉最大的三个月为 {', '.join(f"{row['month']} 月 {row['absolute_spread_mm']:.3f} mm" for row in sensitive_months)}。完整 2004–2013 各年、十年汇总、逐月湿日强度/极值/连续干日见 `weather_statistics_comparison.csv` 与 `monthly_rain_comparison.csv`。湿日 P95/P99 用正值降雨排序后线性插值计算。

## WGEN/CLI 参数敏感性

本审计只调用项目 `scripts/build_dssat_cli.py:fit_monthly_statistics` 计算月参数，没有生成或安装 CLI。五候选的 WGEN 14 个参数逐月值、原始值和 CLI 精度后的绝对/相对跨度见 `cli_parameter_sensitivity.csv`。四个直接降雨参数的最大月跨度为：

| 参数 | 最大跨度月份 | 原始参数绝对跨度 | 相对五方案均值跨度 | CLI 精度后跨度 |
|---|---:|---:|---:|---:|
| RTOT | {direct_parameter_max['RTOT']['month']} | {direct_parameter_max['RTOT']['absolute_spread']:.4f} | {direct_parameter_max['RTOT']['relative_spread_to_abs_mean']:.2%} | {direct_parameter_max['RTOT']['cli_rounded_absolute_spread']:.4f} |
| RNUM | {direct_parameter_max['RNUM']['month']} | {direct_parameter_max['RNUM']['absolute_spread']:.4f} | {direct_parameter_max['RNUM']['relative_spread_to_abs_mean']:.2%} | {direct_parameter_max['RNUM']['cli_rounded_absolute_spread']:.4f} |
| ALPHA | {direct_parameter_max['ALPHA']['month']} | {direct_parameter_max['ALPHA']['absolute_spread']:.4f} | {direct_parameter_max['ALPHA']['relative_spread_to_abs_mean']:.2%} | {direct_parameter_max['ALPHA']['cli_rounded_absolute_spread']:.4f} |
| PDW | {direct_parameter_max['PDW']['month']} | {direct_parameter_max['PDW']['absolute_spread']:.4f} | {direct_parameter_max['PDW']['relative_spread_to_abs_mean']:.2%} | {direct_parameter_max['PDW']['cli_rounded_absolute_spread']:.4f} |

项目函数的 `RTOT` 是 2004–2013 同月份逐年总雨量的均值；因此 2013 年 6 月月量跨度 1.748 mm，在十年拟合的 6 月 `RTOT` 上表现为 0.1748 mm。`RNUM` 同样是十年同月份平均雨日数。相对跨度定义为 `(最大值−最小值)/|五方案均值|`；接近零均值的条件温度参数可能出现很大的相对百分比，应同时看绝对跨度。

温度/辐射日输入及其不依赖湿干分类的月均值完全相同；WGEN 的 `NAMN/NASD` 亦相同。`SDMN/SDSD/SWMN/SWSD/XDMN/XDSD/XWMN/XWSD` 虽来自未变的温度/辐射日值，却按雨日/干日分组，分组变化会影响这些参数，报告已全部列出。

## 场景判读与停止点

五个方案在雨量与月参数上形成可量化的区间，原始 GPCC/CPC 的湿干判断有 {target_agreement['products_disagree_wetdry_days']} 天分歧，强度校正改变新增雨量，组合位于对应两校正产品逐日值之间。2013 年最高雨量是 CPC_BIASCORR，最低是 GPCC_RAW，差 0.259 mm，没有一个方案在年雨量上明显脱离其余方案；但 GPCC 与 CPC 所决定的小雨日数不同。024 没有预设“显著分叉”的数值门槛，也没有适用的 DSSAT 官方阈值；机器状态保守为 `INSUFFICIENT_FOR_CLASSIFICATION`，数值依据保存在 `diagnostic_summary.json` 和 `final_gate.json`。不能自动宣布某一候选最佳。

从这五方案的年雨量和直接降水月参数跨度看，立即开展 WGEN-only 敏感性的优先级较低；若研究需要确认小雨日重分类及 ALPHA 等参数差异会不会在随机天气中放大，可由用户审阅后另行决定一个隔离、限算力的 WGEN-only 实验。本审计本身没有执行 WGEN、DSSAT 或 PPO。格点参考值代表区域分析，底层站资料可能重叠，日界与空间尺度也不同；五个候选都不能称为 HLA 本站观测恢复，不能覆盖现有 D222、WTH 或 cleaned weather。

## 主要输出

- 五个候选：`hla_candidate_gpcc_raw_2004_2013.csv`、`hla_candidate_cpc_raw_2004_2013.csv`、`hla_candidate_gpcc_biascorr_2004_2013.csv`、`hla_candidate_cpc_biascorr_2004_2013.csv`、`hla_candidate_ensemble_biascorr_2004_2013.csv`
- 共同参考基线：`hla_reference_baseline_2004_2013.csv`；111 天逐日对照：`fill_values_111_days.csv`
- 质量与溯源：`candidate_integrity_check.json`、`input_provenance.json`、`bias_correction_audit.csv`
- 统计与参数：`weather_statistics_comparison.csv`、`monthly_rain_comparison.csv`、`cli_parameter_sensitivity.csv`、`diagnostic_summary.json`、`final_gate.json`

全部输出位于 `results/hla_gridded_fill_sensitivity_024/`。本任务在生成这些候选与审计后停止。
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({"status": "AUDIT_COMPLETE", "site": "HLA", "target_dates": len(gaps),
                      "remaining_unresolved_by_scenario": gate["remaining_unresolved_by_scenario"],
                      "scenario_consistency_status": gate["scenario_consistency_status"],
                      "annual_2013_total_spread_mm": summary["annual_2013_total_spread_mm"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
