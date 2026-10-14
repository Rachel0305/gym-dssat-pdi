#!/usr/bin/env python3
"""Read-only parameter and SRAD hard-bound audit for YC task 004_21."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/yc_random_weather_ppo/004_21_yc_wgen_srad_generation_mechanism_and_correction_design"
FITTING = ROOT / "results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv"
CLI = ROOT / "results/yc_wgen_cli_pilot/003_06_05_02/final/CNYC.CLI"
EXPECTED_FITTING_SHA256 = "4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34"
EXPECTED_CLI_SHA256 = "65CF134600A5881706A5D435E1A09B276ED92A21FA5ABE2E18AAF63AF1E3A929"
LATITUDE = 36.83
WGEN_FIELDS = ("SDMN", "SDSD", "SWMN", "SWSD")
THRESHOLDS = (2.0, 2.4682389600000003, 3.0, 4.0, 5.0)
SEASON_MONTHS = range(6, 11)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def rada(doy: int, latitude: float) -> float:
    """DSSAT v4.8.0.24 Weather/WGEN.for RADA implementation."""
    pi = 3.14159
    rad = pi / 180.0
    t = 2.0 * pi * (doy + 10.0) / 365.0
    dec = -23.45 * math.cos(t)
    ssin = math.sin(rad * dec) * math.sin(rad * latitude)
    ccos = math.cos(rad * dec) * math.cos(rad * latitude)
    soc = max(-1.0, min(1.0, ssin / ccos))
    day_length = 12.0 + 24.0 * math.asin(soc) / pi
    dsinb = 3600.0 * (day_length * ssin + 24.0 / pi * ccos * math.sqrt(1.0 - soc * soc))
    solar_constant = 1368.0 * (1.0 + 0.033 * math.cos(2.0 * pi * doy / 365.0))
    return solar_constant * dsinb / 1.0e6


def mean_sd(values: list[float]) -> tuple[float, float]:
    mean = math.fsum(values) / len(values)
    sd = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1))
    return mean, sd


def read_cli_parameters() -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    in_wgen = False
    for line in CLI.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("*WGEN PARAMETERS"):
            in_wgen = True
            continue
        if in_wgen and line.startswith("*"):
            break
        if not in_wgen or not line.strip() or line.lstrip().startswith("@"):
            continue
        cells = line.split()
        if len(cells) != 15:
            raise ValueError(f"Unexpected WGEN row: {line!r}")
        month = int(cells[0])
        names = ("SDMN", "SDSD", "SWMN", "SWSD", "XDMN", "XDSD", "XWMN", "XWSD",
                 "NAMN", "NASD", "ALPHA", "RTOT", "PDW", "RNUM")
        rows[month] = {name: float(value) for name, value in zip(names, cells[1:])}
    if sorted(rows) != list(range(1, 13)):
        raise ValueError(f"Expected CLI WGEN months 1-12; found {sorted(rows)}")
    return rows


def read_fitting() -> dict[int, list[dict[str, float | date]]]:
    by_month: dict[int, list[dict[str, float | date]]] = defaultdict(list)
    with FITTING.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            day = date.fromisoformat(row["DATE"])
            by_month[day.month].append({
                "date": day,
                "SRAD": float(row["SRAD"]),
                "RAIN": float(row["RAIN"]),
            })
    if sum(map(len, by_month.values())) != 3653:
        raise ValueError("Fitting input does not contain the frozen 3653 daily rows")
    return by_month


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parameter_audit(fitting: dict[int, list[dict[str, float | date]]], cli: dict[int, dict[str, float]]) -> list[dict[str, object]]:
    rows = []
    for month in range(1, 13):
        month_days = fitting[month]
        dry = [float(row["SRAD"]) for row in month_days if float(row["RAIN"]) <= 0.0]
        wet = [float(row["SRAD"]) for row in month_days if float(row["RAIN"]) > 0.0]
        all_srad = [float(row["SRAD"]) for row in month_days]
        dry_mean, dry_sd = mean_sd(dry)
        wet_mean, wet_sd = mean_sd(wet)
        all_mean = math.fsum(all_srad) / len(all_srad)
        recomputed = {"SDMN": dry_mean, "SDSD": dry_sd, "SWMN": wet_mean, "SWSD": wet_sd}
        diffs = {name: cli[month][name] - recomputed[name] for name in WGEN_FIELDS}
        samn_cli = cli_monthly_mean(month)
        rows.append({
            "month": month,
            "dry_days": len(dry),
            "wet_days": len(wet),
            "dry_mean": f"{dry_mean:.10f}",
            "dry_sd": f"{dry_sd:.10f}",
            "wet_mean": f"{wet_mean:.10f}",
            "wet_sd": f"{wet_sd:.10f}",
            "all_mean": f"{all_mean:.10f}",
            "cli_SDMN": cli[month]["SDMN"],
            "recomputed_SDMN": f"{dry_mean:.10f}",
            "difference_SDMN_cli_minus_recomputed": f"{diffs['SDMN']:.10f}",
            "cli_SDSD": cli[month]["SDSD"],
            "recomputed_SDSD": f"{dry_sd:.10f}",
            "difference_SDSD_cli_minus_recomputed": f"{diffs['SDSD']:.10f}",
            "cli_SWMN": cli[month]["SWMN"],
            "recomputed_SWMN": f"{wet_mean:.10f}",
            "difference_SWMN_cli_minus_recomputed": f"{diffs['SWMN']:.10f}",
            "cli_SWSD": cli[month]["SWSD"],
            "recomputed_SWSD": f"{wet_sd:.10f}",
            "difference_SWSD_cli_minus_recomputed": f"{diffs['SWSD']:.10f}",
            "cli_SAMN": samn_cli,
            "recomputed_SAMN": f"{all_mean:.10f}",
            "difference_SAMN_cli_minus_recomputed": f"{samn_cli - all_mean:.10f}",
            "serialization_status": "MATCH_WITH_1_DECIMAL_ROUNDING" if all(abs(value) <= 0.0500001 for value in diffs.values()) and abs(samn_cli - all_mean) <= 0.0500001 else "MISMATCH",
        })
    return rows


def cli_monthly_mean(month: int) -> float:
    in_monthly = False
    for line in CLI.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("*MONTHLY AVERAGES"):
            in_monthly = True
            continue
        if in_monthly and line.startswith("*"):
            in_monthly = False
        if in_monthly and line.strip() and not line.lstrip().startswith("@"):
            cells = line.split()
            if len(cells) == 9 and cells[0].isdigit() and int(cells[0]) == month:
                return float(cells[1])
    raise ValueError(f"SAMN not found for month {month}")


def bound_summary() -> dict[int, tuple[float, int, float, int]]:
    values: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for year in range(2004, 2024):
        day = date(year, 1, 1)
        end = date(year, 12, 31)
        while day <= end:
            if day.month in SEASON_MONTHS:
                # WGENGN: RC=0.8*RADA and SRMIN=0.2*RC.
                floor = 0.16 * rada(day.timetuple().tm_yday, LATITUDE)
                values[day.month].append((floor, day.timetuple().tm_yday))
            day += timedelta(days=1)
    return {
        month: (min(items, key=lambda item: item[0])[0], min(items, key=lambda item: item[0])[1],
                max(items, key=lambda item: item[0])[0], max(items, key=lambda item: item[0])[1])
        for month, items in values.items()
    }


def probability_rows(bounds: dict[int, tuple[float, int, float, int]]) -> list[dict[str, object]]:
    rows = []
    for month in sorted(bounds):
        lower_min, min_doy, lower_max, max_doy = bounds[month]
        for threshold in THRESHOLDS:
            if threshold <= lower_min:
                status, probability = "ZERO_BY_HARD_BOUND", "0"
                note = "Every DOY in this calendar month has SRMIN >= threshold; generated SRAD is clamped at or above SRMIN."
            else:
                status, probability = "NOT_COMPUTED_STATE_DEPENDENT", ""
                note = "The hard bound permits crossing on some DOYs; numeric probability requires the correlated AR state, wet/dry path, monthly interpolation, and clipping distribution."
            rows.append({
                "month": month,
                "threshold_SRAD_lt": f"{threshold:.9g}",
                "SRMIN_month_min_MJ_m2_d": f"{lower_min:.9f}",
                "SRMIN_min_DOY": min_doy,
                "SRMIN_month_max_MJ_m2_d": f"{lower_max:.9f}",
                "SRMIN_max_DOY": max_doy,
                "probability": probability,
                "probability_status": status,
                "basis": "DSSAT v4.8.0.24 Weather/WGEN.for; latitude 36.83; month dates across 2004-2023",
                "note": note,
            })
    return rows


def main() -> None:
    if sha256(FITTING) != EXPECTED_FITTING_SHA256:
        raise SystemExit("Frozen fitting CSV SHA256 mismatch; refusing audit")
    if sha256(CLI) != EXPECTED_CLI_SHA256:
        raise SystemExit("Frozen CNYC.CLI SHA256 mismatch; refusing audit")
    OUT.mkdir(parents=True, exist_ok=True)
    fitting = read_fitting()
    cli = read_cli_parameters()
    audit = parameter_audit(fitting, cli)
    bounds = bound_summary()
    write_csv(OUT / "srad_cli_parameter_audit.csv", audit)
    write_csv(OUT / "theoretical_low_srad_probability.csv", probability_rows(bounds))
    if any(row["serialization_status"] != "MATCH_WITH_1_DECIMAL_ROUNDING" for row in audit):
        raise SystemExit("At least one monthly frozen CLI SRAD parameter failed independent recomputation")
    summary = {
        "task_id": "004_21",
        "fitting_sha256": EXPECTED_FITTING_SHA256,
        "cli_sha256": EXPECTED_CLI_SHA256,
        "fitting_rows": 3653,
        "parameter_months_checked": 12,
        "parameters_checked": 5,
        "all_months_match_with_1_decimal_cli_rounding": True,
        "rainfall_wet_day_rule": "RAIN > 0.0 mm",
        "crop_calendar_months": list(SEASON_MONTHS),
        "latitude_deg": LATITUDE,
        "seasonal_SRMIN_range_MJ_m2_d": [
            min(item[0] for item in bounds.values()),
            max(item[2] for item in bounds.values()),
        ],
        "seasonal_min_SRMIN_month": min(bounds, key=lambda month: bounds[month][0]),
        "seasonal_min_SRMIN_DOY": min(bounds.values(), key=lambda item: item[0])[1],
        "theoretical_probabilities": "Exact zeros emitted only where source-derived SRMIN excludes threshold for every DOY in that month; other stateful probabilities intentionally not numerically estimated.",
        "no_wgen_run": True,
        "no_dssat_run": True,
        "no_weather_generated": True,
        "no_ppo_run": True,
        "old_artifacts_modified": False,
    }
    (OUT / "parameter_and_bound_audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    decision = {
        "task_id": "004_21",
        "runtime_srad_mechanism_status": "PARTIALLY_VERIFIED",
        "low_tail_loss_mechanism": "TRUNCATION_OR_BOUND",
        "recommended_correction_family": "RUNTIME_PARAMETER_CHANGE",
        "weather_only_pilot_ready": "YES",
        "pilot_precondition": "Before generation, pin the exact DSSAT executable/source revision matching the runtime package path and verify DailyWeatherCheck; use an isolated runtime copy only.",
        "parameter_audit": {
            "status": "PASS_WITH_1_DECIMAL_SERIALIZATION_ROUNDING",
            "months": 12,
            "parameters_per_month": 5,
            "fields": ["SDMN", "SDSD", "SWMN", "SWSD", "SAMN"],
            "fitting_csv_sha256": EXPECTED_FITTING_SHA256,
            "frozen_cli_sha256": EXPECTED_CLI_SHA256,
        },
        "mechanistic_findings": {
            "source_confirmed_sr_ad_floor": "SRMIN = 0.2 * RC; RC = 0.8 * RADA(DOY, LAT), hence SRMIN = 0.16 * RADA",
            "yc_crop_calendar_months": [6, 7, 8, 9, 10],
            "minimum_floor_MJ_m2_d": min(item[0] for item in bounds.values()),
            "minimum_floor_month": min(bounds, key=lambda month: bounds[month][0]),
            "minimum_floor_doy": min(bounds.values(), key=lambda item: item[0])[1],
            "probability_srad_lt_2_468_crop_season": 0,
            "explicit_cli_lower_bound_field": False,
            "rainfall_influences_srad_by": ["wet/dry conditional parameter selection", "sequential shared PRNG stream"],
            "tmax_tmin_srad_dependence": "shared truncated-normal innovations and prior-day cross-variable state matrices A/B",
        },
        "correction_design": {
            "first_pilot_factor": "change only the SRAD lower-bound rule in an isolated runtime/source copy; leave CLI, wet/dry fits, rainfall, temperatures, seed and context unchanged",
            "candidate_floor": "0.1 MJ m-2 d-1 only if exact runtime source confirms the compiled DailyWeatherCheck guard; otherwise use a separately justified nonnegative floor and label it experimental",
            "no_observed_fitting": True,
            "no_weather_generated": True,
            "no_dssat_run": True,
            "no_ppo_run": True,
        },
        "source_scope_caveat": "The installed binary exposes the v4.8.0.24+2~a4ae650 WGEN source path, while its exact matching WGEN.for and DailyWeatherCheck source/hash are not archived locally; conclusions use the official v4.8.0.24 source plus binary-string/runtime provenance and remain partial at patch-build identity.",
    }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
