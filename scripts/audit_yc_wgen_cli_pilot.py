#!/usr/bin/env python3
"""Audit YC WGEN pilot prerequisites without generating weather or running DSSAT."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual" / "YC"
OUT_DIR = ROOT / "results" / "yc_wgen_cli_pilot"
CLEANED_WEATHER_CSV = ROOT / "weather_clean" / "YCA_weather_cleaned.csv"
LEGACY_CLI = (
    ROOT
    / "DSSAT_auto_validation"
    / "run_CNYC0802_DSSAT480_IC0_null_2000_2023"
    / "pdi_smoke_test"
    / "input_used_by_pdi"
    / "CNYC.CLI"
)
TRAIN_YEARS = list(range(2004, 2014))
KNOWN_VALIDATION_YEARS = list(range(2014, 2024))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def expected_doys(year: int) -> list[int]:
    days = 366 if date(year, 12, 31).timetuple().tm_yday == 366 else 365
    return list(range(1, days + 1))


def parse_weather(path: Path, expected_year: int) -> tuple[dict, list[dict]]:
    lines = path.read_text(encoding="latin-1").splitlines()
    station_id = ""
    daily_columns: list[str] = []
    data_start = -1
    header_error = ""
    for idx, line in enumerate(lines):
        normalized = line.strip().upper()
        if normalized.startswith("@ INSI"):
            for value_line in lines[idx + 1 :]:
                if value_line.strip():
                    station_id = value_line.split()[0].upper()
                    break
        if normalized.startswith("@") and "DATE" in normalized.split():
            daily_columns = [token.upper() for token in line.lstrip()[1:].split()]
            data_start = idx + 1
            break

    required = ["DATE", "SRAD", "TMAX", "TMIN", "RAIN"]
    if not daily_columns or any(name not in daily_columns for name in required):
        header_error = "missing_required_daily_header"

    rows: list[dict] = []
    parse_errors = 0
    if data_start >= 0 and not header_error:
        positions = {name: daily_columns.index(name) for name in required}
        for line in lines[data_start:]:
            tokens = line.split()
            if not tokens or not tokens[0].isdigit():
                continue
            try:
                doy_code = int(tokens[positions["DATE"]])
                values = {
                    name: float(tokens[positions[name]])
                    for name in ("SRAD", "TMAX", "TMIN", "RAIN")
                }
                rows.append({"date_code": doy_code, **values})
            except (IndexError, ValueError):
                parse_errors += 1

    dates = [row["date_code"] for row in rows]
    doys = [value % 1000 for value in dates]
    date_years = [value // 1000 for value in dates]
    duplicates = len(dates) - len(set(dates))
    expected = [expected_year * 1000 + doy for doy in expected_doys(expected_year)]
    missing_dates = sorted(set(expected) - set(dates))
    extra_dates = sorted(set(dates) - set(expected))
    sentinel_count = sum(
        1
        for row in rows
        for name in ("SRAD", "TMAX", "TMIN", "RAIN")
        if not math.isfinite(row[name]) or row[name] <= -90.0
    )
    negative_rain = sum(row["RAIN"] < 0 for row in rows if math.isfinite(row["RAIN"]))
    negative_srad = sum(row["SRAD"] < 0 for row in rows if math.isfinite(row["SRAD"]))
    inverted_temps = sum(
        row["TMAX"] < row["TMIN"]
        for row in rows
        if math.isfinite(row["TMAX"]) and math.isfinite(row["TMIN"])
    )
    non_finite = sum(
        not math.isfinite(row[name])
        for row in rows
        for name in ("SRAD", "TMAX", "TMIN", "RAIN")
    )
    checks = {
        "required_header_present": not header_error,
        "all_records_parse": parse_errors == 0,
        "expected_record_count": len(rows) == len(expected),
        "dates_continuous_and_in_year": not missing_dates and not extra_dates and set(date_years) == {expected_year},
        "no_duplicate_dates": duplicates == 0,
        "no_missing_or_nonfinite_values": sentinel_count == 0 and non_finite == 0,
        "rain_nonnegative": negative_rain == 0,
        "srad_nonnegative": negative_srad == 0,
        "tmax_not_below_tmin": inverted_temps == 0,
        "station_id_is_CNYC": station_id == "CNYC",
        "filename_matches_year": path.name.upper() == f"CNYC{expected_year % 100:02d}01.WTH",
    }
    qc_status = "pass" if all(checks.values()) else "fail"
    annual_rain = sum(row["RAIN"] for row in rows if math.isfinite(row["RAIN"]))
    wet_days = sum(row["RAIN"] > 0.1 for row in rows if math.isfinite(row["RAIN"]))
    max_dry_spell = 0
    current_dry_spell = 0
    for row in rows:
        if row["RAIN"] <= 0.1:
            current_dry_spell += 1
            max_dry_spell = max(max_dry_spell, current_dry_spell)
        else:
            current_dry_spell = 0

    monthly: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        doy = row["date_code"] % 1000
        month = (date(expected_year, 1, 1) + timedelta(days=doy - 1)).month
        monthly[month].append(row)

    month_rows = []
    for month in sorted(monthly):
        subset = monthly[month]
        item = {"site": "YC", "year": expected_year, "month": month, "record_count": len(subset)}
        item["rain_total_mm"] = round(sum(r["RAIN"] for r in subset), 3)
        item["wet_day_count_gt_0p1mm"] = sum(r["RAIN"] > 0.1 for r in subset)
        for name in ("TMAX", "TMIN", "SRAD"):
            values = [r[name] for r in subset]
            item[f"{name.lower()}_mean"] = round(statistics.fmean(values), 4)
            item[f"{name.lower()}_sd_population"] = round(statistics.pstdev(values), 4)
        month_rows.append(item)

    result = {
        "site": "YC",
        "station_code": "YCA",
        "weather_station_id": station_id,
        "year": expected_year,
        "weather_file": path.relative_to(ROOT).as_posix(),
        "file_sha256": sha256(path),
        "file_bytes": path.stat().st_size,
        "record_count": len(rows),
        "expected_record_count": len(expected),
        "first_date_code": min(dates) if dates else None,
        "last_date_code": max(dates) if dates else None,
        "duplicate_date_count": duplicates,
        "missing_date_count": len(missing_dates),
        "extra_date_count": len(extra_dates),
        "parse_error_count": parse_errors,
        "sentinel_or_nonfinite_count": sentinel_count,
        "nonfinite_value_count": non_finite,
        "negative_rain_count": negative_rain,
        "negative_srad_count": negative_srad,
        "tmax_below_tmin_count": inverted_temps,
        "annual_rain_mm": round(annual_rain, 3),
        "wet_day_count_gt_0p1mm": wet_days,
        "max_dry_spell_days_rain_le_0p1mm": max_dry_spell,
        "qc_status": qc_status,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "checks": checks,
        "monthly_statistics": month_rows,
    }
    return result, month_rows


def legacy_cli_evidence() -> dict | None:
    if not LEGACY_CLI.is_file():
        return None
    lines = LEGACY_CLI.read_text(encoding="latin-1").splitlines()
    climate_window = None
    valid_counts = None
    source_label = None
    for idx, line in enumerate(lines):
        if line.strip().upper().startswith("@BEGYR") and idx + 1 < len(lines):
            values = lines[idx + 1].split()
            if len(values) >= 6:
                climate_window = {
                    "start": f"{int(values[0]):04d}-{int(values[1]):02d}-{int(values[2]):02d}",
                    "end": f"{int(values[3]):04d}-{int(values[4]):02d}-{int(values[5]):02d}",
                }
        if line.strip().startswith("Valid"):
            values = [int(value) for value in line.split(":", 1)[1].split() if value.lstrip("+-").isdigit()]
            if values:
                valid_counts = {
                    "all_values": values[0],
                    "rain": values[1] if len(values) > 1 else None,
                    "tmax": values[2] if len(values) > 2 else None,
                    "tmin": values[3] if len(values) > 3 else None,
                    "srad": values[4] if len(values) > 4 else None,
                }
        if "Calculated_from_daily_data" in line:
            source_label = line.split()[-1]
    broad_run_dir = LEGACY_CLI.parents[2]
    same_hash_copy_count = sum(
        1
        for candidate in broad_run_dir.rglob("CNYC.CLI")
        if sha256(candidate) == sha256(LEGACY_CLI)
    )
    return {
        "path": LEGACY_CLI.relative_to(ROOT).as_posix(),
        "sha256": sha256(LEGACY_CLI),
        "same_hash_copy_count_in_broad_null_run": same_hash_copy_count,
        "station_id": "CNYC" if any(line.strip().upper() == "*CLIMATE:CNYC" for line in lines) else None,
        "contains_wgen_parameter_section": any(line.strip().upper() == "*WGEN PARAMETERS" for line in lines),
        "source_label_in_file": source_label,
        "recorded_flagged_data_window": climate_window,
        "valid_daily_records_by_variable": valid_counts,
        "eligible_for_this_pilot": False,
        "exclusion_reason": (
            "The file's recorded window includes validation year 2014, its exact estimator inputs are not documented as 2004-2013 only, "
            "and the associated PDI env_args explicitly set random_weather=false."
        ),
        "source_path_in_current_repo_code": "src/compare_yc_windows480_vs_pdi_yearly_ic0.py references my_data/CNYC.CLI",
        "referenced_my_data_cli_exists": (ROOT / "my_data" / "CNYC.CLI").is_file(),
    }


def crosscheck_cleaned_weather(qc_rows: list[dict]) -> tuple[list[dict], dict]:
    if not CLEANED_WEATHER_CSV.is_file():
        return [], {"status": "source_csv_missing", "path": CLEANED_WEATHER_CSV.relative_to(ROOT).as_posix()}
    totals: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)
    with CLEANED_WEATHER_CSV.open("r", encoding="utf-8-sig", newline="") as stream:
        for item in csv.DictReader(stream):
            if item.get("station", "").upper() != "YCA":
                continue
            year = int(item["year"])
            totals[year] += float(item["RAIN"])
            counts[year] += 1
    checks = []
    for row in qc_rows:
        year = row["year"]
        source_total = totals.get(year)
        delta = None if source_total is None else round(row["annual_rain_mm"] - source_total, 4)
        checks.append(
            {
                "site": "YC",
                "year": year,
                "wth_file": row["weather_file"],
                "wth_sha256": row["file_sha256"],
                "wth_daily_records": row["record_count"],
                "cleaned_csv_daily_records": counts.get(year, 0),
                "wth_annual_rain_mm": row["annual_rain_mm"],
                "cleaned_csv_annual_rain_mm": None if source_total is None else round(source_total, 4),
                "rain_total_delta_mm": delta,
                "match_within_0p1mm": (
                    source_total is not None
                    and counts.get(year, 0) == row["record_count"]
                    and abs(delta) <= 0.1
                ),
            }
        )
    return checks, {
        "status": "pass" if len(checks) == len(qc_rows) and all(item["match_within_0p1mm"] for item in checks) else "fail",
        "path": CLEANED_WEATHER_CSV.relative_to(ROOT).as_posix(),
        "training_years_checked": [item["year"] for item in checks],
    }


def main() -> None:
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"YC input directory not found: {INPUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    historical_files = {
        int(path.name[4:6]): path
        for path in INPUT_DIR.glob("CNYC??01.WTH")
        if path.name[4:6].isdigit()
    }
    available_years = sorted(2000 + yy for yy in historical_files)
    rows = []
    monthly_rows = []
    missing_train_years = []
    for year in TRAIN_YEARS:
        source = INPUT_DIR / f"CNYC{year % 100:02d}01.WTH"
        if not source.is_file():
            missing_train_years.append(year)
            continue
        result, month_rows = parse_weather(source, year)
        rows.append(result)
        monthly_rows.extend(month_rows)

    crosscheck_rows, crosscheck_status = crosscheck_cleaned_weather(rows)
    annual_rains = [row["annual_rain_mm"] for row in rows]
    quartiles = statistics.quantiles(annual_rains, n=4, method="inclusive") if len(annual_rains) >= 2 else [None] * 3
    q1, median_rain, q3 = (quartiles[0], statistics.median(annual_rains), quartiles[-1]) if annual_rains else (None, None, None)
    lower_rain_review_fence = None if q1 is None else q1 - 1.5 * (q3 - q1)
    rainfall_review_years = [
        row["year"] for row in rows
        if lower_rain_review_fence is not None and row["annual_rain_mm"] < lower_rain_review_fence
    ]
    for row in rows:
        row["annual_rainfall_review_flag"] = "review_low_annual_rainfall" if row["year"] in rainfall_review_years else ""

    cli_files = sorted(path.name for path in INPUT_DIR.glob("*.CLI"))
    legacy_cli = legacy_cli_evidence()
    all_train_qc_pass = len(rows) == len(TRAIN_YEARS) and all(row["qc_status"] == "pass" for row in rows)
    source_evidence = {
        "training_years": TRAIN_YEARS,
        "validation_years": KNOWN_VALIDATION_YEARS,
        "configured_independent_test_years": [],
        "available_real_weather_years_in_yc_input": available_years,
        "extra_year_candidates_outside_train_and_validation": sorted(
            set(available_years) - set(TRAIN_YEARS) - set(KNOWN_VALIDATION_YEARS)
        ),
        "independent_test_status": "not_available_or_not_verified",
        "independent_test_reason": (
            "2000-2003 exist as historical weather files, but the repository does not provide a complete, auditable log "
            "of prior model selection and manual comparisons for those years. A 2000-2023 null-run directory also exists, "
            "so they are not certified as untouched test years."
        ),
        "split_evidence": [
            "configs/055_00_yca_lowIC_expanded_action_maskableppo.json",
            "docs/yc_weather_pipeline_audit.md",
        ],
        "broad_null_run_evidence": "DSSAT_auto_validation/run_CNYC0802_DSSAT480_IC0_null_2000_2023",
        "missing_train_year_files": missing_train_years,
        "train_weather_qc_status": "pass" if all_train_qc_pass else "fail",
        "train_annual_rainfall_description_mm": {
            "minimum": min(annual_rains) if annual_rains else None,
            "maximum": max(annual_rains) if annual_rains else None,
            "mean": round(statistics.fmean(annual_rains), 3) if annual_rains else None,
            "median": round(median_rain, 3) if median_rain is not None else None,
            "q1_inclusive": round(q1, 3) if q1 is not None else None,
            "q3_inclusive": round(q3, 3) if q3 is not None else None,
            "low_outlier_review_fence_1p5_iqr": round(lower_rain_review_fence, 3) if lower_rain_review_fence is not None else None,
            "manual_review_years": rainfall_review_years,
            "interpretation": "Descriptive review flag only; it does not invalidate an observed weather year.",
        },
        "cleaned_csv_rainfall_crosscheck": crosscheck_status,
        "source_files_modified": False,
    }
    write_json(OUT_DIR / "split_audit.json", source_evidence)
    write_csv(
        OUT_DIR / "train_weather_qc.csv",
        [
            "site", "station_code", "weather_station_id", "year", "weather_file", "file_sha256", "file_bytes",
            "record_count", "expected_record_count", "first_date_code", "last_date_code", "duplicate_date_count",
            "missing_date_count", "extra_date_count", "parse_error_count", "sentinel_or_nonfinite_count",
            "nonfinite_value_count", "negative_rain_count", "negative_srad_count", "tmax_below_tmin_count",
            "annual_rain_mm", "wet_day_count_gt_0p1mm", "max_dry_spell_days_rain_le_0p1mm", "annual_rainfall_review_flag",
            "qc_status", "failed_checks",
        ],
        [{**row, "failed_checks": ";".join(row["failed_checks"])} for row in rows],
    )
    write_csv(
        OUT_DIR / "train_weather_source_crosscheck.csv",
        [
            "site", "year", "wth_file", "wth_sha256", "wth_daily_records", "cleaned_csv_daily_records",
            "wth_annual_rain_mm", "cleaned_csv_annual_rain_mm", "rain_total_delta_mm", "match_within_0p1mm",
        ],
        crosscheck_rows,
    )
    write_csv(
        OUT_DIR / "train_weather_monthly_statistics.csv",
        [
            "site", "year", "month", "record_count", "rain_total_mm", "wet_day_count_gt_0p1mm",
            "tmax_mean", "tmax_sd_population", "tmin_mean", "tmin_sd_population", "srad_mean", "srad_sd_population",
        ],
        monthly_rows,
    )

    cli_status = {
        "status": "BLOCKED_CLI_GENERATION",
        "site": "YC",
        "station_code": "YCA",
        "wth_station_id": "CNYC",
        "filex_wsta_examples": ["CNYC0801", "CNYC1401"],
        "filex_template": "DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/YC/CNYC0801.MZX",
        "yc_cli_files_in_current_input_directory": cli_files,
        "legacy_yc_cli_candidate_outside_current_input_directory": legacy_cli,
        "cli_path": None,
        "cli_sha256": None,
        "parameter_source_years": None,
        "parameter_estimator": None,
        "parameter_estimator_version": None,
        "parameter_estimator_run": False,
        "in_project_official_weather_tool_found": False,
        "reason": (
            "The YC lowIC input contains no .CLI. A legacy CNYC.CLI exists in the 2000-2023 null-run artifacts, but its internal flagged-data window "
            "is 2008-2014, its training-only estimation provenance is absent, and its associated Gym-DSSAT run set random_weather=false. "
            "The checked-in Gym-DSSAT 0.0.5 source can request DSSAT WGEN and pass a seed, but it does not estimate YC .CLI parameters. "
            "No verified WeatherMan parameter-estimation executable or script is present in the project. Project AGENTS.md limits file inspection "
            "to this project, so external installation directories were not inspected."
        ),
        "source_train_years": TRAIN_YEARS,
        "input_weather_hashes": {str(row["year"]): row["file_sha256"] for row in rows},
        "provenance_verified": False,
        "generator_artifacts_created": 0,
        "weather_realizations_created": 0,
        "dssat_smoke_runs": 0,
        "ppo_runs": 0,
        "wsta_cli_mapping_status": "requires runtime confirmation; current FileX WSTA examples are 8-character year-specific stems",
        "official_steps": [
            "In WeatherMan, select the YC station code CNYC and import only CNYC0401.WTH through CNYC1301.WTH as the IBSNAT3 daily format.",
            "Use Generate > Calculate Parameters and restrict the period to 2004/001 through 2013/365, selecting WGEN parameters.",
            "Export and preserve the generated climate file with WeatherMan version, exact period, source file hashes, and output hash.",
            "Confirm how the resulting CNYC climate file maps to the current FileX WSTA before a DSSAT WGEN smoke run; do not copy another station's CLI.",
        ],
        "official_manual": "https://dssat.net/wp-content/uploads/2011/10/DSSAT-vol3.pdf",
        "official_source": "https://github.com/DSSAT/dssat-csm-os/blob/develop/Weather/WGEN.for",
    }
    write_json(OUT_DIR / "yc_cli_provenance.json", cli_status)

    manifest_fields = [
        "site", "station_code", "weather_realization_id", "weather_generation_seed", "ppo_seed", "weather_source",
        "weather_parameter_artifact", "cli_path_or_id", "cli_sha256", "source_train_years", "generation_timestamp",
        "generator_name", "generator_version", "weather_artifact_path", "weather_artifact_sha256",
    ]
    write_csv(OUT_DIR / "yc_weather_manifest.csv", manifest_fields, [])
    write_csv(
        OUT_DIR / "weather_quality_summary.csv",
        [
            "site", "weather_realization_id", "weather_generation_seed", "weather_artifact_path", "weather_artifact_sha256",
            "hard_gate_status", "statistical_sanity_status", "notes",
        ],
        [],
    )
    write_csv(
        OUT_DIR / "dssat_smoke_summary.csv",
        [
            "site", "weather_realization_id", "weather_generation_seed", "reset_status", "season_status", "planting_date",
            "termination_date", "yield_kg_ha", "warning_log", "fatal_error", "smoke_status",
        ],
        [],
    )
    write_json(
        OUT_DIR / "seed_reproducibility.json",
        {
            "status": "not_tested_cli_generation_blocked",
            "ppo_seed": 0,
            "ppo_seed_used": False,
            "weather_generation_seeds_planned": [101, 102, 103, 104, 105],
            "weather_generation_seeds_attempted": [],
            "same_seed_reproducibility": None,
            "different_seed_weather_difference": None,
            "evidence_artifacts": [],
            "reason": "No credible YC .CLI was available, so no weather generation call was made.",
        },
    )
    write_json(
        OUT_DIR / "logs" / "audit_run.json",
        {
            "task": "003_01_yc_wgen_cli_pilot",
            "command": "python scripts/audit_yc_wgen_cli_pilot.py",
            "source_input_directory": INPUT_DIR.relative_to(ROOT).as_posix(),
            "train_years": TRAIN_YEARS,
            "train_weather_qc_status": "pass" if all_train_qc_pass else "fail",
            "historical_train_files_checked": len(rows),
            "cli_generation_attempted": False,
            "wgen_pilot_attempted": False,
            "dssat_smoke_attempted": False,
            "ppo_training_attempted": False,
            "cleaned_csv_crosscheck": crosscheck_status,
            "legacy_cli_candidate": legacy_cli,
            "qc_rows": len(rows),
            "cli_files_in_yc_input_directory": cli_files,
        },
    )
    print(
        json.dumps(
            {
                "output_directory": OUT_DIR.relative_to(ROOT).as_posix(),
                "available_historical_years": available_years,
                "train_year_count": len(rows),
                "train_weather_qc_status": "pass" if all_train_qc_pass else "fail",
                "failed_years": [row["year"] for row in rows if row["qc_status"] != "pass"],
                "cli_status": cli_status["status"],
                "realizations_created": 0,
                "dssat_smoke_runs": 0,
                "ppo_runs": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
