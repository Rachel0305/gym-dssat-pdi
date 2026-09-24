#!/usr/bin/env python3
"""Build a deterministic, train-only DSSAT CLI candidate from daily weather."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable


EXPECTED_SHA256 = "4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34"
EXPECTED_START = date(2004, 1, 1)
EXPECTED_END = date(2013, 12, 31)
EXPECTED_ROWS = 3653
WET_RULE = "RAIN > 0.0 mm (positive precipitation; original WGEN PAR code)"
WEATHER_VARIABLES = ("SRAD", "TMAX", "TMIN", "RAIN")
FLAG_VARIABLES = ("RAIN", "TMAX", "TMIN", "SRAD", "SUNH", "DEWP", "WIND", "PAR", "TDRY", "TWET", "EVAP", "RHUM")
CLI_SECTIONS = (
    "*CLIMATE",
    "*MONTHLY AVERAGES",
    "*WGEN PARAMETERS",
    "*RANGE CHECK VALUES",
    "*FLAGGED DATA COUNT",
)


@dataclass(frozen=True)
class WeatherDay:
    day: date
    srad: float
    tmax: float
    tmin: float
    rain: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _number(row: dict[str, str], name: str, line_number: int) -> float:
    raw = row.get(name, "")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"第 {line_number} 行 {name} 不是有效数字: {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"第 {line_number} 行 {name} 不是有限值")
    return value


def read_weather(path: Path, expected_sha256: str = EXPECTED_SHA256) -> tuple[list[WeatherDay], str]:
    actual_hash = sha256_file(path)
    if expected_sha256 and actual_hash != expected_sha256.upper():
        raise ValueError(f"输入 SHA256 不匹配: expected={expected_sha256}, actual={actual_hash}")

    records: list[WeatherDay] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing_columns = {"DATE", *WEATHER_VARIABLES}.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"天气输入缺少列: {sorted(missing_columns)}")
        for line_number, row in enumerate(reader, start=2):
            try:
                current_date = date.fromisoformat(row["DATE"].strip())
            except (TypeError, ValueError) as exc:
                raise ValueError(f"第 {line_number} 行 DATE 无效: {row.get('DATE')!r}") from exc
            srad = _number(row, "SRAD", line_number)
            tmax = _number(row, "TMAX", line_number)
            tmin = _number(row, "TMIN", line_number)
            rain = _number(row, "RAIN", line_number)
            if srad < 0:
                raise ValueError(f"{current_date}: SRAD < 0")
            if tmax < tmin:
                raise ValueError(f"{current_date}: TMAX < TMIN")
            if rain < 0:
                raise ValueError(f"{current_date}: RAIN < 0")
            records.append(WeatherDay(current_date, srad, tmax, tmin, rain))

    if len(records) != EXPECTED_ROWS:
        raise ValueError(f"行数错误: expected={EXPECTED_ROWS}, actual={len(records)}")
    if not records or records[0].day != EXPECTED_START or records[-1].day != EXPECTED_END:
        raise ValueError("天气日期范围不匹配冻结的 2004-01-01 至 2013-12-31")
    if len({record.day for record in records}) != len(records):
        raise ValueError("DATE 存在重复")
    if any(right.day.toordinal() - left.day.toordinal() != 1 for left, right in zip(records, records[1:])):
        raise ValueError("DATE 未逐日连续")
    if any(record.day.year < 2004 or record.day.year > 2013 for record in records):
        raise ValueError("检测到冻结训练期以外的年份")
    return records, actual_hash


def _mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("条件子集中无观测，不能估计均值和标准差")
    mean = math.fsum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    sd = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1))
    return mean, sd


def gamma_shape_greenwood_durand(wet_rain: list[float]) -> float:
    """Port of the approximation in Richardson & Wright's WGEN PAR source."""
    if len(wet_rain) < 3 or any(value <= 0 for value in wet_rain):
        raise ValueError("Gamma 形状参数至少需要 3 个正值湿日降水量")
    mean_rain = math.fsum(wet_rain) / len(wet_rain)
    mean_log_rain = math.fsum(math.log(value) for value in wet_rain) / len(wet_rain)
    y = math.log(mean_rain) - mean_log_rain
    if y <= 0:
        raise ValueError(f"Gamma 形状参数不可估计: log(mean)-mean(log)={y}")
    numerator = 8.898919 + 9.05995 * y + 0.9775373 * y * y
    denominator = y * (17.79728 + 11.968477 * y + y * y)
    alpha = numerator / denominator
    if alpha >= 1.0:
        alpha = 0.998
    if alpha <= 0 or not math.isfinite(alpha):
        raise ValueError(f"Gamma 形状参数无效: {alpha}")
    return alpha


def count_monthly_transitions(records: list[WeatherDay], initial_previous_wet: bool = False) -> dict[int, dict[str, int]]:
    transitions = {
        month: {"dry_to_dry": 0, "dry_to_wet": 0, "wet_to_dry": 0, "wet_to_wet": 0}
        for month in range(1, 13)
    }
    previous_wet = initial_previous_wet
    for record in records:
        wet = record.rain > 0.0
        key = ("wet" if previous_wet else "dry") + "_to_" + ("wet" if wet else "dry")
        transitions[record.day.month][key] += 1
        previous_wet = wet
    return transitions


def _monthly_total_means(records: list[WeatherDay]) -> dict[int, tuple[float, float]]:
    totals: dict[tuple[int, int], float] = defaultdict(float)
    wet_counts: dict[tuple[int, int], int] = defaultdict(int)
    for record in records:
        key = (record.day.year, record.day.month)
        totals[key] += record.rain
        wet_counts[key] += record.rain > 0.0
    by_month_total: dict[int, list[float]] = defaultdict(list)
    by_month_wet: dict[int, list[float]] = defaultdict(list)
    for (year, month), total in sorted(totals.items()):
        by_month_total[month].append(total)
        by_month_wet[month].append(float(wet_counts[(year, month)]))
    return {
        month: (math.fsum(by_month_total[month]) / len(by_month_total[month]),
                math.fsum(by_month_wet[month]) / len(by_month_wet[month]))
        for month in range(1, 13)
    }


def fit_monthly_statistics(records: list[WeatherDay]) -> list[dict[str, float | int]]:
    grouped: dict[int, list[WeatherDay]] = defaultdict(list)
    for record in records:
        grouped[record.day.month].append(record)
    totals_and_wetdays = _monthly_total_means(records)
    # The WGEN PAR source initializes its prior-day rain state as dry.
    transitions = count_monthly_transitions(records, initial_previous_wet=False)

    result: list[dict[str, float | int]] = []
    for month in range(1, 13):
        days = grouped[month]
        wet_days = [record for record in days if record.rain > 0.0]
        dry_days = [record for record in days if record.rain <= 0.0]
        if len(wet_days) < 3:
            raise ValueError(f"月份 {month}: 湿日数不足 3，不能估计 ALPHA")
        dry_to_dry = transitions[month]["dry_to_dry"]
        dry_to_wet = transitions[month]["dry_to_wet"]
        wet_to_dry = transitions[month]["wet_to_dry"]
        wet_to_wet = transitions[month]["wet_to_wet"]
        dry_denominator = dry_to_dry + dry_to_wet
        wet_denominator = wet_to_dry + wet_to_wet
        if dry_denominator == 0 or wet_denominator == 0:
            raise ValueError(f"月份 {month}: 转移概率分母为 0")

        srad_dry_mean, srad_dry_sd = _mean_sd([record.srad for record in dry_days])
        srad_wet_mean, srad_wet_sd = _mean_sd([record.srad for record in wet_days])
        tmax_dry_mean, tmax_dry_sd = _mean_sd([record.tmax for record in dry_days])
        tmax_wet_mean, tmax_wet_sd = _mean_sd([record.tmax for record in wet_days])
        tmin_mean, tmin_sd = _mean_sd([record.tmin for record in days])
        monthly_rain, monthly_wet_count = totals_and_wetdays[month]
        result.append({
            "month": month,
            "day_count": len(days),
            "dry_day_count": len(dry_days),
            "wet_day_count": len(wet_days),
            "rain_total_mm": math.fsum(record.rain for record in days),
            "rainy_day_count": len(wet_days),
            "dry_to_dry_count": dry_to_dry,
            "dry_to_wet_count": dry_to_wet,
            "wet_to_dry_count": wet_to_dry,
            "wet_to_wet_count": wet_to_wet,
            "p_wet_given_dry": dry_to_wet / dry_denominator,
            "p_wet_given_wet": wet_to_wet / wet_denominator,
            "SDMN": srad_dry_mean,
            "SDSD": srad_dry_sd,
            "SWMN": srad_wet_mean,
            "SWSD": srad_wet_sd,
            "XDMN": tmax_dry_mean,
            "XDSD": tmax_dry_sd,
            "XWMN": tmax_wet_mean,
            "XWSD": tmax_wet_sd,
            "NAMN": tmin_mean,
            "NASD": tmin_sd,
            "ALPHA": gamma_shape_greenwood_durand([record.rain for record in wet_days]),
            "RTOT": monthly_rain,
            "PDW": dry_to_wet / dry_denominator,
            "RNUM": monthly_wet_count,
        })
    return result


def monthly_weather_summary(records: list[WeatherDay]) -> list[dict[str, float | int]]:
    totals_and_wetdays = _monthly_total_means(records)
    grouped: dict[int, list[WeatherDay]] = defaultdict(list)
    for record in records:
        grouped[record.day.month].append(record)
    result = []
    for month in range(1, 13):
        days = grouped[month]
        rain_total_mean, wet_count_mean = totals_and_wetdays[month]
        result.append({
            "month": month,
            "observed_day_count": len(days),
            "month_year_count": len({record.day.year for record in days}),
            "SRAD_mean_MJ_m2_day": math.fsum(record.srad for record in days) / len(days),
            "TMAX_mean_C": math.fsum(record.tmax for record in days) / len(days),
            "TMIN_mean_C": math.fsum(record.tmin for record in days) / len(days),
            "SAMN": math.fsum(record.srad for record in days) / len(days),
            "XAMN": math.fsum(record.tmax for record in days) / len(days),
            "NAMN": math.fsum(record.tmin for record in days) / len(days),
            "RAIN_total_mm_all_years": math.fsum(record.rain for record in days),
            "RAIN_mean_month_total_mm": rain_total_mean,
            "RTOT": rain_total_mean,
            "wet_day_count_all_years": sum(record.rain > 0.0 for record in days),
            "wet_day_mean_per_month": wet_count_mean,
            "RNUM": wet_count_mean,
        })
    return result


def crosscheck_statistics(records: list[WeatherDay], fitted: list[dict[str, float | int]]) -> dict[str, object]:
    """Independent two-pass checks using statistics.mean/stdev and raw group counts."""
    by_month: dict[int, list[WeatherDay]] = defaultdict(list)
    for record in records:
        by_month[record.day.month].append(record)
    checks: list[dict[str, object]] = []
    for row in fitted:
        month = int(row["month"])
        days = by_month[month]
        wet = [day for day in days if day.rain > 0.0]
        dry = [day for day in days if day.rain <= 0.0]
        tests = {
            "SDMN": (float(row["SDMN"]), statistics.mean(day.srad for day in dry)),
            "SDSD": (float(row["SDSD"]), statistics.stdev(day.srad for day in dry)),
            "SWMN": (float(row["SWMN"]), statistics.mean(day.srad for day in wet)),
            "SWSD": (float(row["SWSD"]), statistics.stdev(day.srad for day in wet)),
            "XDMN": (float(row["XDMN"]), statistics.mean(day.tmax for day in dry)),
            "XDSD": (float(row["XDSD"]), statistics.stdev(day.tmax for day in dry)),
            "XWMN": (float(row["XWMN"]), statistics.mean(day.tmax for day in wet)),
            "XWSD": (float(row["XWSD"]), statistics.stdev(day.tmax for day in wet)),
            "NAMN": (float(row["NAMN"]), statistics.mean(day.tmin for day in days)),
            "NASD": (float(row["NASD"]), statistics.stdev(day.tmin for day in days)),
        }
        mismatches = [name for name, (primary, secondary) in tests.items()
                      if not math.isclose(primary, secondary, rel_tol=1e-12, abs_tol=1e-12)]
        previous_wet = False
        transition_counts = {"dry_to_dry": 0, "dry_to_wet": 0, "wet_to_dry": 0, "wet_to_wet": 0}
        for day in records:
            current_wet = day.rain > 0.0
            if day.day.month == month:
                transition_key = ("wet" if previous_wet else "dry") + "_to_" + ("wet" if current_wet else "dry")
                transition_counts[transition_key] += 1
            previous_wet = current_wet
        dry_n = transition_counts["dry_to_dry"] + transition_counts["dry_to_wet"]
        wet_n = transition_counts["wet_to_dry"] + transition_counts["wet_to_wet"]
        if not math.isclose(float(row["PDW"]), transition_counts["dry_to_wet"] / dry_n, rel_tol=1e-12, abs_tol=1e-12):
            mismatches.append("PDW")
        if not math.isclose(float(row["p_wet_given_wet"]), transition_counts["wet_to_wet"] / wet_n, rel_tol=1e-12, abs_tol=1e-12):
            mismatches.append("PWW_audit")
        year_months: dict[int, tuple[float, int]] = {}
        for day in records:
            if day.day.month == month:
                total, wet_n_month = year_months.get(day.day.year, (0.0, 0))
                year_months[day.day.year] = (total + day.rain, wet_n_month + (day.rain > 0.0))
        if not math.isclose(float(row["RTOT"]), statistics.mean(value[0] for value in year_months.values()), rel_tol=1e-12, abs_tol=1e-12):
            mismatches.append("RTOT")
        if not math.isclose(float(row["RNUM"]), statistics.mean(value[1] for value in year_months.values()), rel_tol=1e-12, abs_tol=1e-12):
            mismatches.append("RNUM")
        wet_vals = [day.rain for day in wet]
        mean_rain = statistics.mean(wet_vals)
        mean_log = statistics.mean(math.log(value) for value in wet_vals)
        y = math.log(mean_rain) - mean_log
        alpha_secondary = (8.898919 + 9.05995 * y + 0.9775373 * y**2) / (y * (17.79728 + 11.968477 * y + y**2))
        if alpha_secondary >= 1:
            alpha_secondary = 0.998
        if not math.isclose(float(row["ALPHA"]), alpha_secondary, rel_tol=1e-12, abs_tol=1e-12):
            mismatches.append("ALPHA")
        checks.append({
            "month": month,
            "independent_statistics_checked": list(tests) + ["ALPHA", "PDW", "PWW_audit", "RTOT", "RNUM"],
            "mismatches": mismatches,
            "wet_count_independent": len(wet),
            "dry_count_independent": len(dry),
        })
    passed = all(not item["mismatches"] for item in checks)
    return {
        "method": "independent statistics.mean/stdev two-pass calculation and separately recounted raw daily groups",
        "months_checked": len(checks),
        "checks_per_month": 15,
        "passed": passed,
        "month_results": checks,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"拒绝写入空 CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def climate_summary(records: list[WeatherDay]) -> dict[str, float]:
    monthly_means: dict[int, list[float]] = defaultdict(list)
    annual_rain: dict[int, float] = defaultdict(float)
    for record in records:
        tmean = (record.tmax + record.tmin) / 2.0
        monthly_means[record.day.month].append(tmean)
        annual_rain[record.day.year] += record.rain
    monthly_tmean = {
        month: math.fsum(values) / len(values)
        for month, values in monthly_means.items()
    }
    return {
        "TAV": math.fsum((record.tmax + record.tmin) / 2.0 for record in records) / len(records),
        "AMP": (max(monthly_tmean.values()) - min(monthly_tmean.values())) / 2.0,
        "SRAY": math.fsum(record.srad for record in records) / len(records),
        "TMXY": math.fsum(record.tmax for record in records) / len(records),
        "TMNY": math.fsum(record.tmin for record in records) / len(records),
        "RAIY": math.fsum(annual_rain.values()) / len(annual_rain),
    }


def _cli_text(
    records: list[WeatherDay], station: str, latitude: float, longitude: float, elevation: float,
    wgen: list[dict[str, float | int]], monthly: list[dict[str, float | int]],
) -> str:
    climate = climate_summary(records)
    lines = [
        f"*CLIMATE:{station}",
        "",
        "@ INSI      LAT     LONG  ELEV   TAV   AMP  SRAY  TMXY  TMNY  RAIY",
        f"  {station:<4} {latitude:8.3f} {longitude:9.3f} {elevation:5.0f} {climate['TAV']:5.1f} {climate['AMP']:5.1f} {climate['SRAY']:5.1f} {climate['TMXY']:5.1f} {climate['TMNY']:5.1f} {climate['RAIY']:6.0f}",
        "@START  DURN  ANGA  ANGB REFHT WNDHT SOURCE",
        f"  {EXPECTED_START.year:4d}    10 -99.0 -99.0 -99.0 -99.0 Calculated_from_train_only_daily_data",
        "@ GSST  GSDU",
        "   -99   -99",
        "",
        "*MONTHLY AVERAGES",
        "@  MTH  SAMN  XAMN  NAMN  RTOT  RNUM  SHMN  AMTH  BMTH",
    ]
    for row in monthly:
        lines.append(
            f"{int(row['month']):6d} {float(row['SAMN']):6.1f} {float(row['XAMN']):6.1f} {float(row['NAMN']):6.1f}"
            f" {float(row['RTOT']):6.1f} {float(row['RNUM']):6.1f} {-99:4d} {-99.0:6.3f} {-99.0:6.3f}"
        )
    lines += [
        "",
        "*WGEN PARAMETERS",
        "@  MTH  SDMN  SDSD  SWMN  SWSD  XDMN  XDSD  XWMN  XWSD  NAMN  NASD ALPHA  RTOT   PDW  RNUM",
    ]
    for row in wgen:
        lines.append(
            f"{int(row['month']):6d}{float(row['SDMN']):6.1f}{float(row['SDSD']):6.1f}"
            f"{float(row['SWMN']):6.1f}{float(row['SWSD']):6.1f}{float(row['XDMN']):7.1f}"
            f"{float(row['XDSD']):6.1f}{float(row['XWMN']):7.1f}{float(row['XWSD']):6.1f}"
            f"{float(row['NAMN']):7.1f}{float(row['NASD']):6.1f}{float(row['ALPHA']):6.3f}"
            f"{float(row['RTOT']):6.1f}{float(row['PDW']):6.3f}{float(row['RNUM']):6.1f}"
        )
    lines += [
        "",
        "*RANGE CHECK VALUES",
        "@      SRAD  TMAX  TMIN  RAIN  DEWP  WIND  SUNH   PAR  TDRY  TWET  EVAP  RHUM",
        "MIN :" + "".join(f" {-99.0:6.1f}" for _ in range(12)),
        "MAX :" + "".join(f" {-99.0:6.1f}" for _ in range(12)),
        "RATE:" + "".join(f" {-99.0:6.1f}" for _ in range(12)),
        "",
        "*FLAGGED DATA COUNT",
        "@BEGYR BEGMN BEGDY ENDYR ENDMN ENDDY",
        f"  {EXPECTED_START.year:4d}     1     1  {EXPECTED_END.year:4d}    12    31",
        "@         TOTAL   RAIN   TMAX   TMIN   SRAD   SUNH   DEWP   WIND    PAR   TDRY   TWET   EVAP   RHUM",
    ]
    core_counts = {"RAIN": len(records), "TMAX": len(records), "TMIN": len(records), "SRAD": len(records)}
    for label in ("Total", "Valid", "Missing", "Error", "Above", "Below", "Rate"):
        row_values: list[int] = []
        for name in FLAG_VARIABLES:
            if name in core_counts:
                if label in ("Total", "Valid"):
                    row_values.append(core_counts[name])
                elif label in ("Missing", "Error"):
                    row_values.append(0)
                else:
                    row_values.append(-99)
            else:
                row_values.append(0 if label in ("Total", "Valid", "Missing", "Error") else -99)
        aggregate = sum(value for value in row_values if value >= 0) if label in ("Total", "Valid", "Missing", "Error") else -99
        lines.append(f"{label:<8}:" + f"{aggregate:7d}" + "".join(f"{value:7d}" for value in row_values))
    return "\n".join(lines) + "\n"


def check_cli_schema(text: str) -> dict[str, object]:
    lines = text.splitlines()
    found_sections = [section for section in CLI_SECTIONS if any(line.startswith(section) for line in lines)]
    month_rows: dict[str, list[int]] = {"monthly_averages": [], "wgen_parameters": []}
    current = ""
    malformed = []
    for number, line in enumerate(lines, start=1):
        if line.startswith("*MONTHLY AVERAGES"):
            current = "monthly_averages"
            continue
        if line.startswith("*WGEN PARAMETERS"):
            current = "wgen_parameters"
            continue
        if line.startswith("*"):
            current = ""
            continue
        if current and line.strip() and not line.lstrip().startswith("@"):
            first = line.split()[0]
            try:
                month = int(first)
            except ValueError:
                continue
            month_rows[current].append(month)
            expected_cols = 9 if current == "monthly_averages" else 15
            if len(line.split()) != expected_cols:
                malformed.append({"line": number, "columns": len(line.split()), "expected": expected_cols})
            else:
                try:
                    parsed = [float(token) for token in line.split()[1:]]
                    if not all(math.isfinite(value) for value in parsed):
                        raise ValueError("non-finite numeric value")
                except ValueError as exc:
                    malformed.append({"line": number, "numeric_parse_error": str(exc)})
        if line.startswith(("MIN :", "MAX :", "RATE:")):
            try:
                parsed = [float(token) for token in line.split(":", 1)[1].split()]
                if len(parsed) != 12 or not all(math.isfinite(value) for value in parsed):
                    raise ValueError("expected 12 finite numeric values")
            except ValueError as exc:
                malformed.append({"line": number, "numeric_parse_error": str(exc)})
        if line.strip().split(":", 1)[0].strip() in {"Total", "Valid", "Missing", "Error", "Above", "Below", "Rate"}:
            try:
                parsed = [int(token) for token in line.split(":", 1)[1].split()]
                if len(parsed) != 13:
                    raise ValueError("expected 13 integer values")
            except ValueError as exc:
                malformed.append({"line": number, "numeric_parse_error": str(exc)})
    climate_data_lines = [line for line in lines if line.startswith("  CNYC")]
    if len(climate_data_lines) != 1:
        malformed.append({"line": None, "numeric_parse_error": "station climate row missing/duplicated"})
    else:
        try:
            climate_values = [float(token) for token in climate_data_lines[0].split()[1:]]
            if len(climate_values) != 9 or not all(math.isfinite(value) for value in climate_values):
                raise ValueError("station climate row must have nine finite numeric values")
        except ValueError as exc:
            malformed.append({"line": None, "numeric_parse_error": str(exc)})
    nonfinite_tokens = [token for token in ("nan", "inf", "-inf") if token in text.lower()]
    return {
        "sections_present": found_sections,
        "sections_missing": [section for section in CLI_SECTIONS if section not in found_sections],
        "monthly_averages_months": month_rows["monthly_averages"],
        "wgen_parameters_months": month_rows["wgen_parameters"],
        "twelve_months_each": month_rows["monthly_averages"] == list(range(1, 13)) and month_rows["wgen_parameters"] == list(range(1, 13)),
        "station_identifier_present": "*CLIMATE:CNYC" in text and "  CNYC" in text,
        "numeric_rows_malformed": malformed,
        "nonfinite_tokens": nonfinite_tokens,
        "passed": len(found_sections) == len(CLI_SECTIONS) and not malformed and not nonfinite_tokens and
                  month_rows["monthly_averages"] == list(range(1, 13)) and
                  month_rows["wgen_parameters"] == list(range(1, 13)) and "*CLIMATE:CNYC" in text,
        "range_values_status": "-99 sentinel: WeatherMan range-check thresholds are editable archive QC metadata, not WGEN parameters; no threshold was invented",
        "flagged_count_status": "ABOVE/BELOW/RATE marked -99 (not evaluated without station range thresholds); no zero flags are fabricated",
    }


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build(
    weather_path: Path, output_cli: Path, station: str = "CNYC", latitude: float = 36.830,
    longitude: float = 116.570, elevation: float = 22, expected_sha256: str = EXPECTED_SHA256,
) -> dict[str, object]:
    records, input_hash = read_weather(weather_path, expected_sha256)
    wgen = fit_monthly_statistics(records)
    monthly = monthly_weather_summary(records)
    cli_text = _cli_text(records, station, latitude, longitude, elevation, wgen, monthly)
    schema = check_cli_schema(cli_text)
    if not schema["passed"]:
        raise ValueError(f"生成 CLI 静态 schema 检查失败: {schema}")
    crosscheck = crosscheck_statistics(records, wgen)
    if not crosscheck["passed"]:
        raise ValueError("独立数值交叉核对失败")

    output_cli.parent.mkdir(parents=True, exist_ok=True)
    output_cli.write_text(cli_text, encoding="ascii", newline="\n")
    monthly_path = output_cli.with_name("monthly_wgen_statistics.csv")
    summary_path = output_cli.with_name("monthly_weather_summary.csv")
    metadata_path = output_cli.with_name("cli_generation_metadata.json")
    log_path = output_cli.with_name("cli_generation_log.txt")
    _write_csv(monthly_path, wgen)
    _write_csv(summary_path, monthly)
    root = Path(__file__).resolve().parents[1]
    metadata = {
        "input_path": str(weather_path),
        "input_sha256": input_hash,
        "expected_input_sha256": expected_sha256.upper(),
        "output_path": str(output_cli),
        "output_sha256": sha256_file(output_cli),
        "station": station,
        "latitude": latitude,
        "longitude": longitude,
        "elevation_m": elevation,
        "start_date": records[0].day.isoformat(),
        "end_date": records[-1].day.isoformat(),
        "row_count": len(records),
        "years_used": list(range(2004, 2014)),
        "validation_data_used": False,
        "algorithm_sources": [
            "Richardson & Wright (1984), WGEN PAR source code, Appendix D: https://support.goldsim.com/hc/en-us/article_attachments/115026531468",
            "DSSAT User's Guide Vol. 3, WeatherMan Appendix B: https://dssat.net/wp-content/uploads/2011/10/DSSAT-vol3.pdf",
            "Soltani & Hoogenboom (2003), DSSAT WeatherMan WGEN adaptation: https://doi.org/10.3354/cr024215",
            "DSSAT public WGEN source: https://github.com/DSSAT/dssat-csm-os/blob/develop/Weather/WGEN.for",
        ],
        "script_git_commit_if_available": _git_commit(root),
        "generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "unresolved_fields": [],
        "non_wgen_missing_metadata": ["ANGA", "ANGB", "REFHT", "WNDHT", "GSST", "GSDU", "SHMN", "AMTH", "BMTH"],
        "range_qc_thresholds": "not available for YC station; -99 sentinel; not used by WGEN",
        "flagged_above_below_rate": "not evaluated and marked -99 because no station range thresholds were provided",
        "wet_day_definition": WET_RULE,
        "transitions": "month assigned by current day; chronological transitions cross month/year boundaries; first record assumes previous dry as WGEN PAR RIM1=0; leap days stay in their calendar month",
        "monthly_standard_deviation": "sample standard deviation (n-1), matching WGEN PAR source estimator",
        "gamma_shape_estimator": "Greenwood-Durand approximation as coded in WGEN PAR; alpha is capped at 0.998 when >=1; WGEN derives beta=(RTOT/RNUM)/ALPHA",
        "cli_status": "CANDIDATE_READY_FOR_WGEN_SMOKE",
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    validation_dir = output_cli.parent.parent / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    (validation_dir / "cli_schema_check.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (validation_dir / "parameter_crosscheck.json").write_text(json.dumps(crosscheck, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log_lines = [
        "YC train-only CNYC.CLI generation",
        f"input={weather_path}",
        f"input_sha256={input_hash}",
        f"rows={len(records)} dates={records[0].day}..{records[-1].day}",
        f"wet_rule={WET_RULE}",
        "validation years used: none",
        "frozen input modified: no",
        "WGEN/DSSAT/PPO executed: no",
        f"CLI sha256={metadata['output_sha256']}",
        "static schema QC: PASS",
        "independent parameter cross-check: PASS",
    ]
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8", newline="\n")
    return metadata


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather", type=Path, default=Path("results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv"))
    parser.add_argument("--station", default="CNYC")
    parser.add_argument("--latitude", type=float, default=36.830)
    parser.add_argument("--longitude", type=float, default=116.570)
    parser.add_argument("--elevation", type=float, default=22)
    parser.add_argument("--start-date", default=EXPECTED_START.isoformat())
    parser.add_argument("--end-date", default=EXPECTED_END.isoformat())
    parser.add_argument("--output", type=Path, default=Path("results/yc_wgen_cli_pilot/003_06_04/generated/CNYC.CLI"))
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.start_date != EXPECTED_START.isoformat() or args.end_date != EXPECTED_END.isoformat():
        print("ERROR: start/end date must match the frozen 2004-01-01..2013-12-31 interval", file=sys.stderr)
        return 2
    if args.station != "CNYC":
        print("ERROR: this train-only frozen input is bound to station CNYC", file=sys.stderr)
        return 2
    try:
        metadata = build(args.weather, args.output, args.station, args.latitude, args.longitude, args.elevation, EXPECTED_SHA256)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"cli_path": metadata["output_path"], "cli_sha256": metadata["output_sha256"], "cli_status": metadata["cli_status"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
