#!/usr/bin/env python3
"""Independently audit frozen YC WGEN parameters without changing the CLI."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_dssat_cli as baseline  # noqa: E402


FITTING_WEATHER = Path("results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv")
FITTING_SHA256 = "4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34"
FROZEN_CLI = Path("results/yc_wgen_cli_pilot/003_06_05_02/final/CNYC.CLI")
FROZEN_CLI_SHA256 = "65CF134600A5881706A5D435E1A09B276ED92A21FA5ABE2E18AAF63AF1E3A929"
PARAMETERS = (
    "SDMN", "SDSD", "SWMN", "SWSD", "XDMN", "XDSD", "XWMN", "XWSD",
    "NAMN", "NASD", "ALPHA", "RTOT", "PDW", "RNUM",
)
PRECISION = {
    "SDMN": 1, "SDSD": 1, "SWMN": 1, "SWSD": 1, "XDMN": 1, "XDSD": 1,
    "XWMN": 1, "XWSD": 1, "NAMN": 1, "NASD": 1, "ALPHA": 3,
    "RTOT": 1, "PDW": 3, "RNUM": 1,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _read_independently(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"DATE", "SRAD", "TMAX", "TMIN", "RAIN"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"天气 CSV 缺少字段: {sorted(missing)}")
        for line_no, row in enumerate(reader, start=2):
            try:
                current = date.fromisoformat(row["DATE"].strip())
                values = {name: float(row[name]) for name in ("SRAD", "TMAX", "TMIN", "RAIN")}
            except (TypeError, ValueError) as exc:
                raise ValueError(f"第 {line_no} 行日期或数值无效") from exc
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"第 {line_no} 行含非有限天气值")
            if values["SRAD"] < 0 or values["RAIN"] < 0 or values["TMAX"] < values["TMIN"]:
                raise ValueError(f"第 {line_no} 行未通过基本物理检查")
            records.append({"date": current, **values})

    if len(records) != 3653 or records[0]["date"] != date(2004, 1, 1) or records[-1]["date"] != date(2013, 12, 31):
        raise ValueError("冻结天气行数或日期范围不符")
    if any((right["date"] - left["date"]).days != 1 for left, right in zip(records, records[1:])):
        raise ValueError("冻结天气日期不连续")
    return records


def _independent_fit(records: list[dict[str, Any]]) -> list[dict[str, float | int]]:
    by_month: dict[int, list[dict[str, Any]]] = defaultdict(list)
    transitions = {
        month: {key: 0 for key in ("DD", "DW", "WD", "WW")}
        for month in range(1, 13)
    }
    monthly_by_year: dict[tuple[int, int], list[float]] = defaultdict(list)
    monthly_wet_by_year: dict[tuple[int, int], int] = defaultdict(int)

    previous_wet = False
    for row in records:
        current_wet = row["RAIN"] > 0.0
        state = ("W" if previous_wet else "D") + ("W" if current_wet else "D")
        transitions[row["date"].month][state] += 1
        by_month[row["date"].month].append(row)
        ym = (row["date"].year, row["date"].month)
        monthly_by_year[ym].append(row["RAIN"])
        monthly_wet_by_year[ym] += int(current_wet)
        previous_wet = current_wet

    monthly_totals: dict[int, list[float]] = defaultdict(list)
    monthly_wet_counts: dict[int, list[float]] = defaultdict(list)
    for (year, month), rainfall in monthly_by_year.items():
        monthly_totals[month].append(math.fsum(rainfall))
        monthly_wet_counts[month].append(float(monthly_wet_by_year[(year, month)]))

    output: list[dict[str, float | int]] = []
    for month in range(1, 13):
        rows = by_month[month]
        wet = [row for row in rows if row["RAIN"] > 0.0]
        dry = [row for row in rows if row["RAIN"] <= 0.0]

        def mean_sd(items: list[dict[str, Any]], name: str) -> tuple[float, float]:
            values = [row[name] for row in items]
            if len(values) < 2:
                raise ValueError(f"月份 {month} 的 {name} 样本不足")
            return statistics.mean(values), statistics.stdev(values)

        sdmn, sdsd = mean_sd(dry, "SRAD")
        swmn, swsd = mean_sd(wet, "SRAD")
        xdmn, xdsd = mean_sd(dry, "TMAX")
        xwmn, xwsd = mean_sd(wet, "TMAX")
        namn, nasd = mean_sd(rows, "TMIN")

        wet_rain = [row["RAIN"] for row in wet]
        if len(wet_rain) < 3:
            raise ValueError(f"月份 {month} 湿日数不足 3，无法估计 ALPHA")
        y = math.log(statistics.mean(wet_rain)) - statistics.mean(math.log(value) for value in wet_rain)
        if y <= 0:
            raise ValueError(f"月份 {month} Greenwood-Durand 参数无效: y={y}")
        alpha = (8.898919 + 9.05995 * y + 0.9775373 * y * y) / (
            y * (17.79728 + 11.968477 * y + y * y)
        )
        if alpha >= 1.0:
            alpha = 0.998

        counts = transitions[month]
        dry_prev = counts["DD"] + counts["DW"]
        if dry_prev == 0:
            raise ValueError(f"月份 {month} 无前日干燥状态样本")
        output.append({
            "month": month,
            "SDMN": sdmn, "SDSD": sdsd, "SWMN": swmn, "SWSD": swsd,
            "XDMN": xdmn, "XDSD": xdsd, "XWMN": xwmn, "XWSD": xwsd,
            "NAMN": namn, "NASD": nasd,
            "ALPHA": alpha,
            "RTOT": statistics.mean(monthly_totals[month]),
            "PDW": counts["DW"] / dry_prev,
            "RNUM": statistics.mean(monthly_wet_counts[month]),
        })
    return output


def _read_cli_parameters(path: Path) -> list[dict[str, float | int]]:
    output: list[dict[str, float | int]] = []
    in_wgen = False
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith("*WGEN PARAMETERS"):
            in_wgen = True
            continue
        if in_wgen and line.startswith("*"):
            break
        if in_wgen and line.strip() and not line.lstrip().startswith("@"):
            output.append(baseline.parse_wgen_parameter_row(line))
    if [int(row["MTH"]) for row in output] != list(range(1, 13)):
        raise ValueError("冻结 CLI 的 WGEN 月份行不完整")
    return output


def _audit_descriptions() -> dict[str, dict[str, str]]:
    grouped = {
        "SDMN": ("干日 SRAD 均值", "SRAD 日值且 RAIN<=0；按历月汇集 2004-2013 所有日", "dry"),
        "SDSD": ("干日 SRAD 样本标准差", "SRAD 日值且 RAIN<=0；样本 SD 使用 n-1", "dry"),
        "SWMN": ("湿日 SRAD 均值", "SRAD 日值且 RAIN>0；按历月汇集 2004-2013 所有日", "wet"),
        "SWSD": ("湿日 SRAD 样本标准差", "SRAD 日值且 RAIN>0；样本 SD 使用 n-1", "wet"),
        "XDMN": ("干日 TMAX 均值", "TMAX 日值且 RAIN<=0；按历月汇集 2004-2013 所有日", "dry"),
        "XDSD": ("干日 TMAX 样本标准差", "TMAX 日值且 RAIN<=0；样本 SD 使用 n-1", "dry"),
        "XWMN": ("湿日 TMAX 均值", "TMAX 日值且 RAIN>0；按历月汇集 2004-2013 所有日", "wet"),
        "XWSD": ("湿日 TMAX 样本标准差", "TMAX 日值且 RAIN>0；样本 SD 使用 n-1", "wet"),
        "NAMN": ("当月全部日 TMIN 均值", "TMIN 不按降雨状态拆分；按历月汇集 2004-2013 所有日", "all"),
        "NASD": ("当月全部日 TMIN 样本标准差", "TMIN 不按降雨状态拆分；样本 SD 使用 n-1", "all"),
        "ALPHA": ("湿日降水 Gamma 形状参数", "以 RAIN>0 湿日降水量计算 Greenwood-Durand 近似；alpha>=1 截为 0.998", "wet-rain"),
        "RTOT": ("平均月降水總量 (mm)", "先算每年各月降水总量，再对 2004-2013 十个对应月份取平均", "monthly-total"),
        "PDW": ("干日後轉為濕日的條件機率", "chronological P(wet today | previous day dry); transition assigned to current day month; first prior state dry", "transition"),
        "RNUM": ("月平均濕日數", "每年每月统计 RAIN>0 天数，再对 2004-2013 十个对应月份取平均", "monthly-wet-count"),
    }
    result: dict[str, dict[str, str]] = {}
    for name, (meaning, logic, condition) in grouped.items():
        threshold_note = condition in {"dry", "wet", "wet-rain", "transition", "monthly-wet-count"}
        reference_formula = logic
        difference = ""
        status = "PASS"
        if threshold_note:
            reference_formula += "; Richardson & Wright 原文叙述另将 wet day 写作 >=0.01 inch，但其 PAR 源码计数条件为 RAIN>0.00"
            difference = "经典 WGEN 文字定义与 WGEN PAR 代码不一致；本实现依照可执行拟合源码 RAIN>0.00。冻结数据有 48 天满足 0<RAIN<0.254 mm，未改阈值。"
            status = "PASS_WITH_NOTE"
        result[name] = {
            "parameter": name,
            "plain_language_meaning": meaning,
            "DSSAT_definition": meaning,
            "reference_source": "Richardson & Wright (1984), WGEN PAR Appendix D; DSSAT User's Guide Vol. 3, Appendix B; DSSAT v4.8.0.24 Weather/WGEN.for",
            "reference_formula_or_logic": reference_formula,
            "wet_or_dry_condition": condition,
            "monthly_or_other_scope": "calendar month; fitting window 2004-2013",
            "our_formula_or_logic": logic + ("; wet flag is strictly RAIN>0.0 mm" if threshold_note else ""),
            "our_code_location": "scripts/build_dssat_cli.py:217-327; independent recheck: scripts/audit_yc_wgen_parameters_004_01.py",
            "matches_reference": "YES_WITH_DOCUMENTED_THRESHOLD_NOTE" if threshold_note else "YES",
            "difference_if_any": difference,
            "impact": "Trace-rain days can change dry/wet grouping and dependent occurrence/conditional estimates; quantified as 48/3653 days. RTOT raw monthly totals are unaffected. No threshold sensitivity or refit performed." if threshold_note else "No method difference identified.",
            "status": status,
        }
    return result


def run_audit(root: Path = ROOT, output_dir: Path | None = None) -> dict[str, Any]:
    input_path = root / FITTING_WEATHER
    cli_path = root / FROZEN_CLI
    input_hash = _sha256(input_path)
    cli_hash = _sha256(cli_path)
    if input_hash != FITTING_SHA256:
        raise ValueError(f"冻结天气 SHA256 不匹配: {input_hash}")
    if cli_hash != FROZEN_CLI_SHA256:
        raise ValueError(f"冻结 CLI SHA256 不匹配: {cli_hash}")

    independent_records = _read_independently(input_path)
    wet_positive = sum(row["RAIN"] > 0.0 for row in independent_records)
    wet_trace = sum(0.0 < row["RAIN"] < 0.254 for row in independent_records)
    independent = _independent_fit(independent_records)

    build_records, _ = baseline.read_weather(input_path)
    build_fit = baseline.fit_monthly_statistics(build_records)
    cli_rows = _read_cli_parameters(cli_path)
    crosschecks: list[dict[str, Any]] = []
    raw_mismatches: list[str] = []
    cli_unexplained: list[str] = []
    for independent_row, build_row, cli_row in zip(independent, build_fit, cli_rows):
        month = int(independent_row["month"])
        for name in PARAMETERS:
            ivalue = float(independent_row[name])
            bvalue = float(build_row[name])
            cvalue = float(cli_row[name])
            raw_diff = ivalue - bvalue
            decimals = PRECISION[name]
            allowed_cli_diff = 0.5 * (10 ** -decimals) + 1e-12
            raw_match = math.isclose(ivalue, bvalue, rel_tol=1e-12, abs_tol=1e-12)
            cli_explained = abs(ivalue - cvalue) <= allowed_cli_diff
            if not raw_match:
                raw_mismatches.append(f"{month:02d}:{name}")
            if not cli_explained:
                cli_unexplained.append(f"{month:02d}:{name}")
            crosschecks.append({
                "month": month,
                "parameter": name,
                "independent_value": f"{ivalue:.15g}",
                "build_dssat_cli_value": f"{bvalue:.15g}",
                "frozen_cli_value": f"{cvalue:.15g}",
                "independent_minus_build": f"{raw_diff:.15g}",
                "independent_minus_cli": f"{ivalue - cvalue:.15g}",
                "cli_decimal_places": decimals,
                "cli_rounding_tolerance": f"{allowed_cli_diff:.15g}",
                "raw_match": "YES" if raw_match else "NO",
                "cli_difference_explained_by_rounding": "YES" if cli_explained else "NO",
            })

    descriptions = _audit_descriptions()
    audit_rows = [descriptions[name] for name in PARAMETERS]
    method_status = "PASS" if not raw_mismatches and not cli_unexplained else "FAIL"
    if method_status == "PASS" and any(row["status"] == "PASS_WITH_NOTE" for row in audit_rows):
        method_status = "PASS_WITH_NOTE"
    summary = {
        "parameter_method_status": method_status,
        "parameters_checked": len(PARAMETERS),
        "parameter_month_values_crosschecked": len(crosschecks),
        "raw_values_exactly_matching": len(crosschecks) - len(raw_mismatches),
        "cli_serialized_values_explained_by_rounding": len(crosschecks) - len(cli_unexplained),
        "raw_mismatch_count": len(raw_mismatches),
        "unexplained_cli_difference_count": len(cli_unexplained),
        "raw_mismatches": raw_mismatches,
        "unexplained_cli_differences": cli_unexplained,
        "fitting_weather": str(FITTING_WEATHER).replace("\\", "/"),
        "fitting_period": "2004-01-01/2013-12-31",
        "fitting_days": len(independent_records),
        "fitting_weather_sha256": input_hash,
        "frozen_cli_sha256": cli_hash,
        "wet_day_rule": "RAIN > 0.0 mm; unchanged",
        "positive_rain_days": wet_positive,
        "positive_rain_below_0_254_mm_days": wet_trace,
        "reference_threshold_note": "Richardson & Wright (1984) prose states >=0.01 inch; WGEN PAR Appendix D code counts RAIN>0.00. Fitting retained the code-aligned rule and did not use validation data.",
        "runtime_alpha_effective_range": [0.01, 0.998],
        "fitted_alpha_range": [min(float(row["ALPHA"]) for row in independent), max(float(row["ALPHA"]) for row in independent)],
        "validation_data_used_for_fitting": False,
        "cli_or_input_modified": False,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(output_dir / "parameter_independent_crosscheck.csv", crosschecks)
        _write_csv(output_dir / "wgen_parameter_audit.csv", audit_rows)
        (output_dir / "parameter_method_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"拒绝写入空文件: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/yc_wgen_cli_pilot/004_01/parameter_audit",
    )
    args = parser.parse_args()
    result = run_audit(ROOT, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["parameter_method_status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
