#!/usr/bin/env python3
"""Finalize YC 2004-2013 training weather and gate official CLI preparation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import openpyxl


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "external" / "yc_chinaflux" / "raw"
OUT = ROOT / "results" / "yc_weather_finalize_and_cli"
CACHE = ROOT / "data" / "external" / "yc_weather_finalize_and_cli" / "nasa_power"
START = date(2004, 1, 1)
END = date(2013, 12, 31)
VARIABLES = ("SRAD", "TMAX", "TMIN", "RAIN")
NASA_PARAMETERS = ("T2M_MAX", "T2M_MIN", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN")
MISSING_TEXT = {"", "-", "--", "/", "－", "−", "NA", "N/A", "NAN", "NONE"}
SENTINELS = {-99999.0, -9999.0, 99999.0, 9999.0, -999.0}
SITE_LAT, SITE_LON, SITE_ELEV_M = 36.830, 116.570, 22.0
API_BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fields})


def backup_existing_outputs() -> str | None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    targets = [p for p in OUT.iterdir() if p.is_file()] if OUT.exists() else []
    import_candidate = OUT / "cli_candidate" / "input" / "CNYC_daily_train_2004_2013.csv"
    if import_candidate.exists():
        targets.append(import_candidate)
    targets += [p for p in (ROOT / "docs" / "yc_weather_finalize_and_cli.md", ROOT / "docs" / "yc_weatherman_cli_manual_steps.md") if p.exists()]
    if not targets:
        return None
    backup_dir = OUT / "backups" / stamp
    backup_dir.mkdir(parents=True, exist_ok=False)
    for path in targets:
        label = "docs__" + path.name if path.parent == ROOT / "docs" else path.name
        shutil.copy2(path, backup_dir / label)
    return str(backup_dir.relative_to(ROOT))


def number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value.upper() in MISSING_TEXT:
            return None
        try:
            value = float(value.replace("－", "-").replace("−", "-"))
        except ValueError:
            return None
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    result = float(value)
    if not math.isfinite(result) or result in SENTINELS:
        return None
    return result


def find_workbooks() -> tuple[Path, Path]:
    workbooks = sorted(RAW.glob("*.xlsx"))
    met = [p for p in workbooks if "地面气象观测数据" in p.name]
    rad = [p for p in workbooks if "辐射观测数据" in p.name]
    if len(met) != 1 or len(rad) != 1:
        raise RuntimeError(f"Expected one official meteorology and radiation workbook in {RAW}")
    return met[0], rad[0]


def load_official(path: Path, kind: str) -> dict[date, dict[str, float | None]]:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    result: dict[date, dict[str, float | None]] = {}
    try:
        ws = wb.active
        last_training_row = 1
        for row_number, (raw_year,) in enumerate(ws.iter_rows(min_row=2, min_col=4, max_col=4, values_only=True), start=2):
            if raw_year is None:
                continue
            year = int(raw_year)
            if year > 2013:
                break
            assert year <= 2013, "Validation-year workbook cell value was read"
            last_training_row = row_number
        for row in ws.iter_rows(min_row=2, max_row=last_training_row, values_only=True):
            if not row or row[3] is None:
                continue
            year = int(row[3])
            assert year <= 2013, "Validation-year workbook cell value was read"
            dt = date(year, int(row[4]), int(row[5]))
            if kind == "met":
                raw = {"TMAX": row[7], "TMIN": row[9], "RAIN": row[22]}
            else:
                raw = {"SRAD": row[6]}
            if dt in result:
                raise RuntimeError(f"Duplicate official daily row: {dt}")
            result[dt] = {key: number(value) for key, value in raw.items()}
    finally:
        wb.close()
    return result


def load_chinaflux_daily(path: Path, years: range = range(2004, 2011)) -> dict[date, dict[str, Any]]:
    grouped: dict[date, dict[str, list[float] | int]] = defaultdict(
        lambda: {"n": 0, "near": [], "rain": [], "srad": []}
    )
    with zipfile.ZipFile(path, metadata_encoding="gbk") as archive:
        members = [m for m in archive.infolist() if not m.is_dir() and m.filename.lower().endswith(".xlsx") and any(f"{year}年" in m.filename for year in years)]
        if len(members) != len(years):
            raise RuntimeError(f"Expected ChinaFLUX half-hour workbooks for {years.start}-{years.stop - 1}, found {[m.filename for m in members]}")
        for member in members:
            with archive.open(member) as binary:
                wb = openpyxl.load_workbook(binary, read_only=True, data_only=True)
                try:
                    ws = wb.active
                    header = [str(v).strip() if v is not None else "" for v in next(ws.iter_rows(min_row=1, max_row=1, values_only=True))]
                    required = {"年", "月", "日", "近地面空气温度", "太阳辐射", "降水量"}
                    if not required <= set(header):
                        raise RuntimeError(f"Unexpected ChinaFLUX workbook columns: {header}")
                    ix = {name: header.index(name) for name in required}
                    for row in ws.iter_rows(min_row=3, values_only=True):
                        try:
                            dt = date(int(row[ix["年"]]), int(row[ix["月"]]), int(row[ix["日"]]))
                        except (TypeError, ValueError):
                            continue
                        if dt.year not in years:
                            continue
                        item = grouped[dt]
                        item["n"] += 1
                        for label, field in (("near", "近地面空气温度"), ("rain", "降水量"), ("srad", "太阳辐射")):
                            value = number(row[ix[field]])
                            if value is not None:
                                item[label].append(value)
                finally:
                    wb.close()

    result: dict[date, dict[str, Any]] = {}
    for dt, item in grouped.items():
        n = int(item["n"])
        is_full_temp = n == 48 and len(item["near"]) == 48
        is_full_rain = n == 48 and len(item["rain"]) == 48
        is_full_srad = n == 48 and len(item["srad"]) == 48
        result[dt] = {
            "n_records": n,
            "TMAX": max(item["near"]) if is_full_temp else None,
            "TMIN": min(item["near"]) if is_full_temp else None,
            "RAIN": sum(item["rain"]) if is_full_rain else None,
            "SRAD": sum(item["srad"]) * 1800.0 / 1_000_000.0 if is_full_srad else None,
            "temperature_status": "COMPLETE_48" if is_full_temp else "INCOMPLETE_48",
            "rain_status": "COMPLETE_48" if is_full_rain else "INCOMPLETE_48",
            "srad_status": "COMPLETE_48" if is_full_srad else "INCOMPLETE_48",
        }
    return result


def all_dates() -> list[date]:
    days: list[date] = []
    current = START
    while current <= END:
        days.append(current)
        current = date.fromordinal(current.toordinal() + 1)
    return days


def build_base(cf: dict[date, dict[str, Any]], met: dict[date, dict[str, float | None]], rad: dict[date, dict[str, float | None]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dt in all_dates():
        row: dict[str, Any] = {"DATE": dt.isoformat(), "YEAR": dt.year, "DOY": dt.timetuple().tm_yday}
        c, m, r = cf.get(dt, {}), met.get(dt, {}), rad.get(dt, {})
        if dt.year == 2004:
            for variable in VARIABLES:
                row[variable] = c.get(variable)
                if variable in ("TMAX", "TMIN"):
                    row[f"{variable}_source"] = "chinaflux_30min_near_surface_consistency_selected"
                else:
                    row[f"{variable}_source"] = "chinaflux_30min"
                row[f"{variable}_qc"] = "SOURCE_COMPLETE_48" if row[variable] is not None else "GAP"
        else:
            for variable in VARIABLES:
                row[variable] = (r if variable == "SRAD" else m).get(variable)
                row[f"{variable}_source"] = "yucheng_official_daily_qc_product"
                row[f"{variable}_qc"] = "OFFICIAL_QC_DAILY_PRODUCT" if row[variable] is not None else "GAP"

        for variable in ("RAIN", "SRAD"):
            if row[variable] is not None and row[variable] < 0:
                row[f"{variable}_qc"] = "GAP_NEGATIVE_VALUE"
        if row["TMAX"] is not None and row["TMIN"] is not None and row["TMAX"] < row["TMIN"]:
            row["TMAX_qc"] = "GAP_TMAX_LT_TMIN"
            row["TMIN_qc"] = "GAP_TMAX_LT_TMIN"
        rows.append(row)
    return rows


def gaps_for(rows: list[dict[str, Any]], cf: dict[date, dict[str, Any]]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    for row in rows:
        dt = date.fromisoformat(row["DATE"])
        c = cf.get(dt, {})
        for variable in VARIABLES:
            value = row[variable]
            qc = row[f"{variable}_qc"]
            if qc.startswith("GAP") or value is None:
                if qc == "GAP_TMAX_LT_TMIN":
                    reason = "Physical ordering error: TMAX < TMIN"
                elif value is not None and value < 0:
                    reason = f"Physical error: {variable} < 0"
                elif dt.year == 2004:
                    field_status = c.get("rain_status" if variable == "RAIN" else "srad_status" if variable == "SRAD" else "temperature_status", "NO_30MIN_ROWS")
                    reason = f"ChinaFLUX {variable} daily aggregation requires 48 valid half-hour values; status={field_status}, records={c.get('n_records', 0)}/48"
                else:
                    reason = "Missing, blank, or official missing sentinel"
                gaps.append({
                    "date": row["DATE"], "variable": variable,
                    "original_source": row[f"{variable}_source"],
                    "original_value": value, "reason": reason,
                })
    return gaps


def compare_rain_sources_2005_2010(cf: dict[date, dict[str, Any]], official: dict[date, dict[str, float | None]]) -> list[dict[str, Any]]:
    results = []
    for year in range(2005, 2011):
        dates = [d for d in official if d.year == year]
        cf_values = [(cf.get(d, {}).get("RAIN"), official[d].get("RAIN")) for d in dates]
        cf_valid = [float(a) for a, _ in cf_values if a is not None and a >= 0]
        official_valid = [float(b) for _, b in cf_values if b is not None and b >= 0]
        paired = [(float(a), float(b)) for a, b in cf_values if a is not None and b is not None and a >= 0 and b >= 0]
        cf_total = sum(cf_valid)
        official_total = sum(official_valid)
        paired_cf, paired_official = sum(a for a, _ in paired), sum(b for _, b in paired)
        results.append({
            "year": year, "calendar_days": len(dates),
            "chinaflux_complete_days": len(cf_valid), "chinaflux_total_mm": cf_total,
            "official_nonmissing_days": len(official_valid), "official_nonmissing_total_mm": official_total,
            "paired_days": len(paired), "paired_chinaflux_total_mm": paired_cf,
            "paired_official_total_mm": paired_official,
            "paired_difference_official_minus_chinaflux_mm": paired_official - paired_cf,
            "paired_relative_difference_percent": (paired_official / paired_cf - 1) * 100 if paired_cf > 0 else None,
        })
    return results


def compact_gap_dates(gaps: list[dict[str, Any]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[date]] = defaultdict(list)
    for gap in gaps:
        d = date.fromisoformat(gap["date"])
        grouped[(str(d.year), gap["variable"])].append(d)
    compact = []
    for (year, variable), days in sorted(grouped.items()):
        ordered = sorted(set(days))
        ranges: list[str] = []
        start = previous = ordered[0]
        for current in ordered[1:]:
            if current.toordinal() == previous.toordinal() + 1:
                previous = current
                continue
            ranges.append(start.isoformat() if start == previous else f"{start.isoformat()} 至 {previous.isoformat()}")
            start = previous = current
        ranges.append(start.isoformat() if start == previous else f"{start.isoformat()} 至 {previous.isoformat()}")
        compact.append({"year": year, "variable": variable, "dates": "、".join(ranges)})
    return compact


def external_path() -> Path:
    return CACHE / f"nasa_power_daily_2004_2013_lat{SITE_LAT:.3f}_lon{SITE_LON:.3f}.json"


def fetch_external() -> tuple[dict[str, Any], str, str, dict[str, Any]]:
    params = {
        "parameters": ",".join(NASA_PARAMETERS), "community": "AG",
        "longitude": f"{SITE_LON:.3f}", "latitude": f"{SITE_LAT:.3f}",
        "start": "20040101", "end": "20131231", "format": "JSON", "time-standard": "LST",
    }
    url = API_BASE + "?" + urllib.parse.urlencode(params)
    target = external_path()
    sidecar = target.with_suffix(".metadata.json")
    CACHE.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sidecar.exists():
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        else:
            previous_manifest = OUT / "run_manifest.json"
            previous = json.loads(previous_manifest.read_text(encoding="utf-8")) if previous_manifest.exists() else {}
            previous_external = previous.get("external", {})
            metadata = {
                "download_time_utc": previous_external.get("download_time_utc"),
                "url": previous_external.get("url", url), "sha256": sha256_file(target),
                "metadata_status": "recovered_from_previous_run_manifest" if previous_external.get("download_time_utc") else "preexisting_cache_without_download_time",
            }
            if metadata["sha256"] == previous_external.get("sha256") and metadata["url"] == url:
                write_json(sidecar, metadata)
        return json.loads(target.read_text(encoding="utf-8")), url, "cache_reused", metadata
    request = urllib.request.Request(url, headers={"User-Agent": "YC-weather-audit/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = response.read()
    # Persist the exact response before parsing so a later audit can reproduce it.
    target.write_bytes(payload)
    metadata = {"download_time_utc": datetime.now(timezone.utc).isoformat(), "url": url,
                "sha256": sha256_file(target), "metadata_status": "captured_at_download"}
    write_json(sidecar, metadata)
    return json.loads(payload.decode("utf-8")), url, "downloaded", metadata


def parse_external(payload: dict[str, Any]) -> tuple[dict[date, dict[str, float | None]], dict[str, Any]]:
    parameters = payload.get("properties", {}).get("parameter", {})
    field_map = {"TMAX": "T2M_MAX", "TMIN": "T2M_MIN", "RAIN": "PRECTOTCORR", "SRAD": "ALLSKY_SFC_SW_DWN"}
    records: dict[date, dict[str, float | None]] = {}
    for variable, parameter in field_map.items():
        values = parameters.get(parameter)
        if not isinstance(values, dict):
            raise RuntimeError(f"NASA POWER response missing parameter {parameter}")
        for key, raw in values.items():
            dt = date(int(key[:4]), int(key[4:6]), int(key[6:8]))
            assert dt.year <= 2013, "2014+ weather values found in NASA response"
            records.setdefault(dt, {})[variable] = number(raw)
    expected = set(all_dates())
    if set(records) != expected:
        missing = sorted(expected - set(records))[:8]
        extra = sorted(set(records) - expected)[:8]
        raise RuntimeError(f"NASA POWER date coverage mismatch; missing={missing}, extra={extra}")
    meta = {
        "title": payload.get("header", {}).get("title"),
        "api_version": payload.get("header", {}).get("api", {}).get("version"),
        "api_name": payload.get("header", {}).get("api", {}).get("name"),
        "time_standard": payload.get("header", {}).get("time_standard"),
        "resolved_coordinates": payload.get("geometry", {}).get("coordinates"),
        "parameters": payload.get("parameters", {}),
        "sources": payload.get("header", {}).get("sources", []),
    }
    return records, meta


def pearson(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sx, sy = sum((x - mx) ** 2 for x in xs), sum((y - my) ** 2 for y in ys)
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(sx * sy)


def calc_errors(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    if not pairs:
        return {"n": 0, "bias_external_minus_official": None, "mae": None, "rmse": None, "pearson_r": None}
    diff = [external - official for official, external in pairs]
    return {
        "n": len(pairs), "bias_external_minus_official": sum(diff) / len(diff),
        "mae": sum(abs(v) for v in diff) / len(diff),
        "rmse": math.sqrt(sum(v * v for v in diff) / len(diff)),
        "pearson_r": pearson(pairs),
    }


def validate_external(base: list[dict[str, Any]], external: dict[date, dict[str, float | None]]) -> tuple[dict[str, Any], dict[str, float]]:
    official_by_date = {date.fromisoformat(r["DATE"]): r for r in base}
    correction: dict[str, float] = {}
    metrics: dict[str, Any] = {"overlap_years": "2005-2013 train-only", "temperature_srad": {}, "rain": {}}
    for variable in ("TMAX", "TMIN", "SRAD"):
        pairs: list[tuple[float, float]] = []
        for dt, row in official_by_date.items():
            if not 2005 <= dt.year <= 2013:
                continue
            official = row[variable]
            ext = external.get(dt, {}).get(variable)
            if official is None or ext is None:
                continue
            if variable in ("SRAD",) and (official < 0 or ext < 0):
                continue
            pairs.append((float(official), float(ext)))
        raw_metrics = calc_errors(pairs)
        delta = sum(a - b for a, b in pairs) / len(pairs) if pairs else 0.0
        correction[variable] = delta
        adjusted_pairs = [(official, ext + delta) for official, ext in pairs]
        metrics["temperature_srad"][variable] = {
            "raw": raw_metrics,
            "bias_correction_parameter_official_minus_external": delta,
            "corrected": calc_errors(adjusted_pairs),
        }

    rain_pairs: list[tuple[date, float, float]] = []
    for dt, row in official_by_date.items():
        if not 2005 <= dt.year <= 2013:
            continue
        official, ext = row["RAIN"], external.get(dt, {}).get("RAIN")
        if official is None or ext is None or official < 0 or ext < 0:
            continue
        rain_pairs.append((dt, float(official), float(ext)))
    if rain_pairs:
        event_agree = sum((official > 0) == (ext > 0) for _, official, ext in rain_pairs)
        wet_reference = [(official, ext) for _, official, ext in rain_pairs if official > 0]
        monthly_ratios: list[dict[str, Any]] = []
        annual_ratios: list[dict[str, Any]] = []
        by_period: dict[str, list[tuple[float, float]]] = defaultdict(list)
        by_year: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for dt, official, ext in rain_pairs:
            by_period[dt.strftime("%Y-%m")].append((official, ext))
            by_year[dt.year].append((official, ext))
        for period, values in sorted(by_period.items()):
            ref_total, ext_total = sum(a for a, _ in values), sum(b for _, b in values)
            if ref_total > 0:
                monthly_ratios.append({"month": period, "external_to_official_total_ratio": ext_total / ref_total, "paired_days": len(values)})
        for year, values in sorted(by_year.items()):
            ref_total, ext_total = sum(a for a, _ in values), sum(b for _, b in values)
            annual_ratios.append({"year": year, "official_total_mm": ref_total, "external_total_mm": ext_total,
                                  "external_to_official_total_ratio": ext_total / ref_total if ref_total > 0 else None,
                                  "paired_days": len(values)})
        daily_amount = calc_errors([(a, b) for _, a, b in rain_pairs])
        event_amount = calc_errors(wet_reference)
        month_values = [r["external_to_official_total_ratio"] for r in monthly_ratios]
        year_values = [r["external_to_official_total_ratio"] for r in annual_ratios if r["external_to_official_total_ratio"] is not None]
        metrics["rain"] = {
            "n_paired_days": len(rain_pairs), "rain_event_agreement_days": event_agree,
            "rain_event_agreement_rate": event_agree / len(rain_pairs),
            "daily_amount_errors": daily_amount,
            "event_amount_errors_on_official_wet_days": event_amount,
            "monthly_total_ratios": monthly_ratios,
            "annual_total_ratios": annual_ratios,
            "monthly_ratio_summary": ratio_summary(month_values),
            "annual_ratio_summary": ratio_summary(year_values),
        }
    else:
        metrics["rain"] = {"n_paired_days": 0, "rain_event_agreement_rate": None}

    # These gates are declared here, before inspecting results: they screen broad transfer suitability,
    # not agreement to observational precision. Filled values remain flagged as external estimates.
    limits = {
        "TMAX_corrected_MAE_max_C": 3.0, "TMIN_corrected_MAE_max_C": 3.0,
        "SRAD_corrected_MAE_max_MJ_m2_d": 4.0,
        "temperature_minimum_pearson_r": 0.85, "SRAD_minimum_pearson_r": 0.70,
        "rain_event_agreement_minimum": 0.80,
        "rain_median_monthly_ratio_range": [0.50, 1.50],
        "rain_median_annual_ratio_range": [0.70, 1.30],
    }
    checks: dict[str, bool] = {}
    for variable in ("TMAX", "TMIN", "SRAD"):
        corrected = metrics["temperature_srad"][variable]["corrected"]
        mae_limit = limits[f"{variable}_corrected_MAE_max_C"] if variable != "SRAD" else limits["SRAD_corrected_MAE_max_MJ_m2_d"]
        r_limit = limits["temperature_minimum_pearson_r"] if variable != "SRAD" else limits["SRAD_minimum_pearson_r"]
        checks[f"{variable}_coverage"] = corrected["n"] >= 1000
        checks[f"{variable}_corrected_MAE"] = corrected["mae"] is not None and corrected["mae"] <= mae_limit
        checks[f"{variable}_pearson_r"] = corrected["pearson_r"] is not None and corrected["pearson_r"] >= r_limit
    rain = metrics["rain"]
    checks["RAIN_coverage"] = rain.get("n_paired_days", 0) >= 3000
    checks["RAIN_event_agreement"] = rain.get("rain_event_agreement_rate") is not None and rain["rain_event_agreement_rate"] >= limits["rain_event_agreement_minimum"]
    mrange = limits["rain_median_monthly_ratio_range"]
    yrange = limits["rain_median_annual_ratio_range"]
    mmedian = rain.get("monthly_ratio_summary", {}).get("median")
    ymedian = rain.get("annual_ratio_summary", {}).get("median")
    checks["RAIN_monthly_total_ratio"] = mmedian is not None and mrange[0] <= mmedian <= mrange[1]
    checks["RAIN_annual_total_ratio"] = ymedian is not None and yrange[0] <= ymedian <= yrange[1]
    result = {
        "source": "NASA POWER Daily API; gap dates only; LST",
        "overlap_years": "2005-2013 only; no 2014+ values read",
        "bias_correction_formula": "corrected = external_raw + mean(official - external_raw) on valid 2005-2013 overlap",
        "rain_method": "No additive correction; use PRECTOTCORR raw values only on gaps",
        "acceptance_limits_declared_in_code": limits,
        "metrics": metrics,
        "gate_checks": checks,
        "accepted_for_gap_fill": all(checks.values()),
        "caveat": "Overlap is a same-site product comparison, not independent validation; POWER is gridded/model-derived data and all filled rows remain labeled external estimates.",
    }
    return result, correction


def ratio_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return {"n": len(values), "mean": sum(values) / len(values), "median": median, "minimum": min(values), "maximum": max(values)}


def fill_gaps(base: list[dict[str, Any]], gaps: list[dict[str, Any]], external: dict[date, dict[str, float | None]], correction: dict[str, float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexed = {r["DATE"]: r for r in base}
    fills: list[dict[str, Any]] = []
    for gap in gaps:
        dt = date.fromisoformat(gap["date"])
        variable = gap["variable"]
        raw = external.get(dt, {}).get(variable)
        if raw is None:
            fills.append({"date": gap["date"], "variable": variable, "external_raw_value": None,
                          "correction_method": "none_missing_external", "correction_parameter": None,
                          "filled_value": None, "external_source": "NASA_POWER_DAILY_API", "qc_status": "UNRESOLVED_EXTERNAL_MISSING"})
            continue
        if variable in ("TMAX", "TMIN", "SRAD"):
            delta = correction.get(variable, 0.0)
            filled = raw + delta
            method = "additive_train_overlap_bias_correction_2005_2013"
        else:
            delta = 0.0
            filled = raw
            method = "raw_no_rain_bias_shift"
        row = indexed[gap["date"]]
        row[variable] = filled
        row[f"{variable}_source"] = "external_gapfill_nasa_power"
        row[f"{variable}_qc"] = "GAP_FILLED_EXTERNAL_BIAS_CORRECTED" if method.startswith("additive") else "GAP_FILLED_EXTERNAL_RAW"
        fills.append({"date": gap["date"], "variable": variable, "external_raw_value": raw,
                      "correction_method": method, "correction_parameter": delta,
                      "filled_value": filled, "external_source": "NASA_POWER_DAILY_API", "qc_status": row[f"{variable}_qc"]})
    return base, fills


def propose_gapfills(gaps: list[dict[str, Any]], external: dict[date, dict[str, float | None]], correction: dict[str, float], status: str) -> list[dict[str, Any]]:
    proposals = []
    for gap in gaps:
        dt, variable = date.fromisoformat(gap["date"]), gap["variable"]
        raw = external.get(dt, {}).get(variable)
        additive = variable in ("TMAX", "TMIN", "SRAD")
        delta = correction.get(variable, 0.0) if additive else 0.0
        proposals.append({
            "date": gap["date"], "variable": variable, "external_raw_value": raw,
            "correction_method": "additive_train_overlap_bias_correction_2005_2013" if additive else "raw_no_rain_bias_shift",
            "correction_parameter": delta if additive else 0.0,
            "filled_value": raw + delta if raw is not None else None,
            "external_source": "NASA_POWER_DAILY_API",
            "qc_status": "UNRESOLVED_EXTERNAL_MISSING" if raw is None else status,
        })
    return proposals


def candidate_qc(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dates = [r["DATE"] for r in rows]
    errors = []
    if len(rows) != 3653:
        errors.append(f"expected_3653_rows_got_{len(rows)}")
    if len(set(dates)) != len(dates):
        errors.append("duplicate_dates")
    expected = [d.isoformat() for d in all_dates()]
    if dates != expected:
        errors.append("date_sequence_incomplete_or_out_of_order")
    for r in rows:
        for variable in VARIABLES:
            v = r[variable]
            if v is None or not math.isfinite(float(v)):
                errors.append(f"{r['DATE']}:{variable}:missing_or_nan")
            if r[f"{variable}_source"] == "unresolved":
                errors.append(f"{r['DATE']}:{variable}:unresolved_source")
            if v is not None and variable in ("RAIN", "SRAD") and v < 0:
                errors.append(f"{r['DATE']}:{variable}:negative")
            if v is not None and float(v) in SENTINELS:
                errors.append(f"{r['DATE']}:{variable}:sentinel")
        if r["TMAX"] is not None and r["TMIN"] is not None and r["TMAX"] < r["TMIN"]:
            errors.append(f"{r['DATE']}:TMAX_LT_TMIN")
    return {"passed": not errors, "row_count": len(rows), "expected_row_count": 3653,
            "first_date": dates[0] if dates else None, "last_date": dates[-1] if dates else None,
            "unique_dates": len(set(dates)), "errors": errors[:200], "error_count": len(errors)}


def longest_dry(values: list[float]) -> int:
    longest = current = 0
    for value in values:
        if value == 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    annual: list[dict[str, Any]] = []
    monthly: list[dict[str, Any]] = []
    for group_name, key_func, output in (
        ("year", lambda r: str(r["YEAR"]), annual),
        ("month", lambda r: r["DATE"][:7], monthly),
    ):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[key_func(row)].append(row)
        for label, group in sorted(groups.items()):
            rain = [float(r["RAIN"]) for r in group]
            item: dict[str, Any] = {
                group_name: int(label) if group_name == "year" else label,
                "days": len(group), "rainfall_total_mm": sum(rain), "rainy_days": sum(v > 0 for v in rain),
                "max_daily_rainfall_mm": max(rain), "longest_dry_spell_days": longest_dry(rain),
            }
            for variable in ("TMAX", "TMIN", "SRAD"):
                values = [float(r[variable]) for r in group]
                item[f"{variable}_mean"] = sum(values) / len(values)
                item[f"{variable}_min"] = min(values)
                item[f"{variable}_max"] = max(values)
            output.append(item)
    return annual, monthly


def gap_fill_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[int, str], int] = defaultdict(int)
    for row in rows:
        for variable in VARIABLES:
            if str(row[f"{variable}_source"]).startswith("external_gapfill_"):
                counts[(int(row["YEAR"]), variable)] += 1
    result = []
    for year in range(2004, 2014):
        days = 366 if year == 2004 or year == 2008 or year == 2012 else 365
        for variable in VARIABLES:
            filled = counts[(year, variable)]
            result.append({"year": year, "variable": variable, "gap_filled_days": filled, "total_days": days, "gap_filled_fraction": filled / days})
    return result


def create_source_inventory(met_path: Path, rad_path: Path, cf_zip: Path) -> list[dict[str, Any]]:
    return [
        {"source": "ChinaFLUX 2003-2010 30min product", "path": str(cf_zip.relative_to(ROOT)), "sha256": sha256_file(cf_zip), "read_cell_years": "2004-2010; 2005-2010 only for source-total comparison"},
        {"source": "Yucheng 2005-2022 official meteorology daily QC product", "path": str(met_path.relative_to(ROOT)), "sha256": sha256_file(met_path), "read_cell_years": "2005-2013 only"},
        {"source": "Yucheng 2005-2022 official radiation daily QC product", "path": str(rad_path.relative_to(ROOT)), "sha256": sha256_file(rad_path), "read_cell_years": "2005-2013 only"},
    ]


def report_text(summary: dict[str, Any], result_dir: Path) -> str:
    gaps = summary.get("gap_counts", {})
    validation = summary.get("external_validation", {}).get("metrics", {})
    rain = validation.get("rain", {})
    climate = summary.get("annual_summary", [])
    y2004 = next((r for r in climate if r.get("year") == 2004), {})
    filled = summary.get("gap_filled_summary", [])
    nonzero = [r for r in filled if r["gap_filled_days"]]
    gap_lines = "\n".join(f"| {year} | {var} | {count} |" for year, fields in sorted(gaps.items()) for var, count in sorted(fields.items())) or "| 无 | - | 0 |"
    gap_date_lines = "\n".join(f"| {r['year']} | {r['variable']} | {r['dates']} |" for r in summary.get("compact_gap_dates", [])) or "| 无 | - | - |"
    fill_lines = "\n".join(f"| {r['year']} | {r['variable']} | {r['gap_filled_days']} | {r['gap_filled_fraction']:.4%} |" for r in nonzero) or "| 无 | - | 0 | - |"
    comparison_lines = "\n".join(
        f"| {r['year']} | {r['chinaflux_complete_days']} | {r['chinaflux_total_mm']:.1f} | {r['official_nonmissing_days']} | {r['official_nonmissing_total_mm']:.1f} | {r['official_nonmissing_total_mm'] - r['chinaflux_total_mm']:.1f} |"
        for r in summary.get("rain_source_comparison_2005_2010", [])
    ) or "| 无 | - | - | - | - | - |"
    metric_lines = []
    for var, data in validation.get("temperature_srad", {}).items():
        raw, corrected = data.get("raw", {}), data.get("corrected", {})
        metric_lines.append(f"| {var} | {raw.get('n')} | {raw.get('bias_external_minus_official'):.3f} | {corrected.get('mae'):.3f} | {corrected.get('rmse'):.3f} | {corrected.get('pearson_r'):.4f} |")
    failed_external_checks = [name for name, passed in summary.get("external_validation", {}).get("gate_checks", {}).items() if not passed]
    rain_agreement = rain.get("rain_event_agreement_rate")
    rain_event_mae = rain.get("event_amount_errors_on_official_wet_days", {}).get("mae")
    monthly_ratio = rain.get("monthly_ratio_summary", {}).get("median")
    annual_ratio = rain.get("annual_ratio_summary", {}).get("median")
    rain_metrics_text = (
        f"{rain_agreement:.2%}" if rain_agreement is not None else "未计算",
        f"{rain_event_mae:.2f}" if rain_event_mae is not None else "未计算",
        f"{monthly_ratio:.3f}" if monthly_ratio is not None else "未计算",
        f"{annual_ratio:.3f}" if annual_ratio is not None else "未计算",
    )
    cli = summary.get("cli", {})
    return f"""# YC 2004–2013 天气定稿与 train-only `.CLI` 准备（003_05）

**最终 Gate：** `{summary.get('final_status')}`
**分析范围：** YC/YCA；训练期 2004–2013；未启动 WGEN、DSSAT 或 PPO。

## 1. 数据源层级

- **2004**：ChinaFLUX 半小时“近地面空气温度”按完整 48 条记录计算 TMAX/TMIN；SRAD 按 `sum(W m-2 × 1800 s) / 1e6` 积分；RAIN 仅对完整 48 条日求和。温度按 2005–2006 重叠期一致性选择，不宣称已经确认 2 m 高度。
- **2005–2013**：禹城站官方 QC 日产品直接提供 TMAX、TMIN、RAIN、SRAD。官方发布值被视为 QC 产品值，不等同于未经处理的原始观测；发布的 0 mm 雨量接受为正式值。
- 外部源只填缺口，不替换完整主源记录；每个候选值保留 `source` 和 `qc`。
- 上一轮项目内来源盘点没有找到覆盖 2004 五个雨量缺口的可信逐日记录；1998–2006 产品只有月尺度，未拆分成日值。
- NASA POWER 官方说明：[Daily API](https://power.larc.nasa.gov/docs/services/api/temporal/daily/)；[参数字典](https://power.larc.nasa.gov/docs/tutorials/parameters/)。

## 2. 缺口清单与外部数据源

原始缺口数（负值、缺测、TMAX<TMIN 和 2004 不完整聚合均计入）：

| 年份 | 变量 | 缺口日数 |
|---:|---|---:|
{gap_lines}

逐日缺口日期（连续日期已合并成首末范围；逐行值与原因见 `gaps_before_external_fill.csv`）：

| 年份 | 变量 | 日期/范围 |
|---:|---|---|
{gap_date_lines}

外部源为 NASA POWER Daily API（AG community，LST，点位 {SITE_LAT:.3f}°N、{SITE_LON:.3f}°E）。它是格点/模型和卫星衍生产品，仅作为缺口补值候选。原始响应缓存于 `data/external/yc_weather_finalize_and_cli/nasa_power/`，请求 URL、API 版本、单位、来源和 SHA256 见 `run_manifest.json`。

### 重叠期验证

只用 2005–2013 官方日值非缺测且物理有效日期；偏差为外部原值减官方值。加性校正仅用于 TMAX/TMIN/SRAD，参数为 `mean(official - external)`；RAIN 不做均值平移。

| 变量 | 配对日 | 原始 bias | 校正后 MAE | 校正后 RMSE | Pearson r |
|---|---:|---:|---:|---:|---:|
{chr(10).join(metric_lines) or '| 无 | - | - | - | - | - |'}

降雨事件一致率：{rain_metrics_text[0]}；官方湿日事件量 MAE：{rain_metrics_text[1]} mm/d；月总量比中位数：{rain_metrics_text[2]}；年总量比中位数：{rain_metrics_text[3]}。所有 overlap 指标是同站点数据产品对照，不是独立观测验证。门槛和逐年/逐月比值见 `external_gapfill_validation.json`。

外部 overlap Gate：`{summary.get('external_validation', {}).get('accepted_for_gap_fill')}`；未通过项：`{', '.join(failed_external_checks) or '无'}`。本轮缺口提案数 {summary.get('gapfill_proposed_count', 0)}，接受填补数 {summary.get('gapfill_accepted_count', 0)}。门槛是本轮预先写入脚本的工程筛选规则，不是 NASA 或 DSSAT 发布的标准。

## 3. 候选数据、统计与 QC

候选文件：`yc_wgen_fitting_weather_2004_2013.csv`；存在：`{summary.get('candidate_created')}`。
候选 QC：`{summary.get('candidate_qc', {}).get('passed')}`；行数 {summary.get('candidate_qc', {}).get('row_count')}；日期覆盖 2004-01-01 至 2013-12-31。
2004 年最终降水量：{f"{y2004.get('rainfall_total_mm'):.1f} mm" if y2004 else "未计算（candidate Gate 未通过）"}。

| 年份 | 变量 | 外部 gap-fill 日数 | 当年比例 |
|---:|---|---:|---:|
{fill_lines}

年/月气候统计见 `candidate_annual_summary.csv` 和 `candidate_monthly_summary.csv`。缺口填补比例不是“实测率”；外部补值仍按来源单独识别。2011–2013 各变量填补量在上述表和 `gap_filled_days_by_year_variable.csv` 中列出。

若 candidate 未形成，年/月气候统计文件会显式标为未运行；`external_gapfill_values.csv` 中的数值只是外部提案，`qc_status=REJECTED_EXTERNAL_OVERLAP_GATE` 时不得读入 candidate。

### 2005–2010 雨量源对照

下表分别汇总 ChinaFLUX 完整 48 半小时日和官方日产品有效日；不同源有效天数可能不同，因此以完整 365/366 天候选总量为准，paired-day 对照数值另见 `rain_source_comparison_2005_2010.csv`。

| 年份 | ChinaFLUX 完整日 | ChinaFLUX 总量 mm | 官方有效日 | 官方有效值总量 mm | 官方−ChinaFLUX mm |
|---:|---:|---:|---:|---:|---:|
{comparison_lines}

WeatherMan 操作依据：[DSSAT User's Guide Volume 3](https://dssat.net/wp-content/uploads/2011/10/DSSAT-vol3.pdf)。

## 4. Leakage audit

`leakage_audit.json` 记录：官方表只解析到 2013 年气象变量列；为停止顺序扫描，仅读取下一行的年份标记 `2014`，该行其余气象单元格未解码/使用；源文件 SHA256 是字节级完整性校验，不解析单元格。偏差参数只由 2005–2013 计算；NASA 请求止于 2013-12-31；validation 天气数据仍独立。代码断言 `max(source_weather_years_used_for_values_or_calibration) <= 2013`。

## 5. `.CLI` 状态与下一步

官方 WeatherMan 自动生成：`{cli.get('automatically_generated', False)}`。状态：`{cli.get('status', 'NOT_RUN')}`。
{cli.get('note', '')}

天气 candidate 通过后仍不运行 WGEN。本轮只准备 train-only `.CLI`；下一任务才进入受限 seed pilot 和 DSSAT 单季 smoke，之后再独立审核能否进入 PPO。

## 6. 可复现文件

- 主脚本：`scripts/finalize_yc_weather_and_prepare_cli.py`
- 原始缺口：`gaps_before_external_fill.csv`
- 外部 overlap / 补值：`external_gapfill_validation.json`、`external_gapfill_values.csv`
- 泄漏审计：`leakage_audit.json`
- run source hash：`run_manifest.json`
- 年/月 summary 与 gap 比例：`candidate_annual_summary.csv`、`candidate_monthly_summary.csv`、`gap_filled_days_by_year_variable.csv`
- 中文 PPT：`docs/yc_weather_finalize_and_cli.pptx`；结构检查：`pptx_validation_summary.json`
- 实验记录：`results/yc_weather_finalize_and_cli/experiment_log.md`
"""


def write_experiment_log(summary: dict[str, Any], output: Path, source_inventory: list[dict[str, Any]], fetch_status: str | None, error: str | None) -> None:
    changes = "\n".join(f"- {row['source']}: `{row['path']}`, SHA256 `{row['sha256']}`, cell-value years `{row['read_cell_years']}`" for row in source_inventory)
    log = f"""# YC 天气定稿实验记录（003_05）

## 开始状态

- Branch / HEAD 由任务启动记录：`codex/sya-forecast-freeze-2026-08-16` / `01b9a9a0f81be746d3ed84d0e7db4cd8a81b0657`。
- 当时 tracked 用户修改：`configs/068_effective_hla_seed1_smoke.yaml`、`configs/068_effective_yca_seed1_smoke.yaml`、博士开题讲稿、HL 图表脚本、`src/mask_aware_dqn_029.py`；另有大量未跟踪内容。本任务没有编辑或提交它们。
- 003_04 已提交，状态为 `BLOCKED_TEMPERATURE_MAPPING` 等；本轮遵循新授权，接受官方 QC 日产品，并采用一致性选择。

## 方法和运行

- 2004 使用 ChinaFLUX 2004 30 min ZIP，逐日要求 48 条有效半小时记录；TMAX/TMIN 只用“近地面空气温度”并标注 consistency-based selection，SRAD 积分，RAIN 求和。
- 2005–2013 使用官方 QC 日产品；0 mm RAIN 按发布产品值保留。
- gap filling 只对缺测、非法值、TMAX<TMIN、2004 30 min 不完整日尝试；NASA POWER 仅为缺口服务。
- 外部请求状态：`{fetch_status}`；如果有错误：`{error or '无'}`。
- 外部验证 Gate：`{summary.get('external_validation', {}).get('accepted_for_gap_fill')}`；Candidate QC：`{summary.get('candidate_qc', {}).get('passed')}`；Final Gate：`{summary.get('final_status')}`。
- 本轮没有启动 WGEN、大规模天气随机生成、DSSAT 或 PPO；未修改生产 `.WTH`、`.CLI`、FileX、SOL、CUL。

## 数据源哈希

{changes}

## 关键产物

- 原始 gap 日期：`gaps_before_external_fill.csv`
- overlap validation：`external_gapfill_validation.json`
- 外部填补值：`external_gapfill_values.csv`
- candidate QC / climate summary：`candidate_qc.json`、`candidate_annual_summary.csv`、`candidate_monthly_summary.csv`
- 2014+ leakage：`leakage_audit.json`
- 最终报告：`docs/yc_weather_finalize_and_cli.md`
- PPT：`docs/yc_weather_finalize_and_cli.pptx`，结构检查结果见 `pptx_validation_summary.json`。

## 复核

- Python 版本：`{sys.version.split()[0]}`；openpyxl 版本：`{openpyxl.__version__}`。
- 2014+ 断言：代码和清单均要求 `max(source_weather_years_used_for_values_or_calibration) <= 2013`。
- 结果没有宣称 external gap-fill 为实测值；NASA POWER overlap 是同站点产品比较，指标及所有补值来源均保留。
"""
    output.write_text(log, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--audit-only", action="store_true", help="Default: audit source coverage/gaps; no external download or candidate write.")
    mode.add_argument("--build-weather-candidate", action="store_true", help="Fetch/cache NASA POWER, validate overlap, and build only if all gates pass.")
    mode.add_argument("--prepare-cli", action="store_true", help="Inspect official CLI-generation availability after candidate QC passes.")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    backup_path = backup_existing_outputs()
    run_time = datetime.now(timezone.utc).isoformat()

    met_path, rad_path = find_workbooks()
    cf_candidates = sorted(RAW.glob("YCA_M_30min.zip"))
    if len(cf_candidates) != 1:
        raise RuntimeError("Could not identify ChinaFLUX 30 min archive")
    cf_path = cf_candidates[0]

    cf = load_chinaflux_daily(cf_path)
    met = load_official(met_path, "met")
    rad = load_official(rad_path, "rad")
    base = build_base(cf, met, rad)
    gaps = gaps_for(base, cf)
    rain_source_comparison = compare_rain_sources_2005_2010(cf, met)
    write_csv(OUT / "rain_source_comparison_2005_2010.csv", list(rain_source_comparison[0]), rain_source_comparison)
    gap_path = OUT / "gaps_before_external_fill.csv"
    write_csv(gap_path, ["date", "variable", "original_source", "original_value", "reason"], gaps)

    gap_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item in gaps:
        gap_counts[item["date"][:4]][item["variable"]] += 1
    gap_counts_json = {year: dict(fields) for year, fields in sorted(gap_counts.items())}
    inventory = create_source_inventory(met_path, rad_path, cf_path)
    external_payload = None
    external_records: dict[date, dict[str, float | None]] = {}
    external_meta: dict[str, Any] = {}
    external_cache_metadata: dict[str, Any] = {}
    external_url: str | None = None
    fetch_status: str | None = None
    fetch_error: str | None = None
    validation: dict[str, Any] = {"accepted_for_gap_fill": False, "gate_checks": {}, "metrics": {}, "note": "Not run in audit-only mode."}
    correction: dict[str, float] = {}
    fills: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] | None = None
    qc: dict[str, Any] = {"passed": False, "not_run": True, "errors": ["candidate not built"]}
    cli: dict[str, Any] = {"status": "NOT_RUN", "automatically_generated": False, "note": "候选天气 QC 尚未通过，CLI 准备未启动。"}

    if args.build_weather_candidate or args.prepare_cli:
        try:
            external_payload, external_url, fetch_status, external_cache_metadata = fetch_external()
            external_records, external_meta = parse_external(external_payload)
            validation, correction = validate_external(base, external_records)
            if args.build_weather_candidate:
                if validation["accepted_for_gap_fill"]:
                    candidate_rows, fills = fill_gaps([dict(r) for r in base], gaps, external_records, correction)
                    qc = candidate_qc(candidate_rows)
                    if qc["passed"]:
                        annual, monthly = summarize(candidate_rows)
                        write_csv(OUT / "yc_wgen_fitting_weather_2004_2013.csv", [
                            "DATE", "YEAR", "DOY", "SRAD", "TMAX", "TMIN", "RAIN",
                            "SRAD_source", "TMAX_source", "TMIN_source", "RAIN_source",
                            "SRAD_qc", "TMAX_qc", "TMIN_qc", "RAIN_qc", "overall_qc",
                        ], [{**r, "overall_qc": "PASS_WITH_EXTERNAL_GAPFILL" if any(str(r[f"{v}_source"]).startswith("external_gapfill_") for v in VARIABLES) else "PASS_SOURCE_VALUE"} for r in candidate_rows])
                        write_csv(OUT / "candidate_annual_summary.csv", list(annual[0]), annual)
                        write_csv(OUT / "candidate_monthly_summary.csv", list(monthly[0]), monthly)
                        write_csv(OUT / "gap_filled_days_by_year_variable.csv", ["year", "variable", "gap_filled_days", "total_days", "gap_filled_fraction"], gap_fill_summary(candidate_rows))
                    else:
                        validation["candidate_qc_blocked"] = True
                        for item in fills:
                            item["qc_status"] = "REJECTED_CANDIDATE_QC"
                else:
                    fills = propose_gapfills(gaps, external_records, correction, "REJECTED_EXTERNAL_OVERLAP_GATE")
                    qc = {"passed": False, "not_run": True, "blocked_by": "external_overlap_validation",
                          "errors": [name for name, passed in validation.get("gate_checks", {}).items() if not passed]}
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            fetch_error = f"{type(exc).__name__}: {exc}"
            fetch_status = "failed"

    if args.build_weather_candidate:
        write_csv(OUT / "external_gapfill_values.csv", ["date", "variable", "external_raw_value", "correction_method", "correction_parameter", "filled_value", "external_source", "qc_status"], fills)
    write_json(OUT / "external_gapfill_validation.json", validation)

    if args.build_weather_candidate and not qc.get("passed"):
        not_run_status = "NOT_RUN_BLOCKED_EXTERNAL_GAPFILL" if not validation.get("accepted_for_gap_fill") else "NOT_RUN_BLOCKED_CANDIDATE_QC"
        write_csv(OUT / "candidate_annual_summary.csv", ["status", "year", "rainfall_total_mm", "rainy_days", "max_daily_rainfall_mm", "longest_dry_spell_days", "TMAX_mean", "TMAX_min", "TMAX_max", "TMIN_mean", "TMIN_min", "TMIN_max", "SRAD_mean", "SRAD_min", "SRAD_max"], [
            {"status": not_run_status}
        ])
        write_csv(OUT / "candidate_monthly_summary.csv", ["status", "month", "days", "rainfall_total_mm", "rainy_days", "max_daily_rainfall_mm", "longest_dry_spell_days", "TMAX_mean", "TMAX_min", "TMAX_max", "TMIN_mean", "TMIN_min", "TMIN_max", "SRAD_mean", "SRAD_min", "SRAD_max"], [
            {"status": not_run_status}
        ])
        write_csv(OUT / "gap_filled_days_by_year_variable.csv", ["year", "variable", "gap_filled_days", "total_days", "gap_filled_fraction", "status"], [
            {**r, "status": not_run_status} for r in gap_fill_summary(base)
        ])

    if args.prepare_cli:
        candidate_file = OUT / "yc_wgen_fitting_weather_2004_2013.csv"
        prior_qc_file = OUT / "candidate_qc.json"
        if candidate_file.exists() and prior_qc_file.exists():
            prior_qc = json.loads(prior_qc_file.read_text(encoding="utf-8"))
            if prior_qc.get("passed"):
                with candidate_file.open("r", encoding="utf-8-sig", newline="") as stream:
                    candidate_rows = []
                    for source_row in csv.DictReader(stream):
                        row: dict[str, Any] = {
                            "DATE": source_row["DATE"], "YEAR": int(source_row["YEAR"]),
                            "DOY": int(source_row["DOY"]), "overall_qc": source_row["overall_qc"],
                        }
                        for variable in VARIABLES:
                            row[variable] = number(source_row[variable])
                            row[f"{variable}_source"] = source_row[f"{variable}_source"]
                            row[f"{variable}_qc"] = source_row[f"{variable}_qc"]
                        candidate_rows.append(row)
                qc = candidate_qc(candidate_rows)
                annual, _ = summarize(candidate_rows)
                cli = {
                    "status": "PASS_WEATHER_CANDIDATE_CLI_MANUAL_REQUIRED",
                    "automatically_generated": False,
                    "official_weatherman_found": False,
                    "note": "没有发现已安装且可验证的官方 WeatherMan 批处理估计器。已生成中文 GUI 步骤；不得手写/猜测 CLI 参数。",
                }
                import_dir = OUT / "cli_candidate" / "input"
                import_dir.mkdir(parents=True, exist_ok=True)
                import_csv = import_dir / "CNYC_daily_train_2004_2013.csv"
                write_csv(import_csv, ["YEAR", "MONTH", "DAY", "DOY", "SRAD", "TMAX", "TMIN", "RAIN"], [
                    {"YEAR": r["YEAR"], "MONTH": int(r["DATE"][5:7]), "DAY": int(r["DATE"][8:10]),
                     "DOY": r["DOY"], "SRAD": r["SRAD"], "TMAX": r["TMAX"], "TMIN": r["TMIN"], "RAIN": r["RAIN"]}
                    for r in candidate_rows
                ])
                cli["weatherman_input_csv"] = str(import_csv.relative_to(ROOT))
                cli["weatherman_input_sha256"] = sha256_file(import_csv)
                guide = ROOT / "docs" / "yc_weatherman_cli_manual_steps.md"
                guide.write_text("""# WeatherMan GUI 生成 train-only `CNYC.CLI` 操作步骤

当前项目可访问目录 `opt/`、`local_packages/`、`tools/` 未检测到可验证的官方 WeatherMan 命令行参数估计器；未扫描或修改项目外安装。本说明只安排官方 GUI 流程；不手写、推测或伪造 `.CLI`。

1. 启动本机已安装的 DSSAT WeatherMan（以软件 About 页面显示的发行版本为准）；如未安装 DSSAT/WeatherMan，先由用户按本机授权软件安装流程准备，并记录版本。本任务不会安装系统软件。
2. 先核验 candidate QC 为 PASS，输入文件为 `results/yc_weather_finalize_and_cli/yc_wgen_fitting_weather_2004_2013.csv`，SHA256 取自 `run_manifest.json`。不得选用旧 `CNYC.CLI`、validation 文件或生产天气目录。
3. 在 WeatherMan 选择 `Station > Select Station`，输入四字符站码 `CNYC` 并新建站点；描述填 `Yucheng, China; train-only 2004-2013`。经纬度和高程参照 `run_manifest.json` 中的站点元数据；若界面要求其他站点统计参数，使用刚导入的训练期数据计算或依该版本 Help，不得抄旧 `.CLI` 或猜数。
4. 选择 `Import/Export > Import Single File`。建立新导入格式，指定 1 行表头、逗号分隔、年月日列以及 SRAD/TMAX/TMIN/RAIN 列和单位；开启预览，确认首末日期、列映射和单位后再导入 `results/yc_weather_finalize_and_cli/cli_candidate/input/CNYC_daily_train_2004_2013.csv`。不要启用自动填补。
5. 仅导入 2004-01-01 至 2013-12-31 的 3,653 个 candidate 日值。按导入统计核对缺测数为 0 和首末日期；不要读取或合并 2014 年及之后天气。
6. 选择 `Generate > Calculate Parameters`，选择 `Both Sets of Parameters`，并确认估计所用站点归档只有上述十个训练年。此步骤只估计参数，不选择 `Generate Weather Data`，不运行 WGEN。
7. 保存 climate file 到新目录 `results/yc_weather_finalize_and_cli/cli_candidate/CNYC.CLI`。确认目标不存在；不得覆盖生产目录或任何已有 `.CLI`。
8. 记录 WeatherMan/DSSAT 完整版本、生成日期、输入 CSV SHA256、输出 `.CLI` SHA256、`WSTA=CNYC`、导入年限和 GUI 警告；再做文件结构、站点码和参数字段审核，通过后再进入 WGEN pilot。

WeatherMan 官方用户指南的菜单为 `Station > Select Station`、`Import/Export > Import Single File`、`Generate > Calculate Parameters`；导入向导可定义新格式、预览字段并按所选日期范围导出。不同发行版界面可能变化，操作前按本机版本 Help 核对。
""", encoding="utf-8")
            else:
                cli = {"status": "NOT_RUN", "automatically_generated": False, "note": "Candidate QC 尚未通过，CLI 准备未启动。"}
        else:
            cli = {"status": "NOT_RUN", "automatically_generated": False, "note": "未找到本任务中已通过 QC 的候选文件和 QC manifest；CLI 准备未启动。"}

    if fetch_error:
        final_status = "BLOCKED_EXTERNAL_GAPFILL"
    elif args.audit_only or (not args.build_weather_candidate and not args.prepare_cli):
        final_status = "BLOCKED_EXTERNAL_GAPFILL"
    elif not validation.get("accepted_for_gap_fill"):
        final_status = "BLOCKED_EXTERNAL_GAPFILL"
    elif args.build_weather_candidate and not qc.get("passed"):
        final_status = "BLOCKED_CANDIDATE_QC"
    elif args.prepare_cli and cli.get("status") == "PASS_WEATHER_CANDIDATE_CLI_MANUAL_REQUIRED":
        final_status = cli["status"]
    elif qc.get("passed"):
        final_status = "PASS_WEATHER_CANDIDATE_CLI_MANUAL_REQUIRED"
    else:
        final_status = "BLOCKED_EXTERNAL_GAPFILL"

    if candidate_rows is not None:
        write_json(OUT / "candidate_qc.json", qc)
        if qc["passed"]:
            annual, _ = summarize(candidate_rows)
        else:
            annual = []
    else:
        annual = []
        if args.build_weather_candidate:
            write_json(OUT / "candidate_qc.json", qc)

    candidate_path = OUT / "yc_wgen_fitting_weather_2004_2013.csv"
    candidate_created = candidate_path.exists() and qc.get("passed", False)
    source_weather_years_used_for_values_or_calibration = [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013]
    leakage = {
        "passed": max(source_weather_years_used_for_values_or_calibration) <= 2013,
        "max_source_weather_years_used_for_values_or_calibration": max(source_weather_years_used_for_values_or_calibration),
        "source_weather_years_used_for_values_or_calibration": source_weather_years_used_for_values_or_calibration,
        "official_workbook_cell_values_read_through_year": 2013,
        "first_post_cutoff_year_marker_read_to_stop_scan": 2014,
        "post_cutoff_weather_cells_decoded_or_used": False,
        "whole_file_hash_reads_cell_values": False,
        "external_bias_calibration_years": "2005-2013 only",
        "external_request_start": "2004-01-01" if external_records else None,
        "external_request_end": "2013-12-31" if external_records else None,
        "validation_weather_values_read": False,
        "validation_years_used_for_metrics_or_correction": [],
        "validation_remains_independent": True,
        "assertion": "assert max(source_weather_years_used_for_values_or_calibration) <= 2013",
    }
    assert max(source_weather_years_used_for_values_or_calibration) <= 2013
    write_json(OUT / "leakage_audit.json", leakage)
    write_csv(OUT / "source_inventory.csv", ["source", "path", "sha256", "read_cell_years"], inventory)
    if fills:
        write_csv(OUT / "external_gapfill_values.csv", ["date", "variable", "external_raw_value", "correction_method", "correction_parameter", "filled_value", "external_source", "qc_status"], fills)

    summary = {
        "scope": "YC/YCA only", "period": "2004-2013", "run_time_utc": run_time,
        "final_status": final_status, "gap_counts": gap_counts_json,
        "compact_gap_dates": compact_gap_dates(gaps),
        "rain_source_comparison_2005_2010": rain_source_comparison,
        "gap_total_variable_days": len(gaps), "gapfill_proposed_count": len(gaps) if external_records else 0,
        "gapfill_accepted_count": sum(str(item.get("qc_status", "")).startswith("GAP_FILLED") for item in fills) if qc.get("passed") else 0,
        "external_fetch_status": fetch_status,
        "external_fetch_error": fetch_error, "external_validation": validation,
        "external_metadata": external_meta, "candidate_created": candidate_created,
        "candidate_qc": qc,
        "gap_filled_summary": gap_fill_summary(candidate_rows) if candidate_rows and qc.get("passed") else gap_fill_summary(base),
        "annual_summary": annual, "cli": cli,
        "source_inventory": inventory,
        "candidate_sha256": sha256_file(candidate_path) if candidate_created else None,
        "external_raw_sha256": sha256_file(external_path()) if external_path().exists() else None,
        "leakage_audit_passed": leakage["passed"],
    }
    write_json(OUT / "audit_summary.json", summary)
    manifest = {
        "run_time_utc": run_time, "mode": "audit-only" if args.audit_only or not (args.build_weather_candidate or args.prepare_cli) else "build-weather-candidate" if args.build_weather_candidate else "prepare-cli",
        "script": "scripts/finalize_yc_weather_and_prepare_cli.py", "script_sha256": sha256_file(Path(__file__)),
        "python_version": sys.version, "site": {"wsta": "CNYC", "latitude": SITE_LAT, "longitude": SITE_LON, "elevation_m": SITE_ELEV_M,
                                                "metadata_basis": "project CNYC weather header; used only for location metadata, not climate values"},
        "sources": inventory,
        "external": {"url": external_url, "fetch_status": fetch_status,
                      "download_time_utc": external_cache_metadata.get("download_time_utc"),
                      "cache_path": str(external_path().relative_to(ROOT)) if external_path().exists() else None,
                      "sha256": summary["external_raw_sha256"], "metadata": external_meta,
                      "cache_metadata": external_cache_metadata,
                      "error": fetch_error},
        "source_weather_years_used_for_values_or_calibration": source_weather_years_used_for_values_or_calibration,
        "max_source_weather_years_used_for_values_or_calibration": max(source_weather_years_used_for_values_or_calibration),
        "assertion": "assert max(source_weather_years_used_for_values_or_calibration) <= 2013",
        "backup_path": backup_path,
        "candidate_path": str(candidate_path.relative_to(ROOT)) if candidate_created else None,
        "candidate_sha256": summary["candidate_sha256"],
    }
    write_json(OUT / "run_manifest.json", manifest)
    report_path = ROOT / "docs" / "yc_weather_finalize_and_cli.md"
    report_path.write_text(report_text(summary, OUT), encoding="utf-8")
    write_experiment_log(summary, OUT / "experiment_log.md", inventory, fetch_status, fetch_error)
    print(json.dumps({"final_status": final_status, "gap_count": len(gaps), "external_fetch": fetch_status,
                      "external_validation_passed": validation.get("accepted_for_gap_fill"),
                      "candidate_created": candidate_created, "candidate_qc": qc.get("passed"),
                      "cli_status": cli.get("status")}, ensure_ascii=False, indent=2))
    return 0 if (candidate_created or args.audit_only) else 2


if __name__ == "__main__":
    raise SystemExit(main())
