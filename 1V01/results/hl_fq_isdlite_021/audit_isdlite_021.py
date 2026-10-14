#!/usr/bin/env python
"""Read-only source audit of the 2013 China ISD-Lite package for HLA/FQA.

The raw ZIP and source workbooks are never extracted, copied, or modified. The
script streams selected ZIP members and writes compact audit products only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
RAW = ROOT / "data" / "external" / "cma_daily_weather" / "raw"
README_DIR = RAW / "_数据说明"
ZIP_PATH = RAW / "china_isd_lite_2013.zip"
TZ_UTC = timezone.utc
TZ_LOCAL = timezone(timedelta(hours=8))
YEAR = 2013
DIRECT_FALLBACK_MAX_KM = 50.0
HLA_KEY_DATES = ["2013-07-30", "2013-08-09", "2013-08-12", "2013-07-21", "2013-08-21"]

HLA_DIR = ROOT / "results" / "hla_random_weather_8seed" / "014_isd50756_recovery"
FQA_SOURCE = ROOT / "results" / "hl_fq_random_weather_8seed" / "source_audit"
D222_AUDIT = FQA_SOURCE / "cern_d222_reaudit"
PRIOR_FINAL = FQA_SOURCE / "weather_source_audit_final.json"
HLA_GHCN = FQA_SOURCE / "ghcn_daily_raw" / "CHM00050756.dly"
FQA_GHCN_DAILY = FQA_SOURCE / "ghcn_FQA_chosen_station_daily.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def file_meta(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path_project_relative": str(path.relative_to(ROOT)).replace("\\", "/"),
        "size_bytes": stat.st_size,
        "sha256": sha256(path),
        "modified_local": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
    }


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return round(value, 6)
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        result = float(text)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def ymd(value: str) -> date:
    return date.fromisoformat(value[:10])


def history_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            usaf = (row.get("USAF") or "").strip()
            wban = (row.get("WBAN") or "").strip()
            if usaf and wban:
                grouped[f"{usaf}-{wban}"].append(row)
    return grouped


def active_2013(row: dict[str, str]) -> bool:
    begin = (row.get("BEGIN") or "").strip()
    end = (row.get("END") or "").strip()
    return bool(begin and end and begin <= "20131231" and end >= "20130101")


def metadata_for(key: str, catalog: dict[str, list[dict[str, str]]]) -> dict[str, str] | None:
    rows = catalog.get(key, [])
    if not rows:
        return None
    active = [row for row in rows if active_2013(row)]
    pool = active or rows
    return max(pool, key=lambda row: ((row.get("END") or ""), (row.get("BEGIN") or "")))


def parse_fqa_wth(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("@ INSI") and i + 1 < len(lines):
            names = line.replace("@", " ").split()
            values = lines[i + 1].split()
            if values and values[0].upper() in {"CNFQ", "FQ"}:
                return {
                    "station_code": values[0],
                    "latitude": float(values[names.index("LAT")]),
                    "longitude": float(values[names.index("LONG")]),
                    "elevation_m": float(values[names.index("ELEV")]),
                    "source_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                }
    raise ValueError(f"Could not read FQA coordinates from weather header: {path}")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def day_bounds(day: date) -> tuple[datetime, datetime]:
    start = datetime.combine(day, time.min, TZ_LOCAL)
    return start, start + timedelta(days=1)


def parse_isd_lite_line(raw: bytes) -> tuple[datetime, list[int]]:
    line = raw.decode("ascii", errors="strict").rstrip("\r\n")
    if len(line) < 61:
        raise ValueError(f"short record ({len(line)} chars)")
    # Date: columns 1-4, 6-7, 9-10, 12-13. Each remaining value is six
    # fixed-width columns; the last two are precipitation at columns 50-61.
    obs = datetime(
        int(line[0:4]), int(line[5:7]), int(line[8:10]), int(line[11:13]), tzinfo=TZ_UTC
    )
    measures = [int(line[i : i + 6].strip()) for i in range(13, 61, 6)]
    if len(measures) != 8:
        raise ValueError("expected 8 six-character measurement columns")
    return obs, measures


def rankdata(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        rank = ((i + 1) + j) / 2.0
        for k in range(i, j):
            ranks[ordered[k][0]] = rank
        i = j
    return ranks


def correlation(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    mx, my = sum(x) / len(x), sum(y) / len(y)
    vx = sum((value - mx) ** 2 for value in x)
    vy = sum((value - my) ** 2 for value in y)
    if vx == 0 or vy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(vx * vy)


def pair_metrics(reference: dict[date, float], candidate: dict[date, float]) -> dict[str, Any]:
    n = len(reference)
    if n == 0:
        return {
            "comparable_days": 0,
            "wet_dry_agreement_pct": None,
            "hit_rate_pct": None,
            "false_wet_pct": None,
            "false_dry_pct": None,
            "mae_mm": None,
            "rmse_mm": None,
            "pearson_r": None,
            "spearman_r": None,
            "cumulative_difference_candidate_minus_reference_mm": None,
            "reference_wet_days": 0,
            "reference_dry_days": 0,
            "major_event_days_ge_50": 0,
            "major_event_wet_hit_pct": None,
            "major_event_ge_50_hit_pct": None,
        }
    refs = [reference[d] for d in sorted(reference)]
    cands = [candidate[d] for d in sorted(reference)]
    wet_ref = [r > 0 for r in refs]
    wet_cand = [c > 0 for c in cands]
    wet_n = sum(wet_ref)
    dry_n = n - wet_n
    errors = [c - r for r, c in zip(refs, cands)]
    major = [(r, c) for r, c in zip(refs, cands) if r >= 50.0]
    return {
        "comparable_days": n,
        "wet_dry_agreement_pct": 100.0 * sum(a == b for a, b in zip(wet_ref, wet_cand)) / n,
        "hit_rate_pct": 100.0 * sum(r > 0 and c > 0 for r, c in zip(refs, cands)) / wet_n if wet_n else None,
        "false_wet_pct": 100.0 * sum(r == 0 and c > 0 for r, c in zip(refs, cands)) / dry_n if dry_n else None,
        "false_dry_pct": 100.0 * sum(r > 0 and c == 0 for r, c in zip(refs, cands)) / wet_n if wet_n else None,
        "mae_mm": sum(abs(e) for e in errors) / n,
        "rmse_mm": math.sqrt(sum(e * e for e in errors) / n),
        "pearson_r": correlation(refs, cands),
        "spearman_r": correlation(rankdata(refs), rankdata(cands)),
        "cumulative_difference_candidate_minus_reference_mm": sum(cands) - sum(refs),
        "reference_wet_days": wet_n,
        "reference_dry_days": dry_n,
        "major_event_days_ge_50": len(major),
        "major_event_wet_hit_pct": 100.0 * sum(c > 0 for _, c in major) / len(major) if major else None,
        "major_event_ge_50_hit_pct": 100.0 * sum(c >= 50 for _, c in major) / len(major) if major else None,
    }


def reconstruct_daily(intervals: list[dict[str, Any]], station_id: str, station_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for day_offset in range(365):
        day = date(YEAR, 1, 1) + timedelta(days=day_offset)
        start, end = day_bounds(day)
        in_day: list[dict[str, Any]] = []
        full: list[dict[str, Any]] = []
        cover: set[int] = set()
        numeric_intervals = trace_intervals = missing_intervals = invalid_intervals = 0
        cross_boundary = 0
        for item in intervals:
            intersects = item["end_local"] > start and item["start_local"] < end
            if not intersects:
                continue
            in_day.append(item)
            contained = item["start_local"] >= start and item["end_local"] <= end
            if not contained:
                if item["kind"] in {"numeric", "trace"}:
                    cross_boundary += 1
                continue
            if item["kind"] == "numeric":
                numeric_intervals += 1
                full.append(item)
            elif item["kind"] == "trace":
                trace_intervals += 1
                full.append(item)
            elif item["kind"] == "missing":
                missing_intervals += 1
            else:
                invalid_intervals += 1
            if item["kind"] in {"numeric", "trace"}:
                lo = int((item["start_local"] - start).total_seconds() // 3600)
                hi = int((item["end_local"] - start).total_seconds() // 3600)
                cover.update(range(max(0, lo), min(24, hi)))

        by_start: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for item in full:
            lo = int((item["start_local"] - start).total_seconds() // 3600)
            hi = int((item["end_local"] - start).total_seconds() // 3600)
            if 0 <= lo < hi <= 24:
                by_start[lo].append({**item, "_end_hour": hi})

        states: list[set[tuple[int, bool]]] = [set() for _ in range(25)]
        states[0].add((0, False))
        overflow = False
        for hour in range(24):
            if not states[hour]:
                continue
            for edge in by_start.get(hour, []):
                for amount, has_trace in tuple(states[hour]):
                    if edge["kind"] == "trace":
                        states[edge["_end_hour"]].add((amount, True))
                    else:
                        states[edge["_end_hour"]].add((amount + edge["raw_tenths"], has_trace))
                    if sum(len(s) for s in states) > 100000:
                        overflow = True
                        break
                if overflow:
                    break
            if overflow:
                break

        terminal = states[24] if not overflow else set()
        totals = sorted({amount for amount, trace in terminal if not trace})
        trace_path = any(trace for _, trace in terminal)
        numeric_path = bool(totals)
        raw_union_hours = len(cover)
        observed = bool(in_day)
        daily_mm: float | None = None
        if overflow:
            status = "OTHER_UNRESOLVED"
            reason = "Exact-cover alternatives exceeded the bounded audit state count."
        elif numeric_path and (len(totals) > 1 or trace_path):
            status = "AMBIGUOUS_OVERLAP"
            reason = "Multiple non-overlapping complete tilings disagree, or a complete tiling depends on trace amount."
        elif numeric_path:
            daily_mm = totals[0] / 10.0
            status = "VALID_DRY_DAY" if daily_mm == 0 else "VALID_COMPLETE_DAY"
            reason = "A complete 00:00-24:00 local-day tiling has one unique numeric total."
        elif trace_path:
            status = "TRACE_ONLY"
            reason = "A complete tiling exists only with one or more unquantified trace intervals."
        elif invalid_intervals:
            status = "OTHER_UNRESOLVED"
            reason = "Unsupported negative precipitation code occurs in a local-day interval."
        elif not observed:
            status = "NO_PRECIP_RECORD"
            reason = "No precipitation occurrence window intersects this local day."
        else:
            status = "INCOMPLETE_COVERAGE"
            reason = "No unique complete numeric tiling; cross-midnight intervals are not split."

        overlap_excess = sum(
            int(item["duration_hours"])
            for item in full
            if item["kind"] in {"numeric", "trace"}
        ) - raw_union_hours
        rows.append(
            {
                "date": day.isoformat(),
                "station_id": station_id,
                "station_name": station_name,
                "daily_status": status,
                "rain_mm": daily_mm,
                "interval_coverage": raw_union_hours,
                "numeric_coverage_hours": raw_union_hours,
                "valid_numeric_intervals": numeric_intervals,
                "trace_intervals": trace_intervals,
                "missing_intervals": missing_intervals,
                "invalid_code_intervals": invalid_intervals,
                "cross_boundary_valid_intervals": cross_boundary,
                "overlap_excess_hours": max(0, overlap_excess),
                "distinct_complete_numeric_totals": len(totals),
                "possible_numeric_totals_mm": ";".join(f"{x / 10:.1f}" for x in totals[:20]),
                "has_complete_trace_path": trace_path,
                "reason": reason,
            }
        )
    return rows


def parse_station_file(
    zf: zipfile.ZipFile,
    entry_name: str,
    station_id: str,
    station_name: str,
    occurrence_writer: csv.DictWriter,
) -> tuple[list[dict[str, Any]], dict[date, dict[str, Any]], dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    record_count = malformed_count = field_count = numeric_count = trace_count = missing_count = 0
    with zf.open(entry_name, "r") as stream:
        for line_no, raw in enumerate(stream, start=1):
            if not raw.strip():
                continue
            record_count += 1
            try:
                obs_utc, measures = parse_isd_lite_line(raw)
            except Exception:
                malformed_count += 1
                continue
            obs_local = obs_utc.astimezone(TZ_LOCAL)
            for field_name, duration, value in (
                ("precip_1h", 1, measures[6]),
                ("precip_6h", 6, measures[7]),
            ):
                field_count += 1
                trace = value == -1
                missing = value == -9999
                if trace:
                    kind = "trace"
                    rain = None
                    trace_count += 1
                elif missing:
                    kind = "missing"
                    rain = None
                    missing_count += 1
                elif value >= 0:
                    kind = "numeric"
                    rain = value / 10.0
                    numeric_count += 1
                else:
                    kind = "invalid"
                    rain = None
                start_utc = obs_utc - timedelta(hours=duration)
                start_local = start_utc.astimezone(TZ_LOCAL)
                intervals.append(
                    {
                        "station_id": station_id,
                        "station_name": station_name,
                        "obs_utc": obs_utc,
                        "obs_local": obs_local,
                        "start_local": start_local,
                        "end_local": obs_local,
                        "duration_hours": duration,
                        "raw_tenths": max(value, 0),
                        "raw_value": value,
                        "kind": kind,
                    }
                )
                occurrence_writer.writerow(
                    {
                        "station_id": station_id,
                        "station_name": station_name,
                        "observation_datetime_utc": obs_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "observation_datetime_local": obs_local.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "accumulation_hours": duration,
                        "rain_mm": "" if rain is None else f"{rain:.1f}",
                        "trace_flag": trace,
                        "missing_flag": missing,
                        "source_field": field_name,
                        "raw_value_tenths_mm": value,
                        "raw_line_number": line_no,
                    }
                )

    daily = reconstruct_daily(intervals, station_id, station_name)
    daily_map = {ymd(row["date"]): row for row in daily}
    by_date: dict[date, dict[str, Any]] = defaultdict(
        lambda: {
            "positive_occurrences": 0,
            "trace_occurrences": 0,
            "max_1h_mm": None,
            "max_6h_mm": None,
            "max_1h_observation_utc": None,
            "max_6h_observation_utc": None,
        }
    )
    for item in intervals:
        if item["kind"] not in {"numeric", "trace"}:
            continue
        d = item["end_local"].date()
        start, end = day_bounds(d)
        if item["start_local"] < start or item["end_local"] > end:
            continue
        detail = by_date[d]
        if item["kind"] == "trace":
            detail["trace_occurrences"] += 1
            continue
        if item["raw_tenths"] > 0:
            detail["positive_occurrences"] += 1
            key = "max_1h_mm" if item["duration_hours"] == 1 else "max_6h_mm"
            value = item["raw_tenths"] / 10.0
            if detail[key] is None or value > detail[key]:
                detail[key] = value
                time_key = "max_1h_observation_utc" if item["duration_hours"] == 1 else "max_6h_observation_utc"
                detail[time_key] = item["obs_utc"].strftime("%Y-%m-%dT%H:%M:%SZ")
    summary = {
        "station_id": station_id,
        "station_name": station_name,
        "raw_2013_record_count": record_count,
        "malformed_record_count": malformed_count,
        "precip_field_occurrence_count": field_count,
        "numeric_precip_field_count": numeric_count,
        "trace_precip_field_count": trace_count,
        "missing_precip_field_count": missing_count,
        "valid_numeric_days": sum(row["daily_status"] in {"VALID_COMPLETE_DAY", "VALID_DRY_DAY"} for row in daily),
    }
    return daily, by_date, summary


def parse_ghcn_dly(path: Path, station_id: str) -> dict[date, float]:
    values: dict[date, float] = {}
    with path.open("r", encoding="ascii", errors="replace") as handle:
        for line in handle:
            if len(line) < 21 or line[:11].strip() != station_id:
                continue
            year = int(line[11:15])
            month = int(line[15:17])
            element = line[17:21]
            if year != YEAR or element != "PRCP":
                continue
            for day_index in range(31):
                offset = 21 + day_index * 8
                cell = line[offset : offset + 8]
                if len(cell) < 8:
                    continue
                raw_value = int(cell[:5])
                qflag = cell[6]
                if raw_value == -9999 or qflag.strip():
                    continue
                try:
                    d = date(year, month, day_index + 1)
                except ValueError:
                    continue
                values[d] = raw_value / 10.0
    return values


def daily_numeric_map(rows: list[dict[str, Any]]) -> dict[date, float]:
    return {
        ymd(row["date"]): float(row["rain_mm"])
        for row in rows
        if row["daily_status"] in {"VALID_COMPLETE_DAY", "VALID_DRY_DAY"} and row["rain_mm"] is not None
    }


def compare_with_shifts(
    reference: dict[date, float],
    candidate_by_date: dict[date, float],
    base: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for shift in (-1, 0, 1):
        aligned_ref: dict[date, float] = {}
        aligned_candidate: dict[date, float] = {}
        for ref_day, ref_value in reference.items():
            cand_day = ref_day + timedelta(days=shift)
            if cand_day in candidate_by_date:
                aligned_ref[ref_day] = ref_value
                aligned_candidate[ref_day] = candidate_by_date[cand_day]
        metrics = pair_metrics(aligned_ref, aligned_candidate)
        rows.append({**base, "candidate_date_equals_reference_date_plus_days": shift, **metrics})
    return rows


def read_local_rain(path: Path, site: str, column: str) -> dict[date, float]:
    result: dict[date, float] = {}
    for row in read_csv(path):
        if row.get("site") != site:
            continue
        value = numeric(row.get(column))
        if value is not None:
            result[ymd(row["date"])] = value
    return result


def source_artifacts() -> list[Path]:
    return [
        HLA_DIR / "unresolved_dates_after_014.csv",
        HLA_DIR / "station_identity.json",
        HLA_DIR / "daily_source_selection_after_014.csv",
        HLA_GHCN,
        PRIOR_FINAL,
        D222_AUDIT / "overlap_same_day.csv",
        D222_AUDIT / "fqa_extreme_rain_audit.csv",
        FQA_GHCN_DAILY,
        ROOT / "my_data" / "CNFQ0701.WTH",
    ]


def nearest_bin(distance: float) -> tuple[str, str]:
    if distance <= 25:
        return "0-25 km", "LOCAL_SPATIAL_CANDIDATE"
    if distance <= 50:
        return "25-50 km", "NEARBY_SPATIAL_CANDIDATE"
    if distance <= 100:
        return "50-100 km", "REFERENCE_ONLY_50_100_KM"
    if distance <= 200:
        return "100-200 km", "REFERENCE_ONLY_100_200_KM"
    return ">200 km", "DISTANT_REFERENCE_ONLY"


def station_id_from_member(name: str) -> str | None:
    base = name.rsplit("/", 1)[-1]
    match = re.fullmatch(r"(\d{6}-\d{5})-2013", base)
    return match.group(1) if match else None


def main() -> None:
    expected_outputs = [
        "input_provenance.json",
        "dataset_identity.json",
        "hla_station_identity.json",
        "fqa_station_candidates.csv",
        "precip_occurrences_2013.csv",
        "hla_isdlite_daily_2013.csv",
        "hla_vs_ghcn_qa.csv",
        "hla_vs_d222_qa.csv",
        "hla_recovery_on_111_days.csv",
        "fqa_candidate_overlap_qa.csv",
        "fqa_candidate_spatial_qa.csv",
        "fqa_recovery_on_2013_gaps.csv",
        "fqa_2013_reference_summary.csv",
        "gap_recovery_summary.csv",
        "fqa_2013_major_event_context.csv",
        "hla_key_event_context.csv",
        "final_gate.json",
        "findings.md",
    ]
    missing = [str(path) for path in [ZIP_PATH, README_DIR / "isd-history.csv", README_DIR / "isd-lite-format.txt", *source_artifacts()] if not path.exists()]
    if missing:
        raise FileNotFoundError("Required local audit input missing: " + ", ".join(missing))

    # The output directory is task-specific and was confirmed empty before this
    # script was created. Only these audit deliverables are written here.
    DOC_FORMAT = DOCS / "hl_fq_isdlite_data_format_021.md"
    DOC_REPORT = DOCS / "hl_fq_isdlite_weather_audit_021.md"

    catalog_path = README_DIR / "isd-history.csv"
    format_path = README_DIR / "isd-lite-format.txt"
    history = history_rows(catalog_path)
    fqa_wth = ROOT / "my_data" / "CNFQ0701.WTH"
    fqa_meta = parse_fqa_wth(fqa_wth)
    prior = json.loads(PRIOR_FINAL.read_text(encoding="utf-8-sig"))
    hla_unresolved_path = HLA_DIR / "unresolved_dates_after_014.csv"
    hla_gap_rows = read_csv(hla_unresolved_path)
    hla_gap_dates = {ymd(row["date"]): row for row in hla_gap_rows}
    hla_d222 = {
        day: value
        for day, value in read_local_rain(HLA_DIR / "daily_source_selection_after_014.csv", "HLA", "local_d222_rain_mm").items()
        if day.year == YEAR
    }
    fqa_d222 = {
        day: value
        for day, value in read_local_rain(D222_AUDIT / "overlap_same_day.csv", "FQA", "d222_rain_raw").items()
        if day.year == YEAR
    }
    fqa_unresolved = [
        ymd(value)
        for value in prior["sites"]["FQA"]["official_ghcn_rain_screen"]["unresolved_local_rain_days"]
        if value.startswith("2013-")
    ]
    hla_identity_014 = json.loads((HLA_DIR / "station_identity.json").read_text(encoding="utf-8-sig"))

    fqa_exact_aliases = {"FENGQIU", "FENGCHIU", "FENG QIU", "封丘"}
    exact_catalog_matches = []
    for station_key, rows in history.items():
        for row in rows:
            name = (row.get("STATION NAME") or "").strip().upper()
            normalized = re.sub(r"\s+", "", name)
            if name in fqa_exact_aliases or normalized in {"FENGQIU", "FENGCHIU", "封丘"}:
                exact_catalog_matches.append({"station_id": station_key, "station_name": name, "country": row.get("CTRY", "")})

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        members = [item for item in zf.infolist() if not item.is_dir() and station_id_from_member(item.filename)]
        member_by_id = {station_id_from_member(item.filename): item for item in members}
        archive_member_ids = set(member_by_id)
        candidates_all: list[dict[str, Any]] = []
        catalog_match_count = 0
        china_member_count = 0
        four_region_member_count = 0
        for station_id, item in member_by_id.items():
            meta = metadata_for(station_id, history)
            if meta:
                catalog_match_count += 1
            country = (meta.get("CTRY", "") if meta else "").strip().upper()
            if country in {"CH", "HK", "MC", "TW"}:
                four_region_member_count += 1
            if country != "CH" or not meta:
                continue
            try:
                lat = float(meta["LAT"])
                lon = float(meta["LON"])
            except (KeyError, ValueError, TypeError):
                continue
            china_member_count += 1
            distance = haversine_km(fqa_meta["latitude"], fqa_meta["longitude"], lat, lon)
            distance_bin, priority = nearest_bin(distance)
            candidates_all.append(
                {
                    "station_id": station_id,
                    "station_name": (meta.get("STATION NAME") or "").strip(),
                    "country": country,
                    "latitude": lat,
                    "longitude": lon,
                    "elevation_m": numeric(meta.get("ELEV(M)")),
                    "distance_to_FQA_km": distance,
                    "distance_bin": distance_bin,
                    "2013_file_present": True,
                    "raw_2013_record_count": None,
                    "candidate_priority": priority,
                    "catalog_active_2013": active_2013(meta),
                    "zip_member": item.filename,
                    "zip_uncompressed_bytes": item.file_size,
                    "notes": "2013 station file exists; distance is from CNFQ0701.WTH header. Remote candidates remain reference-only.",
                }
            )
        candidates_all.sort(key=lambda row: row["distance_to_FQA_km"])
        top20 = candidates_all[:20]
        exact_file_matches = [row for row in candidates_all if row["station_name"].upper().replace(" ", "") in {"FENGQIU", "FENGCHIU"}]
        selected_candidates = list(top20)
        selected_ids = {row["station_id"] for row in selected_candidates}
        for row in exact_file_matches:
            if row["station_id"] not in selected_ids:
                selected_candidates.append(row)
                selected_ids.add(row["station_id"])

        hla_station_id = "507560-99999"
        hla_member = member_by_id.get(hla_station_id)
        hla_history = metadata_for(hla_station_id, history)
        hla_file_present = hla_member is not None
        hla_identity_pass = bool(
            hla_file_present
            and hla_history
            and (hla_history.get("STATION NAME") or "").strip().upper() == "HAILUN"
            and (hla_history.get("CTRY") or "").strip().upper() == "CH"
            and hla_identity_014.get("status") == "PASS"
            and hla_identity_014.get("ghcn_station") == "CHM00050756"
        )

        selected_station_records: dict[str, dict[str, Any]] = {}
        daily_by_station: dict[str, list[dict[str, Any]]] = {}
        occurrence_day_by_station: dict[str, dict[date, dict[str, Any]]] = {}
        occurrence_fields = [
            "station_id",
            "station_name",
            "observation_datetime_utc",
            "observation_datetime_local",
            "accumulation_hours",
            "rain_mm",
            "trace_flag",
            "missing_flag",
            "source_field",
            "raw_value_tenths_mm",
            "raw_line_number",
        ]
        occurrence_path = OUT / "precip_occurrences_2013.csv"
        with occurrence_path.open("w", encoding="utf-8-sig", newline="") as out_handle:
            occurrence_writer = csv.DictWriter(out_handle, fieldnames=occurrence_fields)
            occurrence_writer.writeheader()
            stations_to_process: list[tuple[str, str, str]] = []
            if hla_file_present:
                stations_to_process.append((hla_station_id, hla_history.get("STATION NAME", "HAILUN"), hla_member.filename))
            for candidate in selected_candidates:
                if candidate["station_id"] not in {item[0] for item in stations_to_process}:
                    stations_to_process.append((candidate["station_id"], candidate["station_name"], candidate["zip_member"]))
            for station_id, station_name, member_name in stations_to_process:
                daily, occ_days, summary = parse_station_file(zf, member_name, station_id, station_name, occurrence_writer)
                selected_station_records[station_id] = summary
                daily_by_station[station_id] = daily
                occurrence_day_by_station[station_id] = occ_days
                for candidate in selected_candidates:
                    if candidate["station_id"] == station_id:
                        candidate["raw_2013_record_count"] = summary["raw_2013_record_count"]
                        candidate["malformed_record_count"] = summary["malformed_record_count"]
                        candidate["valid_reconstructed_days"] = summary["valid_numeric_days"]
                        candidate["numeric_precip_occurrences"] = summary["numeric_precip_field_count"]

    # Station inventory and source identities.
    candidate_csv_rows = []
    for candidate in top20:
        candidate_csv_rows.append(candidate)
    candidate_fields = [
        "station_id",
        "station_name",
        "country",
        "latitude",
        "longitude",
        "elevation_m",
        "distance_to_FQA_km",
        "distance_bin",
        "2013_file_present",
        "raw_2013_record_count",
        "candidate_priority",
        "catalog_active_2013",
        "valid_reconstructed_days",
        "numeric_precip_occurrences",
        "notes",
    ]
    write_csv(OUT / "fqa_station_candidates.csv", candidate_csv_rows, candidate_fields)

    format_docs = [p for p in README_DIR.iterdir() if p.is_file()]
    archive_sha = sha256(ZIP_PATH)
    source_meta = [file_meta(path) for path in [ZIP_PATH, *format_docs, *source_artifacts()]]
    input_provenance = {
        "task": "021",
        "created_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "raw_data_access": "read-only ZIP member streaming; no raw file extracted, moved, copied, renamed, or modified",
        "archive": file_meta(ZIP_PATH),
        "source_description": "The supplied Chinese data description states the files derive from NCDC/NOAA ISD-Lite public FTP and were repackaged by year for a regional station subset; the directory name cma_daily_weather is historical and not the source identity.",
        "readme_files": [file_meta(path) for path in sorted(format_docs)],
        "audit_input_artifacts": [file_meta(path) for path in source_artifacts()],
        "project_scope": str(ROOT),
    }
    write_json(OUT / "input_provenance.json", input_provenance)

    # HLA identity links file-name identity to the prior full-ISD/GHCN closure.
    hla_identity = {
        "status": "PASS" if hla_identity_pass else "FAIL",
        "site": "HLA",
        "isd_lite_station_id": hla_station_id,
        "station_name": (hla_history.get("STATION NAME") or "").strip() if hla_history else None,
        "country": (hla_history.get("CTRY") or "").strip() if hla_history else None,
        "latitude": numeric(hla_history.get("LAT")) if hla_history else None,
        "longitude": numeric(hla_history.get("LON")) if hla_history else None,
        "elevation_m": numeric(hla_history.get("ELEV(M)")) if hla_history else None,
        "2013_file_present": hla_file_present,
        "2013_zip_member": hla_member.filename if hla_member else None,
        "raw_2013_record_count": selected_station_records.get(hla_station_id, {}).get("raw_2013_record_count"),
        "ghcn_station_id": "CHM00050756",
        "wmo_station_number": "50756",
        "prior_014_full_isd_identity_status": hla_identity_014.get("status"),
        "prior_014_identity_evidence": hla_identity_014.get("identity_evidence", []),
        "metadata_discrepancy": hla_identity_014.get("metadata_discrepancy"),
        "interpretation": "ISD-Lite filename USAF/WBAN matches 014 full-ISD station 507560-99999; station history calls it HAILUN, CH; 014 linked WMO 50756 and GHCN CHM00050756. ISD-Lite itself contains no station coordinates, so the prior full-ISD/GHCN location comparison is retained as context.",
    }
    write_json(OUT / "hla_station_identity.json", hla_identity)

    dataset_identity = {
        "dataset": "NOAA_ISD_LITE_CHINA_2013",
        "source_organization": "NOAA / NCDC, now NCEI",
        "source_product": "Integrated Surface Data Lite (ISD-Lite), a fixed-width subset of ISD",
        "regional_package_source_note": "The supplied _数据说明.docx identifies NOAA/NCDC public ISD-Lite FTP as the underlying data source and states this regional package is selected/repacked by year; it is not a CMA daily product.",
        "raw_archive_name": ZIP_PATH.name,
        "sha256": archive_sha,
        "archive_bytes": ZIP_PATH.stat().st_size,
        "zip_member_file_count": len(members),
        "unique_station_files": len(archive_member_ids),
        "metadata_matched_station_files": catalog_match_count,
        "files_in_CH_HK_MC_TW_region": four_region_member_count,
        "China_country_code_CH_station_files": china_member_count,
        "2013_files_with_valid_CH_coordinates": len(candidates_all),
        "HLA_507560_file_present": hla_file_present,
        "FQA_exact_name_catalog_matches": exact_catalog_matches,
        "FQA_exact_name_2013_file_matches": exact_file_matches,
        "fqa_station_metadata": fqa_meta,
        "station_file_name_rule": "USAF-WBAN-YYYY; observed HLA member is 507560-99999-2013.",
        "time_basis": "UTC/GMT; the supplied data description says GMT. ISD-Lite timestamps are rounded to nearest whole hour per NCEI technical documentation.",
        "record_resolution": "Observation timestamp is nearest whole hour; actual cadence varies by station. The package description says most recent data are 3-hourly with some 1-hour stations.",
        "record_layout": {
            "line_length_characters": 61,
            "date_columns_1_based": {"year": "1-4", "month": "6-7", "day": "9-10", "hour": "12-13"},
            "measurements": [
                {"field": "air_temperature", "columns_1_based": "14-19"},
                {"field": "dew_point", "columns_1_based": "20-25"},
                {"field": "sea_level_pressure", "columns_1_based": "26-31"},
                {"field": "wind_direction", "columns_1_based": "32-37"},
                {"field": "wind_speed", "columns_1_based": "38-43"},
                {"field": "sky_cover", "columns_1_based": "44-49"},
                {"field": "precip_1h", "columns_1_based": "50-55", "scale": 10, "unit": "mm"},
                {"field": "precip_6h", "columns_1_based": "56-61", "scale": 10, "unit": "mm"},
            ],
            "precip_missing_value": -9999,
            "precip_trace_value": -1,
            "precip_trace_interpretation": "Unquantified trace; not assigned an arbitrary rainfall depth.",
            "accumulation_window_assumption": "For time-window tests, an accumulation at timestamp t covers the preceding duration ending at t: [t-1h,t) or [t-6h,t). The bundled format defines accumulation duration and the official format describes depth measured at observation time; this endpoint reading is stated as an audit assumption because the bundled summary does not spell out interval endpoints.",
            "local_day": "00:00-24:00 Asia/Shanghai (UTC+08:00). Windows crossing a local midnight are not split or assigned to one date.",
        },
        "source_relationship": "GHCN-Daily and ISD-Lite are both NOAA products and may share a national/GTS observation chain. They are product-level cross-checks, not guaranteed independent physical observations. CERN D222 is the local station observation chain.",
        "ncei_technical_document": "https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/isd-lite-technical-document.pdf",
        "ncei_format_document": "https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/isd-lite-format.pdf",
        "archive_members_examined_for_precipitation": sorted(selected_ids | {hla_station_id}),
        "archive_was_extracted": False,
    }
    write_json(OUT / "dataset_identity.json", dataset_identity)

    # Daily outputs for the HLA same station and every one of the nearest 20
    # Chinese stations with an actual 2013 archive member.
    hla_daily = daily_by_station.get(hla_station_id, [])
    daily_fields = [
        "date",
        "station_id",
        "station_name",
        "daily_status",
        "rain_mm",
        "interval_coverage",
        "numeric_coverage_hours",
        "valid_numeric_intervals",
        "trace_intervals",
        "missing_intervals",
        "invalid_code_intervals",
        "cross_boundary_valid_intervals",
        "overlap_excess_hours",
        "distinct_complete_numeric_totals",
        "possible_numeric_totals_mm",
        "has_complete_trace_path",
        "reason",
    ]
    write_csv(OUT / "hla_isdlite_daily_2013.csv", hla_daily, daily_fields)
    for candidate in selected_candidates:
        station_id = candidate["station_id"]
        daily = daily_by_station.get(station_id, [])
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", station_id)
        write_csv(OUT / f"fqa_{safe_id}_daily_2013.csv", daily, daily_fields)

    hla_candidate_daily = daily_numeric_map(hla_daily)
    hla_ghcn = parse_ghcn_dly(HLA_GHCN, "CHM00050756")
    hla_vs_ghcn_rows = compare_with_shifts(
        hla_ghcn,
        hla_candidate_daily,
        {
            "comparison": "ISD-Lite reconstructed local day vs GHCN-Daily PRCP",
            "candidate_station": hla_station_id,
            "reference_station": "CHM00050756",
            "source_relationship": "Both NOAA products; may share GTS observation chain.",
        },
    )
    write_csv(OUT / "hla_vs_ghcn_qa.csv", hla_vs_ghcn_rows, list(hla_vs_ghcn_rows[0].keys()))
    hla_vs_d222_rows = compare_with_shifts(
        hla_d222,
        hla_candidate_daily,
        {
            "comparison": "ISD-Lite reconstructed local day vs explicit HLA D222 value",
            "candidate_station": hla_station_id,
            "reference_station": "HLA CERN D222",
            "source_relationship": "Local D222 numeric values; blank D222 cells are excluded, never set to zero.",
        },
    )
    write_csv(OUT / "hla_vs_d222_qa.csv", hla_vs_d222_rows, list(hla_vs_d222_rows[0].keys()))

    hla_recovery_rows = []
    for day in sorted(hla_gap_dates):
        daily = next((row for row in hla_daily if row["date"] == day.isoformat()), None)
        status = daily["daily_status"] if daily else "NO_PRECIP_RECORD"
        rain = daily["rain_mm"] if daily else None
        recovered = status in {"VALID_COMPLETE_DAY", "VALID_DRY_DAY"} and rain is not None
        hla_recovery_rows.append(
            {
                "date": day.isoformat(),
                "previous_status": hla_gap_dates[day].get("previous_status", "UNRESOLVED"),
                "isdlite_status": status,
                "isdlite_rain_mm": rain,
                "interval_coverage": daily.get("interval_coverage") if daily else 0,
                "recovery_decision": "RECOVERED_BY_SAME_STATION_ISDLITE" if recovered else "UNRESOLVED",
                "reason": daily.get("reason", "No HLA daily row available.") if daily else "No HLA daily row available.",
            }
        )
    write_csv(
        OUT / "hla_recovery_on_111_days.csv",
        hla_recovery_rows,
        ["date", "previous_status", "isdlite_status", "isdlite_rain_mm", "interval_coverage", "recovery_decision", "reason"],
    )

    # FQA candidate overlap and spatial QA. D222 blanks are excluded.
    fqa_overlap_rows: list[dict[str, Any]] = []
    for candidate in top20:
        station_id = candidate["station_id"]
        candidate_daily = daily_numeric_map(daily_by_station.get(station_id, []))
        base = {
            "candidate_station": station_id,
            "candidate_name": candidate["station_name"],
            "distance_to_FQA_km": candidate["distance_to_FQA_km"],
            "elevation_difference_to_FQA_m": (
                candidate["elevation_m"] - fqa_meta["elevation_m"] if candidate["elevation_m"] is not None else None
            ),
            "reference_station": "FQA CERN D222",
            "source_relationship": "Remote NOAA station product; not an independent local gauge.",
            "date_shift_convention": "Candidate date = D222 reference date + shift days.",
        }
        fqa_overlap_rows.extend(compare_with_shifts(fqa_d222, candidate_daily, base))
    fqa_overlap_fields = list(fqa_overlap_rows[0].keys()) if fqa_overlap_rows else []
    write_csv(OUT / "fqa_candidate_overlap_qa.csv", fqa_overlap_rows, fqa_overlap_fields)

    same_day_by_station = {
        row["candidate_station"]: row
        for row in fqa_overlap_rows
        if row["candidate_date_equals_reference_date_plus_days"] == 0
    }
    spatial_rows = []
    for candidate in top20:
        overlap = same_day_by_station[candidate["station_id"]]
        dist = candidate["distance_to_FQA_km"]
        spatial_rows.append(
            {
                "station_id": candidate["station_id"],
                "station_name": candidate["station_name"],
                "country": candidate["country"],
                "distance_to_FQA_km": dist,
                "distance_bin": candidate["distance_bin"],
                "elevation_m": candidate["elevation_m"],
                "elevation_difference_to_FQA_m": candidate["elevation_m"] - fqa_meta["elevation_m"] if candidate["elevation_m"] is not None else None,
                "2013_file_present": candidate["2013_file_present"],
                "raw_2013_record_count": candidate["raw_2013_record_count"],
                "valid_reconstructed_days": candidate["valid_reconstructed_days"],
                "same_day_numeric_overlap_days": overlap["comparable_days"],
                "same_day_wet_dry_agreement_pct": overlap["wet_dry_agreement_pct"],
                "same_day_mae_mm": overlap["mae_mm"],
                "same_day_rmse_mm": overlap["rmse_mm"],
                "spatial_use": "DIRECT_SPATIAL_SCREEN_PASSED" if dist <= DIRECT_FALLBACK_MAX_KM else "REFERENCE_ONLY",
                "interpretation": "Distance is only a spatial screen; a numeric fallback also requires complete local-day reconstruction and adequate explicit D222 overlap QA." if dist <= DIRECT_FALLBACK_MAX_KM else "Beyond the audit's 50 km direct-fallback screen; use only as regional reference even where observations exist.",
            }
        )
    write_csv(OUT / "fqa_candidate_spatial_qa.csv", spatial_rows, list(spatial_rows[0].keys()) if spatial_rows else [])

    best = top20[0] if top20 else None
    accepted_station_ids: set[str] = set()
    for candidate in top20:
        overlap = same_day_by_station[candidate["station_id"]]
        if (
            candidate["distance_to_FQA_km"] <= DIRECT_FALLBACK_MAX_KM
            and overlap["comparable_days"] >= 20
            and overlap["wet_dry_agreement_pct"] is not None
            and overlap["wet_dry_agreement_pct"] >= 80.0
        ):
            accepted_station_ids.add(candidate["station_id"])

    fqa_recovery_rows = []
    for day in sorted(set(fqa_unresolved)):
        recover_candidate = None
        reference_candidate = None
        for candidate in top20:
            station_id = candidate["station_id"]
            daily_row = next((row for row in daily_by_station.get(station_id, []) if row["date"] == day.isoformat()), None)
            if not daily_row or daily_row["daily_status"] not in {"VALID_COMPLETE_DAY", "VALID_DRY_DAY"}:
                continue
            if station_id in accepted_station_ids:
                recover_candidate = (candidate, daily_row)
                break
            if reference_candidate is None:
                reference_candidate = (candidate, daily_row)
        if recover_candidate:
            cand, day_row = recover_candidate
            decision = "RECOVERABLE_BY_ACCEPTED_STATION"
            station = cand["station_id"]
            dist = cand["distance_to_FQA_km"]
            state = day_row["daily_status"]
            rain = day_row["rain_mm"]
            reason = "Candidate passed the audit spatial screen, reconstructed the entire local day, and passed the minimum explicit-D222 overlap gate."
        elif reference_candidate:
            cand, day_row = reference_candidate
            decision = "REFERENCE_ONLY"
            station = cand["station_id"]
            dist = cand["distance_to_FQA_km"]
            state = day_row["daily_status"]
            rain = day_row["rain_mm"]
            reason = "A complete remote-station daily total exists, but no candidate passed the direct-fallback spatial/overlap gate."
        else:
            station = best["station_id"] if best else ""
            dist = best["distance_to_FQA_km"] if best else None
            day_row = next((row for row in daily_by_station.get(station, []) if row["date"] == day.isoformat()), None) if station else None
            state = day_row["daily_status"] if day_row else "NO_PRECIP_RECORD"
            rain = day_row["rain_mm"] if day_row else None
            decision = "UNRESOLVED"
            reason = "No complete numeric local-day reconstruction from an accepted station; FQA D222 blank remains unresolved."
        fqa_recovery_rows.append(
            {
                "date": day.isoformat(),
                "previous_status": "UNRESOLVED_AFTER_008_009",
                "candidate_station": station,
                "distance_km": dist,
                "isdlite_status": state,
                "isdlite_rain_mm": rain,
                "recovery_decision": decision,
                "reason": reason,
            }
        )
    write_csv(
        OUT / "fqa_recovery_on_2013_gaps.csv",
        fqa_recovery_rows,
        ["date", "previous_status", "candidate_station", "distance_km", "isdlite_status", "isdlite_rain_mm", "recovery_decision", "reason"],
    )

    # Event context preserves sub-daily values as event signals, never as daily sums.
    fqa_ghcn_values: dict[date, float] = {}
    for row in read_csv(FQA_GHCN_DAILY):
        d = ymd(row["date"])
        value = numeric(row.get("rain_mm"))
        if d.year == YEAR and value is not None:
            fqa_ghcn_values[d] = value
    fqa_extreme_rows = read_csv(D222_AUDIT / "fqa_extreme_rain_audit.csv")
    best_valid_days = best["valid_reconstructed_days"] if best else 0
    fqa_extreme_2013 = next(
        (row for row in fqa_extreme_rows if row.get("record_type") == "YEAR_SUMMARY" and row.get("year") == "2013"),
        {},
    )
    fqa_events = [
        row for row in fqa_extreme_rows
        if row.get("record_type") == "EXTREME_EVENT" and row.get("date", "").startswith("2013-") and (numeric(row.get("d222_20_20_mm")) or 0) >= 50
    ]
    fqa_event_rows = []
    best_id = best["station_id"] if best else ""
    best_days = occurrence_day_by_station.get(best_id, {})
    best_daily = {row["date"]: row for row in daily_by_station.get(best_id, [])} if best else {}
    for event in fqa_events:
        d = ymd(event["date"])
        day_detail = best_days.get(d, {})
        near_details = [best_days.get(d + timedelta(days=shift), {}) for shift in (-1, 0, 1)]
        fqa_event_rows.append(
            {
                "date": d.isoformat(),
                "d222_rain_mm": numeric(event.get("d222_20_20_mm")),
                "nearest_isdlite_station": best_id,
                "nearest_isdlite_distance_km": best["distance_to_FQA_km"] if best else None,
                "nearest_isdlite_daily_status": best_daily.get(d.isoformat(), {}).get("daily_status", "NO_PRECIP_RECORD"),
                "nearest_isdlite_daily_mm": best_daily.get(d.isoformat(), {}).get("rain_mm"),
                "isdlite_positive_subdaily_occurrences_same_day": day_detail.get("positive_occurrences", 0),
                "isdlite_trace_occurrences_same_day": day_detail.get("trace_occurrences", 0),
                "isdlite_max_1h_mm_same_day": day_detail.get("max_1h_mm"),
                "isdlite_max_1h_observation_utc": day_detail.get("max_1h_observation_utc"),
                "isdlite_max_6h_mm_same_day": day_detail.get("max_6h_mm"),
                "isdlite_max_6h_observation_utc": day_detail.get("max_6h_observation_utc"),
                "isdlite_positive_subdaily_occurrence_within_pm1_day": any(x.get("positive_occurrences", 0) for x in near_details),
                "ghcn_57083_same_day_mm": fqa_ghcn_values.get(d),
                "nasa_same_day_mm": numeric(event.get("nasa_same_day_mm")),
                "nasa_pm1_max_mm": numeric(event.get("nasa_pm1_max_mm")),
                "ghcn_pm1_max_mm_from_prior_audit": numeric(event.get("ghcn_pm1_max_mm")),
                "interpretation": "A positive 1h/6h occurrence indicates nearby rain only; it is not a daily amount and the remote station cannot verify FQA magnitude.",
            }
        )
    write_csv(OUT / "fqa_2013_major_event_context.csv", fqa_event_rows, list(fqa_event_rows[0].keys()) if fqa_event_rows else [])

    d222_fqa_numeric_subtotal = sum(fqa_d222.values())
    fqa_ghcn_same_day_numeric_overlap = len(set(fqa_d222) & set(fqa_ghcn_values))
    fqa_reference_summary_rows = [
        {
            "source": "CERN D222 FQA local chain",
            "2013_numeric_subtotal_or_annual_total_mm": d222_fqa_numeric_subtotal,
            "numeric_days_or_valid_days": len(fqa_d222),
            "expected_days": 365,
            "full_year_numeric_coverage": False,
            "interpretation": "Sum of explicitly numeric D222 days only; 330 blank days are unresolved and are not zeros.",
        },
        {
            "source": "NOAA GHCN-Daily CHM00057083 selected FQA candidate",
            "2013_numeric_subtotal_or_annual_total_mm": sum(fqa_ghcn_values.values()),
            "numeric_days_or_valid_days": len(fqa_ghcn_values),
            "expected_days": 365,
            "full_year_numeric_coverage": len(fqa_ghcn_values) == 365,
            "interpretation": f"Recorded-value subtotal on {len(fqa_ghcn_values)} valid days; the prior source audit's 399.5 mm is not a full-year total. Same-day numeric overlap with D222 is {fqa_ghcn_same_day_numeric_overlap} days.",
        },
        {
            "source": "NASA POWER FQA point daily PRECTOTCORR",
            "2013_numeric_subtotal_or_annual_total_mm": numeric(fqa_extreme_2013.get("nasa_annual_total_mm")),
            "numeric_days_or_valid_days": 365,
            "expected_days": 365,
            "full_year_numeric_coverage": True,
            "interpretation": "Full 2013 gridded point series in existing read-only artifact; auxiliary grid reference, not a local gauge.",
        },
        {
            "source": f"NOAA ISD-Lite {best_id if best else 'NONE'}",
            "2013_numeric_subtotal_or_annual_total_mm": None,
            "numeric_days_or_valid_days": best_valid_days,
            "expected_days": 365,
            "full_year_numeric_coverage": False,
            "interpretation": "No uniquely reconstructed full local days; subdaily occurrences cannot be summed as an annual daily series.",
        },
    ]
    write_csv(
        OUT / "fqa_2013_reference_summary.csv",
        fqa_reference_summary_rows,
        ["source", "2013_numeric_subtotal_or_annual_total_mm", "numeric_days_or_valid_days", "expected_days", "full_year_numeric_coverage", "interpretation"],
    )

    hla_selected_rows = {row["date"]: row for row in read_csv(HLA_DIR / "daily_source_selection_after_014.csv") if row.get("site") == "HLA" and row.get("year") == "2013"}
    hla_occ_days = occurrence_day_by_station.get(hla_station_id, {})
    hla_daily_map = {row["date"]: row for row in hla_daily}
    hla_event_rows = []
    for date_text in HLA_KEY_DATES:
        d = ymd(date_text)
        detail = hla_occ_days.get(d, {})
        selected = hla_selected_rows.get(date_text, {})
        hla_event_rows.append(
            {
                "date": date_text,
                "d222_explicit_rain_mm": numeric(selected.get("local_d222_rain_mm")),
                "ghcn_50756_rain_mm": numeric(selected.get("ghcn_50756_rain_mm")),
                "prior_selected_rain_mm": numeric(selected.get("selected_rain_mm")),
                "prior_selected_source": selected.get("selected_source", ""),
                "isdlite_daily_status": hla_daily_map.get(date_text, {}).get("daily_status", "NO_PRECIP_RECORD"),
                "isdlite_daily_mm": hla_daily_map.get(date_text, {}).get("rain_mm"),
                "positive_subdaily_occurrences": detail.get("positive_occurrences", 0),
                "trace_occurrences": detail.get("trace_occurrences", 0),
                "max_1h_mm": detail.get("max_1h_mm"),
                "max_1h_observation_utc": detail.get("max_1h_observation_utc"),
                "max_6h_mm": detail.get("max_6h_mm"),
                "max_6h_observation_utc": detail.get("max_6h_observation_utc"),
                "interpretation": "Sub-daily occurrence can support event presence but does not supply a complete daily amount unless daily_status is VALID_COMPLETE_DAY/VALID_DRY_DAY.",
            }
        )
    write_csv(OUT / "hla_key_event_context.csv", hla_event_rows, list(hla_event_rows[0].keys()))

    hla_recovered = sum(row["recovery_decision"] == "RECOVERED_BY_SAME_STATION_ISDLITE" for row in hla_recovery_rows)
    hla_remaining = len(hla_recovery_rows) - hla_recovered
    hla_valid_all = sum(row["daily_status"] in {"VALID_COMPLETE_DAY", "VALID_DRY_DAY"} for row in hla_daily)
    hla_status = "ACCEPTED" if len(hla_daily) == 365 and hla_valid_all == 365 else ("PARTIAL" if hla_valid_all else "NOT_ACCEPTED")
    fqa_recovered = sum(row["recovery_decision"] == "RECOVERABLE_BY_ACCEPTED_STATION" for row in fqa_recovery_rows)
    fqa_reference_only = sum(row["recovery_decision"] == "REFERENCE_ONLY" for row in fqa_recovery_rows)
    fqa_remaining = len(fqa_recovery_rows) - fqa_recovered
    fqa_status = "REFERENCE_ONLY" if best and best["distance_to_FQA_km"] > DIRECT_FALLBACK_MAX_KM else ("PARTIAL" if fqa_recovered else "NOT_ACCEPTED")

    hla_d222_same = next((row for row in hla_vs_d222_rows if row["candidate_date_equals_reference_date_plus_days"] == 0), {})
    hla_ghcn_same = next((row for row in hla_vs_ghcn_rows if row["candidate_date_equals_reference_date_plus_days"] == 0), {})
    fqa_best_same = same_day_by_station.get(best_id, {}) if best else {}

    summary_rows = [
        {
            "site": "HLA",
            "previous_unresolved_days": len(hla_gap_dates),
            "valid_isdlite_days_2013": hla_valid_all,
            "numeric_precip_field_occurrences": selected_station_records.get(hla_station_id, {}).get("numeric_precip_field_count", 0),
            "trace_precip_field_occurrences": selected_station_records.get(hla_station_id, {}).get("trace_precip_field_count", 0),
            "missing_precip_field_occurrences": selected_station_records.get(hla_station_id, {}).get("missing_precip_field_count", 0),
            "previous_gap_days_recovered": hla_recovered,
            "remaining_previous_gap_days": hla_remaining,
            "best_candidate_station": hla_station_id,
            "candidate_distance_km": 0,
            "same_day_overlap_days_with_local_D222": hla_d222_same.get("comparable_days", 0),
            "same_day_overlap_days_with_GHCN": hla_ghcn_same.get("comparable_days", 0),
            "local_D222_GHCN_numeric_overlap_days": len(set(hla_d222) & set(hla_ghcn)),
            "outcome": hla_status,
        },
        {
            "site": "FQA",
            "previous_unresolved_days": len(fqa_unresolved),
            "valid_isdlite_days_2013": best_valid_days,
            "previous_gap_days_recovered": fqa_recovered,
            "remaining_previous_gap_days": fqa_remaining,
            "best_candidate_station": best_id,
            "candidate_distance_km": best["distance_to_FQA_km"] if best else None,
            "same_day_overlap_days_with_local_D222": fqa_best_same.get("comparable_days", 0),
            "same_day_overlap_days_with_GHCN": len(set(daily_numeric_map(daily_by_station.get(best_id, []))) & set(fqa_ghcn_values)) if best else 0,
            "local_D222_GHCN_numeric_overlap_days": fqa_ghcn_same_day_numeric_overlap,
            "reference_only_gap_days": fqa_reference_only,
            "outcome": fqa_status,
        },
    ]
    write_csv(OUT / "gap_recovery_summary.csv", summary_rows, list(summary_rows[0].keys()))

    fqa_exact_file = bool(exact_file_matches)
    fqa_2013_sanity = "SUSPICIOUS"
    fqa_overall_provenance = "FAIL"
    gate = {
        "task": "021",
        "dataset": "NOAA_ISD_LITE_CHINA_2013",
        "created_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "HLA": {
            "station_identity": "PASS" if hla_identity_pass else "FAIL",
            "starting_unresolved_days": len(hla_gap_dates),
            "valid_isdlite_days_2013": hla_valid_all,
            "recovered_days": hla_recovered,
            "remaining_unresolved_days": hla_remaining,
            "isdlite_2013_status": hla_status,
            "rain_2013_sanity": "SUSPICIOUS" if hla_status != "ACCEPTED" else "NOT_COMPLETE",
            "rain_provenance": "PASS" if hla_remaining == 0 and hla_status == "ACCEPTED" else "PARTIAL",
            "ready_to_resume_010": False,
            "reason_not_ready": "HLA still has unresolved 2013 rain dates or an earlier 2004 incomplete-record blocker; 021 does not clear the full rain-source gate.",
        },
        "FQA": {
            "exact_station_found": fqa_exact_file,
            "best_candidate_station": best_id if best else "NONE",
            "best_candidate_distance_km": best["distance_to_FQA_km"] if best else None,
            "direct_fallback_distance_screen_km": DIRECT_FALLBACK_MAX_KM,
            "isdlite_2013_status": fqa_status,
            "starting_unresolved_2013_gap_days": len(fqa_unresolved),
            "recovered_2013_gap_days": fqa_recovered,
            "reference_only_2013_gap_days": fqa_reference_only,
            "remaining_2013_gap_days": fqa_remaining,
            "best_candidate_raw_2013_record_count": best.get("raw_2013_record_count") if best else 0,
            "best_candidate_numeric_precip_field_occurrences": best.get("numeric_precip_occurrences") if best else 0,
            "best_candidate_valid_daily_days": best_valid_days,
            "rain_2013_sanity": fqa_2013_sanity,
        "d222_2013_numeric_subtotal_mm": sum(fqa_d222.values()),
        "d222_2013_numeric_days": len(fqa_d222),
        "d222_2013_blank_days": 365 - len(fqa_d222),
        "nasa_2013_annual_total_mm": numeric(fqa_extreme_2013.get("nasa_annual_total_mm")),
        "ghcn_2013_valid_day_subtotal_mm": sum(fqa_ghcn_values.values()),
        "ghcn_2013_valid_days": len(fqa_ghcn_values),
        "local_D222_GHCN_numeric_overlap_days": fqa_ghcn_same_day_numeric_overlap,
            "overall_rain_provenance": fqa_overall_provenance,
            "weather_ready_for_wgen": False,
            "reason_not_ready": "FQA has no exact-name 2013 station file; nearest station is beyond the direct-fallback screen, 2013 D222 magnitude remains inconsistent with NOAA/NASA annual references, and the 2011-2012 provenance blocker is outside this 2013 package.",
        },
        "provenance": {
            "source": "NOAA/NCDC ISD-Lite; bundled region/year package repacked from public ISD-Lite files; not CMA daily data.",
            "ghcn_relationship": "NOAA-derived products; shared GTS/national source chain cannot be ruled out.",
            "d222_relationship": "CERN D222 local observation chain; local blanks were not coerced to zero.",
        },
        "wgen_dssat_ppo_executed": False,
    }
    write_json(OUT / "final_gate.json", gate)

    # Format note and Chinese decision report.
    format_note = f"""# NOAA/NCDC ISD-Lite 数据格式核验（任务 021）

## 来源身份

原始输入为 `{ZIP_PATH.name}`。随包 `_数据说明.docx` 说明其底层来源是 NOAA/NCDC 公开 ISD-Lite FTP，并由资料提供方按年份重打包中国及港澳台站点。项目目录 `cma_daily_weather` 是历史目录名；本 ZIP 不是 CMA 日值数据。

包内有 {len(members)} 个 2013 站点数据文件，ZIP 解压后总量约 {sum(item.file_size for item in members) / 1_000_000:.2f} MB。本审计只通过 ZIP 流式读取 HLA 和 FQA 最近 20 个中国候选站文件，没有整包解压。

## 时间戳与记录格式

- 站点文件名为 `USAF-WBAN-YYYY`；HLA 文件为 `507560-99999-2013`。
- 时间为 GMT/UTC。NCEI 技术说明指出 ISD-Lite 时间戳取最接近整点的观测时刻；站点实际记录频率不一定逐小时，随包说明称近年多为 3 小时、少数为 1 小时。
- 每行 61 个字符：年份列 1–4、月份 6–7、日期 9–10、小时 12–13；之后 8 个六字符气象字段连续排列。
- 降水 1 小时字段列 50–55，6 小时字段列 56–61；两者单位 mm，整数按 10 缩放，即原值除以 10。`-9999` 是缺测，`-1` 是 trace；trace 不赋任意毫米值。
- 随包 `isd-lite-format.txt` 有两处列范围与长度不一致：日期列把 9–11 写成长度 2，露点列把 20–24 写成长度 6。实际 61 字符行表明日期为 9–10、露点为 20–25；本审计按原始样例和固定宽度顺序解析，并校验日期和气象字段数。

## 累计窗和本地日

为可审计地处理重叠，本审计把观测时刻作为累计窗终点：1 小时字段表示 `[t−1h,t)`，6 小时字段表示 `[t−6h,t)`。随包格式文件说明了累计时长，但没有单列区间端点；该端点读法依据“在观测时刻测得累计深度”的 ISD 字段定义，作为本次审计假设明确记录。

所有日界转换到 Asia/Shanghai（UTC+08:00），目标日为本地 00:00–24:00。只有一组不重叠区间能完整覆盖 24 小时且所有有效路径给出唯一数值总量时，才输出有效日值；1 小时与 6 小时区间不相加重复计雨。跨本地午夜的区间不拆分。替代完整路径总量冲突，或完整路径依赖 trace 时，标为 unresolved。

## 参考资料

- 随包 `isd-lite-format.txt`、`isd-history.csv`、`isd-history.txt` 和 `_数据说明.docx`。
- [NCEI ISD-Lite Technical Document](https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/isd-lite-technical-document.pdf)：固定宽度子集、整点时间处理、缺测及 trace 编码、质量限制。
- [NCEI ISD-Lite Format](https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/isd-lite-format.pdf)：字段列、降水单位与缩放因子。
"""
    DOC_FORMAT.write_text(format_note, encoding="utf-8")

    hla_counts = Counter(row["daily_status"] for row in hla_daily)
    fqa_candidate_summary = "; ".join(
        f"{row['station_id']} {row['station_name']} ({row['distance_to_FQA_km']:.1f} km, {row['valid_reconstructed_days']} valid days)"
        for row in top20[:5]
    )
    fqa_event_support = sum(bool(row["isdlite_positive_subdaily_occurrence_within_pm1_day"]) for row in fqa_event_rows)
    fqa_annual_d222 = numeric(fqa_extreme_2013.get("d222_20_20_mm"))
    fqa_annual_nasa = numeric(fqa_extreme_2013.get("nasa_annual_total_mm"))
    fqa_annual_ghcn = sum(fqa_ghcn_values.values())
    hla_events_by_date = {row["date"]: row for row in hla_event_rows}
    hla_aug12 = hla_events_by_date.get("2013-08-12", {})
    hla_jul30 = hla_events_by_date.get("2013-07-30", {})
    max_fqa_event = max(fqa_event_rows, key=lambda row: row["d222_rain_mm"] or 0) if fqa_event_rows else {}
    hla_report = f"""# HLA / FQA 2013 ISD-Lite 降水来源审计（021）

## 结论

现有压缩包确属 NOAA/NCDC ISD-Lite 中国区域 2013 年包，不是 CMA 日值资料。HLA 同站 `507560-99999-2013` 存在，身份可与 014 的 full-ISD `507560-99999`、WMO `50756` 和 GHCN `CHM00050756` 闭合。按完整、不重叠且唯一总量的本地日窗口规则，HLA 2013 有效日数为 **{hla_valid_all}/365**，014 留下的 **111 天恢复 {hla_recovered} 天，仍 unresolved {hla_remaining} 天**。相较 014 的 full-ISD 路线，ISD-Lite 没有更容易形成合法日雨量：014 与本次都没有完整本地日；ISD-Lite 的 1h 降水字段全缺，6h 字段只提供零散窗口，且不含 full-ISD 的其他降水累计记录。它没有改变 HLA 降水来源门禁，也不具备恢复 010 的条件。

FQA 在 ISD history/2013 实际文件中**未找到封丘 exact-name 站**。根据现有 `CNFQ0701.WTH` 天气头坐标（{fqa_meta['latitude']:.3f}, {fqa_meta['longitude']:.3f}, {fqa_meta['elevation_m']:.0f} m），最近站为 **{best_id if best else 'NONE'} / {best['station_name'] if best else '无'}**，距 FQA **{best['distance_to_FQA_km']:.2f} km**。候选前五：{fqa_candidate_summary}。本审计将 50 km 作为直接日值 fallback 的保守空间筛选线；远站只作区域参考。当前正式缺口清单为前序 008/009 机读结果中的 {len(fqa_unresolved)} 天，ISD-Lite 通过已接受站恢复 **{fqa_recovered} 天**，参考站上可用但不能采纳的天数 **{fqa_reference_only} 天**，剩余缺口 **{fqa_remaining} 天**。

FQA 2013 D222 有明确数值的 35 天合计 **{sum(fqa_d222.values()):.1f} mm**，另有 330 天空白；这只是已记录日的 subtotal，空白没有当 0。NASA POWER 提供完整 365 天，总量 {fqa_annual_nasa:.2f} mm。所选 GHCN `CHM00057083` 只有 106 个有效日，记录值 subtotal 为 {fqa_annual_ghcn:.1f} mm，因此它不是完整年总量；它与 D222 明确数值日有 {fqa_ghcn_same_day_numeric_overlap} 天同日数值 overlap。最近 ISD-Lite 站对 D222 明确日的完整日值 overlap 为 {fqa_best_same.get('comparable_days', 0)} 天。六个 D222 ≥50 mm 事件中，最近站 ±1 日内有 {fqa_event_support} 个出现正子日观测；最大事件 {max_fqa_event.get('date', 'NA')} 的 D222 为 {numeric(max_fqa_event.get('d222_rain_mm')) or 0:.1f} mm，而最近站同日最大 6h 量为 {numeric(max_fqa_event.get('isdlite_max_6h_mm_same_day')) or 0:.1f} mm，NASA 为 {numeric(max_fqa_event.get('nasa_same_day_mm')) or 0:.2f} mm、GHCN 为 {numeric(max_fqa_event.get('ghcn_57083_same_day_mm')) or 0:.1f} mm。最近站距离约 84.7 km，只能提示区域降水事件，不能代表 FQA 点位或验证该日量级。FQA 2013 sanity 继续为 **SUSPICIOUS**；2011–2012 provenance 仍不在该 2013 包覆盖范围内。

## 逐站状态

### HLA

- `HLA_ISDLITE_STATION_IDENTITY = {'PASS' if hla_identity_pass else 'FAIL'}`；2013 同站文件存在，实际记录 {selected_station_records.get(hla_station_id, {}).get('raw_2013_record_count', 0)} 条。历史目录位置与 014 中 2013/ GHCN 位置差异已在 identity JSON 中保留，不用静态位置覆盖该差异。
- ISD-Lite 年内 daily 状态：`{json.dumps(dict(hla_counts), ensure_ascii=False)}`。同日可比较完整值：GHCN {hla_ghcn_same.get('comparable_days', 0)} 天，D222 明确数值日 {hla_d222_same.get('comparable_days', 0)} 天；若为 0，误差、相关和事件命中率均不可估计。
- `HLA_ISDLITE_2013_STATUS = {hla_status}`；`HLA_ISDLITE_RECOVERED_DAYS = {hla_recovered}`；`HLA_REMAINING_UNRESOLVED_DAYS = {hla_remaining}`。
- HLA 2013 有效的 1h 降水字段为 0 条；6h 字段中 211 条数值、29 条 trace、2,674 条缺测。五个重点日期中，7 月 30 日 D222=178.2 mm，ISD-Lite 最大完整窗为 39 mm/6h，GHCN=13.7 mm；8 月 12 日 D222=66.9 mm，但同站同一本地日内存在 79 mm/6h，GHCN=100.8 mm。7 月 21 日和 8 月 9 日子日观测支持降雨事件出现，但无法确认 D222 日量级；8 月 12 日的 6h 值还与 D222 日量级不一致。HLA 2013 climate sanity 仍为 SUSPICIOUS。
- `HLA_RAIN_PROVENANCE = {'PASS' if hla_remaining == 0 and hla_status == 'ACCEPTED' else 'PARTIAL'}`；`HLA_READY_TO_RESUME_010 = FALSE`。HLA 2004 连续空白 blocker 也仍需解决。

### FQA

- `FQA_EXACT_ISDLITE_STATION_FOUND = {str(fqa_exact_file).upper()}`。2013 实际中国站最近 20 个候选均列在 `fqa_station_candidates.csv`；HLA 与这些候选的所有降水字段 occurrence 都保存在 `precip_occurrences_2013.csv`。
- `FQA_BEST_ISDLITE_CANDIDATE = {best_id if best else 'NONE'}`，距离 {best['distance_to_FQA_km'] if best else 'NA'} km；最近站高程差 {((best['elevation_m'] - fqa_meta['elevation_m']) if best and best['elevation_m'] is not None else float('nan')):.1f} m。该站 8,744 条记录中 1h 降水字段全部缺测，只有 {best.get('numeric_precip_occurrences', 0) if best else 0} 个 1h/6h数值降水字段，无法完整覆盖任一本地日。站点空间、海拔、完整日数与 overlap QA 见 `fqa_candidate_spatial_qa.csv` 和 `fqa_candidate_overlap_qa.csv`。
- 最近站的完整日总量与 D222 有效 overlap 若不足，湿/干命中、MAE、RMSE、相关和累计差均保留为空值，不把缺测日当干日。D222 空白也未置 0。
- `FQA_ISDLITE_2013_STATUS = {fqa_status}`；`FQA_2013_RECOVERED_DAYS = {fqa_recovered}`；`FQA_2013_RAIN_SANITY = SUSPICIOUS`；`FQA_OVERALL_RAIN_PROVENANCE = FAIL`；`FQA_WEATHER_READY_FOR_WGEN = FALSE`。
- CERN D222 与旧 `HLLCYCFQ降雨数据.xls` 的值链在 009 中有闭合证据，但不视为独立观测。ISD-Lite 与 GHCN 同为 NOAA 产品，可能共享国家/GTS 输入链。最近站也有约 {best['distance_to_FQA_km'] if best else float('nan'):.1f} km 距离，因此只可提供 2013 区域辅助信息。D222 最大事件 {max_fqa_event.get('date', 'NA')} 的 204.0 mm 与远站、NASA、GHCN 的同日量级差异明显，说明事件发生不等于 magnitude 获得验证。它不能独立解除 FQA 2011/2012 的总量和 2012-09-02 240 mm 极端量级 blocker。

## 降水累计窗方法

ISD-Lite 时间戳按 UTC 解析并转换到 Asia/Shanghai。1h/6h值作为观测时刻前的累计窗口；先生成完整不重叠覆盖路径，再按唯一数值日总量决定 `VALID_COMPLETE_DAY` 或 `VALID_DRY_DAY`。同一时段上的 1h、6h值从不直接相加。trace、缺测、跨午夜、路径冲突和不完整覆盖分别保留 unresolved 状态。具体字段及窗口假设见 [格式说明](hl_fq_isdlite_data_format_021.md)。

## 最终 gate 与执行范围

`final_gate.json` 已记录 HLA、FQA 状态。没有运行 fitting weather、CLI、WGEN、DSSAT 或 PPO；没有修改 D222、原始压缩包、原始天气文件或现有实验结果。

机器可读汇总：

| 状态 | HLA | FQA |
|---|---|---|
| ISD-Lite 2013 | {hla_status} | {fqa_status} |
| 身份 / exact 站 | {'PASS' if hla_identity_pass else 'FAIL'} | {'TRUE' if fqa_exact_file else 'FALSE'} |
| 缺口起点 / 恢复 / 剩余 | {len(hla_gap_dates)} / {hla_recovered} / {hla_remaining} | {len(fqa_unresolved)} / {fqa_recovered} / {fqa_remaining} |
| Rain provenance | {'PASS' if hla_remaining == 0 and hla_status == 'ACCEPTED' else 'PARTIAL'} | {fqa_overall_provenance} |
| 可进入 WGEN 前序工作 | FALSE | FALSE |

## 结果文件

- `results/hl_fq_isdlite_021/input_provenance.json`、`dataset_identity.json`、`final_gate.json`
- HLA identity、逐日数据、GHCN/D222 overlap、111 天恢复和重点事件表
- FQA 最近 20 站、逐站逐日数据、空间/overlap QA、157 天恢复、2013 来源覆盖汇总和极端事件上下文表
- `results/hl_fq_isdlite_021/findings.md`
"""
    DOC_REPORT.write_text(hla_report, encoding="utf-8")
    (OUT / "findings.md").write_text(hla_report, encoding="utf-8")

    print(json.dumps({
        "out_dir": str(OUT),
        "hla_station_identity": hla_identity["status"],
        "hla_daily_status_counts": dict(hla_counts),
        "hla_recovered": hla_recovered,
        "fqa_exact_station_found": fqa_exact_file,
        "fqa_best_candidate": best_id,
        "fqa_best_distance_km": best["distance_to_FQA_km"] if best else None,
        "fqa_daily_valid_days_best_candidate": best_valid_days,
        "fqa_gap_count": len(fqa_unresolved),
        "fqa_recovered": fqa_recovered,
        "fqa_reference_only": fqa_reference_only,
        "fqa_remaining": fqa_remaining,
        "wgen_dssat_ppo_executed": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
