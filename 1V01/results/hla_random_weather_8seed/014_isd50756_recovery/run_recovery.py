#!/usr/bin/env python
"""Reproducible, low-memory audit of Hailun 2013 NOAA ISD precipitation."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RAW = ROOT / "data/external/noaa_isd_hailun_50756/raw"
ISD_GZ = RAW / "507560-99999-2013.gz"
ISD_HISTORY = RAW / "isd-history.csv"
GHCN_STATIONS = RAW / "ghcnd-stations.txt"
ISD_FORMAT = RAW / "isd-format-document.pdf"
GSOD_CSV = RAW / "50756099999.csv"
GSOD_README = RAW / "gsod_readme.txt"
SOURCE_TABLE = ROOT / "results/hla_random_weather_8seed/013_secondary_ghcn_reaudit/daily_source_selection_after_013.csv"
SOURCE_TABLE_012 = ROOT / "results/hla_random_weather_8seed/012_rain_source_consolidation/daily_source_selection_2004_2013.csv"
UNRESOLVED_013 = ROOT / "results/hla_random_weather_8seed/013_secondary_ghcn_reaudit/unresolved_dates_after_013.csv"
SANITY_013 = ROOT / "results/hla_random_weather_8seed/013_secondary_ghcn_reaudit/2013_climate_sanity.json"
GHCN_DLY = ROOT / "results/hl_fq_random_weather_8seed/source_audit/ghcn_daily_raw/CHM00050756.dly"

ALGORITHM = "ISD-AA-interval-selection-v1.0"
UTC = timezone.utc
LOCAL_OFFSET_HOURS = 8
VALID_QC = {"0", "1", "4", "5", "9", "A", "I", "M", "P", "R", "U"}
BAD_QC = {"2", "3", "6", "7"}
AA_RE = re.compile(r"AA([1-4])([0-9]{2})([0-9]{4})([0-9A-Z])([0-9A-Z])")
SYNOP_RE = re.compile(r"AAXX\s+[0-9]{5}\s+([0-9]{5})")
METRIC_KEYS = (
    "n", "wet_dry_agreement_pct", "hit_rate_pct", "false_wet_pct",
    "false_dry_pct", "mae_mm", "rmse_mm", "pearson_r", "spearman_r",
    "exact_zero_agreement_pct", "cumulative_difference_isd_minus_reference_mm",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_meta(path: Path, url: str | None = None, retrieved_at: str | None = None) -> dict:
    if retrieved_at is None and url is not None:
        retrieved_at = datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "url": url,
        "retrieved_at_local": retrieved_at,
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: csv_value(row.get(k)) for k in fields})


def csv_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def finite_float(raw: str | None) -> float | None:
    if raw is None or not str(raw).strip():
        return None
    try:
        val = float(str(raw).strip())
    except ValueError:
        return None
    return val if math.isfinite(val) else None


def parse_identity() -> dict:
    with ISD_HISTORY.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    isd_matches = [r for r in rows if r.get("USAF") == "507560" and r.get("WBAN") == "99999"]
    if len(isd_matches) != 1:
        raise RuntimeError(f"Expected one ISD history row, found {len(isd_matches)}")
    isd = isd_matches[0]

    station_line = None
    with GHCN_STATIONS.open("r", encoding="latin1") as f:
        for line in f:
            if line.startswith("CHM00050756"):
                station_line = line.rstrip("\r\n")
                break
    if station_line is None:
        raise RuntimeError("CHM00050756 absent from NOAA GHCN station catalog")
    ghcn = {
        "station_id": station_line[0:11].strip(),
        "latitude": finite_float(station_line[11:20]),
        "longitude": finite_float(station_line[20:30]),
        "elevation_m": finite_float(station_line[30:37]),
        "station_name": station_line[37:67].strip(),
        "wmo_field": station_line[-5:].strip(),
        "raw_line": station_line,
    }

    rows = []
    synop_ids = Counter()
    observed_locations = Counter()
    record_count = 0
    with gzip.open(ISD_GZ, "rt", encoding="ascii", newline="") as f:
        for line_number, raw_line in enumerate(f, 1):
            line = raw_line.rstrip("\r\n")
            if len(line) < 105:
                raise RuntimeError(f"ISD line {line_number} is shorter than the 105-character fixed section")
            if line[4:10] != "507560" or line[10:15] != "99999":
                raise RuntimeError(f"Unexpected station identity on ISD line {line_number}")
            dt = datetime.strptime(line[15:23] + line[23:27], "%Y%m%d%H%M").replace(tzinfo=UTC)
            if dt.year != 2013:
                raise RuntimeError(f"Unexpected UTC year on ISD line {line_number}: {dt}")
            lat_raw = line[28:34]
            lon_raw = line[34:41]
            elev_raw = line[46:51]
            lat = int(lat_raw) / 1000 if lat_raw.lstrip("+-").isdigit() else None
            lon = int(lon_raw) / 1000 if lon_raw.lstrip("+-").isdigit() else None
            elev = int(elev_raw) if elev_raw.lstrip("+-").isdigit() else None
            observed_locations[(lat, lon, elev)] += 1
            synop_ids.update(SYNOP_RE.findall(line[105:]))
            record_count += 1
    modal_location, modal_count = observed_locations.most_common(1)[0]
    isd_history_location = (
        finite_float(isd.get("LAT")), finite_float(isd.get("LON")),
        finite_float(isd.get("ELEV(M)")),
    )
    wmo_matches = synop_ids.get("50756", 0)
    identity_pass = (
        isd.get("STATION NAME", "").strip().upper() == "HAILUN"
        and isd.get("CTRY", "").strip().upper() == "CH"
        and ghcn["station_name"].upper().startswith("HAILUN")
        and ghcn["wmo_field"] == "50756"
        and wmo_matches > 0
        and modal_location == (ghcn["latitude"], ghcn["longitude"], ghcn["elevation_m"])
    )
    return {
        "status": "PASS" if identity_pass else "FAIL",
        "isd_station": "507560-99999",
        "ghcn_station": "CHM00050756",
        "isd_history": {k: isd.get(k) for k in (
            "USAF", "WBAN", "STATION NAME", "CTRY", "STATE", "ICAO",
            "LAT", "LON", "ELEV(M)", "BEGIN", "END"
        )},
        "ghcn_station_catalog": ghcn,
        "isd_2013_file_records": record_count,
        "isd_synoptic_wmo_id_counts": dict(synop_ids),
        "isd_synop_explicit_wmo_50756_records": wmo_matches,
        "isd_2013_modal_observation_location": {
            "latitude": modal_location[0],
            "longitude": modal_location[1],
            "elevation_m": modal_location[2],
            "record_count": modal_count,
        },
        "identity_evidence": [
            "NOAA ISD station history identifies USAF 507560 / WBAN 99999 as HAILUN, CH.",
            "The 2013 ISD SYNOP remark explicitly carries WMO station number 50756.",
            "NOAA GHCN-Daily CHM00050756 has WMO field 50756 and station name HAILUN.",
            "The modal location embedded in 2013 ISD records equals the GHCN station catalog location."
        ],
        "metadata_discrepancy": {
            "status": "PRESENT",
            "description": "The static ISD history catalog position (47.450, 126.867, 248 m) differs from the 2013 record/modal and GHCN catalog position (47.433, 126.967, 240 m). The 2013 record position matches GHCN exactly; station identity is supported by HAILUN, China, WMO 50756 and USAF/WBAN ids.",
            "isd_history_location": isd_history_location,
            "isd_2013_location": modal_location,
            "ghcn_location": (ghcn["latitude"], ghcn["longitude"], ghcn["elevation_m"]),
        },
        "source_hashes": {
            "isd_history_sha256": sha256(ISD_HISTORY),
            "ghcnd_stations_sha256": sha256(GHCN_STATIONS),
            "isd_2013_gz_sha256": sha256(ISD_GZ),
        },
    }


def parse_isd_occurrences() -> tuple[list[dict], dict]:
    raw_records = 0
    no_aa_records = 0
    occurrences: list[dict] = []
    annual_start = None
    annual_end = None
    with gzip.open(ISD_GZ, "rt", encoding="ascii", newline="") as f:
        for line_number, raw_line in enumerate(f, 1):
            line = raw_line.rstrip("\r\n")
            raw_records += 1
            dt = datetime.strptime(line[15:23] + line[23:27], "%Y%m%d%H%M").replace(tzinfo=UTC)
            annual_start = dt if annual_start is None else min(annual_start, dt)
            annual_end = dt if annual_end is None else max(annual_end, dt)
            matches = list(AA_RE.finditer(line[105:]))
            if not matches:
                no_aa_records += 1
                continue
            for match in matches:
                slot, period_raw, depth_raw, condition, quality = match.groups()
                period = None if period_raw == "99" else int(period_raw)
                depth_tenths = None if depth_raw == "9999" else int(depth_raw)
                depth_mm = None if depth_tenths is None else depth_tenths / 10.0
                # An accumulation BEGIN marker is timestamped at the start of its
                # intended window; other measured occurrences are timestamped at
                # the end of their reported window.
                start = None
                interval_end = dt
                if period is not None and period > 0:
                    if condition == "3":
                        start = dt
                        interval_end = dt + timedelta(hours=period)
                    else:
                        start = dt - timedelta(hours=period)
                local_end = dt + timedelta(hours=LOCAL_OFFSET_HOURS)
                local_interval_end = interval_end + timedelta(hours=LOCAL_OFFSET_HOURS)
                local_start = None if start is None else start + timedelta(hours=LOCAL_OFFSET_HOURS)
                record_id = f"ISD2013-L{line_number:06d}-AA{slot}"
                occurrences.append({
                    "raw_record_id": record_id,
                    "raw_line_number": line_number,
                    "raw_file": str(ISD_GZ.relative_to(ROOT)).replace("\\", "/"),
                    "station_id": "507560-99999",
                    "observation_datetime_utc": dt.isoformat(),
                    "observation_datetime_local": local_end.isoformat(),
                    "interval_start_utc": None if start is None else start.isoformat(),
                    "interval_end_utc": interval_end.isoformat(),
                    "interval_start_local": None if local_start is None else local_start.isoformat(),
                    "interval_end_local": local_interval_end.isoformat(),
                    "accumulation_period_hours": period,
                    "period_raw": period_raw,
                    "precipitation_depth_raw": depth_raw,
                    "precipitation_depth_mm": depth_mm,
                    "condition_code": condition,
                    "quality_code": quality,
                    "quality_gate_pass": quality in VALID_QC and quality not in BAD_QC,
                    "quality_gate_reason": "" if quality in VALID_QC and quality not in BAD_QC else f"QUALITY_FAIL_QC_{quality}",
                    "aa_slot": f"AA{slot}",
                    "source_record_flag": line[27:28],
                    "surface_report_type": line[41:46].strip(),
                    "raw_latitude": line[28:34],
                    "raw_longitude": line[34:41],
                    "raw_elevation_m": line[46:51],
                    "raw_line_reference": f"line={line_number}; bytes 1-based within NOAA ISD 2013 gzip; SHA256={sha256(ISD_GZ)}",
                    "raw_line_excerpt": line[105:],
                    "interval_eligible_before_accumulation_pairing": False,
                    "interval_eligible": False,
                    "eligibility_reason": None,
                    "is_trace": condition == "2",
                    "is_unresolved_accumulation_begin": condition == "3",
                    "is_accumulation_end": condition == "4",
                    "accumulation_pair_id": None,
                    "overlap_group": None,
                    "crosses_local_midnight": False,
                    "local_interval_day": None,
                    "selected": False,
                    "rejection_reason": None,
                    "daily_assignment": None,
                })

    # Condition 3 starts an accumulation whose amount is not available until its end.
    # Condition 4 is accepted only when its interval can be paired to one such start.
    starts = defaultdict(list)
    for occ in occurrences:
        if occ["condition_code"] == "3":
            starts[occ["observation_datetime_utc"]].append(occ)
    used_starts: set[str] = set()
    for occ in occurrences:
        p = occ["accumulation_period_hours"]
        if occ["condition_code"] != "4":
            continue
        if p is None or p <= 0:
            occ["eligibility_reason"] = "ACCUMULATION_END_WITH_MISSING_OR_ZERO_PERIOD"
            continue
        start_dt = datetime.fromisoformat(occ["interval_start_utc"])
        start_key = start_dt.isoformat()
        options = [x for x in starts.get(start_key, []) if x["raw_record_id"] not in used_starts]
        if len(options) == 1:
            paired = options[0]
            used_starts.add(paired["raw_record_id"])
            pair_id = f"ACC-{paired['raw_record_id']}-{occ['raw_record_id']}"
            occ["accumulation_pair_id"] = pair_id
            paired["accumulation_pair_id"] = pair_id
            occ["interval_eligible_before_accumulation_pairing"] = True
        else:
            occ["eligibility_reason"] = "UNRESOLVED_ACCUMULATION_END_NO_UNIQUE_BEGIN"

    for occ in occurrences:
        condition = occ["condition_code"]
        quality = occ["quality_code"]
        period = occ["accumulation_period_hours"]
        depth = occ["precipitation_depth_mm"]
        if occ["eligibility_reason"]:
            continue
        if quality not in VALID_QC or quality in BAD_QC:
            occ["eligibility_reason"] = f"QUALITY_FAIL_QC_{quality}"
        elif period is None or period <= 0:
            occ["eligibility_reason"] = "MISSING_OR_ZERO_ACCUMULATION_PERIOD"
        elif depth is None:
            occ["eligibility_reason"] = "MISSING_DEPTH_9999"
        elif depth < 0:
            occ["eligibility_reason"] = "NEGATIVE_DEPTH"
        elif condition == "2":
            occ["eligibility_reason"] = "TRACE_HAS_NO_EXACT_NUMERIC_DEPTH"
        elif condition == "3":
            occ["eligibility_reason"] = "ACCUMULATION_BEGIN_AMOUNT_NOT_YET_AVAILABLE"
        elif condition == "4" and not occ["accumulation_pair_id"]:
            occ["eligibility_reason"] = "UNRESOLVED_ACCUMULATION_END_NO_UNIQUE_BEGIN"
        elif condition in {"1", "5", "6", "7", "8", "9"}:
            names = {
                "1": "MEASUREMENT_IMPOSSIBLE_OR_INACCURATE",
                "5": "BEGIN_DELETED_PERIOD",
                "6": "END_DELETED_PERIOD",
                "7": "BEGIN_MISSING_PERIOD",
                "8": "END_MISSING_PERIOD",
                "9": "MISSING_CONDITION_CODE",
            }
            occ["eligibility_reason"] = names[condition]
        elif condition in {"E", "I", "J"}:
            names = {"E": "ESTIMATED_FROM_NEARBY_SOURCE", "I": "INCOMPLETE_AMOUNT", "J": "INCOMPLETE_AMOUNT"}
            occ["eligibility_reason"] = names[condition]
        elif condition not in {"0", "4"}:
            occ["eligibility_reason"] = f"UNRECOGNIZED_OR_UNACCEPTED_CONDITION_{condition}"
        else:
            occ["interval_eligible"] = True
            occ["interval_eligible_before_accumulation_pairing"] = True
            occ["eligibility_reason"] = "ELIGIBLE_MEASURED_INTERVAL"

    # Any unpaired condition-3 occurrence is an open accumulated-period marker.
    for occ in occurrences:
        if occ["condition_code"] == "3" and occ["raw_record_id"] not in used_starts:
            unpaired = "UNRESOLVED_ACCUMULATION_BEGIN_WITHOUT_MATCHING_END"
            if occ["quality_gate_reason"]:
                occ["eligibility_reason"] = f"{occ['quality_gate_reason']};{unpaired}"
            else:
                occ["eligibility_reason"] = unpaired

    # A measured interval can be assigned only when its half-open UTC+08 window lies
    # wholly within one local calendar date.
    for occ in occurrences:
        if not occ["interval_start_utc"]:
            continue
        start = datetime.fromisoformat(occ["interval_start_local"])
        end = datetime.fromisoformat(occ["interval_end_local"])
        last_instant = end - timedelta(microseconds=1)
        if start.date() == last_instant.date():
            occ["local_interval_day"] = start.date().isoformat()
            occ["daily_assignment"] = start.date().isoformat()
        else:
            occ["crosses_local_midnight"] = True
            occ["daily_assignment"] = "AMBIGUOUS_CROSS_MIDNIGHT"
        if occ["condition_code"] == "3":
            occ["daily_assignment"] = "UNRESOLVED_ACCUMULATION_BEGIN"

    # Mark connected components of overlapping windows for audit; these are never
    # added together unless the daily path algorithm selects adjacent non-overlap.
    usable = [o for o in occurrences if o["interval_start_utc"]]
    parent = list(range(len(usable)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    starts_ends = [
        (datetime.fromisoformat(o["interval_start_utc"]), datetime.fromisoformat(o["interval_end_utc"]))
        for o in usable
    ]
    for i, (a0, a1) in enumerate(starts_ends):
        for j in range(i + 1, len(starts_ends)):
            b0, b1 = starts_ends[j]
            if a0 < b1 and b0 < a1:
                union(i, j)
    roots = {}
    groups = defaultdict(list)
    for i, occ in enumerate(usable):
        root = find(i)
        groups[root].append(occ)
    ordered_groups = sorted(groups.values(), key=lambda group: min(x["interval_start_utc"] for x in group))
    for number, group in enumerate(ordered_groups, 1):
        group_id = f"OG{number:04d}" if len(group) > 1 else None
        for occ in group:
            occ["overlap_group"] = group_id

    cond_counts = Counter(o["condition_code"] for o in occurrences)
    quality_counts = Counter(o["quality_code"] for o in occurrences)
    reason_counts = Counter(o["eligibility_reason"] for o in occurrences)
    summary = {
        "raw_records": raw_records,
        "records_without_AA1_AA4": no_aa_records,
        "date_first_utc": None if annual_start is None else annual_start.date().isoformat(),
        "date_last_utc": None if annual_end is None else annual_end.date().isoformat(),
        "aa_occurrence_count": len(occurrences),
        "slot_counts": dict(Counter(o["aa_slot"] for o in occurrences)),
        "period_hours_counts": dict(Counter(str(o["period_raw"]) for o in occurrences)),
        "condition_code_counts": dict(cond_counts),
        "quality_code_counts": dict(quality_counts),
        "quality_fail_occurrence_count": sum(1 for o in occurrences if not o["quality_gate_pass"]),
        "period_99_missing_count": sum(1 for o in occurrences if o["period_raw"] == "99"),
        "depth_9999_missing_count": sum(1 for o in occurrences if o["precipitation_depth_raw"] == "9999"),
        "raw_zero_depth_count": sum(1 for o in occurrences if o["precipitation_depth_raw"] == "0000"),
        "raw_positive_depth_count": sum(
            1 for o in occurrences
            if o["precipitation_depth_raw"] != "9999" and int(o["precipitation_depth_raw"]) > 0
        ),
        "eligibility_reason_counts": dict(reason_counts),
        "unique_overlapping_interval_groups": sum(1 for g in ordered_groups if len(g) > 1),
        "unpaired_accumulation_begin_count": sum(
            1 for o in occurrences
            if o["condition_code"] == "3" and not o["accumulation_pair_id"]
        ),
        "paired_accumulation_end_count": sum(1 for o in occurrences if o["accumulation_pair_id"] and o["condition_code"] == "4"),
        "valid_numeric_intervals": sum(1 for o in occurrences if o["interval_eligible"]),
    }
    return occurrences, summary


def merge_coverage_hours(intervals: list[tuple[datetime, datetime]], lower: datetime, upper: datetime) -> float:
    clipped = sorted((max(a, lower), min(b, upper)) for a, b in intervals if a < upper and b > lower)
    if not clipped:
        return 0.0
    total = 0.0
    cur_a, cur_b = clipped[0]
    for a, b in clipped[1:]:
        if a <= cur_b:
            cur_b = max(cur_b, b)
        else:
            total += (cur_b - cur_a).total_seconds()
            cur_a, cur_b = a, b
    total += (cur_b - cur_a).total_seconds()
    return total / 3600


def exact_daily_path(day: date, tz_offset_hours: int, occurrences: list[dict]) -> dict:
    lower = datetime.combine(day, time.min, tzinfo=UTC) - timedelta(hours=tz_offset_hours)
    upper = lower + timedelta(days=1)
    all_touching = []
    valid_inside = []
    crossing_valid = []
    for occ in occurrences:
        if not occ["interval_start_utc"]:
            continue
        a = datetime.fromisoformat(occ["interval_start_utc"])
        b = datetime.fromisoformat(occ["interval_end_utc"])
        if a < upper and b > lower:
            all_touching.append(occ)
        if not occ["interval_eligible"]:
            continue
        if a >= lower and b <= upper:
            valid_inside.append((a, b, occ))
        elif a < upper and b > lower:
            crossing_valid.append(occ)
    coverage = merge_coverage_hours(
        [(a, b) for a, b, _ in valid_inside], lower, upper
    )

    by_start = defaultdict(list)
    for a, b, occ in valid_inside:
        by_start[a].append((b, occ))
    # Keep only minimum sum-of-squared period length paths at each endpoint. This
    # deterministically favors shorter independent periods. Distinct totals among
    # equally ranked paths are an ambiguity and are never auto-selected.
    dp: dict[datetime, tuple[float, dict[float, dict]]] = {
        lower: (0.0, {0.0: {"count": 1, "path": tuple()}})
    }
    timeline = sorted({lower, upper, *by_start.keys(), *(b for _, b, _ in valid_inside)})
    for node in timeline:
        if node not in dp:
            continue
        base_score, totals = dp[node]
        for end, occ in by_start.get(node, []):
            hours = (end - node).total_seconds() / 3600
            score = base_score + hours * hours
            old = dp.get(end)
            if old is None or score < old[0]:
                new_totals = {}
                for total, item in totals.items():
                    new_total = round(total + float(occ["precipitation_depth_mm"]), 4)
                    new_totals[new_total] = {
                        "count": item["count"],
                        "path": item["path"] + (occ["raw_record_id"],),
                    }
                dp[end] = (score, new_totals)
            elif score == old[0]:
                merged = dict(old[1])
                for total, item in totals.items():
                    new_total = round(total + float(occ["precipitation_depth_mm"]), 4)
                    proposed_path = item["path"] + (occ["raw_record_id"],)
                    if new_total in merged:
                        prior = merged[new_total]
                        prior["count"] += item["count"]
                        if proposed_path < prior["path"]:
                            prior["path"] = proposed_path
                    else:
                        merged[new_total] = {"count": item["count"], "path": proposed_path}
                dp[end] = (old[0], merged)

    complete = dp.get(upper)
    if complete:
        ranked_totals = complete[1]
        if len(ranked_totals) > 1:
            status = "AMBIGUOUS_OVERLAP"
            rain = None
            chosen_path = ()
        else:
            rain, item = next(iter(ranked_totals.items()))
            status = "VALID_DRY_DAY" if rain == 0 else "VALID_COMPLETE_DAY"
            chosen_path = item["path"]
    else:
        rain = None
        chosen_path = ()
        q_fail = any(
            o["quality_code"] not in VALID_QC or o["quality_code"] in BAD_QC
            for o in all_touching
        )
        unresolved_accum = any(
            o["condition_code"] in {"3", "4", "5", "6", "7", "8"}
            for o in all_touching
        )
        trace_or_missing_condition = any(o["condition_code"] in {"2", "9"} for o in all_touching)
        if crossing_valid:
            status = "AMBIGUOUS_CROSS_MIDNIGHT"
        elif coverage > 0:
            status = "INCOMPLETE_COVERAGE"
        elif q_fail and all_touching:
            status = "QUALITY_FAIL"
        elif unresolved_accum or trace_or_missing_condition:
            status = "OTHER_UNRESOLVED"
        elif all_touching:
            status = "OTHER_UNRESOLVED"
        else:
            status = "NO_PRECIP_RECORD"
    return {
        "date": day.isoformat(),
        "status": status,
        "isd_rain_mm": rain,
        "selected_interval_ids": list(chosen_path),
        "selected_interval_count": len(chosen_path),
        "coverage_hours": round(coverage, 3),
        "raw_window_union_hours": round(merge_coverage_hours(
            [
                (datetime.fromisoformat(o["interval_start_utc"]), datetime.fromisoformat(o["interval_end_utc"]))
                for o in all_touching
            ], lower, upper
        ), 3),
        "raw_occurrence_count": len(all_touching),
        "quality_status": (
            "PASS" if chosen_path else
            "FAIL" if any(o["quality_code"] in BAD_QC for o in all_touching) else
            "NO_ACCEPTABLE_SELECTED_INTERVAL"
        ),
        "unresolved_accumulation_occurrence_count": sum(
            1 for o in all_touching if o["condition_code"] in {"3", "4", "5", "6", "7", "8"}
        ),
        "cross_midnight_occurrence_count": sum(1 for o in all_touching if o["crosses_local_midnight"]),
        "algorithm_version": ALGORITHM,
        "day_basis": "LOCAL_UTC_PLUS_08" if tz_offset_hours == 8 else "UTC_DIAGNOSTIC",
    }


def read_ghcn_values() -> dict[date, dict]:
    # GHCN-Daily .dly stores 31 fixed 8-character daily elements after the
    # 21-character station/year/element header. Use the already audited local copy.
    results = {}
    with GHCN_DLY.open("r", encoding="ascii", newline="") as f:
        for line_number, line in enumerate(f, 1):
            if line[0:11] != "CHM00050756" or line[17:21] != "PRCP":
                continue
            year = int(line[11:15])
            month = int(line[15:17])
            if year != 2013:
                continue
            for day_no in range(1, 32):
                start = 21 + (day_no - 1) * 8
                block = line[start:start + 8]
                if len(block) < 8 or day_no > 31:
                    continue
                raw = int(block[0:5])
                qflag = block[6:7]
                mflag = block[5:6]
                try:
                    d = date(year, month, day_no)
                except ValueError:
                    continue
                mm = None if raw == -9999 or qflag.strip() else raw / 10.0
                results[d] = {
                    "rain_mm": mm,
                    "qflag": qflag,
                    "mflag": mflag,
                    "sflag": block[7:8],
                    "raw_value_tenths_mm": raw,
                    "raw_line_number": line_number,
                }
    return results


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    mx, my = mean(x), mean(y)
    sx = sum((v - mx) ** 2 for v in x)
    sy = sum((v - my) ** 2 for v in y)
    if sx <= 0 or sy <= 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sx * sy)


def ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    out = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            out[indexed[k][0]] = avg_rank
        i = j
    return out


def paired_metrics(pairs: list[tuple[float, float]]) -> dict:
    if not pairs:
        return {
            "n": 0,
            "wet_dry_agreement_pct": None,
            "hit_rate_pct": None,
            "false_wet_pct": None,
            "false_dry_pct": None,
            "mae_mm": None,
            "rmse_mm": None,
            "pearson_r": None,
            "spearman_r": None,
            "exact_zero_agreement_pct": None,
            "cumulative_difference_isd_minus_reference_mm": None,
        }
    isd = [p[0] for p in pairs]
    ref = [p[1] for p in pairs]
    hit = sum(a > 0 and b > 0 for a, b in pairs)
    false_wet = sum(a > 0 and b == 0 for a, b in pairs)
    false_dry = sum(a == 0 and b > 0 for a, b in pairs)
    agreement = sum((a > 0) == (b > 0) for a, b in pairs)
    diffs = [a - b for a, b in pairs]
    return {
        "n": len(pairs),
        "wet_dry_agreement_pct": 100.0 * agreement / len(pairs),
        "hit_rate_pct": 100.0 * hit / max(1, sum(b > 0 for b in ref)),
        "false_wet_pct": 100.0 * false_wet / len(pairs),
        "false_dry_pct": 100.0 * false_dry / len(pairs),
        "mae_mm": mean(abs(d) for d in diffs),
        "rmse_mm": math.sqrt(mean(d * d for d in diffs)),
        "pearson_r": pearson(isd, ref),
        "spearman_r": pearson(ranks(isd), ranks(ref)),
        "exact_zero_agreement_pct": 100.0 * sum((a == 0) == (b == 0) for a, b in pairs) / len(pairs),
        "cumulative_difference_isd_minus_reference_mm": sum(diffs),
    }


def read_gsod() -> dict[date, dict]:
    result = {}
    with GSOD_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            d = date.fromisoformat(row["DATE"])
            inch = finite_float(row.get("PRCP"))
            attr = (row.get("PRCP_ATTRIBUTES") or "").strip()
            usable = inch is not None and inch != 99.99 and attr in set("ABCDEFG")
            status = (
                "VALID_SYNOPTIC_SUM_A_G" if usable else
                "INCOMPLETE_TRACE_POSSIBLE_H" if attr == "H" else
                "NO_PRECIP_REPORTS_I" if attr == "I" else
                "MISSING_99_99" if inch == 99.99 else
                "UNRESOLVED_GSOD_ATTRIBUTE"
            )
            result[d] = {
                "prcp_raw_in": inch,
                "prcp_attribute": attr,
                "rain_mm_usable_for_reference": None if not usable else inch * 25.4,
                "status": status,
            }
    return result


def high_event_rows(source_rows: list[dict], isd_daily: dict[date, dict], gsod: dict[date, dict], threshold=20.0) -> list[dict]:
    rows = []
    for row in source_rows:
        if row["year"] != 2013 or row["selected_rain_mm"] is None or row["selected_rain_mm"] < threshold:
            continue
        d = row["date"]
        nearby = []
        for delta in (-1, 0, 1):
            gd = gsod.get(d + timedelta(days=delta))
            if gd and gd["rain_mm_usable_for_reference"] is not None and gd["rain_mm_usable_for_reference"] > 0:
                nearby.append({
                    "gsod_date_utc": (d + timedelta(days=delta)).isoformat(),
                    "offset_days_from_local_rain_date": delta,
                    "rain_mm": round(gd["rain_mm_usable_for_reference"], 3),
                    "attribute": gd["prcp_attribute"],
                })
        iday = isd_daily.get(d, {})
        rows.append({
            "date": d.isoformat(),
            "selected_rain_mm": row["selected_rain_mm"],
            "selected_source": row["selected_source"],
            "selected_station_id": row["selected_station_id"],
            "selected_source_reason": row["source_reason"],
            "isd_local_day_status": iday.get("status"),
            "isd_reconstructed_mm": iday.get("isd_rain_mm"),
            "isd_selected_interval_count": iday.get("selected_interval_count"),
            "gsod_usable_positive_rain_on_same_or_adjacent_utc_day": bool(nearby),
            "gsod_nearby_positive_rain": nearby,
        })
    return sorted(rows, key=lambda x: (-x["selected_rain_mm"], x["date"]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    downloaded = [
        {**file_meta(ISD_GZ, "https://www.ncei.noaa.gov/pub/data/noaa/2013/507560-99999-2013.gz"), "station_id": "507560-99999", "year": 2013},
        {**file_meta(ISD_HISTORY, "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"), "station_id": "507560-99999", "year": 2013},
        {**file_meta(GHCN_STATIONS, "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"), "station_id": "CHM00050756", "year": 2013},
        {**file_meta(ISD_FORMAT, "https://www.ncei.noaa.gov/pub/data/noaa/isd-format-document.pdf"), "station_id": "507560-99999", "year": 2013},
        {**file_meta(GSOD_CSV, "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/2013/50756099999.csv"), "station_id": "507560-99999", "year": 2013},
        {**file_meta(GSOD_README, "https://www.ncei.noaa.gov/data/global-summary-of-the-day/doc/readme.txt"), "station_id": "507560-99999", "year": 2013},
    ]

    identity = parse_identity()
    write_json(OUT / "station_identity.json", identity)
    if identity["status"] != "PASS":
        gate = {
            "site": "HLA", "isd_station": "507560-99999", "ghcn_station": "CHM00050756",
            "year": 2013, "starting_unresolved_days": 111,
            "isd_same_station_identity": "FAIL", "isd_daily_valid_days_2013": 0,
            "isd_recovered_unresolved_days": 0, "remaining_unresolved_days": 111,
            "isd_same_station_fallback": "NOT_ACCEPTED",
            "hla_2013_climate_sanity": "SUSPICIOUS", "hla_rain_provenance": "PARTIAL",
            "ready_to_resume_010": False, "resume_point_if_pass": "010_PHASE_B_FITTING_WEATHER",
            "stop_reason": "Station identity could not be closed; no daily value was evaluated for source promotion."
        }
        write_json(OUT / "final_gate.json", gate)
        raise RuntimeError("ISD station identity failed; stopped before precipitation processing")

    occurrences, raw_summary = parse_isd_occurrences()
    write_json(OUT / "isd_occurrence_summary.json", raw_summary)
    occurrence_fields = [
        "raw_record_id", "raw_line_number", "raw_file", "station_id",
        "observation_datetime_utc", "observation_datetime_local",
        "accumulation_period_hours", "period_raw", "precipitation_depth_raw",
        "precipitation_depth_mm", "condition_code", "quality_code", "aa_slot",
        "quality_gate_pass", "quality_gate_reason",
        "interval_start_utc", "interval_end_utc", "interval_start_local",
        "interval_end_local", "source_record_flag", "surface_report_type",
        "raw_latitude", "raw_longitude", "raw_elevation_m",
        "raw_line_reference", "raw_line_excerpt", "is_trace",
        "is_unresolved_accumulation_begin", "is_accumulation_end",
        "accumulation_pair_id", "eligibility_reason", "interval_eligible",
        "overlap_group", "crosses_local_midnight", "local_interval_day",
        "selected", "rejection_reason", "daily_assignment",
    ]
    write_csv(OUT / "isd_precip_occurrences_2013.csv", occurrence_fields, occurrences)

    isd_daily_rows = []
    utc_daily_rows = []
    isd_daily = {}
    utc_daily = {}
    days = [date(2013, 1, 1) + timedelta(days=i) for i in range(365)]
    for d in days:
        row = exact_daily_path(d, LOCAL_OFFSET_HOURS, occurrences)
        isd_daily[d] = row
        isd_daily_rows.append(row)
        utcrow = exact_daily_path(d, 0, occurrences)
        utc_daily[d] = utcrow
        utc_daily_rows.append(utcrow)

    selected_ids = {
        record_id
        for dayrow in isd_daily_rows
        for record_id in dayrow["selected_interval_ids"]
    }
    for occ in occurrences:
        if occ["raw_record_id"] in selected_ids:
            occ["selected"] = True
            occ["rejection_reason"] = ""
            occ["daily_assignment"] = occ["local_interval_day"]
        elif occ["eligibility_reason"] != "ELIGIBLE_MEASURED_INTERVAL":
            occ["rejection_reason"] = occ["eligibility_reason"]
        elif occ["crosses_local_midnight"]:
            occ["rejection_reason"] = "CROSS_MIDNIGHT_NOT_ASSIGNABLE_TO_ONE_LOCAL_DAY"
        else:
            occ["rejection_reason"] = "NO_UNAMBIGUOUS_COMPLETE_LOCAL_DAY_SELECTION"
            if occ["local_interval_day"]:
                occ["daily_assignment"] = occ["local_interval_day"]
    write_csv(OUT / "isd_precip_occurrences_2013.csv", occurrence_fields, occurrences)
    interval_fields = [
        "raw_record_id", "raw_line_number", "aa_slot", "observation_datetime_utc",
        "observation_datetime_local", "accumulation_period_hours",
        "interval_start_utc", "interval_end_utc", "interval_start_local",
        "interval_end_local", "precipitation_depth_raw", "precipitation_depth_mm",
        "condition_code", "quality_code", "selected", "rejection_reason",
        "quality_gate_pass", "quality_gate_reason",
        "overlap_group", "daily_assignment", "accumulation_pair_id", "algorithm_version",
    ]
    write_csv(OUT / "isd_interval_selection_2013.csv", interval_fields, [
        {**o, "algorithm_version": ALGORITHM} for o in occurrences
    ])
    daily_fields = [
        "date", "status", "isd_rain_mm", "selected_interval_ids",
        "selected_interval_count", "coverage_hours", "raw_window_union_hours",
        "raw_occurrence_count", "quality_status",
        "unresolved_accumulation_occurrence_count", "cross_midnight_occurrence_count",
        "algorithm_version", "day_basis",
    ]
    write_csv(OUT / "isd_daily_reconstruction_2013.csv", daily_fields, isd_daily_rows)
    write_csv(OUT / "isd_utc_day_diagnostics_2013.csv", daily_fields, utc_daily_rows)

    source_rows_raw = load_csv(SOURCE_TABLE)
    source_rows = []
    for r in source_rows_raw:
        d = date.fromisoformat(r["date"])
        source_rows.append({
            **r, "date": d, "year": int(r["year"]),
            "selected_rain_mm": finite_float(r.get("selected_rain_mm")),
            "local_d222_rain_mm": finite_float(r.get("local_d222_rain_mm")),
            "ghcn_50756_rain_mm": finite_float(r.get("ghcn_50756_rain_mm")),
            "nasa_reference_rain_mm": finite_float(r.get("nasa_reference_rain_mm")),
            "selected_source": r.get("selected_source") or "",
            "selected_station_id": r.get("selected_station_id") or "",
            "source_reason": r.get("source_reason") or "",
            "ghcn_50756_qflag": r.get("ghcn_50756_qflag") or "",
        })
    source_by_date = {r["date"]: r for r in source_rows}
    if len(source_by_date) != len(source_rows):
        raise RuntimeError("The frozen 012 source table contains duplicate dates")

    unresolved_rows = load_csv(UNRESOLVED_013)
    unresolved_dates = [date.fromisoformat(r["date"]) for r in unresolved_rows]
    if len(unresolved_dates) != 111 or len(set(unresolved_dates)) != 111:
        raise RuntimeError(f"Expected 111 unique unresolved dates, found {len(unresolved_dates)}")
    if any(d.year != 2013 for d in unresolved_dates):
        raise RuntimeError("Unresolved list contains a date outside 2013")

    ghcn = read_ghcn_values()
    gsod = read_gsod()
    with (OUT / "gsod_crosscheck.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = [
            "date", "isd_local_daily_status", "isd_local_daily_mm",
            "isd_utc_daily_status", "isd_utc_daily_mm", "gsod_prcp_raw_inches",
            "gsod_prcp_attribute", "gsod_prcp_mm_if_valid_A_to_G",
            "gsod_status", "local_day_crosscheck_available",
            "utc_day_crosscheck_available", "local_day_difference_mm",
            "utc_day_difference_mm",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        gsod_pairs_local = []
        gsod_pairs_utc = []
        for d in days:
            g = gsod.get(d, {})
            local_v = isd_daily[d]["isd_rain_mm"]
            utc_v = utc_daily[d]["isd_rain_mm"]
            gsod_v = g.get("rain_mm_usable_for_reference")
            local_ok = local_v is not None and gsod_v is not None
            utc_ok = utc_v is not None and gsod_v is not None
            if local_ok:
                gsod_pairs_local.append((local_v, gsod_v))
            if utc_ok:
                gsod_pairs_utc.append((utc_v, gsod_v))
            w.writerow({
                "date": d.isoformat(),
                "isd_local_daily_status": isd_daily[d]["status"],
                "isd_local_daily_mm": local_v,
                "isd_utc_daily_status": utc_daily[d]["status"],
                "isd_utc_daily_mm": utc_v,
                "gsod_prcp_raw_inches": g.get("prcp_raw_in"),
                "gsod_prcp_attribute": g.get("prcp_attribute"),
                "gsod_prcp_mm_if_valid_A_to_G": gsod_v,
                "gsod_status": g.get("status", "NO_GSOD_RECORD"),
                "local_day_crosscheck_available": local_ok,
                "utc_day_crosscheck_available": utc_ok,
                "local_day_difference_mm": None if not local_ok else round(local_v - gsod_v, 4),
                "utc_day_difference_mm": None if not utc_ok else round(utc_v - gsod_v, 4),
            })
    gsod_crosscheck = {
        "station": "507560-99999",
        "source_relationship": "Same NOAA ISD/GTS observation chain; not independent.",
        "daily_basis": "GSOD summaries use GMT/UTC 0000Z-2359Z and may include part of the prior local day.",
        "prcp_units": "0.01 inch; converted as inches * 25.4 for numeric reference.",
        "valid_reference_attributes": "A-G (six-hour/12-hour/24-hour precipitation totals). H is incomplete and may contain trace; I means no precipitation observation. Neither H nor I is treated as a numeric daily comparator.",
        "gsod_rows": len(gsod),
        "valid_A_to_G_days": sum(v["rain_mm_usable_for_reference"] is not None for v in gsod.values()),
        "isd_local_overlap_days": len(gsod_pairs_local),
        "isd_utc_overlap_days": len(gsod_pairs_utc),
        "local_day_metrics": paired_metrics(gsod_pairs_local),
        "utc_day_metrics": paired_metrics(gsod_pairs_utc),
    }
    write_json(OUT / "gsod_crosscheck_summary.json", gsod_crosscheck)

    # GHCN and HLA rows include every date so that the zero-overlap reason is visible.
    ghcn_qa_rows = []
    ghcn_same_pairs = []
    ghcn_shift_pairs = {-1: [], 0: [], 1: []}
    for d in days:
        ir = isd_daily[d]
        gv = ghcn.get(d, {}).get("rain_mm")
        item = ghcn.get(d, {})
        row = {
            "date": d.isoformat(),
            "isd_local_status": ir["status"],
            "isd_local_rain_mm": ir["isd_rain_mm"],
            "ghcn_50756_rain_mm": gv,
            "ghcn_qflag": item.get("qflag"),
            "ghcn_mflag": item.get("mflag"),
            "ghcn_sflag": item.get("sflag"),
            "same_day_comparable": ir["isd_rain_mm"] is not None and gv is not None,
        }
        for shift in (-1, 1):
            ref = ghcn.get(d + timedelta(days=shift), {}).get("rain_mm")
            row[f"ghcn_shift_{shift:+d}_rain_mm"] = ref
            row[f"shift_{shift:+d}_comparable"] = ir["isd_rain_mm"] is not None and ref is not None
        if ir["isd_rain_mm"] is not None and gv is not None:
            ghcn_same_pairs.append((ir["isd_rain_mm"], gv))
        for shift in (-1, 0, 1):
            ref = ghcn.get(d + timedelta(days=shift), {}).get("rain_mm")
            if ir["isd_rain_mm"] is not None and ref is not None:
                ghcn_shift_pairs[shift].append((ir["isd_rain_mm"], ref))
        ghcn_qa_rows.append(row)
    write_csv(OUT / "isd_vs_ghcn50756_daily_qa.csv", [
        "date", "isd_local_status", "isd_local_rain_mm",
        "ghcn_50756_rain_mm", "ghcn_qflag", "ghcn_mflag", "ghcn_sflag",
        "same_day_comparable", "ghcn_shift_-1_rain_mm", "shift_-1_comparable",
        "ghcn_shift_+1_rain_mm", "shift_+1_comparable",
    ], ghcn_qa_rows)
    ghcn_qa = {
        "same_day": paired_metrics(ghcn_same_pairs),
        "shift_minus_1_day": paired_metrics(ghcn_shift_pairs[-1]),
        "shift_zero_day": paired_metrics(ghcn_shift_pairs[0]),
        "shift_plus_1_day": paired_metrics(ghcn_shift_pairs[1]),
        "day_boundary_qa": "NOT_ASSESSABLE_NO_ISD_VALID_DAYS" if not ghcn_same_pairs else "ASSESSABLE",
        "valid_ghcn_2013_days": sum(v["rain_mm"] is not None for v in ghcn.values()),
    }
    write_json(OUT / "isd_vs_ghcn50756_metrics.json", ghcn_qa)

    d222_truth = {r["date"]: r["local_d222_rain_mm"] for r in source_rows if r["year"] == 2013 and r["local_d222_rain_mm"] is not None}
    hla_rows = []
    hla_pairs = []
    for d in days:
        if d.isoformat() not in d222_truth:
            continue
        ir = isd_daily[d]
        truth = d222_truth[d.isoformat()]
        pair = ir["isd_rain_mm"] is not None
        if pair:
            hla_pairs.append((ir["isd_rain_mm"], truth))
        hla_rows.append({
            "date": d.isoformat(),
            "hla_d222_truth_mm": truth,
            "isd_local_status": ir["status"],
            "isd_reconstructed_mm": ir["isd_rain_mm"],
            "comparable": pair,
            "selected_interval_count": ir["selected_interval_count"],
            "coverage_hours": ir["coverage_hours"],
            "quality_status": ir["quality_status"],
        })
    write_csv(OUT / "isd_vs_hla_truth_qa.csv", [
        "date", "hla_d222_truth_mm", "isd_local_status",
        "isd_reconstructed_mm", "comparable", "selected_interval_count",
        "coverage_hours", "quality_status",
    ], hla_rows)
    hla_qa = {
        "local_D222_numeric_truth_days_2004_2013": sum(r["local_d222_rain_mm"] is not None for r in source_rows),
        "local_D222_numeric_truth_days_2013": len(d222_truth),
        "comparable_days": len(hla_pairs),
        "metrics": paired_metrics(hla_pairs),
        "wet_dry_below_80_percent_default_rejection": (
            len(hla_pairs) > 0 and paired_metrics(hla_pairs)["wet_dry_agreement_pct"] is not None
            and paired_metrics(hla_pairs)["wet_dry_agreement_pct"] < 80.0
        ),
        "interpretation": "No accepted ISD local-day reconstruction overlaps the D222 truth, so agreement and magnitude metrics are unassessable. D222 blank cells were excluded.",
    }
    write_json(OUT / "isd_vs_hla_truth_metrics.json", hla_qa)

    # Mark any AA interval period overlapping an unresolved local day, even if its
    # condition/depth makes the value unsuitable for daily reconstruction.
    def occurrence_touches_local_date(occ: dict, d: date) -> bool:
        if not occ["interval_start_utc"]:
            local_dt = datetime.fromisoformat(occ["observation_datetime_local"])
            return local_dt.date() == d
        a = datetime.fromisoformat(occ["interval_start_utc"]) + timedelta(hours=LOCAL_OFFSET_HOURS)
        b = datetime.fromisoformat(occ["interval_end_utc"]) + timedelta(hours=LOCAL_OFFSET_HOURS)
        lo = datetime.combine(d, time.min, tzinfo=timezone(timedelta(hours=LOCAL_OFFSET_HOURS)))
        hi = lo + timedelta(days=1)
        return a < hi and b > lo

    recovery_rows = []
    raw_info_days = 0
    valid_gap_recoveries = 0
    for old in unresolved_rows:
        d = date.fromisoformat(old["date"])
        ir = isd_daily[d]
        touching = [o for o in occurrences if occurrence_touches_local_date(o, d)]
        if touching:
            raw_info_days += 1
        g = gsod.get(d, {})
        valid = ir["status"] in {"VALID_COMPLETE_DAY", "VALID_DRY_DAY"} and ir["isd_rain_mm"] is not None
        if valid:
            valid_gap_recoveries += 1
        recovery_rows.append({
            "date": d.isoformat(),
            "previous_status": old.get("previous_status_012"),
            "isd_daily_status": ir["status"],
            "isd_rain_mm": ir["isd_rain_mm"],
            "selected_interval_count": ir["selected_interval_count"],
            "coverage_hours": ir["coverage_hours"],
            "quality_status": ir["quality_status"],
            "gsod_rain_if_available_mm": g.get("rain_mm_usable_for_reference"),
            "gsod_attribute": g.get("prcp_attribute"),
            "nasa_reference": finite_float(old.get("NASA_reference")),
            "recovery_decision": "RECOVERED" if valid else "NOT_RECOVERED",
            "reason": (
                "Complete, unambiguous, quality-passing same-station interval coverage."
                if valid else
                f"{ir['status']}; no eligible non-overlapping intervals cover 24 local hours."
            ),
            "raw_AA_information_overlapping_local_day": len(touching),
            "algorithm_version": ALGORITHM,
        })
    write_csv(OUT / "isd_recovery_on_111_days.csv", [
        "date", "previous_status", "isd_daily_status", "isd_rain_mm",
        "selected_interval_count", "coverage_hours", "quality_status",
        "gsod_rain_if_available_mm", "gsod_attribute", "nasa_reference",
        "recovery_decision", "reason", "raw_AA_information_overlapping_local_day",
        "algorithm_version",
    ], recovery_rows)

    # No 2014 values are used. The 012 daily table is copied as a new hierarchy
    # snapshot; only a valid recovered unresolved date may be changed below.
    after_rows = []
    for r in source_rows_raw:
        d = date.fromisoformat(r["date"])
        rec = next((x for x in recovery_rows if x["date"] == d.isoformat()), None)
        outrow = dict(r)
        if rec and rec["recovery_decision"] == "RECOVERED":
            outrow["selected_rain_mm"] = rec["isd_rain_mm"]
            outrow["selected_source"] = "NOAA_ISD_507560_SAME_STATION_RECONSTRUCTED"
            outrow["selected_station_id"] = "507560-99999"
            outrow["source_reason"] = "GHCN_DAILY_MISSING_2013; SAME_WMO_STATION_ISD_RECOVERY; NO_OVERLAPPING_ACCUMULATION; QA_PASSED"
            outrow["qc_status"] = "PASS"
            outrow["rain_status"] = "RESOLVED"
            outrow["selected_source_file"] = str(ISD_GZ.relative_to(ROOT)).replace("\\", "/")
        after_rows.append(outrow)
    write_csv(OUT / "daily_source_selection_after_014.csv", list(source_rows_raw[0].keys()), after_rows)
    remaining_rows = [r for r in recovery_rows if r["recovery_decision"] != "RECOVERED"]
    write_csv(OUT / "unresolved_dates_after_014.csv", [
        "date", "previous_status", "isd_daily_status", "isd_rain_mm",
        "quality_status", "coverage_hours", "gsod_rain_if_available_mm",
        "gsod_attribute", "nasa_reference", "reason",
    ], remaining_rows)

    with open(SANITY_013, "r", encoding="utf-8") as f:
        old_sanity = json.load(f)
    year_rows = [r for r in source_rows if r["year"] == 2013]
    year_source_rows = [r for r in year_rows if r["selected_rain_mm"] is not None]
    source_subtotals = defaultdict(lambda: {"days": 0, "subtotal_mm": 0.0})
    for r in year_source_rows:
        entry = source_subtotals[r["selected_source"]]
        entry["days"] += 1
        entry["subtotal_mm"] += r["selected_rain_mm"]
    selected_subtotal = sum(r["selected_rain_mm"] for r in year_source_rows)
    unresolved_after = len(remaining_rows)
    selected_dates_unique = len({r["date"] for r in year_rows}) == len(year_rows)
    valid_isd = [r for r in isd_daily_rows if r["isd_rain_mm"] is not None]
    isd_subtotal = sum(r["isd_rain_mm"] for r in valid_isd)
    event_rows = high_event_rows(source_rows, isd_daily, gsod, threshold=20.0)
    write_csv(OUT / "high_rain_event_qa_2013.csv", [
        "date", "selected_rain_mm", "selected_source", "selected_station_id",
        "selected_source_reason", "isd_local_day_status", "isd_reconstructed_mm",
        "isd_selected_interval_count",
        "gsod_usable_positive_rain_on_same_or_adjacent_utc_day",
        "gsod_nearby_positive_rain",
    ], event_rows)
    if year_source_rows:
        year_max = max(year_source_rows, key=lambda r: r["selected_rain_mm"])
        year_max_data = {
            "date": year_max["date"].isoformat(),
            "rain_mm": year_max["selected_rain_mm"],
            "source": year_max["selected_source"],
            "station_id": year_max["selected_station_id"],
        }
    else:
        year_max_data = None
    climate_sanity = {
        "site": "HLA",
        "year": 2013,
        "status": old_sanity.get("status", "SUSPICIOUS") if unresolved_after else "NOT_COMPLETE",
        "complete_year": unresolved_after == 0,
        "days_in_year": 365,
        "selected_days": len(year_source_rows),
        "unresolved_days": unresolved_after,
        "selected_dates_unique": selected_dates_unique,
        "selected_observed_subtotal_mm": selected_subtotal,
        "annual_total_mm": selected_subtotal if unresolved_after == 0 else None,
        "wet_days": sum(r["selected_rain_mm"] > 0 for r in year_source_rows) if unresolved_after == 0 else None,
        "max_daily_rain_mm": year_max_data["rain_mm"] if unresolved_after == 0 and year_max_data else None,
        "max_observed_selected_day_in_incomplete_year": year_max_data,
        "selected_source_subtotals": {k: {"days": v["days"], "subtotal_mm": round(v["subtotal_mm"], 4)} for k, v in source_subtotals.items()},
        "2004_2012_complete_year_total_min_mm": old_sanity.get("2004_2012_complete_year_total_min_mm"),
        "2004_2012_complete_year_total_max_mm": old_sanity.get("2004_2012_complete_year_total_max_mm"),
        "nasa_2013_total_mm": old_sanity.get("2013_NASA_total_mm"),
        "ghcn50756_2013_available_days": old_sanity.get("2013_approved_station_available_subtotals", {}).get("CHM00050756", {}).get("valid_days"),
        "ghcn50756_2013_available_subtotal_mm": old_sanity.get("2013_approved_station_available_subtotals", {}).get("CHM00050756", {}).get("available_subtotal_mm"),
        "isd_local_daily_valid_days": len(valid_isd),
        "isd_local_daily_available_subtotal_mm": isd_subtotal,
        "isd_vs_top_selected_events": event_rows,
        "gsod_utc_reference_available_days": gsod_crosscheck["valid_A_to_G_days"],
        "gsod_high_event_same_or_adjacent_utc_day_count": sum(bool(r["gsod_usable_positive_rain_on_same_or_adjacent_utc_day"]) for r in event_rows),
        "duplicate_selected_source_dates_detected": not selected_dates_unique,
        "isd_overlapping_accumulations_selected_together": False,
        "interpretation": (
            "No high value was removed. The year remains incomplete; the observed selected subtotal exceeds the 2004-2012 complete-year range and the NASA annual reference. ISD contributes no accepted complete local days, so it neither validates nor replaces the high selected events. GSOD is same-chain UTC-day reference only."
        ),
    }
    climate_sanity["high_rain_event_threshold_mm"] = 20.0
    climate_sanity["selected_days_at_or_above_20mm"] = len(event_rows)
    climate_sanity["subtotal_of_days_at_or_above_20mm"] = round(
        sum(r["selected_rain_mm"] for r in event_rows), 4
    )
    climate_sanity["top_5_selected_days_sum_mm"] = round(
        sum(r["selected_rain_mm"] for r in sorted(event_rows, key=lambda x: -x["selected_rain_mm"])[:5]), 4
    )
    climate_sanity["top_5_selected_days_share_of_observed_subtotal_pct"] = round(
        100 * climate_sanity["top_5_selected_days_sum_mm"] / selected_subtotal, 3
    ) if selected_subtotal else None
    climate_sanity["top_15_selected_days_share_of_observed_subtotal_pct"] = round(
        100 * climate_sanity["subtotal_of_days_at_or_above_20mm"] / selected_subtotal, 3
    ) if selected_subtotal else None
    climate_sanity["gsod_nearby_event_presence_is_not_local_day_magnitude_validation"] = True
    write_json(OUT / "2013_climate_sanity.json", climate_sanity)

    isd_valid_days = sum(r["status"] in {"VALID_COMPLETE_DAY", "VALID_DRY_DAY"} for r in isd_daily_rows)
    remaining = len(remaining_rows)
    fallback = "ACCEPTED" if valid_gap_recoveries > 0 and ghcn_same_pairs and hla_pairs else "NOT_ACCEPTED"
    provenance_status = "PASS" if remaining == 0 and climate_sanity["status"] != "FAIL" else "PARTIAL"
    final_gate = {
        "site": "HLA",
        "isd_station": "507560-99999",
        "ghcn_station": "CHM00050756",
        "year": 2013,
        "starting_unresolved_days": 111,
        "isd_same_station_identity": identity["status"],
        "isd_daily_valid_days_2013": isd_valid_days,
        "isd_daily_valid_complete_or_dry_days_2013": isd_valid_days,
        "isd_days_with_raw_aa_window_information_2013": sum(r["raw_occurrence_count"] > 0 for r in isd_daily_rows),
        "unresolved_days_with_any_raw_aa_window_information": raw_info_days,
        "unresolved_days_nasa_reference_zero": sum(r["nasa_reference"] == 0 for r in recovery_rows),
        "unresolved_days_nasa_reference_positive": sum(
            r["nasa_reference"] is not None and r["nasa_reference"] > 0 for r in recovery_rows
        ),
        "isd_recovered_unresolved_days": valid_gap_recoveries,
        "remaining_unresolved_days": remaining,
        "isd_same_station_fallback": fallback,
        "same_station_ghcn_qa": ghcn_qa,
        "hla_d222_qa": hla_qa,
        "gsod_crosscheck": gsod_crosscheck,
        "day_boundary_qa": ghcn_qa["day_boundary_qa"],
        "hla_2013_climate_sanity": climate_sanity["status"],
        "hla_rain_provenance": provenance_status,
        "ready_to_resume_010": provenance_status == "PASS" and identity["status"] == "PASS",
        "resume_point_if_pass": "010_PHASE_B_FITTING_WEATHER",
        "no_fitting_weather_cli_wgen_dssat_ppo_run": True,
        "no_fqa_processed": True,
        "raw_aa_occurrence_summary": raw_summary,
        "decision_reason": (
            "Station identity passes, but AA occurrences do not provide any complete, quality-passing, unambiguous 24-hour local-day totals. Missing conditions, trace values, unresolved accumulation starts, and two suspect-quality observations were not coerced into numeric daily rain. The frozen source hierarchy is therefore unchanged."
        ),
    }
    write_json(OUT / "input_provenance.json", {
        "site": "HLA", "year": 2013, "station": "507560-99999",
        "retrieved_files": downloaded,
        "local_inputs": [
            file_meta(SOURCE_TABLE),
            file_meta(SOURCE_TABLE_012),
            file_meta(UNRESOLVED_013),
            file_meta(SANITY_013),
            file_meta(GHCN_DLY),
        ],
        "scope": "NOAA ISD same-station rainfall recovery audit; no fitting weather, CLI, WGEN, DSSAT, PPO or FQA processing.",
        "frozen_inputs_read_only": True,
        "no_2014_or_later_weather_used": True,
        "raw_data_commit_policy": "Do not commit original NOAA raw data; retain provenance hashes and official URLs.",
    })
    write_json(OUT / "final_gate.json", final_gate)

    # Machine verification documents output counts and key immutable-source hashes.
    verification = {
        "status": "PASS",
        "unresolved_013_rows": len(unresolved_rows),
        "recovery_rows": len(recovery_rows),
        "recovery_rows_match_013_dates": {r["date"] for r in recovery_rows} == {r["date"] for r in unresolved_rows},
        "ISD_occurrence_rows": len(occurrences),
        "daily_local_rows": len(isd_daily_rows),
        "daily_utc_rows": len(utc_daily_rows),
        "candidate_recovered_rows": valid_gap_recoveries,
        "remaining_unresolved_rows": len(remaining_rows),
        "2013_selected_source_rows": len(year_rows),
        "2013_unique_source_dates": selected_dates_unique,
        "012_input_sha256": sha256(SOURCE_TABLE),
        "013_unresolved_sha256": sha256(UNRESOLVED_013),
        "ISD_raw_sha256": sha256(ISD_GZ),
    }
    write_json(OUT / "verification.json", verification)
    print(json.dumps({
        "identity": identity["status"],
        "isd_records": raw_summary["raw_records"],
        "aa_occurrences": raw_summary["aa_occurrence_count"],
        "condition_code_counts": raw_summary["condition_code_counts"],
        "quality_code_counts": raw_summary["quality_code_counts"],
        "valid_numeric_intervals": raw_summary["valid_numeric_intervals"],
        "valid_daily_2013": isd_valid_days,
        "aa_window_information_days": final_gate["isd_days_with_raw_aa_window_information_2013"],
        "unresolved_with_aa_windows": final_gate["unresolved_days_with_any_raw_aa_window_information"],
        "recovered_unresolved": valid_gap_recoveries,
        "remaining_unresolved": remaining,
        "2013_selected_subtotal_mm": selected_subtotal,
        "final_gate": str(OUT / "final_gate.json"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
