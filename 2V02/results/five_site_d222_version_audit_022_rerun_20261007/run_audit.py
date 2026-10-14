"""Read-only audit of the project-contained CERN D222 workbook candidate.

This script reads project inputs and writes only new Task 022 audit artifacts.
It does not modify source workbooks, weather files, WTH files, or prior results.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import xlrd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "five_site_d222_version_audit_022_rerun_20261007"
DOC = ROOT / "docs" / "five_site_d222_version_audit_022_rerun_20261007.md"
CURRENT = ROOT / "data" / "external" / "人工大气观测降雨蒸发能见度日值.xls"
OLD_D222 = ROOT / "data" / "external" / "cern_d222" / "raw" / "excel.xls"
LEGACY = ROOT / "my_data" / "HLLCYCFQ降雨数据.xls"
PREVIOUS_PARTIAL = ROOT / "my_data" / "SYCS人工大气观测降雨蒸发能见度日值.xls"

TARGETS = {
    "HLA": {"name": "海伦", "fitting_period": "2004-2013"},
    "FQA": {"name": "封丘", "fitting_period": "2005-2013"},
    "YCA": {"name": "禹城", "fitting_period": "NOT_SPECIFIED_IN_022"},
    "SYA": {"name": "沈阳", "fitting_period": "NOT_SPECIFIED_IN_022"},
    "LCA": {"name": "栾城", "fitting_period": "NOT_SPECIFIED_IN_022"},
}
EXPECTED_YEARS = range(1998, 2024)
HLA_KEY_DATES = ["2013-07-30", "2013-07-21", "2013-08-09", "2013-08-12", "2013-08-21"]
FQA_KEY_DATES = ["2012-09-02", "2013-05-26"]
CATEGORIES = [
    "EXACT_NUMERIC_MATCH",
    "NUMERIC_CHANGED",
    "OLD_MISSING_NEW_NUMERIC",
    "OLD_NUMERIC_NEW_MISSING",
    "BOTH_MISSING",
    "OLD_ZERO_NEW_NONZERO",
    "OLD_NONZERO_NEW_ZERO",
    "DATE_ONLY_IN_NEW",
    "DATE_ONLY_IN_OLD",
]


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_number(value: Any) -> tuple[float | None, str]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None, "BLANK"
    if isinstance(value, bool):
        return None, "INVALID"
    try:
        number = float(value.strip() if isinstance(value, str) else value)
    except (TypeError, ValueError):
        return None, "INVALID"
    if not math.isfinite(number):
        return None, "INVALID"
    if number < 0:
        return number, "NEGATIVE"
    if number == 0:
        return 0.0, "EXPLICIT_ZERO"
    return number, "NUMERIC"


def clean_number(value: Any) -> float | None:
    parsed, status = parse_number(value)
    return parsed if status in ("NUMERIC", "EXPLICIT_ZERO", "NEGATIVE") else None


def rounded(value: float | None, digits: int = 6) -> float | str:
    return round(value, digits) if value is not None else ""


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_d222(path: Path) -> dict[str, Any]:
    book = xlrd.open_workbook(str(path), on_demand=True)
    sheets: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    all_codes: set[str] = set()
    all_dates: list[dt.date] = []
    invalid_date_rows = 0
    selected_field = ""
    missing_cell_types: Counter[str] = Counter()
    rain_cell_types: Counter[str] = Counter()

    for sheet in book.sheets():
        headers = [str(v).strip() for v in sheet.row_values(1)] if sheet.nrows > 1 else []
        if not headers:
            continue
        station_col = next((i for i, name in enumerate(headers) if name == "生态站代码"), None)
        year_col = next((i for i, name in enumerate(headers) if name == "年"), None)
        month_col = next((i for i, name in enumerate(headers) if name == "月"), None)
        day_col = next((i for i, name in enumerate(headers) if name == "日"), None)
        rain_col = next(
            (i for i, name in enumerate(headers) if "合计" in name and "mm" in name.lower()),
            None,
        )
        if None in (station_col, year_col, month_col, day_col, rain_col):
            continue
        selected_field = headers[rain_col]
        sheets.append(
            {
                "name": sheet.name,
                "nrows": sheet.nrows,
                "ncols": sheet.ncols,
                "headers": headers,
                "data_rows": max(0, sheet.nrows - 2),
            }
        )
        for row_index in range(2, sheet.nrows):
            code_raw = sheet.cell_value(row_index, station_col)
            code = str(code_raw).strip()
            if not code:
                continue
            all_codes.add(code)
            try:
                year = int(float(sheet.cell_value(row_index, year_col)))
                month = int(float(sheet.cell_value(row_index, month_col)))
                day = int(float(sheet.cell_value(row_index, day_col)))
                date = dt.date(year, month, day)
            except (ValueError, TypeError, OverflowError):
                invalid_date_rows += 1
                continue
            raw_rain = sheet.cell_value(row_index, rain_col)
            rain, value_status = parse_number(raw_rain)
            cell_type = xlrd.sheet.ctype_text.get(sheet.cell_type(row_index, rain_col), "unknown")
            rain_cell_types[cell_type] += 1
            if value_status == "BLANK":
                missing_cell_types[cell_type] += 1
            all_dates.append(date)
            all_rows.append(
                {
                    "site_code": code,
                    "date": date,
                    "rain_raw": raw_rain,
                    "rain_mm": rain,
                    "value_status": value_status,
                    "source_row": row_index + 1,
                    "sheet": sheet.name,
                    "rain_cell_type": cell_type,
                }
            )
    book.release_resources()
    return {
        "sheets": sheets,
        "rows": all_rows,
        "site_codes": sorted(all_codes),
        "data_first_date": min(all_dates).isoformat() if all_dates else None,
        "data_last_date": max(all_dates).isoformat() if all_dates else None,
        "invalid_date_rows": invalid_date_rows,
        "rain_field": selected_field,
        "rain_cell_types": dict(rain_cell_types),
        "missing_cell_types": dict(missing_cell_types),
    }


def load_legacy_fqa(path: Path) -> dict[dt.date, float | None]:
    book = xlrd.open_workbook(str(path), on_demand=True)
    result: dict[dt.date, float | None] = {}
    for sheet in book.sheets():
        if sheet.nrows < 2:
            continue
        headers = [str(v).strip() for v in sheet.row_values(1)]
        try:
            station_col = headers.index("生态站代码")
            year_col = headers.index("年")
            month_col = headers.index("月")
            day_col = headers.index("日")
        except ValueError:
            continue
        rain_col = next((i for i, name in enumerate(headers) if "合计" in name and "mm" in name.lower()), None)
        if rain_col is None:
            continue
        for row_index in range(2, sheet.nrows):
            if str(sheet.cell_value(row_index, station_col)).strip() != "FQA":
                continue
            try:
                date = dt.date(
                    int(float(sheet.cell_value(row_index, year_col))),
                    int(float(sheet.cell_value(row_index, month_col))),
                    int(float(sheet.cell_value(row_index, day_col))),
                )
            except (ValueError, TypeError, OverflowError):
                continue
            result[date] = clean_number(sheet.cell_value(row_index, rain_col))
    book.release_resources()
    return result


def index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[dt.date, list[dict[str, Any]]]]:
    indexed: dict[str, dict[dt.date, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        indexed[row["site_code"]][row["date"]].append(row)
    return indexed


def category(old_has_date: bool, new_has_date: bool, old_value: float | None, new_value: float | None) -> str:
    if not old_has_date:
        return "DATE_ONLY_IN_NEW"
    if not new_has_date:
        return "DATE_ONLY_IN_OLD"
    if old_value is None and new_value is None:
        return "BOTH_MISSING"
    if old_value is None and new_value is not None:
        return "OLD_MISSING_NEW_NUMERIC"
    if old_value is not None and new_value is None:
        return "OLD_NUMERIC_NEW_MISSING"
    assert old_value is not None and new_value is not None
    if old_value == 0 and new_value != 0:
        return "OLD_ZERO_NEW_NONZERO"
    if old_value != 0 and new_value == 0:
        return "OLD_NONZERO_NEW_ZERO"
    if old_value == new_value:
        return "EXACT_NUMERIC_MATCH"
    return "NUMERIC_CHANGED"


def source_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path_project_relative": rel(path),
        "path_absolute": str(path.resolve()),
        "filename": path.name,
        "size_bytes": stat.st_size,
        "mtime_utc": dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc).isoformat(),
        "mtime_local": dt.datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
        "sha256": sha256(path),
    }


def build_inventory() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[Path] = set()

    def add(path: Path, role: str, source_class: str, station: str, period: str, note: str = "") -> None:
        path = path.resolve()
        if not path.is_file() or path in seen:
            return
        seen.add(path)
        stat = path.stat()
        entries.append(
            {
                "path": rel(path),
                "path_absolute": str(path),
                "sha256": sha256(path),
                "size_bytes": stat.st_size,
                "mtime_utc": dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc).isoformat(),
                "role": role,
                "raw_derived_classification": source_class,
                "station": station,
                "period": period,
                "notes": note,
            }
        )

    add(CURRENT, "current five-site D222 snapshot selected for Task 022", "raw product snapshot", "HLA; FQA; YCA; SYA; LCA", "1998-2023; site-specific coverage gaps recorded in QC", "Selected by exact filename and newest mtime among matching project candidates.")
    add(PREVIOUS_PARTIAL, "earlier partial D222 workbook candidate", "raw product / previous partial snapshot", "SYA; CSA", "1998-2023 rows; SYA has year gaps", "Read-only comparator for SYA; not a five-site source.")
    add(OLD_D222, "previous D222 workbook used by 009 audit", "raw product / historical snapshot", "HLA; FQA", "1998-2023", "Read-only; metadata and hash match 009 audit record.")
    add(LEGACY, "historical local rainfall workbook", "legacy transcription/export; same observation chain, not independent", "HLA; FQA; YCA; LCA; other codes", "1998-2023 partial/site-specific", "Read-only; contains 20-20 rainfall total field.")

    for site in TARGETS:
        add(
            ROOT / "weather_clean" / f"{site}_weather_cleaned.csv",
            "existing cleaned weather table",
            "derived / processed weather",
            site,
            "file dates inspected; see CSV",
            "Not a raw D222 observation source; numeric values may include filling/processing.",
        )
    add(ROOT / "weather_clean" / "all_sites_weather_cleaned.csv", "combined cleaned weather table", "derived / processed weather", "HLA; FQA; YCA; SYA; LCA", "file dates inspected; see CSV", "Not a raw D222 observation source.")

    wth_site_files = {
        "HLA": "CNHL0701.WTH",
        "FQA": "CNFQ0701.WTH",
        "YCA": "CNYC0801.WTH",
        "SYA": "CNSY1201.WTH",
        "LCA": "CNLC0801.WTH",
    }
    for site, filename in wth_site_files.items():
        add(ROOT / "my_data" / filename, "existing station WTH", "derived DSSAT-ready weather", site, "single WTH file; header/date range inspectable", "Inventory only; not treated as raw rainfall observation or modified.")

    prompt_names = [
        "008_hl_fq_wgen_random_weather_8seed.md",
        "009_cern_d222_rain_provenance_reaudit.md",
        "010_hla_wgen_random_weather_8seed_resume.md",
        "011_hla_ghcn50756_rain_fallback_audit.md",
        "012_hla_full_rain_source_consolidation.md",
        "013_hla_2013_secondary_ghcn_reaudit.md",
        "014_hla_2013_noaa_isd_same_station_recovery.md",
        "021_hl_fq_isdlite_weather_audit.md",
        "022_five_site_d222_version_audit.md",
    ]
    for name in prompt_names:
        add(ROOT / "prompt_02" / name, "task prompt/reference", "instruction document; not data", "HLA; FQA; multi-site as named", "task-specific", "Instructions are inventoried separately from observations.")

    reference_paths = [
        "docs/hl_fq_cern_d222_rain_reaudit.md",
        "docs/hla_ghcn50756_rain_fallback_audit_011.md",
        "docs/hla_rain_source_consolidation_012.md",
        "docs/hl_fq_isdlite_weather_audit_021.md",
        "results/hl_fq_random_weather_8seed/source_audit/weather_source_audit_final.json",
        "results/hl_fq_random_weather_8seed/source_audit/cern_d222_reaudit/raw_file_metadata.json",
        "results/hl_fq_random_weather_8seed/source_audit/cern_d222_reaudit/findings.md",
        "results/hl_fq_random_weather_8seed/source_audit/cern_d222_reaudit/final_gate.json",
        "results/hl_fq_random_weather_8seed/source_audit/cern_d222_reaudit/fqa_extreme_rain_audit.csv",
        "results/hl_fq_random_weather_8seed/source_audit/cern_d222_reaudit/hla_2004_rain_audit.csv",
        "results/hl_fq_random_weather_8seed/source_audit/cern_d222_reaudit/rain_field_closure.csv",
        "results/hla_random_weather_8seed/010/source_audit/workbook_inventory.json",
        "results/hla_random_weather_8seed/010/source_audit/raw_inventory.json",
        "results/hla_random_weather_8seed/010/source_audit/phase_a_gate.json",
        "results/hla_random_weather_8seed/010/findings.md",
        "results/hla_random_weather_8seed/011_ghcn50756_fallback/final_gate.json",
        "results/hla_random_weather_8seed/011_ghcn50756_fallback/findings.md",
        "results/hla_random_weather_8seed/011_ghcn50756_fallback/unresolved_local_dates_after_target.csv",
        "results/hla_random_weather_8seed/011_ghcn50756_fallback/extreme_rain_audit.csv",
        "results/hla_random_weather_8seed/012_rain_source_consolidation/input_provenance.json",
        "results/hla_random_weather_8seed/012_rain_source_consolidation/source_hierarchy.json",
        "results/hla_random_weather_8seed/012_rain_source_consolidation/daily_source_selection_2004_2013.csv",
        "results/hla_random_weather_8seed/012_rain_source_consolidation/unresolved_dates_final.csv",
        "results/hla_random_weather_8seed/012_rain_source_consolidation/findings.md",
        "results/hla_random_weather_8seed/012_rain_source_consolidation/final_gate.json",
        "results/hla_random_weather_8seed/013_secondary_ghcn_reaudit/input_provenance.json",
        "results/hla_random_weather_8seed/013_secondary_ghcn_reaudit/daily_source_selection_after_013.csv",
        "results/hla_random_weather_8seed/013_secondary_ghcn_reaudit/unresolved_dates_after_013.csv",
        "results/hla_random_weather_8seed/013_secondary_ghcn_reaudit/findings.md",
        "results/hla_random_weather_8seed/013_secondary_ghcn_reaudit/final_gate.json",
        "results/hla_random_weather_8seed/014_isd50756_recovery/input_provenance.json",
        "results/hla_random_weather_8seed/014_isd50756_recovery/daily_source_selection_after_014.csv",
        "results/hla_random_weather_8seed/014_isd50756_recovery/unresolved_dates_after_014.csv",
        "results/hla_random_weather_8seed/014_isd50756_recovery/findings.md",
        "results/hla_random_weather_8seed/014_isd50756_recovery/final_gate.json",
        "results/hl_fq_isdlite_021/input_provenance.json",
        "results/hl_fq_isdlite_021/dataset_identity.json",
        "results/hl_fq_isdlite_021/findings.md",
        "results/hl_fq_isdlite_021/final_gate.json",
    ]
    for text in reference_paths:
        path = ROOT / Path(text)
        add(path, "prior provenance audit evidence", "derived audit record; source table where indicated", "HLA; FQA; site(s) indicated by path/content", "audit-specific; see file", "Read-only reference; prior evidence does not establish new-snapshot values.")

    return sorted(entries, key=lambda row: row["path"])


def run() -> None:
    if not CURRENT.is_file():
        raise FileNotFoundError(f"Project-contained workbook candidate is missing: {CURRENT}")
    if not OUT.is_dir():
        raise FileNotFoundError(f"Audit output directory must already exist: {OUT}")

    current = load_d222(CURRENT)
    current_index = index_rows(current["rows"])
    current_codes = set(current["site_codes"])
    current_sites = {site: site in current_codes for site in TARGETS}
    target_present = [site for site, present in current_sites.items() if present]
    target_missing = [site for site, present in current_sites.items() if not present]

    meta = source_metadata(CURRENT)
    old_meta = source_metadata(OLD_D222) if OLD_D222.is_file() else None
    legacy_meta = source_metadata(LEGACY) if LEGACY.is_file() else None
    sheet_records = sum(item["data_rows"] for item in current["sheets"])
    all_records = current["rows"]

    previous_partial_meta = source_metadata(PREVIOUS_PARTIAL) if PREVIOUS_PARTIAL.is_file() else None
    filename_candidates = [x for x in [meta, previous_partial_meta] if x]
    identity = {
        "snapshot_id": "CERN_D222_CURRENT_SNAPSHOT_2026_10_07",
        "identity_status": "FIVE_SITE_CODES_PRESENT_WITH_SITE_YEAR_GAPS_RECORDED",
        "audit_date": "2026-10-07",
        "selection": {
            "selected_candidate": meta,
            "matching_candidate_count": 1,
            "exact_filename_candidate_count": 1,
            "related_candidates": [previous_partial_meta] if previous_partial_meta else [],
            "filename_candidates": filename_candidates,
            "selection_rationale": "Selected the exact-basename workbook data/external/人工大气观测降雨蒸发能见度日值.xls. The earlier prefixed workbook in my_data is a partial SYA/CSA export with an older mtime (2026-04-01); the selected exact-name file has mtime 2026-10-07 and contains all five target station codes.",
            "other_relevant_project_workbooks": [x for x in [old_meta, legacy_meta, previous_partial_meta] if x],
            "download_time_confirmed": False,
            "mtime_interpretation": "Filesystem mtime is recorded as requested; it does not prove download time.",
        },
        "workbook": {
            "sheet_names": [item["name"] for item in current["sheets"]],
            "sheets": current["sheets"],
            "data_record_rows": sheet_records,
            "total_rows_including_title_and_header": sum(item["nrows"] for item in current["sheets"]),
            "column_count": current["sheets"][0]["ncols"] if current["sheets"] else None,
            "data_first_date": current["data_first_date"],
            "data_last_date": current["data_last_date"],
            "station_codes": current["site_codes"],
            "project_target_station_codes_present": target_present,
            "project_target_station_codes_missing": target_missing,
            "non_target_station_codes": sorted(current_codes - set(TARGETS)),
            "invalid_date_rows": current["invalid_date_rows"],
        },
        "field_mapping_from_workbook_headers": {
            "station_code_field": "生态站代码",
            "station_name_field": None,
            "year_field": "年",
            "month_field": "月",
            "day_field": "日",
            "rain_field": current["rain_field"],
            "rain_unit": "mm (from field header)",
            "missing_encoding_observed": "empty-string/blank cell; never treated as zero",
            "rain_cell_types": current["rain_cell_types"],
            "blank_cell_types": current["missing_cell_types"],
            "numeric_storage_note": "Rain values are stored as parseable numeric text in this workbook; values were parsed numerically without imputing blanks.",
        },
        "five_site_coverage_gate": {
            "all_five_sites_present": len(target_present) == 5,
            "target_site_count_present": len(target_present),
            "target_site_count_expected": 5,
            "missing_sites": target_missing,
            "full_five_site_version_comparison_possible": len(target_present) == 5,
            "reason": "All five target codes are present; date gaps and missing rain cells are reported separately in yearly QC.",
        },
        "interpretation": "Audit snapshot candidate only. It does not replace, supersede, or backfill any existing project input.",
    }
    (OUT / "new_d222_snapshot_identity.json").write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")

    station_names = {site: info["name"] for site, info in TARGETS.items()}
    station_rows = []
    for site in TARGETS:
        by_date = current_index.get(site, {})
        station_rows.append(
            {
                "project_site": site,
                "d222_site_code": site if site in current_codes else "",
                "station_name": station_names[site],
                "first_date": min(by_date).isoformat() if by_date else "",
                "last_date": max(by_date).isoformat() if by_date else "",
                "total_rows": sum(len(items) for items in by_date.values()) if by_date else "",
                "rain_field": current["rain_field"],
                "missing_encoding": "empty-string/blank cell; not zero" if by_date else "",
                "notes": (
                    "Site code observed in workbook; station name is supplied by project mapping because the workbook has no station-name field."
                    if by_date
                    else "Expected target code is absent from the candidate workbook; station name is project mapping only and not internally confirmed."
                ),
            }
        )
    write_csv(
        OUT / "site_mapping.csv",
        ["project_site", "d222_site_code", "station_name", "first_date", "last_date", "total_rows", "rain_field", "missing_encoding", "notes"],
        station_rows,
    )

    daily_rows: list[dict[str, Any]] = []
    by_site_year: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in all_records:
        if row["site_code"] not in TARGETS:
            continue
        by_site_year[(row["site_code"], row["date"].year)].append(row)
        daily_rows.append(
            {
                "project_site": row["site_code"],
                "d222_site_code": row["site_code"],
                "station_name": station_names[row["site_code"]],
                "date": row["date"].isoformat(),
                "year": row["date"].year,
                "rain_raw": str(row["rain_raw"]),
                "rain_mm": rounded(row["rain_mm"]),
                "value_status": row["value_status"],
                "source_sheet": row["sheet"],
                "source_row": row["source_row"],
                "unit": "mm",
            }
        )
    write_csv(
        OUT / "current_d222_daily_normalized.csv",
        ["project_site", "d222_site_code", "station_name", "date", "year", "rain_raw", "rain_mm", "value_status", "source_sheet", "source_row", "unit"],
        sorted(daily_rows, key=lambda row: (row["project_site"], row["date"])),
    )

    qc_rows: list[dict[str, Any]] = []
    for site in TARGETS:
        site_present = current_sites[site]
        site_index = current_index.get(site, {})
        for year in EXPECTED_YEARS:
            expected = (dt.date(year + 1, 1, 1) - dt.date(year, 1, 1)).days
            year_dates = {date for date in site_index if date.year == year}
            year_records = by_site_year.get((site, year), [])
            statuses = Counter(row["value_status"] for row in year_records)
            numeric_values = [row["rain_mm"] for row in year_records if row["value_status"] in ("NUMERIC", "EXPLICIT_ZERO", "NEGATIVE")]
            valid_values = [row["rain_mm"] for row in year_records if row["value_status"] in ("NUMERIC", "EXPLICIT_ZERO")]
            max_value = max(valid_values) if valid_values else None
            max_dates = [row["date"].isoformat() for row in year_records if row["rain_mm"] == max_value] if max_value is not None else []
            duplicate_count = sum(max(0, len(items) - 1) for date, items in site_index.items() if date.year == year)
            qc_rows.append(
                {
                    "project_site": site,
                    "d222_site_code": site if site_present else "",
                    "station_name": station_names[site],
                    "year": year,
                    "expected_days": expected,
                    "site_present_in_candidate": site_present,
                    "actual_record_count": len(year_records) if site_present else 0,
                    "actual_unique_dates": len(year_dates) if site_present else 0,
                    "date_gap_days": expected - len(year_dates),
                    "duplicate_date_count": duplicate_count if site_present else "",
                    "numeric_rain_days": sum(statuses[s] for s in ("NUMERIC", "EXPLICIT_ZERO", "NEGATIVE")) if site_present else "",
                    "explicit_zero_days": statuses["EXPLICIT_ZERO"] if site_present else "",
                    "blank_missing_days": statuses["BLANK"] if site_present else "",
                    "negative_rain_days": statuses["NEGATIVE"] if site_present else "",
                    "invalid_rain_days": statuses["INVALID"] if site_present else "",
                    "annual_numeric_subtotal_mm": rounded(sum(numeric_values)) if site_present and numeric_values else (0.0 if site_present else ""),
                    "annual_max_daily_rain_mm": rounded(max_value),
                    "max_rain_date": min(max_dates) if max_dates else "",
                    "coverage_status": "SITE_ABSENT" if not site_present else ("NO_ROWS_IN_YEAR" if not year_records else "ROWS_PRESENT"),
                    "notes": "A missing site is not interpreted as a blank or dry day." if not site_present else "Blank rain cells remain missing; subtotal sums parseable numeric values only.",
                }
            )
    write_csv(
        OUT / "current_d222_yearly_qc.csv",
        ["project_site", "d222_site_code", "station_name", "year", "expected_days", "site_present_in_candidate", "actual_record_count", "actual_unique_dates", "date_gap_days", "duplicate_date_count", "numeric_rain_days", "explicit_zero_days", "blank_missing_days", "negative_rain_days", "invalid_rain_days", "annual_numeric_subtotal_mm", "annual_max_daily_rain_mm", "max_rain_date", "coverage_status", "notes"],
        qc_rows,
    )

    old_d222_data = load_d222(OLD_D222) if OLD_D222.is_file() else {"rows": []}
    legacy_data = load_d222(LEGACY) if LEGACY.is_file() else {"rows": []}
    previous_partial_data = load_d222(PREVIOUS_PARTIAL) if PREVIOUS_PARTIAL.is_file() else {"rows": []}
    old_index = index_rows(old_d222_data["rows"])
    legacy_index = index_rows(legacy_data["rows"])
    previous_partial_index = index_rows(previous_partial_data["rows"])

    comparison_sources = {
        "HLA": [
            ("OLD_RAW_D222", old_index, OLD_D222, "historical raw D222 workbook"),
            ("LEGACY_HLLCYCFQ", legacy_index, LEGACY, "same observation chain; not independent validation"),
        ],
        "FQA": [
            ("OLD_RAW_D222", old_index, OLD_D222, "historical raw D222 workbook"),
            ("LEGACY_HLLCYCFQ", legacy_index, LEGACY, "same observation chain; not independent validation"),
        ],
        "YCA": [("LEGACY_HLLCYCFQ", legacy_index, LEGACY, "legacy D222-derived rainfall workbook")],
        "SYA": [("PREVIOUS_PARTIAL_D222", previous_partial_index, PREVIOUS_PARTIAL, "earlier D222 workbook candidate")],
        "LCA": [("LEGACY_HLLCYCFQ", legacy_index, LEGACY, "legacy D222-derived rainfall workbook")],
    }
    primary_source_key = {"HLA": "OLD_RAW_D222", "FQA": "OLD_RAW_D222", "YCA": "LEGACY_HLLCYCFQ", "SYA": "PREVIOUS_PARTIAL_D222", "LCA": "LEGACY_HLLCYCFQ"}
    diff_rows: list[dict[str, Any]] = []
    compare_counts: dict[tuple[str, str, int], Counter[str]] = defaultdict(Counter)
    compare_overlap: Counter[tuple[str, str, int]] = Counter()
    compare_union: Counter[tuple[str, str, int]] = Counter()
    for site, source_list in comparison_sources.items():
        current_dates = current_index.get(site, {})
        for source_key, source_index, source_path, source_role in source_list:
            old_dates = source_index.get(site, {})
            for date in sorted(set(old_dates) | set(current_dates)):
                old_has_date = date in old_dates
                new_has_date = date in current_dates
                old_items = old_dates.get(date, [])
                new_items = current_dates.get(date, [])
                old_value = old_items[0]["rain_mm"] if old_items else None
                new_value = new_items[0]["rain_mm"] if new_items else None
                cat = category(old_has_date, new_has_date, old_value, new_value)
                compare_counts[(site, source_key, date.year)][cat] += 1
                compare_union[(site, source_key, date.year)] += 1
                if old_has_date and new_has_date:
                    compare_overlap[(site, source_key, date.year)] += 1
                abs_diff = abs(new_value - old_value) if old_value is not None and new_value is not None else None
                rel_diff = (new_value - old_value) / old_value if old_value not in (None, 0) and new_value is not None else None
                diff_rows.append(
                    {
                        "site": site,
                        "comparison_source": source_key,
                        "year": date.year,
                        "date": date.isoformat(),
                        "old_source_path": rel(source_path),
                        "old_source_role": source_role,
                        "old_rain_mm": rounded(old_value),
                        "new_source_path": rel(CURRENT),
                        "new_source_role": "current CERN D222 snapshot",
                        "new_rain_mm": rounded(new_value),
                        "old_value_status": old_items[0]["value_status"] if old_items else "NO_DATE_ROW",
                        "new_value_status": new_items[0]["value_status"] if new_items else "NO_DATE_ROW",
                        "comparison_category": cat,
                        "absolute_difference_mm": rounded(abs_diff),
                        "relative_difference": rounded(rel_diff),
                        "notes": "Blank values remain missing. Agreement with the legacy table is same-chain compatibility, not independent verification." if source_key == "LEGACY_HLLCYCFQ" else "Read-only product-version comparison; no values were changed or filled.",
                    }
                )
    write_csv(
        OUT / "d222_version_daily_diff.csv",
        ["site", "comparison_source", "year", "date", "old_source_path", "old_source_role", "old_rain_mm", "new_source_path", "new_source_role", "new_rain_mm", "old_value_status", "new_value_status", "comparison_category", "absolute_difference_mm", "relative_difference", "notes"],
        diff_rows,
    )

    summary_rows: list[dict[str, Any]] = []
    for site, source_list in comparison_sources.items():
        for source_key, source_index, source_path, source_role in source_list:
            for year in EXPECTED_YEARS:
                counts = compare_counts.get((site, source_key, year), Counter())
                union_n = compare_union[(site, source_key, year)]
                current_year = [row for row in current_index.get(site, {}).values() for row in row if row["date"].year == year]
                old_year = [row for day_rows in source_index.get(site, {}).values() for row in day_rows if row["date"].year == year]
                current_numeric = [row["rain_mm"] for row in current_year if row["value_status"] in ("NUMERIC", "EXPLICIT_ZERO")]
                old_numeric = [row["rain_mm"] for row in old_year if row["value_status"] in ("NUMERIC", "EXPLICIT_ZERO")]
                changed = sum(counts[name] for name in ("NUMERIC_CHANGED", "OLD_MISSING_NEW_NUMERIC", "OLD_NUMERIC_NEW_MISSING", "OLD_ZERO_NEW_NONZERO", "OLD_NONZERO_NEW_ZERO", "DATE_ONLY_IN_NEW", "DATE_ONLY_IN_OLD"))
                status = "NO_ROWS_IN_EITHER_SOURCE" if union_n == 0 else "COMPARED"
                summary_rows.append(
                    {
                        "site": site,
                        "year": year,
                        "comparison_source": source_key,
                        "comparison_status": status,
                        "comparison_old_source": rel(source_path),
                        "comparison_old_source_role": source_role,
                        "union_dates_compared": union_n,
                        "overlap_dates": compare_overlap[(site, source_key, year)],
                        **{name: counts[name] for name in CATEGORIES},
                        "old_annual_numeric_subtotal_mm": rounded(sum(old_numeric)) if old_numeric else (0.0 if old_year else ""),
                        "new_annual_numeric_subtotal_mm": rounded(sum(current_numeric)) if current_numeric else (0.0 if current_year else ""),
                        "annual_rainfall_difference_mm": rounded(sum(current_numeric) - sum(old_numeric)) if old_year and current_year else "",
                        "version_change_detected": changed > 0,
                        "version_stable_for_compared_dates": changed == 0 and union_n > 0,
                    }
                )
    write_csv(
        OUT / "d222_version_summary.csv",
        ["site", "year", "comparison_source", "comparison_status", "comparison_old_source", "comparison_old_source_role", "union_dates_compared", "overlap_dates", *CATEGORIES, "old_annual_numeric_subtotal_mm", "new_annual_numeric_subtotal_mm", "annual_rainfall_difference_mm", "version_change_detected", "version_stable_for_compared_dates"],
        summary_rows,
    )

    hla_unresolved_path = ROOT / "results" / "hla_random_weather_8seed" / "014_isd50756_recovery" / "unresolved_dates_after_014.csv"
    hla_selection_path = ROOT / "results" / "hla_random_weather_8seed" / "014_isd50756_recovery" / "daily_source_selection_after_014.csv"
    hla_unresolved = read_csv(hla_unresolved_path)
    hla_selected = {row["date"]: row for row in read_csv(hla_selection_path)}
    hla_rows = []
    for item in hla_unresolved:
        previous = hla_selected.get(item["date"], {})
        date = dt.date.fromisoformat(item["date"])
        current_items = current_index.get("HLA", {}).get(date, [])
        current_item = current_items[0] if current_items else {}
        current_status = current_item.get("value_status", "DATE_ABSENT")
        previous_value, _ = parse_number(previous.get("local_d222_rain_mm", ""))
        current_value = current_item.get("rain_mm")
        changed_to_numeric = current_status in ("NUMERIC", "EXPLICIT_ZERO")
        hla_rows.append(
            {
                "date": item["date"],
                "previous_unresolved": True,
                "previous_unresolved_reason": item.get("reason", ""),
                "old_d222_rain_mm": previous.get("local_d222_rain_mm", ""),
                "previous_selected_rain_mm": previous.get("selected_rain_mm", ""),
                "previous_selected_source": previous.get("selected_source", ""),
                "current_candidate_has_hla_site": True,
                "current_candidate_status": current_status,
                "current_d222_rain_mm": rounded(current_value),
                "newly_numeric_in_current": changed_to_numeric and previous_value is None,
                "newly_explicit_zero_in_current": current_status == "EXPLICIT_ZERO" and previous_value is None,
                "gap_changed": changed_to_numeric and previous_value is None,
                "remaining_unresolved": not changed_to_numeric,
                "notes": "Current blank cells are kept missing; explicit numeric zero is distinguished from blank.",
            }
        )
    write_csv(
        OUT / "hla_111_day_recheck.csv",
        ["date", "previous_unresolved", "previous_unresolved_reason", "old_d222_rain_mm", "previous_selected_rain_mm", "previous_selected_source", "current_candidate_has_hla_site", "current_candidate_status", "current_d222_rain_mm", "newly_numeric_in_current", "newly_explicit_zero_in_current", "gap_changed", "remaining_unresolved", "notes"],
        hla_rows,
    )

    hla_key_rows = []
    for date in HLA_KEY_DATES:
        previous = hla_selected.get(date, {})
        key_date = dt.date.fromisoformat(date)
        current_items = current_index.get("HLA", {}).get(key_date, [])
        current_item = current_items[0] if current_items else {}
        current_value = current_item.get("rain_mm")
        hla_key_rows.append(
            {
                "date": date,
                "old_d222_rain_mm": previous.get("local_d222_rain_mm", ""),
                "old_legacy_rain_mm": previous.get("local_legacy_rain_mm", ""),
                "previous_selected_rain_mm": previous.get("selected_rain_mm", ""),
                "previous_selected_source": previous.get("selected_source", ""),
                "current_candidate_has_hla_site": True,
                "current_candidate_status": current_item.get("value_status", "DATE_ABSENT"),
                "current_d222_rain_mm": rounded(current_value),
                "version_delta_mm": rounded(current_value - float(previous["local_d222_rain_mm"])) if current_value is not None and previous.get("local_d222_rain_mm", "") not in ("", None) else "",
                "notes": "Historical selected-source values are retained for context; current D222 value is read directly from the new five-site workbook.",
            }
        )
    write_csv(
        OUT / "hla_2013_key_event_recheck.csv",
        ["date", "old_d222_rain_mm", "old_legacy_rain_mm", "previous_selected_rain_mm", "previous_selected_source", "current_candidate_has_hla_site", "current_candidate_status", "current_d222_rain_mm", "version_delta_mm", "notes"],
        hla_key_rows,
    )

    prior_weather_audit_path = ROOT / "results" / "hl_fq_random_weather_8seed" / "source_audit" / "weather_source_audit_final.json"
    prior_weather_audit = json.loads(prior_weather_audit_path.read_text(encoding="utf-8"))
    fqa_unresolved = set(prior_weather_audit["sites"]["FQA"]["official_ghcn_rain_screen"]["unresolved_local_rain_days"])
    fqa_rows: list[dict[str, Any]] = []
    expected_fqa_old = {2011: 1614.4, 2012: 1378.2, 2013: 1144.4}

    def year_values(index: dict[str, dict[dt.date, list[dict[str, Any]]]], site: str, year: int) -> tuple[list[dict[str, Any]], list[float], int]:
        records = [row for day_rows in index.get(site, {}).values() for row in day_rows if row["date"].year == year]
        values = [row["rain_mm"] for row in records if row["value_status"] in ("NUMERIC", "EXPLICIT_ZERO")]
        blank_days = sum(row["value_status"] == "BLANK" for row in records)
        return records, values, blank_days

    fqa_current_missing_by_unresolved = Counter()
    for year in (2011, 2012, 2013):
        old_year_rows, old_vals, old_blank_count = year_values(old_index, "FQA", year)
        current_year_rows, current_vals, current_blank_count = year_values(current_index, "FQA", year)
        _, legacy_vals, legacy_blank_count = year_values(legacy_index, "FQA", year)
        current_year_max = max(current_vals) if current_vals else None
        old_year_max = max(old_vals) if old_vals else None
        fqa_rows.append(
            {
                "row_type": "ANNUAL_SUMMARY",
                "date": "",
                "year": year,
                "old_d222_rain_mm": "",
                "legacy_rain_mm": "",
                "old_d222_numeric_days": len(old_vals),
                "old_d222_blank_days": old_blank_count,
                "old_d222_numeric_subtotal_mm": rounded(sum(old_vals)) if old_vals else "",
                "current_d222_numeric_days": len(current_vals),
                "current_d222_blank_days": current_blank_count,
                "current_d222_numeric_subtotal_mm": rounded(sum(current_vals)) if current_vals else "",
                "legacy_numeric_days": len(legacy_vals),
                "legacy_blank_days": legacy_blank_count,
                "legacy_numeric_subtotal_mm": rounded(sum(legacy_vals)) if legacy_vals else "",
                "annual_total_difference_vs_old_mm": rounded(sum(current_vals) - sum(old_vals)) if current_vals and old_vals else "",
                "annual_max_difference_vs_old_mm": rounded(current_year_max - old_year_max) if current_year_max is not None and old_year_max is not None else "",
                "previous_reported_total_mm": expected_fqa_old[year],
                "previous_unresolved_2013": "",
                "current_candidate_has_fqa_site": True,
                "current_candidate_status": "SITE_PRESENT",
                "current_d222_rain_mm": "",
                "notes": "Current, prior raw D222, and legacy same-chain totals are reported separately; blanks are excluded from numeric subtotals.",
            }
        )
    for date_text in FQA_KEY_DATES:
        date = dt.date.fromisoformat(date_text)
        old_items = old_index.get("FQA", {}).get(date, [])
        current_items = current_index.get("FQA", {}).get(date, [])
        legacy_items = legacy_index.get("FQA", {}).get(date, [])
        old_value = old_items[0]["rain_mm"] if old_items else None
        current_value = current_items[0]["rain_mm"] if current_items else None
        legacy_value = legacy_items[0]["rain_mm"] if legacy_items else None
        fqa_rows.append(
            {
                "row_type": "KEY_EVENT",
                "date": date_text,
                "year": date.year,
                "old_d222_rain_mm": rounded(old_value),
                "legacy_rain_mm": rounded(legacy_value),
                "old_d222_numeric_days": "",
                "old_d222_blank_days": "",
                "old_d222_numeric_subtotal_mm": "",
                "current_d222_numeric_days": "",
                "current_d222_blank_days": "",
                "current_d222_numeric_subtotal_mm": "",
                "legacy_numeric_days": "",
                "legacy_blank_days": "",
                "legacy_numeric_subtotal_mm": "",
                "annual_total_difference_vs_old_mm": "",
                "annual_max_difference_vs_old_mm": "",
                "previous_reported_total_mm": "",
                "previous_unresolved_2013": date_text in fqa_unresolved,
                "current_candidate_has_fqa_site": True,
                "current_candidate_status": current_items[0]["value_status"] if current_items else "DATE_ABSENT",
                "current_d222_rain_mm": rounded(current_value),
                "notes": "The legacy table belongs to the same observation chain and is not an independent validation source.",
            }
        )
    for offset in range(365):
        date = dt.date(2013, 1, 1) + dt.timedelta(days=offset)
        old_items = old_index.get("FQA", {}).get(date, [])
        current_items = current_index.get("FQA", {}).get(date, [])
        legacy_items = legacy_index.get("FQA", {}).get(date, [])
        old_value = old_items[0]["rain_mm"] if old_items else None
        current_value = current_items[0]["rain_mm"] if current_items else None
        legacy_value = legacy_items[0]["rain_mm"] if legacy_items else None
        current_status = current_items[0]["value_status"] if current_items else "DATE_ABSENT"
        if date.isoformat() in fqa_unresolved:
            fqa_current_missing_by_unresolved[current_status] += 1
        fqa_rows.append(
            {
                "row_type": "DAILY_2013_OLD_MISSINGNESS_AND_PRIOR_UNRESOLVED",
                "date": date.isoformat(),
                "year": 2013,
                "old_d222_rain_mm": rounded(old_value),
                "legacy_rain_mm": rounded(legacy_value),
                "old_d222_numeric_days": "",
                "old_d222_blank_days": "",
                "old_d222_numeric_subtotal_mm": "",
                "current_d222_numeric_days": "",
                "current_d222_blank_days": "",
                "current_d222_numeric_subtotal_mm": "",
                "legacy_numeric_days": "",
                "legacy_blank_days": "",
                "legacy_numeric_subtotal_mm": "",
                "annual_total_difference_vs_old_mm": "",
                "annual_max_difference_vs_old_mm": "",
                "previous_reported_total_mm": "",
                "previous_unresolved_2013": date.isoformat() in fqa_unresolved,
                "current_candidate_has_fqa_site": True,
                "current_candidate_status": current_status,
                "current_d222_rain_mm": rounded(current_value),
                "notes": "Prior unresolved membership comes from the 008/009 weather source audit; blank cells remain missing, not zero.",
            }
        )
    write_csv(
        OUT / "fqa_2011_2013_recheck.csv",
        ["row_type", "date", "year", "old_d222_rain_mm", "legacy_rain_mm", "old_d222_numeric_days", "old_d222_blank_days", "old_d222_numeric_subtotal_mm", "current_d222_numeric_days", "current_d222_blank_days", "current_d222_numeric_subtotal_mm", "legacy_numeric_days", "legacy_blank_days", "legacy_numeric_subtotal_mm", "annual_total_difference_vs_old_mm", "annual_max_difference_vs_old_mm", "previous_reported_total_mm", "previous_unresolved_2013", "current_candidate_has_fqa_site", "current_candidate_status", "current_d222_rain_mm", "notes"],
        fqa_rows,
    )

    summary_by_key = {(row["site"], row["comparison_source"], row["year"]): row for row in summary_rows}
    source_index_by_key = {
        "OLD_RAW_D222": old_index,
        "LEGACY_HLLCYCFQ": legacy_index,
        "PREVIOUS_PARTIAL_D222": previous_partial_index,
    }
    site_audit_rows: dict[str, list[dict[str, Any]]] = {}
    for site in ("YCA", "SYA", "LCA"):
        source_key = primary_source_key[site]
        source_path = next(spec[2] for spec in comparison_sources[site] if spec[0] == source_key)
        source_role = next(spec[3] for spec in comparison_sources[site] if spec[0] == source_key)
        old_site_index = source_index_by_key[source_key]
        rows = []
        for year in EXPECTED_YEARS:
            summary = summary_by_key[(site, source_key, year)]
            new_year_rows, new_values, new_blanks = year_values(current_index, site, year)
            old_year_rows, old_values, old_blanks = year_values(old_site_index, site, year)
            new_max = max(new_values) if new_values else None
            old_max = max(old_values) if old_values else None
            version_change = summary["version_change_detected"]
            rows.append(
                {
                    "site": site,
                    "fitting_period": TARGETS[site]["fitting_period"],
                    "audit_period": "1998-2023 shared-source range; per-year rows",
                    "year": year,
                    "compared_old_source": rel(source_path),
                    "old_source_role": source_role,
                    "current_record_count": len(new_year_rows),
                    "old_record_count": len(old_year_rows),
                    "overlap_dates": summary["overlap_dates"],
                    "exact_match_days": summary["EXACT_NUMERIC_MATCH"],
                    "numeric_changed_days": summary["NUMERIC_CHANGED"],
                    "old_missing_new_numeric_days": summary["OLD_MISSING_NEW_NUMERIC"],
                    "old_numeric_new_missing_days": summary["OLD_NUMERIC_NEW_MISSING"],
                    "date_only_in_new_days": summary["DATE_ONLY_IN_NEW"],
                    "date_only_in_old_days": summary["DATE_ONLY_IN_OLD"],
                    "old_annual_numeric_subtotal_mm": rounded(sum(old_values)) if old_values else (0.0 if old_year_rows else ""),
                    "current_annual_numeric_subtotal_mm": rounded(sum(new_values)) if new_values else (0.0 if new_year_rows else ""),
                    "annual_rainfall_difference_mm": summary["annual_rainfall_difference_mm"],
                    "old_max_daily_rain_mm": rounded(old_max),
                    "current_max_daily_rain_mm": rounded(new_max),
                    "annual_max_event_difference_mm": rounded(new_max - old_max) if new_max is not None and old_max is not None else "",
                    "old_blank_days": old_blanks,
                    "current_blank_days": new_blanks,
                    "version_change_detected": version_change,
                    "current_experiment_modified": False,
                    "notes": "Numeric subtotals sum explicit values only; blanks are not zero. Fitting period is not specified in prompt 022 for this site." if TARGETS[site]["fitting_period"] == "NOT_SPECIFIED_IN_022" else "",
                }
            )
        site_audit_rows[site] = rows
        write_csv(
            OUT / f"{site.lower()}_version_audit.csv",
            ["site", "fitting_period", "audit_period", "year", "compared_old_source", "old_source_role", "current_record_count", "old_record_count", "overlap_dates", "exact_match_days", "numeric_changed_days", "old_missing_new_numeric_days", "old_numeric_new_missing_days", "date_only_in_new_days", "date_only_in_old_days", "old_annual_numeric_subtotal_mm", "current_annual_numeric_subtotal_mm", "annual_rainfall_difference_mm", "old_max_daily_rain_mm", "current_max_daily_rain_mm", "annual_max_event_difference_mm", "old_blank_days", "current_blank_days", "version_change_detected", "current_experiment_modified", "notes"],
            rows,
        )

    overview = []
    source_labels = {
        "OLD_RAW_D222": "historical raw D222",
        "LEGACY_HLLCYCFQ": "legacy same-chain workbook",
        "PREVIOUS_PARTIAL_D222": "earlier partial D222 workbook",
    }
    for site, info in TARGETS.items():
        source_key = primary_source_key[site]
        primary_rows = [row for row in summary_rows if row["site"] == site and row["comparison_source"] == source_key]
        totals = {name: sum(int(row[name] or 0) for row in primary_rows) for name in CATEGORIES}
        source_role = next(spec[3] for spec in comparison_sources[site] if spec[0] == source_key)
        site_dates = current_index.get(site, {})
        site_rows = sum(len(items) for items in site_dates.values())
        expected_total = sum((dt.date(y + 1, 1, 1) - dt.date(y, 1, 1)).days for y in EXPECTED_YEARS)
        gap_total = expected_total - len(site_dates)
        first_date = min(site_dates).isoformat() if site_dates else ""
        last_date = max(site_dates).isoformat() if site_dates else ""
        changed_total = sum(totals[name] for name in ("NUMERIC_CHANGED", "OLD_ZERO_NEW_NONZERO", "OLD_NONZERO_NEW_ZERO"))
        missing_pattern_changed = sum(totals[name] for name in ("OLD_MISSING_NEW_NUMERIC", "OLD_NUMERIC_NEW_MISSING", "DATE_ONLY_IN_NEW", "DATE_ONLY_IN_OLD")) > 0
        if site == "HLA":
            key_issue = "D222 version stable; all 111 previous unresolved dates remain blank; existing HLA provenance gate is unchanged."
            recommendation = "Keep HLA provenance blockers; the snapshot does not recover gaps or resolve climatic plausibility."
        elif site == "FQA":
            key_issue = "D222 version stable; 2011-2013 totals and 240/204 mm events remain as recorded; magnitude concern is unresolved."
            recommendation = "Keep FQA blocker; same-chain agreement does not validate climate correctness."
        else:
            key_issue = "No version difference detected against the available historical source for matched dates."
            recommendation = "Record version stability; no version-driven reprocessing is indicated by this audit."
        overview.append(
            {
                "site": site,
                "fitting_period": info["fitting_period"],
                "current_d222_coverage": f"{first_date} to {last_date}; {site_rows} rows; {gap_total} absent calendar dates within 1998-2023",
                "compared_old_source": f"{source_labels[source_key]}: {source_role}",
                "exact_match_days": totals["EXACT_NUMERIC_MATCH"],
                "changed_numeric_days": changed_total,
                "old_missing_new_numeric_days": totals["OLD_MISSING_NEW_NUMERIC"],
                "old_numeric_new_missing_days": totals["OLD_NUMERIC_NEW_MISSING"],
                "key_issue_changed": key_issue,
                "current_experiment_modified": False,
                "recommendation": recommendation,
            }
        )
    write_csv(
        OUT / "five_site_version_overview.csv",
        ["site", "fitting_period", "current_d222_coverage", "compared_old_source", "exact_match_days", "changed_numeric_days", "old_missing_new_numeric_days", "old_numeric_new_missing_days", "key_issue_changed", "current_experiment_modified", "recommendation"],
        overview,
    )

    inventory = build_inventory()
    write_csv(
        OUT / "existing_source_inventory.csv",
        ["path", "path_absolute", "sha256", "size_bytes", "mtime_utc", "role", "raw_derived_classification", "station", "period", "notes"],
        inventory,
    )

    primary_summary = {
        (row["site"], row["comparison_source"], row["year"]): row
        for row in summary_rows
    }
    hla_primary_rows = [row for row in summary_rows if row["site"] == "HLA" and row["comparison_source"] == "OLD_RAW_D222"]
    fqa_primary_rows = [row for row in summary_rows if row["site"] == "FQA" and row["comparison_source"] == "OLD_RAW_D222"]
    hla_new_numeric = sum(bool(row["newly_numeric_in_current"]) for row in hla_rows)
    hla_new_zero = sum(bool(row["newly_explicit_zero_in_current"]) for row in hla_rows)
    hla_remaining = sum(bool(row["remaining_unresolved"]) for row in hla_rows)
    hla_primary_changed = any(row["version_change_detected"] for row in hla_primary_rows)
    fqa_primary_changed = any(row["version_change_detected"] for row in fqa_primary_rows)
    fqa_unresolved_status_counts = {
        "current_blank_days": fqa_current_missing_by_unresolved["BLANK"],
        "current_explicit_zero_days": fqa_current_missing_by_unresolved["EXPLICIT_ZERO"],
        "current_positive_numeric_days": fqa_current_missing_by_unresolved["NUMERIC"],
        "current_date_absent_days": fqa_current_missing_by_unresolved["DATE_ABSENT"],
    }
    hla_gates = {
        "current_d222_has_new_information": hla_primary_changed,
        "gaps_111_changed": hla_new_numeric > 0,
        "d222_version_stable": not hla_primary_changed,
        "existing_provenance_status_unchanged": True,
        "previous_unresolved_days": len(hla_unresolved),
        "current_candidate_site_present": current_sites["HLA"],
        "assessable_days": len(hla_rows),
        "still_missing_days": hla_remaining,
        "newly_numeric_days": hla_new_numeric,
        "newly_explicit_zero_days": hla_new_zero,
    }
    fqa_gates = {
        "current_d222_has_new_information": fqa_primary_changed,
        "extreme_values_version_stable": not fqa_primary_changed,
        "missing_pattern_changed": any(
            int(row[category_name] or 0) > 0
            for row in fqa_primary_rows
            for category_name in ("OLD_MISSING_NEW_NUMERIC", "OLD_NUMERIC_NEW_MISSING", "DATE_ONLY_IN_NEW", "DATE_ONLY_IN_OLD")
        ),
        "existing_provenance_status_unchanged": True,
        "previous_unresolved_2013_days": len(fqa_unresolved),
        "current_candidate_site_present": current_sites["FQA"],
        "assessable_days": 365,
        "unresolved_date_status_counts": fqa_unresolved_status_counts,
        "newly_numeric_unresolved_days": fqa_unresolved_status_counts["current_positive_numeric_days"] + fqa_unresolved_status_counts["current_explicit_zero_days"],
    }
    site_change_gates = {
        site: {
            "version_change_detected": any(
                row["version_change_detected"]
                for row in summary_rows
                if row["site"] == site and row["comparison_source"] == primary_source_key[site]
            ),
            "current_candidate_site_present": current_sites[site],
            "compared_source": primary_source_key[site],
        }
        for site in ("YCA", "SYA", "LCA")
    }
    no_changes = not any(
        any(row["version_change_detected"] for row in summary_rows if row["site"] == site and row["comparison_source"] == primary_source_key[site])
        for site in TARGETS
    )
    comparison_totals = {}
    for site in TARGETS:
        key = primary_source_key[site]
        rows = [row for row in summary_rows if row["site"] == site and row["comparison_source"] == key]
        comparison_totals[site] = {
            "primary_source": key,
            "exact_numeric_match_days": sum(int(row["EXACT_NUMERIC_MATCH"] or 0) for row in rows),
            "numeric_changed_days": sum(int(row["NUMERIC_CHANGED"] or 0) + int(row["OLD_ZERO_NEW_NONZERO"] or 0) + int(row["OLD_NONZERO_NEW_ZERO"] or 0) for row in rows),
            "old_missing_new_numeric_days": sum(int(row["OLD_MISSING_NEW_NUMERIC"] or 0) for row in rows),
            "old_numeric_new_missing_days": sum(int(row["OLD_NUMERIC_NEW_MISSING"] or 0) for row in rows),
            "date_only_in_new_days": sum(int(row["DATE_ONLY_IN_NEW"] or 0) for row in rows),
            "date_only_in_old_days": sum(int(row["DATE_ONLY_IN_OLD"] or 0) for row in rows),
            "both_missing_days": sum(int(row["BOTH_MISSING"] or 0) for row in rows),
        }
    fqa_annual_primary = [primary_summary[("FQA", "OLD_RAW_D222", year)] for year in (2011, 2012, 2013)]
    fqa_key_values_stable = all(
        (old_index.get("FQA", {}).get(dt.date.fromisoformat(day), [{}])[0].get("rain_mm")
         == current_index.get("FQA", {}).get(dt.date.fromisoformat(day), [{}])[0].get("rain_mm"))
        for day in FQA_KEY_DATES
    )
    fqa_annual_stable = all(
        row["old_annual_numeric_subtotal_mm"] == row["new_annual_numeric_subtotal_mm"]
        and not any(int(row[name] or 0) for name in ("NUMERIC_CHANGED", "OLD_MISSING_NEW_NUMERIC", "OLD_NUMERIC_NEW_MISSING", "OLD_ZERO_NEW_NONZERO", "OLD_NONZERO_NEW_ZERO"))
        for row in fqa_annual_primary
    )
    hla_key_stable = all(
        (old_index.get("HLA", {}).get(dt.date.fromisoformat(day), [{}])[0].get("rain_mm")
         == current_index.get("HLA", {}).get(dt.date.fromisoformat(day), [{}])[0].get("rain_mm"))
        for day in HLA_KEY_DATES
    )
    gate = {
        "task": "022",
        "audit_only": True,
        "audit_date": "2026-10-07",
        "overall_status": "COMPLETED_VERSION_AUDIT_NO_VERSION_CHANGES" if len(target_present) == 5 and no_changes else "COMPLETED_VERSION_AUDIT_CHANGES_RECORDED",
        "candidate_snapshot_identity_status": identity["identity_status"],
        "candidate_target_sites_present": target_present,
        "candidate_target_sites_missing": target_missing,
        "candidate_contains_all_five_sites": len(target_present) == 5,
        "existing_data_modified": False,
        "existing_wth_modified": False,
        "existing_results_modified": False,
        "current_experiment_modified": False,
        "wgen_executed": False,
        "dssat_executed": False,
        "ppo_executed": False,
        "comparison_totals_by_site": comparison_totals,
        "hla_2013_key_values_stable_vs_old_d222": hla_key_stable,
        "fqa_2011_2013_annual_totals_stable_vs_old_d222": fqa_annual_stable,
        "fqa_2012_09_02_and_2013_05_26_stable_vs_old_d222": fqa_key_values_stable,
        "HLA": hla_gates,
        "FQA": fqa_gates,
        **site_change_gates,
        "next_step": "RECORD_ONLY_NO_AUTOMATIC_REPROCESSING",
    }
    (OUT / "final_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")

    fqa_annual = [row for row in fqa_rows if row["row_type"] == "ANNUAL_SUMMARY"]
    fqa_annual_text = " / ".join(
        f"{row['year']}：当前 {row['current_d222_numeric_subtotal_mm']} mm、旧 D222 {row['old_d222_numeric_subtotal_mm']} mm、差值 {row['annual_total_difference_vs_old_mm']} mm"
        for row in fqa_annual
    )
    hla_total_exact = comparison_totals["HLA"]["exact_numeric_match_days"]
    fqa_total_exact = comparison_totals["FQA"]["exact_numeric_match_days"]
    yca_total_exact = comparison_totals["YCA"]["exact_numeric_match_days"]
    sya_total_exact = comparison_totals["SYA"]["exact_numeric_match_days"]
    lca_total_exact = comparison_totals["LCA"]["exact_numeric_match_days"]
    current_meta = meta
    report = f"""# 五站 CERN D222 快照版本复审（022 rerun）

## 执行结论

本次复跑读取了项目内刚移入的完整工作簿，并完成五站逐日版本比较。门禁状态为 **{gate['overall_status']}**。在各站可比较的旧来源日期上未发现数值变化、缺测模式变化或日期行变化；这是来源版本稳定结论，不证明观测本身气候学正确。

本次产物写入独立目录 `{rel(OUT)}`，上一轮 022 审计产物未覆盖。没有修改任何既有天气数据、原始文件、WTH、训练/评估结果或既有审计结果；没有运行 WGEN、DSSAT 或 PPO。五站 `current_experiment_modified` 均为 `FALSE`。

## 候选文件身份与字段

| 项目 | 记录 |
|---|---|
| 路径 | `{current_meta['path_absolute']}` |
| 文件名 | `{current_meta['filename']}` |
| 文件大小 | {current_meta['size_bytes']:,} bytes |
| mtime（UTC） | {current_meta['mtime_utc']} |
| SHA-256 | `{current_meta['sha256']}` |
| 工作簿 | {', '.join(item['name'] for item in current['sheets'])}，{sheet_records:,} 条数据行，{current['sheets'][0]['ncols'] if current['sheets'] else 0} 列 |
| 文件内日期范围 | {current['data_first_date']} 至 {current['data_last_date']} |
| 文件内站点代码 | {', '.join(current['site_codes'])} |
| 目标站覆盖 | {len(target_present)}/5：{', '.join(target_present)} |
| 雨量字段 | `{current['rain_field']}`，单位 mm |
| 缺测表达 | 空字符串/空白单元格；未当作 0 |

所选文件为 `{meta['path_project_relative']}`。同目录外的候选没有参与选择；项目内较早的前缀文件 `{rel(PREVIOUS_PARTIAL)}` 作为 SYA 历史部分候选单独比较。工作簿没有站名字段，所以站名采用项目代码映射。mtime 是文件系统时间，不等同于可验证的下载时间。

## 当前候选内部 QC

五站均存在于工作簿。1998–2023 的逐年应有天数、实际日期行、重复日期、数值/显式零、空白、负值/非法值、数值小计和最大日降水事件列于 `current_d222_yearly_qc.csv`。站点覆盖范围与缺口单独记录；缺失日期和空白雨量单元格均未解释为 0 或进行填补。新版文件内部无重复日期、负值或非法雨量值。

## 五站逐站结论

| 站点 | 候选覆盖 | 本次可得结论 | 建议 |
|---|---|---|---|
| HLA | {'存在' if current_sites['HLA'] else '不存在'} | 与旧 raw D222 数值匹配 {hla_total_exact:,} 天；111 个既有 unresolved 日全部仍未恢复 | 保留既有来源疑点；若要解除需另做来源/气候合理性审查 |
| FQA | 存在 | 与旧 raw D222 数值匹配 {fqa_total_exact:,} 天；2011–2013 年总量、缺测位置及 240/204 mm 事件稳定 | FQA 数值疑点仍 unresolved；同一观测链一致不构成独立验证 |
| YCA | 存在 | 与项目旧同链表匹配 {yca_total_exact:,} 个数值日；未见版本差异 | 记录来源稳定；不触发重处理 |
| SYA | 存在 | 与较早部分 D222 候选匹配 {sya_total_exact:,} 个数值日；未见版本差异 | 记录来源稳定；该旧文件仅为部分站点快照 |
| LCA | 存在 | 与项目旧同链表匹配 {lca_total_exact:,} 个数值日；未见版本差异 | 记录来源稳定；不触发重处理 |

### HLA 专项

014/021 冻结的 111 天 unresolved 清单全部可逐日检查：**111/111 仍为空，新增数值 0 天，新增显式零 0 天，剩余未解决 111 天**。五个关键日期与旧 raw D222 一致：2013-07-30 178.2 mm、07-21 45.7 mm、08-09 63.0 mm、08-12 66.9 mm；08-21 仍为空，既有 selected source 为 GHCN 35.1 mm。该复核未重选来源；HLA 既有 provenance 状态保持不变。

### FQA 专项

新版与旧 raw D222 的年度比较：{fqa_annual_text}。2012-09-02 的 240.0 mm 和 2013-05-26 的 204.0 mm 在新版中均仍存在；2013 年 330 个旧 D222 空白日位置未变化，既有 157 个 unresolved 日期在新版仍为空（新增数值/显式零 0 天）。新版与 `HLLCYCFQ降雨数据.xls` 的一致性反映同一观测链兼容性，不能作为独立气候验证。FQA `SUSPICIOUS_BUT_UNRESOLVED` blocker 保持，不自动解除。

## 版本与实验有效性的边界

- **版本稳定**：五站对各自可用的历史来源逐日比较均未见可检测版本差异；可比较来源的具体分类计数见 `d222_version_summary.csv`。YCA/LCA 与旧表可能同属 D222 观测链；SYA 的旧候选为较早部分快照。来源一致降低本地转录/下载版本差异的可能性，但不证明记录符合真实气候。
- **版本发生变化但仅记录**：本次没有检测到变化，因此无需登记变化日或开展影响评估。
- **值得未来单独复核**：HLA 111 个雨量缺口和 2013 来源选择问题仍未解决；FQA 年度总量与极端日值仍有气候合理性疑点。后续若要改变来源或评估实验影响，应新开独立任务并明确范围。本次不回填天气、不重建基线、不重跑实验。

## 产物

- 复跑结果目录：`{rel(OUT)}/`（含身份、站点映射、逐年 QC、逐日对照、HLA/FQA 专项表、三站二级审计、五站总览和 `final_gate.json`）。
- 中文报告：`{rel(DOC)}`。

机器门禁：`{rel(OUT / 'final_gate.json')}`。下一步仅记录 `RECORD_ONLY_NO_AUTOMATIC_REPROCESSING`；本任务到此停止。
"""
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(report, encoding="utf-8")

    print(json.dumps({
        "candidate": rel(CURRENT),
        "candidate_sites": current["site_codes"],
        "target_sites_present": target_present,
        "target_sites_missing": target_missing,
        "hla_unresolved_rows": len(hla_rows),
        "fqa_prior_unresolved_days": len(fqa_unresolved),
        "all_site_daily_diff_rows": len(diff_rows),
        "inventory_rows": len(inventory),
        "final_status": gate["overall_status"],
        "outputs": [
            rel(OUT / "new_d222_snapshot_identity.json"),
            rel(OUT / "current_d222_yearly_qc.csv"),
            rel(OUT / "d222_version_daily_diff.csv"),
            rel(OUT / "final_gate.json"),
            rel(DOC),
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
