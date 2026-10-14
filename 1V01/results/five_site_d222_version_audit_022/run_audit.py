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
OUT = ROOT / "results" / "five_site_d222_version_audit_022"
DOC = ROOT / "docs" / "five_site_d222_version_audit_022.md"
CURRENT = ROOT / "my_data" / "SYCS人工大气观测降雨蒸发能见度日值.xls"
OLD_D222 = ROOT / "data" / "external" / "cern_d222" / "raw" / "excel.xls"
LEGACY = ROOT / "my_data" / "HLLCYCFQ降雨数据.xls"

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

    add(CURRENT, "only project-contained filename-suffix match; audit candidate", "raw product candidate", "SYA; CSA", "1998-2023 rows; gaps by site", "Not a complete five-site snapshot; mtime is filesystem metadata, not download time proof.")
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

    identity = {
        "snapshot_id": "CERN_D222_CURRENT_SNAPSHOT_2026_10_07",
        "identity_status": "INCOMPLETE_CANDIDATE_NOT_CONFIRMED_AS_FULL_CURRENT_SNAPSHOT",
        "audit_date": "2026-10-07",
        "selection": {
            "selected_candidate": meta,
            "matching_candidate_count": 1,
            "selection_rationale": "The only project-contained workbook whose filename ends with the requested base name is my_data/SYCS人工大气观测降雨蒸发能见度日值.xls. It is retained as a candidate only; its internal codes do not cover all five target stations.",
            "other_relevant_project_workbooks": [x for x in [old_meta, legacy_meta] if x],
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
            "full_five_site_version_comparison_possible": False,
            "reason": "The candidate contains only target code SYA; CSA is not one of the five project target sites.",
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

    cleaned_path = ROOT / "weather_clean" / "SYA_weather_cleaned.csv"
    cleaned: dict[dt.date, float | None] = {}
    for row in read_csv(cleaned_path):
        if row.get("station") != "SYA":
            continue
        date = dt.date.fromisoformat(row["date"])
        cleaned[date] = clean_number(row.get("RAIN"))
    sya_rows_by_date = current_index.get("SYA", {})
    diff_rows: list[dict[str, Any]] = []
    compare_counts: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    compare_overlap: Counter[tuple[str, int]] = Counter()
    for date in sorted(set(cleaned) | set(sya_rows_by_date)):
        old_has_date = date in cleaned
        new_has_date = date in sya_rows_by_date
        old_value = cleaned.get(date)
        new_rows = sya_rows_by_date.get(date, [])
        new_value = new_rows[0]["rain_mm"] if len(new_rows) == 1 else None
        cat = category(old_has_date, new_has_date, old_value, new_value)
        if len(new_rows) > 1:
            cat = "NUMERIC_CHANGED"  # duplicate rows are separately flagged in the QC table
        compare_counts[("SYA", date.year)][cat] += 1
        if old_has_date and new_has_date:
            compare_overlap[("SYA", date.year)] += 1
        abs_diff = abs(new_value - old_value) if old_value is not None and new_value is not None else None
        rel_diff = (new_value - old_value) / old_value if old_value not in (None, 0) and new_value is not None else None
        diff_rows.append(
            {
                "site": "SYA",
                "year": date.year,
                "date": date.isoformat(),
                "old_source_path": rel(cleaned_path),
                "old_source_role": "derived_cleaned_weather; not raw D222",
                "old_rain_mm": rounded(old_value),
                "new_source_path": rel(CURRENT),
                "new_source_role": "D222 candidate; target site present",
                "new_rain_mm": rounded(new_value),
                "comparison_category": cat,
                "absolute_difference_mm": rounded(abs_diff),
                "relative_difference": rounded(rel_diff),
                "notes": "This compares raw-like D222 candidate values to processed weather; it cannot establish D222 version change or independent validation.",
            }
        )
    write_csv(
        OUT / "d222_version_daily_diff.csv",
        ["site", "year", "date", "old_source_path", "old_source_role", "old_rain_mm", "new_source_path", "new_source_role", "new_rain_mm", "comparison_category", "absolute_difference_mm", "relative_difference", "notes"],
        diff_rows,
    )

    summary_rows: list[dict[str, Any]] = []
    for site in TARGETS:
        for year in EXPECTED_YEARS:
            counts = compare_counts.get((site, year), Counter())
            if not current_sites[site]:
                status = "NOT_COMPARABLE_SITE_ABSENT_FROM_CURRENT_CANDIDATE"
                source_path = ""
                source_role = ""
                n_dates = n_overlap = ""
                category_counts = {name: "" for name in CATEGORIES}
            elif site == "SYA":
                status = "DERIVED_SOURCE_DIAGNOSTIC_ONLY"
                source_path = rel(cleaned_path)
                source_role = "derived_cleaned_weather; not raw D222"
                n_dates = sum(counts.values())
                n_overlap = compare_overlap[(site, year)]
                category_counts = {name: counts[name] for name in CATEGORIES}
            else:
                status = "NO_COMPARABLE_OLD_SOURCE"
                source_path = ""
                source_role = ""
                n_dates = n_overlap = ""
                category_counts = {name: "" for name in CATEGORIES}
            summary_rows.append(
                {
                    "site": site,
                    "year": year,
                    "comparison_status": status,
                    "comparison_old_source": source_path,
                    "comparison_old_source_role": source_role,
                    "union_dates_compared": n_dates,
                    "overlap_dates": n_overlap,
                    **category_counts,
                    "raw_d222_version_conclusion": "NOT_ASSESSABLE_WITH_AVAILABLE_FILES",
                }
            )
    write_csv(
        OUT / "d222_version_summary.csv",
        ["site", "year", "comparison_status", "comparison_old_source", "comparison_old_source_role", "union_dates_compared", "overlap_dates", *CATEGORIES, "raw_d222_version_conclusion"],
        summary_rows,
    )

    hla_unresolved_path = ROOT / "results" / "hla_random_weather_8seed" / "014_isd50756_recovery" / "unresolved_dates_after_014.csv"
    hla_selection_path = ROOT / "results" / "hla_random_weather_8seed" / "014_isd50756_recovery" / "daily_source_selection_after_014.csv"
    hla_unresolved = read_csv(hla_unresolved_path)
    hla_selected = {row["date"]: row for row in read_csv(hla_selection_path)}
    hla_rows = []
    for item in hla_unresolved:
        previous = hla_selected.get(item["date"], {})
        hla_rows.append(
            {
                "date": item["date"],
                "previous_unresolved": True,
                "previous_unresolved_reason": item.get("reason", ""),
                "old_d222_rain_mm": previous.get("local_d222_rain_mm", ""),
                "previous_selected_rain_mm": previous.get("selected_rain_mm", ""),
                "previous_selected_source": previous.get("selected_source", ""),
                "current_candidate_has_hla_site": False,
                "current_candidate_status": "SITE_ABSENT_FROM_CANDIDATE",
                "current_d222_rain_mm": "",
                "gap_changed": "",
                "notes": "No HLA row can be read from the candidate; absence is not treated as missing/zero rainfall.",
            }
        )
    write_csv(
        OUT / "hla_111_day_recheck.csv",
        ["date", "previous_unresolved", "previous_unresolved_reason", "old_d222_rain_mm", "previous_selected_rain_mm", "previous_selected_source", "current_candidate_has_hla_site", "current_candidate_status", "current_d222_rain_mm", "gap_changed", "notes"],
        hla_rows,
    )

    hla_key_rows = []
    for date in HLA_KEY_DATES:
        previous = hla_selected.get(date, {})
        hla_key_rows.append(
            {
                "date": date,
                "old_d222_rain_mm": previous.get("local_d222_rain_mm", ""),
                "old_legacy_rain_mm": previous.get("local_legacy_rain_mm", ""),
                "previous_selected_rain_mm": previous.get("selected_rain_mm", ""),
                "previous_selected_source": previous.get("selected_source", ""),
                "current_candidate_has_hla_site": False,
                "current_candidate_status": "SITE_ABSENT_FROM_CANDIDATE",
                "current_d222_rain_mm": "",
                "version_delta_mm": "",
                "notes": "Historical values are quoted from the frozen 014 source table; no current HLA value is present in this candidate.",
            }
        )
    write_csv(
        OUT / "hla_2013_key_event_recheck.csv",
        ["date", "old_d222_rain_mm", "old_legacy_rain_mm", "previous_selected_rain_mm", "previous_selected_source", "current_candidate_has_hla_site", "current_candidate_status", "current_d222_rain_mm", "version_delta_mm", "notes"],
        hla_key_rows,
    )

    old_d222_data = load_d222(OLD_D222) if OLD_D222.is_file() else {"rows": []}
    old_index = index_rows(old_d222_data["rows"])
    legacy_fqa = load_legacy_fqa(LEGACY) if LEGACY.is_file() else {}
    prior_weather_audit_path = ROOT / "results" / "hl_fq_random_weather_8seed" / "source_audit" / "weather_source_audit_final.json"
    prior_weather_audit = json.loads(prior_weather_audit_path.read_text(encoding="utf-8"))
    fqa_unresolved = set(prior_weather_audit["sites"]["FQA"]["official_ghcn_rain_screen"]["unresolved_local_rain_days"])
    fqa_rows: list[dict[str, Any]] = []
    expected_fqa_old = {2011: 1614.4, 2012: 1378.2, 2013: 1144.4}
    for year in (2011, 2012, 2013):
        year_rows = [row for row in old_index.get("FQA", {}).items() if row[0].year == year]
        vals = [item[1][0]["rain_mm"] for item in year_rows if item[1] and item[1][0]["rain_mm"] is not None and item[1][0]["value_status"] in ("NUMERIC", "EXPLICIT_ZERO", "NEGATIVE")]
        blank_count = sum(1 for _, items in year_rows if items and items[0]["value_status"] == "BLANK")
        fqa_rows.append(
            {
                "row_type": "ANNUAL_SUMMARY",
                "date": "",
                "year": year,
                "old_d222_rain_mm": "",
                "legacy_rain_mm": "",
                "old_d222_numeric_days": len(vals),
                "old_d222_blank_days": blank_count,
                "old_d222_numeric_subtotal_mm": rounded(sum(vals)) if vals else "",
                "previous_reported_total_mm": expected_fqa_old[year],
                "previous_unresolved_2013": "",
                "current_candidate_has_fqa_site": False,
                "current_candidate_status": "SITE_ABSENT_FROM_CANDIDATE",
                "current_d222_rain_mm": "",
                "notes": "Totals are recomputed from the historical 009 D222 workbook, not from the current candidate; the FQA site is absent from that candidate.",
            }
        )
    for date_text in FQA_KEY_DATES:
        date = dt.date.fromisoformat(date_text)
        old_items = old_index.get("FQA", {}).get(date, [])
        old_value = old_items[0]["rain_mm"] if old_items else None
        fqa_rows.append(
            {
                "row_type": "KEY_EVENT",
                "date": date_text,
                "year": date.year,
                "old_d222_rain_mm": rounded(old_value),
                "legacy_rain_mm": rounded(legacy_fqa.get(date)),
                "old_d222_numeric_days": "",
                "old_d222_blank_days": "",
                "old_d222_numeric_subtotal_mm": "",
                "previous_reported_total_mm": "",
                "previous_unresolved_2013": date_text in fqa_unresolved,
                "current_candidate_has_fqa_site": False,
                "current_candidate_status": "SITE_ABSENT_FROM_CANDIDATE",
                "current_d222_rain_mm": "",
                "notes": "Historical D222/legacy values are same-chain references; no current FQA value is present in this candidate.",
            }
        )
    for offset in range(365):
        date = dt.date(2013, 1, 1) + dt.timedelta(days=offset)
        old_items = old_index.get("FQA", {}).get(date, [])
        old_value = old_items[0]["rain_mm"] if old_items else None
        fqa_rows.append(
            {
                "row_type": "DAILY_2013_OLD_MISSINGNESS_AND_PRIOR_UNRESOLVED",
                "date": date.isoformat(),
                "year": 2013,
                "old_d222_rain_mm": rounded(old_value),
                "legacy_rain_mm": rounded(legacy_fqa.get(date)),
                "old_d222_numeric_days": "",
                "old_d222_blank_days": "",
                "old_d222_numeric_subtotal_mm": "",
                "previous_reported_total_mm": "",
                "previous_unresolved_2013": date.isoformat() in fqa_unresolved,
                "current_candidate_has_fqa_site": False,
                "current_candidate_status": "SITE_ABSENT_FROM_CANDIDATE",
                "current_d222_rain_mm": "",
                "notes": "Exact prior unresolved membership comes from the 008/009 weather source audit; old blank remains blank and is not zero.",
            }
        )
    write_csv(
        OUT / "fqa_2011_2013_recheck.csv",
        ["row_type", "date", "year", "old_d222_rain_mm", "legacy_rain_mm", "old_d222_numeric_days", "old_d222_blank_days", "old_d222_numeric_subtotal_mm", "previous_reported_total_mm", "previous_unresolved_2013", "current_candidate_has_fqa_site", "current_candidate_status", "current_d222_rain_mm", "notes"],
        fqa_rows,
    )

    # Per-site YCA/SYA/LCA annual records: raw D222 version conclusions stay null
    # unless an old raw D222 comparator is available for the same site.
    for site in ("YCA", "SYA", "LCA"):
        rows = []
        for summary in [item for item in summary_rows if item["site"] == site]:
            year = summary["year"]
            qc = next(item for item in qc_rows if item["project_site"] == site and item["year"] == year)
            old_values = [v for date, v in cleaned.items() if date.year == year] if site == "SYA" else []
            new_values = [row["rain_mm"] for row in by_site_year.get((site, year), []) if row["rain_mm"] is not None and row["value_status"] in ("NUMERIC", "EXPLICIT_ZERO")]
            rows.append(
                {
                    "site": site,
                    "fitting_period": TARGETS[site]["fitting_period"],
                    "year": year,
                    "current_site_present": current_sites[site],
                    "old_comparison_source": summary["comparison_old_source"],
                    "old_comparison_source_role": summary["comparison_old_source_role"],
                    "raw_d222_version_comparison_status": "NOT_ASSESSABLE_NO_OLD_RAW_D222_COMPARATOR" if site == "SYA" and current_sites[site] else ("NOT_ASSESSABLE_SITE_ABSENT" if not current_sites[site] else "NOT_ASSESSABLE"),
                    "overlap_dates": summary["overlap_dates"],
                    "exact_match_days": summary["EXACT_NUMERIC_MATCH"],
                    "numeric_changed_days": summary["NUMERIC_CHANGED"],
                    "old_missing_new_numeric_days": summary["OLD_MISSING_NEW_NUMERIC"],
                    "old_numeric_new_missing_days": summary["OLD_NUMERIC_NEW_MISSING"],
                    "old_cleaned_subtotal_mm": rounded(sum(v for v in old_values if v is not None)) if site == "SYA" and old_values else "",
                    "current_candidate_numeric_subtotal_mm": rounded(sum(new_values)) if current_sites[site] and new_values else (0.0 if current_sites[site] else ""),
                    "annual_rainfall_difference_mm": "",
                    "annual_max_event_difference_mm": "",
                    "notes": "Processed weather includes values for days blank in candidate D222; these annual sums/extremes are not directly comparable." if site == "SYA" else "Candidate lacks this site; no current-version difference can be calculated.",
                    "current_experiment_modified": False,
                }
            )
        write_csv(
            OUT / f"{site.lower()}_version_audit.csv",
            ["site", "fitting_period", "year", "current_site_present", "old_comparison_source", "old_comparison_source_role", "raw_d222_version_comparison_status", "overlap_dates", "exact_match_days", "numeric_changed_days", "old_missing_new_numeric_days", "old_numeric_new_missing_days", "old_cleaned_subtotal_mm", "current_candidate_numeric_subtotal_mm", "annual_rainfall_difference_mm", "annual_max_event_difference_mm", "notes", "current_experiment_modified"],
            rows,
        )

    overview = []
    for site, info in TARGETS.items():
        site_summary = [row for row in summary_rows if row["site"] == site]
        site_present = current_sites[site]
        if site == "SYA":
            totals = {name: sum(int(row[name] or 0) for row in site_summary) for name in CATEGORIES}
            compare_source = f"{rel(cleaned_path)} (derived; not old raw D222)"
            key_issue = "Numeric candidate values match the derived cleaned table; its extra filled values do not establish raw D222 version change."
        else:
            totals = {name: "" for name in CATEGORIES}
            compare_source = "Not comparable: target site absent from candidate"
            key_issue = "Current candidate lacks this site; version status cannot be determined."
        overview.append(
            {
                "site": site,
                "fitting_period": info["fitting_period"],
                "current_d222_coverage": (
                    "1998-01-01 to 2023-12-31 rows; 1999-2000 absent" if site == "SYA" else "SITE_ABSENT_FROM_CANDIDATE"
                ),
                "compared_old_source": compare_source,
                "exact_match_days": totals["EXACT_NUMERIC_MATCH"],
                "changed_numeric_days": totals["NUMERIC_CHANGED"],
                "old_missing_new_numeric_days": totals["OLD_MISSING_NEW_NUMERIC"],
                "old_numeric_new_missing_days": totals["OLD_NUMERIC_NEW_MISSING"],
                "key_issue_changed": key_issue,
                "current_experiment_modified": False,
                "recommendation": "Record only; provide the complete current five-site D222 snapshot in the project before any full version comparison.",
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

    hla_gates = {
        "current_d222_has_new_information": None,
        "gaps_111_changed": None,
        "d222_version_stable": None,
        "existing_provenance_status_unchanged": True,
        "previous_unresolved_days": len(hla_unresolved),
        "current_candidate_site_present": False,
        "assessable_days": 0,
    }
    fqa_gates = {
        "current_d222_has_new_information": None,
        "extreme_values_version_stable": None,
        "missing_pattern_changed": None,
        "existing_provenance_status_unchanged": True,
        "previous_unresolved_2013_days": len(fqa_unresolved),
        "current_candidate_site_present": False,
        "assessable_days": 0,
    }
    gate = {
        "task": "022",
        "audit_only": True,
        "audit_date": "2026-10-07",
        "overall_status": "BLOCKED_INCOMPLETE_CURRENT_SNAPSHOT",
        "candidate_snapshot_identity_status": identity["identity_status"],
        "candidate_target_sites_present": target_present,
        "candidate_target_sites_missing": target_missing,
        "candidate_contains_all_five_sites": False,
        "existing_data_modified": False,
        "existing_wth_modified": False,
        "existing_results_modified": False,
        "current_experiment_modified": False,
        "wgen_executed": False,
        "dssat_executed": False,
        "ppo_executed": False,
        "HLA": hla_gates,
        "FQA": fqa_gates,
        "YCA": {"version_change_detected": None, "current_candidate_site_present": False},
        "SYA": {"version_change_detected": None, "current_candidate_site_present": True, "numeric_matches_to_derived_cleaned": 1239, "raw_old_d222_comparator_available": False},
        "LCA": {"version_change_detected": None, "current_candidate_site_present": False},
        "derived_weather_diagnostic": {
            "SYA_exact_numeric_matches": sum(1 for row in diff_rows if row["comparison_category"] == "EXACT_NUMERIC_MATCH"),
            "SYA_old_numeric_new_missing": sum(1 for row in diff_rows if row["comparison_category"] == "OLD_NUMERIC_NEW_MISSING"),
            "SYA_numeric_changed": sum(1 for row in diff_rows if row["comparison_category"] == "NUMERIC_CHANGED"),
            "interpretation": "Derived weather values are not a raw D222 version comparator; blanks and filled values remain distinct.",
        },
        "next_step": "RECORD_ONLY_NO_AUTOMATIC_REPROCESSING",
        "required_input_for_full_audit": "Place the complete current five-site CERN D222 workbook snapshot inside the project directory, then rerun Task 022.",
    }
    (OUT / "final_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")

    fqa_annual = [row for row in fqa_rows if row["row_type"] == "ANNUAL_SUMMARY"]
    fqa_annual_text = " / ".join(f"{row['year']}: {row['old_d222_numeric_subtotal_mm']} mm" for row in fqa_annual)
    current_meta = meta
    report = f"""# 五站 CERN D222 候选快照版本审计（022）

## 执行结论

本次只读审计没有找到项目内可确认覆盖五站的当前 D222 快照。唯一文件名后缀匹配候选是 `{meta['path_project_relative']}`，但工作簿实际站点代码只有 `{', '.join(current['site_codes'])}`；五个目标站中仅 `SYA` 出现，`HLA/FQA/YCA/LCA` 均缺席。因此任务状态为 **BLOCKED_INCOMPLETE_CURRENT_SNAPSHOT**：下文的缺站结果表示“无法比较”，不表示新版沿用旧值或仍为空。

未修改任何既有天气数据、原始文件、WTH、训练/评估结果或既有审计文件；未运行 WGEN、DSSAT 或 PPO。所有 `current_experiment_modified` 均为 `FALSE`。

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

工作簿没有单独的站名字段。`SYA` 到“沈阳”的对应关系来自项目站点映射；没有把这称为工作簿内部的站名确认。mtime 记录的是文件系统时间，不能证明下载时间；它为 2026-04-01，早于本次审计日期。另有旧 D222 原件 `data/external/cern_d222/raw/excel.xls`（009 审计所用）及旧本地表，但它们不能补成同一份当前五站快照。

## 当前候选内部 QC

候选文件中 `SYA` 有 8,765 行：1998 年与 2001–2023 年有日期行，1999–2000 年没有日期行；无重复日期。全期可解析数值 1,239 天（含显式 0 共 59 天），空白 7,526 天，负值 0，非法值 0；数值小计 13,327.8 mm，最大日值 102.0 mm（2012-08-04）。上述小计只加总明示数值，所有空白保持缺测。其他四个目标站在此候选中没有行，不能将其记为 26 年空白观测。

## 五站逐站结论

| 站点 | 候选覆盖 | 本次可得结论 | 建议 |
|---|---|---|---|
| HLA | 不存在 | 111 天旧 unresolved 清单均无法在候选中核查；关键 2013 值也没有新版站点记录 | 提供含 HLA 的当前快照后重审；014/021 的 `PARTIAL` 状态保持 |
| FQA | 不存在 | 2011–2013 新版总量、空白格局及 240/204 mm 事件均无法复核；既有 157 天清单不变 | 提供含 FQA 的当前快照后重审；blocker 不自动解除 |
| YCA | 不存在 | 没有当前候选行，版本变化不可判断 | 完整快照到位后单独复核 |
| SYA | 存在 | 1,239 个候选数值与清洗天气表对应值全数相同；清洗表另有 7,161 个数值落在候选空白日，属于派生填充值对比，不能证明原始 D222 版本变化 | 需要旧的同站 raw D222 才能判版本稳定/变化 |
| LCA | 不存在 | 没有当前候选行，版本变化不可判断 | 完整快照到位后单独复核 |

### HLA 专项

014/021 冻结的 111 天 unresolved 清单已逐行输出到 `hla_111_day_recheck.csv`。本候选不含 `HLA`，所以可复核天数为 0；“新版是否补值”“缺口是否变化”“版本是否稳定”均为 **无法判断**。五个重点日期的历史 D222/selected-source 值另列于 `hla_2013_key_event_recheck.csv`：2013-07-30 为 178.2 mm，07-21 为 45.7 mm，08-09 为 63.0 mm，08-12 为 66.9 mm，08-21 的旧 D222 为空且既有选值为 GHCN 35.1 mm。当前候选没有 HLA 值可比，既有 provenance 状态不变。

### FQA 专项

以下为旧 009 D222 原件重新计算的历史值，不是本次候选的新值：{fqa_annual_text}。旧文件中 2012-09-02 为 240.0 mm、2013-05-26 为 204.0 mm；2013 有 330 个旧 D222 空白日，历史来源审计记录 157 个 unresolved 日。`fqa_2011_2013_recheck.csv` 保留 2013 全年逐日历史值、旧表空白状态和 unresolved 标记。本候选没有 FQA 站，因此无法回答这些极端事件是否仍在新版中；旧 D222 与 `HLLCYCFQ降雨数据.xls` 处于同一观测链，二者一致不构成独立验证。FQA blocker 与先前 `SUSPICIOUS_BUT_UNRESOLVED` 结论保持，不自动解除。

## 版本与实验有效性的边界

- **版本稳定**：目前只能说 SYA 候选中的 1,239 个明确数值与清洗天气表相同；由于后者是派生数据且无旧 SYA raw D222，不能据此判原始版本稳定。
- **版本发生变化但仅记录**：本次没有发现可确认的同站 raw D222 新旧数值变化。清洗表在 D222 空白日仍有数值，属于处理/填补差别，已标明来源类别，不作为原始版本变化。
- **值得未来单独复核**：获取项目内完整的当前五站 D222 快照后，先重做逐日版本审计；若出现变化，再另开影响评估任务。本次不回填天气、不重建基线、不重跑实验。

## 产物

- `results/five_site_d222_version_audit_022/new_d222_snapshot_identity.json`
- `results/five_site_d222_version_audit_022/site_mapping.csv`
- `results/five_site_d222_version_audit_022/existing_source_inventory.csv`
- `results/five_site_d222_version_audit_022/current_d222_yearly_qc.csv`
- `results/five_site_d222_version_audit_022/current_d222_daily_normalized.csv`
- `results/five_site_d222_version_audit_022/d222_version_daily_diff.csv` 与 `d222_version_summary.csv`
- `results/five_site_d222_version_audit_022/hla_111_day_recheck.csv`、`hla_2013_key_event_recheck.csv`、`fqa_2011_2013_recheck.csv`
- `results/five_site_d222_version_audit_022/yca_version_audit.csv`、`sya_version_audit.csv`、`lca_version_audit.csv`、`five_site_version_overview.csv`
- `results/five_site_d222_version_audit_022/final_gate.json`

机器门禁：`{(OUT / 'final_gate.json').relative_to(ROOT).as_posix()}`。下一步仅需将完整的当前五站原始快照放入项目目录，再启动一个独立的只读版本审计；本任务到此停止。
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
        "sya_daily_diff_rows": len(diff_rows),
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
