from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "benchmark_results" / "030_00_weather_qc"


RANGES = {
    "TMAX": (-60.0, 60.0),
    "TMIN": (-80.0, 50.0),
    "RAIN": (0.0, 500.0),
    "SRAD": (0.0, 60.0),
}

MISSING_SENTINELS = {-99.0, -99.00}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def year_from_date_token(token: str) -> int | None:
    token = token.strip()
    if not token.isdigit():
        return None
    if len(token) == 7:
        return int(token[:4])
    if len(token) == 5:
        yy = int(token[:2])
        return 2000 + yy if yy < 50 else 1900 + yy
    return None


def doy_from_date_token(token: str) -> int | None:
    token = token.strip()
    if not token.isdigit():
        return None
    if len(token) == 7:
        return int(token[4:])
    if len(token) == 5:
        return int(token[2:])
    return None


def station_from_path(path: Path, source_group: str) -> str:
    if source_group == "hla_long":
        return "HLA"
    parts = path.parts
    try:
        idx = parts.index("multisite_new_cultivar_inputs_013")
        st = parts[idx + 1]
    except Exception:
        st = path.parent.name
    return "HLA" if st.upper() == "HL" else st.upper()


def discover_weather_files() -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    base = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013"
    if base.exists():
        for station_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
            for path in sorted(station_dir.glob("*.WTH")):
                files.append(
                    {
                        "path": path,
                        "source_group": "multisite_013",
                        "station": station_from_path(path, "multisite_013"),
                    }
                )

    hla_runs = (
        ROOT
        / "DSSAT_auto_validation"
        / "HLA_2004"
        / "candidate_ic055_n025_null_2004_2023"
        / "runs"
    )
    if hla_runs.exists():
        for run_dir in sorted([p for p in hla_runs.iterdir() if p.is_dir()]):
            input_dir = run_dir / "input"
            for path in sorted(input_dir.glob("*.WTH")):
                files.append(
                    {
                        "path": path,
                        "source_group": "hla_long",
                        "station": "HLA",
                    }
                )
    return files


@dataclass
class ParsedWeather:
    header_idx: int
    columns: list[str]
    rows: list[dict[str, Any]]
    lines: list[str]


def parse_wth(path: Path) -> ParsedWeather:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_idx = -1
    columns: list[str] = []
    for i, line in enumerate(lines):
        if line.lstrip().startswith("@") and "DATE" in line.upper():
            header_idx = i
            columns = line.replace("@", " ", 1).split()
            columns = [c.upper() for c in columns]
            break
    if header_idx < 0:
        raise ValueError(f"No weather data header found: {path}")

    rows: list[dict[str, Any]] = []
    for i in range(header_idx + 1, len(lines)):
        line = lines[i]
        if not line.strip() or line.lstrip().startswith(("*", "@", "$", "!")):
            continue
        tokens = line.split()
        if len(tokens) < len(columns):
            continue
        record: dict[str, Any] = {
            "line_index": i,
            "line_no": i + 1,
            "tokens": tokens,
            "values": {},
        }
        for j, col in enumerate(columns):
            value = tokens[j]
            if col == "DATE":
                record["date"] = value
                record["year"] = year_from_date_token(value)
                record["doy"] = doy_from_date_token(value)
            else:
                try:
                    record["values"][col] = float(value)
                except ValueError:
                    record["values"][col] = None
        rows.append(record)
    return ParsedWeather(header_idx=header_idx, columns=columns, rows=rows, lines=lines)


def invalid_reason(var: str, value: float | None) -> str | None:
    if value is None:
        return f"{var}_not_numeric"
    if value in MISSING_SENTINELS:
        return f"{var}_missing_sentinel_-99"
    if var not in RANGES:
        return None
    lo, hi = RANGES[var]
    if value < lo:
        return f"{var}_below_{lo:g}"
    if value > hi:
        return f"{var}_above_{hi:g}"
    return None


def nearest_valid(rows: list[dict[str, Any]], idx: int, var: str, direction: int) -> tuple[int, float] | None:
    j = idx + direction
    while 0 <= j < len(rows):
        val = rows[j]["values"].get(var)
        if invalid_reason(var, val) is None:
            return j, float(val)
        j += direction
    return None


def interpolate(rows: list[dict[str, Any]], idx: int, var: str) -> tuple[float | None, str]:
    prev_val = nearest_valid(rows, idx, var, -1)
    next_val = nearest_valid(rows, idx, var, 1)
    if prev_val and next_val:
        p_idx, p = prev_val
        n_idx, n = next_val
        frac = (idx - p_idx) / (n_idx - p_idx)
        return p + frac * (n - p), "linear_prev_next"
    if prev_val:
        return prev_val[1], "carry_previous_valid"
    if next_val:
        return next_val[1], "carry_next_valid"
    return None, "no_valid_neighbor"


def format_token(old: str, value: float) -> str:
    # Keep DSSAT-style compact one-decimal values without trying to preserve every column.
    if re.fullmatch(r"-?\d+(\.\d+)?", old):
        return f"{value:.1f}"
    return f"{value:.1f}"


def rewrite_lines(parsed: ParsedWeather, corrections: list[dict[str, Any]]) -> list[str]:
    lines = list(parsed.lines)
    by_line_var = {(c["line_index"], c["variable"]): c for c in corrections if c["status"] == "corrected"}
    col_index = {col: i for i, col in enumerate(parsed.columns)}
    rows_by_line = {r["line_index"]: r for r in parsed.rows}
    for line_idx in sorted({k[0] for k in by_line_var}):
        row = rows_by_line[line_idx]
        tokens = list(row["tokens"])
        for var in parsed.columns:
            key = (line_idx, var)
            if key not in by_line_var:
                continue
            j = col_index[var]
            tokens[j] = format_token(tokens[j], float(by_line_var[key]["new_value"]))
        lines[line_idx] = " ".join(tokens)
    return lines


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    corrected_root = OUT_DIR / "weather_corrected"
    corrected_root.mkdir(parents=True, exist_ok=True)
    # Clear only generated corrected WTH copies so stale outputs from a previous
    # QC rule cannot be mistaken for current results.
    for old_copy in corrected_root.rglob("*.WTH"):
        if OUT_DIR not in old_copy.resolve().parents:
            raise RuntimeError(f"Refusing to remove unexpected path: {old_copy}")
        old_copy.unlink()

    weather_files = discover_weather_files()
    file_rows: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    files_with_correction: dict[Path, list[dict[str, Any]]] = {}
    year_candidates: list[dict[str, Any]] = []

    for entry in weather_files:
        path = Path(entry["path"])
        rel = path.relative_to(ROOT).as_posix()
        raw_hash_before = sha256(path)
        try:
            parsed = parse_wth(path)
        except Exception as exc:
            file_rows.append(
                {
                    "station": entry["station"],
                    "source_group": entry["source_group"],
                    "path": rel,
                    "filename": path.name,
                    "n_rows": 0,
                    "years": "",
                    "sha256_raw": raw_hash_before,
                    "parse_status": f"error: {exc}",
                }
            )
            anomalies.append(
                {
                    "station": entry["station"],
                    "source_group": entry["source_group"],
                    "path": rel,
                    "filename": path.name,
                    "line_no": "",
                    "year": "",
                    "doy": "",
                    "variable": "FILE",
                    "old_value": "",
                    "reason": "parse_error_no_weather_DATE_header",
                    "correction_method": "flagged_only",
                    "new_value": "",
                    "status": "flagged_only",
                }
            )
            if raw_hash_before != sha256(path):
                raise RuntimeError(f"Original file changed unexpectedly: {path}")
            continue
        year_counts: dict[int, int] = {}
        year_bad_counts: dict[int, int] = {}
        year_missing_counts: dict[int, int] = {}
        year_corrected_counts: dict[int, int] = {}
        for row in parsed.rows:
            y = row.get("year")
            if y is not None:
                year_counts[int(y)] = year_counts.get(int(y), 0) + 1
        years = sorted({r.get("year") for r in parsed.rows if r.get("year") is not None})
        file_rows.append(
            {
                "station": entry["station"],
                "source_group": entry["source_group"],
                "path": rel,
                "filename": path.name,
                "n_rows": len(parsed.rows),
                "years": ";".join(str(y) for y in years),
                "sha256_raw": raw_hash_before,
                "parse_status": "ok",
            }
        )

        file_corrections: list[dict[str, Any]] = []
        for idx, row in enumerate(parsed.rows):
            row_year = row.get("year")
            for var in ["SRAD", "TMAX", "TMIN", "RAIN"]:
                if var not in parsed.columns:
                    continue
                value = row["values"].get(var)
                reason = invalid_reason(var, value)
                if reason is None:
                    continue
                if value in MISSING_SENTINELS:
                    new_value, method = None, "flag_only_missing_sentinel"
                    if row_year is not None:
                        year_missing_counts[int(row_year)] = year_missing_counts.get(int(row_year), 0) + 1
                else:
                    new_value, method = interpolate(parsed.rows, idx, var)
                    if row_year is not None and new_value is None:
                        year_bad_counts[int(row_year)] = year_bad_counts.get(int(row_year), 0) + 1
                    if row_year is not None and new_value is not None:
                        year_corrected_counts[int(row_year)] = year_corrected_counts.get(int(row_year), 0) + 1
                anomaly = {
                    "station": entry["station"],
                    "source_group": entry["source_group"],
                    "path": rel,
                    "filename": path.name,
                    "line_no": row["line_no"],
                    "year": row.get("year"),
                    "doy": row.get("doy"),
                    "variable": var,
                    "old_value": value,
                    "reason": reason,
                    "correction_method": method,
                    "new_value": new_value,
                    "status": "corrected" if new_value is not None else "flagged_only",
                    "line_index": row["line_index"],
                }
                anomalies.append({k: v for k, v in anomaly.items() if k != "line_index"})
                manifest.append({k: v for k, v in anomaly.items() if k != "line_index"})
                if new_value is not None:
                    file_corrections.append(anomaly)

            tmax = row["values"].get("TMAX")
            tmin = row["values"].get("TMIN")
            if tmax is not None and tmin is not None and invalid_reason("TMAX", tmax) is None and invalid_reason("TMIN", tmin) is None and tmin > tmax:
                if row.get("year") is not None:
                    year_bad_counts[int(row["year"])] = year_bad_counts.get(int(row["year"]), 0) + 1
                anomalies.append(
                    {
                        "station": entry["station"],
                        "source_group": entry["source_group"],
                        "path": rel,
                        "filename": path.name,
                        "line_no": row["line_no"],
                        "year": row.get("year"),
                        "doy": row.get("doy"),
                        "variable": "TMIN_TMAX",
                        "old_value": f"TMIN={tmin};TMAX={tmax}",
                        "reason": "TMIN_above_TMAX",
                        "correction_method": "flag_only_cross_field",
                        "new_value": "",
                        "status": "flagged_only",
                    }
                )

        if file_corrections:
            files_with_correction[path] = file_corrections
            corrected_lines = rewrite_lines(parsed, file_corrections)
            station_dir = OUT_DIR / "weather_corrected" / entry["station"] / entry["source_group"]
            station_dir.mkdir(parents=True, exist_ok=True)
            out_path = station_dir / path.name
            out_path.write_text("\n".join(corrected_lines) + "\n", encoding="utf-8")
            corrected_hash = sha256(out_path)
            for m in manifest:
                if m["path"] == rel and m["status"] == "corrected":
                    m["corrected_path"] = out_path.relative_to(ROOT).as_posix()
                    m["sha256_corrected"] = corrected_hash

        raw_hash_after = sha256(path)
        if raw_hash_before != raw_hash_after:
            raise RuntimeError(f"Original file changed unexpectedly: {path}")

        for y, n in sorted(year_counts.items()):
            unresolved_missing = year_missing_counts.get(y, 0)
            unresolved_bad = year_bad_counts.get(y, 0)
            corrected = year_corrected_counts.get(y, 0)
            usable = n >= 300 and unresolved_missing == 0 and unresolved_bad == 0
            if usable:
                reason = "usable_raw" if corrected == 0 else "usable_with_corrected_copy"
            elif n < 300:
                reason = "too_few_daily_rows"
            elif unresolved_missing:
                reason = "unresolved_missing_sentinel"
            else:
                reason = "unresolved_physical_or_cross_field_anomaly"
            years_in_file = len(year_counts)
            if entry["station"] == "HLA" and entry["source_group"] == "hla_long":
                source_priority = 0
            elif years_in_file == 1 and entry["source_group"] == "multisite_013":
                source_priority = 1
            else:
                source_priority = 2
            year_candidates.append(
                {
                    "station": entry["station"],
                    "year": y,
                    "source_group": entry["source_group"],
                    "filename": path.name,
                    "path": rel,
                    "n_rows_year": n,
                    "missing_sentinel_count": unresolved_missing,
                    "unresolved_anomaly_count": unresolved_bad,
                    "corrected_value_count": corrected,
                    "usable": "yes" if usable else "no",
                    "reason": reason,
                    "source_priority": source_priority,
                    "years_in_file": years_in_file,
                }
            )

    def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    write_csv(
        OUT_DIR / "030_00_weather_files.csv",
        file_rows,
        ["station", "source_group", "path", "filename", "n_rows", "years", "sha256_raw", "parse_status"],
    )
    write_csv(
        OUT_DIR / "030_00_anomalies.csv",
        anomalies,
        [
            "station",
            "source_group",
            "path",
            "filename",
            "line_no",
            "year",
            "doy",
            "variable",
            "old_value",
            "reason",
            "correction_method",
            "new_value",
            "status",
        ],
    )
    write_csv(
        OUT_DIR / "030_00_weather_qc_manifest.csv",
        manifest,
        [
            "station",
            "source_group",
            "path",
            "filename",
            "line_no",
            "year",
            "doy",
            "variable",
            "old_value",
            "reason",
            "correction_method",
            "new_value",
            "status",
            "corrected_path",
            "sha256_corrected",
        ],
    )

    selected_by_station_year: dict[tuple[str, int], dict[str, Any]] = {}
    for row in year_candidates:
        key = (row["station"], int(row["year"]))
        prev = selected_by_station_year.get(key)
        if prev is None:
            selected_by_station_year[key] = row
            continue
        row_key = (int(row["source_priority"]), row["usable"] != "yes", int(row["years_in_file"]))
        prev_key = (int(prev["source_priority"]), prev["usable"] != "yes", int(prev["years_in_file"]))
        if row_key < prev_key:
            selected_by_station_year[key] = row

    usability_rows: list[dict[str, Any]] = []
    for row in sorted(selected_by_station_year.values(), key=lambda r: (r["station"], int(r["year"]))):
        out_row = dict(row)
        out_row["selected_for_next_rl"] = "yes" if row["usable"] == "yes" else "no"
        out_row.pop("source_priority", None)
        usability_rows.append(out_row)

    write_csv(
        OUT_DIR / "030_00_weather_year_usability.csv",
        usability_rows,
        [
            "station",
            "year",
            "selected_for_next_rl",
            "usable",
            "reason",
            "source_group",
            "filename",
            "path",
            "n_rows_year",
            "missing_sentinel_count",
            "unresolved_anomaly_count",
            "corrected_value_count",
            "years_in_file",
        ],
    )

    anomaly_by_station: dict[str, int] = {}
    corrected_by_station: dict[str, int] = {}
    for a in anomalies:
        anomaly_by_station[a["station"]] = anomaly_by_station.get(a["station"], 0) + 1
    for m in manifest:
        if m["status"] == "corrected":
            corrected_by_station[m["station"]] = corrected_by_station.get(m["station"], 0) + 1

    result = {
        "task": "030_00_weather_qc",
        "n_files_scanned": len(file_rows),
        "n_anomaly_records": len(anomalies),
        "n_corrected_values": sum(1 for m in manifest if m["status"] == "corrected"),
        "n_usable_station_years": sum(1 for r in usability_rows if r["selected_for_next_rl"] == "yes"),
        "n_unusable_station_years": sum(1 for r in usability_rows if r["selected_for_next_rl"] == "no"),
        "stations_scanned": sorted({r["station"] for r in file_rows}),
        "anomaly_by_station": anomaly_by_station,
        "corrected_by_station": corrected_by_station,
        "output_dir": OUT_DIR.relative_to(ROOT).as_posix(),
        "original_files_unchanged": True,
    }
    (OUT_DIR / "030_00_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# 030_00 Weather QC record",
        "",
        "## Scope",
        "",
        "- Scanned current mainline `.WTH` files only.",
        "- Did not scan `benchmark_results/` snapshots or archives.",
        "- Did not run DSSAT or RL training.",
        "- Original `.WTH` files were not modified; corrected files are derived copies only.",
        "",
        "## Summary",
        "",
        f"- Files scanned: {result['n_files_scanned']}",
        f"- Anomaly records: {result['n_anomaly_records']}",
        f"- Corrected numeric values: {result['n_corrected_values']}",
        f"- Usable station-years for next RL: {result['n_usable_station_years']}",
        f"- Unusable station-years: {result['n_unusable_station_years']}",
        f"- Stations: {', '.join(result['stations_scanned'])}",
        "",
        "## Outputs",
        "",
        "- `030_00_weather_files.csv`",
        "- `030_00_anomalies.csv`",
        "- `030_00_weather_qc_manifest.csv`",
        "- `030_00_weather_year_usability.csv`",
        "- `weather_corrected/`",
        "- `030_00_result.json`",
        "",
        "## Notes",
        "",
        "- `HL` input naming is reported as `HLA`.",
        "- Physical-range violations are corrected by interpolation in derived copies.",
        "- Cross-field anomalies are flagged unless a physical-range violation identifies the offending variable.",
    ]
    if anomalies:
        md_lines.extend(["", "## Anomaly list", ""])
        for a in anomalies:
            md_lines.append(
                f"- {a['station']} {a['filename']} year={a['year']} doy={a['doy']} "
                f"{a['variable']}={a['old_value']} ({a['reason']}), status={a['status']}"
            )
    (OUT_DIR / "030_00_weather_qc_record.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
