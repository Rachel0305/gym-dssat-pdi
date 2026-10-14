"""Build FQA 2005-2013 fitting weather and a static CLI; no DSSAT/WGEN/PPO run."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import xlrd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
START, END = dt.date(2005, 1, 1), dt.date(2013, 12, 31)
SOURCE_PATHS = {
    "RAIN": ROOT / "data/external/人工大气观测降雨蒸发能见度日值.xls",
    "TMAX_TMIN": ROOT / "my_data/T2.xls",
    "SRAD": ROOT / "my_data/D32.xls",
    "NASA_CANDIDATES": ROOT / "results/hl_fq_random_weather_8seed/source_audit/nasa_candidate_gapfill.csv",
    "CLI_BUILDER": ROOT / "scripts/build_dssat_cli.py",
    "STATION_WTH_HEADER": ROOT / "DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/FQ/CNFQ0701.WTH",
}
FIELDS = ["DATE", "SRAD", "TMAX", "TMIN", "RAIN"]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number(value):
    if value is None or str(value).strip() == "":
        return None
    try:
        x = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"unparseable numeric cell {value!r}")
    if not math.isfinite(x):
        raise ValueError(f"nonfinite numeric cell {value!r}")
    return x


def load_xls(path: Path, columns: dict[str, int], required_header: dict[int, str]):
    book = xlrd.open_workbook(str(path), on_demand=True)
    values = {}
    try:
        for sheet in book.sheets():
            if sheet.nrows < 2:
                continue
            heads = [str(v).strip() for v in sheet.row_values(1)]
            if not all(i < len(heads) and heads[i] == label for i, label in required_header.items()):
                continue
            for row_index in range(2, sheet.nrows):
                row = sheet.row_values(row_index)
                if str(row[0]).strip() != "FQA":
                    continue
                day = dt.date(int(row[1]), int(row[2]), int(row[3]))
                if not START <= day <= END:
                    continue
                if day in values:
                    raise ValueError(f"duplicate FQA date in {path.name}: {day}")
                values[day] = {name: number(row[index]) for name, index in columns.items()}
    finally:
        book.release_resources()
    if not values:
        raise ValueError(f"no matching FQA data in {path}")
    return values


def load_candidates(path: Path):
    candidates = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["site"] != "FQA":
                continue
            day = dt.date.fromisoformat(row["date"])
            if not START <= day <= END:
                raise ValueError(f"candidate outside fit range: {day}")
            key = day, row["variable"]
            if key in candidates:
                raise ValueError(f"duplicate candidate: {key}")
            if row["source_gate_pass"].lower() != "true":
                raise ValueError(f"unapproved candidate: {key}")
            candidates[key] = row
    return candidates


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def cli_module():
    path = SOURCE_PATHS["CLI_BUILDER"]
    spec = importlib.util.spec_from_file_location("fqa_041_cli_fit", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    if any(OUT.glob("fitting_weather.csv")) or any(OUT.glob("CNFQ.CLI")):
        raise RuntimeError("041 outputs already exist; refusing overwrite")
    for path in SOURCE_PATHS.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    dates = [START + dt.timedelta(days=i) for i in range((END - START).days + 1)]
    rain = load_xls(SOURCE_PATHS["RAIN"], {"RAIN": 8}, {0: "生态站代码", 8: "20-20合计(mm)"})
    temp = load_xls(SOURCE_PATHS["TMAX_TMIN"], {"TMAX": 5, "TMIN": 7}, {0: "生态站代码", 5: "日最大值(℃)", 7: "日最小值(℃)"})
    srad = load_xls(SOURCE_PATHS["SRAD"], {"SRAD": 4}, {0: "生态站代码", 4: "总辐射总量(MJ/m^2)"})
    for name, data in (("RAIN", rain), ("TMAX_TMIN", temp), ("SRAD", srad)):
        if set(data) != set(dates):
            raise ValueError(f"{name} date coverage mismatch: missing={len(set(dates)-set(data))}, extra={len(set(data)-set(dates))}")
    candidates = load_candidates(SOURCE_PATHS["NASA_CANDIDATES"])
    station_lines = SOURCE_PATHS["STATION_WTH_HEADER"].read_text(encoding="ascii").splitlines()
    tokens = next(line.split() for line in station_lines if line.startswith("  CNFQ"))
    latitude, longitude, elevation = map(float, tokens[1:4])

    weather_rows, provenance_rows = [], []
    counts = Counter()
    annual = defaultdict(lambda: {"rain_mm": 0.0, "wet_days": 0, "blank_to_zero_days": 0, "candidate_overlay_days": 0})
    used_candidates = set()
    for day in dates:
        raw_rain = rain[day]["RAIN"]
        values = {"RAIN": 0.0 if raw_rain is None else raw_rain, **temp[day], **srad[day]}
        sources = {"RAIN": "D222_BLANK_TO_ZERO" if raw_rain is None else "D222_RECORDED"}
        if raw_rain is None:
            counts["rain_blank_to_zero"] += 1
            annual[day.year]["blank_to_zero_days"] += 1
        else:
            counts["rain_recorded"] += 1
        for variable in ("TMAX", "TMIN", "SRAD"):
            value = values[variable]
            invalid = value is None or (variable == "SRAD" and (value < 0 or value > 60)) or (variable in ("TMAX", "TMIN") and (value < -60 or value > 50))
            key = day, variable
            if invalid:
                row = candidates.get(key)
                if row is None:
                    raise ValueError(f"missing approved candidate for {key}; raw={value}")
                if row["reason"] not in ("raw_missing", "physical_qc_invalid"):
                    raise ValueError(f"unexpected candidate reason for {key}: {row['reason']}")
                if value is not None and row["reason"] != "physical_qc_invalid":
                    raise ValueError(f"candidate reason/raw mismatch: {key}")
                values[variable] = float(row["candidate_value"])
                sources[variable] = "NASA_POWER_BIAS_CORRECTED"
                used_candidates.add(key)
                counts[f"{variable}_candidate"] += 1
                annual[day.year]["candidate_overlay_days"] += 1
            else:
                if key in candidates:
                    raise ValueError(f"candidate attempts to replace valid station {key}")
                sources[variable] = "STATION_RAW"
        if any(not math.isfinite(values[v]) for v in ("RAIN", "SRAD", "TMAX", "TMIN")):
            raise ValueError(f"nonfinite value on {day}")
        if values["RAIN"] < 0 or values["SRAD"] < 0 or values["TMAX"] < values["TMIN"]:
            raise ValueError(f"physical QC failed on {day}: {values}")
        if raw_rain is not None and values["RAIN"] != raw_rain:
            raise ValueError(f"D222 recorded rain changed on {day}")
        if raw_rain is None and values["RAIN"] != 0:
            raise ValueError(f"D222 blank not zero on {day}")
        weather_rows.append({"DATE": day.isoformat(), "SRAD": values["SRAD"], "TMAX": values["TMAX"], "TMIN": values["TMIN"], "RAIN": values["RAIN"]})
        provenance_rows.append({"DATE": day.isoformat(), "RAIN_original_cell": "BLANK" if raw_rain is None else "NUMERIC", "RAIN_original_value": "" if raw_rain is None else raw_rain,
                                "RAIN_source": sources["RAIN"], "SRAD_source": sources["SRAD"], "TMAX_source": sources["TMAX"], "TMIN_source": sources["TMIN"]})
        annual[day.year]["rain_mm"] += values["RAIN"]
        annual[day.year]["wet_days"] += values["RAIN"] > 0
    if used_candidates != set(candidates):
        raise ValueError(f"unused candidates: {sorted(set(candidates)-used_candidates)[:5]}")
    if len(weather_rows) != 3287:
        raise ValueError(f"expected 3287 days, got {len(weather_rows)}")
    fit = cli_module()
    records = [fit.WeatherDay(dt.date.fromisoformat(r["DATE"]), r["SRAD"], r["TMAX"], r["TMIN"], r["RAIN"]) for r in weather_rows]
    wgen = fit.fit_monthly_statistics(records)
    monthly = fit.monthly_weather_summary(records)
    crosscheck = fit.crosscheck_statistics(records, wgen)
    if not crosscheck["passed"]:
        raise ValueError(f"WGEN parameter crosscheck failed: {crosscheck}")
    cli = fit._cli_text(records, "CNFQ", latitude, longitude, elevation, wgen, monthly)
    old_start = "  2004    10 -99.0"
    new_start = "  2005     9 -99.0"
    if cli.count(old_start) != 1:
        raise ValueError("YC hardcoded CLI start row not found exactly once")
    cli = cli.replace(old_start, new_start, 1)
    schema = fit.check_cli_schema(cli.replace("CNFQ", "CNYC"))
    if not schema["passed"] or cli.count("*CLIMATE:CNFQ") != 1:
        raise ValueError(f"CLI schema failed: {schema}")

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "fitting_weather.csv", FIELDS, weather_rows)
    write_csv(OUT / "daily_provenance.csv", ["DATE", "RAIN_original_cell", "RAIN_original_value", "RAIN_source", "SRAD_source", "TMAX_source", "TMIN_source"], provenance_rows)
    write_csv(OUT / "annual_summary.csv", ["year", "rain_mm", "wet_days", "blank_to_zero_days", "candidate_overlay_days"],
              [dict(year=year, **annual[year]) for year in sorted(annual)])
    write_csv(OUT / "monthly_wgen_statistics.csv", list(wgen[0]), wgen)
    write_csv(OUT / "monthly_weather_summary.csv", list(monthly[0]), monthly)
    (OUT / "CNFQ.CLI").write_text(cli, encoding="ascii", newline="\n")
    manifest = {name: {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha(path)} for name, path in SOURCE_PATHS.items()}
    output = {"task": "041", "site": "FQA", "fit_start": START.isoformat(), "fit_end": END.isoformat(), "days": len(dates),
              "rain_rule": "D222 20-20 numeric unchanged; D222 blank -> 0 mm per user decision; converted zero is not observed zero",
              "counts": dict(counts), "station_coordinates_from_wth": {"latitude": latitude, "longitude": longitude, "elevation_m": elevation},
              "input_manifest": manifest, "output_sha256": {name: sha(OUT / name) for name in ("fitting_weather.csv", "daily_provenance.csv", "annual_summary.csv", "CNFQ.CLI")},
              "FQA_FITTING_WEATHER": "PASS", "FQA_STATIC_CLI": "PASS", "FQA_RUNTIME_SMOKE": "NOT_RUN", "FQA_WGEN_DISTRIBUTION_QC": "NOT_RUN", "FQA_PPO": "NOT_RUN",
              "monthly_parameter_crosscheck": crosscheck, "cli_schema": schema}
    (OUT / "final_gate.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"days": len(dates), "counts": dict(counts), "FQA_FITTING_WEATHER": "PASS", "FQA_STATIC_CLI": "PASS"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
