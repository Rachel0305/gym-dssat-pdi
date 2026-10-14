"""Read-only, project-contained audit of the 15 frozen HLA 2013 heavy-rain dates."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import date, timedelta
from pathlib import Path

import xlrd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE014 = ROOT / "results/hla_random_weather_8seed/014_isd50756_recovery"
EVENTS = BASE014 / "high_rain_event_qa_2013.csv"
SELECTED = BASE014 / "daily_source_selection_after_014.csv"
GSOD = BASE014 / "gsod_crosscheck.csv"
OLD = ROOT / "data/external/cern_d222/raw/excel.xls"
CURRENT = ROOT / "data/external/人工大气观测降雨蒸发能见度日值.xls"
GHCN = ROOT / "results/hl_fq_random_weather_8seed/source_audit/ghcn_daily_raw/CHM00050756.dly"
GHCN_SECONDARY = ROOT / "results/hl_fq_random_weather_8seed/source_audit/ghcn_daily_raw/CHM00050854.dly"
GPCC = ROOT / "results/hl_fq_global_precip_023/hla_gpcc_daily_2011_2013.csv"
CPC = ROOT / "results/hl_fq_global_precip_023/hla_cpc_daily_2011_2013.csv"
SOURCES = [EVENTS, SELECTED, GSOD, OLD, CURRENT, GHCN, GHCN_SECONDARY, GPCC, CPC]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def number(raw):
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite number: {raw!r}")
    return value


def original_workbook(path: Path, target_dates: set[str]):
    book = xlrd.open_workbook(str(path), on_demand=True)
    found = {}
    try:
        for sheet in book.sheets():
            if sheet.nrows < 2:
                continue
            header = [str(v).strip() for v in sheet.row_values(1)]
            if len(header) < 10 or "20-8" not in header[6] or "8-20" not in header[7] or "20-20" not in header[8] or "8-8" not in header[9]:
                raise ValueError(f"Unexpected D222 columns: {path.name} {sheet.name}")
            for index in range(2, sheet.nrows):
                row = sheet.row_values(index)
                if str(row[0]).strip() != "HLA" or int(float(row[1])) != 2013:
                    continue
                day = date(int(float(row[1])), int(float(row[2])), int(float(row[3]))).isoformat()
                if day not in target_dates:
                    continue
                if day in found:
                    raise ValueError(f"Duplicate HLA date in {path.name}: {day}")
                values = [number(row[i]) for i in (6, 7, 8, 9)]
                found[day] = {"sheet": sheet.name, "row": index + 1,
                              "night_20_08": values[0], "day_08_20": values[1],
                              "total_20_20": values[2], "day_08_08": values[3]}
    finally:
        book.release_resources()
    if set(found) != target_dates:
        raise ValueError(f"Missing HLA event dates in {path.name}: {sorted(target_dates - set(found))}")
    return found


def ghcn_prcp(path: Path, target_dates: set[str]):
    found = {}
    station = path.stem
    with path.open(encoding="ascii") as stream:
        for line_number, line in enumerate(stream, 1):
            if line[:11] != station or line[17:21] != "PRCP" or line[11:15] != "2013":
                continue
            year, month = int(line[11:15]), int(line[15:17])
            for dom in range(1, 32):
                try:
                    day = date(year, month, dom).isoformat()
                except ValueError:
                    continue
                if day not in target_dates:
                    continue
                segment = line[21 + (dom - 1) * 8:21 + dom * 8]
                raw = int(segment[:5])
                item = {"rain_mm": None if raw == -9999 else raw / 10,
                        "mflag": segment[5], "qflag": segment[6], "sflag": segment[7],
                        "line": line_number, "raw_tenths_mm": raw}
                if day in found:
                    raise ValueError(f"Duplicate GHCN PRCP date: {day}")
                found[day] = item
    return found


def grid_context(table, day):
    dates = [(date.fromisoformat(day) + timedelta(days=offset)).isoformat() for offset in (-1, 0, 1)]
    current = table.get(day)
    nearby = [float(table[d]["neighbor_max_mm"]) for d in dates if d in table and table[d]["neighbor_max_mm"]]
    return (float(current["nearest_grid_precip_mm"]) if current else None,
            float(current["neighbor_max_mm"]) if current else None,
            max(nearby) if nearby else None)


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    if (OUT / "final_gate.json").exists():
        raise FileExistsError("039 final gate exists; refuse to overwrite")
    gate = {"task": "039", "site": "HLA", "status": "STARTED", "event_count": 0,
            "input_gate": False, "wgen_executed": False, "dssat_executed": False,
            "ppo_executed": False, "weather_modified": False,
            "formal_source_selected": False, "ready_for_formal_training": False}
    try:
        event_rows = read_csv(EVENTS)
        dates = {row["date"] for row in event_rows}
        if len(event_rows) != 15 or len(dates) != 15 or any(float(row["selected_rain_mm"]) < 20 for row in event_rows):
            raise ValueError("014 frozen >=20 mm event set differs from 15 unique dates")
        selected = {row["date"]: row for row in read_csv(SELECTED) if row["date"] in dates}
        gsod = {row["date"]: row for row in read_csv(GSOD) if row["date"] in dates or row["date"] in
                {(date.fromisoformat(d) + timedelta(days=o)).isoformat() for d in dates for o in (-1, 1)}}
        gpcc = {row["date"]: row for row in read_csv(GPCC) if row["date"][:4] == "2013"}
        cpc = {row["date"]: row for row in read_csv(CPC) if row["date"][:4] == "2013"}
        old = original_workbook(OLD, dates)
        current = original_workbook(CURRENT, dates)
        ghcn = ghcn_prcp(GHCN, dates)
        ghcn_secondary = ghcn_prcp(GHCN_SECONDARY, dates)
        if len(selected) != 15 or len(gpcc) != 365 or len(cpc) != 365:
            raise ValueError("Frozen selected or gridded dates incomplete")
        manifest = {"site": "HLA", "task": "039", "event_rule": "014 frozen selected >=20 mm, all 15 dates",
                    "inputs": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in SOURCES],
                    "source_chain_note": "Old and current D222 are versions of the same source, not independent observations"}
        gate["input_gate"] = True
        matrix = []
        for event in event_rows:
            day = event["date"]
            o, c = old[day], current[day]
            chosen = selected[day]
            rain = float(event["selected_rain_mm"])
            if not math.isclose(rain, float(chosen["selected_rain_mm"]), abs_tol=1e-9):
                raise ValueError(f"014 event and selected mismatch: {day}")
            local_source = chosen["selected_source"] == "HLA_LOCAL_D222_CHAIN"
            if local_source and (o["total_20_20"] is None or not math.isclose(rain, o["total_20_20"], abs_tol=1e-9)):
                raise ValueError(f"014 selected and old raw D222 mismatch: {day}")
            if not local_source and not chosen["selected_source"].startswith("NOAA_GHCN_DAILY"):
                raise ValueError(f"Unexpected selected source: {day}")
            def closure(raw):
                values = (raw["night_20_08"], raw["day_08_20"], raw["total_20_20"])
                return None if any(x is None for x in values) else round(values[2] - values[0] - values[1], 6)
            oc, cc = closure(o), closure(c)
            stable = all(o[k] == c[k] for k in ("night_20_08", "day_08_20", "total_20_20", "day_08_08"))
            if not local_source:
                component_status = "NOT_APPLICABLE_D222_BLANK"
            elif oc is not None and cc is not None:
                component_status = "BOTH_SNAPSHOTS_CLOSED" if abs(oc) <= .051 and abs(cc) <= .051 else "COMPONENT_MISMATCH"
            else:
                component_status = "INCOMPLETE_COMPONENTS_NO_CLOSURE_CLAIM"
            source_status = ("RAW_SELECTED_MATCH_AND_VERSION_STABLE" if stable else "VERSION_CHANGED") if local_source else "GHCN_FALLBACK_EVENT"
            g = ghcn.get(day)
            if chosen["selected_source"] == "NOAA_GHCN_DAILY" and (not g or g["rain_mm"] is None or g["qflag"].strip() or not math.isclose(rain, g["rain_mm"], abs_tol=1e-9)):
                raise ValueError(f"Selected GHCN 50756 value differs from raw .dly: {day}")
            secondary = ghcn_secondary.get(day)
            if chosen["selected_source"] == "NOAA_GHCN_DAILY_SECONDARY" and (chosen["selected_station_id"] != "CHM00050854" or not secondary or secondary["rain_mm"] is None or secondary["qflag"].strip() or not math.isclose(rain, secondary["rain_mm"], abs_tol=1e-9)):
                raise ValueError(f"Selected secondary GHCN value differs from raw .dly: {day}")
            gs = gsod.get(day)
            gp = grid_context(gpcc, day)
            cp = grid_context(cpc, day)
            nearby_utc = [(date.fromisoformat(day) + timedelta(days=k)).isoformat() for k in (-1, 0, 1)]
            gsod_values = {d: number(gsod[d]["gsod_prcp_mm_if_valid_A_to_G"]) for d in nearby_utc if d in gsod and gsod[d]["gsod_prcp_mm_if_valid_A_to_G"]}
            matrix.append({
                "date": day, "selected_mm": rain, "selected_source": chosen["selected_source"],
                "old_sheet": o["sheet"], "old_row": o["row"], "old_20_08_mm": o["night_20_08"],
                "old_08_20_mm": o["day_08_20"], "old_20_20_mm": o["total_20_20"], "old_08_08_mm": o["day_08_08"],
                "old_component_residual_mm": oc, "current_sheet": c["sheet"], "current_row": c["row"],
                "current_20_08_mm": c["night_20_08"], "current_08_20_mm": c["day_08_20"],
                "current_20_20_mm": c["total_20_20"], "current_08_08_mm": c["day_08_08"],
                "current_component_residual_mm": cc, "source_transcription_status": source_status,
                "component_closure_status": component_status,
                "ghcn50756_raw_mm": g["rain_mm"] if g else None,
                "ghcn50756_raw_tenths_mm": g["raw_tenths_mm"] if g else None,
                "ghcn50756_qflag": g["qflag"] if g else None, "ghcn50756_sflag": g["sflag"] if g else None,
                "ghcn50756_raw_line": g["line"] if g else None,
                "secondary_50854_raw_mm": secondary["rain_mm"] if secondary else None,
                "secondary_50854_qflag": secondary["qflag"] if secondary else None,
                "secondary_50854_raw_line": secondary["line"] if secondary else None,
                "ghcn_minus_d222_mm": round(g["rain_mm"] - rain, 3) if g and g["rain_mm"] is not None and not g["qflag"].strip() else None,
                "gsod_same_utc_mm": gsod_values.get(day), "gsod_utc_minus1_mm": gsod_values.get(nearby_utc[0]),
                "gsod_utc_plus1_mm": gsod_values.get(nearby_utc[2]),
                "gpcc_same_nearest_mm": gp[0], "gpcc_same_3x3_max_mm": gp[1], "gpcc_plusminus1_3x3_max_mm": gp[2],
                "cpc_same_nearest_mm": cp[0], "cpc_same_3x3_max_mm": cp[1], "cpc_plusminus1_3x3_max_mm": cp[2],
                "date_anchor_status": "D222_EXPORT_DATE_ANCHOR_UNDOCUMENTED",
                "magnitude_adjudication_status": "NOT_INDEPENDENTLY_ADJUDICATED",
            })
        write_csv(OUT / "event_evidence_matrix.csv", matrix)
        write_json(OUT / "input_manifest.json", manifest)
        gate.update({"status": "SOURCE_TRANSCRIPTION_AUDITED_MAGNITUDE_UNRESOLVED", "event_count": len(matrix),
                     "source_version_stable_count": sum(x["source_transcription_status"] == "RAW_SELECTED_MATCH_AND_VERSION_STABLE" for x in matrix),
                     "component_closed_count": sum(x["component_closure_status"] == "BOTH_SNAPSHOTS_CLOSED" for x in matrix),
                     "component_incomplete_count": sum(x["component_closure_status"] == "INCOMPLETE_COMPONENTS_NO_CLOSURE_CLAIM" for x in matrix),
                     "component_mismatch_count": sum(x["component_closure_status"] == "COMPONENT_MISMATCH" for x in matrix),
                     "ghcn_raw_numeric_unflagged_count": sum(x["ghcn_minus_d222_mm"] is not None for x in matrix),
                     "selected_15_day_subtotal_mm": round(math.fsum(x["selected_mm"] for x in matrix), 3),
                     "magnitude_independently_adjudicated_count": 0,
                     "next_evidence_needed": "Original gauge log or station QC record, with date interval and units, for disputed 2013 heavy-rain values"})
    except Exception as exc:
        gate.update({"status": "FAILED_INPUT_OR_COMPUTATION_GATE", "error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        write_json(OUT / "final_gate.json", gate)


if __name__ == "__main__":
    main()
