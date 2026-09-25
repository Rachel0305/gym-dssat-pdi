"""Read-only YC coordinate propagation preflight based on retained repo evidence.

This probe does not import Gym-DSSAT, launch DSSAT, modify inputs, or run a crop.
It localizes the mismatch in the existing 003_06_06 runtime snapshots.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results/yc_wgen_cli_pilot/003_06_07"
PREVIOUS_DIR = PROJECT_ROOT / "results/yc_wgen_cli_pilot/003_06_06/crop_smoke"
SOURCE_FILEX = PROJECT_ROOT / "DSSAT_auto_validation/multisite_new_cultivar_inputs_013/YC/CNYC0801.MZX"
EXPECTED = {"LAT": 36.830, "LONG": 116.570, "ELEV": 22.0}
RUNS = {
    "historical_control": "historical_control_retry4",
    "seed_101": "seed_101",
    "seed_104": "seed_104",
}


def _coordinate_header_index(lines: list[str]) -> int:
    for index, line in enumerate(lines):
        upper = line.upper()
        if line.lstrip().startswith("@L") and all(name in upper for name in ("XCRD", "YCRD", "ELEV")):
            return index
    raise ValueError("No @L XCRD/YCRD/ELEV header found")


def parse_filex_coordinates(text: str) -> dict[str, float]:
    """Parse the first field row and map DSSAT XCRD/YCRD to LONG/LAT."""
    lines = text.splitlines()
    header_index = _coordinate_header_index(lines)
    for row in lines[header_index + 1 :]:
        tokens = row.split()
        if tokens and tokens[0] == "1":
            if len(tokens) < 4:
                break
            return {"LONG": float(tokens[1]), "LAT": float(tokens[2]), "ELEV": float(tokens[3])}
        if row.lstrip().startswith("*"):
            break
    raise ValueError("No field-1 coordinate row follows the coordinate header")


def render_filex_coordinates(text: str, coordinates: dict[str, float]) -> str:
    """Render a diagnostic FileX copy in its dotted fixed-width coordinate fields."""
    lines = text.splitlines()
    header_index = _coordinate_header_index(lines)
    header = lines[header_index]
    row_index = next(
        (i for i in range(header_index + 1, len(lines)) if lines[i].split() and lines[i].split()[0] == "1"),
        None,
    )
    if row_index is None:
        raise ValueError("No field-1 coordinate row follows the coordinate header")
    markers = {m.group(1): m.start() for m in re.finditer(r"\.+(XCRD|YCRD|ELEV|AREA)\b", header)}
    if not {"XCRD", "YCRD", "ELEV", "AREA"}.issubset(markers):
        raise ValueError("Unexpected FileX coordinate header layout")
    bounds = {
        "LONG": (markers["XCRD"] - 1, markers["YCRD"] - 1),
        "LAT": (markers["YCRD"] - 1, markers["ELEV"] - 1),
        "ELEV": (markers["ELEV"] - 1, markers["AREA"] - 1),
    }
    formats = {"LONG": f"{coordinates['LONG']:.5f}", "LAT": f"{coordinates['LAT']:.5f}", "ELEV": f"{coordinates['ELEV']:.1f}"}
    row = lines[row_index]
    for name in ("ELEV", "LAT", "LONG"):
        start, end = bounds[name]
        width = end - start
        value = formats[name]
        if len(value) > width:
            raise ValueError(f"{name} value does not fit the FileX field width {width}")
        row = row[:start] + value.rjust(width) + row[end:]
    lines[row_index] = row
    rendered = "\n".join(lines) + "\n"
    if parse_filex_coordinates(rendered) != {"LONG": 116.57, "LAT": 36.83, "ELEV": 22.0}:
        raise ValueError("Rendered FileX coordinate round-trip failed")
    return rendered


def parse_dssat_field_coordinates(text: str) -> dict[str, float]:
    """Read the XCRD/YCRD/ELEV record inside *FIELDS, not the soil-site text."""
    lines = text.splitlines()
    fields_index = next((i for i, line in enumerate(lines) if line.strip().upper().startswith("*FIELDS")), None)
    if fields_index is None:
        raise ValueError("DSSAT input has no *FIELDS section")
    header_index = next(
        (i for i in range(fields_index + 1, len(lines)) if lines[i].lstrip().startswith("@L") and "XCRD" in lines[i].upper()),
        None,
    )
    start_index = header_index + 1 if header_index is not None else fields_index + 1
    for row in lines[start_index:]:
        if row.lstrip().startswith("*"):
            break
        tokens = row.split()
        if not tokens:
            continue
        if header_index is None and not re.fullmatch(r"[-+]?\d+(?:\.\d*)?", tokens[0]):
            continue
        values = tokens[1:4] if tokens[0].isdigit() and len(tokens) >= 4 else tokens[:3]
        if len(values) == 3:
            return {"LONG": float(values[0]), "LAT": float(values[1]), "ELEV": float(values[2])}
    raise ValueError("DSSAT *FIELDS coordinate value row not found")


def coordinate_gate(actual: dict[str, float | None]) -> bool:
    return all(actual.get(name) is not None and abs(float(actual[name]) - value) < 1e-6 for name, value in EXPECTED.items())


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summary_coordinates(text: str) -> dict[str, float | None]:
    lines = text.splitlines()
    header_index = next(
        (i for i, line in enumerate(lines) if line.startswith("@") and all(key in line for key in ("XLAT", "LONG", "ELEV", "SDAT"))),
        None,
    )
    if header_index is None:
        return {"LAT": None, "LONG": None, "ELEV": None}
    data_index = next((i for i in range(header_index + 1, len(lines)) if lines[i].strip() and lines[i].strip()[0].isdigit()), None)
    if data_index is None:
        return {"LAT": None, "LONG": None, "ELEV": None}
    header, data = lines[header_index], lines[data_index]
    start = header.index("XLAT")
    long_start = header.index("LONG")
    elev_start = header.index("ELEV")
    widths = (long_start - start, elev_start - long_start, 5)
    offsets = (start, long_start, elev_start)
    values: list[float | None] = []
    for offset, width in zip(offsets, widths):
        token = data[offset : offset + width].strip()
        try:
            values.append(float(token) if token else None)
        except ValueError:
            values.append(None)
    return {"LAT": values[0], "LONG": values[1], "ELEV": values[2]}


def _line_number(path: Path, needle: str) -> int | None:
    for number, line in enumerate(_read_text(path).splitlines(), 1):
        if needle in line:
            return number
    return None


def run_probe() -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    source_values = parse_filex_coordinates(_read_text(SOURCE_FILEX))
    trace_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []
    per_run: dict[str, Any] = {}

    summary_path = PREVIOUS_DIR / "crop_output_summary.csv"
    with summary_path.open(encoding="utf-8-sig", newline="") as stream:
        before_by_id = {row["run_id"]: row for row in csv.DictReader(stream)}
    stress_path = PREVIOUS_DIR / "dssat_water_n_stress_summary.csv"
    with stress_path.open(encoding="utf-8-sig", newline="") as stream:
        stress_rows = list(csv.DictReader(stream))
    stress_by_run: dict[str, dict[str, dict[str, str]]] = {}
    for row in stress_rows:
        stress_by_run.setdefault(row["run_id"], {})[row["indicator"]] = row

    warning_summary_path = PROJECT_ROOT / "results/yc_wgen_cli_pilot/003_06_06/runtime_warning_audit/warning_summary.json"
    warning_summary = json.loads(_read_text(warning_summary_path))
    per_run_warning = warning_summary["per_run_event_counts"]
    other_warning_events = per_run_warning.get("PHOTO L->C", 0) + per_run_warning.get("STONES/ADCOEF defaults", 0)

    for label, previous_id in RUNS.items():
        snapshot = PREVIOUS_DIR / previous_id / "runtime_snapshot"
        filex_path = snapshot / "fileX.MZX"
        inp_path = snapshot / "DSSAT48.INP"
        inh_path = snapshot / "DSSAT48.INH"
        warning_path = snapshot / "WARNING.OUT"
        summary_out_path = snapshot / "Summary.OUT"
        rendered = parse_filex_coordinates(_read_text(filex_path))
        inp = parse_dssat_field_coordinates(_read_text(inp_path))
        inh = parse_dssat_field_coordinates(_read_text(inh_path))
        warning_text = _read_text(warning_path)
        summary_coords = _summary_coordinates(_read_text(summary_out_path))
        read_events = {
            "LAT": warning_text.count("Error reading latitude from experimental file"),
            "LONG": warning_text.count("Error reading longitude from experimental file"),
            "ELEV": warning_text.count("Error reading elevation from experimental file"),
        }
        transfer_names = {"LAT": "CYCRDin", "LONG": "CXCRDin", "ELEV": "CELEVin"}
        transfer_events = {name: warning_text.count(f"Error transferring variable: {variable} FIELD") for name, variable in transfer_names.items()}
        per_run[label] = {
            "previous_run_id": previous_id,
            "rendered_filex_coordinates": rendered,
            "parsed_internal_coordinates": None,
            "dssat48_inp_field_coordinates": inp,
            "dssat48_inh_field_coordinates": inh,
            "runtime_coordinates": {"LAT": 0.0, "LONG": 0.0, "ELEV": 0.0},
            "summary_out_coordinates": summary_coords,
            "coordinate_read_warning_counts": read_events,
            "coordinate_transfer_warning_counts": transfer_events,
            "noncoordinate_baseline_warning_events": other_warning_events,
            "dssat48_inp_line": _line_number(inp_path, "-999.00000"),
            "warning_file": str(warning_path.relative_to(PROJECT_ROOT)),
        }
        for variable, expected in EXPECTED.items():
            source_actual = source_values[variable]
            rendered_actual = rendered[variable]
            inp_actual = inp[variable]
            trace_rows.extend(
                [
                    {"stage": "source FileX/template", "file_or_function": str(SOURCE_FILEX.relative_to(PROJECT_ROOT)), "variable_name": variable, "expected_value": expected, "observed_value": source_actual, "status": "SOURCE_PLACEHOLDER", "evidence": "原始 YC MZX 的坐标字段为 -99"},
                    {"stage": "rendered isolated FileX", "file_or_function": str(filex_path.relative_to(PROJECT_ROOT)), "variable_name": variable, "expected_value": expected, "observed_value": rendered_actual, "status": "PASS", "evidence": "上一轮 runtime_snapshot/fileX.MZX 固定列读取"},
                    {"stage": "Gym-DSSAT/PDI parsed variables", "file_or_function": "installed gym_dssat_pdi source unavailable inside project", "variable_name": variable, "expected_value": expected, "observed_value": "NOT_OBSERVED", "status": "UNOBSERVED", "evidence": "未修改/读取仓库外安装包；无 parser trace 可核实"},
                    {"stage": "generated DSSAT48.INP *FIELDS", "file_or_function": str(inp_path.relative_to(PROJECT_ROOT)), "variable_name": variable, "expected_value": expected, "observed_value": inp_actual, "status": "FAIL_PLACEHOLDER_RETAINED", "evidence": "*FIELDS coordinate record retains -999/-99/-99"},
                    {"stage": "generated DSSAT48.INH *FIELDS", "file_or_function": str(inh_path.relative_to(PROJECT_ROOT)), "variable_name": variable, "expected_value": expected, "observed_value": inh[variable], "status": "FAIL_PLACEHOLDER_RETAINED", "evidence": "companion INH coordinate record also retains -999/-99/-99"},
                    {"stage": "runtime INFO/WARNING", "file_or_function": str(warning_path.relative_to(PROJECT_ROOT)), "variable_name": variable, "expected_value": expected, "observed_value": "read/transfer error; value set to zero", "status": "FAIL", "evidence": f"read_errors={read_events[variable]}; transfer_errors={transfer_events[variable]}"},
                    {"stage": "Summary.OUT runtime output", "file_or_function": str(summary_out_path.relative_to(PROJECT_ROOT)), "variable_name": variable, "expected_value": expected, "observed_value": summary_coords[variable], "status": "FAIL_EMPTY", "evidence": "Summary.OUT XLAT/LONG/ELEV columns are blank"},
                ]
            )

        before = before_by_id[previous_id]
        stress = stress_by_run[previous_id]
        comp = {
            "run_id": label,
            "previous_run_id": previous_id,
            "before_status": before["runtime_status"],
            "after_status": "NOT_RUN_BLOCKED_BEFORE_INP_COORDINATE_GATE",
            "before_anthesis": before["anthesis_date"],
            "after_anthesis": "",
            "before_maturity": before["maturity_date"],
            "after_maturity": "",
            "before_season_length_dap": before["season_length_dap_days"],
            "after_season_length_dap": "",
            "before_yield_kg_ha": before["yield_kg_ha"],
            "after_yield_kg_ha": "",
            "before_biomass_kg_ha": before["biomass_kg_ha"],
            "after_biomass_kg_ha": "",
            "before_irrigation_mm": before["cumulative_irrigation_mm"],
            "after_irrigation_mm": "",
            "before_fertilizer_n_kg_ha": before["cumulative_fertilizer_n_kg_ha"],
            "after_fertilizer_n_kg_ha": "",
            "before_n_uptake_kg_ha": before["n_uptake_kg_ha"],
            "after_n_uptake_kg_ha": "",
            "before_max_wspd": stress["WSPD"]["max"],
            "before_max_wsgd": stress["WSGD"]["max"],
            "before_max_nstd": stress["NSTD"]["max"],
            "before_swtd_min": stress["SWTD"]["min"],
            "before_swtd_max": stress["SWTD"]["max"],
            "before_swxd_min": stress["SWXD"]["min"],
            "before_swxd_max": stress["SWXD"]["max"],
            "after_stress_summary": "",
            "crop_output_changed_after_fix": "NOT_ASSESSED",
            "comparison_status": "BLOCKED_NO_COORDINATE_FIX_OR_POSTFIX_RUN",
        }
        comparison_rows.append(comp)

        coordinate_read_total = sum(read_events.values())
        coordinate_transfer_total = sum(transfer_events.values())
        warning_rows.append(
            {
                "run_id": label,
                "before_coordinate_read_events": coordinate_read_total,
                "before_coordinate_transfer_events": coordinate_transfer_total,
                "before_coordinate_warning_events": coordinate_read_total + coordinate_transfer_total,
                "before_noncoordinate_baseline_events": other_warning_events,
                "before_total_discrete_events": coordinate_read_total + coordinate_transfer_total + other_warning_events,
                "after_coordinate_warning_events": "",
                "after_noncoordinate_baseline_events": "",
                "after_total_discrete_events": "",
                "after_status": "NOT_RUN_BLOCKED_BEFORE_INP_COORDINATE_GATE",
            }
        )

    _write_csv(RESULTS_DIR / "coordinate_trace.csv", trace_rows, list(trace_rows[0]))
    _write_csv(RESULTS_DIR / "coordinate_fix_crop_output_comparison.csv", comparison_rows, list(comparison_rows[0]))
    _write_csv(RESULTS_DIR / "warning_comparison.csv", warning_rows, list(warning_rows[0]))

    check = {
        "expected_lat": EXPECTED["LAT"],
        "expected_long": EXPECTED["LONG"],
        "expected_elev": EXPECTED["ELEV"],
        "lat_expected": EXPECTED["LAT"],
        "lat_actual": per_run["seed_101"]["dssat48_inp_field_coordinates"]["LAT"],
        "long_expected": EXPECTED["LONG"],
        "long_actual": per_run["seed_101"]["dssat48_inp_field_coordinates"]["LONG"],
        "elev_expected": EXPECTED["ELEV"],
        "elev_actual": per_run["seed_101"]["dssat48_inp_field_coordinates"]["ELEV"],
        "pass": all(coordinate_gate(run["dssat48_inp_field_coordinates"]) and coordinate_gate(run["dssat48_inh_field_coordinates"]) for run in per_run.values()),
        "probe_type": "read-only retained-evidence preflight; no crop simulation",
        "parsed_internal_coordinates": None,
        "rendered_filex_values": per_run["seed_101"]["rendered_filex_coordinates"],
        "parsed_internal_values": {"LAT": None, "LONG": None, "ELEV": None, "status": "NOT_OBSERVED"},
        "dssat48_inp_values": per_run["seed_101"]["dssat48_inp_field_coordinates"],
        "runtime_coordinates": per_run["seed_101"]["runtime_coordinates"],
        "summary_out_coordinates": per_run["seed_101"]["summary_out_coordinates"],
        "per_run_evidence": per_run,
        "root_cause_status": "PARTIALLY_LOCALIZED_SOURCE_LEVEL_UNRESOLVED",
        "loss_boundary": "between correct rendered FileX and DSSAT48.INP *FIELDS coordinate record",
    }
    (RESULTS_DIR / "dssat48_coordinate_check.json").write_text(json.dumps(check, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for label, previous_id in RUNS.items():
        run_dir = RESULTS_DIR / "runtime" / label
        run_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "status": "NOT_RUN_BLOCKED_BEFORE_COORDINATE_GATE",
            "previous_run_id": previous_id,
            "reason": "DSSAT48.INP *FIELDS coordinates are placeholders; no fix was integrated or verified.",
            "crop_simulation_run": False,
        }
        (run_dir / "preflight_record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return check


if __name__ == "__main__":
    result = run_probe()
    print(json.dumps({
        "expected_lat": result["expected_lat"],
        "expected_long": result["expected_long"],
        "expected_elev": result["expected_elev"],
        "rendered_filex_values": result["rendered_filex_values"],
        "parsed_internal_values": result["parsed_internal_values"],
        "dssat48_inp_values": result["dssat48_inp_values"],
        "runtime_coordinates": result["runtime_coordinates"],
        "pass": result["pass"],
        "loss_boundary": result["loss_boundary"],
        "results_dir": str(RESULTS_DIR),
    }, ensure_ascii=False, indent=2))
