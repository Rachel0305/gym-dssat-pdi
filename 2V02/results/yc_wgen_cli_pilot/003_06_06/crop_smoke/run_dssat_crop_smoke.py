from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import sys
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))
TASK_ROOT = PROJECT_ROOT / "results" / "yc_wgen_cli_pilot" / "003_06_06"
OUT = TASK_ROOT / "crop_smoke"
INPUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013" / "YC"
SOURCE_FILEX = INPUT_DIR / "CNYC0801.MZX"
CULTIVAR = INPUT_DIR / "MZCER048.CUL"
SOIL = INPUT_DIR / "SOIL.SOL"
CLI = PROJECT_ROOT / "results" / "yc_wgen_cli_pilot" / "003_06_05_02" / "final" / "CNYC.CLI"
TRAIN_WEATHER = PROJECT_ROOT / "results" / "yc_weather_gapfill_finalize" / "yc_wgen_fitting_weather_2004_2013.csv"
HIST_WTH = PROJECT_ROOT / "my_data" / "CNYC0801.WTH"
EXPECTED_CLI_SHA256 = "65CF134600A5881706A5D435E1A09B276ED92A21FA5ABE2E18AAF63AF1E3A929"
EXPECTED_TRAIN_SHA256 = "4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34"
KNOWN_STATION = {"latitude": 36.830, "longitude": 116.570, "elevation_m": 22.0}
MAX_STEPS = 400
from scripts.run_yc_wgen_seed_pilot import _runtime_class, pdi_seed_configured, prepare_wgen_filex, runtime_filex_mode

OUTPUT_NAMES = {
    "CNYC.CLI", "CNYC0801.WTH", "FILEX.MZX", "DSSAT-PDI.YML", "DSSAT48.INP", "DSSAT48.INH",
    "SUMMARY.OUT", "OVERVIEW.OUT", "PLANTGRO.OUT", "PLANTN.OUT", "SOILWAT.OUT", "SOILNI.OUT",
    "MGMTOPS.OUT", "MGMTEVENT.OUT", "INFO.OUT", "WARNING.OUT", "ERROR.OUT", "RUNLIST.OUT",
    "SOIL.SOL", "MZCER048.CUL",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _json_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def _atomic_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_value(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _coords_in_isolated_filex(text: str) -> str:
    lines = text.splitlines()
    header_index = next((index for index, line in enumerate(lines) if line.lstrip().startswith("@L") and "XCRD" in line.upper()), None)
    if header_index is None:
        raise ValueError("FileX has no XCRD/YCRD/ELEV location header")
    row_index = next((index for index in range(header_index + 1, len(lines)) if lines[index].split() and lines[index].split()[0] == "1"), None)
    if row_index is None:
        raise ValueError("FileX has no field-1 location row")
    line = lines[row_index]
    header = lines[header_index]
    column_matches = {match.group(1): match.start() for match in re.finditer(r"\.+(XCRD|YCRD|ELEV|AREA)\b", header)}
    if not {"XCRD", "YCRD", "ELEV", "AREA"}.issubset(column_matches):
        raise ValueError(f"Unexpected FileX location header: {header!r}")
    bounds = {
        name: (column_matches[name] - 1, column_matches[next_name] - 1)
        for name, next_name in (("XCRD", "YCRD"), ("YCRD", "ELEV"), ("ELEV", "AREA"))
    }
    expected_widths = {"XCRD": 16, "YCRD": 16, "ELEV": 10}
    replacements = {"XCRD": f"{KNOWN_STATION['longitude']:.5f}", "YCRD": f"{KNOWN_STATION['latitude']:.5f}", "ELEV": f"{KNOWN_STATION['elevation_m']:.1f}"}
    for name in ("ELEV", "YCRD", "XCRD"):
        start, end = bounds[name]
        value = replacements[name]
        width = end - start
        if width != expected_widths[name]:
            raise ValueError(f"Unexpected FileX {name} field width {width}; expected {expected_widths[name]}")
        if len(value) > width:
            raise ValueError(f"Coordinate {value} does not fit FileX {name} field width {width}")
        line = line[:start] + value.rjust(width) + line[end:]
    lines[row_index] = line
    result = "\n".join(lines) + "\n"
    actual = _filex_coordinates(result)
    if actual != {"XCRD": 116.57, "YCRD": 36.83, "ELEV": 22.0}:
        raise ValueError(f"Isolated FileX coordinate verification failed: {actual}")
    return result


def _filex_coordinates(text: str) -> dict[str, float]:
    lines = text.splitlines()
    header_index = next((index for index, line in enumerate(lines) if line.lstrip().startswith("@L") and "XCRD" in line.upper()), None)
    if header_index is None:
        return {}
    row = next((lines[index] for index in range(header_index + 1, len(lines)) if lines[index].split() and lines[index].split()[0] == "1"), "")
    tokens = row.split()
    if len(tokens) < 4:
        return {}
    return {"XCRD": float(tokens[1]), "YCRD": float(tokens[2]), "ELEV": float(tokens[3])}


def _normalize_filex(text: str) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if "@N METHODS" in line.upper():
            for row_index in range(index + 1, len(lines)):
                tokens = lines[row_index].split()
                if tokens and tokens[0].isdigit():
                    if len(tokens) > 2:
                        tokens[2] = "WTHER"
                        lines[row_index] = " ".join(tokens)
                    break
    return "\n".join(lines) + "\n"


def _last_finite(values: list[Any]) -> float | None:
    for value in reversed(values):
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            return parsed
    return None


def _state_date(value: Any) -> str | None:
    try:
        yrdoy = int(float(value))
        year, doy = divmod(yrdoy, 1000)
        return (date(year, 1, 1) + timedelta(days=doy - 1)).isoformat()
    except (TypeError, ValueError, OverflowError):
        return None


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summary_from_output(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_line = next((line for line in lines if line.lstrip().startswith("@") and "RUNNO" in line), None)
    if header_line is None:
        return {}
    headers = header_line.split()
    soil_index = next((index for index, name in enumerate(headers) if name.startswith("SOIL_ID")), None)
    if soil_index is None:
        return {}
    data_line = next((line for line in lines if "YC99001200" in line and line.split() and line.split()[0].isdigit()), None)
    if data_line is None:
        return {}
    values = data_line.split()
    try:
        value_soil_index = values.index("YC99001200")
    except ValueError:
        return {}
    # Summary.OUT leaves XLAT/LONG/ELEV blank when DSSAT cannot parse coordinates.
    columns = [name.rstrip(".") for name in headers[soil_index + 1:] if name.rstrip(".") not in {"XLAT", "LONG", "ELEV"}]
    tail = values[value_soil_index + 1:]
    if len(columns) != len(tail):
        raise ValueError(f"Summary.OUT field/value count mismatch after SOIL_ID: {len(columns)} != {len(tail)}")
    result: dict[str, Any] = {
        "RUN": float(values[0]),
        "TRNO": float(values[1]),
        "CROP": values[5] if len(values) > 5 else "",
        "MODEL": values[6] if len(values) > 6 else "",
        "EXNAME": values[7] if len(values) > 7 else "",
        "TNAM": values[8] if len(values) > 8 else "",
        "FNAM": values[9] if len(values) > 9 else "",
    }
    for name, value in zip(columns, tail):
        try:
            result[name] = float(value)
        except ValueError:
            result[name] = value
    return result


def _output_snapshot(temp_folder: Path, run_dir: Path) -> list[str]:
    snapshot = run_dir / "runtime_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in sorted(temp_folder.iterdir()):
        if path.is_file() and (path.name.upper() in OUTPUT_NAMES or path.suffix.upper() == ".OUT" or path.suffix.upper() == ".WTH"):
            target = snapshot / path.name
            shutil.copy2(path, target)
            copied.append(target.name)
    return copied


def _capture_states(states: list[dict], path: Path) -> list[dict]:
    scalar_names = ("dap", "yrdoy", "istage", "vstage", "grnwt", "topwt", "xlai", "swfac", "nstres", "wtnup", "trnu", "totir", "cumsumfert", "rain", "srad", "tmax", "tmin", "wtdep", "rtdep", "runoff")
    rows = []
    for state in states:
        normalized = {str(key).casefold(): _json_value(value) for key, value in state.items()}
        row = {name: normalized.get(name) for name in scalar_names}
        row["date"] = _state_date(normalized.get("yrdoy"))
        row["soil_water_layers"] = json.dumps(normalized.get("sw"), ensure_ascii=True) if normalized.get("sw") is not None else ""
        row["soil_dul_layers"] = json.dumps(normalized.get("dul"), ensure_ascii=True) if normalized.get("dul") is not None else ""
        rows.append(row)
    _write_csv(path, rows)
    return rows


def run_one(run_id: str, mode: str, weather_seed: int | None, template_path: Path) -> dict:
    run_dir = OUT / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing crop-smoke result: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime_log = run_dir / "runtime_log.txt"
    runtime_folder: Path | None = None
    env = None
    original_methods = None
    class_type = None
    states: list[dict] = []
    status = "FAILED"
    errors = None
    started = time.time()

    try:
        import gym
        import gym_dssat_pdi.envs.dssat_pdi as dssat_module
        import zmq

        class_type = _runtime_class(dssat_module)
        original_get_state = class_type._get_state
        original_get_sockets = class_type._get_sockets_
        instance_context: dict[str, Any] = {}

        def capture_state(instance: Any) -> Any:
            result = original_get_state(instance)
            current = getattr(instance, "_state", None)
            if isinstance(current, dict):
                instance._pilot_crop_states.append(copy.deepcopy(current))
            return result

        def capture_sockets(instance: Any) -> Any:
            instance._pilot_crop_states = []
            if weather_seed is not None:
                instance._rseed1 = int(weather_seed)
            instance_context["instance"] = instance
            instance_context["temp_folder"] = Path(instance._tmp_folder)
            result = original_get_sockets(instance)
            if getattr(instance, "_server", None) is not None:
                instance._server.setsockopt(zmq.RCVTIMEO, 30000)
            return result

        class_type._get_state = capture_state
        class_type._get_sockets_ = capture_sockets
        original_methods = (original_get_state, original_get_sockets)
        env = gym.make(
            "gym_dssat_pdi:GymDssatPdi-v0",
            log_saving_path=str(runtime_log),
            mode="all",
            auxiliary_file_paths=[str(CULTIVAR), str(CLI), str(SOIL), str(HIST_WTH)],
            random_weather=(mode == "W"),
            seed=None,
            fileX_template_path=str(template_path),
            experiment_number=1,
            evaluation=True,
            cultivar="maize",
            run_dssat_location="/opt/dssat_pdi/run_dssat",
        ).unwrapped

        runtime_folder = Path(env._tmp_folder)
        runtime_filex = runtime_folder / "fileX.MZX"
        runtime_cli = runtime_folder / "CNYC.CLI"
        runtime_wth = runtime_folder / "CNYC0801.WTH"
        filex_text = runtime_filex.read_text(encoding="utf-8", errors="replace")
        filex_mode = runtime_filex_mode(filex_text)
        if filex_mode != mode:
            raise RuntimeError(f"Runtime WTHER={filex_mode!r}; expected {mode!r}")
        if _filex_coordinates(filex_text) != {"XCRD": 116.57, "YCRD": 36.83, "ELEV": 22.0}:
            raise RuntimeError(f"Runtime FileX location fields were not preserved: {_filex_coordinates(filex_text)}")
        if not runtime_cli.is_file() or sha256_file(runtime_cli) != EXPECTED_CLI_SHA256:
            raise RuntimeError("Runtime CNYC.CLI missing or hash mismatch")
        if mode == "M" and (not runtime_wth.is_file() or sha256_file(runtime_wth) != sha256_file(HIST_WTH)):
            raise RuntimeError("Historical control did not receive byte-identical CNYC0801.WTH")
        pdi_path = runtime_folder / "dssat-pdi.yml"
        pdi_text = pdi_path.read_text(encoding="utf-8", errors="replace") if pdi_path.exists() else ""
        if weather_seed is not None and not pdi_seed_configured(pdi_text, weather_seed):
            raise RuntimeError(f"PDI runtime did not receive weather seed {weather_seed}")

        action = {key: 0.0 for key in env.action_variables}
        steps = 0
        max_rss_mb = 0.0
        while not bool(getattr(env, "done", False)) and steps < MAX_STEPS:
            env.step(action)
            steps += 1
        if not bool(getattr(env, "done", False)):
            raise TimeoutError(f"Crop smoke exceeded {MAX_STEPS} daily steps")
        states = list(getattr(env, "_pilot_crop_states", []))
        copied = _output_snapshot(runtime_folder, run_dir)
        filex_text = runtime_filex.read_text(encoding="utf-8", errors="replace")
        status = "PASS"
    except Exception as exc:
        errors = f"{type(exc).__name__}: {exc}"
        instance = locals().get("instance_context", {}).get("instance")
        temp = locals().get("instance_context", {}).get("temp_folder")
        if runtime_folder is None and isinstance(temp, Path):
            runtime_folder = temp
        if runtime_folder and runtime_folder.exists():
            try:
                copied = _output_snapshot(runtime_folder, run_dir)
            except Exception as snapshot_exc:
                errors += f"; snapshot={type(snapshot_exc).__name__}: {snapshot_exc}"
        states = list(getattr(instance, "_pilot_crop_states", [])) if instance is not None else []
        copied = locals().get("copied", [])
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as close_exc:
                errors = (errors + "; " if errors else "") + f"close={type(close_exc).__name__}: {close_exc}"
        if class_type is not None and original_methods:
            class_type._get_state, class_type._get_sockets_ = original_methods

    state_rows = _capture_states(states, run_dir / "daily_state.csv")
    log_text = runtime_log.read_text(encoding="utf-8", errors="replace") if runtime_log.exists() else ""
    info_path = run_dir / "runtime_snapshot" / "INFO.OUT"
    info_text = info_path.read_text(encoding="utf-8", errors="replace") if info_path.exists() else ""
    summary_path = run_dir / "runtime_snapshot" / "Summary.OUT"
    summary = _summary_from_output(summary_path)
    if not summary and status == "PASS":
        status = "INCOMPLETE_SUMMARY_OUTPUT"
        errors = "DSSAT ended without a parseable seasonal summary row"

    planting_date = _state_date(summary.get("PDAT"))
    anthesis_date = _state_date(summary.get("ADAT"))
    maturity_date = _state_date(summary.get("MDAT"))
    emergence_date = _state_date(summary.get("EDAT"))
    harvest_date = _state_date(summary.get("HDAT"))
    anthesis_dap = (date.fromisoformat(anthesis_date) - date.fromisoformat(planting_date)).days if anthesis_date and planting_date else None
    maturity_dap = (date.fromisoformat(maturity_date) - date.fromisoformat(planting_date)).days if maturity_date and planting_date else None
    final_state = state_rows[-1] if state_rows else {}
    lai_values = [float(row["xlai"]) for row in state_rows if isinstance(row.get("xlai"), (int, float)) and math.isfinite(float(row["xlai"]))]
    swfac_values = [float(row["swfac"]) for row in state_rows if isinstance(row.get("swfac"), (int, float)) and math.isfinite(float(row["swfac"]))]
    nstres_values = [float(row["nstres"]) for row in state_rows if isinstance(row.get("nstres"), (int, float)) and math.isfinite(float(row["nstres"]))]
    sw_values = []
    for row in state_rows:
        value = row.get("soil_water_layers")
        if value:
            try:
                sw_values.extend(float(item) for item in json.loads(value))
            except (TypeError, ValueError, json.JSONDecodeError):
                pass

    yield_kg_ha = summary.get("HWAM")
    biomass_kg_ha = summary.get("CWAM")
    irrigation_mm = summary.get("IRCM") if summary else _last_finite([row.get("totir") for row in state_rows])
    fertilizer_final = _last_finite([row.get("cumsumfert") for row in state_rows])
    n_uptake_kg_ha = summary.get("NUCM") if summary else _last_finite([row.get("wtnup") for row in state_rows])
    water_check = {
        "irrigation_mm": irrigation_mm,
        "scheduled_fertilizer_n_kg_ha": 303.0,
        "runtime_cumulative_fertilizer_state": fertilizer_final,
        "n_uptake_kg_ha": n_uptake_kg_ha,
        "min_swfac": min(swfac_values) if swfac_values else None,
        "max_swfac": max(swfac_values) if swfac_values else None,
        "min_nstres": min(nstres_values) if nstres_values else None,
        "max_nstres": max(nstres_values) if nstres_values else None,
        "min_soil_water_fraction": min(sw_values) if sw_values else None,
        "max_soil_water_fraction": max(sw_values) if sw_values else None,
    }
    phenology_ok = bool(planting_date and emergence_date and anthesis_date and maturity_date and harvest_date and anthesis_dap is not None and maturity_dap is not None and anthesis_dap > 0 and maturity_dap > anthesis_dap)
    yield_ok = bool(yield_kg_ha is not None and biomass_kg_ha is not None and math.isfinite(yield_kg_ha) and math.isfinite(biomass_kg_ha) and yield_kg_ha >= 0 and biomass_kg_ha >= yield_kg_ha and biomass_kg_ha < 100000)
    water_values = [value for key, value in water_check.items() if key not in {"scheduled_fertilizer_n_kg_ha", "runtime_cumulative_fertilizer_state", "n_uptake_kg_ha"} and isinstance(value, (int, float))]
    water_n_finite = all(math.isfinite(float(value)) for value in water_values)
    stress_values = [value for key in ("min_swfac", "max_swfac", "min_nstres", "max_nstres") if isinstance((value := water_check[key]), (int, float))]
    stress_ranges_ok = all(0 <= value <= 1 for value in stress_values)
    sw_range_ok = not sw_values or all(0 <= value <= 1 for value in sw_values)
    water_n_ok = irrigation_mm is not None and irrigation_mm >= 0 and n_uptake_kg_ha is not None and n_uptake_kg_ha >= 0 and water_n_finite and stress_ranges_ok and sw_range_ok

    result = {
        "run_id": run_id,
        "mode": "random_weather_WGEN" if mode == "W" else "historical_weather_measured",
        "random_weather": mode == "W",
        "weather_seed": weather_seed,
        "gym_seed": None,
        "ppo_training": False,
        "runtime_status": status,
        "runtime_steps": len(state_rows),
        "runtime_error": errors,
        "runtime_directory": str(runtime_folder) if runtime_folder else None,
        "runtime_log": str(runtime_log.relative_to(PROJECT_ROOT)),
        "snapshot_files": sorted(locals().get("copied", [])),
        "runtime_wther": mode,
        "runtime_cli_sha256": sha256_file(CLI) if CLI.exists() else None,
        "historical_wth_sha256": sha256_file(HIST_WTH) if mode == "M" else None,
        "runtime_filex_sha256": sha256_file(run_dir / "runtime_snapshot" / "fileX.MZX") if (run_dir / "runtime_snapshot" / "fileX.MZX").exists() else None,
        "runtime_filex_sha256_with_wther_normalized": hashlib.sha256(_normalize_filex((run_dir / "runtime_snapshot" / "fileX.MZX").read_text(encoding="utf-8", errors="replace")).encode()).hexdigest().upper() if (run_dir / "runtime_snapshot" / "fileX.MZX").exists() else None,
        "station_coordinates_in_filex": _filex_coordinates(filex_text) if "filex_text" in locals() else None,
        "planting_date": planting_date,
        "emergence_date": emergence_date,
        "anthesis_dap": anthesis_dap,
        "anthesis_date": anthesis_date,
        "maturity_dap": maturity_dap,
        "maturity_date": maturity_date,
        "harvest_date": harvest_date,
        "season_length_dap_days": maturity_dap,
        "simulation_last_weather_date": next((row["date"] for row in reversed(state_rows) if row.get("date")), None),
        "summary_output": summary,
        "final_grain_yield_kg_ha": yield_kg_ha,
        "final_biomass_kg_ha": biomass_kg_ha,
        "yield_biomass_checks_pass": yield_ok,
        "max_lai": max(lai_values) if lai_values else None,
        "water_n_checks": water_check,
        "water_n_checks_pass": water_n_ok,
        "phenology_checks_pass": phenology_ok,
        "warning_line_count": sum(bool(line.strip()) for line in (run_dir / "runtime_snapshot" / "WARNING.OUT").read_text(encoding="utf-8", errors="replace").splitlines()) if (run_dir / "runtime_snapshot" / "WARNING.OUT").exists() else 0,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    _atomic_json(run_dir / "crop_smoke_status.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Serial, isolated YC DSSAT crop-output smoke")
    parser.add_argument("--runs", nargs="+", choices=("historical_control", "historical_control_retry1", "historical_control_retry2", "historical_control_retry3", "historical_control_retry4", "seed_101", "seed_104"), required=True)
    args = parser.parse_args()
    if sha256_file(CLI) != EXPECTED_CLI_SHA256 or sha256_file(TRAIN_WEATHER) != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("Frozen CLI or training weather hash preflight failed")
    for path in (SOURCE_FILEX, CULTIVAR, SOIL, HIST_WTH):
        if not path.is_file():
            raise FileNotFoundError(path)
    runtime_tmp = OUT / "runtime" / "tmp"
    runtime_tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(runtime_tmp)
    tempfile.tempdir = str(runtime_tmp)

    template_text = _coords_in_isolated_filex(prepare_wgen_filex(SOURCE_FILEX.read_text(encoding="utf-8", errors="replace")))
    template_version = "v4" if "historical_control_retry4" in args.runs or "seed_101" in args.runs or "seed_104" in args.runs else ("v3" if "historical_control_retry3" in args.runs else ("v2" if "historical_control_retry2" in args.runs else "v1"))
    template_path = OUT / "configs" / f"CNYC0801_crop_template_{template_version}.jinja2"
    if template_path.exists() and template_path.read_text(encoding="utf-8") != template_text:
        raise FileExistsError(f"Refusing to overwrite an existing different isolated template: {template_path}")
    if not template_path.exists():
        template_path.write_text(template_text, encoding="utf-8", newline="\n")

    provenance = {
        "scope": "YC 2008 treatment 1 only; random weather seeds 101 and 104 plus one historical 2008 measured-weather control",
        "source_filex": str(SOURCE_FILEX.relative_to(PROJECT_ROOT)),
        "source_filex_sha256": sha256_file(SOURCE_FILEX),
        "isolated_filex_template": str(template_path.relative_to(PROJECT_ROOT)),
        "isolated_filex_template_sha256": sha256_file(template_path),
        "template_change": "Only treatment 1 retained, WTHER rendered by wrapper, and FileX XCRD/YCRD/ELEV replaced in their original fixed-width column positions using existing YC station metadata; PHOTO L left unchanged.",
        "coordinate_source": "my_data/CNYC0801.WTH station header and task-provided YC metadata",
        "station_metadata": KNOWN_STATION,
        "cultivar_file": str(CULTIVAR.relative_to(PROJECT_ROOT)),
        "cultivar_sha256": sha256_file(CULTIVAR),
        "cultivar_code": "MZ/ZD0985",
        "soil_file": str(SOIL.relative_to(PROJECT_ROOT)),
        "soil_sha256": sha256_file(SOIL),
        "soil_profile": "YC99001200",
        "soil_missing_fields_left_unchanged": ["SLCF/STONES", "SADC/ADCOEF"],
        "corrected_cli_sha256": sha256_file(CLI),
        "train_weather_sha256": sha256_file(TRAIN_WEATHER),
        "historical_weather_file": str(HIST_WTH.relative_to(PROJECT_ROOT)),
        "historical_weather_sha256": sha256_file(HIST_WTH),
        "historical_control_mode": "random_weather=False; WTHER=M; byte-identical historical WTH copied as auxiliary input",
        "random_weather_mode": "random_weather=True; WTHER=W; corrected CLI hash checked at runtime",
        "management": "YC CNYC 2008 treatment 1; unchanged source management; constant zero Gym action as in WGEN pilot",
        "runtime": "DSSAT 4.8.0.024 via existing project Docker environment",
        "run_order": args.runs,
        "validation_weather_used": False,
        "wet_day_definition_changed": False,
        "wgen_refit": False,
        "ppo_training": False,
        "other_sites_modified": False,
    }
    provenance_path = OUT / f"crop_smoke_provenance_{'_'.join(args.runs)}.json"
    _atomic_json(provenance_path, provenance)

    run_map = {"historical_control": ("M", None), "historical_control_retry1": ("M", None), "historical_control_retry2": ("M", None), "historical_control_retry3": ("M", None), "historical_control_retry4": ("M", None), "seed_101": ("W", 101), "seed_104": ("W", 104)}
    outcomes = []
    for run_id in args.runs:
        mode, seed = run_map[run_id]
        print(f"[{run_id}] start serial DSSAT crop smoke mode={mode} seed={seed}", flush=True)
        result = run_one(run_id, mode, seed, template_path)
        outcomes.append(result)
        print(f"[{run_id}] status={result['runtime_status']} steps={result['runtime_steps']} yield={result['final_grain_yield_kg_ha']} topwt={result['final_biomass_kg_ha']}", flush=True)
        if result["runtime_status"] != "PASS":
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            return 2
    invocation_slug = "_".join(args.runs)
    _atomic_json(OUT / f"crop_smoke_runs_{invocation_slug}.json", {"runs": outcomes})
    print(json.dumps(outcomes, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
