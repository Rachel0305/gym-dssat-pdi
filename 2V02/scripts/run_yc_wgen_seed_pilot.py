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
import sys
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "results" / "yc_wgen_cli_pilot" / "003_06_05"
CLI_PATH = PROJECT_ROOT / "results" / "yc_wgen_cli_pilot" / "003_06_04" / "generated" / "CNYC.CLI"
TRAIN_WEATHER = PROJECT_ROOT / "results" / "yc_weather_gapfill_finalize" / "yc_wgen_fitting_weather_2004_2013.csv"
YC_INPUT = PROJECT_ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013" / "YC"
SOURCE_FILEX = YC_INPUT / "CNYC0801.MZX"
CULTIVAR_PATH = YC_INPUT / "MZCER048.CUL"
SOIL_PATH = YC_INPUT / "SOIL.SOL"
EXPECTED_CLI_SHA256 = "5ABF5D7BB97EFAAE5E8361ABB4C773E1554B58116CBCA2213E75DCFF838285F0"
EXPECTED_TRAIN_SHA256 = "4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34"
RUN_SEEDS = {
    "seed_101_run_a": 101,
    "seed_101_run_b": 101,
    "seed_102": 102,
    "seed_103": 103,
    "seed_104": 104,
    "seed_105": 105,
}
WEATHER_FIELDS = ("RAIN", "SRAD", "TMAX", "TMIN")
SIMULATION_START = date(2008, 6, 1)
MAX_STEPS = 400


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def prepare_wgen_filex(source_text: str) -> str:
    """Create a one-treatment pilot copy and expose the wrapper's WTHER variable."""
    lines: list[str] = []
    in_climate = False
    for line in source_text.splitlines():
        if line.lstrip().upper().startswith("*CLIMATE"):
            in_climate = True
        if not in_climate and re.match(r"^\s*2\s+", line):
            continue
        lines.append(line)
    text = "\n".join(lines) + "\n"
    text, replacements = re.subn(
        r"(?m)^([ \t]*1[ \t]+ME[ \t]+)\S+",
        r"\g<1>{{ wther }}",
        text,
        count=1,
    )
    if replacements != 1:
        raise ValueError("Could not template WTHER for isolated WGEN pilot")
    if not re.search(r"(?m)^\s*1\s+1\s+1\s+0\s+Sim2008\b", text):
        raise ValueError("YC 2008 treatment row is missing from the pilot FileX")
    if re.search(r"(?m)^\s*2\s+", text.split("*CLIMATE", 1)[0]):
        raise ValueError("Non-2008 treatment rows remain in the isolated pilot FileX")
    return text


def runtime_filex_mode(filex_text: str) -> str | None:
    lines = filex_text.splitlines()
    for index, line in enumerate(lines):
        if line.lstrip().upper().startswith("@N METHODS"):
            for row in lines[index + 1 :]:
                tokens = row.split()
                if tokens and tokens[0].isdigit():
                    return tokens[2] if len(tokens) > 2 else None
                if row.lstrip().startswith("*"):
                    break
    return None


def pdi_seed_configured(pdi_text: str, seed: int) -> bool:
    return bool(re.search(rf"(?m)^\s*rseed1_\s*=\s*{int(seed)}\s*$", pdi_text))


def _scalar(value: Any) -> float | str | None:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return value
        return parsed if math.isfinite(parsed) else None
    return None


def _state_lookup(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    return {str(key).casefold(): value for key, value in state.items()}


def _find_number(state: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = _scalar(state.get(name.casefold()))
        if isinstance(value, float):
            return value
    return None


def daily_weather_from_states(states: list[dict[str, Any]], start_date: date = SIMULATION_START) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, raw_state in enumerate(states):
        state = _state_lookup(raw_state)
        values = {
            "RAIN": _find_number(state, ("rain", "rainf", "precip", "precipitation")),
            "SRAD": _find_number(state, ("srad", "solar_radiation", "rad")),
            "TMAX": _find_number(state, ("tmax", "max_temp", "maximum_temperature")),
            "TMIN": _find_number(state, ("tmin", "min_temp", "minimum_temperature")),
        }
        if not any(value is not None for value in values.values()):
            continue

        explicit_date = state.get("date") or state.get("weather_date")
        explicit_doy = _find_number(state, ("doy", "day_of_year", "jday"))
        dap = _find_number(state, ("dap", "days_after_planting"))
        parsed_date: date | None = None
        if isinstance(explicit_date, str):
            for fmt in ("%Y-%m-%d", "%Y%j", "%y%j"):
                try:
                    parsed_date = datetime.strptime(explicit_date.strip(), fmt).date()
                    break
                except ValueError:
                    continue
        elif isinstance(explicit_date, (int, float)):
            date_text = str(int(explicit_date)).zfill(5)
            try:
                parsed_date = datetime.strptime(date_text, "%y%j").date()
            except ValueError:
                pass

        if parsed_date is None and explicit_doy is not None and 1 <= explicit_doy <= 366:
            year_value = _find_number(state, ("year", "year4", "yyyy"))
            year = int(year_value) if year_value and year_value >= 1000 else start_date.year
            parsed_date = date(year, 1, 1) + timedelta(days=int(explicit_doy) - 1)
        if parsed_date is None:
            parsed_date = start_date + timedelta(days=index)

        rows.append(
            {
                "DATE": parsed_date.isoformat(),
                "DOY": parsed_date.timetuple().tm_yday,
                "RAIN": values["RAIN"],
                "SRAD": values["SRAD"],
                "TMAX": values["TMAX"],
                "TMIN": values["TMIN"],
            }
        )
    return rows


def parse_generated_wth(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header: list[str] | None = None
    rows: list[dict[str, Any]] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("@") and "DATE" in stripped.upper():
            header = [token.upper().lstrip("@") for token in stripped.split()]
            continue
        if not stripped or stripped.startswith("*") or header is None:
            continue
        tokens = stripped.split()
        if len(tokens) < len(header) or not re.fullmatch(r"\d{5}", tokens[0]):
            continue
        values = dict(zip(header, tokens))
        try:
            parsed_date = datetime.strptime(tokens[0], "%y%j").date()
            row = {
                "DATE": parsed_date.isoformat(),
                "DOY": parsed_date.timetuple().tm_yday,
            }
            for field in WEATHER_FIELDS:
                row[field] = float(values[field]) if field in values else None
            if any(row[field] is not None for field in WEATHER_FIELDS):
                rows.append(row)
        except (ValueError, KeyError):
            continue
    return rows


def canonical_weather_bytes(rows: list[dict[str, Any]]) -> bytes:
    columns = ("DATE", "DOY", "RAIN", "SRAD", "TMAX", "TMIN")
    ordered = sorted(rows, key=lambda row: (str(row.get("DATE", "")), int(row.get("DOY") or 0)))
    output = [",".join(columns)]
    for row in ordered:
        values: list[str] = []
        for key in columns:
            value = row.get(key)
            if value is None:
                values.append("")
            elif key == "DATE":
                values.append(str(value))
            elif isinstance(value, (int, float)):
                values.append(format(float(value), ".12g"))
            else:
                values.append(str(value))
        output.append(",".join(values))
    return ("\n".join(output) + "\n").encode("utf-8")


def longest_dry_spell(rows: list[dict[str, Any]]) -> int:
    longest = current = 0
    for row in sorted(rows, key=lambda item: item["DATE"]):
        rain = row.get("RAIN")
        if isinstance(rain, (int, float)) and float(rain) <= 0.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def summarize_weather(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "window_type": "simulation-season statistics",
        "row_count": len(rows),
        "start_date": min((row["DATE"] for row in rows), default=None),
        "end_date": max((row["DATE"] for row in rows), default=None),
    }
    for field in WEATHER_FIELDS:
        values = [float(row[field]) for row in rows if isinstance(row.get(field), (int, float))]
        summary[f"{field.lower()}_n"] = len(values)
        summary[f"mean_{field.lower()}"] = sum(values) / len(values) if values else None
        summary[f"min_{field.lower()}"] = min(values) if values else None
        summary[f"max_{field.lower()}"] = max(values) if values else None
    rains = [float(row["RAIN"]) for row in rows if isinstance(row.get("RAIN"), (int, float))]
    summary["season_rainfall_mm"] = sum(rains) if rains else None
    summary["rainy_days_gt_0mm"] = sum(value > 0.0 for value in rains)
    summary["maximum_daily_rainfall_mm"] = max(rains) if rains else None
    summary["longest_dry_spell_days"] = longest_dry_spell(rows)
    return summary


def physical_sanity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    issues: list[str] = []
    missing_by_field: dict[str, int] = {}
    for field in WEATHER_FIELDS:
        missing_by_field[field] = sum(not isinstance(row.get(field), (int, float)) for row in rows)
    for index, row in enumerate(rows):
        rain, srad, tmax, tmin = (row.get(name) for name in WEATHER_FIELDS)
        if isinstance(rain, (int, float)) and rain < 0:
            issues.append(f"row_{index + 1}: negative RAIN")
        if isinstance(srad, (int, float)) and srad < 0:
            issues.append(f"row_{index + 1}: negative SRAD")
        if isinstance(tmax, (int, float)) and isinstance(tmin, (int, float)) and tmax < tmin:
            issues.append(f"row_{index + 1}: TMAX below TMIN")
        if isinstance(rain, (int, float)) and rain > 500:
            issues.append(f"row_{index + 1}: RAIN exceeds 500 mm/day sanity bound")
        if isinstance(srad, (int, float)) and srad > 45:
            issues.append(f"row_{index + 1}: SRAD exceeds 45 MJ/m2/day sanity bound")
        if isinstance(tmax, (int, float)) and not -60 <= tmax <= 60:
            issues.append(f"row_{index + 1}: TMAX outside [-60, 60] C sanity bound")
        if isinstance(tmin, (int, float)) and not -80 <= tmin <= 50:
            issues.append(f"row_{index + 1}: TMIN outside [-80, 50] C sanity bound")
    no_rain = bool(rows) and not any(
        isinstance(row.get("RAIN"), (int, float)) and row["RAIN"] > 0 for row in rows
    )
    if no_rain:
        issues.append("all-zero rainfall in captured simulation window")
    return {
        "status": (
            "NOT_RUN_NO_WEATHER_OUTPUT"
            if not rows
            else "PASS" if not issues and not any(missing_by_field.values()) else "FAIL_OR_INCOMPLETE"
        ),
        "row_count": len(rows),
        "missing_by_field": missing_by_field,
        "all_zero_rainfall": no_rain,
        "issues": issues,
        "bounds_are_screening_checks_not_calibrated_extreme_value_tests": True,
    }


def compare_weather_rows(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    left_rows = sorted(left, key=lambda row: row["DATE"])
    right_rows = sorted(right, key=lambda row: row["DATE"])
    paired = min(len(left_rows), len(right_rows))
    variable_stats: dict[str, Any] = {}
    for field in WEATHER_FIELDS:
        diffs: list[float] = []
        for a, b in zip(left_rows[:paired], right_rows[:paired]):
            va, vb = a.get(field), b.get(field)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                diffs.append(float(va) - float(vb))
        variable_stats[field] = {
            "compared_values": len(diffs),
            "different_values": sum(abs(value) > 1e-10 for value in diffs),
            "mean_absolute_difference": sum(abs(value) for value in diffs) / len(diffs) if diffs else None,
            "max_absolute_difference": max((abs(value) for value in diffs), default=None),
        }
    dates_aligned = [row["DATE"] for row in left_rows[:paired]] == [row["DATE"] for row in right_rows[:paired]]
    return {
        "left_row_count": len(left_rows),
        "right_row_count": len(right_rows),
        "dates_aligned": dates_aligned,
        "variables": variable_stats,
        "different_weather_values": any(item["different_values"] > 0 for item in variable_stats.values()),
    }


def classify_runtime_error(message: str) -> str:
    text = message.casefold()
    if "cnyc.cli" in text and any(token in text for token in ("not found", "missing", "cannot open", "no such file")):
        return "FAIL_CLI_LOOKUP"
    if any(token in text for token in ("parse error", "schema mismatch", "invalid cli", "malformed cli")) or (
        "wgenin" in text and ("5010" in text or "unknown error" in text)
    ):
        return "FAIL_CLI_PARSE"
    if any(token in text for token in ("missing field", "required field", "field not found")):
        return "FAIL_REQUIRED_FIELD"
    if "rseed1" in text or ("seed" in text and any(token in text for token in ("ignored", "invalid", "interface", "did not receive"))):
        return "FAIL_SEED_INTERFACE"
    if "wgen" in text or "weather generator" in text or "weather generation" in text:
        return "FAIL_WGEN_RUNTIME"
    if any(token in text for token in ("timed out", "zmq.again")):
        return "OTHER_SPECIFIC_FAILURE"
    return "OTHER_SPECIFIC_FAILURE"


def _float_cell(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def read_weather_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for source_row in csv.DictReader(stream):
            date_text = str(source_row.get("DATE", ""))
            try:
                parsed_date = date.fromisoformat(date_text)
            except ValueError:
                continue
            rows.append(
                {
                    "DATE": parsed_date.isoformat(),
                    "DOY": int(source_row.get("DOY") or parsed_date.timetuple().tm_yday),
                    **{field: _float_cell(source_row.get(field)) for field in WEATHER_FIELDS},
                }
            )
    return rows


def frozen_monthly_reference(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, dict[str, list[float]]] = {}
    for row in rows:
        month = date.fromisoformat(row["DATE"]).month
        grouped.setdefault(month, {field: [] for field in WEATHER_FIELDS})
        for field in WEATHER_FIELDS:
            value = row.get(field)
            if isinstance(value, (int, float)):
                grouped[month][field].append(float(value))
    return {
        str(month): {
            field: (sum(values) / len(values) if values else None)
            for field, values in fields.items()
        }
        for month, fields in sorted(grouped.items())
    }


def _runtime_class(module: Any) -> type:
    candidates = {
        value
        for value in vars(module).values()
        if isinstance(value, type)
        and value.__module__ == module.__name__
        and hasattr(value, "_get_sockets_")
        and hasattr(value, "_get_state")
    }
    if len(candidates) != 1:
        raise RuntimeError(f"Could not uniquely identify Gym-DSSAT runtime class; found {len(candidates)}")
    return next(iter(candidates))


def _process_tree_rss_mb() -> float:
    try:
        import psutil

        process = psutil.Process()
        processes = [process, *process.children(recursive=True)]
        return sum(item.memory_info().rss for item in processes if item.is_running()) / (1024 * 1024)
    except Exception:
        return 0.0


def _write_runtime_evidence(path: Path, values: dict[str, Any]) -> None:
    lines = [f"{key}: {value}" for key, value in values.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _snapshot_runtime_files(temp_folder: Path, run_dir: Path) -> list[str]:
    snapshot = run_dir / "runtime_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for path in sorted(temp_folder.iterdir()) if temp_folder.exists() else []:
        if not path.is_file():
            continue
        if path.suffix.upper() == ".WTH" or path.name.upper() in {
            "CNYC.CLI", "FILEX.MZX", "DSSAT-PDI.YML", "ERROR.OUT", "INFO.OUT", "WARNING.OUT",
            "RUNLIST.OUT", "DSSAT48.INP", "DSSAT48.INH",
        }:
            target = snapshot / path.name
            shutil.copy2(path, target)
            copied.append(str(target.relative_to(PROJECT_ROOT)))
    return copied


def run_one(run_id: str, weather_seed: int) -> dict[str, Any]:
    run_dir = OUTPUT_ROOT / "runtime" / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing pilot run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime_log = run_dir / "runtime_log.txt"
    started = time.time()
    cli_hash = sha256_file(CLI_PATH)
    if cli_hash != EXPECTED_CLI_SHA256:
        raise RuntimeError(f"CNYC.CLI hash mismatch before run: {cli_hash}")

    env = None
    original_methods: tuple[Any, Any] | None = None
    class_type: type | None = None
    captured_states: list[dict[str, Any]] = []
    max_tree_rss_mb = 0.0
    runtime_cwd: str | None = None
    copied_files: list[str] = []
    error_text: str | None = None
    status = "OTHER_SPECIFIC_FAILURE"
    cli_runtime_hash: str | None = None
    filex_text: str | None = None
    pdi_yaml_text: str | None = None
    filex_mode: str | None = None
    run_context: dict[str, Any] = {}

    try:
        import gym
        import gym_dssat_pdi.envs.dssat_pdi as dssat_module

        class_type = _runtime_class(dssat_module)
        original_get_sockets = class_type._get_sockets_
        original_get_state = class_type._get_state

        def pilot_get_state(instance: Any) -> Any:
            result = original_get_state(instance)
            state = getattr(instance, "_state", None)
            if isinstance(state, dict):
                instance._pilot_weather_states.append(copy.deepcopy(state))
            return result

        def pilot_get_sockets(instance: Any) -> Any:
            instance._pilot_weather_states = []
            instance._pilot_weather_seed = int(weather_seed)
            instance._rseed1 = int(weather_seed)
            run_context["instance"] = instance
            run_context["temp_folder"] = Path(instance._tmp_folder)
            result = original_get_sockets(instance)
            if getattr(instance, "_server", None) is not None:
                import zmq

                instance._server.setsockopt(zmq.RCVTIMEO, 30000)
            return result

        class_type._get_state = pilot_get_state
        class_type._get_sockets_ = pilot_get_sockets
        original_methods = (original_get_state, original_get_sockets)

        env = gym.make(
            "gym_dssat_pdi:GymDssatPdi-v0",
            log_saving_path=str(runtime_log),
            mode="all",
            auxiliary_file_paths=[str(CULTIVAR_PATH), str(CLI_PATH), str(SOIL_PATH)],
            random_weather=True,
            seed=None,
            fileX_template_path=str(OUTPUT_ROOT / "configs" / "CNYC0801_wgen_template.jinja2"),
            experiment_number=1,
            evaluation=True,
            cultivar="maize",
            run_dssat_location="/opt/dssat_pdi/run_dssat",
        ).unwrapped

        runtime_cwd = str(Path(env._tmp_folder).resolve())
        runtime_cli = Path(env._tmp_folder) / "CNYC.CLI"
        if not runtime_cli.is_file():
            raise FileNotFoundError("CNYC.CLI not found in Gym-DSSAT runtime working directory")
        cli_runtime_hash = sha256_file(runtime_cli)
        if cli_runtime_hash != EXPECTED_CLI_SHA256:
            raise RuntimeError(f"Runtime CNYC.CLI hash mismatch: {cli_runtime_hash}")

        runtime_filex = Path(env._tmp_folder) / "fileX.MZX"
        filex_text = runtime_filex.read_text(encoding="utf-8", errors="replace")
        filex_mode = runtime_filex_mode(filex_text)
        if filex_mode != "W":
            raise RuntimeError(f"Runtime FileX WTHER is {filex_mode!r}, expected 'W'")
        pdi_yaml = Path(env._tmp_folder) / "dssat-pdi.yml"
        pdi_yaml_text = pdi_yaml.read_text(encoding="utf-8", errors="replace") if pdi_yaml.exists() else ""
        if not pdi_seed_configured(pdi_yaml_text, weather_seed):
            raise RuntimeError(f"PDI config did not receive weather_seed={weather_seed} as rseed1")
        if not re.search(r"(?m)^\s*1\s+1\s+1\s+0\s+Sim2008\b", filex_text):
            raise RuntimeError("Runtime FileX lost YC 2008 treatment")

        action = {key: 0.0 for key in env.action_variables}
        max_tree_rss_mb = max(max_tree_rss_mb, _process_tree_rss_mb())
        steps = 0
        while not bool(getattr(env, "done", False)) and steps < MAX_STEPS:
            env.step(action)
            steps += 1
            max_tree_rss_mb = max(max_tree_rss_mb, _process_tree_rss_mb())
        if not bool(getattr(env, "done", False)):
            raise TimeoutError(f"DSSAT WGEN pilot did not finish within {MAX_STEPS} daily steps")

        copied_files = _snapshot_runtime_files(Path(env._tmp_folder), run_dir)
        states = list(getattr(env, "_pilot_weather_states", []))
        weather_rows = daily_weather_from_states(states)
        capture_method = "runtime_daily_state" if weather_rows else ""
        if not weather_rows:
            for relative in copied_files:
                copied_path = PROJECT_ROOT / relative
                if copied_path.suffix.upper() == ".WTH":
                    parsed = parse_generated_wth(copied_path)
                    if parsed:
                        weather_rows.extend(parsed)
            if weather_rows:
                capture_method = "generated_runtime_WTH_file"

        if weather_rows:
            weather_path = OUTPUT_ROOT / "generated_weather" / f"{run_id}.csv"
            weather_path.parent.mkdir(parents=True, exist_ok=True)
            canonical = canonical_weather_bytes(weather_rows)
            weather_path.write_bytes(canonical)
            weather_hash = sha256_bytes(canonical)
            weather_rows = read_weather_csv(weather_path)
            capture_status = "CAPTURED"
        else:
            weather_path = None
            weather_hash = None
            capture_status = "WEATHER_SEQUENCE_NOT_DIRECTLY_OBSERVABLE"

        status = "PASS"
        run_result = {
            "run_id": run_id,
            "station": "YC",
            "weather_seed": weather_seed,
            "ppo_seed": "NOT_APPLICABLE",
            "seed_interface": "pilot weather_seed explicitly injected into Gym-DSSAT PDI rseed1 before runtime launch",
            "steps": steps,
            "runtime_compatibility_status": status,
            "filex_wther": filex_mode,
            "runtime_cli_path": str(runtime_cli),
            "runtime_cli_sha256": cli_runtime_hash,
            "runtime_working_directory": runtime_cwd,
            "weather_capture_method": capture_method or "WEATHER_SEQUENCE_NOT_DIRECTLY_OBSERVABLE",
            "weather_capture_status": capture_status,
            "weather_path": str(weather_path.relative_to(PROJECT_ROOT)) if weather_path else None,
            "weather_sha256": weather_hash,
            "weather_row_count": len(weather_rows),
            "weather_summary": summarize_weather(weather_rows),
            "physical_sanity": physical_sanity(weather_rows),
            "runtime_snapshot_files": copied_files,
            "runtime_log": str(runtime_log.relative_to(PROJECT_ROOT)),
            "peak_process_tree_rss_mb": round(max_tree_rss_mb, 2),
            "elapsed_seconds": round(time.time() - started, 3),
            "runtime_state_keys": sorted({str(key) for state in states for key in state}),
        }
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        failed_instance = run_context.get("instance")
        failed_temp = run_context.get("temp_folder")
        if env is None and isinstance(failed_temp, Path) and failed_temp.exists():
            runtime_cwd = str(failed_temp.resolve())
            copied_files = _snapshot_runtime_files(failed_temp, run_dir)
            runtime_cli = failed_temp / "CNYC.CLI"
            cli_runtime_hash = sha256_file(runtime_cli) if runtime_cli.exists() else None
            runtime_filex = failed_temp / "fileX.MZX"
            if runtime_filex.exists():
                filex_text = runtime_filex.read_text(encoding="utf-8", errors="replace")
                filex_mode = runtime_filex_mode(filex_text)
            pdi_yaml = failed_temp / "dssat-pdi.yml"
            if pdi_yaml.exists():
                pdi_yaml_text = pdi_yaml.read_text(encoding="utf-8", errors="replace")
            diagnostic_text: list[str] = []
            for diagnostic_path in (runtime_log, failed_temp / "ERROR.OUT", failed_temp / "WARNING.OUT"):
                if diagnostic_path.exists():
                    diagnostic_text.append(diagnostic_path.read_text(encoding="utf-8", errors="replace")[-5000:])
            if diagnostic_text:
                error_text += "; runtime_diagnostics=" + "\n".join(diagnostic_text)
            try:
                import psutil

                client_pid = getattr(failed_instance, "_client_process_pid", None)
                if client_pid and psutil.pid_exists(int(client_pid)):
                    process = psutil.Process(int(client_pid))
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except psutil.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=2)
                if failed_instance is not None:
                    failed_instance._client_process_pid = None
                    if getattr(failed_instance, "_server", None) is not None:
                        failed_instance._server.close(0)
                        failed_instance._server = None
                    if getattr(failed_instance, "_zmq_context", None) is not None:
                        failed_instance._zmq_context.term()
                        failed_instance._zmq_context = None
                    failed_instance.closed = True
            except Exception as cleanup_exc:
                error_text += f"; partial_init_cleanup={type(cleanup_exc).__name__}: {cleanup_exc}"
        status = classify_runtime_error(error_text)
        if env is not None:
            try:
                copied_files = _snapshot_runtime_files(Path(env._tmp_folder), run_dir)
            except Exception as snapshot_exc:
                error_text += f"; snapshot_error={type(snapshot_exc).__name__}: {snapshot_exc}"
        run_result = {
            "run_id": run_id,
            "station": "YC",
            "weather_seed": weather_seed,
            "ppo_seed": "NOT_APPLICABLE",
            "runtime_compatibility_status": status,
            "error": error_text,
            "runtime_cli_sha256": cli_runtime_hash,
            "runtime_working_directory": runtime_cwd,
            "filex_wther": filex_mode,
            "pdi_rseed1_configured": bool(pdi_yaml_text and pdi_seed_configured(pdi_yaml_text, weather_seed)),
            "runtime_snapshot_files": copied_files,
            "runtime_log": str(runtime_log.relative_to(PROJECT_ROOT)),
            "peak_process_tree_rss_mb": round(max_tree_rss_mb, 2),
            "elapsed_seconds": round(time.time() - started, 3),
        }
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as close_exc:
                run_result.setdefault("cleanup_warning", f"{type(close_exc).__name__}: {close_exc}")
        if class_type is not None and original_methods is not None:
            class_type._get_state, class_type._get_sockets_ = original_methods
        if CLI_PATH.exists():
            run_result["candidate_cli_sha256_after_run"] = sha256_file(CLI_PATH)
        (run_dir / "wgen_status.json").write_text(
            json.dumps(run_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        evidence = {
            "source_cli_path": str(CLI_PATH.relative_to(PROJECT_ROOT)),
            "source_cli_sha256": cli_hash,
            "runtime_cli_basename": "CNYC.CLI",
            "runtime_cli_sha256": cli_runtime_hash,
            "runtime_working_directory": runtime_cwd,
            "filex_runtime_basename": "fileX.MZX",
            "filex_wsta": "CNYC0801",
            "filex_wther": filex_mode,
            "runtime_pdi_rseed1": weather_seed,
            "weather_seed": weather_seed,
            "ppo_seed": "NOT_APPLICABLE",
            "runtime_log_path": str(runtime_log.relative_to(PROJECT_ROOT)),
            "runtime_snapshot_files": copied_files,
            "compatibility_status": status,
            "error": error_text,
        }
        _write_runtime_evidence(run_dir / "cli_copy_evidence.txt", evidence)

    return run_result


def _load_run_result(run_id: str) -> dict[str, Any] | None:
    run_dir = OUTPUT_ROOT / "runtime" / run_id
    path = run_dir / "wgen_status.json"
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if "pdi_rseed1_received" in result:
        result.pop("pdi_rseed1_received")
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if result.get("runtime_compatibility_status") in {"OTHER_SPECIFIC_FAILURE", "FAIL_CLI_PARSE"} and not result.get("pdi_rseed1_configured"):
        tmp_root = OUTPUT_ROOT / "runtime" / "tmp"
        candidates = sorted(
            (folder for folder in tmp_root.iterdir() if folder.is_dir() and (folder / "ERROR.OUT").exists()),
            key=lambda folder: (folder / "ERROR.OUT").stat().st_mtime,
            reverse=True,
        ) if tmp_root.exists() else []
        for runtime_dir in candidates:
            runtime_cli = runtime_dir / "CNYC.CLI"
            if not runtime_cli.exists() or sha256_file(runtime_cli) != EXPECTED_CLI_SHA256:
                continue
            error_out = (runtime_dir / "ERROR.OUT").read_text(encoding="utf-8", errors="replace")
            warning_path = runtime_dir / "WARNING.OUT"
            warning_out = warning_path.read_text(encoding="utf-8", errors="replace") if warning_path.exists() else ""
            runtime_log_path = run_dir / "runtime_log.txt"
            runtime_log = runtime_log_path.read_text(encoding="utf-8", errors="replace") if runtime_log_path.exists() else ""
            detail = "\n".join(part for part in (runtime_log, error_out, warning_out) if part)
            status = classify_runtime_error(detail)
            copied_files = _snapshot_runtime_files(runtime_dir, run_dir)
            filex_path = runtime_dir / "fileX.MZX"
            filex_text = filex_path.read_text(encoding="utf-8", errors="replace") if filex_path.exists() else ""
            pdi_path = runtime_dir / "dssat-pdi.yml"
            pdi_text = pdi_path.read_text(encoding="utf-8", errors="replace") if pdi_path.exists() else ""
            seed = RUN_SEEDS[run_id]
            result.update(
                {
                    "init_exception": result.get("error"),
                    "error": detail[-12000:],
                    "runtime_compatibility_status": status,
                    "runtime_cli_sha256": sha256_file(runtime_cli),
                    "runtime_working_directory": str(runtime_dir.resolve()),
                    "filex_wther": runtime_filex_mode(filex_text),
                    "pdi_rseed1_configured": pdi_seed_configured(pdi_text, seed),
                    "runtime_snapshot_files": copied_files,
                    "runtime_diagnostics": {
                        "error_out": str((run_dir / "runtime_snapshot" / "ERROR.OUT").relative_to(PROJECT_ROOT)),
                        "warning_out": str((run_dir / "runtime_snapshot" / "WARNING.OUT").relative_to(PROJECT_ROOT)) if warning_out else None,
                        "error_key_wgenin_5010": "WGENIN" in detail and "5010" in detail,
                    },
                }
            )
            evidence = {
                "source_cli_path": str(CLI_PATH.relative_to(PROJECT_ROOT)),
                "source_cli_sha256": EXPECTED_CLI_SHA256,
                "runtime_cli_path": str((run_dir / "runtime_snapshot" / "CNYC.CLI").relative_to(PROJECT_ROOT)),
                "runtime_cli_basename": "CNYC.CLI",
                "runtime_cli_sha256": result["runtime_cli_sha256"],
                "runtime_working_directory": result["runtime_working_directory"],
                "filex_runtime_basename": "fileX.MZX",
                "filex_wsta": "CNYC0801",
                "filex_wther": result["filex_wther"],
                "runtime_pdi_rseed1": seed if result["pdi_rseed1_configured"] else "not confirmed",
                "weather_seed": seed,
                "ppo_seed": "NOT_APPLICABLE",
                "compatibility_status": status,
                "runtime_error": "DSSAT WGENIN error 5010 at CNYC.CLI line 27" if result["runtime_diagnostics"]["error_key_wgenin_5010"] else result.get("init_exception"),
                "runtime_snapshot_files": copied_files,
            }
            _write_runtime_evidence(run_dir / "cli_copy_evidence.txt", evidence)
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            break
    return result


def _load_generated_weather(run_id: str) -> list[dict[str, Any]]:
    path = OUTPUT_ROOT / "generated_weather" / f"{run_id}.csv"
    return read_weather_csv(path) if path.exists() else []


def _write_json(path: Path, content: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def analyze_results() -> dict[str, Any]:
    result_dir = OUTPUT_ROOT
    result_dir.mkdir(parents=True, exist_ok=True)
    results = {run_id: _load_run_result(run_id) for run_id in RUN_SEEDS}
    present = {run_id: value for run_id, value in results.items() if value is not None}
    weather = {run_id: _load_generated_weather(run_id) for run_id in present}
    hashes = {
        run_id: (result.get("weather_sha256") if result else None)
        for run_id, result in present.items()
    }

    same_seed = {
        "seed": 101,
        "run_a": "seed_101_run_a",
        "run_b": "seed_101_run_b",
        "run_a_hash": hashes.get("seed_101_run_a"),
        "run_b_hash": hashes.get("seed_101_run_b"),
        "identical": (
            hashes.get("seed_101_run_a") == hashes.get("seed_101_run_b")
            if hashes.get("seed_101_run_a") and hashes.get("seed_101_run_b")
            else None
        ),
        "variables_compared": list(WEATHER_FIELDS),
        "row_count_run_a": len(weather.get("seed_101_run_a", [])),
        "row_count_run_b": len(weather.get("seed_101_run_b", [])),
        "status": "CHECKED" if hashes.get("seed_101_run_a") and hashes.get("seed_101_run_b") else "PENDING",
    }
    _write_json(result_dir / "reproducibility_check.json", same_seed)

    unique_seeds = [f"seed_{seed}" for seed in range(101, 106)]
    pairwise: list[dict[str, Any]] = []
    for index, left_id in enumerate(unique_seeds):
        for right_id in unique_seeds[index + 1 :]:
            left_hash, right_hash = hashes.get(left_id), hashes.get(right_id)
            pairwise.append(
                {
                    "left_run": left_id,
                    "right_run": right_id,
                    "left_hash": left_hash,
                    "right_hash": right_hash,
                    "identical": left_hash == right_hash if left_hash and right_hash else None,
                    "weather_differences": (
                        compare_weather_rows(weather.get(left_id, []), weather.get(right_id, []))
                        if weather.get(left_id) and weather.get(right_id)
                        else None
                    ),
                }
            )
    distinct_hashes = {hashes.get(run_id) for run_id in unique_seeds if hashes.get(run_id)}
    diversity = {
        "seeds_tested": [RUN_SEEDS[run_id] for run_id in unique_seeds if run_id in present],
        "weather_hashes": {run_id: hashes.get(run_id) for run_id in unique_seeds},
        "pairwise_comparisons": pairwise,
        "distinct_weather_sequences": len(distinct_hashes),
        "all_five_seeds_present": all(run_id in present for run_id in unique_seeds),
        "different_seeds_distinct": (
            len(distinct_hashes) == 5 if all(hashes.get(run_id) for run_id in unique_seeds) else None
        ),
        "not_all_seeds_identical": (
            len(distinct_hashes) > 1 if all(hashes.get(run_id) for run_id in unique_seeds) else None
        ),
    }
    _write_json(result_dir / "seed_diversity_check.json", diversity)

    summary_rows: list[dict[str, Any]] = []
    all_rows: dict[str, list[dict[str, Any]]] = {}
    for run_id, result in present.items():
        rows = weather.get(run_id, [])
        all_rows[run_id] = rows
        summary = result.get("weather_summary", summarize_weather(rows))
        summary_rows.append(
            {
                "run_id": run_id,
                "weather_seed": RUN_SEEDS[run_id],
                "capture_method": result.get("weather_capture_method") or "WEATHER_SEQUENCE_NOT_DIRECTLY_OBSERVABLE",
                "weather_sha256": hashes.get(run_id),
                **summary,
            }
        )
    summary_path = result_dir / "weather_summary_by_seed.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "run_id", "weather_seed", "capture_method", "weather_sha256", "window_type", "row_count",
        "start_date", "end_date", "season_rainfall_mm", "rainy_days_gt_0mm", "maximum_daily_rainfall_mm",
        "longest_dry_spell_days", "mean_srad", "mean_tmax", "mean_tmin", "min_tmin", "max_tmax", "max_srad",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)

    combined_rows = [row for rows in all_rows.values() for row in rows]
    per_seed_sanity = {run_id: result.get("physical_sanity", physical_sanity(all_rows[run_id])) for run_id, result in present.items()}
    train_rows = read_weather_csv(TRAIN_WEATHER) if TRAIN_WEATHER.exists() else []
    train_hash = sha256_file(TRAIN_WEATHER) if TRAIN_WEATHER.exists() else None
    climate = frozen_monthly_reference(train_rows)
    monthly_comparison: list[dict[str, Any]] = []
    for run_id, rows in all_rows.items():
        months: dict[int, dict[str, list[float]]] = {}
        for row in rows:
            month = date.fromisoformat(row["DATE"]).month
            months.setdefault(month, {field: [] for field in WEATHER_FIELDS})
            for field in WEATHER_FIELDS:
                value = row.get(field)
                if isinstance(value, (int, float)):
                    months[month][field].append(float(value))
        for month, fields in sorted(months.items()):
            record: dict[str, Any] = {"run_id": run_id, "month": month}
            for field, values in fields.items():
                generated_mean = sum(values) / len(values) if values else None
                reference_mean = climate.get(str(month), {}).get(field)
                record[f"generated_mean_{field.lower()}"] = generated_mean
                record[f"training_2004_2013_mean_{field.lower()}"] = reference_mean
                record[f"difference_{field.lower()}"] = (
                    generated_mean - reference_mean
                    if generated_mean is not None and reference_mean is not None
                    else None
                )
            monthly_comparison.append(record)
    physical_status = (
        "NOT_RUN_NO_WEATHER_OUTPUT"
        if not combined_rows
        else "PASS" if present and all(item.get("status") == "PASS" for item in per_seed_sanity.values()) else "FAIL_OR_INCOMPLETE"
    )
    physical = {
        "overall_status": physical_status,
        "per_seed": per_seed_sanity,
        "frozen_training_reference": {
            "path": str(TRAIN_WEATHER.relative_to(PROJECT_ROOT)),
            "sha256": train_hash,
            "expected_sha256": EXPECTED_TRAIN_SHA256,
            "hash_verified": train_hash == EXPECTED_TRAIN_SHA256,
            "date_start": min((row["DATE"] for row in train_rows), default=None),
            "date_end": max((row["DATE"] for row in train_rows), default=None),
            "row_count": len(train_rows),
            "monthly_means": climate,
        },
        "monthly_climatology_sanity_comparison": monthly_comparison,
        "interpretation": "Train-period-only sanity reference; descriptive, not a formal fit or acceptance threshold.",
        "validation_data_used_for_fitting": False,
    }
    _write_json(result_dir / "physical_sanity_check.json", physical)

    input_hash_after = sha256_file(CLI_PATH) if CLI_PATH.exists() else None
    training_hash_after = sha256_file(TRAIN_WEATHER) if TRAIN_WEATHER.exists() else None
    pipeline_statuses = [item.get("runtime_compatibility_status") for item in present.values()]
    all_runs_completed = all(run_id in present for run_id in RUN_SEEDS)
    all_runtime_passed = all_runs_completed and all(value == "PASS" for value in pipeline_statuses)
    same_pass = same_seed["identical"] is True
    diversity_pass = diversity["different_seeds_distinct"] is True
    physical_pass = physical["overall_status"] == "PASS"
    capture_pass = all_rows and all(len(rows) > 0 for rows in all_rows.values())
    wgen_status = (
        "WGEN_SEED_PILOT_PASS"
        if all_runtime_passed and same_pass and diversity_pass and physical_pass and capture_pass and input_hash_after == EXPECTED_CLI_SHA256 and training_hash_after == EXPECTED_TRAIN_SHA256
        else "INCOMPLETE_OR_FAILED"
    )
    final = {
        "task": "003_06_05_run_wgen_seed_pilot",
        "station": "YC",
        "cli_path": str(CLI_PATH.relative_to(PROJECT_ROOT)),
        "cli_sha256_expected": EXPECTED_CLI_SHA256,
        "cli_sha256_before": EXPECTED_CLI_SHA256 if CLI_PATH.exists() else None,
        "cli_sha256_after": input_hash_after,
        "cli_hash_verified": input_hash_after == EXPECTED_CLI_SHA256,
        "training_weather_sha256_after": training_hash_after,
        "training_weather_hash_verified": training_hash_after == EXPECTED_TRAIN_SHA256,
        "weather_candidate_modified": training_hash_after != EXPECTED_TRAIN_SHA256,
        "cli_modified": input_hash_after != EXPECTED_CLI_SHA256,
        "validation_data_used_for_fitting": False,
        "ppo_training_run": False,
        "other_sites_modified": False,
        "runs_requested": list(RUN_SEEDS),
        "runs_completed": list(present),
        "run_results": present,
        "runtime_compatibility_status": "PASS" if all_runtime_passed else next((value for value in pipeline_statuses if value != "PASS"), "INCOMPLETE"),
        "weather_capture_method": sorted({item.get("weather_capture_method") or "WEATHER_SEQUENCE_NOT_DIRECTLY_OBSERVABLE" for item in present.values()}),
        "same_seed_reproducible": same_seed["identical"],
        "same_seed_weather_hash": same_seed["run_a_hash"] if same_seed["identical"] else None,
        "different_seeds_distinct": diversity["different_seeds_distinct"],
        "distinct_weather_sequences": diversity["distinct_weather_sequences"],
        "physical_sanity_status": physical["overall_status"],
        "seed_coupling_status": "SEPARATE: pilot weather_seed is explicitly injected as PDI rseed1; no PPO seed used",
        "wgen_seed_pilot_status": wgen_status,
        "peak_process_tree_rss_mb": max((float(item.get("peak_process_tree_rss_mb", 0)) for item in present.values()), default=0.0),
        "remaining_blockers": [
            f"{run_id}: {item.get('error')}" for run_id, item in present.items() if item.get("runtime_compatibility_status") != "PASS"
        ],
    }
    _write_json(result_dir / "pilot_summary.json", final)
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated YC internal-WGEN seed pilot")
    parser.add_argument(
        "--runs",
        nargs="+",
        choices=list(RUN_SEEDS),
        default=list(RUN_SEEDS),
        help="Pilot run IDs; execute serially. Use one ID first as the runtime smoke gate.",
    )
    parser.add_argument("--analysis-only", action="store_true", help="Recompute machine-readable summaries without DSSAT")
    args = parser.parse_args()

    runtime_tmp = OUTPUT_ROOT / "runtime" / "tmp"
    runtime_tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(runtime_tmp)
    tempfile.tempdir = str(runtime_tmp)

    if not args.analysis_only:
        if not CLI_PATH.is_file() or sha256_file(CLI_PATH) != EXPECTED_CLI_SHA256:
            print("FAIL_CLI_PARSE: frozen candidate CLI hash/path preflight failed", file=sys.stderr)
            return 2
        if not TRAIN_WEATHER.is_file() or sha256_file(TRAIN_WEATHER) != EXPECTED_TRAIN_SHA256:
            print("FAIL_PHYSICAL_SANITY: frozen train-weather reference hash/path preflight failed", file=sys.stderr)
            return 2
        for path in (SOURCE_FILEX, CULTIVAR_PATH, SOIL_PATH):
            if not path.is_file():
                print(f"OTHER_SPECIFIC_FAILURE: missing YC input {path}", file=sys.stderr)
                return 2

        configs_dir = OUTPUT_ROOT / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        template_path = configs_dir / "CNYC0801_wgen_template.jinja2"
        template_text = prepare_wgen_filex(SOURCE_FILEX.read_text(encoding="utf-8", errors="replace"))
        if template_path.exists():
            if template_path.read_text(encoding="utf-8") != template_text:
                print("Refusing to overwrite a different isolated FileX template", file=sys.stderr)
                return 2
        else:
            template_path.write_text(template_text, encoding="utf-8", newline="\n")

        provenance = {
            "station": "YC",
            "site_code": "CNYC",
            "source_filex": str(SOURCE_FILEX.relative_to(PROJECT_ROOT)),
            "source_filex_sha256": sha256_file(SOURCE_FILEX),
            "isolated_filex_template": str(template_path.relative_to(PROJECT_ROOT)),
            "cultivar_input": str(CULTIVAR_PATH.relative_to(PROJECT_ROOT)),
            "cultivar_sha256": sha256_file(CULTIVAR_PATH),
            "soil_input": str(SOIL_PATH.relative_to(PROJECT_ROOT)),
            "soil_sha256": sha256_file(SOIL_PATH),
            "cli_input": str(CLI_PATH.relative_to(PROJECT_ROOT)),
            "cli_sha256": sha256_file(CLI_PATH),
            "train_weather_reference": str(TRAIN_WEATHER.relative_to(PROJECT_ROOT)),
            "train_weather_sha256": sha256_file(TRAIN_WEATHER),
            "selected_treatment": "YC/CNYC 2008, treatment 1 (isolated copy only)",
            "filex_wsta": "CNYC0801; DSSAT CLI lookup basename expected CNYC.CLI",
            "random_weather": True,
            "filex_wther_template_value": "{{ wther }} rendered by Gym-DSSAT as W",
            "weather_seed_map": RUN_SEEDS,
            "ppo_seed": "NOT_APPLICABLE",
            "action_policy": "constant zero action for all runtime action variables; no agent or PPO",
            "max_daily_steps": MAX_STEPS,
            "temporary_directory_root": str(runtime_tmp.relative_to(PROJECT_ROOT)),
            "validation_weather_used": False,
            "existing_wrapper_modified": False,
        }
        _write_json(OUTPUT_ROOT / "runtime" / "pilot_input_provenance.json", provenance)

        for run_id in args.runs:
            print(f"[{run_id}] start YC WGEN runtime smoke; weather_seed={RUN_SEEDS[run_id]}", flush=True)
            outcome = run_one(run_id, RUN_SEEDS[run_id])
            _write_json(OUTPUT_ROOT / "runtime" / run_id / "wgen_status.json", outcome)
            print(
                f"[{run_id}] status={outcome.get('runtime_compatibility_status')} "
                f"captured={outcome.get('weather_capture_status', 'NO')} "
                f"rows={outcome.get('weather_row_count', 0)} "
                f"peak_process_tree_rss_mb={outcome.get('peak_process_tree_rss_mb', 0)}",
                flush=True,
            )
            if outcome.get("runtime_compatibility_status") != "PASS":
                break

    final = analyze_results()
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return 0 if final["wgen_seed_pilot_status"] == "WGEN_SEED_PILOT_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
