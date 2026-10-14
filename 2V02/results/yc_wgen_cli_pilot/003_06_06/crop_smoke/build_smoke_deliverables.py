from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
TASK_ROOT = PROJECT_ROOT / "results" / "yc_wgen_cli_pilot" / "003_06_06"
CROP_ROOT = TASK_ROOT / "crop_smoke"
AUDIT_ROOT = TASK_ROOT / "runtime_warning_audit"
CLI_SHA256 = "65CF134600A5881706A5D435E1A09B276ED92A21FA5ABE2E18AAF63AF1E3A929"
TRAIN_SHA256 = "4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34"
RUNS = (
    ("historical_control_retry4", "M", None),
    ("seed_101", "W", 101),
    ("seed_104", "W", 104),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_dssat_table(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_index = next((i for i, line in enumerate(lines) if line.lstrip().startswith("@YEAR")), None)
    if header_index is None:
        raise ValueError(f"No @YEAR table header in {path}")
    headers = [value.lstrip("@") for value in lines[header_index].split()]
    rows = []
    for line in lines[header_index + 1:]:
        values = line.split()
        if values and values[0].isdigit() and len(values) == len(headers):
            rows.append(dict(zip(headers, values)))
    if not rows:
        raise ValueError(f"No data rows in DSSAT output {path}")
    return rows


def finite_stats(rows: list[dict[str, str]], variable: str) -> dict[str, float | int]:
    values = [float(row[variable]) for row in rows if row.get(variable, "")]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"No finite DSSAT values for {variable}")
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "days_nonzero": sum(abs(value) > 1e-9 for value in values),
        "n": len(values),
    }


def copy_evidence(source: Path, target: Path) -> None:
    if target.exists():
        if sha256_file(source) != sha256_file(target):
            raise FileExistsError(f"Refusing to overwrite different evidence: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def event_count(text: str, token: str) -> int:
    return text.count(token)


def main() -> int:
    expected_train = PROJECT_ROOT / "results" / "yc_weather_gapfill_finalize" / "yc_wgen_fitting_weather_2004_2013.csv"
    expected_cli = PROJECT_ROOT / "results" / "yc_wgen_cli_pilot" / "003_06_05_02" / "final" / "CNYC.CLI"
    if sha256_file(expected_cli) != CLI_SHA256 or sha256_file(expected_train) != TRAIN_SHA256:
        raise RuntimeError("Frozen corrected CLI or train-weather hash mismatch")

    crop_rows: list[dict[str, Any]] = []
    phenology_rows: list[dict[str, Any]] = []
    stress_rows: list[dict[str, Any]] = []
    status_by_run: dict[str, dict[str, Any]] = {}
    warning_events: dict[str, dict[str, Any]] = {}
    evidence_hashes: dict[str, str] = {}
    for run_id, mode, seed in RUNS:
        run_dir = CROP_ROOT / run_id
        status = load_json(run_dir / "crop_smoke_status.json")
        status_by_run[run_id] = status
        if status["runtime_status"] != "PASS":
            raise RuntimeError(f"Expected a completed DSSAT run for {run_id}")
        if status["runtime_cli_sha256"] != CLI_SHA256 or status["runtime_wther"] != mode:
            raise RuntimeError(f"Runtime CLI or WTHER preflight mismatch for {run_id}")
        if seed is not None and status.get("weather_seed") != seed:
            raise RuntimeError(f"WGEN seed mismatch for {run_id}")
        if mode == "M" and status.get("historical_wth_sha256") != "39B814537927E012B7EB04A238F88DE9DF8C92C011FDB9854F0B7A121E3B7003":
            raise RuntimeError("Historical 2008 WTH hash mismatch")

        snapshot = run_dir / "runtime_snapshot"
        warning_text = (snapshot / "WARNING.OUT").read_text(encoding="utf-8", errors="replace")
        info_text = (snapshot / "INFO.OUT").read_text(encoding="utf-8", errors="replace")
        evidence_dir = AUDIT_ROOT / "warning_evidence" / run_id
        for name in ("WARNING.OUT", "INFO.OUT", "DSSAT48.INP", "Summary.OUT", "PlantGro.OUT", "SoilWat.OUT", "SoilNi.OUT", "MgmtEvent.OUT", "fileX.MZX"):
            source = snapshot / name
            if source.is_file():
                target = evidence_dir / name
                copy_evidence(source, target)
                evidence_hashes[str(target.relative_to(PROJECT_ROOT))] = sha256_file(target)

        plant_rows = read_dssat_table(snapshot / "PlantGro.OUT")
        soil_rows = read_dssat_table(snapshot / "SoilWat.OUT")
        crop = status["summary_output"]
        stress = {name: finite_stats(plant_rows, name) for name in ("WSPD", "WSGD", "NSTD")}
        soil_water = {name: finite_stats(soil_rows, name) for name in ("SWTD", "SWXD")}
        for name, stats in stress.items():
            stress_rows.append({"run_id": run_id, "mode": mode, "weather_seed": seed, "indicator": name, **stats})
        for name, stats in soil_water.items():
            stress_rows.append({"run_id": run_id, "mode": mode, "weather_seed": seed, "indicator": name, **stats})

        crop_rows.append({
            "run_id": run_id,
            "weather_mode": "historical_measured" if mode == "M" else "random_WGEN",
            "weather_seed": seed,
            "runtime_status": status["runtime_status"],
            "runtime_steps": status["runtime_steps"],
            "runtime_wther": status["runtime_wther"],
            "normalized_filex_sha256": status["runtime_filex_sha256_with_wther_normalized"],
            "planting_date": status["planting_date"],
            "emergence_date": status["emergence_date"],
            "anthesis_date": status["anthesis_date"],
            "anthesis_dap": status["anthesis_dap"],
            "maturity_date": status["maturity_date"],
            "maturity_dap": status["maturity_dap"],
            "harvest_date": status["harvest_date"],
            "season_length_dap_days": status["season_length_dap_days"],
            "yield_kg_ha": status["final_grain_yield_kg_ha"],
            "biomass_kg_ha": status["final_biomass_kg_ha"],
            "max_lai": status["max_lai"],
            "crop_period_rain_mm": crop["PRCM"],
            "cumulative_irrigation_mm": crop["IRCM"],
            "cumulative_fertilizer_n_kg_ha": crop["NICM"],
            "cumulative_et_mm": crop["ETCM"],
            "n_uptake_kg_ha": crop["NUCM"],
            "min_profile_soil_water_mm": soil_water["SWTD"]["min"],
            "max_profile_soil_water_mm": soil_water["SWTD"]["max"],
            "min_extractable_water_mm": soil_water["SWXD"]["min"],
            "max_extractable_water_mm": soil_water["SWXD"]["max"],
            "max_wspd": stress["WSPD"]["max"],
            "max_wsgd": stress["WSGD"]["max"],
            "max_nstd": stress["NSTD"]["max"],
            "management_events_verified": crop["IRCM"] == 120.0 and crop["NICM"] == 303.0,
            "finite_output_checks_pass": all(math.isfinite(float(crop[key])) for key in ("HWAM", "CWAM", "IRCM", "NICM", "NUCM", "ETCM")),
        })
        phenology_rows.append({
            "run_id": run_id,
            "weather_seed": seed,
            "planting_date": status["planting_date"],
            "emergence_date": status["emergence_date"],
            "anthesis_date": status["anthesis_date"],
            "anthesis_dap": status["anthesis_dap"],
            "maturity_date": status["maturity_date"],
            "maturity_dap": status["maturity_dap"],
            "harvest_date": status["harvest_date"],
            "season_length_dap_days": status["season_length_dap_days"],
            "phenology_checks_pass": status["phenology_checks_pass"],
        })

        for name, text, token, category in (
            ("PHOTO L->C", warning_text, "Photosynthesis method (PHOTO in FILEX) has been changed", "B"),
            ("Latitude read error", warning_text, "Error reading latitude from experimental file", "C"),
            ("Longitude read error", warning_text, "Error reading longitude from experimental file", "C"),
            ("Elevation read error", warning_text, "Error reading elevation from experimental file", "C"),
            ("CYCRDin transfer", warning_text, "Error transferring variable: CYCRDin FIELD", "C"),
            ("CXCRDin transfer", warning_text, "Error transferring variable: CXCRDin FIELD", "C"),
            ("CELEVin transfer", warning_text, "Error transferring variable: CELEVin FIELD", "C"),
            ("STONES/ADCOEF defaults", info_text, "The soil file has missing or invalid data.", "B"),
        ):
            count = event_count(text, token)
            if count:
                entry = warning_events.setdefault(name, {"class": category, "occurrences_per_run": count})
                if entry["class"] != category or entry["occurrences_per_run"] != count:
                    raise ValueError(f"Warning frequency changed across primary runs: {name}")

    normalized_hashes = {status["runtime_filex_sha256_with_wther_normalized"] for status in status_by_run.values()}
    if len(normalized_hashes) != 1:
        raise ValueError(f"Treatment/cultivar/soil FileX differs after WTHER normalization: {normalized_hashes}")

    previous_snapshot = PROJECT_ROOT / "results" / "yc_wgen_cli_pilot" / "003_06_05_02" / "runtime" / "seed_101_run_a" / "runtime_snapshot"
    if previous_snapshot.exists():
        for name in ("WARNING.OUT", "INFO.OUT", "DSSAT48.INP", "fileX.MZX"):
            source = previous_snapshot / name
            if source.is_file():
                target = AUDIT_ROOT / "warning_evidence" / "previous_003_06_05_02" / name
                copy_evidence(source, target)
                evidence_hashes[str(target.relative_to(PROJECT_ROOT))] = sha256_file(target)

    row_specs = (
        ("PHOTO L->C", "PHOTO", "FileX.MZX PHOTO=L", "DSSAT reports fallback method C", "Maize photosynthesis method", "B", "既有兼容回退，historical 与 WGEN 一致；不擅自改模型设置。", "WARNING.OUT"),
        ("Latitude read error", "FileX latitude", "YC latitude 36.830; isolated FileX contains 36.83000", "DSSAT48.INP placeholder; Summary.OUT coordinate blank; CYCRDin ultimately set to zero", "Location-dependent model metadata; crop-output influence not isolated", "C", "运行时没有确认 YC 纬度进入 DSSAT；需先修复读取/传递并重跑三组。", "WARNING.OUT; DSSAT48.INP; Summary.OUT"),
        ("Longitude read error", "FileX longitude", "YC longitude 116.570; isolated FileX contains 116.57000", "DSSAT48.INP placeholder; Summary.OUT coordinate blank; CXCRDin ultimately set to zero", "Location-dependent model metadata; crop-output influence not isolated", "C", "运行时没有确认 YC 经度进入 DSSAT；需先修复读取/传递并重跑三组。", "WARNING.OUT; DSSAT48.INP; Summary.OUT"),
        ("Elevation read error", "FileX elevation", "YC elevation 22 m; isolated FileX contains 22.0", "DSSAT48.INP placeholder; Summary.OUT coordinate blank; CELEVin ultimately set to zero", "Location-dependent model metadata; crop-output influence not isolated", "C", "运行时没有确认 YC 海拔进入 DSSAT；需先修复读取/传递并重跑三组。", "WARNING.OUT; DSSAT48.INP; Summary.OUT"),
        ("CYCRDin transfer", "DSSAT field transfer", "latitude 36.830", "transfer errors at initialization/start/end; final GET reports zero", "DSSAT field coordinate state", "C", "与纬度读取错误同一根因；不得把 FileX 文本正确误报为模型读入成功。", "WARNING.OUT"),
        ("CXCRDin transfer", "DSSAT field transfer", "longitude 116.570", "transfer errors at initialization/start/end; final GET reports zero", "DSSAT field coordinate state", "C", "与经度读取错误同一根因；不得把 FileX 文本正确误报为模型读入成功。", "WARNING.OUT"),
        ("CELEVin transfer", "DSSAT field transfer", "elevation 22 m", "transfer errors at initialization/start/end; final GET reports zero", "DSSAT field coordinate state", "C", "与海拔读取错误同一根因；不得把 FileX 文本正确误报为模型读入成功。", "WARNING.OUT"),
        ("STONES/ADCOEF defaults", "YC SOIL.SOL profile YC99001200", "profile values are -99/missing; no measured replacements available", "INFO.OUT confirms defaults; Soil details show STONES/SLCF=0.0% and ADCOEF=0.0", "soil water/root and nitrate-retention/transport processes", "B", "historical control 与随机天气共用同一 SOIL.SOL/default；保留为基线不确定性，不凭空补值。", "INFO.OUT; SOIL.SOL"),
    )
    audit_rows = []
    for signal, source, expected, actual, process, category, note, evidence in row_specs:
        event = warning_events[signal]
        count = event["occurrences_per_run"]
        audit_rows.append({
            "warning_signal": signal,
            "class": category,
            "source": source,
            "expected_value_or_state": expected,
            "actual_runtime_value_or_state": actual,
            "likely_affected_process": process,
            "fatal_to_smoke": "NO; DSSAT ended normally" if category != "C" else "NO immediate failure; potential metadata risk unresolved",
            "crop_output_relevance": "shared compatibility fallback" if signal == "PHOTO L->C" else ("may influence soil and nitrogen processes; same baseline inputs" if category == "B" else "not isolated; cannot rule out location-dependent effects"),
            "historical_control_comparison": "present in historical and both WGEN runs",
            "needs_fix_before_crop_smoke": "NO; smoke completed" if category != "C" else "smoke completed, but runtime read still fails",
            "needs_fix_before_ppo": "NO; freeze and disclose baseline fallback" if category == "B" else "YES",
            "occurrences_per_primary_run": count,
            "occurrences_across_three_primary_runs": count * len(RUNS),
            "evidence_files": evidence,
            "notes": note,
        })
    audit_rows.append({
        "warning_signal": "Cultivar-specific warning not observed",
        "class": "N/A",
        "source": "MZCER048.CUL and DSSAT48.INP",
        "expected_value_or_state": "MZ/ZD0985 is present in cultivar file",
        "actual_runtime_value_or_state": "MZ/ZD0985 resolved; no cultivar-related WARNING.OUT message",
        "likely_affected_process": "phenology and yield cultivar parameters",
        "fatal_to_smoke": "NO",
        "crop_output_relevance": "no warning observed; do not infer parameter calibration validity from this check",
        "historical_control_comparison": "same cultivar in all three runs",
        "needs_fix_before_crop_smoke": "NO",
        "needs_fix_before_ppo": "NO cultivar-warning blocker identified",
        "occurrences_per_primary_run": 0,
        "occurrences_across_three_primary_runs": 0,
        "evidence_files": "DSSAT48.INP; MZCER048.CUL; WARNING.OUT",
        "notes": "Previous-run checklist mentioned cultivar warnings; none were present in the current or copied prior WARNING.OUT evidence.",
    })

    write_csv(CROP_ROOT / "crop_output_summary.csv", crop_rows)
    write_csv(CROP_ROOT / "phenology_summary.csv", phenology_rows)
    write_csv(CROP_ROOT / "dssat_water_n_stress_summary.csv", stress_rows)
    write_csv(AUDIT_ROOT / "runtime_warning_audit.csv", audit_rows)

    warning_total = sum(event["occurrences_per_run"] * len(RUNS) for event in warning_events.values())
    class_totals = {
        category: sum(event["occurrences_per_run"] * len(RUNS) for event in warning_events.values() if event["class"] == category)
        for category in ("A", "B", "C")
    }
    warning_summary = {
        "definition": "Counts discrete warning/error or INFO-default events, not nonempty log lines; counts repeated occurrences over the three primary crop runs.",
        "primary_runs": [run_id for run_id, _, _ in RUNS],
        "runtime_warning_total": warning_total,
        "warning_class_A": class_totals["A"],
        "warning_class_B": class_totals["B"],
        "warning_class_C": class_totals["C"],
        "warnings_blocking_ppo": class_totals["C"] > 0,
        "cultivar_warning_observed": False,
        "per_run_event_counts": {name: event["occurrences_per_run"] for name, event in warning_events.items()},
        "all_primary_runs_completed_normally": all(status["runtime_status"] == "PASS" for status in status_by_run.values()),
        "previous_003_06_05_02_seed_101_warning_evidence_copied": previous_snapshot.exists(),
        "evidence_sha256": evidence_hashes,
        "coordinate_failure_evidence": "FileX.MZX retains isolated values, while DSSAT48.INP retains placeholders, Summary.OUT location columns are blank, and WARNING.OUT reports transfer values set to zero.",
        "soil_defaults_observed": {"STONES_SLCF_percent": 0.0, "ADCOEF_cm3_g": 0.0},
        "class_B_interpretation": "PHOTO L-to-C and STONES/ADCOEF fallbacks were also present in the historical control; no input values were fabricated.",
        "class_C_interpretation": "YC latitude/longitude/elevation were not accepted into DSSAT's runtime field metadata; fix the parser/transfer path and repeat the three-run smoke before PPO.",
    }
    (AUDIT_ROOT / "warning_summary.json").write_text(json.dumps(warning_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    yield_pass = all(row["yield_kg_ha"] >= 0 and row["biomass_kg_ha"] >= row["yield_kg_ha"] and row["finite_output_checks_pass"] for row in crop_rows)
    phenology_pass = all(row["phenology_checks_pass"] for row in phenology_rows)
    management_pass = all(row["management_events_verified"] for row in crop_rows)
    stress_pass = all(
        math.isfinite(float(row["min"])) and math.isfinite(float(row["max"])) and 0 <= float(row["min"]) <= float(row["max"]) <= 1
        for row in stress_rows if row["indicator"] in {"WSPD", "WSGD", "NSTD"}
    )
    crop_summary = {
        "scope": "YC 2008 treatment 1 only; historical 2008 WTH control and WGEN weather seeds 101 and 104; serial runs; no PPO.",
        "runtime_version": "DSSAT 4.8.0.024",
        "corrected_cli_sha256": sha256_file(expected_cli),
        "corrected_cli_hash_verified": sha256_file(expected_cli) == CLI_SHA256,
        "frozen_train_weather_sha256": sha256_file(expected_train),
        "frozen_train_weather_hash_verified": sha256_file(expected_train) == TRAIN_SHA256,
        "validation_weather_used_for_fitting": False,
        "wgen_refit": False,
        "wet_day_definition_changed": False,
        "ppo_training_run": False,
        "other_sites_modified": False,
        "run_ids": [row["run_id"] for row in crop_rows],
        "wther_by_run": {row["run_id"]: row["runtime_wther"] for row in crop_rows},
        "weather_seed_runs": [row["weather_seed"] for row in crop_rows if row["weather_seed"] is not None],
        "same_filex_after_wther_normalization": len(normalized_hashes) == 1,
        "runtime_runs_completed": all(status["runtime_status"] == "PASS" for status in status_by_run.values()),
        "phenology_status": "PASS" if phenology_pass else "FAIL",
        "yield_biomass_status": "PASS" if yield_pass else "FAIL",
        "management_water_n_event_status": "PASS" if management_pass else "FAIL",
        "dssat_stress_factor_finite_and_0_to_1": stress_pass,
        "crop_smoke_status": "PASS" if all((phenology_pass, yield_pass, management_pass, stress_pass)) else "FAIL",
        "warnings_blocking_ppo": warning_summary["warnings_blocking_ppo"],
        "yc_random_weather_dssat_smoke_status": "BLOCKED_BEFORE_PPO" if warning_summary["warnings_blocking_ppo"] else "YC_RANDOM_WEATHER_DSSAT_SMOKE_PASS",
        "recommended_next_step": "修复 DSSAT FileX 坐标读取/传递后，以相同 CLI、天气、土壤和管理串行重跑历史对照与 seeds 101/104；未通过前不启动 PPO。",
        "runs": crop_rows,
        "stress_summary_rows": stress_rows,
    }
    (CROP_ROOT / "crop_smoke_summary.json").write_text(json.dumps(crop_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "crop_smoke_status": crop_summary["crop_smoke_status"],
        "yc_random_weather_dssat_smoke_status": crop_summary["yc_random_weather_dssat_smoke_status"],
        "runtime_warning_total": warning_total,
        "warning_classes": class_totals,
        "crop_output_summary_rows": len(crop_rows),
        "stress_summary_rows": len(stress_rows),
        "same_filex_after_wther_normalization": crop_summary["same_filex_after_wther_normalization"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
