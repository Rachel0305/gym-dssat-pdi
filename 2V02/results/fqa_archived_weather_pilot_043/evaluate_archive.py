"""Verify every archived 043 runtime realization and write a compact manifest/gate."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUNS = ["yc_2007_s1066", *[f"fq_2007_s{i}" for i in range(1001, 1006)], "fq_2007_s1001_repeat"]
FIELDS = ("RAIN", "SRAD", "TMAX", "TMIN")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def check_one(run_id: str):
    folder = OUT / "runtime" / run_id
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    source = json.loads((folder / "input_manifest.json").read_text(encoding="utf-8"))
    weather_file = folder / "runtime_weather_daily.csv"
    if not weather_file.is_file():
        raise FileNotFoundError(weather_file)
    if sha(weather_file) != result["weather_sha256"]:
        raise ValueError(f"weather SHA mismatch: {run_id}")
    for record in source.values():
        path = ROOT / record["path"]
        if not path.is_file() or sha(path) != record["sha256"]:
            raise ValueError(f"source changed or unavailable: {run_id} {record['path']}")
    with weather_file.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["DATE", "DOY", *FIELDS]:
            raise ValueError(f"archive columns mismatch: {run_id}")
        rows = list(reader)
    if len(rows) != result["steps"] or len(rows) != result["weather_row_count"]:
        raise ValueError(f"archive row count mismatch: {run_id}")
    dates = [dt.date.fromisoformat(row["DATE"]) for row in rows]
    if any((b-a).days != 1 for a, b in zip(dates, dates[1:])):
        raise ValueError(f"archive DATE not consecutive: {run_id}")
    if any(int(row["DOY"]) != day.timetuple().tm_yday for row, day in zip(rows, dates)):
        raise ValueError(f"archive DOY mismatch: {run_id}")
    numbers = [{key: float(row[key]) for key in FIELDS} for row in rows]
    if any(not all(math.isfinite(v) for v in row.values()) or row["RAIN"] < 0 or row["SRAD"] < 0 or row["TMAX"] < row["TMIN"] for row in numbers):
        raise ValueError(f"archive physical check failed: {run_id}")
    if result["status"] != "ARCHIVED_RUNTIME_WEATHER" or not result["episode_done"] or not result["runtime_seed_confirmed"] or result["runtime_filex_wther"] != "W":
        raise ValueError(f"runtime gate failed: {run_id}")
    if result["runtime_cli_sha256"] != source["cli"]["sha256"]:
        raise ValueError(f"runtime CLI differs: {run_id}")
    snapshot = folder / "runtime_snapshot"
    inp = (snapshot / "DSSAT48.INP").read_text(encoding="latin-1")
    warning = (snapshot / "WARNING.OUT").read_text(encoding="latin-1")
    native_missing_coords = bool(re.search(r"-999\.00000\s+-99\.00000\s+-99\.00", inp.split("*FIELDS", 1)[1].split("*INITIAL CONDITIONS", 1)[0]))
    ipfld_warnings = all(term in warning for term in ("Error reading latitude", "Error reading longitude", "Error reading elevation"))
    return {
        "run_id": run_id, "site": result["site"], "year": result["year"], "treatment": result["treatment"],
        "weather_seed": result["weather_seed"], "context": result["capture_context"], "days": len(rows),
        "first_date": dates[0].isoformat(), "last_date": dates[-1].isoformat(),
        "weather_archive": str(weather_file.relative_to(ROOT)).replace("\\", "/"), "weather_sha256": sha(weather_file),
        "cli_sha256": source["cli"]["sha256"], "filex_template_sha256": source["template"]["sha256"],
        "runtime_filex_sha256": sha(snapshot / "fileX.MZX"), "native_inp_sha256": sha(snapshot / "DSSAT48.INP"),
        "rain_mm": sum(row["RAIN"] for row in numbers), "wet_days": sum(row["RAIN"] > 0 for row in numbers),
        "max_rain_mm": max(row["RAIN"] for row in numbers), "mean_srad": sum(row["SRAD"] for row in numbers)/len(rows),
        "mean_tmax": sum(row["TMAX"] for row in numbers)/len(rows), "mean_tmin": sum(row["TMIN"] for row in numbers)/len(rows),
        "peak_rss_mb": result["peak_process_tree_rss_mb"],
        "native_field_coords_missing": native_missing_coords, "ipfld_coordinate_warnings": ipfld_warnings,
        "archive_qc": "PASS", "climate_distribution_qc": "NOT_ASSESSED_SMALL_N",
    }


def main():
    if (OUT / "final_gate.json").exists():
        raise FileExistsError("043 final gate exists; refusing overwrite")
    entries = [check_one(run_id) for run_id in RUNS]
    main_entries = [row for row in entries if row["site"] == "FQA" and not row["run_id"].endswith("repeat")]
    repeated_equal = (OUT / "runtime/fq_2007_s1001/runtime_weather_daily.csv").read_bytes() == (OUT / "runtime/fq_2007_s1001_repeat/runtime_weather_daily.csv").read_bytes()
    distinct = len({row["weather_sha256"] for row in main_entries})
    yc = entries[0]
    fq = entries[1:]
    gate = {
        "task": "043", "scope": "one YC control plus five FQA seeds and one repeated FQA seed; 2007 treatment 1; no PPO",
        "all_realizations_archived": len(entries) == 7 and all(row["archive_qc"] == "PASS" for row in entries),
        "yc_control_native_coordinate_status": "MISSING_WITH_IPFLD_WARNINGS" if yc["native_field_coords_missing"] and yc["ipfld_coordinate_warnings"] else "OTHER",
        "fqa_native_coordinate_status": "MISSING_WITH_IPFLD_WARNINGS" if all(row["native_field_coords_missing"] and row["ipfld_coordinate_warnings"] for row in fq) else "OTHER",
        "yc_fqa_native_coordinate_behavior_same_in_this_smoke": yc["native_field_coords_missing"] and yc["ipfld_coordinate_warnings"] and all(row["native_field_coords_missing"] and row["ipfld_coordinate_warnings"] for row in fq),
        "coordinate_functional_impact": "UNRESOLVED",
        "fqa_distinct_weather_hashes_across_5_seeds": distinct,
        "fqa_different_seed_diversity": "PASS_SMALL_PILOT" if distinct == 5 else "FAIL",
        "fqa_seed1001_repeat_exact": repeated_equal,
        "fqa_same_seed_reproducibility": "PASS_THIS_FRESH_PROCESS_CONTEXT" if repeated_equal else "FAIL",
        "fqa_climate_distribution_qc": "NOT_ASSESSED_SMALL_N",
        "fqa_weather_pool_80_20_created": False,
        "fqa_ppo_executed": False,
        "training_entrypoint_per_episode_archive_hook": "CONTRACT_DEFINED_NOT_INTEGRATED",
        "ready_for_formal_ppo": False,
        "manifest": "results/fqa_archived_weather_pilot_043/realization_manifest.csv",
    }
    with (OUT / "realization_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)
    (OUT / "final_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: gate[key] for key in ("all_realizations_archived", "yc_fqa_native_coordinate_behavior_same_in_this_smoke", "fqa_distinct_weather_hashes_across_5_seeds", "fqa_seed1001_repeat_exact", "ready_for_formal_ppo")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
