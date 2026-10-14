"""Freeze 024 HLA scenario inputs and fit five isolated CNHL CLI candidates.

No WGEN, DSSAT, or PPO execution. Existing inputs are read only.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/hla_weather_enhancement_029"
SOURCE = ROOT / "results/hla_gridded_fill_sensitivity_024"
NAMES = ["gpcc_raw", "cpc_raw", "gpcc_biascorr", "cpc_biascorr", "ensemble_biascorr"]
CORE_INPUTS = [
    "data/external/人工大气观测降雨蒸发能见度日值.xls",
    "results/hla_random_weather_8seed/014_isd50756_recovery/daily_source_selection_after_014.csv",
    "results/hl_fq_random_weather_8seed/source_audit/nasa_candidate_gapfill.csv",
    "weather_clean/HLA_weather_cleaned.csv",
    "results/hla_gridded_fill_sensitivity_024/input_provenance.json",
    "results/hla_gridded_fill_sensitivity_024/candidate_integrity_check.json",
    "scripts/build_dssat_cli.py",
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_yc_fit_module():
    path = ROOT / "scripts/build_dssat_cli.py"
    spec = importlib.util.spec_from_file_location("yc_cli_fitting_029", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    if OUT.exists() and any((OUT / "source_freeze").glob("*")):
        raise RuntimeError("029 source freeze already exists; refusing to overwrite")
    freeze_dir = OUT / "source_freeze"
    fit_dir = OUT / "weather_fitting"
    freeze_dir.mkdir(parents=True, exist_ok=True)
    fit_dir.mkdir(parents=True, exist_ok=True)
    assets = CORE_INPUTS + [
        f"results/hla_gridded_fill_sensitivity_024/hla_candidate_{name}_2004_2013.csv"
        for name in NAMES
    ]
    hashes = []
    for relative in assets:
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(path)
        hashes.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha(path)})
    selection = ROOT / CORE_INPUTS[1]
    with selection.open(newline="", encoding="utf-8-sig") as f:
        src = list(csv.DictReader(f))
    source_counts = dict(Counter(row["selected_source"] for row in src if row["rain_status"] == "RESOLVED"))
    source_freeze = {
        "task": "029",
        "site": "HLA",
        "scope": "2004-01-01/2013-12-31",
        "expected_days": 3653,
        "source_status": "FIVE_GRIDDED_REFERENCE_SENSITIVITY_SCENARIOS; NO_FORMAL_SINGLE_RAIN_SOURCE_SELECTED",
        "scenario_selection": "User chose five isolated limited-candidate tracks in this task",
        "rainfall_source_hierarchy": "014 selected HLA D222 / GHCN 50756 / approved secondary GHCN on 3542 dates; five 024 gridded references fill the remaining 111 dates separately",
        "resolved_source_counts_before_111_fills": source_counts,
        "unresolved_014_dates": sum(row["rain_status"] == "UNRESOLVED" for row in src),
        "nonrain_rule": "weather_clean HLA SRAD/TMAX/TMIN plus 008 screened 92-date overlays as frozen in 024",
        "missing_treatment": "Five scenario-specific 024 gridded reference values on 111 dates; no interpolation/replacement elsewhere; outputs remain sensitivity inputs",
        "d222_blank_semantics": "Official D222 export blank code not documented; do not equate every blank with zero outside explicit source hierarchy",
        "date_anchor": "D222 20:00-to-20:00 ending-date convention supported generally; CERN export anchor inferred, not formally documented",
        "2013_climate_sanity": "SUSPICIOUS per 014/024; five scenarios add only about 2.84-3.10 mm to 2013 111 target days",
        "sha256_inputs": hashes,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (freeze_dir / "source_provenance.json").write_text(
        json.dumps(source_freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fit = load_yc_fit_module()
    summaries = []
    for name in NAMES:
        original = SOURCE / f"hla_candidate_{name}_2004_2013.csv"
        scenario = fit_dir / name
        scenario.mkdir(parents=True, exist_ok=True)
        weather = scenario / "fitting_weather.csv"
        shutil.copyfile(original, weather)
        if sha(original) != sha(weather):
            raise ValueError(f"copy hash mismatch for {name}")
        days, _ = fit.read_weather(weather, expected_sha256="")
        wgen = fit.fit_monthly_statistics(days)
        monthly = fit.monthly_weather_summary(days)
        cross = fit.crosscheck_statistics(days, wgen)
        if not cross["passed"]:
            raise ValueError(f"monthly parameter crosscheck failed: {name}")
        cli_text = fit._cli_text(days, "CNHL", 47.450, 126.900, 234.0, wgen, monthly)
        # YC checker is station-bound; substitute only for schema validation.
        schema = fit.check_cli_schema(cli_text.replace("CNHL", "CNYC"))
        if not schema["passed"] or cli_text.count("*CLIMATE:CNHL") != 1:
            raise ValueError(f"CLI schema failed: {name}")
        cli = scenario / "CNHL.CLI"
        cli.write_text(cli_text, encoding="ascii", newline="\n")
        write_csv(scenario / "monthly_wgen_statistics.csv", wgen)
        write_csv(scenario / "monthly_weather_summary.csv", monthly)
        (scenario / "cli_static_validation.json").write_text(
            json.dumps({"schema": schema, "parameter_crosscheck": cross, "station": "CNHL",
                        "validation_note": "YC schema checker station token adapted in memory only; output CLI retains CNHL",
                        "runtime_validated": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        annual_rain = {}
        for day in days:
            annual_rain[day.day.year] = annual_rain.get(day.day.year, 0.0) + day.rain
        summaries.append({
            "scenario": name.upper(), "fitting_weather": weather.relative_to(ROOT).as_posix(),
            "fitting_sha256": sha(weather), "cli": cli.relative_to(ROOT).as_posix(),
            "cli_sha256": sha(cli), "days": len(days), "months": len(wgen),
            "annual_rain_mean_mm": sum(annual_rain.values()) / len(annual_rain),
            "wet_days_gt0": sum(day.rain > 0 for day in days),
            "maximum_rain_mm": max(day.rain for day in days),
            "srad_min": min(day.srad for day in days),
            "tmax_min": min(day.tmax for day in days),
            "tmin_max": max(day.tmin for day in days),
            "static_cli_check": "PASS", "runtime_status": "NOT_RUN",
            "purpose": "SCENARIO_SENSITIVITY_ONLY_NOT_FORMAL_SOURCE",
        })
    write_csv(fit_dir / "scenario_fitting_inventory.csv", summaries)
    print(json.dumps({"source_counts": source_counts, "unresolved_dates": source_freeze["unresolved_014_dates"],
                      "fit_scenarios": len(summaries), "cli_static_pass": sum(row["static_cli_check"] == "PASS" for row in summaries)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
