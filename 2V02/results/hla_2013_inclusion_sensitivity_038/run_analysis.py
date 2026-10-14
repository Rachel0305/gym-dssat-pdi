"""Project-local, fitting-only HLA 2013 inclusion sensitivity audit."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
INV = ROOT / "results/hla_weather_enhancement_029/weather_fitting/scenario_fitting_inventory.csv"
SOURCE_014 = ROOT / "results/hla_random_weather_8seed/014_isd50756_recovery/daily_source_selection_after_014.csv"
SANITY_014 = ROOT / "results/hla_random_weather_8seed/014_isd50756_recovery/2013_climate_sanity.json"
SUMMARY_024 = ROOT / "results/hla_gridded_fill_sensitivity_024/diagnostic_summary.json"
FIT_SCRIPT = ROOT / "scripts/build_dssat_cli.py"
NAMES = ("GPCC_RAW", "CPC_RAW", "GPCC_BIASCORR", "CPC_BIASCORR", "ENSEMBLE_BIASCORR")
RAIN_FIELDS = {"ALPHA", "RTOT", "PDW", "RNUM"}
EXPECTED_DAYS_9 = 3288


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest().upper()


def write_json(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {name}")
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_fit_module():
    spec = importlib.util.spec_from_file_location("frozen_yc_fit_038", FIT_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def group(field: str) -> str:
    if field in RAIN_FIELDS:
        return "DIRECT_RAIN"
    if field in {"NAMN", "NASD"}:
        return "TMIN"
    if field.startswith("S"):
        return "WET_DRY_SRAD"
    return "WET_DRY_TMAX"


def main() -> None:
    if (OUT / "final_gate.json").exists():
        raise FileExistsError("038 final gate already exists; refusing overwrite")
    gate = {
        "task": "038", "site": "HLA", "purpose": "2013_INCLUSION_WGEN_PARAMETER_SENSITIVITY",
        "status": "STARTED", "input_freeze_pass": False, "common_2004_2012_pass": False,
        "parameter_comparison_pass": False, "wgen_executed": False,
        "dssat_executed": False, "ppo_executed": False,
        "formal_rain_source_selected": False, "ready_for_formal_training": False,
    }
    try:
        fit = load_fit_module()
        with INV.open(newline="", encoding="utf-8-sig") as f:
            inventory = {r["scenario"]: r for r in csv.DictReader(f)}
        if set(inventory) != set(NAMES):
            raise ValueError(f"Expected five frozen scenarios, got {sorted(inventory)}")

        archive = {}
        first_nine = None
        manifest = {"task": "038", "site": "HLA", "inputs": [], "comparison_window": "2004-2012 versus 2004-2013", "scenario_role": "SENSITIVITY_ONLY"}
        for path in (INV, SOURCE_014, SANITY_014, SUMMARY_024, FIT_SCRIPT):
            manifest["inputs"].append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)})

        for name in NAMES:
            item = inventory[name]
            weather_path = ROOT / item["fitting_weather"]
            cli_path = ROOT / item["cli"]
            monthly_path = weather_path.parent / "monthly_wgen_statistics.csv"
            for p, expected in ((weather_path, item["fitting_sha256"]), (cli_path, item["cli_sha256"])):
                actual = sha(p)
                if actual != expected.upper():
                    raise ValueError(f"Frozen hash mismatch: {p.relative_to(ROOT)}")
            days, actual = fit.read_weather(weather_path, expected_sha256=item["fitting_sha256"])
            nine = [d for d in days if d.day.year <= 2012]
            if len(nine) != EXPECTED_DAYS_9 or nine[0].day != date(2004, 1, 1) or nine[-1].day != date(2012, 12, 31):
                raise ValueError(f"Invalid 2004-2012 window: {name}")
            if first_nine is None:
                first_nine = nine
            elif nine != first_nine:
                raise ValueError(f"2004-2012 differs between scenarios: {name}")
            ten_fit = fit.fit_monthly_statistics(days)
            with monthly_path.open(newline="", encoding="utf-8-sig") as f:
                frozen_fit = list(csv.DictReader(f))
            if len(frozen_fit) != 12:
                raise ValueError(f"Frozen monthly fit incomplete: {name}")
            for calc, frozen in zip(ten_fit, frozen_fit):
                if int(calc["month"]) != int(frozen["month"]):
                    raise ValueError(f"Month mismatch: {name}")
                for field in fit.WGEN_PARAMETER_FIELDS:
                    if not math.isclose(float(calc[field]), float(frozen[field]), rel_tol=1e-10, abs_tol=1e-10):
                        raise ValueError(f"Frozen fit mismatch: {name} month {calc['month']} {field}")
            archive[name] = {"ten": ten_fit, "nine": nine}
            manifest["inputs"].extend([
                {"path": weather_path.relative_to(ROOT).as_posix(), "sha256": actual},
                {"path": cli_path.relative_to(ROOT).as_posix(), "sha256": sha(cli_path)},
                {"path": monthly_path.relative_to(ROOT).as_posix(), "sha256": sha(monthly_path)},
            ])

        gate["input_freeze_pass"] = True
        gate["common_2004_2012_pass"] = True
        assert first_nine is not None
        nine_fit = fit.fit_monthly_statistics(first_nine)
        base_rows = []
        comparisons = []
        by_key = defaultdict(list)
        for base in nine_fit:
            month = int(base["month"])
            for field in fit.WGEN_PARAMETER_FIELDS:
                val9 = float(base[field])
                prec = fit.WGEN_PARAMETER_PRECISION[field]
                base_rows.append({"month": month, "parameter": field, "group": group(field), "value_full_precision": val9, "value_cli_precision": f"{val9:.{prec}f}"})
                for name in NAMES:
                    val10 = float(archive[name]["ten"][month - 1][field])
                    rounded9 = float(f"{val9:.{prec}f}")
                    rounded10 = float(f"{val10:.{prec}f}")
                    row = {
                        "scenario": name, "month": month, "parameter": field, "group": group(field),
                        "value_2004_2012": val9, "value_2004_2013": val10,
                        "delta_2013_minus_9yr": val10 - val9,
                        "abs_delta": abs(val10 - val9),
                        "relative_delta_to_9yr": (val10 - val9) / abs(val9) if abs(val9) > 1e-12 else "",
                        "cli_value_9yr": rounded9, "cli_value_10yr": rounded10,
                        "cli_delta": rounded10 - rounded9,
                    }
                    comparisons.append(row)
                    by_key[(month, field)].append(row)

        summary = []
        for (month, field), rows in sorted(by_key.items()):
            values10 = [float(r["value_2004_2013"]) for r in rows]
            deltas = [float(r["abs_delta"]) for r in rows]
            cli10 = [float(r["cli_value_10yr"]) for r in rows]
            cli9 = float(rows[0]["cli_value_9yr"])
            summary.append({
                "month": month, "parameter": field, "group": group(field),
                "max_abs_2013_inclusion_delta": max(deltas),
                "min_abs_2013_inclusion_delta": min(deltas),
                "five_scenario_span": max(values10) - min(values10),
                "max_abs_cli_2013_inclusion_delta": max(abs(v - cli9) for v in cli10),
                "five_scenario_cli_span": max(cli10) - min(cli10),
                "inclusion_to_scenario_span_ratio": max(deltas) / (max(values10) - min(values10)) if max(values10) - min(values10) > 1e-12 else "",
            })

        sanity = json.loads(SANITY_014.read_text(encoding="utf-8"))
        diagnostic = json.loads(SUMMARY_024.read_text(encoding="utf-8"))
        with SOURCE_014.open(newline="", encoding="utf-8-sig") as f:
            source_rows = list(csv.DictReader(f))
        selected_2013 = [r for r in source_rows if r["date"].startswith("2013-") and r["rain_status"] == "RESOLVED"]
        selected_sum = math.fsum(float(r["selected_rain_mm"]) for r in selected_2013)
        if len(selected_2013) != 254 or not math.isclose(selected_sum, 1149.8, abs_tol=1e-6):
            raise ValueError("014 selected 2013 rain subtotal changed")
        target = diagnostic["target_summary"]
        target_totals = [float(target[n]["target_rain_subtotal_mm"]) for n in NAMES]
        manifest["rain_2013"] = {
            "selected_days": len(selected_2013), "selected_subtotal_mm_not_full_year": selected_sum,
            "unresolved_days": 111, "gridded_fill_min_mm": min(target_totals),
            "gridded_fill_max_mm": max(target_totals),
            "scenario_2013_total_min_mm": diagnostic["annual_2013_total_min_mm"],
            "scenario_2013_total_max_mm": diagnostic["annual_2013_total_max_mm"],
            "prior_014_sanity_status": sanity["status"],
        }
        write_csv("common_2004_2012_wgen_parameters.csv", base_rows)
        write_csv("monthly_parameter_comparison.csv", comparisons)
        write_csv("parameter_impact_summary.csv", summary)
        write_json("input_manifest.json", manifest)
        gate.update({
            "parameter_comparison_pass": True, "status": "AUDIT_COMPLETE_FORMAL_SOURCE_UNRESOLVED",
            "nine_year_days": EXPECTED_DAYS_9, "scenarios": list(NAMES),
            "parameter_comparison_rows": len(comparisons),
            "recommended_next_step": "Review the measured 2013 inclusion deltas against five-scenario spans; prioritize adjudicating selected 2013 heavy-rain observations over further tweaking the 111-day gridded reference fills. Keep all candidate CLI tracks exploratory until rainfall provenance and weather QA gates are resolved.",
        })
    except Exception as exc:
        gate.update({"status": "FAILED_INPUT_OR_COMPUTATION_GATE", "error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        write_json("final_gate.json", gate)


if __name__ == "__main__":
    main()
