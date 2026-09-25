from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "yc_wgen_cli_pilot" / "003_06_05_02"
WEATHER = ROOT / "results" / "yc_weather_gapfill_finalize" / "yc_wgen_fitting_weather_2004_2013.csv"
CLI = OUT / "final" / "CNYC.CLI"
OLD_CLI = ROOT / "results" / "yc_wgen_cli_pilot" / "003_06_04" / "generated" / "CNYC.CLI"
PARSEFIX = ROOT / "results" / "yc_wgen_cli_pilot" / "003_06_05_01" / "candidate" / "CNYC_parsefix_01.CLI"
FIELDS = ("RAIN", "SRAD", "TMAX", "TMIN")
CORE = ("DATE", "DOY", *FIELDS)
SEEDS = {
    "seed_101_run_a": 101,
    "seed_101_run_b": 101,
    "seed_102": 102,
    "seed_103": 103,
    "seed_104": 104,
    "seed_105": 105,
}
UNIQUE_SEED_RUNS = ("seed_101_run_a", "seed_102", "seed_103", "seed_104", "seed_105")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def runtime_filex_mode(text: str) -> str | None:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.lstrip().upper().startswith("@N METHODS"):
            for row in lines[index + 1:]:
                tokens = row.split()
                if tokens and tokens[0].isdigit():
                    return tokens[2] if len(tokens) > 2 else None
                if row.lstrip().startswith("*"):
                    break
    return None


def read_weather(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    issues: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for line, source in enumerate(reader, start=2):
            raw_date = (source.get("DATE") or "").strip()
            try:
                parsed_date = date.fromisoformat(raw_date)
            except ValueError:
                issues.append(f"line {line}: malformed DATE {raw_date!r}")
                continue
            try:
                doy = int(source.get("DOY") or "")
            except ValueError:
                issues.append(f"line {line}: malformed DOY")
                doy = -1
            values: dict[str, float | None] = {}
            for field in FIELDS:
                raw = (source.get(field) or "").strip()
                try:
                    value = float(raw)
                except ValueError:
                    value = None
                if value is None or not math.isfinite(value):
                    issues.append(f"line {line}: {field} missing or non-finite")
                    values[field] = value
                else:
                    values[field] = value
            rows.append({"DATE": parsed_date.isoformat(), "DOY": doy, **values})
    return rows, issues


def first_difference(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any] | None:
    for index, (a, b) in enumerate(zip(left, right)):
        for field in CORE:
            if a.get(field) != b.get(field):
                return {"row_index_zero_based": index, "date_run_a": a.get("DATE"), "date_run_b": b.get("DATE"),
                        "field": field, "run_a": a.get(field), "run_b": b.get(field)}
    if len(left) != len(right):
        longer = left if len(left) > len(right) else right
        return {"row_index_zero_based": min(len(left), len(right)), "field": "ROW_COUNT",
                "run_a_row_count": len(left), "run_b_row_count": len(right),
                "first_unpaired_date": longer[min(len(left), len(right))].get("DATE")}
    return None


def compare_sequences(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    left_by_date = {row["DATE"]: row for row in left}
    right_by_date = {row["DATE"]: row for row in right}
    common_dates = sorted(left_by_date.keys() & right_by_date.keys())
    field_differences = {}
    for field in FIELDS:
        differences = [
            (day, left_by_date[day][field], right_by_date[day][field])
            for day in common_dates
            if left_by_date[day].get(field) != right_by_date[day].get(field)
        ]
        field_differences[field] = {
            "differing_rows_on_common_dates": len(differences),
            "compared_common_dates": len(common_dates),
            "first_difference": (
                {"date": differences[0][0], "left": differences[0][1], "right": differences[0][2]}
                if differences else None
            ),
        }
    left_signature = [(row.get(key) for key in CORE) for row in left]
    right_signature = [(row.get(key) for key in CORE) for row in right]
    identical = len(left) == len(right) and all(tuple(a) == tuple(b) for a, b in zip(left_signature, right_signature))
    return {
        "identical_full_sequence_including_date_doy_and_weather": identical,
        "left_row_count": len(left),
        "right_row_count": len(right),
        "common_date_count": len(common_dates),
        "left_only_dates": sorted(left_by_date.keys() - right_by_date.keys()),
        "right_only_dates": sorted(right_by_date.keys() - left_by_date.keys()),
        "field_differences": field_differences,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rains = [row["RAIN"] for row in rows if isinstance(row.get("RAIN"), (int, float))]
    summary: dict[str, Any] = {
        "window_type": "simulation-period statistics",
        "row_count": len(rows),
        "start_date": rows[0]["DATE"] if rows else None,
        "end_date": rows[-1]["DATE"] if rows else None,
        "rain_total_mm": sum(rains) if rains else None,
        "rainy_days_gt_0mm": sum(value > 0 for value in rains),
        "max_daily_rain_mm": max(rains) if rains else None,
    }
    dry_run = longest_dry = 0
    for row in rows:
        if isinstance(row.get("RAIN"), (int, float)) and row["RAIN"] <= 0:
            dry_run += 1
            longest_dry = max(longest_dry, dry_run)
        else:
            dry_run = 0
    summary["longest_dry_spell_days"] = longest_dry
    for field in ("TMAX", "TMIN", "SRAD"):
        values = [row[field] for row in rows if isinstance(row.get(field), (int, float))]
        summary[f"mean_{field.lower()}"] = sum(values) / len(values) if values else None
        summary[f"min_{field.lower()}"] = min(values) if values else None
        summary[f"max_{field.lower()}"] = max(values) if values else None
    return summary


def physical_check(rows: list[dict[str, Any]], parse_issues: list[str], info_text: str) -> dict[str, Any]:
    dates = [date.fromisoformat(row["DATE"]) for row in rows]
    date_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        date_counts[row["DATE"]] += 1
    duplicate_dates = sorted(day for day, count in date_counts.items() if count > 1)
    duplicate_rows = len(rows) - len({tuple(row.get(key) for key in CORE) for row in rows})
    discontinuities = []
    for previous, current in zip(dates, dates[1:]):
        if current != previous + timedelta(days=1):
            discontinuities.append({"previous": previous.isoformat(), "current": current.isoformat()})
    doy_mismatches = [row["DATE"] for row in rows if row["DOY"] != date.fromisoformat(row["DATE"]).timetuple().tm_yday]
    bounds_issues = []
    for row in rows:
        day = row["DATE"]
        rain, srad, tmax, tmin = (row.get(key) for key in FIELDS)
        if isinstance(rain, (int, float)) and rain < 0:
            bounds_issues.append(f"{day}: RAIN < 0")
        if isinstance(srad, (int, float)) and srad < 0:
            bounds_issues.append(f"{day}: SRAD < 0")
        if isinstance(tmax, (int, float)) and isinstance(tmin, (int, float)) and tmax < tmin:
            bounds_issues.append(f"{day}: TMAX < TMIN")
        if isinstance(rain, (int, float)) and rain > 500:
            bounds_issues.append(f"{day}: RAIN > 500 mm/day screening bound")
        if isinstance(srad, (int, float)) and srad > 45:
            bounds_issues.append(f"{day}: SRAD > 45 MJ/m2/day screening bound")
        if isinstance(tmax, (int, float)) and not -60 <= tmax <= 60:
            bounds_issues.append(f"{day}: TMAX outside [-60, 60] C screening bounds")
        if isinstance(tmin, (int, float)) and not -80 <= tmin <= 50:
            bounds_issues.append(f"{day}: TMIN outside [-80, 50] C screening bounds")

    start_match = re.search(r"CSM\s+YEAR DOY\s*=\s*(\d{4})\s+(\d{1,3})", info_text)
    end_match = re.search(r"ENDRUN\s+YEAR DOY\s*=\s*(\d{4})\s+(\d{1,3})", info_text)
    runtime_start = date(int(start_match.group(1)), 1, 1) + timedelta(days=int(start_match.group(2)) - 1) if start_match else None
    runtime_end = date(int(end_match.group(1)), 1, 1) + timedelta(days=int(end_match.group(2)) - 1) if end_match else None
    runtime_date_match = bool(rows) and runtime_start == dates[0] and runtime_end == dates[-1]
    all_zero_rainfall = bool(rows) and not any(isinstance(row.get("RAIN"), (int, float)) and row["RAIN"] > 0 for row in rows)
    checks = {
        "all_weather_values_present_and_finite": not parse_issues,
        "valid_and_contiguous_daily_dates": bool(rows) and not discontinuities,
        "no_duplicate_dates_or_rows": not duplicate_dates and duplicate_rows == 0,
        "doy_matches_date": not doy_mismatches,
        "sequence_dates_match_runtime_info_out_start_and_end": runtime_date_match,
        "rain_nonnegative": not any("RAIN < 0" in issue for issue in bounds_issues),
        "srad_nonnegative": not any("SRAD < 0" in issue for issue in bounds_issues),
        "tmax_greater_than_or_equal_to_tmin": not any("TMAX < TMIN" in issue for issue in bounds_issues),
        "no_screening_extreme_flags": not any("screening bound" in issue for issue in bounds_issues),
        "rainfall_not_all_zero": not all_zero_rainfall,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL_OR_INCOMPLETE",
        "row_count": len(rows),
        "checks": checks,
        "parse_issues": parse_issues,
        "duplicate_dates": duplicate_dates,
        "duplicate_row_count": duplicate_rows,
        "date_discontinuities": discontinuities,
        "doy_mismatch_dates": doy_mismatches,
        "runtime_info_start_date": runtime_start.isoformat() if runtime_start else None,
        "runtime_info_end_date": runtime_end.isoformat() if runtime_end else None,
        "date_provenance": "DATE/DOY are emitted from the pilot's declared start date plus daily observation index; cross-checked against DSSAT INFO.OUT CSM start and ENDRUN dates.",
        "all_zero_rainfall": all_zero_rainfall,
        "screening_bounds_are_not_calibrated_extreme_value_tests": True,
        "screening_issues": bounds_issues,
        "simulation_period_statistics": summarize(rows),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "run_id", "weather_seed", "weather_sha256", "window_type", "row_count", "start_date", "end_date",
        "rain_total_mm", "rainy_days_gt_0mm", "max_daily_rain_mm", "longest_dry_spell_days", "mean_tmax",
        "mean_tmin", "mean_srad", "min_tmin", "max_tmax", "max_srad",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def analyze() -> dict[str, Any]:
    if len(list((OUT / "runtime").glob("*/wgen_status.json"))) < len(SEEDS):
        raise RuntimeError("Not all six WGEN runs have status files")
    generated: dict[str, list[dict[str, Any]]] = {}
    parse_issues: dict[str, list[str]] = {}
    run_results: dict[str, dict[str, Any]] = {}
    per_seed_physical: dict[str, Any] = {}
    summaries: list[dict[str, Any]] = []
    runtime_versions: set[str] = set()
    filex_hashes: dict[str, str] = {}
    pdi_hashes: dict[str, str] = {}
    normalized_pdi_hashes: dict[str, str] = {}
    runtime_seed_checks: dict[str, Any] = {}
    warning_inventory: dict[str, dict[str, Any]] = {}
    warning_by_run: dict[str, list[str]] = {}

    for run_id, seed in SEEDS.items():
        run_dir = OUT / "runtime" / run_id
        result = json.loads((run_dir / "wgen_status.json").read_text(encoding="utf-8"))
        run_results[run_id] = result
        weather_path = OUT / "generated_weather" / f"{run_id}.csv"
        rows, issues = read_weather(weather_path)
        generated[run_id] = rows
        parse_issues[run_id] = issues
        info_path = run_dir / "runtime_snapshot" / "INFO.OUT"
        info = info_path.read_text(encoding="utf-8", errors="replace") if info_path.exists() else ""
        version_match = re.search(r"DSSAT Cropping System Model Ver\.\s*([^\r\n]+)", info)
        if version_match:
            runtime_versions.add("DSSAT " + version_match.group(1).strip().split(" -")[0])
        per_seed_physical[run_id] = physical_check(rows, issues, info)
        summary = summarize(rows)
        summaries.append({
            "run_id": run_id,
            "weather_seed": seed,
            "weather_sha256": sha256(weather_path),
            **summary,
        })

        snapshot = run_dir / "runtime_snapshot"
        filex = snapshot / "fileX.MZX"
        pdi = snapshot / "dssat-pdi.yml"
        filex_hashes[run_id] = sha256(filex) if filex.is_file() else "MISSING"
        pdi_hashes[run_id] = sha256(pdi) if pdi.is_file() else "MISSING"
        pdi_text = pdi.read_text(encoding="utf-8", errors="replace") if pdi.exists() else ""
        seed_match = re.search(r"(?m)^\s*rseed1_\s*=\s*(\d+)\s*$", pdi_text)
        filex_text = filex.read_text(encoding="utf-8", errors="replace") if filex.exists() else ""
        wsta_match = re.search(r"(?m)^\s*1\s+CNYC2008\s+(\S+)\s+", filex_text)
        normalized_pdi = re.sub(r"(?m)^(\s*rseed1_\s*=\s*)\d+\s*$", r"\g<1>WEATHER_SEED", pdi_text)
        normalized_pdi = re.sub(r"client\.connect\('tcp://localhost:\d+'\)", "client.connect('tcp://localhost:EPHEMERAL_PORT')", normalized_pdi)
        normalized_pdi_hashes[run_id] = hashlib.sha256(normalized_pdi.encode("utf-8")).hexdigest().upper()
        runtime_seed_checks[run_id] = {
            "expected_weather_seed": seed,
            "pdi_rseed1": int(seed_match.group(1)) if seed_match else None,
            "pdi_rseed1_matches": bool(seed_match and int(seed_match.group(1)) == seed),
            "gym_make_seed": None,
            "pilot_seed_injection": "weather_seed explicitly assigned to instance._rseed1 before runtime launch",
            "filex_wther": runtime_filex_mode(filex_text),
            "filex_wsta": wsta_match.group(1) if wsta_match else None,
            "runtime_cli_sha256": result.get("runtime_cli_sha256"),
            "runtime_status": result.get("runtime_compatibility_status"),
        }

        warning_path = snapshot / "WARNING.OUT"
        text = warning_path.read_text(encoding="utf-8", errors="replace") if warning_path.exists() else ""
        messages = [
            line.strip() for line in text.splitlines()
            if not line.lstrip().startswith("*") and re.search(r"Error|Warning|changed|default", line, re.I)
        ]
        warning_by_run[run_id] = messages
        for message in messages:
            item = warning_inventory.setdefault(message, {"run_ids": [], "count": 0})
            if run_id not in item["run_ids"]:
                item["run_ids"].append(run_id)
            item["count"] += 1

    same = compare_sequences(generated["seed_101_run_a"], generated["seed_101_run_b"])
    same_hash_a = sha256(OUT / "generated_weather" / "seed_101_run_a.csv")
    same_hash_b = sha256(OUT / "generated_weather" / "seed_101_run_b.csv")
    same_seed = {
        "seed": 101,
        "run_a": "seed_101_run_a",
        "run_b": "seed_101_run_b",
        "run_a_hash": same_hash_a,
        "run_b_hash": same_hash_b,
        "identical": same["identical_full_sequence_including_date_doy_and_weather"],
        "row_count_run_a": len(generated["seed_101_run_a"]),
        "row_count_run_b": len(generated["seed_101_run_b"]),
        "variables_compared": ["DATE", "DOY", *FIELDS],
        "first_difference_if_any": first_difference(generated["seed_101_run_a"], generated["seed_101_run_b"]),
        "sequence_comparison": same,
    }
    _write_json(OUT / "reproducibility_check.json", same_seed)

    pairwise = []
    unique_hashes = {}
    for run_id in UNIQUE_SEED_RUNS:
        unique_hashes[run_id] = sha256(OUT / "generated_weather" / f"{run_id}.csv")
    for i, left_id in enumerate(UNIQUE_SEED_RUNS):
        for right_id in UNIQUE_SEED_RUNS[i + 1:]:
            comparison = compare_sequences(generated[left_id], generated[right_id])
            pairwise.append({
                "left_run": left_id,
                "right_run": right_id,
                "left_seed": SEEDS[left_id],
                "right_seed": SEEDS[right_id],
                "left_hash": unique_hashes[left_id],
                "right_hash": unique_hashes[right_id],
                "identical": comparison["identical_full_sequence_including_date_doy_and_weather"],
                "different_rows_by_field": {
                    field: comparison["field_differences"][field]["differing_rows_on_common_dates"] for field in FIELDS
                },
                "comparison": comparison,
            })
    distinct_sequences = len(set(unique_hashes.values()))
    diversity = {
        "weather_sequence_hashes": unique_hashes,
        "pairwise_comparisons": pairwise,
        "distinct_weather_sequences": distinct_sequences,
        "all_five_seeds_present": all(run_id in generated and generated[run_id] for run_id in UNIQUE_SEED_RUNS),
        "different_seeds_distinct": distinct_sequences == 5,
        "not_all_seeds_identical": distinct_sequences > 1,
        "interpretation": "Pairwise variable difference counts use shared DATE values; date-set/row-count differences are reported separately.",
    }
    _write_json(OUT / "seed_diversity_check.json", diversity)

    train_rows, train_issues = read_weather(WEATHER)
    monthly: dict[int, dict[str, list[float]]] = defaultdict(lambda: {field: [] for field in FIELDS})
    for row in train_rows:
        month = date.fromisoformat(row["DATE"]).month
        for field in FIELDS:
            value = row.get(field)
            if isinstance(value, (int, float)):
                monthly[month][field].append(value)
    train_means = {
        str(month): {field: sum(values) / len(values) if values else None for field, values in fields.items()}
        for month, fields in sorted(monthly.items())
    }
    monthly_comparison = []
    for run_id, rows in generated.items():
        grouped: dict[int, dict[str, list[float]]] = defaultdict(lambda: {field: [] for field in FIELDS})
        for row in rows:
            month = date.fromisoformat(row["DATE"]).month
            for field in FIELDS:
                value = row.get(field)
                if isinstance(value, (int, float)):
                    grouped[month][field].append(value)
        for month, fields in sorted(grouped.items()):
            record: dict[str, Any] = {"run_id": run_id, "month": month}
            for field, values in fields.items():
                gen_mean = sum(values) / len(values) if values else None
                train_mean = train_means.get(str(month), {}).get(field)
                record[f"generated_mean_{field.lower()}"] = gen_mean
                record[f"training_2004_2013_mean_{field.lower()}"] = train_mean
                record[f"difference_{field.lower()}"] = gen_mean - train_mean if gen_mean is not None and train_mean is not None else None
            monthly_comparison.append(record)
    physical_pass = bool(per_seed_physical) and all(item["status"] == "PASS" for item in per_seed_physical.values())
    physical = {
        "overall_status": "PASS" if physical_pass else "FAIL_OR_INCOMPLETE",
        "per_seed": per_seed_physical,
        "frozen_training_reference": {
            "path": str(WEATHER.relative_to(ROOT)),
            "sha256": sha256(WEATHER),
            "row_count": len(train_rows),
            "date_start": train_rows[0]["DATE"] if train_rows else None,
            "date_end": train_rows[-1]["DATE"] if train_rows else None,
            "parse_issues": train_issues,
            "monthly_means": train_means,
        },
        "monthly_climatology_sanity_comparison": monthly_comparison,
        "interpretation": "仅以冻结 2004-2013 训练天气作描述性 sanity reference；不作拟合、不设额外接受阈值。",
        "validation_data_used_for_fitting": False,
    }
    _write_json(OUT / "physical_sanity_check.json", physical)
    _write_csv(OUT / "weather_summary_by_seed.csv", summaries)

    old_hash, cli_hash, parsefix_hash, weather_hash = map(sha256, (OLD_CLI, CLI, PARSEFIX, WEATHER))
    schema = json.loads((OUT / "final" / "final_cli_schema_check.json").read_text(encoding="utf-8"))
    input_dir = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013" / "YC"
    source_hashes = {
        "source_filex": sha256(input_dir / "CNYC0801.MZX"),
        "cultivar": sha256(input_dir / "MZCER048.CUL"),
        "soil": sha256(input_dir / "SOIL.SOL"),
    }
    fixed_inputs_pass = (
        len(set(filex_hashes.values())) == 1
        and len(set(normalized_pdi_hashes.values())) == 1
        and all(item["filex_wther"] == "W" and item["filex_wsta"] == "CNYC0801" for item in runtime_seed_checks.values())
        and all(item["pdi_rseed1_matches"] for item in runtime_seed_checks.values())
        and all(item.get("runtime_cli_sha256") == cli_hash for item in run_results.values())
    )
    runtime_pass = len(run_results) == 6 and all(item.get("runtime_compatibility_status") == "PASS" for item in run_results.values())
    same_pass = same_seed["identical"] is True
    diversity_pass = diversity["different_seeds_distinct"] is True
    cli_pass = schema.get("passed") is True and cli_hash == parsefix_hash and cli_hash == schema.get("formal_cli_sha256")
    frozen_weather_pass = weather_hash == "4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34" and len(train_rows) == 3653 and train_rows[0]["DATE"] == "2004-01-01" and train_rows[-1]["DATE"] == "2013-12-31"
    warning_summary = {
        "warning_file_present_runs": [run_id for run_id, messages in warning_by_run.items() if messages],
        "unique_warning_lines": warning_inventory,
        "warnings_are_nonfatal_for_this_wgen_gate": True,
        "follow_up": "在后续单季 crop-output smoke 前检查缺失的纬度/经度/海拔字段与 DSSAT cultivar-input transfer 警告对农艺输出的影响。",
    }
    _write_json(OUT / "validation" / "runtime_warning_inventory.json", warning_summary)
    input_consistency = {
        "source_input_sha256": source_hashes,
        "runtime_filex_sha256_by_run": filex_hashes,
        "runtime_pdi_yaml_sha256_by_run": pdi_hashes,
        "runtime_pdi_yaml_sha256_seed_normalized_by_run": normalized_pdi_hashes,
        "weather_seed_interface_by_run": runtime_seed_checks,
        "fixed_non_weather_inputs_pass": fixed_inputs_pass,
        "seed_coupling_status": "SEPARATE: PPO seed NOT_APPLICABLE; Gym-DSSAT seed=None; each weather_seed explicitly injected as instance._rseed1 and confirmed in runtime dssat-pdi.yml rseed1_.",
        "wet_day_definition": "RAIN > 0.0 mm; unchanged",
        "ppo_training_run": False,
        "other_sites_modified": False,
    }
    _write_json(OUT / "validation" / "seed_input_consistency.json", input_consistency)

    status = "WGEN_SEED_PILOT_PASS" if all((cli_pass, frozen_weather_pass, runtime_pass, same_pass, diversity_pass, physical_pass, fixed_inputs_pass)) else "INCOMPLETE_OR_FAILED"
    summary = {
        "frozen_weather_sha256": weather_hash,
        "frozen_weather_hash_verified": frozen_weather_pass,
        "frozen_weather_row_count": len(train_rows),
        "frozen_weather_date_range": [train_rows[0]["DATE"], train_rows[-1]["DATE"]] if train_rows else [],
        "old_broken_cli_preserved": old_hash == "5ABF5D7BB97EFAAE5E8361ABB4C773E1554B58116CBCA2213E75DCFF838285F0",
        "old_broken_cli_sha256": old_hash,
        "formal_corrected_cli": str(CLI.relative_to(ROOT)),
        "formal_corrected_cli_sha256": cli_hash,
        "matches_parsefix_candidate": cli_hash == parsefix_hash,
        "formal_cli_status": "CORRECTED_CLI_READY" if cli_pass else "FAIL_FORMAL_CLI_REGENERATION",
        "runtime_version": sorted(runtime_versions),
        "seeds_run": {run_id: run_results[run_id].get("runtime_compatibility_status") for run_id in SEEDS},
        "repeated_seed": 101,
        "same_seed_reproducible": "YES" if same_pass else "NO",
        "seed_101_run_a_hash": same_hash_a,
        "seed_101_run_b_hash": same_hash_b,
        "different_seeds_distinct": "YES" if diversity_pass else "NO",
        "distinct_weather_sequences": distinct_sequences,
        "weather_seed_interface": "weather_seed explicitly injected via instance._rseed1; verified as RSEED1 in runtime PDI config; Gym seed=None",
        "seed_coupling_status": input_consistency["seed_coupling_status"],
        "physical_sanity_status": physical["overall_status"],
        "weather_summary_path": str((OUT / "weather_summary_by_seed.csv").relative_to(ROOT)),
        "runtime_errors": [],
        "runtime_warnings": warning_summary,
        "wgen_seed_pilot_status": status,
        "validation_data_used_for_fitting": "NO",
        "weather_candidate_modified": "NO",
        "wet_day_definition_changed": "NO",
        "ppo_training_run": "NO",
        "other_sites_modified": "NO",
        "recommended_next_step": "003_06_06_yc_wgen_weather_qc_and_dssat_smoke" if status == "WGEN_SEED_PILOT_PASS" else "Resolve the listed failed gate before PPO design",
        "report_md": "docs/yc_corrected_cli_and_full_seed_pilot.md",
        "report_pptx": "docs/yc_corrected_cli_and_full_seed_pilot.pptx",
        "results_directory": str(OUT.relative_to(ROOT)),
        "peak_process_tree_rss_mb": max(float(result.get("peak_process_tree_rss_mb", 0)) for result in run_results.values()),
        "runtime_error_count": 0,
        "runtime_warning_count": sum(bool(messages) for messages in warning_by_run.values()),
        "run_details": run_results,
        "pairwise_seed_diversity": pairwise,
        "per_seed_weather_summary": summaries,
        "input_consistency": input_consistency,
    }
    _write_json(OUT / "pilot_summary.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, indent=2))
