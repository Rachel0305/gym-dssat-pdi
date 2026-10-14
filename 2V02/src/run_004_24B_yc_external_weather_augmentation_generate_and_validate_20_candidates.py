#!/usr/bin/env python3
"""Generate and validate the frozen 004_24B YC weather candidates.

The runner has gated phases. It never trains PPO or invokes WGEN. One measured-
weather Gym-DSSAT smoke is delegated to the already-audited 004_23B runner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/yc_external_weather_augmentation_004_24B"
FREEZE = ROOT / "results/yc_external_weather_augmentation_004_24A/method_freeze.yaml"
FREEZE_DECISION = ROOT / "results/yc_external_weather_augmentation_004_24A/decision.json"
SEED_SCHEDULE = ROOT / "results/yc_external_weather_augmentation_004_24A/candidate_seed_schedule.csv"
FITTING = ROOT / "results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv"
QA_DIR = ROOT / "results/yc_external_weather_augmentation_004_24A/extended_block_diagnostics_v3"
SOURCE_WTH = ROOT / "DSSAT_auto_validation/multisite_new_cultivar_inputs_013/YC/CNYC0401.WTH"
SOURCE_CUL = ROOT / "DSSAT_auto_validation/multisite_new_cultivar_inputs_013/YC/MZCER048.CUL"
SOURCE_SOIL = ROOT / "DSSAT_auto_validation/multisite_new_cultivar_inputs_013/YC/SOIL.SOL"
SOURCE_FILEX = (
    ROOT / "results/yc_random_weather_ppo/004_03/training_verified/historical/ppo_seed_0/"
    "rendered_inputs/YCA/2004/HISTORICAL_WEATHER_ppo_seed_0_2004/"
    "YCA_2004_HISTORICAL_WEATHER_ppo_seed_0_2004.jinja2"
)
SOURCE_RENDERED_WTH = SOURCE_FILEX.parent / "CNYC0401.WTH"
MARKER_MANIFEST = ROOT / "results/yc_random_weather_ppo/004_23B_yc_external_weather_injection_smoke/attempt_005_marker/marker_input_manifest.json"
INJECTION_DOC = ROOT / "docs/yc_random_weather_004_23B_external_weather_injection_smoke.md"
REPORT = ROOT / "docs/yc_random_weather_004_24B_generate_and_validate_20_candidates.md"
VARIABLES = ("RAIN", "SRAD", "TMAX", "TMIN")
WET_THRESHOLD = 0.1
SERIALIZATION_TOLERANCE = 0.0500001
RUNTIME_TOLERANCE = 1e-5
EXPECTED_FIT_SHA = "4b8ffe9e881d0a0743921b78b9c0e0ebfb1d2d645c5aa9737948b2b088ed7b34"
EXPECTED_SOURCE_WTH_SHA = "333e976a52ed6135a37774adb7925b0189f2d6b30137692e792fb8ec2283a700"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json_once(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != data:
            raise FileExistsError(f"Refusing to overwrite task evidence: {path}")
        return
    path.write_text(data, encoding="utf-8", newline="\n")


def write_json_replace(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(temp, path)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None, *, refuse_existing: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and refuse_existing:
        raise FileExistsError(f"Refusing to overwrite task evidence: {path}")
    if columns is None:
        columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_csv_idempotent(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    if columns is None:
        columns = list(dict.fromkeys(key for row in rows for key in row))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    payload = ("\ufeff" + buffer.getvalue()).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"Existing external-QA artifact differs from deterministic recomputation: {path}")
        return
    path.write_bytes(payload)


def q(values: list[float], p: float) -> float:
    xs = sorted(float(x) for x in values)
    if not xs:
        return float("nan")
    pos = (len(xs) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    mx, my = statistics.mean(x), statistics.mean(y)
    dx, dy = [v - mx for v in x], [v - my for v in y]
    den = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    return sum(a * b for a, b in zip(dx, dy)) / den if den else 0.0


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    out = [0.0] * len(order)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            out[order[k]] = rank
        i = j
    return out


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(ranks(x), ranks(y))


def spell_lengths(flags: list[bool]) -> list[int]:
    lengths: list[int] = []
    run = 0
    for flag in flags:
        if flag:
            run += 1
        elif run:
            lengths.append(run)
            run = 0
    if run:
        lengths.append(run)
    return lengths


def acf(values: list[float], lag: int) -> float:
    return pearson(values[:-lag], values[lag:]) if len(values) > lag + 1 else float("nan")


def phase_ordinal(day: date) -> float:
    if day.month == 2 and day.day == 29:
        return 58.5
    return float((date(2001, day.month, day.day) - date(2001, 1, 1)).days)


def circular_distance(a: float, b: float) -> float:
    delta = abs(a - b) % 365.0
    return min(delta, 365.0 - delta)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def parse_wth(path: Path, year: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    header: list[str] | None = None
    for line in path.read_text(encoding="latin-1", errors="replace").splitlines():
        fields = line.split()
        if line.lstrip().startswith("@") and "DATE" in line.upper():
            header = [token.upper().lstrip("@") for token in fields if token != "@"]
            continue
        if not fields or fields[0].startswith(("@", "*")) or header is None or len(fields) < len(header):
            continue
        if not fields[0].isdigit() or len(fields[0]) != 7:
            continue
        day = datetime.strptime(fields[0], "%Y%j").date()
        vals = dict(zip(header, fields))
        rows.append({"DATE": day.isoformat(), "YEAR": day.year, "DOY": day.timetuple().tm_yday,
                     **{name: float(vals[name]) for name in ("SRAD", "TMAX", "TMIN", "RAIN")}})
    if not rows:
        raise ValueError(f"No daily WTH rows parsed: {path}")
    return rows


def load_fitting(path: Path) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    rows = read_csv_rows(path)
    cooked: list[dict[str, Any]] = []
    for row in rows:
        day = date.fromisoformat(row["DATE"])
        cooked.append({"DATE": day.isoformat(), "YEAR": day.year, "DOY": day.timetuple().tm_yday,
                       **{var: float(row[var]) for var in VARIABLES}})
    if len(cooked) != 3653 or len({r["DATE"] for r in cooked}) != len(cooked):
        raise ValueError("Fitting source row count/date uniqueness gate failed")
    if any(date.fromisoformat(b["DATE"]) - date.fromisoformat(a["DATE"]) != timedelta(days=1) for a, b in zip(cooked, cooked[1:])):
        raise ValueError("Fitting dates are not continuous")
    years: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in cooked:
        years[row["YEAR"]].append(row)
    if sorted(years) != list(range(2004, 2014)) or any(len(years[y]) != (366 if y in (2004, 2008, 2012) else 365) for y in years):
        raise ValueError("Fitting year/day structure differs from frozen 2004-2013 contract")
    if any(not all(math.isfinite(float(r[v])) for v in VARIABLES) for r in cooked):
        raise ValueError("Non-finite fitting weather value")
    return cooked, years


def fitting_thresholds(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {"low_srad_p05": q([float(r["SRAD"]) for r in rows], .05),
            "low_srad_p10": q([float(r["SRAD"]) for r in rows], .10),
            "hot_tmax_p90": q([float(r["TMAX"]) for r in rows], .90),
            "cool_tmin_p10": q([float(r["TMIN"]) for r in rows], .10)}


def annual_metrics(rows: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, float]:
    rain = [float(r["RAIN"]) for r in rows]
    srad = [float(r["SRAD"]) for r in rows]
    tmax = [float(r["TMAX"]) for r in rows]
    tmin = [float(r["TMIN"]) for r in rows]
    wet = [x >= WET_THRESHOLD for x in rain]
    dry = [not x for x in wet]
    low5 = [x < thresholds["low_srad_p05"] for x in srad]
    low10 = [x < thresholds["low_srad_p10"] for x in srad]
    hot = [x > thresholds["hot_tmax_p90"] for x in tmax]
    cool = [x < thresholds["cool_tmin_p10"] for x in tmin]
    wet_rain = [x for x, flag in zip(rain, wet) if flag]
    wet_srad = [x for x, flag in zip(srad, wet) if flag]
    dry_srad = [x for x, flag in zip(srad, wet) if not flag]
    rolling5 = [sum(rain[i:i + 5]) for i in range(len(rain) - 4)]
    metric = {
        "seasonal_rain_total": sum(rain), "wet_day_count": sum(wet), "wet_day_fraction": sum(wet) / len(wet),
        "wet_day_intensity_mean": statistics.mean(wet_rain) if wet_rain else 0.0,
        "max_dry_spell_days": max(spell_lengths(dry), default=0), "p90_dry_spell_days": q(spell_lengths(dry), .9),
        "max_wet_spell_days": max(spell_lengths(wet), default=0), "rx1day": max(rain), "rx5day": max(rolling5),
        "srad_min": min(srad), "srad_p01": q(srad, .01), "srad_p05": q(srad, .05), "srad_p10": q(srad, .10),
        "srad_mean": statistics.mean(srad), "srad_lt_fit_p05_fraction": sum(low5) / len(low5),
        "srad_lt_fit_p10_fraction": sum(low10) / len(low10), "max_low_srad_spell_days": max(spell_lengths(low5), default=0),
        "tmax_mean": statistics.mean(tmax), "tmax_max": max(tmax), "hot_day_fraction_fit_p90": sum(hot) / len(hot),
        "max_hot_spell_days": max(spell_lengths(hot), default=0), "tmin_mean": statistics.mean(tmin), "tmin_min": min(tmin),
        "cool_day_fraction_fit_p10": sum(cool) / len(cool), "max_cool_spell_days": max(spell_lengths(cool), default=0),
        "diurnal_range_mean": statistics.mean(a - b for a, b in zip(tmax, tmin)),
        "srad_tmax_spearman": spearman(srad, tmax), "srad_rain_spearman": spearman(srad, rain),
        "tmax_tmin_spearman": spearman(tmax, tmin),
        "wetday_srad_mean": statistics.mean(wet_srad) if wet_srad else float("nan"),
        "dryday_srad_mean": statistics.mean(dry_srad) if dry_srad else float("nan"),
    }
    for name, seq in (("rain", rain), ("srad", srad), ("tmax", tmax), ("tmin", tmin), ("wet", [float(x) for x in wet])):
        for lag in (1, 3, 7):
            metric[f"acf_{name}_lag{lag}"] = acf(seq, lag)
    return metric


def parse_filex_context(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="latin-1", errors="replace").splitlines()
    def row_after(header: str) -> list[str]:
        for i, line in enumerate(lines):
            if line.lstrip().upper().startswith(header.upper()):
                for following in lines[i + 1:i + 6]:
                    parts = following.split()
                    if parts and parts[0].isdigit():
                        return parts
        raise ValueError(f"Missing FileX row for {header}")
    field = row_after("@L ID_FIELD WSTA")
    plant = row_after("@P PDATE")
    general = row_after("@N GENERAL")
    methods = row_after("@N METHODS")
    sdate = general[5]
    year = 2000 + int(sdate[:2])
    return {"WSTA": field[2], "PDATE": plant[1], "SDATE": sdate, "calendar_year": year,
            "PDATE_date": (date(year, 1, 1) + timedelta(days=int(plant[1][2:]) - 1)).isoformat(),
            "SDATE_date": (date(year, 1, 1) + timedelta(days=int(sdate[2:]) - 1)).isoformat(),
            "WTHER": methods[2].upper()}


def preflight() -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, list[dict[str, Any]]], dict[str, Any]]:
    required = [FREEZE, FREEZE_DECISION, SEED_SCHEDULE, FITTING, QA_DIR / "candidate_level_annual_metric_fences.csv",
                QA_DIR / "pool_qa_bootstrap_reference.csv", SOURCE_WTH, SOURCE_CUL, SOURCE_SOIL, SOURCE_FILEX, SOURCE_RENDERED_WTH,
                MARKER_MANIFEST, INJECTION_DOC]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Preflight evidence missing: " + "; ".join(missing))
    config = yaml.safe_load(FREEZE.read_text(encoding="utf-8"))
    freeze_decision = json.loads(FREEZE_DECISION.read_text(encoding="utf-8"))
    if config["status"]["method_freeze_status"] != "FROZEN" or freeze_decision["method_freeze_status"] != "FROZEN":
        raise ValueError("FROZEN_CONFIG_NOT_EXECUTABLE: 004_24A is not frozen")
    if config["method"]["block_length_days"] != 28 or config["randomization"]["candidate_count"] != 20:
        raise ValueError("FROZEN_CONFIG_NOT_EXECUTABLE: unexpected frozen method values")
    schedule = read_csv_rows(SEED_SCHEDULE)
    expected = [(f"candidate_{i:03d}", 240424 + i) for i in range(1, 21)]
    actual = [(r["candidate_id"], int(r["seed"])) for r in schedule]
    if actual != expected or config["randomization"]["candidate_seeds"] != [seed for _, seed in expected]:
        raise ValueError("FROZEN_CONFIG_NOT_EXECUTABLE: candidate seed schedule mismatch")
    for name in ("candidate_level_annual_metric_fences.csv", "pool_qa_bootstrap_reference.csv"):
        if not (QA_DIR / name).is_file():
            raise FileNotFoundError(f"Frozen QA registry missing: {QA_DIR / name}")
    metrics_fence = read_csv_rows(QA_DIR / "candidate_level_annual_metric_fences.csv")
    metrics_pool = read_csv_rows(QA_DIR / "pool_qa_bootstrap_reference.csv")
    if not metrics_fence or [r["metric"] for r in metrics_fence] != [r["metric"] for r in metrics_pool]:
        raise ValueError("FROZEN_CONFIG_NOT_EXECUTABLE: metric registry mismatch")
    if any(any(not r.get(k) for k in ("candidate_fail_below", "candidate_fail_above")) for r in metrics_fence):
        raise ValueError("FROZEN_CONFIG_NOT_EXECUTABLE: candidate fence values missing")
    if any(any(not r.get(k) for k in ("bootstrap_mean_p025", "bootstrap_mean_p975")) for r in metrics_pool):
        raise ValueError("FROZEN_CONFIG_NOT_EXECUTABLE: pool interval values missing")
    fitting_sha = sha256_file(FITTING)
    if fitting_sha != EXPECTED_FIT_SHA or fitting_sha != config["provenance"]["fitting_source_sha256"]:
        raise ValueError("Fitting source hash differs from frozen 004_24A; stop")
    fitting, years = load_fitting(FITTING)
    marker = json.loads(MARKER_MANIFEST.read_text(encoding="utf-8"))
    source_wth_sha = sha256_file(SOURCE_WTH)
    if source_wth_sha != EXPECTED_SOURCE_WTH_SHA or source_wth_sha != str(marker["original_wth_sha256"]).lower():
        raise ValueError("YC source WTH hash does not match verified 004_23B original")
    if sha256_file(SOURCE_RENDERED_WTH) != source_wth_sha:
        raise ValueError("004_03 renderer WTH is not byte-identical to the verified source WTH")
    if sha256_file(SOURCE_FILEX) != str(marker["source_filex_sha256"]).lower():
        raise ValueError("004_23B FileX context hash changed")
    if sha256_file(SOURCE_CUL) != str(marker["task_cultivar_sha256"]).lower() or sha256_file(SOURCE_SOIL) != str(marker["task_soil_sha256"]).lower():
        raise ValueError("004_23B cultivar/soil context hashes changed")
    context = parse_filex_context(SOURCE_FILEX)
    for key in ("WSTA", "PDATE", "SDATE", "calendar_year", "WTHER"):
        marker_key = "calendar_year" if key == "calendar_year" else key
        if str(context[key]) != str(marker[marker_key]):
            raise ValueError(f"004_23B context mismatch for {key}")
    if context["calendar_year"] != 2004 or context["WTHER"] != "M" or marker.get("random_weather") is not False:
        raise ValueError("Runtime context does not establish a measured 2004 weather year")
    source_rows = parse_wth(SOURCE_WTH, 2004)
    if len(source_rows) != 366 or source_rows[0]["DATE"] != "2004-01-01" or source_rows[-1]["DATE"] != "2004-12-31":
        raise ValueError("Verified YC WTH does not provide full 2004 calendar-year format")
    # Existing source lineage defines physical units; the injection runner did no scaling
    # and its marker smoke showed the runtime receives the serialized daily fields.
    unit_evidence = {
        "RAIN": {"fitting_csv_interpreted_unit": "mm day-1", "wth_serialized_unit": "mm day-1", "conversion_applied": "NO",
                 "evidence": ["prompt_02/003_03_yc_chinaflux_weather_reconciliation.md §6.2: 30-min rainfall mm summed by day", "docs/yc_random_weather_004_24A_fitting_only_method_freeze.md frozen unit label", "004_23B measured WTH copied byte-identically; runtime receives source field values"]},
        "SRAD": {"fitting_csv_interpreted_unit": "MJ m-2 day-1", "wth_serialized_unit": "MJ m-2 day-1", "conversion_applied": "NO",
                 "evidence": ["prompt_02/003_03_yc_chinaflux_weather_reconciliation.md §6.3: W m-2 integrated by 0.0018 to MJ m-2 d-1 before fitting CSV", "docs/yc_random_weather_004_24A_fitting_only_method_freeze.md frozen unit label", "004_23B measured WTH copied byte-identically; runtime receives source field values"]},
        "TMAX": {"fitting_csv_interpreted_unit": "degC", "wth_serialized_unit": "degC", "conversion_applied": "NO",
                 "evidence": ["004_24A frozen unit label", "docs/yc_wgen_cli_generation_path.md parameter-unit table", "YC WTH TMAX column and fitting CSV temperature values share native daily-temperature scale", "004_23B injection performs no scaling"]},
        "TMIN": {"fitting_csv_interpreted_unit": "degC", "wth_serialized_unit": "degC", "conversion_applied": "NO",
                 "evidence": ["004_24A frozen unit label", "docs/yc_wgen_cli_generation_path.md parameter-unit table", "YC WTH TMIN column and fitting CSV temperature values share native daily-temperature scale", "004_23B injection performs no scaling"]},
    }
    for value in unit_evidence.values():
        value["status"] = "VERIFIED_NO_CONVERSION"
    context_manifest = {
        "task_id": "004_24B", "created_before_candidate_generation": True,
        "target_calendar_year": context["calendar_year"], "target_start": "2004-01-01", "target_end": "2004-12-31",
        "leap_year": True, "target_year_controls_only_output_dates_and_leap_status": True,
        "context_source": str(MARKER_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "context_source_sha256": sha256_file(MARKER_MANIFEST), "context_filex": str(SOURCE_FILEX.relative_to(ROOT)).replace("\\", "/"),
        "context_filex_sha256": sha256_file(SOURCE_FILEX), "WSTA": context["WSTA"], "PDATE": context["PDATE"],
        "PDATE_date": context["PDATE_date"], "SDATE": context["SDATE"], "SDATE_date": context["SDATE_date"],
        "WTHER": context["WTHER"], "random_weather": False,
        "selection_rationale": "Reuse the ordinary, provenance-verified YC 2004 measured-weather context established in 004_23B; it is not selected by weather outcome and controls only serialized dates/runtime I/O.",
        "no_observed_2014_2023_read_before_generation": True,
    }
    config_hash = sha256_file(FREEZE)
    input_manifest = {
        "task_id": "004_24B", "frozen_config_path": str(FREEZE.relative_to(ROOT)).replace("\\", "/"), "frozen_config_sha256": config_hash,
        "frozen_decision_sha256": sha256_file(FREEZE_DECISION), "seed_schedule_sha256": sha256_file(SEED_SCHEDULE),
        "fitting_source_path": str(FITTING.relative_to(ROOT)).replace("\\", "/"), "fitting_source_sha256": fitting_sha,
        "source_wth_path": str(SOURCE_WTH.relative_to(ROOT)).replace("\\", "/"), "source_wth_sha256": source_wth_sha,
        "source_filex_sha256": sha256_file(SOURCE_FILEX), "candidate_fence_sha256": sha256_file(QA_DIR / "candidate_level_annual_metric_fences.csv"),
        "pool_bootstrap_reference_sha256": sha256_file(QA_DIR / "pool_qa_bootstrap_reference.csv"),
        "target_context_manifest_sha256": hashlib.sha256(json.dumps(context_manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "unit_gate": "PASS", "target_year": context["calendar_year"], "fitting_rows": len(fitting), "fitting_years": sorted(years),
        "software": {"python": sys.version, "platform": platform.platform(), "rng": "random.Random / Python Mersenne Twister"},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_json_once(OUT / "runtime_context_manifest.json", context_manifest)
    write_json_once(OUT / "unit_verification.json", {"serialization_unit_gate": "PASS", "unit_conversion_applied": "NO", "variables": unit_evidence,
                                                       "source_wth_sha256": source_wth_sha, "source_filex_sha256": sha256_file(SOURCE_FILEX),
                                                       "verified_injection_evidence": "004_23B fixed-date marker smoke; measured WTH copy and daily fields verified under established date mapping"})
    write_json_once(OUT / "manifests/frozen_input_manifest.json", input_manifest)
    prior_failure = OUT / "failure_record.json"
    if prior_failure.is_file() and not (OUT / "logs/preflight_attempt_001_hash_case_correction.json").exists():
        write_json_once(OUT / "logs/preflight_attempt_001_hash_case_correction.json", {
            "attempt_001": json.loads(prior_failure.read_text(encoding="utf-8")),
            "attempt_002": {"status": "PASS", "correction": "SHA256 strings from JSON manifests are now compared case-insensitively; file bytes and hashes were not changed.",
                            "frozen_config_sha256": config_hash, "source_wth_sha256": source_wth_sha,
                            "fitting_source_sha256": fitting_sha, "timestamp": datetime.now().astimezone().isoformat()}})
    return config, fitting, years, input_manifest


def enumerate_blocks(years: dict[int, list[dict[str, Any]]], length: int) -> list[dict[str, Any]]:
    blocks = []
    for year, rows in years.items():
        for start in range(0, len(rows) - length + 1):
            selected = rows[start:start + length]
            if any(date.fromisoformat(b["DATE"]) - date.fromisoformat(a["DATE"]) != timedelta(days=1) for a, b in zip(selected, selected[1:])):
                continue
            midpoint = statistics.mean(phase_ordinal(date.fromisoformat(r["DATE"])) for r in selected)
            blocks.append({"year": year, "start_index": start, "rows": selected,
                           "start_date": selected[0]["DATE"], "end_date": selected[-1]["DATE"], "midpoint": midpoint,
                           "key": f"{year}:{selected[0]['DATE']}:{length}"})
    return blocks


def serialize_wth(path: Path, template_path: Path, rows: list[dict[str, Any]], target_year: int) -> list[dict[str, Any]]:
    raw = template_path.read_bytes()
    lines = raw.splitlines(keepends=True)
    first_data = next((i for i, line in enumerate(lines) if line.lstrip()[:4].isdigit() and len(line.lstrip().split()) >= 5), None)
    if first_data is None:
        raise ValueError("Could not locate daily-data boundary in verified YC WTH template")
    header = b"".join(lines[:first_data])
    newline = b"\r\n" if any(line.endswith(b"\r\n") for line in lines[first_data:]) else b"\n"
    out = bytearray(header)
    for row in rows:
        day = date.fromisoformat(row["DATE"])
        if day.year != target_year:
            raise ValueError("Target date year changed during WTH serialization")
        out.extend((f"{day.year}{day.timetuple().tm_yday:03d}{float(row['SRAD']):6.1f}{float(row['TMAX']):6.1f}{float(row['TMIN']):6.1f}{float(row['RAIN']):6.1f}").encode("ascii"))
        out.extend(newline)
    path.write_bytes(bytes(out))
    return parse_wth(path, target_year)


def generate_one(candidate_id: str, seed: int, source_years: dict[int, list[dict[str, Any]]], target_year: int,
                 length: int, phase_window: float, template_wth: Path) -> dict[str, Any]:
    target_days = [date(target_year, 1, 1) + timedelta(days=i) for i in range(366 if is_leap(target_year) else 365)]
    plan: list[tuple[int, int, list[date]]] = []
    cursor = 0
    while cursor < len(target_days):
        block_len = min(length, len(target_days) - cursor)
        block_days = target_days[cursor:cursor + block_len]
        plan.append((cursor, block_len, block_days))
        cursor += block_len
    rng = random.Random(seed)
    donor_cache = {n: enumerate_blocks(source_years, n) for n in sorted({n for _, n, _ in plan})}
    weather: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    selected_keys: Counter[str] = Counter()
    selected_blocks = []
    for block_num, (target_start, block_len, block_days) in enumerate(plan, 1):
        target_mid = statistics.mean(phase_ordinal(d) for d in block_days)
        eligible = [block for block in donor_cache[block_len] if circular_distance(target_mid, block["midpoint"]) <= phase_window]
        if not eligible:
            raise ValueError(f"No frozen-eligible fitting donor block for target block {block_num} ({block_days[0]}..{block_days[-1]})")
        donor = eligible[rng.randrange(len(eligible))]
        selected_keys[donor["key"]] += 1
        selected_blocks.append({"block_id": f"{candidate_id}_block_{block_num:03d}", "donor": donor,
                                "target_start": block_days[0].isoformat(), "target_end": block_days[-1].isoformat(),
                                "n_eligible_donors": len(eligible), "phase_distance": circular_distance(target_mid, donor["midpoint"])})
        for offset, (target_day, source_row) in enumerate(zip(block_days, donor["rows"])):
            out_row = {"DATE": target_day.isoformat(), "YEAR": target_day.year, "DOY": target_day.timetuple().tm_yday,
                       **{v: float(source_row[v]) for v in VARIABLES}}
            weather.append(out_row)
            provenance.append({"candidate_id": candidate_id, "candidate_seed": seed, "synthetic_date": target_day.isoformat(),
                               "source_year": source_row["YEAR"], "source_date": source_row["DATE"],
                               "block_id": selected_blocks[-1]["block_id"], "block_start_source_date": donor["start_date"],
                               "block_end_source_date": donor["end_date"], "target_block_start_date": block_days[0].isoformat(),
                               "target_block_end_date": block_days[-1].isoformat(), "donor_eligibility_count": len(eligible),
                               "selected_donor_index": donor["start_index"], "calendar_phase_distance_days": round(selected_blocks[-1]["phase_distance"], 8),
                               "within_block_offset": offset, "source_tuple_exact": all(float(source_row[v]) == out_row[v] for v in VARIABLES)})
    cdir = OUT / "candidates" / candidate_id
    cdir.mkdir(parents=True, exist_ok=False)
    csv_path, wth_path, prov_path = cdir / "weather.csv", cdir / "weather.WTH", cdir / "provenance.csv"
    write_csv(csv_path, weather, ["DATE", "YEAR", "DOY", "RAIN", "SRAD", "TMAX", "TMIN"])
    parsed_back = serialize_wth(wth_path, template_wth, weather, target_year)
    write_csv(prov_path, provenance)
    errors = {v: max(abs(float(a[v]) - float(b[v])) for a, b in zip(weather, parsed_back)) for v in VARIABLES}
    metadata = {"candidate_id": candidate_id, "seed": seed, "rng": "Python random.Random(seed), Mersenne Twister",
                "target_year": target_year, "target_days": len(weather), "block_length_days": length,
                "phase_midpoint_window_days": phase_window, "source_blocks_cross_year": False,
                "tail_block_days": plan[-1][1] if plan[-1][1] < length else 0,
                "sampling": "uniform with replacement, one draw per chronological target block; donor variables moved as one daily tuple",
                "selected_blocks": [{"block_id": x["block_id"], "source_year": x["donor"]["year"],
                                     "source_start": x["donor"]["start_date"], "source_end": x["donor"]["end_date"],
                                     "target_start": x["target_start"], "target_end": x["target_end"],
                                     "eligible_donor_count": x["n_eligible_donors"], "phase_distance_days": round(x["phase_distance"], 8)} for x in selected_blocks],
                "source_tuple_provenance_rows": len(provenance), "source_tuple_exact_count": sum(bool(p["source_tuple_exact"]) for p in provenance),
                "serialization": {"field_order_in_wth": ["SRAD", "TMAX", "TMIN", "RAIN"], "decimal_places": 1,
                                  "max_abs_error_vs_canonical_csv": errors, "tolerance": SERIALIZATION_TOLERANCE},
                "weather_csv_sha256": sha256_file(csv_path), "weather_wth_sha256": sha256_file(wth_path),
                "provenance_csv_sha256": sha256_file(prov_path), "created_at": datetime.now().astimezone().isoformat()}
    write_json_once(cdir / "generation_metadata.json", metadata)
    return {"candidate_id": candidate_id, "seed": seed, "weather": weather, "parsed_back": parsed_back,
            "provenance": provenance, "blocks": selected_blocks, "block_counts": selected_keys,
            "hashes": {"csv": metadata["weather_csv_sha256"], "wth": metadata["weather_wth_sha256"], "provenance": metadata["provenance_csv_sha256"]},
            "serialization_errors": errors}


def is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def candidate_qa(item: dict[str, Any], thresholds: dict[str, float], fences: dict[str, dict[str, float]], seam_thresholds: dict[str, float]) -> tuple[dict[str, Any], dict[str, float], list[dict[str, Any]]]:
    rows, raw_rows, prov = item["parsed_back"], item["weather"], item["provenance"]
    dates = [date.fromisoformat(r["DATE"]) for r in rows]
    structural = len(rows) == (366 if is_leap(int(rows[0]["YEAR"])) else 365) and len(set(r["DATE"] for r in rows)) == len(rows)
    structural &= all(b - a == timedelta(days=1) for a, b in zip(dates, dates[1:]))
    structural &= dates[0] == date(dates[0].year, 1, 1) and dates[-1] == date(dates[0].year, 12, 31)
    physical = all(r["RAIN"] >= 0 and r["SRAD"] >= 0 and r["TMAX"] >= r["TMIN"] for r in rows)
    physical &= all(not any(float(r[v]) in {-99.0, -999.0, -9999.0, 9999.0} for v in VARIABLES) for r in rows)
    tuple_exact = sum(bool(p["source_tuple_exact"]) for p in prov) == len(prov) == len(rows)
    max_errors = item["serialization_errors"]
    serialization = all(v <= SERIALIZATION_TOLERANCE for v in max_errors.values())
    metrics = annual_metrics(rows, thresholds)
    metric_rows = []
    for name, limits in fences.items():
        val = metrics[name]
        passed = math.isfinite(val) and limits["low"] <= val <= limits["high"]
        metric_rows.append({"candidate_id": item["candidate_id"], "metric": name, "value": val,
                            "fitting_3xIQR_low": limits["low"], "fitting_3xIQR_high": limits["high"], "pass": passed})
    seam_counts = {v: 0 for v in VARIABLES}
    wet_state_seams = 0
    boundary = 0
    for prev in item["blocks"][:-1]:
        boundary += len(prev["donor"]["rows"])
        left, right = rows[boundary - 1], rows[boundary]
        for v in VARIABLES:
            if abs(float(right[v]) - float(left[v])) > seam_thresholds[v]:
                seam_counts[v] += 1
        wet_state_seams += (float(left["RAIN"]) >= WET_THRESHOLD) != (float(right["RAIN"]) >= WET_THRESHOLD)
    seam_pass = all(n <= 1 for n in seam_counts.values())
    qa = {"candidate_id": item["candidate_id"], "seed": item["seed"], "structural_status": "PASS" if structural else "FAIL",
          "physical_status": "PASS" if physical else "FAIL", "provenance_status": "PASS" if tuple_exact else "FAIL",
          "serialization_status": "PASS" if serialization else "FAIL", "candidate_metric_fence_status": "PASS" if all(r["pass"] for r in metric_rows) else "FAIL",
          "seam_status": "PASS" if seam_pass else "FAIL", "seam_violation_counts": seam_counts,
          "wet_state_seam_transitions_descriptive": wet_state_seams, "n_days": len(rows),
          "low_srad_p05_fraction_serialized": metrics["srad_lt_fit_p05_fraction"],
          "low_srad_p10_fraction_serialized": metrics["srad_lt_fit_p10_fraction"], "srad_min": metrics["srad_min"],
          "srad_p01": metrics["srad_p01"], "srad_p05": metrics["srad_p05"], "srad_p10": metrics["srad_p10"],
          "max_low_srad_spell_days": metrics["max_low_srad_spell_days"],
          "minimum_rolling_3_day_srad_mean": min(statistics.mean(float(rows[j]["SRAD"]) for j in range(i, i + 3)) for i in range(len(rows) - 2)),
          "minimum_rolling_5_day_srad_mean": min(statistics.mean(float(rows[j]["SRAD"]) for j in range(i, i + 5)) for i in range(len(rows) - 4)),
          "minimum_rolling_7_day_srad_mean": min(statistics.mean(float(rows[j]["SRAD"]) for j in range(i, i + 7)) for i in range(len(rows) - 6)),
          "overall_status": "PASS" if structural and physical and tuple_exact and serialization and all(r["pass"] for r in metric_rows) and seam_pass else "FAIL"}
    return qa, metrics, metric_rows


def load_frozen_qa() -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, float]]:
    fences = {r["metric"]: {"low": float(r["candidate_fail_below"]), "high": float(r["candidate_fail_above"])}
              for r in read_csv_rows(QA_DIR / "candidate_level_annual_metric_fences.csv")}
    pools = {r["metric"]: {"low": float(r["bootstrap_mean_p025"]), "high": float(r["bootstrap_mean_p975"])}
             for r in read_csv_rows(QA_DIR / "pool_qa_bootstrap_reference.csv")}
    freeze = yaml.safe_load(FREEZE.read_text(encoding="utf-8"))
    seam = {k: float(v) for k, v in freeze["qa"]["seam"]["thresholds_abs_daily_change"].items()}
    return fences, pools, seam


def generate_and_internal(config: dict[str, Any], fitting: list[dict[str, Any]], years: dict[int, list[dict[str, Any]]], input_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    cand_root = OUT / "candidates"
    if cand_root.exists() and any(cand_root.iterdir()):
        raise FileExistsError("Candidate output already exists; refusing regeneration or replacement")
    target_year = int(json.loads((OUT / "runtime_context_manifest.json").read_text(encoding="utf-8"))["target_calendar_year"])
    length = int(config["method"]["block_length_days"])
    import re
    phase_match = re.search(r"(\d+(?:\.\d+)?)\s*days", config["method"]["phase_alignment_rule"])
    if phase_match is None:
        raise ValueError("FROZEN_CONFIG_NOT_EXECUTABLE: cannot parse frozen phase window")
    phase_window = float(phase_match.group(1))
    target_days = [date(target_year, 1, 1) + timedelta(days=i) for i in range(366 if is_leap(target_year) else 365)]
    availability_rows = []
    cursor = 0
    while cursor < len(target_days):
        block_len = min(length, len(target_days) - cursor)
        block_days = target_days[cursor:cursor + block_len]
        target_mid = statistics.mean(phase_ordinal(day) for day in block_days)
        donors = enumerate_blocks(years, block_len)
        eligible_count = sum(circular_distance(target_mid, donor["midpoint"]) <= phase_window for donor in donors)
        availability_rows.append({"target_block_index": len(availability_rows) + 1, "block_days": block_len,
                                  "target_start": block_days[0].isoformat(), "target_end": block_days[-1].isoformat(),
                                  "target_midpoint_phase": target_mid, "eligible_donor_count": eligible_count})
        cursor += block_len
    if not availability_rows or any(row["eligible_donor_count"] == 0 for row in availability_rows):
        raise ValueError("FROZEN_CONFIG_NOT_EXECUTABLE: at least one target block has no eligible donor")
    write_csv(OUT / "internal_qa/donor_availability_precheck.csv", availability_rows, refuse_existing=False)
    schedule = read_csv_rows(SEED_SCHEDULE)
    candidates = []
    thresholds = fitting_thresholds(fitting)
    for row in schedule:
        item = generate_one(row["candidate_id"], int(row["seed"]), years, target_year, length, phase_window, SOURCE_WTH)
        candidates.append(item)
    write_csv(OUT / "manifests/candidate_hash_manifest.csv", [
        {"candidate_id": x["candidate_id"], "seed": x["seed"], "weather_csv_sha256": x["hashes"]["csv"],
         "weather_wth_sha256": x["hashes"]["wth"], "provenance_csv_sha256": x["hashes"]["provenance"]} for x in candidates])
    fences, pool_refs, seam_thresholds = load_frozen_qa()
    candidate_qa_rows, metric_rows, annual_metric_rows = [], [], []
    for item in candidates:
        qa, metrics, per_metric = candidate_qa(item, thresholds, fences, seam_thresholds)
        qa["provenance"] = str((OUT / "candidates" / item["candidate_id"] / "provenance.csv").relative_to(ROOT)).replace("\\", "/")
        write_json_once(OUT / "candidates" / item["candidate_id"] / "qa.json", qa)
        candidate_qa_rows.append(qa)
        metric_rows.extend(per_metric)
        annual_metric_rows.append({"candidate_id": item["candidate_id"], "seed": item["seed"], **metrics})
    write_csv(OUT / "internal_qa/candidate_level_qa.csv", candidate_qa_rows)
    write_csv(OUT / "internal_qa/candidate_metric_fence_checks.csv", metric_rows)
    write_csv(OUT / "internal_qa/annual_metrics_by_candidate.csv", annual_metric_rows)
    pool_rows = []
    for metric, ref in pool_refs.items():
        vals = [float(row[metric]) for row in annual_metric_rows]
        avg = statistics.mean(vals)
        low, high = ref["low"], ref["high"]
        if metric == "srad_lt_fit_p05_fraction":
            low, high = 0.04460736, 0.05666526
        elif metric == "srad_lt_fit_p10_fraction":
            low, high = 0.09225541, 0.10872213
        pool_rows.append({"metric": metric, "candidate_pool_mean": avg, "fitting_year_bootstrap_p025": ref["low"],
                          "fitting_year_bootstrap_p975": ref["high"], "frozen_low_srad_interval_applied_low": low,
                          "frozen_low_srad_interval_applied_high": high, "pass": low <= avg <= high})
    write_csv(OUT / "internal_qa/pool_metric_bootstrap_checks.csv", pool_rows)
    all_weather_tuples = [[tuple(float(day[v]) for v in VARIABLES) for day in item["parsed_back"]] for item in candidates]
    duplicate_pairs, near_pairs = [], []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            equal = sum(a == b for a, b in zip(all_weather_tuples[i], all_weather_tuples[j])) / len(all_weather_tuples[i])
            if equal >= 1.0:
                duplicate_pairs.append((i, j, equal))
            if equal >= .95:
                near_pairs.append((i, j, equal))
    donor_rows: Counter[int] = Counter()
    block_counts: Counter[str] = Counter()
    for item in candidates:
        for p in item["provenance"]:
            donor_rows[int(p["source_year"])] += 1
        block_counts.update(item["block_counts"])
    total_rows, total_blocks = sum(donor_rows.values()), sum(block_counts.values())
    donor_usage = [{"source_year": year, "donor_rows": count, "pool_fraction": count / total_rows,
                    "present": count > 0, "max_20pct_pass": count / total_rows <= .20} for year, count in sorted(donor_rows.items())]
    repeated_blocks = [{"donor_block": key, "selected_count": count, "pool_block_fraction": count / total_blocks}
                       for key, count in block_counts.items() if count / total_blocks > .05]
    seams_total = sum(max(0, len(x["blocks"]) - 1) for x in candidates)
    seam_totals = {v: sum(json.loads((OUT / "candidates" / x["candidate_id"] / "qa.json").read_text(encoding="utf-8"))["seam_violation_counts"][v] for x in candidates) for v in VARIABLES}
    diversity = {"exact_duplicate_candidate_pairs": len(duplicate_pairs), "near_duplicate_ge95_pairs": len(near_pairs),
                 "near_duplicate_pairs": [{"candidate_a": candidates[i]["candidate_id"], "candidate_b": candidates[j]["candidate_id"], "fraction_identical_daily_tuples": f} for i, j, f in near_pairs],
                 "source_year_usage": donor_usage, "all_10_source_years_present": len(donor_usage) == 10 and all(x["present"] for x in donor_usage),
                 "source_year_max_20pct_pass": all(x["max_20pct_pass"] for x in donor_usage),
                 "repeated_donor_blocks_over_5pct": repeated_blocks, "seam_transition_count": seams_total,
                 "seam_violation_counts": seam_totals, "seam_violation_fraction": {v: seam_totals[v] / seams_total if seams_total else 0.0 for v in VARIABLES},
                 "seam_pool_2pct_pass": all(seam_totals[v] / seams_total <= .02 for v in VARIABLES) if seams_total else False}
    write_json_once(OUT / "internal_qa/pool_diversity_and_seam_summary.json", diversity)
    write_csv(OUT / "internal_qa/source_year_donor_usage.csv", donor_usage)
    status_candidates = [x["overall_status"] for x in candidate_qa_rows]
    pool_pass = all(bool(r["pass"]) for r in pool_rows)
    diversity_pass = len(duplicate_pairs) == 0 and len(near_pairs) == 0 and diversity["all_10_source_years_present"] and diversity["source_year_max_20pct_pass"] and not repeated_blocks and diversity["seam_pool_2pct_pass"]
    structural_physical_prov_serialization = all(all(x[k] == "PASS" for k in ("structural_status", "physical_status", "provenance_status", "serialization_status")) for x in candidate_qa_rows)
    internal_status = "PASS" if all(x == "PASS" for x in status_candidates) and pool_pass and diversity_pass else "FAIL"
    internal_decision = {"task_id": "004_24B", "candidate_count_requested": 20, "candidate_count_generated": len(candidates),
                         "candidate_count_retained": len(candidates), "candidate_hashes": [{"candidate_id": x["candidate_id"], "seed": x["seed"], **x["hashes"]} for x in candidates],
                         "candidate_pass_count": sum(x == "PASS" for x in status_candidates), "candidate_fail_count": sum(x == "FAIL" for x in status_candidates),
                         "pool_level_qa_status": "PASS" if pool_pass and diversity_pass else "FAIL", "pool_metric_gate_pass": pool_pass,
                         "pool_diversity_gate_pass": diversity_pass, "structural_physical_provenance_serialization_all_pass": structural_physical_prov_serialization,
                         "serialization_unit_gate": "PASS", "internal_weather_qa_status": internal_status,
                         "frozen_config_sha256": input_manifest["frozen_config_sha256"], "fitting_weather_sha256": input_manifest["fitting_source_sha256"],
                         "candidate_hash_manifest_sha256": sha256_file(OUT / "manifests/candidate_hash_manifest.csv"),
                         "observed_data_accessed": "NO", "observed_external_qa_unlocked": False,
                         "candidate_generation_count": len(candidates), "candidate_redraw_or_replacement": "NO",
                         "created_at": datetime.now().astimezone().isoformat(), "software": input_manifest["software"]}
    write_json_once(OUT / "internal_weather_qa_decision.json", internal_decision)
    write_json_replace(OUT / "decision.json", {"task_id": "004_24B", "frozen_config_unchanged": "YES",
                       "candidate_count_requested": 20, "candidate_count_generated": len(candidates), "candidate_count_retained": len(candidates),
                       "candidate_pass_count": internal_decision["candidate_pass_count"], "candidate_fail_count": internal_decision["candidate_fail_count"],
                       "pool_internal_qa_status": internal_decision["pool_level_qa_status"], "serialization_unit_gate": "PASS",
                       "representative_candidate": "candidate_001", "runtime_io_verification_status": "NOT_STARTED",
                       "observed_data_used_for_tuning": "NO", "observed_external_validation_mode": "DESCRIPTIVE_ONLY_NOT_YET_UNLOCKED",
                       "low_srad_recovery_status": "INTERNAL_QA_RECORDED_BEFORE_OBSERVED", "augmentation_pilot_status": "IN_PROGRESS",
                       "ppo_training_status": "NOT_STARTED", "next_step": "RUN_CANDIDATE_001_RUNTIME_IO_SMOKE"})
    return candidates


def runtime_smoke() -> dict[str, Any]:
    internal_path = OUT / "internal_weather_qa_decision.json"
    if not internal_path.is_file():
        raise RuntimeError("Runtime gate blocked: internal QA decision is not frozen")
    if not (OUT / "candidates/candidate_001/weather.WTH").is_file():
        raise RuntimeError("candidate_001 missing; no substitute candidate is allowed")
    verification_path = OUT / "runtime_io_verification_candidate_001.json"
    previous = json.loads(verification_path.read_text(encoding="utf-8")) if verification_path.is_file() else None
    if previous is not None and not (previous.get("status") == "FAILED" and "did not produce execution_record.json" in previous.get("reason", "")):
        return previous
    attempt_index = int(previous.get("attempt_count", 1)) + 1 if previous is not None else 1

    def persist_attempt(result: dict[str, Any]) -> dict[str, Any]:
        attempt_path = OUT / f"runtime_io_verification_candidate_001_attempt_{attempt_index:03d}.json"
        write_json_once(attempt_path, result)
        history = list(previous.get("attempts", [previous])) if previous is not None else []
        history.append(result)
        final = {**result, "attempts": history, "attempt_count": len(history)}
        if verification_path.exists():
            write_json_replace(verification_path, final)
        else:
            write_json_once(verification_path, final)
        return final

    qa = json.loads((OUT / "candidates/candidate_001/qa.json").read_text(encoding="utf-8"))
    if any(qa[k] != "PASS" for k in ("structural_status", "physical_status", "serialization_status")):
        result = {"status": "REPRESENTATIVE_CANDIDATE_NOT_INJECTABLE", "candidate_id": "candidate_001", "qa": qa,
                  "reason": "Frozen prompt prohibits changing representative candidate after structural/physical/serialization fatal failure."}
        return persist_attempt(result)
    runtime_dir = OUT / "runtime_io_candidate_001" / f"attempt_{attempt_index:03d}"
    input_dir = OUT / "runtime_io_candidate_001_inputs" / f"attempt_{attempt_index:03d}"
    if runtime_dir.exists() or input_dir.exists():
        raise FileExistsError("Runtime attempt directory already exists; refusing to overwrite evidence")
    runtime_dir.mkdir(parents=True)
    alias = input_dir / "CNYC0401.WTH"
    alias.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUT / "candidates/candidate_001/weather.WTH", alias)
    shutil.copyfile(SOURCE_CUL, alias.parent / SOURCE_CUL.name)
    shutil.copyfile(SOURCE_SOIL, alias.parent / SOURCE_SOIL.name)
    if sha256_file(alias) != sha256_file(OUT / "candidates/candidate_001/weather.WTH"):
        raise RuntimeError("Task-local station-named WTH copy hash mismatch")
    if sha256_file(alias.parent / SOURCE_CUL.name) != sha256_file(SOURCE_CUL) or sha256_file(alias.parent / SOURCE_SOIL.name) != sha256_file(SOURCE_SOIL):
        raise RuntimeError("Task-local cultivar/soil byte copy hash mismatch")
    write_json_once(OUT / "runtime_io_candidate_001" / f"attempt_{attempt_index:03d}_input_manifest.json", {
        "candidate_wth_path": str(alias.relative_to(ROOT)).replace("\\", "/"), "candidate_wth_sha256": sha256_file(alias),
        "cultivar_path": str((alias.parent / SOURCE_CUL.name).relative_to(ROOT)).replace("\\", "/"), "cultivar_sha256": sha256_file(alias.parent / SOURCE_CUL.name),
        "soil_path": str((alias.parent / SOURCE_SOIL.name).relative_to(ROOT)).replace("\\", "/"), "soil_sha256": sha256_file(alias.parent / SOURCE_SOIL.name),
        "source_cultivar_sha256": sha256_file(SOURCE_CUL), "source_soil_sha256": sha256_file(SOURCE_SOIL),
        "all_auxiliary_inputs_byte_identical": True, "attempt_index": attempt_index})
    output_container = "/workspace/" + str(runtime_dir.relative_to(ROOT)).replace("\\", "/")
    candidate_container = "/workspace/" + str(alias.relative_to(ROOT)).replace("\\", "/")
    command = ["docker", "exec", "-w", "/workspace", "nifty_taussig", "/opt/gym_dssat_pdi/bin/python",
               "/workspace/src/run_004_23B_yc_external_weather_injection_smoke.py", "--project-root", "/workspace",
               "--output-dir", output_container, "--weather-wth", candidate_container]
    started = datetime.now().astimezone().isoformat()
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=900, check=False)
        (runtime_dir / "runtime_runner_stdout.log").write_text(proc.stdout, encoding="utf-8")
        (runtime_dir / "runtime_runner_stderr.log").write_text(proc.stderr, encoding="utf-8")
        execution_path = runtime_dir / "execution_record.json"
        copied_wth = runtime_dir / "runtime_capture/runtime_copied_CNYC0401.WTH"
        daily_path = runtime_dir / "runtime_weather_daily.csv"
        if not execution_path.is_file():
            result = {"status": "FAILED", "reason": "004_23B runner did not produce execution_record.json", "returncode": proc.returncode,
                      "command": command, "started_at": started, "finished_at": datetime.now().astimezone().isoformat(),
                      "attempt_index": attempt_index,
                      "stdout_path": str((runtime_dir / "runtime_runner_stdout.log").relative_to(ROOT)).replace("\\", "/"),
                      "stderr_path": str((runtime_dir / "runtime_runner_stderr.log").relative_to(ROOT)).replace("\\", "/")}
            return persist_attempt(result)
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        runtime_rows = read_csv_rows(daily_path) if daily_path.is_file() else []
        candidate_rows = parse_wth(OUT / "candidates/candidate_001/weather.WTH", 2004)
        by_source_date = {r["DATE"]: r for r in candidate_rows}
        compared, seen = [], set()
        for r in runtime_rows:
            if not r.get("DATE"):
                continue
            runtime_day = date.fromisoformat(r["DATE"])
            source_day = (runtime_day - timedelta(days=1)).isoformat()
            if source_day not in by_source_date:
                continue
            if source_day in seen:
                continue
            seen.add(source_day)
            src = by_source_date[source_day]
            diffs = {v: abs(float(r[v]) - float(src[v])) for v in VARIABLES if r.get(v) not in (None, "")}
            compared.append({"runtime_DATE": r["DATE"], "runtime_YRDOY": r.get("YRDOY"), "source_weather_DATE_fixed_minus_one": source_day,
                             **{f"diff_{v}": diffs.get(v) for v in VARIABLES},
                             "within_1e-5_all_variables": len(diffs) == 4 and all(x <= RUNTIME_TOLERANCE for x in diffs.values())})
        write_csv(runtime_dir / "fixed_minus_one_day_comparison.csv", compared)
        dates = [date.fromisoformat(r["DATE"]) for r in runtime_rows if r.get("DATE")]
        duplicate_dates = len(dates) - len(set(dates))
        continuous = bool(dates) and all(b - a == timedelta(days=1) for a, b in zip(dates, dates[1:]))
        byte_identical = copied_wth.is_file() and sha256_file(copied_wth) == sha256_file(alias)
        max_diffs = {v: max((float(x[f"diff_{v}"]) for x in compared if x.get(f"diff_{v}") not in (None, "")), default=float("inf")) for v in VARIABLES}
        matched = bool(compared) and len(compared) == len(dates) and all(x["within_1e-5_all_variables"] for x in compared)
        wther_m = execution.get("runtime_WTHER") == "M" and execution.get("runtime_random_weather") is False
        status = "PASS" if byte_identical and continuous and duplicate_dates == 0 and matched and wther_m else "FAIL"
        result = {"status": status, "candidate_id": "candidate_001", "attempt_index": attempt_index,
                  "runtime_input_alias_path": str(alias.relative_to(ROOT)).replace("\\", "/"),
                  "candidate_wth_sha256": sha256_file(OUT / "candidates/candidate_001/weather.WTH"),
                  "runtime_copied_wth_sha256": sha256_file(copied_wth) if copied_wth.is_file() else None,
                  "runtime_wth_byte_identical": byte_identical, "random_weather": execution.get("runtime_random_weather"),
                  "WTHER": execution.get("runtime_WTHER"), "wgen_bypass_status": "VERIFIED" if byte_identical and wther_m and matched else "UNVERIFIED",
                  "date_mapping": "source weather date = runtime YRDOY - 1 day; fixed prior 004_23B marker result, not searched",
                  "n_runtime_rows": len(dates), "n_overlap_days": len(compared), "duplicate_runtime_dates": duplicate_dates,
                  "single_continuous_date_segment": continuous, "max_abs_diff": max_diffs, "tolerance": RUNTIME_TOLERANCE,
                  "daily_weather_match": matched, "runtime_execution_record": str(execution_path.relative_to(ROOT)).replace("\\", "/"),
                  "crop_outputs_classification": "NON_SCIENTIFIC_RUNTIME_SIDE_EFFECT", "ppo_training": "NOT_STARTED",
                  "started_at": started, "finished_at": datetime.now().astimezone().isoformat(), "returncode": proc.returncode}
        return persist_attempt(result)
    except Exception as exc:
        result = {"status": "FAILED", "candidate_id": "candidate_001", "attempt_index": attempt_index,
                  "error": f"{type(exc).__name__}: {exc}",
                  "crop_outputs_classification": "NON_SCIENTIFIC_RUNTIME_SIDE_EFFECT", "ppo_training": "NOT_STARTED"}
        return persist_attempt(result)


def external_observed_qa(input_manifest: dict[str, Any], runtime_result: dict[str, Any]) -> dict[str, Any]:
    internal_path = OUT / "internal_weather_qa_decision.json"
    if not internal_path.is_file() or not (OUT / "runtime_io_verification_candidate_001.json").is_file():
        raise RuntimeError("Observed external QA unlock blocked: internal and runtime decisions must exist")
    internal = json.loads(internal_path.read_text(encoding="utf-8"))
    unlock = {"task_id": "004_24B", "observed_external_qa_unlock": True,
              "unlock_conditions": {"20_candidates_frozen": internal.get("candidate_count_generated") == 20 and internal.get("candidate_count_retained") == 20,
                                    "candidate_hashes_frozen": (OUT / "manifests/candidate_hash_manifest.csv").is_file() and len(read_csv_rows(OUT / "manifests/candidate_hash_manifest.csv")) == 20,
                                    "internal_decision_written": True, "runtime_io_completed": True,
                                    "frozen_config_unchanged": sha256_file(FREEZE) == input_manifest["frozen_config_sha256"]},
              "frozen_config_sha256": input_manifest["frozen_config_sha256"], "fitting_sha256": input_manifest["fitting_source_sha256"],
              "candidate_hash_manifest_sha256": sha256_file(OUT / "manifests/candidate_hash_manifest.csv"),
              "internal_decision_sha256": sha256_file(internal_path),
              "runtime_verification_sha256": sha256_file(OUT / "runtime_io_verification_candidate_001.json"),
              "runtime_status": runtime_result.get("status"), "observed_data_used_for_tuning": "NO",
              "written_before_first_observed_weather_read": True, "created_at": datetime.now().astimezone().isoformat()}
    if not all(unlock["unlock_conditions"].values()):
        unlock["observed_external_qa_unlock"] = False
        write_json_once(OUT / "external_qa_unlock.json", unlock)
        raise RuntimeError("Observed external QA gate failed")
    unlock_path = OUT / "external_qa_unlock.json"
    if unlock_path.exists():
        prior = json.loads(unlock_path.read_text(encoding="utf-8"))
        for key in ("unlock_conditions", "frozen_config_sha256", "fitting_sha256", "candidate_hash_manifest_sha256",
                    "internal_decision_sha256", "runtime_verification_sha256", "runtime_status"):
            if prior.get(key) != unlock.get(key):
                raise RuntimeError(f"Existing observed QA unlock evidence changed: {key}")
    else:
        write_json_once(unlock_path, unlock)

    prior_manifest = OUT / "external_qa_2014_2023/observed_source_manifest.csv"
    prior_daily_summary = OUT / "external_qa_2014_2023/daily_marginal_summary.csv"
    if prior_manifest.is_file() and not prior_daily_summary.exists():
        failure_log = OUT / "logs/external_attempt_001_failure.json"
        if not failure_log.exists():
            write_json_once(failure_log, {
                "attempt": 1, "stage": "post-unlock descriptive aggregation", "status": "INTERRUPTED_BY_RUNNER_ERROR",
                "error": "KeyError: candidate_id while assigning realization labels to candidate CSV rows",
                "observed_source_manifest_was_written": True, "candidate_or_config_changed": False,
                "resolution": "Join the frozen candidate order to loaded rows by candidate_001..candidate_020, then recompute the same descriptive tables.",
                "no_weather_or_parameter_retuning": True})

    # First observed-period access occurs below, after external_qa_unlock.json is durable.
    observed_root = ROOT / "results/yc_random_weather_ppo/004_05/evaluation/runs/observed_weather/random_weather/ppo_seed_5/rendered_inputs/YCA"
    observed_rows: list[dict[str, Any]] = []
    observed_hashes = []
    for year in range(2014, 2024):
        folder = observed_root / str(year)
        wths = list(folder.rglob("*.WTH"))
        if len(wths) != 1:
            raise FileNotFoundError(f"Expected exactly one official observed WTH for {year}, found {len(wths)}")
        rows = parse_wth(wths[0], year)
        if len(rows) != (366 if is_leap(year) else 365):
            raise ValueError(f"Observed WTH is not a complete calendar year: {year} has {len(rows)} rows")
        observed_hashes.append({"year": year, "wth_path": str(wths[0].relative_to(ROOT)).replace("\\", "/"), "wth_sha256": sha256_file(wths[0]), "n_days": len(rows)})
        for row in rows:
            observed_rows.append({**row, "dataset": "observed", "realization": str(year)})
    write_csv_idempotent(OUT / "external_qa_2014_2023/observed_source_manifest.csv", observed_hashes)
    fitting, fitting_years = load_fitting(FITTING)
    fit_rows = [{**r, "dataset": "fitting", "realization": str(r["YEAR"])} for r in fitting]
    candidate_rows = []
    for index, item in enumerate(candidates_from_disk(), 1):
        candidate_id = f"candidate_{index:03d}"
        for row in item:
            candidate_rows.append({**row, "dataset": "candidate_pool", "realization": candidate_id})
    groups = {"fitting": fit_rows, "candidate_pool": candidate_rows, "observed": observed_rows}
    daily_summary = []
    quantiles = (.01, .05, .10, .50, .90, .95, .99)
    for dataset, records in groups.items():
        for var in VARIABLES:
            vals = [float(r[var]) for r in records]
            daily_summary.append({"dataset": dataset, "variable": var, "n_days": len(vals), "mean": statistics.mean(vals),
                                  "sd": statistics.stdev(vals), "min": min(vals), "max": max(vals),
                                  **{f"p{int(p*100):02d}": q(vals, p) for p in quantiles}})
    write_csv_idempotent(OUT / "external_qa_2014_2023/daily_marginal_summary.csv", daily_summary)
    thresholds = fitting_thresholds(fitting)
    realization_rows = []
    for dataset, records in groups.items():
        by_real: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in records:
            by_real[str(row["realization"])].append(row)
        for realization, series in sorted(by_real.items()):
            m = annual_metrics(series, thresholds)
            realization_rows.append({"dataset": dataset, "realization": realization, "n_days": len(series), **m})
    write_csv_idempotent(OUT / "external_qa_2014_2023/metrics_by_realization.csv", realization_rows)
    # Candidate inputs are complete calendar years; compare like-for-like only with
    # fitting and observed calendar years. Historical WGEN outputs are crop-season
    # captures and are summarized separately from the official 004_19/004_20 audit.
    paired_support = []
    for metric in ("seasonal_rain_total", "max_dry_spell_days", "rx1day", "rx5day", "srad_min", "srad_p01", "srad_p05", "srad_p10",
                   "srad_lt_fit_p05_fraction", "srad_lt_fit_p10_fraction", "tmax_mean", "tmax_max", "tmin_mean", "tmin_min",
                   "srad_mean", "srad_rain_spearman", "srad_tmax_spearman", "tmax_tmin_spearman"):
        for dataset in ("fitting", "candidate_pool", "observed"):
            vals = [float(r[metric]) for r in realization_rows if r["dataset"] == dataset]
            paired_support.append({"metric": metric, "dataset": dataset, "n_realizations": len(vals), "min": min(vals),
                                   "p05": q(vals, .05), "median": q(vals, .50), "p95": q(vals, .95), "max": max(vals),
                                   "support_preserving_vs_fitting": min(vals) >= min(float(r[metric]) for r in realization_rows if r["dataset"] == "fitting") - 1e-9 and max(vals) <= max(float(r[metric]) for r in realization_rows if r["dataset"] == "fitting") + 1e-9})
    write_csv_idempotent(OUT / "external_qa_2014_2023/realization_support_comparison.csv", paired_support)
    # Read only frozen prior conclusions for old-WGEN context; do not rebuild that pool.
    old_report = ROOT / "docs/yc_random_weather_004_20_wgen_srad_tail_joint_diagnosis.md"
    old_context = {"source_report": str(old_report.relative_to(ROOT)).replace("\\", "/"),
                   "source_report_sha256": sha256_file(old_report) if old_report.is_file() else None,
                   "interpretation": "Use the already-audited same-season result as context; 004_17 WGEN archive rows are crop-episode captures, not full-calendar-year series, so no invalid duration-mismatched pooling."}
    old_context_path = OUT / "external_qa_2014_2023/old_wgen_context.json"
    if old_context_path.exists():
        if json.loads(old_context_path.read_text(encoding="utf-8")) != old_context:
            raise RuntimeError("Existing old-WGEN context evidence changed")
    else:
        write_json_once(old_context_path, old_context)
    return {"observed_realization_count": 10, "observed_daily_rows": len(observed_rows), "observed_source_hashes": observed_hashes,
            "descriptive_external_qa_only": True, "observed_data_used_for_tuning": "NO",
            "candidate_pool_preserves_fitting_daily_support": all(x["support_preserving_vs_fitting"] for x in paired_support if x["dataset"] == "candidate_pool"),
            "candidate_pool_low_srad_p05_fraction": next(x for x in daily_summary if x["dataset"] == "candidate_pool" and x["variable"] == "SRAD"),
            "old_wgen_context": old_context, "unlock_sha256": sha256_file(OUT / "external_qa_unlock.json")}


def candidates_from_disk() -> list[list[dict[str, Any]]]:
    items = []
    for i in range(1, 21):
        path = OUT / "candidates" / f"candidate_{i:03d}" / "weather.csv"
        rows = read_csv_rows(path)
        items.append([{**r, "YEAR": int(r["YEAR"]), **{v: float(r[v]) for v in VARIABLES}} for r in rows])
    return items


def render_report(decision: dict[str, Any]) -> None:
    internal = json.loads((OUT / "internal_weather_qa_decision.json").read_text(encoding="utf-8")) if (OUT / "internal_weather_qa_decision.json").is_file() else {}
    runtime = json.loads((OUT / "runtime_io_verification_candidate_001.json").read_text(encoding="utf-8")) if (OUT / "runtime_io_verification_candidate_001.json").is_file() else {}
    external = json.loads((OUT / "external_qa_2014_2023/external_qa_summary.json").read_text(encoding="utf-8")) if (OUT / "external_qa_2014_2023/external_qa_summary.json").is_file() else {}
    external_rows = read_csv_rows(OUT / "external_qa_2014_2023/metrics_by_realization.csv") if (OUT / "external_qa_2014_2023/metrics_by_realization.csv").is_file() else []
    metric_columns = ("seasonal_rain_total", "wet_day_fraction", "max_dry_spell_days", "srad_min", "srad_p05", "srad_mean", "tmax_mean", "tmin_mean", "srad_rain_spearman", "srad_tmax_spearman", "tmax_tmin_spearman", "acf_rain_lag1", "acf_wet_lag1", "acf_srad_lag1")
    external_medians: dict[str, dict[str, float]] = {}
    for dataset in ("fitting", "candidate_pool", "observed"):
        subset = [row for row in external_rows if row.get("dataset") == dataset]
        external_medians[dataset] = {
            metric: statistics.median(float(row[metric]) for row in subset if row.get(metric) not in (None, ""))
            for metric in metric_columns if any(row.get(metric) not in (None, "") for row in subset)
        }
    pool_checks = read_csv_rows(OUT / "internal_qa/pool_metric_bootstrap_checks.csv") if (OUT / "internal_qa/pool_metric_bootstrap_checks.csv").is_file() else []
    failed_pool_metrics = [row["metric"] for row in pool_checks if row.get("pass", "").lower() == "false"]
    candidate_qa_rows = read_csv_rows(OUT / "internal_qa/candidate_level_qa.csv") if (OUT / "internal_qa/candidate_level_qa.csv").is_file() else []
    metric_fence_fail_count = sum(row.get("candidate_metric_fence_status") == "FAIL" for row in candidate_qa_rows)
    seam_fail_count = sum(row.get("seam_status") == "FAIL" for row in candidate_qa_rows)
    structural_all_pass = all(row.get("structural_status") == "PASS" and row.get("physical_status") == "PASS"
                              and row.get("provenance_status") == "PASS" and row.get("serialization_status") == "PASS"
                              for row in candidate_qa_rows)
    diversity_path = OUT / "internal_qa/pool_diversity_and_seam_summary.json"
    diversity = json.loads(diversity_path.read_text(encoding="utf-8")) if diversity_path.is_file() else {}
    first_external_failure = OUT / "logs/external_attempt_001_failure.json"
    freeze_sha = sha256_file(FREEZE)
    fitting_sha = sha256_file(FITTING)
    lines = [
        "# 004_24B YC 外部天气增强：20 套候选生成与验证", "",
        "## 1. 冻结配置", "",
        f"- 004_24A frozen config：`{FREEZE.relative_to(ROOT).as_posix()}`，SHA256 `{freeze_sha}`。",
        f"- fitting source：`{FITTING.relative_to(ROOT).as_posix()}`，SHA256 `{fitting_sha}`，2004–2013 共3653日。",
        "- 方法：多变量 moving-block bootstrap，完整 `[RAIN, SRAD, TMAX, TMIN]` 向量；块长28日；calendar midpoint ±7日；源块不跨年；尾块按剩余日数；原样拼接，不平滑、不插值、不裁剪。",
        "- 目标日期上下文：复用004_23B已核验的YC 2004 measured-weather FileX（CNYC0401，SDATE 04165，PDATE 04169，WTHER=M）；完整目标日历年为2004闰年。上下文只控制日期/闰年及代表候选I/O，不参与donor选择。",
        "- Seed：240425–240444；候选数固定20；Python `random.Random(seed)`按时间顺序逐块等概率有放回抽样，未重抽。", "",
        "## 2. 生成", "",
        f"- 实际生成并保留 `{decision.get('candidate_count_generated', 0)}/20` 套；没有替补或删除。逐日源年、源日期、块边界、目标日期和选择信息见 `candidates/candidate_NNN/provenance.csv`。",
        "- 生成值保存在 `weather.csv`；DSSAT 文件保存在 `weather.WTH`；逐套 SHA256 在 `manifests/candidate_hash_manifest.csv`。", "",
        "## 3. 单位与序列化", "",
        "- 004_24A冻结的变量单位沿用fitting/WTH约定；ChinaFLUX降雨按日累积mm，SRAD按已记录的0.0018积分转换为MJ m⁻² d⁻¹，candidate阶段无二次换算；TMAX/TMIN保持°C。",
        "- 004_23B输入WTH曾byte-identical进入runtime，字段无scale变换；candidate WTH按YC模板列序SRAD/TMAX/TMIN/RAIN、一位小数序列化，CSV↔WTH最大误差和容差见各candidate `generation_metadata.json`。", "",
        "## 4. Candidate 与 pool 内部 QA", "",
        f"- 内部QA：`{internal.get('internal_weather_qa_status', 'UNAVAILABLE')}`；candidate PASS/FAIL `{internal.get('candidate_pass_count', 0)}/{internal.get('candidate_fail_count', 0)}`；pool `{internal.get('pool_level_qa_status', 'UNAVAILABLE')}`。",
        f"- Candidate gate拆分：结构/物理/provenance/序列化全通过=`{structural_all_pass}`；指标fence失败 `{metric_fence_fail_count}/20`；seam失败 `{seam_fail_count}/20`；overall失败14/20。",
        "- 候选年度指标3×IQR、20套均值cluster-bootstrap、低SRAD P5/P10比例、seam、来源年覆盖、重复块和候选重复均严格使用004_24A冻结阈值；结果分别在 `internal_qa/` 下。失败candidate仍完整保留。",
        f"- pool冻结指标未通过项：{', '.join(failed_pool_metrics) if failed_pool_metrics else '无'}；seam检查 `seam_pool_2pct_pass={diversity.get('seam_pool_2pct_pass', 'UNKNOWN')}`，违规比例 `{diversity.get('seam_violation_fraction', {})}`。候选重复数 `{diversity.get('exact_duplicate_candidate_pairs', 'UNKNOWN')}`。",
        "- `internal_weather_qa_decision.json` 在读取Observed之前写入并冻结；不根据Observed结果改候选或阈值。", "",
        "## 5. candidate_001 runtime I/O", "",
        f"- status：`{runtime.get('status', 'NOT_RUN')}`；runtime复制WTH byte-identical：`{runtime.get('runtime_wth_byte_identical', 'UNKNOWN')}`；random_weather={runtime.get('random_weather', 'UNKNOWN')}；WTHER={runtime.get('WTHER', 'UNKNOWN')}。",
        f"- 固定映射按004_23B marker证据使用 source date = runtime YRDOY−1；匹配日数 `{runtime.get('n_overlap_days', 0)}`，重复runtime日期 `{runtime.get('duplicate_runtime_dates', 'NA')}`，逐日字段容差 `1e-5`。crop过程输出均为 `NON_SCIENTIFIC_RUNTIME_SIDE_EFFECT`。", "",
        "## 6. Observed 描述性外部 QA", "",
        f"- QA解锁：`{(OUT / 'external_qa_unlock.json').is_file()}`；Observed 2014–2023仅在内部QA、candidate hashes及candidate_001 runtime I/O证据写入后读取。Observed没有用于调参、筛选、替补或重抽。",
        f"- Observed日数 `{external.get('observed_daily_rows', 'NOT_ACCESSED')}`，10个年份的WTH哈希及来源见 `external_qa_2014_2023/observed_source_manifest.csv`。逐日分布、逐年天气指标和support对照见对应CSV。",
        "- 新候选pool是否保留fitting日尺度支持，按每变量年际范围在 `realization_support_comparison.csv` 中描述；它不要求20套覆盖所有Observed极端。旧WGEN为episode-season capture，本报告不将其与全年candidate作时长不匹配的直接池化；旧WGEN低SRAD缺口引用004_20正式诊断。", "",
        "### 描述性分布摘录（realization中位数）", "",
        "| 指标 | fitting 2004–2013 | 新候选20套 | Observed 2014–2023 |",
        "|---|---:|---:|---:|",
        f"| 年总降雨 (mm) | {external_medians.get('fitting', {}).get('seasonal_rain_total', float('nan')):.1f} | {external_medians.get('candidate_pool', {}).get('seasonal_rain_total', float('nan')):.1f} | {external_medians.get('observed', {}).get('seasonal_rain_total', float('nan')):.1f} |",
        f"| wet-day fraction | {external_medians.get('fitting', {}).get('wet_day_fraction', float('nan')):.3f} | {external_medians.get('candidate_pool', {}).get('wet_day_fraction', float('nan')):.3f} | {external_medians.get('observed', {}).get('wet_day_fraction', float('nan')):.3f} |",
        f"| 最大连续无雨日 | {external_medians.get('fitting', {}).get('max_dry_spell_days', float('nan')):.1f} | {external_medians.get('candidate_pool', {}).get('max_dry_spell_days', float('nan')):.1f} | {external_medians.get('observed', {}).get('max_dry_spell_days', float('nan')):.1f} |",
        f"| SRAD season minimum | {external_medians.get('fitting', {}).get('srad_min', float('nan')):.3f} | {external_medians.get('candidate_pool', {}).get('srad_min', float('nan')):.3f} | {external_medians.get('observed', {}).get('srad_min', float('nan')):.3f} |",
        f"| SRAD season P5 | {external_medians.get('fitting', {}).get('srad_p05', float('nan')):.3f} | {external_medians.get('candidate_pool', {}).get('srad_p05', float('nan')):.3f} | {external_medians.get('observed', {}).get('srad_p05', float('nan')):.3f} |",
        f"| mean Tmax (°C) | {external_medians.get('fitting', {}).get('tmax_mean', float('nan')):.3f} | {external_medians.get('candidate_pool', {}).get('tmax_mean', float('nan')):.3f} | {external_medians.get('observed', {}).get('tmax_mean', float('nan')):.3f} |",
        f"| mean Tmin (°C) | {external_medians.get('fitting', {}).get('tmin_mean', float('nan')):.3f} | {external_medians.get('candidate_pool', {}).get('tmin_mean', float('nan')):.3f} | {external_medians.get('observed', {}).get('tmin_mean', float('nan')):.3f} |",
        f"| Spearman SRAD–RAIN | {external_medians.get('fitting', {}).get('srad_rain_spearman', float('nan')):.3f} | {external_medians.get('candidate_pool', {}).get('srad_rain_spearman', float('nan')):.3f} | {external_medians.get('observed', {}).get('srad_rain_spearman', float('nan')):.3f} |",
        f"| Spearman SRAD–Tmax | {external_medians.get('fitting', {}).get('srad_tmax_spearman', float('nan')):.3f} | {external_medians.get('candidate_pool', {}).get('srad_tmax_spearman', float('nan')):.3f} | {external_medians.get('observed', {}).get('srad_tmax_spearman', float('nan')):.3f} |",
        f"| RAIN lag-1 ACF | {external_medians.get('fitting', {}).get('acf_rain_lag1', float('nan')):.3f} | {external_medians.get('candidate_pool', {}).get('acf_rain_lag1', float('nan')):.3f} | {external_medians.get('observed', {}).get('acf_rain_lag1', float('nan')):.3f} |",
        f"| wet/dry lag-1 ACF | {external_medians.get('fitting', {}).get('acf_wet_lag1', float('nan')):.3f} | {external_medians.get('candidate_pool', {}).get('acf_wet_lag1', float('nan')):.3f} | {external_medians.get('observed', {}).get('acf_wet_lag1', float('nan')):.3f} |",
        f"| SRAD lag-1 ACF | {external_medians.get('fitting', {}).get('acf_srad_lag1', float('nan')):.3f} | {external_medians.get('candidate_pool', {}).get('acf_srad_lag1', float('nan')):.3f} | {external_medians.get('observed', {}).get('acf_srad_lag1', float('nan')):.3f} |", "",
        "- 20候选池出现 fitting 低SRAD支持：逐日池 P5=`2.315`，fitting逐日 P5=`2.468` MJ m⁻² d⁻¹；候选 realization 的季节SRAD P5中位数=`2.357`，fitting=`2.606`。结合004_20中旧WGEN作物季在fitting P5以下为0日，这说明低尾在新候选中重新出现；但其频率/持续性仍须服从冻结QA，且内部pool gate当前失败，不能据此判方法通过。",
        "- 全年候选与fitting/Observed的总体分布并非处处重合：candidate年降雨中位数高于两参照，SRAD–Tmax关联中位数也高于fitting与Observed；这些为描述性差异，不代表显著性或因果。详细范围及所有固定依赖/持久性指标见 `metrics_by_realization.csv` 和 `realization_support_comparison.csv`。", "",
        "## 7. 限制", "",
        "- moving-block bootstrap不能生成超出2004–2013 donor日支持的新单日四变量组合；统计相似不等于天气因果，也不证明20套足够PPO训练。",
        "- 工程尝试记录：首次预检因SHA文本大小写比较误报停止，改为大小写无关核验后通过；runtime attempt_001因复用目录中已有输入被保护逻辑拒绝，attempt_002因缺少同目录CUL/SOL未启动，attempt_003完成唯一一次episode并通过。各次原始stderr与attempt记录均保留在 `runtime_io_candidate_001/`。",
        "- observed汇总首次解锁后处理因runner漏接candidate编号触发 `KeyError`；异常证据保存在 `logs/external_attempt_001_failure.json`。只修复行标签join后，对同一20套固定candidate和同一Observed源重新计算描述统计，没有重生成、重抽、替换或修改任何阈值。",
        "- 代表candidate只执行了一次no-op runtime I/O smoke；其crop/yield/stress输出不作为科学结果。未训练PPO、未运行WGEN。", "",
        "## 8. 最终判定", "",
        f"- `augmentation_pilot_status = {decision.get('augmentation_pilot_status', 'UNAVAILABLE')}`",
        f"- `weather_generation_method_status = {decision.get('weather_generation_method_status', 'UNAVAILABLE')}`",
        f"- `next_step = {decision.get('next_step', 'UNAVAILABLE')}`",
        "- PPO training：`NOT_STARTED`；git commit/push：`NOT_PERFORMED`。", "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "generate", "runtime", "external", "all"), default="all")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        config, fitting, years, input_manifest = preflight()
        if args.phase == "preflight":
            print(json.dumps({"preflight": "PASS", "frozen_config_sha256": input_manifest["frozen_config_sha256"],
                              "fitting_sha256": input_manifest["fitting_source_sha256"], "target_year": input_manifest["target_year"],
                              "unit_gate": "PASS"}, ensure_ascii=False))
            return 0
        if args.phase in ("generate", "all"):
            generate_and_internal(config, fitting, years, input_manifest)
            if args.phase == "generate":
                print((OUT / "internal_weather_qa_decision.json").read_text(encoding="utf-8"))
                return 0
        if args.phase in ("runtime", "all"):
            if not (OUT / "internal_weather_qa_decision.json").is_file():
                raise RuntimeError("Run --phase generate first; no implicit candidate generation in runtime-only phase")
            runtime_result = runtime_smoke()
            write_json_replace(OUT / "decision.json", {**json.loads((OUT / "decision.json").read_text(encoding="utf-8")),
                               "runtime_io_verification_status": runtime_result.get("status"),
                               "candidate_001_runtime_wth_byte_identical": runtime_result.get("runtime_wth_byte_identical"),
                               "candidate_001_runtime_daily_weather_match": runtime_result.get("daily_weather_match"),
                               "observed_external_validation_mode": "DESCRIPTIVE_ONLY_NOT_YET_UNLOCKED",
                               "next_step": "RUN_OBSERVED_DESCRIPTIVE_QA" if runtime_result.get("status") == "PASS" else "STOP_RUNTIME_IO_FAILURE"})
            if args.phase == "runtime":
                print(json.dumps(runtime_result, ensure_ascii=False, indent=2))
                return 0 if runtime_result.get("status") == "PASS" else 2
        if args.phase in ("external", "all"):
            runtime_result = json.loads((OUT / "runtime_io_verification_candidate_001.json").read_text(encoding="utf-8"))
            if runtime_result.get("status") != "PASS":
                raise RuntimeError("Observed QA remains locked because candidate_001 runtime I/O did not PASS")
            external = external_observed_qa(input_manifest, runtime_result)
            write_json_once(OUT / "external_qa_2014_2023/external_qa_summary.json", external)
            decision = json.loads((OUT / "decision.json").read_text(encoding="utf-8"))
            internal = json.loads((OUT / "internal_weather_qa_decision.json").read_text(encoding="utf-8"))
            external_structure_ok = external.get("candidate_pool_preserves_fitting_daily_support") is True
            all_gates = (internal.get("internal_weather_qa_status") == "PASS" and runtime_result.get("status") == "PASS"
                         and external_structure_ok)
            if all_gates:
                pilot, method, next_step = "PASS", "PILOT_VALIDATED", "DESIGN_FORMAL_TRAINING_WEATHER_POOL"
            elif internal.get("internal_weather_qa_status") == "PASS" and runtime_result.get("status") == "PASS":
                pilot, method, next_step = "PARTIAL", "PILOT_PARTIALLY_VALIDATED", "REVIEW_INTERNAL_QA_FAILURES_BEFORE_NEW_TASK"
            else:
                pilot, method, next_step = "FAIL", "NOT_VALIDATED", "STOP_AND_REVISE_METHOD_IN_NEW_TASK"
            candidate_low_tail = external.get("candidate_pool_low_srad_p05_fraction", {}).get("p05", float("inf")) < 2.46823896
            decision.update({"observed_external_validation_mode": "DESCRIPTIVE_ONLY", "observed_data_used_for_tuning": "NO",
                             "observed_external_qa": "COMPLETED_DESCRIPTIVE_ONLY",
                             "old_wgen_low_srad_gap_comparison_status": "CONFIRMED_BY_004_20",
                             "candidate_low_srad_fitting_support_status": "PRESENT_IN_CANDIDATE_POOL" if candidate_low_tail else "NOT_DEMONSTRATED",
                             "low_srad_recovery_status": ("PRESENT_BUT_INTERNAL_POOL_GATE_FAIL" if candidate_low_tail and not internal.get("pool_metric_gate_pass")
                                                           else "PRESERVED_FROM_FITTING" if candidate_low_tail else "NOT_PRESERVED"),
                             "external_candidate_pool_preserves_fitting_annual_support": external_structure_ok,
                             "augmentation_pilot_status": pilot, "weather_generation_method_status": method,
                             "ppo_training_status": "NOT_STARTED", "next_step": next_step})
            write_json_replace(OUT / "decision.json", decision)
            render_report(decision)
            print(json.dumps({"decision": decision, "external_summary": external}, ensure_ascii=False, indent=2))
            return 0 if pilot == "PASS" else (1 if pilot == "PARTIAL" else 2)
    except Exception as exc:
        failure = {"task_id": "004_24B", "status": "BLOCKED_OR_FAILED", "error": f"{type(exc).__name__}: {exc}",
                   "observed_data_accessed": "NO" if not (OUT / "external_qa_unlock.json").is_file() else "UNLOCKED",
                   "ppo_training_status": "NOT_STARTED", "wgen_run": "NOT_STARTED_BY_004_24B", "dssat_crop_experiment": "NOT_STARTED_BY_004_24B",
                   "created_at": datetime.now().astimezone().isoformat()}
        if not (OUT / "failure_record.json").exists():
            write_json_once(OUT / "failure_record.json", failure)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
