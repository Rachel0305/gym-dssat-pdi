from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
RESULTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_yc_wgen_seed_pilot as pilot


SOURCE_CLI = ROOT / "results/yc_wgen_cli_pilot/003_06_04/generated/CNYC.CLI"
FIXED_CLI = RESULTS / "candidate/CNYC_parsefix_01.CLI"
RUNTIME_INPUT_CLI = RESULTS / "candidate/runtime_input/CNYC.CLI"
TRAIN_WEATHER = ROOT / "results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv"
TEMPLATE_SOURCE = ROOT / "results/yc_wgen_cli_pilot/003_06_05/configs/CNYC0801_wgen_template.jinja2"
TEMPLATE_TARGET = RESULTS / "configs/CNYC0801_wgen_template.jinja2"
SOURCE_CLI_SHA256 = "5ABF5D7BB97EFAAE5E8361ABB4C773E1554B58116CBCA2213E75DCFF838285F0"
FIXED_CLI_SHA256 = "65CF134600A5881706A5D435E1A09B276ED92A21FA5ABE2E18AAF63AF1E3A929"
TRAIN_WEATHER_SHA256 = "4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34"
RUN_ID = "seed_101_parsefix_01_basename_ok"


def sha256(path: Path) -> str:
    return pilot.sha256_file(path)


def main() -> int:
    if sha256(SOURCE_CLI) != SOURCE_CLI_SHA256:
        raise SystemExit("Frozen original CNYC.CLI hash changed; refusing runtime smoke")
    if sha256(FIXED_CLI) != FIXED_CLI_SHA256:
        raise SystemExit("Isolated fixed candidate hash mismatch; refusing runtime smoke")
    if sha256(TRAIN_WEATHER) != TRAIN_WEATHER_SHA256:
        raise SystemExit("Frozen training-weather hash changed; refusing runtime smoke")
    if not TEMPLATE_SOURCE.is_file():
        raise SystemExit(f"Previous isolated FileX template not found: {TEMPLATE_SOURCE}")

    RUNTIME_INPUT_CLI.parent.mkdir(parents=True, exist_ok=True)
    if RUNTIME_INPUT_CLI.exists():
        if sha256(RUNTIME_INPUT_CLI) != FIXED_CLI_SHA256:
            raise SystemExit("Refusing to overwrite a different runtime-input CNYC.CLI")
    else:
        shutil.copy2(FIXED_CLI, RUNTIME_INPUT_CLI)

    first_attempt_path = RESULTS / "runtime_smoke_summary.json"
    first_attempt_archive = RESULTS / "runtime/attempt_01_basename_lookup_failure.json"
    if first_attempt_path.is_file() and not first_attempt_archive.exists():
        first_attempt = json.loads(first_attempt_path.read_text(encoding="utf-8"))
        first_attempt["initial_wrapper_assessment"] = {
            "runtime_smoke_status": first_attempt.get("runtime_smoke_status"),
            "wgenin_5010_resolved": first_attempt.get("wgenin_5010_resolved"),
        }
        first_attempt["runtime_smoke_status"] = "NOT_A_WGEN_TEST_CLI_LOOKUP_FAILURE"
        first_attempt["wgenin_5010_present_after_fix"] = None
        first_attempt["wgenin_5010_resolved"] = "NOT_TESTED"
        first_attempt["entered_weather_or_later_stage"] = False
        first_attempt["scope_note"] = (
            "The runtime could not locate CNYC.CLI and stopped in MAKEFW before WGENIN; "
            "this attempt provides no evidence about WGENIN parsing."
        )
        first_attempt_archive.parent.mkdir(parents=True, exist_ok=True)
        first_attempt_archive.write_text(
            json.dumps(first_attempt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    TEMPLATE_TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TEMPLATE_TARGET.exists():
        if sha256(TEMPLATE_TARGET) != sha256(TEMPLATE_SOURCE):
            raise SystemExit("Refusing to overwrite a different isolated FileX template")
    else:
        shutil.copy2(TEMPLATE_SOURCE, TEMPLATE_TARGET)

    runtime_temp = RESULTS / "runtime/tmp"
    runtime_temp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(runtime_temp)
    tempfile.tempdir = str(runtime_temp)

    pilot.OUTPUT_ROOT = RESULTS
    pilot.CLI_PATH = RUNTIME_INPUT_CLI
    pilot.EXPECTED_CLI_SHA256 = FIXED_CLI_SHA256
    pilot.TRAIN_WEATHER = TRAIN_WEATHER

    provenance = {
        "task": "003_06_05_01_fix_wgenin_5010",
        "station": "YC/CNYC",
        "source_cli": str(SOURCE_CLI.relative_to(ROOT)),
        "source_cli_sha256_before": sha256(SOURCE_CLI),
        "fixed_cli": str(FIXED_CLI.relative_to(ROOT)),
        "fixed_cli_sha256_before": sha256(FIXED_CLI),
        "runtime_input_cli": str(RUNTIME_INPUT_CLI.relative_to(ROOT)),
        "runtime_input_cli_basename": RUNTIME_INPUT_CLI.name,
        "runtime_input_cli_sha256_before": sha256(RUNTIME_INPUT_CLI),
        "filex_template_source": str(TEMPLATE_SOURCE.relative_to(ROOT)),
        "filex_template_sha256": sha256(TEMPLATE_TARGET),
        "train_weather_reference": str(TRAIN_WEATHER.relative_to(ROOT)),
        "train_weather_sha256_before": sha256(TRAIN_WEATHER),
        "weather_seed": 101,
        "ppo_seed": "NOT_APPLICABLE",
        "random_weather": True,
        "wther": "W",
        "wsta": "CNYC0801",
        "selected_treatment": "YC/CNYC 2008 treatment 1",
        "actions": "constant zero actions; no PPO/agent",
        "run_limit": "one serial runtime run only",
    }
    (RESULTS / "runtime/pilot_input_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    outcome = pilot.run_one(RUN_ID, 101)
    source_after = sha256(SOURCE_CLI)
    fixed_after = sha256(FIXED_CLI)
    runtime_input_after = sha256(RUNTIME_INPUT_CLI)
    weather_after = sha256(TRAIN_WEATHER)
    error = str(outcome.get("error") or "")
    wgenin_5010_present = "WGENIN" in error and "5010" in error
    status = outcome.get("runtime_compatibility_status")
    entered_weather_stage = bool(outcome.get("steps", 0) or outcome.get("weather_row_count", 0))
    if status == "PASS" and entered_weather_stage and not wgenin_5010_present:
        smoke_status = "PARSE_FIX_RUNTIME_PASS"
    elif wgenin_5010_present:
        smoke_status = "FAIL_SAME_WGENIN_5010"
    else:
        smoke_status = "FAIL_NEW_RUNTIME_BLOCKER"

    if smoke_status == "PARSE_FIX_RUNTIME_PASS":
        wgenin_resolution = "YES"
    elif wgenin_5010_present:
        wgenin_resolution = "NO"
    else:
        wgenin_resolution = "NOT_CONFIRMED"

    warning_path = RESULTS / "runtime" / RUN_ID / "runtime_snapshot" / "WARNING.OUT"
    warning_text = warning_path.read_text(encoding="utf-8", errors="replace") if warning_path.exists() else ""
    version_match = re.search(r"DSSAT Cropping System Model Ver\.\s*([^\s]+)", warning_text)

    summary = {
        "task": "003_06_05_01_fix_wgenin_5010",
        "runtime_version": version_match.group(1) if version_match else "NOT_CAPTURED",
        "original_error": "WGENIN 5010 at CNYC.CLI line 27",
        "runtime_smoke_status": smoke_status,
        "runtime_compatibility_status": status,
        "wgenin_5010_present_after_fix": wgenin_5010_present,
        "wgenin_5010_resolved": wgenin_resolution,
        "entered_weather_or_later_stage": entered_weather_stage,
        "new_runtime_error": outcome.get("error"),
        "outcome": outcome,
        "original_cli_sha256_before": SOURCE_CLI_SHA256,
        "original_cli_sha256_after": source_after,
        "original_cli_unchanged": source_after == SOURCE_CLI_SHA256,
        "fixed_cli_sha256_before": FIXED_CLI_SHA256,
        "fixed_cli_sha256_after": fixed_after,
        "fixed_cli_unchanged_during_smoke": fixed_after == FIXED_CLI_SHA256,
        "runtime_input_cli_sha256_after": runtime_input_after,
        "runtime_input_cli_unchanged_during_smoke": runtime_input_after == FIXED_CLI_SHA256,
        "train_weather_sha256_before": TRAIN_WEATHER_SHA256,
        "train_weather_sha256_after": weather_after,
        "train_weather_unchanged": weather_after == TRAIN_WEATHER_SHA256,
        "additional_seeds_run": [],
        "ppo_run": False,
    }
    (RESULTS / "runtime_smoke_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({key: summary[key] for key in (
        "runtime_smoke_status", "runtime_compatibility_status", "wgenin_5010_resolved",
        "original_cli_unchanged", "fixed_cli_unchanged_during_smoke",
        "runtime_input_cli_unchanged_during_smoke", "train_weather_unchanged",
    )}, ensure_ascii=False, indent=2))
    return 0 if smoke_status == "PARSE_FIX_RUNTIME_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
