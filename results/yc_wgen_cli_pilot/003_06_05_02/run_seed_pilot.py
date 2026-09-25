from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TASK_ROOT = ROOT / "results" / "yc_wgen_cli_pilot" / "003_06_05_02"
WEATHER = ROOT / "results" / "yc_weather_gapfill_finalize" / "yc_wgen_fitting_weather_2004_2013.csv"
BROKEN_CLI = ROOT / "results" / "yc_wgen_cli_pilot" / "003_06_04" / "generated" / "CNYC.CLI"
PARSEFIX_CLI = ROOT / "results" / "yc_wgen_cli_pilot" / "003_06_05_01" / "candidate" / "CNYC_parsefix_01.CLI"
TEMPLATE_SOURCE = ROOT / "results" / "yc_wgen_cli_pilot" / "003_06_05_01" / "configs" / "CNYC0801_wgen_template.jinja2"
CLI = TASK_ROOT / "final" / "CNYC.CLI"
EXPECTED_WEATHER = "4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34"
EXPECTED_BROKEN = "5ABF5D7BB97EFAAE5E8361ABB4C773E1554B58116CBCA2213E75DCFF838285F0"
EXPECTED_FIXED = "65CF134600A5881706A5D435E1A09B276ED92A21FA5ABE2E18AAF63AF1E3A929"
RUNS = {
    "seed_101_run_a": 101,
    "seed_101_run_b": 101,
    "seed_102": 102,
    "seed_103": 103,
    "seed_104": 104,
    "seed_105": 105,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def require_hash(path: Path, expected: str) -> None:
    if not path.is_file() or sha256(path) != expected:
        raise RuntimeError(f"Input integrity check failed: {path}")


def main() -> int:
    require_hash(WEATHER, EXPECTED_WEATHER)
    require_hash(BROKEN_CLI, EXPECTED_BROKEN)
    require_hash(PARSEFIX_CLI, EXPECTED_FIXED)
    require_hash(CLI, EXPECTED_FIXED)
    schema = json.loads((TASK_ROOT / "final" / "final_cli_schema_check.json").read_text(encoding="utf-8"))
    if schema.get("passed") is not True:
        raise RuntimeError("Formal CLI schema gate did not pass")

    sys.path.insert(0, str(ROOT))
    import scripts.run_yc_wgen_seed_pilot as pilot

    config_dir = TASK_ROOT / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    template = config_dir / TEMPLATE_SOURCE.name
    if template.exists():
        if template.read_bytes() != TEMPLATE_SOURCE.read_bytes():
            raise RuntimeError(f"Refusing to overwrite different isolated template: {template}")
    else:
        shutil.copy2(TEMPLATE_SOURCE, template)

    output_root = TASK_ROOT
    runtime_tmp = output_root / "runtime" / "tmp"
    runtime_tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(runtime_tmp)
    tempfile.tempdir = str(runtime_tmp)
    pilot.OUTPUT_ROOT = output_root
    pilot.CLI_PATH = CLI
    pilot.EXPECTED_CLI_SHA256 = EXPECTED_FIXED
    pilot.TRAIN_WEATHER = WEATHER

    provenance = {
        "task": "003_06_05_02_finalize_cli_and_resume_seed_pilot",
        "station": "YC",
        "runtime": "DSSAT 4.8.0.024",
        "formal_cli_path": str(CLI.relative_to(ROOT)),
        "formal_cli_sha256": sha256(CLI),
        "formal_cli_matches_parsefix_candidate": sha256(CLI) == sha256(PARSEFIX_CLI),
        "broken_cli_preserved_path": str(BROKEN_CLI.relative_to(ROOT)),
        "broken_cli_sha256": sha256(BROKEN_CLI),
        "frozen_training_weather_path": str(WEATHER.relative_to(ROOT)),
        "frozen_training_weather_sha256": sha256(WEATHER),
        "isolated_filex_template_path": str(template.relative_to(ROOT)),
        "isolated_filex_template_sha256": sha256(template),
        "weather_seed_runs": RUNS,
        "ppo_seed": "NOT_APPLICABLE",
        "ppo_training": False,
        "validation_weather_used_for_fitting": False,
        "other_sites_modified": False,
        "execution_order": list(RUNS),
        "execution_mode": "serial",
    }
    provenance_path = output_root / "runtime" / "formal_seed_pilot_provenance.json"
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    outcomes = []
    for run_id, seed in RUNS.items():
        require_hash(CLI, EXPECTED_FIXED)
        require_hash(WEATHER, EXPECTED_WEATHER)
        require_hash(BROKEN_CLI, EXPECTED_BROKEN)
        print(f"[{run_id}] START weather_seed={seed}; PPO=NOT_APPLICABLE", flush=True)
        outcome = pilot.run_one(run_id, seed)
        outcomes.append(outcome)
        status = outcome.get("runtime_compatibility_status")
        print(
            f"[{run_id}] {status}; captured={outcome.get('weather_capture_status', 'NO')}; "
            f"rows={outcome.get('weather_row_count', 0)}; "
            f"peak_process_tree_rss_mb={outcome.get('peak_process_tree_rss_mb', 0)}; "
            f"elapsed_seconds={outcome.get('elapsed_seconds', 0)}",
            flush=True,
        )
        if outcome.get("candidate_cli_sha256_after_run") not in (None, EXPECTED_FIXED):
            raise RuntimeError("Formal CLI changed during runtime; stopping further runs")
        require_hash(CLI, EXPECTED_FIXED)
        require_hash(WEATHER, EXPECTED_WEATHER)
        require_hash(BROKEN_CLI, EXPECTED_BROKEN)

    status = {
        "runs_requested": list(RUNS),
        "runs_finished": [item.get("run_id") for item in outcomes],
        "all_six_invoked": len(outcomes) == 6,
        "runtime_statuses": {item.get("run_id"): item.get("runtime_compatibility_status") for item in outcomes},
        "formal_cli_sha256_after": sha256(CLI),
        "frozen_training_weather_sha256_after": sha256(WEATHER),
        "broken_cli_sha256_after": sha256(BROKEN_CLI),
        "all_integrity_checks_pass": (
            sha256(CLI) == EXPECTED_FIXED
            and sha256(WEATHER) == EXPECTED_WEATHER
            and sha256(BROKEN_CLI) == EXPECTED_BROKEN
        ),
    }
    (output_root / "runtime" / "formal_seed_pilot_execution.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return 0 if status["all_six_invoked"] and status["all_integrity_checks_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
