from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_ROOT = PROJECT_ROOT / "Leave_One_experiments" / "smoke_tests_debug"
ORIGINAL_SUMMARY = PROJECT_ROOT / "Leave_One_experiments" / "smoke_tests" / "evaluation" / "smoke_test_summary.csv"
RERUN_SUMMARY = DEBUG_ROOT / "evaluation" / "smoke_test_timeout_rerun_summary.csv"
RUNNER = PROJECT_ROOT / "src" / "run_smoke_tests.py"


def main() -> None:
    (DEBUG_ROOT / "evaluation").mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(ORIGINAL_SUMMARY)
    failures = original[original["run_status"] != "ok"].copy()
    rows = []
    env = os.environ.copy()
    env["SMOKE_TEST_OUTPUT_ROOT"] = str(DEBUG_ROOT)
    for _, failed in failures.iterrows():
        station = failed["station"]
        year = int(failed["year"])
        policy = failed["policy_name"]
        cmd = [
            sys.executable,
            str(RUNNER),
            "--single",
            "--station",
            station,
            "--year",
            str(year),
            "--policy",
            policy,
        ]
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                timeout=300,
                text=True,
                capture_output=True,
                env=env,
            )
            single_path = DEBUG_ROOT / "evaluation" / "single" / f"{station}_{year}_{policy}.json"
            if completed.returncode == 0 and single_path.exists():
                row = json.loads(single_path.read_text(encoding="utf-8"))
                rerun_status = row.get("run_status", "failed")
                error_message = row.get("error_message", "")
            else:
                row = failed.to_dict()
                rerun_status = "failed"
                error_message = (completed.stderr or completed.stdout)[-1200:]
        except subprocess.TimeoutExpired:
            row = failed.to_dict()
            rerun_status = "timeout"
            error_message = "timeout_after_300s"
        row.update(
            {
                "original_run_status": failed["run_status"],
                "rerun_status": rerun_status,
                "run_status": rerun_status,
                "error_message": error_message,
                "fix_applied": "rendered_template_management_sections_fixed",
                "debug_notes": (
                    "MI/MF enabled in temporary rendered file; original static "
                    "irrigation/fertilizer rows replaced with safe zero rows; "
                    "missing irrigation section inserted when required."
                ),
            }
        )
        rows.append(row)
        pd.DataFrame(rows).to_csv(RERUN_SUMMARY, index=False, encoding="utf-8-sig")
        print(f"{station} {year} {policy}: {rerun_status}")
    print(RERUN_SUMMARY.relative_to(PROJECT_ROOT))
    print(pd.DataFrame(rows)["rerun_status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
