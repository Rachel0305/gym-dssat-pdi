"""Corrected entrypoint for the transparent task-181 linear workflow.

The initial run stopped because DAP 110 had only no-op legal before the
decision.  This file deliberately contains no hidden experiment logic: it
only changes the second target to DAP 111 and redirects outputs.  The full
step-by-step loops remain visible in the imported task-181 source.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import run_181_sya_late_stage_short_horizon_counterfactual as task181


DEFAULT_CONFIG = ROOT / "configs" / "181_sya_late_stage_short_horizon_counterfactual_corrected.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config

    task181.TASK_NAME = "sya_late_stage_short_horizon_counterfactual_corrected"
    task181.TARGET_DAPS = [96, 111]
    task181.OUT = ROOT / "benchmark_results" / "181_sya_late_stage_short_horizon_counterfactual_corrected"
    task181.DOC = ROOT / "docs" / "181_sya_late_stage_short_horizon_counterfactual_corrected_record.md"
    task181.PROMPT = ROOT / "prompts" / "2026-08-20_sya_late_stage_short_horizon_counterfactual_corrected.md"

    print(json.dumps(task181.run(config_path.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
