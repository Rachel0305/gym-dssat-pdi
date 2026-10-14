"""061_01: longer LC weather-augmentation MaskablePPO run.

This thin wrapper reuses the audited 061_00 runner implementation but writes to
061_01-specific output paths. It keeps the same weather perturbation design and
validates only on the original LC 2014-2023 years.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_061_lca_lowIC_weather_augmentation_smoke_maskableppo as base


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "061_01"
TASK_NAME = "lca_lowIC_weather_augmentation_500k_maskableppo"
DEFAULT_CONFIG = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg_path = args.config if args.config.is_absolute() else (Path.cwd() / args.config).resolve()
    base.TASK_ID = TASK_ID
    base.TASK_NAME = TASK_NAME
    base.DEFAULT_CONFIG = DEFAULT_CONFIG
    base.OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
    base.DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
    base.PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"
    print(json.dumps(base.run(cfg_path, dry_run=args.dry_run), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
