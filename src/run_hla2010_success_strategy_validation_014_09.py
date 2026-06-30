from __future__ import annotations

import argparse
from pathlib import Path

from run_hla2010_action_space_sensitivity_probe_014_08 import (
    ACTION_TABLE_9,
    OUT_DIR as BASE_OUT_DIR,
    train_and_eval,
)


YEAR = 2010
OUT_DIR = BASE_OUT_DIR.parent / "hla2010_success_strategy_validation_014_09"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Reuse the 9-action baseline-relative probe, but keep a separate output root.
    from run_hla2010_action_space_sensitivity_probe_014_08 import OUT_DIR as PROBE_OUT_DIR

    # Temporarily point the base module to the new output dir by using a symlink-style move:
    # we call the underlying training helper and then relocate the generated run into 014_09.
    # The helper writes under PROBE_OUT_DIR / YEAR / action9_seed...
    run_path = train_and_eval("action9", ACTION_TABLE_9, args.timesteps, args.seed)

    new_root = OUT_DIR / str(YEAR)
    new_root.mkdir(parents=True, exist_ok=True)
    target = new_root / run_path.name
    if target.exists():
        import shutil

        shutil.rmtree(target)
    if run_path.exists():
        import shutil

        shutil.move(str(run_path), str(target))
    print(str(target))


if __name__ == "__main__":
    main()
