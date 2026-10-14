"""Offline regression check using the preserved failed 10K replay audit."""

from __future__ import annotations

import json
import pickle
import sys
from collections import deque
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.reward_scaling import AuditableTrainingRewardScaleWrapper


CHECKPOINT = (
    ROOT
    / "benchmark_results/021_14"
    / "021_14_sy2014_reward_scale_seed1_25k__sy_2014_seed1"
    / "checkpoints/checkpoint_10000"
)


def main() -> None:
    wrapper = object.__new__(AuditableTrainingRewardScaleWrapper)
    records = pd.read_csv(CHECKPOINT / "reward_scale_records.csv").to_dict("records")
    wrapper.audit_records = deque(records, maxlen=50001)
    wrapper.callback_boundary_uncommitted_steps = {5000, 10000}
    wrapper.total_steps = 10000
    with (CHECKPOINT / "replay_buffer.pkl").open("rb") as handle:
        replay = pickle.load(handle)
    result = wrapper.replay_consistency(replay)
    result["passed"] = bool(
        result.get("checked")
        and result.get("all_finite")
        and result.get("max_abs_error", 1.0) < 1e-4
        and result.get("uncommitted_boundary_steps") == [5000, 10000]
    )
    output = ROOT / "benchmark_results/021_14/021_14_multiboundary_regression_test.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
