from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "021_26"
DEMO = ROOT / "benchmark_results/021_20/021_20_demonstration_transitions.npz"


def main() -> None:
    data = np.load(DEMO)
    actions = data["actions"].reshape(-1)
    rewards = data["rewards"].reshape(-1)
    dones = data["dones"].reshape(-1)
    terminal_indices = np.where(dones != 0)[0]
    if len(terminal_indices) != 1:
        raise RuntimeError(f"Expected one terminal transition, got {terminal_indices}")
    terminal = int(terminal_indices[0])
    nonzero_indices = np.where(actions != 0)[0]
    rows = []
    for index in nonzero_indices:
        rows.append({
            "transition_index": int(index),
            "dap": int(round(float(data["observations"][index, 1]))),
            "action": int(actions[index]),
            "immediate_reward": float(rewards[index]),
            "n_step_return": float(data["n_step_returns"][index]),
            "n_step_horizon": int(data["n_step_horizons"][index]),
            "steps_to_terminal": terminal - int(index),
            "terminal_reward": float(rewards[terminal]),
            "terminal_reward_inside_n_step_window": bool(
                terminal - int(index) < int(data["n_step_horizons"][index])
            ),
            "n_step_contains_any_positive_reward": bool(
                np.any(rewards[index : index + int(data["n_step_horizons"][index])] > 0)
            ),
            "absolute_cost_to_margin_0p8_ratio": abs(float(rewards[index])) / 0.8,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "021_26_nonzero_demo_credit_assignment_audit.csv", index=False, encoding="utf-8-sig")
    summary = {
        "status": "completed_offline_no_training",
        "terminal_transition_index": terminal,
        "terminal_reward": float(rewards[terminal]),
        "nonzero_action_count": int(len(frame)),
        "all_nonzero_actions_have_negative_immediate_reward": bool((frame.immediate_reward < 0).all()),
        "all_nonzero_nstep_returns_equal_immediate_cost": bool(
            np.allclose(frame.n_step_return, frame.immediate_reward)
        ),
        "any_nonzero_action_sees_terminal_reward_in_nstep": bool(
            frame.terminal_reward_inside_n_step_window.any()
        ),
        "minimum_steps_from_nonzero_action_to_terminal": int(frame.steps_to_terminal.min()),
        "maximum_steps_from_nonzero_action_to_terminal": int(frame.steps_to_terminal.max()),
        "interpretation": "with frozen target and 5-step returns, demonstration resource actions receive cost-only targets and no harvest-yield credit",
    }
    (OUT / "021_26_credit_assignment_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(frame.to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
