from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_sy2014_dqfd_real_network_loss_diagnostic_021_22 as base
import run_sy2014_fixed_observation_standardization_diagnostic_021_24 as normbase
from frozen_nstep_dqn_config_020_11 import dqn_kwargs
from run_sy2014_demo_return_to_go_ab_021_27 import full_return_to_go
from run_sy2014_standardized_demo_pretraining_curve_021_26 import OfflineShapeEnv
from stable_baselines3 import DQN


OUT = ROOT / "benchmark_results" / "021_27"


def huber_slope_wrt_prediction(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.clamp(prediction - target, min=-1.0, max=1.0)


def main() -> None:
    np.random.seed(0)
    torch.manual_seed(0)
    raw, standardized, validation = normbase.prepare_demonstrations()
    treatment = full_return_to_go(standardized)
    env = OfflineShapeEnv(standardized["observations"].shape[1])
    model = DQN("MlpPolicy", env, verbose=0, **dqn_kwargs(seed=0))
    target_1, target_n = base.compute_targets(model, treatment)
    observations = torch.as_tensor(standardized["observations"], dtype=torch.float32)
    actions = torch.as_tensor(standardized["actions"], dtype=torch.long).reshape(-1)
    with torch.no_grad():
        q = model.q_net(observations)
        chosen = q.gather(1, actions[:, None]).squeeze(1)
    slope_1 = huber_slope_wrt_prediction(chosen, target_1)
    slope_n = huber_slope_wrt_prediction(chosen, target_n)
    nonzero_indices = np.where(np.asarray(raw["actions"]).reshape(-1) != 0)[0]
    rows = []
    for index in nonzero_indices:
        rows.append({
            "transition_index": int(index),
            "dap": int(round(float(raw["observations"][index, 1]))),
            "action": int(actions[index]),
            "initial_chosen_q": float(chosen[index]),
            "target_1_step": float(target_1[index]),
            "target_full_return": float(target_n[index]),
            "huber_slope_td1_wrt_q": float(slope_1[index]),
            "huber_slope_full_return_wrt_q": float(slope_n[index]),
            "equal_weight_td_slope_sum": float(slope_1[index] + slope_n[index]),
            "td_slopes_opposite": bool(float(slope_1[index] * slope_n[index]) < 0),
            "both_slopes_saturated": bool(abs(float(slope_1[index])) == 1 and abs(float(slope_n[index])) == 1),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "021_27_nonzero_td_gradient_conflict_audit.csv", index=False, encoding="utf-8-sig")
    summary = {
        "status": "completed_offline_no_training",
        "scaler_validation_passed": bool(validation["all_pre_run_checks_pass"]),
        "nonzero_event_count": int(len(frame)),
        "all_td_slopes_opposite": bool(frame.td_slopes_opposite.all()),
        "all_opposing_slopes_saturated": bool(frame.both_slopes_saturated.all()),
        "max_abs_equal_weight_td_slope_sum": float(frame.equal_weight_td_slope_sum.abs().max()),
        "interpretation": "at initialization, equal-weight 1-step and full-return Huber gradients cancel exactly on every nonzero demonstration event",
    }
    (OUT / "021_27_td_gradient_conflict_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    env.close()
    print(frame.to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
