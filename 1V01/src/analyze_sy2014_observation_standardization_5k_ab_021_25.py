from __future__ import annotations

import sys
import json
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
from stable_baselines3 import DQN


OUT = ROOT / "benchmark_results" / "021_25"


def main() -> None:
    raw, standardized, validation = normbase.prepare_demonstrations()
    actions = np.asarray(raw["actions"], dtype=np.int64).reshape(-1)
    nonzero = actions != 0
    rows = []
    for arm, use_standardized in (
        ("control_raw_observation", False),
        ("treatment_standardized_observation", True),
    ):
        observations = standardized["observations"] if use_standardized else raw["observations"]
        obs_tensor = torch.as_tensor(observations, dtype=torch.float32)
        for checkpoint in range(1000, 5001, 1000):
            model = DQN.load(
                str(OUT / "training" / arm / "checkpoints" / f"checkpoint_{checkpoint}.zip"),
                device="cpu",
            )
            with torch.no_grad():
                q = model.q_net(obs_tensor)
            prediction = q.argmax(dim=1).cpu().numpy()
            expert_q = q[torch.arange(len(actions)), torch.as_tensor(actions)]
            other = q.clone()
            other[torch.arange(len(actions)), torch.as_tensor(actions)] = -torch.inf
            margin = expert_q - other.max(dim=1).values
            rows.append({
                "arm": arm,
                "checkpoint": checkpoint,
                "overall_demo_action_accuracy": float(np.mean(prediction == actions)),
                "noop_state_accuracy": float(np.mean(prediction[~nonzero] == 0)),
                "nonzero_demo_action_recall": float(np.mean(prediction[nonzero] == actions[nonzero])),
                "predicted_nonzero_fraction": float(np.mean(prediction != 0)),
                "predicted_action0_fraction": float(np.mean(prediction == 0)),
                "expert_margin_mean_all": float(margin.mean()),
                "expert_margin_mean_nonzero": float(margin[torch.as_tensor(nonzero)].mean()),
                "expert_margin_positive_fraction_nonzero": float(
                    (margin[torch.as_tensor(nonzero)] > 0).float().mean()
                ),
                "q_abs_mean": float(q.abs().mean()),
                "q_abs_max": float(q.abs().max()),
                "demo_nonzero_count": int(nonzero.sum()),
                "scaler_validation_passed": bool(validation["all_pre_run_checks_pass"]),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "021_25_checkpoint_demo_q_audit.csv", index=False, encoding="utf-8-sig")
    treatment = frame[frame.arm == "treatment_standardized_observation"]
    summary = {
        "status": "completed_offline_no_training",
        "demo_transition_count": int(len(actions)),
        "demo_nonzero_action_count": int(nonzero.sum()),
        "always_noop_naive_accuracy": float(np.mean(actions == 0)),
        "treatment_nonzero_recall_all_checkpoints_zero": bool(
            (treatment.nonzero_demo_action_recall == 0).all()
        ),
        "treatment_predicted_noop_fraction_all_checkpoints_one": bool(
            (treatment.predicted_action0_fraction == 1).all()
        ),
        "treatment_nonzero_expert_margin_1k": float(
            treatment.set_index("checkpoint").loc[1000, "expert_margin_mean_nonzero"]
        ),
        "treatment_nonzero_expert_margin_5k": float(
            treatment.set_index("checkpoint").loc[5000, "expert_margin_mean_nonzero"]
        ),
        "interpretation": "standardization fixed numerical gradients but the network did not recall any of five sparse nonzero demonstration actions",
    }
    (OUT / "021_25_posthoc_q_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
