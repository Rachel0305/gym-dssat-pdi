from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import audit_sy2014_pairwise_mc_gradient_conflict_022_12 as audit12
from run_sy2014_stage_mc_dqn_seed1_short_022_02 import QNetwork


OUT = ROOT / "benchmark_results" / "022_14"
BETA = 1.0
VIRTUAL_LR = 1e-4


def positive_geometric_steps(initial_error: float, threshold_error: float, fractional_reduction: float) -> int:
    if not (0 < threshold_error < initial_error and 0 < fractional_reduction < 1):
        raise ValueError("invalid geometric-step inputs")
    return int(math.ceil(math.log(threshold_error / initial_error) / math.log(1.0 - fractional_reduction)))


def main() -> None:
    torch.set_num_threads(min(4, torch.get_num_threads()))
    OUT.mkdir(parents=True, exist_ok=True)
    previous = pd.read_csv(ROOT / "benchmark_results" / "022_12" / "022_12_seed_gradient_summary.csv")
    lambda_fixed = float(previous["lambda_norm"].median())
    if not math.isfinite(lambda_fixed) or lambda_fixed <= 0:
        raise ValueError("invalid fixed lambda")

    dataset = np.load(audit12.NPZ_PATH)
    observations = dataset["observations"]
    actions = dataset["actions"]
    targets = dataset["targets"]
    scenarios = dataset["scenarios"].astype(str)
    manifest = pd.read_csv(audit12.MANIFEST_PATH)
    split = pd.read_csv(audit12.SPLIT_PATH)
    train_scenarios = set(split.loc[split["split"] == "train", "scenario"].astype(str))
    train_mask = np.asarray([scenario in train_scenarios for scenario in scenarios])
    pair_table, pair_obs, pair_targets = audit12.build_pair_targets(observations, manifest)

    train_obs = torch.tensor(observations[train_mask], dtype=torch.float32)
    train_actions = torch.tensor(actions[train_mask], dtype=torch.long)
    train_targets = torch.tensor(targets[train_mask], dtype=torch.float32)
    rows: list[dict[str, float | int | str]] = []

    for seed in range(3):
        payload = torch.load(
            audit12.CHECKPOINT_DIR / f"offline_mc_q_seed{seed}.pt",
            map_location="cpu",
            weights_only=False,
        )
        model = QNetwork()
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        params = list(model.parameters())
        mc_loss, pair_loss = audit12.evaluate_losses(
            model, train_obs, train_actions, train_targets, pair_obs, pair_targets
        )
        grads_mc = audit12.gradient_list(mc_loss, params)
        grads_pair = audit12.gradient_list(pair_loss, params)
        virtual = copy.deepcopy(model)
        with torch.no_grad():
            for param, grad_mc, grad_pair in zip(virtual.parameters(), grads_mc, grads_pair):
                param.add_(-VIRTUAL_LR * (grad_mc + lambda_fixed * grad_pair))
            q0 = model(pair_obs).numpy()
            q1 = virtual(pair_obs).numpy()

        for index, pair in pair_table.iterrows():
            target = float(pair_targets[index])
            margin0 = float(q0[index, audit12.ACTION_CONTROL] - q0[index, audit12.ACTION_TREATMENT])
            margin1 = float(q1[index, audit12.ACTION_CONTROL] - q1[index, audit12.ACTION_TREATMENT])
            delta = margin1 - margin0
            error0 = target - margin0
            error1 = target - margin1
            if not all(math.isfinite(v) for v in (target, margin0, margin1, delta, error0, error1)) or delta <= 0:
                raise ValueError(f"seed{seed} {pair['prefix']}: invalid/nonpositive improvement")
            region = "quadratic" if abs(error0) < BETA else "linear"
            linear_flip = int(math.ceil(max(0.0, -margin0) / delta))
            fractional = delta / error0
            geometric_flip = positive_geometric_steps(error0, target, fractional)
            geometric_90pct = positive_geometric_steps(error0, 0.1 * target, fractional)
            rows.append(
                {
                    "seed": seed,
                    "prefix": str(pair["prefix"]),
                    "lambda_fixed": lambda_fixed,
                    "target_margin": target,
                    "margin_before": margin0,
                    "margin_after_one_virtual_step": margin1,
                    "one_step_improvement": delta,
                    "residual_before": error0,
                    "residual_after": error1,
                    "smooth_l1_region_before": region,
                    "local_fractional_residual_reduction": fractional,
                    "linear_steps_to_sign_flip": linear_flip,
                    "quadratic_local_steps_to_sign_flip": geometric_flip,
                    "quadratic_local_steps_to_90pct_target": geometric_90pct,
                }
            )

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "022_14_pairwise_budget_estimates.csv", index=False)
    max_flip = int(table["quadratic_local_steps_to_sign_flip"].max())
    recommended = int(math.ceil(max_flip / 1000.0) * 1000)
    result = {
        "status": "completed",
        "lambda_fixed": lambda_fixed,
        "lambda_source": "median of 022_12 seed-specific shared-gradient norm ratios",
        "all_initial_residuals_in_smooth_l1_quadratic_region": bool(
            (table["smooth_l1_region_before"] == "quadratic").all()
        ),
        "maximum_linear_sign_flip_estimate": int(table["linear_steps_to_sign_flip"].max()),
        "maximum_quadratic_local_sign_flip_estimate": max_flip,
        "recommended_single_offline_training_updates": recommended,
        "maximum_quadratic_local_90pct_target_estimate": int(
            table["quadratic_local_steps_to_90pct_target"].max()
        ),
        "dqn_formal_training_steps": 0,
        "dssat_calls": 0,
        "saved_updated_checkpoint": False,
        "limitation": (
            "Local extrapolation only: gradients, Jacobian, MC interaction and optimizer dynamics "
            "can change during actual training; the update budget is not a convergence guarantee."
        ),
    }
    (OUT / "022_14_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
