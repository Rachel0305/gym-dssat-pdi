from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import audit_sy2014_pairwise_mc_gradient_conflict_022_12 as audit12
from run_sy2014_stage_mc_dqn_seed1_short_022_02 import QNetwork


OUT = ROOT / "benchmark_results" / "022_18"
ALPHAS = np.round(np.linspace(0.0, 1.0, 1001), 3)
SEEDS = (0, 1, 2)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    data = np.load(audit12.NPZ_PATH)
    obs = data["observations"]
    actions = data["actions"]
    targets = data["targets"]
    scenarios = data["scenarios"].astype(str)
    stage_indices = data["stage_indices"]
    daps = np.asarray([(1, 30, 50, 65, 85, 110)[int(i)] for i in stage_indices])
    manifest = pd.read_csv(audit12.MANIFEST_PATH)
    split = pd.read_csv(audit12.SPLIT_PATH)
    train_scenarios = set(split.loc[split["split"] == "train", "scenario"].astype(str))
    test_scenarios = set(split.loc[split["split"] == "test", "scenario"].astype(str))
    clean_scenarios = test_scenarios - {"W120_critical__N200_early"}
    train_mask = np.asarray([s in train_scenarios for s in scenarios])
    clean_mask = np.asarray([s in clean_scenarios for s in scenarios])

    pair_table, pair_obs, pair_targets = audit12.build_pair_targets(obs, manifest)
    controlled = pd.read_csv(ROOT / "benchmark_results" / "022_17" / "022_17_controlled_action_pairs.csv")
    protected_scenarios = sorted(controlled.loc[~controlled["execution_alias"], "scenario"].astype(str))
    if len(protected_scenarios) != 8:
        raise ValueError(f"expected 8 non-alias protected DAP110 scenarios, got {len(protected_scenarios)}")
    protected_indices = []
    for scenario in protected_scenarios:
        idx = np.where((scenarios == scenario) & (daps == 110))[0]
        if len(idx) != 1:
            raise ValueError(f"{scenario}: DAP110 state count={len(idx)}")
        protected_indices.append(int(idx[0]))
    protected_obs = torch.tensor(obs[protected_indices], dtype=torch.float32)

    train_obs = torch.tensor(obs[train_mask], dtype=torch.float32)
    train_actions = torch.tensor(actions[train_mask], dtype=torch.long)
    train_targets = torch.tensor(targets[train_mask], dtype=torch.float32)
    clean_obs = torch.tensor(obs[clean_mask], dtype=torch.float32)
    clean_daps = daps[clean_mask]
    support = audit12.support_map()

    rows: list[dict[str, float | int | bool]] = []
    summaries: list[dict[str, float | int | bool | None]] = []
    for seed in SEEDS:
        p0 = torch.load(
            ROOT / "benchmark_results" / "022_15" / "checkpoints" / f"pairwise_mc_seed{seed}_update0.pt",
            map_location="cpu", weights_only=False,
        )["model_state_dict"]
        p500 = torch.load(
            ROOT / "benchmark_results" / "022_15" / "checkpoints" / f"pairwise_mc_seed{seed}_update500.pt",
            map_location="cpu", weights_only=False,
        )["model_state_dict"]
        model = QNetwork()
        model.load_state_dict(p0)
        baseline_choices, _ = audit12.support_argmax(model, clean_obs, clean_daps, support)
        feasible_alphas: list[float] = []
        first_pair_correct: float | None = None
        first_protection_failure: float | None = None
        for alpha in ALPHAS:
            interpolated = {
                key: (1.0 - float(alpha)) * p0[key] + float(alpha) * p500[key]
                for key in p0
            }
            model.load_state_dict(interpolated)
            model.eval()
            with torch.no_grad():
                pair_q = model(pair_obs)
                margins = pair_q[:, audit12.ACTION_CONTROL] - pair_q[:, audit12.ACTION_TREATMENT]
                protected_q = model(protected_obs)
                protected_actions = torch.argmax(protected_q[:, [0, 1]], dim=1)  # 0->action0, 1->action1
                selected = model(train_obs).gather(1, train_actions[:, None]).squeeze(1)
                mc_loss = torch.nn.functional.smooth_l1_loss(selected, train_targets)
            choices, _ = audit12.support_argmax(model, clean_obs, clean_daps, support)
            pair_ok = bool(torch.all(margins > 0).item())
            protection_ok = bool(torch.all(protected_actions == 0).item())
            feasible = bool(pair_ok and protection_ok)
            if pair_ok and first_pair_correct is None:
                first_pair_correct = float(alpha)
            if not protection_ok and first_protection_failure is None:
                first_protection_failure = float(alpha)
            if feasible:
                feasible_alphas.append(float(alpha))
            rows.append(
                {
                    "seed": seed,
                    "alpha": float(alpha),
                    "minimum_dap65_pair_margin": float(margins.min().item()),
                    "dap65_all_three_correct": pair_ok,
                    "dap110_action0_count_of_8": int((protected_actions == 0).sum().item()),
                    "dap110_all_eight_protected": protection_ok,
                    "both_causal_conditions": feasible,
                    "mc_train_loss": float(mc_loss.item()),
                    "dap1_clean_argmax_changes": int(np.sum((choices != baseline_choices) & (clean_daps == 1))),
                    "dap85_clean_argmax_changes": int(np.sum((choices != baseline_choices) & (clean_daps == 85))),
                }
            )
        summaries.append(
            {
                "seed": seed,
                "has_feasible_alpha": bool(feasible_alphas),
                "feasible_alpha_min": min(feasible_alphas) if feasible_alphas else None,
                "feasible_alpha_max": max(feasible_alphas) if feasible_alphas else None,
                "feasible_grid_point_count": len(feasible_alphas),
                "first_alpha_all_dap65_correct": first_pair_correct,
                "first_alpha_dap110_protection_failure": first_protection_failure,
            }
        )

    audit_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summaries)
    audit_df.to_csv(OUT / "022_18_parameter_path_audit.csv", index=False)
    summary_df.to_csv(OUT / "022_18_seed_feasible_intervals.csv", index=False)
    feasible_seed_count = int(summary_df["has_feasible_alpha"].sum())
    branch = (
        "A_conservative_path_feasible"
        if feasible_seed_count >= 2
        else ("B_single_seed_feasible" if feasible_seed_count == 1 else "C_no_feasible_path")
    )
    result = {
        "status": "completed",
        "branch": branch,
        "feasible_seed_count": feasible_seed_count,
        "total_seeds": 3,
        "alpha_grid_min": 0.0,
        "alpha_grid_max": 1.0,
        "alpha_grid_step": 0.001,
        "protected_dap110_scenario_count": len(protected_scenarios),
        "dqn_training_steps": 0,
        "dssat_calls": 0,
        "saved_interpolated_checkpoint": False,
        "limitation": "Linear parameter interpolation is a path feasibility audit, not a trained model or convergence proof.",
    }
    (OUT / "022_18_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for seed, group in audit_df.groupby("seed"):
        axes[0].plot(group["alpha"], group["minimum_dap65_pair_margin"], label=f"seed{seed}")
        axes[1].plot(group["alpha"], group["dap110_action0_count_of_8"], label=f"seed{seed}")
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set(title="Worst controlled DAP65 margin", xlabel="Interpolation alpha", ylabel="min Q(a1)-Q(a7)")
    axes[1].axhline(8, color="black", linestyle="--", linewidth=1)
    axes[1].set(title="Protected DAP110 states choosing action0", xlabel="Interpolation alpha", ylabel="Count of 8")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "022_18_conservative_path_feasibility.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "022_18_conservative_path_feasibility.svg", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
