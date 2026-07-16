from __future__ import annotations

import json
import math
import random
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


OUT = ROOT / "benchmark_results" / "022_16"
SEEDS = (0, 1, 2)
UPDATES = 3000
CHECKPOINTS = (0, 500, 1500, 3000)
LR = 1e-4
CONTAMINATED_SCENARIO = "W120_critical__N200_early"


def evaluate_choices(
    model: QNetwork,
    observations: torch.Tensor,
    daps: np.ndarray,
    support: dict[int, list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    return audit12.support_argmax(model, observations, daps, support)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    (OUT / "checkpoints").mkdir()
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
    clean_scenarios = test_scenarios - {CONTAMINATED_SCENARIO}
    train_mask = np.asarray([s in train_scenarios for s in scenarios])
    clean_mask = np.asarray([s in clean_scenarios for s in scenarios])
    if int(train_mask.sum()) != 216 or int(clean_mask.sum()) != 66:
        raise ValueError("unexpected train/clean-test sizes")

    pair_table, pair_obs, pair_targets = audit12.build_pair_targets(obs, manifest)
    train_obs = torch.tensor(obs[train_mask], dtype=torch.float32)
    train_actions = torch.tensor(actions[train_mask], dtype=torch.long)
    train_targets = torch.tensor(targets[train_mask], dtype=torch.float32)
    clean_obs = torch.tensor(obs[clean_mask], dtype=torch.float32)
    clean_daps = daps[clean_mask]
    clean_scenario_array = scenarios[clean_mask]
    support = audit12.support_map()

    dap110_total = int((daps == 110).sum())
    dap110_counts = {
        int(action): int(((daps == 110) & (actions == action)).sum()) for action in support[110]
    }
    if dap110_total != 48 or dap110_counts != {0: 40, 1: 8}:
        raise ValueError(f"DAP110 support provenance mismatch: total={dap110_total}, counts={dap110_counts}")

    losses: list[dict[str, float | int]] = []
    checkpoints: list[dict[str, float | int | bool]] = []
    stages: list[dict[str, float | int]] = []
    margins: list[dict[str, float | int | str | bool]] = []
    dap110_states: list[dict[str, float | int | str | bool]] = []

    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        payload = torch.load(
            audit12.CHECKPOINT_DIR / f"offline_mc_q_seed{seed}.pt",
            map_location="cpu",
            weights_only=False,
        )
        model = QNetwork()
        model.load_state_dict(payload["model_state_dict"])
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        baseline_choices, baseline_q = evaluate_choices(model, clean_obs, clean_daps, support)
        initial_mc, _ = audit12.evaluate_losses(
            model, train_obs, train_actions, train_targets, pair_obs, pair_targets
        )
        initial_mc_value = float(initial_mc.detach())

        for update in range(UPDATES + 1):
            if update in CHECKPOINTS:
                model.eval()
                mc_loss, pair_loss = audit12.evaluate_losses(
                    model, train_obs, train_actions, train_targets, pair_obs, pair_targets
                )
                choices, q = evaluate_choices(model, clean_obs, clean_daps, support)
                with torch.no_grad():
                    pair_q = model(pair_obs).cpu().numpy()
                dap1_changes = int(np.sum((choices != baseline_choices) & (clean_daps == 1)))
                dap110_changes = int(np.sum((choices != baseline_choices) & (clean_daps == 110)))
                checkpoints.append(
                    {
                        "seed": seed,
                        "update": update,
                        "mc_loss": float(mc_loss.detach()),
                        "mc_loss_relative_change": (
                            float(mc_loss.detach()) - initial_mc_value
                        ) / max(abs(initial_mc_value), 1e-12),
                        "descriptive_pair_loss": float(pair_loss.detach()),
                        "dap1_clean_test_argmax_changes": dap1_changes,
                        "dap110_clean_test_argmax_changes": dap110_changes,
                        "finite": bool(np.isfinite(q).all()),
                    }
                )
                for index, pair in pair_table.iterrows():
                    predicted = float(
                        pair_q[index, audit12.ACTION_CONTROL]
                        - pair_q[index, audit12.ACTION_TREATMENT]
                    )
                    margins.append(
                        {
                            "seed": seed,
                            "update": update,
                            "prefix": str(pair["prefix"]),
                            "target_margin": float(pair_targets[index]),
                            "predicted_margin": predicted,
                            "ranking_correct": predicted > 0,
                        }
                    )
                for dap in sorted(support):
                    mask = clean_daps == dap
                    stages.append(
                        {
                            "seed": seed,
                            "update": update,
                            "dap": dap,
                            "support_argmax_changes_from_update0": int(
                                np.sum(choices[mask] != baseline_choices[mask])
                            ),
                        }
                    )
                mask110 = clean_daps == 110
                for local_index in np.where(mask110)[0]:
                    dap110_states.append(
                        {
                            "seed": seed,
                            "update": update,
                            "scenario": str(clean_scenario_array[local_index]),
                            "q_action0": float(q[local_index, 0]),
                            "q_action1": float(q[local_index, 1]),
                            "q0_minus_q1": float(q[local_index, 0] - q[local_index, 1]),
                            "argmax_action": int(choices[local_index]),
                            "baseline_argmax_action": int(baseline_choices[local_index]),
                            "changed_from_update0": bool(
                                choices[local_index] != baseline_choices[local_index]
                            ),
                            "baseline_abs_q_gap": float(
                                abs(baseline_q[local_index, 0] - baseline_q[local_index, 1])
                            ),
                        }
                    )
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "seed": seed,
                        "additional_updates": update,
                        "loss": "MC_only",
                        "optimizer_reinitialized": True,
                    },
                    OUT / "checkpoints" / f"mc_only_seed{seed}_update{update}.pt",
                )
                model.train()
            if update == UPDATES:
                break
            q_all = model(train_obs)
            selected = q_all.gather(1, train_actions[:, None]).squeeze(1)
            mc_loss = torch.nn.functional.smooth_l1_loss(selected, train_targets)
            optimizer.zero_grad(set_to_none=True)
            mc_loss.backward()
            optimizer.step()
            losses.append({"seed": seed, "update": update + 1, "mc_loss_before_step": float(mc_loss.detach())})

    loss_df = pd.DataFrame(losses)
    checkpoint_df = pd.DataFrame(checkpoints)
    stage_df = pd.DataFrame(stages)
    margin_df = pd.DataFrame(margins)
    state_df = pd.DataFrame(dap110_states)
    loss_df.to_csv(OUT / "022_16_mc_only_training_loss.csv", index=False)
    checkpoint_df.to_csv(OUT / "022_16_checkpoint_metrics.csv", index=False)
    stage_df.to_csv(OUT / "022_16_stage_argmax_trajectory.csv", index=False)
    margin_df.to_csv(OUT / "022_16_pair_margin_descriptive.csv", index=False)
    state_df.to_csv(OUT / "022_16_dap110_state_q_trajectory.csv", index=False)

    final = checkpoint_df.loc[checkpoint_df["update"] == UPDATES].copy()
    drift_seed_count = int((final["dap110_clean_test_argmax_changes"] > 0).sum())
    branch = (
        "A_pairwise_not_necessary_for_dap110_drift"
        if drift_seed_count >= 2
        else (
            "B_pairwise_necessary_in_current_setup"
            if drift_seed_count == 0
            else "C_mixed_seed_evidence"
        )
    )

    comparison_rows: list[dict[str, int | str | bool | float]] = []
    for seed in SEEDS:
        combined_payload = torch.load(
            ROOT / "benchmark_results" / "022_15" / "checkpoints" / f"pairwise_mc_seed{seed}_update3000.pt",
            map_location="cpu",
            weights_only=False,
        )
        combined = QNetwork()
        combined.load_state_dict(combined_payload["model_state_dict"])
        combined_choices, _ = evaluate_choices(combined, clean_obs, clean_daps, support)
        baseline_payload = torch.load(
            audit12.CHECKPOINT_DIR / f"offline_mc_q_seed{seed}.pt", map_location="cpu", weights_only=False
        )
        baseline = QNetwork()
        baseline.load_state_dict(baseline_payload["model_state_dict"])
        base_choices, _ = evaluate_choices(baseline, clean_obs, clean_daps, support)
        mc_payload = torch.load(
            OUT / "checkpoints" / f"mc_only_seed{seed}_update3000.pt", map_location="cpu", weights_only=False
        )
        mc_model = QNetwork()
        mc_model.load_state_dict(mc_payload["model_state_dict"])
        mc_choices, _ = evaluate_choices(mc_model, clean_obs, clean_daps, support)
        mask110 = clean_daps == 110
        for idx in np.where(mask110)[0]:
            mc_changed = bool(mc_choices[idx] != base_choices[idx])
            combined_changed = bool(combined_choices[idx] != base_choices[idx])
            comparison_rows.append(
                {
                    "seed": seed,
                    "scenario": str(clean_scenario_array[idx]),
                    "baseline_action": int(base_choices[idx]),
                    "mc_only_action": int(mc_choices[idx]),
                    "pairwise_mc_action": int(combined_choices[idx]),
                    "mc_only_changed": mc_changed,
                    "pairwise_mc_changed": combined_changed,
                    "both_changed": bool(mc_changed and combined_changed),
                    "same_final_action": bool(mc_choices[idx] == combined_choices[idx]),
                }
            )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUT / "022_16_dap110_mc_only_vs_pairwise_comparison.csv", index=False)

    result = {
        "status": "completed",
        "branch": branch,
        "mc_only_dap110_drift_seed_count": drift_seed_count,
        "total_seeds": 3,
        "updates_per_seed": UPDATES,
        "total_offline_updates": UPDATES * len(SEEDS),
        "dssat_calls": 0,
        "online_interactions": 0,
        "dap110_total_transitions": dap110_total,
        "dap110_action_counts": dap110_counts,
        "scope": "Tests whether pairwise is necessary for drift; does not identify the full drift mechanism.",
        "guardrail_not_changed": True,
        "warmstart_started": False,
    }
    (OUT / "022_16_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for seed, group in checkpoint_df.groupby("seed"):
        axes[0].plot(group["update"], group["dap110_clean_test_argmax_changes"], marker="o", label=f"seed{seed}")
        axes[1].plot(group["update"], group["mc_loss_relative_change"] * 100, marker="o", label=f"seed{seed}")
    axes[0].set(title="DAP110 drift under MC-only continuation", xlabel="Updates", ylabel="Changed clean-test states")
    axes[1].set(title="MC loss trajectory", xlabel="Updates", ylabel="Change from update0 (%)")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "022_16_mc_only_drift_control.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "022_16_mc_only_drift_control.svg", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
