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
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import audit_sy2014_pairwise_mc_gradient_conflict_022_12 as audit12
from run_sy2014_stage_mc_dqn_seed1_short_022_02 import QNetwork


OUT = ROOT / "benchmark_results" / "022_15"
LAMBDA = 0.0060048738917845
UPDATES = 3000
CHECKPOINTS = (0, 500, 1500, 3000)
SEEDS = (0, 1, 2)
LR = 1e-4


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    (OUT / "checkpoints").mkdir()
    torch.set_num_threads(min(4, torch.get_num_threads()))

    dataset = np.load(audit12.NPZ_PATH)
    obs = dataset["observations"]
    actions = dataset["actions"]
    targets = dataset["targets"]
    scenarios = dataset["scenarios"].astype(str)
    stage_indices = dataset["stage_indices"]
    stage_daps = np.asarray([(1, 30, 50, 65, 85, 110)[int(i)] for i in stage_indices])
    manifest = pd.read_csv(audit12.MANIFEST_PATH)
    split = pd.read_csv(audit12.SPLIT_PATH)
    train_scenarios = set(split.loc[split["split"] == "train", "scenario"].astype(str))
    test_scenarios = set(split.loc[split["split"] == "test", "scenario"].astype(str))
    train_mask = np.asarray([s in train_scenarios for s in scenarios])
    original_test_mask = np.asarray([s in test_scenarios for s in scenarios])
    contaminated_scenario = "W120_critical__N200_early"
    if contaminated_scenario not in test_scenarios:
        raise ValueError("preregistered contaminated pair scenario is not in original test split")
    clean_test_scenarios = test_scenarios - {contaminated_scenario}
    clean_test_mask = np.asarray([s in clean_test_scenarios for s in scenarios])
    if int(train_mask.sum()) != 216 or int(original_test_mask.sum()) != 72 or int(clean_test_mask.sum()) != 66:
        raise ValueError("unexpected train/test/clean-test sizes")

    pair_table, pair_obs, pair_targets = audit12.build_pair_targets(obs, manifest)
    if set(pair_table["prefix"]) & test_scenarios != {contaminated_scenario}:
        raise ValueError("pair/test overlap differs from preregistration")
    train_obs = torch.tensor(obs[train_mask], dtype=torch.float32)
    train_actions = torch.tensor(actions[train_mask], dtype=torch.long)
    train_targets = torch.tensor(targets[train_mask], dtype=torch.float32)
    clean_obs = torch.tensor(obs[clean_test_mask], dtype=torch.float32)
    clean_daps = stage_daps[clean_test_mask]
    support = audit12.support_map()

    loss_rows: list[dict[str, float | int]] = []
    checkpoint_rows: list[dict[str, float | int | bool | str]] = []
    margin_rows: list[dict[str, float | int | str]] = []
    stage_rows: list[dict[str, float | int]] = []
    seed_final: list[dict[str, float | int | bool | str]] = []

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

        with torch.no_grad():
            baseline_choices, _ = audit12.support_argmax(model, clean_obs, clean_daps, support)
        initial_mc, _ = audit12.evaluate_losses(
            model, train_obs, train_actions, train_targets, pair_obs, pair_targets
        )
        initial_mc_value = float(initial_mc.detach())
        intermediate_pass = False

        for update in range(0, UPDATES + 1):
            if update in CHECKPOINTS:
                model.eval()
                mc_loss, pair_loss = audit12.evaluate_losses(
                    model, train_obs, train_actions, train_targets, pair_obs, pair_targets
                )
                with torch.no_grad():
                    pair_q = model(pair_obs).cpu().numpy()
                    choices, q_clean = audit12.support_argmax(model, clean_obs, clean_daps, support)
                margins = pair_q[:, audit12.ACTION_CONTROL] - pair_q[:, audit12.ACTION_TREATMENT]
                all_margins_positive = bool((margins > 0).all())
                mc_relative = (float(mc_loss.detach()) - initial_mc_value) / max(abs(initial_mc_value), 1e-12)
                dap1_changes = int(np.sum((choices != baseline_choices) & (clean_daps == 1)))
                dap110_changes = int(np.sum((choices != baseline_choices) & (clean_daps == 110)))
                finite = bool(
                    np.isfinite(margins).all()
                    and math.isfinite(float(mc_loss.detach()))
                    and math.isfinite(float(pair_loss.detach()))
                    and np.isfinite(q_clean).all()
                )
                passed = bool(
                    all_margins_positive
                    and mc_relative <= 0.01
                    and dap1_changes == 0
                    and dap110_changes == 0
                    and finite
                )
                if update not in (0, UPDATES) and passed:
                    intermediate_pass = True
                checkpoint_rows.append(
                    {
                        "seed": seed,
                        "update": update,
                        "mc_loss": float(mc_loss.detach()),
                        "pair_loss": float(pair_loss.detach()),
                        "mc_loss_relative_change_from_update0": mc_relative,
                        "all_three_pair_margins_positive": all_margins_positive,
                        "dap1_clean_test_argmax_changes": dap1_changes,
                        "dap110_clean_test_argmax_changes": dap110_changes,
                        "finite": finite,
                        "checkpoint_pass": passed,
                    }
                )
                for index, pair in pair_table.iterrows():
                    margin_rows.append(
                        {
                            "seed": seed,
                            "update": update,
                            "prefix": str(pair["prefix"]),
                            "target_margin": float(pair_targets[index]),
                            "predicted_margin": float(margins[index]),
                            "ranking_correct": bool(margins[index] > 0),
                        }
                    )
                for dap in sorted(support):
                    dap_mask = clean_daps == dap
                    stage_rows.append(
                        {
                            "seed": seed,
                            "update": update,
                            "dap": dap,
                            "clean_test_state_count": int(dap_mask.sum()),
                            "support_argmax_changes_from_update0": int(
                                np.sum(choices[dap_mask] != baseline_choices[dap_mask])
                            ),
                        }
                    )
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "seed": seed,
                        "additional_updates": update,
                        "lambda_pairwise": LAMBDA,
                        "optimizer_reinitialized": True,
                        "source_checkpoint": str(
                            audit12.CHECKPOINT_DIR / f"offline_mc_q_seed{seed}.pt"
                        ),
                    },
                    OUT / "checkpoints" / f"pairwise_mc_seed{seed}_update{update}.pt",
                )
                model.train()
            if update == UPDATES:
                break
            mc_loss, pair_loss = audit12.evaluate_losses(
                model, train_obs, train_actions, train_targets, pair_obs, pair_targets
            )
            total_loss = mc_loss + LAMBDA * pair_loss
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            loss_rows.append(
                {
                    "seed": seed,
                    "update": update + 1,
                    "mc_loss_before_step": float(mc_loss.detach()),
                    "pair_loss_before_step": float(pair_loss.detach()),
                    "combined_loss_before_step": float(total_loss.detach()),
                }
            )

        final_row = checkpoint_rows[-1]
        if int(final_row["seed"]) != seed or int(final_row["update"]) != UPDATES:
            raise RuntimeError("final checkpoint bookkeeping error")
        final_pass = bool(final_row["checkpoint_pass"])
        seed_final.append(
            {
                "seed": seed,
                "final_pass": final_pass,
                "intermediate_pass": intermediate_pass,
                "intermediate_pass_then_regressed": bool(intermediate_pass and not final_pass),
                "final_mc_loss_relative_change": float(final_row["mc_loss_relative_change_from_update0"]),
                "final_all_pair_margins_positive": bool(final_row["all_three_pair_margins_positive"]),
                "final_dap1_changes": int(final_row["dap1_clean_test_argmax_changes"]),
                "final_dap110_changes": int(final_row["dap110_clean_test_argmax_changes"]),
            }
        )

    loss_df = pd.DataFrame(loss_rows)
    checkpoint_df = pd.DataFrame(checkpoint_rows)
    margin_df = pd.DataFrame(margin_rows)
    stage_df = pd.DataFrame(stage_rows)
    final_df = pd.DataFrame(seed_final)
    loss_df.to_csv(OUT / "022_15_training_loss.csv", index=False)
    checkpoint_df.to_csv(OUT / "022_15_checkpoint_metrics.csv", index=False)
    margin_df.to_csv(OUT / "022_15_pair_margin_trajectory.csv", index=False)
    stage_df.to_csv(OUT / "022_15_stage_argmax_trajectory.csv", index=False)
    final_df.to_csv(OUT / "022_15_seed_final_status.csv", index=False)
    pair_table.to_csv(OUT / "022_15_controlled_pair_targets.csv", index=False)

    pass_count = int(final_df["final_pass"].sum())
    branch = "A_offline_pairwise_effective" if pass_count >= 2 else (
        "B_single_seed_only" if pass_count == 1 else "C_no_seed_passed"
    )
    result = {
        "status": "completed",
        "branch": branch,
        "final_seed_pass_count": pass_count,
        "total_seeds": 3,
        "lambda_fixed": LAMBDA,
        "updates_per_seed": UPDATES,
        "total_offline_updates": UPDATES * len(SEEDS),
        "checkpoint_updates": list(CHECKPOINTS),
        "optimizer_reinitialized_because_022_08_optimizer_state_absent": True,
        "mc_training_transitions": int(train_mask.sum()),
        "pairwise_states": len(pair_table),
        "original_test_scenarios": len(test_scenarios),
        "clean_guardrail_test_scenarios": len(clean_test_scenarios),
        "pairwise_overlap_original_test_scenario": contaminated_scenario,
        "dssat_calls": 0,
        "online_interactions": 0,
        "warmstart_022_13_started": False,
        "next_step_allowed": branch == "A_offline_pairwise_effective",
    }
    (OUT / "022_15_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for (seed, prefix), group in margin_df.groupby(["seed", "prefix"]):
        axes[0].plot(group["update"], group["predicted_margin"], marker="o", label=f"s{seed} {prefix.split('__')[0]}")
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Additional full-batch updates", ylabel="Q(a1)-Q(a7)", title="Controlled DAP65 pair margins")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    for seed, group in checkpoint_df.groupby("seed"):
        axes[1].plot(group["update"], group["mc_loss_relative_change_from_update0"] * 100, marker="o", label=f"seed{seed}")
    axes[1].axhline(1.0, color="#C00000", linestyle="--", linewidth=1, label="+1% guardrail")
    axes[1].set(xlabel="Additional full-batch updates", ylabel="MC loss change (%)", title="MC regression guardrail")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False)
    fig.suptitle("SY2014 fixed pairwise+MC offline training (022_15)")
    fig.tight_layout()
    fig.savefig(OUT / "022_15_offline_pairwise_training.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "022_15_offline_pairwise_training.svg", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
