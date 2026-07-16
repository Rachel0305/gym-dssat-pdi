from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

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

import run_sy2014_stage_mc_dqn_seed1_short_022_02 as network_base
import run_sy2014_fixed_grid_offline_mc_q_generalization_022_07 as audit07


OUT_ROOT = ROOT / "benchmark_results" / "022_08"
SEEDS = (0, 1, 2)
UPDATES = 2000
TEST_SCENARIOS = 12
SPLIT_SEED = 20260716
MIN_SUPPORT = 2


def make_identifiable_support(
    manifest: pd.DataFrame, full_support: dict[int, tuple[int, ...]]
) -> tuple[dict[int, tuple[int, ...]], pd.DataFrame, set[str]]:
    counts = (
        manifest.groupby(["dap", "action_index"], as_index=False)
        .agg(total_transition_count=("scenario", "size"))
    )
    counts["primary_evaluable"] = counts["total_transition_count"] >= MIN_SUPPORT
    counts["status"] = np.where(counts["primary_evaluable"], "evaluable", "not_estimable_singleton")
    singleton_pairs = counts.loc[~counts["primary_evaluable"], ["dap", "action_index"]]
    singleton_scenarios: set[str] = set()
    for row in singleton_pairs.itertuples():
        singleton_scenarios.update(
            manifest.loc[
                manifest["dap"].eq(row.dap) & manifest["action_index"].eq(row.action_index), "scenario"
            ].tolist()
        )
    identifiable = {
        dap: tuple(
            action
            for action in actions
            if bool(
                counts.loc[
                    counts["dap"].eq(dap) & counts["action_index"].eq(action), "primary_evaluable"
                ].iloc[0]
            )
        )
        for dap, actions in full_support.items()
    }
    return identifiable, counts, singleton_scenarios


def fast_split(
    manifest: pd.DataFrame,
    identifiable: dict[int, tuple[int, ...]],
    mandatory_train: set[str],
) -> tuple[set[str], set[str], int]:
    scenarios = np.array(sorted(manifest["scenario"].unique()))
    scenario_to_index = {scenario: index for index, scenario in enumerate(scenarios)}
    pairs = [(dap, action) for dap, actions in identifiable.items() for action in actions]
    matrix = np.zeros((len(scenarios), len(pairs)), dtype=np.int8)
    for column, (dap, action) in enumerate(pairs):
        present = manifest.loc[manifest["dap"].eq(dap) & manifest["action_index"].eq(action), "scenario"].unique()
        for scenario in present:
            matrix[scenario_to_index[scenario], column] = 1
    totals = matrix.sum(axis=0)
    if (totals < 2).any():
        raise RuntimeError("Identifiable support still contains a pair with fewer than two samples")
    eligible_test = np.array([s for s in scenarios if s not in mandatory_train])
    labels = manifest.groupby("scenario")["success"].first().to_dict()
    rng = np.random.default_rng(SPLIT_SEED)
    for attempt in range(1, 200001):
        test_array = rng.choice(eligible_test, size=TEST_SCENARIOS, replace=False)
        test_indices = np.array([scenario_to_index[s] for s in test_array], dtype=int)
        test_counts = matrix[test_indices].sum(axis=0)
        if not bool(((test_counts >= 1) & (test_counts <= totals - 1)).all()):
            continue
        test = set(test_array.tolist())
        train = set(scenarios.tolist()) - test
        if len({labels[s] for s in test}) != 2 or len({labels[s] for s in train}) != 2:
            continue
        return train, test, attempt
    raise RuntimeError("No feasible 36/12 split found after 200000 target-blind attempts")


def main() -> None:
    if OUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT_ROOT}")
    OUT_ROOT.mkdir(parents=True)
    (OUT_ROOT / "checkpoints").mkdir()
    obs, actions, targets, manifest, full_support = audit07.load_dataset()
    identifiable, support_counts, singleton_scenarios = make_identifiable_support(manifest, full_support)
    support_counts.to_csv(OUT_ROOT / "022_08_stage_action_identifiability.csv", index=False)
    if support_counts.loc[~support_counts["primary_evaluable"]].shape[0] != 1:
        raise RuntimeError("Expected exactly one singleton stage-action pair")
    singleton = support_counts.loc[~support_counts["primary_evaluable"]].iloc[0]
    if int(singleton["dap"]) != 85 or int(singleton["action_index"]) != 5:
        raise RuntimeError("Singleton differs from preregistered DAP85/action5")

    train_scenarios, test_scenarios, split_attempt = fast_split(manifest, identifiable, singleton_scenarios)
    train_frame = manifest[manifest["scenario"].isin(train_scenarios)].copy()
    test_frame = manifest[manifest["scenario"].isin(test_scenarios)].copy()
    if len(train_frame) != 216 or len(test_frame) != 72:
        raise RuntimeError("Unexpected transition split size")
    if not singleton_scenarios.issubset(train_scenarios):
        raise RuntimeError("Singleton scenario was not retained in training")

    scenario_labels = manifest.groupby("scenario", as_index=False)["success"].first()
    scenario_labels["split"] = scenario_labels["scenario"].map(
        lambda value: "test" if value in test_scenarios else "train"
    )
    scenario_labels["mandatory_train_singleton_source"] = scenario_labels["scenario"].isin(singleton_scenarios)
    scenario_labels.to_csv(OUT_ROOT / "022_08_scenario_split.csv", index=False)

    coverage_rows: list[dict[str, Any]] = []
    for split_name, frame in (("train", train_frame), ("test", test_frame)):
        for dap, stage_actions in identifiable.items():
            for action in stage_actions:
                coverage_rows.append(
                    {
                        "split": split_name,
                        "dap": dap,
                        "action_index": action,
                        "transition_count": int(((frame["dap"] == dap) & (frame["action_index"] == action)).sum()),
                    }
                )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(OUT_ROOT / "022_08_identifiable_stage_action_split_coverage.csv", index=False)
    if (coverage["transition_count"] < 1).any():
        raise RuntimeError("Identifiable stage-action coverage failed")

    train_idx = train_frame["row_index"].to_numpy(dtype=int)
    train_obs = torch.from_numpy(obs[train_idx])
    train_actions = torch.from_numpy(actions[train_idx]).long()
    train_targets = torch.from_numpy(targets[train_idx]).float()
    train_target_mean = float(train_targets.mean().item())
    metric_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []

    torch.set_num_threads(4)
    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = network_base.QNetwork()
        control_metrics, control_stages = audit07.evaluate_model(
            model, obs, actions, targets, test_frame, identifiable, train_target_mean, seed, "untrained_control"
        )
        metric_rows.append(control_metrics)
        stage_rows.extend(control_stages)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        model.train()
        for update in range(1, UPDATES + 1):
            q_all = model(train_obs)
            q_selected = q_all.gather(1, train_actions.unsqueeze(1)).squeeze(1)
            loss = nn.functional.smooth_l1_loss(q_selected, train_targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_rows.append({"seed": seed, "update": update, "loss": float(loss.item())})
        trained_metrics, trained_stages = audit07.evaluate_model(
            model, obs, actions, targets, test_frame, identifiable, train_target_mean, seed, "offline_mc_q_2000"
        )
        metric_rows.append(trained_metrics)
        stage_rows.extend(trained_stages)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "seed": seed,
                "updates": UPDATES,
                "minimum_support": MIN_SUPPORT,
                "identifiable_support": identifiable,
                "singleton_scenarios": sorted(singleton_scenarios),
                "train_scenarios": sorted(train_scenarios),
                "test_scenarios": sorted(test_scenarios),
            },
            OUT_ROOT / "checkpoints" / f"offline_mc_q_seed{seed}.pt",
        )
        print(
            f"seed={seed} rho={trained_metrics['test_recorded_action_spearman']:.3f} "
            f"mae={trained_metrics['test_recorded_action_mae']:.3f} "
            f"dap1={trained_metrics['dap1_best_action_match']} "
            f"stage_rho={trained_metrics['median_stage_aggregate_spearman']:.3f} "
            f"pass={trained_metrics['all_preregistered_conditions']}",
            flush=True,
        )

    metrics = pd.DataFrame(metric_rows)
    stages = pd.DataFrame(stage_rows)
    losses = pd.DataFrame(loss_rows)
    metrics.to_csv(OUT_ROOT / "022_08_seed_test_metrics.csv", index=False)
    stages.to_csv(OUT_ROOT / "022_08_seed_stage_action_q_target.csv", index=False)
    losses.to_csv(OUT_ROOT / "022_08_training_loss.csv", index=False)
    trained = metrics[metrics["arm"].eq("offline_mc_q_2000")]
    pass_count = int(trained["all_preregistered_conditions"].sum())
    majority_mae_improved = int(trained["mae_better_than_constant"].sum()) >= 2
    majority_rank_insufficient = int((trained["test_recorded_action_spearman"] < 0.60).sum()) >= 2
    if pass_count >= 2:
        branch, next_step = "A_offline_ranking_learning_feasible", "allow_preregistered_short_online_warmstart_smoke"
    elif pass_count >= 1 or (majority_mae_improved and not majority_rank_insufficient):
        branch, next_step = "B_partial_or_seed_unstable", "report_only_no_automatic_online_training"
    else:
        branch, next_step = "C_offline_supervision_not_generalizing", "stop_offline_ranking_branch"

    checks = {
        "exactly_one_preregistered_singleton": bool(
            len(support_counts.loc[~support_counts["primary_evaluable"]]) == 1
            and int(singleton["dap"]) == 85
            and int(singleton["action_index"]) == 5
        ),
        "singleton_scenario_train_only": singleton_scenarios.issubset(train_scenarios),
        "scenario_disjoint": not bool(train_scenarios & test_scenarios),
        "train_36_test_12": len(train_scenarios) == 36 and len(test_scenarios) == 12,
        "identifiable_actions_covered_both_splits": bool((coverage["transition_count"] >= 1).all()),
        "success_failure_both_splits": train_frame["success"].nunique() == 2 and test_frame["success"].nunique() == 2,
        "three_seeds": set(trained["seed"].astype(int)) == set(SEEDS),
        "updates_2000_each": bool((losses.groupby("seed").size() == UPDATES).all()),
        "finite_metrics": bool(
            np.isfinite(
                trained[["test_recorded_action_mae", "test_recorded_action_rmse", "test_recorded_action_spearman"]].to_numpy()
            ).all()
        ),
    }
    if not all(checks.values()):
        branch, next_step = "D_implementation_failure", "stop_and_fix_without_scientific_interpretation"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for seed, group in losses.groupby("seed"):
        axes[0].plot(group["update"], group["loss"], label=f"seed{seed}")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Full-batch update")
    axes[0].set_ylabel("Training SmoothL1 loss")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)
    ordered = trained.sort_values("seed")
    x = np.arange(len(ordered))
    axes[1].bar(x, ordered["test_recorded_action_spearman"], color="#1f4e79")
    axes[1].axhline(0.60, color="#b22222", linestyle=":", label="threshold")
    axes[1].set_xticks(x, [f"seed{s}" for s in ordered["seed"]])
    axes[1].set_ylabel("Held-out recorded-action Spearman")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False)
    fig.suptitle("SY2014 minimum-support offline MC-Q generalization")
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "022_08_offline_mc_q_generalization.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT_ROOT / "022_08_offline_mc_q_generalization.svg", bbox_inches="tight")
    plt.close(fig)

    payload = {
        "status": "completed" if all(checks.values()) else "failed",
        "branch": branch,
        "next_step": next_step,
        "zero_dssat_calls": True,
        "zero_online_interactions": True,
        "minimum_support": MIN_SUPPORT,
        "singleton_pairs": support_counts.loc[~support_counts["primary_evaluable"]].to_dict(orient="records"),
        "singleton_scenarios": sorted(singleton_scenarios),
        "split_seed": SPLIT_SEED,
        "split_search_attempt": split_attempt,
        "checks": checks,
        "preregistered_seed_pass_count": pass_count,
        "trained_seed_metrics": trained.to_dict(orient="records"),
        "guardrail": "DAP30+ aggregate action rankings are history-confounded descriptive metrics, not causal counterfactuals.",
    }
    (OUT_ROOT / "022_08_result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
