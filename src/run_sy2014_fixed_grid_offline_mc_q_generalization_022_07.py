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

import run_sy2014_stage_mc_dqn_seed1_short_022_02 as base


OUT_ROOT = ROOT / "benchmark_results" / "022_07"
SOURCE = ROOT / "benchmark_results" / "022_03"
SUPPORT_SOURCE = ROOT / "benchmark_results" / "022_05" / "022_05_pretraining_checks.json"
SEEDS = (0, 1, 2)
UPDATES = 2000
TEST_SCENARIOS = 12
SPLIT_SEED = 20260715


def load_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, dict[int, tuple[int, ...]]]:
    raw = np.load(SOURCE / "022_03_fixed_grid_replay_dataset.npz")
    obs = raw["observations"].astype(np.float32)
    actions = raw["actions"].astype(np.int64)
    targets = raw["targets"].astype(np.float32)
    manifest = pd.read_csv(SOURCE / "022_03_fixed_grid_transition_manifest.csv")
    support_json = json.loads(SUPPORT_SOURCE.read_text(encoding="utf-8"))["derived_support"]
    support = {int(dap): tuple(int(x) for x in values) for dap, values in support_json.items()}
    if obs.shape != (288, 25) or actions.shape != (288,) or targets.shape != (288,) or len(manifest) != 288:
        raise RuntimeError("Unexpected fixed-grid dataset dimensions")
    if not np.array_equal(actions, manifest["action_index"].to_numpy(dtype=np.int64)):
        raise RuntimeError("NPZ actions and manifest do not align")
    if not np.allclose(targets, manifest["target_scaled"].to_numpy(dtype=np.float32), atol=1e-6, rtol=0):
        raise RuntimeError("NPZ targets and manifest do not align")
    manifest = manifest.copy()
    manifest["row_index"] = np.arange(len(manifest), dtype=int)
    manifest["success"] = manifest["primary_scientific_pass_022_01"].astype(str).str.lower().eq("true")
    return obs, actions, targets, manifest, support


def split_scenarios(manifest: pd.DataFrame) -> tuple[set[str], set[str], int]:
    scenarios = np.array(sorted(manifest["scenario"].unique()))
    pairs = sorted({(int(r.dap), int(r.action_index)) for r in manifest.itertuples()})
    rng = np.random.default_rng(SPLIT_SEED)
    for attempt in range(1, 100001):
        test = set(rng.choice(scenarios, size=TEST_SCENARIOS, replace=False).tolist())
        train = set(scenarios) - test
        test_frame = manifest[manifest["scenario"].isin(test)]
        train_frame = manifest[manifest["scenario"].isin(train)]
        coverage_ok = all(
            bool(((test_frame["dap"] == dap) & (test_frame["action_index"] == action)).any())
            and bool(((train_frame["dap"] == dap) & (train_frame["action_index"] == action)).any())
            for dap, action in pairs
        )
        label_ok = test_frame["success"].nunique() == 2 and train_frame["success"].nunique() == 2
        if coverage_ok and label_ok:
            return train, test, attempt
    raise RuntimeError("Unable to find a scenario split satisfying preregistered coverage")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return float(pd.Series(x).rank(method="average").corr(pd.Series(y).rank(method="average")))


def evaluate_model(
    model: base.QNetwork,
    obs: np.ndarray,
    actions: np.ndarray,
    targets: np.ndarray,
    frame: pd.DataFrame,
    support: dict[int, tuple[int, ...]],
    train_target_mean: float,
    seed: int,
    arm: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    indices = frame["row_index"].to_numpy(dtype=int)
    with torch.no_grad():
        q_all = model(torch.from_numpy(obs[indices])).cpu().numpy()
    chosen_q = q_all[np.arange(len(indices)), actions[indices]]
    truth = targets[indices]
    mae = float(np.mean(np.abs(chosen_q - truth)))
    rmse = float(np.sqrt(np.mean((chosen_q - truth) ** 2)))
    constant_mae = float(np.mean(np.abs(train_target_mean - truth)))

    stage_rows: list[dict[str, Any]] = []
    stage_spearman: list[float] = []
    dap1_predicted_best = dap1_target_best = None
    for dap in sorted(support):
        local_mask = frame["dap"].to_numpy(dtype=int) == dap
        local_q = q_all[local_mask]
        local_indices = indices[local_mask]
        action_rows: list[dict[str, Any]] = []
        for action in support[dap]:
            recorded = actions[local_indices] == action
            if not recorded.any():
                raise RuntimeError(f"Held-out set lacks DAP{dap} action{action}")
            row = {
                "seed": seed,
                "arm": arm,
                "dap": dap,
                "action_index": action,
                "mean_predicted_q_across_test_stage_states": float(local_q[:, action].mean()),
                "recorded_target_mean": float(targets[local_indices[recorded]].mean()),
                "recorded_target_count": int(recorded.sum()),
            }
            action_rows.append(row)
            stage_rows.append(row)
        action_df = pd.DataFrame(action_rows)
        rho = spearman(
            action_df["mean_predicted_q_across_test_stage_states"].to_numpy(),
            action_df["recorded_target_mean"].to_numpy(),
        )
        stage_spearman.append(rho)
        if dap == 1:
            dap1_predicted_best = int(
                action_df.loc[action_df["mean_predicted_q_across_test_stage_states"].idxmax(), "action_index"]
            )
            dap1_target_best = int(action_df.loc[action_df["recorded_target_mean"].idxmax(), "action_index"])

    metrics = {
        "seed": seed,
        "arm": arm,
        "test_recorded_action_mae": mae,
        "test_recorded_action_rmse": rmse,
        "test_recorded_action_spearman": spearman(chosen_q, truth),
        "constant_train_mean_baseline_mae": constant_mae,
        "mae_improvement_over_constant": constant_mae - mae,
        "mae_better_than_constant": mae < constant_mae,
        "median_stage_aggregate_spearman": float(np.nanmedian(stage_spearman)),
        "dap1_predicted_best_action": dap1_predicted_best,
        "dap1_recorded_target_best_action": dap1_target_best,
        "dap1_best_action_match": dap1_predicted_best == dap1_target_best,
    }
    metrics["all_preregistered_conditions"] = bool(
        metrics["test_recorded_action_spearman"] >= 0.60
        and metrics["mae_better_than_constant"]
        and metrics["dap1_best_action_match"]
        and metrics["median_stage_aggregate_spearman"] >= 0.50
    )
    return metrics, stage_rows


def main() -> None:
    if OUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT_ROOT}")
    OUT_ROOT.mkdir(parents=True)
    (OUT_ROOT / "checkpoints").mkdir()
    obs, actions, targets, manifest, support = load_dataset()
    pair_counts = (
        manifest.groupby(["dap", "action_index"], as_index=False)
        .size()
        .rename(columns={"size": "total_transition_count"})
    )
    pair_counts.to_csv(OUT_ROOT / "022_07_global_stage_action_counts.csv", index=False)
    impossible_pairs = pair_counts[pair_counts["total_transition_count"] < 2]
    if not impossible_pairs.empty:
        payload = {
            "status": "failed",
            "branch": "D_implementation_or_split_failure",
            "next_step": "redesign_split_preregistration_before_any_training",
            "zero_dssat_calls": True,
            "zero_online_interactions": True,
            "zero_gradient_updates": True,
            "reason": "The preregistered requirement that every stage-action pair appear in both train and test is mathematically impossible when a pair has fewer than two total samples.",
            "impossible_stage_action_pairs": impossible_pairs.to_dict(orient="records"),
        }
        (OUT_ROOT / "022_07_result.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
        return
    train_scenarios, test_scenarios, split_attempt = split_scenarios(manifest)
    train_frame = manifest[manifest["scenario"].isin(train_scenarios)].copy()
    test_frame = manifest[manifest["scenario"].isin(test_scenarios)].copy()
    if set(train_scenarios) & set(test_scenarios):
        raise RuntimeError("Scenario leakage detected")
    if len(train_frame) != 216 or len(test_frame) != 72:
        raise RuntimeError("Unexpected train/test transition counts")

    scenario_rows = []
    scenario_labels = manifest.groupby("scenario", as_index=False)["success"].first()
    for row in scenario_labels.itertuples():
        scenario_rows.append(
            {"scenario": row.scenario, "split": "test" if row.scenario in test_scenarios else "train", "success": row.success}
        )
    pd.DataFrame(scenario_rows).to_csv(OUT_ROOT / "022_07_scenario_split.csv", index=False)

    coverage_rows = []
    for split_name, split_frame in (("train", train_frame), ("test", test_frame)):
        for dap, actions_at_stage in support.items():
            for action in actions_at_stage:
                coverage_rows.append(
                    {
                        "split": split_name,
                        "dap": dap,
                        "action_index": action,
                        "transition_count": int(((split_frame["dap"] == dap) & (split_frame["action_index"] == action)).sum()),
                    }
                )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(OUT_ROOT / "022_07_stage_action_split_coverage.csv", index=False)
    if (coverage["transition_count"] < 1).any():
        raise RuntimeError("A stage-action pair is absent from one split")

    train_idx = train_frame["row_index"].to_numpy(dtype=int)
    train_obs = torch.from_numpy(obs[train_idx])
    train_actions = torch.from_numpy(actions[train_idx]).long()
    train_targets = torch.from_numpy(targets[train_idx]).float()
    train_target_mean = float(train_targets.mean().item())
    metric_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []

    # This changes execution speed only; data, updates, seeds and optimizer remain frozen.
    torch.set_num_threads(4)
    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = base.QNetwork()
        initial_metrics, initial_stage = evaluate_model(
            model, obs, actions, targets, test_frame, support, train_target_mean, seed, "untrained_control"
        )
        metric_rows.append(initial_metrics)
        stage_rows.extend(initial_stage)
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
        trained_metrics, trained_stage = evaluate_model(
            model, obs, actions, targets, test_frame, support, train_target_mean, seed, "offline_mc_q_2000"
        )
        metric_rows.append(trained_metrics)
        stage_rows.extend(trained_stage)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "seed": seed,
                "updates": UPDATES,
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
    metrics.to_csv(OUT_ROOT / "022_07_seed_test_metrics.csv", index=False)
    stages.to_csv(OUT_ROOT / "022_07_seed_stage_action_q_target.csv", index=False)
    losses.to_csv(OUT_ROOT / "022_07_training_loss.csv", index=False)

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

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for seed, group in losses.groupby("seed"):
        axes[0].plot(group["update"], group["loss"], label=f"seed{seed}")
    axes[0].set_xlabel("Full-batch update")
    axes[0].set_ylabel("Training SmoothL1 loss")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)
    x = np.arange(len(SEEDS))
    initial = metrics[metrics["arm"].eq("untrained_control")].sort_values("seed")
    final = trained.sort_values("seed")
    axes[1].bar(x - 0.2, initial["test_recorded_action_spearman"], width=0.4, label="untrained")
    axes[1].bar(x + 0.2, final["test_recorded_action_spearman"], width=0.4, label="offline MC-Q")
    axes[1].axhline(0.60, color="#b22222", linestyle=":", label="preregistered threshold")
    axes[1].set_xticks(x, [f"seed{s}" for s in SEEDS])
    axes[1].set_ylabel("Held-out recorded-action Spearman")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False)
    fig.suptitle("SY2014 fixed-grid offline MC-Q scenario-held-out test")
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "022_07_offline_mc_q_generalization.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT_ROOT / "022_07_offline_mc_q_generalization.svg", bbox_inches="tight")
    plt.close(fig)

    checks = {
        "scenario_disjoint": not bool(set(train_scenarios) & set(test_scenarios)),
        "train_36_test_12_scenarios": len(train_scenarios) == 36 and len(test_scenarios) == 12,
        "train_216_test_72_transitions": len(train_frame) == 216 and len(test_frame) == 72,
        "all_stage_actions_in_both_splits": bool((coverage["transition_count"] >= 1).all()),
        "both_success_labels_in_both_splits": train_frame["success"].nunique() == 2 and test_frame["success"].nunique() == 2,
        "three_fixed_seeds": set(trained["seed"].astype(int)) == set(SEEDS),
        "exactly_2000_updates_each": bool((losses.groupby("seed").size() == UPDATES).all()),
        "finite_metrics": bool(np.isfinite(trained[["test_recorded_action_mae", "test_recorded_action_rmse", "test_recorded_action_spearman"]].to_numpy()).all()),
    }
    if not all(checks.values()):
        branch, next_step = "D_implementation_or_split_failure", "stop_and_fix_without_interpreting_science"
    payload: dict[str, Any] = {
        "status": "completed" if all(checks.values()) else "failed",
        "branch": branch,
        "next_step": next_step,
        "zero_dssat_calls": True,
        "zero_online_interactions": True,
        "split_seed": SPLIT_SEED,
        "split_search_attempt": split_attempt,
        "checks": checks,
        "preregistered_seed_pass_count": pass_count,
        "trained_seed_metrics": trained.to_dict(orient="records"),
    }
    (OUT_ROOT / "022_07_result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
