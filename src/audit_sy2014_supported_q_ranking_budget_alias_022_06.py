from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

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

import run_sy2014_stage_mc_dqn_seed1_short_022_02 as base
import run_sy2014_fixed_grid_replay_prefill_dqn_seed1_022_03 as exp03
from stage_based_dqn_core_022 import (
    ACTION_TABLE_9,
    EXECUTABLE_STAGE_DAPS,
    IRRIGATION_BUDGET,
    NITROGEN_BUDGET,
    execute_stage_action,
)


OUT_ROOT = ROOT / "benchmark_results" / "022_06"
SOURCE_022_03 = ROOT / "benchmark_results" / "022_03"
SOURCE_022_05 = ROOT / "benchmark_results" / "022_05"
CHECKPOINTS = (15, 30, 45, 60)


def load_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, dict[int, tuple[int, ...]], pd.DataFrame]:
    data = np.load(SOURCE_022_03 / "022_03_fixed_grid_replay_dataset.npz")
    observations = data["observations"].astype(np.float32)
    actions = data["actions"].astype(int)
    targets = data["targets"].astype(float)
    manifest = pd.read_csv(SOURCE_022_03 / "022_03_fixed_grid_transition_manifest.csv")
    stage_actions = pd.read_csv(SOURCE_022_05 / "022_05_checkpoint_stage_actions.csv")
    prechecks = json.loads((SOURCE_022_05 / "022_05_pretraining_checks.json").read_text(encoding="utf-8"))
    support = {int(dap): tuple(int(a) for a in values) for dap, values in prechecks["derived_support"].items()}

    if observations.shape != (288, 25) or actions.shape != (288,) or targets.shape != (288,):
        raise RuntimeError("Unexpected 022_03 dataset shape")
    if len(manifest) != 288 or len(stage_actions) != 24:
        raise RuntimeError("Unexpected manifest or checkpoint stage row count")
    if not np.array_equal(actions, manifest["action_index"].to_numpy(dtype=int)):
        raise RuntimeError("Dataset actions do not align with manifest")
    if not np.allclose(targets, manifest["target_scaled"].to_numpy(dtype=float), atol=1e-6, rtol=0):
        raise RuntimeError("Dataset targets do not align with manifest")
    if set(support) != set(EXECUTABLE_STAGE_DAPS):
        raise RuntimeError("Support stages do not match executable stages")
    return observations, actions, targets, manifest, support, stage_actions


def q_ranking_audit(
    observations: np.ndarray,
    actions: np.ndarray,
    targets: np.ndarray,
    manifest: pd.DataFrame,
    support: dict[int, tuple[int, ...]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    q_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    dap1_rows: list[dict[str, Any]] = []
    dap_values = manifest["dap"].to_numpy(dtype=int)
    success_values = manifest["primary_scientific_pass_022_01"].astype(str).str.lower().eq("true").to_numpy()

    dap1_obs = observations[dap_values == 1]
    dap1_max_abs_state_difference = float(np.max(np.abs(dap1_obs - dap1_obs[0])))
    if dap1_max_abs_state_difference > 1e-7:
        raise RuntimeError(f"DAP1 states are not identical: max diff {dap1_max_abs_state_difference}")

    selected = pd.read_csv(SOURCE_022_05 / "022_05_checkpoint_stage_actions.csv")
    for checkpoint_season in CHECKPOINTS:
        payload = torch.load(
            SOURCE_022_05 / "checkpoints" / f"season_{checkpoint_season:03d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        model = base.QNetwork()
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        with torch.no_grad():
            all_q = model(torch.from_numpy(observations)).cpu().numpy()

        for dap in EXECUTABLE_STAGE_DAPS:
            state_mask = dap_values == dap
            supported = support[int(dap)]
            recorded_selected = int(
                selected.loc[
                    selected["episode"].eq(f"eval_s{checkpoint_season}") & selected["dap"].eq(dap),
                    "action_index",
                ].iloc[0]
            )
            action_summaries: list[dict[str, Any]] = []
            for action in supported:
                recorded_mask = state_mask & (actions == action)
                row = {
                    "checkpoint_season": checkpoint_season,
                    "dap": int(dap),
                    "action_index": int(action),
                    "requested_irrigation": ACTION_TABLE_9[action]["amir"],
                    "requested_nitrogen": ACTION_TABLE_9[action]["anfer"],
                    "mean_q_across_48_stage_states": float(all_q[state_mask, action].mean()),
                    "std_q_across_48_stage_states": float(all_q[state_mask, action].std()),
                    "recorded_target_mean": float(targets[recorded_mask].mean()),
                    "recorded_target_std": float(targets[recorded_mask].std()),
                    "recorded_sample_count": int(recorded_mask.sum()),
                    "recorded_success_fraction": float(success_values[recorded_mask].mean()),
                    "checkpoint_selected_action": recorded_selected,
                }
                action_summaries.append(row)
                q_rows.append(row)

            action_df = pd.DataFrame(action_summaries)
            q_rank = action_df["mean_q_across_48_stage_states"].rank(method="average")
            target_rank = action_df["recorded_target_mean"].rank(method="average")
            spearman = float(q_rank.corr(target_rank)) if len(action_df) >= 2 else float("nan")
            q_best = int(action_df.loc[action_df["mean_q_across_48_stage_states"].idxmax(), "action_index"])
            target_best = int(action_df.loc[action_df["recorded_target_mean"].idxmax(), "action_index"])
            ranking_rows.append(
                {
                    "checkpoint_season": checkpoint_season,
                    "dap": int(dap),
                    "supported_action_count": len(supported),
                    "mean_q_target_spearman": spearman,
                    "q_best_action": q_best,
                    "recorded_target_best_action": target_best,
                    "q_best_matches_target_best": q_best == target_best,
                    "checkpoint_selected_action": recorded_selected,
                    "checkpoint_selected_matches_target_best": recorded_selected == target_best,
                    "later_stage_targets_are_history_confounded": dap != 1,
                }
            )
            if dap == 1:
                for row in action_summaries:
                    dap1_rows.append(
                        {
                            **row,
                            "identical_initial_state_max_abs_difference": dap1_max_abs_state_difference,
                            "q_best_action": q_best,
                            "target_best_action": target_best,
                        }
                    )

    q_df = pd.DataFrame(q_rows)
    ranking_df = pd.DataFrame(ranking_rows)
    dap1_df = pd.DataFrame(dap1_rows)
    summary = {
        "dap1_identical_state_max_abs_difference": dap1_max_abs_state_difference,
        "ranking_comparisons": int(len(ranking_df)),
        "q_best_target_best_match_count": int(ranking_df["q_best_matches_target_best"].sum()),
        "checkpoint_selected_target_best_match_count": int(ranking_df["checkpoint_selected_matches_target_best"].sum()),
        "dap1_checkpoint_selected_target_best_match_count": int(
            ranking_df.loc[ranking_df["dap"].eq(1), "checkpoint_selected_matches_target_best"].sum()
        ),
        "dap1_comparison_count": int(ranking_df["dap"].eq(1).sum()),
        "median_stage_spearman": float(ranking_df["mean_q_target_spearman"].median()),
    }
    return q_df, ranking_df, dap1_df, summary


def budget_alias_audit(
    stage_actions: pd.DataFrame,
    support: dict[int, tuple[int, ...]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    season_rows: list[dict[str, Any]] = []
    for checkpoint_season in CHECKPOINTS:
        episode = stage_actions.loc[stage_actions["episode"].eq(f"eval_s{checkpoint_season}")].sort_values("stage_index")
        if len(episode) != 6:
            raise RuntimeError(f"Checkpoint {checkpoint_season} does not have six stage rows")
        first_n300_dap: int | None = None
        clipped_count = 0
        alias_group_count = 0
        for _, source in episode.iterrows():
            dap = int(source["dap"])
            used_i_before = float(source["cumulative_irrigation"] - source["executed_irrigation"])
            used_n_before = float(source["cumulative_nitrogen"] - source["executed_nitrogen"])
            execution_groups: dict[tuple[float, float], list[int]] = {}
            for candidate in support[dap]:
                executed = execute_stage_action(candidate, dap, used_i_before, used_n_before)
                key = (executed.executed_irrigation, executed.executed_nitrogen)
                execution_groups.setdefault(key, []).append(candidate)
            selected_key = (float(source["executed_irrigation"]), float(source["executed_nitrogen"]))
            aliases = execution_groups[selected_key]
            clipped = bool(
                float(source["requested_irrigation"]) != float(source["executed_irrigation"])
                or float(source["requested_nitrogen"]) != float(source["executed_nitrogen"])
            )
            if clipped:
                clipped_count += 1
            if len(aliases) > 1:
                alias_group_count += 1
            if first_n300_dap is None and float(source["cumulative_nitrogen"]) >= NITROGEN_BUDGET:
                first_n300_dap = dap
            rows.append(
                {
                    "checkpoint_season": checkpoint_season,
                    "stage_index": int(source["stage_index"]),
                    "dap": dap,
                    "selected_action": int(source["action_index"]),
                    "remaining_irrigation_before": IRRIGATION_BUDGET - used_i_before,
                    "remaining_nitrogen_before": NITROGEN_BUDGET - used_n_before,
                    "requested_irrigation": float(source["requested_irrigation"]),
                    "requested_nitrogen": float(source["requested_nitrogen"]),
                    "executed_irrigation": float(source["executed_irrigation"]),
                    "executed_nitrogen": float(source["executed_nitrogen"]),
                    "cumulative_irrigation": float(source["cumulative_irrigation"]),
                    "cumulative_nitrogen": float(source["cumulative_nitrogen"]),
                    "selected_action_was_clipped": clipped,
                    "same_execution_supported_actions": json.dumps(aliases),
                    "same_execution_alias_count": len(aliases),
                    "selected_execution_has_alias": len(aliases) > 1,
                    "nitrogen_budget_already_exhausted_before_action": used_n_before >= NITROGEN_BUDGET,
                }
            )
        season_rows.append(
            {
                "checkpoint_season": checkpoint_season,
                "first_dap_reaching_n300": first_n300_dap,
                "final_executed_nitrogen": float(episode["executed_nitrogen"].sum()),
                "final_executed_irrigation": float(episode["executed_irrigation"].sum()),
                "clipped_selected_action_count": clipped_count,
                "selected_execution_alias_stage_count": alias_group_count,
                "n300_was_reached_by_positive_executed_n_before_or_at_saturation": bool(
                    episode.loc[episode["cumulative_nitrogen"].le(NITROGEN_BUDGET), "executed_nitrogen"].sum()
                    >= NITROGEN_BUDGET
                ),
            }
        )
    detail = pd.DataFrame(rows)
    seasons = pd.DataFrame(season_rows)
    summary = {
        "stage_decisions": int(len(detail)),
        "clipped_selected_action_count": int(detail["selected_action_was_clipped"].sum()),
        "selected_execution_alias_stage_count": int(detail["selected_execution_has_alias"].sum()),
        "post_n_budget_exhaustion_decision_count": int(detail["nitrogen_budget_already_exhausted_before_action"].sum()),
        "all_checkpoints_reach_n300": bool((seasons["final_executed_nitrogen"] == 300.0).all()),
        "n300_is_real_executed_total_not_accounting_illusion": bool(
            seasons["n300_was_reached_by_positive_executed_n_before_or_at_saturation"].all()
        ),
        "first_dap_reaching_n300": {
            str(int(row.checkpoint_season)): int(row.first_dap_reaching_n300)
            for row in seasons.itertuples()
        },
    }
    return detail, seasons, summary


def plot_q_target(q_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, dap in zip(axes.ravel(), EXECUTABLE_STAGE_DAPS):
        subset = q_df.loc[q_df["dap"].eq(dap)]
        target = subset.groupby("action_index", as_index=False)["recorded_target_mean"].first()
        ax.plot(target["action_index"], target["recorded_target_mean"], "ko--", label="recorded MC target mean")
        for season, group in subset.groupby("checkpoint_season"):
            ax.plot(
                group["action_index"], group["mean_q_across_48_stage_states"], marker="o", label=f"Q s{season}"
            )
        ax.set_title(f"DAP {dap}")
        ax.set_xlabel("Supported action index")
        ax.set_ylabel("Scaled value")
        ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("SY2014 supported-action Q ranking vs fixed-grid MC target means")
    fig.savefig(OUT_ROOT / "022_06_supported_q_vs_target_ranking.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT_ROOT / "022_06_supported_q_vs_target_ranking.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if OUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT_ROOT}")
    OUT_ROOT.mkdir(parents=True)
    observations, actions, targets, manifest, support, stage_actions = load_inputs()
    q_df, ranking_df, dap1_df, q_summary = q_ranking_audit(
        observations, actions, targets, manifest, support
    )
    alias_df, alias_seasons, alias_summary = budget_alias_audit(stage_actions, support)

    q_df.to_csv(OUT_ROOT / "022_06_checkpoint_stage_action_q_target_summary.csv", index=False)
    ranking_df.to_csv(OUT_ROOT / "022_06_stage_ranking_summary.csv", index=False)
    dap1_df.to_csv(OUT_ROOT / "022_06_dap1_identical_state_q_target_audit.csv", index=False)
    alias_df.to_csv(OUT_ROOT / "022_06_checkpoint_budget_alias_detail.csv", index=False)
    alias_seasons.to_csv(OUT_ROOT / "022_06_checkpoint_budget_alias_summary.csv", index=False)
    plot_q_target(q_df)

    dap1_mismatch = q_summary["dap1_checkpoint_selected_target_best_match_count"] < q_summary["dap1_comparison_count"]
    majority_ranking_mismatch = q_summary["q_best_target_best_match_count"] < (q_summary["ranking_comparisons"] / 2)
    alias_drives_n300 = not alias_summary["n300_is_real_executed_total_not_accounting_illusion"]
    if (dap1_mismatch or majority_ranking_mismatch) and alias_drives_n300:
        branch = "C_q_ranking_and_budget_alias_coexist"
    elif dap1_mismatch or majority_ranking_mismatch:
        branch = "A_supported_q_ranking_misalignment_primary"
    elif alias_drives_n300:
        branch = "B_budget_alias_primary"
    else:
        branch = "D_evidence_insufficient"
    payload = {
        "status": "completed",
        "branch": branch,
        "zero_training": True,
        "zero_dssat_calls": True,
        "q_ranking_summary": q_summary,
        "budget_alias_summary": alias_summary,
        "interpretation_guardrail": (
            "DAP1 is an identical-state action comparison; later-stage target means are history-confounded and descriptive only."
        ),
    }
    (OUT_ROOT / "022_06_result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
