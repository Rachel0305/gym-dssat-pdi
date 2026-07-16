from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmark_results" / "022_08" / "022_08_seed_stage_action_q_target.csv"
OUT_ROOT = ROOT / "benchmark_results" / "022_09"
SEEDS = (0, 1, 2)
EXPECTED_ACTIONS = (1, 2, 4, 5, 7, 8)


def main() -> None:
    if OUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT_ROOT}")
    OUT_ROOT.mkdir(parents=True)
    source = pd.read_csv(SOURCE)
    frame = source[
        source["arm"].eq("offline_mc_q_2000")
        & source["dap"].eq(65)
        & source["seed"].isin(SEEDS)
    ].copy()
    checks = {
        "three_seeds": set(frame["seed"].astype(int)) == set(SEEDS),
        "six_actions_each_seed": bool((frame.groupby("seed").size() == 6).all()),
        "exact_expected_actions": all(
            tuple(sorted(group["action_index"].astype(int))) == EXPECTED_ACTIONS
            for _, group in frame.groupby("seed")
        ),
        "finite_q_and_target": bool(
            np.isfinite(
                frame[["mean_predicted_q_across_test_stage_states", "recorded_target_mean"]].to_numpy()
            ).all()
        ),
    }
    if not all(checks.values()):
        payload = {"status": "failed", "branch": "D_data_failure", "checks": checks}
        (OUT_ROOT / "022_09_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        raise RuntimeError(f"DAP65 input checks failed: {checks}")

    summaries = []
    for seed, group in frame.groupby("seed", sort=True):
        rho = float(
            group["mean_predicted_q_across_test_stage_states"]
            .rank(method="average")
            .corr(group["recorded_target_mean"].rank(method="average"))
        )
        action1 = group[group["action_index"].eq(1)].iloc[0]
        action7 = group[group["action_index"].eq(7)].iloc[0]
        q_best = int(group.loc[group["mean_predicted_q_across_test_stage_states"].idxmax(), "action_index"])
        target_best = int(group.loc[group["recorded_target_mean"].idxmax(), "action_index"])
        summaries.append(
            {
                "seed": int(seed),
                "dap65_q_target_spearman": rho,
                "q_best_action": q_best,
                "recorded_target_best_action": target_best,
                "q_action7_minus_action1": float(
                    action7["mean_predicted_q_across_test_stage_states"]
                    - action1["mean_predicted_q_across_test_stage_states"]
                ),
                "target_action7_minus_action1": float(
                    action7["recorded_target_mean"] - action1["recorded_target_mean"]
                ),
                "q_prefers_action1_over_action7": bool(
                    action1["mean_predicted_q_across_test_stage_states"]
                    > action7["mean_predicted_q_across_test_stage_states"]
                ),
                "spearman_improved_over_022_06_minus_0p6": rho > -0.6,
            }
        )
    summary = pd.DataFrame(summaries)
    frame.to_csv(OUT_ROOT / "022_09_dap65_seed_action_q_target.csv", index=False)
    summary.to_csv(OUT_ROOT / "022_09_dap65_seed_summary.csv", index=False)

    spearman_pass = int((summary["dap65_q_target_spearman"] >= 0.50).sum()) >= 2
    critical_pair_pass = int(summary["q_prefers_action1_over_action7"].sum()) >= 2
    all_improved = bool(summary["spearman_improved_over_022_06_minus_0p6"].all())
    if spearman_pass and critical_pair_pass:
        branch, next_step = "A_dap65_pathology_repaired", "allow_warmstart_online_draft"
    elif all_improved:
        branch, next_step = "B_partial_improvement_critical_pair_reversed", "pause_warmstart_run_controlled_dap65_action_swap"
    else:
        branch, next_step = "C_no_reliable_improvement", "stop_current_warmstart_initialization"

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (seed, group) in zip(axes, frame.groupby("seed", sort=True)):
        ax.plot(
            group["action_index"],
            group["recorded_target_mean"],
            "ko--",
            label="recorded target mean",
        )
        ax.plot(
            group["action_index"],
            group["mean_predicted_q_across_test_stage_states"],
            "o-",
            color="#1f4e79",
            label="offline Q",
        )
        rho = summary.loc[summary["seed"].eq(seed), "dap65_q_target_spearman"].iloc[0]
        ax.set_title(f"seed{seed}, rho={rho:.3f}")
        ax.set_xlabel("Action index")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Scaled value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("SY2014 DAP65 offline checkpoint ranking audit")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT_ROOT / "022_09_dap65_offline_ranking.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT_ROOT / "022_09_dap65_offline_ranking.svg", bbox_inches="tight")
    plt.close(fig)

    payload = {
        "status": "completed",
        "branch": branch,
        "next_step": next_step,
        "zero_training": True,
        "zero_dssat_calls": True,
        "checks": checks,
        "spearman_at_least_0p5_seed_count": int((summary["dap65_q_target_spearman"] >= 0.50).sum()),
        "q_prefers_action1_over_action7_seed_count": int(summary["q_prefers_action1_over_action7"].sum()),
        "seed_summaries": summary.to_dict(orient="records"),
        "guardrail": "DAP65 grouped targets are history-confounded descriptive evidence, not same-state causal counterfactuals.",
    }
    (OUT_ROOT / "022_09_result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
