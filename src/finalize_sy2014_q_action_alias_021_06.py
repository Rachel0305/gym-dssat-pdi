from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "021_06"


def main() -> None:
    q = pd.read_csv(OUT / "021_06_q_values_long.csv")
    flips = pd.read_csv(OUT / "021_06_rank_flip_summary.csv")
    alias = pd.read_csv(OUT / "021_06_action_alias_summary.csv")

    comparison = q[q["evaluated_checkpoint"].isin([15000, 20000])].copy()
    argmax = comparison.loc[
        comparison.groupby(["state_id", "evaluated_checkpoint"])["online_q"].idxmax(),
        [
            "state_id",
            "source_checkpoint",
            "target_dap",
            "actual_dap",
            "evaluated_checkpoint",
            "action_index",
            "requested_irrigation",
            "requested_nitrogen",
            "online_q",
        ],
    ].copy()
    argmax = argmax.merge(
        alias[["state_id", "unique_executed_action_count", "alias_fraction"]],
        on="state_id",
        how="left",
    )
    argmax.to_csv(OUT / "021_06_online_argmax_by_fixed_state.csv", index=False, encoding="utf-8-sig")

    wide = argmax.pivot(index="state_id", columns="evaluated_checkpoint", values="action_index")
    alias_by_state = alias.set_index("state_id")
    changed = wide[15000].ne(wide[20000])
    unaliased = alias_by_state.loc[wide.index, "unique_executed_action_count"].eq(9)

    joined = flips.merge(
        alias[["state_id", "unique_executed_action_count"]], on="state_id", how="left"
    )
    target = comparison.pivot(
        index=["state_id", "action_index"], columns="evaluated_checkpoint", values="target_q"
    )
    target_max_difference = float((target[20000] - target[15000]).abs().max())

    initial = argmax[argmax["state_id"].eq("ref15000_target1_dap1")].set_index(
        "evaluated_checkpoint"
    )
    evidence = {
        "online_rank_flips_total": int(
            flips.query("network == 'online' and rank_flip == True").shape[0]
        ),
        "online_robust_rank_flips_total": int(
            flips.query("network == 'online' and robust_rank_flip == True").shape[0]
        ),
        "online_rank_flips_in_unaliased_states": int(
            joined.query(
                "network == 'online' and rank_flip == True and unique_executed_action_count == 9"
            ).shape[0]
        ),
        "online_robust_rank_flips_in_unaliased_states": int(
            joined.query(
                "network == 'online' and robust_rank_flip == True and unique_executed_action_count == 9"
            ).shape[0]
        ),
        "target_rank_flips_total": int(
            flips.query("network == 'target' and rank_flip == True").shape[0]
        ),
        "target_q_max_abs_difference_15k_vs_20k": target_max_difference,
        "online_argmax_changed_states": int(changed.sum()),
        "online_argmax_changed_unaliased_states": int((changed & unaliased).sum()),
        "states_total": int(len(wide)),
        "states_with_action_aliasing": int(
            alias["unique_executed_action_count"].lt(9).sum()
        ),
        "states_all_requests_alias_to_one": int(
            alias["unique_executed_action_count"].eq(1).sum()
        ),
        "initial_dap1_argmax_15k": {
            "action_index": int(initial.loc[15000, "action_index"]),
            "irrigation": float(initial.loc[15000, "requested_irrigation"]),
            "nitrogen": float(initial.loc[15000, "requested_nitrogen"]),
        },
        "initial_dap1_argmax_20k": {
            "action_index": int(initial.loc[20000, "action_index"]),
            "irrigation": float(initial.loc[20000, "requested_irrigation"]),
            "nitrogen": float(initial.loc[20000, "requested_nitrogen"]),
        },
        "interpretation": (
            "Both online-Q ordering drift and state-dependent action aliasing are present. "
            "The DAP1 argmax changes from action 8 (I30/N100) at 15K to action 0 "
            "(I0/N0) at 20K while all nine actions are executable, so wrapper aliasing alone "
            "cannot explain nitrogen abandonment. Target-Q values are identical between the "
            "15K and 20K checkpoints on all fixed states, localizing the observed ordering "
            "change to the online network during this interval; causality remains unproven."
        ),
    }
    (OUT / "021_06_decisive_evidence_summary.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(evidence, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
