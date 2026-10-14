"""Task 183 interpretation addendum.

This post-hoc audit reads completed task-183 tables only.  It turns raw masks,
probabilities, checkpoint changes, and task-182 ablation references into a
concise root-cause interpretation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "183_sya_ppo_decision_point_rootcause_audit"
DECISIONS = OUT / "evaluation" / "183_decision_state_observations.csv"
SUMMARY = OUT / "evaluation" / "183_policy_concentration_summary.csv"
CHECKPOINTS = OUT / "evaluation" / "183_checkpoint_comparison.csv"
WEATHER_L1 = OUT / "evaluation" / "183_weather_scenario_probability_l1.csv"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> None:
    decisions = pd.read_csv(DECISIONS)
    summary = pd.read_csv(SUMMARY)
    checkpoints = pd.read_csv(CHECKPOINTS)
    weather_l1 = pd.read_csv(WEATHER_L1)

    ckpt2000 = decisions[decisions["checkpoint_step"].eq(2000)].copy()
    ckpt1000 = decisions[decisions["checkpoint_step"].eq(1000)].copy()
    ckpt2000["i45_minus_noop_prob"] = ckpt2000["prob_action12_i45_n0"] - ckpt2000["prob_action0_noop"]
    ckpt1000["noop_minus_i45_prob"] = ckpt1000["prob_action0_noop"] - ckpt1000["prob_action12_i45_n0"]

    by_dap = (
        ckpt2000.groupby("target_dap", as_index=False)
        .agg(
            rows=("weather_variant", "count"),
            selected_i45_rows=("selected_action", lambda s: int((s == 12).sum())),
            noop_legal_rows=("noop_legal", lambda s: int(pd.Series(s).astype(bool).sum())),
            i45_legal_rows=("i45_n0_legal", lambda s: int(pd.Series(s).astype(bool).sum())),
            mean_top_probability=("top_probability", "mean"),
            min_top_probability=("top_probability", "min"),
            max_top_probability=("top_probability", "max"),
            mean_normalized_entropy=("legal_policy_entropy_normalized", "mean"),
            mean_i45_minus_noop_prob=("i45_minus_noop_prob", "mean"),
            min_i45_minus_noop_prob=("i45_minus_noop_prob", "min"),
            max_i45_minus_noop_prob=("i45_minus_noop_prob", "max"),
            mean_drop_event_profit_delta=("task182_delta_profit_when_event_removed", "mean"),
            min_drop_event_profit_delta=("task182_delta_profit_when_event_removed", "min"),
            max_drop_event_profit_delta=("task182_delta_profit_when_event_removed", "max"),
        )
    )
    by_dap_1000 = (
        ckpt1000.groupby("target_dap", as_index=False)
        .agg(
            selected_noop_rows=("selected_action", lambda s: int((s == 0).sum())),
            mean_top_probability=("top_probability", "mean"),
            mean_noop_minus_i45_prob=("noop_minus_i45_prob", "mean"),
        )
    )
    by_dap = by_dap.merge(by_dap_1000, on="target_dap", how="left", suffixes=("_ckpt2000", "_ckpt1000"))

    weather_summary = (
        weather_l1.groupby(["checkpoint_step", "target_dap"], as_index=False)
        .agg(
            max_probability_l1_vs_original=("probability_l1_vs_original", "max"),
            all_masks_same=("mask_same_as_original", lambda s: bool(pd.Series(s).astype(bool).all())),
        )
    )

    changed_rows = int(checkpoints["selected_action_changed"].astype(bool).sum())
    task182_profit_positive_rows = int((ckpt2000["task182_delta_profit_when_event_removed"] > 1e-6).sum())
    mask_forced_rows = int((ckpt2000["selected_action"].eq(12) & ~ckpt2000["noop_legal"].astype(bool)).sum())
    selected_i45_noop_legal_rows = int((ckpt2000["selected_action"].eq(12) & ckpt2000["noop_legal"].astype(bool)).sum())
    high_concentration_rows = int((ckpt2000["top_probability"] >= 0.7).sum())

    by_dap.to_csv(
        OUT / "evaluation" / "183_interpretation_by_dap.csv",
        index=False,
        encoding="utf-8-sig",
    )
    weather_summary.to_csv(
        OUT / "evaluation" / "183_interpretation_weather_probability_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    result = {
        "task": "183_sya_ppo_decision_point_interpretation",
        "status": "completed",
        "mode": "read_only_posthoc_interpretation",
        "inputs": {
            "decisions": rel(DECISIONS),
            "policy_summary": rel(SUMMARY),
            "checkpoint_comparison": rel(CHECKPOINTS),
            "weather_probability_l1": rel(WEATHER_L1),
        },
        "primary_label": "near_uniform_policy_argmax_drift_to_low_value_i45",
        "evidence": {
            "ckpt2000_selected_i45_noop_legal_rows": selected_i45_noop_legal_rows,
            "ckpt2000_mask_forced_i45_rows": mask_forced_rows,
            "ckpt2000_high_concentration_rows_top_prob_ge_0p7": high_concentration_rows,
            "checkpoint1000_to_2000_selected_action_changed_rows": changed_rows,
            "ckpt2000_task182_drop_event_profit_positive_rows": task182_profit_positive_rows,
            "ckpt2000_top_probability_min": float(ckpt2000["top_probability"].min()),
            "ckpt2000_top_probability_max": float(ckpt2000["top_probability"].max()),
            "ckpt2000_normalized_entropy_min": float(ckpt2000["legal_policy_entropy_normalized"].min()),
            "ckpt2000_normalized_entropy_max": float(ckpt2000["legal_policy_entropy_normalized"].max()),
            "ckpt2000_i45_minus_noop_prob_min": float(ckpt2000["i45_minus_noop_prob"].min()),
            "ckpt2000_i45_minus_noop_prob_max": float(ckpt2000["i45_minus_noop_prob"].max()),
            "weather_probability_l1_max": float(weather_l1["probability_l1_vs_original"].max()),
        },
        "interpretation": {
            "mask": "No-op and I45/N0 were legal at all checkpoint-2000 DAP97/104 states; I45 was not mask-forced.",
            "concentration": "The policy was almost flat over the four legal actions: top probabilities were about 0.264-0.268 and normalized entropy was about 0.997. This is not a strong one-action collapse.",
            "training_process": "All 12 scenario-DAP rows changed from no-op at checkpoint 1000 to I45/N0 at checkpoint 2000 through small probability shifts.",
            "weather_state_use": "Masks and action rankings were invariant across weather scenarios; probability L1 distance from original was at most 0.000197. At these DAPs, current rain was identical across scenarios and residual crop-state differences barely affected probabilities.",
            "payoff_conflict": "Task 182 showed that removing the checkpoint-2000 DAP97/104 irrigation events improved profit/reward in all 12 event-level references.",
            "root_cause": "The immediate cause is a learned near-tie deterministic argmax drift toward I45/N0, not mask forcing. The deeper cause is likely weak or ambiguous value learning/credit assignment under this short 2K training setup.",
        },
        "outputs": {
            "by_dap": rel(OUT / "evaluation" / "183_interpretation_by_dap.csv"),
            "weather_probability_summary": rel(OUT / "evaluation" / "183_interpretation_weather_probability_summary.csv"),
        },
    }
    (OUT / "183_interpretation_addendum.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
