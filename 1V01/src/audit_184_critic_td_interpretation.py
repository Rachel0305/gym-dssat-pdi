"""Task 184 post-hoc interpretation.

This script reads the completed task-184 critic/one-step TD tables only.
It does not train PPO and does not rerun DSSAT.  Its only job is to turn the
raw actor-vs-critic rows into a compact diagnosis that is easy to explain to a
supervisor:

1. What did the actor choose at DAP97/104?
2. Which action had the highest local one-step TD proxy?
3. Was the actor's I45 choice locally supported by the critic?

Reminder: PPO has a state-value critic V(s), not a Q(s, a) critic.  The action
score used in task 184 is the local proxy:

    TD(a) = reward_scaled(s, a) + gamma * V(s')

So this is a local critic-consistency audit, not a full-season payoff test.
Use task 182 for full-season action ablation payoff.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "184_sya_ppo_critic_td_action_value_audit"
EVAL = OUT / "evaluation"
ALIGNMENT = EVAL / "184_actor_critic_alignment.csv"
GROUP_SUMMARY = EVAL / "184_actor_critic_group_summary.csv"
TD_ROWS = EVAL / "184_action_one_step_td.csv"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def as_int(value: object) -> int:
    return int(value)


def summarize_by_checkpoint_dap(alignment: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (checkpoint_step, target_dap), group in alignment.groupby(["checkpoint_step", "target_dap"]):
        actor_i45 = group["actor_selected_pair"].eq("I45/N0")
        actor_noop = group["actor_selected_pair"].eq("I0/N0")
        td_noop = group["td_top_pair"].eq("I0/N0")
        td_i45 = group["td_top_pair"].eq("I45/N0")
        td_i30 = group["td_top_pair"].eq("I30/N0")
        aligned = group["actor_selected_is_td_top"].astype(bool)
        i45_delta = group["i45_td_delta_vs_noop_scaled"]

        rows.append(
            {
                "checkpoint_step": as_int(checkpoint_step),
                "target_dap": as_int(target_dap),
                "scenario_count": int(len(group)),
                "actor_i45_rows": int(actor_i45.sum()),
                "actor_noop_rows": int(actor_noop.sum()),
                "td_top_noop_rows": int(td_noop.sum()),
                "td_top_i45_rows": int(td_i45.sum()),
                "td_top_i30_rows": int(td_i30.sum()),
                "actor_td_aligned_rows": int(aligned.sum()),
                "i45_td_delta_positive_rows": int((i45_delta > 1e-12).sum()),
                "i45_td_delta_negative_rows": int((i45_delta < -1e-12).sum()),
                "mean_i45_td_delta_vs_noop_scaled": float(i45_delta.mean()),
                "min_i45_td_delta_vs_noop_scaled": float(i45_delta.min()),
                "max_i45_td_delta_vs_noop_scaled": float(i45_delta.max()),
                "mean_i45_reward_unscaled_delta_vs_noop": float(
                    group["i45_reward_unscaled_delta_vs_noop"].mean()
                ),
                "mean_i45_critic_value_after_delta_vs_noop": float(
                    group["i45_critic_value_after_delta_vs_noop"].mean()
                ),
                "mean_actor_i45_minus_noop_prob": float(group["actor_i45_minus_noop_prob"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["checkpoint_step", "target_dap"])


def summarize_ckpt2000_detail(alignment: pd.DataFrame) -> pd.DataFrame:
    ckpt2000 = alignment[alignment["checkpoint_step"].eq(2000)].copy()
    columns = [
        "weather_variant",
        "target_dap",
        "actor_selected_pair",
        "td_top_pair",
        "actor_selected_is_td_top",
        "i45_td_delta_vs_noop_scaled",
        "i45_reward_unscaled_delta_vs_noop",
        "i45_critic_value_after_delta_vs_noop",
        "actor_i45_minus_noop_prob",
    ]
    detail = ckpt2000[columns].copy()
    detail["local_interpretation"] = detail.apply(
        lambda row: (
            "actor_i45_supported_by_td"
            if row["actor_selected_pair"] == "I45/N0" and bool(row["actor_selected_is_td_top"])
            else (
                "actor_i45_not_td_top"
                if row["actor_selected_pair"] == "I45/N0"
                else "actor_not_i45"
            )
        ),
        axis=1,
    )
    return detail.sort_values(["target_dap", "weather_variant"])


def main() -> None:
    if not ALIGNMENT.exists():
        raise FileNotFoundError(ALIGNMENT)
    if not GROUP_SUMMARY.exists():
        raise FileNotFoundError(GROUP_SUMMARY)
    if not TD_ROWS.exists():
        raise FileNotFoundError(TD_ROWS)

    alignment = pd.read_csv(ALIGNMENT)
    group_summary = pd.read_csv(GROUP_SUMMARY)
    td_rows = pd.read_csv(TD_ROWS)

    by_checkpoint_dap = summarize_by_checkpoint_dap(alignment)
    ckpt2000_detail = summarize_ckpt2000_detail(alignment)

    by_checkpoint_dap_path = EVAL / "184_interpretation_by_checkpoint_dap.csv"
    ckpt2000_detail_path = EVAL / "184_interpretation_ckpt2000_detail.csv"
    by_checkpoint_dap.to_csv(by_checkpoint_dap_path, index=False, encoding="utf-8-sig")
    ckpt2000_detail.to_csv(ckpt2000_detail_path, index=False, encoding="utf-8-sig")

    ckpt1000 = alignment[alignment["checkpoint_step"].eq(1000)]
    ckpt2000 = alignment[alignment["checkpoint_step"].eq(2000)]
    ckpt2000_dap97 = ckpt2000[ckpt2000["target_dap"].eq(97)]
    ckpt2000_dap104 = ckpt2000[ckpt2000["target_dap"].eq(104)]

    td_top_counts_2000 = ckpt2000["td_top_pair"].value_counts().to_dict()
    actor_counts_2000 = ckpt2000["actor_selected_pair"].value_counts().to_dict()

    result = {
        "task": "184_sya_ppo_critic_td_interpretation",
        "status": "completed",
        "mode": "read_only_posthoc_interpretation",
        "inputs": {
            "alignment": rel(ALIGNMENT),
            "group_summary": rel(GROUP_SUMMARY),
            "action_one_step_td": rel(TD_ROWS),
        },
        "primary_label": "actor_i45_mostly_not_supported_by_one_step_critic_td",
        "evidence": {
            "target_state_rows": int(len(alignment)),
            "candidate_td_rows": int(len(td_rows)),
            "ckpt1000_actor_td_aligned_rows": int(ckpt1000["actor_selected_is_td_top"].astype(bool).sum()),
            "ckpt1000_rows": int(len(ckpt1000)),
            "ckpt2000_actor_selected_counts": {str(k): int(v) for k, v in actor_counts_2000.items()},
            "ckpt2000_td_top_counts": {str(k): int(v) for k, v in td_top_counts_2000.items()},
            "ckpt2000_actor_td_aligned_rows": int(ckpt2000["actor_selected_is_td_top"].astype(bool).sum()),
            "ckpt2000_rows": int(len(ckpt2000)),
            "ckpt2000_i45_td_delta_positive_rows": int(
                (ckpt2000["i45_td_delta_vs_noop_scaled"] > 1e-12).sum()
            ),
            "ckpt2000_i45_td_delta_negative_rows": int(
                (ckpt2000["i45_td_delta_vs_noop_scaled"] < -1e-12).sum()
            ),
            "ckpt2000_dap97_td_top_noop_rows": int(ckpt2000_dap97["td_top_pair"].eq("I0/N0").sum()),
            "ckpt2000_dap97_i45_td_delta_vs_noop_scaled_unique": sorted(
                {
                    round(float(x), 6)
                    for x in ckpt2000_dap97["i45_td_delta_vs_noop_scaled"].unique()
                }
            ),
            "ckpt2000_dap104_td_top_counts": {
                str(k): int(v) for k, v in ckpt2000_dap104["td_top_pair"].value_counts().items()
            },
            "ckpt2000_i45_reward_unscaled_delta_vs_noop_mean": float(
                ckpt2000["i45_reward_unscaled_delta_vs_noop"].mean()
            ),
        },
        "interpretation": {
            "what_actor_did": "At checkpoint 2000 the actor selected I45/N0 at all 12 target states: 6 weather scenarios x DAP97/104.",
            "what_local_td_said": "The one-step TD proxy selected no-op in 10/12 checkpoint-2000 states, I45/N0 in 1/12, and I30/N0 in 1/12.",
            "dap97": "DAP97 is the cleanest case: no-op had the highest one-step TD in all six weather scenarios, while actor still selected I45/N0. I45 was lower than no-op by 0.0495 scaled reward, exactly matching the 45 mm water cost penalty after reward scaling.",
            "dap104": "DAP104 was mixed: no-op was TD-best in four scenarios, I45/N0 in one scenario, and I30/N0 in one scenario. Therefore even here the critic did not broadly support the actor's universal I45 choice.",
            "checkpoint_change": "Checkpoint 1000 actor and one-step TD were aligned in all 12 states and selected no-op. By checkpoint 2000 the actor drifted to I45, but one-step TD mostly stayed with no-op.",
            "root_cause": "The immediate cause is actor/critic mismatch under near-tie policy probabilities: the deterministic actor argmax chose I45 even though the local critic TD proxy mostly did not prefer it.",
            "boundary": "This does not prove the exact training-time advantage estimates were wrong, because PPO uses rollout advantages over trajectories; it proves that the final checkpoint's local critic is not a strong explanation for the late I45 actions.",
        },
        "outputs": {
            "by_checkpoint_dap": rel(by_checkpoint_dap_path),
            "ckpt2000_detail": rel(ckpt2000_detail_path),
        },
    }

    addendum_path = OUT / "184_interpretation_addendum.json"
    addendum_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
