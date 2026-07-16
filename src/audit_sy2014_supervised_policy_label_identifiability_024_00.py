from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "024_00"
SUPPORT = ROOT / "benchmark_results" / "022_08" / "022_08_stage_action_identifiability.csv"
MANIFEST = ROOT / "benchmark_results" / "022_03" / "022_03_fixed_grid_transition_manifest.csv"
STRICT = ROOT / "benchmark_results" / "022_01" / "022_01_strict_success_candidates.csv"
PAIR65 = ROOT / "benchmark_results" / "022_19" / "022_19_dap65_causal_targets.csv"
PAIR110 = ROOT / "benchmark_results" / "022_19" / "022_19_dap110_causal_targets.csv"
MIN_SUPPORT = 2


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    support = pd.read_csv(SUPPORT)
    support = support.loc[support["primary_evaluable"].astype(str).str.lower().isin(("true", "1"))].copy()
    manifest = pd.read_csv(MANIFEST)
    strict_scenarios = set(pd.read_csv(STRICT)["scenario"].astype(str))
    causal65 = set(pd.read_csv(PAIR65)["prefix"].astype(str))
    causal110 = set(pd.read_csv(PAIR110)["scenario"].astype(str))
    causal_scenarios = causal65 | causal110
    imitation_scenarios = strict_scenarios - causal_scenarios

    if causal65 & causal110:
        raise ValueError("DAP65 and DAP110 causal scenario sets unexpectedly overlap")
    imitation = manifest.loc[manifest["scenario"].astype(str).isin(imitation_scenarios)].copy()
    counts = imitation.groupby(["dap", "action_index"]).size().rename("strict_imitation_count").reset_index()
    rows: list[dict[str, object]] = []
    for item in support.itertuples(index=False):
        dap, action = int(item.dap), int(item.action_index)
        match = counts.loc[(counts["dap"] == dap) & (counts["action_index"] == action), "strict_imitation_count"]
        imitation_count = int(match.iloc[0]) if len(match) else 0
        causal_role = "none"
        causal_count = 0
        if dap == 65 and action in (1, 7):
            causal_role, causal_count = "independent_test_pair", len(causal65)
        elif dap == 110 and action in (0, 1):
            causal_role, causal_count = "independent_test_pair", len(causal110)
        if imitation_count >= MIN_SUPPORT:
            status = "train_validation_supported"
        elif imitation_count == 1:
            status = "not_estimable_singleton"
        elif causal_count > 0:
            status = "causal_test_only_no_training_support"
        else:
            status = "unknown_no_reliable_label"
        rows.append(
            {
                "dap": dap,
                "action_index": action,
                "behavior_support_total_count": int(item.total_transition_count),
                "strict_imitation_count_after_causal_holdout": imitation_count,
                "causal_role": causal_role,
                "causal_state_count": causal_count,
                "label_status": status,
                "main_train_validation_supported": status == "train_validation_supported",
            }
        )
    coverage = pd.DataFrame(rows)
    coverage.to_csv(OUT / "024_00_stage_action_label_coverage.csv", index=False)

    stage_rows: list[dict[str, object]] = []
    for dap, group in coverage.groupby("dap", sort=True):
        stage_rows.append(
            {
                "dap": int(dap),
                "behavior_supported_action_count": int(len(group)),
                "train_validation_supported_action_count": int(group["main_train_validation_supported"].sum()),
                "causal_test_only_action_count": int((group["label_status"] == "causal_test_only_no_training_support").sum()),
                "singleton_action_count": int((group["label_status"] == "not_estimable_singleton").sum()),
                "unknown_action_count": int((group["label_status"] == "unknown_no_reliable_label").sum()),
                "all_behavior_actions_identifiable": bool(
                    group["label_status"].isin(("train_validation_supported", "causal_test_only_no_training_support")).all()
                ),
            }
        )
    stage_summary = pd.DataFrame(stage_rows)
    stage_summary.to_csv(OUT / "024_00_stage_label_summary.csv", index=False)

    leakage = bool(imitation_scenarios & causal_scenarios)
    all_identifiable = bool(stage_summary["all_behavior_actions_identifiable"].all())
    every_stage_trainable = bool((stage_summary["train_validation_supported_action_count"] >= 1).all())
    causal_test_complete = len(causal65) == 3 and len(causal110) == 8
    passed = bool(all_identifiable and every_stage_trainable and causal_test_complete and not leakage)
    branch = "A_labels_identifiable" if passed else "C_labels_insufficient"
    result = {
        "status": "completed",
        "branch": branch,
        "strict_success_scenarios_total": len(strict_scenarios),
        "causal_test_scenarios_total": len(causal_scenarios),
        "strict_success_scenarios_remaining_for_imitation": len(imitation_scenarios),
        "dap65_causal_test_states": len(causal65),
        "dap110_causal_test_states": len(causal110),
        "behavior_supported_actions_total": int(len(coverage)),
        "train_validation_supported_actions": int(coverage["main_train_validation_supported"].sum()),
        "singleton_actions": int((coverage["label_status"] == "not_estimable_singleton").sum()),
        "unknown_actions": int((coverage["label_status"] == "unknown_no_reliable_label").sum()),
        "causal_test_only_actions": int((coverage["label_status"] == "causal_test_only_no_training_support").sum()),
        "all_behavior_actions_identifiable": all_identifiable,
        "every_stage_has_trainable_label": every_stage_trainable,
        "causal_test_complete": causal_test_complete,
        "scenario_leakage": leakage,
        "formal_supervised_training_steps": 0,
        "dssat_calls": 0,
        "next_step_allowed": passed,
        "scope": "Label provenance and coverage only; historical MC associations are not treated as causal labels.",
    }
    (OUT / "024_00_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    order = ["train_validation_supported", "causal_test_only_no_training_support", "not_estimable_singleton", "unknown_no_reliable_label"]
    colors = ["#2f6f4e", "#3b6fb6", "#d49a00", "#b22222"]
    pivot = coverage.groupby(["dap", "label_status"]).size().unstack(fill_value=0).reindex(columns=order, fill_value=0)
    ax = pivot.plot(kind="bar", stacked=True, color=colors, figsize=(9, 4.8))
    ax.set(xlabel="Stage DAP", ylabel="Behavior-supported actions", title="SY2014 supervised-label identifiability")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    ax.grid(axis="y", alpha=0.2)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(OUT / "024_00_label_identifiability.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "024_00_label_identifiability.svg", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
