from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_sy2014_stage_mc_dqn_seed1_short_022_02 as base


OUT = ROOT / "benchmark_results" / "022_17"
MANIFEST = ROOT / "benchmark_results" / "022_03" / "022_03_fixed_grid_transition_manifest.csv"
CHANGES = ROOT / "benchmark_results" / "022_16" / "022_16_dap110_mc_only_vs_pairwise_comparison.csv"
TOLERANCE = 1.0


def run_schedule(
    label: str,
    actions: tuple[int, ...],
    scaler: base.FixedObservationScaler,
    thresholds: dict[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    env = base.make_env(OUT / "runtime" / label, seed=1)
    runner = base.StageSeasonRunner(env, scaler)

    def choose(stage_index: int, _dap: int, _obs: Any, _valid: tuple[int, ...]) -> int:
        return int(actions[stage_index])

    try:
        result = runner.run(choose, label, keep_daily=True)
        metrics = base.summary_metrics(env, result, thresholds)
    finally:
        env.close()
    summary = {
        "label": label,
        "scenario": label.rsplit("__a", 1)[0],
        "dap110_requested_action": int(actions[-1]),
        "action_sequence": json.dumps(actions),
        "final_yield": result["final_yield"],
        "final_biomass": result["final_biomass"],
        "irrigation_total": result["irrigation_total"],
        "nitrogen_total": result["nitrogen_total"],
        "g0_raw": result["returns_raw"][0],
        "g0_scaled": result["returns_scaled"][0],
        **metrics,
    }
    for row in result["stage_rows"]:
        row["label"] = label
        row["scenario"] = summary["scenario"]
    for row in result["daily_rows"]:
        row["label"] = label
        row["scenario"] = summary["scenario"]
    return summary, result["stage_rows"], result["daily_rows"]


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    base.OUT_ROOT = OUT
    changes = pd.read_csv(CHANGES)
    changed = changes.loc[changes["mc_only_changed"] | changes["pairwise_mc_changed"]].copy()
    scenarios = sorted(changed["scenario"].unique().tolist())
    if len(scenarios) != 9:
        raise ValueError(f"expected 9 preregistered changed scenarios, got {len(scenarios)}")
    manifest = pd.read_csv(MANIFEST)
    scaler = base.FixedObservationScaler(base.SCALER_PATH)
    thresholds = json.loads(base.THRESHOLDS_PATH.read_text(encoding="utf-8"))

    summaries: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    daily: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        rows = manifest.loc[manifest["scenario"].astype(str) == scenario].sort_values("stage_index")
        if len(rows) != 6 or rows["dap"].astype(int).tolist() != [1, 30, 50, 65, 85, 110]:
            raise ValueError(f"{scenario}: invalid six-stage source")
        prefix = rows["action_index"].astype(int).tolist()[:5]
        arm_results: dict[int, dict[str, Any]] = {}
        arm_stages: dict[int, pd.DataFrame] = {}
        for final_action in (0, 1):
            actions = tuple(prefix + [final_action])
            label = f"{scenario}__a{final_action}"
            summary, stage_rows, daily_rows = run_schedule(label, actions, scaler, thresholds)
            summaries.append(summary)
            stages.extend(stage_rows)
            daily.extend(daily_rows)
            arm_results[final_action] = summary
            arm_stages[final_action] = pd.DataFrame(stage_rows).sort_values("stage_index")
            print(
                f"{label}: yield={summary['final_yield']:.3f} I={summary['irrigation_total']:.0f} "
                f"G0={summary['g0_raw']:.3f}",
                flush=True,
            )
        s0, s1 = arm_stages[0].iloc[-1], arm_stages[1].iloc[-1]
        execution_alias = bool(
            float(s0["executed_irrigation"]) == float(s1["executed_irrigation"])
            and float(s0["executed_nitrogen"]) == float(s1["executed_nitrogen"])
        )
        delta_g0 = float(arm_results[1]["g0_raw"] - arm_results[0]["g0_raw"])
        pair_rows.append(
            {
                "scenario": scenario,
                "prefix_actions": json.dumps(prefix),
                "pre_dap110_irrigation": float(arm_stages[0].iloc[-2]["cumulative_irrigation"]),
                "action0_executed_irrigation": float(s0["executed_irrigation"]),
                "action1_executed_irrigation": float(s1["executed_irrigation"]),
                "execution_alias": execution_alias,
                "yield_action0": float(arm_results[0]["final_yield"]),
                "yield_action1": float(arm_results[1]["final_yield"]),
                "delta_yield_a1_minus_a0": float(arm_results[1]["final_yield"] - arm_results[0]["final_yield"]),
                "g0_action0": float(arm_results[0]["g0_raw"]),
                "g0_action1": float(arm_results[1]["g0_raw"]),
                "delta_g0_a1_minus_a0": delta_g0,
                "wp_action0": float(arm_results[0]["WP_ET_kg_m3"]),
                "wp_action1": float(arm_results[1]["WP_ET_kg_m3"]),
                "pfp_action0": float(arm_results[0]["PFP_N_kg_kg"]),
                "pfp_action1": float(arm_results[1]["PFP_N_kg_kg"]),
            }
        )

    summary_df = pd.DataFrame(summaries)
    stage_df = pd.DataFrame(stages)
    daily_df = pd.DataFrame(daily)
    pair_df = pd.DataFrame(pair_rows)
    consequence_rows: list[dict[str, Any]] = []
    pair_lookup = pair_df.set_index("scenario")
    for row in changed.itertuples(index=False):
        pair = pair_lookup.loc[str(row.scenario)]
        for condition, changed_flag, new_action in (
            ("mc_only", bool(row.mc_only_changed), int(row.mc_only_action)),
            ("pairwise_mc", bool(row.pairwise_mc_changed), int(row.pairwise_mc_action)),
        ):
            if not changed_flag:
                continue
            old_action = int(row.baseline_action)
            old_g0 = float(pair[f"g0_action{old_action}"])
            new_g0 = float(pair[f"g0_action{new_action}"])
            delta = new_g0 - old_g0
            if bool(pair["execution_alias"]):
                outcome = "alias"
            elif delta > TOLERANCE:
                outcome = "improved"
            elif delta < -TOLERANCE:
                outcome = "degraded"
            else:
                outcome = "neutral"
            consequence_rows.append(
                {
                    "seed": int(row.seed),
                    "scenario": str(row.scenario),
                    "training_condition": condition,
                    "old_action": old_action,
                    "new_action": new_action,
                    "old_g0": old_g0,
                    "new_g0": new_g0,
                    "delta_g0_new_minus_old": delta,
                    "execution_alias": bool(pair["execution_alias"]),
                    "outcome": outcome,
                }
            )
    consequence_df = pd.DataFrame(consequence_rows)
    counts = consequence_df["outcome"].value_counts().to_dict()
    improved = int(counts.get("improved", 0))
    degraded = int(counts.get("degraded", 0))
    if degraded == 0 and improved >= 1:
        branch = "A_zero_change_guardrail_too_strict"
    elif degraded >= 1 and improved == 0:
        branch = "B_guardrail_detects_real_degradation"
    elif degraded >= 1 and improved >= 1:
        branch = "C_mixed_consequences"
    else:
        branch = "D_all_neutral_alias_or_failure"

    summary_df.to_csv(OUT / "022_17_season_summary.csv", index=False)
    stage_df.to_csv(OUT / "022_17_stage_actions.csv", index=False)
    daily_df.to_csv(OUT / "022_17_daily_values.csv", index=False)
    pair_df.to_csv(OUT / "022_17_controlled_action_pairs.csv", index=False)
    consequence_df.to_csv(OUT / "022_17_model_change_consequences.csv", index=False)
    result = {
        "status": "completed",
        "branch": branch,
        "scenario_count": len(scenarios),
        "dssat_seasons": len(summary_df),
        "dqn_training_steps": 0,
        "outcome_counts": {str(k): int(v) for k, v in counts.items()},
        "execution_alias_scenarios": pair_df.loc[pair_df["execution_alias"], "scenario"].tolist(),
        "tolerance_g0": TOLERANCE,
        "guardrail_changed": False,
        "warmstart_started": False,
    }
    (OUT / "022_17_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_df = pair_df.sort_values("delta_g0_a1_minus_a0")
    colors = ["#808080" if alias else ("#2E7D32" if value > 0 else "#C62828") for value, alias in zip(plot_df["delta_g0_a1_minus_a0"], plot_df["execution_alias"])]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(plot_df["scenario"], plot_df["delta_g0_a1_minus_a0"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(TOLERANCE, color="#777777", linestyle="--", linewidth=0.8)
    ax.axvline(-TOLERANCE, color="#777777", linestyle="--", linewidth=0.8)
    ax.set_xlabel("G0(action1) - G0(action0)")
    ax.set_title("SY2014 controlled DAP110 action swap (022_17)")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "022_17_dap110_controlled_reward_difference.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "022_17_dap110_controlled_reward_difference.svg", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
