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


OUT_ROOT = ROOT / "benchmark_results" / "022_10"
CONTROL_ACTIONS = (3, 4, 7, 1, 1, 0)
TREATMENT_ACTIONS = (3, 4, 7, 7, 1, 0)
REFERENCE_YIELD = 11202.00439453125


def run_arm(
    arm: str,
    actions: tuple[int, ...],
    scaler: base.FixedObservationScaler,
    thresholds: dict[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    env = base.make_env(OUT_ROOT / "runtime" / arm, seed=1)
    runner = base.StageSeasonRunner(env, scaler)

    def choose(stage_index: int, _dap: int, _obs: Any, _valid: tuple[int, ...]) -> int:
        return int(actions[stage_index])

    try:
        result = runner.run(choose, arm, keep_daily=True)
        metrics = base.summary_metrics(env, result, thresholds)
    finally:
        env.close()
    row = {
        "arm": arm,
        "action_sequence": json.dumps(actions),
        "final_yield": result["final_yield"],
        "final_biomass": result["final_biomass"],
        "irrigation_total": result["irrigation_total"],
        "nitrogen_total": result["nitrogen_total"],
        "g0_raw": result["returns_raw"][0],
        "g0_scaled": result["returns_scaled"][0],
        **metrics,
    }
    for stage in result["stage_rows"]:
        stage["arm"] = arm
    for daily in result["daily_rows"]:
        daily["arm"] = arm
    return row, result["stage_rows"], result["daily_rows"]


def main() -> None:
    if OUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT_ROOT}")
    OUT_ROOT.mkdir(parents=True)
    base.OUT_ROOT = OUT_ROOT
    scaler = base.FixedObservationScaler(base.SCALER_PATH)
    thresholds = json.loads(base.THRESHOLDS_PATH.read_text(encoding="utf-8"))
    summaries: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    daily: list[dict[str, Any]] = []
    for arm, actions in (
        ("control_action1_I15_N0", CONTROL_ACTIONS),
        ("treatment_action7_I15_N100", TREATMENT_ACTIONS),
    ):
        summary, stage_rows, daily_rows = run_arm(arm, actions, scaler, thresholds)
        summaries.append(summary)
        stages.extend(stage_rows)
        daily.extend(daily_rows)
        print(
            f"{arm}: yield={summary['final_yield']:.3f} I={summary['irrigation_total']:.0f} "
            f"N={summary['nitrogen_total']:.0f} G0={summary['g0_raw']:.3f}",
            flush=True,
        )

    summary_df = pd.DataFrame(summaries)
    stage_df = pd.DataFrame(stages)
    daily_df = pd.DataFrame(daily)
    control = summary_df.iloc[0]
    treatment = summary_df.iloc[1]
    control_stages = stage_df[stage_df["arm"].eq(control["arm"])].sort_values("stage_index")
    treatment_stages = stage_df[stage_df["arm"].eq(treatment["arm"])].sort_values("stage_index")
    non65 = control_stages["dap"].ne(65).to_numpy()
    checks = {
        "control_replays_022_01_within_1kg": abs(float(control["final_yield"]) - REFERENCE_YIELD) <= 1.0,
        "six_stages_each": len(control_stages) == 6 and len(treatment_stages) == 6,
        "only_selected_action_difference_is_dap65": bool(
            (control_stages.loc[non65, "action_index"].to_numpy() == treatment_stages.loc[non65, "action_index"].to_numpy()).all()
            and int(control_stages.loc[control_stages["dap"].eq(65), "action_index"].iloc[0]) == 1
            and int(treatment_stages.loc[treatment_stages["dap"].eq(65), "action_index"].iloc[0]) == 7
        ),
        "dap65_prebudget_i30_n200": bool(
            float(control_stages.loc[control_stages["dap"].eq(50), "cumulative_irrigation"].iloc[0]) == 30.0
            and float(control_stages.loc[control_stages["dap"].eq(50), "cumulative_nitrogen"].iloc[0]) == 200.0
        ),
        "dap65_actions_execute_without_clipping": bool(
            float(control_stages.loc[control_stages["dap"].eq(65), "executed_nitrogen"].iloc[0]) == 0.0
            and float(treatment_stages.loc[treatment_stages["dap"].eq(65), "executed_nitrogen"].iloc[0]) == 100.0
            and float(control_stages.loc[control_stages["dap"].eq(65), "executed_irrigation"].iloc[0]) == 15.0
            and float(treatment_stages.loc[treatment_stages["dap"].eq(65), "executed_irrigation"].iloc[0]) == 15.0
        ),
        "same_irrigation_i60": float(control["irrigation_total"]) == 60.0 and float(treatment["irrigation_total"]) == 60.0,
        "expected_n200_vs_n300": float(control["nitrogen_total"]) == 200.0 and float(treatment["nitrogen_total"]) == 300.0,
        "summary_matches_both": float(control["summary_match_score"]) <= 1.0 and float(treatment["summary_match_score"]) <= 1.0,
    }
    if not all(checks.values()):
        payload = {"status": "failed", "branch": "D_implementation_failure", "checks": checks}
        (OUT_ROOT / "022_10_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        raise RuntimeError(f"Controlled comparison checks failed: {checks}")

    delta = {
        "treatment_minus_control_yield": float(treatment["final_yield"] - control["final_yield"]),
        "treatment_minus_control_biomass": float(treatment["final_biomass"] - control["final_biomass"]),
        "treatment_minus_control_irrigation": float(treatment["irrigation_total"] - control["irrigation_total"]),
        "treatment_minus_control_nitrogen": float(treatment["nitrogen_total"] - control["nitrogen_total"]),
        "treatment_minus_control_g0_raw": float(treatment["g0_raw"] - control["g0_raw"]),
        "treatment_minus_control_wp_et": float(treatment["WP_ET_kg_m3"] - control["WP_ET_kg_m3"]),
        "treatment_minus_control_pfp_n": float(treatment["PFP_N_kg_kg"] - control["PFP_N_kg_kg"]),
    }
    both_yield_pass = bool(control["final_yield"] >= thresholds["yield_min"] and treatment["final_yield"] >= thresholds["yield_min"])
    if both_yield_pass and delta["treatment_minus_control_g0_raw"] < 0 and delta["treatment_minus_control_yield"] <= 0:
        branch = "A_action1_dominates_at_fixed_state"
    elif delta["treatment_minus_control_g0_raw"] > 0:
        branch = "B_action7_dominates_at_fixed_state"
    elif delta["treatment_minus_control_yield"] > 0 and delta["treatment_minus_control_g0_raw"] < 0:
        branch = "C_yield_gain_but_reward_efficiency_loss"
    else:
        branch = "A_action1_reward_dominates_with_negligible_yield_difference"

    summary_df.to_csv(OUT_ROOT / "022_10_controlled_summary.csv", index=False)
    stage_df.to_csv(OUT_ROOT / "022_10_controlled_stage_actions.csv", index=False)
    daily_df.to_csv(OUT_ROOT / "022_10_controlled_daily_values.csv", index=False)
    pd.DataFrame([delta]).to_csv(OUT_ROOT / "022_10_treatment_minus_control.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    labels = ["action1\nI60/N200", "action7\nI60/N300"]
    axes[0].bar(labels, summary_df["final_yield"], color=["#0072B2", "#D55E00"])
    axes[0].axhline(thresholds["yield_min"], color="#333333", linestyle=":")
    axes[0].set_ylabel("HWAM (kg/ha)")
    axes[1].bar(labels, summary_df["g0_raw"], color=["#0072B2", "#D55E00"])
    axes[1].set_ylabel("G0 raw reward")
    axes[2].bar(labels, summary_df["PFP_N_kg_kg"], color=["#0072B2", "#D55E00"])
    axes[2].set_ylabel("PFP-N (kg/kg)")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("SY2014 controlled DAP65 action swap")
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "022_10_dap65_action_swap.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT_ROOT / "022_10_dap65_action_swap.svg", bbox_inches="tight")
    plt.close(fig)

    payload = {
        "status": "completed",
        "branch": branch,
        "zero_training": True,
        "dssat_forward_seasons": 2,
        "checks": checks,
        "control": summaries[0],
        "treatment": summaries[1],
        "treatment_minus_control": delta,
    }
    (OUT_ROOT / "022_10_result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
