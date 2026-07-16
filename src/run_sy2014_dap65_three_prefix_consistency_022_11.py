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


OUT_ROOT = ROOT / "benchmark_results" / "022_11"
RESULT_022_10 = ROOT / "benchmark_results" / "022_10" / "022_10_result.json"
MATERIAL_G0_THRESHOLD = -250.0
PREFIXES = {
    "prefix_b_w75_uniform_pre90": {
        "control": (4, 4, 7, 1, 1, 0),
        "treatment": (4, 4, 7, 7, 1, 0),
        "pre_dap65_i": 45.0,
        "pre_dap65_n": 200.0,
        "final_i": 75.0,
    },
    "prefix_c_w120_critical_context": {
        "control": (3, 5, 8, 1, 2, 0),
        "treatment": (3, 5, 8, 7, 2, 0),
        "pre_dap65_i": 60.0,
        "pre_dap65_n": 200.0,
        "final_i": 105.0,
    },
}


def run_schedule(
    label: str,
    actions: tuple[int, ...],
    scaler: base.FixedObservationScaler,
    thresholds: dict[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    env = base.make_env(OUT_ROOT / "runtime" / label, seed=1)
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
    for row in result["daily_rows"]:
        row["label"] = label
    return summary, result["stage_rows"], result["daily_rows"]


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
    pair_rows: list[dict[str, Any]] = []
    implementation_checks: dict[str, bool] = {}

    for prefix, spec in PREFIXES.items():
        pair_summaries = {}
        pair_stages = {}
        for arm in ("control", "treatment"):
            label = f"{prefix}__{arm}"
            summary, stage_rows, daily_rows = run_schedule(label, spec[arm], scaler, thresholds)
            summary["prefix"] = prefix
            summary["arm"] = arm
            summaries.append(summary)
            stages.extend(stage_rows)
            daily.extend(daily_rows)
            pair_summaries[arm] = summary
            pair_stages[arm] = pd.DataFrame(stage_rows).sort_values("stage_index")
            print(
                f"{label}: yield={summary['final_yield']:.3f} I={summary['irrigation_total']:.0f} "
                f"N={summary['nitrogen_total']:.0f} G0={summary['g0_raw']:.3f}",
                flush=True,
            )

        control = pair_summaries["control"]
        treatment = pair_summaries["treatment"]
        cs = pair_stages["control"]
        ts = pair_stages["treatment"]
        non65 = cs["dap"].ne(65).to_numpy()
        checks = {
            "six_stages": len(cs) == 6 and len(ts) == 6,
            "only_action_difference_dap65": bool(
                (cs.loc[non65, "action_index"].to_numpy() == ts.loc[non65, "action_index"].to_numpy()).all()
                and int(cs.loc[cs["dap"].eq(65), "action_index"].iloc[0]) == 1
                and int(ts.loc[ts["dap"].eq(65), "action_index"].iloc[0]) == 7
            ),
            "prebudget_matches": bool(
                float(cs.loc[cs["dap"].eq(50), "cumulative_irrigation"].iloc[0]) == spec["pre_dap65_i"]
                and float(cs.loc[cs["dap"].eq(50), "cumulative_nitrogen"].iloc[0]) == spec["pre_dap65_n"]
            ),
            "dap65_no_clipping": bool(
                float(cs.loc[cs["dap"].eq(65), "executed_irrigation"].iloc[0]) == 15.0
                and float(ts.loc[ts["dap"].eq(65), "executed_irrigation"].iloc[0]) == 15.0
                and float(cs.loc[cs["dap"].eq(65), "executed_nitrogen"].iloc[0]) == 0.0
                and float(ts.loc[ts["dap"].eq(65), "executed_nitrogen"].iloc[0]) == 100.0
            ),
            "same_expected_irrigation": float(control["irrigation_total"]) == spec["final_i"]
            and float(treatment["irrigation_total"]) == spec["final_i"],
            "n200_vs_n300": float(control["nitrogen_total"]) == 200.0
            and float(treatment["nitrogen_total"]) == 300.0,
            "summary_match": float(control["summary_match_score"]) <= 1.0
            and float(treatment["summary_match_score"]) <= 1.0,
        }
        for key, value in checks.items():
            implementation_checks[f"{prefix}__{key}"] = value
        pair_rows.append(
            {
                "prefix": prefix,
                "control_yield": control["final_yield"],
                "treatment_yield": treatment["final_yield"],
                "delta_yield": treatment["final_yield"] - control["final_yield"],
                "control_biomass": control["final_biomass"],
                "treatment_biomass": treatment["final_biomass"],
                "delta_biomass": treatment["final_biomass"] - control["final_biomass"],
                "control_g0": control["g0_raw"],
                "treatment_g0": treatment["g0_raw"],
                "delta_g0": treatment["g0_raw"] - control["g0_raw"],
                "control_wp": control["WP_ET_kg_m3"],
                "treatment_wp": treatment["WP_ET_kg_m3"],
                "control_pfp": control["PFP_N_kg_kg"],
                "treatment_pfp": treatment["PFP_N_kg_kg"],
                "both_yield_pass": bool(
                    control["final_yield"] >= thresholds["yield_min"]
                    and treatment["final_yield"] >= thresholds["yield_min"]
                ),
                "material_action1_dominance": bool(
                    control["final_yield"] >= thresholds["yield_min"]
                    and treatment["final_yield"] >= thresholds["yield_min"]
                    and treatment["g0_raw"] - control["g0_raw"] <= MATERIAL_G0_THRESHOLD
                ),
            }
        )

    if not all(implementation_checks.values()):
        payload = {"status": "failed", "branch": "D_implementation_failure", "checks": implementation_checks}
        (OUT_ROOT / "022_11_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        raise RuntimeError(f"022_11 implementation checks failed: {implementation_checks}")

    previous = json.loads(RESULT_022_10.read_text(encoding="utf-8"))
    p = previous["treatment_minus_control"]
    previous_row = {
        "prefix": "prefix_a_w60_critical_022_10",
        "control_yield": previous["control"]["final_yield"],
        "treatment_yield": previous["treatment"]["final_yield"],
        "delta_yield": p["treatment_minus_control_yield"],
        "control_biomass": previous["control"]["final_biomass"],
        "treatment_biomass": previous["treatment"]["final_biomass"],
        "delta_biomass": p["treatment_minus_control_biomass"],
        "control_g0": previous["control"]["g0_raw"],
        "treatment_g0": previous["treatment"]["g0_raw"],
        "delta_g0": p["treatment_minus_control_g0_raw"],
        "control_wp": previous["control"]["WP_ET_kg_m3"],
        "treatment_wp": previous["treatment"]["WP_ET_kg_m3"],
        "control_pfp": previous["control"]["PFP_N_kg_kg"],
        "treatment_pfp": previous["treatment"]["PFP_N_kg_kg"],
        "both_yield_pass": bool(previous["control"]["primary_pass"] and previous["treatment"]["primary_pass"]),
        "material_action1_dominance": bool(
            previous["control"]["final_yield"] >= thresholds["yield_min"]
            and previous["treatment"]["final_yield"] >= thresholds["yield_min"]
            and p["treatment_minus_control_g0_raw"] <= MATERIAL_G0_THRESHOLD
        ),
    }
    combined = pd.DataFrame([previous_row, *pair_rows])
    consistent_count = int(combined["material_action1_dominance"].sum())
    if consistent_count == 3:
        branch = "A_three_of_three_material_action1_dominance"
        next_step = "allow_one_offline_pairwise_mc_gradient_conflict_audit"
    else:
        branch = "B_context_dependent_stop_pairwise_route"
        next_step = "stop_no_fourth_prefix"

    pd.DataFrame(summaries).to_csv(OUT_ROOT / "022_11_new_prefix_summary.csv", index=False)
    pd.DataFrame(stages).to_csv(OUT_ROOT / "022_11_new_prefix_stage_actions.csv", index=False)
    pd.DataFrame(daily).to_csv(OUT_ROOT / "022_11_new_prefix_daily_values.csv", index=False)
    combined.to_csv(OUT_ROOT / "022_11_three_prefix_consistency.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    labels = ["W60 critical", "W75 uniform", "W120 critical"]
    axes[0].bar(labels, combined["delta_yield"], color="#1f4e79")
    axes[0].axhline(0, color="#333333", linewidth=0.8)
    axes[0].set_ylabel("Action7 - action1 yield (kg/ha)")
    axes[1].bar(labels, combined["delta_g0"], color="#D55E00")
    axes[1].axhline(MATERIAL_G0_THRESHOLD, color="#b22222", linestyle=":", label="-250 criterion")
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_ylabel("Action7 - action1 G0")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("SY2014 DAP65 action1/action7 three-prefix consistency")
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "022_11_three_prefix_consistency.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT_ROOT / "022_11_three_prefix_consistency.svg", bbox_inches="tight")
    plt.close(fig)

    payload = {
        "status": "completed",
        "branch": branch,
        "next_step": next_step,
        "dssat_forward_seasons_this_task": 4,
        "dqn_training_updates": 0,
        "material_g0_threshold": MATERIAL_G0_THRESHOLD,
        "implementation_checks": implementation_checks,
        "material_action1_dominance_count": consistent_count,
        "total_prefix_count": 3,
        "three_prefix_rows": combined.to_dict(orient="records"),
        "mechanism_guardrail": "Biomass gain without grain gain supports but does not prove source-sink saturation.",
    }
    (OUT_ROOT / "022_11_result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
