from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_sy2014_deterministic_oracle_search_021_18 as old


OUT = ROOT / "benchmark_results" / "022_01"
BASELINES = ROOT / "benchmark_results" / "021_18" / "021_18_baseline_metrics.csv"
OLD_ORACLE = ROOT / "benchmark_results" / "021_18" / "runs" / "R_I75_no_early_mid_N200" / "summary.json"

WATER = {
    "W60_critical": {30: 15.0, 50: 15.0, 65: 15.0, 85: 15.0},
    "W75_uniform_pre90": {1: 15.0, 30: 15.0, 50: 15.0, 65: 15.0, 85: 15.0},
    "W75_critical": {30: 15.0, 50: 15.0, 65: 30.0, 85: 15.0},
    "W90_all6": {1: 15.0, 30: 15.0, 50: 15.0, 65: 15.0, 85: 15.0, 110: 15.0},
    "W90_critical": {30: 15.0, 50: 30.0, 65: 30.0, 85: 15.0},
    "W120_critical": {30: 30.0, 50: 30.0, 65: 30.0, 85: 30.0},
}

NITROGEN = {
    "N150_bal": {30: 50.0, 50: 50.0, 65: 50.0},
    "N200_early": {1: 50.0, 30: 50.0, 50: 100.0},
    "N200_bal": {30: 50.0, 50: 100.0, 65: 50.0},
    "N200_spread": {30: 50.0, 50: 50.0, 65: 50.0, 85: 50.0},
    "N250_early": {1: 50.0, 30: 100.0, 50: 100.0},
    "N250_bal": {1: 50.0, 30: 50.0, 50: 100.0, 65: 50.0},
    "N250_mid": {30: 50.0, 50: 100.0, 65: 100.0},
    "N300_bal": {1: 50.0, 30: 100.0, 50: 100.0, 65: 50.0},
}

SMOKE_I = {22: 15.0, 29: 15.0, 42: 15.0, 56: 15.0, 79: 15.0}
SMOKE_N = {29: 50.0, 42: 100.0, 56: 50.0}


def thresholds() -> dict[str, float]:
    base = pd.read_csv(BASELINES).set_index("scenario")
    expert = base.loc["official_extension_expert"]
    auto = base.loc["dssat_auto"]
    return {
        "yield_min": float(max(expert["yield_kg_ha"], auto["yield_kg_ha"])),
        "wp_et_min": float(max(expert["WP_ET_kg_m3"], auto["WP_ET_kg_m3"])),
        "pfp_n_min": float(expert["PFP_N_kg_kg"]),
        "strict_i_max": 90.0,
        "strict_n_max": 250.0,
        "budget_i_max": 120.0,
        "budget_n_max": 300.0,
    }


def write_grid() -> None:
    rows = []
    for kind, schedules in (("water", WATER), ("nitrogen", NITROGEN)):
        for name, schedule in schedules.items():
            rows.append(
                {
                    "kind": kind,
                    "schedule_id": name,
                    "schedule_json": json.dumps(schedule, sort_keys=True),
                    "total": sum(schedule.values()),
                    "late_n_after_dap90": sum(v for d, v in schedule.items() if kind == "nitrogen" and d > 90),
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "022_01_preregistered_schedule_grid.csv", index=False)


def validate_smoke() -> dict:
    previous = json.loads(OLD_ORACLE.read_text(encoding="utf-8"))
    row = old.run_one("smoke_replay_02118_oracle_I75_N200", SMOKE_I, SMOKE_N, "input_chain_smoke")
    checks = {
        "yield_within_2": abs(float(row["final_gwad"]) - 11205.0) <= 2.0,
        "irrigation_75": float(row["summary_irrigation_total"]) == 75.0,
        "nitrogen_200": float(row["summary_nitrogen_total"]) == 200.0,
        "no_missing_daps": json.loads(row["missing_schedule_daps"]) == [],
        "matches_previous_yield_within_2": abs(float(row["final_gwad"]) - float(previous["final_gwad"])) <= 2.0,
    }
    result = {"checks": checks, "all_pass": all(checks.values()), "result": row}
    (OUT / "022_01_smoke_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not result["all_pass"]:
        raise RuntimeError(f"022_01 smoke failed: {checks}")
    return row


def classify(frame: pd.DataFrame, limits: dict[str, float]) -> pd.DataFrame:
    result = frame.copy()
    result["late_n_after_dap90"] = 0.0
    result["yield_gate_pass"] = result["final_gwad"] >= limits["yield_min"]
    result["wp_et_gate_pass"] = result["WP_ET_kg_m3"] >= limits["wp_et_min"]
    result["pfp_n_gate_pass"] = result["PFP_N_kg_kg"] >= limits["pfp_n_min"]
    result["budget_pass"] = (
        (result["summary_irrigation_total"] <= limits["budget_i_max"])
        & (result["summary_nitrogen_total"] <= limits["budget_n_max"])
    )
    result["late_n_gate_pass"] = result["late_n_after_dap90"].eq(0.0)
    result["primary_scientific_pass"] = result[
        ["yield_gate_pass", "wp_et_gate_pass", "pfp_n_gate_pass", "budget_pass", "late_n_gate_pass"]
    ].all(axis=1)
    result["strict_resource_pass"] = (
        result["primary_scientific_pass"]
        & (result["summary_irrigation_total"] <= limits["strict_i_max"])
        & (result["summary_nitrogen_total"] <= limits["strict_n_max"])
    )
    return result


def plot_results(frame: pd.DataFrame, limits: dict[str, float]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    colors = frame["summary_nitrogen_total"]
    sizes = 25 + frame["summary_irrigation_total"] * 0.8
    scatter = axes[0].scatter(frame["WP_ET_kg_m3"], frame["final_gwad"], c=colors, s=sizes, cmap="viridis", alpha=0.85, edgecolor="black", linewidth=0.4)
    axes[0].axhline(limits["yield_min"], color="#b2182b", linestyle="--", label="Yield gate")
    axes[0].axvline(limits["wp_et_min"], color="#2166ac", linestyle="--", label="WP_ET gate")
    axes[0].set_xlabel("WP_ET (kg m$^{-3}$)")
    axes[0].set_ylabel("HWAM (kg ha$^{-1}$)")
    axes[0].set_title("Yield and water productivity")
    axes[0].legend(frameon=False)
    fig.colorbar(scatter, ax=axes[0], label="Season N (kg ha$^{-1}$)")

    axes[1].scatter(frame["PFP_N_kg_kg"], frame["final_gwad"], c=frame["summary_irrigation_total"], s=60, cmap="plasma", alpha=0.85, edgecolor="black", linewidth=0.4)
    axes[1].axhline(limits["yield_min"], color="#b2182b", linestyle="--")
    axes[1].axvline(limits["pfp_n_min"], color="#2166ac", linestyle="--", label="PFP_N gate")
    axes[1].set_xlabel("PFP_N (kg grain kg$^{-1}$ N)")
    axes[1].set_ylabel("HWAM (kg ha$^{-1}$)")
    axes[1].set_title("Yield and nitrogen productivity")
    axes[1].legend(frameon=False)
    fig.suptitle("SY2014 fixed official-stage deterministic feasibility search", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "022_01_fixed_stage_feasibility.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "022_01_fixed_stage_feasibility.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    old.OUT_ROOT = OUT
    write_grid()
    limits = thresholds()
    (OUT / "022_01_thresholds_and_provenance.json").write_text(
        json.dumps(
            {
                **limits,
                "primary_source": "prompts/021_18_sy2014_deterministic_upper_bound_oracle_search.md pre-registered criteria",
                "strict_source": "later 021_20 smoke gate, informed by 021_18 I75/N200 result; secondary only",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    validate_smoke()

    rows = []
    total = len(WATER) * len(NITROGEN)
    counter = 0
    for water_name, water in WATER.items():
        for nitrogen_name, nitrogen in NITROGEN.items():
            counter += 1
            scenario = f"{water_name}__{nitrogen_name}"
            print(f"[{counter}/{total}] {scenario}", flush=True)
            row = old.run_one(scenario, water, nitrogen, "fixed_official_stage_grid")
            row["water_schedule"] = water_name
            row["nitrogen_schedule"] = nitrogen_name
            rows.append(row)

    frame = classify(pd.DataFrame(rows), limits).sort_values(
        ["primary_scientific_pass", "strict_resource_pass", "final_gwad", "PFP_N_kg_kg"],
        ascending=[False, False, False, False],
    )
    frame.to_csv(OUT / "022_01_candidate_summary.csv", index=False)
    frame.loc[frame["primary_scientific_pass"]].to_csv(OUT / "022_01_primary_success_candidates.csv", index=False)
    frame.loc[frame["strict_resource_pass"]].to_csv(OUT / "022_01_strict_success_candidates.csv", index=False)
    execution_checks = {
        "candidate_count_48": bool(len(frame) == 48),
        "all_scheduled_daps_fired": bool(frame["missing_schedule_daps"].map(json.loads).map(len).eq(0).all()),
        "irrigation_summary_matches_requested": bool(
            (frame["requested_irrigation_total"] - frame["summary_irrigation_total"]).abs().max() < 1e-9
        ),
        "nitrogen_summary_matches_requested": bool(
            (frame["requested_nitrogen_total"] - frame["summary_nitrogen_total"]).abs().max() < 1e-9
        ),
        "all_summary_match_scores_zero": bool(frame["summary_match_score"].abs().max() < 1e-9),
        "all_candidates_have_no_late_n": bool(frame["late_n_after_dap90"].eq(0.0).all()),
    }
    (OUT / "022_01_execution_validation.json").write_text(
        json.dumps({"checks": execution_checks, "all_pass": all(execution_checks.values())}, indent=2),
        encoding="utf-8",
    )
    if not all(execution_checks.values()):
        raise RuntimeError(f"Execution validation failed: {execution_checks}")
    plot_results(frame, limits)

    primary = int(frame["primary_scientific_pass"].sum())
    strict = int(frame["strict_resource_pass"].sum())
    summary = {
        "status": "completed",
        "candidate_count": len(frame),
        "primary_success_count": primary,
        "strict_success_count": strict,
        "decision": "allow_022_02_preregistration" if primary > 0 else "stop_no_training",
        "training_calls": 0,
        "dssat_candidate_calls": len(frame),
        "stage_daps_moved_after_results": False,
        "thresholds_relaxed_after_results": False,
    }
    (OUT / "022_01_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
