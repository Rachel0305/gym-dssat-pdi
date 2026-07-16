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
from stage_based_dqn_core_022 import EXECUTABLE_STAGE_DAPS, valid_action_indices


OUT_ROOT = ROOT / "benchmark_results" / "022_04"
SOURCE_ROOT = ROOT / "benchmark_results" / "022_03"
MANIFEST = SOURCE_ROOT / "022_03_fixed_grid_transition_manifest.csv"
CHECKPOINTS = (15, 30, 45, 60)
EXPECTED_SUPPORT = {
    1: (0, 1, 3, 4),
    30: (4, 5, 7, 8),
    50: (4, 5, 7, 8),
    65: (1, 2, 4, 5, 7, 8),
    85: (1, 2, 4, 5),
    110: (0, 1),
}

base.OUT_ROOT = OUT_ROOT


def derive_support() -> tuple[dict[int, tuple[int, ...]], pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(MANIFEST)
    if len(frame) != 288:
        raise RuntimeError(f"Expected 288 fixed-grid transitions, got {len(frame)}")
    support: dict[int, tuple[int, ...]] = {}
    rows = []
    for dap in EXECUTABLE_STAGE_DAPS:
        subset = frame[frame["dap"].eq(dap)]
        actions = tuple(sorted(int(value) for value in subset["action_index"].unique()))
        support[dap] = actions
        rows.append(
            {
                "dap": dap,
                "supported_actions": json.dumps(actions),
                "transition_count": len(subset),
                "scenario_count": subset["scenario"].nunique(),
                "contains_primary_success": bool(subset["primary_scientific_pass_022_01"].astype(bool).any()),
                "contains_primary_failure": bool((~subset["primary_scientific_pass_022_01"].astype(bool)).any()),
                "original_valid_actions": json.dumps(valid_action_indices(dap)),
                "support_is_subset_of_original": set(actions).issubset(valid_action_indices(dap)),
            }
        )
    support_df = pd.DataFrame(rows)
    checks = {
        "exactly_288_transitions": len(frame) == 288,
        "six_stage_support_sets": len(support) == 6,
        "support_matches_preregistered_expected": support == EXPECTED_SUPPORT,
        "every_stage_has_success_and_failure_sources": bool(
            support_df["contains_primary_success"].all()
            and support_df["contains_primary_failure"].all()
        ),
        "all_support_actions_originally_legal": bool(support_df["support_is_subset_of_original"].all()),
        "four_checkpoints_exist": all(
            (SOURCE_ROOT / "checkpoints" / f"season_{season:03d}.pt").exists()
            for season in CHECKPOINTS
        ),
    }
    return support, support_df, checks


def evaluate_treatment(
    season: int,
    support: dict[int, tuple[int, ...]],
    scaler: base.FixedObservationScaler,
    thresholds: dict[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    checkpoint_path = SOURCE_ROOT / "checkpoints" / f"season_{season:03d}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = base.QNetwork()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    env = base.make_env(OUT_ROOT / "treatment_runtime" / f"season_{season:03d}")
    runner = base.StageSeasonRunner(env, scaler)
    q_rows: list[dict[str, Any]] = []

    def choose(stage_index: int, dap: int, obs: np.ndarray, original_valid: tuple[int, ...]) -> int:
        supported = tuple(action for action in support[dap] if action in original_valid)
        if supported != support[dap]:
            raise RuntimeError(f"Support/original legality mismatch at DAP {dap}")
        with torch.no_grad():
            q = model(torch.from_numpy(obs).float().unsqueeze(0)).squeeze(0).cpu().numpy()
        original_action = base.masked_argmax(q, original_valid)
        supported_action = base.masked_argmax(q, supported)
        row = {
            "checkpoint_season": season,
            "stage_index": stage_index,
            "dap": dap,
            "original_valid_actions": json.dumps(original_valid),
            "supported_actions": json.dumps(supported),
            "original_argmax_action": original_action,
            "support_argmax_action": supported_action,
            "original_argmax_was_out_of_support": original_action not in supported,
        }
        row.update({f"q_action_{index}": float(q[index]) for index in range(9)})
        q_rows.append(row)
        return supported_action

    try:
        result = runner.run(choose, f"support_eval_s{season}", keep_daily=True)
        metrics = base.summary_metrics(env, result, thresholds)
    finally:
        env.close()
    summary = {
        "checkpoint_season": season,
        "arm": "treatment_fixed_grid_support_mask",
        "final_yield": result["final_yield"],
        "final_biomass": result["final_biomass"],
        "irrigation_total": result["irrigation_total"],
        "nitrogen_total": result["nitrogen_total"],
        "g0_raw": result["returns_raw"][0],
        "g0_scaled": result["returns_scaled"][0],
        "action_sequence": json.dumps(result["selected_actions"]),
        **metrics,
    }
    for row in result["stage_rows"]:
        row.update({"checkpoint_season": season, "arm": "treatment_fixed_grid_support_mask"})
    for row in result["daily_rows"]:
        row.update({"checkpoint_season": season, "arm": "treatment_fixed_grid_support_mask"})
    return summary, result["stage_rows"], result["daily_rows"], q_rows


def main() -> None:
    if OUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT_ROOT}")
    OUT_ROOT.mkdir(parents=True)
    support, support_df, checks = derive_support()
    support_df.to_csv(OUT_ROOT / "022_04_stage_action_support.csv", index=False)
    (OUT_ROOT / "022_04_stage_action_support.json").write_text(
        json.dumps({str(dap): list(actions) for dap, actions in support.items()}, indent=2),
        encoding="utf-8",
    )
    if not all(checks.values()):
        payload = {"status": "failed", "branch": "D_implementation_failure", "checks": checks}
        (OUT_ROOT / "022_04_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        raise RuntimeError(f"Pre-run checks failed: {checks}")

    thresholds = json.loads(base.THRESHOLDS_PATH.read_text(encoding="utf-8"))
    scaler = base.FixedObservationScaler(base.SCALER_PATH)
    control = pd.read_csv(SOURCE_ROOT / "022_03_checkpoint_evaluation_summary.csv")
    control.insert(1, "arm", "control_original_mask_reused_022_03")
    treatment_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    q_rows: list[dict[str, Any]] = []
    for season in CHECKPOINTS:
        summary, stages, daily, q = evaluate_treatment(season, support, scaler, thresholds)
        treatment_rows.append(summary)
        stage_rows.extend(stages)
        daily_rows.extend(daily)
        q_rows.extend(q)
        print(
            f"season={season} yield={summary['final_yield']:.0f} I={summary['irrigation_total']:.0f} "
            f"N={summary['nitrogen_total']:.0f} primary={summary['primary_pass']} strict={summary['strict_pass']}",
            flush=True,
        )
    treatment = pd.DataFrame(treatment_rows)
    combined = pd.concat([control, treatment], ignore_index=True, sort=False)
    combined.to_csv(OUT_ROOT / "022_04_control_treatment_summary.csv", index=False)
    pd.DataFrame(stage_rows).to_csv(OUT_ROOT / "022_04_treatment_stage_actions.csv", index=False)
    pd.DataFrame(daily_rows).to_csv(OUT_ROOT / "022_04_treatment_daily_values.csv", index=False)
    q_df = pd.DataFrame(q_rows)
    q_df.to_csv(OUT_ROOT / "022_04_q_support_audit.csv", index=False)

    primary_count = int(treatment["primary_pass"].sum())
    strict_count = int(treatment["strict_pass"].sum())
    season60 = bool(treatment.loc[treatment.checkpoint_season.eq(60), "primary_pass"].iloc[0])
    yield_preserved = bool((treatment["final_yield"] >= thresholds["yield_min"]).all())
    if primary_count >= 3 and season60 and yield_preserved:
        branch, next_step = "A_support_extrapolation_confirmed", "allow_022_05_training_with_frozen_support_mask"
    elif primary_count >= 1:
        branch, next_step = "B_partial_support_effect", "report_only_no_automatic_training"
    else:
        branch, next_step = "C_support_mask_not_sufficient", "stop_support_mask_branch"

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for arm, group in combined.groupby("arm"):
        label = "control" if arm.startswith("control") else "support mask"
        style = "--" if label == "control" else "-"
        axes[0].plot(group.checkpoint_season, group.final_yield, marker="o", linestyle=style, label=label)
        axes[1].plot(group.checkpoint_season, group.irrigation_total, marker="o", linestyle=style, label=label)
        axes[2].plot(group.checkpoint_season, group.nitrogen_total, marker="o", linestyle=style, label=label)
    axes[0].axhline(thresholds["yield_min"], color="#b22222", linestyle=":")
    axes[0].set_ylabel("HWAM (kg/ha)")
    axes[1].set_ylabel("Irrigation (mm)")
    axes[2].set_ylabel("Nitrogen (kg/ha)")
    for ax in axes:
        ax.set_xlabel("Checkpoint season")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.suptitle("SY2014 checkpoint audit: original vs fixed-grid action support")
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "022_04_support_mask_checkpoint_audit.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT_ROOT / "022_04_support_mask_checkpoint_audit.svg", bbox_inches="tight")
    plt.close(fig)

    payload = {
        "status": "completed",
        "branch": branch,
        "next_step": next_step,
        "checks": checks,
        "treatment_primary_checkpoint_count": primary_count,
        "treatment_strict_checkpoint_count": strict_count,
        "season60_primary": season60,
        "yield_threshold_preserved_all_checkpoints": yield_preserved,
        "original_argmax_out_of_support_count": int(q_df["original_argmax_was_out_of_support"].sum()),
        "q_audit_state_count": len(q_df),
        "treatment_checkpoints": treatment_rows,
    }
    (OUT_ROOT / "022_04_result.json").write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
