"""Consolidate the failed-first and corrected 021_13 reward-scale smoke audits."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "benchmark_results/021_13"
FAILED = BASE / "021_13_sy2014_reward_scale_smoke__sy_2014_seed1"
PASSED = BASE / "021_13_sy2014_reward_scale_smoke_retry__sy_2014_seed1"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    failed_audit = read_json(FAILED / "checkpoints/checkpoint_200/reward_scale_audit.json")
    passed_audit = read_json(PASSED / "checkpoints/checkpoint_200/reward_scale_audit.json")
    state = read_json(PASSED / "checkpoints/checkpoint_200/training_state.json")
    season = pd.read_csv(PASSED / "evaluations/season_summary.csv").iloc[0]
    verification = pd.DataFrame(
        [
            {
                "experiment_id": state["experiment_id"],
                "config_hash": state["config_hash"],
                "steps": state["completed_steps"],
                "reward_scale": passed_audit["reward_scale"],
                "max_scale_identity_error": passed_audit["max_scale_identity_error"],
                "max_component_identity_error": passed_audit["max_component_identity_error"],
                "component_fields_missing": passed_audit["component_fields_missing"],
                "replay_stored_count": passed_audit["replay_consistency"]["stored_count"],
                "wrapper_record_count": passed_audit["replay_consistency"]["record_count"],
                "callback_boundary_uncommitted_final_step": passed_audit[
                    "replay_consistency"
                ]["callback_boundary_uncommitted_final_step"],
                "replay_max_abs_error": passed_audit["replay_consistency"]["max_abs_error"],
                "replay_tolerance": passed_audit["replay_abs_error_tolerance"],
                "all_rewards_finite": passed_audit["all_rewards_finite"],
                "audit_passed": passed_audit["passed"],
                "exploration_rate": state["exploration_rate"],
                "sb3_internal_total_timesteps": state["sb3_internal_total_timesteps"],
                "evaluation_reward_scope": "raw_unscaled_economic_reward",
                "evaluation_reward_total": float(season["reward_total"]),
            }
        ]
    )
    verification.to_csv(
        BASE / "021_13_smoke_verification.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "status": "smoke_passed_after_callback_boundary_fix",
        "formal_25k_started": False,
        "training_or_dssat_scope": "200-step smoke only",
        "first_attempt": {
            "status": "failed_safely",
            "reason": (
                "replay audit aligned 199 stored transitions with the last 199 of 200 wrapper "
                "records; SB3 stop callback observes one final env step before replay insertion"
            ),
            "reward_formula_checks_were_already_exact": bool(
                failed_audit["max_scale_identity_error"] == 0
                and failed_audit["max_component_identity_error"] == 0
            ),
        },
        "retry": verification.iloc[0].to_dict(),
        "scientific_interpretation": (
            "Implementation chain verified only.  The 200-step agricultural outcome is not "
            "evidence that reward scaling improves stability or policy quality."
        ),
        "next_gate": (
            "User approval is required before a SY2014 seed1 25K single-variable comparison; "
            "that run must repeat 021_12 parameter-L2 and fixed-state Q/ranking metrics."
        ),
    }
    (BASE / "021_13_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

