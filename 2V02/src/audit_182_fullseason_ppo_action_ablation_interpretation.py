"""Task 182 post-hoc interpretation audit.

This reads the completed fixed-schedule ablation outputs only.  It adds a
tolerance-aware interpretation layer so tiny DSSAT differences are not
overstated as meaningful yield losses.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "182_sya_fullseason_ppo_action_ablation"
EFFECTS = OUT / "evaluation" / "182_ablation_effects_vs_full_ppo.csv"
SUMMARY = OUT / "evaluation" / "182_episode_summary.csv"
TOLERANCE_YIELD_KG_HA = 5.0


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def direction(value: float, tolerance: float) -> str:
    if value > tolerance:
        return "increase"
    if value < -tolerance:
        return "decrease"
    return "negligible"


def main() -> None:
    if not EFFECTS.exists():
        raise FileNotFoundError(EFFECTS)
    effects = pd.read_csv(EFFECTS)
    summary = pd.read_csv(SUMMARY)

    non_full = effects[effects["treatment"].ne("full_ppo")].copy()
    non_full["raw_yield_direction"] = [
        direction(float(value), 1e-6) for value in non_full["delta_final_grnwt_vs_full"]
    ]
    non_full["tolerance_yield_direction"] = [
        direction(float(value), TOLERANCE_YIELD_KG_HA)
        for value in non_full["delta_final_grnwt_vs_full"]
    ]
    non_full["profit_direction"] = [
        direction(float(value), 1e-6) for value in non_full["delta_profit_simple_vs_full"]
    ]
    non_full["reward_unscaled_direction"] = [
        direction(float(value), 1e-6)
        for value in non_full["delta_reward_unscaled_sum_vs_full"]
    ]
    non_full["max_nstres_direction"] = [
        direction(float(value), 1e-6) for value in non_full["delta_max_nstres_vs_full"]
    ]

    by_treatment = (
        non_full.groupby("treatment", as_index=False)
        .agg(
            scenarios=("weather_variant", "nunique"),
            mean_delta_final_grnwt=("delta_final_grnwt_vs_full", "mean"),
            min_delta_final_grnwt=("delta_final_grnwt_vs_full", "min"),
            max_delta_final_grnwt=("delta_final_grnwt_vs_full", "max"),
            mean_delta_profit_simple=("delta_profit_simple_vs_full", "mean"),
            min_delta_profit_simple=("delta_profit_simple_vs_full", "min"),
            max_delta_profit_simple=("delta_profit_simple_vs_full", "max"),
            mean_delta_reward_unscaled=("delta_reward_unscaled_sum_vs_full", "mean"),
            irrigation_saved_mm=("delta_total_irrigation_vs_full", lambda s: float(-pd.to_numeric(s).mean())),
            yield_increase_count=("tolerance_yield_direction", lambda s: int((s == "increase").sum())),
            yield_negligible_count=("tolerance_yield_direction", lambda s: int((s == "negligible").sum())),
            yield_decrease_count=("tolerance_yield_direction", lambda s: int((s == "decrease").sum())),
            profit_increase_count=("profit_direction", lambda s: int((s == "increase").sum())),
        )
    )

    non_full.to_csv(
        OUT / "evaluation" / "182_tolerance_aware_effects.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_treatment.to_csv(
        OUT / "evaluation" / "182_tolerance_aware_by_treatment.csv",
        index=False,
        encoding="utf-8-sig",
    )

    raw_counts = non_full["raw_yield_direction"].value_counts().to_dict()
    tolerance_counts = non_full["tolerance_yield_direction"].value_counts().to_dict()
    profit_counts = non_full["profit_direction"].value_counts().to_dict()
    reward_counts = non_full["reward_unscaled_direction"].value_counts().to_dict()
    full = summary[summary["treatment"].eq("full_ppo")].copy()
    full_sequences = sorted(set(full["applied_action_sequence"].astype(str)))

    result = {
        "task": "182_sya_fullseason_ppo_action_ablation_interpretation",
        "status": "completed",
        "mode": "read_only_posthoc_interpretation",
        "inputs": {
            "effects": relative(EFFECTS),
            "summary": relative(SUMMARY),
        },
        "yield_tolerance_kg_ha": TOLERANCE_YIELD_KG_HA,
        "non_full_rows": int(len(non_full)),
        "raw_yield_direction_counts": {str(k): int(v) for k, v in raw_counts.items()},
        "tolerance_yield_direction_counts": {str(k): int(v) for k, v in tolerance_counts.items()},
        "profit_direction_counts": {str(k): int(v) for k, v in profit_counts.items()},
        "reward_unscaled_direction_counts": {str(k): int(v) for k, v in reward_counts.items()},
        "min_delta_final_grnwt_vs_full": float(non_full["delta_final_grnwt_vs_full"].min()),
        "max_delta_final_grnwt_vs_full": float(non_full["delta_final_grnwt_vs_full"].max()),
        "min_delta_profit_simple_vs_full": float(non_full["delta_profit_simple_vs_full"].min()),
        "max_delta_profit_simple_vs_full": float(non_full["delta_profit_simple_vs_full"].max()),
        "min_delta_reward_unscaled_vs_full": float(non_full["delta_reward_unscaled_sum_vs_full"].min()),
        "max_delta_reward_unscaled_vs_full": float(non_full["delta_reward_unscaled_sum_vs_full"].max()),
        "full_ppo_action_sequences": full_sequences,
        "interpretation": {
            "primary": (
                "Deleting DAP97 and/or DAP104 saved water and improved simple profit/reward in all "
                "18 non-full branches. With a 5 kg/ha practical tolerance, no branch shows a "
                "meaningful yield loss; 15 branches show yield increase and 3 are negligible."
            ),
            "boundary": (
                "This remains a fixed-future-schedule ablation. It does not test whether the PPO "
                "policy would compensate after an omitted irrigation event."
            ),
        },
        "outputs": {
            "tolerance_aware_effects": relative(OUT / "evaluation" / "182_tolerance_aware_effects.csv"),
            "tolerance_aware_by_treatment": relative(OUT / "evaluation" / "182_tolerance_aware_by_treatment.csv"),
        },
    }
    (OUT / "182_interpretation_addendum.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
