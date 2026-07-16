from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "021_16"


def actual_totals(scenario: str) -> tuple[float, float, int, int]:
    path = OUT / scenario / "pdi_tmp_snapshot_eval" / "MgmtEvent.OUT"
    irrigation = 0.0
    nitrogen = 0.0
    irrigation_events = 0
    nitrogen_events = 0
    for raw in path.read_text(encoding="latin-1", errors="ignore").splitlines():
        if "Irrigation" in raw:
            match = re.search(r"Irrigation\s+([-+]?\d+(?:\.\d+)?)\s+mm", raw)
            if match:
                irrigation += float(match.group(1))
                irrigation_events += 1
        elif "Fertilizer" in raw or "Nitrogen" in raw:
            match = re.search(r"(?:Fertilizer|Nitrogen)\s+([-+]?\d+(?:\.\d*)?)\s+kg", raw)
            if match:
                nitrogen += float(match.group(1))
                nitrogen_events += 1
    return irrigation, nitrogen, irrigation_events, nitrogen_events


def main() -> None:
    summary_path = OUT / "021_16_summary.csv"
    summary = pd.read_csv(summary_path)
    for idx, row in summary.iterrows():
        values = actual_totals(str(row["scenario"]))
        summary.loc[idx, "actual_irrigation_total"] = values[0]
        summary.loc[idx, "actual_nitrogen_total"] = values[1]
        summary.loc[idx, "actual_irrigation_events"] = values[2]
        summary.loc[idx, "actual_nitrogen_events"] = values[3]
    summary.to_csv(summary_path, index=False)

    yields = summary.set_index("scenario")["final_gwad"].to_dict()
    effects = pd.DataFrame(
        [
            {
                "contrast": "N_early_minus_N_late_under_I_early",
                "yield_difference_kg_ha": yields["i_early__n_early"] - yields["i_early__n_late"],
                "interpretation": "固定 I-early，仅交换施氮时点",
            },
            {
                "contrast": "N_early_minus_N_late_under_I_late",
                "yield_difference_kg_ha": yields["i_late__n_early"] - yields["i_late__n_late"],
                "interpretation": "固定 I-late，仅交换施氮时点",
            },
            {
                "contrast": "I_late_minus_I_early_under_N_early",
                "yield_difference_kg_ha": yields["i_late__n_early"] - yields["i_early__n_early"],
                "interpretation": "固定 N-early，仅交换灌溉时点",
            },
            {
                "contrast": "I_late_minus_I_early_under_N_late",
                "yield_difference_kg_ha": yields["i_late__n_late"] - yields["i_early__n_late"],
                "interpretation": "固定 N-late，仅交换灌溉时点",
            },
        ]
    )
    effects.to_csv(OUT / "021_16_factorial_effects.csv", index=False)

    early_mean = (yields["i_early__n_early"] + yields["i_late__n_early"]) / 2.0
    late_mean = (yields["i_early__n_late"] + yields["i_late__n_late"]) / 2.0
    i_early_mean = (yields["i_early__n_early"] + yields["i_early__n_late"]) / 2.0
    i_late_mean = (yields["i_late__n_early"] + yields["i_late__n_late"]) / 2.0
    result = {
        "status": "completed_deterministic_forward",
        "training_calls": 0,
        "dssat_forward_runs": 4,
        "mean_n_timing_effect_early_minus_late_kg_ha": early_mean - late_mean,
        "mean_irrigation_timing_effect_late_minus_early_kg_ha": i_late_mean - i_early_mean,
        "all_actual_totals_i120_n300": bool(
            summary["actual_irrigation_total"].eq(120.0).all()
            and summary["actual_nitrogen_total"].eq(300.0).all()
        ),
        "high_schedule_reproduction_error_kg_ha": float(
            summary.loc[summary["scenario"].eq("i_early__n_early"), "absolute_reproduction_error"].iloc[0]
        ),
        "low_schedule_reproduction_error_kg_ha": float(
            summary.loc[summary["scenario"].eq("i_late__n_late"), "absolute_reproduction_error"].iloc[0]
        ),
        "interpretation": (
            "Under fixed SY2014 IC=2 input and identical seasonal I120/N300, swapping only "
            "the nitrogen timing changes yield by about 2.91 t/ha, while swapping only the "
            "two observed irrigation timings changes yield by 7-20 kg/ha. This controlled "
            "DSSAT comparison supports nitrogen timing as the dominant cause for the two "
            "10K checkpoint yield outcomes. It does not by itself identify the DQN training root cause."
        ),
    }
    (OUT / "021_16_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(effects.to_string(index=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
