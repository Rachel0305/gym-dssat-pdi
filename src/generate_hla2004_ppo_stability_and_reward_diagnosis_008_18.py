from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ppo_safe_rendering import PROJECT_ROOT


OUTPUT_ROOT = PROJECT_ROOT / "Leave_One_experiments" / "hla2004_ppo_stability_reward_diagnosis_008_18"
DOC_MD = PROJECT_ROOT / "docs" / "2026-06-14_008_18_hla2004_ppo_stability_and_reward_diagnosis_report.md"


def read_first(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return df.iloc[0].to_dict() if not df.empty else {}


def read_hla(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if {"station", "year"}.issubset(df.columns):
        df = df[(df["station"].astype(str).eq("HLA")) & (pd.to_numeric(df["year"], errors="coerce").eq(2004))]
    return df.iloc[0].to_dict() if not df.empty else {}


def stage_pattern(path: Path) -> tuple[str, dict[str, float]]:
    if not path.exists():
        return "", {}
    df = pd.read_csv(path)
    pieces = []
    vals: dict[str, float] = {}
    for _, row in df.iterrows():
        sid = str(row.get("stage_id"))
        raw = float(pd.to_numeric(row.get("raw_stage_action_irrigation"), errors="coerce")) if "raw_stage_action_irrigation" in df.columns else np.nan
        irr = float(pd.to_numeric(row.get("stage_action_amir"), errors="coerce")) if "stage_action_amir" in df.columns else float(pd.to_numeric(row.get("irrigation"), errors="coerce"))
        vals[f"{sid}_raw_i"] = raw
        vals[f"{sid}_irrigation"] = irr
        if np.isfinite(raw):
            pieces.append(f"{sid}:raw={raw:.2f},I={irr:.1f}")
        else:
            pieces.append(f"{sid}:I={irr:.1f}")
    return "; ".join(pieces), vals


def ratio(value: Any, denominator: float = 7062.4658203125) -> float:
    try:
        return float(value) / denominator
    except Exception:
        return np.nan


def build_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    rule = read_hla(PROJECT_ROOT / "Leave_One_experiments" / "forecast_gate_rule_replay_008_11" / "evaluation" / "008_11_forecast_gate_rule_replay_summary.csv")
    rows.append(
        {
            "experiment": "008_11_pure_forecast_rule",
            "seed": "none",
            "design": "forecast gate pure rule, 30 mm when triggered",
            "total_irrigation": rule.get("total_irrigation"),
            "total_n": rule.get("total_n"),
            "final_grnwt": rule.get("final_grnwt"),
            "grnwt_fraction_vs_fixed_I120": ratio(rule.get("final_grnwt")),
            "swfac_days_gt_0p05": rule.get("swfac_days_gt_0p05"),
            "ppo_autonomous_contribution": 0.0,
            "raw_action_pattern": "not applicable",
            "strategy_structure": "rule triggered S2-S5 irrigation",
            "interpretation": "Agronomically interpretable forecast rule, but no PPO decision.",
        }
    )

    hard = read_hla(PROJECT_ROOT / "Leave_One_experiments" / "ppo_contribution_forecast_gate_008_12" / "evaluation" / "008_12_ppo_contribution_summary.csv")
    rows.append(
        {
            "experiment": "008_12_hard_min_gate_ppo",
            "seed": 0,
            "design": "PPO under hard-minimum forecast gate",
            "total_irrigation": hard.get("total_irrigation"),
            "total_n": hard.get("total_n"),
            "final_grnwt": hard.get("final_grnwt"),
            "grnwt_fraction_vs_fixed_I120": ratio(hard.get("final_grnwt")),
            "swfac_days_gt_0p05": hard.get("swfac_days_gt_0p05"),
            "ppo_autonomous_contribution": hard.get("total_ppo_extra_above_min", 0.0),
            "raw_action_pattern": "PPO extra above gate minimum = 0",
            "strategy_structure": "rule-dominated",
            "interpretation": "Good yield but PPO contribution was zero.",
        }
    )

    soft0 = read_first(PROJECT_ROOT / "Leave_One_experiments" / "hla2004_irrigation_only_soft_stress_ppo_008_15_10k" / "evaluation" / "HLA_2004_stress_aware_stage_ppo_summary.csv")
    pattern0, _ = stage_pattern(PROJECT_ROOT / "Leave_One_experiments" / "hla2004_irrigation_only_soft_stress_ppo_008_15_10k" / "daily_outputs" / "HLA" / "HLA_2004_seed0_stress_aware_stage_steps.csv")
    rows.append(
        {
            "experiment": "008_15_soft_stress_ppo_10k",
            "seed": 0,
            "design": "no hard minimum, soft SWFAC penalty 1.0/0.25",
            "total_irrigation": soft0.get("total_irrigation"),
            "total_n": soft0.get("total_n"),
            "final_grnwt": soft0.get("final_grnwt"),
            "grnwt_fraction_vs_fixed_I120": soft0.get("grnwt_fraction_vs_fixed_I120", ratio(soft0.get("final_grnwt"))),
            "swfac_days_gt_0p05": soft0.get("swfac_days_gt_0p05"),
            "ppo_autonomous_contribution": soft0.get("total_irrigation"),
            "raw_action_pattern": pattern0,
            "strategy_structure": "S4/S5 full late irrigation",
            "interpretation": "PPO made autonomous nonzero decisions and reached 96% of fixed I120 yield with 80 mm irrigation.",
        }
    )

    seed16 = pd.read_csv(PROJECT_ROOT / "Leave_One_experiments" / "hla2004_irrigation_only_soft_stress_ppo_008_16_seed_stability" / "evaluation" / "008_16_seed_stability_summary.csv")
    seed1 = seed16[pd.to_numeric(seed16["seed"], errors="coerce").eq(1)].iloc[0].to_dict()
    rows.append(
        {
            "experiment": "008_16_soft_stress_seed1_5k",
            "seed": 1,
            "design": "no hard minimum, soft SWFAC penalty 1.0/0.25",
            "total_irrigation": seed1.get("total_irrigation"),
            "total_n": seed1.get("total_n"),
            "final_grnwt": seed1.get("final_grnwt"),
            "grnwt_fraction_vs_fixed_I120": seed1.get("grnwt_fraction_vs_fixed_I120", ratio(seed1.get("final_grnwt"))),
            "swfac_days_gt_0p05": seed1.get("swfac_days_gt_0p05"),
            "ppo_autonomous_contribution": seed1.get("total_irrigation"),
            "raw_action_pattern": "S1-S5 raw=-1.00",
            "strategy_structure": "zero-irrigation degeneration",
            "interpretation": "Original soft penalty did not remove the bad zero-irrigation attractor for seed1.",
        }
    )

    strong = read_first(PROJECT_ROOT / "Leave_One_experiments" / "hla2004_stronger_soft_stress_penalty_ppo_008_17" / "evaluation" / "008_17_stronger_penalty_recovery_summary.csv")
    rows.append(
        {
            "experiment": "008_17_stronger_penalty_seed1_5k",
            "seed": 1,
            "design": "no hard minimum, stronger SWFAC penalty 3.0/1.0",
            "total_irrigation": strong.get("total_irrigation"),
            "total_n": strong.get("total_n"),
            "final_grnwt": strong.get("final_grnwt"),
            "grnwt_fraction_vs_fixed_I120": strong.get("grnwt_fraction_vs_fixed_I120", ratio(strong.get("final_grnwt"))),
            "swfac_days_gt_0p05": strong.get("swfac_days_gt_0p05"),
            "ppo_autonomous_contribution": strong.get("total_irrigation"),
            "raw_action_pattern": "S1-S3 raw=-1.00; S4/S5 raw=0.34",
            "strategy_structure": "S4/S5 partial late irrigation",
            "interpretation": "Stronger penalty recovered seed1 from zero irrigation but did not reach seed0 yield level.",
        }
    )

    return pd.DataFrame(rows)


def df_md(df: pd.DataFrame) -> str:
    out = df.copy()
    for col in out.select_dtypes(include=["float", "int"]).columns:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(4)
    header = "| " + " | ".join(out.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(out.columns)) + " |"
    rows = ["| " + " | ".join("" if pd.isna(x) else str(x) for x in row) + " |" for row in out.values.tolist()]
    return "\n".join([header, sep, *rows])


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "evaluation").mkdir(parents=True, exist_ok=True)
    DOC_MD.parent.mkdir(parents=True, exist_ok=True)
    summary = build_rows()
    summary_path = OUTPUT_ROOT / "evaluation" / "008_18_hla2004_core_experiment_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    lines = [
        "# 008_18 HLA 2004 PPO Stability And Reward Diagnosis Report",
        "",
        "## Purpose",
        "",
        "This report consolidates the HLA 2004 forecast/stress-aware irrigation PPO experiment chain from 008_11 to 008_17.",
        "No new training or DSSAT simulation was performed.",
        "",
        "## Core Experiment Chain",
        "",
        df_md(summary),
        "",
        "## Proven Result",
        "",
        "The forecast/stress-aware stage decision framework can produce agronomically interpretable irrigation behavior in HLA 2004.",
        "The pure forecast rule responds to dry forecast windows, and the no-hard-minimum PPO design demonstrates that PPO can make autonomous irrigation decisions when the rule only allows or blocks irrigation.",
        "",
        "The strongest positive case is 008_15 seed0: PPO applied 80 mm in S4/S5, reached 6777.93 kg/ha GRNWT, and achieved 95.97% of the fixed I120_N150 reference yield without hard-minimum rule support.",
        "",
        "## Unresolved Limitation",
        "",
        "The PPO policy is seed-sensitive under the current reward design.",
        "Seed0 found a strong S4/S5 late-irrigation strategy, but seed1 degenerated to zero irrigation under the original soft penalty.",
        "A stronger SWFAC penalty recovered seed1 to a partial S4/S5 irrigation strategy, but yield reached only 79.0% of fixed I120_N150.",
        "",
        "This means the framework is promising, but the current PPO configuration should not yet be presented as a production-level stable policy.",
        "",
        "## Interpretation For The Paper",
        "",
        "Two layers should be separated in the manuscript:",
        "",
        "1. Proven: a weather-forecast and stress-diagnosis assisted stage decision framework can make irrigation decisions agronomically interpretable and allows PPO to contribute when hard minimum gate support is removed.",
        "2. Not yet solved: PPO water amount is sensitive to random initialization, so reward scaling and training stability require further study.",
        "",
        "## Recommended Next Step",
        "",
        "Do not continue blind penalty scanning on HLA 2004.",
        "Use this diagnosis as a staged HLA 2004 result, then run a minimal FQA 2016 second-year validation using the 008_17 stronger penalty configuration without tuning.",
        "",
        "## Files",
        "",
        f"- Core summary CSV: `{summary_path.relative_to(PROJECT_ROOT)}`",
    ]
    DOC_MD.write_text("\n".join(lines), encoding="utf-8")
    print(summary_path.relative_to(PROJECT_ROOT))
    print(DOC_MD.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()

