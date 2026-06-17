from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_all_year_direct_action_safe_ppo as direct
import run_hla2004_stress_aware_stage_ppo_008_07 as base_runner
from ppo_safe_rendering import PROJECT_ROOT
from soft_stress_gate_wrapper_008_14 import SoftStressForecastGateDapStageActionWrapper


CONFIG_PATH = PROJECT_ROOT / "experiments" / "ppo_observed_years" / "config_008_14_hla2004_irrigation_only_ppo_soft_stress_reward.yaml"
RULE_SUMMARY = PROJECT_ROOT / "Leave_One_experiments" / "forecast_gate_rule_replay_008_11" / "evaluation" / "008_11_forecast_gate_rule_replay_summary.csv"
HARD_MIN_SUMMARY = PROJECT_ROOT / "Leave_One_experiments" / "ppo_contribution_forecast_gate_008_12" / "evaluation" / "008_12_ppo_contribution_summary.csv"


def make_env(config: dict[str, Any], env_config: dict[str, Any], station: str, year: int, seed: int, tag: str, evaluation: bool):
    base = direct.make_base_env(env_config, station, year, seed, tag, evaluation=evaluation)
    return SoftStressForecastGateDapStageActionWrapper(base, config)


def _df_md(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return ""
    out = df.head(max_rows).copy()
    for col in out.select_dtypes(include=["float", "int"]).columns:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(4)
    header = "| " + " | ".join(out.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(out.columns)) + " |"
    rows = ["| " + " | ".join("" if pd.isna(x) else str(x) for x in row) + " |" for row in out.values.tolist()]
    return "\n".join([header, sep, *rows])


def _read_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def build_custom_report(config: dict[str, Any]) -> None:
    root = base_runner.root(config)
    summary_path = root / "evaluation" / "HLA_2004_stress_aware_stage_ppo_summary.csv"
    train_path = root / "evaluation" / "HLA_2004_stress_aware_stage_ppo_training.csv"
    stage_path = root / "daily_outputs" / "HLA" / "HLA_2004_seed0_stress_aware_stage_steps.csv"
    summary = _read_optional(summary_path)
    train = _read_optional(train_path)
    stages = _read_optional(stage_path)

    comparisons: list[dict[str, Any]] = []
    run_label = "008_15_soft_stress_ppo" if "008_15" in str(root) or "008_15" in str(base_runner.doc_md(config)) else "008_14_soft_stress_ppo"
    if not summary.empty:
        r = summary.iloc[0]
        comparisons.append(
            {
                "scenario": run_label,
                "total_irrigation": r.get("total_irrigation"),
                "total_n": r.get("total_n"),
                "final_grnwt": r.get("final_grnwt"),
                "swfac_days_gt_0p05": r.get("swfac_days_gt_0p05"),
                "note": "PPO controls irrigation; forecast gate only allows/blocks; no hard minimum.",
            }
        )
        for scenario in ["n_only_medium", "fixed_I60_N150", "fixed_I120_N150", "expert_reference_recorded"]:
            g = r.get(f"{scenario}_grnwt")
            i = r.get(f"{scenario}_irrigation")
            n = r.get(f"{scenario}_n")
            if pd.notna(g):
                comparisons.append(
                    {
                        "scenario": scenario,
                        "total_irrigation": i,
                        "total_n": n,
                        "final_grnwt": g,
                        "swfac_days_gt_0p05": np.nan,
                        "note": "Existing 008_06 reference.",
                    }
                )
    rule = _read_optional(RULE_SUMMARY)
    if not rule.empty:
        hit = rule[(rule["station"].astype(str).eq("HLA")) & (pd.to_numeric(rule["year"], errors="coerce").eq(2004))]
        if not hit.empty:
            rr = hit.iloc[0]
            comparisons.append(
                {
                    "scenario": "008_11_pure_forecast_rule",
                    "total_irrigation": rr.get("total_irrigation"),
                    "total_n": rr.get("total_n"),
                    "final_grnwt": rr.get("final_grnwt"),
                    "swfac_days_gt_0p05": rr.get("swfac_days_gt_0p05"),
                    "note": "Pure rule replay with 30 mm hard event amount.",
                }
            )
    hard = _read_optional(HARD_MIN_SUMMARY)
    if not hard.empty:
        hit = hard[(hard["station"].astype(str).eq("HLA")) & (pd.to_numeric(hard["year"], errors="coerce").eq(2004))]
        if not hit.empty:
            hr = hit.iloc[0]
            comparisons.append(
                {
                    "scenario": "008_12_hard_min_gate_ppo",
                    "total_irrigation": hr.get("total_irrigation"),
                    "total_n": hr.get("total_n"),
                    "final_grnwt": hr.get("final_grnwt"),
                    "swfac_days_gt_0p05": hr.get("swfac_days_gt_0p05"),
                    "note": "PPO under hard minimum gate; contribution previously measured as zero.",
                }
            )

    comp = pd.DataFrame(comparisons)
    comp_path = root / "evaluation" / "008_14_reference_comparison.csv"
    comp.to_csv(comp_path, index=False, encoding="utf-8-sig")

    stage_keep = [
        "stage_id",
        "decision_dap",
        "raw_stage_action_irrigation",
        "raw_stage_action_nitrogen",
        "stage_action_amir",
        "stage_action_anfer",
        "irrigation_before_gate",
        "irrigation_after_gate",
        "gate_future_rain",
        "gate_forecast_trigger",
        "gate_swfac_at_decision",
        "growth_reward",
        "terminal_reward",
        "soft_swfac_stress_days",
        "soft_swfac_penalty",
        "reward",
    ]
    stage_table = stages[[c for c in stage_keep if c in stages.columns]].copy() if not stages.empty else stages
    extra = 0.0
    if not stages.empty and "irrigation_before_gate" in stages and "irrigation_after_gate" in stages:
        before = pd.to_numeric(stages["irrigation_before_gate"], errors="coerce").fillna(0.0)
        after = pd.to_numeric(stages["irrigation_after_gate"], errors="coerce").fillna(0.0)
        extra = float(after.sum())
        blocked = float(np.maximum(before - after, 0.0).sum())
    else:
        blocked = 0.0
    reward_diag = {}
    if not stages.empty:
        growth_sum = float(pd.to_numeric(stages.get("growth_reward", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
        terminal_sum = float(pd.to_numeric(stages.get("terminal_reward", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
        stress_penalty_sum = float(pd.to_numeric(stages.get("soft_swfac_penalty", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
        water_cost_sum = float(pd.to_numeric(stages.get("stage_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()) * float(config["reward"].get("water_cost", 0.0))
        denominator = abs(growth_sum) + abs(terminal_sum) + abs(stress_penalty_sum) + abs(water_cost_sum)
        reward_diag = {
            "growth_reward_sum": growth_sum,
            "terminal_reward_sum": terminal_sum,
            "soft_swfac_penalty_sum": stress_penalty_sum,
            "water_cost_sum": water_cost_sum,
            "terminal_reward_share_abs_components": terminal_sum / denominator if denominator else np.nan,
        }

    lines = [
        f"# {config.get('report_title', 'HLA 2004 Irrigation-Only PPO With Soft Stress Reward')}",
        "",
        "## Purpose",
        "",
        "This smoke test checks whether PPO can learn irrigation in a clearly water-limited year without a hard minimum forecast-gate irrigation amount.",
        "Nitrogen is fixed at N150. PPO only has a meaningful irrigation decision.",
        "",
        "## Training",
        "",
        _df_md(train),
        "",
        "## Evaluation Summary",
        "",
        _df_md(summary),
        "",
        "## Reference Comparison",
        "",
        _df_md(comp),
        "",
        "## Stage Decisions",
        "",
        _df_md(stage_table),
        "",
        "## Reward Component Diagnostic",
        "",
        _df_md(pd.DataFrame([reward_diag]) if reward_diag else pd.DataFrame()),
        "",
        "## Interpretation",
        "",
    ]
    if not summary.empty:
        r = summary.iloc[0]
        total_i = float(r.get("total_irrigation", np.nan))
        frac = float(r.get("grnwt_fraction_vs_fixed_I120", np.nan))
        lines.extend(
            [
                f"- PPO irrigation total: {total_i:.2f} mm.",
                f"- Final GRNWT: {float(r.get('final_grnwt', np.nan)):.2f}.",
                f"- GRNWT / fixed_I120_N150: {frac:.3f}.",
                f"- SWFAC stress days: {int(r.get('swfac_days_gt_0p05', 0))}.",
                f"- Gate-blocked irrigation request: {blocked:.2f} mm.",
                f"- Effective PPO irrigation after gate: {extra:.2f} mm.",
            ]
        )
        if reward_diag:
            lines.append(f"- Terminal reward share of absolute tracked components: {float(reward_diag['terminal_reward_share_abs_components']):.3f}.")
        if not stage_table.empty and "raw_stage_action_irrigation" in stage_table.columns:
            raw_pattern = ", ".join(
                f"{row['stage_id']}={float(row['raw_stage_action_irrigation']):.2f}"
                for _, row in stage_table.iterrows()
                if pd.notna(row.get("raw_stage_action_irrigation"))
            )
            lines.append(f"- Raw PPO irrigation intention by stage: {raw_pattern}.")
        if total_i <= 1.0:
            lines.append("- PPO did not learn to irrigate under this 2000-step smoke setting; next test should adjust reward scale or timesteps.")
        elif frac >= 0.70:
            lines.append("- PPO learned a nonzero irrigation policy without hard-minimum gate support.")
        else:
            lines.append("- PPO used some irrigation but yield remains low; reward/action scaling still needs diagnosis.")
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Training summary: `{train_path.relative_to(PROJECT_ROOT)}`",
            f"- Evaluation summary: `{summary_path.relative_to(PROJECT_ROOT)}`",
            f"- Reference comparison: `{comp_path.relative_to(PROJECT_ROOT)}`",
            f"- Stage CSV: `{stage_path.relative_to(PROJECT_ROOT)}`",
            f"- Figures: `{(root / 'figures').relative_to(PROJECT_ROOT)}`",
        ]
    )
    base_runner.doc_md(config).write_text("\n".join(lines), encoding="utf-8")


def run(config_path: Path, report_only: bool = False) -> None:
    base_runner.make_env = make_env
    base_runner.run(config_path, report_only=report_only)
    config = direct.load_yaml(config_path)
    build_custom_report(config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    run(config_path, report_only=args.report_only)


if __name__ == "__main__":
    main()
