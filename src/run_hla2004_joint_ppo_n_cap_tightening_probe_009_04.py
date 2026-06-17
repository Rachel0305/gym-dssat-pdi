from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

import run_all_year_direct_action_safe_ppo as direct
import run_hla2004_joint_ppo_reward_balance_probe_009_03 as probe_00903
import run_hla2004_stage_level_water_nitrogen_joint_ppo_009_02 as joint_runner
from ppo_safe_rendering import PROJECT_ROOT


BASE_CONFIG = PROJECT_ROOT / "experiments" / "ppo_observed_years" / "config_009_02_hla2004_stage_level_water_nitrogen_joint_ppo_smoke.yaml"
OUTPUT_ROOT = PROJECT_ROOT / "Leave_One_experiments" / "hla2004_joint_ppo_n_cap_tightening_probe_009_04"
DOC_MD = PROJECT_ROOT / "docs" / "2026-06-17_009_04_hla2004_joint_ppo_n_cap_tightening_probe_report.md"
CONFIG_PATH = OUTPUT_ROOT / "configs" / "config_009_04_n_cap_tightening.yaml"

REF_00903 = PROJECT_ROOT / "Leave_One_experiments" / "hla2004_joint_ppo_reward_balance_probe_009_03" / "evaluation" / "009_03_reward_balance_probe_summary.csv"
REF_00819 = PROJECT_ROOT / "Leave_One_experiments" / "hla2004_four_scenario_process_plots_008_19" / "evaluation" / "008_19_hla2004_four_scenario_summary.csv"


def configure_00904(base: dict[str, Any]) -> dict[str, Any]:
    cfg = {
        **base,
        "paths": dict(base["paths"]),
        "reward": dict(base["reward"]),
        "runtime": dict(base["runtime"]),
        "ppo": dict(base["ppo"]),
        "stage_action": {"stages": [dict(x) for x in base["stage_action"]["stages"]]},
        "forecast_stress_gate": dict(base["forecast_stress_gate"]),
        "soft_stress_reward": dict(base["soft_stress_reward"]),
        "soft_nstres_reward": dict(base["soft_nstres_reward"]),
        "economics": dict(base["economics"]),
        "action_safety": dict(base["action_safety"]),
        "success_criteria": dict(base["success_criteria"]),
        "station_years": [dict(x) for x in base["station_years"]],
    }
    cfg["report_title"] = "009_04 HLA 2004 Joint PPO N-Cap Tightening Probe"
    cfg["paths"]["output_root"] = str(OUTPUT_ROOT.relative_to(PROJECT_ROOT))
    cfg["paths"]["doc_md"] = str(DOC_MD.relative_to(PROJECT_ROOT))
    cfg["paths"]["doc_ppt"] = "docs/022_009_04_hla2004_joint_ppo_n_cap_tightening_probe.pptx"
    cfg["reward"]["water_cost"] = 0.030
    cfg["reward"]["nitrogen_cost"] = 0.030
    cfg["success_criteria"]["cap_saturation_threshold_n"] = 150.0
    cfg["success_criteria"]["min_total_n"] = 110.0

    new_n_caps = {"S1": 50.0, "S2": 50.0, "S3": 40.0, "S4": 10.0, "S5": 0.0}
    for stage in cfg["stage_action"]["stages"]:
        sid = str(stage["stage_id"])
        if sid in new_n_caps:
            stage["nitrogen_cap"] = new_n_caps[sid]
    return cfg


def run_00904() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "configs").mkdir(parents=True, exist_ok=True)
    base = direct.load_yaml(BASE_CONFIG)
    cfg = configure_00904(base)
    direct.write_yaml(cfg, CONFIG_PATH)
    joint_runner.run(CONFIG_PATH, report_only=False)


def load_main_row() -> dict[str, Any]:
    summary_path = OUTPUT_ROOT / "evaluation" / "HLA_2004_joint_ppo_summary.csv"
    stage_path = OUTPUT_ROOT / "daily_outputs" / "HLA" / "HLA_2004_seed0_soft_stress_stage_steps.csv"
    return probe_00903.read_summary(OUTPUT_ROOT, "009_04_n_cap_tightening", 0.030, 0.030) | {
        "summary_path": str(summary_path.relative_to(PROJECT_ROOT)),
        "stage_path": str(stage_path.relative_to(PROJECT_ROOT)),
    }


def comparison_df(row_00904: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if REF_00903.exists():
        ref = pd.read_csv(REF_00903)
        rows.extend(ref.to_dict("records"))
    rows.append(row_00904)
    out = pd.DataFrame(rows)
    wanted = [
        "scenario",
        "water_cost",
        "nitrogen_cost",
        "total_irrigation",
        "total_n",
        "total_base_n",
        "total_ppo_extra_n",
        "final_grnwt",
        "grnwt_fraction_vs_00815_irrigation_only",
        "swfac_days_gt_0p05",
        "nstres_days_gt_0p05",
        "soft_swfac_penalty_total",
        "soft_nstres_penalty_total",
        "S3_irrigation",
        "S4_irrigation",
        "S5_irrigation",
        "S3_extra_n",
        "S4_extra_n",
        "S5_extra_n",
        "n_cap_saturated",
        "debug_joint_promising",
    ]
    return out[[c for c in wanted if c in out.columns]]


def fixed_i120_grnwt() -> float | None:
    if not REF_00819.exists():
        return None
    df = pd.read_csv(REF_00819)
    for col in ["scenario", "management", "policy"]:
        if col in df.columns:
            hit = df[df[col].astype(str).str.contains("I120|fixed_I120", case=False, regex=True, na=False)]
            if not hit.empty and "final_grnwt" in hit.columns:
                return float(pd.to_numeric(hit["final_grnwt"], errors="coerce").dropna().iloc[0])
    return None


def write_report(comp: pd.DataFrame, row_00904: dict[str, Any]) -> None:
    report_comp = comp.copy()
    for col in report_comp.columns:
        if col != "scenario":
            report_comp[col] = pd.to_numeric(report_comp[col], errors="ignore")
    fixed = fixed_i120_grnwt()
    lines = [
        "# 009_04 HLA 2004 Joint PPO N-Cap Tightening Probe",
        "",
        "## Purpose",
        "",
        "009_03 showed that lower water cost restored irrigation, but nitrogen remained near 190 kg/ha even when nitrogen cost was increased. 009_04 tests whether tightening supplemental N caps can reduce high N input while preserving yield.",
        "",
        "## Configuration",
        "",
        "- HLA 2004, seed 0, 5000 timesteps.",
        "- `water_cost=0.030`, `nitrogen_cost=0.030`.",
        "- Base N remains S1=50 kg/ha and S2=50 kg/ha.",
        "- PPO supplemental N caps are tightened to S3=40, S4=10, S5=0 kg/ha.",
        "- Total N upper bound is therefore 150 kg/ha.",
        "",
        "## Comparison",
        "",
        direct.df_to_markdown(report_comp),
        "",
        "## Interpretation",
        "",
    ]
    r = row_00904
    total_n = float(r["total_n"])
    total_i = float(r["total_irrigation"])
    grnwt = float(r["final_grnwt"])
    lines.extend(
        [
            f"- 009_04 total N: {total_n:.2f} kg/ha.",
            f"- 009_04 total irrigation: {total_i:.2f} mm.",
            f"- 009_04 final GRNWT: {grnwt:.2f} kg/ha.",
            f"- N cap saturated: {r.get('n_cap_saturated')}.",
            f"- SWFAC stress days: {int(float(r.get('swfac_days_gt_0p05', 0)))}; NSTRES stress days: {int(float(r.get('nstres_days_gt_0p05', 0)))}.",
        ]
    )
    if fixed is not None:
        lines.append(f"- GRNWT / fixed I120_N150 reference: {grnwt / fixed:.3f}.")

    lines.extend(
        [
            "",
            "## Decision",
            "",
        ]
    )
    if total_n <= 160 and grnwt >= 6900 and 70 <= total_i <= 90:
        lines.append("- 009_04 meets the practical target: lower N around 150 kg/ha, recovered irrigation, and preserved yield.")
    elif total_n <= 160 and grnwt < 6900:
        lines.append("- 009_04 reduced N but lost too much yield; consider a slightly wider S3 or S4 N cap.")
    elif total_n > 160:
        lines.append("- 009_04 still uses too much N; further cap tightening or revised N timing is needed.")
    else:
        lines.append("- 009_04 needs inspection before expanding to seed stability.")

    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Summary: `{(OUTPUT_ROOT / 'evaluation' / 'HLA_2004_joint_ppo_summary.csv').relative_to(PROJECT_ROOT)}`",
            f"- Stage decisions: `{(OUTPUT_ROOT / 'daily_outputs' / 'HLA' / 'HLA_2004_seed0_soft_stress_stage_steps.csv').relative_to(PROJECT_ROOT)}`",
            f"- Daily CSV: `{(OUTPUT_ROOT / 'daily_outputs' / 'HLA' / 'HLA_2004_seed0_soft_stress_stage_daily.csv').relative_to(PROJECT_ROOT)}`",
            f"- Figure folder: `{(OUTPUT_ROOT / 'figures').relative_to(PROJECT_ROOT)}`",
        ]
    )
    DOC_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    run_00904()
    row = load_main_row()
    comp = comparison_df(row)
    comp_path = OUTPUT_ROOT / "evaluation" / "009_04_n_cap_tightening_comparison.csv"
    comp_path.parent.mkdir(parents=True, exist_ok=True)
    comp.to_csv(comp_path, index=False, encoding="utf-8-sig")
    write_report(comp, row)
    print(comp_path)
    print(DOC_MD)


if __name__ == "__main__":
    main()
