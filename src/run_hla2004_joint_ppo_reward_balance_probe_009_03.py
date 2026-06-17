from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

import run_all_year_direct_action_safe_ppo as direct
import run_hla2004_stage_level_water_nitrogen_joint_ppo_009_02 as joint_runner
from ppo_safe_rendering import PROJECT_ROOT


BASE_CONFIG = PROJECT_ROOT / "experiments" / "ppo_observed_years" / "config_009_02_hla2004_stage_level_water_nitrogen_joint_ppo_smoke.yaml"
OUTPUT_ROOT = PROJECT_ROOT / "Leave_One_experiments" / "hla2004_joint_ppo_reward_balance_probe_009_03"
DOC_MD = PROJECT_ROOT / "docs" / "2026-06-17_009_03_hla2004_joint_ppo_reward_balance_probe_report.md"

SCENARIOS = [
    {
        "scenario": "009_02_baseline",
        "source": PROJECT_ROOT / "Leave_One_experiments" / "hla2004_stage_level_water_nitrogen_joint_ppo_009_02",
        "water_cost": 0.050,
        "nitrogen_cost": 0.030,
        "run": False,
    },
    {
        "scenario": "009_03A_lower_water_cost",
        "water_cost": 0.030,
        "nitrogen_cost": 0.030,
        "run": True,
    },
    {
        "scenario": "009_03B_higher_nitrogen_cost",
        "water_cost": 0.050,
        "nitrogen_cost": 0.050,
        "run": True,
    },
]


def scenario_root(name: str) -> Path:
    return OUTPUT_ROOT / "scenarios" / name


def write_config(base: dict[str, Any], scenario: dict[str, Any]) -> Path:
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
    name = scenario["scenario"]
    cfg["report_title"] = f"{name} HLA 2004 Joint PPO Reward Balance Probe"
    cfg["reward"]["water_cost"] = float(scenario["water_cost"])
    cfg["reward"]["nitrogen_cost"] = float(scenario["nitrogen_cost"])
    cfg["paths"]["output_root"] = str(scenario_root(name).relative_to(PROJECT_ROOT))
    cfg["paths"]["doc_md"] = f"docs/2026-06-17_{name}_hla2004_joint_ppo_reward_balance_probe_report.md"
    cfg["paths"]["doc_ppt"] = f"docs/{name}_hla2004_joint_ppo_reward_balance_probe.pptx"
    config_path = OUTPUT_ROOT / "configs" / f"config_{name}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    direct.write_yaml(cfg, config_path)
    return config_path


def read_summary(root: Path, scenario: str, water_cost: float, nitrogen_cost: float) -> dict[str, Any]:
    summary_path = root / "evaluation" / "HLA_2004_joint_ppo_summary.csv"
    stage_path = root / "daily_outputs" / "HLA" / "HLA_2004_seed0_soft_stress_stage_steps.csv"
    row = pd.read_csv(summary_path).iloc[0].to_dict()
    stages = pd.read_csv(stage_path) if stage_path.exists() else pd.DataFrame()
    row["scenario"] = scenario
    row["water_cost"] = water_cost
    row["nitrogen_cost"] = nitrogen_cost
    if not stages.empty:
        for sid in ["S3", "S4", "S5"]:
            hit = stages[stages["stage_id"].astype(str).eq(sid)]
            if not hit.empty:
                r = hit.iloc[0]
                row[f"{sid}_irrigation"] = float(pd.to_numeric(r.get("stage_action_amir"), errors="coerce"))
                row[f"{sid}_extra_n"] = float(pd.to_numeric(r.get("ppo_extra_n_applied", r.get("stage_extra_anfer")), errors="coerce"))
                row[f"{sid}_raw_i"] = float(pd.to_numeric(r.get("raw_stage_action_irrigation"), errors="coerce"))
                row[f"{sid}_raw_n"] = float(pd.to_numeric(r.get("raw_stage_action_nitrogen"), errors="coerce"))
    row["summary_path"] = str(summary_path.relative_to(PROJECT_ROOT))
    row["stage_path"] = str(stage_path.relative_to(PROJECT_ROOT))
    return row


def df_md(df: pd.DataFrame, max_rows: int = 20) -> str:
    out = df.head(max_rows).copy()
    keep = [
        "scenario",
        "water_cost",
        "nitrogen_cost",
        "total_irrigation",
        "total_n",
        "total_ppo_extra_n",
        "final_grnwt",
        "grnwt_fraction_vs_00815_irrigation_only",
        "swfac_days_gt_0p05",
        "nstres_days_gt_0p05",
        "soft_swfac_penalty_total",
        "soft_nstres_penalty_total",
        "S4_irrigation",
        "S5_irrigation",
        "S3_extra_n",
        "S4_extra_n",
        "S5_extra_n",
        "debug_joint_promising",
    ]
    out = out[[c for c in keep if c in out.columns]]
    for col in out.select_dtypes(include=["float", "int"]).columns:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(4)
    header = "| " + " | ".join(out.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(out.columns)) + " |"
    rows = ["| " + " | ".join("" if pd.isna(x) else str(x) for x in row) + " |" for row in out.values.tolist()]
    return "\n".join([header, sep, *rows])


def write_report(summary: pd.DataFrame, summary_path: Path) -> None:
    lines = [
        "# 009_03 HLA 2004 Joint PPO Reward Balance Probe",
        "",
        "## Purpose",
        "",
        "This probe checks whether 009_02 used too little irrigation and too much nitrogen because water and nitrogen reward costs were imbalanced.",
        "",
        "## Results",
        "",
        df_md(summary),
        "",
        "## Interpretation",
        "",
    ]
    base = summary[summary["scenario"].eq("009_02_baseline")]
    loww = summary[summary["scenario"].eq("009_03A_lower_water_cost")]
    highn = summary[summary["scenario"].eq("009_03B_higher_nitrogen_cost")]
    if not base.empty and not loww.empty:
        b = base.iloc[0]
        a = loww.iloc[0]
        lines.append(
            f"- Lowering water cost changed irrigation from {float(b['total_irrigation']):.2f} to {float(a['total_irrigation']):.2f} mm."
        )
        lines.append(
            f"- S4/S5 irrigation changed from {float(b.get('S4_irrigation', 0)):.2f}/{float(b.get('S5_irrigation', 0)):.2f} to {float(a.get('S4_irrigation', 0)):.2f}/{float(a.get('S5_irrigation', 0)):.2f} mm."
        )
        lines.append(
            f"- SWFAC penalty changed from {float(b['soft_swfac_penalty_total']):.2f} to {float(a['soft_swfac_penalty_total']):.2f}."
        )
    if not base.empty and not highn.empty:
        b = base.iloc[0]
        c = highn.iloc[0]
        lines.append(
            f"- Increasing nitrogen cost changed total N from {float(b['total_n']):.2f} to {float(c['total_n']):.2f} kg/ha."
        )
        lines.append(
            f"- NSTRES penalty changed from {float(b['soft_nstres_penalty_total']):.2f} to {float(c['soft_nstres_penalty_total']):.2f}."
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Combined summary: `{summary_path.relative_to(PROJECT_ROOT)}`",
            f"- Output root: `{OUTPUT_ROOT.relative_to(PROJECT_ROOT)}`",
        ]
    )
    DOC_MD.parent.mkdir(parents=True, exist_ok=True)
    DOC_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "configs").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "evaluation").mkdir(parents=True, exist_ok=True)
    base = direct.load_yaml(BASE_CONFIG)
    rows = []
    for scenario in SCENARIOS:
        if scenario.get("run", False):
            cfg_path = write_config(base, scenario)
            joint_runner.run(cfg_path, report_only=False)
            root = scenario_root(scenario["scenario"])
        else:
            root = scenario["source"]
        rows.append(read_summary(root, scenario["scenario"], float(scenario["water_cost"]), float(scenario["nitrogen_cost"])))
    summary = pd.DataFrame(rows)
    summary_path = OUTPUT_ROOT / "evaluation" / "009_03_reward_balance_probe_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    write_report(summary, summary_path)
    print(summary_path)
    print(DOC_MD)


if __name__ == "__main__":
    main()

