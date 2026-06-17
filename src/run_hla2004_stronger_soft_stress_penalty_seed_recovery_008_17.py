from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_all_year_direct_action_safe_ppo as direct
import run_hla2004_irrigation_only_ppo_soft_stress_reward_008_14 as soft_runner
from ppo_safe_rendering import PROJECT_ROOT


CONFIG_PATH = PROJECT_ROOT / "experiments" / "ppo_observed_years" / "config_008_17_hla2004_stronger_soft_stress_penalty_seed_recovery.yaml"


def root(config: dict[str, Any]) -> Path:
    return PROJECT_ROOT / config["paths"]["output_root"]


def doc_md(config: dict[str, Any]) -> Path:
    return PROJECT_ROOT / config["paths"]["doc_md"]


def seed_config(base: dict[str, Any], seed: int) -> tuple[dict[str, Any], Path]:
    cfg = {
        **base,
        "paths": dict(base["paths"]),
        "runtime": dict(base["runtime"]),
        "ppo": dict(base["ppo"]),
        "reward": dict(base["reward"]),
        "soft_stress_reward": dict(base["soft_stress_reward"]),
        "forecast_stress_gate": dict(base["forecast_stress_gate"]),
        "stage_action": {"stages": [dict(x) for x in base["stage_action"]["stages"]]},
        "success_criteria": dict(base["success_criteria"]),
        "economics": dict(base["economics"]),
        "action_safety": dict(base["action_safety"]),
        "station_years": [dict(x) for x in base["station_years"]],
    }
    cfg["seed"] = int(seed)
    cfg["seeds"] = [int(seed)]
    cfg["total_timesteps"] = 5000
    cfg["report_title"] = f"008_17 HLA 2004 Seed {seed} Stronger Soft-Stress Penalty"
    cfg["paths"]["output_root"] = str((root(base) / f"seed{seed}_5k").relative_to(PROJECT_ROOT))
    cfg["paths"]["doc_md"] = f"docs/2026-06-14_008_17_seed{seed}_hla2004_stronger_soft_stress_penalty_report.md"
    cfg["paths"]["doc_ppt"] = f"docs/019_008_17_seed{seed}_hla2004_stronger_soft_stress_penalty.pptx"
    out = root(base) / "configs" / f"config_008_17_seed{seed}_5k.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    direct.write_yaml(cfg, out)
    return cfg, out


def reward_diagnostic(stage_path: Path, water_cost: float) -> dict[str, float]:
    stages = pd.read_csv(stage_path) if stage_path.exists() else pd.DataFrame()
    if stages.empty:
        return {}
    growth = float(pd.to_numeric(stages.get("growth_reward", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    terminal = float(pd.to_numeric(stages.get("terminal_reward", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    stress = float(pd.to_numeric(stages.get("soft_swfac_penalty", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    water = float(pd.to_numeric(stages.get("stage_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()) * float(water_cost)
    denom = abs(growth) + abs(terminal) + abs(stress) + abs(water)
    out: dict[str, float] = {
        "growth_reward_sum": growth,
        "terminal_reward_sum": terminal,
        "soft_swfac_penalty_sum": stress,
        "water_cost_sum": water,
        "terminal_reward_share_abs_components": terminal / denom if denom else np.nan,
        "stress_penalty_share_abs_components": stress / denom if denom else np.nan,
    }
    for _, row in stages.iterrows():
        sid = str(row.get("stage_id"))
        out[f"{sid}_raw_i"] = float(pd.to_numeric(row.get("raw_stage_action_irrigation"), errors="coerce"))
        out[f"{sid}_irrigation"] = float(pd.to_numeric(row.get("stage_action_amir"), errors="coerce"))
    return out


def classify_recovery(row: pd.Series) -> str:
    total_i = float(row.get("total_irrigation", 0.0))
    saturated = bool(str(row.get("irrigation_cap_saturated", "False")).lower() == "true" or row.get("irrigation_cap_saturated") is True)
    frac = float(row.get("grnwt_fraction_vs_fixed_I120", np.nan))
    fixed_i60 = float(row.get("fixed_I60_N150_grnwt", np.nan))
    final = float(row.get("final_grnwt", np.nan))
    raw_values = [float(row.get(f"S{i}_raw_i", -1.0)) for i in range(1, 6)]
    if saturated:
        return "over_aggressive_saturated"
    if total_i <= 1e-6:
        return "not_recovered_zero_irrigation"
    if total_i > 120.0:
        return "over_aggressive_high_irrigation"
    if total_i > 20.0 and any(x > -0.9 for x in raw_values) and (final > fixed_i60 or frac >= 0.85):
        return "recovered"
    return "partial_recovery"


def read_result(seed: int, run_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    summary_path = run_root / "evaluation" / "HLA_2004_stress_aware_stage_ppo_summary.csv"
    stage_path = run_root / "daily_outputs" / "HLA" / f"HLA_2004_seed{seed}_stress_aware_stage_steps.csv"
    row = pd.read_csv(summary_path).iloc[0].to_dict()
    row.update(reward_diagnostic(stage_path, water_cost=float(config["reward"].get("water_cost", 0.0))))
    row["penalty_config"] = f"excess={config['soft_stress_reward']['swfac_excess_cost']};day={config['soft_stress_reward']['swfac_day_cost']}"
    row["seed_outcome"] = classify_recovery(pd.Series(row))
    row["stage_csv_path"] = str(stage_path.relative_to(PROJECT_ROOT))
    return row


def run_seed(base: dict[str, Any], seed: int) -> dict[str, Any]:
    cfg, cfg_path = seed_config(base, seed)
    soft_runner.run(cfg_path, report_only=False)
    return read_result(seed, PROJECT_ROOT / cfg["paths"]["output_root"], cfg)


def df_md(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return ""
    out = df.head(max_rows).copy()
    for col in out.select_dtypes(include=["float", "int"]).columns:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(4)
    keep = [
        "seed",
        "penalty_config",
        "seed_outcome",
        "total_irrigation",
        "total_n",
        "final_grnwt",
        "grnwt_fraction_vs_fixed_I120",
        "swfac_days_gt_0p05",
        "soft_swfac_penalty_sum",
        "stress_penalty_share_abs_components",
        "terminal_reward_share_abs_components",
        "S1_raw_i",
        "S2_raw_i",
        "S3_raw_i",
        "S4_raw_i",
        "S5_raw_i",
        "S1_irrigation",
        "S2_irrigation",
        "S3_irrigation",
        "S4_irrigation",
        "S5_irrigation",
    ]
    out = out[[c for c in keep if c in out.columns]]
    header = "| " + " | ".join(out.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(out.columns)) + " |"
    rows = ["| " + " | ".join("" if pd.isna(x) else str(x) for x in row) + " |" for row in out.values.tolist()]
    return "\n".join([header, sep, *rows])


def write_report(config: dict[str, Any], combined: pd.DataFrame, decision_log: list[str]) -> None:
    out_root = root(config)
    eval_dir = out_root / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    combined_path = eval_dir / "008_17_stronger_penalty_recovery_summary.csv"
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
    lines = [
        "# 008_17 HLA 2004 Stronger Soft-Stress Penalty Seed Recovery",
        "",
        "## Purpose",
        "",
        "This task tests whether stronger SWFAC soft-stress penalty can recover seed1 from the zero-irrigation degeneration observed in 008_16.",
        "",
        "## Decision Log",
        "",
        *[f"- {x}" for x in decision_log],
        "",
        "## Summary",
        "",
        df_md(combined),
        "",
        "## Interpretation",
        "",
    ]
    if not combined.empty:
        outcomes = set(combined["seed_outcome"].astype(str))
        if "recovered" in outcomes:
            lines.append("- At least one degenerated seed recovered under stronger SWFAC penalty.")
        if "over_aggressive_saturated" in outcomes or "over_aggressive_high_irrigation" in outcomes:
            lines.append("- Stronger penalty may be too aggressive because irrigation became saturated or excessive.")
        if "not_recovered_zero_irrigation" in outcomes:
            lines.append("- Stronger penalty did not remove the zero-irrigation attractor for at least one seed.")
        if len(combined) >= 2 and all(x == "recovered" for x in combined["seed_outcome"].astype(str)):
            lines.append("- Both tested seeds recovered; the next step can be seed2 or second-year validation.")
        elif len(combined) == 1:
            lines.append("- Only seed1 was run because the second seed depends on seed1 recovery.")
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Combined summary: `{combined_path.relative_to(PROJECT_ROOT)}`",
            f"- Output root: `{out_root.relative_to(PROJECT_ROOT)}`",
        ]
    )
    doc_md(config).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = direct.load_yaml(config_path)
    root(config).mkdir(parents=True, exist_ok=True)
    (root(config) / "configs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, root(config) / "configs" / config_path.name)

    rows = []
    decision_log: list[str] = []
    seed1 = run_seed(config, 1)
    rows.append(seed1)
    decision_log.append(f"Seed1 5k completed: {seed1['seed_outcome']}.")
    if seed1["seed_outcome"] == "recovered":
        seed0 = run_seed(config, 0)
        rows.append(seed0)
        decision_log.append(f"Seed0 5k completed under the same stronger penalty: {seed0['seed_outcome']}.")
    else:
        decision_log.append("Seed0 was not rerun because seed1 did not recover cleanly.")
    combined = pd.DataFrame(rows)
    write_report(config, combined, decision_log)
    print(root(config) / "evaluation" / "008_17_stronger_penalty_recovery_summary.csv")
    print(doc_md(config))


if __name__ == "__main__":
    main()

