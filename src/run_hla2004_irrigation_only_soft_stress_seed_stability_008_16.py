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


CONFIG_PATH = PROJECT_ROOT / "experiments" / "ppo_observed_years" / "config_008_16_hla2004_irrigation_only_soft_stress_ppo_seed_stability.yaml"
SEED0_ROOT = PROJECT_ROOT / "Leave_One_experiments" / "hla2004_irrigation_only_soft_stress_ppo_008_15_10k"
SEED0_SUMMARY = SEED0_ROOT / "evaluation" / "HLA_2004_stress_aware_stage_ppo_summary.csv"
SEED0_STAGE = SEED0_ROOT / "daily_outputs" / "HLA" / "HLA_2004_seed0_stress_aware_stage_steps.csv"


def root(config: dict[str, Any]) -> Path:
    return PROJECT_ROOT / config["paths"]["output_root"]


def doc_md(config: dict[str, Any]) -> Path:
    return PROJECT_ROOT / config["paths"]["doc_md"]


def load_stage_pattern(stage_path: Path) -> dict[str, Any]:
    stages = pd.read_csv(stage_path) if stage_path.exists() else pd.DataFrame()
    out: dict[str, Any] = {}
    if stages.empty:
        return out
    for _, row in stages.iterrows():
        sid = str(row.get("stage_id"))
        out[f"{sid}_raw_i"] = float(pd.to_numeric(row.get("raw_stage_action_irrigation"), errors="coerce"))
        out[f"{sid}_irrigation"] = float(pd.to_numeric(row.get("stage_action_amir"), errors="coerce"))
    return out


def classify_seed(row: pd.Series) -> str:
    total_i = float(row.get("total_irrigation", np.nan))
    saturated = bool(str(row.get("irrigation_cap_saturated", "False")).lower() == "true" or row.get("irrigation_cap_saturated") is True)
    if total_i <= 1e-6 or saturated:
        return "degeneration"
    s1 = float(row.get("S1_raw_i", np.nan))
    s2 = float(row.get("S2_raw_i", np.nan))
    s3 = float(row.get("S3_raw_i", np.nan))
    s4 = float(row.get("S4_raw_i", np.nan))
    s5 = float(row.get("S5_raw_i", np.nan))
    s2_i = float(row.get("S2_irrigation", 0.0))
    s3_i = float(row.get("S3_irrigation", 0.0))
    if (60.0 <= total_i <= 100.0) and all(x < 0 for x in [s1, s2, s3]) and all(x > 0 for x in [s4, s5]):
        return "stable"
    if total_i < 20.0 or total_i > 110.0 or ((s2 > 0 and s2_i > 0) or (s3 > 0 and s3_i > 0)):
        return "drift"
    return "partial_stable"


def read_seed_result(seed: int, run_root: Path, source_label: str) -> dict[str, Any]:
    summary_path = run_root / "evaluation" / "HLA_2004_stress_aware_stage_ppo_summary.csv"
    stage_path = run_root / "daily_outputs" / "HLA" / f"HLA_2004_seed{seed}_stress_aware_stage_steps.csv"
    summary = pd.read_csv(summary_path).iloc[0].to_dict()
    summary.update(load_stage_pattern(stage_path))
    summary["source_label"] = source_label
    summary["seed_classification"] = classify_seed(pd.Series(summary))
    summary["stage_csv_path"] = str(stage_path.relative_to(PROJECT_ROOT))
    return summary


def seed_config(base: dict[str, Any], seed: int) -> tuple[dict[str, Any], Path]:
    cfg = direct.copy.deepcopy(base) if hasattr(direct, "copy") else {**base}
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
    cfg["report_title"] = f"008_16 HLA 2004 Seed {seed} Soft-Stress PPO 5k"
    cfg["paths"]["output_root"] = str((root(base) / f"seed{seed}_5k").relative_to(PROJECT_ROOT))
    cfg["paths"]["doc_md"] = f"docs/2026-06-14_008_16_seed{seed}_hla2004_soft_stress_ppo_5k_report.md"
    cfg["paths"]["doc_ppt"] = f"docs/018_008_16_seed{seed}_hla2004_soft_stress_ppo_5k.pptx"
    out = root(base) / "configs" / f"config_008_16_seed{seed}_5k.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    direct.write_yaml(cfg, out)
    return cfg, out


def run_seed(base: dict[str, Any], seed: int) -> dict[str, Any]:
    cfg, cfg_path = seed_config(base, seed)
    soft_runner.run(cfg_path, report_only=False)
    return read_seed_result(seed, PROJECT_ROOT / cfg["paths"]["output_root"], f"008_16_seed{seed}_5k")


def df_md(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return ""
    out = df.head(max_rows).copy()
    for col in out.select_dtypes(include=["float", "int"]).columns:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(4)
    keep = [
        "seed",
        "source_label",
        "seed_classification",
        "total_irrigation",
        "total_n",
        "final_grnwt",
        "grnwt_fraction_vs_fixed_I120",
        "first_irrigation_dap",
        "swfac_days_gt_0p05",
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
    out_root.mkdir(parents=True, exist_ok=True)
    eval_dir = out_root / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    combined_path = eval_dir / "008_16_seed_stability_summary.csv"
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
    completed = combined[combined["source_label"].astype(str).str.contains("008_16|008_15", regex=True)]
    stable_count = int((completed["seed_classification"] == "stable").sum()) if not completed.empty else 0
    lines = [
        "# 008_16 HLA 2004 Irrigation-Only Soft-Stress PPO Seed Stability",
        "",
        "## Purpose",
        "",
        "This task checks whether the 008_14/008_15 late-irrigation PPO strategy is reproducible across random seeds.",
        "Seed0 is read from 008_15 and not rerun. Seed1 is run first; seed2 is run only if seed1 is stable.",
        "",
        "## Decision Log",
        "",
        *[f"- {x}" for x in decision_log],
        "",
        "## Seed Stability Summary",
        "",
        df_md(combined),
        "",
        "## Interpretation",
        "",
    ]
    lines.append(f"- Stable seeds among completed rows: {stable_count}/{len(completed)}.")
    if not combined.empty and (combined["seed_classification"] == "stable").all():
        lines.append("- The S4/S5 late-irrigation pattern is reproducible under the tested seeds.")
        lines.append("- This supports proceeding to 008_17 second water-stress year validation.")
    elif "degeneration" in set(combined["seed_classification"].astype(str)):
        lines.append("- At least one seed degenerated; do not proceed to 008_17 before diagnosing reward/action design.")
    else:
        lines.append("- Seed behavior is mixed; inspect stage raw actions before deciding whether to extend timesteps.")
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
    seed0 = read_seed_result(0, SEED0_ROOT, "008_15_seed0_10k")
    rows.append(seed0)
    decision_log.append(f"Seed0 read from 008_15: {seed0['seed_classification']}.")

    seed1 = run_seed(config, 1)
    rows.append(seed1)
    decision_log.append(f"Seed1 5k completed: {seed1['seed_classification']}.")

    if seed1["seed_classification"] == "stable":
        seed2 = run_seed(config, 2)
        rows.append(seed2)
        decision_log.append(f"Seed2 5k completed because seed1 was stable: {seed2['seed_classification']}.")
    elif seed1["seed_classification"] == "drift":
        decision_log.append("Seed1 drifted; seed2 was not run. Inspect before increasing timesteps.")
    else:
        decision_log.append("Seed1 degenerated or was not stable; seed2 was not run.")

    combined = pd.DataFrame(rows)
    write_report(config, combined, decision_log)
    print(root(config) / "evaluation" / "008_16_seed_stability_summary.csv")
    print(doc_md(config))


if __name__ == "__main__":
    main()

