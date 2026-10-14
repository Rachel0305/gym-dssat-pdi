from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_lc_multiyear_free_timing_ppo_smoke_032_10 as eval_base


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
PROMPT = ROOT / "prompts" / "032_12_lc_multiyear_75k_future_year_transfer.md"
SOURCE_MODEL = ROOT / "benchmark_results" / "032_11_lc_multiyear_free_timing_ppo_training_length" / "models" / "LCA" / "LCA_multiyear_2005_2010_stress_aware_maskableppo_seed0_ckpt75000.zip"
BASELINE_SUMMARY = ROOT / "benchmark_results" / "031_36_missing_dssat_auto_completion_for_03134" / "evaluation" / "031_36_full_completed_template_aware_unified_baseline_summary.csv"
OUT = ROOT / "benchmark_results" / "032_12_lc_multiyear_75k_future_year_transfer"
DOC = ROOT / "docs" / "032_12_lc_multiyear_75k_future_year_transfer_record.md"
POOL = ROOT / "Leave_One_experiments" / "all_year_weather_calibration_validation" / "scenario_pool" / "all_year_weather_scenario_pool.csv"

STATION = "LCA"
SITE = "LC"
SOURCE_SEED = 0
SOURCE_CHECKPOINT = 75000
TARGET_YEARS = list(range(2011, 2021))


def ensure_dirs() -> None:
    for rel in ["configs", "daily_outputs/LCA", "evaluation", "tables"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "No rows."
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(3)
    work = work.astype(object).where(pd.notna(work), "")
    header = "| " + " | ".join(map(str, work.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in work.to_numpy().tolist()]
    return "\n".join([header, sep, *rows])


def load_config() -> dict[str, Any]:
    cfg = direct_ppo.load_yaml(CONFIG)
    cfg = json.loads(json.dumps(cfg))
    cfg["seed"] = SOURCE_SEED
    cfg["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    cfg["runtime"]["smoke_station"] = STATION
    cfg["runtime"]["smoke_year"] = TARGET_YEARS[0]
    return cfg


def make_selection() -> pd.DataFrame:
    pool = pd.read_csv(POOL)
    rows = pool[(pool["station_code"].eq(STATION)) & (pool["year"].astype(int).isin(TARGET_YEARS))].copy()
    found = sorted(rows["year"].astype(int).unique().tolist())
    if found != TARGET_YEARS:
        raise RuntimeError(f"Expected LC target years {TARGET_YEARS}, found {found}")
    rows["selected_for_train"] = False
    rows["selected_for_eval"] = True
    rows["selection_reason"] = "032_12_lc_multiyear_75k_future_year_transfer"
    return rows.sort_values(["station_code", "year"]).reset_index(drop=True)


def patch_eval_globals() -> None:
    eval_base.OUT = OUT
    eval_base.DOC = DOC
    eval_base.SEED = SOURCE_SEED
    eval_base.STATION = STATION


def run_frozen_eval(config: dict[str, Any], env_config: dict[str, Any]) -> pd.DataFrame:
    row = pd.Series(
        {
            "station_code": STATION,
            "year": "",
            "seed": SOURCE_SEED,
            "checkpoint_step": SOURCE_CHECKPOINT,
            "run_status": "ok",
            "model_path": str(SOURCE_MODEL.relative_to(ROOT)).replace("\\", "/"),
        }
    )
    rows = [eval_base.evaluate_checkpoint(config, env_config, row, year) for year in TARGET_YEARS]
    out = pd.DataFrame(rows)
    out["target_year"] = out.get("year", pd.Series(TARGET_YEARS)).astype(int)
    out["source_training_years"] = "2005-2010"
    out["source_checkpoint_step"] = SOURCE_CHECKPOINT
    return out


def pfp(yield_kg_ha: float, n_kg_ha: float) -> float:
    if not np.isfinite(yield_kg_ha) or not np.isfinite(n_kg_ha) or n_kg_ha <= 0:
        return np.nan
    return float(yield_kg_ha / n_kg_ha)


def compare_to_baselines(eval_df: pd.DataFrame) -> pd.DataFrame:
    if not BASELINE_SUMMARY.exists():
        out = eval_df.copy()
        out["baseline_status"] = "missing_baseline_summary"
        return out
    base = pd.read_csv(BASELINE_SUMMARY)
    base = base[(base["station_code"].eq(STATION)) & (base["year"].astype(int).isin(TARGET_YEARS))].copy()
    metric_rows: list[dict[str, Any]] = []
    for year, rl_row in eval_df.groupby("target_year"):
        rl = rl_row.iloc[0]
        b = base[base["year"].astype(int).eq(int(year))]
        expert = b[b["scenario"].eq("official_extension_expert")]
        four = b[b["scenario"].isin(["null", "recorded_farmer", "dssat_auto", "official_extension_expert"])]
        rl_y = float(rl.get("final_grnwt", np.nan))
        rl_i = float(rl.get("total_irrigation", np.nan))
        rl_n = float(rl.get("total_n", np.nan))
        rl_pfp = pfp(rl_y, rl_n)
        row: dict[str, Any] = {
            "site": SITE,
            "station_code": STATION,
            "year": int(year),
            "rl_yield_kg_ha": rl_y,
            "rl_irrigation_mm": rl_i,
            "rl_nitrogen_kg_ha": rl_n,
            "rl_PFP_N_kg_kg": rl_pfp,
            "rl_max_water_stress": rl.get("max_swfac", np.nan),
            "rl_max_nitrogen_stress": rl.get("max_nstres", np.nan),
            "rl_action_sequence": rl.get("action_sequence", ""),
            "baseline_rows": int(len(four)),
        }
        if not expert.empty:
            ex = expert.iloc[0]
            row.update(
                {
                    "expert_yield_kg_ha": float(ex.get("grain_yield_kg_ha", np.nan)),
                    "expert_irrigation_mm": float(ex.get("actual_irrigation_mm", np.nan)),
                    "expert_nitrogen_kg_ha": float(ex.get("actual_nitrogen_kg_ha", np.nan)),
                    "expert_PFP_N_kg_kg": float(ex.get("PFP_N_kg_kg", np.nan)),
                    "yield_gap_vs_expert": rl_y - float(ex.get("grain_yield_kg_ha", np.nan)),
                    "water_saving_vs_expert": float(ex.get("actual_irrigation_mm", np.nan)) - rl_i,
                    "n_saving_vs_expert": float(ex.get("actual_nitrogen_kg_ha", np.nan)) - rl_n,
                    "PFP_N_gap_vs_expert": rl_pfp - float(ex.get("PFP_N_kg_kg", np.nan)),
                }
            )
        if not four.empty:
            row.update(
                {
                    "four_max_yield_kg_ha": float(pd.to_numeric(four["grain_yield_kg_ha"], errors="coerce").max()),
                    "four_max_PFP_N_kg_kg": float(pd.to_numeric(four["PFP_N_kg_kg"], errors="coerce").max()),
                    "four_min_irrigation_mm_positive_yield": float(
                        pd.to_numeric(four.loc[pd.to_numeric(four["grain_yield_kg_ha"], errors="coerce") > 0, "actual_irrigation_mm"], errors="coerce").min()
                    ),
                    "four_min_nitrogen_kg_ha_positive_yield": float(
                        pd.to_numeric(four.loc[pd.to_numeric(four["grain_yield_kg_ha"], errors="coerce") > 0, "actual_nitrogen_kg_ha"], errors="coerce").min()
                    ),
                }
            )
            row["yield_gap_vs_four_max"] = rl_y - row["four_max_yield_kg_ha"]
            row["PFP_N_gap_vs_four_max"] = rl_pfp - row["four_max_PFP_N_kg_kg"] if np.isfinite(rl_pfp) else np.nan
        metric_rows.append(row)
    comp = pd.DataFrame(metric_rows)
    comp["yield_ge_expert"] = comp["yield_gap_vs_expert"] >= -1e-9
    comp["saves_water_vs_expert"] = comp["water_saving_vs_expert"] > 1e-9
    comp["saves_n_vs_expert"] = comp["n_saving_vs_expert"] > 1e-9
    comp["PFP_N_ge_expert"] = comp["PFP_N_gap_vs_expert"] >= -1e-9
    return comp


def write_record(eval_df: pd.DataFrame, comp_df: pd.DataFrame) -> None:
    pass_eval = bool(len(eval_df) == len(TARGET_YEARS) and eval_df["run_status"].astype(str).str.startswith("ok").all())
    pass_baseline = bool(len(comp_df) == len(TARGET_YEARS) and (comp_df.get("baseline_rows", pd.Series(dtype=int)).fillna(0) >= 4).all())
    key_cols = [
        "year",
        "rl_yield_kg_ha",
        "expert_yield_kg_ha",
        "yield_gap_vs_expert",
        "rl_irrigation_mm",
        "expert_irrigation_mm",
        "water_saving_vs_expert",
        "rl_nitrogen_kg_ha",
        "expert_nitrogen_kg_ha",
        "n_saving_vs_expert",
        "rl_PFP_N_kg_kg",
        "expert_PFP_N_kg_kg",
        "PFP_N_gap_vs_expert",
        "rl_max_water_stress",
        "rl_max_nitrogen_stress",
    ]
    eval_cols = [
        "target_year",
        "run_status",
        "final_grnwt",
        "total_irrigation",
        "total_n",
        "PFP_N",
        "reward_stress_aware_sum",
        "max_swfac",
        "max_nstres",
        "action_sequence",
    ]
    summary = {
        "years": len(comp_df),
        "yield_ge_expert": int(comp_df.get("yield_ge_expert", pd.Series(dtype=bool)).sum()),
        "water_saving_vs_expert": int(comp_df.get("saves_water_vs_expert", pd.Series(dtype=bool)).sum()),
        "n_saving_vs_expert": int(comp_df.get("saves_n_vs_expert", pd.Series(dtype=bool)).sum()),
        "PFP_N_ge_expert": int(comp_df.get("PFP_N_ge_expert", pd.Series(dtype=bool)).sum()),
        "mean_yield_gap_vs_expert": float(pd.to_numeric(comp_df.get("yield_gap_vs_expert", pd.Series(dtype=float)), errors="coerce").mean()),
        "mean_water_saving_vs_expert": float(pd.to_numeric(comp_df.get("water_saving_vs_expert", pd.Series(dtype=float)), errors="coerce").mean()),
        "mean_n_saving_vs_expert": float(pd.to_numeric(comp_df.get("n_saving_vs_expert", pd.Series(dtype=float)), errors="coerce").mean()),
    }
    lines = [
        "# 032_12 LC multi-year 75k free-timing MaskablePPO future-year transfer record",
        "",
        "## Status",
        "",
        f"- Evaluation pass: `{pass_eval}`.",
        f"- Four-baseline comparison table complete: `{pass_baseline}`.",
        "- Training in this task: 0 timesteps.",
        "",
        "## Source model",
        "",
        f"- Model: `{SOURCE_MODEL.relative_to(ROOT).as_posix()}`.",
        "- Source training years: LC2005-LC2010.",
        "- Source checkpoint: 75,000 timesteps from 032_11.",
        "- Algorithm: no-forecast free-timing MaskablePPO.",
        "",
        "## Target years",
        "",
        f"- {', '.join(map(str, TARGET_YEARS))}.",
        "",
        "## Summary versus official expert",
        "",
        md_table(pd.DataFrame([summary])),
        "",
        "## Per-year comparison versus official expert",
        "",
        md_table(comp_df[[c for c in key_cols if c in comp_df.columns]], max_rows=80),
        "",
        "## Frozen PPO endpoint details",
        "",
        md_table(eval_df[[c for c in eval_cols if c in eval_df.columns]], max_rows=80),
        "",
        "## Interpretation boundary",
        "",
        "- This task evaluates one frozen 75k checkpoint only.",
        "- It does not retrain on 2011-2020.",
        "- It does not include weather forecast features.",
        "- WP_ET for RL is not reported here unless ET denominator is available in the replay outputs; no ET value is inferred.",
    ]
    text = "\n".join(lines) + "\n"
    DOC.write_text(text, encoding="utf-8")
    (OUT / DOC.name).write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    if not SOURCE_MODEL.exists():
        raise FileNotFoundError(SOURCE_MODEL)
    shutil.copyfile(CONFIG, OUT / "configs" / CONFIG.name)
    if PROMPT.exists():
        shutil.copyfile(PROMPT, OUT / "configs" / PROMPT.name)
    config = load_config()
    selection = make_selection()
    selection.to_csv(OUT / "configs" / "032_12_lc_target_year_selection.csv", index=False, encoding="utf-8-sig")
    direct_ppo.OUTPUT_ROOT = OUT
    patch_eval_globals()
    env_config = direct_ppo.build_env_config(config, selection)
    direct_ppo.write_yaml(env_config, OUT / "configs" / "032_12_resolved_env_config.yaml")
    eval_df = run_frozen_eval(config, env_config)
    eval_df.to_csv(OUT / "evaluation" / "032_12_lc2011_2020_frozen_75k_eval_summary.csv", index=False, encoding="utf-8-sig")
    comp_df = compare_to_baselines(eval_df)
    comp_df.to_csv(OUT / "evaluation" / "032_12_lc2011_2020_frozen_75k_vs_baselines.csv", index=False, encoding="utf-8-sig")
    write_record(eval_df, comp_df)
    result = {
        "task": "032_12_lc_multiyear_75k_future_year_transfer",
        "training_run": False,
        "source_model": str(SOURCE_MODEL.relative_to(ROOT)).replace("\\", "/"),
        "target_years": TARGET_YEARS,
        "record_md": str(DOC.relative_to(ROOT)).replace("\\", "/"),
        "eval_summary": str((OUT / "evaluation" / "032_12_lc2011_2020_frozen_75k_eval_summary.csv").relative_to(ROOT)).replace("\\", "/"),
        "comparison_summary": str((OUT / "evaluation" / "032_12_lc2011_2020_frozen_75k_vs_baselines.csv").relative_to(ROOT)).replace("\\", "/"),
    }
    (OUT / "032_12_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(comp_df[["year", "rl_yield_kg_ha", "yield_gap_vs_expert", "water_saving_vs_expert", "n_saving_vs_expert", "rl_PFP_N_kg_kg", "PFP_N_gap_vs_expert"]].to_string(index=False))


if __name__ == "__main__":
    main()
