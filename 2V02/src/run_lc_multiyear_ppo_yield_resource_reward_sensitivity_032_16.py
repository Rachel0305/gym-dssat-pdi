from __future__ import annotations

import json
import shutil
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_lc_multiyear_free_timing_ppo_smoke_032_10 as m


ROOT = Path(__file__).resolve().parents[1]
PROMPT = ROOT / "prompts" / "032_16_lc_multiyear_ppo_yield_resource_reward_sensitivity.md"
BASE_CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
BASELINE_SUMMARY = ROOT / "benchmark_results" / "031_36_missing_dssat_auto_completion_for_03134" / "evaluation" / "031_36_full_completed_template_aware_unified_baseline_summary.csv"
POOL = ROOT / "Leave_One_experiments" / "all_year_weather_calibration_validation" / "scenario_pool" / "all_year_weather_scenario_pool.csv"
OUT = ROOT / "benchmark_results" / "032_16_lc_multiyear_ppo_yield_resource_reward_sensitivity"
DOC = ROOT / "docs" / "032_16_lc_multiyear_ppo_yield_resource_reward_sensitivity_record.md"

STATION = "LCA"
SITE = "LC"
SEED = 0
TRAIN_YEARS = [2005, 2006, 2007, 2008, 2009, 2010]
TRANSFER_YEARS = list(range(2011, 2021))
ALL_YEARS = TRAIN_YEARS + TRANSFER_YEARS
TIMESTEPS = 50000
CHECKPOINTS = [25000, 50000]

VARIANTS = [
    {
        "variant": "current_50k",
        "reuse_model": ROOT / "benchmark_results" / "032_11_lc_multiyear_free_timing_ppo_training_length" / "models" / STATION / f"{STATION}_multiyear_2005_2010_stress_aware_maskableppo_seed0_ckpt50000.zip",
        "yield_coef": 0.158,
        "water_cost": 1.1,
        "nitrogen_cost": 1.58,
    },
    {
        "variant": "yield_plus_50k",
        "reuse_model": None,
        "yield_coef": 0.180,
        "water_cost": 1.1,
        "nitrogen_cost": 1.58,
    },
    {
        "variant": "resource_cheaper_50k",
        "reuse_model": None,
        "yield_coef": 0.158,
        "water_cost": 0.8,
        "nitrogen_cost": 1.2,
    },
]


def ensure_dirs() -> None:
    for rel in ["configs", "evaluation", "tables", "logs"]:
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


def make_selection() -> pd.DataFrame:
    pool = pd.read_csv(POOL)
    rows = pool[(pool["station_code"].eq(STATION)) & (pool["year"].astype(int).isin(ALL_YEARS))].copy()
    found = sorted(rows["year"].astype(int).unique().tolist())
    if found != ALL_YEARS:
        raise RuntimeError(f"Expected LC years {ALL_YEARS}, found {found}")
    rows["selected_for_train"] = rows["year"].astype(int).isin(TRAIN_YEARS)
    rows["selected_for_eval"] = True
    rows["selection_reason"] = "032_16_lc_multiyear_ppo_yield_resource_reward_sensitivity"
    return rows.sort_values(["station_code", "year"]).reset_index(drop=True)


def variant_out(name: str) -> Path:
    return OUT / "variants" / name


def patch_training_globals(vout: Path, prompt: Path) -> None:
    m.OUT = vout
    m.DOC = vout / "variant_record.md"
    m.PROMPT = prompt
    m.STATION = STATION
    m.TRAIN_YEARS = TRAIN_YEARS
    m.SEED = SEED
    m.TOTAL_TIMESTEPS = TIMESTEPS
    m.CHECKPOINT_STEPS = CHECKPOINTS


def load_variant_config(v: dict[str, Any], vout: Path) -> dict[str, Any]:
    config = direct_ppo.load_yaml(BASE_CONFIG)
    config = json.loads(json.dumps(config))
    config["seed"] = SEED
    config["total_timesteps"] = TIMESTEPS
    config["paths"]["output_root"] = str(vout.relative_to(ROOT)).replace("\\", "/")
    config["runtime"]["smoke_station"] = STATION
    config["runtime"]["smoke_year"] = TRAIN_YEARS[0]
    config["reward"]["yield_coef"] = float(v["yield_coef"])
    config["reward"]["water_cost"] = float(v["water_cost"])
    config["reward"]["nitrogen_cost"] = float(v["nitrogen_cost"])
    return config


def model_row_for_variant(v: dict[str, Any], vout: Path) -> pd.DataFrame:
    if v["reuse_model"] is not None:
        model = Path(v["reuse_model"])
        if not model.exists():
            raise FileNotFoundError(model)
        return pd.DataFrame(
            [
                {
                    "station_code": STATION,
                    "train_years": ",".join(map(str, TRAIN_YEARS)),
                    "seed": SEED,
                    "checkpoint_step": 50000,
                    "run_status": "ok_reused_from_032_11",
                    "model_path": str(model.relative_to(ROOT)).replace("\\", "/"),
                    "model_sha256": m.sha256_file(model),
                }
            ]
        )
    train_df, reset_df = m.train(load_variant_config(v, vout), direct_ppo.build_env_config(load_variant_config(v, vout), make_selection()))
    reset_df.to_csv(vout / "logs" / "032_16_training_year_reset_counts.csv", index=False, encoding="utf-8-sig")
    return train_df


def pfp(y: float, n: float) -> float:
    if not np.isfinite(y) or not np.isfinite(n) or n <= 0:
        return np.nan
    return float(y / n)


def compare_to_baselines(eval_df: pd.DataFrame) -> pd.DataFrame:
    base = pd.read_csv(BASELINE_SUMMARY)
    base = base[(base["station_code"].eq(STATION)) & (base["scenario"].isin(["null", "recorded_farmer", "dssat_auto", "official_extension_expert"]))].copy()
    rows = []
    for row in eval_df.itertuples(index=False):
        year = int(row.year)
        b = base[base["year"].astype(int).eq(year)].copy()
        expert = b[b["scenario"].eq("official_extension_expert")]
        rl_y = float(getattr(row, "final_grnwt", np.nan))
        rl_i = float(getattr(row, "total_irrigation", np.nan))
        rl_n = float(getattr(row, "total_n", np.nan))
        rl_pfp = pfp(rl_y, rl_n)
        item = {
            "variant": getattr(row, "variant"),
            "split": "train_2005_2010" if year in TRAIN_YEARS else "transfer_2011_2020",
            "year": year,
            "run_status": getattr(row, "run_status"),
            "rl_yield_kg_ha": rl_y,
            "rl_irrigation_mm": rl_i,
            "rl_nitrogen_kg_ha": rl_n,
            "rl_PFP_N_kg_kg": rl_pfp,
            "rl_max_swfac": float(getattr(row, "max_swfac", np.nan)),
            "rl_max_nstres": float(getattr(row, "max_nstres", np.nan)),
            "action_sequence": getattr(row, "action_sequence", ""),
            "four_max_yield_kg_ha": float(pd.to_numeric(b["grain_yield_kg_ha"], errors="coerce").max()) if len(b) else np.nan,
            "four_max_PFP_N_kg_kg": float(pd.to_numeric(b["PFP_N_kg_kg"], errors="coerce").max()) if len(b) else np.nan,
        }
        if len(expert):
            ex = expert.iloc[0]
            item.update(
                {
                    "expert_yield_kg_ha": float(ex["grain_yield_kg_ha"]),
                    "expert_irrigation_mm": float(ex["actual_irrigation_mm"]),
                    "expert_nitrogen_kg_ha": float(ex["actual_nitrogen_kg_ha"]),
                    "expert_PFP_N_kg_kg": float(ex["PFP_N_kg_kg"]),
                    "yield_gap_vs_expert": rl_y - float(ex["grain_yield_kg_ha"]),
                    "water_saving_vs_expert": float(ex["actual_irrigation_mm"]) - rl_i,
                    "n_saving_vs_expert": float(ex["actual_nitrogen_kg_ha"]) - rl_n,
                    "PFP_N_gap_vs_expert": rl_pfp - float(ex["PFP_N_kg_kg"]) if np.isfinite(rl_pfp) else np.nan,
                }
            )
        item["yield_gap_vs_four_max"] = rl_y - item["four_max_yield_kg_ha"]
        item["PFP_N_gap_vs_four_max"] = rl_pfp - item["four_max_PFP_N_kg_kg"] if np.isfinite(rl_pfp) else np.nan
        item["yield_win_four"] = item["yield_gap_vs_four_max"] > 1e-9
        item["PFP_N_win_four"] = item["PFP_N_gap_vs_four_max"] > 1e-9 if np.isfinite(item["PFP_N_gap_vs_four_max"]) else False
        item["any_metric_win_four"] = bool(item["yield_win_four"] or item["PFP_N_win_four"])
        rows.append(item)
    return pd.DataFrame(rows)


def summarize(comp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, split), g in comp.groupby(["variant", "split"]):
        rows.append(
            {
                "variant": variant,
                "split": split,
                "n_years": int(len(g)),
                "mean_yield": float(g["rl_yield_kg_ha"].mean()),
                "mean_yield_gap_vs_four_max": float(g["yield_gap_vs_four_max"].mean()),
                "yield_win_four": int(g["yield_win_four"].sum()),
                "mean_irrigation": float(g["rl_irrigation_mm"].mean()),
                "mean_nitrogen": float(g["rl_nitrogen_kg_ha"].mean()),
                "mean_PFP_N": float(g["rl_PFP_N_kg_kg"].mean()),
                "mean_PFP_N_gap_vs_four_max": float(g["PFP_N_gap_vs_four_max"].mean()),
                "PFP_N_win_four": int(g["PFP_N_win_four"].sum()),
                "mean_water_saving_vs_expert": float(g["water_saving_vs_expert"].mean()),
                "mean_n_saving_vs_expert": float(g["n_saving_vs_expert"].mean()),
                "mean_max_nstres": float(g["rl_max_nstres"].mean()),
            }
        )
    for variant, g in comp.groupby("variant"):
        rows.append(
            {
                "variant": variant,
                "split": "all_2005_2020",
                "n_years": int(len(g)),
                "mean_yield": float(g["rl_yield_kg_ha"].mean()),
                "mean_yield_gap_vs_four_max": float(g["yield_gap_vs_four_max"].mean()),
                "yield_win_four": int(g["yield_win_four"].sum()),
                "mean_irrigation": float(g["rl_irrigation_mm"].mean()),
                "mean_nitrogen": float(g["rl_nitrogen_kg_ha"].mean()),
                "mean_PFP_N": float(g["rl_PFP_N_kg_kg"].mean()),
                "mean_PFP_N_gap_vs_four_max": float(g["PFP_N_gap_vs_four_max"].mean()),
                "PFP_N_win_four": int(g["PFP_N_win_four"].sum()),
                "mean_water_saving_vs_expert": float(g["water_saving_vs_expert"].mean()),
                "mean_n_saving_vs_expert": float(g["n_saving_vs_expert"].mean()),
                "mean_max_nstres": float(g["rl_max_nstres"].mean()),
            }
        )
    return pd.DataFrame(rows)


def run_variant(v: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    name = str(v["variant"])
    vout = variant_out(name)
    for rel in ["configs", "models/LCA", "daily_outputs/LCA", "evaluation", "logs", "tensorboard/LCA"]:
        (vout / rel).mkdir(parents=True, exist_ok=True)
    patch_training_globals(vout, PROMPT)
    config = load_variant_config(v, vout)
    selection = make_selection()
    selection.to_csv(vout / "configs" / "032_16_lc_year_selection.csv", index=False, encoding="utf-8-sig")
    direct_ppo.write_yaml(config, vout / "configs" / "032_16_variant_config.yaml")
    env_config = direct_ppo.build_env_config(config, selection)
    direct_ppo.write_yaml(env_config, vout / "configs" / "032_16_resolved_env_config.yaml")
    direct_ppo.OUTPUT_ROOT = vout

    try:
        train_df = model_row_for_variant(v, vout)
        train_df["variant"] = name
        train_df.to_csv(vout / "evaluation" / "032_16_training_checkpoint_inventory.csv", index=False, encoding="utf-8-sig")
        target_rows = train_df[pd.to_numeric(train_df["checkpoint_step"], errors="coerce").eq(50000)]
        eval_rows = []
        for _, model_row in target_rows.iterrows():
            for year in ALL_YEARS:
                r = m.evaluate_checkpoint(config, env_config, model_row, year)
                r["variant"] = name
                eval_rows.append(r)
        eval_df = pd.DataFrame(eval_rows)
        eval_df.to_csv(vout / "evaluation" / "032_16_all_year_eval_summary.csv", index=False, encoding="utf-8-sig")
        comp_df = compare_to_baselines(eval_df)
        comp_df.to_csv(vout / "evaluation" / "032_16_vs_four_baselines.csv", index=False, encoding="utf-8-sig")
        return train_df, eval_df, comp_df
    except Exception:
        err = traceback.format_exc()
        (vout / "logs" / "032_16_variant_error.txt").write_text(err, encoding="utf-8")
        return pd.DataFrame([{"variant": name, "run_status": "failed", "notes": err[-4000:]}]), pd.DataFrame(), pd.DataFrame()


def write_record(train_all: pd.DataFrame, eval_all: pd.DataFrame, comp_all: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# 032_16 LC multiyear PPO yield-resource reward sensitivity record",
        "",
        "## Status",
        "",
        "- Screening completed.",
        f"- Variants: {', '.join(v['variant'] for v in VARIANTS)}.",
        "- This task is not final model selection.",
        "",
        "## Scope",
        "",
        "- Station: LC / LCA.",
        f"- Train years: {', '.join(map(str, TRAIN_YEARS))}.",
        f"- Transfer years: {', '.join(map(str, TRANSFER_YEARS))}.",
        "- Algorithm/action/mask framework inherited from 032_11/032_12.",
        f"- New training timesteps per non-reused variant: {TIMESTEPS}.",
        "- Evaluation checkpoint: 50k endpoint only.",
        "",
        "## Variant definitions",
        "",
        md_table(pd.DataFrame([{k: str(v[k]) for k in ["variant", "yield_coef", "water_cost", "nitrogen_cost", "reuse_model"]} for v in VARIANTS])),
        "",
        "## Variant summary",
        "",
        md_table(summary, max_rows=80),
        "",
        "## Per-year comparison",
        "",
        md_table(
            comp_all[
                [
                    "variant",
                    "split",
                    "year",
                    "rl_yield_kg_ha",
                    "yield_gap_vs_four_max",
                    "rl_irrigation_mm",
                    "rl_nitrogen_kg_ha",
                    "rl_PFP_N_kg_kg",
                    "PFP_N_gap_vs_four_max",
                    "water_saving_vs_expert",
                    "n_saving_vs_expert",
                    "action_sequence",
                ]
            ],
            max_rows=200,
        )
        if not comp_all.empty
        else "No comparison rows.",
        "",
        "## Interpretation boundary",
        "",
        "- A yield-biased variant should only be extended if it improves mean yield gap while retaining water and N savings versus official expert.",
        "- Do not add more variants based on these results without a new pre-registration.",
    ]
    text = "\n".join(lines) + "\n"
    DOC.write_text(text, encoding="utf-8")
    (OUT / DOC.name).write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    shutil.copyfile(PROMPT, OUT / "configs" / PROMPT.name)
    shutil.copyfile(BASE_CONFIG, OUT / "configs" / BASE_CONFIG.name)
    train_parts = []
    eval_parts = []
    comp_parts = []
    for variant in VARIANTS:
        print(f"RUN_VARIANT {variant['variant']}", flush=True)
        train_df, eval_df, comp_df = run_variant(variant)
        train_parts.append(train_df)
        if not eval_df.empty:
            eval_parts.append(eval_df)
        if not comp_df.empty:
            comp_parts.append(comp_df)
    train_all = pd.concat(train_parts, ignore_index=True, sort=False) if train_parts else pd.DataFrame()
    eval_all = pd.concat(eval_parts, ignore_index=True, sort=False) if eval_parts else pd.DataFrame()
    comp_all = pd.concat(comp_parts, ignore_index=True, sort=False) if comp_parts else pd.DataFrame()
    summary = summarize(comp_all) if not comp_all.empty else pd.DataFrame()
    train_all.to_csv(OUT / "tables" / "032_16_all_variant_training_inventory.csv", index=False, encoding="utf-8-sig")
    eval_all.to_csv(OUT / "tables" / "032_16_all_variant_eval_summary.csv", index=False, encoding="utf-8-sig")
    comp_all.to_csv(OUT / "tables" / "032_16_all_variant_vs_four_baselines.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "tables" / "032_16_variant_summary.csv", index=False, encoding="utf-8-sig")
    write_record(train_all, eval_all, comp_all, summary)
    result = {
        "task": "032_16_lc_multiyear_ppo_yield_resource_reward_sensitivity",
        "training_run": True,
        "variants": [v["variant"] for v in VARIANTS],
        "record_md": str(DOC.relative_to(ROOT)).replace("\\", "/"),
        "summary_csv": str((OUT / "tables" / "032_16_variant_summary.csv").relative_to(ROOT)).replace("\\", "/"),
    }
    (OUT / "032_16_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
