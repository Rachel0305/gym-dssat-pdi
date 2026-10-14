"""206HLA: replay 204HLA PPO on all 10x6 augmented training climates.

This is an operational response audit, not a fixed-base-state causal swap.  For
each source year, it compares the complete PPO action sequence under each RAIN
perturbation with that source year's original-weather action sequence.  Reward,
actions, masks, renderer and model weights are unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for module_path in [ROOT, ROOT / "src"]:
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base03222
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine


TASK_ID = "206HLA"
TASK_NAME = "hla_augmented_weather_policy_response"
RUN_ROOT = ROOT / "benchmark_results" / "204HLA_hla_lowIC_10y_six_weather_balanced_aug_maskableppo"
CONFIG = ROOT / "configs" / "204HLA_hla_lowIC_10y_six_weather_balanced_aug_maskableppo.json"
SPLIT = RUN_ROOT / "configs" / "204HLA_training_validation_split.csv"
INVENTORY = RUN_ROOT / "evaluation" / "204HLA_training_checkpoint_inventory.csv"
INPUT_ROOT = ROOT / "DSSAT_auto_validation" / "hla_lowIC_weather_scenario_bank_202_v1"
CHECKPOINTS = [10_000, 25_000, 50_000, 100_000]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def output_root(smoke: bool, smoke_source_year: int) -> Path:
    suffix = f"_smoke{smoke_source_year}" if smoke else ""
    return ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}{suffix}"


def record_path(smoke: bool, smoke_source_year: int) -> Path:
    suffix = f"_smoke{smoke_source_year}" if smoke else ""
    return ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}{suffix}_record.md"


def md_table(frame: pd.DataFrame) -> str:
    work = frame.copy()
    for column in work.select_dtypes(include=["number"]).columns:
        work[column] = pd.to_numeric(work[column], errors="coerce").round(4)
    return work.to_markdown(index=False)


def run_audit(smoke: bool, smoke_source_year: int = 2004) -> dict[str, Any]:
    out = output_root(smoke, smoke_source_year)
    if out.exists():
        raise FileExistsError(f"Refuse to overwrite existing audit: {rel(out)}")
    required = [CONFIG, SPLIT, INVENTORY, RUN_ROOT / "204HLA_formal_result.json", INPUT_ROOT]
    missing = [rel(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing 206HLA inputs: " + "; ".join(missing))

    split = pd.read_csv(SPLIT, keep_default_na=False)
    train = split[split["split"].astype(str).eq("train")].copy()
    train["year"] = pd.to_numeric(train["year"], errors="raise").astype(int)
    train["source_year"] = pd.to_numeric(train["source_year"], errors="raise").astype(int)
    expected_pairs = set(
        (year, variant)
        for year in range(2004, 2014)
        for variant in ["original", "early_dry", "mid_dry", "late_dry", "early_wet", "mid_wet"]
    )
    actual_pairs = set(zip(train["source_year"], train["weather_variant"].astype(str)))
    if actual_pairs != expected_pairs or len(train) != 60 or train["year"].nunique() != 60:
        raise ValueError("206HLA requires the audited complete 10x6 training scenario bank")
    inventory = pd.read_csv(INVENTORY, keep_default_na=False)
    inventory["checkpoint_step"] = pd.to_numeric(inventory["checkpoint_step"], errors="raise").astype(int)
    checkpoints = [10_000] if smoke else CHECKPOINTS
    scenarios = train[train["source_year"].eq(smoke_source_year)] if smoke else train
    if smoke and len(scenarios) != 6:
        raise ValueError(f"Smoke source year {smoke_source_year} does not have six scenarios")

    for sub in ["configs", "evaluation", "daily_outputs", "logs"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONFIG, out / "configs" / CONFIG.name)
    scenarios.to_csv(out / "configs" / f"{TASK_ID}_evaluated_scenarios.csv", index=False, encoding="utf-8-sig")

    old = {
        "engine_lowic": engine.LOWIC_INPUT_ROOT,
        "engine_i": list(engine.BINARY_IRRIGATION_LEVELS),
        "engine_n": list(engine.BINARY_NITROGEN_LEVELS),
        "base_out": base03222.OUT,
        "base_seed": base03222.SEED,
        "base_sites": list(base03222.SITES),
        "base_site_names": dict(base03222.SITE_NAMES),
        "renderer_root": ppo_safe_rendering.MULTISITE_INPUT_ROOT,
    }
    rows: list[dict[str, Any]] = []
    try:
        engine.LOWIC_INPUT_ROOT = INPUT_ROOT
        engine.BINARY_IRRIGATION_LEVELS = [0.0, 15.0, 30.0, 45.0]
        engine.BINARY_NITROGEN_LEVELS = [0.0, 40.0, 80.0, 120.0]
        base03222.OUT = out
        base03222.SEED = 0
        base03222.SITES = ["HLA"]
        base03222.SITE_NAMES.clear()
        base03222.SITE_NAMES.update({"HLA": "HLA"})
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = INPUT_ROOT
        config = engine.load_config()
        env_config = direct_ppo.build_env_config(config, split)
        for checkpoint in checkpoints:
            model_rows = inventory[
                inventory["checkpoint_step"].eq(checkpoint)
                & inventory["run_status"].astype(str).str.startswith("ok")
            ]
            if len(model_rows) != 1:
                raise RuntimeError(f"Expected one inventory row for checkpoint {checkpoint}")
            model_row = model_rows.iloc[0]
            for scenario in scenarios.itertuples(index=False):
                try:
                    result = base03222.evaluate_checkpoint(config, env_config, model_row, int(scenario.year))
                    result.update(
                        {
                            "source_year": int(scenario.source_year),
                            "weather_variant": str(scenario.weather_variant),
                            "rain_multiplier": float(scenario.rain_multiplier),
                            "perturbation_window": str(scenario.perturbation_window),
                        }
                    )
                    rows.append(result)
                except Exception:
                    rows.append(
                        {
                            "checkpoint_step": checkpoint,
                            "year": int(scenario.year),
                            "source_year": int(scenario.source_year),
                            "weather_variant": str(scenario.weather_variant),
                            "run_status": "failed",
                            "notes": traceback.format_exc()[-4000:],
                        }
                    )
    finally:
        engine.LOWIC_INPUT_ROOT = old["engine_lowic"]
        engine.BINARY_IRRIGATION_LEVELS = old["engine_i"]
        engine.BINARY_NITROGEN_LEVELS = old["engine_n"]
        base03222.OUT = old["base_out"]
        base03222.SEED = old["base_seed"]
        base03222.SITES = old["base_sites"]
        base03222.SITE_NAMES.clear()
        base03222.SITE_NAMES.update(old["base_site_names"])
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old["renderer_root"]

    details = pd.DataFrame(rows)
    details.to_csv(out / "evaluation" / f"{TASK_ID}_scenario_episode_metrics.csv", index=False, encoding="utf-8-sig")
    failed = details[~details["run_status"].astype(str).str.startswith("ok")]
    if not failed.empty:
        raise RuntimeError(f"206HLA had {len(failed)} failed scenario replays")
    for column in ["checkpoint_step", "source_year", "final_grnwt", "total_irrigation", "total_n", "PFP_N"]:
        details[column] = pd.to_numeric(details[column], errors="coerce")

    pair_rows: list[dict[str, Any]] = []
    for (checkpoint, source_year), group in details.groupby(["checkpoint_step", "source_year"]):
        original = group[group["weather_variant"].astype(str).eq("original")]
        if len(original) != 1:
            raise RuntimeError(f"Missing original reference for {checkpoint}/{source_year}")
        ref = original.iloc[0]
        for row in group.itertuples(index=False):
            pair_rows.append(
                {
                    "checkpoint_step": int(checkpoint),
                    "source_year": int(source_year),
                    "weather_variant": str(row.weather_variant),
                    "action_changed_vs_original": str(row.action_sequence) != str(ref["action_sequence"]),
                    "irrigation_changed_vs_original": not np.isclose(float(row.total_irrigation), float(ref["total_irrigation"])),
                    "nitrogen_changed_vs_original": not np.isclose(float(row.total_n), float(ref["total_n"])),
                    "yield_delta_vs_original_kg_ha": float(row.final_grnwt) - float(ref["final_grnwt"]),
                    "action_sequence": str(row.action_sequence),
                    "total_irrigation": float(row.total_irrigation),
                    "total_n": float(row.total_n),
                    "final_grnwt": float(row.final_grnwt),
                    "PFP_N": float(row.PFP_N),
                }
            )
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(out / "evaluation" / f"{TASK_ID}_paired_weather_response.csv", index=False, encoding="utf-8-sig")
    non_original = pairs[~pairs["weather_variant"].eq("original")].copy()
    summary_rows: list[dict[str, Any]] = []
    for checkpoint, group in pairs.groupby("checkpoint_step"):
        changed = non_original[non_original["checkpoint_step"].eq(checkpoint)]
        source_response = changed.groupby("source_year")["action_changed_vs_original"].any()
        summary_rows.append(
            {
                "checkpoint_step": int(checkpoint),
                "scenario_rows": int(len(group)),
                "unique_full_action_sequences_across_60": int(group["action_sequence"].nunique()),
                "action_change_pairs_vs_original": int(changed["action_changed_vs_original"].sum()),
                "action_change_rate_vs_original": float(changed["action_changed_vs_original"].mean()),
                "source_years_with_any_action_response": int(source_response.sum()),
                "irrigation_change_rate_vs_original": float(changed["irrigation_changed_vs_original"].mean()),
                "nitrogen_change_rate_vs_original": float(changed["nitrogen_changed_vs_original"].mean()),
                "mean_abs_yield_delta_vs_original_kg_ha": float(changed["yield_delta_vs_original_kg_ha"].abs().mean()),
                "physical_yield_response_pairs_gt_1kg_ha": int(changed["yield_delta_vs_original_kg_ha"].abs().gt(1.0).sum()),
                "mean_yield_kg_ha": float(group["final_grnwt"].mean()),
                "mean_irrigation_mm": float(group["total_irrigation"].mean()),
                "mean_nitrogen_kg_ha": float(group["total_n"].mean()),
                "mean_PFP_N_kg_kg": float(group["PFP_N"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("checkpoint_step")
    summary.to_csv(out / "evaluation" / f"{TASK_ID}_checkpoint_weather_response_summary.csv", index=False, encoding="utf-8-sig")

    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "phase": "smoke" if smoke else "full_10x6",
        "status": "completed",
        "audit_type": "operational scenario replay; not fixed-base-state causal swap",
        "checkpoints": checkpoints,
        "scenario_rows_per_checkpoint": int(len(scenarios)),
        "summary": rel(out / "evaluation" / f"{TASK_ID}_checkpoint_weather_response_summary.csv"),
        "paired_response": rel(out / "evaluation" / f"{TASK_ID}_paired_weather_response.csv"),
        "record": rel(record_path(smoke, smoke_source_year)),
    }
    (out / f"{TASK_ID}_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    record = [
        "# 206HLA HLA augmented-weather policy response audit",
        "",
        f"- 阶段：`{result['phase']}`；状态：`completed`。",
        "- 读取 204HLA 已冻结 checkpoint；不训练、不改 reward、不改 action mask。",
        "- 对同一源年的 5 个降雨扰动情景与 original 情景比较完整 DAP 动作序列。",
        "- 这是闭环轨迹层面的天气响应审计；天气改变后 DSSAT 状态轨迹也会改变，因此不能替代固定基础状态的 causal swap。",
        "",
        "## Checkpoint summary",
        "",
        md_table(summary),
        "",
        "## Reproduction",
        "",
        f"- Script: `{rel(Path(__file__))}`.",
        f"- Input split: `{rel(SPLIT)}`.",
        f"- Model inventory: `{rel(INVENTORY)}`.",
        f"- Detail: `{result['paired_response']}`.",
    ]
    record_path(smoke, smoke_source_year).write_text("\n".join(record) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-source-year", type=int, default=2004)
    args = parser.parse_args()
    run_audit(args.smoke, args.smoke_source_year)


if __name__ == "__main__":
    main()
