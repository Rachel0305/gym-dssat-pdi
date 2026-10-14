"""098_00: read-only replay of FQ 2K/5K/10K checkpoints from 097."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_096_fqa_reward_ncost25_ckpt25k_five_scenario as base


CONFIG = ROOT / "configs/098_00_fqa_ncost25_early_checkpoint_replay_2k_5k_10k.json"
PROMPT = ROOT / "prompts/098_00_fqa_ncost25_early_checkpoint_replay_2k_5k_10k.md"
SOURCE_ROOT = ROOT / "benchmark_results/097_00_fqa_originIC_reward_ncost25_checkpoint_onset_diagnostic"
ORIGINAL_INPUT = ROOT / "DSSAT_auto_validation/multisite_new_cultivar_inputs_013"
CORRECTED_INPUT = ROOT / "benchmark_results/088_00_fq2018_input_quality_correction/corrected_input"
BASELINE_ROOT = ROOT / "benchmark_results/051_02_fqa_originIC_four_baselines_static_level1"
CORRECTED_BASELINE_ROOT = ROOT / "benchmark_results/088_00_fq2018_input_quality_correction_run_corrected2018"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_cfg() -> dict[str, Any]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def existing_row(cfg: dict[str, Any], out: Path, checkpoint: int, year: int, input_root: Path, variant: str, strict: bool) -> dict[str, Any]:
    """Use 096 replay code with 097 inventory/summary aliases, without training."""
    return base.replay_candidate(
        {
            "task_id": cfg["task_id"],
            "task_name": cfg["task_name"],
            "seed": cfg["seed"],
            "training": {"total_timesteps": 25000},
        },
        out,
        out,
        ROOT / "configs/097_00_effective_reward_ncost25_checkpoint_onset_ppo_config.yaml",
        checkpoint,
        year,
        input_root,
        variant,
        strict,
    )


def tables_for_checkpoint(cfg: dict[str, Any], out: Path, checkpoint: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    years = list(range(2014, 2024))
    row_map = {(int(row["year"]), row["variant"]): ROOT / row["snapshot_path"] for row in rows}
    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for year in years:
        baseline_root = CORRECTED_BASELINE_ROOT if year == 2018 else BASELINE_ROOT
        candidate_variant = f"ckpt{checkpoint}_corrected2018" if year == 2018 else f"ckpt{checkpoint}_original"
        scenario_paths = {
            "null": baseline_root / "snapshots/FQA" / str(year) / "null",
            "recorded_farmer_template": baseline_root / "snapshots/FQA" / str(year) / "recorded_farmer_template",
            "dssat_auto": baseline_root / "snapshots/FQA" / str(year) / "dssat_auto",
            "official_extension_expert": baseline_root / "snapshots/FQA" / str(year) / "official_extension_expert",
            "ppo": row_map[(year, candidate_variant)],
        }
        missing = [rel(path) for path in scenario_paths.values() if not (path / "Summary.OUT").exists()]
        if missing:
            raise FileNotFoundError(f"missing snapshots for checkpoint={checkpoint}, year={year}: {missing}")
        year_frames = [base.figures.daily_from_snapshot(path, year, scenario) for scenario, path in scenario_paths.items()]
        merged = base.add_reward_columns(pd.concat(year_frames, ignore_index=True))
        frames.append(merged)
        for scenario, subset in merged.groupby("scenario"):
            last = subset.iloc[-1]
            metrics = base.baseline.metrics_from_snapshot(scenario_paths[scenario], float(last["grain_yield_kg_ha"]))
            n_total = float(metrics["actual_nitrogen_kg_ha"])
            summaries.append(
                {
                    "checkpoint_step": checkpoint,
                    "station_code": "FQA",
                    "site": "FQ",
                    "year": year,
                    "scenario": scenario,
                    "grain_yield_kg_ha": float(last["grain_yield_kg_ha"]),
                    "biomass_kg_ha": float(last["biomass_kg_ha"]),
                    "actual_irrigation_mm": float(metrics["actual_irrigation_mm"]),
                    "actual_nitrogen_kg_ha": n_total,
                    "etcp_mm": float(metrics["etcp_mm"]),
                    "WP_ET_kg_m3": float(metrics["WP_ET_kg_m3"]),
                    "PFP_N_kg_kg": float(metrics["PFP_N_kg_kg"]) if pd.notna(metrics["PFP_N_kg_kg"]) and n_total > 0 else np.nan,
                    "max_wspd": float(pd.to_numeric(subset["water_stress_index_wspd"], errors="coerce").max()),
                    "max_nstd": float(pd.to_numeric(subset["nitrogen_stress_index_nstd"], errors="coerce").max()),
                    "snapshot_path": rel(scenario_paths[scenario]),
                }
            )
    daily = pd.concat(frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    events = daily[(daily["irrigation_executed_mm"].gt(0)) | (daily["nitrogen_executed_kg_ha"].gt(0))].copy()
    out_dir = out / "tables" / f"ckpt{checkpoint}"
    out_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_dir / f"098_00_ckpt{checkpoint}_five_scenario_daily.csv", index=False, encoding="utf-8-sig")
    events.to_csv(out_dir / f"098_00_ckpt{checkpoint}_management_events.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / f"098_00_ckpt{checkpoint}_season_summary.csv", index=False, encoding="utf-8-sig")
    gaps: list[dict[str, Any]] = []
    for year, subset in summary.groupby("year"):
        candidate = subset[subset["scenario"].eq("ppo")].iloc[0]
        four = subset[~subset["scenario"].eq("ppo")]
        for metric in ["grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg"]:
            maximum = pd.to_numeric(four[metric], errors="coerce").max(skipna=True)
            value = float(candidate[metric]) if pd.notna(candidate[metric]) else np.nan
            gaps.append({"checkpoint_step": checkpoint, "year": int(year), "metric": metric, "ppo_value": value, "four_baseline_max": maximum, "ppo_minus_four_max": value - maximum if pd.notna(maximum) else np.nan})
    gaps_df = pd.DataFrame(gaps)
    gaps_df.to_csv(out_dir / f"098_00_ckpt{checkpoint}_metric_gaps.csv", index=False, encoding="utf-8-sig")
    wins = gaps_df.groupby("metric")["ppo_minus_four_max"].apply(lambda values: int((pd.to_numeric(values, errors="coerce") > 0).sum())).to_dict()
    means = summary.groupby("scenario")[["grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg", "actual_irrigation_mm", "actual_nitrogen_kg_ha"]].mean(numeric_only=True).reset_index()
    means.to_csv(out_dir / f"098_00_ckpt{checkpoint}_site_means.csv", index=False, encoding="utf-8-sig")
    ppo = means[means["scenario"].eq("ppo")].iloc[0].to_dict()
    return {"checkpoint": checkpoint, "rows": int(len(summary)), "ppo_win_counts": wins, "ppo_site_mean": ppo, "season_summary": rel(out_dir / f"098_00_ckpt{checkpoint}_season_summary.csv"), "metric_gaps": rel(out_dir / f"098_00_ckpt{checkpoint}_metric_gaps.csv")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_cfg()
    out = ROOT / cfg["output_root"]
    checkpoints = list(map(int, cfg["checkpoints"]))
    if args.dry_run:
        print(json.dumps({"task_id": cfg["task_id"], "training_run": False, "checkpoints": checkpoints, "years": list(range(2014, 2024)), "external_n_allowed": False, "corrected_2018_exists": (CORRECTED_INPUT / "FQ/CNFQ1801.WTH").exists(), "source_exists": SOURCE_ROOT.exists(), "prompt_exists": PROMPT.exists()}, ensure_ascii=False, indent=2))
        return
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    (out / "configs").mkdir(parents=True, exist_ok=False)
    (out / "evaluation").mkdir(parents=True, exist_ok=True)
    (out / "audits").mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONFIG, out / "configs" / CONFIG.name)
    shutil.copy2(PROMPT, out / "configs" / PROMPT.name)
    shutil.copy2(SOURCE_ROOT / "evaluation/097_00_checkpoint_validation_summary.csv", out / "evaluation/096_00_checkpoint_validation_summary.csv")
    shutil.copy2(SOURCE_ROOT / "evaluation/097_00_training_checkpoint_inventory.csv", out / "evaluation/096_00_training_checkpoint_inventory.csv")

    replay_rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        for year in range(2014, 2024):
            replay_rows.append(existing_row(cfg, out, checkpoint, year, ORIGINAL_INPUT, f"ckpt{checkpoint}_original", True))
        replay_rows.append(existing_row(cfg, out, checkpoint, 2018, CORRECTED_INPUT, f"ckpt{checkpoint}_corrected2018", False))
    pd.DataFrame(replay_rows).to_csv(out / "098_00_replay_provenance.csv", index=False, encoding="utf-8-sig")

    tables = [tables_for_checkpoint(cfg, out, checkpoint, [row for row in replay_rows if row["variant"].startswith(f"ckpt{checkpoint}_")]) for checkpoint in checkpoints]
    result = {"task_id": cfg["task_id"], "status": "completed_read_only_early_checkpoint_replay", "training_run": False, "external_n_included": False, "weather_forecast_included": False, "checkpoints": checkpoints, "replay_rows": len(replay_rows), "five_scenario_tables": tables, "output_root": rel(out), "next_step_allowed": False, "next_step_note": "Keep 097 strict gate labels unchanged; no automatic freeze or long training."}
    (out / "098_00_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "098_00_manifest.json").write_text(json.dumps({"config": cfg, "result": result}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = ["# 098_00 FQ early-checkpoint replay (2K/5K/10K)", "", "- 只读复用 097 模型；未训练、未使用 external-N。", "- 原始 10 年 replay 用于 endpoint oracle；corrected 2018 单独作为输入 provenance。", "", "## Five-scenario summaries", ""]
    lines.extend(f"- `{table}`" for table in tables)
    lines.extend(["", "- 097 的严格 gate 状态不在本任务中修改。", ""])
    (ROOT / "docs/098_00_fqa_ncost25_early_checkpoint_replay_2k_5k_10k_record.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
