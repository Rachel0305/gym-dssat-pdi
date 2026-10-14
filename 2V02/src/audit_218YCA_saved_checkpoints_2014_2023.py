"""Replay saved 218YCA checkpoints on the untouched 2014-2023 YC weather.

This is a read-only audit of the partial 218YCA run.  It does not retrain or
rewrite the failed formal-run summary; replay outputs are written below a
separate ``replay_2014_2023`` directory inside the 218YCA result directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_lc_multiyear_free_timing_ppo_smoke_032_10 as lc_multi
import run_218YCA_yca_lowIC_10y_six_weather_simple_profit_aug_maskableppo_500k as runner
from run_fqa2014_project_profit_reward_ppo_checkpoint_guardrail_035_04 import ProjectProfitDiscreteWrapper


TASK_ROOT = ROOT / "benchmark_results" / "218YCA_yca_lowIC_10y_six_weather_simple_profit_aug_maskableppo_500k"
REPLAY_ROOT = TASK_ROOT / "replay_2014_2023"
CONFIG_PATH = TASK_ROOT / "configs" / "218YCA_yca_lowIC_10y_six_weather_simple_profit_aug_maskableppo_500k.json"
CHECKPOINT_STEPS = [2000, 5000, 10000, 25000, 50000, 100000, 150000, 200000]
VALIDATION_YEARS = list(range(2014, 2024))


def main() -> None:
    run_contract = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest = pd.read_csv(
        TASK_ROOT / "configs" / "218YCA_selected_manifest.csv",
        keep_default_na=False,
    )
    split = runner.base.make_split(manifest)
    # The formal JSON is the experiment contract, while the inherited engine
    # YAML supplies the DSSAT paths and the actual 16-action environment grid.
    # This is the same separation used by the 218YCA training runner.
    config = runner.base03222.load_config()
    config["reward"] = dict(runner.prior.simple_profit_engine_config()["reward"])
    config["seed"] = int(run_contract["seed"])
    env_config = direct_ppo.build_env_config(config, split)

    REPLAY_ROOT.mkdir(parents=True, exist_ok=True)
    (REPLAY_ROOT / "daily_outputs" / "YCA").mkdir(parents=True, exist_ok=True)
    (REPLAY_ROOT / "evaluation" / "YCA").mkdir(parents=True, exist_ok=True)

    # Recreate the training/evaluation wrapper contract without changing the
    # original formal-run files.
    runner.base03222.base.StressAwareDiscreteWrapper = ProjectProfitDiscreteWrapper
    old_out = lc_multi.OUT
    old_station = lc_multi.STATION
    old_seed = lc_multi.SEED
    old_renderer_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    lc_multi.OUT = REPLAY_ROOT
    lc_multi.STATION = "YCA"
    lc_multi.SEED = int(run_contract["seed"])
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = ROOT / run_contract["input_root"]

    rows: list[dict] = []
    try:
        for checkpoint_step in CHECKPOINT_STEPS:
            model_path = TASK_ROOT / "models" / "YCA" / (
                f"YCA_half_split_stress_aware_maskableppo_seed0_ckpt{checkpoint_step}.zip"
            )
            if not model_path.exists():
                continue
            model_row = pd.Series(
                {
                    "station_code": "YCA",
                    "checkpoint_step": checkpoint_step,
                    "model_path": model_path.relative_to(ROOT).as_posix(),
                }
            )
            for year in VALIDATION_YEARS:
                print(f"[218YCA replay] checkpoint={checkpoint_step} year={year}", flush=True)
                row = lc_multi.evaluate_checkpoint(config, env_config, model_row, year)
                rows.append(row)
    finally:
        lc_multi.OUT = old_out
        lc_multi.STATION = old_station
        lc_multi.SEED = old_seed
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_renderer_root

    detail = pd.DataFrame(rows)
    detail_path = REPLAY_ROOT / "evaluation" / "218YCA_replay_validation_summary.csv"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")

    numeric_cols = [
        "final_grnwt",
        "total_irrigation",
        "total_n",
        "profit_simple",
        "PFP_N",
        "irrigation_event_count",
        "n_event_count",
        "swfac_stress_days_gt_0p05",
        "nstres_days_gt_0p05",
    ]
    for col in numeric_cols:
        if col in detail.columns:
            detail[col] = pd.to_numeric(detail[col], errors="coerce")
    summary = (
        detail.groupby("checkpoint_step", as_index=False)
        .agg(
            validation_years=("year", "nunique"),
            mean_yield_kg_ha=("final_grnwt", "mean"),
            std_yield_kg_ha=("final_grnwt", "std"),
            mean_irrigation_mm=("total_irrigation", "mean"),
            mean_n_kg_ha=("total_n", "mean"),
            mean_simple_profit=("profit_simple", "mean"),
            mean_pfp_n=("PFP_N", "mean"),
            mean_irrigation_events=("irrigation_event_count", "mean"),
            mean_n_events=("n_event_count", "mean"),
            mean_water_stress_days_gt_0p05=("swfac_stress_days_gt_0p05", "mean"),
            mean_n_stress_days_gt_0p05=("nstres_days_gt_0p05", "mean"),
            unique_action_signatures=("action_sequence", "nunique"),
        )
        .sort_values("checkpoint_step")
    )
    summary_path = REPLAY_ROOT / "evaluation" / "218YCA_replay_checkpoint_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    result = {
        "task": "218YCA_saved_checkpoint_replay_2014_2023",
        "source_formal_run": str(TASK_ROOT.relative_to(ROOT)).replace("\\", "/"),
        "replay_root": str(REPLAY_ROOT.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_steps_evaluated": [int(x) for x in summary["checkpoint_step"].tolist()],
        "validation_years": VALIDATION_YEARS,
        "detail_csv": str(detail_path.relative_to(ROOT)).replace("\\", "/"),
        "summary_csv": str(summary_path.relative_to(ROOT)).replace("\\", "/"),
        "wp_et_status": "pending exact Summary.OUT/ETCP replay",
    }
    (REPLAY_ROOT / "218YCA_replay_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
