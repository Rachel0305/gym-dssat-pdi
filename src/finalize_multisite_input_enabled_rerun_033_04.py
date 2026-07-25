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

import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base
import run_multisite_input_enabled_five_site_half_split_maskableppo_rerun_033_04 as rerun


def main() -> None:
    split = rerun.load_split_available_weather()
    train_df = pd.read_csv(rerun.OUT / "evaluation" / "032_22_training_checkpoint_inventory.csv", keep_default_na=False)
    reset_path = rerun.OUT / "logs" / "032_22_training_year_reset_counts.csv"
    reset_df = pd.read_csv(reset_path, keep_default_na=False) if reset_path.exists() else pd.DataFrame()
    eval_df = pd.read_csv(rerun.OUT / "evaluation" / "032_22_checkpoint_validation_summary.csv", keep_default_na=False)
    by_station = pd.read_csv(rerun.OUT / "evaluation" / "032_22_validation_summary_by_station_checkpoint.csv", keep_default_na=False)

    rerun.write_record(split, train_df, reset_df, eval_df, by_station)
    result = {
        "task": "033_04_multisite_input_enabled_five_site_half_split_maskableppo_rerun",
        "record_md": str(rerun.DOC.relative_to(ROOT)).replace("\\", "/"),
        "train_inventory": str((rerun.OUT / "evaluation" / "032_22_training_checkpoint_inventory.csv").relative_to(ROOT)).replace("\\", "/"),
        "validation_summary": str((rerun.OUT / "evaluation" / "032_22_checkpoint_validation_summary.csv").relative_to(ROOT)).replace("\\", "/"),
        "by_station": str((rerun.OUT / "evaluation" / "032_22_validation_summary_by_station_checkpoint.csv").relative_to(ROOT)).replace("\\", "/"),
        "filtered_split": str(rerun.FILTERED_SPLIT.relative_to(ROOT)).replace("\\", "/"),
        "skipped_missing_multisite_wth": str(rerun.SKIPPED_WTH.relative_to(ROOT)).replace("\\", "/"),
        "input_source": "DSSAT_auto_validation/multisite_new_cultivar_inputs_013",
        "stations": base.SITES,
        "total_timesteps_per_station": base.TOTAL_TIMESTEPS,
        "baseline_comparison": "not_computed_old_baselines_use_old_input_chain",
    }
    out = rerun.OUT / "033_04_result.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
