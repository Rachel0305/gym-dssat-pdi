"""Seed-2 wrapper for the cached DSSAT WP_ET/PFP_N replay."""

from pathlib import Path
import pandas as pd
import run_e4_fqa_warmstart_weather_residual_2k as e4
import run_e4_fqa_wp_et_replay as replay

ROOT = Path(__file__).resolve().parents[1]
e4.SEED = 2
e4.SOURCE = ROOT / "benchmark_results/180FQA_fqa_originIC_dual_branch_noforecast_control_2k_seed2_smoke2k/models/FQA/FQA_half_split_stress_aware_maskableppo_seed2_ckpt2000.zip"
e4.CFG_PATHS.clear()
e4.CFG_PATHS.update({
    "forecast": ROOT / "configs/184FQA_fqa_originIC_e4_warmstart_weather_residual_2k_seed2.json",
    "noforecast": ROOT / "configs/185FQA_fqa_originIC_e4_warmstart_noforecast_control_2k_seed2.json",
})
e4.RESULT_BASENAME = "184_185_FQA_e4_seed2_warmstart_weather_residual_2k_smoke_result"
e4.DOC_BASENAME = "184_185_FQA_e4_seed2_warmstart_weather_residual_2k_record"
replay.SEED = 2
replay.REPLAY_BASENAME = "184_185_FQA_e4_seed2_wp_et_replay"

if __name__ == "__main__":
    # The seed-2 smoke was completed before the filename template was fixed;
    # repair only the generated provenance fields after the model files were
    # renamed to include seed2.
    for kind in ["forecast", "noforecast"]:
        cfg = e4.read_json(e4.CFG_PATHS[kind])
        out = e4.output_root(kind)
        inventory = pd.read_csv(out / "evaluation" / "training_checkpoint_inventory.csv", keep_default_na=False)
        model_path = str(inventory.iloc[0]["model_path"])
        summary_path = out / "evaluation" / f"{cfg['task_id']}_checkpoint_validation_summary.csv"
        summary = pd.read_csv(summary_path, keep_default_na=False)
        summary["model_path"] = model_path
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    replay.main()
