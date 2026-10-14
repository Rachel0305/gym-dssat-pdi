"""Seed-2 wrapper for the final FQ E4 paired 2K stability smoke."""

from pathlib import Path
import run_e4_fqa_warmstart_weather_residual_2k as e4

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

if __name__ == "__main__":
    raise SystemExit(e4.main())
