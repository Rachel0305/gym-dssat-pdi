# FQA 048 figure set

These seven figures summarize the two paired 2007 WGEN heldout episodes in 048, using the purple/green PPO palette from the HL 055_03 reference set.

- `fqa_seed0_2k_vs_10k_heldout_metrics.png`: grain yield, PFP_N, and evaluation return for both weather seeds.
- `fqa_seed0_2k_vs_10k_heldout_management.png`: wrapper cumulative irrigation and nitrogen.
- `fqa_seed0_2k_vs_10k_heldout_reward.png`: evaluation episode return by weather seed.
- `fqa_seed0_1081_heldout_management_bars.png` and `fqa_seed0_1100_heldout_management_bars.png`: individual paired management bars.
- `fqa_seed0_1081_heldout_daily_weather.png` and `fqa_seed0_1100_heldout_daily_weather.png`: archived daily rain, temperature, and solar radiation.

048 did not save daily crop-state trajectories or daily action traces. The daily figures therefore show archived weather only; they do not reconstruct crop stress, biomass, yield, or management events. WP_ET is not shown because no ETCP replay was performed. PFP_N and management totals use safety-wrapper cumulative inputs and were not closed against DSSAT Summary.OUT in 048.
