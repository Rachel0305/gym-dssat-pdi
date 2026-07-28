# 032_15 LC2018 75k PPO five-scenario daily figure

## Purpose

Build one advisor-facing five-scenario daily process figure for the current LC multiyear 75k frozen MaskablePPO candidate.

## Scope

- Station: LC / LCA.
- Year: 2018.
- RL candidate: 032_12 frozen transfer evaluation from the 032_11 LC2005-2010 multiyear 75k checkpoint.
- Four baselines: null, recorded_farmer, dssat_auto, official_extension_expert.
- No training.
- No DSSAT rerun.
- No model reselection.

## Inputs

- PPO daily output:
  - `benchmark_results/032_12_lc_multiyear_75k_future_year_transfer/daily_outputs/LCA/LCA_2018_seed0_ckpt75000_daily.csv`
- Baseline daily outputs:
  - `benchmark_results/031_35_missing_four_baseline_completion_for_03134/evaluation/031_35_full_generated_baseline_daily.csv`
  - `benchmark_results/031_36_missing_dssat_auto_completion_for_03134/evaluation/031_36_full_generated_dssat_auto_daily.csv`

## Figure panels

Use the previous five-scenario style, but do not plot unavailable SWTD/soil-water storage as if it existed.

Required panels:

1. rainfall + Tmax/Tmin;
2. water stress index;
3. nitrogen stress index;
4. irrigation events;
5. nitrogen application events;
6. grain and biomass trajectories;
7. cumulative common reward, recomputed with one formula for all five scenarios;
8. endpoint metric summary.

## Common reward for plotting

Do not use source `reward` columns because previous 032_02 audit showed mixed reward scales across PPO and baselines.

For all scenarios, recompute a common scaled daily reward:

```text
common_reward_step =
  0.001 * (
      0.158 * grnwt
      - 1.1 * irrigation_mm
      - 1.58 * nitrogen_kg_ha
      + 10.0 * irrigation_mm * max(previous_wspd - current_wspd, 0)
      + 5.0 * nitrogen_kg_ha * max(previous_nstd - current_nstd, 0)
  )
```

Then plot the cumulative sum by DAP.

## Outputs

Write to:

`benchmark_results/032_15_lc2018_75k_ppo_five_scenario_daily_figure/`

Required:

- `tables/032_15_lc2018_75k_ppo_five_scenario_daily.csv`
- `tables/032_15_lc2018_75k_ppo_five_scenario_summary.csv`
- `figures/032_15_lc2018_75k_ppo_five_scenario_daily.png`
- `docs/032_15_lc2018_75k_ppo_five_scenario_daily_figure_record.md`

## Interpretation boundary

- This is a plot packaging task only.
- Do not claim new training success.
- Report that the cumulative reward is a recomputed common plotting reward, not the mixed source reward.
