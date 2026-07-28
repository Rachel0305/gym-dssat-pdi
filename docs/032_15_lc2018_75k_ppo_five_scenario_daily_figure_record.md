# 032_15 LC2018 75k PPO five-scenario daily figure record

## Status

- Completed.
- Training run: 0.
- DSSAT run: 0.
- Model reselection: none.

## Inputs

- PPO daily: `benchmark_results/032_12_lc_multiyear_75k_future_year_transfer/daily_outputs/LCA/LCA_2018_seed0_ckpt75000_daily.csv`
- Baseline daily: `benchmark_results/031_35_missing_four_baseline_completion_for_03134/evaluation/031_35_full_generated_baseline_daily.csv`
- DSSAT-auto daily: `benchmark_results/031_36_missing_dssat_auto_completion_for_03134/evaluation/031_36_full_generated_dssat_auto_daily.csv`

## Outputs

- Daily table: `benchmark_results/032_15_lc2018_75k_ppo_five_scenario_daily_figure/tables/032_15_lc2018_75k_ppo_five_scenario_daily.csv`
- Summary table: `benchmark_results/032_15_lc2018_75k_ppo_five_scenario_daily_figure/tables/032_15_lc2018_75k_ppo_five_scenario_summary.csv`
- Figure: `benchmark_results/032_15_lc2018_75k_ppo_five_scenario_daily_figure/figures/032_15_lc2018_75k_ppo_five_scenario_daily.png`

## Scenario summary

| site | station_code | year | scenario | label | final_grain_kg_ha | final_biomass_kg_ha | irrigation_total_mm | nitrogen_total_kg_ha | PFP_N_kg_kg | max_wspd | max_nstd | irrigation_event_count | nitrogen_event_count | final_common_reward_scaled |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LC | LCA | 2018 | null | Null | 2676.067 | 4938.943 | 0.0 | 0.0 |  | 0.0 | 0.477 | 0 | 0 | 7.343 |
| LC | LCA | 2018 | recorded_farmer | Recorded farmer | 7913.131 | 16247.34 | 130.0 | 250.0 | 31.653 | 0.0 | 0.008 | 2 | 2 | 21.762 |
| LC | LCA | 2018 | dssat_auto | DSSAT auto | 2713.999 | 4967.417 | 0.0 | 0.0 |  | 0.0 | 0.477 | 0 | 0 | 7.444 |
| LC | LCA | 2018 | official_extension_expert | Official expert | 7917.404 | 16269.525 | 198.75 | 247.5 | 31.99 | 0.0 | 0.008 | 5 | 5 | 21.705 |
| LC | LCA | 2018 | rl_candidate | 75k PPO candidate | 8233.118 | 16173.058 | 75.0 | 200.0 | 41.166 | 0.106 | 0.012 | 3 | 3 | 22.016 |

## Reward plotting note

- Source reward columns are not used for cross-scenario comparison.
- `common_reward_cumulative_scaled` is recomputed using one 032 stress-aware formula for all five scenarios.
