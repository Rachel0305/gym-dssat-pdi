# 032_07 LC2010 seed0/50k cross-year transfer record

## Scope

- Frozen source model: LC2010 MaskablePPO seed0 checkpoint 50k from 032_04.
- Target years: LC2012-LC2015.
- No retraining, no reward change, no checkpoint reselection.
- LC2011 is excluded because current completed-baseline daily files do not contain DSSAT auto for that year.
- Mixed cumulative reward is not plotted; figures use cumulative irrigation/nitrogen per 032_02.

## Transfer evaluation summary

| algorithm | policy_name | run_status | episode_length | final_grnwt | total_irrigation | total_n | profit_simple | PFP_N | reward_stress_aware_sum | stress_relief_bonus_sum_unscaled | early_dap1_10_irrigation | early_dap1_10_n | reached_both_caps_by_dap10 | irrigation_event_count | n_event_count | first_irrigation_dap | first_n_dap | swfac_stress_days_gt_0p05 | nstres_days_gt_0p05 | action_sequence | model_path | daily_csv_path | station_code | target_year | source_year | seed | checkpoint_step | max_swfac | swfac_days_gt_0p001 | swfac_days_gt_0p01 | swfac_days_gt_0p05 | max_nstres | nstres_days_gt_0p001 | nstres_days_gt_0p01 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MaskablePPO | maskableppo_stress_aware_5k_seed0 | ok | 117 | 8972.615 | 45.0 | 160.0 | 8670.315 | 56.079 | 1.268 | 152.135 | 45.0 | 160.0 | False | 1 | 2 | 1 | 2 | 0 | 24 | DAP1 I45/N0; DAP2 I0/N40; DAP8 I0/N120 | benchmark_results/032_04_lc2010_stress_aware_ppo_multiseed_200k/models/LCA/LCA_2010_stress_aware_maskableppo_seed0_ckpt50000.zip | benchmark_results/032_07_lc2010_seed0_50k_cross_year_transfer/daily_outputs/LCA/LCA_2012_source2010_seed0_ckpt50000_transfer_daily.csv | LCA | 2012 | 2010 | 0 | 50000 | 0.0 | 0 | 0 | 0 | 0.19 | 29 | 26 |
| MaskablePPO | maskableppo_stress_aware_5k_seed0 | ok | 96 | 7445.668 | 60.0 | 80.0 | 7253.268 | 93.071 | 1.134 | 150.262 | 60.0 | 80.0 | False | 2 | 2 | 1 | 2 | 0 | 50 | DAP1 I45/N0; DAP2 I0/N40; DAP8 I0/N40; DAP9 I15/N0 | benchmark_results/032_04_lc2010_stress_aware_ppo_multiseed_200k/models/LCA/LCA_2010_stress_aware_maskableppo_seed0_ckpt50000.zip | benchmark_results/032_07_lc2010_seed0_50k_cross_year_transfer/daily_outputs/LCA/LCA_2013_source2010_seed0_ckpt50000_transfer_daily.csv | LCA | 2013 | 2010 | 0 | 50000 | 0.031 | 1 | 1 | 0 | 0.504 | 57 | 54 |
| MaskablePPO | maskableppo_stress_aware_5k_seed0 | ok | 102 | 9923.693 | 45.0 | 160.0 | 9621.393 | 62.023 | 1.419 | 153.585 | 45.0 | 160.0 | False | 1 | 2 | 1 | 2 | 2 | 21 | DAP1 I45/N0; DAP2 I0/N40; DAP8 I0/N120 | benchmark_results/032_04_lc2010_stress_aware_ppo_multiseed_200k/models/LCA/LCA_2010_stress_aware_maskableppo_seed0_ckpt50000.zip | benchmark_results/032_07_lc2010_seed0_50k_cross_year_transfer/daily_outputs/LCA/LCA_2014_source2010_seed0_ckpt50000_transfer_daily.csv | LCA | 2014 | 2010 | 0 | 50000 | 0.477 | 2 | 2 | 2 | 0.274 | 28 | 25 |
| MaskablePPO | maskableppo_stress_aware_5k_seed0 | ok | 99 | 9870.453 | 45.0 | 160.0 | 9568.153 | 61.69 | 1.407 | 149.521 | 45.0 | 160.0 | False | 1 | 2 | 1 | 2 | 0 | 14 | DAP1 I45/N0; DAP2 I0/N40; DAP8 I0/N120 | benchmark_results/032_04_lc2010_stress_aware_ppo_multiseed_200k/models/LCA/LCA_2010_stress_aware_maskableppo_seed0_ckpt50000.zip | benchmark_results/032_07_lc2010_seed0_50k_cross_year_transfer/daily_outputs/LCA/LCA_2015_source2010_seed0_ckpt50000_transfer_daily.csv | LCA | 2015 | 2010 | 0 | 50000 | 0.0 | 0 | 0 | 0 | 0.326 | 21 | 18 |

## Five-scenario endpoint summary

| year | scenario | final_grain_kg_ha | irrigation_event_total_mm | nitrogen_event_total_kg_ha | PFP_N | max_water_stress_wspd | max_nitrogen_stress_nstd | irrigation_event_count | nitrogen_event_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2012 | dssat_auto | 2935.771 | 0.0 | 0.0 |  | 0.0 | 0.473 | 0 | 0 |
| 2012 | null | 2935.771 | 0.0 | 0.0 |  | 0.0 | 0.473 | 0 | 0 |
| 2012 | official_extension_expert | 9017.703 | 228.75 | 247.5 | 36.435 | 0.0 | 0.012 | 6 | 5 |
| 2012 | recorded_farmer | 8985.804 | 130.0 | 250.0 | 35.943 | 0.0 | 0.012 | 2 | 2 |
| 2012 | rl_candidate | 8972.615 | 45.0 | 160.0 | 56.079 | 0.0 | 0.19 | 1 | 2 |
| 2013 | dssat_auto | 2971.06 | 0.0 | 0.0 |  | 0.0 | 0.494 | 0 | 0 |
| 2013 | null | 2971.06 | 0.0 | 0.0 |  | 0.0 | 0.494 | 0 | 0 |
| 2013 | official_extension_expert | 8723.492 | 198.75 | 247.5 | 35.246 | 0.0 | 0.012 | 5 | 5 |
| 2013 | recorded_farmer | 8702.307 | 130.0 | 250.0 | 34.809 | 0.023 | 0.012 | 2 | 2 |
| 2013 | rl_candidate | 7445.668 | 60.0 | 80.0 | 93.071 | 0.031 | 0.504 | 2 | 2 |
| 2014 | dssat_auto | 3103.243 | 0.0 | 0.0 |  | 0.0 | 0.486 | 0 | 0 |
| 2014 | null | 3145.846 | 0.0 | 0.0 |  | 0.0 | 0.486 | 0 | 0 |
| 2014 | official_extension_expert | 10522.427 | 198.75 | 247.5 | 42.515 | 0.0 | 0.024 | 5 | 5 |
| 2014 | recorded_farmer | 10516.671 | 130.0 | 250.0 | 42.067 | 0.0 | 0.012 | 2 | 2 |
| 2014 | rl_candidate | 9923.693 | 45.0 | 160.0 | 62.023 | 0.477 | 0.274 | 1 | 2 |
| 2015 | dssat_auto | 3063.429 | 0.0 | 0.0 |  | 0.0 | 0.485 | 0 | 0 |
| 2015 | null | 3063.429 | 0.0 | 0.0 |  | 0.0 | 0.485 | 0 | 0 |
| 2015 | official_extension_expert | 10371.388 | 198.75 | 247.5 | 41.905 | 0.0 | 0.012 | 5 | 5 |
| 2015 | recorded_farmer | 10051.912 | 130.0 | 250.0 | 40.208 | 0.0 | 0.012 | 2 | 2 |
| 2015 | rl_candidate | 9870.453 | 45.0 | 160.0 | 61.69 | 0.0 | 0.326 | 1 | 2 |

## PPO transfer vs official expert / best baseline

| year | ppo_yield | expert_yield | delta_yield_vs_expert | delta_yield_vs_best_baseline | ppo_irrigation | expert_irrigation | irrigation_saving_vs_expert | ppo_n | expert_n | n_saving_vs_expert | ppo_PFP_N | expert_PFP_N | delta_PFP_N_vs_expert | delta_PFP_N_vs_best_baseline | ppo_max_wspd | ppo_max_nstd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2012 | 8972.615 | 9017.703 | -45.089 | -45.089 | 45.0 | 228.75 | 183.75 | 160.0 | 247.5 | 87.5 | 56.079 | 36.435 | 19.644 | 19.644 | 0.0 | 0.19 |
| 2013 | 7445.668 | 8723.492 | -1277.823 | -1277.823 | 60.0 | 198.75 | 138.75 | 80.0 | 247.5 | 167.5 | 93.071 | 35.246 | 57.824 | 57.824 | 0.031 | 0.504 |
| 2014 | 9923.693 | 10522.427 | -598.734 | -598.734 | 45.0 | 198.75 | 153.75 | 160.0 | 247.5 | 87.5 | 62.023 | 42.515 | 19.508 | 19.508 | 0.477 | 0.274 |
| 2015 | 9870.453 | 10371.388 | -500.935 | -500.935 | 45.0 | 198.75 | 153.75 | 160.0 | 247.5 | 87.5 | 61.69 | 41.905 | 19.786 | 19.786 | 0.0 | 0.326 |

## Checks

| year | check | passed | value |
| --- | --- | --- | --- |
| 2012 | five_scenarios_present | True | dssat_auto,null,official_extension_expert,recorded_farmer,rl_candidate |
| 2013 | five_scenarios_present | True | dssat_auto,null,official_extension_expert,recorded_farmer,rl_candidate |
| 2014 | five_scenarios_present | True | dssat_auto,null,official_extension_expert,recorded_farmer,rl_candidate |
| 2015 | five_scenarios_present | True | dssat_auto,null,official_extension_expert,recorded_farmer,rl_candidate |

## Outputs

| year | daily_csv | summary_csv | figures |
| --- | --- | --- | --- |
| 2012 | benchmark_results/032_07_lc2010_seed0_50k_cross_year_transfer/tables/032_07_lc2012_five_scenario_daily.csv | benchmark_results/032_07_lc2010_seed0_50k_cross_year_transfer/tables/032_07_lc2012_five_scenario_summary.csv | benchmark_results/032_07_lc2010_seed0_50k_cross_year_transfer/figures/032_07_lc2012_source2010_seed0_ckpt50000_transfer_five_scenario_daily.png;benchmark_results/032_07_lc2010_seed0_50k_cross_year_transfer/figures/032_07_lc2012_source2010_seed0_ckpt50000_transfer_five_scenario_daily.svg |
| 2013 | benchmark_results/032_07_lc2010_seed0_50k_cross_year_transfer/tables/032_07_lc2013_five_scenario_daily.csv | benchmark_results/032_07_lc2010_seed0_50k_cross_year_transfer/tables/032_07_lc2013_five_scenario_summary.csv | benchmark_results/032_07_lc2010_seed0_50k_cross_year_transfer/figures/032_07_lc2013_source2010_seed0_ckpt50000_transfer_five_scenario_daily.png;benchmark_results/032_07_lc2010_seed0_50k_cross_year_transfer/figures/032_07_lc2013_source2010_seed0_ckpt50000_transfer_five_scenario_daily.svg |
| 2014 | benchmark_results/032_07_lc2010_seed0_50k_cross_year_transfer/tables/032_07_lc2014_five_scenario_daily.csv | benchmark_results/032_07_lc2010_seed0_50k_cross_year_transfer/tables/032_07_lc2014_five_scenario_summary.csv | benchmark_results/032_07_lc2010_seed0_50k_cross_year_transfer/figures/032_07_lc2014_source2010_seed0_ckpt50000_transfer_five_scenario_daily.png;benchmark_results/032_07_lc2010_seed0_50k_cross_year_transfer/figures/032_07_lc2014_source2010_seed0_ckpt50000_transfer_five_scenario_daily.svg |
| 2015 | benchmark_results/032_07_lc2010_seed0_50k_cross_year_transfer/tables/032_07_lc2015_five_scenario_daily.csv | benchmark_results/032_07_lc2010_seed0_50k_cross_year_transfer/tables/032_07_lc2015_five_scenario_summary.csv | benchmark_results/032_07_lc2010_seed0_50k_cross_year_transfer/figures/032_07_lc2015_source2010_seed0_ckpt50000_transfer_five_scenario_daily.png;benchmark_results/032_07_lc2010_seed0_50k_cross_year_transfer/figures/032_07_lc2015_source2010_seed0_ckpt50000_transfer_five_scenario_daily.svg |

## Interpretation boundary

- This is a frozen-policy transfer diagnostic, not a new training result.
- Any favorable or unfavorable year should be interpreted as LC2010-policy transfer behavior.
- Soil water SWTD is not available in the completed generated baseline daily files used here; the soil-water panel is therefore marked unavailable rather than fabricated.
