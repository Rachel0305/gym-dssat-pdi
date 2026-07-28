# 032_08 LC2010 seed0/50k remaining-year transfer record

## Scope

- Frozen source model: LC2010 MaskablePPO seed0 checkpoint 50k from 032_04.
- Target years: LC2005-LC2007 and LC2016-LC2023.
- No retraining, no reward change, no checkpoint reselection.
- LC2008, LC2009, and LC2011 are excluded because current completed-baseline daily files do not contain DSSAT auto for those years.
- Mixed cumulative reward is not plotted; figures use cumulative irrigation/nitrogen per 032_02.

## Transfer evaluation summary

| algorithm | policy_name | run_status | episode_length | final_grnwt | total_irrigation | total_n | profit_simple | PFP_N | reward_stress_aware_sum | stress_relief_bonus_sum_unscaled | early_dap1_10_irrigation | early_dap1_10_n | reached_both_caps_by_dap10 | irrigation_event_count | n_event_count | first_irrigation_dap | first_n_dap | swfac_stress_days_gt_0p05 | nstres_days_gt_0p05 | action_sequence | model_path | daily_csv_path | station_code | target_year | source_year | seed | checkpoint_step | max_swfac | swfac_days_gt_0p001 | swfac_days_gt_0p01 | swfac_days_gt_0p05 | max_nstres | nstres_days_gt_0p001 | nstres_days_gt_0p01 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MaskablePPO | maskableppo_stress_aware_5k_seed0 | ok | 107 | 8131.811 | 60.0 | 80.0 | 7939.411 | 101.648 | 1.241 | 148.5 | 60.0 | 80.0 | False | 2 | 2 | 1 | 2 | 0 | 66 | DAP1 I45/N0; DAP2 I0/N40; DAP8 I0/N40; DAP9 I15/N0 | benchmark_results/032_04_lc2010_stress_aware_ppo_multiseed_200k/models/LCA/LCA_2010_stress_aware_maskableppo_seed0_ckpt50000.zip | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/daily_outputs/LCA/LCA_2005_source2010_seed0_ckpt50000_transfer_daily.csv | LCA | 2005 | 2010 | 0 | 50000 | 0.0 | 0 | 0 | 0 | 0.494 | 71 | 68 |
| MaskablePPO | maskableppo_stress_aware_5k_seed0 | ok | 98 | 9224.913 | 45.0 | 160.0 | 8922.613 | 57.656 | 1.304 | 148.5 | 45.0 | 160.0 | False | 1 | 2 | 1 | 2 | 0 | 4 | DAP1 I45/N0; DAP2 I0/N40; DAP8 I0/N120 | benchmark_results/032_04_lc2010_stress_aware_ppo_multiseed_200k/models/LCA/LCA_2010_stress_aware_maskableppo_seed0_ckpt50000.zip | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/daily_outputs/LCA/LCA_2006_source2010_seed0_ckpt50000_transfer_daily.csv | LCA | 2006 | 2010 | 0 | 50000 | 0.0 | 0 | 0 | 0 | 0.128 | 8 | 6 |
| MaskablePPO | maskableppo_stress_aware_5k_seed0 | ok | 98 | 8279.055 | 45.0 | 160.0 | 7976.755 | 51.744 | 1.159 | 153.455 | 45.0 | 160.0 | False | 1 | 2 | 1 | 2 | 0 | 4 | DAP1 I45/N0; DAP2 I0/N40; DAP8 I0/N120 | benchmark_results/032_04_lc2010_stress_aware_ppo_multiseed_200k/models/LCA/LCA_2010_stress_aware_maskableppo_seed0_ckpt50000.zip | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/daily_outputs/LCA/LCA_2007_source2010_seed0_ckpt50000_transfer_daily.csv | LCA | 2007 | 2010 | 0 | 50000 | 0.0 | 0 | 0 | 0 | 0.053 | 7 | 6 |
| MaskablePPO | maskableppo_stress_aware_5k_seed0 | ok | 93 | 7090.336 | 60.0 | 80.0 | 6897.936 | 88.629 | 1.076 | 148.5 | 60.0 | 80.0 | False | 2 | 2 | 1 | 2 | 0 | 43 | DAP1 I45/N0; DAP2 I0/N40; DAP8 I0/N40; DAP9 I15/N0 | benchmark_results/032_04_lc2010_stress_aware_ppo_multiseed_200k/models/LCA/LCA_2010_stress_aware_maskableppo_seed0_ckpt50000.zip | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/daily_outputs/LCA/LCA_2016_source2010_seed0_ckpt50000_transfer_daily.csv | LCA | 2016 | 2010 | 0 | 50000 | 0.0 | 0 | 0 | 0 | 0.522 | 53 | 50 |
| MaskablePPO | maskableppo_stress_aware_5k_seed0 | ok | 94 | 7340.232 | 60.0 | 80.0 | 7147.832 | 91.753 | 1.116 | 148.549 | 60.0 | 80.0 | False | 2 | 2 | 1 | 2 | 0 | 56 | DAP1 I45/N0; DAP2 I0/N40; DAP8 I0/N40; DAP9 I15/N0 | benchmark_results/032_04_lc2010_stress_aware_ppo_multiseed_200k/models/LCA/LCA_2010_stress_aware_maskableppo_seed0_ckpt50000.zip | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/daily_outputs/LCA/LCA_2017_source2010_seed0_ckpt50000_transfer_daily.csv | LCA | 2017 | 2010 | 0 | 50000 | 0.0 | 0 | 0 | 0 | 0.458 | 62 | 59 |
| MaskablePPO | maskableppo_stress_aware_5k_seed0 | ok | 87 | 6999.602 | 60.0 | 80.0 | 6807.202 | 87.495 | 1.062 | 148.5 | 60.0 | 80.0 | False | 2 | 2 | 1 | 2 | 0 | 46 | DAP1 I45/N0; DAP2 I0/N40; DAP8 I0/N40; DAP9 I15/N0 | benchmark_results/032_04_lc2010_stress_aware_ppo_multiseed_200k/models/LCA/LCA_2010_stress_aware_maskableppo_seed0_ckpt50000.zip | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/daily_outputs/LCA/LCA_2018_source2010_seed0_ckpt50000_transfer_daily.csv | LCA | 2018 | 2010 | 0 | 50000 | 0.0 | 0 | 0 | 0 | 0.458 | 51 | 49 |
| MaskablePPO | maskableppo_stress_aware_5k_seed0 | ok | 92 | 7876.842 | 60.0 | 80.0 | 7684.442 | 98.461 | 1.201 | 148.5 | 60.0 | 80.0 | False | 2 | 2 | 1 | 2 | 0 | 55 | DAP1 I45/N0; DAP2 I0/N40; DAP8 I0/N40; DAP9 I15/N0 | benchmark_results/032_04_lc2010_stress_aware_ppo_multiseed_200k/models/LCA/LCA_2010_stress_aware_maskableppo_seed0_ckpt50000.zip | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/daily_outputs/LCA/LCA_2019_source2010_seed0_ckpt50000_transfer_daily.csv | LCA | 2019 | 2010 | 0 | 50000 | 0.0 | 0 | 0 | 0 | 0.54 | 59 | 57 |
| MaskablePPO | maskableppo_stress_aware_5k_seed0 | ok | 92 | 6740.631 | 60.0 | 80.0 | 6548.231 | 84.258 | 1.022 | 149.18 | 60.0 | 80.0 | False | 2 | 2 | 1 | 2 | 0 | 50 | DAP1 I45/N0; DAP2 I0/N40; DAP8 I0/N40; DAP9 I15/N0 | benchmark_results/032_04_lc2010_stress_aware_ppo_multiseed_200k/models/LCA/LCA_2010_stress_aware_maskableppo_seed0_ckpt50000.zip | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/daily_outputs/LCA/LCA_2020_source2010_seed0_ckpt50000_transfer_daily.csv | LCA | 2020 | 2010 | 0 | 50000 | 0.0 | 0 | 0 | 0 | 0.521 | 55 | 52 |
| MaskablePPO | maskableppo_stress_aware_5k_seed0 | ok | 96 | 8264.215 | 60.0 | 80.0 | 8071.815 | 103.303 | 1.262 | 148.5 | 60.0 | 80.0 | False | 2 | 2 | 1 | 2 | 0 | 56 | DAP1 I45/N0; DAP2 I0/N40; DAP8 I0/N40; DAP9 I15/N0 | benchmark_results/032_04_lc2010_stress_aware_ppo_multiseed_200k/models/LCA/LCA_2010_stress_aware_maskableppo_seed0_ckpt50000.zip | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/daily_outputs/LCA/LCA_2021_source2010_seed0_ckpt50000_transfer_daily.csv | LCA | 2021 | 2010 | 0 | 50000 | 0.0 | 0 | 0 | 0 | 0.507 | 63 | 60 |
| MaskablePPO | maskableppo_stress_aware_5k_seed0 | ok | 95 | 9767.673 | 45.0 | 160.0 | 9465.373 | 61.048 | 1.389 | 148.5 | 45.0 | 160.0 | False | 1 | 2 | 1 | 2 | 0 | 13 | DAP1 I45/N0; DAP2 I0/N40; DAP8 I0/N120 | benchmark_results/032_04_lc2010_stress_aware_ppo_multiseed_200k/models/LCA/LCA_2010_stress_aware_maskableppo_seed0_ckpt50000.zip | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/daily_outputs/LCA/LCA_2022_source2010_seed0_ckpt50000_transfer_daily.csv | LCA | 2022 | 2010 | 0 | 50000 | 0.0 | 0 | 0 | 0 | 0.434 | 16 | 15 |
| MaskablePPO | maskableppo_stress_aware_5k_seed0 | ok | 88 | 6759.474 | 60.0 | 80.0 | 6567.074 | 84.493 | 1.024 | 148.5 | 60.0 | 80.0 | False | 2 | 2 | 1 | 2 | 0 | 51 | DAP1 I45/N0; DAP2 I0/N40; DAP8 I0/N40; DAP9 I15/N0 | benchmark_results/032_04_lc2010_stress_aware_ppo_multiseed_200k/models/LCA/LCA_2010_stress_aware_maskableppo_seed0_ckpt50000.zip | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/daily_outputs/LCA/LCA_2023_source2010_seed0_ckpt50000_transfer_daily.csv | LCA | 2023 | 2010 | 0 | 50000 | 0.0 | 0 | 0 | 0 | 0.509 | 54 | 53 |

## Five-scenario endpoint summary

| year | scenario | final_grain_kg_ha | irrigation_event_total_mm | nitrogen_event_total_kg_ha | PFP_N | max_water_stress_wspd | max_nitrogen_stress_nstd | irrigation_event_count | nitrogen_event_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2005 | dssat_auto | 3192.892 | 0.0 | 0.0 |  | 0.0 | 0.49 | 0 | 0 |
| 2005 | null | 3192.892 | 0.0 | 0.0 |  | 0.0 | 0.49 | 0 | 0 |
| 2005 | official_extension_expert | 12084.387 | 228.75 | 247.5 | 48.826 | 0.0 | 0.008 | 6 | 5 |
| 2005 | recorded_farmer | 11768.635 | 130.0 | 250.0 | 47.075 | 0.0 | 0.008 | 2 | 2 |
| 2005 | rl_candidate | 8131.811 | 60.0 | 80.0 | 101.648 | 0.0 | 0.494 | 2 | 2 |
| 2006 | dssat_auto | 2981.537 | 0.0 | 0.0 |  | 0.0 | 0.504 | 0 | 0 |
| 2006 | null | 2981.537 | 0.0 | 0.0 |  | 0.0 | 0.504 | 0 | 0 |
| 2006 | official_extension_expert | 9444.844 | 198.75 | 247.5 | 38.161 | 0.0 | 0.007 | 5 | 5 |
| 2006 | recorded_farmer | 9189.549 | 130.0 | 250.0 | 36.758 | 0.0 | 0.007 | 2 | 2 |
| 2006 | rl_candidate | 9224.913 | 45.0 | 160.0 | 57.656 | 0.0 | 0.128 | 1 | 2 |
| 2007 | dssat_auto | 2804.122 | 0.0 | 0.0 |  | 0.0 | 0.455 | 0 | 0 |
| 2007 | null | 2804.122 | 0.0 | 0.0 |  | 0.0 | 0.455 | 0 | 0 |
| 2007 | official_extension_expert | 8105.514 | 198.75 | 247.5 | 32.75 | 0.0 | 0.012 | 5 | 5 |
| 2007 | recorded_farmer | 8048.226 | 130.0 | 250.0 | 32.193 | 0.0 | 0.012 | 2 | 2 |
| 2007 | rl_candidate | 8279.055 | 45.0 | 160.0 | 51.744 | 0.0 | 0.053 | 1 | 2 |
| 2016 | dssat_auto | 2874.945 | 0.0 | 0.0 |  | 0.0 | 0.461 | 0 | 0 |
| 2016 | null | 2874.945 | 0.0 | 0.0 |  | 0.0 | 0.461 | 0 | 0 |
| 2016 | official_extension_expert | 8565.475 | 198.75 | 247.5 | 34.608 | 0.0 | 0.007 | 5 | 5 |
| 2016 | recorded_farmer | 8364.429 | 130.0 | 250.0 | 33.458 | 0.0 | 0.025 | 2 | 2 |
| 2016 | rl_candidate | 7090.336 | 60.0 | 80.0 | 88.629 | 0.0 | 0.522 | 2 | 2 |
| 2017 | dssat_auto | 2915.772 | 0.0 | 0.0 |  | 0.0 | 0.488 | 0 | 0 |
| 2017 | null | 2914.419 | 0.0 | 0.0 |  | 0.0 | 0.488 | 0 | 0 |
| 2017 | official_extension_expert | 9064.438 | 198.75 | 247.5 | 36.624 | 0.0 | 0.008 | 5 | 5 |
| 2017 | recorded_farmer | 9041.351 | 130.0 | 250.0 | 36.165 | 0.0 | 0.006 | 2 | 2 |
| 2017 | rl_candidate | 7340.232 | 60.0 | 80.0 | 91.753 | 0.0 | 0.458 | 2 | 2 |
| 2018 | dssat_auto | 2713.999 | 0.0 | 0.0 |  | 0.0 | 0.477 | 0 | 0 |
| 2018 | null | 2676.067 | 0.0 | 0.0 |  | 0.0 | 0.477 | 0 | 0 |
| 2018 | official_extension_expert | 7917.404 | 198.75 | 247.5 | 31.99 | 0.0 | 0.008 | 5 | 5 |
| 2018 | recorded_farmer | 7913.131 | 130.0 | 250.0 | 31.653 | 0.0 | 0.008 | 2 | 2 |
| 2018 | rl_candidate | 6999.602 | 60.0 | 80.0 | 87.495 | 0.0 | 0.458 | 2 | 2 |
| 2019 | dssat_auto | 2982.423 | 0.0 | 0.0 |  | 0.0 | 0.504 | 0 | 0 |
| 2019 | null | 2982.423 | 0.0 | 0.0 |  | 0.0 | 0.504 | 0 | 0 |
| 2019 | official_extension_expert | 9725.751 | 198.75 | 247.5 | 39.296 | 0.0 | 0.046 | 5 | 5 |
| 2019 | recorded_farmer | 9805.405 | 130.0 | 250.0 | 39.222 | 0.0 | 0.005 | 2 | 2 |
| 2019 | rl_candidate | 7876.842 | 60.0 | 80.0 | 98.461 | 0.0 | 0.54 | 2 | 2 |
| 2020 | dssat_auto | 2475.929 | 0.0 | 0.0 |  | 0.0 | 0.49 | 0 | 0 |
| 2020 | null | 2475.929 | 0.0 | 0.0 |  | 0.0 | 0.49 | 0 | 0 |
| 2020 | official_extension_expert | 7775.217 | 198.75 | 247.5 | 31.415 | 0.0 | 0.012 | 5 | 5 |
| 2020 | recorded_farmer | 7762.199 | 130.0 | 250.0 | 31.049 | 0.0 | 0.012 | 2 | 2 |
| 2020 | rl_candidate | 6740.631 | 60.0 | 80.0 | 84.258 | 0.0 | 0.521 | 2 | 2 |
| 2021 | dssat_auto | 3216.673 | 0.0 | 0.0 |  | 0.0 | 0.5 | 0 | 0 |
| 2021 | null | 3216.673 | 0.0 | 0.0 |  | 0.0 | 0.5 | 0 | 0 |
| 2021 | official_extension_expert | 11254.507 | 198.75 | 247.5 | 45.473 | 0.0 | 0.009 | 5 | 5 |
| 2021 | recorded_farmer | 11041.3 | 130.0 | 250.0 | 44.165 | 0.0 | 0.009 | 2 | 2 |
| 2021 | rl_candidate | 8264.215 | 60.0 | 80.0 | 103.303 | 0.0 | 0.507 | 2 | 2 |
| 2022 | dssat_auto | 2967.667 | 0.0 | 0.0 |  | 0.0 | 0.501 | 0 | 0 |
| 2022 | null | 2967.667 | 0.0 | 0.0 |  | 0.0 | 0.501 | 0 | 0 |
| 2022 | official_extension_expert | 10310.088 | 198.75 | 247.5 | 41.657 | 0.0 | 0.036 | 5 | 5 |
| 2022 | recorded_farmer | 10389.916 | 130.0 | 250.0 | 41.56 | 0.0 | 0.008 | 2 | 2 |
| 2022 | rl_candidate | 9767.673 | 45.0 | 160.0 | 61.048 | 0.0 | 0.434 | 1 | 2 |
| 2023 | dssat_auto | 2473.385 | 0.0 | 0.0 |  | 0.0 | 0.503 | 0 | 0 |
| 2023 | null | 2501.165 | 0.0 | 0.0 |  | 0.0 | 0.503 | 0 | 0 |
| 2023 | official_extension_expert | 7899.227 | 198.75 | 247.5 | 31.916 | 0.0 | 0.05 | 5 | 5 |
| 2023 | recorded_farmer | 7935.174 | 130.0 | 250.0 | 31.741 | 0.0 | 0.007 | 2 | 2 |
| 2023 | rl_candidate | 6759.474 | 60.0 | 80.0 | 84.493 | 0.0 | 0.509 | 2 | 2 |

## PPO transfer vs official expert / best baseline

| year | ppo_yield | expert_yield | delta_yield_vs_expert | delta_yield_vs_best_baseline | ppo_irrigation | expert_irrigation | irrigation_saving_vs_expert | ppo_n | expert_n | n_saving_vs_expert | ppo_PFP_N | expert_PFP_N | delta_PFP_N_vs_expert | delta_PFP_N_vs_best_baseline | ppo_max_wspd | ppo_max_nstd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2005 | 8131.811 | 12084.387 | -3952.576 | -3952.576 | 60.0 | 228.75 | 168.75 | 80.0 | 247.5 | 167.5 | 101.648 | 48.826 | 52.822 | 52.822 | 0.0 | 0.494 |
| 2006 | 9224.913 | 9444.844 | -219.931 | -219.931 | 45.0 | 198.75 | 153.75 | 160.0 | 247.5 | 87.5 | 57.656 | 38.161 | 19.495 | 19.495 | 0.0 | 0.128 |
| 2007 | 8279.055 | 8105.514 | 173.541 | 173.541 | 45.0 | 198.75 | 153.75 | 160.0 | 247.5 | 87.5 | 51.744 | 32.75 | 18.995 | 18.995 | 0.0 | 0.053 |
| 2016 | 7090.336 | 8565.475 | -1475.139 | -1475.139 | 60.0 | 198.75 | 138.75 | 80.0 | 247.5 | 167.5 | 88.629 | 34.608 | 54.021 | 54.021 | 0.0 | 0.522 |
| 2017 | 7340.232 | 9064.438 | -1724.206 | -1724.206 | 60.0 | 198.75 | 138.75 | 80.0 | 247.5 | 167.5 | 91.753 | 36.624 | 55.129 | 55.129 | 0.0 | 0.458 |
| 2018 | 6999.602 | 7917.404 | -917.802 | -917.802 | 60.0 | 198.75 | 138.75 | 80.0 | 247.5 | 167.5 | 87.495 | 31.99 | 55.506 | 55.506 | 0.0 | 0.458 |
| 2019 | 7876.842 | 9725.751 | -1848.909 | -1928.563 | 60.0 | 198.75 | 138.75 | 80.0 | 247.5 | 167.5 | 98.461 | 39.296 | 59.165 | 59.165 | 0.0 | 0.54 |
| 2020 | 6740.631 | 7775.217 | -1034.586 | -1034.586 | 60.0 | 198.75 | 138.75 | 80.0 | 247.5 | 167.5 | 84.258 | 31.415 | 52.843 | 52.843 | 0.0 | 0.521 |
| 2021 | 8264.215 | 11254.507 | -2990.292 | -2990.292 | 60.0 | 198.75 | 138.75 | 80.0 | 247.5 | 167.5 | 103.303 | 45.473 | 57.83 | 57.83 | 0.0 | 0.507 |
| 2022 | 9767.673 | 10310.088 | -542.415 | -622.243 | 45.0 | 198.75 | 153.75 | 160.0 | 247.5 | 87.5 | 61.048 | 41.657 | 19.391 | 19.391 | 0.0 | 0.434 |
| 2023 | 6759.474 | 7899.227 | -1139.753 | -1175.7 | 60.0 | 198.75 | 138.75 | 80.0 | 247.5 | 167.5 | 84.493 | 31.916 | 52.577 | 52.577 | 0.0 | 0.509 |

## Checks

| year | check | passed | value |
| --- | --- | --- | --- |
| 2005 | five_scenarios_present | True | dssat_auto,null,official_extension_expert,recorded_farmer,rl_candidate |
| 2006 | five_scenarios_present | True | dssat_auto,null,official_extension_expert,recorded_farmer,rl_candidate |
| 2007 | five_scenarios_present | True | dssat_auto,null,official_extension_expert,recorded_farmer,rl_candidate |
| 2016 | five_scenarios_present | True | dssat_auto,null,official_extension_expert,recorded_farmer,rl_candidate |
| 2017 | five_scenarios_present | True | dssat_auto,null,official_extension_expert,recorded_farmer,rl_candidate |
| 2018 | five_scenarios_present | True | dssat_auto,null,official_extension_expert,recorded_farmer,rl_candidate |
| 2019 | five_scenarios_present | True | dssat_auto,null,official_extension_expert,recorded_farmer,rl_candidate |
| 2020 | five_scenarios_present | True | dssat_auto,null,official_extension_expert,recorded_farmer,rl_candidate |
| 2021 | five_scenarios_present | True | dssat_auto,null,official_extension_expert,recorded_farmer,rl_candidate |
| 2022 | five_scenarios_present | True | dssat_auto,null,official_extension_expert,recorded_farmer,rl_candidate |
| 2023 | five_scenarios_present | True | dssat_auto,null,official_extension_expert,recorded_farmer,rl_candidate |

## Outputs

| year | daily_csv | summary_csv | figures |
| --- | --- | --- | --- |
| 2005 | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/tables/032_08_lc2005_five_scenario_daily.csv | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/tables/032_08_lc2005_five_scenario_summary.csv | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/figures/032_08_lc2005_source2010_seed0_ckpt50000_transfer_five_scenario_daily.png;benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/figures/032_08_lc2005_source2010_seed0_ckpt50000_transfer_five_scenario_daily.svg |
| 2006 | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/tables/032_08_lc2006_five_scenario_daily.csv | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/tables/032_08_lc2006_five_scenario_summary.csv | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/figures/032_08_lc2006_source2010_seed0_ckpt50000_transfer_five_scenario_daily.png;benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/figures/032_08_lc2006_source2010_seed0_ckpt50000_transfer_five_scenario_daily.svg |
| 2007 | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/tables/032_08_lc2007_five_scenario_daily.csv | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/tables/032_08_lc2007_five_scenario_summary.csv | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/figures/032_08_lc2007_source2010_seed0_ckpt50000_transfer_five_scenario_daily.png;benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/figures/032_08_lc2007_source2010_seed0_ckpt50000_transfer_five_scenario_daily.svg |
| 2016 | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/tables/032_08_lc2016_five_scenario_daily.csv | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/tables/032_08_lc2016_five_scenario_summary.csv | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/figures/032_08_lc2016_source2010_seed0_ckpt50000_transfer_five_scenario_daily.png;benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/figures/032_08_lc2016_source2010_seed0_ckpt50000_transfer_five_scenario_daily.svg |
| 2017 | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/tables/032_08_lc2017_five_scenario_daily.csv | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/tables/032_08_lc2017_five_scenario_summary.csv | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/figures/032_08_lc2017_source2010_seed0_ckpt50000_transfer_five_scenario_daily.png;benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/figures/032_08_lc2017_source2010_seed0_ckpt50000_transfer_five_scenario_daily.svg |
| 2018 | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/tables/032_08_lc2018_five_scenario_daily.csv | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/tables/032_08_lc2018_five_scenario_summary.csv | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/figures/032_08_lc2018_source2010_seed0_ckpt50000_transfer_five_scenario_daily.png;benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/figures/032_08_lc2018_source2010_seed0_ckpt50000_transfer_five_scenario_daily.svg |
| 2019 | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/tables/032_08_lc2019_five_scenario_daily.csv | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/tables/032_08_lc2019_five_scenario_summary.csv | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/figures/032_08_lc2019_source2010_seed0_ckpt50000_transfer_five_scenario_daily.png;benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/figures/032_08_lc2019_source2010_seed0_ckpt50000_transfer_five_scenario_daily.svg |
| 2020 | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/tables/032_08_lc2020_five_scenario_daily.csv | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/tables/032_08_lc2020_five_scenario_summary.csv | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/figures/032_08_lc2020_source2010_seed0_ckpt50000_transfer_five_scenario_daily.png;benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/figures/032_08_lc2020_source2010_seed0_ckpt50000_transfer_five_scenario_daily.svg |
| 2021 | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/tables/032_08_lc2021_five_scenario_daily.csv | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/tables/032_08_lc2021_five_scenario_summary.csv | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/figures/032_08_lc2021_source2010_seed0_ckpt50000_transfer_five_scenario_daily.png;benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/figures/032_08_lc2021_source2010_seed0_ckpt50000_transfer_five_scenario_daily.svg |
| 2022 | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/tables/032_08_lc2022_five_scenario_daily.csv | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/tables/032_08_lc2022_five_scenario_summary.csv | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/figures/032_08_lc2022_source2010_seed0_ckpt50000_transfer_five_scenario_daily.png;benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/figures/032_08_lc2022_source2010_seed0_ckpt50000_transfer_five_scenario_daily.svg |
| 2023 | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/tables/032_08_lc2023_five_scenario_daily.csv | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/tables/032_08_lc2023_five_scenario_summary.csv | benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/figures/032_08_lc2023_source2010_seed0_ckpt50000_transfer_five_scenario_daily.png;benchmark_results/032_08_lc2010_seed0_50k_remaining_year_transfer/figures/032_08_lc2023_source2010_seed0_ckpt50000_transfer_five_scenario_daily.svg |

## Interpretation boundary

- This is a frozen-policy transfer diagnostic, not a new training result.
- Any favorable or unfavorable year should be interpreted as LC2010-policy transfer behavior.
- Soil water SWTD is not available in the completed generated baseline daily files used here; the soil-water panel is therefore marked unavailable rather than fabricated.
