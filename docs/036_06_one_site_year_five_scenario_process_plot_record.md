# 036_06 单站点五情景过程图记录

## 任务边界

- 不训练。
- 不重跑DSSAT。
- PPO daily来自036_01修复后正式重跑。
- 四情景daily来自031_35/031_36基线daily源，与036_03比较口径一致。
- 累计奖励面板使用统一公式 `ΔGRNWT - 1.1*irrigation - 1.58*nitrogen` 重算，不使用各源原始reward。

## 五情景终值摘要

| station_code   |   year | scenario                  | label                 |   final_grain_kg_ha |   final_biomass_kg_ha |   total_irrigation_mm |   total_nitrogen_kg_ha |   PFP_N_kg_kg |   irrigation_event_count |   nitrogen_event_count |   first_irrigation_dap |   first_n_dap |   max_water_stress_index_wspd |   max_nitrogen_stress_index_nstd |   final_cumulative_common_reward | source_file                                                                                                                                           |
|:---------------|-------:|:--------------------------|:----------------------|--------------------:|----------------------:|----------------------:|-----------------------:|--------------:|-------------------------:|-----------------------:|-----------------------:|--------------:|------------------------------:|---------------------------------:|---------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------|
| LCA            |   2021 | null                      | Null                  |             3216.67 |               5371.28 |                  0    |                    0   |      nan      |                        0 |                      0 |                    nan |           nan |                             0 |                           0.4998 |                          3216.67 | benchmark_results\031_35_missing_four_baseline_completion_for_03134\evaluation\031_35_full_generated_baseline_daily.csv                               |
| LCA            |   2021 | recorded_farmer           | Recorded farmer       |            11041.3  |              19780.5  |                200    |                  300   |       36.8043 |                        3 |                      3 |                      2 |             1 |                             0 |                           0.0122 |                         10347.3  | benchmark_results\031_35_missing_four_baseline_completion_for_03134\evaluation\031_35_full_generated_baseline_daily.csv                               |
| LCA            |   2021 | dssat_auto                | DSSAT auto            |             3216.67 |               5371.28 |                  0    |                    0   |      nan      |                        0 |                      0 |                    nan |           nan |                             0 |                           0.4998 |                          3216.67 | benchmark_results\031_36_missing_dssat_auto_completion_for_03134\evaluation\031_36_full_generated_dssat_auto_daily.csv                                |
| LCA            |   2021 | official_extension_expert | Official expert       |            11254.5  |              20034.5  |                198.75 |                  247.5 |       45.4728 |                        5 |                      5 |                      7 |             7 |                             0 |                           0.0122 |                         10644.8  | benchmark_results\031_35_missing_four_baseline_completion_for_03134\evaluation\031_35_full_generated_baseline_daily.csv                               |
| LCA            |   2021 | ppo_candidate             | MaskablePPO candidate |            11258.1  |              20103.8  |                 45    |                   40   |      281.452  |                        1 |                      1 |                      1 |             1 |                             0 |                           0.0122 |                         11145.4  | benchmark_results\036_01_original_free_timing_maskableppo_ic1_linked_five_site_half_split_rerun\daily_outputs\LCA\LCA_2021_seed0_ckpt100000_daily.csv |

## 输出

- `benchmark_results\036_06_one_site_year_five_scenario_process_plot\figures\036_06_lca2021_five_scenario_process.png`
- `benchmark_results\036_06_one_site_year_five_scenario_process_plot\figures\036_06_lca2021_five_scenario_process.svg`
- `benchmark_results\036_06_one_site_year_five_scenario_process_plot\tables\036_06_lca2021_five_scenario_daily.csv`
- `benchmark_results\036_06_one_site_year_five_scenario_process_plot\tables\036_06_lca2021_five_scenario_summary.csv`