# 040_42 SYA lowIC 040_40 checkpoint100k 验证年份五情景指标柱状图记录

## 结论先说

- 验证年份：10 年（2014–2023）。
- PPO 至少一项指标达到四情景最高：10 / 10 年。
- 产量超过四情景最高：5 / 10 年。
- WP_ET 超过四情景最高：3 / 10 年。
- PFP_N 超过四情景最高：10 / 10 年。
- 本任务不训练；PPO 指标由 040_40 checkpoint100000 动作序列固定重放 DSSAT 后从 Summary.OUT 补齐。

## 指标差距表

| station_code | site | year | ppo_yield | four_max_yield | four_max_yield_scenario | gap_yield_vs_four_max | win_yield_vs_four_max | ppo_wp_et | four_max_wp_et | four_max_wp_et_scenario | gap_wp_et_vs_four_max | win_wp_et_vs_four_max | ppo_pfp_n | four_max_pfp_n | four_max_pfp_n_scenario | gap_pfp_n_vs_four_max | win_pfp_n_vs_four_max | any_metric_win_four | ppo_irrigation | ppo_nitrogen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SYA | SY | 2014 | 10210.0989 | 10840.3467 | official_extension_expert | -630.2478 | False | 2.07 | 2.15 | official_extension_expert | -0.08 | False | 42.5 | 36.5 | official_extension_expert | 6.0 | True | True | 240.0 | 240.0 |
| SYA | SY | 2015 | 11104.5337 | 10857.8674 | official_extension_expert | 246.6663 | True | 2.26 | 2.27 | official_extension_expert | -0.01 | False | 46.3 | 36.6 | official_extension_expert | 9.7 | True | True | 240.0 | 240.0 |
| SYA | SY | 2016 | 7856.7194 | 7871.7786 | official_extension_expert | -15.0592 | False | 1.63 | 1.63 | official_extension_expert | 0.0 | True | 32.7 | 26.5 | official_extension_expert | 6.2 | True | True | 240.0 | 240.0 |
| SYA | SY | 2017 | 9862.9419 | 10896.0376 | official_extension_expert | -1033.0957 | False | 2.33 | 2.45 | official_extension_expert | -0.12 | False | 41.1 | 36.7 | official_extension_expert | 4.4 | True | True | 240.0 | 240.0 |
| SYA | SY | 2018 | 8261.3611 | 8094.176 | official_extension_expert | 167.1851 | True | 1.82 | 1.8 | official_extension_expert | 0.02 | True | 34.4 | 27.3 | official_extension_expert | 7.1 | True | True | 240.0 | 240.0 |
| SYA | SY | 2019 | 10411.5015 | 10364.5337 | official_extension_expert | 46.9678 | True | 2.08 | 2.12 | official_extension_expert | -0.04 | False | 43.4 | 34.9 | official_extension_expert | 8.5 | True | True | 240.0 | 240.0 |
| SYA | SY | 2020 | 9911.2677 | 9870.6927 | official_extension_expert | 40.575 | True | 2.09 | 2.1 | official_extension_expert | -0.01 | False | 49.6 | 33.2 | official_extension_expert | 16.4 | True | True | 240.0 | 200.0 |
| SYA | SY | 2021 | 10021.5253 | 10079.975 | official_extension_expert | -58.4497 | False | 2.18 | 2.14 | official_extension_expert | 0.04 | True | 50.1 | 33.9 | official_extension_expert | 16.2 | True | True | 240.0 | 200.0 |
| SYA | SY | 2022 | 10763.075 | 10922.4487 | official_extension_expert | -159.3738 | False | 2.31 | 2.53 | recorded_farmer_template | -0.22 | False | 44.8 | 36.8 | official_extension_expert | 8.0 | True | True | 240.0 | 240.0 |
| SYA | SY | 2023 | 11006.2439 | 10936.7651 | official_extension_expert | 69.4788 | True | 2.26 | 2.27 | official_extension_expert | -0.01 | False | 45.9 | 36.8 | official_extension_expert | 9.1 | True | True | 240.0 | 240.0 |

## 五情景指标列表

| year | scenario | grain_yield_kg_ha | WP_ET_kg_m3 | PFP_N_kg_kg | actual_irrigation_mm | actual_nitrogen_kg_ha | source_status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2014 | null | 1572.7153 | 0.53 |  | 0.0 | 0.0 | generated_040_21_lowIC_four_baseline |
| 2014 | recorded_farmer_template | 3324.2075 | 1.09 | 5.7 | 0.0 | 586.0 | generated_040_21_lowIC_four_baseline |
| 2014 | dssat_auto | 2820.1126 | 0.63 |  | 230.0 | 0.0 | generated_040_21_lowIC_four_baseline |
| 2014 | official_extension_expert | 10840.3467 | 2.15 | 36.5 | 266.0 | 297.0 | generated_040_21_lowIC_four_baseline |
| 2014 | rl_candidate | 10210.0989 | 2.07 | 42.5 | 240.0 | 240.0 | 040_42_fixed_replay_of_040_40_daily_actions |
| 2015 | null | 1432.6886 | 0.48 |  | 0.0 | 0.0 | generated_040_21_lowIC_four_baseline |
| 2015 | recorded_farmer_template | 4080.3482 | 1.2 | 7.0 | 0.0 | 586.0 | generated_040_21_lowIC_four_baseline |
| 2015 | dssat_auto | 2787.0963 | 0.66 |  | 162.0 | 0.0 | generated_040_21_lowIC_four_baseline |
| 2015 | official_extension_expert | 10857.8674 | 2.27 | 36.6 | 266.0 | 297.0 | generated_040_21_lowIC_four_baseline |
| 2015 | rl_candidate | 11104.5337 | 2.26 | 46.3 | 240.0 | 240.0 | 040_42_fixed_replay_of_040_40_daily_actions |
| 2016 | null | 1402.7704 | 0.4 |  | 0.0 | 0.0 | generated_040_21_lowIC_four_baseline |
| 2016 | recorded_farmer_template | 5535.8435 | 1.37 | 9.4 | 0.0 | 586.0 | generated_040_21_lowIC_four_baseline |
| 2016 | dssat_auto | 2602.3975 | 0.61 |  | 126.0 | 0.0 | generated_040_21_lowIC_four_baseline |
| 2016 | official_extension_expert | 7871.7786 | 1.63 | 26.5 | 266.0 | 297.0 | generated_040_21_lowIC_four_baseline |
| 2016 | rl_candidate | 7856.7194 | 1.63 | 32.7 | 240.0 | 240.0 | 040_42_fixed_replay_of_040_40_daily_actions |
| 2017 | null | 0.0 | 0.0 |  | 0.0 | 0.0 | generated_040_21_lowIC_four_baseline |
| 2017 | recorded_farmer_template | 0.0 | 0.0 |  | 0.0 | 242.0 | generated_040_21_lowIC_four_baseline |
| 2017 | dssat_auto | 2639.7546 | 0.66 |  | 273.0 | 0.0 | generated_040_21_lowIC_four_baseline |
| 2017 | official_extension_expert | 10896.0376 | 2.45 | 36.7 | 266.0 | 297.0 | generated_040_21_lowIC_four_baseline |
| 2017 | rl_candidate | 9862.9419 | 2.33 | 41.1 | 240.0 | 240.0 | 040_42_fixed_replay_of_040_40_daily_actions |
| 2018 | null | 1539.5903 | 0.52 |  | 0.0 | 0.0 | generated_040_21_lowIC_four_baseline |
| 2018 | recorded_farmer_template | 4907.3325 | 1.53 | 8.4 | 0.0 | 586.0 | generated_040_21_lowIC_four_baseline |
| 2018 | dssat_auto | 2407.1362 | 0.62 |  | 128.0 | 0.0 | generated_040_21_lowIC_four_baseline |
| 2018 | official_extension_expert | 8094.176 | 1.8 | 27.3 | 266.0 | 297.0 | generated_040_21_lowIC_four_baseline |
| 2018 | rl_candidate | 8261.3611 | 1.82 | 34.4 | 240.0 | 240.0 | 040_42_fixed_replay_of_040_40_daily_actions |
| 2019 | null | 0.0 | 0.0 |  | 0.0 | 0.0 | generated_040_21_lowIC_four_baseline |
| 2019 | recorded_farmer_template | 0.0 | 0.0 |  | 0.0 | 242.0 | generated_040_21_lowIC_four_baseline |
| 2019 | dssat_auto | 2399.4113 | 0.54 |  | 198.0 | 0.0 | generated_040_21_lowIC_four_baseline |
| 2019 | official_extension_expert | 10364.5337 | 2.12 | 34.9 | 266.0 | 297.0 | generated_040_21_lowIC_four_baseline |
| 2019 | rl_candidate | 10411.5015 | 2.08 | 43.4 | 240.0 | 240.0 | 040_42_fixed_replay_of_040_40_daily_actions |
| 2020 | null | 1286.805 | 0.49 |  | 0.0 | 0.0 | generated_040_21_lowIC_four_baseline |
| 2020 | recorded_farmer_template | 2567.6343 | 0.95 | 4.4 | 0.0 | 586.0 | generated_040_21_lowIC_four_baseline |
| 2020 | dssat_auto | 3246.402 | 0.72 |  | 232.0 | 0.0 | generated_040_21_lowIC_four_baseline |
| 2020 | official_extension_expert | 9870.6927 | 2.1 | 33.2 | 266.0 | 297.0 | generated_040_21_lowIC_four_baseline |
| 2020 | rl_candidate | 9911.2677 | 2.09 | 49.6 | 240.0 | 200.0 | 040_42_fixed_replay_of_040_40_daily_actions |
| 2021 | null | 1849.967 | 0.52 |  | 0.0 | 0.0 | generated_040_21_lowIC_four_baseline |
| 2021 | recorded_farmer_template | 4966.5359 | 1.27 | 8.5 | 0.0 | 586.0 | generated_040_21_lowIC_four_baseline |
| 2021 | dssat_auto | 3409.7958 | 0.78 |  | 133.0 | 0.0 | generated_040_21_lowIC_four_baseline |
| 2021 | official_extension_expert | 10079.975 | 2.14 | 33.9 | 266.0 | 297.0 | generated_040_21_lowIC_four_baseline |
| 2021 | rl_candidate | 10021.5253 | 2.18 | 50.1 | 240.0 | 200.0 | 040_42_fixed_replay_of_040_40_daily_actions |
| 2022 | null | 3773.6255 | 1.01 |  | 0.0 | 0.0 | generated_040_21_lowIC_four_baseline |
| 2022 | recorded_farmer_template | 10528.1946 | 2.53 | 18.0 | 0.0 | 586.0 | generated_040_21_lowIC_four_baseline |
| 2022 | dssat_auto | 4181.409 | 1.0 |  | 62.0 | 0.0 | generated_040_21_lowIC_four_baseline |
| 2022 | official_extension_expert | 10922.4487 | 2.36 | 36.8 | 266.0 | 297.0 | generated_040_21_lowIC_four_baseline |
| 2022 | rl_candidate | 10763.075 | 2.31 | 44.8 | 240.0 | 240.0 | 040_42_fixed_replay_of_040_40_daily_actions |
| 2023 | null | 2292.8418 | 0.82 |  | 0.0 | 0.0 | generated_040_21_lowIC_four_baseline |
| 2023 | recorded_farmer_template | 5407.3499 | 1.69 | 9.2 | 0.0 | 586.0 | generated_040_21_lowIC_four_baseline |
| 2023 | dssat_auto | 2624.4464 | 0.58 |  | 198.0 | 0.0 | generated_040_21_lowIC_four_baseline |
| 2023 | official_extension_expert | 10936.7651 | 2.27 | 36.8 | 266.0 | 297.0 | generated_040_21_lowIC_four_baseline |
| 2023 | rl_candidate | 11006.2439 | 2.26 | 45.9 | 240.0 | 240.0 | 040_42_fixed_replay_of_040_40_daily_actions |

## 输出图

- `benchmark_results/040_42_sya_lowIC_04040_ckpt100k_validation_five_scenario_metric_bars/figures/040_42_sya_lowIC_04040_ckpt100k_validation_five_scenario_metrics.png`
- `benchmark_results/040_42_sya_lowIC_04040_ckpt100k_validation_five_scenario_metric_bars/figures/040_42_sya_lowIC_04040_ckpt100k_validation_five_scenario_resources.png`
