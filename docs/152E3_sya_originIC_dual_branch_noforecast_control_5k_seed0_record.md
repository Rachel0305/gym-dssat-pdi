# 152E3 SYA originIC dual-branch 无天气信号对照记录

## 设计

- 保留 E2 dual-branch 结构：25 维基础分支 + 12 维天气分支，96 维 latent。
- 12 个天气分支输入固定为 0，不读取任何天气数据。
- reward、动作网格、安全约束、输入、年份和 PPO 设置保持不变。
- 当前阶段：`100K formal`；输出目录：`benchmark_results/152E3_sya_originIC_dual_branch_noforecast_control_5k_seed0`。

## gate

- `final_checkpoint`: `5000`
- `expected_validation_rows`: `10`
- `observed_validation_rows`: `10`
- `all_daily_files_exist`: `True`
- `all_actions_on_declared_grid`: `True`
- `positive_actions_present`: `True`
- `not_all_positive_actions_at_dap1`: `True`
- `zero_weather_columns_present_and_all_zero`: `True`
- `next_step_allowed`: `True`

## observation / zero-weather audit

|   checkpoint_step |   year | daily_exists   |   row_count |   off_grid_irrigation_rows |   off_grid_nitrogen_rows |   positive_action_rows |   positive_action_rows_after_dap1 |   forecast_required_column_count |   forecast_present_column_count | zero_weather_columns_all_zero   |
|------------------:|-------:|:---------------|------------:|---------------------------:|-------------------------:|-----------------------:|----------------------------------:|---------------------------------:|--------------------------------:|:--------------------------------|
|              2000 |   2014 | True           |         144 |                          0 |                        0 |                      8 |                                 7 |                               24 |                              24 | True                            |
|              2000 |   2015 | True           |         141 |                          0 |                        0 |                      8 |                                 7 |                               24 |                              24 | True                            |
|              2000 |   2016 | True           |         141 |                          0 |                        0 |                      8 |                                 7 |                               24 |                              24 | True                            |
|              2000 |   2017 | True           |         130 |                          0 |                        0 |                      8 |                                 7 |                               24 |                              24 | True                            |
|              2000 |   2018 | True           |         129 |                          0 |                        0 |                      8 |                                 7 |                               24 |                              24 | True                            |
|              2000 |   2019 | True           |         140 |                          0 |                        0 |                      8 |                                 7 |                               24 |                              24 | True                            |
|              2000 |   2020 | True           |         138 |                          0 |                        0 |                      8 |                                 7 |                               24 |                              24 | True                            |
|              2000 |   2021 | True           |         138 |                          0 |                        0 |                      8 |                                 7 |                               24 |                              24 | True                            |
|              2000 |   2022 | True           |         142 |                          0 |                        0 |                      8 |                                 7 |                               24 |                              24 | True                            |
|              2000 |   2023 | True           |         138 |                          0 |                        0 |                      8 |                                 7 |                               24 |                              24 | True                            |
|              5000 |   2014 | True           |         144 |                          0 |                        0 |                     13 |                                12 |                               24 |                              24 | True                            |
|              5000 |   2015 | True           |         141 |                          0 |                        0 |                     12 |                                11 |                               24 |                              24 | True                            |
|              5000 |   2016 | True           |         141 |                          0 |                        0 |                     13 |                                12 |                               24 |                              24 | True                            |
|              5000 |   2017 | True           |         130 |                          0 |                        0 |                     12 |                                11 |                               24 |                              24 | True                            |
|              5000 |   2018 | True           |         129 |                          0 |                        0 |                     12 |                                11 |                               24 |                              24 | True                            |
|              5000 |   2019 | True           |         140 |                          0 |                        0 |                     12 |                                11 |                               24 |                              24 | True                            |
|              5000 |   2020 | True           |         138 |                          0 |                        0 |                     13 |                                12 |                               24 |                              24 | True                            |
|              5000 |   2021 | True           |         138 |                          0 |                        0 |                     13 |                                12 |                               24 |                              24 | True                            |
|              5000 |   2022 | True           |         142 |                          0 |                        0 |                     12 |                                11 |                               24 |                              24 | True                            |
|              5000 |   2023 | True           |         138 |                          0 |                        0 |                     13 |                                12 |                               24 |                              24 | True                            |

## 解释边界

- 这是 E2 的 matched-architecture no-forecast 对照，不是最终 forecast 优势证明。
- 若 E2 相对本对照仍有优势，才说明天气信号可能带来额外价值。

## smoke verification

- `smoke_result_exists`: `True`
- `smoke_manifest_exists`: `True`
- `smoke_status_completed`: `True`
- `smoke_gate_passed`: `True`
- `smoke_timesteps_match`: `True`
- `smoke_checkpoints_match`: `True`
- `smoke_input_profile_match`: `True`
- `smoke_station_match`: `True`
- `smoke_actions_match`: `True`
- `smoke_forecast_features_match`: `True`
- `smoke_output_root`: `benchmark_results/152E3_sya_originIC_dual_branch_noforecast_control_5k_seed0_smoke2k`
- `next_step_allowed`: `True`
