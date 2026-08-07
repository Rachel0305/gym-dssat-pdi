# 046_10 SYA originIC 扩展动作空间 MaskablePPO 记录

## 设计

- 参考实验：`046_02_sya_originIC_binary_timing_ppo`。
- 唯一实验变量：动作档位，灌溉 `[0.0, 15.0, 30.0, 45.0]` mm，施氮 `[0.0, 40.0, 80.0, 120.0]` kg/ha，共 16 个组合。
- 未改变：originIC 输入、原始 observation、无天气预报、无归一化、reward、安全约束、训练/验证年份和随机种子。
- 训练：`100000` steps；checkpoints：`[25000, 50000, 75000, 100000]`。

## 100K 正式训练后动作审计

- `final_checkpoint`: `100000`
- `expected_validation_rows`: `10`
- `observed_validation_rows`: `10`
- `all_daily_files_exist`: `True`
- `all_actions_on_declared_grid`: `True`
- `positive_actions_present`: `True`
- `positive_actions_transmitted`: `True`
- `not_all_positive_actions_at_dap1`: `True`
- `multiple_nonzero_action_pairs`: `True`
- `new_action_levels_used`: `True`
- `next_step_allowed`: `True`

## 正式训练前 smoke 复核

- `smoke_result_exists`: `True`
- `smoke_manifest_exists`: `True`
- `smoke_status_completed`: `True`
- `smoke_gate_passed`: `True`
- `smoke_timesteps_match`: `True`
- `smoke_checkpoints_match`: `True`
- `smoke_input_profile_match`: `True`
- `smoke_actions_match`: `True`
- `smoke_observation_contract_match`: `True`
- `smoke_output_root`: `benchmark_results/046_10_sya_originIC_expanded_action_maskableppo_smoke2k`
- `next_step_allowed`: `True`

## 按年动作审计

|   checkpoint_step |   year | daily_exists   |   row_count |   off_grid_irrigation_rows |   off_grid_nitrogen_rows |   positive_action_rows |   positive_action_rows_after_dap1 |   transmission_mismatch_rows |   novel_level_event_rows |   unique_nonzero_action_pairs | nonzero_action_pairs             |
|------------------:|-------:|:---------------|------------:|---------------------------:|-------------------------:|-----------------------:|----------------------------------:|-----------------------------:|-------------------------:|------------------------------:|:---------------------------------|
|             25000 |   2014 | True           |         144 |                          0 |                        0 |                     13 |                                12 |                            0 |                       12 |                             4 | I15/N0; I15/N80; I30/N80; I45/N0 |
|             25000 |   2015 | True           |         141 |                          0 |                        0 |                     13 |                                12 |                            0 |                       12 |                             4 | I15/N0; I15/N80; I30/N80; I45/N0 |
|             25000 |   2016 | True           |         141 |                          0 |                        0 |                     13 |                                12 |                            0 |                       12 |                             4 | I15/N0; I15/N80; I30/N80; I45/N0 |
|             25000 |   2017 | True           |         130 |                          0 |                        0 |                     15 |                                14 |                            0 |                       15 |                             3 | I15/N0; I15/N80; I30/N80         |
|             25000 |   2018 | True           |         129 |                          0 |                        0 |                     15 |                                14 |                            0 |                       15 |                             3 | I15/N0; I15/N80; I30/N80         |
|             25000 |   2019 | True           |         140 |                          0 |                        0 |                     13 |                                12 |                            0 |                       12 |                             4 | I15/N0; I15/N80; I30/N80; I45/N0 |
|             25000 |   2020 | True           |         138 |                          0 |                        0 |                     13 |                                12 |                            0 |                       12 |                             4 | I15/N0; I15/N80; I30/N80; I45/N0 |
|             25000 |   2021 | True           |         138 |                          0 |                        0 |                     13 |                                12 |                            0 |                       12 |                             4 | I15/N0; I15/N80; I30/N80; I45/N0 |
|             25000 |   2022 | True           |         142 |                          0 |                        0 |                     13 |                                12 |                            0 |                       12 |                             4 | I15/N0; I15/N80; I30/N80; I45/N0 |
|             25000 |   2023 | True           |         138 |                          0 |                        0 |                     13 |                                12 |                            0 |                       12 |                             4 | I15/N0; I15/N80; I30/N80; I45/N0 |
|             50000 |   2014 | True           |         144 |                          0 |                        0 |                     13 |                                12 |                            0 |                       13 |                             4 | I0/N40; I15/N0; I15/N40; I30/N40 |
|             50000 |   2015 | True           |         141 |                          0 |                        0 |                     13 |                                12 |                            0 |                       13 |                             4 | I0/N40; I15/N0; I15/N40; I30/N40 |
|             50000 |   2016 | True           |         141 |                          0 |                        0 |                     12 |                                11 |                            0 |                       12 |                             3 | I15/N0; I15/N40; I30/N40         |
|             50000 |   2017 | True           |         130 |                          0 |                        0 |                     13 |                                12 |                            0 |                       13 |                             4 | I0/N40; I15/N0; I15/N40; I30/N40 |
|             50000 |   2018 | True           |         129 |                          0 |                        0 |                     13 |                                12 |                            0 |                       13 |                             4 | I0/N40; I15/N0; I15/N40; I30/N40 |
|             50000 |   2019 | True           |         140 |                          0 |                        0 |                     13 |                                12 |                            0 |                       13 |                             4 | I0/N40; I15/N0; I15/N40; I30/N40 |
|             50000 |   2020 | True           |         138 |                          0 |                        0 |                     12 |                                11 |                            0 |                       12 |                             3 | I15/N0; I15/N40; I30/N40         |
|             50000 |   2021 | True           |         138 |                          0 |                        0 |                     12 |                                11 |                            0 |                       12 |                             3 | I15/N0; I15/N40; I30/N40         |
|             50000 |   2022 | True           |         142 |                          0 |                        0 |                     13 |                                12 |                            0 |                       13 |                             4 | I0/N40; I15/N0; I15/N40; I30/N40 |
|             50000 |   2023 | True           |         138 |                          0 |                        0 |                     13 |                                12 |                            0 |                       13 |                             4 | I0/N40; I15/N0; I15/N40; I30/N40 |
|             75000 |   2014 | True           |         144 |                          0 |                        0 |                      4 |                                 3 |                            0 |                        4 |                             2 | I30/N40; I45/N40                 |
|             75000 |   2015 | True           |         141 |                          0 |                        0 |                      4 |                                 3 |                            0 |                        4 |                             2 | I30/N40; I45/N40                 |
|             75000 |   2016 | True           |         141 |                          0 |                        0 |                      4 |                                 3 |                            0 |                        4 |                             2 | I30/N40; I45/N40                 |
|             75000 |   2017 | True           |         130 |                          0 |                        0 |                      4 |                                 3 |                            0 |                        4 |                             2 | I30/N40; I45/N40                 |
|             75000 |   2018 | True           |         129 |                          0 |                        0 |                      4 |                                 3 |                            0 |                        4 |                             2 | I30/N40; I45/N40                 |
|             75000 |   2019 | True           |         140 |                          0 |                        0 |                      4 |                                 3 |                            0 |                        4 |                             2 | I30/N40; I45/N40                 |
|             75000 |   2020 | True           |         138 |                          0 |                        0 |                      4 |                                 3 |                            0 |                        4 |                             2 | I30/N40; I45/N40                 |
|             75000 |   2021 | True           |         138 |                          0 |                        0 |                      4 |                                 3 |                            0 |                        4 |                             2 | I30/N40; I45/N40                 |
|             75000 |   2022 | True           |         142 |                          0 |                        0 |                      4 |                                 3 |                            0 |                        4 |                             2 | I30/N40; I45/N40                 |
|             75000 |   2023 | True           |         138 |                          0 |                        0 |                      4 |                                 3 |                            0 |                        4 |                             2 | I30/N40; I45/N40                 |
|            100000 |   2014 | True           |         144 |                          0 |                        0 |                     11 |                                10 |                            0 |                       10 |                             4 | I0/N80; I15/N0; I30/N40; I45/N40 |
|            100000 |   2015 | True           |         141 |                          0 |                        0 |                     11 |                                10 |                            0 |                       10 |                             4 | I0/N80; I15/N0; I30/N40; I45/N40 |
|            100000 |   2016 | True           |         141 |                          0 |                        0 |                     11 |                                10 |                            0 |                       10 |                             4 | I0/N80; I15/N0; I30/N40; I45/N40 |
|            100000 |   2017 | True           |         130 |                          0 |                        0 |                     11 |                                10 |                            0 |                       10 |                             4 | I0/N80; I15/N0; I30/N40; I45/N40 |
|            100000 |   2018 | True           |         129 |                          0 |                        0 |                     13 |                                12 |                            0 |                       12 |                             4 | I0/N80; I15/N0; I30/N40; I45/N40 |
|            100000 |   2019 | True           |         140 |                          0 |                        0 |                     11 |                                10 |                            0 |                       10 |                             4 | I0/N80; I15/N0; I30/N40; I45/N40 |
|            100000 |   2020 | True           |         138 |                          0 |                        0 |                     11 |                                10 |                            0 |                       10 |                             4 | I0/N80; I15/N0; I30/N40; I45/N40 |
|            100000 |   2021 | True           |         138 |                          0 |                        0 |                     11 |                                10 |                            0 |                       10 |                             4 | I0/N80; I15/N0; I30/N40; I45/N40 |
|            100000 |   2022 | True           |         142 |                          0 |                        0 |                     11 |                                10 |                            0 |                       10 |                             4 | I0/N80; I15/N0; I30/N40; I45/N40 |
|            100000 |   2023 | True           |         138 |                          0 |                        0 |                     11 |                                10 |                            0 |                       10 |                             4 | I0/N80; I15/N0; I30/N40; I45/N40 |

## 规范化产物

- `evaluation/046_10_training_checkpoint_inventory.csv` <- `evaluation/042_10_training_checkpoint_inventory.csv`
- `evaluation/046_10_checkpoint_validation_summary.csv` <- `evaluation/042_10_checkpoint_validation_summary.csv`
- `evaluation/046_10_validation_summary_by_station_checkpoint.csv` <- `evaluation/042_10_validation_summary_by_station_checkpoint.csv`
- `046_10_engine_result.json` <- `042_10_result.json`
