# 046_10 SYA originIC 扩展动作空间 MaskablePPO 记录

## 设计

- 参考实验：`046_02_sya_originIC_binary_timing_ppo`。
- 唯一实验变量：动作档位，灌溉 `[0.0, 15.0, 30.0, 45.0]` mm，施氮 `[0.0, 40.0, 80.0, 120.0]` kg/ha，共 16 个组合。
- 未改变：originIC 输入、原始 observation、无天气预报、无归一化、reward、安全约束、训练/验证年份和随机种子。
- 训练：`2000` steps；checkpoints：`[1000, 2000]`。

## Smoke 验收

- `final_checkpoint`: `2000`
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

## 按年动作审计

|   checkpoint_step |   year | daily_exists   |   row_count |   off_grid_irrigation_rows |   off_grid_nitrogen_rows |   positive_action_rows |   positive_action_rows_after_dap1 |   transmission_mismatch_rows |   novel_level_event_rows |   unique_nonzero_action_pairs | nonzero_action_pairs                             |
|------------------:|-------:|:---------------|------------:|---------------------------:|-------------------------:|-----------------------:|----------------------------------:|-----------------------------:|-------------------------:|------------------------------:|:-------------------------------------------------|
|              1000 |   2014 | True           |         144 |                          0 |                        0 |                     11 |                                10 |                            0 |                        9 |                             4 | I15/N0; I15/N120; I30/N0; I45/N0                 |
|              1000 |   2015 | True           |         141 |                          0 |                        0 |                     12 |                                11 |                            0 |                       10 |                             3 | I15/N0; I15/N120; I45/N0                         |
|              1000 |   2016 | True           |         141 |                          0 |                        0 |                     12 |                                11 |                            0 |                       10 |                             3 | I15/N0; I15/N120; I45/N0                         |
|              1000 |   2017 | True           |         130 |                          0 |                        0 |                     12 |                                11 |                            0 |                       10 |                             3 | I15/N0; I15/N120; I45/N0                         |
|              1000 |   2018 | True           |         129 |                          0 |                        0 |                     12 |                                11 |                            0 |                       10 |                             3 | I15/N0; I15/N120; I45/N0                         |
|              1000 |   2019 | True           |         140 |                          0 |                        0 |                     12 |                                11 |                            0 |                       10 |                             3 | I15/N0; I15/N120; I45/N0                         |
|              1000 |   2020 | True           |         138 |                          0 |                        0 |                     12 |                                11 |                            0 |                       10 |                             3 | I15/N0; I15/N120; I45/N0                         |
|              1000 |   2021 | True           |         138 |                          0 |                        0 |                     12 |                                11 |                            0 |                       10 |                             3 | I15/N0; I15/N120; I45/N0                         |
|              1000 |   2022 | True           |         142 |                          0 |                        0 |                     12 |                                11 |                            0 |                       10 |                             3 | I15/N0; I15/N120; I45/N0                         |
|              1000 |   2023 | True           |         138 |                          0 |                        0 |                     12 |                                11 |                            0 |                       10 |                             3 | I15/N0; I15/N120; I45/N0                         |
|              2000 |   2014 | True           |         144 |                          0 |                        0 |                     12 |                                11 |                            0 |                        9 |                             6 | I0/N80; I15/N0; I30/N0; I30/N40; I45/N0; I45/N40 |
|              2000 |   2015 | True           |         141 |                          0 |                        0 |                     12 |                                11 |                            0 |                        9 |                             6 | I0/N80; I15/N0; I30/N0; I30/N40; I45/N0; I45/N40 |
|              2000 |   2016 | True           |         141 |                          0 |                        0 |                     12 |                                11 |                            0 |                        9 |                             6 | I0/N40; I0/N80; I15/N0; I30/N0; I45/N0; I45/N40  |
|              2000 |   2017 | True           |         130 |                          0 |                        0 |                     13 |                                12 |                            0 |                       10 |                             6 | I0/N40; I0/N80; I15/N0; I30/N0; I45/N0; I45/N40  |
|              2000 |   2018 | True           |         129 |                          0 |                        0 |                     12 |                                11 |                            0 |                        9 |                             6 | I0/N80; I15/N0; I30/N0; I30/N40; I45/N0; I45/N40 |
|              2000 |   2019 | True           |         140 |                          0 |                        0 |                     12 |                                11 |                            0 |                        9 |                             6 | I0/N80; I15/N0; I30/N0; I30/N40; I45/N0; I45/N40 |
|              2000 |   2020 | True           |         138 |                          0 |                        0 |                     11 |                                10 |                            0 |                        8 |                             6 | I0/N80; I15/N0; I30/N0; I30/N40; I45/N0; I45/N40 |
|              2000 |   2021 | True           |         138 |                          0 |                        0 |                     11 |                                10 |                            0 |                        8 |                             6 | I0/N80; I15/N0; I30/N0; I30/N40; I45/N0; I45/N40 |
|              2000 |   2022 | True           |         142 |                          0 |                        0 |                     11 |                                10 |                            0 |                        8 |                             6 | I0/N80; I15/N0; I30/N0; I30/N40; I45/N0; I45/N40 |
|              2000 |   2023 | True           |         138 |                          0 |                        0 |                     12 |                                11 |                            0 |                        9 |                             6 | I0/N80; I15/N0; I30/N0; I30/N40; I45/N0; I45/N40 |

## 规范化产物

- `evaluation/046_10_training_checkpoint_inventory.csv` <- `evaluation/042_10_training_checkpoint_inventory.csv`
- `evaluation/046_10_checkpoint_validation_summary.csv` <- `evaluation/042_10_checkpoint_validation_summary.csv`
- `evaluation/046_10_validation_summary_by_station_checkpoint.csv` <- `evaluation/042_10_validation_summary_by_station_checkpoint.csv`
- `046_10_engine_result.json` <- `042_10_result.json`
