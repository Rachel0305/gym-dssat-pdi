# 047_00 SYA originIC 扩展动作空间 MaskablePPO 记录

## 设计

- 参考实验：`046_10_sya_originIC_expanded_action_maskableppo`。
- 唯一实验变量：动作档位，灌溉 `[0.0, 15.0, 30.0, 45.0]` mm，施氮 `[0.0, 10.0, 20.0, 40.0]` kg/ha，共 16 个组合。
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
- `not_all_positive_actions_at_dap1`: `False`
- `multiple_nonzero_action_pairs`: `False`
- `new_action_levels_used`: `True`
- `next_step_allowed`: `False`

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
- `smoke_teacher_guidance_disabled`: `True`
- `smoke_output_root`: `benchmark_results/047_00_sya_originIC_small_n_action_maskableppo_smoke2k`
- `next_step_allowed`: `True`

## 按年动作审计

|   checkpoint_step |   year | daily_exists   |   row_count |   off_grid_irrigation_rows |   off_grid_nitrogen_rows |   positive_action_rows |   positive_action_rows_after_dap1 |   transmission_mismatch_rows |   novel_level_event_rows |   unique_nonzero_action_pairs | nonzero_action_pairs   |
|------------------:|-------:|:---------------|------------:|---------------------------:|-------------------------:|-----------------------:|----------------------------------:|-----------------------------:|-------------------------:|------------------------------:|:-----------------------|
|             25000 |   2014 | True           |         144 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|             25000 |   2015 | True           |         141 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|             25000 |   2016 | True           |         141 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|             25000 |   2017 | True           |         130 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|             25000 |   2018 | True           |         129 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|             25000 |   2019 | True           |         140 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|             25000 |   2020 | True           |         138 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|             25000 |   2021 | True           |         138 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|             25000 |   2022 | True           |         142 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|             25000 |   2023 | True           |         138 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|             50000 |   2014 | True           |         144 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|             50000 |   2015 | True           |         141 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|             50000 |   2016 | True           |         141 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|             50000 |   2017 | True           |         130 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|             50000 |   2018 | True           |         129 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|             50000 |   2019 | True           |         140 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|             50000 |   2020 | True           |         138 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|             50000 |   2021 | True           |         138 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|             50000 |   2022 | True           |         142 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|             50000 |   2023 | True           |         138 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|             75000 |   2014 | True           |         144 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|             75000 |   2015 | True           |         141 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|             75000 |   2016 | True           |         141 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|             75000 |   2017 | True           |         130 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|             75000 |   2018 | True           |         129 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|             75000 |   2019 | True           |         140 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|             75000 |   2020 | True           |         138 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|             75000 |   2021 | True           |         138 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|             75000 |   2022 | True           |         142 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|             75000 |   2023 | True           |         138 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|            100000 |   2014 | True           |         144 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|            100000 |   2015 | True           |         141 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|            100000 |   2016 | True           |         141 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|            100000 |   2017 | True           |         130 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|            100000 |   2018 | True           |         129 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|            100000 |   2019 | True           |         140 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|            100000 |   2020 | True           |         138 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|            100000 |   2021 | True           |         138 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|            100000 |   2022 | True           |         142 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |
|            100000 |   2023 | True           |         138 |                          0 |                        0 |                      1 |                                 0 |                            0 |                        1 |                             1 | I45/N10                |

## 规范化产物

- `evaluation/047_00_training_checkpoint_inventory.csv` <- `evaluation/042_10_training_checkpoint_inventory.csv`
- `evaluation/047_00_checkpoint_validation_summary.csv` <- `evaluation/042_10_checkpoint_validation_summary.csv`
- `evaluation/047_00_validation_summary_by_station_checkpoint.csv` <- `evaluation/042_10_validation_summary_by_station_checkpoint.csv`
- `047_00_engine_result.json` <- `042_10_result.json`

## 结果统计与解释（2026-08-06 追加）

### 核心结论

`047_00` 没有改善 `046_10` 的 PPO 策略。把施氮动作空间从 `[0, 40, 80, 120]` 改成 `[0, 10, 20, 40]` 后，正式 100K 训练在所有 checkpoint 上都塌缩成同一个固定动作：

```text
DAP1 I45/N10
```

也就是只在 DAP1 灌溉 45 mm、施氮 10 kg/ha，之后全季不再响应水分或氮胁迫。因此 `047_00` 不是一个成功候选策略，也不应继续通过增加训练步数来挽救。

### 047_00 checkpoint 均值

| checkpoint | 平均产量 kg/ha | 总灌溉 mm | 总施氮 kg/ha | PFP_N kg/kg | reward | 灌溉次数 | 施氮次数 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 25K | 6697.28 | 45.0 | 10.0 | 669.73 | 0.91 | 1.0 | 1.0 |
| 50K | 6697.28 | 45.0 | 10.0 | 669.73 | 0.91 | 1.0 | 1.0 |
| 75K | 6697.28 | 45.0 | 10.0 | 669.73 | 0.91 | 1.0 | 1.0 |
| 100K | 6697.28 | 45.0 | 10.0 | 669.73 | 0.91 | 1.0 | 1.0 |

### 047_00 100K 相对 046_10 100K

| 指标 | 047_00 - 046_10 |
|---|---:|
| 平均产量 | -3375.29 kg/ha |
| 总灌溉 | -195.0 mm |
| 总施氮 | -226.0 kg/ha |
| PFP_N | +627.08 kg/kg |
| simple profit | -2803.71 |
| reward | -0.16 |

这个结果说明，小氮档位本身并没有让 PPO 学会更好的动态追肥，而是把 PPO 从 `046_10` 的“高投入保险策略”推向了另一个极端：极低投入、低产、全季躺平。

### 与主要情景均值对照

| 情景 | 平均产量 kg/ha | 总灌溉 mm | 总施氮 kg/ha | PFP_N kg/kg | WP_ET kg/m3 |
|---|---:|---:|---:|---:|---:|
| null | 5697.67 | 0.0 | 0.0 | NA | 1.38 |
| recorded_farmer_template | 7894.86 | 0.0 | 293.0 | 26.93 | 1.89 |
| official_extension_expert | 9994.21 | 266.0 | 297.0 | 33.65 | 2.11 |
| dssat_auto_external_n, threshold 0.05 | 10049.15 | 111.5 | 157.5 | 65.22 | 2.16 |
| 046_10 PPO, 100K | 10072.57 | 240.0 | 236.0 | 42.65 | not recomputed here |
| 047_00 PPO, 100K | 6697.28 | 45.0 | 10.0 | 669.73 | not recomputed here |

### PFP_N 解释边界

`047_00` 的 PFP_N 极高不是因为策略优秀，而是因为 PFP_N 的公式通常是：

```text
PFP_N = grain_yield_kg_ha / total_nitrogen_kg_ha
```

当总施氮量非常低时，即使产量很差，PFP_N 也会被分母压得很高。`047_00` 只施 10 kg/ha 氮，因此得到约 669.73 kg/kg 的 PFP_N；这是一种“低投入导致的比值膨胀”，不是高产高效管理。

因此，PFP_N 不能单独作为策略优劣判断指标。后续报告中应增加产量下限或综合效率指标，例如：

```text
yield-constrained PFP_N
只在产量 >= 某个阈值时解释 PFP_N

或

PFP_N 与产量、WP_ET、总灌溉、总施氮并列展示
```

对于 `046_05` 系列中 `0.5-auto` 的高 PFP_N，也应采用同样解释边界：它的 PFP_N 高主要来自低施氮量，而不是说明其综合表现必然优于 PPO 或 expert。正式冻结结果时，不应修改 PFP_N 原始公式来让 PPO 显得更好，而应在图表和文字中明确说明 PFP_N 是部分生产力指标，必须结合产量一起解释。

### 下一步建议

不要继续延长 `047_00` 训练。更合理的下一步是建立 `048`：

```text
small-N action grid + auto-0.05 teacher guidance
```

`048` 的 teacher guidance 应重点防止 `047_00` 暴露出的坏模式：

```text
只在 DAP1 做一次小动作，之后全季不再响应胁迫。
```

也就是说，下一步不只是让 PPO 使用小剂量，而是要教它在后期根据氮胁迫进行动态响应。
