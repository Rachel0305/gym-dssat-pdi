# 048_00 SYA originIC 扩展动作空间 MaskablePPO 记录

## 设计

- 参考实验：`046_10_sya_originIC_expanded_action_maskableppo`。
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
- `smoke_teacher_guidance_enabled`: `True`
- `smoke_output_root`: `benchmark_results/048_00_sya_originIC_auto005_teacher_guided_maskableppo_smoke2k`
- `next_step_allowed`: `True`

## 按年动作审计

|   checkpoint_step |   year | daily_exists   |   row_count |   off_grid_irrigation_rows |   off_grid_nitrogen_rows |   positive_action_rows |   positive_action_rows_after_dap1 |   transmission_mismatch_rows |   novel_level_event_rows |   unique_nonzero_action_pairs | nonzero_action_pairs                       |
|------------------:|-------:|:---------------|------------:|---------------------------:|-------------------------:|-----------------------:|----------------------------------:|-----------------------------:|-------------------------:|------------------------------:|:-------------------------------------------|
|             25000 |   2014 | True           |         144 |                          0 |                        0 |                      3 |                                 2 |                            0 |                        3 |                             3 | I0/N40; I30/N80; I45/N120                  |
|             25000 |   2015 | True           |         141 |                          0 |                        0 |                      3 |                                 2 |                            0 |                        3 |                             3 | I0/N40; I30/N80; I45/N120                  |
|             25000 |   2016 | True           |         141 |                          0 |                        0 |                      3 |                                 2 |                            0 |                        3 |                             3 | I0/N40; I30/N80; I45/N120                  |
|             25000 |   2017 | True           |         130 |                          0 |                        0 |                      3 |                                 2 |                            0 |                        3 |                             3 | I0/N40; I30/N80; I45/N120                  |
|             25000 |   2018 | True           |         129 |                          0 |                        0 |                      3 |                                 2 |                            0 |                        3 |                             3 | I0/N40; I30/N80; I45/N120                  |
|             25000 |   2019 | True           |         140 |                          0 |                        0 |                      3 |                                 2 |                            0 |                        3 |                             3 | I0/N40; I30/N80; I45/N120                  |
|             25000 |   2020 | True           |         138 |                          0 |                        0 |                      3 |                                 2 |                            0 |                        3 |                             3 | I0/N40; I30/N80; I45/N120                  |
|             25000 |   2021 | True           |         138 |                          0 |                        0 |                      3 |                                 2 |                            0 |                        3 |                             3 | I0/N40; I30/N80; I45/N120                  |
|             25000 |   2022 | True           |         142 |                          0 |                        0 |                      3 |                                 2 |                            0 |                        3 |                             3 | I0/N40; I30/N80; I45/N120                  |
|             25000 |   2023 | True           |         138 |                          0 |                        0 |                      3 |                                 2 |                            0 |                        3 |                             3 | I0/N40; I30/N80; I45/N120                  |
|             50000 |   2014 | True           |         144 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        8 |                             5 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N120  |
|             50000 |   2015 | True           |         141 |                          0 |                        0 |                      6 |                                 5 |                            0 |                        6 |                             5 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N120  |
|             50000 |   2016 | True           |         141 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        8 |                             5 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N120  |
|             50000 |   2017 | True           |         130 |                          0 |                        0 |                      6 |                                 5 |                            0 |                        6 |                             5 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N120  |
|             50000 |   2018 | True           |         129 |                          0 |                        0 |                      7 |                                 6 |                            0 |                        7 |                             5 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N120  |
|             50000 |   2019 | True           |         140 |                          0 |                        0 |                      7 |                                 6 |                            0 |                        7 |                             5 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N120  |
|             50000 |   2020 | True           |         138 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        8 |                             5 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N120  |
|             50000 |   2021 | True           |         138 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        8 |                             5 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N120  |
|             50000 |   2022 | True           |         142 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        8 |                             5 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N120  |
|             50000 |   2023 | True           |         138 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        8 |                             5 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N120  |
|             75000 |   2014 | True           |         144 |                          0 |                        0 |                      9 |                                 8 |                            0 |                        9 |                             4 | I15/N0; I30/N0; I30/N40; I45/N120          |
|             75000 |   2015 | True           |         141 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        8 |                             5 | I15/N0; I30/N0; I30/N40; I45/N40; I45/N120 |
|             75000 |   2016 | True           |         141 |                          0 |                        0 |                      9 |                                 8 |                            0 |                        9 |                             4 | I15/N0; I30/N0; I30/N40; I45/N120          |
|             75000 |   2017 | True           |         130 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        8 |                             5 | I15/N0; I30/N0; I30/N40; I45/N40; I45/N120 |
|             75000 |   2018 | True           |         129 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        8 |                             5 | I15/N0; I30/N0; I30/N40; I45/N40; I45/N120 |
|             75000 |   2019 | True           |         140 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        8 |                             5 | I15/N0; I30/N0; I30/N40; I45/N40; I45/N120 |
|             75000 |   2020 | True           |         138 |                          0 |                        0 |                      9 |                                 8 |                            0 |                        9 |                             4 | I15/N0; I30/N0; I30/N40; I45/N120          |
|             75000 |   2021 | True           |         138 |                          0 |                        0 |                      9 |                                 8 |                            0 |                        9 |                             4 | I15/N0; I30/N0; I30/N40; I45/N120          |
|             75000 |   2022 | True           |         142 |                          0 |                        0 |                      8 |                                 7 |                            0 |                        8 |                             5 | I15/N0; I30/N0; I30/N40; I45/N40; I45/N120 |
|             75000 |   2023 | True           |         138 |                          0 |                        0 |                      9 |                                 8 |                            0 |                        9 |                             4 | I15/N0; I30/N0; I30/N40; I45/N120          |
|            100000 |   2014 | True           |         144 |                          0 |                        0 |                     13 |                                12 |                            0 |                       13 |                             4 | I0/N40; I15/N0; I30/N40; I45/N120          |
|            100000 |   2015 | True           |         141 |                          0 |                        0 |                     13 |                                12 |                            0 |                       13 |                             4 | I0/N40; I15/N0; I30/N40; I45/N120          |
|            100000 |   2016 | True           |         141 |                          0 |                        0 |                     13 |                                12 |                            0 |                       13 |                             4 | I0/N40; I15/N0; I30/N40; I45/N120          |
|            100000 |   2017 | True           |         130 |                          0 |                        0 |                     13 |                                12 |                            0 |                       13 |                             4 | I0/N40; I15/N0; I30/N40; I45/N120          |
|            100000 |   2018 | True           |         129 |                          0 |                        0 |                     14 |                                13 |                            0 |                       13 |                             4 | I0/N80; I15/N0; I30/N40; I45/N120          |
|            100000 |   2019 | True           |         140 |                          0 |                        0 |                     13 |                                12 |                            0 |                       13 |                             4 | I0/N40; I15/N0; I30/N40; I45/N120          |
|            100000 |   2020 | True           |         138 |                          0 |                        0 |                     11 |                                10 |                            0 |                       11 |                             3 | I15/N0; I30/N40; I45/N120                  |
|            100000 |   2021 | True           |         138 |                          0 |                        0 |                     13 |                                12 |                            0 |                       13 |                             4 | I0/N40; I15/N0; I30/N40; I45/N120          |
|            100000 |   2022 | True           |         142 |                          0 |                        0 |                     13 |                                12 |                            0 |                       13 |                             4 | I0/N40; I15/N0; I30/N40; I45/N120          |
|            100000 |   2023 | True           |         138 |                          0 |                        0 |                     13 |                                12 |                            0 |                       13 |                             4 | I0/N40; I15/N0; I30/N40; I45/N120          |

## 规范化产物

- `evaluation/048_00_training_checkpoint_inventory.csv` <- `evaluation/042_10_training_checkpoint_inventory.csv`
- `evaluation/048_00_checkpoint_validation_summary.csv` <- `evaluation/042_10_checkpoint_validation_summary.csv`
- `evaluation/048_00_validation_summary_by_station_checkpoint.csv` <- `evaluation/042_10_validation_summary_by_station_checkpoint.csv`
- `048_00_engine_result.json` <- `042_10_result.json`

## 结果统计与解释（2026-08-06 追加）

### 核心结论

`048_00` 的 auto-0.05 teacher guidance 成功避免了 `047_00` 那种 `DAP1 I45/N10` 后全季躺平的失败模式；smoke 和 formal 的动作审计均通过。

但是，`048_00` 并没有真正学到 auto-0.05 的低投入高效率策略。正式 100K 后，PPO 仍然接近 `046_10` 的高投入保险策略，平均总灌溉 240 mm、总施氮 240 kg/ha；产量略低于 `046_10`，PFP_N 也没有改善。

最重要的行为问题是：teacher shaping 虽然进入了 daily 输出，但策略仍然在 DAP1 大量施氮：

```text
典型 100K 动作：
DAP1 I45/N120
DAP8 I30/N40
DAP31 I30/N40
DAP51-59 左右再追 N40 或 N80
之后主要继续灌溉
```

这说明当前 teacher shaping 太弱，或者与继承的 reward / safety / DAP90 后禁氮约束仍然不匹配。`048_00` 不应作为优于 `046_10` 的候选结果。

### 048_00 checkpoint 均值

| checkpoint | 平均产量 kg/ha | 总灌溉 mm | 总施氮 kg/ha | PFP_N kg/kg | reward | 灌溉次数 | 施氮次数 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 25K | 9322.58 | 75.0 | 240.0 | 38.84 | -0.04 | 2.0 | 3.0 |
| 50K | 9956.24 | 153.0 | 240.0 | 41.48 | 0.13 | 5.4 | 4.0 |
| 75K | 10059.92 | 240.0 | 240.0 | 41.92 | 0.08 | 8.5 | 4.0 |
| 100K | 9949.81 | 240.0 | 240.0 | 41.46 | 0.13 | 12.0 | 3.9 |

### 048_00 相对 046_10

| checkpoint | 产量差 kg/ha | 灌溉差 mm | 施氮差 kg/ha | PFP_N 差 kg/kg | reward 差 |
|---:|---:|---:|---:|---:|---:|
| 25K | -688.70 | -165.0 | 0.0 | -2.87 | -1.06 |
| 50K | -83.11 | -87.0 | 0.0 | -0.35 | -0.89 |
| 75K | +82.16 | +91.5 | +80.0 | -20.44 | -1.18 |
| 100K | -122.76 | 0.0 | +4.0 | -1.19 | -0.94 |

100K 时，`048_00` 与 `046_10` 的资源投入几乎一样，但平均产量低约 123 kg/ha，PFP_N 低约 1.19 kg/kg。因此，在当前参数下，teacher guidance 没有带来净改善。

### 048_00 相对 047_00

`048_00` 明显修复了 `047_00` 的低投入低产塌缩：

| checkpoint | 产量差 kg/ha | 灌溉差 mm | 施氮差 kg/ha | PFP_N 差 kg/kg |
|---:|---:|---:|---:|---:|
| 25K | +2625.30 | +30.0 | +230.0 | -630.89 |
| 50K | +3258.96 | +108.0 | +230.0 | -628.25 |
| 75K | +3362.64 | +195.0 | +230.0 | -627.81 |
| 100K | +3252.53 | +195.0 | +230.0 | -628.27 |

这说明 `048_00` 的 teacher/reward 结构没有让 PPO 躺平，但也没有让它节氮。

### 动作审计

`048_00_smoke_result.json`：

```text
next_step_allowed: True
not_all_positive_actions_at_dap1: True
multiple_nonzero_action_pairs: True
new_action_levels_used: True
```

`048_00_formal_result.json`：

```text
next_step_allowed: True
not_all_positive_actions_at_dap1: True
multiple_nonzero_action_pairs: True
new_action_levels_used: True
```

因此，`048_00` 从动作多样性和执行链路上是合格的；失败点不在于动作不可达，而在于学到的管理方向仍然偏高投入。

### Teacher shaping 是否生效

100K daily 输出中存在 teacher 字段：

```text
teacher_reward_unscaled
teacher_reward_scaled
teacher_prev_nstres
reward_after_teacher_guidance
```

各验证年 teacher reward scaled 总和约为：

```text
-0.90 至 -0.98
```

这说明 teacher shaping 确实进入了训练/评估链路，并对早期高氮动作产生了负项；但这个负项没有强到足以改变 PPO 对早期高氮和满灌溉的偏好。

### 与主要基线的解释

均值对照：

| 情景 | 平均产量 kg/ha | 总灌溉 mm | 总施氮 kg/ha | PFP_N kg/kg | WP_ET kg/m3 |
|---|---:|---:|---:|---:|---:|
| auto-0.05 | 10049.15 | 111.5 | 157.5 | 65.22 | 2.16 |
| auto-0.5 | 8052.07 | 112.0 | 47.5 | 176.67 | 1.73 |
| expert | 9994.21 | 266.0 | 297.0 | 33.65 | 2.11 |
| 046_10 PPO | 10072.57 | 240.0 | 236.0 | 42.66 | 2.11 |
| 048_00 PPO | 9949.81 | 240.0 | 240.0 | 41.46 | not recomputed here |

`048_00` 没有接近 auto-0.05 的低投入策略，也没有超过 `046_10`。因此 `048_00` 不能替代 `046_10` 作为 SY 当前阶段性 PPO 结果。

### 解释与下一步

`048_00` 的失败不是完全失败：它证明 teacher wrapper 能进入训练链路，也能避免 `047_00` 的 DAP1 小剂量躺平。但是，当前 teacher shaping 太弱，且继承的安全约束仍然阻止 PPO 在 DAP90 后追氮，而 auto-0.05 的许多施氮事件恰好发生在 DAP90 后。

下一步如继续尝试，应新开 `048_01` 或 `049`，只撤销 DAP90 后禁氮，同时保留：

```text
season N cap
minimum days between fertilization
action mask
teacher guidance
```

实验问题应明确写成：

```text
Does removing only the inherited DAP90 late-N ban allow PPO to learn the auto-0.05 stress-responsive late nitrogen strategy?
```

如果 `048_01` 仍不能明显改善水氮效率，则建议 SY 站点停止继续调参，冻结 `046_05_sya_originIC_046_10_sya_originIC_expanded_action_maskableppo_nstd050_minimal_auto_nstd050_minimal_reporting_ckpt100000` 作为阶段性结果，并转向其他站点。
