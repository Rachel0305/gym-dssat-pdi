# 048_01 SYA originIC 扩展动作空间 MaskablePPO 记录

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
- `smoke_action_safety_override_match`: `True`
- `smoke_output_root`: `benchmark_results/048_01_sya_originIC_auto005_teacher_guided_no_late_n_ban_maskableppo_smoke2k`
- `next_step_allowed`: `True`

## 按年动作审计

|   checkpoint_step |   year | daily_exists   |   row_count |   off_grid_irrigation_rows |   off_grid_nitrogen_rows |   positive_action_rows |   positive_action_rows_after_dap1 |   positive_n_rows_after_dap90 |   nitrogen_kg_ha_after_dap90 |   transmission_mismatch_rows |   novel_level_event_rows |   unique_nonzero_action_pairs | nonzero_action_pairs                             |
|------------------:|-------:|:---------------|------------:|---------------------------:|-------------------------:|-----------------------:|----------------------------------:|------------------------------:|-----------------------------:|-----------------------------:|-------------------------:|------------------------------:|:-------------------------------------------------|
|             25000 |   2014 | True           |         144 |                          0 |                        0 |                     14 |                                13 |                             0 |                            0 |                            0 |                       13 |                             6 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N0; I45/N40 |
|             25000 |   2015 | True           |         141 |                          0 |                        0 |                     14 |                                13 |                             0 |                            0 |                            0 |                       13 |                             6 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N0; I45/N40 |
|             25000 |   2016 | True           |         141 |                          0 |                        0 |                     15 |                                14 |                             0 |                            0 |                            0 |                       14 |                             5 | I0/N40; I15/N0; I30/N0; I45/N0; I45/N40          |
|             25000 |   2017 | True           |         130 |                          0 |                        0 |                     15 |                                14 |                             0 |                            0 |                            0 |                       14 |                             5 | I0/N40; I15/N0; I30/N0; I45/N0; I45/N40          |
|             25000 |   2018 | True           |         129 |                          0 |                        0 |                     14 |                                13 |                             0 |                            0 |                            0 |                       13 |                             6 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N0; I45/N40 |
|             25000 |   2019 | True           |         140 |                          0 |                        0 |                     15 |                                14 |                             0 |                            0 |                            0 |                       14 |                             5 | I0/N40; I15/N0; I30/N0; I45/N0; I45/N40          |
|             25000 |   2020 | True           |         138 |                          0 |                        0 |                     14 |                                13 |                             0 |                            0 |                            0 |                       13 |                             6 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N0; I45/N40 |
|             25000 |   2021 | True           |         138 |                          0 |                        0 |                     15 |                                14 |                             0 |                            0 |                            0 |                       14 |                             5 | I0/N40; I15/N0; I30/N0; I45/N0; I45/N40          |
|             25000 |   2022 | True           |         142 |                          0 |                        0 |                     15 |                                14 |                             0 |                            0 |                            0 |                       14 |                             5 | I0/N40; I15/N0; I30/N0; I45/N0; I45/N40          |
|             25000 |   2023 | True           |         138 |                          0 |                        0 |                     14 |                                13 |                             0 |                            0 |                            0 |                       13 |                             6 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N0; I45/N40 |
|             50000 |   2014 | True           |         144 |                          0 |                        0 |                     15 |                                14 |                             0 |                            0 |                            0 |                       15 |                             5 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N40         |
|             50000 |   2015 | True           |         141 |                          0 |                        0 |                     15 |                                14 |                             0 |                            0 |                            0 |                       15 |                             5 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N40         |
|             50000 |   2016 | True           |         141 |                          0 |                        0 |                     15 |                                14 |                             0 |                            0 |                            0 |                       15 |                             5 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N40         |
|             50000 |   2017 | True           |         130 |                          0 |                        0 |                     15 |                                14 |                             0 |                            0 |                            0 |                       15 |                             5 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N40         |
|             50000 |   2018 | True           |         129 |                          0 |                        0 |                     15 |                                14 |                             0 |                            0 |                            0 |                       15 |                             5 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N40         |
|             50000 |   2019 | True           |         140 |                          0 |                        0 |                     15 |                                14 |                             0 |                            0 |                            0 |                       15 |                             5 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N40         |
|             50000 |   2020 | True           |         138 |                          0 |                        0 |                     15 |                                14 |                             0 |                            0 |                            0 |                       15 |                             5 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N40         |
|             50000 |   2021 | True           |         138 |                          0 |                        0 |                     15 |                                14 |                             0 |                            0 |                            0 |                       15 |                             5 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N40         |
|             50000 |   2022 | True           |         142 |                          0 |                        0 |                     15 |                                14 |                             0 |                            0 |                            0 |                       15 |                             5 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N40         |
|             50000 |   2023 | True           |         138 |                          0 |                        0 |                     15 |                                14 |                             0 |                            0 |                            0 |                       15 |                             5 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N40         |
|             75000 |   2014 | True           |         144 |                          0 |                        0 |                     15 |                                14 |                             0 |                            0 |                            0 |                       15 |                             5 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N40         |
|             75000 |   2015 | True           |         141 |                          0 |                        0 |                     15 |                                14 |                             0 |                            0 |                            0 |                       15 |                             5 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N40         |
|             75000 |   2016 | True           |         141 |                          0 |                        0 |                     15 |                                14 |                             0 |                            0 |                            0 |                       15 |                             5 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N40         |
|             75000 |   2017 | True           |         130 |                          0 |                        0 |                     15 |                                14 |                             0 |                            0 |                            0 |                       15 |                             5 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N40         |
|             75000 |   2018 | True           |         129 |                          0 |                        0 |                     15 |                                14 |                             0 |                            0 |                            0 |                       15 |                             5 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N40         |
|             75000 |   2019 | True           |         140 |                          0 |                        0 |                     15 |                                14 |                             0 |                            0 |                            0 |                       15 |                             5 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N40         |
|             75000 |   2020 | True           |         138 |                          0 |                        0 |                     15 |                                14 |                             0 |                            0 |                            0 |                       15 |                             5 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N40         |
|             75000 |   2021 | True           |         138 |                          0 |                        0 |                     15 |                                14 |                             0 |                            0 |                            0 |                       15 |                             5 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N40         |
|             75000 |   2022 | True           |         142 |                          0 |                        0 |                     15 |                                14 |                             0 |                            0 |                            0 |                       15 |                             5 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N40         |
|             75000 |   2023 | True           |         138 |                          0 |                        0 |                     15 |                                14 |                             0 |                            0 |                            0 |                       15 |                             5 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N40         |
|            100000 |   2014 | True           |         144 |                          0 |                        0 |                     13 |                                12 |                             0 |                            0 |                            0 |                       12 |                             6 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N0; I45/N40 |
|            100000 |   2015 | True           |         141 |                          0 |                        0 |                     13 |                                12 |                             0 |                            0 |                            0 |                       12 |                             6 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N0; I45/N40 |
|            100000 |   2016 | True           |         141 |                          0 |                        0 |                     13 |                                12 |                             0 |                            0 |                            0 |                       12 |                             6 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N0; I45/N40 |
|            100000 |   2017 | True           |         130 |                          0 |                        0 |                     13 |                                12 |                             0 |                            0 |                            0 |                       12 |                             6 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N0; I45/N40 |
|            100000 |   2018 | True           |         129 |                          0 |                        0 |                     13 |                                12 |                             0 |                            0 |                            0 |                       12 |                             6 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N0; I45/N40 |
|            100000 |   2019 | True           |         140 |                          0 |                        0 |                     13 |                                12 |                             0 |                            0 |                            0 |                       12 |                             6 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N0; I45/N40 |
|            100000 |   2020 | True           |         138 |                          0 |                        0 |                     13 |                                12 |                             0 |                            0 |                            0 |                       12 |                             6 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N0; I45/N40 |
|            100000 |   2021 | True           |         138 |                          0 |                        0 |                     13 |                                12 |                             0 |                            0 |                            0 |                       12 |                             6 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N0; I45/N40 |
|            100000 |   2022 | True           |         142 |                          0 |                        0 |                     13 |                                12 |                             0 |                            0 |                            0 |                       12 |                             6 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N0; I45/N40 |
|            100000 |   2023 | True           |         138 |                          0 |                        0 |                     13 |                                12 |                             0 |                            0 |                            0 |                       12 |                             6 | I0/N40; I15/N0; I30/N0; I30/N40; I45/N0; I45/N40 |

## 规范化产物

- `evaluation/048_01_training_checkpoint_inventory.csv` <- `evaluation/042_10_training_checkpoint_inventory.csv`
- `evaluation/048_01_checkpoint_validation_summary.csv` <- `evaluation/042_10_checkpoint_validation_summary.csv`
- `evaluation/048_01_validation_summary_by_station_checkpoint.csv` <- `evaluation/042_10_validation_summary_by_station_checkpoint.csv`
- `048_01_engine_result.json` <- `042_10_result.json`
## 结果分析（2026-08-06）

### 实验是否按预注册条件执行

- 2K smoke 与 100K formal 均完成，formal action gate 全部通过：10/10 验证年日文件完整、动作均在声明网格上、动作均被 DSSAT 传递、且并非只在 DAP1 动作。
- 配置与 formal manifest 均记录 `fertilization_allowed_dap_range=[1,150]`。因此，DAP90 后禁氮确实被撤销；本实验不是因为旧的 DAP90 掩码仍在生效而没有晚期施氮。
- 然而 25K、50K、75K、100K 四个 checkpoint 的 40 个验证年-模型组合中，`positive_n_rows_after_dap90=0`、`nitrogen_kg_ha_after_dap90=0`。策略没有使用新增的晚期施氮机会。

### Checkpoint 均值（2014-2023，单 seed）

| checkpoint | 产量 kg/ha | 灌溉 mm | 施氮 kg/ha | PFP_N kg/kg | reward | 平均施氮事件数 |
|---:|---:|---:|---:|---:|---:|---:|
| 25K | 10211.55 | 240.0 | 240.0 | 42.55 | 0.057 | 6.0 |
| 50K | 10027.23 | 240.0 | 240.0 | 41.78 | 0.028 | 6.0 |
| 75K | 10027.23 | 240.0 | 240.0 | 41.78 | 0.028 | 6.0 |
| 100K | 10035.89 | 240.0 | 240.0 | 41.82 | 0.054 | 6.0 |

100K 的逐年产量范围为 7795.48--11175.06 kg/ha。25K 的平均产量最高，但四个 checkpoint 都打满 240 mm 灌溉和 240 kg/ha 施氮，不能作为“接近 auto-0.05 的低投入、胁迫响应策略”的证据。

### 100K 与先前 PPO 的对照

| 策略 | 产量 kg/ha | 灌溉 mm | 施氮 kg/ha | PFP_N kg/kg | reward |
|---|---:|---:|---:|---:|---:|
| 046_10（无 teacher） | 10072.57 | 240.0 | 236.0 | 42.65 | 1.069 |
| 048_00（teacher，仍禁晚氮） | 9949.81 | 240.0 | 240.0 | 41.46 | 0.128 |
| 048_01（teacher，允许至 DAP150） | 10035.89 | 240.0 | 240.0 | 41.82 | 0.054 |

相对 046_10，048_01 在 100K 的产量为 -36.68 kg/ha、施氮为 +4.0 kg/ha、PFP_N 为 -0.83 kg/kg，且 reward 为 -1.015。相对 048_00，048_01 恢复了约 86.08 kg/ha 产量，但水氮投入没有下降，也没有出现晚期施氮；这不足以证明撤销 DAP90 禁氮带来了有效 teacher 学习。

### 动作与 teacher 诊断

- 100K 的十个验证年均在 DAP1、8、15、22、29、36 各施 N40，至 DAP36 已累计 240 kg/ha；之后没有剩余氮预算可用于 NSTRES 驱动的晚期响应。
- 100K 每年的代表性水分动作仍延续到 DAP91、98、105，但这些均为灌溉而非施氮。
- 100K 平均 teacher shaping 为 -1.004/年，而最终总 reward 均值仍只有 0.054，说明现有局部 teacher 奖惩没有改变“早期打满氮上限”的吸引力，反而与基础 reward 链形成了明显冲突。

### 结论与停止规则

048_01 是一个有效的负结果：撤销 DAP90 后禁氮本身没有使 PPO 学到 auto-0.05 的晚期、胁迫响应施氮。当前瓶颈不再是晚期氮的可行动性，而是策略在早期耗尽氮预算，以及 teacher shaping 与基础回报/预算机制不一致。

因此不应继续在 SYA 上仅靠增加训练步数、继续放宽时间窗口或微调该 teacher 系数。保留 046_05 的 auto-0.5 五情景报告包作为 SYA 阶段性结果；048_00 与 048_01 作为已记录的 teacher-guidance 负对照。后续若重启 SYA，必须另行预注册“显式早期氮预算保留或重新设计的多目标 reward”实验，而不能将其与本 048 系列混合比较。
