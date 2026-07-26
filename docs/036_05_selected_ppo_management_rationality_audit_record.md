# 036_05 PPO措施合理性审计记录

## 任务边界

- 不训练。
- 不修改reward、模型、动作空间或DSSAT输入。
- 只审计036_04选中的每站点代表checkpoint对应的50个验证年。

## 审计规则

- 早期集中投入：DAP1-10 使用水或氮占季节总量 >= 80%。
- 胁迫响应：操作前3天最大 `swfac/nstres` >= 0.05 记为响应性操作。
- 预防性/无胁迫操作：操作前3天最大 `swfac/nstres` <= 0.001。
- 最小间隔检查：同类水/氮操作间隔 < 7 天记为违反。
- 后期施氮：DAP > 90 仍施氮记为风险。

## 站点级审计汇总

| station_code   |   years |   any_metric_win_years |   mean_total_irrigation |   mean_total_n |   mean_irrigation_event_count |   mean_n_event_count |   mean_first_irrigation_dap |   mean_first_n_dap |   early_dump_warning_years |   interval_warning_years |   late_n_warning_years |   zero_yield_warning_years |   mostly_preventive_warning_years |   mean_management_risk_flag_count |   max_swfac |   max_nstres |
|:---------------|--------:|-----------------------:|------------------------:|---------------:|------------------------------:|---------------------:|----------------------------:|-------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------------:|----------------------------------:|----------------------------------:|------------:|-------------:|
| FQA            |      10 |                      7 |                      45 |            160 |                             1 |                  2   |                           1 |                  2 |                         10 |                        8 |                      0 |                          1 |                                10 |                               2.9 |      0.6055 |       0.0122 |
| HLA            |      10 |                      7 |                     150 |            240 |                             4 |                  3   |                           1 |                  1 |                         10 |                        0 |                      0 |                          0 |                                10 |                               2   |      0      |       0.0284 |
| LCA            |      10 |                     10 |                      45 |             40 |                             1 |                  1   |                           1 |                  1 |                         10 |                        0 |                      0 |                          0 |                                10 |                               2   |      0      |       0.0122 |
| SYA            |      10 |                     10 |                     150 |            212 |                             4 |                  4.3 |                           1 |                  1 |                          0 |                        0 |                      0 |                          0 |                                10 |                               1   |      0.6165 |       0.0437 |
| YCA            |      10 |                     10 |                      45 |             80 |                             1 |                  1   |                           1 |                  2 |                         10 |                        0 |                      0 |                          0 |                                10 |                               2   |      0      |       0.1341 |

## 风险年份样例（最多20行）

| station_code   |   year |   checkpoint_step |   final_grnwt |   total_irrigation |   total_n |   irrigation_event_count |   n_event_count |   first_irrigation_dap |   first_n_dap |   early_irrigation_share |   early_n_share |   max_swfac |   max_nstres | any_metric_win_four_max   | management_risk_flags                                                            |
|:---------------|-------:|------------------:|--------------:|-------------------:|----------:|-------------------------:|----------------:|-----------------------:|--------------:|-------------------------:|----------------:|------------:|-------------:|:--------------------------|:---------------------------------------------------------------------------------|
| FQA            |   2018 |             25000 |          0    |                 45 |       160 |                        1 |               2 |                      1 |             2 |                      1   |          1      |      0      |       0.0122 | False                     | early_dump_warning;interval_warning;zero_yield_warning;mostly_preventive_warning |
| FQA            |   2016 |             25000 |       7803.56 |                 45 |       160 |                        1 |               2 |                      1 |             2 |                      1   |          1      |      0.4567 |       0.0122 | True                      | early_dump_warning;interval_warning;mostly_preventive_warning                    |
| FQA            |   2017 |             25000 |       7704.48 |                 45 |       160 |                        1 |               2 |                      1 |             2 |                      1   |          1      |      0      |       0.0122 | True                      | early_dump_warning;interval_warning;mostly_preventive_warning                    |
| FQA            |   2019 |             25000 |       7859.62 |                 45 |       160 |                        1 |               2 |                      1 |             2 |                      1   |          1      |      0.6055 |       0.0122 | True                      | early_dump_warning;interval_warning;mostly_preventive_warning                    |
| FQA            |   2020 |             25000 |       8963.34 |                 45 |       160 |                        1 |               2 |                      1 |             2 |                      1   |          1      |      0      |       0.0122 | False                     | early_dump_warning;interval_warning;mostly_preventive_warning                    |
| FQA            |   2021 |             25000 |       7463.09 |                 45 |       160 |                        1 |               2 |                      1 |             2 |                      1   |          1      |      0      |       0.0122 | True                      | early_dump_warning;interval_warning;mostly_preventive_warning                    |
| FQA            |   2022 |             25000 |       6687.14 |                 45 |       160 |                        1 |               2 |                      1 |             2 |                      1   |          1      |      0      |       0.0122 | True                      | early_dump_warning;interval_warning;mostly_preventive_warning                    |
| FQA            |   2023 |             25000 |       9099.09 |                 45 |       160 |                        1 |               2 |                      1 |             2 |                      1   |          1      |      0      |       0.0122 | True                      | early_dump_warning;interval_warning;mostly_preventive_warning                    |
| FQA            |   2014 |             25000 |       8300.73 |                 45 |       160 |                        1 |               2 |                      1 |             2 |                      1   |          1      |      0      |       0.0122 | False                     | early_dump_warning;mostly_preventive_warning                                     |
| FQA            |   2015 |             25000 |       8005.62 |                 45 |       160 |                        1 |               2 |                      1 |             2 |                      1   |          1      |      0      |       0.0122 | True                      | early_dump_warning;mostly_preventive_warning                                     |
| HLA            |   2014 |             25000 |       7267.42 |                150 |       240 |                        4 |               3 |                      1 |             1 |                      0.6 |          0.8333 |      0      |       0.0148 | True                      | early_dump_warning;mostly_preventive_warning                                     |
| HLA            |   2015 |             25000 |       7633.33 |                150 |       240 |                        4 |               3 |                      1 |             1 |                      0.6 |          0.8333 |      0      |       0.0145 | False                     | early_dump_warning;mostly_preventive_warning                                     |
| HLA            |   2016 |             25000 |       7446.33 |                150 |       240 |                        4 |               3 |                      1 |             1 |                      0.6 |          0.8333 |      0      |       0.0284 | False                     | early_dump_warning;mostly_preventive_warning                                     |
| HLA            |   2017 |             25000 |       6187.46 |                150 |       240 |                        4 |               3 |                      1 |             1 |                      0.6 |          0.8333 |      0      |       0.0122 | True                      | early_dump_warning;mostly_preventive_warning                                     |
| HLA            |   2018 |             25000 |       6488.87 |                150 |       240 |                        4 |               3 |                      1 |             1 |                      0.6 |          0.8333 |      0      |       0.0158 | True                      | early_dump_warning;mostly_preventive_warning                                     |
| HLA            |   2019 |             25000 |       5740.03 |                150 |       240 |                        4 |               3 |                      1 |             1 |                      0.6 |          0.8333 |      0      |       0.0149 | True                      | early_dump_warning;mostly_preventive_warning                                     |
| HLA            |   2020 |             25000 |       7426.46 |                150 |       240 |                        4 |               3 |                      1 |             1 |                      0.6 |          0.8333 |      0      |       0.0149 | True                      | early_dump_warning;mostly_preventive_warning                                     |
| HLA            |   2021 |             25000 |       5363.97 |                150 |       240 |                        4 |               3 |                      1 |             1 |                      0.6 |          0.8333 |      0      |       0.0138 | True                      | early_dump_warning;mostly_preventive_warning                                     |
| HLA            |   2022 |             25000 |       7882.81 |                150 |       240 |                        4 |               3 |                      1 |             1 |                      0.6 |          0.8333 |      0      |       0.0122 | False                     | early_dump_warning;mostly_preventive_warning                                     |
| HLA            |   2023 |             25000 |       6513.96 |                150 |       240 |                        4 |               3 |                      1 |             1 |                      0.6 |          0.8333 |      0      |       0.0143 | True                      | early_dump_warning;mostly_preventive_warning                                     |

## 初步结论

- 在 50 个验证年中，早期集中投入风险年份为 40 个。
- 操作多为无前置胁迫/预防性投入的年份为 50 个。
- 同类操作间隔小于7天的年份为 8 个。
- DAP90后施氮年份为 0 个。
- 零产量年份为 1 个。
- 因此，036_04 的“指标候选成功”不能直接等价为“措施过程合理”。需要将过程合理性作为单独结论汇报。

## 输出图件

- `benchmark_results\036_05_selected_ppo_management_rationality_audit\figures\036_05_ppo_event_timeline_by_station_year.png`
- `benchmark_results\036_05_selected_ppo_management_rationality_audit\figures\036_05_ppo_event_timeline_by_station_year.svg`
- `benchmark_results\036_05_selected_ppo_management_rationality_audit\figures\036_05_management_risk_counts_by_station.png`
- `benchmark_results\036_05_selected_ppo_management_rationality_audit\figures\036_05_management_risk_counts_by_station.svg`

## 输出表格

- `benchmark_results\036_05_selected_ppo_management_rationality_audit\tables\036_05_event_level_audit.csv`
- `benchmark_results\036_05_selected_ppo_management_rationality_audit\tables\036_05_year_level_management_audit.csv`
- `benchmark_results\036_05_selected_ppo_management_rationality_audit\tables\036_05_station_level_management_audit.csv`