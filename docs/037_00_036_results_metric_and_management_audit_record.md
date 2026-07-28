# 037_00：036 主线结果指标表现与措施合理性总审计记录
## 任务边界
- 本任务不训练模型、不重跑 DSSAT、不修改 reward 或动作约束。
- 输入为 036_04 指标对照表和 036_05 措施合理性审计表。
- 本任务目的：把 036 主线结果整理成“指标表现 + 措施合理性”的可汇报总览。

## 036 主线配置回顾
- 数据源：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013`。
- 输入与执行修复：`IC=1`，动态管理为 `IRRIG=L, FERTI=L`。
- 算法：自由时序 `MaskablePPO`。
- 灌溉档位：`{0, 15, 30, 45}` mm；施氮档位：`{0, 40, 80, 120}` kg/ha。
- 约束：灌溉/施氮最小间隔 7 天、单季水氮上限、DAP90 后禁氮。
- checkpoint：25K/50K/75K/100K；每站点按 036_04 预定义规则选一个 checkpoint。

## 总体摘要
|   station_count |   year_count |   any_metric_win_years |   yield_win_years |   wp_et_win_years |   pfp_n_win_years |   two_or_more_metric_win_years |   hard_risk_years |   hard_risk_clean_years |   mostly_preventive_years |
|----------------:|-------------:|-----------------------:|------------------:|------------------:|------------------:|-------------------------------:|------------------:|------------------------:|--------------------------:|
|               5 |           50 |                     44 |                22 |                32 |                30 |                             32 |                40 |                      10 |                        50 |

## 每站点指标与措施总览
| station_code   |   selected_checkpoint |   years |   any_metric_win_years |   yield_win_years |   wp_et_win_years |   pfp_n_win_years |   mean_yield_gap_vs_four_max |   mean_water_saving_vs_expert_mm |   mean_n_saving_vs_expert_kg_ha |   hard_risk_years |   mostly_preventive_years |
|:---------------|----------------------:|--------:|-----------------------:|------------------:|------------------:|------------------:|-----------------------------:|---------------------------------:|--------------------------------:|------------------:|--------------------------:|
| FQA            |                 25000 |      10 |                      7 |                 1 |                 7 |                 0 |                     -249.333 |                          147.133 |                          82.167 |                10 |                        10 |
| HLA            |                 25000 |      10 |                      7 |                 7 |                 7 |                 0 |                      302.136 |                           79     |                         -52     |                10 |                        10 |
| LCA            |                100000 |      10 |                     10 |                 7 |                 9 |                10 |                      -42.936 |                          154     |                         208     |                10 |                        10 |
| SYA            |                 25000 |      10 |                     10 |                 5 |                 0 |                10 |                      -14.274 |                          116     |                          88     |                 0 |                        10 |
| YCA            |                100000 |      10 |                     10 |                 2 |                 9 |                10 |                       -1.522 |                          159.98  |                         167.9   |                10 |                        10 |

## 代表性成功年份（按胜出指标数和产量 gap 排序，最多 12 行）
| station_code   |   year |   checkpoint_step |   final_grnwt |   total_irrigation |   total_n |   WP_ET_kg_m3 |   PFP_N_kg_kg |   yield_gap_vs_four_max |   wp_et_gap_vs_four_max |   pfp_n_gap_vs_four_max |   win_metric_count | 措施结论   |
|:---------------|-------:|------------------:|--------------:|-------------------:|----------:|--------------:|--------------:|------------------------:|------------------------:|------------------------:|-------------------:|:-----------|
| YCA            |   2019 |            100000 |       7606.58 |                 45 |        80 |          2.21 |          95.1 |                  17.467 |                    0.02 |                    64.4 |                  3 | 存在硬风险 |
| LCA            |   2022 |            100000 |      10405.3  |                 45 |        40 |          2.9  |         260.1 |                  15.377 |                    0.09 |                   218.4 |                  3 | 存在硬风险 |
| LCA            |   2023 |            100000 |       7946.47 |                 45 |        40 |          2.2  |         198.7 |                  11.299 |                    0.07 |                   166.8 |                  3 | 存在硬风险 |
| YCA            |   2021 |            100000 |       8623.96 |                 45 |        80 |          2.43 |         107.8 |                   6.017 |                    0.24 |                    73.1 |                  3 | 存在硬风险 |
| LCA            |   2021 |            100000 |      11258.1  |                 45 |        40 |          3.47 |         281.5 |                   3.561 |                    0.14 |                   236   |                  3 | 存在硬风险 |
| LCA            |   2014 |            100000 |      10524.2  |                 45 |        40 |          3.2  |         263.1 |                   1.78  |                    0.11 |                   220.6 |                  3 | 存在硬风险 |
| LCA            |   2019 |            100000 |       9807    |                 45 |        40 |          2.89 |         245.2 |                   1.595 |                    0.08 |                   205.9 |                  3 | 存在硬风险 |
| LCA            |   2018 |            100000 |       7917.72 |                 45 |        40 |          2.67 |         198   |                   0.317 |                    0.1  |                   166   |                  3 | 存在硬风险 |
| HLA            |   2014 |             25000 |       7267.42 |                150 |       240 |          1.54 |          30.3 |                 600.247 |                    0.17 |                    -4.2 |                  2 | 存在硬风险 |
| HLA            |   2020 |             25000 |       7426.46 |                150 |       240 |          1.6  |          30.9 |                 562.886 |                    0.11 |                    -3.8 |                  2 | 存在硬风险 |
| HLA            |   2017 |             25000 |       6187.46 |                150 |       240 |          1.32 |          25.8 |                 470.954 |                    0.05 |                    -3   |                  2 | 存在硬风险 |
| HLA            |   2023 |             25000 |       6513.96 |                150 |       240 |          1.48 |          27.1 |                 435.485 |                    0.06 |                    -3.4 |                  2 | 存在硬风险 |

## 需要重点解释或复核的年份（按硬风险数量排序，最多 12 行）
| station_code   |   year |   checkpoint_step |   final_grnwt |   total_irrigation |   total_n |   irrigation_event_count |   n_event_count |   first_irrigation_dap |   first_n_dap |   hard_risk_count | mostly_preventive_warning   | management_risk_flags                                                            |
|:---------------|-------:|------------------:|--------------:|-------------------:|----------:|-------------------------:|----------------:|-----------------------:|--------------:|------------------:|:----------------------------|:---------------------------------------------------------------------------------|
| FQA            |   2018 |             25000 |          0    |                 45 |       160 |                        1 |               2 |                      1 |             2 |                 3 | True                        | early_dump_warning;interval_warning;zero_yield_warning;mostly_preventive_warning |
| FQA            |   2020 |             25000 |       8963.34 |                 45 |       160 |                        1 |               2 |                      1 |             2 |                 2 | True                        | early_dump_warning;interval_warning;mostly_preventive_warning                    |
| FQA            |   2016 |             25000 |       7803.56 |                 45 |       160 |                        1 |               2 |                      1 |             2 |                 2 | True                        | early_dump_warning;interval_warning;mostly_preventive_warning                    |
| FQA            |   2017 |             25000 |       7704.48 |                 45 |       160 |                        1 |               2 |                      1 |             2 |                 2 | True                        | early_dump_warning;interval_warning;mostly_preventive_warning                    |
| FQA            |   2019 |             25000 |       7859.62 |                 45 |       160 |                        1 |               2 |                      1 |             2 |                 2 | True                        | early_dump_warning;interval_warning;mostly_preventive_warning                    |
| FQA            |   2022 |             25000 |       6687.14 |                 45 |       160 |                        1 |               2 |                      1 |             2 |                 2 | True                        | early_dump_warning;interval_warning;mostly_preventive_warning                    |
| FQA            |   2023 |             25000 |       9099.09 |                 45 |       160 |                        1 |               2 |                      1 |             2 |                 2 | True                        | early_dump_warning;interval_warning;mostly_preventive_warning                    |
| FQA            |   2021 |             25000 |       7463.09 |                 45 |       160 |                        1 |               2 |                      1 |             2 |                 2 | True                        | early_dump_warning;interval_warning;mostly_preventive_warning                    |
| FQA            |   2014 |             25000 |       8300.73 |                 45 |       160 |                        1 |               2 |                      1 |             2 |                 1 | True                        | early_dump_warning;mostly_preventive_warning                                     |
| HLA            |   2015 |             25000 |       7633.33 |                150 |       240 |                        4 |               3 |                      1 |             1 |                 1 | True                        | early_dump_warning;mostly_preventive_warning                                     |
| HLA            |   2016 |             25000 |       7446.33 |                150 |       240 |                        4 |               3 |                      1 |             1 |                 1 | True                        | early_dump_warning;mostly_preventive_warning                                     |
| HLA            |   2022 |             25000 |       7882.81 |                150 |       240 |                        4 |               3 |                      1 |             1 |                 1 | True                        | early_dump_warning;mostly_preventive_warning                                     |

## 解释边界
- `any_metric_win_four_max` 表示至少一个指标超过四情景最高值，不表示三个指标全部同时超过。
- `mostly_preventive_warning` 不直接等于错误；它表示操作前 3 天未出现明显胁迫，需要用过程图或反事实试验解释。
- 硬风险包括早期集中投入、间隔违规、DAP90 后施氮、零产量；这些是下一轮改进或汇报时优先说明的问题。
- 与 expert 的节水节氮用于说明资源投入变化；与四情景最高值的 gap 用于说明是否真正超过所有对照。

## 输出图件
- `benchmark_results\037_00_036_results_metric_and_management_audit\figures\037_00_station_metric_win_counts.png`
- `benchmark_results\037_00_036_results_metric_and_management_audit\figures\037_00_station_metric_win_counts.svg`
- `benchmark_results\037_00_036_results_metric_and_management_audit\figures\037_00_station_management_audit_counts.png`
- `benchmark_results\037_00_036_results_metric_and_management_audit\figures\037_00_station_management_audit_counts.svg`
- `benchmark_results\037_00_036_results_metric_and_management_audit\figures\037_00_station_mean_gap_and_savings.png`
- `benchmark_results\037_00_036_results_metric_and_management_audit\figures\037_00_station_mean_gap_and_savings.svg`

## 输出表格
- `benchmark_results/037_00_036_results_metric_and_management_audit/tables/037_00_year_level_metric_management_overview.csv`
- `benchmark_results/037_00_036_results_metric_and_management_audit/tables/037_00_station_level_metric_management_overview.csv`
- `benchmark_results/037_00_036_results_metric_and_management_audit/tables/037_00_overall_summary.csv`
