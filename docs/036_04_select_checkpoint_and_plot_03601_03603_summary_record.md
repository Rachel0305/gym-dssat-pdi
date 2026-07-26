# 036_04 checkpoint选择与五站点汇总绘图记录

## 任务边界

- 本任务不训练模型。
- 本任务不修改reward、动作空间、约束条件或DSSAT输入。
- 本任务只整理036_01正式重跑与036_03 replay后的结果。
- FQA2018零产量异常保留，不静默删除。

## 固定checkpoint选择规则

每站点代表checkpoint按以下顺序选择：`any_metric_win_count`、`yield_win_count`、`wp_et_win_count`、`pfp_n_win_count`、`mean_gap_yield + 1000*mean_gap_wp_et + 10*mean_gap_pfp_n`，最后选更早checkpoint。

## 选中的checkpoint

| station_code   |   checkpoint_step |   validation_years |   any_metric_win_count |   yield_win_count |   wp_et_win_count |   pfp_n_win_count |   mean_gap_yield |   mean_gap_wp_et |   mean_gap_pfp_n |
|:---------------|------------------:|-------------------:|-----------------------:|------------------:|------------------:|------------------:|-----------------:|-----------------:|-----------------:|
| FQA            |             25000 |                 10 |                      7 |                 1 |                 7 |                 0 |        -249.333  |           0.097  |          -5.6719 |
| HLA            |             25000 |                 10 |                      7 |                 7 |                 7 |                 0 |         302.136  |           0.034  |          -6.89   |
| LCA            |            100000 |                 10 |                     10 |                 7 |                 9 |                10 |         -42.9355 |           0.077  |         195.21   |
| SYA            |             25000 |                 10 |                     10 |                 5 |                 0 |                10 |         -14.2737 |          -0.096  |           8.47   |
| YCA            |            100000 |                 10 |                     10 |                 2 |                 9 |                10 |          -1.5222 |           0.1833 |          69.4374 |

## 站点汇总

| station_code   |   checkpoint_step |   validation_years |   any_metric_win_count |   yield_win_count |   wp_et_win_count |   pfp_n_win_count |   mean_final_grnwt |   mean_total_irrigation |   mean_total_n |   mean_WP_ET_kg_m3 |   mean_PFP_N_kg_kg |   mean_water_saving_vs_expert_mm |   std_water_saving_vs_expert_mm |   mean_n_saving_vs_expert_kg_ha |   std_n_saving_vs_expert_kg_ha |   mean_yield_gap_vs_four_max |   mean_wp_et_gap_vs_four_max |   mean_pfp_n_gap_vs_four_max | zero_yield_years   |
|:---------------|------------------:|-------------------:|-----------------------:|------------------:|------------------:|------------------:|-------------------:|------------------------:|---------------:|-------------------:|-------------------:|---------------------------------:|--------------------------------:|--------------------------------:|-------------------------------:|-----------------------------:|-----------------------------:|-----------------------------:|:-------------------|
| FQA            |             25000 |                 10 |                      7 |                 1 |                 7 |                 0 |            7188.67 |                      45 |            160 |              2.202 |            49.9222 |                          147.133 |                          16.722 |                         82.1667 |                        13.8046 |                    -249.333  |                       0.097  |                      -5.6719 | 2018               |
| HLA            |             25000 |                 10 |                      7 |                 7 |                 7 |                 0 |            6795.06 |                     150 |            240 |              1.471 |            28.3    |                           79     |                           0     |                        -52      |                         0      |                     302.136  |                       0.034  |                      -6.89   |                    |
| LCA            |            100000 |                 10 |                     10 |                 7 |                 9 |                10 |            9317.2  |                      45 |             40 |              2.851 |           232.95   |                          154     |                           0     |                        208      |                         0      |                     -42.9355 |                       0.077  |                     195.21   |                    |
| SYA            |             25000 |                 10 |                     10 |                 5 |                 0 |                10 |           10141.3  |                     150 |            212 |              2.128 |            48.07   |                          116     |                           0     |                         88      |                        19.3218 |                     -14.2737 |                      -0.096  |                       8.47   |                    |
| YCA            |            100000 |                 10 |                     10 |                 2 |                 9 |                10 |            8202.01 |                      45 |             80 |              2.305 |           102.53   |                          159.98  |                          12.607 |                        167.9    |                         0.3162 |                      -1.5222 |                       0.1833 |                      69.4374 |                    |

## 零产量/异常保留

| station_code   |   year |   checkpoint_step |   final_grnwt |   total_irrigation |   total_n |   WP_ET_kg_m3 |   PFP_N_kg_kg |
|:---------------|-------:|------------------:|--------------:|-------------------:|----------:|--------------:|--------------:|
| FQA            |   2018 |             25000 |             0 |                 45 |       160 |             0 |           nan |

## 输出图件

- `benchmark_results\036_04_select_checkpoint_and_plot_03601_03603_summary\figures\036_04_fqa_selected_checkpoint_metric_gaps.png`
- `benchmark_results\036_04_select_checkpoint_and_plot_03601_03603_summary\figures\036_04_fqa_selected_checkpoint_metric_gaps.svg`
- `benchmark_results\036_04_select_checkpoint_and_plot_03601_03603_summary\figures\036_04_hla_selected_checkpoint_metric_gaps.png`
- `benchmark_results\036_04_select_checkpoint_and_plot_03601_03603_summary\figures\036_04_hla_selected_checkpoint_metric_gaps.svg`
- `benchmark_results\036_04_select_checkpoint_and_plot_03601_03603_summary\figures\036_04_lca_selected_checkpoint_metric_gaps.png`
- `benchmark_results\036_04_select_checkpoint_and_plot_03601_03603_summary\figures\036_04_lca_selected_checkpoint_metric_gaps.svg`
- `benchmark_results\036_04_select_checkpoint_and_plot_03601_03603_summary\figures\036_04_sya_selected_checkpoint_metric_gaps.png`
- `benchmark_results\036_04_select_checkpoint_and_plot_03601_03603_summary\figures\036_04_sya_selected_checkpoint_metric_gaps.svg`
- `benchmark_results\036_04_select_checkpoint_and_plot_03601_03603_summary\figures\036_04_yca_selected_checkpoint_metric_gaps.png`
- `benchmark_results\036_04_select_checkpoint_and_plot_03601_03603_summary\figures\036_04_yca_selected_checkpoint_metric_gaps.svg`
- `benchmark_results\036_04_select_checkpoint_and_plot_03601_03603_summary\figures\036_04_overall_water_n_saving_vs_expert.png`
- `benchmark_results\036_04_select_checkpoint_and_plot_03601_03603_summary\figures\036_04_overall_water_n_saving_vs_expert.svg`

## 解释边界

- 图中“超过四情景最高值”使用036_03的available-baseline最大值口径；若同一年存在多个recorded/template版本，取可用基线最大值是保守口径，但不等同于单一canonical recorded farmer。
- 节水/节氮图默认相对official expert计算，因为导师当前关注PPO是否能在专家措施基础上节水节氮；若后续要求相对四情景最省资源值，可另行生成。