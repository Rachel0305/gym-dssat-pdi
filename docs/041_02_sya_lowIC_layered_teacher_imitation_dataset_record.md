# 041_02 SYA lowIC 分层 teacher imitation 数据集记录

## 结论

- 分支：`A_layered_teacher_dataset_ready`
- 选中年份数：10
- strong teacher 年份数：5
- near-miss teacher 年份数：5
- 每日 imitation 样本数：1398
- 非零动作样本数：73

## 选中 teacher

| year | source_task | teacher_tier | candidate_id | grain_yield_kg_ha | WP_ET_kg_m3 | PFP_N_kg_kg | summary_irrigation_total | summary_nitrogen_total | gap_yield_vs_four_max | gap_wp_et_vs_four_max | gap_pfp_n_vs_four_max | win_count | deficit_sum_scaled |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2014 | 041_00 | strong_all3 | W240_mid_late__N240_three80 | 12824.0 | 2.54 | 53.4 | 240.0 | 240.0 | 1983.6533 | 0.39 | 16.9 | 3 | 0.0 |
| 2015 | 041_01 | near_miss | W210_drop_dap8__N240_three80 | 11081.0 | 2.27 | 46.2 | 210.0 | 240.0 | 223.1326 | 0.0 | 9.6 | 2 | 0.0 |
| 2016 | 041_00 | strong_all3 | W150_late_saving__N160_two80 | 8052.0 | 1.75 | 50.3 | 150.0 | 160.0 | 180.2214 | 0.12 | 23.8 | 3 | 0.0 |
| 2017 | 041_00 | near_miss | W240_mid_late__N240_three80 | 10718.0 | 2.5 | 44.7 | 240.0 | 240.0 | -178.0376 | 0.05 | 8.0 | 2 | 0.178 |
| 2018 | 041_00 | strong_all3 | W240_late_balanced__N200_mid_late | 8334.0 | 1.85 | 41.7 | 240.0 | 200.0 | 239.824 | 0.05 | 14.4 | 3 | 0.0 |
| 2019 | 041_00 | near_miss | W225_ppo_like__N240_three80 | 10415.0 | 2.05 | 43.4 | 225.0 | 240.0 | 50.4663 | -0.07 | 8.5 | 2 | 0.7 |
| 2020 | 041_00 | strong_all3 | W240_mid_late__N200_mid_late | 9961.0 | 2.17 | 49.8 | 240.0 | 200.0 | 90.3073 | 0.07 | 16.6 | 3 | 0.0 |
| 2021 | 041_00 | strong_all3 | W150_late_saving__N240_three80 | 10500.0 | 2.3 | 43.8 | 150.0 | 240.0 | 420.025 | 0.16 | 9.9 | 3 | 0.0 |
| 2022 | 041_01 | near_miss | W195_no_dap8_late_save__N240_two120 | 11305.0 | 2.45 | 47.1 | 195.0 | 240.0 | 382.5513 | -0.08 | 10.3 | 2 | 0.8 |
| 2023 | 041_00 | near_miss | W240_ppo_plus_late__N240_three80 | 11005.0 | 2.24 | 45.9 | 240.0 | 240.0 | 68.2349 | -0.03 | 9.1 | 2 | 0.3 |

## 动作分布

| teacher_tier | teacher_action_index | requested_irrigation_mm_action | requested_nitrogen_kg_ha_action | samples | weighted_samples |
| --- | --- | --- | --- | --- | --- |
| near_miss | 0 | 0.0 | 0.0 | 665 | 66.5 |
| near_miss | 1 | 0.0 | 80.0 | 6 | 3.0 |
| near_miss | 2 | 0.0 | 120.0 | 1 | 0.5 |
| near_miss | 3 | 30.0 | 0.0 | 10 | 5.0 |
| near_miss | 6 | 45.0 | 0.0 | 11 | 5.5 |
| near_miss | 7 | 45.0 | 80.0 | 6 | 3.0 |
| near_miss | 8 | 45.0 | 120.0 | 1 | 0.5 |
| strong_all3 | 0 | 0.0 | 0.0 | 660 | 132.0 |
| strong_all3 | 1 | 0.0 | 80.0 | 10 | 10.0 |
| strong_all3 | 2 | 0.0 | 120.0 | 2 | 2.0 |
| strong_all3 | 3 | 30.0 | 0.0 | 10 | 10.0 |
| strong_all3 | 6 | 45.0 | 0.0 | 16 | 16.0 |

## 说明

- 本任务不训练 PPO，只构建 warm-start 数据集。
- strong teacher 是三项全超；near-miss teacher 是较弱辅助信号，不能当作三项全优真值。
- 后续 041_03 若训练，必须保留 strong/near-miss 分层权重，不能混成同等标签。

## 输出文件

- 选中 teacher：`benchmark_results/041_02_sya_lowIC_layered_teacher_imitation_dataset/tables/041_02_selected_teacher_trajectories.csv`
- 每日 imitation 数据：`benchmark_results/041_02_sya_lowIC_layered_teacher_imitation_dataset/tables/041_02_imitation_daily_dataset.csv`
- 非零动作日：`benchmark_results/041_02_sya_lowIC_layered_teacher_imitation_dataset/tables/041_02_imitation_action_days.csv`
- 动作分布：`benchmark_results/041_02_sya_lowIC_layered_teacher_imitation_dataset/tables/041_02_action_distribution.csv`
- JSON 结果：`benchmark_results/041_02_sya_lowIC_layered_teacher_imitation_dataset/041_02_result.json`
