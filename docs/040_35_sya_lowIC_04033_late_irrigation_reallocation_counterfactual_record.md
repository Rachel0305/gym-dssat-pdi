# 040_35 SYA lowIC 040_33 后期灌溉重分配反事实记录

## 先说结论

- 2017 最优后移分支为 `move_DAP68_to_96`：产量变化 706.61 kg/ha，SWFAC>0.05 天数变化 -3 天。
- 本任务未训练模型，只是固定日程 DSSAT 反事实，结果只能作为下一轮 PPO 约束/reward 的依据。

## 预注册动作计划

| station_code | year | branch | ppo_source | source_move_dap | target_requested_dap | target_actual_dap | planned_total_irrigation | planned_total_n | irrigation_dap1_30 | irrigation_dap31_60 | irrigation_dap61_90 | irrigation_dap91_plus | action_sequence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SYA | 2014 | original_replay | 040_33 checkpoint 25k |  |  |  | 240.0 | 240.0 | 60.0 | 90.0 | 90.0 | 0.0 | DAP1 I30/N120; DAP8 I30/N120; DAP40 I45/N0; DAP47 I45/N0; DAP61 I45/N0; DAP68 I45/N0 |
| SYA | 2014 | move_DAP68_to_96 | 040_33 checkpoint 25k | 68 | 96 | 96 | 240.0 | 240.0 | 60.0 | 90.0 | 45.0 | 45.0 | DAP1 I30/N120; DAP8 I30/N120; DAP40 I45/N0; DAP47 I45/N0; DAP61 I45/N0; DAP96 I45/N0 |
| SYA | 2014 | move_DAP68_to_103 | 040_33 checkpoint 25k | 68 | 103 | 103 | 240.0 | 240.0 | 60.0 | 90.0 | 45.0 | 45.0 | DAP1 I30/N120; DAP8 I30/N120; DAP40 I45/N0; DAP47 I45/N0; DAP61 I45/N0; DAP103 I45/N0 |
| SYA | 2014 | move_DAP68_to_110 | 040_33 checkpoint 25k | 68 | 110 | 110 | 240.0 | 240.0 | 60.0 | 90.0 | 45.0 | 45.0 | DAP1 I30/N120; DAP8 I30/N120; DAP40 I45/N0; DAP47 I45/N0; DAP61 I45/N0; DAP110 I45/N0 |
| SYA | 2017 | original_replay | 040_33 checkpoint 25k |  |  |  | 240.0 | 240.0 | 60.0 | 90.0 | 90.0 | 0.0 | DAP1 I30/N120; DAP8 I30/N120; DAP31 I45/N0; DAP40 I45/N0; DAP61 I45/N0; DAP68 I45/N0 |
| SYA | 2017 | move_DAP68_to_96 | 040_33 checkpoint 25k | 68 | 96 | 96 | 240.0 | 240.0 | 60.0 | 90.0 | 45.0 | 45.0 | DAP1 I30/N120; DAP8 I30/N120; DAP31 I45/N0; DAP40 I45/N0; DAP61 I45/N0; DAP96 I45/N0 |
| SYA | 2017 | move_DAP68_to_103 | 040_33 checkpoint 25k | 68 | 103 | 103 | 240.0 | 240.0 | 60.0 | 90.0 | 45.0 | 45.0 | DAP1 I30/N120; DAP8 I30/N120; DAP31 I45/N0; DAP40 I45/N0; DAP61 I45/N0; DAP103 I45/N0 |
| SYA | 2017 | move_DAP68_to_110 | 040_33 checkpoint 25k | 68 | 110 | 110 | 240.0 | 240.0 | 60.0 | 90.0 | 45.0 | 45.0 | DAP1 I30/N120; DAP8 I30/N120; DAP31 I45/N0; DAP40 I45/N0; DAP61 I45/N0; DAP110 I45/N0 |
| SYA | 2022 | original_replay | 040_33 checkpoint 25k |  |  |  | 240.0 | 240.0 | 60.0 | 90.0 | 90.0 | 0.0 | DAP1 I30/N120; DAP8 I30/N120; DAP40 I45/N0; DAP47 I45/N0; DAP61 I45/N0; DAP68 I45/N0 |
| SYA | 2022 | move_DAP68_to_96 | 040_33 checkpoint 25k | 68 | 96 | 96 | 240.0 | 240.0 | 60.0 | 90.0 | 45.0 | 45.0 | DAP1 I30/N120; DAP8 I30/N120; DAP40 I45/N0; DAP47 I45/N0; DAP61 I45/N0; DAP96 I45/N0 |
| SYA | 2022 | move_DAP68_to_103 | 040_33 checkpoint 25k | 68 | 103 | 103 | 240.0 | 240.0 | 60.0 | 90.0 | 45.0 | 45.0 | DAP1 I30/N120; DAP8 I30/N120; DAP40 I45/N0; DAP47 I45/N0; DAP61 I45/N0; DAP103 I45/N0 |
| SYA | 2022 | move_DAP68_to_110 | 040_33 checkpoint 25k | 68 | 110 | 110 | 240.0 | 240.0 | 60.0 | 90.0 | 45.0 | 45.0 | DAP1 I30/N120; DAP8 I30/N120; DAP40 I45/N0; DAP47 I45/N0; DAP61 I45/N0; DAP110 I45/N0 |

## 动作编辑明细

| station_code | year | branch | source_dap | target_requested_dap | target_actual_dap | moved_irrigation_mm | source_had_n_kg_ha |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SYA | 2014 | move_DAP68_to_96 | 68 | 96 | 96 | 45.0 | 0.0 |
| SYA | 2014 | move_DAP68_to_103 | 68 | 103 | 103 | 45.0 | 0.0 |
| SYA | 2014 | move_DAP68_to_110 | 68 | 110 | 110 | 45.0 | 0.0 |
| SYA | 2017 | move_DAP68_to_96 | 68 | 96 | 96 | 45.0 | 0.0 |
| SYA | 2017 | move_DAP68_to_103 | 68 | 103 | 103 | 45.0 | 0.0 |
| SYA | 2017 | move_DAP68_to_110 | 68 | 110 | 110 | 45.0 | 0.0 |
| SYA | 2022 | move_DAP68_to_96 | 68 | 96 | 96 | 45.0 | 0.0 |
| SYA | 2022 | move_DAP68_to_103 | 68 | 103 | 103 | 45.0 | 0.0 |
| SYA | 2022 | move_DAP68_to_110 | 68 | 110 | 110 | 45.0 | 0.0 |

## DSSAT 反事实结果

| algorithm | policy_name | run_status | episode_length | final_grnwt | total_irrigation | total_n | profit_simple | PFP_N | reward_stress_aware_sum | stress_relief_bonus_sum_unscaled | early_dap1_10_irrigation | early_dap1_10_n | reached_both_caps_by_dap10 | irrigation_event_count | n_event_count | first_irrigation_dap | first_n_dap | swfac_stress_days_gt_0p05 | nstres_days_gt_0p05 | action_sequence | model_path | daily_csv_path | station_code | year | branch | max_swfac | swfac_days_gt_0p001 | swfac_days_gt_0p01 | swfac_days_gt_0p05 | max_nstres | nstres_days_gt_0p001 | nstres_days_gt_0p01 | delta_yield_vs_original_replay | delta_swfac_days_gt_0p05_vs_original_replay | delta_max_swfac_vs_original_replay | delta_total_irrigation_vs_original_replay | delta_total_n_vs_original_replay |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_schedule_counterfactual | fixed_schedule_counterfactual_stress_aware_5k_seed0 | ok | 144 | 10148.3179 | 240.0 | 240.0 | 9505.1179 | 42.2847 | 1.0037 | 43.5 | 60.0 | 240.0 | False | 6 | 2 | 1 | 1 | 9 | 7 | DAP1 I30/N120; DAP8 I30/N120; DAP40 I45/N0; DAP47 I45/N0; DAP61 I45/N0; DAP68 I45/N0 | not_a_model.zip | benchmark_results/040_35_sya_lowIC_04033_late_irrigation_reallocation_counterfactual/daily_outputs/SYA_2014_original_replay_daily.csv | SYA | 2014 | original_replay | 0.8214 | 9 | 9 | 9 | 0.1113 | 19 | 12 | 0.0 | 0 | 0.0 | 0.0 | 0.0 |
| fixed_schedule_counterfactual | fixed_schedule_counterfactual_stress_aware_5k_seed0 | ok | 144 | 10241.7102 | 240.0 | 240.0 | 9598.5102 | 42.6738 | 1.0185 | 43.5 | 60.0 | 240.0 | False | 6 | 2 | 1 | 1 | 8 | 5 | DAP1 I30/N120; DAP8 I30/N120; DAP40 I45/N0; DAP47 I45/N0; DAP61 I45/N0; DAP96 I45/N0 | not_a_model.zip | benchmark_results/040_35_sya_lowIC_04033_late_irrigation_reallocation_counterfactual/daily_outputs/SYA_2014_move_DAP68_to_96_daily.csv | SYA | 2014 | move_DAP68_to_96 | 0.7587 | 8 | 8 | 8 | 0.0836 | 17 | 9 | 93.3923 | -1 | -0.0628 | 0.0 | 0.0 |
| fixed_schedule_counterfactual | fixed_schedule_counterfactual_stress_aware_5k_seed0 | ok | 144 | 10222.8473 | 240.0 | 240.0 | 9579.6473 | 42.5952 | 1.0155 | 43.5 | 60.0 | 240.0 | False | 6 | 2 | 1 | 1 | 8 | 6 | DAP1 I30/N120; DAP8 I30/N120; DAP40 I45/N0; DAP47 I45/N0; DAP61 I45/N0; DAP103 I45/N0 | not_a_model.zip | benchmark_results/040_35_sya_lowIC_04033_late_irrigation_reallocation_counterfactual/daily_outputs/SYA_2014_move_DAP68_to_103_daily.csv | SYA | 2014 | move_DAP68_to_103 | 0.7975 | 8 | 8 | 8 | 0.099 | 18 | 11 | 74.5294 | -1 | -0.024 | 0.0 | 0.0 |
| fixed_schedule_counterfactual | fixed_schedule_counterfactual_stress_aware_5k_seed0 | ok | 144 | 10118.0719 | 240.0 | 240.0 | 9474.8719 | 42.1586 | 0.999 | 43.5 | 60.0 | 240.0 | False | 6 | 2 | 1 | 1 | 9 | 9 | DAP1 I30/N120; DAP8 I30/N120; DAP40 I45/N0; DAP47 I45/N0; DAP61 I45/N0; DAP110 I45/N0 | not_a_model.zip | benchmark_results/040_35_sya_lowIC_04033_late_irrigation_reallocation_counterfactual/daily_outputs/SYA_2014_move_DAP68_to_110_daily.csv | SYA | 2014 | move_DAP68_to_110 | 0.7518 | 10 | 10 | 9 | 0.1592 | 20 | 12 | -30.246 | 0 | -0.0697 | 0.0 | 0.0 |
| fixed_schedule_counterfactual | fixed_schedule_counterfactual_stress_aware_5k_seed0 | ok | 130 | 9060.1514 | 240.0 | 240.0 | 8416.9514 | 37.7506 | 0.8318 | 43.5 | 60.0 | 240.0 | False | 6 | 2 | 1 | 1 | 12 | 0 | DAP1 I30/N120; DAP8 I30/N120; DAP31 I45/N0; DAP40 I45/N0; DAP61 I45/N0; DAP68 I45/N0 | not_a_model.zip | benchmark_results/040_35_sya_lowIC_04033_late_irrigation_reallocation_counterfactual/daily_outputs/SYA_2017_original_replay_daily.csv | SYA | 2017 | original_replay | 0.9381 | 12 | 12 | 12 | 0.0214 | 10 | 6 | 0.0 | 0 | 0.0 | 0.0 | 0.0 |
| fixed_schedule_counterfactual | fixed_schedule_counterfactual_stress_aware_5k_seed0 | ok | 130 | 9766.759 | 240.0 | 240.0 | 9123.559 | 40.6948 | 0.9434 | 43.5 | 60.0 | 240.0 | False | 6 | 2 | 1 | 1 | 9 | 4 | DAP1 I30/N120; DAP8 I30/N120; DAP31 I45/N0; DAP40 I45/N0; DAP61 I45/N0; DAP96 I45/N0 | not_a_model.zip | benchmark_results/040_35_sya_lowIC_04033_late_irrigation_reallocation_counterfactual/daily_outputs/SYA_2017_move_DAP68_to_96_daily.csv | SYA | 2017 | move_DAP68_to_96 | 0.8278 | 9 | 9 | 9 | 0.1287 | 14 | 8 | 706.6077 | -3 | -0.1104 | 0.0 | 0.0 |
| fixed_schedule_counterfactual | fixed_schedule_counterfactual_stress_aware_5k_seed0 | ok | 130 | 8595.6207 | 240.0 | 240.0 | 7952.4207 | 35.8151 | 0.7584 | 43.5 | 60.0 | 240.0 | False | 6 | 2 | 1 | 1 | 10 | 0 | DAP1 I30/N120; DAP8 I30/N120; DAP31 I45/N0; DAP40 I45/N0; DAP61 I45/N0; DAP103 I45/N0 | not_a_model.zip | benchmark_results/040_35_sya_lowIC_04033_late_irrigation_reallocation_counterfactual/daily_outputs/SYA_2017_move_DAP68_to_103_daily.csv | SYA | 2017 | move_DAP68_to_103 | 0.8998 | 10 | 10 | 10 | 0.0144 | 7 | 3 | -464.5306 | -2 | -0.0384 | 0.0 | 0.0 |
| fixed_schedule_counterfactual | fixed_schedule_counterfactual_stress_aware_5k_seed0 | ok | 130 | 7877.1167 | 240.0 | 240.0 | 7233.9167 | 32.8213 | 0.6449 | 43.5 | 60.0 | 240.0 | False | 6 | 2 | 1 | 1 | 13 | 0 | DAP1 I30/N120; DAP8 I30/N120; DAP31 I45/N0; DAP40 I45/N0; DAP61 I45/N0; DAP110 I45/N0 | not_a_model.zip | benchmark_results/040_35_sya_lowIC_04033_late_irrigation_reallocation_counterfactual/daily_outputs/SYA_2017_move_DAP68_to_110_daily.csv | SYA | 2017 | move_DAP68_to_110 | 0.9133 | 13 | 13 | 13 | 0.0144 | 7 | 3 | -1183.0347 | 1 | -0.0249 | 0.0 | 0.0 |
| fixed_schedule_counterfactual | fixed_schedule_counterfactual_stress_aware_5k_seed0 | ok | 142 | 10138.7805 | 240.0 | 240.0 | 9495.5805 | 42.2449 | 1.0022 | 43.5 | 60.0 | 240.0 | False | 6 | 2 | 1 | 1 | 0 | 0 | DAP1 I30/N120; DAP8 I30/N120; DAP40 I45/N0; DAP47 I45/N0; DAP61 I45/N0; DAP68 I45/N0 | not_a_model.zip | benchmark_results/040_35_sya_lowIC_04033_late_irrigation_reallocation_counterfactual/daily_outputs/SYA_2022_original_replay_daily.csv | SYA | 2022 | original_replay | 0.0 | 0 | 0 | 0 | 0.0122 | 6 | 2 | 0.0 | 0 | 0.0 | 0.0 | 0.0 |
| fixed_schedule_counterfactual | fixed_schedule_counterfactual_stress_aware_5k_seed0 | ok | 142 | 10259.9463 | 240.0 | 240.0 | 9616.7463 | 42.7498 | 1.0214 | 43.5 | 60.0 | 240.0 | False | 6 | 2 | 1 | 1 | 0 | 0 | DAP1 I30/N120; DAP8 I30/N120; DAP40 I45/N0; DAP47 I45/N0; DAP61 I45/N0; DAP96 I45/N0 | not_a_model.zip | benchmark_results/040_35_sya_lowIC_04033_late_irrigation_reallocation_counterfactual/daily_outputs/SYA_2022_move_DAP68_to_96_daily.csv | SYA | 2022 | move_DAP68_to_96 | 0.0 | 0 | 0 | 0 | 0.0122 | 6 | 2 | 121.1658 | 0 | 0.0 | 0.0 | 0.0 |
| fixed_schedule_counterfactual | fixed_schedule_counterfactual_stress_aware_5k_seed0 | ok | 142 | 10259.9463 | 240.0 | 240.0 | 9616.7463 | 42.7498 | 1.0214 | 43.5 | 60.0 | 240.0 | False | 6 | 2 | 1 | 1 | 0 | 0 | DAP1 I30/N120; DAP8 I30/N120; DAP40 I45/N0; DAP47 I45/N0; DAP61 I45/N0; DAP103 I45/N0 | not_a_model.zip | benchmark_results/040_35_sya_lowIC_04033_late_irrigation_reallocation_counterfactual/daily_outputs/SYA_2022_move_DAP68_to_103_daily.csv | SYA | 2022 | move_DAP68_to_103 | 0.0 | 0 | 0 | 0 | 0.0122 | 6 | 2 | 121.1658 | 0 | 0.0 | 0.0 | 0.0 |
| fixed_schedule_counterfactual | fixed_schedule_counterfactual_stress_aware_5k_seed0 | ok | 142 | 10259.9463 | 240.0 | 240.0 | 9616.7463 | 42.7498 | 1.0214 | 43.5 | 60.0 | 240.0 | False | 6 | 2 | 1 | 1 | 0 | 0 | DAP1 I30/N120; DAP8 I30/N120; DAP40 I45/N0; DAP47 I45/N0; DAP61 I45/N0; DAP110 I45/N0 | not_a_model.zip | benchmark_results/040_35_sya_lowIC_04033_late_irrigation_reallocation_counterfactual/daily_outputs/SYA_2022_move_DAP68_to_110_daily.csv | SYA | 2022 | move_DAP68_to_110 | 0.0 | 0 | 0 | 0 | 0.0122 | 6 | 2 | 121.1658 | 0 | 0.0 | 0.0 | 0.0 |

## 判读边界

- 如果后移分支改善 2017 的产量和后期水分胁迫，说明 040_33 的主要问题可能是水分预算过早用完。
- 如果后移分支不改善，则不应继续围绕简单后移灌溉去调 PPO。
