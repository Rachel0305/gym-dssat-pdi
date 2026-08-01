# 041_06 SYA lowIC teacher 轨迹差异与 PPO 策略输入敏感性审计记录

## 结论

- 分支：`A_teacher_policy_sensitivity_audit_completed`
- 本任务没有训练，也没有重新跑 DSSAT 季节；只读取 041_04 的 BC dataset 与模型 checkpoint。
- 当前 observation 中 RAIN 直接可见：False；TMIN 直接可见：False。
- 因此不能说当前 PPO 直接根据当天降雨或 Tmin 决策；它最多通过 DSSAT 状态变量间接受历史天气影响。

## observation 覆盖

| concept | directly_in_observation | observation_names | observation_indices | note |
| --- | --- | --- | --- | --- |
| DAP | True | dap | 1 |  |
| TMAX | True | tmax | 19 |  |
| TMIN | False |  |  | not directly visible to policy in current 25-dim observation |
| SRAD | True | srad | 8 |  |
| RAIN | False |  |  | not directly visible to policy in current 25-dim observation |
| soil_water | True | sw_1;sw_2;sw_3;sw_4;sw_5;sw_6;sw_7;sw_8;sw_9 | 9;10;11;12;13;14;15;16;17 |  |
| SWFAC | True | swfac | 18 |  |
| NSTRES | True | nstres | 6 |  |
| cumulative_irrigation | True | totir | 21 |  |
| cumulative_nitrogen | True | cumsumfert | 0 |  |
| yield_state | True | grnwt | 4 |  |
| biomass_state | True | topwt | 20 |  |

## teacher 跨年份轨迹摘要

| year | teacher_tier | candidate_id | sample_days | nonzero_event_count | irrigation_total | nitrogen_total | event_days | event_actions | action_sequence_hashlike |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2014 | strong_all3 | W240_mid_late__N240_three80 | 152 | 9 | 240.0 | 240.0 | 5;34;47;49;64;65;79;99;114 | a1(I0,N80);a6(I45,N0);a1(I0,N80);a6(I45,N0);a6(I45,N0);a1(I0,N80);a3(I30,N0);a6(I45,N0);a3(I30,N0) | 0-0-0-0-0-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-6-0-0-0-0-0-0-0-0-0-0-0-0-1-0-6-0-0-0-0-0-0-0-0-0-0-0-0-0-0-6-1-0-0-0-0-0-0-0-0-0-0-0-0-0-3-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-6-0-0-0-0-0-0-0-0-0-0-0-0-0-0-3-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0 |
| 2015 | near_miss | W210_drop_dap8__N240_three80 | 141 | 6 | 210.0 | 240.0 | 5;35;42;47;65;95 | a7(I45,N80);a6(I45,N0);a3(I30,N0);a1(I0,N80);a7(I45,N80);a6(I45,N0) | 0-0-0-0-0-7-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-6-0-0-0-0-0-0-3-0-0-0-0-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-7-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-6-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0 |
| 2016 | strong_all3 | W150_late_saving__N160_two80 | 141 | 6 | 150.0 | 160.0 | 5;47;49;79;99;119 | a1(I0,N80);a1(I0,N80);a3(I30,N0);a6(I45,N0);a6(I45,N0);a3(I30,N0) | 0-0-0-0-0-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-1-0-3-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-6-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-6-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-3-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0 |
| 2017 | near_miss | W240_mid_late__N240_three80 | 139 | 9 | 240.0 | 240.0 | 5;34;47;49;64;65;79;99;114 | a1(I0,N80);a6(I45,N0);a1(I0,N80);a6(I45,N0);a6(I45,N0);a1(I0,N80);a3(I30,N0);a6(I45,N0);a3(I30,N0) | 0-0-0-0-0-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-6-0-0-0-0-0-0-0-0-0-0-0-0-1-0-6-0-0-0-0-0-0-0-0-0-0-0-0-0-0-6-1-0-0-0-0-0-0-0-0-0-0-0-0-0-3-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-6-0-0-0-0-0-0-0-0-0-0-0-0-0-0-3-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0 |
| 2018 | strong_all3 | W240_late_balanced__N200_mid_late | 129 | 8 | 240.0 | 200.0 | 5;34;47;64;65;94;109;124 | a3(I30,N0);a6(I45,N0);a1(I0,N80);a6(I45,N0);a2(I0,N120);a6(I45,N0);a6(I45,N0);a3(I30,N0) | 0-0-0-0-0-3-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-6-0-0-0-0-0-0-0-0-0-0-0-0-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-6-2-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-6-0-0-0-0-0-0-0-0-0-0-0-0-0-0-6-0-0-0-0-0-0-0-0-0-0-0-0-0-0-3-0-0-0-0 |
| 2019 | near_miss | W225_ppo_like__N240_three80 | 140 | 7 | 225.0 | 240.0 | 5;12;35;42;47;65;95 | a7(I45,N80);a3(I30,N0);a6(I45,N0);a3(I30,N0);a1(I0,N80);a7(I45,N80);a3(I30,N0) | 0-0-0-0-0-7-0-0-0-0-0-0-3-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-6-0-0-0-0-0-0-3-0-0-0-0-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-7-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-3-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0 |
| 2020 | strong_all3 | W240_mid_late__N200_mid_late | 138 | 8 | 240.0 | 200.0 | 34;47;49;64;65;79;99;114 | a6(I45,N0);a1(I0,N80);a6(I45,N0);a6(I45,N0);a2(I0,N120);a3(I30,N0);a6(I45,N0);a3(I30,N0) | 0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-6-0-0-0-0-0-0-0-0-0-0-0-0-1-0-6-0-0-0-0-0-0-0-0-0-0-0-0-0-0-6-2-0-0-0-0-0-0-0-0-0-0-0-0-0-3-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-6-0-0-0-0-0-0-0-0-0-0-0-0-0-0-3-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0 |
| 2021 | strong_all3 | W150_late_saving__N240_three80 | 138 | 7 | 150.0 | 240.0 | 5;47;49;65;79;99;119 | a1(I0,N80);a1(I0,N80);a3(I30,N0);a1(I0,N80);a6(I45,N0);a6(I45,N0);a3(I30,N0) | 0-0-0-0-0-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-1-0-3-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-1-0-0-0-0-0-0-0-0-0-0-0-0-0-6-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-6-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-3-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0 |
| 2022 | near_miss | W195_no_dap8_late_save__N240_two120 | 142 | 6 | 195.0 | 240.0 | 5;35;47;65;95;114 | a8(I45,N120);a6(I45,N0);a2(I0,N120);a6(I45,N0);a3(I30,N0);a3(I30,N0) | 0-0-0-0-0-8-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-6-0-0-0-0-0-0-0-0-0-0-0-2-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-6-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-3-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-3-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0 |
| 2023 | near_miss | W240_ppo_plus_late__N240_three80 | 138 | 7 | 240.0 | 240.0 | 5;12;35;42;47;65;95 | a7(I45,N80);a3(I30,N0);a6(I45,N0);a3(I30,N0);a1(I0,N80);a7(I45,N80);a6(I45,N0) | 0-0-0-0-0-7-0-0-0-0-0-0-3-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-6-0-0-0-0-0-0-3-0-0-0-0-1-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-7-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-6-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0 |

## teacher 轨迹两两差异摘要

| index | year_a | year_b | compared_days | hamming_rate_all_days | different_nonzero_days |
| --- | --- | --- | --- | --- | --- |
| count | 45.0 | 45.0 | 45.0 | 45.0 | 45.0 |
| mean | 2016.6667 | 2020.3333 | 136.9111 | 0.0609 | 8.3111 |
| std | 2.2361 | 2.2361 | 4.1496 | 0.0261 | 3.5215 |
| min | 2014.0 | 2015.0 | 129.0 | 0.0 | 0.0 |
| 25% | 2015.0 | 2019.0 | 138.0 | 0.0496 | 7.0 |
| 50% | 2016.0 | 2021.0 | 138.0 | 0.0709 | 10.0 |
| 75% | 2018.0 | 2022.0 | 139.0 | 0.0797 | 11.0 |
| max | 2022.0 | 2023.0 | 142.0 | 0.087 | 12.0 |

## BC init 与 PPO100K 在 teacher 状态上的动作多样性

| policy_label | sample_count | unique_pred_actions | nonzero_pred_rate | teacher_match_rate_all | teacher_match_rate_on_teacher_nonzero | mean_top_prob | mean_nonzero_prob | mean_pred_irrigation | mean_pred_nitrogen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bc_init | 1398 | 4 | 0.6588 | 0.3684 | 0.5205 | 0.5923 | 0.5335 | 12.7361 | 27.7539 |
| ppo100k | 1398 | 8 | 0.6144 | 0.4063 | 0.3973 | 0.737 | 0.5268 | 20.8155 | 33.2475 |

## 输入扰动敏感性

| policy_label | perturbation_group | direction | obs_indices | argmax_change_rate | mean_total_variation_distance | p95_total_variation_distance | mean_original_argmax_prob_change | mean_nonzero_prob_change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bc_init | grain_yield_state_grnwt | high_max | 4 | 0.309 | 0.0608 | 0.1301 | -0.0488 | -0.0187 |
| bc_init | grain_yield_state_grnwt | high_p90 | 4 | 0.2797 | 0.0568 | 0.1271 | -0.0449 | -0.015 |
| bc_init | crop_biomass_topwt | low_p10 | 20 | 0.0944 | 0.0408 | 0.1179 | -0.0363 | -0.0356 |
| bc_init | crop_biomass_topwt | high_max | 20 | 0.1974 | 0.0362 | 0.1158 | -0.0084 | 0.0185 |
| bc_init | crop_biomass_topwt | high_p90 | 20 | 0.1974 | 0.036 | 0.1158 | -0.0086 | 0.0185 |
| bc_init | cumulative_nitrogen_cumsumfert | high_p90 | 0 | 0.1116 | 0.0178 | 0.0665 | -0.0128 | -0.0024 |
| bc_init | cumulative_nitrogen_cumsumfert | high_max | 0 | 0.1116 | 0.0178 | 0.0665 | -0.0128 | -0.0024 |
| bc_init | cumulative_irrigation_totir | high_max | 21 | 0.1974 | 0.0174 | 0.0652 | -0.0112 | -0.0004 |
| bc_init | cumulative_irrigation_totir | high_p90 | 21 | 0.1946 | 0.0169 | 0.0637 | -0.0107 | -0.0001 |
| bc_init | dap | high_max | 1 | 0.0172 | 0.0119 | 0.0544 | -0.006 | -0.0028 |
| bc_init | dap | high_p90 | 1 | 0.0143 | 0.01 | 0.0411 | -0.0045 | -0.0024 |
| bc_init | cumulative_nitrogen_cumsumfert | low_p10 | 0 | 0.0107 | 0.0061 | 0.0307 | 0.0008 | -0.0013 |
| bc_init | cumulative_irrigation_totir | low_p10 | 21 | 0.0057 | 0.0039 | 0.0228 | -0.0 | -0.0007 |
| bc_init | dap | low_p10 | 1 | 0.0072 | 0.0031 | 0.0152 | -0.0001 | 0.0006 |
| bc_init | srad | low_p10 | 8 | 0.0079 | 0.0023 | 0.0108 | -0.0004 | -0.0002 |
| bc_init | tmax | high_max | 19 | 0.0086 | 0.002 | 0.0107 | -0.0006 | 0.0001 |
| bc_init | srad | high_max | 8 | 0.0064 | 0.0017 | 0.0081 | 0.0001 | 0.0001 |
| bc_init | grain_yield_state_grnwt | low_p10 | 4 | 0.0 | 0.0016 | 0.0113 | 0.0007 | -0.0003 |
| bc_init | tmax | high_p90 | 19 | 0.0079 | 0.0012 | 0.0068 | -0.0003 | 0.0001 |
| bc_init | srad | high_p90 | 8 | 0.005 | 0.0012 | 0.0065 | 0.0001 | 0.0001 |
| bc_init | tmax | low_p10 | 19 | 0.0007 | 0.0008 | 0.0044 | 0.0001 | -0.0 |
| bc_init | swfac | high_max | 18 | 0.0007 | 0.0001 | 0.0007 | -0.0 | 0.0 |
| bc_init | soil_water_sw_all_layers | high_max | 9;10;11;12;13;14;15;16;17 | 0.0 | 0.0001 | 0.0005 | -0.0 | -0.0 |
| bc_init | soil_water_sw_all_layers | high_p90 | 9;10;11;12;13;14;15;16;17 | 0.0007 | 0.0001 | 0.0004 | -0.0 | -0.0 |
| bc_init | soil_water_sw_all_layers | low_p10 | 9;10;11;12;13;14;15;16;17 | 0.0 | 0.0 | 0.0002 | 0.0 | -0.0 |
| bc_init | nstres | high_max | 6 | 0.0 | 0.0 | 0.0002 | 0.0 | -0.0 |
| bc_init | swfac | low_p10 | 18 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| bc_init | swfac | high_p90 | 18 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| bc_init | nstres | low_p10 | 6 | 0.0 | 0.0 | 0.0 | -0.0 | 0.0 |
| bc_init | nstres | high_p90 | 6 | 0.0 | 0.0 | 0.0 | -0.0 | 0.0 |
| ppo100k | grain_yield_state_grnwt | high_max | 4 | 0.4313 | 0.2308 | 0.8562 | -0.2156 | -0.0151 |
| ppo100k | grain_yield_state_grnwt | high_p90 | 4 | 0.407 | 0.2293 | 0.8562 | -0.2129 | -0.0148 |
| ppo100k | crop_biomass_topwt | high_max | 20 | 0.372 | 0.2222 | 0.8405 | -0.21 | -0.0297 |
| ppo100k | crop_biomass_topwt | high_p90 | 20 | 0.372 | 0.2202 | 0.8405 | -0.2108 | -0.0277 |
| ppo100k | crop_biomass_topwt | low_p10 | 20 | 0.2511 | 0.1213 | 0.5405 | -0.0472 | 0.0233 |
| ppo100k | cumulative_irrigation_totir | high_max | 21 | 0.0279 | 0.0887 | 0.3196 | -0.0536 | 0.0084 |
| ppo100k | cumulative_irrigation_totir | high_p90 | 21 | 0.0265 | 0.0865 | 0.3125 | -0.0523 | 0.0083 |
| ppo100k | dap | high_max | 1 | 0.0236 | 0.0452 | 0.193 | -0.0279 | -0.0003 |
| ppo100k | cumulative_nitrogen_cumsumfert | high_p90 | 0 | 0.0172 | 0.0349 | 0.1586 | -0.0224 | -0.0007 |
| ppo100k | cumulative_nitrogen_cumsumfert | high_max | 0 | 0.0172 | 0.0349 | 0.1586 | -0.0224 | -0.0007 |

仅显示前 40 行，共 60 行。

## 单维度敏感性 Top

| policy_label | obs_index | obs_name | argmax_change_rate_p90 | mean_total_variation_distance_p90 | std |
| --- | --- | --- | --- | --- | --- |
| bc_init | 4 | grnwt | 0.2797 | 0.0568 | 3180.1689 |
| bc_init | 20 | topwt | 0.1974 | 0.036 | 6002.8408 |
| bc_init | 0 | cumsumfert | 0.1116 | 0.0178 | 82.7506 |
| bc_init | 21 | totir | 0.1946 | 0.0169 | 80.6165 |
| bc_init | 1 | dap | 0.0143 | 0.01 | 40.4141 |
| bc_init | 7 | rtdep | 0.0072 | 0.0066 | 30.9033 |
| bc_init | 22 | vstage | 0.0064 | 0.0027 | 8.0799 |
| bc_init | 2 | dtt | 0.0064 | 0.0019 | 4.9057 |
| bc_init | 19 | tmax | 0.0079 | 0.0012 | 4.6051 |
| bc_init | 8 | srad | 0.005 | 0.0012 | 6.909 |
| bc_init | 5 | istage | 0.0021 | 0.001 | 2.1982 |
| bc_init | 3 | ep | 0.0064 | 0.0008 | 1.864 |
| bc_init | 24 | xlai | 0.0 | 0.0007 | 1.2074 |
| bc_init | 23 | wtdep | 0.0 | 0.0007 | 8.4273 |
| bc_init | 12 | sw_4 | 0.0007 | 0.0 | 0.0993 |
| bc_init | 13 | sw_5 | 0.0 | 0.0 | 0.0891 |
| bc_init | 9 | sw_1 | 0.0007 | 0.0 | 0.0877 |
| bc_init | 10 | sw_2 | 0.0 | 0.0 | 0.0846 |
| bc_init | 15 | sw_7 | 0.0 | 0.0 | 0.0607 |
| bc_init | 11 | sw_3 | 0.0 | 0.0 | 0.0857 |
| bc_init | 14 | sw_6 | 0.0 | 0.0 | 0.0771 |
| bc_init | 16 | sw_8 | 0.0 | 0.0 | 0.0443 |
| bc_init | 18 | swfac | 0.0 | 0.0 | 0.0853 |
| bc_init | 17 | sw_9 | 0.0 | 0.0 | 0.0267 |
| bc_init | 6 | nstres | 0.0 | 0.0 | 0.014 |
| ppo100k | 4 | grnwt | 0.407 | 0.2293 | 3180.1689 |
| ppo100k | 20 | topwt | 0.372 | 0.2202 | 6002.8408 |
| ppo100k | 21 | totir | 0.0265 | 0.0865 | 80.6165 |
| ppo100k | 7 | rtdep | 0.0093 | 0.0362 | 30.9033 |
| ppo100k | 0 | cumsumfert | 0.0172 | 0.0349 | 82.7506 |
| ppo100k | 1 | dap | 0.0193 | 0.0335 | 40.4141 |
| ppo100k | 22 | vstage | 0.005 | 0.0102 | 8.0799 |
| ppo100k | 8 | srad | 0.0014 | 0.0053 | 6.909 |
| ppo100k | 2 | dtt | 0.0007 | 0.0052 | 4.9057 |
| ppo100k | 19 | tmax | 0.0007 | 0.004 | 4.6051 |
| ppo100k | 5 | istage | 0.0036 | 0.0021 | 2.1982 |
| ppo100k | 3 | ep | 0.0007 | 0.002 | 1.864 |
| ppo100k | 24 | xlai | 0.0014 | 0.0016 | 1.2074 |
| ppo100k | 23 | wtdep | 0.0 | 0.0009 | 8.4273 |
| ppo100k | 13 | sw_5 | 0.0 | 0.0001 | 0.0891 |

## 输出文件

- teacher_summary: `benchmark_results/041_06_sya_lowIC_teacher_policy_sensitivity_audit/tables/041_06_teacher_trajectory_summary.csv`
- teacher_pairwise_distance: `benchmark_results/041_06_sya_lowIC_teacher_policy_sensitivity_audit/tables/041_06_teacher_pairwise_action_distance.csv`
- observation_coverage: `benchmark_results/041_06_sya_lowIC_teacher_policy_sensitivity_audit/tables/041_06_observation_variable_coverage.csv`
- observation_ranges: `benchmark_results/041_06_sya_lowIC_teacher_policy_sensitivity_audit/tables/041_06_observation_dimension_ranges.csv`
- policy_states: `benchmark_results/041_06_sya_lowIC_teacher_policy_sensitivity_audit/tables/041_06_policy_predictions_on_teacher_states.csv`
- policy_diversity: `benchmark_results/041_06_sya_lowIC_teacher_policy_sensitivity_audit/tables/041_06_policy_diversity_summary.csv`
- policy_sensitivity: `benchmark_results/041_06_sya_lowIC_teacher_policy_sensitivity_audit/tables/041_06_policy_perturbation_sensitivity.csv`
- top_sensitive_dimensions: `benchmark_results/041_06_sya_lowIC_teacher_policy_sensitivity_audit/tables/041_06_top_sensitive_observation_dimensions.csv`
- figures: `benchmark_results/041_06_sya_lowIC_teacher_policy_sensitivity_audit/figures/041_06_teacher_resource_diversity.png`
- figures: `benchmark_results/041_06_sya_lowIC_teacher_policy_sensitivity_audit/figures/041_06_policy_sensitivity_top20.png`
- figures: `benchmark_results/041_06_sya_lowIC_teacher_policy_sensitivity_audit/figures/041_06_teacher_nonzero_match_rate.png`
- record_md: `docs/041_06_sya_lowIC_teacher_policy_sensitivity_audit_record.md`
