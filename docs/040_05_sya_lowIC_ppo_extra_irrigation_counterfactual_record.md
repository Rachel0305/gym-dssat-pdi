# 040_05 SYA lowIC PPO 额外灌溉反事实审计记录

## 结论先说

- 本任务固定 PPO checkpoint 75000，不重新训练。
- 目标是判断 2014/2017 的水分胁迫失败是否可由一次中后期额外灌溉缓解。
- 因 PPO 原策略已用 I150，而原季节软上限为 I160、单次最小正灌溉为 I15，本任务的加灌分支仅作为机制反事实，临时放宽灌溉上限到 I205。

## 预注册干预计划

| station_code | year | ppo_checkpoint_step | original_total_irrigation | original_total_n | original_swfac_days_gt_0p05 | original_yield | intervention_dap | intervention_irrigation_mm | intervention_n_kg_ha | relaxed_irrigation_limit | original_action_sequence | original_daily_csv_path | nonzero_action_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SYA | 2014 | 75000 | 150.0 | 240.0 | 27 | 5506.1548 | 86 | 45.0 | 0.0 | 205.0 | DAP1 I45/N40; DAP8 I45/N40; DAP15 I30/N0; DAP16 I0/N40; DAP22 I30/N0; DAP23 I0/N40; DAP30 I0/N40; DAP37 I0/N40 | benchmark_results/040_00_sya_lowIC_free_timing_maskableppo/daily_outputs/SYA/SYA_2014_seed0_ckpt75000_daily.csv | 8 |
| SYA | 2017 | 75000 | 150.0 | 240.0 | 31 | 5204.1901 | 68 | 45.0 | 0.0 | 205.0 | DAP1 I45/N40; DAP8 I45/N40; DAP15 I45/N40; DAP22 I0/N40; DAP29 I15/N80 | benchmark_results/040_00_sya_lowIC_free_timing_maskableppo/daily_outputs/SYA/SYA_2017_seed0_ckpt75000_daily.csv | 5 |

## 反事实结果汇总

| algorithm | policy_name | run_status | episode_length | final_grnwt | total_irrigation | total_n | profit_simple | PFP_N | reward_stress_aware_sum | stress_relief_bonus_sum_unscaled | early_dap1_10_irrigation | early_dap1_10_n | reached_both_caps_by_dap10 | irrigation_event_count | n_event_count | first_irrigation_dap | first_n_dap | swfac_stress_days_gt_0p05 | nstres_days_gt_0p05 | action_sequence | model_path | daily_csv_path | station_code | year | branch | relaxed_irrigation_limit | max_swfac | swfac_days_gt_0p001 | swfac_days_gt_0p01 | swfac_days_gt_0p05 | max_nstres | nstres_days_gt_0p001 | nstres_days_gt_0p01 | delta_yield_vs_original_replay | delta_swfac_days_gt_0p05_vs_original_replay | delta_total_irrigation_vs_original_replay |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_schedule_counterfactual | fixed_schedule_counterfactual_stress_aware_5k_seed0 | ok | 144 | 5506.1548 | 150.0 | 240.0 | 4961.9548 | 22.9423 | 0.3913 | 65.5034 | 90.0 | 80.0 | False | 4 | 6 | 1 | 1 | 27 | 0 | DAP1 I45/N40; DAP8 I45/N40; DAP15 I30/N0; DAP16 I0/N40; DAP22 I30/N0; DAP23 I0/N40; DAP30 I0/N40; DAP37 I0/N40 | not_a_model.zip | benchmark_results/040_05_sya_lowIC_ppo_extra_irrigation_counterfactual/daily_outputs/SYA_2014_ppo_original_replay_daily.csv | SYA | 2014 | ppo_original_replay |  | 1.0 | 27 | 27 | 27 | 0.0122 | 9 | 2 | 0.0 | 0 | 0.0 |
| fixed_schedule_counterfactual | fixed_schedule_counterfactual_stress_aware_5k_seed0 | ok | 144 | 8218.8867 | 195.0 | 240.0 | 7625.1867 | 34.2454 | 0.7704 | 65.5034 | 90.0 | 80.0 | False | 5 | 6 | 1 | 1 | 18 | 14 | DAP1 I45/N40; DAP8 I45/N40; DAP15 I30/N0; DAP16 I0/N40; DAP22 I30/N0; DAP23 I0/N40; DAP30 I0/N40; DAP37 I0/N40; DAP86 I45/N0 | not_a_model.zip | benchmark_results/040_05_sya_lowIC_ppo_extra_irrigation_counterfactual/daily_outputs/SYA_2014_ppo_plus_i45_first_water_stress_relaxed_cap_daily.csv | SYA | 2014 | ppo_plus_i45_first_water_stress_relaxed_cap | 205.0 | 0.9444 | 19 | 19 | 18 | 0.1289 | 24 | 17 | 2712.7319 | -9 | 45.0 |
| fixed_schedule_counterfactual | fixed_schedule_counterfactual_stress_aware_5k_seed0 | ok | 130 | 5204.1901 | 150.0 | 240.0 | 4659.9901 | 21.6841 | 0.3437 | 65.6529 | 90.0 | 80.0 | False | 4 | 5 | 1 | 1 | 31 | 0 | DAP1 I45/N40; DAP8 I45/N40; DAP15 I45/N40; DAP22 I0/N40; DAP29 I15/N80 | not_a_model.zip | benchmark_results/040_05_sya_lowIC_ppo_extra_irrigation_counterfactual/daily_outputs/SYA_2017_ppo_original_replay_daily.csv | SYA | 2017 | ppo_original_replay |  | 0.9491 | 31 | 31 | 31 | 0.0144 | 7 | 3 | 0.0 | 0 | 0.0 |
| fixed_schedule_counterfactual | fixed_schedule_counterfactual_stress_aware_5k_seed0 | ok | 130 | 6432.2784 | 195.0 | 240.0 | 5838.5784 | 26.8012 | 0.4883 | 65.6529 | 90.0 | 80.0 | False | 5 | 5 | 1 | 1 | 19 | 0 | DAP1 I45/N40; DAP8 I45/N40; DAP15 I45/N40; DAP22 I0/N40; DAP29 I15/N80; DAP68 I45/N0 | not_a_model.zip | benchmark_results/040_05_sya_lowIC_ppo_extra_irrigation_counterfactual/daily_outputs/SYA_2017_ppo_plus_i45_first_water_stress_relaxed_cap_daily.csv | SYA | 2017 | ppo_plus_i45_first_water_stress_relaxed_cap | 205.0 | 0.999 | 19 | 19 | 19 | 0.0144 | 7 | 3 | 1228.0884 | -12 | 45.0 |

## 解释边界

- 如果加灌改善明显，只能说明当前 PPO 策略存在水量/时机不足的机制风险；不能直接作为正式训练新约束。
- 如果加灌改善不明显，则说明失败不只是中后期水分不足，下一步应检查早期生长轨迹或 reward/策略表达。
