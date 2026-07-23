# 031_08 SY2014 free-timing reward v3 PPO smoke record

## Scope

- SYA2014 seed0 only.
- PPO only, 5,000 timesteps.
- Daily free timing; no expert DAP windows.
- Reward v3 removes TOPWT reward and adds stronger early/no-stress/repeat penalties.

## Training summary

| algorithm | station_code | train_years | seed | total_timesteps | run_status | model_path | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PPO | SYA | 2014 | 0.0000 | 5000.0000 | ok | benchmark_results/031_08_sy2014_free_timing_reward_v3_ppo_smoke/ppo/models/SYA/ppo_free_timing_reward_v3_seed0.zip |  |

## Evaluation summary

| algorithm | policy_name | split | run_status | episode_length | final_grnwt | total_irrigation | total_n | profit_simple | reward_v3_sum | timing_penalty_sum | early_dap1_10_irrigation | early_dap1_10_n | reached_both_caps_by_dap10 | irrigation_event_count | n_event_count | first_irrigation_dap | first_n_dap | swfac_stress_days_gt_0p05 | nstres_days_gt_0p05 | daily_csv_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPO | ppo_reward_v3_5k | train | ok | 144.0000 | 10368.4595 | 160.0000 | 250.0000 | 8958.4595 | 7235.4217 | 1723.0378 | 160.0000 | 250.0000 | True | 9.0000 | 6.0000 | 1.0000 | 1.0000 | 8.0000 | 10.0000 | benchmark_results/031_08_sy2014_free_timing_reward_v3_ppo_smoke/ppo/daily_outputs/SYA/2014_train_ppo_reward_v3_daily.csv |
| PPO | ppo_reward_v3_5k | eval | ok | 144.0000 | 10368.4595 | 160.0000 | 250.0000 | 8958.4595 | 7235.4217 | 1723.0378 | 160.0000 | 250.0000 | True | 9.0000 | 6.0000 | 1.0000 | 1.0000 | 8.0000 | 10.0000 | benchmark_results/031_08_sy2014_free_timing_reward_v3_ppo_smoke/ppo/daily_outputs/SYA/2014_eval_ppo_reward_v3_daily.csv |

## Smoke pass signals

- Does not reach both I160/N250 by DAP10: False.
- Final grain >= 031_05 stress_triggered 9761 kg/ha: True.

## Comparison subset

| source | policy_name | final_grnwt | total_irrigation | total_n | profit_simple | first_irrigation_dap | first_n_dap | irrigation_event_count | n_event_count | swfac_stress_days_gt_0p05 | nstres_days_gt_0p05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 031_08 | ppo_reward_v3_5k | 10368.4595 | 160.0000 | 250.0000 | 8958.4595 | 1.0000 | 1.0000 | 9.0000 | 6.0000 | 8.0000 | 10.0000 |
| 031_04 | ppo_5k | 10387.8589 | 160.0000 | 250.0000 | 8977.8589 | 1.0000 | 1.0000 | 9.0000 | 7.0000 | 8.0000 | 10.0000 |
| 031_04 | dqn_5k | 6926.1597 | 160.0000 | 250.0000 | 5516.1597 | 102.0000 | 83.0000 | 8.0000 | 7.0000 | 0.0000 | 42.0000 |
| 031_03 | continuous_noop | 2729.5013 | 0.0000 | 0.0000 | 2729.5013 |  |  | 0.0000 | 0.0000 | 0.0000 | 97.0000 |
| 031_03 | continuous_random | 10365.5566 | 160.0000 | 250.0000 | 8955.5566 | 1.0000 | 1.0000 | 8.0000 | 7.0000 | 8.0000 | 10.0000 |
| 031_03 | dqn_noop | 2729.5013 | 0.0000 | 0.0000 | 2729.5013 |  |  | 0.0000 | 0.0000 | 0.0000 | 97.0000 |
| 031_03 | dqn_random | 10316.4087 | 160.0000 | 250.0000 | 8906.4087 | 1.0000 | 1.0000 | 5.0000 | 5.0000 | 8.0000 | 9.0000 |
| evaluation | early_dump | 10091.8402 | 160.0000 | 250.0000 | 8681.8402 | 1.0000 | 1.0000 | 4.0000 | 4.0000 | 9.0000 | 9.0000 |
| evaluation | uniform_spread | 10908.6035 | 160.0000 | 250.0000 | 9498.6035 | 1.0000 | 1.0000 | 6.0000 | 5.0000 | 0.0000 | 0.0000 |
| evaluation | expert_window_budget | 10908.6035 | 160.0000 | 250.0000 | 9498.6035 | 1.0000 | 1.0000 | 6.0000 | 5.0000 | 0.0000 | 0.0000 |
| evaluation | stress_triggered | 9761.7120 | 120.0000 | 160.0000 | 8841.7120 | 1.0000 | 41.0000 | 3.0000 | 2.0000 | 2.0000 | 27.0000 |
| evaluation | delayed_late | 6926.1597 | 160.0000 | 250.0000 | 5516.1597 | 102.0000 | 83.0000 | 8.0000 | 7.0000 | 0.0000 | 42.0000 |
| 031_06 | ppo_reward_v2_5k | 10292.6953 | 160.0000 | 250.0000 | 8882.6953 | 1.0000 | 1.0000 | 7.0000 | 6.0000 | 8.0000 | 10.0000 |
| 031_06 | dqn_reward_v2_5k | 9697.9248 | 160.0000 | 250.0000 | 8287.9248 | 66.0000 | 66.0000 | 5.0000 | 4.0000 | 0.0000 | 24.0000 |
| 031_04 | ppo_5k | 10387.8589 | 160.0000 | 250.0000 | 8977.8589 | 1.0000 | 1.0000 | 9.0000 | 7.0000 | 8.0000 | 10.0000 |
| 031_04 | dqn_5k | 6926.1597 | 160.0000 | 250.0000 | 5516.1597 | 102.0000 | 83.0000 | 8.0000 | 7.0000 | 0.0000 | 42.0000 |
| 031_05 | early_dump | 10091.8402 | 160.0000 | 250.0000 | 8681.8402 | 1.0000 | 1.0000 | 4.0000 | 4.0000 | 9.0000 | 9.0000 |
| 031_05 | uniform_spread | 10908.6035 | 160.0000 | 250.0000 | 9498.6035 | 1.0000 | 1.0000 | 6.0000 | 5.0000 | 0.0000 | 0.0000 |
| 031_05 | expert_window_budget | 10908.6035 | 160.0000 | 250.0000 | 9498.6035 | 1.0000 | 1.0000 | 6.0000 | 5.0000 | 0.0000 | 0.0000 |
| 031_05 | stress_triggered | 9761.7120 | 120.0000 | 160.0000 | 8841.7120 | 1.0000 | 41.0000 | 3.0000 | 2.0000 | 2.0000 | 27.0000 |

## Interpretation boundary

This is a single-seed reward smoke, not a cross-seed or cross-year claim. Do not tune coefficients in-place based on this result.
