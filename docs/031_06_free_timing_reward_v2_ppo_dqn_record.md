# 031_06 Free-timing reward v2 PPO/DQN record

## Scope

- SYA2014 seed0 only.
- Daily free timing; no expert DAP windows.
- PPO and DQN, 5,000 timesteps each.
- Reward v2 includes TOPWT delta plus soft timing penalties for repeated, early-excess, and late operations.

## PPO eval

algorithm      policy_name split run_status  episode_length  final_grnwt  total_irrigation  total_n  profit_simple  reward_v2_sum  timing_penalty_sum  irrigation_event_count  n_event_count  first_irrigation_dap  first_n_dap  swfac_stress_days_gt_0p05  nstres_days_gt_0p05                                                                                                  daily_csv_path
      PPO ppo_reward_v2_5k train         ok             144 10292.695312             160.0    250.0    8882.695312   10468.848619           264.71681                       7              6                     1            1                          8                   10 benchmark_results/031_06_free_timing_reward_v2_ppo_dqn/ppo/daily_outputs/SYA/2014_train_ppo_reward_v2_daily.csv
      PPO ppo_reward_v2_5k  eval         ok             144 10292.695312             160.0    250.0    8882.695312   10468.848619           264.71681                       7              6                     1            1                          8                   10  benchmark_results/031_06_free_timing_reward_v2_ppo_dqn/ppo/daily_outputs/SYA/2014_eval_ppo_reward_v2_daily.csv

## DQN eval

algorithm      policy_name split run_status  episode_length  final_grnwt  total_irrigation  total_n  profit_simple  reward_v2_sum  timing_penalty_sum  irrigation_event_count  n_event_count  first_irrigation_dap  first_n_dap  swfac_stress_days_gt_0p05  nstres_days_gt_0p05                                                                                                  daily_csv_path
      DQN dqn_reward_v2_5k train         ok             144  9697.924805             160.0    250.0    8287.924805    9782.818412          124.285714                       5              4                    66           66                          0                   24 benchmark_results/031_06_free_timing_reward_v2_ppo_dqn/dqn/daily_outputs/SYA/2014_train_dqn_reward_v2_daily.csv
      DQN dqn_reward_v2_5k  eval         ok             144  9697.924805             160.0    250.0    8287.924805    9782.818412          124.285714                       5              4                    66           66                          0                   24  benchmark_results/031_06_free_timing_reward_v2_ppo_dqn/dqn/daily_outputs/SYA/2014_eval_dqn_reward_v2_daily.csv

## Comparison

source          policy_name run_status  final_grnwt  total_irrigation  total_n  profit_simple  reward_v2_sum  timing_penalty_sum  irrigation_event_count  n_event_count  first_irrigation_dap  first_n_dap  swfac_stress_days_gt_0p05  nstres_days_gt_0p05                                                                                                     daily_csv_path
031_06     ppo_reward_v2_5k         ok 10292.695312             160.0    250.0    8882.695312   10468.848619          264.716810                       7              6                   1.0          1.0                          8                   10     benchmark_results/031_06_free_timing_reward_v2_ppo_dqn/ppo/daily_outputs/SYA/2014_eval_ppo_reward_v2_daily.csv
031_06     dqn_reward_v2_5k         ok  9697.924805             160.0    250.0    8287.924805    9782.818412          124.285714                       5              4                  66.0         66.0                          0                   24     benchmark_results/031_06_free_timing_reward_v2_ppo_dqn/dqn/daily_outputs/SYA/2014_eval_dqn_reward_v2_daily.csv
031_04               ppo_5k         ok 10387.858887             160.0    250.0    8977.858887            NaN                 NaN                       9              7                   1.0          1.0                          8                   10              benchmark_results/031_04_free_daily_original_reward_5k_train/ppo/daily_outputs/SYA/2014_ppo_daily.csv
031_04               dqn_5k         ok  6926.159668             160.0    250.0    5516.159668            NaN                 NaN                       8              7                 102.0         83.0                          0                   42         benchmark_results/031_04_free_daily_original_reward_5k_train/dqn/daily_outputs/SYA/2014_eval_dqn_daily.csv
031_05           early_dump         ok 10091.840210             160.0    250.0    8681.840210            NaN                 NaN                       4              4                   1.0          1.0                          9                    9           benchmark_results/031_05_sy2014_free_timing_rule_sensitivity/daily_outputs/SYA/2014_early_dump_daily.csv
031_05       uniform_spread         ok 10908.603516             160.0    250.0    9498.603516            NaN                 NaN                       6              5                   1.0          1.0                          0                    0       benchmark_results/031_05_sy2014_free_timing_rule_sensitivity/daily_outputs/SYA/2014_uniform_spread_daily.csv
031_05 expert_window_budget         ok 10908.603516             160.0    250.0    9498.603516            NaN                 NaN                       6              5                   1.0          1.0                          0                    0 benchmark_results/031_05_sy2014_free_timing_rule_sensitivity/daily_outputs/SYA/2014_expert_window_budget_daily.csv
031_05     stress_triggered         ok  9761.712036             120.0    160.0    8841.712036            NaN                 NaN                       3              2                   1.0         41.0                          2                   27     benchmark_results/031_05_sy2014_free_timing_rule_sensitivity/daily_outputs/SYA/2014_stress_triggered_daily.csv
031_05         delayed_late         ok  6926.159668             160.0    250.0    5516.159668            NaN                 NaN                       8              7                 102.0         83.0                          0                   42         benchmark_results/031_05_sy2014_free_timing_rule_sensitivity/daily_outputs/SYA/2014_delayed_late_daily.csv

## Interpretation boundary

This is a first reward-v2 smoke. It can show whether the two algorithms move away from 031_04 bad modes, but it is not a cross-seed or cross-year claim.

## Main finding

Reward v2 did not produce a successful free-timing policy on SYA2014 seed0.

- PPO remained an early-dump policy: it reached I160/N250 by DAP7, with first irrigation and fertilization both at DAP1. Its yield (10292.70 kg/ha) was lower than the 031_04 original-reward PPO 5k result (10387.86 kg/ha) and lower than the fixed timing `uniform_spread`/`expert_window_budget` references (10908.60 kg/ha).
- DQN improved relative to the bad delayed-late 031_04 DQN result (9697.92 vs 6926.16 kg/ha) by moving first operation from DAP83/102 to DAP66, but it still saturated I160/N250 and remained below the better fixed timing references.
- Therefore, this smoke supports a limited conclusion: the added soft timing penalties partially moved DQN away from the extreme delayed-late failure, but did not solve free-timing RL. PPO still behaves like early cap saturation.

## Actual non-zero actions

PPO evaluation actions:

| DAP | irrigation | nitrogen | cumulative I | cumulative N |
|---:|---:|---:|---:|---:|
| 1 | 21.7909 | 61.0763 | 21.7909 | 61.0763 |
| 2 | 24.2129 | 53.8859 | 46.0037 | 114.9622 |
| 3 | 27.1015 | 43.4615 | 73.1052 | 158.4237 |
| 4 | 26.2839 | 38.9830 | 99.3892 | 197.4067 |
| 5 | 26.8196 | 39.0851 | 126.2087 | 236.4918 |
| 6 | 27.1491 | 13.5082 | 153.3579 | 250.0000 |
| 7 | 6.6421 | 0.0000 | 160.0000 | 250.0000 |

DQN evaluation actions:

| DAP | action index | irrigation | nitrogen | cumulative I | cumulative N |
|---:|---:|---:|---:|---:|---:|
| 66 | 8 | 40.0000 | 80.0000 | 40.0000 | 80.0000 |
| 67 | 8 | 40.0000 | 80.0000 | 80.0000 | 160.0000 |
| 68 | 8 | 40.0000 | 80.0000 | 120.0000 | 240.0000 |
| 69 | 4 | 20.0000 | 10.0000 | 140.0000 | 250.0000 |
| 70 | 4 | 20.0000 | 0.0000 | 160.0000 | 250.0000 |

## Next decision

Do not expand this exact reward-v2 setup to all sites/years yet. The next free-timing task should either:

1. revise the reward so that timing quality is more directly learned, or
2. test whether a longer PPO/DQN budget with the same reward changes the above failure modes.

Any next run should remain pre-registered and should keep random/rule baselines in the comparison table.
