# 032_01 LC2010 rule-timing counterfactual audit record

## Scope

- LCA2010 only.
- No PPO/DQN training.
- Same coarse grid, caps, 7-day interval, and stress-aware reward accounting as 032_00.
- Deterministic rule policies only.

## Summary

            rule run_status  episode_length  final_grnwt  total_irrigation  total_n  profit_simple_032  reward_stress_aware_sum  stress_relief_bonus_sum_unscaled  early_dap1_10_irrigation  early_dap1_10_n  irrigation_event_count  n_event_count  first_irrigation_dap  first_n_dap  swfac_stress_days_gt_0p05  nstres_days_gt_0p05                                                          action_sequence                                                                                               daily_csv_path
 early_frontload         ok              95  8318.429565              60.0    240.0        7873.229565                 0.968112                         99.000004                      30.0            240.0                       2              2                     1            1                          0                    0                                DAP1 I30/N120; DAP8 I0/N120; DAP15 I30/N0  benchmark_results/032_01_lc2010_rule_timing_counterfactual/daily_outputs/LCA/2010_early_frontload_daily.csv
  uniform_spread         ok              95  8287.680054              75.0    200.0        7889.180054                 0.960453                         49.500002                      15.0             40.0                       5              5                     1            1                          0                    8 DAP1 I15/N40; DAP22 I15/N40; DAP43 I15/N40; DAP64 I15/N40; DAP85 I15/N40   benchmark_results/032_01_lc2010_rule_timing_counterfactual/daily_outputs/LCA/2010_uniform_spread_daily.csv
 midseason_shift         ok              95  5675.250244              90.0    240.0        5197.050244                 0.434625                         16.135001                       0.0              0.0                       3              3                    36           36                          0                   23                              DAP36 I30/N80; DAP50 I30/N80; DAP64 I30/N80  benchmark_results/032_01_lc2010_rule_timing_counterfactual/daily_outputs/LCA/2010_midseason_shift_daily.csv
    late_delayed         ok              95  3153.670959              90.0    240.0        2675.470959                 0.020080                          0.000000                       0.0              0.0                       3              3                    70           70                          0                   57                              DAP70 I30/N80; DAP77 I30/N80; DAP84 I30/N80     benchmark_results/032_01_lc2010_rule_timing_counterfactual/daily_outputs/LCA/2010_late_delayed_daily.csv
stress_triggered         ok              95  8273.043213              30.0    240.0        7860.843213                 1.033552                        138.611344                      30.0              0.0                       1              3                     1           14                          0                   10                    DAP1 I30/N0; DAP14 I0/N80; DAP40 I0/N80; DAP47 I0/N80 benchmark_results/032_01_lc2010_rule_timing_counterfactual/daily_outputs/LCA/2010_stress_triggered_daily.csv
ppo_03200_replay         ok              95  8348.992310              60.0    240.0        7903.792310                 0.972941                         99.000004                      45.0            240.0                       3              2                     1            1                          0                    0                   DAP1 I30/N120; DAP8 I0/N120; DAP9 I15/N0; DAP16 I15/N0 benchmark_results/032_01_lc2010_rule_timing_counterfactual/daily_outputs/LCA/2010_ppo_03200_replay_daily.csv

## Ranges

- Yield range across rules: 5195.32 kg/ha.
- 032 reward-sum range across rules: 1.013472.
- Highest 032 reward rule: stress_triggered.

## Interpretation boundary

- This audit diagnoses the LC2010 timing/reward landscape.
- It does not select a final management policy and does not train RL.

## First-read interpretation

- LC2010 strongly penalizes delayed water/N: `midseason_shift` and `late_delayed` lose roughly 2.6-5.2 t/ha relative to the early/high-yield rules.
- Therefore, the early tendency in 032_00 is not purely nonsensical. In this site-year, waiting until mid-season or late season is genuinely harmful.
- However, the 032 reward accounting ranks `stress_triggered` highest despite slightly lower yield, because it uses less irrigation and earns more stress-relief bonus.
- The 032_00 MaskablePPO replay has the highest yield among these rules, but not the highest reward and not the lowest water use.
- This points to a learning/search issue under the current 5k PPO smoke: a better reward-ranked deterministic rule exists, but PPO did not discover it.
- Next step should not be coefficient tuning. A more useful next check is whether longer PPO training or multi-seed PPO can discover the lower-water stress-triggered/near-triggered region, while still avoiding late failure.
