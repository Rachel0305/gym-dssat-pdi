# 032_00 free-timing stress-aware PPO/DQN smoke record

## Scope

- Site-year: LCA2010.
- Seed0 only.
- Final model after 5000 timesteps; no checkpoint selection.
- Algorithms: MaskablePPO and DQN.
- Daily observation; no expert-DAP windows.
- Coarse action grid and 7-day operation interval retained as feasible-operation constraints.

## Training summary

  algorithm station_code train_years  seed  total_timesteps run_status                                                                                                    model_path notes
MaskablePPO          LCA        2010     0             5000         ok benchmark_results/032_00_free_timing_stress_aware_ppo_dqn_smoke/models/LCA/maskableppo_stress_aware_seed0.zip      
        DQN          LCA        2010     0             5000         ok         benchmark_results/032_00_free_timing_stress_aware_ppo_dqn_smoke/models/LCA/dqn_stress_aware_seed0.zip      

## Evaluation summary

  algorithm                       policy_name run_status  episode_length  final_grnwt  total_irrigation  total_n  profit_simple     PFP_N  reward_stress_aware_sum  stress_relief_bonus_sum_unscaled  early_dap1_10_irrigation  early_dap1_10_n  reached_both_caps_by_dap10  irrigation_event_count  n_event_count  first_irrigation_dap  first_n_dap  swfac_stress_days_gt_0p05  nstres_days_gt_0p05                                                                                                                                                                  action_sequence                                                                                                    model_path                                                                                                                 daily_csv_path station_code  year  seed
MaskablePPO maskableppo_stress_aware_5k_seed0         ok              95  8348.992310              60.0    240.0    7903.792310 34.787468                 0.972941                         99.000004                      45.0            240.0                       False                       3              2                     1            1                          0                    0                                                                                                                           DAP1 I30/N120; DAP8 I0/N120; DAP9 I15/N0; DAP16 I15/N0 benchmark_results/032_00_free_timing_stress_aware_ppo_dqn_smoke/models/LCA/maskableppo_stress_aware_seed0.zip benchmark_results/032_00_free_timing_stress_aware_ppo_dqn_smoke/daily_outputs/LCA/2010_maskableppo_stress_aware_eval_daily.csv          LCA  2010     0
        DQN         dqn_stress_aware_5k_seed0         ok              95  8338.466797             150.0    240.0    7794.266797 34.743612                 0.773278                          0.000000                      60.0            120.0                       False                       8              5                     3            2                          0                    0 DAP2 I0/N80; DAP3 I45/N0; DAP8 I0/N40; DAP9 I15/N0; DAP15 I0/N40; DAP16 I15/N0; DAP22 I0/N40; DAP23 I15/N0; DAP29 I0/N40; DAP30 I15/N0; DAP37 I15/N0; DAP44 I15/N0; DAP51 I15/N0         benchmark_results/032_00_free_timing_stress_aware_ppo_dqn_smoke/models/LCA/dqn_stress_aware_seed0.zip         benchmark_results/032_00_free_timing_stress_aware_ppo_dqn_smoke/daily_outputs/LCA/2010_dqn_stress_aware_eval_daily.csv          LCA  2010     0

## Interpretation boundary

- This is an execution/behavior smoke only.
- It does not prove cross-seed, cross-year, or cross-site performance.
- Coefficients are frozen for this task and must not be tuned in-place from these results.

## Non-scientific execution note

- First execution attempt failed before training because `runtime.mode` was incorrectly set to `smoke`; gym-DSSAT accepts only `all`, `fertilization`, or `irrigation`.
- The config was corrected to `runtime.mode: all` and rerun. This was an infrastructure/configuration fix, not a reward or algorithm change.

## First-read interpretation

- Coarse irrigation levels removed the exact 6 mm micro-irrigation failure mode by construction.
- MaskablePPO reduced LC2010 irrigation from the previous selected PPO candidate pattern of many 6 mm events to 3 irrigation events and 60 mm total irrigation in this seed0 smoke.
- However, MaskablePPO still front-loaded management: DAP1 I30/N120, DAP8 N120, DAP9 I15, DAP16 I15. This is not yet a convincing free-timing agronomic policy.
- DQN remained more fragmented than PPO: 8 irrigation events, 5 nitrogen events, I150/N240.
- The same-day stress-relief bonus is probably too sparse or too local to explain/prefer preventive timing: both evaluated policies kept W stress and N stress near zero, so stress-relief evidence is limited.
- Branch outcome: partial engineering improvement, not success. Do not expand this exact v2-smoke reward to all sites/years without another design step.
