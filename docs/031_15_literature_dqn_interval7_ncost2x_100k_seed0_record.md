# 031_15 Literature DQN interval7 ncost2x 100k seed0 record

## Scope

- SYA2014 only.
- Seed0 only.
- Same DQN/reward/action constraints as 031_13.
- Single changed variable relative to 031_13: total_timesteps 5000 -> 100000.
- This is not a training-step scan and not a new nitrogen-cost scan.

## Training summary

                       algorithm station_code train_years  seed  total_timesteps run_status                                                                                                                 model_path notes                                               task
literature_DQN_interval7_ncost2x          SYA        2014     0           100000         ok benchmark_results/031_15_literature_dqn_interval7_ncost2x_100k_seed0/models/SYA/literature_dqn_interval7_ncost2x_seed0.zip       031_15_literature_dqn_interval7_ncost2x_100k_seed0

## Evaluation summary

                       algorithm                          policy_name station_code  year  seed split run_status  episode_length  final_grnwt  total_irrigation  total_n  profit_simple     PFP_N  literature_reward_sum  early_dap1_10_irrigation  early_dap1_10_n  reached_both_caps_by_dap10  irrigation_event_count  n_event_count  first_irrigation_dap  first_n_dap  swfac_stress_days_gt_0p05  nstres_days_gt_0p05                                                                                                                                                                                           action_sequence                                                                                                                 model_path                                                                                                                                    daily_csv_path                                               task
literature_DQN_interval7_ncost2x lit_dqn_interval7_ncost2x_100k_seed0          SYA  2014     0  eval         ok             144 10842.489014             160.0    250.0    9432.489014 43.369956            1142.113264                      30.0            250.0                       False                      14              2                     1            1                          0                   11 DAP1 I24/N160; DAP8 I0/N90; DAP9 I6/N0; DAP16 I18/N0; DAP23 I18/N0; DAP30 I6/N0; DAP37 I6/N0; DAP44 I6/N0; DAP51 I6/N0; DAP58 I18/N0; DAP65 I12/N0; DAP72 I12/N0; DAP79 I6/N0; DAP86 I12/N0; DAP93 I10/N0 benchmark_results/031_15_literature_dqn_interval7_ncost2x_100k_seed0/models/SYA/literature_dqn_interval7_ncost2x_seed0.zip benchmark_results/031_15_literature_dqn_interval7_ncost2x_100k_seed0/daily_outputs/SYA/2014_seed0_eval_literature_dqn_interval7_ncost2x_daily.csv 031_15_literature_dqn_interval7_ncost2x_100k_seed0

## Interpretation boundary

- If seed0 still uses N250 with early-heavy nitrogen, longer training alone did not fix the learned ranking.
- If seed0 moves toward staged N200 while preserving yield, then 5k was likely too short and seed1/2 should be tested next under the same 100k setup.

## Result interpretation

- Branch: longer training alone did not fix the learned ranking.
- The 100k seed0 policy still used the full N cap: N=250 kg/ha.
- It also increased irrigation to the full I cap: I=160 mm.
- Early nitrogen remained severe: DAP1-10 N=250 kg/ha.
- Compared with 031_13 seed0 5k, 100k yield increased from 10689.64 to 10842.49 kg/ha, but it remained below the 031_10 5k seed0 baseline at 10955.97 kg/ha and still worse than the 031_12 staged-N200 counterfactual profit/PFP_N target.
- Therefore the evidence does not support "5k was simply too short" as the main explanation for the early-N250 behavior.

## Files

- Training summary: `benchmark_results/031_15_literature_dqn_interval7_ncost2x_100k_seed0/evaluation/training_run_summary.csv`
- Evaluation summary: `benchmark_results/031_15_literature_dqn_interval7_ncost2x_100k_seed0/evaluation/eval_summary.csv`
- Prior-context comparison: `benchmark_results/031_15_literature_dqn_interval7_ncost2x_100k_seed0/evaluation/031_15_with_prior_context.csv`
