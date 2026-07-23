# 031_23 Double-Dueling DQN SY2014 seed1/seed2 repeat

## Scope

- Repeats only the 031_22 mask-aware Double-Dueling DQN configuration.
- SYA2014, seeds 1 and 2, 20k timesteps each.
- No reward/action/constraint/hyperparameter changes.
- Final model only; no checkpoint selection.

## Correction note

- The first generated console comparison mislabeled seed1/seed2 as seed0 because the inherited 031_22 summarizer wrote seed=0.
- This record and CSVs correct only the labels using seed_repeat; no training or scientific values were rerun or changed.

## Training summary

                    algorithm station_code  train_years  seed  total_timesteps run_status                                                                                                                                                model_path  optimizer_updates  target_updates  notes  seed_repeat
mask_aware_double_dueling_DQN          SYA         2014     1            20000         ok benchmark_results/031_22_literature_aligned_ppo_dqn_ddqn_sy2014_smoke/models/SYA/free_timing_mask_aware_double_dueling_dqn_ncost2x_scaled_reward_seed0.pt              19001              20    NaN            1
mask_aware_double_dueling_DQN          SYA         2014     2            20000         ok benchmark_results/031_22_literature_aligned_ppo_dqn_ddqn_sy2014_smoke/models/SYA/free_timing_mask_aware_double_dueling_dqn_ncost2x_scaled_reward_seed0.pt              19001              20    NaN            2

## Evaluation summary

 seed  final_grnwt  total_irrigation  total_n  profit_simple     PFP_N  early_dap1_10_n  nstres_days_gt_0p05                                                                                                                                           action_sequence
    1 10814.505615             156.0    240.0    9458.505615 45.060440              0.0                    7 DAP12 I18/N0; DAP35 I6/N120; DAP42 I0/N120; DAP43 I18/N0; DAP50 I18/N0; DAP57 I18/N0; DAP64 I18/N0; DAP71 I18/N0; DAP78 I18/N0; DAP85 I18/N0; DAP92 I6/N0
    2 10990.451660              90.0    240.0    9700.451660 45.793549             40.0                    0                  DAP1 I6/N40; DAP19 I6/N0; DAP26 I6/N0; DAP33 I6/N0; DAP46 I6/N0; DAP53 I18/N40; DAP60 I12/N160; DAP67 I6/N0; DAP109 I6/N0; DAP118 I18/N0

## With 031_22 seed0 context

                            comparison_label  seed  final_grnwt  total_irrigation  total_n  profit_simple     PFP_N  early_dap1_10_n  nstres_days_gt_0p05                                                                                                                                                      action_sequence
mask_aware_DoubleDuelingDQN_031_22_20k_seed0     0 10964.654541             156.0    240.0    9608.654541 45.686061             40.0                    5 DAP1 I18/N40; DAP23 I6/N0; DAP30 I6/N80; DAP37 I6/N80; DAP44 I12/N0; DAP55 I18/N0; DAP58 I0/N40; DAP62 I12/N0; DAP69 I24/N0; DAP76 I24/N0; DAP83 I24/N0; DAP91 I6/N0
mask_aware_DoubleDuelingDQN_031_23_20k_seed1     1 10814.505615             156.0    240.0    9458.505615 45.060440              0.0                    7            DAP12 I18/N0; DAP35 I6/N120; DAP42 I0/N120; DAP43 I18/N0; DAP50 I18/N0; DAP57 I18/N0; DAP64 I18/N0; DAP71 I18/N0; DAP78 I18/N0; DAP85 I18/N0; DAP92 I6/N0
mask_aware_DoubleDuelingDQN_031_23_20k_seed2     2 10990.451660              90.0    240.0    9700.451660 45.793549             40.0                    0                             DAP1 I6/N40; DAP19 I6/N0; DAP26 I6/N0; DAP33 I6/N0; DAP46 I6/N0; DAP53 I18/N40; DAP60 I12/N160; DAP67 I6/N0; DAP109 I6/N0; DAP118 I18/N0

## Interpretation boundary

- This is still same-year training/evaluation, not cross-year transfer.
- If seed1/2 remain promising, next step should be frozen transfer to SY2012/SY2015.
- If seed1/2 collapse or front-load N, seed0 should be treated as insufficient evidence.
