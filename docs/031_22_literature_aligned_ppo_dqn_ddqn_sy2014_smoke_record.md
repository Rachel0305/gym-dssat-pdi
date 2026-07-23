# 031_22 Literature-aligned PPO / DQN / Double-Dueling DQN comparison smoke

## Scope

- SYA2014 seed0 only.
- PPO and vanilla DQN rows are existing 20k outputs from 031_17 and 031_19; they are not retrained.
- New training in this task: project-local mask-aware Double-Dueling DQN, 20k steps.
- Same free-timing discrete action grid, reward, caps, 7-day intervals, irrigation DAP 1-120, fertilization DAP 1-90.
- Final model only; no post-hoc checkpoint selection.

## DDQN training summary

                    algorithm station_code train_years  seed  total_timesteps run_status                                                                                                                                                model_path  optimizer_updates  target_updates notes
mask_aware_double_dueling_DQN          SYA        2014     0            20000         ok benchmark_results/031_22_literature_aligned_ppo_dqn_ddqn_sy2014_smoke/models/SYA/free_timing_mask_aware_double_dueling_dqn_ncost2x_scaled_reward_seed0.pt              19001              20      

## Algorithm comparison

                            comparison_label  final_grnwt  total_irrigation  total_n  profit_simple     PFP_N  early_dap1_10_irrigation  early_dap1_10_n  swfac_stress_days_gt_0p05  nstres_days_gt_0p05                                                                                                                                                                                                                               action_sequence
                MaskablePPO_031_17_20k_seed0 10943.095703             108.0    240.0    9635.095703 45.596232                      12.0            240.0                          0                    7 DAP1 I6/N120; DAP8 I6/N120; DAP15 I6/N0; DAP22 I6/N0; DAP29 I6/N0; DAP36 I6/N0; DAP43 I6/N0; DAP50 I6/N0; DAP57 I6/N0; DAP64 I6/N0; DAP71 I6/N0; DAP78 I6/N0; DAP85 I6/N0; DAP92 I6/N0; DAP99 I6/N0; DAP106 I6/N0; DAP113 I6/N0; DAP120 I6/N0
            SB3_vanilla_DQN_031_19_20k_seed0 10678.765869             156.0    240.0    9322.765869 44.494858                      36.0            160.0                          0                   15                                                                                                   DAP1 I0/N120; DAP2 I24/N0; DAP8 I12/N40; DAP15 I12/N40; DAP22 I12/N40; DAP29 I24/N0; DAP36 I12/N0; DAP43 I24/N0; DAP50 I12/N0; DAP57 I24/N0
mask_aware_DoubleDuelingDQN_031_22_20k_seed0 10964.654541             156.0    240.0    9608.654541 45.686061                      18.0             40.0                          0                    5                                                                          DAP1 I18/N40; DAP23 I6/N0; DAP30 I6/N80; DAP37 I6/N80; DAP44 I12/N0; DAP55 I18/N0; DAP58 I0/N40; DAP62 I12/N0; DAP69 I24/N0; DAP76 I24/N0; DAP83 I24/N0; DAP91 I6/N0

## Interpretation boundary

- This is an algorithm-family smoke, not a full site/year result.
- It can justify whether Double-Dueling DQN deserves more runs under the current free-timing setup.
- It cannot by itself prove cross-year or cross-site robustness.
