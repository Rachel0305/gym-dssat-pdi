# 031_24 Double-Dueling DQN 50k vs 100k training-length sensitivity

## Scope

- SYA2014 training, seeds 0/1/2.
- Step budgets: 50k and 100k.
- Frozen deterministic evaluation on SYA2014, SYA2012, SYA2015.
- Same reward/action/constraint settings as 031_22/031_23.
- Final models only; no checkpoint selection.

## Training summary

                    algorithm station_code  train_year  seed  step_budget run_status                                                                                                                               model_path                                                     model_sha256  optimizer_updates  target_updates                                                      online_hash                                                      target_hash notes
mask_aware_double_dueling_DQN          SYA        2014     0        50000         ok  benchmark_results/031_24_ddqn_training_length_50k_100k_sy_crossyear/models/SYA/free_timing_mask_aware_double_dueling_dqn_seed0_50000.pt 12347ff5e81f29a331d60ef80645f8532af182e1bd365f37fc1773a4310cb74d              49001              50 a6b05fef1d5125db886b1e392cff69e41153c0a65ddaadc0b62fa22bca72f24a a6b05fef1d5125db886b1e392cff69e41153c0a65ddaadc0b62fa22bca72f24a      
mask_aware_double_dueling_DQN          SYA        2014     1        50000         ok  benchmark_results/031_24_ddqn_training_length_50k_100k_sy_crossyear/models/SYA/free_timing_mask_aware_double_dueling_dqn_seed1_50000.pt f79f4a252a260da71d7a7daad19881f37cc521914b27945a0fda8a37eedd5756              49001              50 20d48938a98e4be38abf385fd5f0e03c6bb5a590b58d1bd3503844ffac87f85c 20d48938a98e4be38abf385fd5f0e03c6bb5a590b58d1bd3503844ffac87f85c      
mask_aware_double_dueling_DQN          SYA        2014     2        50000         ok  benchmark_results/031_24_ddqn_training_length_50k_100k_sy_crossyear/models/SYA/free_timing_mask_aware_double_dueling_dqn_seed2_50000.pt f2f69cb77686c51eb92d8d8a33e93a8751e868811c6112977ef8c2f5c27fef55              49001              50 9ff3eb07ac57f3ac2c8bc69b7d9490dc6231d4625395117cc2b5f3013c7cf367 9ff3eb07ac57f3ac2c8bc69b7d9490dc6231d4625395117cc2b5f3013c7cf367      
mask_aware_double_dueling_DQN          SYA        2014     0       100000         ok benchmark_results/031_24_ddqn_training_length_50k_100k_sy_crossyear/models/SYA/free_timing_mask_aware_double_dueling_dqn_seed0_100000.pt d36516c43de5931fe9b459c82207f6715696ea6b49919b5df7f2147ab4c84dcd              99001             100 146a12115107c80c1bc2f0cad2946a4c90a69b859eb033d77113dd39ccc32765 146a12115107c80c1bc2f0cad2946a4c90a69b859eb033d77113dd39ccc32765      
mask_aware_double_dueling_DQN          SYA        2014     1       100000         ok benchmark_results/031_24_ddqn_training_length_50k_100k_sy_crossyear/models/SYA/free_timing_mask_aware_double_dueling_dqn_seed1_100000.pt de5137c861c4fe08e20f34e9242126caf6cc815613fc280ff147d542d3e9e900              99001             100 fe555f16cfb74592e32c93d614cb6b31150dd888551527aaf1e1d23d32826b88 fe555f16cfb74592e32c93d614cb6b31150dd888551527aaf1e1d23d32826b88      
mask_aware_double_dueling_DQN          SYA        2014     2       100000         ok benchmark_results/031_24_ddqn_training_length_50k_100k_sy_crossyear/models/SYA/free_timing_mask_aware_double_dueling_dqn_seed2_100000.pt c7c6bea21886e6f064d1ac0cd0d1698cfa914e0e2552c98d60488947a50890d1              99001             100 b423e530bec65d983151a3833dcfab245871fe79c004ce199d1ef1e2dff255de b423e530bec65d983151a3833dcfab245871fe79c004ce199d1ef1e2dff255de      

## Evaluation summary

 year  seed  step_budget  final_grain_kg_ha  wp_et_kg_m3  pfp_n_kg_kg  irrigation_event_total_mm  nitrogen_event_total_kg_ha  early_dap1_10_n  nstres_days_gt_0p05                                                                                                                                                                                                                                                 action_sequence
 2014     0        50000       10812.923584         2.02         45.1                      156.0                       240.0             40.0                   11                                                                              DAP3 I18/N0; DAP8 I6/N40; DAP16 I0/N80; DAP17 I18/N0; DAP23 I0/N80; DAP24 I18/N0; DAP30 I0/N40; DAP32 I18/N0; DAP39 I18/N0; DAP46 I24/N0; DAP53 I12/N0; DAP60 I12/N0; DAP67 I12/N0
 2012     0        50000        9543.787231         1.99         39.8                      156.0                       240.0            120.0                    6                                                                               DAP1 I0/N40; DAP2 I18/N0; DAP8 I6/N80; DAP16 I12/N40; DAP23 I0/N80; DAP25 I18/N0; DAP32 I18/N0; DAP39 I18/N0; DAP46 I24/N0; DAP53 I12/N0; DAP60 I12/N0; DAP67 I12/N0; DAP98 I6/N0
 2015     0        50000       10508.492432         2.08         43.8                      156.0                       240.0             40.0                   11                                                   DAP1 I0/N40; DAP2 I18/N0; DAP8 I12/N0; DAP14 I0/N80; DAP15 I18/N0; DAP21 I0/N80; DAP22 I18/N0; DAP28 I0/N40; DAP29 I18/N0; DAP36 I18/N0; DAP43 I12/N0; DAP50 I12/N0; DAP57 I12/N0; DAP64 I12/N0; DAP107 I6/N0
 2014     1        50000       10847.484131         2.10         45.2                      156.0                       240.0              0.0                   12                                                                                          DAP2 I12/N0; DAP8 I18/N0; DAP15 I18/N0; DAP22 I18/N0; DAP29 I18/N0; DAP36 I18/N0; DAP43 I18/N0; DAP46 I0/N40; DAP50 I18/N0; DAP53 I0/N120; DAP60 I12/N40; DAP67 I6/N40
 2012     1        50000        9837.178955         2.02         41.0                      156.0                       240.0              0.0                   10                                                                                                                    DAP1 I18/N0; DAP9 I18/N0; DAP16 I18/N0; DAP23 I18/N0; DAP30 I18/N0; DAP37 I18/N0; DAP44 I12/N120; DAP51 I12/N120; DAP99 I6/N0; DAP111 I18/N0
 2015     1        50000       10625.792236         2.09         44.3                      156.0                       240.0              0.0                    8                                                                                         DAP1 I18/N0; DAP9 I18/N0; DAP16 I18/N0; DAP23 I18/N0; DAP30 I18/N0; DAP37 I18/N0; DAP42 I0/N40; DAP44 I18/N0; DAP49 I0/N120; DAP56 I12/N40; DAP63 I12/N40; DAP100 I6/N0
 2014     2        50000       10933.337402         2.14         45.6                      156.0                       240.0             80.0                    7                                           DAP1 I0/N40; DAP7 I6/N0; DAP8 I0/N40; DAP15 I0/N80; DAP22 I6/N40; DAP29 I18/N0; DAP30 I0/N40; DAP36 I6/N0; DAP47 I6/N0; DAP54 I24/N0; DAP61 I24/N0; DAP68 I6/N0; DAP75 I24/N0; DAP82 I24/N0; DAP89 I6/N0; DAP96 I6/N0
 2012     2        50000        9573.035278         2.03         39.9                      156.0                       240.0             80.0                    8 DAP1 I0/N40; DAP8 I0/N40; DAP12 I6/N0; DAP15 I0/N40; DAP21 I18/N0; DAP22 I0/N40; DAP28 I6/N0; DAP29 I0/N80; DAP35 I6/N0; DAP42 I24/N0; DAP51 I6/N0; DAP58 I6/N0; DAP65 I12/N0; DAP72 I6/N0; DAP79 I24/N0; DAP86 I24/N0; DAP93 I6/N0; DAP102 I6/N0; DAP109 I6/N0
 2015     2        50000       10553.778076         2.14         44.0                      156.0                       240.0             80.0                    9                             DAP1 I0/N40; DAP7 I6/N0; DAP8 I0/N40; DAP15 I0/N40; DAP19 I12/N0; DAP22 I0/N80; DAP26 I6/N0; DAP31 I0/N40; DAP33 I6/N0; DAP40 I24/N0; DAP47 I6/N0; DAP54 I24/N0; DAP61 I12/N0; DAP68 I6/N0; DAP75 I24/N0; DAP82 I24/N0; DAP89 I6/N0
 2014     0       100000       10987.617188         2.20         45.8                      102.0                       240.0            240.0                    5                                                                                                                                     DAP1 I24/N160; DAP8 I12/N0; DAP9 I0/N80; DAP41 I6/N0; DAP62 I6/N0; DAP69 I6/N0; DAP100 I12/N0; DAP107 I24/N0; DAP115 I12/N0
 2012     0       100000        9776.350708         2.10         40.7                      132.0                       240.0            240.0                    6                                                                 DAP1 I18/N0; DAP2 I0/N160; DAP9 I12/N80; DAP16 I12/N0; DAP40 I6/N0; DAP48 I12/N0; DAP56 I6/N0; DAP67 I6/N0; DAP74 I6/N0; DAP89 I12/N0; DAP98 I6/N0; DAP105 I12/N0; DAP112 I12/N0; DAP119 I12/N0
 2015     0       100000       10926.271973         2.28         45.5                      126.0                       240.0            240.0                    8                                                                                           DAP1 I24/N160; DAP8 I12/N0; DAP9 I0/N80; DAP15 I12/N0; DAP39 I6/N0; DAP62 I6/N0; DAP69 I6/N0; DAP83 I12/N0; DAP99 I12/N0; DAP106 I12/N0; DAP113 I12/N0; DAP120 I12/N0
 2014     1       100000        2756.176758         0.74          NaN                       30.0                         0.0              0.0                   98                                                                                                                                                                                                                                       DAP1 I24/N0; DAP101 I6/N0
 2012     1       100000        2635.533752         0.72          NaN                       30.0                         0.0              0.0                   97                                                                                                                                                                                                                                        DAP1 I24/N0; DAP99 I6/N0
 2015     1       100000        2662.340088         0.74          NaN                       30.0                         0.0              0.0                  101                                                                                                                                                                                                                                       DAP1 I24/N0; DAP100 I6/N0
 2014     2       100000       10950.382080         2.13         45.6                      150.0                       240.0            200.0                    8                                                                              DAP1 I0/N80; DAP2 I12/N0; DAP8 I6/N120; DAP15 I18/N40; DAP35 I12/N0; DAP46 I12/N0; DAP53 I12/N0; DAP60 I12/N0; DAP67 I12/N0; DAP74 I12/N0; DAP81 I12/N0; DAP91 I24/N0; DAP98 I6/N0
 2012     2       100000        9674.418335         2.15         40.3                      156.0                       240.0            200.0                    8                                                               DAP1 I0/N80; DAP2 I12/N0; DAP8 I0/N120; DAP15 I6/N40; DAP44 I12/N0; DAP51 I12/N0; DAP58 I12/N0; DAP65 I12/N0; DAP72 I12/N0; DAP81 I12/N0; DAP88 I12/N0; DAP98 I6/N0; DAP105 I24/N0; DAP112 I24/N0
 2015     2       100000       10748.220215         2.19         44.8                      150.0                       240.0            200.0                   10                                                                             DAP1 I0/N80; DAP5 I12/N0; DAP8 I6/N120; DAP15 I18/N40; DAP42 I12/N0; DAP49 I12/N0; DAP56 I12/N0; DAP63 I12/N0; DAP70 I12/N0; DAP78 I12/N0; DAP85 I12/N0; DAP93 I24/N0; DAP100 I6/N0

## SY2012/SY2015 gap versus four baselines

 year  seed  step_budget    gap_yield  gap_wp_et  gap_pfp_n  advisor_any_metric_strict_winner winning_metrics
 2012     0        50000  -622.212769      -0.37       -1.4                             False                
 2015     0        50000  -382.507568      -0.25        7.5                              True           PFP_N
 2012     1        50000  -328.821045      -0.34       -0.2                             False                
 2015     1        50000  -265.207764      -0.24        8.0                              True           PFP_N
 2012     2        50000  -592.964722      -0.33       -1.3                             False                
 2015     2        50000  -337.221924      -0.19        7.7                              True           PFP_N
 2012     0       100000  -389.649292      -0.26       -0.5                             False                
 2015     0       100000    35.271973      -0.05        9.2                              True     yield;PFP_N
 2012     1       100000 -7530.466248      -1.64        NaN                             False                
 2015     1       100000 -8228.659912      -1.59        NaN                             False                
 2012     2       100000  -491.581665      -0.21       -0.9                             False                
 2015     2       100000  -142.779785      -0.14        8.5                              True           PFP_N

## 20k DDQN context

                            comparison_label  seed  final_grnwt  total_irrigation  total_n  profit_simple     PFP_N  early_dap1_10_n  nstres_days_gt_0p05
mask_aware_DoubleDuelingDQN_031_22_20k_seed0     0 10964.654541             156.0    240.0    9608.654541 45.686061             40.0                    5
mask_aware_DoubleDuelingDQN_031_23_20k_seed1     1 10814.505615             156.0    240.0    9458.505615 45.060440              0.0                    7
mask_aware_DoubleDuelingDQN_031_23_20k_seed2     2 10990.451660              90.0    240.0    9700.451660 45.793549             40.0                    0

## Interpretation boundary

- Better SYA2014 performance alone is not success; SYA2012/SYA2015 frozen transfer must also be checked.
- If 100k worsens transfer relative to 50k/20k, treat it as possible overfitting to SYA2014.
- No hyperparameter or reward conclusion should be drawn from this task alone.
