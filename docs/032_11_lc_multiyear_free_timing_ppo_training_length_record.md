# 032_11 LC multi-year free-timing MaskablePPO training-length record

## Status

- Run pass: `True`.
- All train years sampled: `True`.
- All checkpoints saved: `True`.
- All checkpoint-year evaluations complete: `True`.

## Scope

- Station: LCA / LC.
- Training years: 2005, 2006, 2007, 2008, 2009, 2010.
- Algorithm: MaskablePPO only.
- Weather forecast: not included.
- Seed: 0.
- Total timesteps: 100000.
- Checkpoints: 25000, 50000, 75000, 100000.
- Reward/actions/constraints inherited unchanged from 032_10 / 032_00.

## Training checkpoint inventory

station_code                   train_years  seed  checkpoint_step run_status                                                                                                                                             model_path                                                     model_sha256 notes
         LCA 2005,2006,2007,2008,2009,2010     0            25000         ok  benchmark_results/032_11_lc_multiyear_free_timing_ppo_training_length/models/LCA/LCA_multiyear_2005_2010_stress_aware_maskableppo_seed0_ckpt25000.zip ef450511f04d97597de48d7666a66e954fd81b87f39e2f891a03f5763ba409c2      
         LCA 2005,2006,2007,2008,2009,2010     0            50000         ok  benchmark_results/032_11_lc_multiyear_free_timing_ppo_training_length/models/LCA/LCA_multiyear_2005_2010_stress_aware_maskableppo_seed0_ckpt50000.zip 222b5b7ecfac1e1b0faf51109bc39bd05bf685558fd7db8d8c76e08ba2e2b176      
         LCA 2005,2006,2007,2008,2009,2010     0            75000         ok  benchmark_results/032_11_lc_multiyear_free_timing_ppo_training_length/models/LCA/LCA_multiyear_2005_2010_stress_aware_maskableppo_seed0_ckpt75000.zip 66d1486029d23db3505c3ee1e70773750e430627c670f70f9c8e90e4ef6e0795      
         LCA 2005,2006,2007,2008,2009,2010     0           100000         ok benchmark_results/032_11_lc_multiyear_free_timing_ppo_training_length/models/LCA/LCA_multiyear_2005_2010_stress_aware_maskableppo_seed0_ckpt100000.zip 3d0c1667d982fa1587a907b1abd245abcaaa89e391f21e3cd1a32e963cd01d1d      

## Training year sampling

 year  episode_count
 2005            151
 2006            162
 2007            147
 2008            182
 2009            181
 2010            172

## Mean performance by checkpoint across LC2005-2010

 checkpoint_step  mean_final_grnwt  mean_total_irrigation  mean_total_n  mean_PFP_N  mean_reward  max_nstres  max_swfac
           25000       9533.474833                  150.0         240.0   39.722812     1.112269    0.012191   0.422022
           50000       9495.228271                  135.0         240.0   39.563451     1.122166    0.012191   0.134291
           75000       9560.169881                   75.0         200.0   47.800849     1.261627    0.170611   0.134291
          100000       5540.467733                   45.0          40.0  138.511693     0.911194    0.475641   0.000000

## Early-action pattern by checkpoint

 checkpoint_step  mean_first_irrigation_dap  mean_first_n_dap  max_first_irrigation_dap  max_first_n_dap  mean_irrigation_event_count  mean_n_event_count
           25000                        1.0               1.0                       1.0              1.0                          4.0                 4.0
           50000                        1.0               1.0                       1.0              1.0                          5.0                 4.0
           75000                        1.0               1.0                       1.0              1.0                          3.0                 3.0
          100000                        1.0               1.0                       1.0              1.0                          1.0                 1.0

## Per-year checkpoint evaluation

 year  checkpoint_step run_status  final_grnwt  total_irrigation  total_n      PFP_N  reward_stress_aware_sum  max_swfac  max_nstres  swfac_days_gt_0p05  nstres_days_gt_0p05  irrigation_event_count  n_event_count  first_irrigation_dap  first_n_dap                                                        action_sequence
 2005            25000         ok 11771.059570             150.0    240.0  49.046082                 1.464127   0.000000    0.012191                   0                    0                       4              4                     1            1 DAP1 I45/N40; DAP8 I45/N120; DAP15 I45/N40; DAP22 I15/N0; DAP23 I0/N40
 2006            25000         ok  9117.434692             150.0    240.0  37.989311                 1.044855   0.000000    0.012191                   0                    0                       4              4                     1            1 DAP1 I45/N40; DAP8 I45/N120; DAP15 I45/N40; DAP22 I15/N0; DAP23 I0/N40
 2007            25000         ok  8051.004639             150.0    240.0  33.545853                 0.881314   0.000000    0.012191                   0                    0                       4              4                     1            1 DAP1 I45/N40; DAP8 I45/N120; DAP15 I45/N40; DAP22 I0/N40; DAP23 I15/N0
 2008            25000         ok 10766.018066             150.0    240.0  44.858409                 1.310419   0.422022    0.012191                   2                    0                       4              4                     1            1 DAP1 I45/N40; DAP8 I45/N120; DAP15 I45/N40; DAP22 I0/N40; DAP23 I15/N0
 2009            25000         ok  9219.428711             150.0    240.0  38.414286                 1.061008   0.000000    0.012191                   0                    0                       4              4                     1            1 DAP1 I45/N40; DAP8 I45/N120; DAP15 I45/N40; DAP22 I15/N0; DAP23 I0/N40
 2010            25000         ok  8275.903320             150.0    240.0  34.482931                 0.911893   0.000000    0.012191                   0                    0                       4              4                     1            1 DAP1 I45/N40; DAP8 I45/N120; DAP15 I45/N40; DAP22 I15/N0; DAP23 I0/N40
 2005            50000         ok 11792.752686             135.0    240.0  49.136470                 1.484055   0.000000    0.012191                   0                    0                       5              4                     1            1 DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80; DAP22 I30/N40; DAP29 I30/N0
 2006            50000         ok  9120.916138             135.0    240.0  38.003817                 1.061905   0.000000    0.012191                   0                    0                       5              4                     1            1 DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80; DAP22 I30/N0; DAP29 I30/N40
 2007            50000         ok  8055.982056             135.0    240.0  33.566592                 0.896949   0.000000    0.012191                   0                    0                       5              4                     1            1 DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80; DAP22 I30/N0; DAP29 I30/N40
 2008            50000         ok 10468.126221             135.0    240.0  43.617193                 1.278156   0.134291    0.012191                   1                    0                       5              4                     1            1 DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80; DAP22 I30/N0; DAP29 I30/N40
 2009            50000         ok  9239.771118             135.0    240.0  38.499046                 1.080709   0.000000    0.012191                   0                    0                       5              4                     1            1 DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80; DAP22 I30/N40; DAP29 I30/N0
 2010            50000         ok  8293.821411             135.0    240.0  34.557589                 0.931224   0.000000    0.012191                   0                    0                       5              4                     1            1 DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80; DAP22 I30/N40; DAP29 I30/N0
 2005            75000         ok 11674.674072              75.0    200.0  58.373370                 1.594599   0.000000    0.170611                   0                   13                       3              3                     1            1                              DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80
 2006            75000         ok  9180.161743              75.0    200.0  45.900809                 1.200466   0.000000    0.012191                   0                    0                       3              3                     1            1                              DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80
 2007            75000         ok  8100.561523              75.0    200.0  40.502808                 1.033192   0.000000    0.012191                   0                    0                       3              3                     1            1                              DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80
 2008            75000         ok 10769.094238              75.0    200.0  53.845471                 1.454909   0.134291    0.120946                   1                    6                       3              3                     1            1                              DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80
 2009            75000         ok  9287.845459              75.0    200.0  46.439227                 1.217505   0.000000    0.012191                   0                    0                       3              3                     1            1                              DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80
 2010            75000         ok  8348.682251              75.0    200.0  41.743411                 1.069092   0.084845    0.012191                   2                    0                       3              3                     1            1                              DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80
 2005           100000         ok  5796.917114              45.0     40.0 144.922928                 0.951713   0.000000    0.450803                   0                   75                       1              1                     1            1                                                           DAP1 I45/N40
 2006           100000         ok  5669.234619              45.0     40.0 141.730865                 0.931539   0.000000    0.475641                   0                   65                       1              1                     1            1                                                           DAP1 I45/N40
 2007           100000         ok  5422.731934              45.0     40.0 135.568298                 0.892592   0.000000    0.396814                   0                   62                       1              1                     1            1                                                           DAP1 I45/N40
 2008           100000         ok  5537.579956              45.0     40.0 138.439499                 0.910738   0.000000    0.439748                   0                   69                       1              1                     1            1                                                           DAP1 I45/N40
 2009           100000         ok  5448.051147              45.0     40.0 136.201279                 0.896592   0.000000    0.389712                   0                   70                       1              1                     1            1                                                           DAP1 I45/N40
 2010           100000         ok  5368.291626              45.0     40.0 134.207291                 0.883990   0.000000    0.381657                   0                   61                       1              1                     1            1                                                           DAP1 I45/N40

## Interpretation boundary

- This task only tests whether longer training helps the same no-forecast multi-year setup.
- It is not a final cross-year model-selection result.
- It does not evaluate LC2011-2020 or LC2021-2023.
- It does not tune reward, constraints, or checkpoint-selection rules in-place.
