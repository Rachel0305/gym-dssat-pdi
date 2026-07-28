# 032_16 LC multiyear PPO yield-resource reward sensitivity record

## Status

- Screening completed.
- Variants: current_50k, yield_plus_50k, resource_cheaper_50k.
- This task is not final model selection.

## Scope

- Station: LC / LCA.
- Train years: 2005, 2006, 2007, 2008, 2009, 2010.
- Transfer years: 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020.
- Algorithm/action/mask framework inherited from 032_11/032_12.
- New training timesteps per non-reused variant: 50000.
- Evaluation checkpoint: 50k endpoint only.

## Variant definitions

| variant | yield_coef | water_cost | nitrogen_cost | reuse_model |
| --- | --- | --- | --- | --- |
| current_50k | 0.158 | 1.1 | 1.58 | /workspaces/gym-dssat-pdi/benchmark_results/032_11_lc_multiyear_free_timing_ppo_training_length/models/LCA/LCA_multiyear_2005_2010_stress_aware_maskableppo_seed0_ckpt50000.zip |
| yield_plus_50k | 0.18 | 1.1 | 1.58 | None |
| resource_cheaper_50k | 0.158 | 0.8 | 1.2 | None |

## Variant summary

| variant | split | n_years | mean_yield | mean_yield_gap_vs_four_max | yield_win_four | mean_irrigation | mean_nitrogen | mean_PFP_N | mean_PFP_N_gap_vs_four_max | PFP_N_win_four | mean_water_saving_vs_expert | mean_n_saving_vs_expert | mean_max_nstres |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_50k | train_2005_2010 | 6 | 9495.228 | -256.058 | 0 | 135.0 | 240.0 | 39.563 | 0.133 | 3 | 68.967 | 7.833 | 0.012 |
| current_50k | transfer_2011_2020 | 10 | 8991.139 | -119.156 | 1 | 135.0 | 240.0 | 37.463 | -0.026 | 8 | 67.0 | 8.0 | 0.013 |
| resource_cheaper_50k | train_2005_2010 | 6 | 9518.085 | -233.201 | 1 | 150.0 | 240.0 | 39.659 | 0.229 | 4 | 53.967 | 7.833 | 0.012 |
| resource_cheaper_50k | transfer_2011_2020 | 10 | 9031.715 | -78.58 | 1 | 150.0 | 240.0 | 37.632 | 0.143 | 9 | 52.0 | 8.0 | 0.013 |
| yield_plus_50k | train_2005_2010 | 6 | 5421.848 | -4329.438 | 0 | 150.0 | 40.0 | 135.546 | 96.116 | 6 | 53.967 | 207.833 | 0.416 |
| yield_plus_50k | transfer_2011_2020 | 10 | 5311.795 | -3798.499 | 0 | 150.0 | 44.0 | 123.373 | 85.883 | 10 | 52.0 | 204.0 | 0.439 |
| current_50k | all_2005_2020 | 16 | 9180.173 | -170.494 | 1 | 135.0 | 240.0 | 38.251 | 0.033 | 11 | 67.738 | 7.938 | 0.013 |
| resource_cheaper_50k | all_2005_2020 | 16 | 9214.104 | -136.563 | 2 | 150.0 | 240.0 | 38.392 | 0.175 | 13 | 52.738 | 7.938 | 0.013 |
| yield_plus_50k | all_2005_2020 | 16 | 5353.065 | -3997.601 | 0 | 150.0 | 42.5 | 127.938 | 89.72 | 16 | 52.738 | 205.438 | 0.43 |

## Per-year comparison

| variant | split | year | rl_yield_kg_ha | yield_gap_vs_four_max | rl_irrigation_mm | rl_nitrogen_kg_ha | rl_PFP_N_kg_kg | PFP_N_gap_vs_four_max | water_saving_vs_expert | n_saving_vs_expert | action_sequence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_50k | train_2005_2010 | 2005 | 11792.753 | -291.635 | 135.0 | 240.0 | 49.136 | 0.336 | 94.0 | 8.0 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80; DAP22 I30/N40; DAP29 I30/N0 |
| current_50k | train_2005_2010 | 2006 | 9120.916 | -323.928 | 135.0 | 240.0 | 38.004 | -0.196 | 64.0 | 8.0 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80; DAP22 I30/N0; DAP29 I30/N40 |
| current_50k | train_2005_2010 | 2007 | 8055.982 | -49.532 | 135.0 | 240.0 | 33.567 | 0.767 | 64.0 | 8.0 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80; DAP22 I30/N0; DAP29 I30/N40 |
| current_50k | train_2005_2010 | 2008 | 10468.126 | -395.46 | 135.0 | 240.0 | 43.617 | -0.283 | 64.0 | 8.0 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80; DAP22 I30/N0; DAP29 I30/N40 |
| current_50k | train_2005_2010 | 2009 | 9239.771 | -30.614 | 135.0 | 240.0 | 38.499 | 0.999 | 64.0 | 8.0 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80; DAP22 I30/N40; DAP29 I30/N0 |
| current_50k | train_2005_2010 | 2010 | 8293.821 | -445.179 | 135.0 | 240.0 | 34.558 | -0.823 | 63.8 | 7.0 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80; DAP22 I30/N40; DAP29 I30/N0 |
| current_50k | transfer_2011_2020 | 2011 | 9017.245 | -322.755 | 135.0 | 240.0 | 37.572 | -7.423 | 64.0 | 8.0 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80; DAP22 I30/N0; DAP29 I30/N40 |
| current_50k | transfer_2011_2020 | 2012 | 8982.095 | -35.609 | 135.0 | 240.0 | 37.425 | 1.025 | 94.0 | 8.0 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80; DAP22 I30/N0; DAP29 I30/N40 |
| current_50k | transfer_2011_2020 | 2013 | 8646.954 | -76.538 | 135.0 | 240.0 | 36.029 | 0.829 | 64.0 | 8.0 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80; DAP22 I30/N0; DAP29 I30/N40 |
| current_50k | transfer_2011_2020 | 2014 | 10523.115 | 0.688 | 135.0 | 240.0 | 43.846 | 1.346 | 64.0 | 8.0 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80; DAP22 I30/N0; DAP29 I30/N40 |
| current_50k | transfer_2011_2020 | 2015 | 10060.99 | -310.398 | 135.0 | 240.0 | 41.921 | 0.021 | 64.0 | 8.0 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80; DAP22 I30/N0; DAP29 I30/N40 |
| current_50k | transfer_2011_2020 | 2016 | 8297.512 | -267.963 | 135.0 | 240.0 | 34.573 | -0.027 | 64.0 | 8.0 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80; DAP22 I30/N40; DAP29 I30/N0 |
| current_50k | transfer_2011_2020 | 2017 | 9042.615 | -21.823 | 135.0 | 240.0 | 37.678 | 1.078 | 64.0 | 8.0 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80; DAP22 I30/N40; DAP29 I30/N0 |
| current_50k | transfer_2011_2020 | 2018 | 7871.586 | -45.817 | 135.0 | 240.0 | 32.798 | 0.798 | 64.0 | 8.0 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80; DAP22 I30/N40; DAP29 I30/N0 |
| current_50k | transfer_2011_2020 | 2019 | 9805.219 | -0.186 | 135.0 | 240.0 | 40.855 | 1.555 | 64.0 | 8.0 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80; DAP22 I30/N40; DAP29 I30/N0 |
| current_50k | transfer_2011_2020 | 2020 | 7664.061 | -111.156 | 135.0 | 240.0 | 31.934 | 0.534 | 64.0 | 8.0 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80; DAP22 I30/N0; DAP29 I30/N40 |
| yield_plus_50k | train_2005_2010 | 2005 | 5725.987 | -6358.4 | 150.0 | 40.0 | 143.15 | 94.35 | 79.0 | 208.0 | DAP1 I45/N0; DAP8 I15/N40; DAP16 I15/N0; DAP23 I15/N0; DAP30 I15/N0; DAP37 I15/N0; DAP44 I15/N0; DAP51 I15/N0 |
| yield_plus_50k | train_2005_2010 | 2006 | 5290.887 | -4153.957 | 150.0 | 40.0 | 132.272 | 94.072 | 49.0 | 208.0 | DAP1 I45/N0; DAP8 I15/N40; DAP16 I15/N0; DAP23 I15/N0; DAP30 I15/N0; DAP37 I15/N0; DAP44 I15/N0; DAP51 I15/N0 |
| yield_plus_50k | train_2005_2010 | 2007 | 5413.128 | -2692.386 | 150.0 | 40.0 | 135.328 | 102.528 | 49.0 | 208.0 | DAP1 I45/N0; DAP8 I15/N40; DAP18 I15/N0; DAP25 I15/N0; DAP32 I15/N0; DAP39 I15/N0; DAP46 I15/N0; DAP53 I15/N0 |
| yield_plus_50k | train_2005_2010 | 2008 | 5493.044 | -5370.542 | 150.0 | 40.0 | 137.326 | 93.426 | 49.0 | 208.0 | DAP1 I45/N0; DAP8 I15/N40; DAP18 I15/N0; DAP25 I15/N0; DAP32 I15/N0; DAP39 I15/N0; DAP46 I15/N0; DAP53 I15/N0 |
| yield_plus_50k | train_2005_2010 | 2009 | 5368.966 | -3901.419 | 150.0 | 40.0 | 134.224 | 96.724 | 49.0 | 208.0 | DAP1 I45/N0; DAP8 I15/N40; DAP16 I15/N0; DAP23 I15/N0; DAP30 I15/N0; DAP37 I15/N0; DAP44 I15/N0; DAP51 I15/N0 |
| yield_plus_50k | train_2005_2010 | 2010 | 5239.077 | -3499.923 | 150.0 | 40.0 | 130.977 | 95.596 | 48.8 | 207.0 | DAP1 I45/N0; DAP8 I15/N40; DAP15 I15/N0; DAP22 I15/N0; DAP29 I15/N0; DAP36 I15/N0; DAP43 I15/N0; DAP50 I15/N0 |
| yield_plus_50k | transfer_2011_2020 | 2011 | 5118.206 | -4221.794 | 150.0 | 40.0 | 127.955 | 82.96 | 49.0 | 208.0 | DAP1 I45/N0; DAP8 I15/N40; DAP16 I15/N0; DAP23 I15/N0; DAP30 I15/N0; DAP37 I15/N0; DAP44 I15/N0; DAP51 I15/N0 |
| yield_plus_50k | transfer_2011_2020 | 2012 | 7537.797 | -1479.907 | 150.0 | 80.0 | 94.222 | 57.822 | 79.0 | 168.0 | DAP1 I45/N0; DAP8 I15/N40; DAP15 I15/N40; DAP22 I15/N0; DAP29 I15/N0; DAP36 I15/N0; DAP43 I15/N0; DAP50 I15/N0 |
| yield_plus_50k | transfer_2011_2020 | 2013 | 4823.54 | -3899.952 | 150.0 | 40.0 | 120.589 | 85.389 | 49.0 | 208.0 | DAP1 I45/N0; DAP8 I15/N40; DAP18 I15/N0; DAP25 I15/N0; DAP32 I15/N0; DAP39 I15/N0; DAP46 I15/N0; DAP53 I15/N0 |
| yield_plus_50k | transfer_2011_2020 | 2014 | 5594.259 | -4928.168 | 150.0 | 40.0 | 139.856 | 97.356 | 49.0 | 208.0 | DAP1 I45/N0; DAP8 I15/N40; DAP17 I15/N0; DAP24 I15/N0; DAP31 I15/N0; DAP38 I15/N0; DAP45 I15/N0; DAP52 I15/N0 |
| yield_plus_50k | transfer_2011_2020 | 2015 | 5639.602 | -4731.786 | 150.0 | 40.0 | 140.99 | 99.09 | 49.0 | 208.0 | DAP1 I45/N0; DAP8 I15/N40; DAP17 I15/N0; DAP24 I15/N0; DAP31 I15/N0; DAP38 I15/N0; DAP45 I15/N0; DAP52 I15/N0 |
| yield_plus_50k | transfer_2011_2020 | 2016 | 5208.309 | -3357.166 | 150.0 | 40.0 | 130.208 | 95.608 | 49.0 | 208.0 | DAP1 I45/N0; DAP8 I15/N40; DAP16 I15/N0; DAP23 I15/N0; DAP30 I15/N0; DAP37 I15/N0; DAP44 I15/N0; DAP51 I15/N0 |
| yield_plus_50k | transfer_2011_2020 | 2017 | 5480.949 | -3583.489 | 150.0 | 40.0 | 137.024 | 100.424 | 49.0 | 208.0 | DAP1 I45/N0; DAP8 I15/N40; DAP16 I15/N0; DAP23 I15/N0; DAP30 I15/N0; DAP37 I15/N0; DAP44 I15/N0; DAP51 I15/N0 |
| yield_plus_50k | transfer_2011_2020 | 2018 | 4706.563 | -3210.84 | 150.0 | 40.0 | 117.664 | 85.664 | 49.0 | 208.0 | DAP1 I45/N0; DAP8 I15/N40; DAP15 I15/N0; DAP22 I15/N0; DAP29 I15/N0; DAP36 I15/N0; DAP43 I15/N0; DAP50 I15/N0 |
| yield_plus_50k | transfer_2011_2020 | 2019 | 4890.871 | -4914.534 | 150.0 | 40.0 | 122.272 | 82.972 | 49.0 | 208.0 | DAP1 I45/N0; DAP8 I15/N40; DAP15 I15/N0; DAP22 I15/N0; DAP29 I15/N0; DAP36 I15/N0; DAP43 I15/N0; DAP50 I15/N0 |
| yield_plus_50k | transfer_2011_2020 | 2020 | 4117.859 | -3657.359 | 150.0 | 40.0 | 102.946 | 71.546 | 49.0 | 208.0 | DAP1 I45/N0; DAP8 I15/N40; DAP17 I15/N0; DAP24 I15/N0; DAP31 I15/N0; DAP38 I15/N0; DAP45 I15/N0; DAP52 I15/N0 |
| resource_cheaper_50k | train_2005_2010 | 2005 | 11801.306 | -283.081 | 150.0 | 240.0 | 49.172 | 0.372 | 79.0 | 8.0 | DAP1 I45/N80; DAP8 I15/N40; DAP15 I15/N40; DAP22 I0/N40; DAP23 I15/N0; DAP29 I0/N40; DAP30 I15/N0; DAP37 I45/N0 |
| resource_cheaper_50k | train_2005_2010 | 2006 | 9171.9 | -272.944 | 150.0 | 240.0 | 38.216 | 0.016 | 49.0 | 8.0 | DAP1 I45/N80; DAP8 I15/N120; DAP15 I15/N40; DAP22 I15/N0; DAP29 I15/N0; DAP36 I15/N0; DAP43 I15/N0; DAP50 I15/N0 |
| resource_cheaper_50k | train_2005_2010 | 2007 | 8053.099 | -52.415 | 150.0 | 240.0 | 33.555 | 0.755 | 49.0 | 8.0 | DAP1 I45/N80; DAP8 I15/N120; DAP15 I15/N40; DAP22 I30/N0; DAP29 I15/N0; DAP36 I15/N0; DAP43 I15/N0 |
| resource_cheaper_50k | train_2005_2010 | 2008 | 10478.341 | -385.245 | 150.0 | 240.0 | 43.66 | -0.24 | 49.0 | 8.0 | DAP1 I45/N80; DAP8 I15/N120; DAP15 I15/N40; DAP22 I30/N0; DAP29 I15/N0; DAP36 I15/N0; DAP43 I15/N0 |
| resource_cheaper_50k | train_2005_2010 | 2009 | 9274.54 | 4.155 | 150.0 | 240.0 | 38.644 | 1.144 | 49.0 | 8.0 | DAP1 I45/N80; DAP8 I15/N40; DAP15 I15/N40; DAP22 I0/N40; DAP23 I15/N0; DAP29 I0/N40; DAP30 I15/N0; DAP37 I15/N0; DAP44 I15/N0; DAP51 I15/N0 |
| resource_cheaper_50k | train_2005_2010 | 2010 | 8329.326 | -409.674 | 150.0 | 240.0 | 34.706 | -0.675 | 48.8 | 7.0 | DAP1 I45/N80; DAP8 I15/N120; DAP15 I0/N40; DAP16 I30/N0; DAP23 I15/N0; DAP30 I15/N0; DAP37 I15/N0; DAP44 I15/N0 |
| resource_cheaper_50k | transfer_2011_2020 | 2011 | 9029.811 | -310.189 | 150.0 | 240.0 | 37.624 | -7.371 | 49.0 | 8.0 | DAP1 I45/N80; DAP8 I15/N120; DAP15 I15/N40; DAP22 I15/N0; DAP29 I15/N0; DAP36 I15/N0; DAP43 I15/N0; DAP50 I15/N0 |
| resource_cheaper_50k | transfer_2011_2020 | 2012 | 8979.681 | -38.022 | 150.0 | 240.0 | 37.415 | 1.015 | 79.0 | 8.0 | DAP1 I45/N80; DAP8 I15/N120; DAP15 I15/N40; DAP22 I30/N0; DAP29 I15/N0; DAP36 I15/N0; DAP43 I15/N0 |
| resource_cheaper_50k | transfer_2011_2020 | 2013 | 8621.569 | -101.923 | 150.0 | 240.0 | 35.923 | 0.723 | 49.0 | 8.0 | DAP1 I45/N80; DAP8 I15/N120; DAP15 I15/N40; DAP22 I15/N0; DAP29 I15/N0; DAP36 I15/N0; DAP43 I15/N0; DAP50 I15/N0 |
| resource_cheaper_50k | transfer_2011_2020 | 2014 | 10522.673 | 0.247 | 150.0 | 240.0 | 43.844 | 1.344 | 49.0 | 8.0 | DAP1 I45/N80; DAP8 I15/N120; DAP15 I15/N40; DAP22 I15/N0; DAP29 I15/N0; DAP36 I45/N0 |
| resource_cheaper_50k | transfer_2011_2020 | 2015 | 10368.701 | -2.687 | 150.0 | 240.0 | 43.203 | 1.303 | 49.0 | 8.0 | DAP1 I45/N80; DAP8 I15/N120; DAP15 I15/N40; DAP22 I15/N0; DAP29 I15/N0; DAP36 I15/N0; DAP43 I15/N0; DAP50 I15/N0 |
| resource_cheaper_50k | transfer_2011_2020 | 2016 | 8347.053 | -218.422 | 150.0 | 240.0 | 34.779 | 0.179 | 49.0 | 8.0 | DAP1 I45/N80; DAP8 I15/N40; DAP15 I15/N40; DAP22 I0/N40; DAP23 I15/N0; DAP29 I0/N40; DAP30 I15/N0; DAP37 I15/N0; DAP44 I15/N0; DAP51 I15/N0 |
| resource_cheaper_50k | transfer_2011_2020 | 2017 | 9020.624 | -43.814 | 150.0 | 240.0 | 37.586 | 0.986 | 49.0 | 8.0 | DAP1 I45/N80; DAP8 I15/N120; DAP15 I15/N40; DAP22 I15/N0; DAP29 I15/N0; DAP36 I45/N0 |
| resource_cheaper_50k | transfer_2011_2020 | 2018 | 7913.564 | -3.839 | 150.0 | 240.0 | 32.973 | 0.973 | 49.0 | 8.0 | DAP1 I45/N80; DAP8 I15/N120; DAP15 I0/N40; DAP16 I30/N0; DAP23 I15/N0; DAP30 I15/N0; DAP37 I15/N0; DAP44 I15/N0 |
| resource_cheaper_50k | transfer_2011_2020 | 2019 | 9782.43 | -22.975 | 150.0 | 240.0 | 40.76 | 1.46 | 49.0 | 8.0 | DAP1 I45/N80; DAP8 I15/N120; DAP15 I0/N40; DAP16 I30/N0; DAP23 I15/N0; DAP30 I15/N0; DAP37 I15/N0; DAP44 I15/N0 |
| resource_cheaper_50k | transfer_2011_2020 | 2020 | 7731.04 | -44.177 | 150.0 | 240.0 | 32.213 | 0.813 | 49.0 | 8.0 | DAP1 I45/N80; DAP8 I15/N120; DAP15 I15/N40; DAP22 I15/N0; DAP29 I15/N0; DAP36 I15/N0; DAP43 I15/N0; DAP50 I15/N0 |

## Interpretation boundary

- A yield-biased variant should only be extended if it improves mean yield gap while retaining water and N savings versus official expert.
- Do not add more variants based on these results without a new pre-registration.
