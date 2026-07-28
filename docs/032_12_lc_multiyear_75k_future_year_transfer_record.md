# 032_12 LC multi-year 75k free-timing MaskablePPO future-year transfer record

## Status

- Evaluation pass: `True`.
- Four-baseline comparison table complete: `False`.
- Training in this task: 0 timesteps.

## Source model

- Model: `benchmark_results/032_11_lc_multiyear_free_timing_ppo_training_length/models/LCA/LCA_multiyear_2005_2010_stress_aware_maskableppo_seed0_ckpt75000.zip`.
- Source training years: LC2005-LC2010.
- Source checkpoint: 75,000 timesteps from 032_11.
- Algorithm: no-forecast free-timing MaskablePPO.

## Target years

- 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020.

## Summary versus official expert

| years | yield_ge_expert | water_saving_vs_expert | n_saving_vs_expert | PFP_N_ge_expert | mean_yield_gap_vs_expert | mean_water_saving_vs_expert | mean_n_saving_vs_expert |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 3 | 10 | 10 | 10 | -14.742 | 127.0 | 48.0 |

## Per-year comparison versus official expert

| year | rl_yield_kg_ha | expert_yield_kg_ha | yield_gap_vs_expert | rl_irrigation_mm | expert_irrigation_mm | water_saving_vs_expert | rl_nitrogen_kg_ha | expert_nitrogen_kg_ha | n_saving_vs_expert | rl_PFP_N_kg_kg | expert_PFP_N_kg_kg | PFP_N_gap_vs_expert | rl_max_water_stress | rl_max_nitrogen_stress |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2011 | 9049.867 | 9103.966 | -54.099 | 75.0 | 199.0 | 124.0 | 200.0 | 248.0 | 48.0 | 45.249 | 36.8 | 8.449 | 0.0 | 0.023 |
| 2012 | 8995.84 | 9017.703 | -21.863 | 75.0 | 229.0 | 154.0 | 200.0 | 248.0 | 48.0 | 44.979 | 36.4 | 8.579 | 0.0 | 0.012 |
| 2013 | 8765.942 | 8723.492 | 42.45 | 75.0 | 199.0 | 124.0 | 200.0 | 248.0 | 48.0 | 43.83 | 35.2 | 8.63 | 0.081 | 0.012 |
| 2014 | 10487.562 | 10522.427 | -34.865 | 75.0 | 199.0 | 124.0 | 200.0 | 248.0 | 48.0 | 52.438 | 42.5 | 9.938 | 0.0 | 0.088 |
| 2015 | 10112.574 | 10371.388 | -258.813 | 75.0 | 199.0 | 124.0 | 200.0 | 248.0 | 48.0 | 50.563 | 41.9 | 8.663 | 0.0 | 0.012 |
| 2016 | 8389.412 | 8565.475 | -176.063 | 75.0 | 199.0 | 124.0 | 200.0 | 248.0 | 48.0 | 41.947 | 34.6 | 7.347 | 0.0 | 0.012 |
| 2017 | 9047.879 | 9064.438 | -16.559 | 75.0 | 199.0 | 124.0 | 200.0 | 248.0 | 48.0 | 45.239 | 36.6 | 8.639 | 0.0 | 0.012 |
| 2018 | 8233.118 | 7917.404 | 315.715 | 75.0 | 199.0 | 124.0 | 200.0 | 248.0 | 48.0 | 41.166 | 32.0 | 9.166 | 0.106 | 0.012 |
| 2019 | 9807.0 | 9725.751 | 81.249 | 75.0 | 199.0 | 124.0 | 200.0 | 248.0 | 48.0 | 49.035 | 39.3 | 9.735 | 0.0 | 0.117 |
| 2020 | 7750.649 | 7775.217 | -24.568 | 75.0 | 199.0 | 124.0 | 200.0 | 248.0 | 48.0 | 38.753 | 31.4 | 7.353 | 0.0 | 0.012 |

## Frozen PPO endpoint details

| target_year | run_status | final_grnwt | total_irrigation | total_n | PFP_N | reward_stress_aware_sum | max_swfac | max_nstres | action_sequence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2011 | ok | 9049.867 | 75.0 | 200.0 | 45.249 | 1.18 | 0.0 | 0.023 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80 |
| 2012 | ok | 8995.84 | 75.0 | 200.0 | 44.979 | 1.174 | 0.0 | 0.012 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80 |
| 2013 | ok | 8765.942 | 75.0 | 200.0 | 43.83 | 1.139 | 0.081 | 0.012 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80 |
| 2014 | ok | 10487.562 | 75.0 | 200.0 | 52.438 | 1.41 | 0.0 | 0.088 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80 |
| 2015 | ok | 10112.574 | 75.0 | 200.0 | 50.563 | 1.348 | 0.0 | 0.012 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80 |
| 2016 | ok | 8389.412 | 75.0 | 200.0 | 41.947 | 1.076 | 0.0 | 0.012 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80 |
| 2017 | ok | 9047.879 | 75.0 | 200.0 | 45.239 | 1.18 | 0.0 | 0.012 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80 |
| 2018 | ok | 8233.118 | 75.0 | 200.0 | 41.166 | 1.052 | 0.106 | 0.012 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80 |
| 2019 | ok | 9807.0 | 75.0 | 200.0 | 49.035 | 1.301 | 0.0 | 0.117 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80 |
| 2020 | ok | 7750.649 | 75.0 | 200.0 | 38.753 | 0.976 | 0.0 | 0.012 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80 |

## Interpretation boundary

- This task evaluates one frozen 75k checkpoint only.
- It does not retrain on 2011-2020.
- It does not include weather forecast features.
- WP_ET for RL is not reported here unless ET denominator is available in the replay outputs; no ET value is inferred.
