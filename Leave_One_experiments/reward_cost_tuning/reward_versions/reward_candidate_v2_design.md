# Reward candidate v2 design

Generated at: 2026-06-06

This stage does not overwrite the original gym-DSSAT reward. Candidate rewards are injected at runtime from `src/reward_candidates_v2.py`.

## Candidate families

- D: stronger linear water and nitrogen costs.
- E: linear costs plus cumulative quadratic costs over total seasonal water and nitrogen used so far.
- F: target interval penalty, with upper bounds 220 mm irrigation and 320 kg ha-1 N.
- G: normalized economic proxy using daily positive grain-yield increments minus water and nitrogen costs.

The G family is not a real RMB economic return yet. It is a normalized debug proxy and should later be replaced by interpretable grain, water, and fertilizer prices.

## Candidate table

| reward_version | reward_family | batch | kind | description | irrigation_cost_coef | nitrogen_cost_coef | cumulative_irrigation_coef | cumulative_n_coef | target_irrigation_upper | target_n_upper | target_irrigation_penalty_coef | target_n_penalty_coef | grain_value | water_cost | n_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| candidate_D1 | candidate_D_strong_linear_cost | batch_1_D | base_minus_cost | Strong linear cost D1. | 0.5 | 0.25 | 0.0 | 0.0 | 220.0 | 320.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| candidate_D2 | candidate_D_strong_linear_cost | batch_1_D | base_minus_cost | Strong linear cost D2. | 1.0 | 0.5 | 0.0 | 0.0 | 220.0 | 320.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| candidate_D3 | candidate_D_strong_linear_cost | batch_1_D | base_minus_cost | Strong linear cost D3. | 2.0 | 1.0 | 0.0 | 0.0 | 220.0 | 320.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| candidate_D4 | candidate_D_strong_linear_cost | batch_1_D | base_minus_cost | Strong linear cost D4. | 5.0 | 2.5 | 0.0 | 0.0 | 220.0 | 320.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| candidate_E1 | candidate_E_cumulative_quadratic_cost | batch_2_E | base_minus_cumulative | Cumulative quadratic cost E1. | 0.5 | 0.25 | 0.0005 | 0.0002 | 220.0 | 320.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| candidate_E2 | candidate_E_cumulative_quadratic_cost | batch_2_E | base_minus_cumulative | Cumulative quadratic cost E2. | 0.5 | 0.25 | 0.001 | 0.0005 | 220.0 | 320.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| candidate_E3 | candidate_E_cumulative_quadratic_cost | batch_2_E | base_minus_cumulative | Cumulative quadratic cost E3. | 0.5 | 0.25 | 0.005 | 0.002 | 220.0 | 320.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| candidate_E4 | candidate_E_cumulative_quadratic_cost | batch_2_E | base_minus_cumulative | Cumulative quadratic cost E4. | 0.5 | 0.25 | 0.01 | 0.005 | 220.0 | 320.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| candidate_F1 | candidate_F_target_interval_penalty | batch_3_F | base_minus_target_penalty | Weak target interval upper-bound penalty. | 0.5 | 0.25 | 0.0 | 0.0 | 220.0 | 320.0 | 0.02 | 0.01 | 0.0 | 0.0 | 0.0 |
| candidate_F2 | candidate_F_target_interval_penalty | batch_3_F | base_minus_target_penalty | Medium target interval upper-bound penalty. | 0.5 | 0.25 | 0.0 | 0.0 | 220.0 | 320.0 | 0.1 | 0.05 | 0.0 | 0.0 | 0.0 |
| candidate_F3 | candidate_F_target_interval_penalty | batch_3_F | base_minus_target_penalty | Strong target interval upper-bound penalty. | 0.5 | 0.25 | 0.0 | 0.0 | 220.0 | 320.0 | 0.5 | 0.25 | 0.0 | 0.0 | 0.0 |
| candidate_G1 | candidate_G_economic_proxy | batch_4_G | economic_proxy | Economic proxy G1. | 0.0 | 0.0 | 0.0 | 0.0 | 220.0 | 320.0 | 0.0 | 0.0 | 0.01 | 0.2 | 0.1 |
| candidate_G2 | candidate_G_economic_proxy | batch_4_G | economic_proxy | Economic proxy G2. | 0.0 | 0.0 | 0.0 | 0.0 | 220.0 | 320.0 | 0.0 | 0.0 | 0.01 | 0.5 | 0.25 |
| candidate_G3 | candidate_G_economic_proxy | batch_4_G | economic_proxy | Economic proxy G3. | 0.0 | 0.0 | 0.0 | 0.0 | 220.0 | 320.0 | 0.0 | 0.0 | 0.005 | 0.5 | 0.25 |
| candidate_G4 | candidate_G_economic_proxy | batch_4_G | economic_proxy | Economic proxy G4. | 0.0 | 0.0 | 0.0 | 0.0 | 220.0 | 320.0 | 0.0 | 0.0 | 0.005 | 1.0 | 0.5 |