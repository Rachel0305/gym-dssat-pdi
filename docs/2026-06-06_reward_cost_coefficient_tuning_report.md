# Reward cost coefficient tuning report

Generated at: 2026-06-06

## Goal

This stage continues from 006_03 and tests stronger cost coefficients under HLA 2011 action-safe PPO with cap 300 mm irrigation / 450 kg ha-1 N. It does not modify the original site-packages reward and does not train PPO without action safety.

## Why 006_03 failed

All previous A/B/C candidates completed evaluation, but all used 300 mm irrigation and 450 kg ha-1 N. Stronger reward costs are needed, or the reward structure must be redesigned around a clearer season-level objective.

## Batches completed

batch_1_D, batch_2_E, batch_3_F, batch_4_G

## HLA comparison

| reward_version | reward_family | batch | eval_count | ok_count | mean_yield | std_yield | mean_reward | mean_irrigation | mean_n | mean_irrigation_saturation_ratio | mean_n_saturation_ratio | mean_swfac | mean_nstres | yield_loss_vs_baseline | input_reduction_vs_baseline | strict_pass | relaxed_pass | partial_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| candidate_D1 | candidate_D_strong_linear_cost | batch_1_D | 3 | 3 | 6843.8576 | 272.3337 | 65.2329 | 300.0 | 450.0 | 1.0 | 1.0 | 0.0 | 0.0007 | -0.0 | 0.0 | False | False | False |
| candidate_D2 | candidate_D_strong_linear_cost | batch_1_D | 3 | 3 | 6842.6005 | 273.4587 | 63.5326 | 300.0 | 450.0 | 1.0 | 1.0 | 0.0 | 0.0007 | 0.0001 | 0.0 | False | False | False |
| candidate_D3 | candidate_D_strong_linear_cost | batch_1_D | 3 | 3 | 6843.4416 | 272.5996 | 60.2551 | 300.0 | 450.0 | 1.0 | 1.0 | 0.0 | 0.0007 | 0.0 | 0.0 | False | False | False |
| candidate_D4 | candidate_D_strong_linear_cost | batch_1_D | 3 | 3 | 6839.8775 | 273.9532 | 50.2975 | 300.0 | 450.0 | 1.0 | 1.0 | 0.0 | 0.0007 | 0.0005 | 0.0 | False | False | False |
| candidate_E1 | candidate_E_cumulative_quadratic_cost | batch_2_E | 3 | 3 | 6877.3773 | 263.674 | 7.9724 | 300.0 | 450.0 | 1.0 | 1.0 | 0.0 | 0.0007 | -0.0049 | 0.0 | False | False | False |
| candidate_E2 | candidate_E_cumulative_quadratic_cost | batch_2_E | 3 | 3 | 6950.3805 | 266.2136 | -60.8583 | 300.0 | 450.0 | 1.0 | 1.0 | 0.0 | 0.0007 | -0.0156 | 0.0 | False | False | False |
| candidate_E3 | candidate_E_cumulative_quadratic_cost | batch_2_E | 3 | 3 | 6962.7024 | 264.7555 | -505.4392 | 300.0 | 450.0 | 1.0 | 1.0 | 0.0 | 0.0007 | -0.0174 | 0.0 | False | False | False |
| candidate_E4 | candidate_E_cumulative_quadratic_cost | batch_2_E | 3 | 3 | 6927.9366 | 264.4147 | -1252.0035 | 300.0 | 450.0 | 1.0 | 1.0 | 0.0 | 0.0007 | -0.0123 | 0.0 | False | False | False |
| candidate_F1 | candidate_F_target_interval_penalty | batch_3_F | 3 | 3 | 7025.4574 | 278.7463 | -92.7633 | 300.0 | 450.0 | 1.0 | 1.0 | 0.0 | 0.0006 | -0.0266 | 0.0 | False | False | False |
| candidate_F2 | candidate_F_target_interval_penalty | batch_3_F | 3 | 3 | 6938.8424 | 262.8576 | -798.4141 | 300.0 | 450.0 | 1.0 | 1.0 | 0.0 | 0.0007 | -0.0139 | 0.0 | False | False | False |
| candidate_F3 | candidate_F_target_interval_penalty | batch_3_F | 3 | 3 | 6900.4388 | 260.5327 | -4332.5399 | 300.0 | 450.0 | 1.0 | 1.0 | 0.0 | 0.0007 | -0.0083 | 0.0 | False | False | False |
| candidate_G1 | candidate_G_economic_proxy | batch_4_G | 3 | 3 | 6840.3776 | 274.2291 | -0.2301 | 300.0 | 450.0 | 1.0 | 1.0 | 0.0 | 0.0007 | 0.0005 | 0.0 | False | False | False |
| candidate_G2 | candidate_G_economic_proxy | batch_4_G | 3 | 3 | 6839.4826 | 265.9058 | -1.2233 | 300.0 | 450.0 | 1.0 | 1.0 | 0.0 | 0.0007 | 0.0006 | 0.0 | False | False | False |
| candidate_G3 | candidate_G_economic_proxy | batch_4_G | 3 | 3 | 6845.767 | 264.3655 | -1.4391 | 300.0 | 450.0 | 1.0 | 1.0 | 0.0 | 0.0007 | -0.0003 | 0.0 | False | False | False |
| candidate_G4 | candidate_G_economic_proxy | batch_4_G | 3 | 3 | 6840.6236 | 266.0175 | -3.0946 | 300.0 | 450.0 | 1.0 | 1.0 | 0.0 | 0.0007 | 0.0004 | 0.0 | False | False | False |

## Pass summary

- Strict pass candidates: None
- Relaxed pass candidates: None
- Partial pass candidates: None
- Recommended candidate: `None`

## SYA/LCA extension

| selected_reward | run_status | notes |
| --- | --- | --- |
|  | skipped | No HLA candidate passed strict or relaxed criteria. |

## Interpretation

If no candidate becomes unsaturated, coefficient tuning alone is insufficient. The likely next step is to redesign reward as an episode-level objective, such as final yield or profit minus total water and nitrogen costs, and expose terminal season information explicitly instead of approximating terminal reward from step-wise callbacks.

## Next step

Do not enter multi-seed or rainfall-scaling budget scenario unless HLA and the small SYA/LCA extension are no longer dominated by cap saturation.
