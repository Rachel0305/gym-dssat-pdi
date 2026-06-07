# Constrained PPO multiseed with augmented prior report

Generated at: 2026-06-06

## Scope

This stage used only `BC_random_forest_regressor_augmented` as the prior, only HLA/SYA/LCA, and seeds `[0, 1, 2]`. It did not train unrestricted PPO, did not train FQA/YCA, and did not enter rainfall-scaling.

## Pretrain smoke checks

| method | policy_name | count | ok |
| --- | --- | --- | --- |
| MS0_augmented_RF_prior_replay | fixed_low_input | 9 | 9 |
| MS0_augmented_RF_prior_replay | null_zero | 9 | 9 |
| MS0_augmented_RF_prior_replay | prior_replay | 9 | 9 |
| MS1_residual_augmented_RF_prior_strict | fixed_low_input | 9 | 9 |
| MS1_residual_augmented_RF_prior_strict | null_zero | 9 | 9 |
| MS1_residual_augmented_RF_prior_strict | prior_replay | 9 | 9 |
| MS2_guardrail_augmented_RF_100_200 | fixed_low_input | 9 | 9 |
| MS2_guardrail_augmented_RF_100_200 | null_zero | 9 | 9 |
| MS2_guardrail_augmented_RF_100_200 | prior_replay | 9 | 9 |

## Multiseed method comparison

| method | mean_yield | mean_profit | mean_irrigation | mean_n | mean_yield_loss_vs_ppo | failure_rate | cap_regression_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MS0_augmented_RF_prior_replay | 6500.5829 | 44.259 | 0.0 | 82.9872 | 0.0546 | 0.0 | 0.0 |
| MS1_residual_augmented_RF_prior_strict | 6462.8087 | 34.2437 | 17.2507 | 87.0362 | 0.0601 | 0.0 | 0.0 |
| MS2_guardrail_augmented_RF_100_200 | 8498.0336 | -15.0197 | 100.0 | 200.0 | -0.2359 | 0.0 | 0.0 |

## Policy comparison

| policy_type | policy_name | eval_count | mean_yield | mean_irrigation | mean_n | mean_profit |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_or_prior | best_expert_schedule_replay | 10 | 8225.2271 | 0.0 | 150.0 | 44.7523 |
| baseline_or_prior | BC_random_forest_regressor_augmented | 10 | 6503.2767 | 0.0 | 81.8291 | 44.5755 |
| baseline_or_prior | BC_two_stage_classifier_regressor_original | 10 | 8091.8537 | 0.0 | 153.575 | 42.5248 |
| multiseed_constrained_ppo | MS0_augmented_RF_prior_replay | 30 | 6503.2767 | 0.0 | 81.8291 | 44.5755 |
| multiseed_constrained_ppo | MS1_residual_augmented_RF_prior_strict | 30 | 6435.6918 | 16.2374 | 84.6443 | 35.0771 |
| multiseed_constrained_ppo | MS2_guardrail_augmented_RF_100_200 | 30 | 8529.1433 | 100.0 | 200.0 | -14.7086 |
| ppo_cap_saturated_baseline | old_ppo_cap_saturated_baseline | 90 | 6875.8716 | 300.0 | 450.0 | -193.7413 |

## Recommended policy

| method | sites | site_pass_count | mean_yield | mean_profit | mean_irrigation | mean_n | mean_yield_loss_vs_ppo | failure_rate | cap_regression_rate | cv_yield | passes_gate | recommendation | next_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MS0_augmented_RF_prior_replay | 3 | 2 | 6500.5829 | 44.259 | 0.0 | 82.9872 | 0.0546 | 0.0 | 0.0 | 0.0355 | False | retain_augmented_rf_prior_as_current_best_profit_policy | do_not_enter_rainfall_scaling; run_irrigation_responsive_expert_search_and_fix_HLA_yield_gap |

## Interpretation

If MS1 or MS2 does not clearly improve profit/yield over MS0 and the augmented RF prior, the safer recommendation is to keep the augmented RF prior. Rainfall-scaling remains premature when irrigation actions are still near zero; the next stage should search for irrigation-responsive expert schedules before stress-scenario RL.
