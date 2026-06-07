# Multiseed prior and method plan

Generated at: 2026-06-06

## Prior choice

`BC_random_forest_regressor_augmented` is selected because 006_12 recommended it as the best augmented learned prior: it passed all HLA/SYA/LCA evaluations with mean irrigation 0.0, mean N about 81.83 kg/ha, yield loss vs PPO about 5.4%, and profit above the original BC prior.

`BC_mlp_regressor_augmented` is not used because it learned an almost zero-input policy with unacceptable yield loss. `BC_two_stage_classifier_regressor_augmented` is retained only as a backup because it passed the input gate but had lower yield/profit than the random forest prior.

Rainfall-scaling is still premature because 006_12 showed that irrigation labels remain sparse. This stage tests stability of the low-input learned prior before any rainfall stress scenario.

## Methods

- MS0_augmented_RF_prior_replay: pure prior replay, no PPO training.
- MS1_residual_augmented_RF_prior_strict: PPO learns a small residual around the RF prior under 100/200 season guardrails.
- MS2_guardrail_augmented_RF_100_200: PPO uses imitation penalty and strict 100/200 season guardrails.

Failure is defined as failed episodes, mean irrigation >100, mean N >200, yield loss vs PPO >15%, profit below 90% of the RF prior, or any regression toward 300/450 high-input behavior.
