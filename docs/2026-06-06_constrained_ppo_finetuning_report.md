# Constrained PPO fine-tuning report

Generated at: 2026-06-06

## Goal

This stage tests constrained PPO fine-tuning from imitation priors. It does not train unrestricted PPO, does not run multi-seed PPO, does not enter rainfall scaling, and does not modify my_data or previous 006_03-006_09 outputs.

## Prior selection

Main prior: `BC_two_stage_classifier_regressor`. Backup prior: `BC_random_forest_regressor`. `BC_constant_schedule_baseline` is only an expert replay reference, not the main learned prior.

## Methods tested

- FT0_BC_two_stage_replay: prior replay, no PPO training.
- FT1_bc_penalty_lambda_0p5: PPO with action deviation penalty around BC prior.
- FT2_residual_bc_prior: PPO residual action around BC prior.
- FT3_budget_guardrail_100_200: PPO with imitation penalty and strict 100/200 seasonal guardrail.

## Method summary

| finetune_method | eval_count | mean_yield | mean_profit | mean_irrigation | mean_n | mean_yield_loss_vs_ppo | mean_bc_deviation | max_i_sat | max_n_sat | profit_gap_vs_bc_two_stage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FT0_BC_two_stage_replay | 10 | 4447.8684 | 30.4787 | 0.0 | 56.0 | 0.3531 | 0.0053 | 0.0 | 0.1778 | -12.0461 |
| FT1_bc_penalty_lambda_0p5 | 10 | 8554.6497 | -132.4281 | 212.3623 | 447.1737 | -0.2442 | 0.0464 | 1.0 | 1.0 | -174.9529 |
| FT2_residual_bc_prior | 10 | 5075.0034 | -2.2712 | 74.8013 | 62.4825 | 0.2619 | 0.0126 | 0.548 | 0.18 | -44.796 |
| FT3_budget_guardrail_100_200 | 10 | 8557.4186 | -14.4258 | 100.0 | 200.0 | -0.2446 | 0.0257 | 0.3333 | 0.4444 | -56.9506 |

## Recommended method

No fine-tuned PPO method passed all gates.

## Interpretation

Methods are judged by run success, irrigation <= 100 mm, N <= 200 kg/ha, yield loss vs PPO baseline <= 15%, mean profit at least 90% of the BC_two_stage prior profit, and no 300/450 saturation. If a fine-tuned method increases inputs substantially without meaningful yield/profit gain over the BC prior, it is not recommended.

## Next step

If a recommended fine-tuning method exists, run constrained PPO multi-seed only for that method. If no fine-tuned PPO method passes all gates, do not run multi-seed yet; return to expert dataset augmentation, stronger profit/cost calibration, or a stricter residual/guardrail formulation. Rainfall-scaling should remain later, after the constrained method is stable.
