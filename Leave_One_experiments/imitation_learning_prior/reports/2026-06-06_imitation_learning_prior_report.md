# Imitation learning prior report

Generated at: 2026-06-06

## Goal

This stage trains supervised behavior cloning priors from offline deterministic expert schedules. It does not train PPO, does not run multi-seed PPO, does not enter rainfall scaling, and does not modify my_data or the original reward files.

## Expert schedules

- HLA: HLA2011_S0157
- SYA: SYA2012_S0443
- LCA: LCA2010_S0313

The original 006_08 imitation dataset contained HLA only, so this stage rebuilt the clean BC dataset from HLA/SYA/LCA best-schedule daily outputs.

## Supervised metrics

| model_name | split | n_rows | irrigation_mae | irrigation_rmse | irrigation_event_precision | irrigation_event_recall | irrigation_event_f1 | nitrogen_mae | nitrogen_rmse | nitrogen_event_precision | nitrogen_event_recall | nitrogen_event_f1 | nonzero_action_accuracy | zero_action_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BC_constant_schedule_baseline | train | 396 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| BC_random_forest_regressor | train | 396 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.7265 | 7.1013 | 0.8 | 1.0 | 0.8889 | 0.9975 | 0.9974 |
| BC_mlp_regressor | train | 396 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.1364 | 11.9183 | 0.0 | 0.0 | 0.0 | 0.9899 | 1.0 |
| BC_two_stage_classifier_regressor | train | 396 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1496 | 1.5237 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| BC_constant_schedule_baseline | validation | 904 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| BC_random_forest_regressor | validation | 904 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.1292 | 9.2659 | 0.4286 | 0.6 | 0.5 | 0.9867 | 0.9911 |
| BC_mlp_regressor | validation | 904 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.1615 | 11.7001 | 0.0 | 0.0 | 0.0 | 0.9889 | 1.0 |
| BC_two_stage_classifier_regressor | validation | 904 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.664 | 8.3521 | 1.0 | 0.5 | 0.6667 | 0.9945 | 1.0 |

## DSSAT/gym-DSSAT evaluation summary

| policy_name | eval_count | mean_yield | mean_irrigation | mean_n | mean_yield_loss_vs_expert | mean_yield_loss_vs_ppo |
| --- | --- | --- | --- | --- | --- | --- |
| BC_constant_schedule_baseline | 10 | 8225.2271 | 0.0 | 150.0 | 0.0 | -0.1962 |
| BC_mlp_regressor | 10 | 2059.4187 | 0.0 | 0.0 | 0.7641 | 0.7005 |
| BC_random_forest_regressor | 10 | 6115.2526 | 0.0 | 85.8948 | 0.26 | 0.1106 |
| BC_two_stage_classifier_regressor | 10 | 8091.8537 | 0.0 | 153.575 | 0.0184 | -0.1768 |

## Policy comparison

| policy_type | policy_name | final_grnwt | total_irrigation | total_n_fertilizer | profit_score |
| --- | --- | --- | --- | --- | --- |
| expert_schedule | HLA2011_S0157 | 6752.7806 | 0.0 | 150.0 | 30.0278 |
| expert_schedule | LCA2010_S0313 | 8757.0909 | 0.0 | 150.0 | 50.0709 |
| expert_schedule | SYA2012_S0443 | 8988.5219 | 0.0 | 150.0 | 52.3852 |
| imitation_policy | BC_constant_schedule_baseline | 8225.2271 | 0.0 | 150.0 | 44.7523 |
| imitation_policy | BC_mlp_regressor | 2059.4187 | 0.0 | 0.0 | 20.5942 |
| imitation_policy | BC_random_forest_regressor | 6115.2526 | 0.0 | 85.8948 | 39.6788 |
| imitation_policy | BC_two_stage_classifier_regressor | 8091.8537 | 0.0 | 153.575 | 42.5248 |
| ppo_cap_saturated_baseline | reference_300mm_450kgN | 6875.8716 | 300.0 | 450.0 | -193.7413 |

## Cap saturation check

No successful imitation policy exceeded the 300 mm irrigation / 450 kg ha-1 N safety cap. The main practical threshold for the next stage is stricter: mean irrigation <= 100 mm, mean N <= 200 kg/ha, and mean yield loss vs PPO reference <= 15%.

Policies passing the constrained PPO fine-tuning gate:

| policy_name | eval_count | mean_yield | mean_irrigation | mean_n | mean_yield_loss_vs_expert | mean_yield_loss_vs_ppo |
| --- | --- | --- | --- | --- | --- | --- |
| BC_constant_schedule_baseline | 10 | 8225.2271 | 0.0 | 150.0 | 0.0 | -0.1962 |
| BC_random_forest_regressor | 10 | 6115.2526 | 0.0 | 85.8948 | 0.26 | 0.1106 |
| BC_two_stage_classifier_regressor | 10 | 8091.8537 | 0.0 | 153.575 | 0.0184 | -0.1768 |

## Interpretation

BC_constant_schedule_baseline is an expert replay baseline, not a learned general policy. Random forest, MLP, and the two-stage classifier-regressor are supervised priors. Because action samples are extremely sparse and irrigation is zero in the selected expert schedules, these results should be treated as an interpretable policy prior rather than a final RL policy.

## Recommendation

If at least one learned imitation policy passes the gate, the next step can be constrained PPO fine-tuning from imitation prior. If only the constant schedule passes, the next step should be expert dataset augmentation before RL fine-tuning.
