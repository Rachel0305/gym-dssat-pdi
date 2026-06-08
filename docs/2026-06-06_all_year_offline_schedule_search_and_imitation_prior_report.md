# All-Year Offline Schedule Search and Imitation Prior Report

Generated at: 2026-06-06

## Scope

This stage uses the 006_15 all-year weather scenario pool to run deterministic schedule search and train imitation priors. It does not train PPO, does not enter rainfall scaling, and does not modify my_data.

## Station-Year Selection

- Selected station-years: 55
- Selected water-stress years: 20
- Selected irrigation-responsive years: 6
- Selected 2020-2023 calibration/validation candidates: 20

| station_code | station_years | water_stress | irrigation_responsive |
| --- | --- | --- | --- |
| FQA | 17 | 11 | 3 |
| HLA | 11 | 1 | 1 |
| LCA | 8 | 1 | 0 |
| SYA | 9 | 4 | 1 |
| YCA | 10 | 3 | 1 |

## Schedule Search

- Schedule evaluations completed or cached: 2180 / 2180
- Expert schedules retained: 217
- Expert schedules with nonzero irrigation: 124

The expert library intentionally keeps top yield, top low-water-cost profit, Pareto-balanced, low-input-within-10%-yield-loss, and irrigation-positive schedules. It therefore does not collapse to profit-only or irrigation=0-only schedules.

## Imitation Dataset

| dataset | path | rows | station_years | nonzero_irrigation_rows | nonzero_n_rows |
| --- | --- | --- | --- | --- | --- |
| 00609_original | Leave_One_experiments/imitation_learning_prior/datasets/imitation_dataset_clean.csv | 1300 | 10 | 0 | 14 |
| 00609_original_alt | Leave_One_experiments/offline_schedule_search/expert_policy/imitation_dataset.csv | 477 | 3 | 0 | 3 |
| 00612_augmented | Leave_One_experiments/expert_dataset_augmentation/imitation_dataset/imitation_dataset_augmented.csv | 19500 | 10 | 87 | 252 |
| 00616_all_year | Leave_One_experiments/all_year_offline_schedule_search/imitation_dataset/all_year_imitation_dataset.csv | 25298 | 55 | 148 | 405 |

## Supervised Imitation Metrics

| policy_name | train_rows | validation_rows | irrigation_mae | irrigation_rmse | irrigation_event_precision | irrigation_event_recall | irrigation_event_f1 | nitrogen_mae | nitrogen_rmse | nitrogen_event_precision | nitrogen_event_recall | nitrogen_event_f1 | nonzero_action_accuracy | zero_action_accuracy | water_stress_irrigation_event_recall | irrigation_responsive_irrigation_event_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BC_random_forest_regressor_all_year | 25298 | 11212 | 0.0323 | 0.6254 | 0.8923 | 0.9508 | 0.9206 | 0.299 | 3.1123 | 0.8533 | 1.0 | 0.9209 | 0.9962 | 0.9964 | 0.9167 | 0.7857 |
| BC_two_stage_classifier_regressor_all_year | 25298 | 11212 | 0.0032 | 0.1217 | 1.0 | 1.0 | 1.0 | 0.278 | 3.5462 | 0.8976 | 0.9583 | 0.927 | 0.9974 | 0.9981 | 1.0 | 1.0 |
| BC_mlp_regressor_all_year | 25298 | 11212 | 0.1589 | 2.1503 | 0.65 | 0.2131 | 0.321 | 0.9276 | 5.7527 | 0.418 | 0.8229 | 0.5544 | 0.9724 | 0.9793 | 0.0833 | 0.0 |

## DSSAT/gym-DSSAT Prior Evaluation

| policy_name | ok | mean_yield | mean_irrigation | mean_n | mean_yield_loss |
| --- | --- | --- | --- | --- | --- |
| BC_mlp_regressor_all_year | 24 | 6025.4486 | 0.0 | 118.4163 | -0.397 |
| BC_random_forest_regressor_all_year | 24 | 6410.4891 | 0.0 | 128.7098 | -0.419 |
| BC_two_stage_classifier_regressor_all_year | 24 | 4040.2756 | 0.0 | 64.615 | 0.1222 |

## Recommended Prior

| policy_name | recommended | mean_irrigation | mean_n | mean_yield_loss_vs_best_expert | water_stress_nonzero_irrigation_rate | supervised_irrigation_event_recall | reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BC_random_forest_regressor_all_year | False | 0.0 | 128.7098 | -0.419 | 0.0 | 0.9508 | does_not_pass_basic_all_year_prior_gate |
| BC_mlp_regressor_all_year | False | 0.0 | 118.4163 | -0.397 | 0.0 | 0.2131 | does_not_pass_basic_all_year_prior_gate |
| BC_two_stage_classifier_regressor_all_year | False | 0.0 | 64.615 | 0.1222 | 0.0 | 1.0 | does_not_pass_basic_all_year_prior_gate |

## Next Step

Recommended next route: `improve_all_year_expert_library_before_constrained_ppo`.