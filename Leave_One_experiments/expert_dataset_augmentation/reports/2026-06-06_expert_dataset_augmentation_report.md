# Expert dataset augmentation report

Generated at: 2026-06-06

## Why not PPO multi-seed yet

006_11 fixed the FT0 replay mismatch, but fixed FT2 and FT3 still did not outperform the BC prior. The current limitation is not seed instability; it is that the imitation prior was trained from very few schedules. This stage therefore augments expert schedules before returning to constrained PPO.

## Selected expert schedules

| station | selected_schedules |
| --- | --- |
| HLA | 15 |
| LCA | 15 |
| SYA | 15 |

## Dataset coverage

| station | year |
| --- | --- |
| HLA | 2007 |
| HLA | 2009 |
| HLA | 2011 |
| LCA | 2008 |
| LCA | 2009 |
| LCA | 2010 |
| LCA | 2011 |
| SYA | 2012 |
| SYA | 2014 |
| SYA | 2015 |

## Action distribution

| dataset | rows | stations | station_years | schedules | nonzero_irrigation_rows | nonzero_n_rows | unique_irrigation_amounts | unique_n_amounts | mean_positive_irrigation | mean_positive_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 00609_original | 1300 | 3 | 10 | 3 | 0 | 14 | 1 | 3 | 0.0 | 107.1429 |
| 00612_augmented | 19500 | 3 | 10 | 45 | 87 | 252 | 2 | 3 | 50.0 | 98.8095 |

## Supervised BC metrics

| model_name | split | n_rows | irrigation_mae | irrigation_rmse | irrigation_event_precision | irrigation_event_recall | irrigation_event_f1 | nitrogen_mae | nitrogen_rmse | nitrogen_event_precision | nitrogen_event_recall | nitrogen_event_f1 | nonzero_action_accuracy | zero_action_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BC_constant_schedule_baseline_augmented | train_year | 5940 | 0.6734 | 5.8026 | 0.2381 | 1.0 | 0.3846 | 0.7323 | 7.785 | 0.7733 | 0.7632 | 0.7682 | 0.9806 | 0.9834 |
| BC_random_forest_regressor_augmented | train_year | 5940 | 0.0306 | 1.0007 | 1.0 | 0.92 | 0.9583 | 0.2664 | 3.3104 | 0.8837 | 1.0 | 0.9383 | 0.998 | 0.9983 |
| BC_mlp_regressor_augmented | train_year | 5940 | 0.2104 | 3.2437 | 0.0 | 0.0 | 0.0 | 1.2626 | 11.8386 | 0.0 | 0.0 | 0.0 | 0.983 | 1.0 |
| BC_two_stage_classifier_regressor_augmented | train_year | 5940 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.1061 | 2.4874 | 0.962 | 1.0 | 0.9806 | 0.9995 | 0.9995 |
| BC_constant_schedule_baseline_augmented | cross_year_validation | 13560 | 0.7117 | 5.9651 | 0.2431 | 1.0 | 0.3912 | 0.7412 | 7.8354 | 0.7879 | 0.7386 | 0.7625 | 0.9798 | 0.9829 |
| BC_random_forest_regressor_augmented | cross_year_validation | 13560 | 0.0918 | 1.918 | 0.8033 | 0.7903 | 0.7967 | 0.8009 | 6.5878 | 0.5612 | 0.8864 | 0.6872 | 0.9877 | 0.9899 |
| BC_mlp_regressor_augmented | cross_year_validation | 13560 | 0.2286 | 3.3809 | 0.0 | 0.0 | 0.0 | 1.2832 | 11.9457 | 0.0 | 0.0 | 0.0 | 0.9824 | 1.0 |
| BC_two_stage_classifier_regressor_augmented | cross_year_validation | 13560 | 0.1659 | 2.8804 | 1.0 | 0.2742 | 0.4304 | 0.6427 | 8.0661 | 0.9674 | 0.5057 | 0.6642 | 0.99 | 0.9998 |

## DSSAT/gym-DSSAT policy gate

| policy_name | eval_count | mean_yield | mean_irrigation | mean_n | mean_profit | mean_yield_loss_vs_ppo | max_irrigation | max_n | mean_input_reduction_vs_ppo | passes_gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BC_mlp_regressor_augmented | 10 | 2059.4187 | 0.0 | 0.0 | 20.5942 | 0.7005 | 0.0 | 0.0 | 1.0 | False |
| BC_random_forest_regressor_augmented | 10 | 6503.2767 | 0.0 | 81.8291 | 44.5755 | 0.0542 | 0.0 | 108.5183 | 0.8909 | True |
| BC_two_stage_classifier_regressor_augmented | 10 | 6273.5947 | 0.0 | 75.0 | 43.9859 | 0.0876 | 0.0 | 75.0 | 0.9 | True |
| BC_two_stage_classifier_regressor_original | 10 | 8091.8537 | 0.0 | 153.575 | 42.5248 | -0.1768 | 0.0 | 180.5 | 0.7952 | True |
| best_expert_schedule_replay | 10 | 8225.2271 | 0.0 | 150.0 | 44.7523 | -0.1962 | 0.0 | 150.0 | 0.8 | True |

## Recommended policy

| policy_name | eval_count | mean_yield | mean_irrigation | mean_n | mean_profit | mean_yield_loss_vs_ppo | max_irrigation | max_n | mean_input_reduction_vs_ppo | passes_gate | recommendation | next_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BC_random_forest_regressor_augmented | 10 | 6503.2767 | 0.0 | 81.8291 | 44.5755 | 0.0542 | 0.0 | 108.5183 | 0.8909 | True | augmented_learned_policy | can_consider_00613_constrained_ppo_multiseed |

## Interpretation

The augmented dataset is richer than 006_09 in schedule count, station-year trajectories, nonzero action rows, and unique water/N action levels. If no augmented learned policy passes the profit and input gate, the correct next step is still to retain original BC/expert replay and improve the expert library, rather than running constrained PPO multi-seed prematurely.

For this run, an augmented learned policy can be considered for the next constrained PPO multi-seed stage only if it appears in the recommended policy table. Rainfall-scaling is still premature because irrigation events remain sparse in the augmented expert library; the next water-stress work should first add or search schedules from genuinely irrigation-responsive years/scenarios.
