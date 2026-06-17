# 008_20 FQA 2016 Irrigation-Only Soft-Stress PPO Second Water-Stress Year Validation

## Purpose

This lightweight run tests whether the HLA 2004 soft-stress stage PPO framework can produce an interpretable irrigation policy in a second water-stress year.
Nitrogen is fixed at N150 by stage prior. PPO controls irrigation only. No hard minimum irrigation gate is used.

## Training Summary

| station | year | seed | total_timesteps | run_status | model_path | notes |
| --- | --- | --- | --- | --- | --- | --- |
| FQA | 2016 | 0 | 5000 | ok | Leave_One_experiments/fqa2016_irrigation_only_soft_stress_ppo_008_20/models/FQA/ppo_soft_stress_FQA_2016_seed0.zip |  |

## Evaluation Summary

| station | year | seed | run_status | stage_steps | daily_steps | total_irrigation | total_n | final_grnwt | final_topwt | profit_low_water_cost | max_swfac | swfac_days_gt_0p05 | max_nstres | nstres_days_gt_0p05 | rain_total_in_episode | first_irrigation_dap | irrigation_cap_saturated | n_cap_saturated | early_irrigation_before_allowed_window | gate_blocked_irrigation | daily_csv_path | stage_csv_path | model_path | rule_replay_total_irrigation | rule_replay_total_n | rule_replay_final_grnwt | rule_replay_swfac_days_gt_0p05 | grnwt_fraction_vs_rule_replay | pool_scenario_type | pool_has_water_stress | pool_irrigation_responsive | pool_yield_gain_from_irrigation_at_same_N | pool_swfac_stress_days_gt_0p05 | pool_max_swfac | debug_promising |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FQA | 2016 | 0 | ok | 4 | 98 | 80.0 | 150.0 | 6999.2743 | 13056.6638 | 24.4927 | 0.4309 | 9 | 0.0122 | 0 | 224.8 | 46.0 | False | False | False | 0.0 | Leave_One_experiments/fqa2016_irrigation_only_soft_stress_ppo_008_20/daily_outputs/FQA/FQA_2016_seed0_soft_stress_stage_daily.csv | Leave_One_experiments/fqa2016_irrigation_only_soft_stress_ppo_008_20/daily_outputs/FQA/FQA_2016_seed0_soft_stress_stage_steps.csv | Leave_One_experiments/fqa2016_irrigation_only_soft_stress_ppo_008_20/models/FQA/ppo_soft_stress_FQA_2016_seed0.zip | 90.0 | 150.0 | 6641.8073 | 12.0 | 1.0538 | normal_year;water_stress_year;nitrogen_stress_year;irrigation_responsive_year | True | True | 2081.2744 | 22.0 | 0.5806 | True |

## Stage Decisions

| stage_id | decision_dap | raw_stage_action_irrigation | stage_action_amir | stage_action_anfer | irrigation_before_gate | irrigation_after_gate | gate_future_rain | gate_forecast_trigger | gate_swfac_at_decision | soft_swfac_penalty | reward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | 1 | -1.0 | 0.0 | 50.0 | 0.0 | 0.0 | 2.9 | False | 0.21 | 0.0 | 0.1127 |
| S2 | 21 | -1.0 | 0.0 | 50.0 | 0.0 | 0.0 | 1.1 | True | 0.0 | 0.0 | 2.2966 |
| S3 | 46 | 1.0 | 40.0 | 50.0 | 40.0 | 40.0 | 0.0 | True | 0.0 | 6.5814 | 56.7694 |
| S4 | 76 | 1.0 | 40.0 | 0.0 | 40.0 | 40.0 | 14.0 | True | 0.4309 | 6.5824 | 146.6921 |

## Interpretation

- Total irrigation: 80.00 mm; total N: 150.00 kg/ha.
- Final GRNWT: 6999.27.
- GRNWT / 008_11 rule replay: 1.054.
- SWFAC stress days: 9; max SWFAC: 0.431.
- First irrigation DAP: 46.0.
- Debug promising: True.
- PPO produced a nonzero, non-saturated irrigation policy.

## Files

- Training summary: `Leave_One_experiments/fqa2016_irrigation_only_soft_stress_ppo_008_20/evaluation/FQA_2016_soft_stress_ppo_training.csv`
- Evaluation summary: `Leave_One_experiments/fqa2016_irrigation_only_soft_stress_ppo_008_20/evaluation/FQA_2016_soft_stress_ppo_summary.csv`
- Daily outputs: `Leave_One_experiments/fqa2016_irrigation_only_soft_stress_ppo_008_20/daily_outputs/FQA`
- Figures: `Leave_One_experiments/fqa2016_irrigation_only_soft_stress_ppo_008_20/figures`