# 009_02 HLA 2004 Stage-Level Water-Nitrogen Joint PPO Smoke Test

## Purpose

This smoke test reintroduces PPO-controlled supplemental nitrogen while preserving the 008 stage-level soft-stress irrigation framework.
S1/S2 use diagnostic agronomic base nitrogen. S3-S5 allow PPO supplemental N. PPO also controls irrigation under the no-hard-minimum forecast/stress gate.

## Training Summary

| station | year | seed | total_timesteps | run_status | model_path | notes |
| --- | --- | --- | --- | --- | --- | --- |
| HLA | 2004 | 0 | 5000 | ok | Leave_One_experiments/hla2004_stage_level_water_nitrogen_joint_ppo_009_02/models/HLA/ppo_joint_water_n_HLA_2004_seed0.zip |  |

## Evaluation Summary

| station | year | seed | run_status | stage_steps | daily_steps | total_irrigation | total_ppo_extra_irrigation | total_n | total_base_n | total_ppo_extra_n | final_grnwt | final_topwt | grnwt_fraction_vs_00815_irrigation_only | profit_low_water_cost | max_swfac | swfac_days_gt_0p05 | max_nstres | nstres_days_gt_0p05 | soft_swfac_penalty_total | soft_nstres_penalty_total | rain_total_in_episode | first_irrigation_dap | first_ppo_n_dap | irrigation_cap_saturated | n_cap_saturated | early_irrigation_before_allowed_window | gate_blocked_irrigation | nstres_warning | debug_joint_promising | daily_csv_path | stage_csv_path | model_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HLA | 2004 | 0 | ok | 5 | 175 | 31.7846 | 31.7846 | 188.9366 | 100.0 | 88.9366 | 5305.6812 | 14526.3806 | 0.7828 | 2.6442 | 0.9474 | 39 | 0.166 | 7 | 119.9664 | 4.287 | 51.5 | 76.0 | 46.0 | False | False | False | 0.0 | False | True | Leave_One_experiments/hla2004_stage_level_water_nitrogen_joint_ppo_009_02/daily_outputs/HLA/HLA_2004_seed0_soft_stress_stage_daily.csv | Leave_One_experiments/hla2004_stage_level_water_nitrogen_joint_ppo_009_02/daily_outputs/HLA/HLA_2004_seed0_soft_stress_stage_steps.csv | Leave_One_experiments/hla2004_stage_level_water_nitrogen_joint_ppo_009_02/models/HLA/ppo_joint_water_n_HLA_2004_seed0.zip |

## Stage Decisions

| stage_id | decision_dap | raw_stage_action_irrigation | raw_stage_action_nitrogen | stage_action_amir | stage_action_anfer | stage_nitrogen_base | ppo_extra_n_applied | irrigation_before_gate | irrigation_after_gate | gate_future_rain | gate_forecast_trigger | gate_swfac_at_decision | soft_swfac_penalty | soft_nstres_penalty | reward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | 1 | -1.0 | 0.698 | 0.0 | 50.0 | 50.0 | 0.0 | 0.0 | 0.0 | 0.0 | False | 0.39 | 0.0 | 0.0 | 0.0196 |
| S2 | 21 | -1.0 | 1.0 | 0.0 | 50.0 | 50.0 | 0.0 | 0.0 | 0.0 | 0.0 | True | 0.0 | 0.0 | 0.0 | 0.4172 |
| S3 | 46 | -1.0 | 1.0 | 0.0 | 50.0 | 0.0 | 50.0 | 0.0 | 0.0 | 0.0 | True | 0.0 | 0.0 | 3.6129 | -0.6318 |
| S4 | 76 | -0.2054 | 0.9468 | 15.8923 | 19.4683 | 0.0 | 19.4683 | 15.8923 | 15.8923 | 0.0 | True | 0.0 | 7.7483 | 0.6741 | -1.4859 |
| S5 | 101 | -0.2054 | 0.9468 | 15.8923 | 19.4683 | 0.0 | 19.4683 | 15.8923 | 15.8923 | 0.0 | True | 0.6104 | 112.2181 | 0.0 | 46.8667 |

## Interpretation

- Total irrigation: 31.78 mm.
- Total N: 188.94 kg/ha = base 100.00 + PPO extra 88.94.
- Final GRNWT: 5305.68.
- GRNWT / 008_15 irrigation-only PPO: 0.783.
- SWFAC stress days: 39; NSTRES stress days: 7.
- Soft SWFAC penalty total: 119.97; soft NSTRES penalty total: 4.29.
- First irrigation DAP: 76.0; first PPO N DAP: 46.0.
- Debug joint promising: True.
- The first water-nitrogen joint smoke test is promising under the predefined criteria.

## Files

- Training summary: `Leave_One_experiments/hla2004_stage_level_water_nitrogen_joint_ppo_009_02/evaluation/HLA_2004_joint_ppo_training.csv`
- Evaluation summary: `Leave_One_experiments/hla2004_stage_level_water_nitrogen_joint_ppo_009_02/evaluation/HLA_2004_joint_ppo_summary.csv`
- Daily outputs: `Leave_One_experiments/hla2004_stage_level_water_nitrogen_joint_ppo_009_02/daily_outputs/HLA`
- Figures: `Leave_One_experiments/hla2004_stage_level_water_nitrogen_joint_ppo_009_02/figures`