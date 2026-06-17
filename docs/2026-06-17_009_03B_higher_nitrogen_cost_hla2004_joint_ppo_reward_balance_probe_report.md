# 009_03B_higher_nitrogen_cost HLA 2004 Joint PPO Reward Balance Probe

## Purpose

This smoke test reintroduces PPO-controlled supplemental nitrogen while preserving the 008 stage-level soft-stress irrigation framework.
S1/S2 use diagnostic agronomic base nitrogen. S3-S5 allow PPO supplemental N. PPO also controls irrigation under the no-hard-minimum forecast/stress gate.

## Training Summary

| station | year | seed | total_timesteps | run_status | model_path | notes |
| --- | --- | --- | --- | --- | --- | --- |
| HLA | 2004 | 0 | 5000 | ok | Leave_One_experiments/hla2004_joint_ppo_reward_balance_probe_009_03/scenarios/009_03B_higher_nitrogen_cost/models/HLA/ppo_joint_water_n_HLA_2004_seed0.zip |  |

## Evaluation Summary

| station | year | seed | run_status | stage_steps | daily_steps | total_irrigation | total_ppo_extra_irrigation | total_n | total_base_n | total_ppo_extra_n | final_grnwt | final_topwt | grnwt_fraction_vs_00815_irrigation_only | profit_low_water_cost | max_swfac | swfac_days_gt_0p05 | max_nstres | nstres_days_gt_0p05 | soft_swfac_penalty_total | soft_nstres_penalty_total | rain_total_in_episode | first_irrigation_dap | first_ppo_n_dap | irrigation_cap_saturated | n_cap_saturated | early_irrigation_before_allowed_window | gate_blocked_irrigation | nstres_warning | debug_joint_promising | daily_csv_path | stage_csv_path | model_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HLA | 2004 | 0 | ok | 5 | 175 | 81.1406 | 81.1406 | 190.0 | 100.0 | 90.0 | 7067.1625 | 17161.2964 | 1.0427 | 15.0576 | 1.0 | 23 | 0.4354 | 51 | 68.7242 | 39.8429 | 51.5 | 21.0 | 46.0 | False | True | False | 0.0 | True | False | Leave_One_experiments/hla2004_joint_ppo_reward_balance_probe_009_03/scenarios/009_03B_higher_nitrogen_cost/daily_outputs/HLA/HLA_2004_seed0_soft_stress_stage_daily.csv | Leave_One_experiments/hla2004_joint_ppo_reward_balance_probe_009_03/scenarios/009_03B_higher_nitrogen_cost/daily_outputs/HLA/HLA_2004_seed0_soft_stress_stage_steps.csv | Leave_One_experiments/hla2004_joint_ppo_reward_balance_probe_009_03/scenarios/009_03B_higher_nitrogen_cost/models/HLA/ppo_joint_water_n_HLA_2004_seed0.zip |

## Stage Decisions

| stage_id | decision_dap | raw_stage_action_irrigation | raw_stage_action_nitrogen | stage_action_amir | stage_action_anfer | stage_nitrogen_base | ppo_extra_n_applied | irrigation_before_gate | irrigation_after_gate | gate_future_rain | gate_forecast_trigger | gate_swfac_at_decision | soft_swfac_penalty | soft_nstres_penalty | reward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | 1 | -1.0 | -0.7033 | 0.0 | 50.0 | 50.0 | 0.0 | 0.0 | 0.0 | 0.0 | False | 0.39 | 0.0 | 0.0 | 0.0196 |
| S2 | 21 | -0.943 | 1.0 | 1.1406 | 50.0 | 50.0 | 0.0 | 1.1406 | 1.1406 | 0.0 | True | 0.0 | 0.0 | 0.0 | 0.3601 |
| S3 | 46 | -1.0 | 1.0 | 0.0 | 50.0 | 0.0 | 50.0 | 0.0 | 0.0 | 0.0 | True | 0.0 | 0.0 | 3.0175 | -0.9708 |
| S4 | 76 | 1.0 | 1.0 | 40.0 | 20.0 | 0.0 | 20.0 | 40.0 | 40.0 | 0.0 | True | 0.0 | 0.0 | 0.6615 | 6.3163 |
| S5 | 101 | 1.0 | 1.0 | 40.0 | 20.0 | 0.0 | 20.0 | 40.0 | 40.0 | 0.0 | True | 0.0 | 68.7242 | 36.1639 | 106.3264 |

## Interpretation

- Total irrigation: 81.14 mm.
- Total N: 190.00 kg/ha = base 100.00 + PPO extra 90.00.
- Final GRNWT: 7067.16.
- GRNWT / 008_15 irrigation-only PPO: 1.043.
- SWFAC stress days: 23; NSTRES stress days: 51.
- Soft SWFAC penalty total: 68.72; soft NSTRES penalty total: 39.84.
- First irrigation DAP: 21.0; first PPO N DAP: 46.0.
- Debug joint promising: False.
- The first water-nitrogen joint smoke test did not fully meet the predefined criteria; inspect action space before tuning reward.

## Files

- Training summary: `Leave_One_experiments/hla2004_joint_ppo_reward_balance_probe_009_03/scenarios/009_03B_higher_nitrogen_cost/evaluation/HLA_2004_joint_ppo_training.csv`
- Evaluation summary: `Leave_One_experiments/hla2004_joint_ppo_reward_balance_probe_009_03/scenarios/009_03B_higher_nitrogen_cost/evaluation/HLA_2004_joint_ppo_summary.csv`
- Daily outputs: `Leave_One_experiments/hla2004_joint_ppo_reward_balance_probe_009_03/scenarios/009_03B_higher_nitrogen_cost/daily_outputs/HLA`
- Figures: `Leave_One_experiments/hla2004_joint_ppo_reward_balance_probe_009_03/scenarios/009_03B_higher_nitrogen_cost/figures`