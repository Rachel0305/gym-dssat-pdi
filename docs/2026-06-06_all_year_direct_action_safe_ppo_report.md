# All-Year Direct Action-Safe PPO Simple Baseline Report

## Scope

This stage trains direct action-safe PPO from gym-DSSAT states. It does not use RF, behavior cloning, expert replay, offline schedule search, constrained PPO from prior, rainfall scaling, reward grids, episode-level profit reward, low-frequency action design, seasonal budget action, or scheduled event action.

## Fixed Settings

- Timesteps per station: 5000
- Daily action scale: irrigation <= 40.0 mm/day; N <= 80.0 kg/ha/day.
- Season action safety cap: irrigation <= 160.0 mm; N <= 250.0 kg/ha.
- Reward: `0.1 * delta_topwt + delta_grnwt - 0.1 * irrigation - 0.25 * nitrogen`.

## Selected Train/Eval Years

| station_code | year | scenario_type | selected_for_train | selected_for_eval | selection_reason |
| --- | --- | --- | --- | --- | --- |
| FQA | 2005 | wet_year;water_stress_year;nitrogen_stress_year | False | True | selected_eval_priority_all_year_pool;water_stress_year;nitrogen_stress_year |
| FQA | 2007 | normal_year;water_stress_year;nitrogen_stress_year;irrigation_responsive_year | True | False | selected_train_priority_all_year_pool;water_stress_year;irrigation_responsive_year;nitrogen_stress_year |
| FQA | 2008 | normal_year;water_stress_year;nitrogen_stress_year;irrigation_responsive_year | True | False | selected_train_priority_all_year_pool;water_stress_year;irrigation_responsive_year;nitrogen_stress_year |
| FQA | 2011 | wet_year;water_stress_year;nitrogen_stress_year | False | True | selected_eval_priority_all_year_pool;water_stress_year;nitrogen_stress_year |
| FQA | 2015 | normal_year;water_stress_year;nitrogen_stress_year | False | True | selected_eval_priority_all_year_pool;water_stress_year;nitrogen_stress_year |
| FQA | 2016 | normal_year;water_stress_year;nitrogen_stress_year;irrigation_responsive_year | True | False | selected_train_priority_all_year_pool;water_stress_year;irrigation_responsive_year;nitrogen_stress_year |
| FQA | 2019 | dry_year;water_stress_year;nitrogen_stress_year | False | True | selected_eval_priority_all_year_pool;water_stress_year;nitrogen_stress_year |
| FQA | 2020 | normal_year;water_stress_year;nitrogen_stress_year | True | False | selected_train_priority_all_year_pool;water_stress_year;nitrogen_stress_year;2020_2023_calibration_validation_candidate |
| FQA | 2023 | normal_year;water_stress_year;nitrogen_stress_year | True | False | selected_train_priority_all_year_pool;water_stress_year;nitrogen_stress_year;2020_2023_calibration_validation_candidate |
| HLA | 2004 | dry_year;water_stress_year;nitrogen_stress_year;irrigation_responsive_year | True | False | selected_train_priority_all_year_pool;water_stress_year;irrigation_responsive_year;nitrogen_stress_year |
| HLA | 2012 | wet_year;nitrogen_stress_year;low_response_year | False | True | selected_eval_priority_all_year_pool;nitrogen_stress_year |
| HLA | 2013 | wet_year;nitrogen_stress_year;low_response_year | False | True | selected_eval_priority_all_year_pool;nitrogen_stress_year |
| HLA | 2015 | dry_year;nitrogen_stress_year;low_response_year | False | True | selected_eval_priority_all_year_pool;nitrogen_stress_year |
| HLA | 2018 | wet_year;nitrogen_stress_year;low_response_year | False | True | selected_eval_priority_all_year_pool;nitrogen_stress_year |
| HLA | 2020 | wet_year;nitrogen_stress_year;low_response_year | True | False | selected_train_priority_all_year_pool;nitrogen_stress_year;2020_2023_calibration_validation_candidate |
| HLA | 2021 | wet_year;nitrogen_stress_year;low_response_year | True | False | selected_train_priority_all_year_pool;nitrogen_stress_year;2020_2023_calibration_validation_candidate |
| HLA | 2022 | normal_year;nitrogen_stress_year;low_response_year | True | False | selected_train_priority_all_year_pool;nitrogen_stress_year;2020_2023_calibration_validation_candidate |
| HLA | 2023 | wet_year;nitrogen_stress_year;low_response_year | True | False | selected_train_priority_all_year_pool;nitrogen_stress_year;2020_2023_calibration_validation_candidate |
| LCA | 2006 | wet_year;nitrogen_stress_year;low_response_year | False | True | selected_eval_priority_all_year_pool;nitrogen_stress_year |
| LCA | 2008 | wet_year;nitrogen_stress_year;low_response_year | False | True | selected_eval_priority_all_year_pool;nitrogen_stress_year |
| LCA | 2009 | wet_year;nitrogen_stress_year;low_response_year | False | True | selected_eval_priority_all_year_pool;nitrogen_stress_year |
| LCA | 2013 | wet_year;nitrogen_stress_year;low_response_year | False | True | selected_eval_priority_all_year_pool;nitrogen_stress_year |
| LCA | 2017 | dry_year;water_stress_year;nitrogen_stress_year | True | False | selected_train_priority_all_year_pool;water_stress_year;nitrogen_stress_year |
| LCA | 2020 | normal_year;nitrogen_stress_year;low_response_year | True | False | selected_train_priority_all_year_pool;nitrogen_stress_year;2020_2023_calibration_validation_candidate |
| LCA | 2021 | wet_year;nitrogen_stress_year;low_response_year | True | False | selected_train_priority_all_year_pool;nitrogen_stress_year;2020_2023_calibration_validation_candidate |
| LCA | 2022 | wet_year;nitrogen_stress_year;low_response_year | True | False | selected_train_priority_all_year_pool;nitrogen_stress_year;2020_2023_calibration_validation_candidate |
| LCA | 2023 | wet_year;nitrogen_stress_year;low_response_year | True | False | selected_train_priority_all_year_pool;nitrogen_stress_year;2020_2023_calibration_validation_candidate |
| SYA | 2008 | wet_year;nitrogen_stress_year;low_response_year | False | True | selected_eval_priority_all_year_pool;nitrogen_stress_year |
| SYA | 2009 | normal_year;water_stress_year;nitrogen_stress_year | True | False | selected_train_priority_all_year_pool;water_stress_year;nitrogen_stress_year |
| SYA | 2010 | wet_year;nitrogen_stress_year;low_response_year | False | True | selected_eval_priority_all_year_pool;nitrogen_stress_year |
| SYA | 2012 | wet_year;nitrogen_stress_year;low_response_year | False | True | selected_eval_priority_all_year_pool;nitrogen_stress_year |
| SYA | 2014 | normal_year;water_stress_year;nitrogen_stress_year | True | False | selected_train_priority_all_year_pool;water_stress_year;nitrogen_stress_year |
| SYA | 2017 | dry_year;water_stress_year;nitrogen_stress_year;irrigation_responsive_year | True | False | selected_train_priority_all_year_pool;water_stress_year;irrigation_responsive_year;nitrogen_stress_year |
| SYA | 2020 | normal_year;water_stress_year;nitrogen_stress_year | True | False | selected_train_priority_all_year_pool;water_stress_year;nitrogen_stress_year;2020_2023_calibration_validation_candidate |
| SYA | 2022 | wet_year;nitrogen_stress_year;low_response_year | True | False | selected_train_priority_all_year_pool;nitrogen_stress_year;2020_2023_calibration_validation_candidate |
| SYA | 2023 | normal_year;nitrogen_stress_year;low_response_year | False | True | selected_eval_priority_all_year_pool;nitrogen_stress_year;2020_2023_calibration_validation_candidate |
| YCA | 2004 | dry_year;nitrogen_stress_year;irrigation_responsive_year | True | False | selected_train_priority_all_year_pool;irrigation_responsive_year;nitrogen_stress_year |
| YCA | 2005 | wet_year;nitrogen_stress_year;low_response_year | False | True | selected_eval_priority_all_year_pool;nitrogen_stress_year |
| YCA | 2010 | wet_year;nitrogen_stress_year;low_response_year | False | True | selected_eval_priority_all_year_pool;nitrogen_stress_year |
| YCA | 2012 | wet_year;nitrogen_stress_year;low_response_year | False | True | selected_eval_priority_all_year_pool;nitrogen_stress_year |
| YCA | 2014 | normal_year;water_stress_year;nitrogen_stress_year | True | False | selected_train_priority_all_year_pool;water_stress_year;nitrogen_stress_year |
| YCA | 2015 | normal_year;water_stress_year;nitrogen_stress_year | True | False | selected_train_priority_all_year_pool;water_stress_year;nitrogen_stress_year |
| YCA | 2019 | dry_year;water_stress_year;nitrogen_stress_year | True | False | selected_train_priority_all_year_pool;water_stress_year;nitrogen_stress_year |
| YCA | 2022 | wet_year;nitrogen_stress_year;low_response_year | True | False | selected_train_priority_all_year_pool;nitrogen_stress_year;2020_2023_calibration_validation_candidate |
| YCA | 2023 | normal_year;nitrogen_stress_year;low_response_year | False | True | selected_eval_priority_all_year_pool;nitrogen_stress_year;2020_2023_calibration_validation_candidate |

## Training Summary

| station_code | train_years | seed | total_timesteps | run_status | model_path | mean_train_reward | final_train_reward | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FQA | 2007,2008,2016,2020,2023 | 0 | 5000 | ok | Leave_One_experiments/all_year_direct_action_safe_ppo/models/FQA/ppo_direct_action_safe_seed0.zip |  |  |  |
| HLA | 2004,2020,2021,2022,2023 | 0 | 5000 | ok | Leave_One_experiments/all_year_direct_action_safe_ppo/models/HLA/ppo_direct_action_safe_seed0.zip |  |  |  |
| LCA | 2017,2020,2021,2022,2023 | 0 | 5000 | ok | Leave_One_experiments/all_year_direct_action_safe_ppo/models/LCA/ppo_direct_action_safe_seed0.zip |  |  |  |
| SYA | 2009,2014,2017,2020,2022 | 0 | 5000 | ok | Leave_One_experiments/all_year_direct_action_safe_ppo/models/SYA/ppo_direct_action_safe_seed0.zip |  |  |  |
| YCA | 2004,2014,2015,2019,2022 | 0 | 5000 | ok | Leave_One_experiments/all_year_direct_action_safe_ppo/models/YCA/ppo_direct_action_safe_seed0.zip |  |  |  |

## Evaluation Summary

| station_code | year | split | run_status | final_grnwt | total_irrigation | total_n | profit_simple | swfac_stress_days_gt_0p05 | nstres_days_gt_0p05 | daily_csv_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FQA | 2007 | train | ok | 7078.9526 | 160.0 | 250.0 | -7.7105 | 0 | 0 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/FQA/2007_ppo_daily.csv |
| FQA | 2008 | train | ok | 7234.4885 | 160.0 | 250.0 | -6.1551 | 0 | 0 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/FQA/2008_ppo_daily.csv |
| FQA | 2016 | train | ok | 7358.6218 | 160.0 | 250.0 | -4.9138 | 0 | 0 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/FQA/2016_ppo_daily.csv |
| FQA | 2020 | train | ok | 7990.318 | 160.0 | 250.0 | 1.4032 | 0 | 0 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/FQA/2020_ppo_daily.csv |
| FQA | 2023 | train | ok | 8224.3243 | 160.0 | 250.0 | 3.7432 | 0 | 0 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/FQA/2023_ppo_daily.csv |
| FQA | 2005 | eval | ok | 7339.7455 | 160.0 | 250.0 | -5.1025 | 0 | 0 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/FQA/2005_ppo_daily.csv |
| FQA | 2011 | eval | ok | 5627.3279 | 160.0 | 250.0 | -22.2267 | 0 | 0 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/FQA/2011_ppo_daily.csv |
| FQA | 2015 | eval | ok | 6931.5387 | 160.0 | 250.0 | -9.1846 | 0 | 0 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/FQA/2015_ppo_daily.csv |
| FQA | 2019 | eval | ok | 7603.8507 | 160.0 | 250.0 | -2.4615 | 0 | 0 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/FQA/2019_ppo_daily.csv |
| HLA | 2004 | train | ok | 6024.1089 | 160.0 | 250.0 | -18.2589 | 33 | 0 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/HLA/2004_train_ppo_daily.csv |
| HLA | 2020 | train | ok | 6866.9489 | 160.0 | 250.0 | -9.8305 | 0 | 20 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/HLA/2020_train_ppo_daily.csv |
| HLA | 2021 | train | ok | 5059.6362 | 160.0 | 250.0 | -27.9036 | 0 | 0 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/HLA/2021_train_ppo_daily.csv |
| HLA | 2022 | train | ok | 7107.605 | 160.0 | 250.0 | -7.424 | 0 | 5 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/HLA/2022_train_ppo_daily.csv |
| HLA | 2023 | train | ok | 6077.9376 | 160.0 | 250.0 | -17.7206 | 0 | 0 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/HLA/2023_train_ppo_daily.csv |
| HLA | 2012 | eval | ok | 0.0 | 160.0 | 250.0 | -78.5 | 0 | 0 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/HLA/2012_ppo_daily.csv |
| HLA | 2013 | eval | ok | 5172.4792 | 160.0 | 250.0 | -26.7752 | 0 | 0 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/HLA/2013_ppo_daily.csv |
| HLA | 2015 | eval | ok | 6969.3353 | 160.0 | 250.0 | -8.8066 | 0 | 21 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/HLA/2015_ppo_daily.csv |
| HLA | 2018 | eval | ok | 6115.9259 | 160.0 | 250.0 | -17.3407 | 0 | 12 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/HLA/2018_ppo_daily.csv |
| LCA | 2017 | train | ok | 9008.5815 | 160.0 | 250.0 | 11.5858 | 0 | 0 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/LCA/2017_ppo_daily.csv |
| LCA | 2020 | train | ok | 7666.1603 | 160.0 | 250.0 | -1.8384 | 0 | 0 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/LCA/2020_ppo_daily.csv |
| LCA | 2021 | train | ok | 10986.4722 | 160.0 | 250.0 | 31.3647 | 0 | 0 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/LCA/2021_ppo_daily.csv |
| LCA | 2022 | train | ok | 9990.0183 | 160.0 | 250.0 | 21.4002 | 0 | 0 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/LCA/2022_ppo_daily.csv |
| LCA | 2023 | train | ok | 7888.3557 | 160.0 | 250.0 | 0.3836 | 0 | 0 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/LCA/2023_ppo_daily.csv |
| LCA | 2006 | eval | ok | 8996.9904 | 160.0 | 250.0 | 11.4699 | 0 | 0 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/LCA/2006_ppo_daily.csv |
| LCA | 2008 | eval | ok | 10768.1665 | 160.0 | 250.0 | 29.1817 | 1 | 0 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/LCA/2008_ppo_daily.csv |
| LCA | 2009 | eval | ok | 9222.0728 | 160.0 | 250.0 | 13.7207 | 0 | 0 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/LCA/2009_ppo_daily.csv |
| LCA | 2013 | eval | ok | 8756.2109 | 160.0 | 250.0 | 9.0621 | 1 | 0 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/LCA/2013_ppo_daily.csv |
| SYA | 2009 | train | ok | 12165.1257 | 160.0 | 250.0 | 43.1513 | 7 | 7 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/SYA/2009_ppo_daily.csv |
| SYA | 2014 | train | ok | 10363.6267 | 160.0 | 250.0 | 25.1363 | 8 | 9 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/SYA/2014_ppo_daily.csv |
| SYA | 2017 | train | ok | 10518.1335 | 160.0 | 250.0 | 26.6813 | 4 | 0 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/SYA/2017_ppo_daily.csv |
| SYA | 2020 | train | ok | 10000.9149 | 160.0 | 250.0 | 21.5091 | 0 | 0 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/SYA/2020_ppo_daily.csv |
| SYA | 2022 | train | ok | 10597.2278 | 160.0 | 250.0 | 27.4723 | 0 | 9 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/SYA/2022_ppo_daily.csv |
| SYA | 2008 | eval | ok | 11073.4253 | 160.0 | 250.0 | 32.2343 | 0 | 0 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/SYA/2008_ppo_daily.csv |
| SYA | 2010 | eval | ok | 8490.4968 | 160.0 | 250.0 | 6.405 | 0 | 0 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/SYA/2010_ppo_daily.csv |
| SYA | 2012 | eval | ok | 10106.0101 | 160.0 | 250.0 | 22.5601 | 0 | 0 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/SYA/2012_ppo_daily.csv |
| SYA | 2023 | eval | ok | 10738.2886 | 160.0 | 250.0 | 28.8829 | 0 | 0 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/SYA/2023_ppo_daily.csv |
| YCA | 2004 | train | ok | 8064.6497 | 160.0 | 250.0 | 2.1465 | 7 | 16 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/YCA/2004_ppo_daily.csv |
| YCA | 2014 | train | ok | 9381.2738 | 160.0 | 250.0 | 15.3127 | 0 | 0 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/YCA/2014_ppo_daily.csv |
| YCA | 2015 | train | ok | 9082.2668 | 160.0 | 250.0 | 12.3227 | 0 | 0 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/YCA/2015_ppo_daily.csv |
| YCA | 2019 | train | ok | 7598.8336 | 160.0 | 250.0 | -2.5117 | 1 | 0 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/YCA/2019_ppo_daily.csv |
| YCA | 2022 | train | ok | 8496.1908 | 160.0 | 250.0 | 6.4619 | 0 | 8 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/YCA/2022_ppo_daily.csv |
| YCA | 2005 | eval | ok | 9111.3995 | 160.0 | 250.0 | 12.614 | 1 | 0 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/YCA/2005_ppo_daily.csv |
| YCA | 2010 | eval | ok | 7246.4764 | 160.0 | 250.0 | -6.0352 | 1 | 0 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/YCA/2010_ppo_daily.csv |
| YCA | 2012 | eval | ok | 7861.5112 | 160.0 | 250.0 | 0.1151 | 1 | 0 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/YCA/2012_ppo_daily.csv |
| YCA | 2023 | eval | ok | 9577.1661 | 160.0 | 250.0 | 17.2717 | 0 | 5 | Leave_One_experiments/all_year_direct_action_safe_ppo/daily_outputs/YCA/2023_ppo_daily.csv |

## Decision Reasonableness

| station_code | year | scenario_type | total_irrigation | total_n | irrigation_event_count | n_event_count | swfac_stress_days_gt_0p05 | nstres_days_gt_0p05 | irrigation_during_or_before_swfac_stress | fertilization_during_or_before_nstres | first_irrigation_dap | first_n_dap | peak_swfac_dap | peak_nstres_dap | final_grnwt | final_topwt | profit_simple | decision_reasonableness_label | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FQA | 2007 | normal_year;water_stress_year;nitrogen_stress_year;irrigation_responsive_year | 160.0 | 250.0 | 8 | 6 | 0 | 0 | True | True | 1 | 1 | 1 | 5 | 7078.9526 | 13001.0681 | -7.7105 | cap_saturated |  |
| FQA | 2008 | normal_year;water_stress_year;nitrogen_stress_year;irrigation_responsive_year | 160.0 | 250.0 | 8 | 6 | 0 | 0 | True | True | 1 | 1 | 1 | 6 | 7234.4885 | 12876.2927 | -6.1551 | cap_saturated |  |
| FQA | 2016 | normal_year;water_stress_year;nitrogen_stress_year;irrigation_responsive_year | 160.0 | 250.0 | 8 | 6 | 0 | 0 | True | True | 1 | 1 | 1 | 5 | 7358.6218 | 13380.6445 | -4.9138 | cap_saturated |  |
| FQA | 2020 | normal_year;water_stress_year;nitrogen_stress_year | 160.0 | 250.0 | 8 | 6 | 0 | 0 | True | True | 1 | 1 | 1 | 6 | 7990.318 | 13846.0181 | 1.4032 | cap_saturated |  |
| FQA | 2023 | normal_year;water_stress_year;nitrogen_stress_year | 160.0 | 250.0 | 8 | 6 | 0 | 0 | True | True | 1 | 1 | 1 | 5 | 8224.3243 | 15156.759 | 3.7432 | cap_saturated |  |
| FQA | 2005 | wet_year;water_stress_year;nitrogen_stress_year | 160.0 | 250.0 | 8 | 6 | 0 | 0 | True | True | 1 | 1 | 1 | 5 | 7339.7455 | 13150.2612 | -5.1025 | cap_saturated |  |
| FQA | 2011 | wet_year;water_stress_year;nitrogen_stress_year | 160.0 | 250.0 | 8 | 6 | 0 | 0 | True | True | 1 | 1 | 1 | 6 | 5627.3279 | 11107.0703 | -22.2267 | cap_saturated |  |
| FQA | 2015 | normal_year;water_stress_year;nitrogen_stress_year | 160.0 | 250.0 | 8 | 6 | 0 | 0 | True | True | 1 | 1 | 1 | 6 | 6931.5387 | 12318.8391 | -9.1846 | cap_saturated |  |
| FQA | 2019 | dry_year;water_stress_year;nitrogen_stress_year | 160.0 | 250.0 | 8 | 6 | 0 | 0 | True | True | 1 | 1 | 1 | 11 | 7603.8507 | 14116.1902 | -2.4615 | cap_saturated |  |
| HLA | 2004 | dry_year;water_stress_year;nitrogen_stress_year;irrigation_responsive_year | 160.0 | 250.0 | 9 | 6 | 33 | 0 | True | True | 1 | 1 | 132 | 15 | 6024.1089 | 16486.1938 | -18.2589 | cap_saturated |  |
| HLA | 2020 | wet_year;nitrogen_stress_year;low_response_year | 160.0 | 250.0 | 9 | 6 | 0 | 20 | True | True | 1 | 1 | 1 | 136 | 6866.9489 | 16591.7212 | -9.8305 | cap_saturated |  |
| HLA | 2021 | wet_year;nitrogen_stress_year;low_response_year | 160.0 | 250.0 | 9 | 6 | 0 | 0 | True | True | 1 | 1 | 1 | 14 | 5059.6362 | 13995.8813 | -27.9036 | cap_saturated |  |
| HLA | 2022 | normal_year;nitrogen_stress_year;low_response_year | 160.0 | 250.0 | 9 | 6 | 0 | 5 | True | True | 1 | 1 | 1 | 135 | 7107.605 | 17494.5532 | -7.424 | cap_saturated |  |
| HLA | 2023 | wet_year;nitrogen_stress_year;low_response_year | 160.0 | 250.0 | 9 | 6 | 0 | 0 | True | True | 1 | 1 | 1 | 14 | 6077.9376 | 16521.1926 | -17.7206 | cap_saturated |  |
| HLA | 2012 | wet_year;nitrogen_stress_year;low_response_year | 160.0 | 250.0 | 9 | 6 | 0 | 0 | True | True | 1 | 1 | 1 | 15 | 0.0 | 1474.6779 | -78.5 | cap_saturated |  |
| HLA | 2013 | wet_year;nitrogen_stress_year;low_response_year | 160.0 | 250.0 | 9 | 6 | 0 | 0 | True | True | 1 | 1 | 1 | 12 | 5172.4792 | 13653.8831 | -26.7752 | cap_saturated |  |
| HLA | 2015 | dry_year;nitrogen_stress_year;low_response_year | 160.0 | 250.0 | 9 | 6 | 0 | 21 | True | True | 1 | 1 | 1 | 156 | 6969.3353 | 17900.3882 | -8.8066 | cap_saturated |  |
| HLA | 2018 | wet_year;nitrogen_stress_year;low_response_year | 160.0 | 250.0 | 9 | 6 | 0 | 12 | True | True | 1 | 1 | 1 | 128 | 6115.9259 | 15486.3757 | -17.3407 | cap_saturated |  |
| LCA | 2017 | dry_year;water_stress_year;nitrogen_stress_year | 160.0 | 250.0 | 9 | 6 | 0 | 0 | True | True | 1 | 1 | 1 | 5 | 9008.5815 | 17173.4814 | 11.5858 | cap_saturated |  |
| LCA | 2020 | normal_year;nitrogen_stress_year;low_response_year | 160.0 | 250.0 | 9 | 6 | 0 | 0 | True | True | 1 | 1 | 1 | 6 | 7666.1603 | 16590.3369 | -1.8384 | cap_saturated |  |
| LCA | 2021 | wet_year;nitrogen_stress_year;low_response_year | 160.0 | 250.0 | 9 | 6 | 0 | 0 | True | True | 1 | 1 | 1 | 5 | 10986.4722 | 19569.8279 | 31.3647 | cap_saturated |  |
| LCA | 2022 | wet_year;nitrogen_stress_year;low_response_year | 160.0 | 250.0 | 9 | 6 | 0 | 0 | True | True | 1 | 1 | 1 | 5 | 9990.0183 | 18429.2505 | 21.4002 | cap_saturated |  |
| LCA | 2023 | wet_year;nitrogen_stress_year;low_response_year | 160.0 | 250.0 | 9 | 6 | 0 | 0 | True | True | 1 | 1 | 14 | 5 | 7888.3557 | 17361.4465 | 0.3836 | cap_saturated |  |
| LCA | 2006 | wet_year;nitrogen_stress_year;low_response_year | 160.0 | 250.0 | 9 | 6 | 0 | 0 | True | True | 1 | 1 | 1 | 5 | 8996.9904 | 15945.564 | 11.4699 | cap_saturated |  |
| LCA | 2008 | wet_year;nitrogen_stress_year;low_response_year | 160.0 | 250.0 | 9 | 6 | 1 | 0 | True | True | 1 | 1 | 18 | 6 | 10768.1665 | 19138.1238 | 29.1817 | cap_saturated |  |
| LCA | 2009 | wet_year;nitrogen_stress_year;low_response_year | 160.0 | 250.0 | 9 | 6 | 0 | 0 | True | True | 1 | 1 | 15 | 6 | 9222.0728 | 17791.1902 | 13.7207 | cap_saturated |  |
| LCA | 2013 | wet_year;nitrogen_stress_year;low_response_year | 160.0 | 250.0 | 9 | 6 | 1 | 0 | True | True | 1 | 1 | 18 | 6 | 8756.2109 | 16625.7715 | 9.0621 | cap_saturated |  |
| SYA | 2009 | normal_year;water_stress_year;nitrogen_stress_year | 160.0 | 250.0 | 8 | 7 | 7 | 7 | True | True | 1 | 1 | 136 | 135 | 12165.1257 | 20060.0378 | 43.1513 | cap_saturated |  |
| SYA | 2014 | normal_year;water_stress_year;nitrogen_stress_year | 160.0 | 250.0 | 8 | 7 | 8 | 9 | True | True | 1 | 1 | 136 | 136 | 10363.6267 | 18626.0315 | 25.1363 | cap_saturated |  |
| SYA | 2017 | dry_year;water_stress_year;nitrogen_stress_year;irrigation_responsive_year | 160.0 | 250.0 | 8 | 7 | 4 | 0 | True | True | 1 | 1 | 117 | 12 | 10518.1335 | 18310.8301 | 26.6813 | cap_saturated |  |
| SYA | 2020 | normal_year;water_stress_year;nitrogen_stress_year | 160.0 | 250.0 | 8 | 7 | 0 | 0 | True | True | 1 | 1 | 1 | 13 | 10000.9149 | 18198.5339 | 21.5091 | cap_saturated |  |
| SYA | 2022 | wet_year;nitrogen_stress_year;low_response_year | 160.0 | 250.0 | 8 | 7 | 0 | 9 | True | True | 1 | 1 | 1 | 134 | 10597.2278 | 16819.2688 | 27.4723 | cap_saturated |  |
| SYA | 2008 | wet_year;nitrogen_stress_year;low_response_year | 160.0 | 250.0 | 8 | 7 | 0 | 0 | True | True | 1 | 1 | 1 | 15 | 11073.4253 | 17889.8865 | 32.2343 | cap_saturated |  |
| SYA | 2010 | wet_year;nitrogen_stress_year;low_response_year | 160.0 | 250.0 | 8 | 7 | 0 | 0 | True | True | 1 | 1 | 1 | 18 | 8490.4968 | 15639.0259 | 6.405 | cap_saturated |  |
| SYA | 2012 | wet_year;nitrogen_stress_year;low_response_year | 160.0 | 250.0 | 8 | 7 | 0 | 0 | True | True | 1 | 1 | 1 | 13 | 10106.0101 | 16730.6738 | 22.5601 | cap_saturated |  |
| SYA | 2023 | normal_year;nitrogen_stress_year;low_response_year | 160.0 | 250.0 | 8 | 7 | 0 | 0 | True | True | 1 | 1 | 1 | 15 | 10738.2886 | 18630.0891 | 28.8829 | cap_saturated |  |
| YCA | 2004 | dry_year;nitrogen_stress_year;irrigation_responsive_year | 160.0 | 250.0 | 12 | 8 | 7 | 16 | True | True | 1 | 1 | 109 | 89 | 8064.6497 | 18240.0574 | 2.1465 | cap_saturated |  |
| YCA | 2014 | normal_year;water_stress_year;nitrogen_stress_year | 160.0 | 250.0 | 12 | 8 | 0 | 0 | True | True | 1 | 1 | 79 | 6 | 9381.2738 | 19913.0945 | 15.3127 | cap_saturated |  |
| YCA | 2015 | normal_year;water_stress_year;nitrogen_stress_year | 160.0 | 250.0 | 12 | 8 | 0 | 0 | True | True | 1 | 1 | 1 | 6 | 9082.2668 | 20207.3474 | 12.3227 | cap_saturated |  |
| YCA | 2019 | dry_year;water_stress_year;nitrogen_stress_year | 160.0 | 250.0 | 12 | 8 | 1 | 0 | True | True | 1 | 1 | 11 | 5 | 7598.8336 | 17785.5127 | -2.5117 | cap_saturated |  |
| YCA | 2022 | wet_year;nitrogen_stress_year;low_response_year | 160.0 | 250.0 | 12 | 8 | 0 | 8 | True | True | 1 | 1 | 1 | 94 | 8496.1908 | 18689.8425 | 6.4619 | cap_saturated |  |
| YCA | 2005 | wet_year;nitrogen_stress_year;low_response_year | 160.0 | 250.0 | 11 | 8 | 1 | 0 | True | True | 1 | 1 | 15 | 5 | 9111.3995 | 19096.8359 | 12.614 | cap_saturated |  |
| YCA | 2010 | wet_year;nitrogen_stress_year;low_response_year | 160.0 | 250.0 | 12 | 8 | 1 | 0 | True | True | 1 | 1 | 15 | 6 | 7246.4764 | 17177.0618 | -6.0352 | cap_saturated |  |
| YCA | 2012 | wet_year;nitrogen_stress_year;low_response_year | 160.0 | 250.0 | 12 | 8 | 1 | 0 | True | True | 1 | 1 | 13 | 5 | 7861.5112 | 17109.3738 | 0.1151 | cap_saturated |  |
| YCA | 2023 | normal_year;nitrogen_stress_year;low_response_year | 160.0 | 250.0 | 12 | 8 | 0 | 5 | True | True | 1 | 1 | 12 | 90 | 9577.1661 | 22438.1592 | 17.2717 | cap_saturated |  |

## Main Finding

- Successful station models: 5 / 5.
- Successful evaluations: 45 / 45.
- Cap-saturated cases: 45.
- If many cases are cap_saturated or decisions do not occur near SWFAC/NSTRES stress, this simple direct PPO baseline should be reported as a baseline attempt, not as a final optimized policy.

## Figure Outputs

| station_code | year | four_panel_path |
| --- | --- | --- |
| FQA | 2007 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\FQA_2007_four_panel.png |
| FQA | 2008 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\FQA_2008_four_panel.png |
| FQA | 2016 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\FQA_2016_four_panel.png |
| FQA | 2020 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\FQA_2020_four_panel.png |
| FQA | 2023 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\FQA_2023_four_panel.png |
| FQA | 2005 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\FQA_2005_four_panel.png |
| FQA | 2011 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\FQA_2011_four_panel.png |
| FQA | 2015 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\FQA_2015_four_panel.png |
| FQA | 2019 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\FQA_2019_four_panel.png |
| HLA | 2004 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\HLA_2004_four_panel.png |
| HLA | 2020 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\HLA_2020_four_panel.png |
| HLA | 2021 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\HLA_2021_four_panel.png |
| HLA | 2022 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\HLA_2022_four_panel.png |
| HLA | 2023 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\HLA_2023_four_panel.png |
| HLA | 2012 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\HLA_2012_four_panel.png |
| HLA | 2013 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\HLA_2013_four_panel.png |
| HLA | 2015 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\HLA_2015_four_panel.png |
| HLA | 2018 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\HLA_2018_four_panel.png |
| LCA | 2017 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\LCA_2017_four_panel.png |
| LCA | 2020 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\LCA_2020_four_panel.png |
| LCA | 2021 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\LCA_2021_four_panel.png |
| LCA | 2022 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\LCA_2022_four_panel.png |
| LCA | 2023 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\LCA_2023_four_panel.png |
| LCA | 2006 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\LCA_2006_four_panel.png |
| LCA | 2008 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\LCA_2008_four_panel.png |
| LCA | 2009 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\LCA_2009_four_panel.png |
| LCA | 2013 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\LCA_2013_four_panel.png |
| SYA | 2009 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\SYA_2009_four_panel.png |
| SYA | 2014 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\SYA_2014_four_panel.png |
| SYA | 2017 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\SYA_2017_four_panel.png |
| SYA | 2020 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\SYA_2020_four_panel.png |
| SYA | 2022 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\SYA_2022_four_panel.png |
| SYA | 2008 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\SYA_2008_four_panel.png |
| SYA | 2010 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\SYA_2010_four_panel.png |
| SYA | 2012 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\SYA_2012_four_panel.png |
| SYA | 2023 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\SYA_2023_four_panel.png |
| YCA | 2004 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\YCA_2004_four_panel.png |
| YCA | 2014 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\YCA_2014_four_panel.png |
| YCA | 2015 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\YCA_2015_four_panel.png |
| YCA | 2019 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\YCA_2019_four_panel.png |
| YCA | 2022 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\YCA_2022_four_panel.png |
| YCA | 2005 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\YCA_2005_four_panel.png |
| YCA | 2010 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\YCA_2010_four_panel.png |
| YCA | 2012 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\YCA_2012_four_panel.png |
| YCA | 2023 | Leave_One_experiments\all_year_direct_action_safe_ppo\figures\four_panel\YCA_2023_four_panel.png |