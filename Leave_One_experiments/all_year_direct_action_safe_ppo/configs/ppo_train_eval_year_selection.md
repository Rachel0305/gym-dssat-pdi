# PPO Train/Eval Year Selection

Selection is based on the 006_15 all-year weather scenario pool.
Priority: water-stress, irrigation-responsive, nitrogen-stress, 2020-2023 calibration/validation candidates, and dry/normal/wet mix.

| station_code | year | scenario_type | selected_for_train | selected_for_eval | selection_reason | growing_season_rain | max_swfac |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FQA | 2005 | wet_year;water_stress_year;nitrogen_stress_year | False | True | selected_eval_priority_all_year_pool;water_stress_year;nitrogen_stress_year | 535.1 | 0.3782 |
| FQA | 2007 | normal_year;water_stress_year;nitrogen_stress_year;irrigation_responsive_year | True | False | selected_train_priority_all_year_pool;water_stress_year;irrigation_responsive_year;nitrogen_stress_year | 325.8 | 0.4622 |
| FQA | 2008 | normal_year;water_stress_year;nitrogen_stress_year;irrigation_responsive_year | True | False | selected_train_priority_all_year_pool;water_stress_year;irrigation_responsive_year;nitrogen_stress_year | 221.8 | 0.6031 |
| FQA | 2011 | wet_year;water_stress_year;nitrogen_stress_year | False | True | selected_eval_priority_all_year_pool;water_stress_year;nitrogen_stress_year | 1051.2 | 0.4627 |
| FQA | 2015 | normal_year;water_stress_year;nitrogen_stress_year | False | True | selected_eval_priority_all_year_pool;water_stress_year;nitrogen_stress_year | 299.6 | 0.5991 |
| FQA | 2016 | normal_year;water_stress_year;nitrogen_stress_year;irrigation_responsive_year | True | False | selected_train_priority_all_year_pool;water_stress_year;irrigation_responsive_year;nitrogen_stress_year | 262.8 | 0.5806 |
| FQA | 2019 | dry_year;water_stress_year;nitrogen_stress_year | False | True | selected_eval_priority_all_year_pool;water_stress_year;nitrogen_stress_year | 201.3 | 0.5568 |
| FQA | 2020 | normal_year;water_stress_year;nitrogen_stress_year | True | False | selected_train_priority_all_year_pool;water_stress_year;nitrogen_stress_year;2020_2023_calibration_validation_candidate | 411.0 | 0.2008 |
| FQA | 2023 | normal_year;water_stress_year;nitrogen_stress_year | True | False | selected_train_priority_all_year_pool;water_stress_year;nitrogen_stress_year;2020_2023_calibration_validation_candidate | 389.9 | 0.3482 |
| HLA | 2004 | dry_year;water_stress_year;nitrogen_stress_year;irrigation_responsive_year | True | False | selected_train_priority_all_year_pool;water_stress_year;irrigation_responsive_year;nitrogen_stress_year | 51.5 | 1.0 |
| HLA | 2012 | wet_year;nitrogen_stress_year;low_response_year | False | True | selected_eval_priority_all_year_pool;nitrogen_stress_year | 546.5 | 0.0 |
| HLA | 2013 | wet_year;nitrogen_stress_year;low_response_year | False | True | selected_eval_priority_all_year_pool;nitrogen_stress_year | 825.1 | 0.0 |
| HLA | 2015 | dry_year;nitrogen_stress_year;low_response_year | False | True | selected_eval_priority_all_year_pool;nitrogen_stress_year | 342.1 | 0.0 |
| HLA | 2018 | wet_year;nitrogen_stress_year;low_response_year | False | True | selected_eval_priority_all_year_pool;nitrogen_stress_year | 824.4 | 0.0 |
| HLA | 2020 | wet_year;nitrogen_stress_year;low_response_year | True | False | selected_train_priority_all_year_pool;nitrogen_stress_year;2020_2023_calibration_validation_candidate | 782.8 | 0.0 |
| HLA | 2021 | wet_year;nitrogen_stress_year;low_response_year | True | False | selected_train_priority_all_year_pool;nitrogen_stress_year;2020_2023_calibration_validation_candidate | 646.4 | 0.0 |
| HLA | 2022 | normal_year;nitrogen_stress_year;low_response_year | True | False | selected_train_priority_all_year_pool;nitrogen_stress_year;2020_2023_calibration_validation_candidate | 368.1 | 0.0 |
| HLA | 2023 | wet_year;nitrogen_stress_year;low_response_year | True | False | selected_train_priority_all_year_pool;nitrogen_stress_year;2020_2023_calibration_validation_candidate | 740.8 | 0.0 |
| LCA | 2006 | wet_year;nitrogen_stress_year;low_response_year | False | True | selected_eval_priority_all_year_pool;nitrogen_stress_year | 347.2 | 0.0 |
| LCA | 2008 | wet_year;nitrogen_stress_year;low_response_year | False | True | selected_eval_priority_all_year_pool;nitrogen_stress_year | 393.8 | 0.0 |
| LCA | 2009 | wet_year;nitrogen_stress_year;low_response_year | False | True | selected_eval_priority_all_year_pool;nitrogen_stress_year | 417.1 | 0.0 |
| LCA | 2013 | wet_year;nitrogen_stress_year;low_response_year | False | True | selected_eval_priority_all_year_pool;nitrogen_stress_year | 417.6 | 0.0 |
| LCA | 2017 | dry_year;water_stress_year;nitrogen_stress_year | True | False | selected_train_priority_all_year_pool;water_stress_year;nitrogen_stress_year | 175.2 | 0.1204 |
| LCA | 2020 | normal_year;nitrogen_stress_year;low_response_year | True | False | selected_train_priority_all_year_pool;nitrogen_stress_year;2020_2023_calibration_validation_candidate | 293.5 | 0.0 |
| LCA | 2021 | wet_year;nitrogen_stress_year;low_response_year | True | False | selected_train_priority_all_year_pool;nitrogen_stress_year;2020_2023_calibration_validation_candidate | 425.3 | 0.0 |
| LCA | 2022 | wet_year;nitrogen_stress_year;low_response_year | True | False | selected_train_priority_all_year_pool;nitrogen_stress_year;2020_2023_calibration_validation_candidate | 451.3 | 0.0 |
| LCA | 2023 | wet_year;nitrogen_stress_year;low_response_year | True | False | selected_train_priority_all_year_pool;nitrogen_stress_year;2020_2023_calibration_validation_candidate | 497.2 | 0.0 |
| SYA | 2008 | wet_year;nitrogen_stress_year;low_response_year | False | True | selected_eval_priority_all_year_pool;nitrogen_stress_year | 584.4 | 0.0 |
| SYA | 2009 | normal_year;water_stress_year;nitrogen_stress_year | True | False | selected_train_priority_all_year_pool;water_stress_year;nitrogen_stress_year | 332.0 | 0.5943 |
| SYA | 2010 | wet_year;nitrogen_stress_year;low_response_year | False | True | selected_eval_priority_all_year_pool;nitrogen_stress_year | 796.6 | 0.0 |
| SYA | 2012 | wet_year;nitrogen_stress_year;low_response_year | False | True | selected_eval_priority_all_year_pool;nitrogen_stress_year | 708.0 | 0.0 |
| SYA | 2014 | normal_year;water_stress_year;nitrogen_stress_year | True | False | selected_train_priority_all_year_pool;water_stress_year;nitrogen_stress_year | 331.8 | 0.8005 |
| SYA | 2017 | dry_year;water_stress_year;nitrogen_stress_year;irrigation_responsive_year | True | False | selected_train_priority_all_year_pool;water_stress_year;irrigation_responsive_year;nitrogen_stress_year | 301.7 | 0.8179 |
| SYA | 2020 | normal_year;water_stress_year;nitrogen_stress_year | True | False | selected_train_priority_all_year_pool;water_stress_year;nitrogen_stress_year;2020_2023_calibration_validation_candidate | 570.7 | 0.6626 |
| SYA | 2022 | wet_year;nitrogen_stress_year;low_response_year | True | False | selected_train_priority_all_year_pool;nitrogen_stress_year;2020_2023_calibration_validation_candidate | 687.1 | 0.0 |
| SYA | 2023 | normal_year;nitrogen_stress_year;low_response_year | False | True | selected_eval_priority_all_year_pool;nitrogen_stress_year;2020_2023_calibration_validation_candidate | 491.7 | 0.0 |
| YCA | 2004 | dry_year;nitrogen_stress_year;irrigation_responsive_year | True | False | selected_train_priority_all_year_pool;irrigation_responsive_year;nitrogen_stress_year | 27.6 | 0.0 |
| YCA | 2005 | wet_year;nitrogen_stress_year;low_response_year | False | True | selected_eval_priority_all_year_pool;nitrogen_stress_year | 551.4 | 0.0 |
| YCA | 2010 | wet_year;nitrogen_stress_year;low_response_year | False | True | selected_eval_priority_all_year_pool;nitrogen_stress_year | 664.5 | 0.0 |
| YCA | 2012 | wet_year;nitrogen_stress_year;low_response_year | False | True | selected_eval_priority_all_year_pool;nitrogen_stress_year | 460.3 | 0.0 |
| YCA | 2014 | normal_year;water_stress_year;nitrogen_stress_year | True | False | selected_train_priority_all_year_pool;water_stress_year;nitrogen_stress_year | 210.7 | 0.6202 |
| YCA | 2015 | normal_year;water_stress_year;nitrogen_stress_year | True | False | selected_train_priority_all_year_pool;water_stress_year;nitrogen_stress_year | 295.8 | 0.1004 |
| YCA | 2019 | dry_year;water_stress_year;nitrogen_stress_year | True | False | selected_train_priority_all_year_pool;water_stress_year;nitrogen_stress_year | 198.7 | 0.1173 |
| YCA | 2022 | wet_year;nitrogen_stress_year;low_response_year | True | False | selected_train_priority_all_year_pool;nitrogen_stress_year;2020_2023_calibration_validation_candidate | 709.0 | 0.0 |
| YCA | 2023 | normal_year;nitrogen_stress_year;low_response_year | False | True | selected_eval_priority_all_year_pool;nitrogen_stress_year;2020_2023_calibration_validation_candidate | 413.2 | 0.0 |