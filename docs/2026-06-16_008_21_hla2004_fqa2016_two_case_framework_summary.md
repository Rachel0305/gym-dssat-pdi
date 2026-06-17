# 008_21 HLA 2004 + FQA 2016 Two-Case Framework Summary

## Purpose

This report summarizes the current two-case evidence that the weather/stress-assisted stage PPO framework is not limited to one year.
HLA 2004 is the primary dry-year demonstration case. FQA 2016 is the second water-stress validation case.

## Two-Case PPO Summary

| case_id | case_role | station | year | scenario_label | total_irrigation | total_n | final_grnwt | max_swfac | swfac_days_gt_0p05 | max_nstres | nstres_days_gt_0p05 | total_reward_for_plot | framework_signal | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HLA 2004 | primary_demo | HLA | 2004 | PPO soft-stress seed0 | 80.0 | 150.0 | 6777.9266 | 1.0 | 22 | 0.5545 | 74 | 13089.9578 | nonzero_non_saturated_interpretable | Current HLA PPO diagnostic policy: no hard minimum gate, fixed N150, PPO controls irrigation. |
| FQA 2016 | second_validation | FQA | 2016 | PPO soft-stress seed0 | 80.0 | 150.0 | 6999.2743 | 0.4309 | 9 | 0.0122 | 0 | 4403.7733 | nonzero_non_saturated_interpretable | 008_20 FQA 2016 soft-stress stage PPO; fixed N150, PPO controls irrigation. |

## FQA 2016 Four-Scenario Summary

| station | year | scenario_key | scenario_label | total_irrigation | total_n | final_grnwt | final_topwt | max_swfac | swfac_days_gt_0p05 | max_nstres | nstres_days_gt_0p05 | total_reward_for_plot | mean_daily_reward_for_plot | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FQA | 2016 | null_zero | Null zero | 0.0 | 0.0 | 2660.7129 | 5048.7579 | 0.0 | 0 | 0.4237 | 61 | 784.9032 | 8.4398 | No irrigation and no nitrogen. |
| FQA | 2016 | expert_reference_recorded | Recorded expert | 75.0 | 144.0 | 7180.9351 | 13248.9758 | 0.1967 | 2 | 0.339 | 10 | 2087.9123 | 22.4507 | Single-year FQA observed management record; not a year-specific optimum. |
| FQA | 2016 | dssat_auto_attempt | DSSAT auto attempt | 0.0 | 0.0 | 2682.9407 | 5078.4558 | 0.0 | 0 | 0.4218 | 61 | 790.8336 | 8.5036 | Automatic-management attempt through gym-DSSAT; retained as diagnostic. |
| FQA | 2016 | ppo_soft_stress_seed0 | PPO soft-stress seed0 | 80.0 | 150.0 | 6999.2743 | 13056.6638 | 0.4309 | 9 | 0.0122 | 0 | 4403.7733 | 47.3524 | 008_20 FQA 2016 soft-stress stage PPO; fixed N150, PPO controls irrigation. |

## Interpretation

- HLA 2004 and FQA 2016 both show nonzero and non-saturated PPO irrigation.
- FQA 2016 extends the framework evidence beyond the primary HLA 2004 demonstration case.
- This still does not prove universal optimality or full water-nitrogen simultaneous optimization.
- The current claim should remain: the framework can generate interpretable irrigation decisions in representative water-stress years under fixed adequate nitrogen supply.

## Figures

- FQA 2016 four-scenario seven-panel process plot: `Leave_One_experiments/two_case_framework_summary_008_21/figures/FQA_2016_four_scenario_combined_process_comparison.png`
- HLA 2004 + FQA 2016 two-case PPO summary plot: `Leave_One_experiments/two_case_framework_summary_008_21/figures/HLA2004_FQA2016_two_case_ppo_summary.png`

## Files

- FQA daily values for plots: `Leave_One_experiments/two_case_framework_summary_008_21/evaluation/008_21_fqa2016_four_scenario_daily_values_for_plots.csv`
- FQA four-scenario summary: `Leave_One_experiments/two_case_framework_summary_008_21/evaluation/008_21_fqa2016_four_scenario_summary.csv`
- HLA daily values copy: `Leave_One_experiments/two_case_framework_summary_008_21/evaluation/008_21_hla2004_four_scenario_daily_values_for_plots_copy.csv`
- HLA summary copy: `Leave_One_experiments/two_case_framework_summary_008_21/evaluation/008_21_hla2004_four_scenario_summary_copy.csv`
- Two-case PPO summary: `Leave_One_experiments/two_case_framework_summary_008_21/evaluation/008_21_hla2004_fqa2016_two_case_ppo_summary.csv`