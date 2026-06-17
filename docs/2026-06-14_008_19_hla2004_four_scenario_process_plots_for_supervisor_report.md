# 008_19 HLA 2004 Four-Scenario Process Plots For Supervisor

## Purpose

This report reorganizes existing HLA 2004 outputs into the four management scenarios requested by the supervisor.
No PPO training and no DSSAT rerun were performed.

## Scenarios

- `null_zero`: no irrigation and no nitrogen.
- `expert_reference_recorded`: single-year observed HLA management reference.
- `dssat_auto_attempt`: gym-DSSAT automatic-management attempt; retained as diagnostic because it behaved like null.
- `ppo_soft_stress_seed0_00815`: current HLA PPO result with no hard-minimum gate and fixed N150.

## Summary

| scenario_key | scenario_label | total_irrigation | total_n | final_grnwt | final_topwt | max_swfac | swfac_days_gt_0p05 | max_nstres | nstres_days_gt_0p05 | total_reward_for_plot | mean_daily_reward_for_plot | note | grnwt_fraction_vs_fixed_I120 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| null_zero | Null zero | 0.0 | 0.0 | 418.5661 | 778.2182 | 0.0 | 0 | 0.7988 | 122 | 786.2112 | 4.5977 | No irrigation and no nitrogen; strict lower-bound baseline. | 0.0593 |
| expert_reference_recorded | Recorded expert | 30.0 | 165.0 | 5150.1758 | 14788.2825 | 1.0 | 40 | 0.1824 | 38 | 1766.6473 | 10.392 | Single-year observed site management record, not a year-specific optimum. | 0.7292 |
| dssat_auto_attempt | DSSAT auto attempt | 0.0 | 0.0 | 418.5661 | 778.2182 | 0.0 | 0 | 0.7988 | 122 | 122.6531 | 0.7215 | Automatic-management attempt through gym-DSSAT; action totals were not fully exposed and output behaved like null. | 0.0593 |
| ppo_soft_stress_seed0_00815 | PPO soft-stress seed0 | 80.0 | 150.0 | 6777.9266 | 16184.6899 | 1.0 | 22 | 0.5545 | 74 | 13089.9578 | 76.9998 | Current HLA PPO diagnostic policy: no hard minimum gate, fixed N150, PPO controls irrigation. | 0.9597 |

## Interpretation

- HLA 2004 is the current main water-limited demonstration case.
- PPO should be represented by 008_15 soft-stress PPO, not by the older FQA transfer replay.
- The PPO case used 80 mm irrigation and achieved about 96% of the fixed I120_N150 reference yield.
- `dssat_auto_attempt` should not be described as a fully verified native DSSAT automatic-management baseline.
- PPO contribution is visible in this scenario set, but seed stability remains a documented limitation from 008_16/008_17.
- Reward curves are included for process comparison. They use the available saved daily reward column for each scenario, prioritizing diagnostic reward when available.

## Figures

- `Leave_One_experiments/hla2004_four_scenario_process_plots_008_19/figures/null_zero_01_weather_rainfall.png`
- `Leave_One_experiments/hla2004_four_scenario_process_plots_008_19/figures/null_zero_02_stress_swfac_nstres.png`
- `Leave_One_experiments/hla2004_four_scenario_process_plots_008_19/figures/null_zero_03_management_actions.png`
- `Leave_One_experiments/hla2004_four_scenario_process_plots_008_19/figures/null_zero_04_growth_topwt_grnwt.png`
- `Leave_One_experiments/hla2004_four_scenario_process_plots_008_19/figures/null_zero_05_daily_reward.png`
- `Leave_One_experiments/hla2004_four_scenario_process_plots_008_19/figures/expert_reference_recorded_01_weather_rainfall.png`
- `Leave_One_experiments/hla2004_four_scenario_process_plots_008_19/figures/expert_reference_recorded_02_stress_swfac_nstres.png`
- `Leave_One_experiments/hla2004_four_scenario_process_plots_008_19/figures/expert_reference_recorded_03_management_actions.png`
- `Leave_One_experiments/hla2004_four_scenario_process_plots_008_19/figures/expert_reference_recorded_04_growth_topwt_grnwt.png`
- `Leave_One_experiments/hla2004_four_scenario_process_plots_008_19/figures/expert_reference_recorded_05_daily_reward.png`
- `Leave_One_experiments/hla2004_four_scenario_process_plots_008_19/figures/dssat_auto_attempt_01_weather_rainfall.png`
- `Leave_One_experiments/hla2004_four_scenario_process_plots_008_19/figures/dssat_auto_attempt_02_stress_swfac_nstres.png`
- `Leave_One_experiments/hla2004_four_scenario_process_plots_008_19/figures/dssat_auto_attempt_03_management_actions.png`
- `Leave_One_experiments/hla2004_four_scenario_process_plots_008_19/figures/dssat_auto_attempt_04_growth_topwt_grnwt.png`
- `Leave_One_experiments/hla2004_four_scenario_process_plots_008_19/figures/dssat_auto_attempt_05_daily_reward.png`
- `Leave_One_experiments/hla2004_four_scenario_process_plots_008_19/figures/ppo_soft_stress_seed0_00815_01_weather_rainfall.png`
- `Leave_One_experiments/hla2004_four_scenario_process_plots_008_19/figures/ppo_soft_stress_seed0_00815_02_stress_swfac_nstres.png`
- `Leave_One_experiments/hla2004_four_scenario_process_plots_008_19/figures/ppo_soft_stress_seed0_00815_03_management_actions.png`
- `Leave_One_experiments/hla2004_four_scenario_process_plots_008_19/figures/ppo_soft_stress_seed0_00815_04_growth_topwt_grnwt.png`
- `Leave_One_experiments/hla2004_four_scenario_process_plots_008_19/figures/ppo_soft_stress_seed0_00815_05_daily_reward.png`
- `Leave_One_experiments/hla2004_four_scenario_process_plots_008_19/figures/HLA_2004_four_scenario_combined_process_comparison.png`

## Files

- Summary CSV: `Leave_One_experiments/hla2004_four_scenario_process_plots_008_19/evaluation/008_19_hla2004_four_scenario_summary_20260615_140049.csv`
- Daily values CSV for all process plots: `Leave_One_experiments/hla2004_four_scenario_process_plots_008_19/evaluation/008_19_hla2004_four_scenario_daily_values_for_plots.csv`