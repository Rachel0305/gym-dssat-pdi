# 032_17 LC 75k PPO all-year five-scenario daily package record

## Status

- Completed.
- Training run: 0.
- DSSAT run: 0.
- Model reselection: none.
- Supplemental auto source: LC2008, LC2009, and LC2011 `dssat_auto` daily traces are filled by 032_18.
- LC2010 baseline daily traces are reused from the existing 027_05 five-scenario daily table because 031_35/031_36 did not contain LC2010 daily baselines.

## Scope

- Station: LC / LCA.
- Frozen candidate: LC2005-LC2010 multiyear no-forecast free-timing MaskablePPO seed0 checkpoint 75k.
- Years: LC2005-LC2020.
- LC2005-LC2010 use 032_11 train-year deterministic evaluation daily files.
- LC2011-LC2020 use 032_12 frozen transfer daily files.

## Counts

| n_years | n_years_five_scenario_daily_complete | n_years_missing_dssat_auto_daily | yield_win_four | PFP_N_win_four | any_available_metric_win_four | mean_yield_gap_vs_four_max | mean_water_saving_vs_expert | mean_n_saving_vs_expert |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 16 | 0 | 4 | 15 | 15 | -87.152 | 127.503 | 47.469 |

## Manifest

| year | split | daily_csv | summary_csv | figure_png | figure_svg | five_scenario_daily_complete | missing_daily_scenarios |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2005 | train_2005_2010 | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/tables/032_17_lc2005_75k_ppo_five_scenario_daily.csv | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/tables/032_17_lc2005_75k_ppo_five_scenario_summary.csv | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/figures/032_17_lc2005_75k_ppo_five_scenario_daily.png | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/figures/032_17_lc2005_75k_ppo_five_scenario_daily.svg | True |  |
| 2006 | train_2005_2010 | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/tables/032_17_lc2006_75k_ppo_five_scenario_daily.csv | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/tables/032_17_lc2006_75k_ppo_five_scenario_summary.csv | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/figures/032_17_lc2006_75k_ppo_five_scenario_daily.png | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/figures/032_17_lc2006_75k_ppo_five_scenario_daily.svg | True |  |
| 2007 | train_2005_2010 | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/tables/032_17_lc2007_75k_ppo_five_scenario_daily.csv | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/tables/032_17_lc2007_75k_ppo_five_scenario_summary.csv | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/figures/032_17_lc2007_75k_ppo_five_scenario_daily.png | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/figures/032_17_lc2007_75k_ppo_five_scenario_daily.svg | True |  |
| 2008 | train_2005_2010 | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/tables/032_17_lc2008_75k_ppo_five_scenario_daily.csv | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/tables/032_17_lc2008_75k_ppo_five_scenario_summary.csv | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/figures/032_17_lc2008_75k_ppo_five_scenario_daily.png | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/figures/032_17_lc2008_75k_ppo_five_scenario_daily.svg | True |  |
| 2009 | train_2005_2010 | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/tables/032_17_lc2009_75k_ppo_five_scenario_daily.csv | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/tables/032_17_lc2009_75k_ppo_five_scenario_summary.csv | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/figures/032_17_lc2009_75k_ppo_five_scenario_daily.png | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/figures/032_17_lc2009_75k_ppo_five_scenario_daily.svg | True |  |
| 2010 | train_2005_2010 | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/tables/032_17_lc2010_75k_ppo_five_scenario_daily.csv | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/tables/032_17_lc2010_75k_ppo_five_scenario_summary.csv | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/figures/032_17_lc2010_75k_ppo_five_scenario_daily.png | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/figures/032_17_lc2010_75k_ppo_five_scenario_daily.svg | True |  |
| 2011 | transfer_2011_2020 | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/tables/032_17_lc2011_75k_ppo_five_scenario_daily.csv | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/tables/032_17_lc2011_75k_ppo_five_scenario_summary.csv | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/figures/032_17_lc2011_75k_ppo_five_scenario_daily.png | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/figures/032_17_lc2011_75k_ppo_five_scenario_daily.svg | True |  |
| 2012 | transfer_2011_2020 | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/tables/032_17_lc2012_75k_ppo_five_scenario_daily.csv | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/tables/032_17_lc2012_75k_ppo_five_scenario_summary.csv | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/figures/032_17_lc2012_75k_ppo_five_scenario_daily.png | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/figures/032_17_lc2012_75k_ppo_five_scenario_daily.svg | True |  |
| 2013 | transfer_2011_2020 | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/tables/032_17_lc2013_75k_ppo_five_scenario_daily.csv | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/tables/032_17_lc2013_75k_ppo_five_scenario_summary.csv | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/figures/032_17_lc2013_75k_ppo_five_scenario_daily.png | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/figures/032_17_lc2013_75k_ppo_five_scenario_daily.svg | True |  |
| 2014 | transfer_2011_2020 | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/tables/032_17_lc2014_75k_ppo_five_scenario_daily.csv | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/tables/032_17_lc2014_75k_ppo_five_scenario_summary.csv | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/figures/032_17_lc2014_75k_ppo_five_scenario_daily.png | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/figures/032_17_lc2014_75k_ppo_five_scenario_daily.svg | True |  |
| 2015 | transfer_2011_2020 | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/tables/032_17_lc2015_75k_ppo_five_scenario_daily.csv | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/tables/032_17_lc2015_75k_ppo_five_scenario_summary.csv | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/figures/032_17_lc2015_75k_ppo_five_scenario_daily.png | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/figures/032_17_lc2015_75k_ppo_five_scenario_daily.svg | True |  |
| 2016 | transfer_2011_2020 | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/tables/032_17_lc2016_75k_ppo_five_scenario_daily.csv | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/tables/032_17_lc2016_75k_ppo_five_scenario_summary.csv | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/figures/032_17_lc2016_75k_ppo_five_scenario_daily.png | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/figures/032_17_lc2016_75k_ppo_five_scenario_daily.svg | True |  |
| 2017 | transfer_2011_2020 | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/tables/032_17_lc2017_75k_ppo_five_scenario_daily.csv | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/tables/032_17_lc2017_75k_ppo_five_scenario_summary.csv | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/figures/032_17_lc2017_75k_ppo_five_scenario_daily.png | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/figures/032_17_lc2017_75k_ppo_five_scenario_daily.svg | True |  |
| 2018 | transfer_2011_2020 | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/tables/032_17_lc2018_75k_ppo_five_scenario_daily.csv | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/tables/032_17_lc2018_75k_ppo_five_scenario_summary.csv | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/figures/032_17_lc2018_75k_ppo_five_scenario_daily.png | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/figures/032_17_lc2018_75k_ppo_five_scenario_daily.svg | True |  |
| 2019 | transfer_2011_2020 | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/tables/032_17_lc2019_75k_ppo_five_scenario_daily.csv | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/tables/032_17_lc2019_75k_ppo_five_scenario_summary.csv | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/figures/032_17_lc2019_75k_ppo_five_scenario_daily.png | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/figures/032_17_lc2019_75k_ppo_five_scenario_daily.svg | True |  |
| 2020 | transfer_2011_2020 | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/tables/032_17_lc2020_75k_ppo_five_scenario_daily.csv | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/tables/032_17_lc2020_75k_ppo_five_scenario_summary.csv | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/figures/032_17_lc2020_75k_ppo_five_scenario_daily.png | benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/figures/032_17_lc2020_75k_ppo_five_scenario_daily.svg | True |  |

## Per-year comparison summary

| year | split | rl_yield_kg_ha | four_max_yield_kg_ha | yield_gap_vs_four_max | rl_irrigation_mm | expert_irrigation_mm | water_saving_vs_expert | rl_nitrogen_kg_ha | expert_nitrogen_kg_ha | n_saving_vs_expert | rl_PFP_N_kg_kg | four_max_PFP_N_kg_kg | PFP_N_gap_vs_four_max | action_pattern | yield_win_four | PFP_N_win_four | any_available_metric_win_four |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2005 | train_2005_2010 | 11674.674 | 12084.387 | -409.713 | 75.0 | 228.75 | 153.75 | 200.0 | 247.5 | 47.5 | 58.373 | 48.826 | 9.548 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80 | False | True | True |
| 2006 | train_2005_2010 | 9180.162 | 9444.844 | -264.683 | 75.0 | 198.75 | 123.75 | 200.0 | 247.5 | 47.5 | 45.901 | 38.161 | 7.74 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80 | False | True | True |
| 2007 | train_2005_2010 | 8100.562 | 8105.514 | -4.952 | 75.0 | 198.75 | 123.75 | 200.0 | 247.5 | 47.5 | 40.503 | 32.75 | 7.753 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80 | False | True | True |
| 2008 | train_2005_2010 | 10769.094 | 10863.586 | -94.492 | 75.0 | 198.75 | 123.75 | 200.0 | 247.5 | 47.5 | 53.845 | 43.893 | 9.952 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80 | False | True | True |
| 2009 | train_2005_2010 | 9287.845 | 9270.385 | 17.46 | 75.0 | 198.75 | 123.75 | 200.0 | 247.5 | 47.5 | 46.439 | 37.456 | 8.983 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80 | True | True | True |
| 2010 | train_2005_2010 | 8348.682 | 8739.0 | -390.318 | 75.0 | 198.8 | 123.8 | 200.0 | 247.0 | 47.0 | 41.743 | 43.66 | -1.917 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80 | False | False | False |
| 2011 | transfer_2011_2020 | 9049.867 | 9124.623 | -74.756 | 75.0 | 198.75 | 123.75 | 200.0 | 247.5 | 47.5 | 45.249 | 36.784 | 8.466 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80 | False | True | True |
| 2012 | transfer_2011_2020 | 8995.84 | 9017.703 | -21.863 | 75.0 | 228.75 | 153.75 | 200.0 | 247.5 | 47.5 | 44.979 | 36.435 | 8.544 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80 | False | True | True |
| 2013 | transfer_2011_2020 | 8765.942 | 8723.492 | 42.45 | 75.0 | 198.75 | 123.75 | 200.0 | 247.5 | 47.5 | 43.83 | 35.246 | 8.583 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80 | True | True | True |
| 2014 | transfer_2011_2020 | 10487.562 | 10522.427 | -34.865 | 75.0 | 198.75 | 123.75 | 200.0 | 247.5 | 47.5 | 52.438 | 42.515 | 9.923 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80 | False | True | True |
| 2015 | transfer_2011_2020 | 10112.574 | 10371.388 | -258.813 | 75.0 | 198.75 | 123.75 | 200.0 | 247.5 | 47.5 | 50.563 | 41.905 | 8.658 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80 | False | True | True |
| 2016 | transfer_2011_2020 | 8389.412 | 8565.475 | -176.063 | 75.0 | 198.75 | 123.75 | 200.0 | 247.5 | 47.5 | 41.947 | 34.608 | 7.339 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80 | False | True | True |
| 2017 | transfer_2011_2020 | 9047.879 | 9064.438 | -16.559 | 75.0 | 198.75 | 123.75 | 200.0 | 247.5 | 47.5 | 45.239 | 36.624 | 8.615 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80 | False | True | True |
| 2018 | transfer_2011_2020 | 8233.118 | 7917.404 | 315.715 | 75.0 | 198.75 | 123.75 | 200.0 | 247.5 | 47.5 | 41.166 | 31.99 | 9.176 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80 | True | True | True |
| 2019 | transfer_2011_2020 | 9807.0 | 9805.405 | 1.595 | 75.0 | 198.75 | 123.75 | 200.0 | 247.5 | 47.5 | 49.035 | 39.296 | 9.739 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80 | True | True | True |
| 2020 | transfer_2011_2020 | 7750.649 | 7775.217 | -24.568 | 75.0 | 198.75 | 123.75 | 200.0 | 247.5 | 47.5 | 38.753 | 31.415 | 7.338 | DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80 | False | True | True |

## Reward plotting note

- Source reward columns are not used for cross-scenario comparison.
- `common_reward_cumulative_scaled` is recomputed using one 032 stress-aware formula for available scenarios.
- SWTD/soil-water storage is not plotted because the current completed baseline daily files do not provide a reliable soil-water storage column.
- After 032_18 supplementation, all LC2005-LC2020 figures contain the five scenarios: null, recorded farmer, DSSAT auto, official expert, and 75k PPO candidate.
