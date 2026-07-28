# 032_14 LC 75k PPO four-baseline success figures record

## Status

- Completed.
- Training run: 0.
- DSSAT run: 0.
- Model reselection: none.

## Figures

- Yearly comparison: `benchmark_results/032_14_lc_75k_ppo_four_baseline_success_figures/figures/032_14_lc_75k_ppo_yearly_vs_four_baseline.png`
- Mean/std summary: `benchmark_results/032_14_lc_75k_ppo_four_baseline_success_figures/figures/032_14_lc_75k_ppo_crossyear_std_summary.png`

## Success counts

| split | n_years | yield_win_four | PFP_N_win_four | any_available_metric_win_four | mean_yield_gap_vs_four_max | std_yield_gap_vs_four_max | mean_PFP_N_gap_vs_four_max | std_PFP_N_gap_vs_four_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train_2005_2010 | 6 | 1 | 6 | 6 | -191.116 | 189.934 | 8.371 | 1.355 |
| transfer_2011_2020 | 10 | 3 | 10 | 10 | -46.31 | 171.518 | 7.83 | 2.794 |
| all_2005_2020 | 16 | 4 | 16 | 16 | -100.613 | 186.864 | 8.033 | 2.317 |

## Actual PPO metric mean/std

| split | metric | unit | n_years | mean | std |
| --- | --- | --- | --- | --- | --- |
| train_2005_2010 | PPO grain yield | kg/ha | 6 | 9560.17 | 1396.378 |
| train_2005_2010 | PPO PFP_N | kg/kg N | 6 | 47.801 | 6.982 |
| train_2005_2010 | PPO irrigation | mm | 6 | 75.0 | 0.0 |
| train_2005_2010 | PPO nitrogen | kg/ha | 6 | 200.0 | 0.0 |
| transfer_2011_2020 | PPO grain yield | kg/ha | 10 | 9063.984 | 859.092 |
| transfer_2011_2020 | PPO PFP_N | kg/kg N | 10 | 45.32 | 4.295 |
| transfer_2011_2020 | PPO irrigation | mm | 10 | 75.0 | 0.0 |
| transfer_2011_2020 | PPO nitrogen | kg/ha | 10 | 200.0 | 0.0 |
| all_2005_2020 | PPO grain yield | kg/ha | 16 | 9250.054 | 1074.398 |
| all_2005_2020 | PPO PFP_N | kg/kg N | 16 | 46.25 | 5.372 |
| all_2005_2020 | PPO irrigation | mm | 16 | 75.0 | 0.0 |
| all_2005_2020 | PPO nitrogen | kg/ha | 16 | 200.0 | 0.0 |

## Per-year comparison

| split | year | yield_gap_vs_four_max_kg_ha | PFP_N_gap_vs_four_max | water_saving_vs_expert_mm | n_saving_vs_expert_kg_ha | yield_win_four | PFP_N_win_four | any_available_metric_win_four |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train_2005_2010 | 2005 | -409.713 | 9.573 | 154.0 | 48.0 | False | True | True |
| train_2005_2010 | 2006 | -264.683 | 7.701 | 124.0 | 48.0 | False | True | True |
| train_2005_2010 | 2007 | -4.952 | 7.703 | 124.0 | 48.0 | False | True | True |
| train_2005_2010 | 2008 | -94.492 | 9.945 | 124.0 | 48.0 | False | True | True |
| train_2005_2010 | 2009 | 17.46 | 8.939 | 124.0 | 48.0 | True | True | True |
| train_2005_2010 | 2010 | -390.318 | 6.363 | 123.8 | 47.0 | False | True | True |
| transfer_2011_2020 | 2011 | -290.133 | 0.254 | 124.0 | 48.0 | False | True | True |
| transfer_2011_2020 | 2012 | -21.863 | 8.579 | 154.0 | 48.0 | False | True | True |
| transfer_2011_2020 | 2013 | 42.45 | 8.63 | 124.0 | 48.0 | True | True | True |
| transfer_2011_2020 | 2014 | -34.865 | 9.938 | 124.0 | 48.0 | False | True | True |
| transfer_2011_2020 | 2015 | -258.813 | 8.663 | 124.0 | 48.0 | False | True | True |
| transfer_2011_2020 | 2016 | -176.063 | 7.347 | 124.0 | 48.0 | False | True | True |
| transfer_2011_2020 | 2017 | -16.559 | 8.639 | 124.0 | 48.0 | False | True | True |
| transfer_2011_2020 | 2018 | 315.715 | 9.166 | 124.0 | 48.0 | True | True | True |
| transfer_2011_2020 | 2019 | 1.595 | 9.735 | 124.0 | 48.0 | True | True | True |
| transfer_2011_2020 | 2020 | -24.568 | 7.353 | 124.0 | 48.0 | False | True | True |

## Metric definition boundary

- PFP_N is computed as grain yield divided by fertilizer N applied; N=0 would be undefined and must not be coerced to infinity.
- WP_ET is not computed for the PPO candidate in this figure package because the current 032_11/032_12 RL outputs do not include a defensible ET denominator.
- Water saving and N saving are reported versus official expert, not versus the four-scenario maximum/minimum envelope.
